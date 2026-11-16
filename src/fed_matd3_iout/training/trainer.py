"""Off-policy (MATD3-family) training loop with optional asynchronous federation of the actors.

Training information: critics use simulator global state + joint actions (CTDE).  Execution: actors use local obs.
Experience budget is identical for every method: `experiment.train_episodes` x `env.episode_steps` env steps.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from ..agents.actor import flatten_params, load_flat_params
from ..agents.exploration import NoiseSchedule
from ..agents.matd3_agent import MATD3Agent
from ..agents.replay_buffer import JointStore
from ..environment.multi_auv_iout_env import MultiAUVIoUTEnv
from ..utils.logger import RunLogger
from ..utils.reproducibility import atomic_write_json
from ..utils.seed import make_rng, seed_everything
from .checkpoint import save_checkpoint
from .federated_training import build_federation
from .local_training import local_update

TRAIN_OFFSET_STRIDE = 10_000


class Trainer:
    def __init__(self, cfg, device, run_dir: str | Path | None = None, train_seed: int = 0, logger: RunLogger | None = None):
        self.cfg, self.device = cfg, device
        self.seed = int(train_seed)
        seed_everything(self.seed)
        self.env = MultiAUVIoUTEnv(cfg)
        e, m = self.env, cfg["matd3"]
        self.agents = [MATD3Agent(i, e.M, e.obs_dim, e.act_dim, e.state_dim, m, device) for i in range(e.M)]
        self.store = JointStore(int(m["replay_capacity"]), e.state_dim, e.M, e.obs_dim, e.act_dim)
        self.rng_noise = make_rng(self.seed, "policy_noise")
        use_sem = bool(cfg["semantic"]["use_in_decisions"])
        self.clients, self.server, self.proto = build_federation(cfg, self.agents, self.store,
                                                                 make_rng(self.seed, "comm", 7), use_sem)
        self.noise = NoiseSchedule(m["expl_noise_start"], m["expl_noise_end"], m["expl_decay_episodes"])
        self.run_dir = Path(run_dir) if run_dir else None
        self.log = logger or RunLogger(self.run_dir, verbose=False)
        self.curve: list[dict] = []
        self.global_time = 0.0
        self.total_steps = 0

    # ------------------------------------------------------------------ main loop
    def train(self, n_episodes: int | None = None, eval_fn=None, eval_every: int = 0, ckpt_every: int = 0) -> list[dict]:
        cfg, env = self.cfg, self.env
        n_ep = int(n_episodes or cfg["experiment"]["train_episodes"])
        m = cfg["matd3"]
        tgt = [a.actor_t for a in self.agents]
        e_grad = float(cfg["energy"]["e_grad_j"])
        for ep in range(n_ep):
            obs, info = env.reset(self.seed * TRAIN_OFFSET_STRIDE + ep)
            state = info["state"]
            std = self.noise.std(ep)
            ep_ret = np.zeros(env.M)
            t0 = time.time()
            for _ in range(env.T):
                intr = env.in_transit_mask()
                act = np.stack([self.agents[i].act(obs[i], std, self.rng_noise) for i in range(env.M)])
                obs2, r, term, trunc, info = env.step(act)
                state2 = info["state"]
                tid = self.store.add(state, obs, act, r, state2, obs2, float(term))   # truncation is NOT terminal
                for c in self.clients:
                    if intr[c.id]:
                        c.replay.add(tid)
                        c.trans_since += 1
                self.total_steps += 1
                if self.total_steps % int(m["update_every"]) == 0:
                    for c in self.clients:
                        local_update(c, tgt, self.device, int(m["warmup"]), int(m["batch_size"]), e_grad, env)
                self.global_time += env.dt
                if self.proto is not None:
                    for rec in self.proto.step(env, self.global_time):
                        self.log.log("aggregation", rec)
                ep_ret += r
                obs, state = obs2, state2
                if trunc:
                    break
            acc = env.accounting()
            rec = {"episode": ep, "return_mean": float(ep_ret.mean()), "U": env.ledger.U, "delivered_mbit": acc["delivered"],
                   "lost_mbit": acc["lost"], "E_comp_j": float(sum(env.energy.comp)),
                   "E_comm_j": float(sum(env.energy.tx) + sum(env.energy.rx)), "E_prop_j": float(sum(env.energy.prop)),
                   "noise": std, "global_version": self.server.version if self.server else 0,
                   "sec": time.time() - t0, "steps_total": self.total_steps}
            self.curve.append(rec)
            self.log.log("train", rec)
            if eval_fn and eval_every and (ep + 1) % eval_every == 0:
                self.curve[-1]["eval"] = eval_fn(self)
            if ckpt_every and self.run_dir and (ep + 1) % ckpt_every == 0:
                self.save(f"ep{ep + 1}")
        return self.curve

    # ------------------------------------------------------------------ policies for evaluation
    def local_policies(self):
        return [lambda o, a=a: a.act(o) for a in self.agents]

    def global_policies(self):
        """The global actor deployed to every AUV (federated methods only)."""
        if self.server is None:
            return self.local_policies()
        from ..agents.actor import Actor
        ref = self.agents[0]
        net = Actor(ref.obs_dim, ref.act_dim, int(self.cfg["matd3"]["hidden"])).to(self.device)
        load_flat_params(net, self.server.theta)
        def pol(o, net=net):
            with torch.no_grad():
                return np.clip(net(torch.as_tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0))[0].cpu().numpy(), -1, 1)
        return [pol for _ in self.agents]

    # ------------------------------------------------------------------ checkpoint
    def save(self, tag: str) -> Path:
        payload = {"tag": tag, "seed": self.seed, "actors": [flatten_params(a.actor) for a in self.agents],
                   "critics": [a.critic.state_dict() for a in self.agents],
                   "global_actor": None if self.server is None else self.server.theta,
                   "global_version": 0 if self.server is None else self.server.version,
                   "local_base_versions": [c.base_version for c in self.clients],
                   "local_update_versions": [c.local_version for c in self.clients],
                   "curve": self.curve}
        p = save_checkpoint(self.run_dir / "checkpoints", f"{tag}", payload)
        if self.server is not None:
            atomic_write_json(self.run_dir / f"aggregation_history_{tag}.json", self.server.history)
            atomic_write_json(self.run_dir / f"communication_history_{tag}.json",
                              {"summary": self.proto.summary(), "server_events": self.server.events})
        return p
