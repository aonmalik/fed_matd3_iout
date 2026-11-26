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
from ..utils.compute_energy import ComputeMeter
from ..utils.logger import RunLogger
from ..utils.reproducibility import atomic_write_json
from ..utils.seed import make_rng, seed_everything
from .checkpoint import save_checkpoint
from .federated_training import build_federation
from .local_training import local_update

TRAIN_OFFSET_STRIDE = 10_000


# D8: what a drain step does (documented in docs/PROTOCOL_2.2_DECISION_LOG.md; asserted by tests/test_d8_drain_semantics.py)
DRAIN_SEMANTICS = {
    "communication_progress": {"transfers_progress": True, "aggregation_and_terminal_messages_progress": True, "new_federated_updates": False},
    "mission_state_evolution": {"sensor_arrivals": True, "task_allocation": True, "auv_motion": True, "service_and_collection": True,
                                "sensor_data_uplink": True, "link_process": True, "actors_used": "current deterministic local actors (no exploration noise)"},
    "training_and_learning": {"gradient_steps": False, "policy_optimisation": False, "replay_writes": False, "new_episodes": False},
    "energy": {"drain_energy_in_reported_episode_energy": False, "drain_energy_in_evaluation_energy": False, "drain_energy_recorded_separately": True},
    "evaluation_starts": "after the drain ends (all resolved, or the bounded budget expired), on freshly reset scenario episodes",
}


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
        self.termination: dict | None = None
        self.meter = ComputeMeter()            # filled by finalize(): cutoff / drain / evaluation-start census

    def compute_report(self) -> dict:
        """Measured training compute time/energy of the local updates (D9).  Distinct from the MODELLED energy.comp (e_grad_j)."""
        return self.meter.report()

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
                    with self.meter:
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

    # ------------------------------------------------------------------ finite-horizon termination (Protocol 2.2)
    def finalize(self, drain_s: float | None = None) -> dict:
        """Explicit end-of-training policy.  Records the TRAINING CUTOFF census, optionally runs a bounded COMMUNICATION DRAIN,
        and records the census at EVALUATION START.

        Drain (only if ``federated.final_drain_s`` > 0): the environment keeps stepping with the current deterministic local
        actors so link states keep evolving, but NO local update is trained, NO transition is stored, and ``proto.accept_new``
        is False so no new federated update is created.  Only transfers/aggregations that already exist can resolve.  The drain
        stops at the first of: nothing left to resolve, or ``drain_s`` seconds.  Nothing is forced: updates still unresolved
        afterwards stay unresolved (their statistics stay "held": exactly-once settlement is not bypassed) and are reported.
        The training budget (episodes x steps) is unchanged; drain steps are not training steps.
        """
        if self.proto is None:
            return {"federated": False}
        budget = float(self.cfg["federated"].get("final_drain_s", 0.0) if drain_s is None else drain_s)
        cut = self.proto.pending_report()
        t_cutoff = self.global_time
        agg_cut = self.proto.n_status["aggregated"]
        env = self.env
        steps = 0
        snap0 = self._drain_snapshot()
        if budget > 0.0 and self.proto.drain_active():
            obs = env._obs_all()
            pols = self.local_policies()
            previous_accept_new = self.proto.accept_new
            self.proto.accept_new = False
            try:
                while steps * env.dt < budget and self.proto.drain_active():
                    act = np.stack([pols[i](obs[i]) for i in range(env.M)])
                    obs, _, _, _, _ = env.step(act)
                    self.global_time += env.dt
                    for rec in self.proto.step(env, self.global_time):
                        self.log.log("aggregation", rec)
                    steps += 1
            finally:
                self.proto.accept_new = previous_accept_new
        end = self.proto.pending_report()
        snap1 = self._drain_snapshot()
        self.termination = {
            "federated": True, "drain_budget_s": budget, "drain_steps": steps, "drain_s": steps * env.dt,
            "t_training_cutoff_s": t_cutoff, "t_evaluation_start_s": self.global_time,
            "pending_at_cutoff": cut, "pending_at_evaluation_start": end,
            "aggregated_during_drain": self.proto.n_status["aggregated"] - agg_cut,
            "global_version_at_cutoff": cut["global_version"], "global_version_at_evaluation_start": end["global_version"],
            "drain_exhausted_budget": bool(budget > 0.0 and steps * env.dt >= budget and self.proto.drain_active()),
            # ---- explicit stage record (training cutoff -> census -> drain start -> bounded drain -> drain end -> evaluation start)
            "stages": {"1_training_cutoff_s": t_cutoff, "2_census_unresolved": cut["unresolved_total"],
                       "3_drain_start_s": t_cutoff if steps > 0 else None, "4_drain_budget_s": budget,
                       "5_drain_end_s": self.global_time, "6_evaluation_start_s": self.global_time},
            "drain_terminated_by": ("drain_not_requested" if budget <= 0.0 else
                                    "nothing_pending_at_cutoff" if steps == 0 else
                                    "budget_exhausted" if self.proto.drain_active() else "all_resolved"),
            "resolved_during_drain": cut["unresolved_total"] - end["unresolved_total"],
            "unresolved_at_drain_expiry": end["unresolved_total"],
            "evaluation_started_with_unresolved": bool(end["unresolved_total"] > 0),
            "statistics_accounting": self.proto.statistics_accounting(),
            "drain_semantics": dict(DRAIN_SEMANTICS),
            "drain_mission": self._drain_delta(snap0, snap1)}
        return self.termination

    def _drain_snapshot(self) -> dict | None:
        """Mission/energy counters of the training environment (None if the environment exposes none, e.g. test stubs)."""
        env = self.env
        en = getattr(env, "energy", None)
        if en is None or not hasattr(env, "accounting"):
            return None
        acc = env.accounting()
        return {"step_idx": int(getattr(env, "step_idx", 0)), "generated_mbit": float(acc["G"]), "delivered_mbit": float(acc["delivered"]),
                "lost_mbit": float(acc["lost"]), "E_prop_j": float(sum(en.prop)), "E_tx_j": float(sum(en.tx)),
                "E_rx_j": float(sum(en.rx)), "E_comp_j": float(sum(en.comp))}

    @staticmethod
    def _drain_delta(a: dict | None, b: dict | None) -> dict:
        if a is None or b is None:
            return {"available": False}
        d = {k: b[k] - a[k] for k in a}
        d.update(available=True, included_in_reported_training_energy=False, included_in_evaluation_energy=False,
                 note="accrued on the training environment after the last logged episode; evaluation resets fresh scenarios")
        return d

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
