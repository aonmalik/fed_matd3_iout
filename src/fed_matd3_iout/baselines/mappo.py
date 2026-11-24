"""MAPPO baseline (genuine on-policy PPO, NOT MATD3-derived).

Shared-parameter TANH-SQUASHED Gaussian actor (Protocol 2.2): latent u ~ N(mu(o), sigma^2) (state-independent log-std),
executed action a = tanh(u) in (-1, 1).  The PPO likelihood is the likelihood of the EXECUTED action:
    log pi(a|o) = log N(u; mu, sigma) - sum_d log(1 - tanh(u_d)^2)        (change of variables; Jacobian of tanh)
evaluated with the numerically stable identity  log(1 - tanh(u)^2) = 2 (log 2 - u - softplus(-2u)).
The rollout buffer stores the LATENT u (not the possibly saturated a), so the log-probability can be recomputed exactly in
every PPO epoch.  (Protocol 2.1 sampled an unbounded Gaussian, stored its log-prob, and executed clip(a, -1, 1): the stored
likelihood described a different action than the one executed whenever clipping happened.)
Entropy bonus: entropy of the latent Gaussian (the exact entropy of the squashed distribution has no closed form); this is
a regulariser on the latent spread, documented as such.
Centralised value V(s, onehot(i)),
GAE(gamma, lambda), clipped surrogate, entropy bonus.  Same environment, reward and experience budget
(train_episodes x episode_steps) as the off-policy methods.  Not mathematically comparable in sample efficiency:
on-policy data is used for `epochs` passes and then discarded; there is no replay and no federation.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..environment.multi_auv_iout_env import MultiAUVIoUTEnv
from ..training.checkpoint import save_checkpoint
from ..utils.compute_energy import ComputeMeter
from ..utils.logger import RunLogger
from ..utils.seed import make_rng, seed_everything

NAME = "mappo"


LOG_RATIO_CLAMP = 20.0       # numerical safeguard only: exp(+-20) is finite in float32; never active for sane ratios


def tanh_log_det(u):
    """log |d tanh(u)/du| = log(1 - tanh(u)^2), stable for any |u| (no epsilon, no clipping)."""
    return 2.0 * (math.log(2.0) - u - torch.nn.functional.softplus(-2.0 * u))


class GaussianActor(nn.Module):
    """Tanh-squashed Gaussian policy.  ``mu`` outputs the LATENT mean (linear head); actions are tanh(latent)."""

    def __init__(self, obs_dim, act_dim, hidden, init_log_std):
        super().__init__()
        self.mu = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                nn.Linear(hidden, act_dim))
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))

    def dist(self, o):
        """Distribution of the LATENT variable u."""
        return torch.distributions.Normal(self.mu(o), self.log_std.exp().expand(o.shape[0], -1))

    def log_prob_latent(self, o, u):
        """log pi(tanh(u)|o) = log N(u) - sum log(1 - tanh(u)^2); shape (N,)."""
        return (self.dist(o).log_prob(u) - tanh_log_det(u)).sum(-1)

    def sample(self, o):
        """Returns (action in [-1,1], latent u, log pi(action|o))."""
        d = self.dist(o)
        u = d.sample()
        return torch.tanh(u), u, (d.log_prob(u) - tanh_log_det(u)).sum(-1)

    def mean_action(self, o):
        return torch.tanh(self.mu(o))


class ValueNet(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.v = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.v(x).squeeze(-1)


class MAPPOTrainer:
    def __init__(self, cfg, device, run_dir=None, train_seed=0, logger=None):
        self.cfg, self.device, self.seed = cfg, device, int(train_seed)
        seed_everything(self.seed)
        self.env = MultiAUVIoUTEnv(cfg)
        e, p, h = self.env, cfg["matd3"]["ppo"], int(cfg["matd3"]["hidden"])
        self.p = p
        self.meter = ComputeMeter()
        self.actor = GaussianActor(e.obs_dim, e.act_dim, h, p["init_log_std"]).to(device)
        self.value = ValueNet(e.state_dim + e.M, h).to(device)
        self.opt = torch.optim.Adam(list(self.actor.parameters()) + list(self.value.parameters()), lr=float(p["lr"]))
        self.rng = make_rng(self.seed, "policy_noise")
        self.run_dir = Path(run_dir) if run_dir else None
        self.log = logger or RunLogger(self.run_dir, verbose=False)
        self.curve, self.server, self.proto = [], None, None
        self.eye = torch.eye(e.M, device=device)

    def _v(self, states):            # states: (T, sd) -> (T, M)
        T = states.shape[0]
        x = torch.cat([states.unsqueeze(1).expand(T, self.env.M, -1), self.eye.unsqueeze(0).expand(T, -1, -1)], -1)
        return self.value(x)

    def _update(self, buf):
        g, lam = float(self.cfg["matd3"]["gamma"]), float(self.p["gae_lambda"])
        S = torch.as_tensor(np.array(buf["s"]), device=self.device)
        S2 = torch.as_tensor(np.array(buf["s2"]), device=self.device)
        O = torch.as_tensor(np.array(buf["o"]), device=self.device)
        A = torch.as_tensor(np.array(buf["u"]), device=self.device)          # LATENT samples (see module docstring)
        LP = torch.as_tensor(np.array(buf["lp"]), device=self.device)
        R = torch.as_tensor(np.array(buf["r"]), device=self.device)
        last = torch.as_tensor(np.array(buf["last"]), device=self.device, dtype=torch.float32)
        with torch.no_grad():
            V, V2 = self._v(S), self._v(S2)
            delta = R + g * V2 - V
            adv = torch.zeros_like(delta)
            run = torch.zeros(self.env.M, device=self.device)
            for t in reversed(range(len(delta))):
                run = delta[t] + g * lam * (1.0 - last[t]) * run
                adv[t] = run
            ret = adv + V
            advn = (adv - adv.mean()) / (adv.std() + 1e-8)
        T, M = O.shape[:2]
        info = {}
        for _ in range(int(self.p["epochs"])):
            idx = torch.randperm(T, device=self.device)
            for mb in idx.split(max(T // 4, 1)):
                d = self.actor.dist(O[mb].reshape(-1, O.shape[-1]))
                lp = self.actor.log_prob_latent(O[mb].reshape(-1, O.shape[-1]), A[mb].reshape(-1, A.shape[-1])).reshape(len(mb), M)
                ratio = (lp - LP[mb]).clamp(-LOG_RATIO_CLAMP, LOG_RATIO_CLAMP).exp()
                c = float(self.p["clip"])
                pl = -torch.min(ratio * advn[mb], ratio.clamp(1 - c, 1 + c) * advn[mb]).mean()
                vl = ((self._v(S[mb]) - ret[mb]) ** 2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pl + 0.5 * vl - float(self.p["entropy_coef"]) * ent
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.value.parameters()), 10.0)
                self.opt.step()
                info = {"policy_loss": float(pl.detach()), "value_loss": float(vl.detach())}
        return info

    def compute_report(self) -> dict:
        """Training compute time / energy of the PPO updates (D9): energy is measured or ``not_measured`` -- never a substituted 0."""
        return self.meter.report()

    def train(self, n_episodes=None, eval_fn=None, eval_every=0, ckpt_every=0):
        cfg, env = self.cfg, self.env
        n_ep = int(n_episodes or cfg["experiment"]["train_episodes"])
        buf = {k: [] for k in ("s", "s2", "o", "a", "u", "lp", "r", "last")}
        for ep in range(n_ep):
            obs, info = env.reset(self.seed * 10_000 + ep)
            state, ret = info["state"], np.zeros(env.M)
            for k in range(env.T):
                with torch.no_grad():
                    a, u, lp = self.actor.sample(torch.as_tensor(obs, device=self.device))
                obs2, r, _, trunc, info = env.step(a.cpu().numpy())          # tanh output: already in [-1, 1], NOT clipped
                for key, val in (("s", state), ("s2", info["state"]), ("o", obs), ("a", a.cpu().numpy()), ("u", u.cpu().numpy()),
                                 ("lp", lp.cpu().numpy()), ("r", r), ("last", float(trunc or k == env.T - 1))):
                    buf[key].append(val)
                ret += r
                obs, state = obs2, info["state"]
                if len(buf["s"]) >= int(self.p["rollout_steps"]):
                    with self.meter:
                        self._update(buf)
                    buf = {k_: [] for k_ in buf}
                if trunc:
                    break
            acc = env.accounting()
            rec = {"episode": ep, "return_mean": float(ret.mean()), "U": env.ledger.U, "delivered_mbit": acc["delivered"],
                   "lost_mbit": acc["lost"], "global_version": 0}
            self.curve.append(rec)
            self.log.log("train", rec)
            if eval_fn and eval_every and (ep + 1) % eval_every == 0:
                self.curve[-1]["eval"] = eval_fn(self)
            if ckpt_every and self.run_dir and (ep + 1) % ckpt_every == 0:
                self.save(f"ep{ep + 1}")
        return self.curve

    def local_policies(self):
        def pol(o):
            with torch.no_grad():
                return self.actor.mean_action(torch.as_tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0))[0].cpu().numpy()
        return [pol for _ in range(self.env.M)]

    global_policies = local_policies

    def save(self, tag):
        return save_checkpoint(self.run_dir / "checkpoints", tag, {"actor": self.actor.state_dict(),
                                                                   "value": self.value.state_dict(), "curve": self.curve})
