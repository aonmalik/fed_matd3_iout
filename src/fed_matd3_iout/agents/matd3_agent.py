"""One MATD3 agent (actor, twin critics, target nets) -- Eqs. M1-M6.

  a_i       = mu_i(o_i)                                                         (M1)
  a'_k      = clip(mu'_k(o'_k) + clip(eps, -c, c), -1, 1)   for ALL agents k    (M2)  target-policy smoothing
  y_i       = r_i + gamma (1-d) min_k Q'_{i,k}(s', a'_1..a'_M)                  (M3)  clipped double-Q
  L_{Q_ik}  = E[(Q_ik(s, a_1..a_M) - y_i)^2]                                    (M4)
  grad_th J = E[ grad_a Q_{i,1}(s, a_1..mu_i(o_i)..a_M) grad_th mu_i(o_i) ]     (M5)  DPG, other agents' a from replay
  th'      <- tau th + (1-tau) th'    after every `policy_delay`-th critic update (M6)  Polyak

critic_mode='centralized': Q input = [s, joint action] (CTDE; simulator state used at training time only).
critic_mode='independent': Q input = [o_i, a_i]  (Independent TD3 baseline; no other-agent information).
twin=False + policy_delay=1 + target_noise=0 reproduces MADDPG (single critic, no TD3 stabilisers).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from .actor import Actor
from .critic import TwinCritic


def polyak_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.mul_(1.0 - tau).add_(sp, alpha=tau)


class MATD3Agent:
    def __init__(self, idx: int, n_agents: int, obs_dim: int, act_dim: int, state_dim: int, mcfg, device):
        self.i, self.M, self.obs_dim, self.act_dim, self.state_dim = idx, n_agents, obs_dim, act_dim, state_dim
        self.cfg = mcfg
        self.device = device
        self.mode = mcfg["critic_mode"]
        self.twin = bool(mcfg["twin"])
        h = int(mcfg["hidden"])
        self.actor = Actor(obs_dim, act_dim, h).to(device)
        self.actor_t = copy.deepcopy(self.actor)
        in_dim = (state_dim + n_agents * act_dim) if self.mode == "centralized" else (obs_dim + act_dim)
        self.critic = TwinCritic(in_dim, h, self.twin).to(device)
        self.critic_t = copy.deepcopy(self.critic)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=float(mcfg["actor_lr"]))
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=float(mcfg["critic_lr"]))
        self.critic_updates = 0
        self.actor_updates = 0
        self.last_info: dict = {}

    # -------------------------------------------------------------- acting
    @torch.no_grad()
    def act(self, obs: np.ndarray, noise_std: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
        a = self.actor(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))[0].cpu().numpy()
        if noise_std > 0:
            a = a + rng.normal(0.0, noise_std, size=a.shape)
        return np.clip(a, -1.0, 1.0)

    # -------------------------------------------------------------- critic input
    def critic_input(self, s, o_i, joint_a) -> torch.Tensor:
        if self.mode == "centralized":
            return torch.cat([s, joint_a.reshape(joint_a.shape[0], -1)], dim=-1)
        return torch.cat([o_i, joint_a[:, self.i, :]], dim=-1)

    # -------------------------------------------------------------- targets
    @torch.no_grad()
    def target_joint_action(self, o2: torch.Tensor, target_actors: list, noise: torch.Tensor | None = None) -> torch.Tensor:
        """(M2): uses ALL target actors (or this agent's own target actor if critic_other_actors == 'own')."""
        c = float(self.cfg["noise_clip"])
        sig = float(self.cfg["target_noise"])
        acts = []
        for k in range(self.M):
            actor_k = target_actors[k] if (self.cfg["critic_other_actors"] == "peers" or k == self.i) else self.actor_t
            mu = actor_k(o2[:, k, :])
            eps = (torch.randn_like(mu) * sig) if noise is None else noise[:, k, :]
            acts.append(torch.clamp(mu + torch.clamp(eps, -c, c), -1.0, 1.0))
        return torch.stack(acts, dim=1)

    @torch.no_grad()
    def compute_target(self, batch: dict, target_actors: list, noise: torch.Tensor | None = None) -> torch.Tensor:
        """(M3). Terminal transitions (d=1) do not bootstrap."""
        a2 = self.target_joint_action(batch["o2"], target_actors, noise)
        x2 = self.critic_input(batch["s2"], batch["o2"][:, self.i, :], a2)
        q1, q2 = self.critic_t(x2)
        qmin = torch.min(q1, q2) if self.twin else q1
        return batch["r"][:, self.i] + float(self.cfg["gamma"]) * (1.0 - batch["d"]) * qmin

    # -------------------------------------------------------------- update
    def update(self, batch: dict, agents_actor_t: list, all_actors: list | None = None, noise=None) -> dict:
        """One critic update; every `policy_delay`-th call also updates the actor and Polyak-averages targets."""
        y = self.compute_target(batch, agents_actor_t, noise)
        x = self.critic_input(batch["s"], batch["o"][:, self.i, :], batch["a"])
        q1, q2 = self.critic(x)
        loss_c = F.mse_loss(q1, y) + (F.mse_loss(q2, y) if self.twin else 0.0)
        self.opt_c.zero_grad(set_to_none=True)
        loss_c.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), float(self.cfg["grad_clip"]))
        self.opt_c.step()
        self.critic_updates += 1
        info = {"critic_loss": float(loss_c.detach()), "q_mean": float(q1.mean().detach()), "actor_loss": None}
        if self.critic_updates % int(self.cfg["policy_delay"]) == 0:
            a_i = self.actor(batch["o"][:, self.i, :])
            ja = batch["a"].clone()
            ja[:, self.i, :] = a_i
            loss_a = -self.critic.q1_only(self.critic_input(batch["s"], batch["o"][:, self.i, :], ja)).mean()
            self.opt_a.zero_grad(set_to_none=True)
            loss_a.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float(self.cfg["grad_clip"]))
            self.opt_a.step()
            tau = float(self.cfg["tau"])
            polyak_update(self.actor_t, self.actor, tau)
            polyak_update(self.critic_t, self.critic, tau)
            self.actor_updates += 1
            info["actor_loss"] = float(loss_a.detach())
        self.last_info = info
        return info
