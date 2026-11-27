"""Small deterministic fixtures: a fake environment surface, tiny clients, and an instrumented harness with global invariants."""
from __future__ import annotations

import copy

import numpy as np
import torch

from ..agents.actor import flatten_params
from ..agents.matd3_agent import MATD3Agent
from ..agents.replay_buffer import ClientReplay, JointStore
from ..communication.packet_delivery import LinkState
from ..federated.communication_protocol import FederatedProtocol
from ..federated.federated_client import FederatedClient
from ..federated.usv_server import USVServer

KEYS = ("coll_mbit", "owned_loss_mbit", "steps", "U_mature", "V_mature", "units_del_mature", "units_coll_mature",
        "n_mature_batches")
ACC = ("coll_mbit", "owned_loss_mbit", "V_mature", "U_mature", "units_del_mature", "units_coll_mature", "n_mature_batches")              # keys whose conservation is checked


def link(up=True, success=True, rate=1.0, prop=0.0):
    return LinkState(in_range=up, outage=not up, up=up, success=success, dist_m=10.0, rate_mbps=rate,
                     p_success=1.0 if success else 0.0, prop_delay_s=prop)


class FakeEnv:
    """Exactly the environment surface the protocol uses, with the real episode-guard semantics, plus conservation bookkeeping."""

    def __init__(self, n):
        self.M = n
        self.links = [link() for _ in range(n)]
        self.episode_id = 1
        self.win = {k: np.zeros(n) for k in KEYS}
        self.model_traffic_active = [False] * n
        self.model_uplink_active = [False] * n
        self.sensor_uplink_airtime_s = [0.0] * n
        self.stats_refused_episode = 0
        self.stats_returned = 0
        self.injected = {k: 0.0 for k in KEYS}
        self.dropped = {k: 0.0 for k in KEYS}          # window content discarded by an episode reset

    def inject(self, i, **kv):
        for k, v in kv.items():
            self.win[k][i] += v
            self.injected[k] += v

    def pop_window_stats(self, i):
        out = {k: float(v[i]) for k, v in self.win.items()}
        for v in self.win.values():
            v[i] = 0.0
        return out

    def return_window_stats(self, i, stats, episode_id):
        if int(episode_id) != self.episode_id:
            self.stats_refused_episode += 1
            return False
        for k in KEYS:
            if k in stats:
                self.win[k][i] += float(stats[k])
        self.stats_returned += 1
        return True

    def new_episode(self):
        for k, v in self.win.items():
            self.dropped[k] += float(v.sum())
        self.episode_id += 1
        self.win = {k: np.zeros(self.M) for k in KEYS}

    def add_comm_energy(self, i, tx, rx):
        pass


def make_clients(cfg, n):
    cfg.matd3.hidden = 8
    cs = []
    for i in range(n):
        torch.manual_seed(0)                                    # identical initial actors
        ag = MATD3Agent(i, 2, 4, 2, 6, cfg.matd3, "cpu")
        st = JointStore(10, 6, 2, 4, 2)
        cs.append(FederatedClient(i, ag, ClientReplay(st, 10, np.random.default_rng(0)), cfg.federated, 16, 1.0))
    return cs


def perturb(client, delta=0.1):
    with torch.no_grad():
        for p in client.agent.actor.parameters():
            p.add_(delta)


def ready(c, steps=5):
    c.steps_since, c.trans_since = steps, steps


class Harness:
    """Federation + fake env + recorders. ``check_invariants`` evaluates the global correctness conditions."""

    def __init__(self, cfg, n=2, mode="async", copy_cfg=True, **fed):
        self.cfg = copy.deepcopy(cfg) if copy_cfg else cfg
        f = self.cfg.federated
        f.update(dict(mode=mode, local_steps_per_update=1, min_local_transitions=0, agg_buffer_size=1, agg_timeout_s=1e9,
                      sync_timeout_s=1e9, upload_timeout_s=1e9, max_buffer_age_s=1e12, clip_norm=None, max_client_weight=1.0,
                      allow_fragmentation=True))
        f.update(fed)
        self.clients = make_clients(self.cfg, n)
        theta0 = flatten_params(self.clients[0].agent.actor)
        for c in self.clients:
            c.reference = theta0.clone()
        self.server = USVServer(theta0, f, n, True)
        self.env = FakeEnv(n)
        self.proto = FederatedProtocol(self.clients, self.server, f, self.cfg["acoustic"]["usv_link"], 1.0)
        self.n = n
        self.t = 0
        self.snapshots: dict[str, torch.Tensor] = {}
        self.pkts: dict[str, object] = {}
        self.thetas = {0: theta0.clone()}
        self.theta_before: dict[int, torch.Tensor] = {}
        self.deltas: dict[str, torch.Tensor] = {}
        self._wrap()
        self.model_bytes = self.proto.model_bytes

    def _wrap(self):
        for c in self.clients:
            orig = c.make_update
            def mk(t, ws, ep=-1, _o=orig):
                p = _o(t, ws, ep)
                self.snapshots[p.uid] = p.snapshot.clone()
                self.deltas[p.uid] = p.delta.clone()
                self.pkts[p.uid] = p
                return p
            c.make_update = mk
        agg = self.server.aggregate_now
        def aggn(t, _a=agg):
            before = self.server.theta.clone()
            v0 = self.server.version
            r = _a(t)
            if r is not None:
                self.theta_before[self.server.version] = before
                self.thetas[self.server.version] = self.server.theta.clone()
            return r
        self.server.aggregate_now = aggn

    def hold(self, on=True):
        """Prevent (on) / allow (off) further updates by raising/restoring the local-step threshold; counters are NOT modified."""
        self.cfg.federated["local_steps_per_update"] = 10 ** 6 if on else 1

    def rate_for_steps(self, k):
        """Link rate [Mbit/s] such that an upload needs about k steps."""
        share = float(self.cfg["acoustic"]["usv_link"]["model_share"])
        return self.model_bytes * 8 / 1e6 / (share * k)

    def set_links(self, **kw):
        for i in range(self.n):
            self.env.links[i] = link(**kw)

    def step(self):
        self.proto.step(self.env, float(self.t))
        self.t += 1

    def run(self, n):
        for _ in range(n):
            self.step()

    def run_until(self, cond, limit=500):
        for _ in range(limit):
            if cond():
                return True
            self.step()
        return cond()

    # ------------------------------------------------------------------ invariants
    def allowed_references(self, c):
        out = [self.thetas[0]] + [self.snapshots[u] for u, p in self.pkts.items()
                                  if p.client_id == c.id and p.status.value == "aggregated"]
        out += [self.thetas[v] for v in self.thetas]
        return out

    def check_invariants(self) -> dict:
        r: dict[str, bool] = {}
        p, s, env = self.proto, self.server, self.env
        acc = p.statistics_accounting()
        r["update_accounting_balanced"] = bool(acc["balanced"])
        uids = [u["uid"] for u in p.update_log]
        r["each_update_settled_at_most_once"] = len(uids) == len(set(uids))
        r["update_ids_unique"] = len(self.pkts) == len(set(self.pkts))
        r["at_most_one_outstanding_per_client"] = all(
            sum(1 for q in self.pkts.values() if q.client_id == c.id and not q.client_resolved
                and q.status.value in ("uploading", "in_flight", "buffered") or
                (q.client_id == c.id and q.terminal and not q.client_resolved)) <= 1 for c in self.clients)
        r["all_parameters_finite"] = bool(torch.isfinite(s.theta).all()) and all(
            bool(torch.isfinite(c.reference).all()) and all(bool(torch.isfinite(x).all()) for x in c.agent.actor.parameters())
            for c in self.clients)
        r["reference_is_committed_or_global_never_rejected"] = all(
            any(torch.equal(c.reference, a) for a in self.allowed_references(c)) for c in self.clients)
        noncommit = [q for q in self.pkts.values() if q.status.value in ("rejected", "expired")]
        r["no_reference_equals_noncommitted_snapshot"] = all(
            not torch.equal(self.clients[q.client_id].reference, self.snapshots[q.uid]) for q in noncommit)
        r["version_equals_rounds"] = s.version == len(s.history)
        held = {k: sum(q.stats.get(k, 0.0) for q in self.pkts.values() if q.stats_disposition == "held" and q.status.value
                       != "created") for k in ACC}
        for k in ACC:
            have = float(env.win[k].sum()) + held[k] + p.stats_mass["consumed"][k] + p.stats_mass["discarded_episode"][k] + env.dropped[k]
            r[f"stats_conserved[{k}]"] = abs(have - env.injected[k]) < 1e-9
        return r
