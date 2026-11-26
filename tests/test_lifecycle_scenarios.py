"""Final validation item 2: deterministic adverse-condition scenarios + mutation checks proving the scenarios can fail."""
import pytest
import torch

from fed_matd3_iout.federated.federated_client import FederatedClient
from fed_matd3_iout.federated.usv_server import USVServer
from fed_matd3_iout.federated.update_packet import UpdateStatus
from fed_matd3_iout.validation import lifecycle_scenarios as L


@pytest.mark.parametrize("name", list(L.SCENARIOS))
def test_scenario_passes(cfg, name):
    r = L.SCENARIOS[name](cfg)
    failed = [k for k, v in r["checks"].items() if not v]
    assert not failed, f"{name}: {failed}"


def test_scenarios_are_deterministic(cfg):
    a = {k: v["checks"] for k, v in L.run_all(cfg).items()}
    b = {k: v["checks"] for k, v in L.run_all(cfg).items()}
    assert a == b


def detected(fn, cfg) -> bool:
    """A mutation is detected if the scenario reports a failed check OR the state machine itself raises."""
    try:
        return not fn(cfg)["passed"]
    except Exception:
        return True


# ---------------------------------------------------------------- mutations: re-introduce the audited bugs, scenarios MUST notice
def test_mutation_receipt_advances_reference_is_detected(cfg, monkeypatch):
    orig = FederatedClient.handle_control

    def bad(self, msg):
        if msg.kind == "receipt" and self.outstanding is not None:
            self.reference = self.outstanding.snapshot.clone()          # the audited bug (C): ACK advances the reference
        return orig(self, msg)
    monkeypatch.setattr(FederatedClient, "handle_control", bad)
    assert detected(L.client_reconnection, cfg)


def test_mutation_reject_advances_reference_is_detected(cfg, monkeypatch):
    orig = FederatedClient.handle_control

    def bad(self, msg):
        if msg.kind == "reject" and self.outstanding is not None:
            self.reference = self.outstanding.snapshot.clone()          # the audited bug (D)
        return orig(self, msg)
    monkeypatch.setattr(FederatedClient, "handle_control", bad)
    assert detected(L.server_rejection, cfg)


def test_mutation_no_duplicate_guard_is_detected(cfg, monkeypatch):
    def bad(self, pkt, t):                                               # accepts every delivery
        self.n_received += 1
        pkt.t_received = t
        self.buffer.append(pkt)
        return True
    monkeypatch.setattr(USVServer, "receive", bad)
    assert detected(L.duplicate_delivery, cfg)


def test_mutation_cross_episode_restore_is_detected(cfg, monkeypatch):
    from fed_matd3_iout.validation import fixtures as F
    monkeypatch.setattr(F.FakeEnv, "return_window_stats",
                        lambda self, i, stats, ep: [self.win[k].__setitem__(i, self.win[k][i] + stats.get(k, 0.0)) for k in F.KEYS] and True)
    assert detected(L.episode_transition, cfg)


# ---------------------------------------------------------------- real Trainer: updates that straddle episode boundaries
@pytest.mark.parametrize("bpp,expect", [(100, "aggregated_across_episodes"), (300, "discarded_across_episodes")])
def test_real_training_episode_transitions(smoke_cfg, bpp, expect):
    from fed_matd3_iout.baselines.registry import build_cfg
    from fed_matd3_iout.training.trainer import Trainer
    cfg = build_cfg(smoke_cfg, "s3a_fedmatd3")
    cfg.federated.model_bytes_per_param = bpp                            # slow transfers: 1-3 episodes per upload
    cfg.federated.upload_timeout_s = 60.0
    cfg.federated.max_buffer_age_s = 70.0
    t = Trainer(cfg, torch.device("cpu"), None, 0)
    t.train(8)
    acc, s = t.proto.statistics_accounting(), t.proto.summary()
    T = cfg.env.episode_steps * cfg.env.dt_s
    cross = [u for u in t.proto.update_log if int(u["t_end"] // T) != int(u["t_created"] // T)]
    assert acc["balanced"] and cross
    assert all(torch.isfinite(c.reference).all() and all(torch.isfinite(x).all() for x in c.agent.actor.parameters())
               for c in t.clients)
    assert torch.isfinite(t.server.theta).all()
    if expect == "aggregated_across_episodes":
        assert any(u["status"] == "aggregated" for u in cross) and s["stats_consumed"] == s["aggregated_updates"]
    else:
        assert s["stats_discarded_episode"] == len(cross) > 0 and s["stats_returned"] == 0   # never restored into another episode
