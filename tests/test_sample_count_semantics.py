"""Protocol 2.2: what the aggregation sample count n_i means (documented in docs/ASSUMPTIONS_AND_LIMITATIONS.md, section S).

n_i = (gradient steps since the client's reference) x (minibatch size) = REPLAY-SAMPLE PRESENTATIONS that shaped Delta_i.
It is neither the number of distinct transitions nor the replay-buffer size.  With a common minibatch size it is proportional to
the number of gradient steps, so FedAvg weights are proportional to local gradient steps (compute actually spent on Delta_i).
"""
from types import SimpleNamespace

import numpy as np
import pytest

from fed_fixtures import Harness, link, perturb
from fed_matd3_iout.federated.aggregation import normalise_with_cap, raw_weights
from fed_matd3_iout.training.local_training import local_update

BATCH = 16                                                       # the Harness clients use batch_size 16


class _Replay(list):
    def sample(self, b, dev):
        return ("batch", b)


def _stub_client(speed):
    steps = []
    return SimpleNamespace(id=0, speed=speed, credit=0.0, replay=_Replay(range(1000)), steps_since=0, total_steps=0,
                           agent=SimpleNamespace(update=lambda batch, tgt: steps.append(batch))), steps


def _run(c, n=200):
    env = SimpleNamespace(add_compute_energy=lambda i, j: None)
    for _ in range(n):
        local_update(c, [], "cpu", warmup=1, batch_size=BATCH, e_grad_j=0.0, env=env)


def test_gradient_steps_scale_with_client_compute_speed():
    fast, _ = _stub_client(1.0)
    slow, _ = _stub_client(0.5)
    _run(fast); _run(slow)
    assert fast.steps_since == 200 and slow.steps_since == 100


def test_packet_n_samples_equals_gradient_steps_times_batch_and_is_not_transition_count(cfg):
    h = Harness(cfg, 2, "async")
    h.hold(False)
    c = h.clients[0]
    perturb(c)
    c.steps_since, c.trans_since = 7, 40                         # 7 gradient steps, 40 distinct transitions seen
    h.env.links[0] = link(rate=h.rate_for_steps(50))
    h.step()
    p = c.outstanding
    assert p.local_steps == 7 and p.n_transitions == 40
    assert p.n_samples == 7 * BATCH != p.n_transitions           # presentations, NOT unique transitions


def test_fedavg_weights_are_proportional_to_local_gradient_steps_for_unequal_clients(cfg):
    h = Harness(cfg, 2, "async", agg_buffer_size=2, aggregation="fedavg", kappa=0.9, eta=5.0)
    h.hold(False)
    for i, steps in enumerate((3, 9)):
        perturb(h.clients[i], 0.1 + 0.05 * i)
        h.clients[i].steps_since = h.clients[i].trans_since = steps
        h.env.links[i] = link(rate=h.rate_for_steps(2))
    h.run_until(lambda: h.server.version == 1)
    rec = h.server.history[0]
    w = dict(zip(rec["clients"], rec["w"]))
    assert w[0] == pytest.approx(3 / 12) and w[1] == pytest.approx(9 / 12)       # n_i-proportional; staleness/relevance ignored


def test_fedavg_weight_ignores_batch_size_when_common_to_all_clients():
    n_small, n_big = np.array([3, 9]) * 16.0, np.array([3, 9]) * 64.0
    w1 = normalise_with_cap(raw_weights(n_small, [1, 1], [0, 0], 1.0, 0.0, 0.1))
    w2 = normalise_with_cap(raw_weights(n_big, [1, 1], [0, 0], 1.0, 0.0, 0.1))
    assert np.allclose(w1, w2) and np.allclose(w1, [0.25, 0.75])


def test_repeated_presentations_count_each_time_not_unique_samples():
    """Same replay content, twice the gradient steps => twice the weight (the quantity counts presentations)."""
    w = normalise_with_cap(raw_weights([10 * BATCH, 20 * BATCH], [1, 1], [1, 1], 1.0, 0.0, 0.1))
    assert w[1] == pytest.approx(2 * w[0])


def test_rejected_update_keeps_progress_so_next_n_samples_covers_all_steps_since_reference(cfg):
    """Steps are only subtracted on COMMIT, so a retry after a rejection reports the cumulative steps behind its delta."""
    h = Harness(cfg, 2, "async")
    h.hold(False)
    c = h.clients[0]
    perturb(c)
    c.steps_since = c.trans_since = 5
    h.env.links[0] = link(rate=h.rate_for_steps(2))
    h.run_until(lambda: c.outstanding is not None)
    p = c.outstanding
    assert p.n_samples == 5 * BATCH and c.steps_since == 5       # steps are NOT consumed by creating/transmitting the update
