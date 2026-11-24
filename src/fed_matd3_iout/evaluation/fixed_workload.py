"""Fixed-workload completion time (D10): a metric DISTINCT from ``backlog_clearance_time_s`` (which is left untouched).

Experiment
  * Workload: the initial buffer contents of every sensor in the evaluation scenario (``env.initial_fill`` x ``b_max`` units, packet
    categories drawn from the scenario's arrival stream).  Scenario k uses the same seeds for every method, so EVERY method receives
    the identical workload (size, per-sensor split, category composition).  ``W`` = ``env.accounting()['G']`` right after reset.
  * New arrivals are DISABLED (traffic.scale = 0) for this measurement; ``G`` is checked to stay equal to ``W`` (otherwise
    ``workload_violated``).
  * Task completion event (authoritative): a data unit is complete when it has been DELIVERED to the USV (``chunk.status ==
    'delivered'`` in ``MultiAUVIoUTEnv._uplink``; counted in ``accounting()['delivered']``).  Being collected, assigned, onboard, in
    transmission, or acknowledged does NOT complete a task.
  * Clock: starts at t = 0 of the evaluation episode (reset); stops at the end of the first step after which delivered >= W - tol.
    Resolution = one step (dt); the sub-step propagation delay of the final delivery is not added.  Includes sensor->AUV collection,
    AUV travel/hover, onboard queueing and the uplink to the USV (incl. airtime shared with model traffic); excludes nothing else.
  * Timeout: the episode horizon T*dt.  If the workload is not complete by then the observation is RIGHT-CENSORED: value = timeout,
    ``fixed_workload_censored = 1``.  Censored runs are kept, never dropped.
  * Failed/dropped tasks: with arrivals disabled no overflow can occur; if any unit is nevertheless lost (``lost > tol``) the workload
    can no longer be completed -> censored with reason ``failed_units``.
  * Zero workload (W = 0): completes at t = 0, uncensored, reason ``zero_workload`` (vacuous; flagged so it can be excluded).
  * Aggregation: per scenario -> mean over the matched scenarios within a training seed -> training seeds are the independent units
    (existing report machinery).  Because censored values equal the timeout, the mean is a LOWER bound of the true mean; always
    report ``fixed_workload_censored`` (censoring rate) and ``fixed_workload_delivered_fraction`` beside it.
  * Comparability: only valid between methods run under identical workload and measurement conditions (same scenarios, same dt,
    same timeout, arrivals off).  Methods differ in the declared non-equivalences (N1-N11) as for every other metric.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from ..environment.multi_auv_iout_env import MultiAUVIoUTEnv

TOL = 1e-9


class FixedWorkloadTracker:
    def __init__(self, workload_mbit: float, tol: float = TOL):
        self.W, self.tol = float(workload_mbit), float(tol)
        self.time_s: float | None = None
        self.reason: str | None = None
        self.delivered = 0.0
        self.violated = False
        if not math.isfinite(self.W) or self.W < 0:
            raise ValueError("workload must be finite and >= 0")
        if self.W <= self.tol:                                      # zero-task edge case
            self.time_s, self.reason = 0.0, "zero_workload"

    @property
    def done(self) -> bool:
        return self.reason is not None

    def update(self, t_end_s: float, delivered_mbit: float, lost_mbit: float = 0.0, generated_mbit: float | None = None) -> None:
        if generated_mbit is not None and generated_mbit > self.W + 1e-6 * max(self.W, 1.0):
            self.violated = True                                    # arrivals were not disabled: the workload is not fixed
        if self.done:
            return
        self.delivered = float(delivered_mbit)
        if lost_mbit > self.tol:
            self.reason = "failed_units"                            # units lost -> the workload can never complete
        elif self.delivered >= self.W - self.tol * max(self.W, 1.0):
            self.time_s, self.reason = float(t_end_s), "completed"

    def result(self, timeout_s: float) -> dict:
        completed = self.reason in ("completed", "zero_workload")
        reason = self.reason or "timeout"
        frac = 1.0 if self.W <= self.tol else min(1.0, self.delivered / self.W)
        return {"fixed_workload_completion_time_s": float(self.time_s) if completed else float(timeout_s),
                "fixed_workload_censored": 0.0 if completed else 1.0,
                "fixed_workload_delivered_fraction": 1.0 if completed else frac,
                "fixed_workload_mbit": self.W, "fixed_workload_violated": float(self.violated),
                "fixed_workload_reason": reason}


def fixed_workload_cfg(cfg):
    c = copy.deepcopy(cfg)
    c["traffic"]["scale"] = 0.0                                     # no new arrivals during the measurement
    return c


def rollout_fixed_workload(cfg, policies, scenario: int, env: MultiAUVIoUTEnv | None = None) -> dict:
    env = env or MultiAUVIoUTEnv(fixed_workload_cfg(cfg))
    obs, _ = env.reset(int(cfg["seeds"]["eval_scenario_base"]) + int(scenario))
    trk = FixedWorkloadTracker(env.accounting()["G"])
    for t in range(env.T):
        if trk.done:
            break
        act = np.stack([policies[i](obs[i]) for i in range(env.M)])
        obs, _, _, trunc, _ = env.step(act)
        acc = env.accounting()
        trk.update((t + 1) * env.dt, acc["delivered"], acc["lost"], acc["G"])
        if trunc:
            break
    return trk.result(env.T * env.dt)


def evaluate_fixed_workload(cfg, policies, n_scenarios: int | None = None) -> list[dict]:
    n = int(n_scenarios or cfg["experiment"]["eval_scenarios"])
    env = MultiAUVIoUTEnv(fixed_workload_cfg(cfg))
    return [rollout_fixed_workload(cfg, policies, k, env) for k in range(n)]
