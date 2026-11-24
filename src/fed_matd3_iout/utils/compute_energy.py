"""Training compute-energy reporting (D9, provisional decision: compute energy is reported in JOULES, or marked ``not_measured``).

Rules (no invented numbers):
  * Energy comes ONLY from a real measurement: integrated power samples or the delta of a hardware energy counter.
  * No wattage/conversion factor is ever assumed here.  Without a usable instrument the status is ``not_measured`` and the value
    is ``None`` -- never 0.  Training compute TIME is still reported separately (it is measured, not converted).
  * Invalid readings (NaN/inf/negative) and gaps larger than ``max_gap_s`` make the covered interval *missing*; the report says how
    much time was covered/missing and downgrades the status to ``partial`` (never silently completes).
Boundary: the measured interval should be the gradient-update computation of the learner (same boundary for every method).  The
instrument available in this build environment is NONE (no RAPL powercap, no GPU power interface), so the pipeline reports
``not_measured`` here.  The modelled ``energy.e_grad_j`` charged by MATD3 trainers (an ASSUMPTION in configs/environment.yaml, not a
measurement) is a different quantity and is not mixed into these reports.
"""
from __future__ import annotations

import glob
import math
import time
from pathlib import Path

UNIT_TO_W = {"W": 1.0, "mW": 1e-3, "kW": 1e3}
UNIT_TO_J = {"J": 1.0, "mJ": 1e-3, "uJ": 1e-6, "kJ": 1e3, "Wh": 3600.0}
NOT_MEASURED = {"status": "not_measured", "energy_j": None, "covered_s": 0.0, "missing_s": None, "n_invalid": 0, "n_samples": 0}


def _valid(x) -> bool:
    try:
        return math.isfinite(float(x)) and float(x) >= 0.0
    except (TypeError, ValueError):
        return False


def integrate_power_joules(times_s, power, unit: str = "W", max_gap_s: float | None = None) -> dict:
    """Trapezoidal integration of power samples to joules.  Intervals touching an invalid sample, a non-increasing timestamp or a gap
    > ``max_gap_s`` are *missing* (not interpolated, not zero-filled).  Returns a report dict (see module docstring)."""
    if unit not in UNIT_TO_W:
        raise ValueError(f"unknown power unit {unit!r}; known: {sorted(UNIT_TO_W)}")
    t, p = list(times_s), list(power)
    if len(t) != len(p):
        raise ValueError("times and power must have the same length")
    n_invalid = sum(1 for x in p if not _valid(x)) + sum(1 for x in t if not math.isfinite(float(x)))
    if len(t) < 2:
        return {**NOT_MEASURED, "n_invalid": n_invalid, "n_samples": len(t)}
    e = covered = missing = 0.0
    for k in range(len(t) - 1):
        dt = float(t[k + 1]) - float(t[k])
        ok = (math.isfinite(dt) and dt > 0 and _valid(p[k]) and _valid(p[k + 1]) and (max_gap_s is None or dt <= max_gap_s))
        if ok:
            e += 0.5 * (float(p[k]) + float(p[k + 1])) * UNIT_TO_W[unit] * dt
            covered += dt
        elif math.isfinite(dt) and dt > 0:
            missing += dt
    if covered <= 0.0:
        return {**NOT_MEASURED, "missing_s": missing, "n_invalid": n_invalid, "n_samples": len(t)}
    return {"status": "measured" if missing == 0.0 and n_invalid == 0 else "partial", "energy_j": e, "covered_s": covered,
            "missing_s": missing, "n_invalid": n_invalid, "n_samples": len(t), "method": "trapezoid"}


def counter_delta_joules(start, end, unit: str = "uJ", wrap=None) -> dict:
    """Energy from a monotonic hardware energy counter (e.g. RAPL ``energy_uj``).  ``wrap`` = counter range for one wraparound."""
    if unit not in UNIT_TO_J:
        raise ValueError(f"unknown energy unit {unit!r}; known: {sorted(UNIT_TO_J)}")
    if not (_valid(start) and _valid(end)):
        return {**NOT_MEASURED, "n_invalid": 1}
    d = float(end) - float(start)
    if d < 0:
        if wrap is None or not _valid(wrap):
            return {**NOT_MEASURED, "n_invalid": 1}
        d += float(wrap)
    return {"status": "measured", "energy_j": d * UNIT_TO_J[unit], "covered_s": None, "missing_s": 0.0, "n_invalid": 0,
            "n_samples": 2, "method": "counter_delta"}


def rapl_counters() -> list[str]:
    return sorted(glob.glob("/sys/class/powercap/intel-rapl:[0-9]*/energy_uj"))


class ComputeMeter:
    """Accumulates the wall-clock and CPU time of the learner's update computation and, when a hardware energy counter is readable,
    its energy.  Without an instrument the energy stays ``not_measured`` (None)."""

    def __init__(self, counters: list[str] | None = None):
        self.paths = [p for p in (rapl_counters() if counters is None else counters) if self._readable(p)]
        self.wall_s = self.cpu_s = 0.0
        self.n_intervals = 0
        self._e = 0.0
        self._bad = 0

    @staticmethod
    def _readable(p) -> bool:
        try:
            float(Path(p).read_text())
            return True
        except Exception:
            return False

    def _read(self):
        try:
            return [float(Path(p).read_text()) for p in self.paths]
        except Exception:
            return None

    def __enter__(self):
        self._t0, self._c0, self._r0 = time.perf_counter(), time.process_time(), (self._read() if self.paths else None)
        return self

    def __exit__(self, *exc):
        self.wall_s += time.perf_counter() - self._t0
        self.cpu_s += time.process_time() - self._c0
        self.n_intervals += 1
        if self.paths:
            r1 = self._read()
            if self._r0 is None or r1 is None:
                self._bad += 1
            else:
                for a, b in zip(self._r0, r1):
                    d = counter_delta_joules(a, b, "uJ")
                    if d["status"] == "measured":
                        self._e += d["energy_j"]
                    else:
                        self._bad += 1
        return False

    def report(self) -> dict:
        base = {"compute_time_wall_s": self.wall_s, "compute_time_cpu_s": self.cpu_s, "n_update_intervals": self.n_intervals,
                "boundary": "learner gradient-update computation only (env stepping, evaluation and communication excluded)",
                "idle_power_subtracted": False}
        if not self.paths:
            return {**base, **NOT_MEASURED, "instrument": None,
                    "note": "no hardware energy counter available; energy NOT measured (not zero); no conversion from time was applied"}
        st = "measured" if self._bad == 0 else "partial"
        return {**base, "status": st, "energy_j": self._e if self.n_intervals else None, "instrument": "intel-rapl energy_uj (package counters)",
                "n_invalid": self._bad, "covered_s": None, "missing_s": None,
                "note": "whole-package counter: includes other processes on the host; idle power NOT subtracted"}
