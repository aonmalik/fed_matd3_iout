"""Baseline 'greedy_matd3' -- thin declaration; the definition lives in baselines/registry.py (single source of truth)."""
from .registry import METHODS

NAME = "greedy_matd3"
OVERRIDES = METHODS[NAME]["overrides"]
DESCRIPTION = METHODS[NAME]["desc"]
COMPARABILITY = METHODS[NAME]["note"]
