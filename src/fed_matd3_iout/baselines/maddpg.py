"""Baseline 'maddpg' -- thin declaration; the definition lives in baselines/registry.py (single source of truth)."""
from .registry import METHODS

NAME = "maddpg"
OVERRIDES = METHODS[NAME]["overrides"]
DESCRIPTION = METHODS[NAME]["desc"]
COMPARABILITY = METHODS[NAME]["note"]
