#!/usr/bin/env python3
"""Compatibility import path for the shared HBM efficiency model."""

from __future__ import annotations

from pathlib import Path
import sys

_WORKSPACE_ROOT = Path(__file__).resolve().parents[5]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from analytical_backend.hbm_efficiency import (  # noqa: E402
    B1_HBM_EFFICIENCY_PCT,
    DEFAULT_PEAK_HBM_GBPS,
    OUTLIER_B_VALUES,
    hbm_efficiency_pct,
    hbm_gbps,
    relative_hbm_efficiency_vs_b1,
)

__all__ = [
    "B1_HBM_EFFICIENCY_PCT",
    "DEFAULT_PEAK_HBM_GBPS",
    "OUTLIER_B_VALUES",
    "hbm_efficiency_pct",
    "hbm_gbps",
    "relative_hbm_efficiency_vs_b1",
]
