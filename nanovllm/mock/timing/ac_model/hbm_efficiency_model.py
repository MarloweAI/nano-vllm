#!/usr/bin/env python3
"""Compatibility import path for the shared HBM efficiency model."""

from __future__ import annotations

from analytical_backend.hbm_efficiency import (
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
