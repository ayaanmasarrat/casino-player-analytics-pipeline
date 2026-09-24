"""
casino_analytics.segmentation
================================
Two-level behavioural segmentation of players.

L1 — Value/Engagement Tier  (based on Average Monthly Value)
    High   : AMV >= SETTINGS.l1_high_amv
    Medium : SETTINGS.l1_med_amv <= AMV < l1_high_amv
    Low    : AMV < SETTINGS.l1_med_amv

L2 — Activity Tier  (based on distinct visit days in the observation period)
    Frequent   : n_visit_days >= SETTINGS.l2_frequent_days
    Occasional : SETTINGS.l2_occasional_days <= n_visit_days < l2_frequent_days
    Lapsed     : n_visit_days < SETTINGS.l2_occasional_days
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from casino_analytics.config import SETTINGS


def assign_l1_segment(player: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``l1_segment`` column (High / Medium / Low) based on ``amv``.

    Fully vectorised: uses ``np.select`` over the ``amv`` Series.
    """
    df = player.copy()
    conditions = [
        df["amv"] >= SETTINGS.l1_high_amv,
        df["amv"] >= SETTINGS.l1_med_amv,
    ]
    choices = ["High", "Medium"]
    df["l1_segment"] = pd.Categorical(
        np.select(conditions, choices, default="Low"),
        categories=["High", "Medium", "Low"],
        ordered=True,
    )
    return df


def assign_l2_segment(player: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``l2_segment`` column (Frequent / Occasional / Lapsed)
    based on ``n_visit_days``.
    """
    df = player.copy()
    conditions = [
        df["n_visit_days"] >= SETTINGS.l2_frequent_days,
        df["n_visit_days"] >= SETTINGS.l2_occasional_days,
    ]
    choices = ["Frequent", "Occasional"]
    df["l2_segment"] = pd.Categorical(
        np.select(conditions, choices, default="Lapsed"),
        categories=["Frequent", "Occasional", "Lapsed"],
        ordered=True,
    )
    return df


def assign_segments(player: pd.DataFrame) -> pd.DataFrame:
    """Apply both L1 and L2 segmentation in sequence."""
    return assign_l2_segment(assign_l1_segment(player))
