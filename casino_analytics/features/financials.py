"""
casino_analytics.features.financials
======================================
Derive hold-percentage metrics and the no-slot-play flag.

All operations are vectorised; zero-coin-in guard prevents division errors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from casino_analytics.config import SETTINGS


def add_financials(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Add three financial metric columns to the daily-player table.

    Columns added
    -------------
    theo_hold_pct
        Theoretical hold percentage: ``theo_win / coin_in * 100``.
        Set to ``NaN`` where ``coin_in < SETTINGS.min_coin_in_for_hold``.
    act_hold_pct
        Actual hold percentage: ``actual_win / coin_in * 100``.
        Same guard applied.
    no_slot_play_flag
        Integer flag (1 / 0): 1 when the player's primary game that day
        was *not* a slot machine.
    """
    df = daily.copy()
    guard = df["coin_in"] >= SETTINGS.min_coin_in_for_hold

    df["theo_hold_pct"] = np.where(
        guard,
        df["theo_win"] / df["coin_in"] * 100,
        np.nan,
    )
    df["act_hold_pct"] = np.where(
        guard,
        df["actual_win"] / df["coin_in"] * 100,
        np.nan,
    )

    if "game_type" in df.columns:
        df["no_slot_play_flag"] = (~df["game_type"].isin(["slot"])).astype(np.int8)
    else:
        df["no_slot_play_flag"] = np.int8(0)

    return df
