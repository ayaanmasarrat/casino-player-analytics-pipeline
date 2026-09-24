"""
casino_analytics.features.player
===================================
Player-level rollup from the daily table, and player-month panel.

Outputs
-------
player_df  : one row per player_id with lifetime metrics.
panel_df   : one row per (player_id, year, month) — the balanced panel used
             for causal inference.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def build_player_rollup(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the daily table to one row per ``player_id``.

    Metrics computed
    ----------------
    total_coin_in   : sum of coin_in across all visits
    total_theo_win  : sum of theo_win
    total_actual_win: sum of actual_win
    n_visit_days    : number of distinct play days
    first_visit     : earliest accounting_date
    last_visit      : most recent accounting_date
    amv             : Average Monthly Value = total_coin_in / active_months
    adt             : Average Daily Trips = n_visit_days / active_months
    age_at_visit    : median age across all visits (rounded to nearest int)
    age_bucket      : mode age bucket
    gender          : first value (constant within player)
    state           : first value
    zip             : first value
    distance_miles  : first value (constant within player)
    mileage_band    : first value
    """
    carry_first = [
        c for c in (
            "gender", "state", "zip", "distance_miles", "mileage_band",
            "date_of_birth", "enrollment_date"
        )
        if c in daily.columns
    ]

    agg: dict = {
        "total_coin_in": ("coin_in", "sum"),
        "total_theo_win": ("theo_win", "sum"),
        "total_actual_win": ("actual_win", "sum"),
        "n_visit_days": ("accounting_date", "nunique"),
        "first_visit": ("accounting_date", "min"),
        "last_visit": ("accounting_date", "max"),
    }

    if "age_at_visit" in daily.columns:
        agg["age_at_visit_median"] = ("age_at_visit", "median")
    if "age_bucket" in daily.columns:
        agg["age_bucket"] = ("age_bucket", lambda s: s.mode().iat[0])
    if "no_slot_play_flag" in daily.columns:
        agg["no_slot_play_days"] = ("no_slot_play_flag", "sum")

    for col in carry_first:
        agg[col] = (col, "first")

    player = (
        daily
        .groupby("player_id", sort=False)
        .agg(**agg)
        .reset_index()
    )

    # Active months = span from first to last visit (at least 1)
    span_days = (player["last_visit"] - player["first_visit"]).dt.days.clip(lower=0)
    active_months = (span_days / 30.44).clip(lower=1)

    player["amv"] = (player["total_coin_in"] / active_months).round(2)
    player["adt"] = (player["n_visit_days"] / active_months).round(4)

    return player


def build_player_month_panel(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to a player-month panel: one row per (player_id, year, month).

    Used as the unit of analysis in the DiD regression.

    Metrics
    -------
    coin_in       : monthly sum
    theo_win      : monthly sum
    actual_win    : monthly sum
    visit_days    : distinct play days in the month
    n_txn         : total transactions in the month
    """
    df = daily.copy()
    df["year"] = df["accounting_date"].dt.year
    df["month"] = df["accounting_date"].dt.month

    panel = (
        df.groupby(["player_id", "year", "month"], sort=False)
        .agg(
            coin_in=("coin_in", "sum"),
            theo_win=("theo_win", "sum"),
            actual_win=("actual_win", "sum"),
            visit_days=("accounting_date", "nunique"),
            n_txn=("n_txn", "sum") if "n_txn" in df.columns else ("coin_in", "count"),
        )
        .reset_index()
    )
    return panel
