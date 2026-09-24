"""
casino_analytics.features.daily
=================================
Merge player demographics into play data and aggregate to
daily-player level (one row per player × accounting_date).

All aggregation is vectorised via groupby; no ``.apply()`` is used.
"""
from __future__ import annotations

import pandas as pd


def merge_demographics(play: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join play transactions with player demographics on ``player_id``.

    Parameters
    ----------
    play : DataFrame
        Play-transaction table (one row per raw transaction).
    players : DataFrame
        Deduplicated player demographics.

    Returns
    -------
    DataFrame
        Merged table retaining all play rows.
    """
    # Keep only the demographic columns needed downstream
    demo_cols = [
        "player_id",
        "date_of_birth",
        "gender",
        "state",
        "zip",
        "zip_lat",
        "zip_lon",
        "enrollment_date",
    ]
    available = [c for c in demo_cols if c in players.columns]
    return play.merge(players[available], on="player_id", how="left")


def aggregate_daily(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse raw transactions to one row per (player_id, accounting_date).

    Aggregations
    ------------
    - ``coin_in``   : sum
    - ``theo_win``  : sum
    - ``actual_win``: sum
    - ``n_txn``     : count of raw transactions
    - ``game_type`` : mode (most frequent game played that day)

    Demographic columns are carried forward via ``first()`` — they are
    constant within a player so any row is equivalent.
    """
    demo_carry = [
        c for c in ("date_of_birth", "gender", "state", "zip",
                    "zip_lat", "zip_lon", "enrollment_date")
        if c in merged.columns
    ]

    agg_dict: dict = {
        "coin_in": ("coin_in", "sum"),
        "theo_win": ("theo_win", "sum"),
        "actual_win": ("actual_win", "sum"),
        "n_txn": ("coin_in", "count"),
    }

    for col in demo_carry:
        agg_dict[col] = (col, "first")

    daily = (
        merged
        .groupby(["player_id", "accounting_date"], sort=False)
        .agg(**agg_dict)
        .reset_index()
    )

    # Add game_type mode separately to avoid lambda in named-agg dict
    if "game_type" in merged.columns:
        game_mode = (
            merged
            .groupby(["player_id", "accounting_date"], sort=False)["game_type"]
            .agg(lambda s: s.mode().iat[0])
            .reset_index()
            .rename(columns={"game_type": "game_type"})
        )
        daily = daily.merge(game_mode, on=["player_id", "accounting_date"], how="left")
    return daily
