"""
casino_analytics.analytics.trends
====================================
Monthly Active User (MAU) aggregation and year-over-year comparisons.

Operates on the player-month panel produced by ``features.player``.
"""
from __future__ import annotations

import pandas as pd


def monthly_active_users(panel: pd.DataFrame, player: pd.DataFrame) -> pd.DataFrame:
    """
    Compute MAU by month, with optional L1 segment breakdown.

    Parameters
    ----------
    panel : DataFrame
        Player-month panel (one row per player × year × month).
    player : DataFrame
        Player rollup table — must contain ``player_id`` and (optionally)
        ``l1_segment``.

    Returns
    -------
    DataFrame with columns: year, month, period, l1_segment (if available),
    mau (distinct active players), total_coin_in, avg_coin_in_per_player.
    """
    df = panel.copy()

    # Attach segment if available
    if "l1_segment" in player.columns:
        seg = player[["player_id", "l1_segment"]]
        df = df.merge(seg, on="player_id", how="left")
        group_cols = ["year", "month", "l1_segment"]
    else:
        group_cols = ["year", "month"]

    mau = (
        df.groupby(group_cols, observed=True)
        .agg(
            mau=("player_id", "nunique"),
            total_coin_in=("coin_in", "sum"),
        )
        .reset_index()
    )
    mau["avg_coin_in_per_player"] = (mau["total_coin_in"] / mau["mau"]).round(2)
    mau["period"] = pd.to_datetime(
        mau["year"].astype(str) + "-" + mau["month"].astype(str).str.zfill(2) + "-01"
    )
    return mau.sort_values(["year", "month"]).reset_index(drop=True)


def yoy_comparison(mau: pd.DataFrame) -> pd.DataFrame:
    """
    Year-over-year change for each calendar month.

    Merges MAU on itself shifted by one year and computes:
    - ``mau_yoy_delta``   : absolute MAU change
    - ``mau_yoy_pct``     : percentage change in MAU
    - ``coin_yoy_pct``    : percentage change in total coin-in

    Requires the MAU DataFrame to **not** be broken out by segment
    (aggregate first if needed).

    Parameters
    ----------
    mau : DataFrame
        Output of ``monthly_active_users`` (no segment breakdown).

    Returns
    -------
    DataFrame with year-over-year columns added.
    """
    df = mau.copy()
    # Build a prior-year lookup
    prev = df[["year", "month", "mau", "total_coin_in"]].copy()
    prev["year"] = prev["year"] + 1
    prev = prev.rename(columns={"mau": "mau_py", "total_coin_in": "coin_in_py"})

    merged = df.merge(prev, on=["year", "month"], how="left")
    merged["mau_yoy_delta"] = merged["mau"] - merged["mau_py"]
    merged["mau_yoy_pct"] = (merged["mau_yoy_delta"] / merged["mau_py"] * 100).round(2)
    merged["coin_yoy_pct"] = (
        (merged["total_coin_in"] - merged["coin_in_py"]) / merged["coin_in_py"] * 100
    ).round(2)
    return merged.drop(columns=["mau_py", "coin_in_py"])
