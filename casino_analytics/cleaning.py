"""
casino_analytics.cleaning
==========================
Data-quality routines applied before any feature engineering.

Steps
-----
1. ``clean_players``  — deduplicate to one row per ``player_id``, parse dates.
2. ``parse_play_dates`` — ensure ``accounting_date`` is timezone-naive datetime.
"""
from __future__ import annotations

import pandas as pd


def clean_players(players: pd.DataFrame) -> pd.DataFrame:
    """
    Return a player table with exactly one row per ``player_id``.

    Deduplication strategy:
    - Sort by ``enrollment_date`` ascending so the *earliest* record is kept.
    - Call ``drop_duplicates(subset=["player_id"], keep="first")`` — fully
      vectorised, no Python loops.

    Date columns ``date_of_birth`` and ``enrollment_date`` are coerced to
    timezone-naive datetime64[ns] (UTC timestamps are stripped of tz-info).
    """
    df = players.copy()

    # Coerce dates to timezone-naive (handles both naive and tz-aware inputs)
    for col in ("date_of_birth", "enrollment_date"):
        if col in df.columns:
            df[col] = (
                pd.to_datetime(df[col], utc=True)
                .dt.tz_localize(None)
                if pd.api.types.is_datetime64_any_dtype(df[col])
                and df[col].dt.tz is not None
                else pd.to_datetime(df[col])
            )

    # Keep earliest enrollment record for each player
    df = (
        df.sort_values("enrollment_date", ascending=True, na_position="last")
        .drop_duplicates(subset=["player_id"], keep="first")
        .reset_index(drop=True)
    )

    return df


def parse_play_dates(play: pd.DataFrame) -> pd.DataFrame:
    """
    Return play data with ``accounting_date`` as timezone-naive datetime64[ns].

    If the column already is timezone-naive datetime it is returned unchanged.
    UTC strings / tz-aware datetimes have the timezone information stripped
    rather than converted, matching the source data's intent (local business
    date, not a UTC instant).
    """
    df = play.copy()
    col = df["accounting_date"]

    if not pd.api.types.is_datetime64_any_dtype(col):
        col = pd.to_datetime(col)

    if col.dt.tz is not None:
        col = col.dt.tz_localize(None)

    df["accounting_date"] = col
    return df
