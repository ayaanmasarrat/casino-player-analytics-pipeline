"""
casino_analytics.loaders
=========================
Load raw CSV files into DataFrames with explicit dtype schemas.

All four CSVs are read with `encoding="utf-8-sig"` so that files saved from
Excel (with a BOM) work without pre-processing.

Returns
-------
Named DataFrames accessed via the ``load_all()`` convenience function.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from casino_analytics.config import SETTINGS

# ---------------------------------------------------------------------------
# Explicit column dtype maps
# ---------------------------------------------------------------------------

_PLAY_DTYPES: dict[str, str] = {
    "player_id": "str",
    "accounting_date": "str",
    "year": "Int16",
    "month": "Int8",
    "game_type": "category",
    "coin_in": "float64",
    "theo_win": "float64",
    "actual_win": "float64",
}

_PLAYER_DTYPES: dict[str, str] = {
    "player_id": "str",
    "date_of_birth": "str",
    "gender": "category",
    "state": "category",
    "zip": "str",
    "zip_lat": "float64",
    "zip_lon": "float64",
    "enrollment_date": "str",
}

_JACKPOT_DTYPES: dict[str, str] = {
    "player_id": "str",
    "jackpot_date": "str",
    "amount": "float64",
}

_APP_EVENTS_DTYPES: dict[str, str] = {
    "player_id": "str",
    "adoption_date": "str",
}


def _read(filename: str, dtypes: dict[str, str], data_dir: Path | None = None) -> pd.DataFrame:
    """Read a single CSV with the given dtype map."""
    path = (data_dir or SETTINGS.data_dir) / filename
    return pd.read_csv(path, dtype=dtypes, encoding="utf-8-sig")


def load_play(data_dir: Path | None = None) -> pd.DataFrame:
    """Load play transaction data."""
    df = _read(SETTINGS.play_csv, _PLAY_DTYPES, data_dir)
    df["accounting_date"] = pd.to_datetime(df["accounting_date"])
    return df


def load_players(data_dir: Path | None = None) -> pd.DataFrame:
    """Load player demographics."""
    df = _read(SETTINGS.player_csv, _PLAYER_DTYPES, data_dir)
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"])
    df["enrollment_date"] = pd.to_datetime(df["enrollment_date"])
    return df


def load_jackpots(data_dir: Path | None = None) -> pd.DataFrame:
    """Load jackpot events."""
    df = _read(SETTINGS.jackpot_csv, _JACKPOT_DTYPES, data_dir)
    df["jackpot_date"] = pd.to_datetime(df["jackpot_date"])
    return df


def load_app_events(data_dir: Path | None = None) -> pd.DataFrame:
    """Load mobile-app adoption events."""
    df = _read(SETTINGS.app_events_csv, _APP_EVENTS_DTYPES, data_dir)
    df["adoption_date"] = pd.to_datetime(df["adoption_date"])
    return df


def load_all(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load all four tables and return them as a named dict."""
    return {
        "play": load_play(data_dir),
        "players": load_players(data_dir),
        "jackpots": load_jackpots(data_dir),
        "app_events": load_app_events(data_dir),
    }
