"""
casino_analytics.features.geo
================================
Vectorised haversine distance from each player's home ZIP centroid to the
venue, and assignment to a mileage band (Local / Regional / National).

No external geocoding service is required: the synthetic data generator
stores ``zip_lat`` / ``zip_lon`` directly on each player row; the venue
coordinates come from ``SETTINGS``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from casino_analytics.config import SETTINGS

# Earth's mean radius in miles
_EARTH_RADIUS_MI = 3_958.8


def haversine_miles(
    lat1: pd.Series | np.ndarray,
    lon1: pd.Series | np.ndarray,
    lat2: float,
    lon2: float,
) -> pd.Series:
    """
    Vectorised haversine distance (miles) from an array of points to a single
    destination (the venue).

    Parameters
    ----------
    lat1, lon1 : array-like of floats
        Origin coordinates in decimal degrees.
    lat2, lon2 : float
        Destination coordinates in decimal degrees (venue).

    Returns
    -------
    pd.Series of float
        Distance in miles.
    """
    lat1_r = np.radians(np.asarray(lat1, dtype=float))
    lon1_r = np.radians(np.asarray(lon1, dtype=float))
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return pd.Series(c * _EARTH_RADIUS_MI, index=np.arange(len(lat1_r)), dtype="float64")


def mileage_band(miles: pd.Series) -> pd.Series:
    """
    Assign each distance to a named band.

    Bands
    -----
    Local    : 0 – LOCAL_MAX  (< 30 miles)
    Regional : LOCAL_MAX – REGIONAL_MAX  (30 – 150 miles)
    National : > REGIONAL_MAX  (> 150 miles)
    """
    conditions = [
        miles < SETTINGS.local_max_miles,
        miles < SETTINGS.regional_max_miles,
    ]
    choices = ["Local", "Regional"]
    return pd.Series(
        np.select(conditions, choices, default="National"),
        index=miles.index,
        dtype="category",
    )


def add_geo(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``distance_miles`` and ``mileage_band`` columns to the daily table.

    Expects columns ``zip_lat`` and ``zip_lon`` to already be present
    (populated during the demographics merge step).
    """
    df = daily.copy()

    dist = haversine_miles(
        df["zip_lat"], df["zip_lon"],
        SETTINGS.venue_lat, SETTINGS.venue_lon,
    )
    dist.index = df.index
    df["distance_miles"] = dist.round(2)
    df["mileage_band"] = mileage_band(dist)
    return df
