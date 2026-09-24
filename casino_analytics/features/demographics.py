"""
casino_analytics.features.demographics
========================================
Compute age at each visit and assign players to age buckets.

Vectorised: uses pandas arithmetic on datetime Series, no ``.apply()``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from casino_analytics.config import SETTINGS


def age_at_visit(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``age_at_visit`` column: exact age in full years on ``accounting_date``.

    Formula (vectorised)
    --------------------
    years_diff = accounting_date.year - dob.year
    age = years_diff - (month-day of visit < month-day of birthday)
    """
    df = daily.copy()
    dob: pd.Series = pd.to_datetime(df["date_of_birth"])
    visit: pd.Series = pd.to_datetime(df["accounting_date"])

    years_diff = visit.dt.year - dob.dt.year
    # 1 if the birthday hasn't happened yet this year, else 0
    not_yet = (
        (visit.dt.month < dob.dt.month) |
        ((visit.dt.month == dob.dt.month) & (visit.dt.day < dob.dt.day))
    ).astype(np.int8)

    df["age_at_visit"] = (years_diff - not_yet).astype("Int16")
    return df


def age_bucket(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``age_bucket`` column using the bin edges defined in ``SETTINGS``.

    Ages below the minimum edge (21) are labelled ``"<21"`` and kept for
    completeness; they can be filtered out before analysis if needed.
    """
    df = daily.copy()
    edges = list(SETTINGS.age_bin_edges)
    labels = list(SETTINGS.age_bin_labels)

    df["age_bucket"] = pd.cut(
        df["age_at_visit"],
        bins=edges,
        labels=labels,
        right=False,       # [left, right)
        include_lowest=True,
    )
    # Label players outside the bin range
    df["age_bucket"] = df["age_bucket"].cat.add_categories("<21")
    df.loc[df["age_at_visit"] < edges[0], "age_bucket"] = "<21"
    return df


def add_demographics(daily: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper: apply both age steps in sequence."""
    return age_bucket(age_at_visit(daily))
