"""
Unit tests for feature engineering modules.

Tests cover:
- Haversine formula accuracy against known city-pair distances
- Age-bucket boundary conditions
- Hold-percentage edge cases (zero coin-in)
- Panel structure (year/month are present and within expected ranges)
- Player rollup (no duplicate player_ids)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Haversine
# ---------------------------------------------------------------------------

from casino_analytics.features.geo import haversine_miles, mileage_band
from casino_analytics.config import SETTINGS


class TestHaversine:
    def test_zero_distance(self) -> None:
        """A point to itself is 0 miles."""
        d = haversine_miles(
            pd.Series([SETTINGS.venue_lat]),
            pd.Series([SETTINGS.venue_lon]),
            SETTINGS.venue_lat,
            SETTINGS.venue_lon,
        )
        assert d.iloc[0] == pytest.approx(0.0, abs=0.01)

    def test_known_distance_portland_seattle(self) -> None:
        """
        Portland (45.5231, -122.6765) to Seattle (47.6062, -122.3321)
        ≈ 145 miles straight-line distance.
        """
        d = haversine_miles(
            pd.Series([45.5231]),
            pd.Series([-122.6765]),
            47.6062, -122.3321,
        )
        assert d.iloc[0] == pytest.approx(145, abs=5)

    def test_vectorised_shape(self) -> None:
        n = 100
        lats = pd.Series(np.random.default_rng(1).uniform(40, 50, n))
        lons = pd.Series(np.random.default_rng(2).uniform(-125, -115, n))
        d = haversine_miles(lats, lons, SETTINGS.venue_lat, SETTINGS.venue_lon)
        assert len(d) == n
        assert (d >= 0).all()


class TestMileageBand:
    def test_local(self) -> None:
        band = mileage_band(pd.Series([5.0, 15.0, 29.9]))
        assert (band == "Local").all()

    def test_regional(self) -> None:
        band = mileage_band(pd.Series([30.0, 80.0, 149.9]))
        assert (band == "Regional").all()

    def test_national(self) -> None:
        band = mileage_band(pd.Series([150.0, 500.0]))
        assert (band == "National").all()


# ---------------------------------------------------------------------------
# Age buckets
# ---------------------------------------------------------------------------

from casino_analytics.features.demographics import age_at_visit, age_bucket


def _make_daily(dob_year: int, visit_year: int, n: int = 1) -> pd.DataFrame:
    return pd.DataFrame({
        "player_id": [f"P{i}" for i in range(n)],
        "date_of_birth": pd.to_datetime([f"{dob_year}-06-15"] * n),
        "accounting_date": pd.to_datetime([f"{visit_year}-07-01"] * n),
        "coin_in": [100.0] * n,
        "theo_win": [8.0] * n,
        "actual_win": [5.0] * n,
    })


class TestAgeBuckets:
    def test_exact_age_boundary(self) -> None:
        """Player born 1980-06-15 visiting on 2025-06-15 is exactly 45."""
        df = _make_daily(1980, 2025)
        df["accounting_date"] = pd.Timestamp("2025-06-15")
        result = age_at_visit(df)
        assert result["age_at_visit"].iloc[0] == 45

    def test_birthday_not_yet(self) -> None:
        """Player born 1980-09-01 visiting on 2025-06-15 is still 44."""
        df = _make_daily(1980, 2025)
        df["date_of_birth"] = pd.Timestamp("1980-09-01")
        df["accounting_date"] = pd.Timestamp("2025-06-15")
        result = age_at_visit(df)
        assert result["age_at_visit"].iloc[0] == 44

    def test_bucket_boundaries(self) -> None:
        """Players at bin edges map to the correct bucket."""
        edges_ages = [21, 30, 40, 50, 60, 70, 80]
        expected_labels = ["21-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
        df = pd.DataFrame({
            "player_id": [f"P{a}" for a in edges_ages],
            "date_of_birth": pd.to_datetime([f"{2025 - a}-06-15" for a in edges_ages]),
            "accounting_date": pd.Timestamp("2025-07-01"),
            "coin_in": [100.0] * len(edges_ages),
        })
        df = age_at_visit(df)
        df = age_bucket(df)
        for i, age in enumerate(edges_ages):
            row = df.loc[df["age_at_visit"] == age]
            assert not row.empty, f"no row with age {age}"
            assert str(row["age_bucket"].iloc[0]) == expected_labels[i], (
                f"age {age} → expected '{expected_labels[i]}', got '{row['age_bucket'].iloc[0]}'"
            )


# ---------------------------------------------------------------------------
# Financial metrics
# ---------------------------------------------------------------------------

from casino_analytics.features.financials import add_financials


class TestFinancials:
    def test_zero_coin_in_guard(self) -> None:
        """Hold % should be NaN when coin_in < min threshold."""
        df = _make_daily(1975, 2024)
        df["coin_in"] = 0.0
        result = add_financials(df)
        assert pd.isna(result["theo_hold_pct"].iloc[0])
        assert pd.isna(result["act_hold_pct"].iloc[0])

    def test_normal_hold_pct(self) -> None:
        df = _make_daily(1975, 2024)
        df["coin_in"] = 200.0
        df["theo_win"] = 10.0
        df["actual_win"] = 8.0
        result = add_financials(df)
        assert result["theo_hold_pct"].iloc[0] == pytest.approx(5.0, abs=0.01)
        assert result["act_hold_pct"].iloc[0] == pytest.approx(4.0, abs=0.01)

    def test_no_slot_flag(self) -> None:
        df = _make_daily(1970, 2024)
        df["game_type"] = "table"
        result = add_financials(df)
        assert result["no_slot_play_flag"].iloc[0] == 1

    def test_slot_flag_zero(self) -> None:
        df = _make_daily(1970, 2024)
        df["game_type"] = "slot"
        result = add_financials(df)
        assert result["no_slot_play_flag"].iloc[0] == 0


# ---------------------------------------------------------------------------
# Player rollup
# ---------------------------------------------------------------------------

from casino_analytics.features.player import build_player_rollup, build_player_month_panel


def _make_full_daily(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    n_players = 10
    player_ids = [f"P{i}" for i in range(n_players)]
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "player_id": rng.choice(player_ids, size=n),
        "accounting_date": rng.choice(dates, size=n),
        "coin_in": rng.exponential(100, size=n),
        "theo_win": rng.exponential(8, size=n),
        "actual_win": rng.exponential(7, size=n),
        "n_txn": rng.integers(1, 5, size=n),
            "age_at_visit": pd.array(rng.integers(21, 80, size=n), dtype="Int16"),
        "distance_miles": rng.uniform(5, 200, size=n),
        "date_of_birth": pd.Timestamp("1975-01-01"),
        "enrollment_date": pd.Timestamp("2022-01-01"),
        "gender": "M",
        "state": "WA",
        "zip": "90000",
    })
    df["age_bucket"] = pd.Categorical(
        pd.cut(df["age_at_visit"], bins=[21,30,40,50,60,70,80,120], labels=[
            "21-29","30-39","40-49","50-59","60-69","70-79","80+"
        ], right=False, include_lowest=True).astype(str),
        categories=["21-29","30-39","40-49","50-59","60-69","70-79","80+"],
    )
    return df


class TestPlayerRollup:
    def test_no_duplicate_player_ids(self) -> None:
        daily = _make_full_daily()
        player = build_player_rollup(daily)
        assert player["player_id"].duplicated().sum() == 0

    def test_amv_positive(self) -> None:
        daily = _make_full_daily()
        player = build_player_rollup(daily)
        assert (player["amv"] > 0).all()

    def test_panel_year_month_types(self) -> None:
        daily = _make_full_daily()
        panel = build_player_month_panel(daily)
        assert "year" in panel.columns
        assert "month" in panel.columns
        assert panel["month"].between(1, 12).all()


# ---------------------------------------------------------------------------
# PSM caliper enforcement
# ---------------------------------------------------------------------------

from casino_analytics.causal.psm import match_nearest_neighbour


class TestPSMCaliper:
    def test_caliper_respected(self) -> None:
        """All matched pairs must be within the caliper distance."""
        rng = np.random.default_rng(0)
        n = 300
        df = pd.DataFrame({
            "player_id": [f"P{i}" for i in range(n)],
            "propensity": rng.uniform(0.1, 0.9, size=n),
            "treated": np.where(rng.random(n) < 0.4, 1, 0),
            "amv": rng.exponential(200, size=n),
        })
        caliper = 0.05
        matched = match_nearest_neighbour(df, n_controls=3, caliper=caliper)

        for group_id, grp in matched.groupby("match_group"):
            t_prop = grp.loc[grp["treated"] == 1, "propensity"].iloc[0]
            c_props = grp.loc[grp["treated"] == 0, "propensity"]
            for cp in c_props:
                assert abs(t_prop - cp) <= caliper + 1e-9, (
                    f"group {group_id}: |{t_prop:.4f} - {cp:.4f}| > caliper {caliper}"
                )
