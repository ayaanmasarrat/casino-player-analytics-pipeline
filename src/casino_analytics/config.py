"""
casino_analytics.config
=======================
Frozen pipeline settings.  Import `SETTINGS` anywhere in the package.
All venue references are fictional ("Cedar Ridge Casino").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root (two levels up from this file: src/casino_analytics/config.py)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    data_dir: Path = _REPO_ROOT / "data" / "raw"
    output_dir: Path = _REPO_ROOT / "output"
    figures_dir: Path = _REPO_ROOT / "output" / "figures"

    play_csv: str = "play_by_month.csv"
    player_csv: str = "players.csv"
    jackpot_csv: str = "jackpots.csv"
    app_events_csv: str = "app_events.csv"

    excel_filename: str = "player_analytics.xlsx"

    # ------------------------------------------------------------------
    # Venue — fictional Cedar Ridge Casino
    # ------------------------------------------------------------------
    venue_lat: float = 45.5231   # somewhere in the Pacific Northwest (fictional)
    venue_lon: float = -122.6765

    # ------------------------------------------------------------------
    # Mileage bands  (miles from venue)
    # ------------------------------------------------------------------
    local_max_miles: float = 30.0
    regional_max_miles: float = 150.0
    # > 150 miles → National

    # ------------------------------------------------------------------
    # Age buckets  (right edge exclusive on the left bucket boundary)
    # ------------------------------------------------------------------
    # Labels: 21-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80+
    age_bin_edges: tuple[int, ...] = (21, 30, 40, 50, 60, 70, 80, 120)
    age_bin_labels: tuple[str, ...] = (
        "21-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"
    )

    # ------------------------------------------------------------------
    # Financial metric guards
    # ------------------------------------------------------------------
    min_coin_in_for_hold: float = 1.0   # avoid divide-by-zero in hold %

    # ------------------------------------------------------------------
    # Segmentation thresholds
    # ------------------------------------------------------------------
    # L1 — value tier  (based on player AMV, average monthly visits)
    l1_high_amv: float = 500.0
    l1_med_amv: float = 150.0

    # L2 — days-active tier  (distinct play days in the period)
    l2_frequent_days: int = 8
    l2_occasional_days: int = 3

    # ------------------------------------------------------------------
    # Causal inference
    # ------------------------------------------------------------------
    psm_caliper: float = 0.05          # max absolute propensity difference
    psm_n_controls: int = 5            # controls per treated unit (1:5)
    psm_random_state: int = 42

    did_cluster_col: str = "player_id"

    # ------------------------------------------------------------------
    # Synthetic data generator
    # ------------------------------------------------------------------
    synth_seed: int = 0
    synth_n_players: int = 48_000
    synth_n_play_rows: int = 40_000
    synth_n_jackpots: int = 350
    synth_treatment_share: float = 0.30   # share of players who adopt the app


SETTINGS = Settings()
