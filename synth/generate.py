"""
synth.generate
==============
Seeded synthetic casino dataset generator.

Produces four CSV files under data/raw/:
  - play_by_month.csv   (~40 k rows)   daily play transactions
  - players.csv         (~48 k rows)   player demographics
  - jackpots.csv        (~350 rows)    jackpot events
  - app_events.csv                     mobile-app adoption events

All coordinates, player IDs, and values are entirely invented.
No real venue, no real player data.

Usage:
    python synth/generate.py
    # or via the CLI:
    python -m casino_analytics.cli generate
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as a script without installing the package
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from casino_analytics.config import SETTINGS  # noqa: E402


# ---------------------------------------------------------------------------
# Fictional synthetic ZIP codes and their centroids
# (completely invented lat/lon values near the Pacific Northwest)
# ---------------------------------------------------------------------------
_N_ZIPS = 200

# ---------------------------------------------------------------------------

def _build_synthetic_zips(rng: np.random.Generator) -> pd.DataFrame:
    """Create a lookup table of fictional ZIP codes with centroids."""
    zips = [f"{90000 + i:05d}" for i in range(_N_ZIPS)]
    # Scatter them around the venue with a realistic distance distribution
    angles = rng.uniform(0, 2 * np.pi, _N_ZIPS)
    # Distance distribution: most players within 200 miles, some further
    distances_miles = rng.exponential(scale=60, size=_N_ZIPS).clip(1, 800)
    # 1 degree of latitude ≈ 69 miles
    lat_offsets = distances_miles / 69 * np.cos(angles)
    lon_offsets = distances_miles / (69 * np.cos(np.radians(SETTINGS.venue_lat))) * np.sin(angles)
    return pd.DataFrame({
        "zip": zips,
        "zip_lat": SETTINGS.venue_lat + lat_offsets,
        "zip_lon": SETTINGS.venue_lon + lon_offsets,
    })


def generate(seed: int | None = None, output_dir: Path | None = None) -> None:
    """Generate all four synthetic CSV files."""
    seed = seed if seed is not None else SETTINGS.synth_seed
    rng = np.random.default_rng(seed)
    out = output_dir or SETTINGS.data_dir
    out.mkdir(parents=True, exist_ok=True)

    n_players = SETTINGS.synth_n_players
    n_plays = SETTINGS.synth_n_play_rows
    n_jackpots = SETTINGS.synth_n_jackpots

    zip_df = _build_synthetic_zips(rng)

    # ------------------------------------------------------------------
    # 1. PLAYERS  (demographics)
    # ------------------------------------------------------------------
    player_ids = [f"P{i:07d}" for i in range(1, n_players + 1)]

    # Ages: casino players skew 45-70
    ages = (rng.beta(a=3, b=2, size=n_players) * (85 - 21) + 21).astype(int).clip(21, 95)

    # Birth dates consistent with age (approximate)
    ref_year = 2025
    birth_years = ref_year - ages
    birth_months = rng.integers(1, 13, size=n_players)
    birth_days = rng.integers(1, 29, size=n_players)
    dob = pd.to_datetime({
        "year": birth_years,
        "month": birth_months,
        "day": birth_days,
    })

    dob_str = dob.dt.strftime("%Y-%m-%d")

    genders = rng.choice(["M", "F", "O"], size=n_players, p=[0.47, 0.50, 0.03])
    states = rng.choice(
        ["WA", "OR", "CA", "ID", "MT", "NV", "AZ", "TX", "FL"],
        size=n_players,
        p=[0.35, 0.20, 0.15, 0.07, 0.05, 0.05, 0.05, 0.04, 0.04],
    )
    zip_indices = rng.integers(0, _N_ZIPS, size=n_players)
    player_zips = zip_df["zip"].iloc[zip_indices].values
    player_zip_lat = zip_df["zip_lat"].iloc[zip_indices].values
    player_zip_lon = zip_df["zip_lon"].iloc[zip_indices].values

    # Enrollment date: uniform over a 4-year window
    enroll_days_offset = rng.integers(0, 4 * 365, size=n_players)
    enroll_dates = pd.Timestamp("2021-01-01") + pd.to_timedelta(enroll_days_offset, unit="D")

    players_df = pd.DataFrame({
        "player_id": player_ids,
        "date_of_birth": dob_str,
        "gender": genders,
        "state": states,
        "zip": player_zips,
        "zip_lat": player_zip_lat.round(6),
        "zip_lon": player_zip_lon.round(6),
        "enrollment_date": enroll_dates.strftime("%Y-%m-%d"),
    })
    players_df.to_csv(out / SETTINGS.player_csv, index=False)
    print(f"  players.csv          {len(players_df):>8,} rows")

    # ------------------------------------------------------------------
    # 2. PLAY DATA  (daily transactions — one row per player-day-month)
    # ------------------------------------------------------------------
    # Sample players with replacement (frequent visitors appear more)
    visit_weights = rng.pareto(a=1.5, size=n_players) + 1
    visit_weights /= visit_weights.sum()
    play_player_ids = rng.choice(player_ids, size=n_plays, replace=True, p=visit_weights)

    # Dates spanning 24 months ending May 2025
    start_date = pd.Timestamp("2023-06-01")
    end_date = pd.Timestamp("2025-05-31")
    date_range = pd.date_range(start_date, end_date, freq="D")
    play_dates = rng.choice(date_range, size=n_plays)

    # Coin-in: heavy-tail (Pareto) — most plays are small, a few are large
    coin_in = (rng.pareto(a=1.2, size=n_plays) * 50 + 20).round(2)

    # Theoretical win ≈ coin_in * hold% (house edge ~5–12 %)
    theo_hold_pct = rng.uniform(0.05, 0.12, size=n_plays)
    theo_win = (coin_in * theo_hold_pct).round(2)

    # Actual win: random around theo_win (can be negative = player won)
    noise = rng.normal(loc=0, scale=0.3, size=n_plays)
    actual_win = (coin_in * (theo_hold_pct + noise)).round(2)

    # Game type
    game_types = rng.choice(
        ["slot", "table", "video_poker", "kiosk"],
        size=n_plays,
        p=[0.65, 0.20, 0.10, 0.05],
    )

    play_df = pd.DataFrame({
        "player_id": play_player_ids,
        "accounting_date": pd.Series(play_dates).dt.strftime("%Y-%m-%d"),
        "year": pd.Series(play_dates).dt.year,
        "month": pd.Series(play_dates).dt.month,
        "game_type": game_types,
        "coin_in": coin_in,
        "theo_win": theo_win,
        "actual_win": actual_win,
    })
    play_df.to_csv(out / SETTINGS.play_csv, index=False)
    print(f"  play_by_month.csv    {len(play_df):>8,} rows")

    # ------------------------------------------------------------------
    # 3. JACKPOTS
    # ------------------------------------------------------------------
    jackpot_player_ids = rng.choice(player_ids, size=n_jackpots, replace=True)
    jackpot_dates = rng.choice(date_range, size=n_jackpots)
    jackpot_amounts = (rng.exponential(scale=5_000, size=n_jackpots) + 1_200).round(2)

    jackpots_df = pd.DataFrame({
        "player_id": jackpot_player_ids,
        "jackpot_date": pd.Series(jackpot_dates).dt.strftime("%Y-%m-%d"),
        "amount": jackpot_amounts,
    })
    jackpots_df.to_csv(out / SETTINGS.jackpot_csv, index=False)
    print(f"  jackpots.csv         {len(jackpots_df):>8,} rows")

    # ------------------------------------------------------------------
    # 4. APP EVENTS  (mobile-app adoption — used for causal section)
    # ------------------------------------------------------------------
    # Treatment group: ~30 % of active players adopt the app
    treatment_share = SETTINGS.synth_treatment_share
    n_treated = int(n_players * treatment_share)

    # Treated players tend to be younger and higher coin-in (selection bias)
    young_players = players_df[players_df["date_of_birth"].apply(
        lambda d: (pd.Timestamp("2025-01-01") - pd.Timestamp(d)).days / 365 < 50
    )]["player_id"].values

    if len(young_players) >= n_treated // 2:
        treated_young = rng.choice(young_players, size=n_treated // 2, replace=False)
        remaining = list(set(player_ids) - set(treated_young))
        treated_other = rng.choice(remaining, size=n_treated - len(treated_young), replace=False)
        treated_ids = np.concatenate([treated_young, treated_other])
    else:
        treated_ids = rng.choice(player_ids, size=n_treated, replace=False)

    # Adoption dates: 2024-01 through 2024-06 (treatment window)
    adoption_start = pd.Timestamp("2024-01-01")
    adoption_end = pd.Timestamp("2024-06-30")
    adoption_date_range = pd.date_range(adoption_start, adoption_end, freq="D")
    adoption_dates = rng.choice(adoption_date_range, size=len(treated_ids))

    app_events_df = pd.DataFrame({
        "player_id": treated_ids,
        "adoption_date": pd.Series(adoption_dates).dt.strftime("%Y-%m-%d"),
    })
    app_events_df.to_csv(out / SETTINGS.app_events_csv, index=False)
    print(f"  app_events.csv       {len(app_events_df):>8,} rows")
    print(f"\nData written to: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic casino dataset")
    parser.add_argument("--seed", type=int, default=SETTINGS.synth_seed)
    parser.add_argument("--output-dir", type=Path, default=SETTINGS.data_dir)
    args = parser.parse_args()
    generate(seed=args.seed, output_dir=args.output_dir)
