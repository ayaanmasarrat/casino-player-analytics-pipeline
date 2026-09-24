"""
casino_analytics.cli
======================
Command-line entry points for the pipeline.

Commands
--------
generate
    Generate synthetic data files (data/raw/*.csv).
run-all
    Load data, run all feature-engineering + causal steps, write Excel and figures.

Usage
-----
    python -m casino_analytics.cli generate
    python -m casino_analytics.cli run-all
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from casino_analytics.config import SETTINGS


def _log(msg: str) -> None:
    print(f"  {msg}", flush=True)


# ---------------------------------------------------------------------------
# generate command
# ---------------------------------------------------------------------------

def cmd_generate(args: argparse.Namespace) -> None:
    from synth.generate import generate  # type: ignore[import]

    seed = getattr(args, "seed", SETTINGS.synth_seed)
    output_dir = getattr(args, "output_dir", SETTINGS.data_dir)
    print("Generating synthetic dataset...")
    generate(seed=seed, output_dir=Path(output_dir))
    print("Done.\n")


# ---------------------------------------------------------------------------
# run-all command
# ---------------------------------------------------------------------------

def cmd_run_all(args: argparse.Namespace) -> None:  # noqa: C901
    from casino_analytics.loaders import load_all
    from casino_analytics.cleaning import clean_players, parse_play_dates
    from casino_analytics.features.daily import merge_demographics, aggregate_daily
    from casino_analytics.features.demographics import add_demographics
    from casino_analytics.features.financials import add_financials
    from casino_analytics.features.geo import add_geo
    from casino_analytics.features.player import build_player_rollup, build_player_month_panel
    from casino_analytics.segmentation import assign_segments
    from casino_analytics.analytics.trends import monthly_active_users, yoy_comparison
    from casino_analytics.causal.psm import (
        estimate_propensity, trim_common_support,
        match_nearest_neighbour, balance_table,
    )
    from casino_analytics.causal.did import (
        add_post_indicator, did_basic, did_with_month_fe,
        build_event_study_panel, event_study, parallel_trends_f_test,
    )
    from casino_analytics.plots import (
        plot_mau_trend, plot_mau_by_segment, plot_age_distribution,
        plot_propensity_overlap, plot_balance, plot_event_study,
    )

    t0 = time.perf_counter()
    print("=" * 60)
    print("Casino Player Analytics Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Section 1 — Feature Engineering
    # ------------------------------------------------------------------
    print("\n[1/4] Feature Engineering")

    _log("Loading CSVs...")
    tables = load_all()
    play = parse_play_dates(tables["play"])
    players = clean_players(tables["players"])
    app_events = tables["app_events"]
    _log(f"  play: {len(play):,} rows  |  players: {len(players):,} rows")

    _log("Merging and aggregating to daily level...")
    merged = merge_demographics(play, players)
    daily = aggregate_daily(merged)
    daily = add_demographics(daily)
    daily = add_financials(daily)
    daily = add_geo(daily)
    _log(f"  daily table: {len(daily):,} rows × {daily.shape[1]} cols")

    _log("Building player rollup and panel...")
    player = build_player_rollup(daily)
    player = assign_segments(player)
    panel = build_player_month_panel(daily)
    _log(f"  player: {len(player):,} rows | panel: {len(panel):,} rows")

    # ------------------------------------------------------------------
    # Section 2 — Trend Analytics
    # ------------------------------------------------------------------
    print("\n[2/4] Trend Analytics")

    mau_seg = monthly_active_users(panel, player)
    mau_total = (
        mau_seg.groupby(["year", "month", "period"])
        .agg(mau=("mau", "sum"), total_coin_in=("total_coin_in", "sum"))
        .reset_index()
    )
    yoy = yoy_comparison(mau_total)
    _log(f"  MAU range: {mau_total['mau'].min():,} – {mau_total['mau'].max():,} players/month")

    p1 = plot_mau_trend(mau_total)
    p2 = plot_mau_by_segment(mau_seg)
    p3 = plot_age_distribution(player)
    _log(f"  Charts saved: {Path(p1).name}, {Path(p2).name}, {Path(p3).name}")

    # ------------------------------------------------------------------
    # Section 3 — Causal Inference (PSM + DiD)
    # ------------------------------------------------------------------
    print("\n[3/4] Causal Inference")

    treated_ids = set(app_events["player_id"])
    player["treated"] = player["player_id"].isin(treated_ids).astype(int)

    feature_cols = ["amv", "adt", "n_visit_days", "distance_miles"]
    player_clean = player[feature_cols + ["player_id", "treated"]].dropna()
    _log(f"  {player_clean['treated'].sum():,} treated | {(player_clean['treated']==0).sum():,} control")

    player_ps = estimate_propensity(player_clean, feature_cols)
    player_trim = trim_common_support(player_ps)
    matched = match_nearest_neighbour(player_trim)
    _log(f"  Matched: {matched['treated'].sum():,} treated, {(matched['treated']==0).sum():,} controls")

    btable = balance_table(player_trim, matched, feature_cols)
    p4 = plot_propensity_overlap(matched)
    p5 = plot_balance(btable)
    _log(f"  Balance charts: {Path(p4).name}, {Path(p5).name}")

    # DiD
    panel_m = panel.merge(player_clean[["player_id", "treated"]], on="player_id", how="inner")
    panel_m = add_post_indicator(panel_m, 2024, 4)
    basic_res = did_basic(panel_m)
    fe_res = did_with_month_fe(panel_m)
    att_basic = basic_res.params.get("treat_post", float("nan"))
    att_fe = fe_res.params.get("treat_post", float("nan"))
    _log(f"  DiD ATT (basic): {att_basic:+.2f} | (FE): {att_fe:+.2f}  (coin-in per player-month)")

    es_panel = build_event_study_panel(panel_m, app_events, max_pre=4, max_post=4)
    es_coefs = event_study(es_panel)
    pt = parallel_trends_f_test(es_panel)
    p6 = plot_event_study(es_coefs)
    _log(f"  Parallel trends F-test: F={pt['f_stat']}, p={pt['p_value']} "
         f"({'PASS' if pt.get('passes_at_05') else 'FAIL'} at α=0.05)")
    _log(f"  Event study chart: {Path(p6).name}")

    # ------------------------------------------------------------------
    # Section 4 — Outputs
    # ------------------------------------------------------------------
    print("\n[4/4] Writing outputs")

    SETTINGS.output_dir.mkdir(parents=True, exist_ok=True)
    excel_path = SETTINGS.output_dir / SETTINGS.excel_filename

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        daily.to_excel(writer, sheet_name="daily", index=False)
        player.to_excel(writer, sheet_name="player", index=False)
        panel.to_excel(writer, sheet_name="panel", index=False)
    _log(f"  Excel written: {excel_path}")

    elapsed = time.perf_counter() - t0
    print(f"\nPipeline complete in {elapsed:.1f}s")
    print(f"Figures: {SETTINGS.figures_dir}")
    print(f"Excel:   {excel_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="casino-analytics",
        description="Casino Player Analytics Pipeline CLI",
    )
    sub = parser.add_subparsers(dest="command")

    # generate
    gen_p = sub.add_parser("generate", help="Generate synthetic data")
    gen_p.add_argument("--seed", type=int, default=SETTINGS.synth_seed)
    gen_p.add_argument("--output-dir", type=Path, default=SETTINGS.data_dir, dest="output_dir")

    # run-all
    sub.add_parser("run-all", help="Run the full pipeline")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "run-all":
        cmd_run_all(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
