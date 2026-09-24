"""
casino_analytics.plots
=========================
Chart helpers that write PNG files to ``SETTINGS.figures_dir``.

All functions accept DataFrames produced by earlier pipeline stages
and return the figure path (str) for logging/notebook display.

Matplotlib is used throughout; no Seaborn dependency.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe in CLI and CI
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from casino_analytics.config import SETTINGS


def _savefig(fig: plt.Figure, name: str, out_dir: Path | None = None) -> str:
    """Save figure and return absolute path string."""
    out = out_dir or SETTINGS.figures_dir
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_mau_trend(mau: pd.DataFrame, out_dir: Path | None = None) -> str:
    """
    Line chart of total MAU over time.

    Parameters
    ----------
    mau : DataFrame
        Aggregated (no segment) MAU DataFrame from ``analytics.trends``.
    """
    df = mau.copy()
    if "l1_segment" in df.columns:
        df = df.groupby("period", observed=True).agg(mau=("mau", "sum")).reset_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["period"], df["mau"], marker="o", linewidth=1.8, color="#2563EB")
    ax.set_title("Monthly Active Players", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Active Players")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    return _savefig(fig, "mau_trend.png", out_dir)


def plot_mau_by_segment(mau_seg: pd.DataFrame, out_dir: Path | None = None) -> str:
    """
    Stacked area chart of MAU broken out by L1 segment.
    """
    if "l1_segment" not in mau_seg.columns:
        raise ValueError("mau_seg must contain l1_segment column")

    pivot = (
        mau_seg
        .groupby(["period", "l1_segment"], observed=True)
        .agg(mau=("mau", "sum"))
        .reset_index()
        .pivot(index="period", columns="l1_segment", values="mau")
        .fillna(0)
    )
    # Order columns so High is at top
    ordered = [c for c in ("High", "Medium", "Low") if c in pivot.columns]
    pivot = pivot[ordered]

    fig, ax = plt.subplots(figsize=(10, 4))
    pivot.plot.area(ax=ax, alpha=0.75, colormap="Blues_r")
    ax.set_title("MAU by L1 Segment (Value Tier)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Active Players")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(title="Segment")
    fig.tight_layout()
    return _savefig(fig, "mau_by_segment.png", out_dir)


def plot_age_distribution(player: pd.DataFrame, out_dir: Path | None = None) -> str:
    """
    Horizontal bar chart of player count by age bucket.
    """
    if "age_bucket" not in player.columns:
        raise ValueError("player must contain age_bucket column")

    counts = (
        player["age_bucket"]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(counts.index.astype(str), counts.values, color="#1D4ED8")
    ax.set_title("Player Age Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Players")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    return _savefig(fig, "age_distribution.png", out_dir)


def plot_propensity_overlap(
    df_matched: pd.DataFrame,
    out_dir: Path | None = None,
) -> str:
    """
    Histogram of propensity scores for treated vs control players.

    Expects ``df_matched`` to have columns ``propensity`` and ``treated``
    (1 / 0).
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 40)
    treated = df_matched.loc[df_matched["treated"] == 1, "propensity"]
    control = df_matched.loc[df_matched["treated"] == 0, "propensity"]
    ax.hist(control.values, bins=bins, alpha=0.55, label="Control", color="#6B7280")
    ax.hist(treated.values, bins=bins, alpha=0.55, label="Treated", color="#2563EB")
    ax.set_title("Propensity Score Overlap (Common Support)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Propensity Score")
    ax.set_ylabel("Players")
    ax.legend()
    fig.tight_layout()
    return _savefig(fig, "propensity_overlap.png", out_dir)


def plot_balance(balance: pd.DataFrame, out_dir: Path | None = None) -> str:
    """
    Love plot (dot plot) showing standardised differences before and after
    matching.

    Expects ``balance`` to have columns: ``covariate``,
    ``std_diff_before``, ``std_diff_after``.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    y = range(len(balance))
    ax.scatter(balance["std_diff_before"], y, marker="o", color="#6B7280", label="Before", zorder=3)
    ax.scatter(balance["std_diff_after"], y, marker="D", color="#2563EB", label="After", zorder=3)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(0.1, color="red", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.axvline(-0.1, color="red", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.set_yticks(list(y))
    ax.set_yticklabels(balance["covariate"].tolist(), fontsize=9)
    ax.set_xlabel("Standardised Mean Difference")
    ax.set_title("Covariate Balance Before vs After Matching", fontsize=12, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return _savefig(fig, "balance_plot.png", out_dir)


def plot_event_study(event_study: pd.DataFrame, out_dir: Path | None = None) -> str:
    """
    Event-study coefficient plot with 95 % confidence intervals.

    Expects ``event_study`` to have columns: ``relative_period``,
    ``coef``, ``ci_lo``, ``ci_hi``.
    """
    df = event_study.copy().sort_values("relative_period")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.errorbar(
        df["relative_period"],
        df["coef"],
        yerr=[df["coef"] - df["ci_lo"], df["ci_hi"] - df["coef"]],
        fmt="o",
        capsize=4,
        color="#2563EB",
        ecolor="#93C5FD",
        linewidth=1.5,
        label="DiD coefficient ± 95% CI",
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(-0.5, color="grey", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.set_xlabel("Months relative to treatment")
    ax.set_ylabel("Effect on coin-in")
    ax.set_title("Event Study: Parallel Trends & Treatment Effect", fontsize=12, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return _savefig(fig, "event_study.png", out_dir)
