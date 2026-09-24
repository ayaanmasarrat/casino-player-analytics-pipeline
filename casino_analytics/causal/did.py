"""Difference-in-differences estimators and event-study parallel-trends test."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from casino_analytics.config import SETTINGS


def add_post_indicator(
    panel: pd.DataFrame,
    treatment_year: int,
    treatment_month: int,
) -> pd.DataFrame:
    """Add post (1/0) and treat_post interaction columns."""
    df = panel.copy()
    df["post"] = (
        (df["year"] > treatment_year) |
        ((df["year"] == treatment_year) & (df["month"] >= treatment_month))
    ).astype(np.int8)
    df["treat_post"] = (df["treated"] * df["post"]).astype(np.int8)
    return df


def did_basic(
    panel: pd.DataFrame,
    outcome: str = "coin_in",
    cluster_col: str | None = None,
) -> "statsmodels.regression.linear_model.RegressionResultsWrapper":  # noqa: F821
    """Basic two-way DiD: y ~ treated + post + treat_post, SEs clustered on player."""
    cc = cluster_col or SETTINGS.did_cluster_col
    formula = f"{outcome} ~ treated + post + treat_post"
    return smf.ols(formula, data=panel).fit(
        cov_type="cluster",
        cov_kwds={"groups": panel[cc]},
    )


def did_with_month_fe(
    panel: pd.DataFrame,
    outcome: str = "coin_in",
    cluster_col: str | None = None,
) -> "statsmodels.regression.linear_model.RegressionResultsWrapper":  # noqa: F821
    """DiD with calendar-month fixed effects absorbed, SEs clustered on player."""
    cc = cluster_col or SETTINGS.did_cluster_col
    df = panel.copy()
    df["period_label"] = (
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    )
    formula = f"{outcome} ~ treated + treat_post + C(period_label)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smf.ols(formula, data=df).fit(
            cov_type="cluster",
            cov_kwds={"groups": df[cc]},
        )


def build_event_study_panel(
    panel: pd.DataFrame,
    adoption: pd.DataFrame,
    max_pre: int = 6,
    max_post: int = 6,
) -> pd.DataFrame:
    """
    Add relative_period (months since adoption) and filter to the event window.
    Period -1 is dropped (reference category).
    """
    df = panel.copy()
    adp = adoption[["player_id", "adoption_date"]].copy()
    adp["adoption_year"] = pd.to_datetime(adp["adoption_date"]).dt.year
    adp["adoption_month"] = pd.to_datetime(adp["adoption_date"]).dt.month

    df = df.merge(adp[["player_id", "adoption_year", "adoption_month"]], on="player_id", how="inner")
    df["relative_period"] = (
        (df["year"] - df["adoption_year"]) * 12 +
        (df["month"] - df["adoption_month"])
    )
    df = df.loc[
        (df["relative_period"] >= -max_pre) &
        (df["relative_period"] <= max_post) &
        (df["relative_period"] != -1)
    ].copy()
    return df


def event_study(
    es_panel: pd.DataFrame,
    outcome: str = "coin_in",
    cluster_col: str | None = None,
) -> pd.DataFrame:
    """Run event-study regression; return tidy coefficient table with 95% CIs."""
    cc = cluster_col or SETTINGS.did_cluster_col
    df = es_panel.copy()
    df["rel_period_cat"] = pd.Categorical(df["relative_period"])
    formula = f"{outcome} ~ C(rel_period_cat)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.ols(formula, data=df).fit(
            cov_type="cluster",
            cov_kwds={"groups": df[cc]} if cc in df.columns else {},
        )

    rows = []
    ci = model.conf_int()
    for name in model.params.index:
        if "rel_period_cat" not in name:
            continue
        period_str = name.replace("C(rel_period_cat)[T.", "").rstrip("]")
        try:
            period = int(float(period_str))
        except ValueError:
            continue
        rows.append({
            "relative_period": period,
            "coef": round(model.params[name], 4),
            "se": round(model.bse[name], 4),
            "ci_lo": round(ci.loc[name, 0], 4),
            "ci_hi": round(ci.loc[name, 1], 4),
            "p_value": round(model.pvalues[name], 4),
        })

    return pd.DataFrame(rows).sort_values("relative_period").reset_index(drop=True)


def parallel_trends_f_test(es_panel: pd.DataFrame, outcome: str = "coin_in") -> dict:
    """Joint F-test of pre-treatment coefficients (H0: parallel trends holds)."""
    pre = es_panel.loc[es_panel["relative_period"] < 0].copy()
    pre["rel_period_cat"] = pd.Categorical(pre["relative_period"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.ols(f"{outcome} ~ C(rel_period_cat)", data=pre).fit()

    period_terms = [t for t in model.params.index if "rel_period_cat" in t]
    if not period_terms:
        return {"f_stat": np.nan, "p_value": np.nan, "passes_at_05": None}

    f_test = model.f_test(" = ".join([f"{t} = 0" for t in period_terms]))
    f_stat = float(f_test.fvalue.flat[0]) if hasattr(f_test.fvalue, "flat") else float(f_test.fvalue)
    p_val = float(f_test.pvalue)

    return {
        "f_stat": round(f_stat, 4),
        "p_value": round(p_val, 4),
        "df_numerator": len(period_terms),
        "passes_at_05": bool(p_val > 0.05),
    }
