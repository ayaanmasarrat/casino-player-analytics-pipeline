"""
casino_analytics.causal.did
==============================
Difference-in-Differences (DiD) regression estimators and an event-study
parallel-trends test.

Models
------
1. **Basic DiD** — two-way interaction of ``post × treated``:
       y_it = α + β1·treated_i + β2·post_t + β3·(treated × post) + ε_it

2. **DiD with month fixed effects** — absorbs calendar-time variation:
       y_it = α_t + β1·treated_i + β3·(treated × post) + ε_it
       Standard errors clustered at ``player_id`` level.

3. **Event study** — one interaction dummy per relative time period to test
   the parallel-trends assumption in pre-treatment periods and trace the
   dynamic treatment effect post-adoption.

All regressions use ``statsmodels.formula.api``; clustering via
``cov_type="cluster"``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from casino_analytics.config import SETTINGS


def _make_panel(
    matched: pd.DataFrame,
    panel: pd.DataFrame,
    treatment_col: str = "treated",
) -> pd.DataFrame:
    """
    Merge treatment status onto the player-month panel for matched players.

    Only players present in ``matched`` are retained.

    Returns
    -------
    DataFrame with columns including ``treated``, ``post``, ``treat_post``,
    ``year``, ``month``, ``period_label``, ``coin_in``.
    """
    ids = matched[["player_id", treatment_col]].drop_duplicates("player_id")
    df = panel.merge(ids, on="player_id", how="inner")
    return df


def add_post_indicator(
    panel: pd.DataFrame,
    treatment_year: int,
    treatment_month: int,
) -> pd.DataFrame:
    """
    Add a ``post`` column (1 after treatment onset, 0 before).

    Parameters
    ----------
    treatment_year, treatment_month : int
        First post-treatment period.
    """
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
    """
    Estimate a basic two-way DiD regression.

    Expects ``treated``, ``post``, and ``treat_post`` columns.
    """
    cc = cluster_col or SETTINGS.did_cluster_col
    formula = f"{outcome} ~ treated + post + treat_post"
    model = smf.ols(formula, data=panel).fit(
        cov_type="cluster",
        cov_kwds={"groups": panel[cc]},
    )
    return model


def did_with_month_fe(
    panel: pd.DataFrame,
    outcome: str = "coin_in",
    cluster_col: str | None = None,
) -> "statsmodels.regression.linear_model.RegressionResultsWrapper":  # noqa: F821
    """
    DiD regression absorbing calendar-month fixed effects.

    Adds ``C(period_label)`` dummies; the ``post`` main effect is dropped
    (absorbed by month FE).
    """
    cc = cluster_col or SETTINGS.did_cluster_col
    df = panel.copy()
    df["period_label"] = (
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    )
    formula = f"{outcome} ~ treated + treat_post + C(period_label)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.ols(formula, data=df).fit(
            cov_type="cluster",
            cov_kwds={"groups": df[cc]},
        )
    return model


def build_event_study_panel(
    panel: pd.DataFrame,
    adoption: pd.DataFrame,
    max_pre: int = 6,
    max_post: int = 6,
) -> pd.DataFrame:
    """
    Build a relative-time panel for the event study.

    For each player-month, ``relative_period`` = months since adoption.
    Periods outside ``[-max_pre, max_post]`` are dropped.

    Parameters
    ----------
    panel : DataFrame
        Player-month panel with ``year`` and ``month``.
    adoption : DataFrame
        Must contain ``player_id`` and ``adoption_date``.
    max_pre, max_post : int
        Window around adoption to keep.
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
        (df["relative_period"] >= -max_pre) & (df["relative_period"] <= max_post)
    ].copy()
    # Omit relative_period == -1 as reference
    df = df.loc[df["relative_period"] != -1].copy()
    return df


def event_study(
    es_panel: pd.DataFrame,
    outcome: str = "coin_in",
    cluster_col: str | None = None,
) -> pd.DataFrame:
    """
    Run the event-study regression and return a tidy coefficient table.

    The regression includes dummies for each ``relative_period`` (reference
    period −1 is dropped).  Returns a DataFrame with columns:
    ``relative_period``, ``coef``, ``se``, ``ci_lo``, ``ci_hi``, ``p_value``.
    """
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
    for name, coef, se, pval, ci_lo, ci_hi in zip(
        model.params.index,
        model.params.values,
        model.bse.values,
        model.pvalues.values,
        model.conf_int().iloc[:, 0].values,
        model.conf_int().iloc[:, 1].values,
    ):
        if "rel_period_cat" not in name:
            continue
        period_str = name.replace("C(rel_period_cat)[T.", "").rstrip("]")
        try:
            period = int(float(period_str))
        except ValueError:
            continue
        rows.append({
            "relative_period": period,
            "coef": round(coef, 4),
            "se": round(se, 4),
            "ci_lo": round(ci_lo, 4),
            "ci_hi": round(ci_hi, 4),
            "p_value": round(pval, 4),
        })

    result = pd.DataFrame(rows).sort_values("relative_period").reset_index(drop=True)
    return result


def parallel_trends_f_test(
    es_panel: pd.DataFrame,
    outcome: str = "coin_in",
) -> dict:
    """
    Joint F-test of pre-treatment coefficients from the event study.

    Tests H0: all relative_period < 0 coefficients are jointly zero
    (the parallel-trends assumption).

    Returns
    -------
    dict with keys: f_stat, p_value, df_numerator, df_denominator, passes_at_05.
    """
    pre = es_panel.loc[es_panel["relative_period"] < 0].copy()
    pre["rel_period_cat"] = pd.Categorical(pre["relative_period"])

    formula = f"{outcome} ~ C(rel_period_cat)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.ols(formula, data=pre).fit()

    # Test all period dummies jointly
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
