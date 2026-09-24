"""Propensity-score matching utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from casino_analytics.config import SETTINGS


def estimate_propensity(
    df: pd.DataFrame,
    feature_cols: list[str],
    treatment_col: str = "treated",
    random_state: int | None = None,
) -> pd.DataFrame:
    """Fit logistic regression on feature_cols and append a propensity column."""
    rs = random_state if random_state is not None else SETTINGS.psm_random_state
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[treatment_col].to_numpy(dtype=int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=1_000, random_state=rs, solver="lbfgs")
    lr.fit(X_scaled, y)

    out = df.copy()
    out["propensity"] = lr.predict_proba(X_scaled)[:, 1]
    return out


def trim_common_support(
    df: pd.DataFrame,
    treatment_col: str = "treated",
    trim_pct: float = 0.01,
) -> pd.DataFrame:
    """Drop observations outside the common support region (trim bottom/top trim_pct tails)."""
    treated_scores = df.loc[df[treatment_col] == 1, "propensity"]
    lo = treated_scores.quantile(trim_pct)
    hi = treated_scores.quantile(1 - trim_pct)
    return df.loc[(df["propensity"] >= lo) & (df["propensity"] <= hi)].copy()


def match_nearest_neighbour(
    df: pd.DataFrame,
    treatment_col: str = "treated",
    n_controls: int | None = None,
    caliper: float | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    1:N nearest-neighbour matching with a propensity caliper (no replacement).
    Uses binary search on sorted control propensities for efficiency.
    """
    nc = n_controls if n_controls is not None else SETTINGS.psm_n_controls
    cal = caliper if caliper is not None else SETTINGS.psm_caliper
    rs = random_state if random_state is not None else SETTINGS.psm_random_state

    rng = np.random.default_rng(rs)

    treated = df.loc[df[treatment_col] == 1].copy()
    controls = df.loc[df[treatment_col] == 0].copy()

    ctrl_sorted = controls.sort_values("propensity").reset_index(drop=True)
    ctrl_props = ctrl_sorted["propensity"].to_numpy()

    used_indices: set[int] = set()
    records: list[pd.DataFrame] = []
    group_id = 0

    treated_shuffled = treated.sample(frac=1, random_state=rs)

    for _, t_row in treated_shuffled.iterrows():
        p = t_row["propensity"]
        lo_idx = int(np.searchsorted(ctrl_props, p - cal, side="left"))
        hi_idx = int(np.searchsorted(ctrl_props, p + cal, side="right"))

        candidates = [i for i in range(lo_idx, hi_idx) if i not in used_indices]
        if not candidates:
            continue

        chosen = rng.choice(candidates, size=min(nc, len(candidates)), replace=False).tolist()
        for idx in chosen:
            used_indices.add(idx)

        matched_controls = ctrl_sorted.iloc[chosen].copy()
        matched_controls["match_group"] = group_id

        t_df = t_row.to_frame().T.copy()
        t_df["match_group"] = group_id

        records.append(t_df)
        records.append(matched_controls)
        group_id += 1

    if not records:
        return pd.DataFrame(columns=df.columns.tolist() + ["match_group"])

    return pd.concat(records, ignore_index=True)


def standardised_difference(treated_vals: pd.Series, control_vals: pd.Series) -> float:
    """Standardised mean difference between treated and control groups."""
    mean_diff = treated_vals.mean() - control_vals.mean()
    pooled_sd = np.sqrt((treated_vals.std() ** 2 + control_vals.std() ** 2) / 2)
    if pooled_sd == 0:
        return float("nan")
    return mean_diff / pooled_sd


def balance_table(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    covariates: list[str],
    treatment_col: str = "treated",
) -> pd.DataFrame:
    """Return a DataFrame with SMD before and after matching for each covariate."""
    rows = []
    for cov in covariates:
        if cov not in before_df.columns:
            continue
        t_b = before_df.loc[before_df[treatment_col] == 1, cov].dropna()
        c_b = before_df.loc[before_df[treatment_col] == 0, cov].dropna()
        t_a = after_df.loc[after_df[treatment_col] == 1, cov].dropna()
        c_a = after_df.loc[after_df[treatment_col] == 0, cov].dropna()
        rows.append({
            "covariate": cov,
            "std_diff_before": round(standardised_difference(t_b, c_b), 4),
            "std_diff_after": round(standardised_difference(t_a, c_a), 4),
        })
    return pd.DataFrame(rows)
