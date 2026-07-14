from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class ReturnEstimates:
    expected_returns: pd.Series
    covariance: pd.DataFrame
    sample_expected_returns: pd.Series
    shrink_target: float
    mean_shrinkage: float
    covariance_shrinkage: float
    observations: int
    periods_per_year: int


def _complete_joint_window(returns: pd.DataFrame, min_observations: int) -> pd.DataFrame:
    numeric = returns.apply(pd.to_numeric, errors="coerce")
    complete = numeric.dropna(axis=0, how="any")
    if len(complete) < min_observations:
        raise ValueError(
            f"need {min_observations} complete joint scenarios; found {len(complete)}"
        )
    if complete.shape[1] < 2:
        raise ValueError("return estimation requires at least two assets")
    if not np.isfinite(complete.to_numpy()).all():
        raise ValueError("return window contains non-finite observations")
    return complete


def estimate_returns_and_covariance(
    returns: pd.DataFrame,
    *,
    mean_shrinkage: float = 0.50,
    periods_per_year: int = 12,
    min_observations: int = 18,
) -> ReturnEstimates:
    """Estimate annual moments using transparent mean and Ledoit-Wolf shrinkage.

    Expected asset returns are sample arithmetic means shrunk toward the
    cross-sectional grand mean. ``mean_shrinkage=0`` is the raw sample estimate;
    ``1`` gives every asset the same grand-mean estimate. Covariance is the
    Ledoit-Wolf constant-correlation-free shrinkage estimator implemented by
    scikit-learn and is annualized by ``periods_per_year``.
    """

    if not 0.0 <= mean_shrinkage <= 1.0:
        raise ValueError("mean_shrinkage must lie in [0, 1]")
    complete = _complete_joint_window(returns, min_observations)
    sample_monthly = complete.mean(axis=0)
    target = float(sample_monthly.mean())
    shrunk_monthly = (1.0 - mean_shrinkage) * sample_monthly + mean_shrinkage * target
    model = LedoitWolf(assume_centered=False).fit(complete.to_numpy(dtype=float))
    covariance = model.covariance_ * periods_per_year
    covariance = (covariance + covariance.T) / 2.0
    minimum_eigenvalue = float(np.linalg.eigvalsh(covariance).min())
    if minimum_eigenvalue < -1e-10:
        raise ValueError(f"Ledoit-Wolf covariance is not PSD: eigenvalue={minimum_eigenvalue}")
    columns = complete.columns
    return ReturnEstimates(
        expected_returns=(shrunk_monthly * periods_per_year).rename("expected_return"),
        covariance=pd.DataFrame(covariance, index=columns, columns=columns),
        sample_expected_returns=(sample_monthly * periods_per_year).rename(
            "sample_expected_return"
        ),
        shrink_target=target * periods_per_year,
        mean_shrinkage=float(mean_shrinkage),
        covariance_shrinkage=float(model.shrinkage_),
        observations=len(complete),
        periods_per_year=periods_per_year,
    )


def trailing_joint_window(
    returns: pd.DataFrame,
    decision_month: str | pd.Timestamp,
    *,
    lookback_months: int = 36,
    min_observations: int = 18,
    require_previous_month: bool = True,
) -> pd.DataFrame:
    """Select assets and joint scenarios using information strictly before a decision."""

    decision = pd.Timestamp(decision_month).to_period("M").to_timestamp("M")
    history = returns.loc[returns.index < decision].sort_index().tail(lookback_months)
    if history.empty:
        raise ValueError("no observations precede the decision month")
    counts = history.notna().sum()
    eligible = counts.index[counts.ge(min_observations)]
    if require_previous_month:
        previous = decision - pd.offsets.MonthEnd(1)
        if previous not in history.index:
            raise ValueError(f"previous calendar month {previous.date()} is absent")
        eligible = eligible.intersection(history.columns[history.loc[previous].notna()])
    if len(eligible) < 2:
        raise ValueError("fewer than two assets satisfy lagged eligibility")
    joint = history.loc[:, eligible].dropna(axis=0, how="any")
    if len(joint) < min_observations:
        # Preserve actual scenarios: iteratively remove the asset responsible for
        # the most missing observations; never replace missing returns with zero.
        candidates = list(eligible)
        while len(candidates) > 2:
            candidate_frame = history.loc[:, candidates]
            complete_count = len(candidate_frame.dropna(axis=0, how="any"))
            if complete_count >= min_observations:
                break
            missing = candidate_frame.isna().sum()
            candidates.remove(missing.sort_values(ascending=False).index[0])
        joint = history.loc[:, candidates].dropna(axis=0, how="any")
    if len(joint) < min_observations:
        raise ValueError(
            f"only {len(joint)} complete joint scenarios remain; need {min_observations}"
        )
    return joint
