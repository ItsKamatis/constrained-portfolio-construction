from __future__ import annotations

import numpy as np
import pandas as pd

from .factors import FACTOR_COLUMNS


def holdings_contribution(
    start_weights: pd.Series,
    asset_returns: pd.Series,
) -> pd.DataFrame:
    """Arithmetic one-period holdings contribution using start-of-period weights."""

    weights = start_weights[start_weights.abs() > 1e-12]
    missing = weights.index.difference(asset_returns.dropna().index)
    if len(missing):
        raise ValueError(f"held assets have missing returns: {missing.tolist()}")
    aligned = pd.concat(
        [weights.rename("start_weight"), asset_returns.reindex(weights.index).rename("asset_return")],
        axis=1,
    )
    aligned["gross_contribution"] = aligned["start_weight"] * aligned["asset_return"]
    aligned.index.name = "asset"
    return aligned


def portfolio_factor_exposure(
    weights: pd.Series,
    asset_exposures: pd.DataFrame,
) -> pd.Series:
    held = weights[weights.abs() > 1e-12]
    missing = held.index.difference(asset_exposures.index)
    if len(missing):
        raise ValueError(f"factor exposures missing held assets: {missing.tolist()}")
    factors = [column for column in FACTOR_COLUMNS if column in asset_exposures.columns]
    if not factors:
        raise ValueError("asset exposures contain no supported factors")
    return asset_exposures.loc[held.index, factors].multiply(held, axis=0).sum().rename(
        "exposure"
    )


def active_return_attribution(
    *,
    portfolio_gross_return: float,
    benchmark_return: float,
    transaction_cost: float,
    portfolio_exposure: pd.Series,
    benchmark_exposure: pd.Series,
    factor_returns: pd.Series,
    tolerance: float = 1e-10,
) -> pd.DataFrame:
    """Reconcile active net return into factors, specific return, and costs."""

    common = portfolio_exposure.index.intersection(benchmark_exposure.index).intersection(
        factor_returns.index
    )
    if common.empty:
        raise ValueError("no common factors for active attribution")
    active_exposure = portfolio_exposure.loc[common] - benchmark_exposure.loc[common]
    factor_contributions = active_exposure * factor_returns.loc[common]
    gross_active = float(portfolio_gross_return - benchmark_return)
    factor_total = float(factor_contributions.sum())
    specific = gross_active - factor_total
    cost_contribution = -float(transaction_cost)
    active_net = gross_active + cost_contribution
    rows = [
        {
            "component": f"factor:{factor}",
            "contribution": float(value),
            "active_exposure": float(active_exposure[factor]),
        }
        for factor, value in factor_contributions.items()
    ]
    rows.extend(
        [
            {"component": "specific", "contribution": specific, "active_exposure": np.nan},
            {
                "component": "transaction_cost",
                "contribution": cost_contribution,
                "active_exposure": np.nan,
            },
            {"component": "active_net_return", "contribution": active_net, "active_exposure": np.nan},
        ]
    )
    result = pd.DataFrame(rows)
    components = result.loc[result["component"] != "active_net_return", "contribution"].sum()
    if abs(float(components) - active_net) > tolerance:
        raise ValueError("active return attribution does not reconcile")
    return result


def aggregate_attribution(attribution: pd.DataFrame) -> pd.DataFrame:
    required = {"month", "method", "component", "contribution"}
    missing = required.difference(attribution.columns)
    if missing:
        raise ValueError(f"attribution missing columns: {sorted(missing)}")
    return (
        attribution.groupby(["method", "component"], as_index=False, observed=True)
        .agg(
            sum_monthly_arithmetic_contribution=("contribution", "sum"),
            months=("month", "nunique"),
        )
        .sort_values(["method", "component"])
    )
