from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .attribution import (
    active_return_attribution,
    holdings_contribution,
    portfolio_factor_exposure,
)
from .data import split_label
from .estimates import estimate_returns_and_covariance, trailing_joint_window
from .factors import FACTOR_COLUMNS, estimate_factor_exposures
from .optimization import (
    NoPositiveExcessReturn,
    PortfolioSolution,
    benchmark_relative,
    equal_weight,
    exact_cardinality_cvar,
    global_minimum_variance,
    historical_cvar,
    long_short_130_30_appendix,
    maximum_sharpe_frontier,
    mean_variance,
)


CORE_METHODS = (
    "equal_weight",
    "global_minimum_variance",
    "shrinkage_mean_variance",
    "maximum_sharpe_with_gmv_fallback",
    "historical_joint_cvar",
    "benchmark_relative_constrained",
    "exact_cardinality_cvar",
)
APPENDIX_METHOD = "130_30_mathematical_appendix"


@dataclass(frozen=True)
class WalkForwardConfig:
    start: str = "2020-01-31"
    end: str = "2025-12-31"
    lookback_months: int = 36
    minimum_monthly_scenarios: int = 18
    mean_shrinkage: float = 0.50
    max_weight: float = 0.20
    risk_aversion: float = 4.0
    cvar_confidence: float = 0.95
    cardinality: int = 8
    cardinality_minimum_weight: float = 0.01
    cardinality_maximum_weight: float = 0.25
    tracking_error_limit: float = 0.08
    turnover_limit: float = 0.30
    factor_active_tolerance: float = 0.15
    turnover_penalty: float = 0.001
    transaction_cost_bps_per_one_way_turnover: float = 10.0
    frontier_points: int = 15
    factor_window_months: int = 60
    factor_minimum_observations: int = 18


@dataclass(frozen=True)
class BacktestResult:
    periods: pd.DataFrame
    weights: pd.DataFrame
    holdings_attribution: pd.DataFrame
    active_attribution: pd.DataFrame
    optimizer_diagnostics: pd.DataFrame
    config: WalkForwardConfig


def _validate_inputs(
    returns: pd.DataFrame,
    eligibility: pd.DataFrame | None,
    config: WalkForwardConfig,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DatetimeIndex]:
    frame = returns.copy().sort_index()
    frame.index = pd.DatetimeIndex(frame.index).to_period("M").to_timestamp("M")
    if frame.index.duplicated().any() or not frame.columns.is_unique:
        raise ValueError("return matrix must have unique month and asset labels")
    frame = frame.apply(pd.to_numeric, errors="coerce")
    if np.isinf(frame.to_numpy()).any():
        raise ValueError("return matrix contains infinite values")
    start = pd.Timestamp(config.start).to_period("M").to_timestamp("M")
    end = pd.Timestamp(config.end).to_period("M").to_timestamp("M")
    months = frame.index[(frame.index >= start) & (frame.index <= end)]
    if months.empty:
        raise ValueError("no holding months fall inside the requested study interval")
    mask = None
    if eligibility is not None:
        mask = eligibility.copy()
        mask.index = pd.DatetimeIndex(mask.index).to_period("M").to_timestamp("M")
        mask = mask.reindex(index=frame.index, columns=frame.columns, fill_value=False).astype(bool)
        missing_months = months.difference(mask.index)
        if len(missing_months):
            raise ValueError(f"eligibility is missing study months: {missing_months.tolist()}")
    return frame, mask, months


def _prior_on_current_universe(
    prior: pd.Series | None,
    benchmark: pd.Series,
) -> tuple[pd.Series, float]:
    if prior is None:
        raise ValueError("prior-on-universe conversion requires an existing portfolio")
    current = prior.reindex(benchmark.index, fill_value=0.0)
    removed = float(prior.loc[~prior.index.isin(benchmark.index)].abs().sum())
    return current, removed


def _drift_weights(weights: pd.Series, returns: pd.Series, gross_return: float) -> pd.Series:
    denominator = 1.0 + float(gross_return)
    if denominator <= 0:
        raise ValueError("portfolio wealth is non-positive after the holding-period return")
    drifted = weights * (1.0 + returns.reindex(weights.index)) / denominator
    if drifted.isna().any():
        raise ValueError("cannot drift weights across missing held-asset returns")
    if abs(float(drifted.sum()) - 1.0) > 1e-8:
        raise ValueError("drifted weights do not sum to one")
    return drifted


def _trade_statistics(
    target: pd.Series, prior: pd.Series | None
) -> tuple[float, float, bool]:
    if prior is None:
        # Inception starts from cash: absolute target notionals capture 100% long
        # investment (and 160% gross for the 130/30 appendix).
        traded_notional = float(target.abs().sum())
        return traded_notional, np.nan, True
    else:
        union = target.index.union(prior.index)
        traded_notional = float(
            (target.reindex(union, fill_value=0.0) - prior.reindex(union, fill_value=0.0))
            .abs()
            .sum()
        )
    return traded_notional, 0.5 * traded_notional, False


def _solution_for_method(
    method: str,
    joint_scenarios: pd.DataFrame,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    benchmark: pd.Series,
    prior: pd.Series | None,
    factor_exposures: pd.DataFrame | None,
    risk_free_rate_annual: float,
    config: WalkForwardConfig,
) -> tuple[PortfolioSolution, float]:
    if method == "equal_weight":
        return equal_weight(joint_scenarios.columns), 0.0
    if method == "global_minimum_variance":
        return global_minimum_variance(covariance, max_weight=config.max_weight), 0.0
    if method == "shrinkage_mean_variance":
        return (
            mean_variance(
                expected_returns,
                covariance,
                risk_aversion=config.risk_aversion,
                max_weight=config.max_weight,
            ),
            0.0,
        )
    if method == "maximum_sharpe_with_gmv_fallback":
        try:
            solution = maximum_sharpe_frontier(
                expected_returns,
                covariance,
                risk_free_rate=risk_free_rate_annual,
                max_weight=config.max_weight,
                grid_size=config.frontier_points,
            )
        except NoPositiveExcessReturn:
            gmv = global_minimum_variance(covariance, max_weight=config.max_weight)
            diagnostics = {
                **dict(gmv.diagnostics),
                "allocation_rule": "gmv_fallback_no_positive_expected_excess_return",
                "risk_free_rate_annual": risk_free_rate_annual,
            }
            solution = PortfolioSolution(
                "maximum_sharpe_with_gmv_fallback",
                gmv.weights,
                gmv.status,
                gmv.objective,
                gmv.solver,
                diagnostics,
            )
        return solution, 0.0
    if method == "historical_joint_cvar":
        return (
            historical_cvar(
                joint_scenarios,
                confidence=config.cvar_confidence,
                max_weight=config.max_weight,
            ),
            0.0,
        )
    if method == "benchmark_relative_constrained":
        if factor_exposures is None:
            raise ValueError("benchmark-relative factor constraints require factor data")
        if prior is None:
            solver_prior = None
            outside_prior_absolute_weight = 0.0
        else:
            solver_prior, outside_prior_absolute_weight = _prior_on_current_universe(
                prior, benchmark
            )
        return (
            benchmark_relative(
                expected_returns,
                covariance,
                benchmark,
                solver_prior,
                factor_exposures=factor_exposures[list(FACTOR_COLUMNS)],
                factor_active_tolerance=config.factor_active_tolerance,
                tracking_error_limit=config.tracking_error_limit,
                turnover_limit=config.turnover_limit,
                max_weight=config.max_weight,
                risk_aversion=config.risk_aversion / 2.0,
                transaction_cost_penalty=config.turnover_penalty,
                outside_prior_absolute_weight=outside_prior_absolute_weight,
            ),
            outside_prior_absolute_weight,
        )
    if method == "exact_cardinality_cvar":
        cardinality = min(config.cardinality, joint_scenarios.shape[1])
        return (
            exact_cardinality_cvar(
                joint_scenarios,
                cardinality=cardinality,
                confidence=config.cvar_confidence,
                minimum_weight=config.cardinality_minimum_weight,
                maximum_weight=config.cardinality_maximum_weight,
            ),
            0.0,
        )
    if method == APPENDIX_METHOD:
        return (
            long_short_130_30_appendix(
                expected_returns,
                covariance,
                risk_aversion=config.risk_aversion,
            ),
            0.0,
        )
    raise ValueError(f"unknown method: {method}")


def run_walk_forward(
    returns: pd.DataFrame,
    *,
    factors: pd.DataFrame | None = None,
    eligibility: pd.DataFrame | None = None,
    methods: tuple[str, ...] | list[str] = CORE_METHODS,
    include_130_30_appendix: bool = False,
    config: WalkForwardConfig | None = None,
) -> BacktestResult:
    """Run lagged monthly construction, drift, costs, and attribution."""

    settings = config or WalkForwardConfig()
    frame, eligibility_mask, months = _validate_inputs(returns, eligibility, settings)
    selected_methods = list(methods)
    if include_130_30_appendix and APPENDIX_METHOD not in selected_methods:
        selected_methods.append(APPENDIX_METHOD)
    if not selected_methods or len(set(selected_methods)) != len(selected_methods):
        raise ValueError("methods must be a nonempty unique list")
    if "benchmark_relative_constrained" in selected_methods and factors is None:
        raise ValueError("factor data are required for the factor-constrained optimizer")
    factor_frame = None
    if factors is not None:
        factor_frame = factors.copy().sort_index()
        factor_frame.index = pd.DatetimeIndex(factor_frame.index).to_period("M").to_timestamp("M")

    states: dict[str, pd.Series] = {}
    period_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    holdings_rows: list[dict[str, object]] = []
    active_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []

    for month in months:
        eligible_columns = frame.columns
        if eligibility_mask is not None:
            eligible_columns = frame.columns[eligibility_mask.loc[month].to_numpy()]
        if len(eligible_columns) < 2:
            raise ValueError(f"fewer than two eligible assets at {month.date()}")
        joint = trailing_joint_window(
            frame.loc[:, eligible_columns],
            month,
            lookback_months=settings.lookback_months,
            min_observations=settings.minimum_monthly_scenarios,
        )
        if joint.index.max() >= month:
            raise AssertionError("lookahead violation: estimation includes the holding month")
        estimates = estimate_returns_and_covariance(
            joint,
            mean_shrinkage=settings.mean_shrinkage,
            min_observations=settings.minimum_monthly_scenarios,
        )
        benchmark = equal_weight(joint.columns).weights
        asset_factor_exposures = None
        if factor_frame is not None:
            asset_factor_exposures = estimate_factor_exposures(
                frame.loc[:, joint.columns],
                factor_frame,
                month,
                window_months=settings.factor_window_months,
                min_observations=settings.factor_minimum_observations,
            )
            missing_exposures = joint.columns.difference(asset_factor_exposures.index)
            if len(missing_exposures):
                raise ValueError(
                    f"factor regressions unavailable for eligible assets: {missing_exposures.tolist()}"
                )
            if "RF" not in factor_frame.columns:
                raise ValueError("factor data must include RF for risk-free-rate handling")
            lagged_rf = pd.to_numeric(factor_frame.reindex(joint.index)["RF"], errors="coerce")
            if lagged_rf.isna().any():
                raise ValueError("factor RF is missing in the lagged construction window")
            if month not in factor_frame.index or pd.isna(factor_frame.at[month, "RF"]):
                raise ValueError(f"factor RF is missing for holding month {month.date()}")
            construction_risk_free_rate = float(lagged_rf.mean() * 12.0)
            realized_risk_free_return = float(factor_frame.at[month, "RF"])
        else:
            construction_risk_free_rate = 0.0
            realized_risk_free_return = 0.0

        holding_returns = frame.loc[month]
        benchmark_contribution = holdings_contribution(benchmark, holding_returns)
        benchmark_return = float(benchmark_contribution["gross_contribution"].sum())

        for method in selected_methods:
            prior = states.get(method)
            solution, forced_turnover = _solution_for_method(
                method,
                joint,
                estimates.expected_returns,
                estimates.covariance,
                benchmark,
                prior,
                asset_factor_exposures,
                construction_risk_free_rate,
                settings,
            )
            target = solution.weights
            holdings = holdings_contribution(target, holding_returns)
            gross_return = float(holdings["gross_contribution"].sum())
            traded_notional, one_way_turnover, inception = _trade_statistics(target, prior)
            cost_base = traded_notional if inception else one_way_turnover
            transaction_cost = (
                cost_base
                * settings.transaction_cost_bps_per_one_way_turnover
                / 10_000.0
            )
            if (
                method == "benchmark_relative_constrained"
                and not inception
                and one_way_turnover > settings.turnover_limit + 5e-6
            ):
                raise ValueError("realized union turnover exceeds the optimizer limit")
            net_return = gross_return - transaction_cost
            end_weights = _drift_weights(target, holding_returns, gross_return)
            states[method] = end_weights
            period_rows.append(
                {
                    "month": month,
                    "split": split_label(month),
                    "method": method,
                    "eligible_assets": len(joint.columns),
                    "estimation_start": joint.index.min(),
                    "estimation_end": joint.index.max(),
                    "joint_scenarios": len(joint),
                    "gross_return": gross_return,
                    "transaction_cost": transaction_cost,
                    "net_return": net_return,
                    "benchmark_return": benchmark_return,
                    "active_net_return": net_return - benchmark_return,
                    "construction_rule": str(
                        solution.diagnostics.get("allocation_rule", solution.method)
                    ),
                    "construction_risk_free_rate_annual": construction_risk_free_rate,
                    "realized_risk_free_return": realized_risk_free_return,
                    "traded_notional": traded_notional,
                    "one_way_turnover": one_way_turnover,
                    "inception": inception,
                    "forced_universe_turnover": forced_turnover,
                }
            )
            union = target.index.union(prior.index if prior is not None else pd.Index([]))
            for asset in union:
                weight_rows.append(
                    {
                        "month": month,
                        "method": method,
                        "asset": asset,
                        "pretrade_weight": float(prior.get(asset, 0.0)) if prior is not None else 0.0,
                        "target_weight": float(target.get(asset, 0.0)),
                        "end_weight": float(end_weights.get(asset, 0.0)),
                    }
                )
            for asset, row in holdings.iterrows():
                holdings_rows.append(
                    {
                        "month": month,
                        "method": method,
                        "asset": asset,
                        **row.to_dict(),
                    }
                )
            diagnostic_rows.append(
                {
                    "month": month,
                    "method": method,
                    "status": solution.status,
                    "solver": solution.solver,
                    "objective": solution.objective,
                    "mean_shrinkage": estimates.mean_shrinkage,
                    "covariance_shrinkage": estimates.covariance_shrinkage,
                    "shrink_target": estimates.shrink_target,
                    "diagnostics_json": json.dumps(dict(solution.diagnostics), sort_keys=True),
                }
            )

            if factor_frame is not None and month in factor_frame.index:
                assert asset_factor_exposures is not None
                portfolio_exposure = portfolio_factor_exposure(target, asset_factor_exposures)
                benchmark_exposure = portfolio_factor_exposure(benchmark, asset_factor_exposures)
                attribution = active_return_attribution(
                    portfolio_gross_return=gross_return,
                    benchmark_return=benchmark_return,
                    transaction_cost=transaction_cost,
                    portfolio_exposure=portfolio_exposure,
                    benchmark_exposure=benchmark_exposure,
                    factor_returns=factor_frame.loc[month],
                )
                for _, row in attribution.iterrows():
                    active_rows.append({"month": month, "method": method, **row.to_dict()})

    return BacktestResult(
        periods=pd.DataFrame(period_rows),
        weights=pd.DataFrame(weight_rows),
        holdings_attribution=pd.DataFrame(holdings_rows),
        active_attribution=pd.DataFrame(active_rows),
        optimizer_diagnostics=pd.DataFrame(diagnostic_rows),
        config=settings,
    )
