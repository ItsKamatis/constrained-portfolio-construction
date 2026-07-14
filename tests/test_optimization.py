from __future__ import annotations

import itertools

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from constrained_portfolio.estimates import estimate_returns_and_covariance
from constrained_portfolio.optimization import (
    NoPositiveExcessReturn,
    OptimizationError,
    benchmark_relative,
    equal_weight,
    exact_cardinality_cvar,
    global_minimum_variance,
    historical_cvar,
    long_short_130_30_appendix,
    maximum_sharpe_frontier,
    mean_variance,
    validate_weights,
)


@pytest.fixture
def scenarios() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    common = rng.normal(0.004, 0.03, size=(48, 1))
    values = common + rng.normal(0.0, 0.02, size=(48, 8)) + np.linspace(-0.001, 0.002, 8)
    return pd.DataFrame(values, columns=list("ABCDEFGH"))


def test_full_curated_optimizer_matrix_satisfies_constraints(scenarios: pd.DataFrame) -> None:
    estimates = estimate_returns_and_covariance(scenarios)
    solutions = [
        equal_weight(scenarios.columns),
        global_minimum_variance(estimates.covariance, max_weight=0.30),
        mean_variance(estimates.expected_returns, estimates.covariance, max_weight=0.30),
        maximum_sharpe_frontier(
            estimates.expected_returns, estimates.covariance, max_weight=0.30, grid_size=7
        ),
        historical_cvar(scenarios, max_weight=0.30),
        exact_cardinality_cvar(
            scenarios, cardinality=4, minimum_weight=0.05, maximum_weight=0.35
        ),
        long_short_130_30_appendix(estimates.expected_returns, estimates.covariance),
    ]
    for solution in solutions[:-1]:
        assert solution.weights.sum() == pytest.approx(1.0, abs=5e-6)
    cardinality = solutions[-2]
    assert (cardinality.weights > 1e-7).sum() == 4
    appendix = solutions[-1]
    assert appendix.weights.clip(lower=0).sum() == pytest.approx(1.3, abs=5e-6)
    assert -appendix.weights.clip(upper=0).sum() == pytest.approx(0.3, abs=5e-6)


def test_benchmark_relative_validates_factor_tolerances_elementwise_and_union_turnover(
    scenarios: pd.DataFrame,
) -> None:
    estimates = estimate_returns_and_covariance(scenarios[["A", "B", "C"]])
    benchmark = pd.Series(1 / 3, index=["A", "B", "C"])
    prior = pd.Series({"A": 0.5, "B": 0.3, "X": 0.2})
    exposures = pd.DataFrame(
        {"strict": [1.0, 0.0, -1.0], "loose": [0.2, -0.1, 0.3]},
        index=["A", "B", "C"],
    )
    tolerances = pd.Series({"strict": 0.01, "loose": 0.25})
    solution = benchmark_relative(
        estimates.expected_returns,
        estimates.covariance,
        benchmark,
        prior.reindex(benchmark.index, fill_value=0.0),
        outside_prior_absolute_weight=0.2,
        factor_exposures=exposures,
        factor_active_tolerance=tolerances,
        tracking_error_limit=0.50,
        turnover_limit=0.40,
        max_weight=0.60,
    )
    active_factor = exposures.T @ (solution.weights - benchmark)
    assert (active_factor.abs() <= tolerances + 5e-6).all()
    union = solution.weights.index.union(prior.index)
    actual_one_way = 0.5 * (
        solution.weights.reindex(union, fill_value=0.0)
        - prior.reindex(union, fill_value=0.0)
    ).abs().sum()
    assert actual_one_way <= 0.40 + 5e-6
    assert solution.diagnostics["one_way_turnover"] == pytest.approx(actual_one_way)


def test_invalid_solver_weights_are_rejected() -> None:
    with pytest.raises(OptimizationError, match="residual"):
        validate_weights(pd.Series({"A": 0.8, "B": 0.8}))


def test_cvar_rejects_incomplete_scenarios(scenarios: pd.DataFrame) -> None:
    scenarios.loc[0, "A"] = np.nan
    with pytest.raises(ValueError, match="joint and complete"):
        historical_cvar(scenarios)


def test_maximum_sharpe_frontier_includes_region_near_gmv() -> None:
    expected = pd.Series({"high_risk": 0.10, "low_risk": 0.05})
    covariance = pd.DataFrame(
        [[1.0, 0.0], [0.0, 0.0001]],
        index=expected.index,
        columns=expected.index,
    )

    solution = maximum_sharpe_frontier(expected, covariance, grid_size=101)

    assert solution.weights["low_risk"] > 0.999
    assert solution.diagnostics["estimated_sharpe"] > 4.9
    assert solution.diagnostics["frontier_lower_target"] < expected.mean()


def test_maximum_sharpe_rejects_no_positive_expected_excess_return() -> None:
    expected = pd.Series({"low_vol": -0.01, "high_vol": -0.01})
    covariance = pd.DataFrame(
        [[0.0001, 0.0], [0.0, 1.0]],
        index=expected.index,
        columns=expected.index,
    )

    with pytest.raises(NoPositiveExcessReturn, match="no feasible portfolio"):
        maximum_sharpe_frontier(expected, covariance, risk_free_rate=0.0)


def test_cardinality_milp_matches_independent_enumerated_continuous_solutions() -> None:
    scenarios = pd.DataFrame(
        [
            [0.02, 0.01, -0.03, 0.00],
            [-0.04, 0.02, 0.01, -0.01],
            [0.01, -0.02, 0.03, 0.02],
            [-0.01, 0.00, -0.02, 0.04],
            [0.03, -0.01, 0.00, -0.03],
            [-0.02, 0.04, -0.01, 0.01],
        ],
        columns=list("ABCD"),
    )
    confidence = 0.80
    minimum_weight = 0.10
    maximum_weight = 0.90
    milp_solution = exact_cardinality_cvar(
        scenarios,
        cardinality=2,
        confidence=confidence,
        minimum_weight=minimum_weight,
        maximum_weight=maximum_weight,
    )

    enumerated_objectives: list[float] = []
    for selected in itertools.combinations(scenarios.columns, 2):
        values = scenarios.loc[:, selected].to_numpy()
        weights = cp.Variable(2)
        alpha = cp.Variable()
        excess = cp.Variable(len(values), nonneg=True)
        cvar = alpha + cp.sum(excess) / ((1 - confidence) * len(values))
        problem = cp.Problem(
            cp.Minimize(cvar),
            [
                cp.sum(weights) == 1,
                weights >= minimum_weight,
                weights <= maximum_weight,
                excess >= -values @ weights - alpha,
            ],
        )
        problem.solve(solver="CLARABEL")
        assert problem.status == cp.OPTIMAL
        enumerated_objectives.append(float(problem.value))

    assert milp_solution.objective == pytest.approx(min(enumerated_objectives), abs=1e-7)
