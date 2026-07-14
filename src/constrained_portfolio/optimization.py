from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


SOLVER_TOLERANCE = 5e-6


@dataclass(frozen=True)
class PortfolioSolution:
    method: str
    weights: pd.Series
    status: str
    objective: float
    solver: str
    diagnostics: Mapping[str, float | int | str] = field(default_factory=dict)


class OptimizationError(RuntimeError):
    pass


class NoPositiveExcessReturn(OptimizationError):
    """Raised when a fully invested maximum-Sharpe problem has no positive forecast excess return."""


def _aligned_moments(
    expected_returns: pd.Series, covariance: pd.DataFrame
) -> tuple[pd.Index, np.ndarray, np.ndarray]:
    assets = pd.Index(expected_returns.index)
    if assets.empty or not assets.is_unique:
        raise ValueError("expected return index must be nonempty and unique")
    if set(assets) != set(covariance.index) or set(assets) != set(covariance.columns):
        raise ValueError("expected returns and covariance assets do not match")
    matrix = covariance.loc[assets, assets].to_numpy(dtype=float)
    matrix = (matrix + matrix.T) / 2.0
    mu = expected_returns.loc[assets].to_numpy(dtype=float)
    if not np.isfinite(mu).all() or not np.isfinite(matrix).all():
        raise ValueError("moments contain non-finite values")
    if np.linalg.eigvalsh(matrix).min() < -1e-8:
        raise ValueError("covariance must be positive semidefinite")
    return assets, mu, matrix


def _solve(problem: cp.Problem, *, conic: bool = False) -> tuple[str, str]:
    preferred = ["CLARABEL", "SCS"] if conic else ["CLARABEL", "OSQP", "SCS"]
    installed = set(cp.installed_solvers())
    errors: list[str] = []
    inaccurate_snapshot: tuple[str, list[np.ndarray | None]] | None = None
    for solver in preferred:
        if solver not in installed:
            continue
        try:
            kwargs: dict[str, object] = {"solver": solver, "verbose": False}
            if solver == "SCS":
                kwargs.update({"eps": 1e-7, "max_iters": 100_000})
            problem.solve(**kwargs)
        except cp.error.SolverError as exc:
            errors.append(f"{solver}: {exc}")
            continue
        if problem.status == cp.OPTIMAL:
            return problem.status, solver
        if problem.status == cp.OPTIMAL_INACCURATE:
            inaccurate_snapshot = (
                solver,
                [None if variable.value is None else np.array(variable.value, copy=True) for variable in problem.variables()],
            )
            errors.append(f"{solver}: {problem.status}; trying another solver")
            continue
        errors.append(f"{solver}: {problem.status}")
    if inaccurate_snapshot is not None:
        solver, values = inaccurate_snapshot
        for variable, value in zip(problem.variables(), values, strict=True):
            variable.value = value
        return cp.OPTIMAL_INACCURATE, solver
    raise OptimizationError("no continuous solver produced a solution: " + "; ".join(errors))


def validate_weights(
    weights: pd.Series,
    *,
    lower_bound: float | pd.Series = 0.0,
    upper_bound: float | pd.Series = 1.0,
    expected_sum: float = 1.0,
    tolerance: float = SOLVER_TOLERANCE,
) -> dict[str, float]:
    values = pd.to_numeric(weights, errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy()).all():
        raise OptimizationError("optimizer returned non-finite weights")
    lower = (
        pd.Series(float(lower_bound), index=values.index)
        if np.isscalar(lower_bound)
        else pd.Series(lower_bound).reindex(values.index)
    )
    upper = (
        pd.Series(float(upper_bound), index=values.index)
        if np.isscalar(upper_bound)
        else pd.Series(upper_bound).reindex(values.index)
    )
    diagnostics = {
        "budget_residual": abs(float(values.sum()) - expected_sum),
        "lower_bound_residual": float(np.maximum(lower.to_numpy() - values.to_numpy(), 0).max()),
        "upper_bound_residual": float(np.maximum(values.to_numpy() - upper.to_numpy(), 0).max()),
    }
    if max(diagnostics.values()) > tolerance:
        raise OptimizationError(f"solver residual validation failed: {diagnostics}")
    return diagnostics


def equal_weight(assets: pd.Index | list[object]) -> PortfolioSolution:
    index = pd.Index(assets, name="asset")
    if index.empty or not index.is_unique:
        raise ValueError("equal weight requires a nonempty unique asset universe")
    weights = pd.Series(1.0 / len(index), index=index, name="weight")
    diagnostics = validate_weights(weights)
    return PortfolioSolution("equal_weight", weights, "analytic", 0.0, "analytic", diagnostics)


def global_minimum_variance(
    covariance: pd.DataFrame,
    *,
    max_weight: float = 1.0,
) -> PortfolioSolution:
    expected = pd.Series(0.0, index=covariance.index)
    assets, _, sigma = _aligned_moments(expected, covariance)
    if max_weight * len(assets) < 1.0 - 1e-12:
        raise ValueError("max_weight makes the budget infeasible")
    w = cp.Variable(len(assets))
    problem = cp.Problem(
        cp.Minimize(cp.quad_form(w, cp.psd_wrap(sigma))),
        [cp.sum(w) == 1, w >= 0, w <= max_weight],
    )
    status, solver = _solve(problem)
    weights = pd.Series(np.asarray(w.value).ravel(), index=assets, name="weight")
    diagnostics = validate_weights(weights, upper_bound=max_weight)
    diagnostics["annual_variance"] = float(weights @ covariance.loc[assets, assets] @ weights)
    return PortfolioSolution(
        "global_minimum_variance", weights, status, float(problem.value), solver, diagnostics
    )


def mean_variance(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    *,
    risk_aversion: float = 4.0,
    max_weight: float = 1.0,
) -> PortfolioSolution:
    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive")
    assets, mu, sigma = _aligned_moments(expected_returns, covariance)
    if max_weight * len(assets) < 1.0 - 1e-12:
        raise ValueError("max_weight makes the budget infeasible")
    w = cp.Variable(len(assets))
    utility = mu @ w - risk_aversion * cp.quad_form(w, cp.psd_wrap(sigma))
    problem = cp.Problem(cp.Maximize(utility), [cp.sum(w) == 1, w >= 0, w <= max_weight])
    status, solver = _solve(problem)
    weights = pd.Series(np.asarray(w.value).ravel(), index=assets, name="weight")
    diagnostics = validate_weights(weights, upper_bound=max_weight)
    diagnostics.update(
        {
            "expected_return": float(weights @ expected_returns.loc[assets]),
            "annual_variance": float(weights @ covariance.loc[assets, assets] @ weights),
        }
    )
    return PortfolioSolution("shrinkage_mean_variance", weights, status, float(problem.value), solver, diagnostics)


def _maximum_feasible_return(mu: np.ndarray, max_weight: float) -> float:
    remaining = 1.0
    total = 0.0
    for value in np.sort(mu)[::-1]:
        allocation = min(max_weight, remaining)
        total += allocation * value
        remaining -= allocation
        if remaining <= 1e-12:
            return float(total)
    raise ValueError("max_weight makes the budget infeasible")


def maximum_sharpe_frontier(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    *,
    risk_free_rate: float = 0.0,
    max_weight: float = 1.0,
    grid_size: int = 25,
) -> PortfolioSolution:
    """Select maximum estimated Sharpe from a constrained efficient-frontier grid."""

    if grid_size < 5:
        raise ValueError("grid_size must be at least five")
    assets, mu, sigma = _aligned_moments(expected_returns, covariance)
    gmv = global_minimum_variance(covariance.loc[assets, assets], max_weight=max_weight)
    gmv_values = gmv.weights.loc[assets].to_numpy(dtype=float)
    lower_target = float(gmv_values @ mu)
    upper_target = _maximum_feasible_return(mu, max_weight)
    if upper_target <= risk_free_rate + 1e-12:
        raise NoPositiveExcessReturn(
            "no feasible portfolio has positive expected excess return; "
            "a maximum-Sharpe efficient-frontier search is not valid"
        )
    if upper_target < lower_target - 1e-10:
        raise OptimizationError("maximum feasible return is below the GMV portfolio return")
    upper_target = max(upper_target, lower_target)
    candidates: list[tuple[float, np.ndarray, float, str, str]] = []
    for target in np.linspace(lower_target, upper_target, grid_size):
        w = cp.Variable(len(assets))
        problem = cp.Problem(
            cp.Minimize(cp.quad_form(w, cp.psd_wrap(sigma))),
            [cp.sum(w) == 1, w >= 0, w <= max_weight, mu @ w >= target],
        )
        try:
            status, solver = _solve(problem)
        except OptimizationError:
            continue
        values = np.asarray(w.value).ravel()
        variance = float(values @ sigma @ values)
        realized_mu = float(values @ mu)
        sharpe = (realized_mu - risk_free_rate) / np.sqrt(variance) if variance > 0 else -np.inf
        if np.isfinite(sharpe):
            candidates.append((sharpe, values, float(problem.value), status, solver))
    if not candidates:
        raise OptimizationError("efficient frontier contained no feasible finite solution")
    sharpe, values, objective, status, solver = max(candidates, key=lambda row: row[0])
    weights = pd.Series(values, index=assets, name="weight")
    diagnostics = validate_weights(weights, upper_bound=max_weight)
    diagnostics["estimated_sharpe"] = float(sharpe)
    diagnostics["frontier_points_solved"] = len(candidates)
    diagnostics["frontier_lower_target"] = lower_target
    diagnostics["frontier_upper_target"] = upper_target
    diagnostics["risk_free_rate_annual"] = float(risk_free_rate)
    diagnostics["positive_excess_return_feasible"] = bool(upper_target > risk_free_rate)
    diagnostics["allocation_rule"] = "maximum_sharpe_frontier_grid"
    return PortfolioSolution("maximum_sharpe_frontier", weights, status, objective, solver, diagnostics)


def historical_cvar(
    scenarios: pd.DataFrame,
    *,
    confidence: float = 0.95,
    max_weight: float = 1.0,
    minimum_expected_return: float | None = None,
) -> PortfolioSolution:
    """Minimize empirical loss CVaR using complete joint historical scenarios."""

    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must lie in (0.5, 1)")
    complete = scenarios.dropna(axis=0, how="any")
    if len(complete) != len(scenarios):
        raise ValueError("CVaR scenarios must be joint and complete; do not zero-fill")
    returns = complete.to_numpy(dtype=float)
    if not np.isfinite(returns).all():
        raise ValueError("CVaR scenarios contain non-finite values")
    assets = complete.columns
    n_obs, n_assets = returns.shape
    if n_obs < 2 or max_weight * n_assets < 1.0 - 1e-12:
        raise ValueError("insufficient scenarios or infeasible max_weight")
    w = cp.Variable(n_assets)
    value_at_risk = cp.Variable()
    excess_loss = cp.Variable(n_obs, nonneg=True)
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= max_weight,
        excess_loss >= -returns @ w - value_at_risk,
    ]
    if minimum_expected_return is not None:
        constraints.append(complete.mean(axis=0).to_numpy() @ w >= minimum_expected_return)
    cvar = value_at_risk + cp.sum(excess_loss) / ((1.0 - confidence) * n_obs)
    problem = cp.Problem(cp.Minimize(cvar), constraints)
    status, solver = _solve(problem)
    weights = pd.Series(np.asarray(w.value).ravel(), index=assets, name="weight")
    diagnostics = validate_weights(weights, upper_bound=max_weight)
    losses = -(returns @ weights.to_numpy())
    var = float(np.quantile(losses, confidence, method="higher"))
    diagnostics.update(
        {
            "empirical_var": var,
            "empirical_tail_mean": float(losses[losses >= var].mean()),
            "joint_scenarios": n_obs,
        }
    )
    return PortfolioSolution("historical_joint_cvar", weights, status, float(problem.value), solver, diagnostics)


def benchmark_relative(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    benchmark_weights: pd.Series,
    prior_weights: pd.Series | None,
    *,
    factor_exposures: pd.DataFrame | None = None,
    factor_active_tolerance: float | pd.Series = 0.10,
    tracking_error_limit: float = 0.08,
    turnover_limit: float = 0.30,
    max_weight: float = 0.20,
    risk_aversion: float = 2.0,
    transaction_cost_penalty: float = 0.001,
    outside_prior_absolute_weight: float = 0.0,
) -> PortfolioSolution:
    assets, mu, sigma = _aligned_moments(expected_returns, covariance)
    benchmark = benchmark_weights.reindex(assets, fill_value=0.0).to_numpy(dtype=float)
    prior = (
        prior_weights.reindex(assets, fill_value=0.0).to_numpy(dtype=float)
        if prior_weights is not None
        else None
    )
    if abs(benchmark.sum() - 1.0) > SOLVER_TOLERANCE:
        raise ValueError("benchmark weights must sum to one on the eligible universe")
    if outside_prior_absolute_weight < 0:
        raise ValueError("outside_prior_absolute_weight cannot be negative")
    if tracking_error_limit <= 0 or turnover_limit < 0 or max_weight <= 0:
        raise ValueError("tracking-error/max-weight limits must be positive and turnover nonnegative")
    if transaction_cost_penalty < 0:
        raise ValueError("transaction_cost_penalty cannot be negative")
    if prior is not None:
        if (prior < -SOLVER_TOLERANCE).any():
            raise ValueError("benchmark-relative prior weights must be long-only")
        union_budget = float(prior.sum() + outside_prior_absolute_weight)
        if abs(union_budget - 1.0) > SOLVER_TOLERANCE:
            raise ValueError(
                "prior weights plus outside-universe absolute weight must sum to one"
            )
    w = cp.Variable(len(assets))
    active = w - benchmark
    constraints: list[cp.Constraint] = [
        cp.sum(w) == 1,
        w >= 0,
        w <= max_weight,
        cp.quad_form(active, cp.psd_wrap(sigma)) <= tracking_error_limit**2,
    ]
    if prior is not None:
        traded = w - prior
        one_way_turnover_expression = 0.5 * (
            cp.norm1(traded) + outside_prior_absolute_weight
        )
        constraints.append(one_way_turnover_expression <= turnover_limit)
    else:
        traded = None
        one_way_turnover_expression = 0.0
    factor_diagnostics: dict[str, float] = {}
    factor_tolerance_values: np.ndarray | None = None
    if factor_exposures is not None:
        missing = set(assets).difference(factor_exposures.index)
        if missing:
            raise ValueError(f"factor exposures missing assets: {sorted(missing)}")
        exposures = factor_exposures.loc[assets].to_numpy(dtype=float)
        active_factor = exposures.T @ active
        if np.isscalar(factor_active_tolerance):
            tolerance = np.full(exposures.shape[1], float(factor_active_tolerance))
        else:
            tolerance = pd.Series(factor_active_tolerance).reindex(
                factor_exposures.columns
            ).to_numpy(dtype=float)
        if not np.isfinite(tolerance).all() or (tolerance < 0).any():
            raise ValueError("factor tolerances must be finite, nonnegative, and cover every factor")
        factor_tolerance_values = tolerance
        constraints.extend([active_factor <= tolerance, active_factor >= -tolerance])
    utility = mu @ w - risk_aversion * cp.quad_form(w, cp.psd_wrap(sigma))
    if traded is not None:
        # The coefficient is per unit of one-way turnover (0.5 * union L1).
        utility -= transaction_cost_penalty * one_way_turnover_expression
    objective = cp.Maximize(utility)
    problem = cp.Problem(objective, constraints)
    status, solver = _solve(problem, conic=True)
    weights = pd.Series(np.asarray(w.value).ravel(), index=assets, name="weight")
    diagnostics = validate_weights(weights, upper_bound=max_weight)
    active_values = weights.to_numpy() - benchmark
    diagnostics.update(
        {
            "tracking_error": float(np.sqrt(max(active_values @ sigma @ active_values, 0.0))),
            "one_way_turnover": (
                float(
                    0.5
                    * (
                        np.abs(weights.to_numpy() - prior).sum()
                        + outside_prior_absolute_weight
                    )
                )
                if prior is not None
                else np.nan
            ),
        }
    )
    if factor_exposures is not None:
        active_factor_values = factor_exposures.loc[assets].to_numpy().T @ active_values
        for name, value in zip(factor_exposures.columns, active_factor_values, strict=True):
            factor_diagnostics[f"active_factor_{name}"] = float(value)
        assert factor_tolerance_values is not None
        factor_residual = np.maximum(
            np.abs(active_factor_values) - factor_tolerance_values, 0.0
        )
        diagnostics["factor_tolerance_residual"] = float(factor_residual.max())
        if factor_residual.max() > SOLVER_TOLERANCE:
            raise OptimizationError("factor exposure residual validation failed")
    diagnostics.update(factor_diagnostics)
    if diagnostics["tracking_error"] > tracking_error_limit + SOLVER_TOLERANCE:
        raise OptimizationError("tracking-error residual validation failed")
    if (
        prior is not None
        and diagnostics["one_way_turnover"] > turnover_limit + SOLVER_TOLERANCE
    ):
        raise OptimizationError("turnover residual validation failed")
    return PortfolioSolution("benchmark_relative_constrained", weights, status, float(problem.value), solver, diagnostics)


def exact_cardinality_cvar(
    scenarios: pd.DataFrame,
    *,
    cardinality: int,
    confidence: float = 0.95,
    minimum_weight: float = 0.01,
    maximum_weight: float = 0.25,
    time_limit_seconds: float = 30.0,
) -> PortfolioSolution:
    """Solve exact-cardinality historical CVaR as a SciPy/HiGHS MILP."""

    complete = scenarios.dropna(axis=0, how="any")
    if len(complete) != len(scenarios):
        raise ValueError("MILP scenarios must be joint and complete; do not zero-fill")
    returns = complete.to_numpy(dtype=float)
    observations, n_assets = returns.shape
    if not 1 <= cardinality <= n_assets:
        raise ValueError("cardinality must be between one and the asset count")
    if cardinality * maximum_weight < 1.0 - 1e-12:
        raise ValueError("cardinality and maximum_weight cannot fund the portfolio")
    if cardinality * minimum_weight > 1.0 + 1e-12:
        raise ValueError("cardinality and minimum_weight exceed the budget")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must lie in (0.5, 1)")

    w_slice = slice(0, n_assets)
    z_slice = slice(n_assets, 2 * n_assets)
    alpha_index = 2 * n_assets
    u_slice = slice(alpha_index + 1, alpha_index + 1 + observations)
    n_variables = u_slice.stop
    objective = np.zeros(n_variables)
    objective[alpha_index] = 1.0
    objective[u_slice] = 1.0 / ((1.0 - confidence) * observations)
    lower = np.zeros(n_variables)
    upper = np.full(n_variables, np.inf)
    upper[w_slice] = maximum_weight
    upper[z_slice] = 1.0
    alpha_bound = float(np.max(np.abs(returns)) + 1.0)
    lower[alpha_index] = -alpha_bound
    upper[alpha_index] = alpha_bound
    integrality = np.zeros(n_variables, dtype=int)
    integrality[z_slice] = 1

    rows: list[np.ndarray] = []
    lb: list[float] = []
    ub: list[float] = []
    budget = np.zeros(n_variables)
    budget[w_slice] = 1.0
    rows.append(budget)
    lb.append(1.0)
    ub.append(1.0)
    count = np.zeros(n_variables)
    count[z_slice] = 1.0
    rows.append(count)
    lb.append(float(cardinality))
    ub.append(float(cardinality))
    for asset in range(n_assets):
        upper_link = np.zeros(n_variables)
        upper_link[asset] = 1.0
        upper_link[n_assets + asset] = -maximum_weight
        rows.append(upper_link)
        lb.append(-np.inf)
        ub.append(0.0)
        lower_link = np.zeros(n_variables)
        lower_link[asset] = -1.0
        lower_link[n_assets + asset] = minimum_weight
        rows.append(lower_link)
        lb.append(-np.inf)
        ub.append(0.0)
    for scenario in range(observations):
        slack = np.zeros(n_variables)
        slack[w_slice] = -returns[scenario]
        slack[alpha_index] = -1.0
        slack[u_slice.start + scenario] = -1.0
        rows.append(slack)
        lb.append(-np.inf)
        ub.append(0.0)
    constraints = LinearConstraint(np.vstack(rows), np.asarray(lb), np.asarray(ub))
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={"time_limit": float(time_limit_seconds), "mip_rel_gap": 1e-8},
    )
    if not result.success or result.x is None:
        raise OptimizationError(f"HiGHS MILP failed: status={result.status}, {result.message}")
    weights = pd.Series(result.x[w_slice], index=complete.columns, name="weight")
    diagnostics = validate_weights(weights, upper_bound=maximum_weight)
    selected = result.x[z_slice]
    diagnostics.update(
        {
            "selected_count": int(np.rint(selected).sum()),
            "integrality_residual": float(np.abs(selected - np.rint(selected)).max()),
            "minimum_selected_weight_residual": float(
                np.maximum(minimum_weight * np.rint(selected) - weights.to_numpy(), 0).max()
            ),
            "mip_gap": float(getattr(result, "mip_gap", np.nan)),
            "joint_scenarios": observations,
        }
    )
    if diagnostics["selected_count"] != cardinality:
        raise OptimizationError("MILP exact-cardinality residual validation failed")
    if diagnostics["integrality_residual"] > SOLVER_TOLERANCE:
        raise OptimizationError("MILP integrality residual validation failed")
    if diagnostics["minimum_selected_weight_residual"] > SOLVER_TOLERANCE:
        raise OptimizationError("MILP minimum-position residual validation failed")
    return PortfolioSolution(
        "exact_cardinality_cvar",
        weights,
        "optimal" if result.status == 0 else "feasible",
        float(result.fun),
        "scipy_highs_milp",
        diagnostics,
    )


def long_short_130_30_appendix(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    *,
    risk_aversion: float = 4.0,
    long_position_cap: float = 0.35,
    short_position_cap: float = 0.15,
) -> PortfolioSolution:
    """Exact 130/30 exposure illustration with signs selected from lagged estimates.

    This is deliberately kept outside the main long-only comparison. It ignores
    locate availability, borrow fees, margin, financing, and recall risk.
    """

    assets, mu, sigma = _aligned_moments(expected_returns, covariance)
    order = np.argsort(mu)
    split = len(assets) // 2
    short_idx = order[:split]
    long_idx = order[split:]
    if len(long_idx) * long_position_cap < 1.3 - 1e-12:
        raise ValueError("long position cap makes 130% long exposure infeasible")
    if len(short_idx) * short_position_cap < 0.3 - 1e-12:
        raise ValueError("short position cap makes 30% short exposure infeasible")
    w = cp.Variable(len(assets))
    constraints = [
        cp.sum(w[long_idx]) == 1.3,
        cp.sum(w[short_idx]) == -0.3,
        w[long_idx] >= 0,
        w[long_idx] <= long_position_cap,
        w[short_idx] <= 0,
        w[short_idx] >= -short_position_cap,
    ]
    objective = cp.Maximize(mu @ w - risk_aversion * cp.quad_form(w, cp.psd_wrap(sigma)))
    problem = cp.Problem(objective, constraints)
    status, solver = _solve(problem)
    weights = pd.Series(np.asarray(w.value).ravel(), index=assets, name="weight")
    lower = pd.Series(-short_position_cap, index=assets)
    upper = pd.Series(long_position_cap, index=assets)
    diagnostics = validate_weights(weights, lower_bound=lower, upper_bound=upper)
    diagnostics.update(
        {
            "long_exposure": float(weights.clip(lower=0).sum()),
            "short_exposure": float(-weights.clip(upper=0).sum()),
            "gross_exposure": float(weights.abs().sum()),
        }
    )
    for key, expected in {"long_exposure": 1.3, "short_exposure": 0.3, "gross_exposure": 1.6}.items():
        if abs(float(diagnostics[key]) - expected) > SOLVER_TOLERANCE:
            raise OptimizationError(f"130/30 residual validation failed for {key}")
    return PortfolioSolution(
        "130_30_mathematical_appendix", weights, status, float(problem.value), solver, diagnostics
    )
