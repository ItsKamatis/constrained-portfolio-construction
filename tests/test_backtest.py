from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from constrained_portfolio.backtest import WalkForwardConfig, run_walk_forward
from constrained_portfolio.data import synthetic_monthly_eligibility, synthetic_monthly_panel


def _small_config(**overrides: object) -> WalkForwardConfig:
    values: dict[str, object] = {
        "start": "2020-04-30",
        "end": "2020-05-31",
        "lookback_months": 3,
        "minimum_monthly_scenarios": 2,
        "factor_minimum_observations": 2,
        "factor_window_months": 3,
        "frontier_points": 5,
        "cardinality": 2,
        "cardinality_maximum_weight": 0.6,
        "max_weight": 0.6,
    }
    values.update(overrides)
    return WalkForwardConfig(**values)


def test_drift_turnover_cost_and_net_return_reconcile_exactly() -> None:
    dates = pd.date_range("2020-01-31", periods=5, freq="ME")
    returns = pd.DataFrame(
        [[0.01, 0.02], [0.00, 0.01], [0.02, -0.01], [0.10, 0.00], [0.00, 0.00]],
        index=dates,
        columns=["A", "B"],
    )
    result = run_walk_forward(
        returns,
        methods=["equal_weight"],
        config=_small_config(transaction_cost_bps_per_one_way_turnover=10.0),
    )
    first, second = result.periods.sort_values("month").to_dict("records")
    assert first["inception"] is True
    assert np.isnan(first["one_way_turnover"])
    assert first["transaction_cost"] == pytest.approx(0.001)
    expected_end_a = 0.5 * 1.10 / 1.05
    expected_turnover = abs(expected_end_a - 0.5)
    assert second["one_way_turnover"] == pytest.approx(expected_turnover)
    assert second["transaction_cost"] == pytest.approx(0.001 * expected_turnover)
    assert second["net_return"] == pytest.approx(
        second["gross_return"] - second["transaction_cost"]
    )
    holdings = result.holdings_attribution.groupby(["month", "method"])[
        "gross_contribution"
    ].sum()
    for row in result.periods.itertuples():
        assert holdings.loc[(row.month, row.method)] == pytest.approx(row.gross_return)


def test_weights_do_not_use_holding_month_return() -> None:
    returns, factors = synthetic_monthly_panel(seed=11, n_assets=8)
    eligibility = synthetic_monthly_eligibility(returns)
    config = WalkForwardConfig(
        start="2020-01-31",
        end="2020-01-31",
        frontier_points=5,
        cardinality=4,
        cardinality_maximum_weight=0.4,
        max_weight=0.4,
    )
    baseline = run_walk_forward(
        returns,
        factors=factors,
        eligibility=eligibility,
        methods=["shrinkage_mean_variance"],
        config=config,
    )
    altered = returns.copy()
    altered.loc["2020-01-31"] = np.linspace(-0.9, 0.9, altered.shape[1])
    rerun = run_walk_forward(
        altered,
        factors=factors,
        eligibility=eligibility,
        methods=["shrinkage_mean_variance"],
        config=config,
    )
    left = baseline.weights.sort_values(["month", "method", "asset"])["target_weight"].to_numpy()
    right = rerun.weights.sort_values(["month", "method", "asset"])["target_weight"].to_numpy()
    np.testing.assert_allclose(left, right, atol=1e-9)
    assert (baseline.periods["estimation_end"] < baseline.periods["month"]).all()


def test_maximum_sharpe_uses_lagged_rf_not_holding_month_rf() -> None:
    returns, factors = synthetic_monthly_panel(seed=23, n_assets=8)
    eligibility = synthetic_monthly_eligibility(returns)
    config = WalkForwardConfig(
        start="2020-01-31",
        end="2020-01-31",
        frontier_points=9,
        cardinality=4,
        cardinality_maximum_weight=0.4,
        max_weight=0.4,
    )
    baseline = run_walk_forward(
        returns,
        factors=factors,
        eligibility=eligibility,
        methods=["maximum_sharpe_with_gmv_fallback"],
        config=config,
    )
    altered_factors = factors.copy()
    altered_factors.loc["2020-01-31", "RF"] = 0.25
    rerun = run_walk_forward(
        returns,
        factors=altered_factors,
        eligibility=eligibility,
        methods=["maximum_sharpe_with_gmv_fallback"],
        config=config,
    )

    np.testing.assert_allclose(
        baseline.weights["target_weight"], rerun.weights["target_weight"], atol=1e-9
    )
    assert baseline.periods["construction_risk_free_rate_annual"].iloc[0] == pytest.approx(
        rerun.periods["construction_risk_free_rate_annual"].iloc[0]
    )
    assert baseline.periods["realized_risk_free_return"].iloc[0] != pytest.approx(
        rerun.periods["realized_risk_free_return"].iloc[0]
    )


def test_maximum_sharpe_walk_forward_labels_gmv_fallback() -> None:
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    returns = pd.DataFrame(
        [[-0.01, -0.02], [-0.01, -0.02], [-0.01, -0.02], [0.00, 0.00]],
        index=dates,
        columns=["A", "B"],
    )

    result = run_walk_forward(
        returns,
        methods=["maximum_sharpe_with_gmv_fallback"],
        config=_small_config(start="2020-04-30", end="2020-04-30"),
    )

    assert result.periods["construction_rule"].iloc[0] == (
        "gmv_fallback_no_positive_expected_excess_return"
    )


def test_missing_held_return_is_not_zero_filled() -> None:
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    returns = pd.DataFrame(
        [[0.01, 0.02], [0.00, 0.01], [0.02, -0.01], [0.10, np.nan]],
        index=dates,
        columns=["A", "B"],
    )
    with pytest.raises(ValueError, match="missing returns"):
        run_walk_forward(
            returns,
            methods=["equal_weight"],
            config=_small_config(start="2020-04-30", end="2020-04-30"),
        )


def test_active_attribution_and_reproducibility() -> None:
    returns, factors = synthetic_monthly_panel(seed=19, n_assets=8)
    eligibility = synthetic_monthly_eligibility(returns)
    config = WalkForwardConfig(
        start="2020-01-31",
        end="2020-02-29",
        frontier_points=5,
        cardinality=4,
        cardinality_maximum_weight=0.4,
        max_weight=0.4,
    )
    kwargs = dict(
        factors=factors,
        eligibility=eligibility,
        methods=["global_minimum_variance", "benchmark_relative_constrained"],
        config=config,
    )
    first = run_walk_forward(returns, **kwargs)
    second = run_walk_forward(returns, **kwargs)
    pd.testing.assert_frame_equal(first.periods, second.periods, check_exact=False, atol=1e-8)
    grouped = first.active_attribution.groupby(["month", "method"])
    for key, frame in grouped:
        total = frame.loc[frame["component"].ne("active_net_return"), "contribution"].sum()
        reported = frame.loc[frame["component"].eq("active_net_return"), "contribution"].iloc[0]
        assert total == pytest.approx(reported, abs=1e-10), key
