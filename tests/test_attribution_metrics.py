from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from constrained_portfolio.attribution import (
    active_return_attribution,
    portfolio_factor_exposure,
)
from constrained_portfolio.factors import FACTOR_COLUMNS, estimate_factor_exposures
from constrained_portfolio.metrics import performance_metrics
from constrained_portfolio.uncertainty import moving_block_bootstrap


def test_zero_weight_asset_without_exposure_is_ignored() -> None:
    weights = pd.Series({"A": 1.0, "B": 0.0})
    exposure = pd.DataFrame({"MKT_RF": [1.2]}, index=["A"])
    result = portfolio_factor_exposure(weights, exposure)
    assert result["MKT_RF"] == pytest.approx(1.2)


def test_factor_exposure_estimation_is_strictly_lagged() -> None:
    dates = pd.date_range("2018-01-31", periods=30, freq="ME")
    rng = np.random.default_rng(3)
    factors = pd.DataFrame(
        rng.normal(0, 0.02, size=(30, 7)), index=dates, columns=[*FACTOR_COLUMNS, "RF"]
    )
    assets = pd.DataFrame(
        {"A": factors["RF"] + 1.1 * factors["MKT_RF"] + 0.2 * factors["MOM"]},
        index=dates,
    )
    decision = dates[24]
    first = estimate_factor_exposures(assets, factors, decision, min_observations=12)
    changed = assets.copy()
    changed.loc[decision:, "A"] = 100.0
    second = estimate_factor_exposures(changed, factors, decision, min_observations=12)
    pd.testing.assert_frame_equal(first, second)
    assert pd.Timestamp(first.attrs["last_asset_observation"]) < decision


def test_active_attribution_reconciles_cost_and_specific_return() -> None:
    result = active_return_attribution(
        portfolio_gross_return=0.03,
        benchmark_return=0.02,
        transaction_cost=0.001,
        portfolio_exposure=pd.Series({"MKT_RF": 1.1, "HML": 0.2}),
        benchmark_exposure=pd.Series({"MKT_RF": 1.0, "HML": 0.1}),
        factor_returns=pd.Series({"MKT_RF": 0.04, "HML": -0.01}),
    )
    components = result.loc[result["component"].ne("active_net_return"), "contribution"].sum()
    assert components == pytest.approx(0.009)


def test_metric_tail_key_matches_requested_confidence() -> None:
    metrics = performance_metrics(pd.Series([0.01, -0.02, 0.03, -0.01]), cvar_confidence=0.90)
    assert "monthly_loss_var_0_9" in metrics
    assert "monthly_loss_cvar_0_9" in metrics
    assert metrics["loss_tail_confidence"] == 0.90


def test_drawdown_includes_initial_capital_peak() -> None:
    metrics = performance_metrics(pd.Series([-0.10, 0.05]))
    assert metrics["maximum_drawdown"] == pytest.approx(-0.10)


def test_block_bootstrap_is_seed_reproducible() -> None:
    returns = pd.Series(np.linspace(-0.02, 0.03, 36))
    first = moving_block_bootstrap(returns, samples=25, seed=9)
    second = moving_block_bootstrap(returns, samples=25, seed=9)
    pd.testing.assert_frame_equal(first, second)
