from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from constrained_portfolio.data import (
    PrivateCRSPAdapter,
    monthly_return_matrix,
    synthetic_monthly_eligibility,
    synthetic_monthly_panel,
)


def _write_fixture(root: Path, *, daily: pd.DataFrame, mapping: pd.DataFrame) -> None:
    root.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(root / "wrds_equity_daily.parquet", index=False)
    mapping.to_parquet(root / "wrds_symbol_map.parquet", index=False)


def _ohlcv(permno: int, dates: pd.DatetimeIndex, closes: list[float] | np.ndarray) -> pd.DataFrame:
    values = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "Permno": permno,
            "Date": dates,
            "Open": values,
            "High": values,
            "Low": values,
            "Close": values,
            "Volume": 1_000.0,
        }
    )


def test_asof_repair_prevents_historical_ticker_identity_merges(tmp_path: Path) -> None:
    dates = pd.to_datetime(["2020-01-31", "2020-02-28"])
    daily = pd.concat(
        [
            _ohlcv(10, dates, [10, 11]),
            _ohlcv(11, dates, [100, 101]),
            _ohlcv(20, dates, [20, 21]),
            _ohlcv(21, dates, [200, 202]),
        ],
        ignore_index=True,
    )
    mapping = pd.DataFrame(
        {
            "Permno": [10, 11, 20, 21],
            "Ticker": ["META", "META", "AXON", "AXON"],
            "Name_Start": pd.to_datetime(
                ["2022-01-01", "2020-01-01", "2021-01-01", "2017-01-01"]
            ),
            "Name_End": pd.to_datetime([None, "2021-12-31", None, "2019-12-31"]),
        }
    )
    _write_fixture(tmp_path, daily=daily, mapping=mapping)

    adapter = PrivateCRSPAdapter(tmp_path)
    current = adapter.current_mapping()
    panel = adapter.repaired_daily_panel()

    assert set(map(tuple, current[["Permno", "Ticker"]].to_numpy())) == {
        (10, "META"),
        (20, "AXON"),
    }
    assert set(panel["permno"]) == {10, 20}
    assert panel.groupby("ticker")["permno"].nunique().eq(1).all()


def test_open_ended_name_interval_is_active(tmp_path: Path) -> None:
    daily = _ohlcv(1, pd.to_datetime(["2025-12-31"]), [10])
    mapping = pd.DataFrame(
        {
            "Permno": [1],
            "Ticker": ["OPEN"],
            "Name_Start": pd.to_datetime(["2020-01-01"]),
            "Name_End": pd.to_datetime([None]),
        }
    )
    _write_fixture(tmp_path, daily=daily, mapping=mapping)
    assert PrivateCRSPAdapter(tmp_path).current_mapping().iloc[0]["Ticker"] == "OPEN"


def test_monthly_return_does_not_bridge_missing_calendar_month(tmp_path: Path) -> None:
    dates = pd.to_datetime(["2025-01-31", "2025-03-31", "2025-04-30"])
    daily = _ohlcv(1, dates, [100, 120, 126])
    mapping = pd.DataFrame(
        {
            "Permno": [1],
            "Ticker": ["GAP"],
            "Name_Start": pd.to_datetime(["2020-01-01"]),
            "Name_End": pd.to_datetime([None]),
        }
    )
    _write_fixture(tmp_path, daily=daily, mapping=mapping)
    monthly = PrivateCRSPAdapter(tmp_path).monthly_returns()

    assert np.isnan(monthly.loc[monthly["month"].eq("2025-03-31"), "return"].iloc[0])
    assert monthly.loc[monthly["month"].eq("2025-04-30"), "return"].iloc[0] == pytest.approx(
        0.05
    )


def test_daily_eligibility_uses_252_prior_observations_and_no_future(tmp_path: Path) -> None:
    dates = pd.bdate_range("2024-12-02", "2025-12-31")
    daily = _ohlcv(1, dates, np.linspace(100, 130, len(dates)))
    mapping = pd.DataFrame(
        {
            "Permno": [1],
            "Ticker": ["ELIG"],
            "Name_Start": pd.to_datetime(["2020-01-01"]),
            "Name_End": pd.to_datetime([None]),
        }
    )
    _write_fixture(tmp_path, daily=daily, mapping=mapping)
    adapter = PrivateCRSPAdapter(tmp_path)

    eligibility = adapter.monthly_eligibility(
        pd.to_datetime(["2025-12-31", "2026-01-31"]), minimum_daily_returns=252
    )

    earlier = adapter.monthly_eligibility(
        pd.to_datetime(["2025-11-30"]), minimum_daily_returns=252
    )
    assert not bool(earlier.loc["2025-11-30", 1])
    assert bool(eligibility.loc["2026-01-31", 1])
    assert "252" in eligibility.attrs["rule"]


def test_synthetic_fixture_and_eligibility_are_reproducible_and_lagged() -> None:
    first_returns, first_factors = synthetic_monthly_panel(seed=7)
    second_returns, second_factors = synthetic_monthly_panel(seed=7)
    pd.testing.assert_frame_equal(first_returns, second_returns)
    pd.testing.assert_frame_equal(first_factors, second_factors)

    eligibility = synthetic_monthly_eligibility(first_returns, minimum_prior_months=12)
    assert not eligibility.loc[first_returns.index[11]].any()
    assert eligibility.loc[first_returns.index[12]].all()
    altered = first_returns.copy()
    altered.loc[first_returns.index[12]] = np.nan
    altered_eligibility = synthetic_monthly_eligibility(altered, minimum_prior_months=12)
    pd.testing.assert_series_equal(
        eligibility.loc[first_returns.index[12]], altered_eligibility.loc[first_returns.index[12]]
    )


def test_monthly_matrix_keeps_missing_values() -> None:
    panel = pd.DataFrame(
        {
            "month": pd.to_datetime(["2025-01-31", "2025-01-31", "2025-02-28"]),
            "permno": [1, 2, 1],
            "return": [0.01, np.nan, 0.02],
        }
    )
    matrix = monthly_return_matrix(panel)
    assert np.isnan(matrix.loc["2025-01-31", 2])
    assert np.isnan(matrix.loc["2025-02-28", 2])


@pytest.mark.private
def test_retained_private_snapshot_invariants_when_configured() -> None:
    root = os.environ.get("CPC_LICENSED_DATA_ROOT")
    if not root:
        pytest.skip("CPC_LICENSED_DATA_ROOT is not configured")
    adapter = PrivateCRSPAdapter(root)
    panel = adapter.repaired_daily_panel(verify_known_snapshot=True)
    assert len(panel) == 192_388
    assert panel["permno"].nunique() == 101
    for historically_reused_ticker in ("META", "AXON"):
        assert panel.loc[panel["ticker"].eq(historically_reused_ticker), "permno"].nunique() == 1
