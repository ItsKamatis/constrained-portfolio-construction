from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd


DAILY_FILE: Final = "wrds_equity_daily.parquet"
SYMBOL_FILE: Final = "wrds_symbol_map.parquet"
DEFAULT_AS_OF: Final = pd.Timestamp("2025-12-31")


@dataclass(frozen=True)
class DataLimitations:
    """Limitations that must travel with every private-panel result."""

    return_definition: str = (
        "Split-adjusted close-to-close price returns; dividends are not included."
    )
    universe_definition: str = (
        "Securities with a unique ticker-PERMNO mapping active on 2025-12-31, "
        "backfilled through available history."
    )
    survivorship: str = (
        "The current-constituent backfill creates survivorship and selection bias."
    )
    benchmark: str = (
        "No historical index weights or market caps are available; the reference "
        "portfolio is eligible-universe equal weight, not an index replica."
    )
    classifications: str = (
        "No point-in-time sector or fundamental history is available; sector "
        "constraints and Brinson attribution are omitted."
    )
    evaluation: str = (
        "2024-2025 has already been inspected and is labeled evaluation, not a "
        "pristine unseen test."
    )
    provenance: str = "licensed_user_input"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class PrivateDataError(ValueError):
    """Raised when a licensed input violates the documented data contract."""


class PrivateCRSPAdapter:
    """Read retained licensed files from an external directory.

    The adapter never writes licensed rows. Callers choose an ignored output
    directory if they persist derived summaries or panels.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        configured = root or os.environ.get("CPC_LICENSED_DATA_ROOT")
        if not configured:
            raise PrivateDataError(
                "set CPC_LICENSED_DATA_ROOT or pass root=; licensed data is not bundled"
            )
        self.root = Path(configured).expanduser().resolve()

    def _path(self, filename: str) -> Path:
        path = self.root / filename
        if not path.is_file():
            raise FileNotFoundError(f"required licensed input not found: {path}")
        return path

    def load_daily(self) -> pd.DataFrame:
        frame = pd.read_parquet(self._path(DAILY_FILE))
        required = {"Permno", "Date", "Open", "High", "Low", "Close", "Volume"}
        missing = required.difference(frame.columns)
        if missing:
            raise PrivateDataError(f"daily file missing columns: {sorted(missing)}")
        return frame.loc[:, sorted(required)].copy()

    def load_symbol_map(self) -> pd.DataFrame:
        frame = pd.read_parquet(self._path(SYMBOL_FILE))
        required = {"Permno", "Ticker", "Name_Start", "Name_End"}
        missing = required.difference(frame.columns)
        if missing:
            raise PrivateDataError(f"symbol map missing columns: {sorted(missing)}")
        return frame.loc[:, ["Permno", "Ticker", "Name_Start", "Name_End"]].copy()

    def current_mapping(self, as_of: str | pd.Timestamp = DEFAULT_AS_OF) -> pd.DataFrame:
        cutoff = pd.Timestamp(as_of).normalize()
        mapping = self.load_symbol_map()
        mapping["Name_Start"] = pd.to_datetime(mapping["Name_Start"])
        mapping["Name_End"] = pd.to_datetime(mapping["Name_End"])
        active = mapping.loc[
            mapping["Name_Start"].le(cutoff)
            & (mapping["Name_End"].isna() | mapping["Name_End"].ge(cutoff)),
            ["Permno", "Ticker"],
        ].drop_duplicates()
        if active.empty:
            raise PrivateDataError(f"no ticker mappings active on {cutoff.date()}")
        dup_ticker = active.loc[active.duplicated("Ticker", keep=False), "Ticker"].unique()
        dup_permno = active.loc[active.duplicated("Permno", keep=False), "Permno"].unique()
        if len(dup_ticker) or len(dup_permno):
            raise PrivateDataError(
                "as-of mapping is not one-to-one; "
                f"duplicate tickers={dup_ticker.tolist()}, PERMNOs={dup_permno.tolist()}"
            )
        return active.sort_values(["Ticker", "Permno"]).reset_index(drop=True)

    def repaired_daily_panel(
        self,
        as_of: str | pd.Timestamp = DEFAULT_AS_OF,
        *,
        verify_known_snapshot: bool = False,
    ) -> pd.DataFrame:
        daily = self.load_daily()
        mapping = self.current_mapping(as_of)
        panel = daily.merge(mapping, on="Permno", how="inner", validate="many_to_one")
        panel = panel.rename(
            columns={
                "Permno": "permno",
                "Ticker": "ticker",
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        panel["date"] = pd.to_datetime(panel["date"])
        panel = panel.sort_values(["permno", "date"], kind="stable").reset_index(drop=True)
        if panel.duplicated(["permno", "date"]).any():
            raise PrivateDataError("repaired panel contains duplicate PERMNO-date rows")
        ticker_counts = panel.groupby("permno")["ticker"].nunique()
        if ticker_counts.gt(1).any():
            raise PrivateDataError("PERMNO identity changed after current-map repair")
        if verify_known_snapshot:
            expected = {
                "rows": 192_388,
                "securities": 101,
                "start": pd.Timestamp("2018-01-02"),
                "end": pd.Timestamp("2025-12-31"),
            }
            observed = {
                "rows": len(panel),
                "securities": panel["permno"].nunique(),
                "start": panel["date"].min(),
                "end": panel["date"].max(),
            }
            if observed != expected:
                raise PrivateDataError(
                    f"retained snapshot invariant failed: expected {expected}, got {observed}"
                )
        return panel[
            ["date", "permno", "ticker", "open", "high", "low", "close", "volume"]
        ]

    def monthly_returns(
        self,
        as_of: str | pd.Timestamp = DEFAULT_AS_OF,
        *,
        verify_known_snapshot: bool = False,
    ) -> pd.DataFrame:
        """Create monthly close-to-close price returns without zero imputation."""

        panel = self.repaired_daily_panel(as_of, verify_known_snapshot=verify_known_snapshot)
        panel["month"] = panel["date"].dt.to_period("M").dt.to_timestamp("M")
        monthly = (
            panel.sort_values("date")
            .groupby(["permno", "ticker", "month"], as_index=False, observed=True)
            .agg(close=("close", "last"), observation_date=("date", "last"))
            .sort_values(["permno", "month"])
        )
        monthly["return"] = monthly.groupby("permno", observed=True)["close"].pct_change(
            fill_method=None
        )
        prior_month = monthly.groupby("permno", observed=True)["month"].shift(1)
        is_consecutive = prior_month.eq(monthly["month"] - pd.offsets.MonthEnd(1))
        monthly.loc[~is_consecutive, "return"] = np.nan
        return monthly[["month", "permno", "ticker", "observation_date", "close", "return"]]

    def monthly_eligibility(
        self,
        decision_months: pd.DatetimeIndex | list[pd.Timestamp],
        as_of: str | pd.Timestamp = DEFAULT_AS_OF,
        *,
        minimum_daily_returns: int = 252,
        verify_known_snapshot: bool = False,
    ) -> pd.DataFrame:
        """Point-in-time eligibility based on daily history available before each month.

        An asset needs at least ``minimum_daily_returns`` valid daily close returns
        and its latest price observation must occur in the immediately preceding
        calendar month. Counts and recency are evaluated strictly before the first
        day of the holding month.
        """

        if minimum_daily_returns < 2:
            raise ValueError("minimum_daily_returns must be at least two")
        panel = self.repaired_daily_panel(as_of, verify_known_snapshot=verify_known_snapshot)
        panel["daily_return"] = panel.groupby("permno", observed=True)["close"].pct_change(
            fill_method=None
        )
        rows: list[pd.Series] = []
        normalized_months = pd.DatetimeIndex(decision_months).to_period("M").to_timestamp("M")
        assets = pd.Index(sorted(panel["permno"].unique()), name="permno")
        for month in normalized_months:
            month_start = month.to_period("M").to_timestamp()
            available = panel.loc[panel["date"] < month_start]
            valid_counts = available.groupby("permno", observed=True)["daily_return"].count()
            last_dates = available.groupby("permno", observed=True)["date"].max()
            previous_period = (month_start - pd.Timedelta(days=1)).to_period("M")
            recent = last_dates.dt.to_period("M").eq(previous_period)
            eligible = valid_counts.ge(minimum_daily_returns) & recent
            rows.append(eligible.reindex(assets, fill_value=False).rename(month))
        result = pd.DataFrame(rows, index=normalized_months, columns=assets, dtype=bool)
        result.index.name = "month"
        result.attrs.update(
            {
                "rule": (
                    f">={minimum_daily_returns} valid daily close returns available before "
                    "the holding month and an observation in the immediately prior month"
                ),
                "provenance": "licensed_user_input",
                "lookahead_control": "all eligibility inputs are dated before the holding month",
            }
        )
        return result


def monthly_return_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    required = {"month", "permno", "return"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"monthly panel missing columns: {sorted(missing)}")
    if panel.duplicated(["month", "permno"]).any():
        raise ValueError("monthly panel has duplicate month-PERMNO observations")
    matrix = panel.pivot(index="month", columns="permno", values="return").sort_index()
    matrix.index = pd.DatetimeIndex(matrix.index, name="month")
    matrix.columns.name = "permno"
    return matrix.astype(float)


def synthetic_monthly_panel(
    *,
    seed: int = 20260713,
    n_assets: int = 12,
    start: str = "2018-01-31",
    end: str = "2025-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a deterministic public-safe asset panel and factor fixture."""

    if n_assets < 6:
        raise ValueError("synthetic fixture needs at least six assets")
    dates = pd.date_range(start, end, freq="ME")
    rng = np.random.default_rng(seed)
    factor_names = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]
    factor_mean = np.array([0.0050, 0.0010, 0.0012, 0.0010, 0.0008, 0.0020])
    factor_scale = np.array([0.040, 0.025, 0.025, 0.018, 0.018, 0.030])
    factors = rng.normal(factor_mean, factor_scale, size=(len(dates), len(factor_names)))
    factor_frame = pd.DataFrame(factors, index=dates, columns=factor_names)
    factor_frame["RF"] = 0.002 / 12
    factor_frame.attrs["provenance"] = "synthetic"

    betas = rng.normal(0.0, 0.35, size=(n_assets, len(factor_names)))
    betas[:, 0] += 1.0
    alpha = np.linspace(-0.001, 0.002, n_assets)
    idiosyncratic = rng.normal(0.0, 0.025, size=(len(dates), n_assets))
    values = factor_frame[factor_names].to_numpy() @ betas.T + alpha + idiosyncratic
    assets = pd.DataFrame(
        values,
        index=dates,
        columns=pd.Index(range(900_001, 900_001 + n_assets), name="permno"),
    )
    assets.index.name = "month"
    assets.attrs.update(
        {
            "provenance": "synthetic",
            "description": "Deterministic correlated monthly returns; no real securities.",
        }
    )
    return assets, factor_frame


def synthetic_monthly_eligibility(
    returns: pd.DataFrame,
    *,
    minimum_prior_months: int = 12,
) -> pd.DataFrame:
    """Public-demo analogue to daily eligibility.

    Synthetic fixtures contain no daily observations, so the public workflow
    requires a fixed number of valid *monthly* observations strictly before the
    holding month. This is not presented as equivalent to the private 252-daily-
    observation rule.
    """

    if minimum_prior_months < 2:
        raise ValueError("minimum_prior_months must be at least two")
    eligibility = pd.DataFrame(False, index=returns.index, columns=returns.columns)
    for month in returns.index:
        history = returns.loc[returns.index < month]
        previous = month - pd.offsets.MonthEnd(1)
        prior_is_current = (
            history.loc[previous].notna()
            if previous in history.index
            else pd.Series(False, index=returns.columns)
        )
        eligibility.loc[month] = history.notna().sum().ge(minimum_prior_months) & prior_is_current
    eligibility.attrs.update(
        {
            "rule": f">={minimum_prior_months} prior valid monthly returns",
            "provenance": "synthetic",
            "analogue_not_equivalence": "No daily synthetic data exist; this is not the 252-daily rule.",
        }
    )
    return eligibility.astype(bool)


def split_label(date: str | pd.Timestamp) -> str:
    timestamp = pd.Timestamp(date)
    if timestamp < pd.Timestamp("2020-01-01"):
        return "warmup"
    if timestamp < pd.Timestamp("2024-01-01"):
        return "development"
    if timestamp <= pd.Timestamp("2025-12-31"):
        return "evaluation_already_inspected"
    return "out_of_design_range"
