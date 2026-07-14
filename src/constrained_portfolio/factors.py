from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import requests


FF5_URL: Final = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_CSV.zip"
)
MOMENTUM_URL: Final = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_CSV.zip"
)
FACTOR_COLUMNS: Final = ("MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _parse_french_monthly_zip(payload: bytes, expected_columns: int) -> pd.DataFrame:
    """Parse the monthly section of a Kenneth French CSV zip archive."""

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"expected one CSV in French archive, found {csv_names}")
        text = archive.read(csv_names[0]).decode("latin-1")
    rows: list[list[float | pd.Timestamp]] = []
    for line in text.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if not fields or not re.fullmatch(r"\d{6}", fields[0]):
            continue
        if len(fields) < expected_columns + 1:
            continue
        date = pd.to_datetime(fields[0], format="%Y%m").to_period("M").to_timestamp("M")
        try:
            values = [float(value) / 100.0 for value in fields[1 : expected_columns + 1]]
        except ValueError:
            continue
        rows.append([date, *values])
    if not rows:
        raise ValueError("French archive contained no parseable monthly observations")
    return pd.DataFrame(rows).set_index(0).sort_index()


def fetch_french_factors(
    cache_dir: str | Path,
    *,
    refresh: bool = False,
    timeout_seconds: float = 30.0,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch/cache monthly FF5 plus momentum with checksums and provenance.

    Values are converted from percent to decimal. The combined cache is public
    and reproducible; it contains no licensed CRSP rows.
    """

    cache = Path(cache_dir)
    combined_path = cache / "ff5_momentum_monthly.csv"
    provenance_path = cache / "provenance.json"
    if combined_path.is_file() and provenance_path.is_file() and not refresh:
        frame = pd.read_csv(combined_path, parse_dates=["month"]).set_index("month")
        frame.index = pd.DatetimeIndex(frame.index, name="month")
        frame.attrs.update(json.loads(provenance_path.read_text(encoding="utf-8")))
        return frame

    client = session or requests.Session()
    payloads: dict[str, bytes] = {}
    for label, url in {"ff5": FF5_URL, "momentum": MOMENTUM_URL}.items():
        response = client.get(url, timeout=timeout_seconds)
        response.raise_for_status()
        payloads[label] = response.content
    ff5 = _parse_french_monthly_zip(payloads["ff5"], expected_columns=6)
    ff5.columns = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "RF"]
    momentum = _parse_french_monthly_zip(payloads["momentum"], expected_columns=1)
    momentum.columns = ["MOM"]
    combined = ff5.join(momentum, how="inner").sort_index()
    combined.index.name = "month"
    if combined.empty or combined.isna().any().any():
        raise ValueError("combined factor dataset is empty or incomplete")
    provenance: dict[str, object] = {
        "provenance": "public",
        "provider": "Kenneth R. French Data Library",
        "dataset": "Fama-French 5 Factors (2x3) plus Momentum",
        "units": "decimal monthly returns",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "ff5": {"url": FF5_URL, "sha256": _sha256(payloads["ff5"])},
            "momentum": {
                "url": MOMENTUM_URL,
                "sha256": _sha256(payloads["momentum"]),
            },
        },
        "start": str(combined.index.min().date()),
        "end": str(combined.index.max().date()),
        "observations": len(combined),
    }
    cache.mkdir(parents=True, exist_ok=True)
    combined.reset_index().to_csv(combined_path, index=False)
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    for label, payload in payloads.items():
        (cache / f"{label}.zip").write_bytes(payload)
    combined.attrs.update(provenance)
    return combined


def estimate_factor_exposures(
    asset_returns: pd.DataFrame,
    factors: pd.DataFrame,
    decision_month: str | pd.Timestamp,
    *,
    window_months: int = 60,
    min_observations: int = 24,
) -> pd.DataFrame:
    """Estimate trailing asset betas using only months before ``decision_month``."""

    regression_columns = [*FACTOR_COLUMNS, "RF"]
    missing = set(regression_columns).difference(factors.columns)
    if missing:
        raise ValueError(f"factor data missing columns: {sorted(missing)}")
    decision = pd.Timestamp(decision_month).to_period("M").to_timestamp("M")
    past_assets = asset_returns.loc[asset_returns.index < decision].sort_index().tail(window_months)
    past_factors = factors.loc[factors.index < decision].sort_index().tail(window_months)
    if past_assets.empty or past_factors.empty:
        raise ValueError("no asset/factor observations precede the decision month")
    rows: list[dict[str, float | int]] = []
    for asset in past_assets.columns:
        joined = pd.concat(
            [past_assets[asset].rename("asset_return"), past_factors[regression_columns]], axis=1
        ).dropna()
        if len(joined) < min_observations:
            continue
        y = joined["asset_return"].to_numpy(dtype=float) - joined["RF"].to_numpy(dtype=float)
        x = joined[list(FACTOR_COLUMNS)].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(joined)), x])
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        record: dict[str, float | int] = {
            "asset": asset,
            "alpha": float(coefficients[0]),
            "observations": len(joined),
        }
        record.update(
            {name: float(value) for name, value in zip(FACTOR_COLUMNS, coefficients[1:], strict=True)}
        )
        rows.append(record)
    if not rows:
        raise ValueError("no assets have enough observations for factor regression")
    exposure = pd.DataFrame(rows).set_index("asset")
    exposure.index = pd.Index(exposure.index, name=asset_returns.columns.name or "asset")
    exposure.attrs.update(
        {
            "decision_month": str(decision.date()),
            "last_asset_observation": str(past_assets.index.max().date()),
            "last_factor_observation": str(past_factors.index.max().date()),
            "lookahead_control": "all regression observations are strictly before decision_month",
        }
    )
    return exposure
