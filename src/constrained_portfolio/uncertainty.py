from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import performance_metrics


def moving_block_bootstrap(
    returns: pd.Series,
    *,
    risk_free_returns: pd.Series | float = 0.0,
    block_length: int = 6,
    samples: int = 500,
    seed: int = 20260713,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Moving-block bootstrap intervals for dependent monthly returns."""

    return_series = pd.to_numeric(returns, errors="coerce").rename("return")
    if np.isscalar(risk_free_returns):
        risk_free_series = pd.Series(float(risk_free_returns), index=return_series.index, name="rf")
    else:
        risk_free_series = pd.to_numeric(risk_free_returns, errors="coerce").rename("rf")
    aligned = pd.concat([return_series, risk_free_series], axis=1).dropna()
    clean_returns = aligned["return"].to_numpy(dtype=float)
    clean_rf = aligned["rf"].to_numpy(dtype=float)
    n_obs = len(clean_returns)
    if not 1 <= block_length <= n_obs:
        raise ValueError("block_length must be between one and the sample length")
    if samples < 20:
        raise ValueError("samples must be at least 20")
    rng = np.random.default_rng(seed)
    starts = np.arange(0, n_obs - block_length + 1)
    draws: list[dict[str, float | int]] = []
    blocks_needed = int(np.ceil(n_obs / block_length))
    for draw in range(samples):
        selected = rng.choice(starts, size=blocks_needed, replace=True)
        sample_returns = np.concatenate(
            [clean_returns[start : start + block_length] for start in selected]
        )[:n_obs]
        sample_rf = np.concatenate([clean_rf[start : start + block_length] for start in selected])[
            :n_obs
        ]
        metrics = performance_metrics(pd.Series(sample_returns), risk_free_returns=pd.Series(sample_rf))
        draws.append({"draw": draw, **metrics})
    frame = pd.DataFrame(draws)
    alpha = (1.0 - confidence) / 2.0
    rows = []
    for metric in ["annualized_return", "annualized_volatility", "sharpe", "maximum_drawdown"]:
        values = frame[metric].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "metric": metric,
                "estimate": float(
                    performance_metrics(
                        pd.Series(clean_returns), risk_free_returns=pd.Series(clean_rf)
                    )[metric]
                ),
                "lower": float(values.quantile(alpha)),
                "upper": float(values.quantile(1.0 - alpha)),
                "confidence": confidence,
                "samples": samples,
                "block_length": block_length,
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)
