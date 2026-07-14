from __future__ import annotations

import numpy as np
import pandas as pd


def performance_metrics(
    returns: pd.Series,
    *,
    benchmark_returns: pd.Series | None = None,
    risk_free_returns: pd.Series | float = 0.0,
    periods_per_year: int = 12,
    cvar_confidence: float = 0.95,
) -> dict[str, float | int]:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty or (clean <= -1).any():
        raise ValueError("returns must be nonempty, finite, and greater than -100%")
    if np.isscalar(risk_free_returns):
        excess = clean - float(risk_free_returns)
    else:
        excess = clean - pd.Series(risk_free_returns).reindex(clean.index)
        excess = excess.dropna()
    wealth = (1.0 + clean).cumprod()
    years = len(clean) / periods_per_year
    annual_return = float(wealth.iloc[-1] ** (1.0 / years) - 1.0)
    annual_volatility = float(clean.std(ddof=1) * np.sqrt(periods_per_year))
    annual_excess = float(excess.mean() * periods_per_year)
    sharpe = annual_excess / annual_volatility if annual_volatility > 0 else np.nan
    running_peak = np.maximum.accumulate(np.concatenate([[1.0], wealth.to_numpy(dtype=float)]))[1:]
    drawdown = wealth / running_peak - 1.0
    losses = -clean.to_numpy(dtype=float)
    value_at_risk = float(np.quantile(losses, cvar_confidence, method="higher"))
    cvar = float(losses[losses >= value_at_risk].mean())
    confidence_label = f"{cvar_confidence:.4f}".rstrip("0").rstrip(".").replace(".", "_")
    result: dict[str, float | int] = {
        "months": len(clean),
        "cumulative_return": float(wealth.iloc[-1] - 1.0),
        "annualized_return": annual_return,
        "annualized_volatility": annual_volatility,
        "sharpe": float(sharpe),
        "maximum_drawdown": float(drawdown.min()),
        f"monthly_loss_var_{confidence_label}": value_at_risk,
        f"monthly_loss_cvar_{confidence_label}": cvar,
        "loss_tail_confidence": cvar_confidence,
    }
    if benchmark_returns is not None:
        benchmark = pd.to_numeric(benchmark_returns, errors="coerce").reindex(clean.index)
        active = (clean - benchmark).dropna()
        tracking_error = float(active.std(ddof=1) * np.sqrt(periods_per_year))
        information_ratio = (
            float(active.mean() * periods_per_year / tracking_error)
            if tracking_error > 0
            else np.nan
        )
        result.update(
            {
                "annualized_tracking_error": tracking_error,
                "information_ratio": information_ratio,
            }
        )
    return result


def backtest_summary(periods: pd.DataFrame) -> pd.DataFrame:
    required = {
        "month",
        "method",
        "net_return",
        "benchmark_return",
        "one_way_turnover",
        "transaction_cost",
    }
    missing = required.difference(periods.columns)
    if missing:
        raise ValueError(f"period results missing columns: {sorted(missing)}")
    rows: list[dict[str, float | int | str]] = []
    for method, frame in periods.groupby("method", observed=True):
        indexed = frame.set_index("month")
        risk_free = (
            indexed["realized_risk_free_return"]
            if "realized_risk_free_return" in indexed.columns
            else 0.0
        )
        metrics = performance_metrics(
            indexed["net_return"],
            benchmark_returns=indexed["benchmark_return"],
            risk_free_returns=risk_free,
        )
        rows.append(
            {
                "method": method,
                **metrics,
                "average_monthly_one_way_turnover": float(frame["one_way_turnover"].mean()),
                "total_transaction_cost": float(frame["transaction_cost"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)
