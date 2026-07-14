from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import pandas as pd

from .attribution import aggregate_attribution
from .backtest import BacktestResult
from .metrics import backtest_summary
from .uncertainty import moving_block_bootstrap


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No observations._"
    headers = [str(column) for column in frame.columns]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in frame.iterrows():
        formatted = []
        for value in row:
            if isinstance(value, float):
                formatted.append("" if pd.isna(value) else f"{value:.6f}")
            else:
                formatted.append(str(value))
        rows.append("| " + " | ".join(formatted) + " |")
    return "\n".join(rows)


def build_research_report(
    result: BacktestResult,
    summary: pd.DataFrame,
    split_summary: pd.DataFrame,
    uncertainty: pd.DataFrame,
    *,
    limitations: Mapping[str, str],
    provenance: Mapping[str, object],
) -> str:
    period_start = pd.to_datetime(result.periods["month"]).min().date()
    period_end = pd.to_datetime(result.periods["month"]).max().date()
    display_columns = [
        "method",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "maximum_drawdown",
        "annualized_tracking_error",
        "information_ratio",
        "average_monthly_one_way_turnover",
        "total_transaction_cost",
    ]
    core_evaluation = summary.loc[
        summary["method"].ne("130_30_mathematical_appendix"),
        [
            *display_columns,
        ]
    ]
    appendix_evaluation = summary.loc[
        summary["method"].eq("130_30_mathematical_appendix"), display_columns
    ]
    factor_summary = (
        aggregate_attribution(result.active_attribution)
        if not result.active_attribution.empty
        else pd.DataFrame()
    )
    scenario_counts = result.periods.groupby("month", observed=True)["joint_scenarios"].first()
    scenario_min = int(scenario_counts.min())
    scenario_median = float(scenario_counts.median())
    scenario_max = int(scenario_counts.max())
    sharpe_rows = result.periods.loc[
        result.periods["method"].eq("maximum_sharpe_with_gmv_fallback")
    ]
    sharpe_fallbacks = int(
        sharpe_rows["construction_rule"]
        .eq("gmv_fallback_no_positive_expected_excess_return")
        .sum()
    )
    sharpe_months = int(len(sharpe_rows))
    sharpe_fallback_note = (
        "The maximum-Sharpe comparison used its explicit global-minimum-variance "
        f"fallback in {sharpe_fallbacks} of {sharpe_months} months because no feasible "
        "portfolio had positive forecast excess return in those windows."
        if sharpe_months
        else "The maximum-Sharpe comparison was not selected for this run."
    )
    config_table = pd.DataFrame(
        [{"setting": key, "value": value} for key, value in asdict(result.config).items()]
    )
    limitation_lines = "\n".join(f"- **{key}:** {value}" for key, value in limitations.items())
    return f"""# Constrained Portfolio Construction Study

## Summary

This report covers monthly holding periods from {period_start} through
{period_end}. Weights are estimated only from prior observations. The reference
is the monthly rebalanced eligible-universe equal-weight portfolio; it is not an
index replica. Arithmetic returns are compounded when reporting cumulative and
annualized performance.

Construction-time Sharpe uses the annualized mean `RF` observed only in the
lagged estimation window when factor data are supplied (zero otherwise).
Reported Sharpe uses the realized monthly `RF` from the declared factor source.
{sharpe_fallback_note}

{_markdown_table(core_evaluation)}

These estimates are descriptive and sensitive to the selected universe,
lookback, cost model, and already-inspected evaluation interval. Bootstrap
intervals below measure sampling variability; they do not remove selection or
survivorship bias.

The empirical CVaR optimizers use {scenario_min} to {scenario_max} complete joint
monthly scenarios (median {scenario_median:.0f}) at the configured
{result.config.cvar_confidence:.0%} confidence. That leaves roughly one or two
observations in the tail in many windows, so CVaR is a conservative empirical
ranking here, not a stable tail-distribution estimate.

## Walk-forward configuration

{_markdown_table(config_table)}

The {result.config.tracking_error_limit:.0%} tracking-error limit is an ex-ante
constraint under each decision month's covariance estimate. The tracking error
reported above is ex-post realized tracking error and can exceed that input.

## Block-bootstrap uncertainty

{_markdown_table(uncertainty)}

## Summed monthly arithmetic active-return attribution

Factor attribution uses lagged rolling asset regressions with FF5+momentum factor
definitions. The factor-data source is stated in provenance (synthetic for the
demo; Kenneth French data for the private study). `specific` is a reconciliation
residual, not proof of security-selection skill. Transaction costs are separate.
These are sums of one-period arithmetic contributions; they are not geometric
multi-period attribution and need not equal compounded cumulative active return.

{_markdown_table(factor_summary)}

## Development and evaluation periods

The 2024-2025 rows remain labeled `evaluation_already_inspected`; they are not
described as a pristine test.

{_markdown_table(split_summary)}

## Data and interpretation limitations

{limitation_lines}

## Provenance

```json
{json.dumps(dict(provenance), indent=2, default=str)}
```

## Method note

The 130/30 result, when present, is reported separately below as a mathematical
appendix. It assumes exact
130% long and 30% short exposure but omits borrow availability, fees, recalls,
margin, and financing. It is not directly comparable to the long-only methods.

{_markdown_table(appendix_evaluation)}
"""


def write_study_outputs(
    result: BacktestResult,
    output_dir: str | Path,
    *,
    limitations: Mapping[str, str],
    provenance: Mapping[str, object],
    bootstrap_samples: int = 200,
    bootstrap_seed: int = 20260713,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = backtest_summary(result.periods)
    split_rows: list[pd.DataFrame] = []
    for split, periods in result.periods.groupby("split", observed=True):
        split_frame = backtest_summary(periods)
        split_frame.insert(0, "split", split)
        split_rows.append(split_frame)
    split_summary = pd.concat(split_rows, ignore_index=True)
    uncertainty_rows: list[pd.DataFrame] = []
    for method, periods in result.periods.groupby("method", observed=True):
        risk_free = (
            periods.set_index("month")["realized_risk_free_return"]
            if "realized_risk_free_return" in periods.columns
            else 0.0
        )
        interval = moving_block_bootstrap(
            periods.set_index("month")["net_return"],
            risk_free_returns=risk_free,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        interval.insert(0, "method", method)
        uncertainty_rows.append(interval)
    uncertainty = pd.concat(uncertainty_rows, ignore_index=True)
    paths = {
        "periods": output / "monthly_results.csv",
        "weights": output / "weights.csv",
        "holdings_attribution": output / "holdings_attribution.csv",
        "active_attribution": output / "active_attribution.csv",
        "optimizer_diagnostics": output / "optimizer_diagnostics.csv",
        "summary": output / "summary.csv",
        "split_summary": output / "split_summary.csv",
        "uncertainty": output / "bootstrap_uncertainty.csv",
        "report": output / "research_report.md",
        "provenance": output / "provenance.json",
    }
    result.periods.to_csv(paths["periods"], index=False)
    result.weights.to_csv(paths["weights"], index=False)
    result.holdings_attribution.to_csv(paths["holdings_attribution"], index=False)
    result.active_attribution.to_csv(paths["active_attribution"], index=False)
    result.optimizer_diagnostics.to_csv(paths["optimizer_diagnostics"], index=False)
    summary.to_csv(paths["summary"], index=False)
    split_summary.to_csv(paths["split_summary"], index=False)
    uncertainty.to_csv(paths["uncertainty"], index=False)
    effective_provenance = {
        **dict(provenance),
        "walk_forward_config": asdict(result.config),
        "risk_free_policy": (
            "lagged factor RF annualized for maximum-Sharpe construction and realized monthly "
            "factor RF for reported Sharpe; zero only when no factor data are supplied"
        ),
    }
    paths["provenance"].write_text(
        json.dumps(effective_provenance, indent=2, default=str), encoding="utf-8"
    )
    paths["report"].write_text(
        build_research_report(
            result,
            summary,
            split_summary,
            uncertainty,
            limitations=limitations,
            provenance=effective_provenance,
        ),
        encoding="utf-8",
    )
    return paths
