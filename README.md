# Constrained Portfolio Construction & Factor Attribution

This repository studies the layer between noisy return estimates and an
investable portfolio. It compares a deliberately small set of construction
methods under position, turnover, tracking-error, factor-exposure, transaction
cost, and cardinality constraints, then explains active returns using holdings,
cost, and factor attribution.

The primary empirical dataset is **not included**. The private adapter reads an
external, user-supplied CRSP-derived directory. The repair stays in memory and
persists only aggregate validation; no licensed or transformed row-level panel
is copied into this repository. Study summaries, weights, and diagnostics are
written only to ignored private-output paths. A deterministic synthetic panel
exercises the entire public workflow.

## Research design

- Monthly decisions are lagged: weights for month `t` use only information
  dated through month `t-1`.
- 2018–2019 is the estimation warm-up, 2020–2023 is the development period,
  and 2024–2025 is an already-inspected evaluation period—not a pristine test.
- Missing returns are never replaced with zero. A missing held-asset return is
  a hard data error rather than an assumed flat price.
- Covariance uses Ledoit–Wolf shrinkage. Expected returns use an explicit
  configurable shrink toward the cross-sectional grand mean.
- Maximum-Sharpe construction uses a constrained efficient-frontier grid that
  begins at the constrained global-minimum-variance portfolio. With factor data,
  its risk-free rate is the annualized mean `RF` from the lagged estimation window;
  reported Sharpe uses the realized monthly `RF`. If no feasible portfolio has
  positive forecast excess return, the walk-forward comparison uses and labels a
  global-minimum-variance fallback instead of misreporting a negative-ratio search.
- Optimizer output is rejected unless budget, bound, exposure, turnover, and
  (where relevant) mixed-integer residual checks pass.
- The reference portfolio is eligible-universe equal weight. It is not called
  the Nasdaq-100 and does not claim to reproduce an index.

## Methods

1. Eligible-universe equal weight.
2. Long-only global minimum variance.
3. Long-only shrinkage mean–variance.
4. Maximum estimated Sharpe with an explicit no-positive-excess GMV fallback.
5. Historical joint-scenario CVaR.
6. Benchmark-relative optimization with tracking-error, turnover, factor, and
   transaction-cost controls.
7. Exact-cardinality historical CVaR solved as a SciPy/HiGHS MILP.
8. A separately labeled 130/30 mathematical appendix.

## Quick start

```powershell
python -m pip install -e ".[dev]"
python -m constrained_portfolio demo --output outputs/public/demo
pytest
```

To validate the retained private files without copying them:

```powershell
$env:CPC_LICENSED_DATA_ROOT = "D:\path\to\licensed\data\raw"
python -m constrained_portfolio repair-private --output outputs/private/repair
python -m constrained_portfolio evaluate-private --output outputs/private/study
```

The private adapter expects `wrds_equity_daily.parquet` and
`wrds_symbol_map.parquet`. See [docs/data_policy.md](docs/data_policy.md) and
[docs/methodology.md](docs/methodology.md) before interpreting results.

## Honest limitations

The retained panel contains split-adjusted **price** history, not
dividend-inclusive total returns. Its universe is selected using constituents
active on 2025-12-31 and backfilled through history, creating survivorship and
current-constituent bias. It has no historical benchmark weights, market caps,
fundamentals, or historical sectors. Sector constraints and Brinson attribution
are therefore intentionally omitted. Results are research illustrations, not
investment advice or a representation of a live investable strategy.

The 95% empirical CVaR estimates often have only one or two observations in
their realized tail because the complete joint windows are short. The 8%
tracking-error limit is ex-ante under the estimated covariance matrix; reported
tracking error is ex-post and can exceed that input.
