# Constrained Portfolio Construction

I built this Python project to compare portfolio methods under position limits, turnover limits, trading costs, and risk constraints.

It includes equal weight, minimum variance, mean variance, maximum Sharpe, historical CVaR, and benchmark-relative portfolios. The exact-cardinality CVaR model uses binary variables to select a fixed number of holdings and is solved with SciPy/HiGHS. The continuous models use CVXPY. There is also a separate 130/30 example.

Each month's weights use data available through the previous month. I use covariance shrinkage and check the solver's output against the constraints before accepting it. The reports include factor attribution and bootstrap uncertainty estimates.

## Run

Use Python 3.11 or newer. The public demo uses synthetic data and writes its results to `outputs/public/demo`.

```sh
python -m pip install -e ".[dev]"
python -m constrained_portfolio demo
pytest
```

To use your own retained data, set `CPC_LICENSED_DATA_ROOT` to the folder containing `wrds_equity_daily.parquet` and `wrds_symbol_map.parquet`, then run:

```sh
python -m constrained_portfolio repair-private
python -m constrained_portfolio evaluate-private
```

The repair runs in memory. Licensed data are not copied into this repo, and private results go to ignored folders.

## Notes

The retained data are split-adjusted prices, so returns exclude dividends. The universe uses constituents active on December 31, 2025, which creates survivorship bias. The equal-weight comparison is not an index replica.

The 2024 to 2025 evaluation period has already been inspected. Some CVaR estimates have only one or two tail observations, and realized tracking error can exceed the model's estimated limit. These results show how the methods behave, not a proven trading strategy.

More detail is in the [methodology](docs/methodology.md), [data notes](docs/data_policy.md), and [research context](docs/research_context_2026-07-13.md).
