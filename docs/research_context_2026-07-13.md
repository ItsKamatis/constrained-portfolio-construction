# Research context - 2026-07-13

## Why this repository exists

This is the curated successor to the KRNN tail-risk portfolio work. The earlier
model's weak out-of-sample predictive R-squared remains a research finding; it is
not hidden or converted into an alpha claim. The narrower question here is how
noisy estimates, realistic constraints, costs, and attribution affect portfolio
construction.

The KRNN forecasting model, independent per-asset EVT simulations, the old
"tail-risk parity" label, claims of automatic distributional robustness, and
claims to a Nasdaq-100 backtest were not migrated.

## Data available at this checkpoint

- The retained CRSP-derived price panel is read only from an external directory.
  No live WRDS access is required and no licensed row-level panel is stored here.
- The verified repair has 192,388 daily rows for 101 securities from 2018-01-02
  through 2025-12-31. The universe is based on mappings active at the end date.
- Returns are split-adjusted price returns, not dividend-inclusive total returns.
- There are no point-in-time index weights, market caps, sectors, or fundamentals.
- Kenneth French FF5, momentum, and RF data are public inputs with cached source
  URLs and hashes. The public demo uses deterministic synthetic assets and factors.

These facts require current-constituent and survivorship-bias disclosures, an
eligible-universe equal-weight reference rather than an index claim, and omission
of sector constraints and Brinson attribution.

## Current research design

The code compares seven long-only construction rules plus a separately labeled
130/30 mathematical appendix. Decisions are monthly and use only prior data.
Solutions are rejected when numerical residuals violate their constraints.

Maximum estimated Sharpe is searched on a constrained efficient-frontier grid
beginning at the constrained minimum-variance portfolio. Construction uses only
lagged factor RF. When no feasible portfolio has positive forecast excess return,
the walk-forward rule records a minimum-variance fallback instead of presenting a
negative-ratio frontier result as maximum Sharpe.

The 95% historical CVaR objectives have only 18 to 29 complete joint monthly
scenarios in the retained study, so their tail is often one or two observations.
The 8% tracking-error constraint is ex-ante; reported tracking error is ex-post.
Both limitations are stated in generated reports.

The planned split is 2018-2019 warm-up, 2020-2023 development, and 2024-2025
already-inspected evaluation. The last interval is not a pristine test.

## Verified state

- Public test configuration: 28 passed, 1 licensed-data test skipped.
- Retained-data configuration: 29 passed.
- Public and private workflows each produce 72 months by 8 methods.
- The private workflow has no lookahead violations; return, turnover, attribution,
  and solver checks reconcile within documented tolerances.
- Public and private generated outputs are ignored. Private outputs should not be
  published without a separate review of the governing data terms.

## Focus for a later revision

Do not add more optimizers merely to increase method count. The highest-value next
step is a legally reusable total-return dataset with point-in-time benchmark and
classification data. That would support a true benchmark-relative study, sector
constraints, and Brinson attribution. A longer or higher-frequency joint history
would be needed before making stronger empirical CVaR claims. Forecast research
should resume only after those data and evaluation controls improve.
