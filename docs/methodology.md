# Methodology

## Timeline and estimates

The study rebalances monthly. A weight used for month `t` is estimated from data
through `t-1`; diagnostic output records both the estimation end and holding
month so this invariant can be tested. The planned periods are:

- 2018–2019: estimation warm-up;
- 2020–2023: development;
- 2024–2025: already-inspected evaluation.

Monthly returns are arithmetic price returns. Cumulative performance compounds
`1 + return`; it is never obtained by summing log returns. Expected returns are
sample arithmetic means shrunk halfway toward the cross-sectional grand mean by
default. Covariance is the Ledoit–Wolf estimator, annualized by 12. CVaR uses
complete historical **joint** scenarios, preserving cross-asset dependence.

## Curated construction methods

- Equal weight is both a strategy and the reference portfolio.
- Global minimum variance is long-only.
- Mean–variance uses the documented shrunk means.
- Maximum estimated Sharpe is selected from solved points on a constrained
  efficient-frontier grid running from the constrained global-minimum-variance
  return to the maximum feasible return; it is not a nonconvex ratio shortcut.
  If no feasible portfolio has positive expected excess return, the pure
  maximum-Sharpe function rejects the problem and the walk-forward comparison
  switches to an explicitly labeled global-minimum-variance risk fallback.
- Historical CVaR minimizes empirical loss CVaR and makes no claim of automatic
  distributional robustness.
- The benchmark-relative portfolio constrains annualized tracking error,
  one-way turnover, and active FF5+momentum exposures, and penalizes one-way
  turnover in its objective.
- Exact-cardinality CVaR uses binary inclusion variables and SciPy/HiGHS MILP.

Every solution passes numerical residual checks for its budget, bounds, and
special constraints. Mixed-integer output additionally checks exact selected
count, integrality, and minimum selected position.

When factor data are supplied, the maximum-Sharpe construction rate is the
annualized arithmetic mean of `RF` observations in the same lagged joint window;
the holding month's `RF` is not used to choose weights. Reported Sharpe uses the
realized monthly `RF`. Workflows without factor data fall back explicitly to zero.

The private study has 18 to 29 complete monthly joint scenarios (median 21).
At 95% confidence, empirical CVaR is therefore driven by roughly one or two tail
observations in many windows. It remains a valid historical-scenario objective,
but it is not treated as a stable estimate of a tail distribution.

The 8% tracking-error constraint is ex-ante under the covariance estimate used
at each rebalance. Reported tracking error is an ex-post realized statistic and
can exceed the ex-ante input without indicating a solver-constraint violation.

## Turnover, cost, and drift

For a transition between fully invested portfolios, one-way turnover is
`0.5 * sum(abs(target - drifted_prior))` across the union of both universes.
Universe exits therefore cannot disappear through renormalization. The default
cost is 10 basis points per unit of one-way turnover, matching the optimizer's
turnover-penalty convention. Inception is labeled separately: it has no one-way
turnover statistic and its cost base is the portfolio's initial gross traded
notional (1.0 for long-only and 1.6 for the 130/30 appendix).

After each holding period, weights drift as
`w_i * (1 + r_i) / (1 + portfolio_return)` before the next rebalance. Gross
return minus transaction cost equals net return exactly.

## Attribution and uncertainty

Rolling asset exposures regress excess asset returns on MKT-RF, SMB, HML, RMW,
CMA, and momentum using only pre-decision observations, with at least 18
complete monthly observations (enough for the intercept plus six slopes). Active factor
contribution is active exposure times realized factor return. Specific return is
the arithmetic reconciliation residual; cost is separate. Holdings
contributions reconcile to gross return. Aggregated attribution tables sum
one-period arithmetic contributions. They are not a geometric multi-period
attribution and are not expected to equal compounded cumulative active return.

Moving-block bootstrap intervals preserve short-run monthly dependence. They
describe sampling uncertainty only and do not repair survivorship, selection,
or model-choice bias.

## 130/30 appendix

The appendix forms lagged long and short candidate sets and solves for exactly
130% long and 30% short exposure. It omits borrow availability, borrow fees,
recall risk, financing, and margin. It is an optimization illustration rather
than an investable backtest and is reported separately from long-only methods.
