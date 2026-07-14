# Data policy and empirical limitations

## Public repository boundary

The distributable repository contains code, documentation, and deterministic
synthetic fixtures only. It does not contain CRSP/WRDS rows, extracts, or
transformed row-level panels. The private adapter reads an external path supplied
by `CPC_LICENSED_DATA_ROOT` or `--data-root`; the repair command validates its
panel in memory and persists only aggregate diagnostics. Private study outputs
such as weights and summaries remain ignored and must not be published without
review of the governing data terms.

Every licensed-data-derived output must be written below a directory literally
named `private`; the command-line interface enforces this. `data/private/`,
`outputs/private/`, and `cache/private/` are ignored. Users remain responsible
for the terms governing their own data access.

## Offline identity repair

The retained files can be repaired without querying WRDS:

1. Filter the symbol map to intervals active on 2025-12-31 (inclusive).
2. Require a one-to-one ticker–PERMNO mapping at that date.
3. Join those PERMNOs to the complete daily price file by PERMNO.
4. Keep PERMNO as the internal security identifier and ticker as display-only.

For the verified retained snapshot this gives 192,388 daily rows for 101
securities from 2018-01-02 through 2025-12-31. Private diagnostics additionally
verify that historically reused ticker strings do not cross PERMNO identities.
These are snapshot integrity checks, not general rules for other CRSP extracts.

## Eligibility and no-lookahead rule

At each private-data holding month an asset must have at least 252 valid daily
close returns available before the first day of that month, and its latest price
observation must fall in the immediately preceding month. Monthly estimation
then uses only dates strictly before the holding month. Missing held-asset
returns are errors; they are never replaced by zero.

The public synthetic panel has no daily source. Its eligibility rule is a
deliberately labeled twelve-prior-month analogue and is not treated as
equivalent evidence.

## Limitations that must accompany results

- Returns are split-adjusted close-to-close price returns, not
  dividend-inclusive total returns.
- The universe is selected at 2025-12-31 and backfilled historically, producing
  current-constituent and survivorship bias.
- No historical index weights or market caps are available. Eligible-universe
  equal weight is a reference portfolio, not the Nasdaq-100 or another index.
- No point-in-time sectors or fundamentals are available. Sector constraints
  and Brinson attribution are intentionally omitted.
- 2024–2025 has already been inspected. It is an evaluation interval, not a
  pristine unseen test.

Kenneth French FF5 and momentum returns are fetched from the public
[Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).
The downloader stores source URLs, retrieval time, date coverage, and SHA-256
checksums with the cache.
