from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backtest import WalkForwardConfig, run_walk_forward
from .data import (
    DataLimitations,
    PrivateCRSPAdapter,
    monthly_return_matrix,
    synthetic_monthly_eligibility,
    synthetic_monthly_panel,
)
from .factors import fetch_french_factors
from .report import write_study_outputs


def _private_output(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    repository = Path(__file__).resolve().parents[2]
    try:
        relative = resolved.relative_to(repository)
    except ValueError:
        relative = None
    if relative is not None:
        permitted_root = repository / "outputs" / "private"
        try:
            resolved.relative_to(permitted_root)
        except ValueError as exc:
            raise ValueError(
                "inside the repository, licensed outputs are restricted to outputs/private"
            ) from exc
    if "private" not in {part.lower() for part in resolved.parts}:
        raise ValueError("licensed-data-derived output path must contain a 'private' directory")
    return resolved


def _write_repair(adapter: PrivateCRSPAdapter, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    daily = adapter.repaired_daily_panel(verify_known_snapshot=True)
    monthly = adapter.monthly_returns(verify_known_snapshot=True)
    summary = {
        "provenance": "licensed_user_input",
        "rows": len(daily),
        "securities": int(daily["permno"].nunique()),
        "start": str(daily["date"].min().date()),
        "end": str(daily["date"].max().date()),
        "identity_mapping_one_to_one": bool(
            daily.groupby("ticker", observed=True)["permno"].nunique().eq(1).all()
        ),
        "limitations": DataLimitations().to_dict(),
        "source_location": "external configurable directory; not copied into repository",
        "persistence": (
            "The repaired daily and monthly panels were produced and validated in memory; "
            "no row-level licensed or transformed panel was written."
        ),
        "monthly_rows_validated_in_memory": len(monthly),
    }
    (output / "repair_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cpc", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("demo", help="run the deterministic public-safe synthetic study")
    demo.add_argument("--output", default="outputs/public/demo")
    demo.add_argument("--bootstrap-samples", type=int, default=200)
    demo.add_argument("--no-appendix", action="store_true")

    repair = commands.add_parser("repair-private", help="repair and validate external licensed files")
    repair.add_argument("--data-root")
    repair.add_argument("--output", default="outputs/private/repair")

    evaluate = commands.add_parser(
        "evaluate-private", help="run the full study on the retained external panel"
    )
    evaluate.add_argument("--data-root")
    evaluate.add_argument("--output", default="outputs/private/study")
    evaluate.add_argument("--factor-cache", default="data/cache/french")
    evaluate.add_argument("--refresh-factors", action="store_true")
    evaluate.add_argument("--bootstrap-samples", type=int, default=200)
    evaluate.add_argument("--no-appendix", action="store_true")

    factors = commands.add_parser("fetch-factors", help="cache public FF5 plus momentum data")
    factors.add_argument("--cache", default="data/cache/french")
    factors.add_argument("--refresh", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "fetch-factors":
        factors = fetch_french_factors(args.cache, refresh=args.refresh)
        print(
            json.dumps(
                {
                    "rows": len(factors),
                    "start": str(factors.index.min().date()),
                    "end": str(factors.index.max().date()),
                    "provenance": factors.attrs,
                },
                indent=2,
                default=str,
            )
        )
        return 0
    if args.command == "repair-private":
        adapter = PrivateCRSPAdapter(args.data_root)
        output = _private_output(args.output)
        _write_repair(adapter, output)
        print(f"validated private repair written to {output}")
        return 0
    if args.command == "demo":
        returns, factors = synthetic_monthly_panel()
        eligibility = synthetic_monthly_eligibility(returns, minimum_prior_months=12)
        result = run_walk_forward(
            returns,
            factors=factors,
            eligibility=eligibility,
            include_130_30_appendix=not args.no_appendix,
        )
        paths = write_study_outputs(
            result,
            args.output,
            limitations={
                "provenance": "Synthetic asset and factor returns; no real securities.",
                "eligibility": (
                    "Twelve-prior-month synthetic analogue; not equivalent to the private "
                    "252-daily-observation rule."
                ),
                "use": "Engineering demonstration only; no empirical investment claim.",
            },
            provenance={"asset_data": "synthetic", "factor_data": "synthetic", "seed": 20260713},
            bootstrap_samples=args.bootstrap_samples,
        )
        print(f"synthetic study written to {paths['report'].parent.resolve()}")
        return 0
    if args.command == "evaluate-private":
        adapter = PrivateCRSPAdapter(args.data_root)
        monthly = adapter.monthly_returns(verify_known_snapshot=True)
        returns = monthly_return_matrix(monthly)
        study_months = returns.index[
            (returns.index >= "2020-01-31") & (returns.index <= "2025-12-31")
        ]
        eligibility = adapter.monthly_eligibility(
            study_months,
            minimum_daily_returns=252,
            verify_known_snapshot=True,
        )
        factors = fetch_french_factors(args.factor_cache, refresh=args.refresh_factors)
        result = run_walk_forward(
            returns,
            factors=factors,
            eligibility=eligibility,
            include_130_30_appendix=not args.no_appendix,
        )
        output = _private_output(args.output)
        paths = write_study_outputs(
            result,
            output,
            limitations=DataLimitations().to_dict(),
            provenance={
                "asset_data": "licensed_user_input",
                "asset_data_location": "external configurable directory; not copied",
                "factor_data": factors.attrs,
                "eligibility": eligibility.attrs,
            },
            bootstrap_samples=args.bootstrap_samples,
        )
        print(f"private study written to {paths['report'].parent.resolve()}")
        return 0
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
