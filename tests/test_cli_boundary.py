from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from constrained_portfolio.cli import _private_output, _write_repair


def test_repository_private_output_is_restricted_to_ignored_output_root() -> None:
    repository = Path(__file__).resolve().parents[1]
    permitted = _private_output(repository / "outputs" / "private" / "study")
    assert permitted == (repository / "outputs" / "private" / "study").resolve()
    with pytest.raises(ValueError, match="restricted"):
        _private_output(repository / "docs" / "private" / "leak")
    assert "**/private/**" in (repository / ".gitignore").read_text(encoding="utf-8")


def test_repair_summary_contains_aggregate_diagnostics_only(tmp_path: Path) -> None:
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-12-30", "2025-12-31"]),
            "permno": [701, 803],
            "ticker": ["AAA", "BBB"],
        }
    )

    class FakeAdapter:
        def repaired_daily_panel(self, *, verify_known_snapshot: bool) -> pd.DataFrame:
            assert verify_known_snapshot
            return daily

        def monthly_returns(self, *, verify_known_snapshot: bool) -> pd.DataFrame:
            assert verify_known_snapshot
            return pd.DataFrame({"month": [pd.Timestamp("2025-12-31")], "return": [0.0]})

    _write_repair(FakeAdapter(), tmp_path)
    serialized = (tmp_path / "repair_summary.json").read_text(encoding="utf-8")
    summary = json.loads(serialized)

    assert summary["identity_mapping_one_to_one"] is True
    assert not any(key.endswith("_permno") for key in summary)
    assert "701" not in serialized
    assert "803" not in serialized
