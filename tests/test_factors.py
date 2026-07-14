from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from constrained_portfolio.factors import fetch_french_factors


def _archive(name: str, text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, text)
    return buffer.getvalue()


class _Response:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


class _Session:
    def __init__(self, payloads: list[bytes]) -> None:
        self.payloads = list(payloads)
        self.calls = 0

    def get(self, url: str, timeout: float) -> _Response:
        del url, timeout
        payload = self.payloads[self.calls]
        self.calls += 1
        return _Response(payload)


def test_french_factor_fetch_cache_units_and_provenance(tmp_path: Path) -> None:
    ff5 = _archive(
        "ff5.csv",
        "notes\n,Mkt-RF,SMB,HML,RMW,CMA,RF\n202001,1.00,2.00,3.00,4.00,5.00,0.10\n"
        "202002,-1.00,1.50,2.50,3.50,4.50,0.10\nAnnual Factors\n",
    )
    momentum = _archive(
        "mom.csv",
        "notes\n,Mom\n202001,6.00\n202002,-2.00\nAnnual Factors\n",
    )
    session = _Session([ff5, momentum])
    factors = fetch_french_factors(tmp_path, session=session)

    assert session.calls == 2
    assert factors.loc["2020-01-31", "MKT_RF"] == pytest.approx(0.01)
    assert factors.loc["2020-01-31", "MOM"] == pytest.approx(0.06)
    provenance = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["provenance"] == "public"
    assert len(provenance["sources"]["ff5"]["sha256"]) == 64

    cached = fetch_french_factors(tmp_path, session=_Session([]))
    assert cached.attrs["provider"] == "Kenneth R. French Data Library"
    assert len(cached) == 2
