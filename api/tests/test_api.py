"""SPEC.md §15: every route's response validates against its schema.

Runs against whatever is in artifacts/ when the test suite executes. In this
repository that's the real Euro 2024 run (`make reproduce` populates it); a
fresh checkout with no artifacts still gets meaningful coverage of the
"no data yet" paths, which `/health` and `/matches` are specifically designed
to answer without raising (SPEC.md §12: "Must work with no artifacts present").
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from xdr.serve.app import app

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
HAS_ARTIFACTS = (ARTIFACTS_DIR / "xdr.duckdb").exists()

client = TestClient(app)


def test_health_always_responds():
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] in ("ok", "no_artifacts")


def test_matches_returns_list_even_with_no_data():
    res = client.get("/matches")
    assert res.status_code == 200
    assert isinstance(res.json(), list)


@pytest.mark.skipif(not HAS_ARTIFACTS, reason="requires a built artifacts/xdr.duckdb")
def test_matches_are_ordered_by_mean_value_desc():
    res = client.get("/matches")
    assert res.status_code == 200
    values = [m["mean_value"] for m in res.json()]
    assert values == sorted(values, reverse=True)


@pytest.mark.skipif(not HAS_ARTIFACTS, reason="requires a built artifacts/xdr.duckdb")
def test_match_moments_validates_against_schema():
    match_id = client.get("/matches").json()[0]["match_id"]
    res = client.get(f"/matches/{match_id}/moments")
    assert res.status_code == 200
    moments = res.json()
    assert len(moments) > 0
    first = moments[0]
    assert first["regret"] is None  # no counterfactual layer in this pass
    assert first["options"] == []
    assert "value" in first["chosen"]


@pytest.mark.skipif(not HAS_ARTIFACTS, reason="requires a built artifacts/xdr.duckdb")
def test_moment_latest_resolves():
    res = client.get("/moments/latest")
    assert res.status_code == 200
    assert res.json()["moment_id"]


def test_moment_unknown_id_is_404():
    res = client.get("/moments/not-a-real-id")
    assert res.status_code == 404


@pytest.mark.skipif(not HAS_ARTIFACTS, reason="requires a built artifacts/xdr.duckdb")
def test_calibration_validates_against_schema():
    res = client.get("/calibration?split=test")
    assert res.status_code == 200
    body = res.json()
    assert 0 <= body["brier"] <= 1
    assert 0 <= body["ece"] <= 1
    assert body["n"] > 0


def test_calibration_unknown_split_is_404():
    res = client.get("/calibration?split=not-a-real-split")
    assert res.status_code == 404
