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
    assert "value" in first["chosen"]
    assert isinstance(first["options"], list)
    # regret is either genuinely null (unsupported) or a non-negative float
    # (SPEC.md §10/§13) -- never coerced to 0.0.
    assert first["regret"] is None or first["regret"] >= 0
    for option in first["options"]:
        assert option["scored"] is (option["value"] is not None)


@pytest.mark.skipif(not HAS_ARTIFACTS, reason="requires a built artifacts/xdr.duckdb")
def test_counterfactual_layer_produces_some_real_regret():
    """M8 landed: at least one served moment, somewhere, must have a real
    (non-null) regret with a real `best` alternative -- otherwise the
    counterfactual layer would be wired in but silently inert."""
    match_id = client.get("/matches").json()[0]["match_id"]
    moments = client.get(f"/matches/{match_id}/moments").json()
    with_regret = [m for m in moments if m["regret"] is not None]
    assert len(with_regret) > 0
    assert all(m["best"] is not None for m in with_regret)


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
