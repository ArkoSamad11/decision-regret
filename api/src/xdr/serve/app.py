"""FastAPI routes (SPEC.md §12). Read-only, no authentication, stateless --
scores to zero, backed by a DuckDB file baked into the image (SPEC.md §5, §14).
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from xdr.config import load_config
from xdr.serve import store
from xdr.serve.schemas import (
    CalibrationReport,
    HealthResponse,
    MatchSummary,
    Moment,
    TransferReport,
)

REPO_ROOT = Path(__file__).resolve().parents[4]

app = FastAPI(title="xDR API", description="Expected decision regret", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

_config = load_config("base.yaml")
_artifacts_dir = REPO_ROOT / _config.paths.artifacts


def _connect() -> duckdb.DuckDBPyConnection | None:
    db_path = _artifacts_dir / "xdr.duckdb"
    if not db_path.exists():
        return None
    return duckdb.connect(str(db_path), read_only=True)


def _run_meta() -> dict | None:
    run_path = _artifacts_dir / "run.json"
    if not run_path.exists():
        return None
    return json.loads(run_path.read_text(encoding="utf-8"))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    meta = _run_meta()
    db_path = _artifacts_dir / "xdr.duckdb"
    status = "ok" if db_path.exists() and meta else "no_artifacts"
    return HealthResponse(
        status=status,
        model_version=meta.get("config_hash") if meta else None,
        run_id=meta.get("run_id") if meta else None,
    )


@app.get("/matches", response_model=list[MatchSummary])
def matches() -> list[MatchSummary]:
    con = _connect()
    if con is None:
        return []
    try:
        return [MatchSummary(**row) for row in store.list_matches(con)]
    finally:
        con.close()


@app.get("/matches/{match_id}/moments", response_model=list[Moment])
def match_moments(match_id: str, min_value: float | None = None, limit: int = 100) -> list[Moment]:
    con = _connect()
    if con is None:
        raise HTTPException(status_code=404, detail="No scored data available")
    try:
        moments = store.list_moments(con, match_id, min_value=min_value, limit=limit)
    finally:
        con.close()
    if not moments:
        raise HTTPException(status_code=404, detail=f"No moments found for match {match_id}")
    return [Moment(**m) for m in moments]


@app.get("/moments/{moment_id}", response_model=Moment)
def moment(moment_id: str) -> Moment:
    con = _connect()
    if con is None:
        raise HTTPException(status_code=404, detail="No scored data available")
    try:
        m = store.get_moment(con, moment_id)
    finally:
        con.close()
    if m is None:
        raise HTTPException(status_code=404, detail=f"Moment {moment_id} not found")
    return Moment(**m)


@app.get("/calibration", response_model=CalibrationReport)
def calibration(split: str = "test") -> CalibrationReport:
    meta = _run_meta()
    if meta is None:
        raise HTTPException(status_code=404, detail="No trained run available")
    report_path = _artifacts_dir / f"report_{split}.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"No report for split '{split}'")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    # SPEC.md §11.2: report scoring and conceding side by side; the headline
    # calibration route surfaces `label_scores`, the harder-to-calibrate,
    # lower-base-rate head.
    head = report["label_scores"]["calibrated"]
    curve = [
        {
            "bin_lower": b["bin_lower"],
            "bin_upper": b["bin_upper"],
            "mean_predicted": b["mean_predicted"],
            "mean_observed": b["mean_observed"],
            "count": b["count"],
        }
        for b in report["label_scores"]["reliability_curve"]
    ]
    return CalibrationReport(
        split=split,
        brier=head["brier"],
        reliability=head["reliability"],
        resolution=head["resolution"],
        uncertainty=head["uncertainty"],
        ece=head["ece"],
        curve=curve,
        n=report["label_scores"]["n"],
    )


@app.get("/transfer", response_model=TransferReport)
def transfer() -> TransferReport:
    path = _artifacts_dir / "transfer.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No transfer study available")
    report = json.loads(path.read_text(encoding="utf-8"))
    return TransferReport(**report)
