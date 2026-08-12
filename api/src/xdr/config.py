"""Pydantic settings loaded from YAML, with a single-level `extends` merge.

Every config in configs/ extends base.yaml and overrides only what differs
(SPEC.md §5, "Config files named {purpose}_{source}_to_{target}.yaml"). This
module is the only place that resolves `extends` -- callers always get a fully
merged, validated XdrConfig back.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

# Everything the project reads or writes -- configs/, data/, artifacts/, runs/,
# docs/ -- is addressed relative to the repo root, so the root has to be
# discoverable at runtime. `parents[3]` walks src/xdr/config.py back up to the
# checkout, which is correct for an editable install and for `python -m` runs
# from a clone.
#
# It is wrong for a non-editable install: `pip install ./api` copies the package
# into site-packages, where the same walk lands on the interpreter's lib/
# directory instead. The API image does exactly that (see api/Dockerfile), and
# the failure is quiet -- configs/ raises at import, but a missing artifacts/
# just makes /health report "no_artifacts" and every route return empty.
#
# XDR_ROOT is the override for that case. The image sets it to /app.
REPO_ROOT = Path(os.environ.get("XDR_ROOT") or Path(__file__).resolve().parents[3]).resolve()
CONFIG_DIR = REPO_ROOT / "configs"


class Competition(BaseModel):
    competition_id: int
    season_id: int
    name: str


class Competitions(BaseModel):
    train: list[Competition] = []
    test: list[Competition] = []


class Paths(BaseModel):
    raw: str = "data/raw"
    parquet: str = "data/parquet"
    artifacts: str = "artifacts"
    runs: str = "runs"
    docs: str = "docs"


class Split(BaseModel):
    unit: str = "match"
    train: float = 0.60
    validation: float = 0.15
    calibration: float = 0.10
    test: float = 0.15


class Features(BaseModel):
    action_window: int = 3
    max_players: int = 22


class Labels(BaseModel):
    horizon: int = 10


class LightGBMConfig(BaseModel):
    n_estimators: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 20
    early_stopping_rounds: int = 30


class ModelConfig(BaseModel):
    lightgbm: LightGBMConfig = LightGBMConfig()


class CalibrationConfig(BaseModel):
    method: str = "isotonic"


class SupportConfig(BaseModel):
    k_neighbors: int = 20
    min_support: float = 0.35


class EvaluationConfig(BaseModel):
    recalibrate_on_target: bool = False
    recalibration_fraction: float = 0.2
    bootstrap_unit: str = "match"
    bootstrap_draws: int = 10000
    report: list[str] = []


class ServeConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class DataConfig(BaseModel):
    competitions: Competitions = Competitions()


class XdrConfig(BaseModel):
    seed: int
    paths: Paths = Paths()
    data: DataConfig = DataConfig()
    split: Split = Split()
    features: Features = Features()
    labels: Labels = Labels()
    model: ModelConfig = ModelConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    support: SupportConfig = SupportConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    serve: ServeConfig = ServeConfig()

    config_name: str = "base"
    config_hash: str = ""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw(path: Path) -> dict[str, Any]:
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    extends = raw.get("extends")
    if extends:
        parent = _load_raw((path.parent / extends).resolve())
        return _deep_merge(parent, raw)
    return raw


def load_config(name_or_path: str) -> XdrConfig:
    """Load a config by filename (resolved against configs/) or an absolute path."""
    path = Path(name_or_path)
    if not path.is_absolute():
        path = CONFIG_DIR / name_or_path
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    raw = _load_raw(path)
    resolved = yaml.safe_dump(raw, sort_keys=True).encode()

    import hashlib

    config_hash = hashlib.sha256(resolved).hexdigest()[:12]
    return XdrConfig(**raw, config_name=path.stem, config_hash=config_hash)
