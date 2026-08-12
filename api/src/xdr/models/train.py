"""LightGBM baseline (SPEC.md §8.2, §8.3): action-features-only scoring and
conceding models, split by match, early stopping on a held-out validation set.

SPEC.md §16, M3: "This alone is a defensible artifact. If everything after
this fails, the project still stands." The DeepSets frame encoder (M6) is
judged against this baseline's numbers, not the other way around.

CLI: python -m xdr.models.train --config base.yaml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from xdr.config import LightGBMConfig, Split, XdrConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[4]

ID_COLUMNS = [
    "game_id",
    "action_id",
    "team_id",
    "period_id",
    "time_seconds",
    "competition_id",
    "season_id",
]
LABEL_COLUMNS = ["label_scores", "label_concedes"]


def git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


def load_features(config: XdrConfig) -> pd.DataFrame:
    """Loads every configured *train* competition's feature Parquet.

    This pass evaluates in-competition only (config.data.competitions.test is
    empty in base.yaml); a transfer config populates `test` too, and
    evaluation/transfer.py (M5, not built in this pass) is what will read it.
    """
    feature_dir = REPO_ROOT / config.paths.parquet / "features"
    frames = []
    for comp in config.data.competitions.train:
        path = feature_dir / f"{comp.competition_id}_{comp.season_id}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing feature file {path}. Run `make features` first.")
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def match_level_split(game_ids: np.ndarray, split: Split, seed: int) -> dict[str, list[int]]:
    """SPEC.md §8.3: split by match, never by action -- actions within a match
    are dependent, and an action-level split leaks."""
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(game_ids)
    n = len(shuffled)
    n_train = int(round(n * split.train))
    n_val = int(round(n * split.validation))
    n_calib = int(round(n * split.calibration))
    return {
        "train": shuffled[:n_train].tolist(),
        "validation": shuffled[n_train : n_train + n_val].tolist(),
        "calibration": shuffled[n_train + n_val : n_train + n_val + n_calib].tolist(),
        # remainder -> test, so rounding never silently drops a match
        "test": shuffled[n_train + n_val + n_calib :].tolist(),
    }


def feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = set(ID_COLUMNS) | set(LABEL_COLUMNS)
    return [c for c in df.columns if c not in exclude and not c.startswith("display_")]


def prepare_matrix(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Coerce object columns to pandas `category` dtype so LightGBM uses its
    native categorical handling (SPEC.md §5) rather than one-hot expansion.
    socceraction's category columns (actiontype, bodypart, result) are already
    `pd.Categorical` over the *full* fixed taxonomy, not just values seen in
    this split, so train/val/calibration/test never disagree on categories.
    """
    X = df[columns].copy()
    cat_cols = []
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) == "category":
            X[c] = X[c].astype("category")
            cat_cols.append(c)
    return X, cat_cols


def train_head(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cat_cols: list[str],
    cfg: LightGBMConfig,
) -> lgb.Booster:
    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "min_child_samples": cfg.min_child_samples,
        "verbosity": -1,
        "seed": 0,
        "deterministic": True,
    }
    return lgb.train(
        params,
        train_set,
        num_boost_round=cfg.n_estimators,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="base.yaml")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    df = load_features(config)

    game_ids = df["game_id"].unique()
    splits = match_level_split(game_ids, config.split, config.seed)

    cols = feature_columns(df)
    X, cat_cols = prepare_matrix(df, cols)

    masks = {name: df["game_id"].isin(ids) for name, ids in splits.items()}

    heads = {}
    for label in LABEL_COLUMNS:
        y = df[label].astype(int)
        booster = train_head(
            X[masks["train"]],
            y[masks["train"]],
            X[masks["validation"]],
            y[masks["validation"]],
            cat_cols,
            config.model.lightgbm,
        )
        heads[label] = booster
        val_logloss = booster.best_score["valid_0"]["binary_logloss"]
        print(f"{label}: best iteration {booster.best_iteration}, best val logloss {val_logloss:.5f}")

    run_id = f"{config.config_name}_{config.config_hash}_{int(time.time())}"
    run_dir = REPO_ROOT / config.paths.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = REPO_ROOT / config.paths.artifacts
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for label, booster in heads.items():
        model_name = label.replace("label_", "") + "_lgbm.txt"
        booster.save_model(str(run_dir / model_name))
        booster.save_model(str(artifacts_dir / model_name))

    run_meta = {
        "run_id": run_id,
        "config_name": config.config_name,
        "config_hash": config.config_hash,
        "git_commit": git_commit(),
        "seed": config.seed,
        "feature_columns": cols,
        "categorical_columns": cat_cols,
        "splits": splits,
        "n_rows": {name: int(mask.sum()) for name, mask in masks.items()},
        "best_iteration": {label: b.best_iteration for label, b in heads.items()},
    }
    (run_dir / "run.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    (artifacts_dir / "run.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print(f"Wrote run artifacts to {run_dir} and {artifacts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
