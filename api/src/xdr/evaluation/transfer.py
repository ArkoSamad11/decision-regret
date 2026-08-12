"""Cross-competition transfer study (SPEC.md §11.2 item 2). SPEC.md §16 calls
M5 "the resume line": whether calibration measured in-competition survives
being evaluated on a competition the model never trained on is the project's
actual thesis, not the counterfactual layer.

No retraining happens here. `config.data.competitions.train` is the same
Euro 2024 list, with the same seed, in both `base.yaml` and this transfer
config -- the match-level split (and therefore the trained boosters and
source calibrators already on disk in `artifacts/`) is identical either way.
This module only: (1) applies that already-trained, already-calibrated model
to the target competition, (2) fits a *new* calibrator on a held-out 20%
slice of the target, disjoint from the slice used for evaluation, and
(3) reports Brier/ECE before and after, with the recovered fraction.

CLI: python -m xdr.evaluation.transfer --config transfer_euro24_to_weuro25.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from xdr.config import REPO_ROOT, XdrConfig, load_config
from xdr.evaluation.metrics import (
    brier_decomposition,
    expected_calibration_error,
    match_level_bootstrap,
    reliability_curve,
)
from xdr.models.calibrate import MIN_ROWS_FOR_ISOTONIC, fit_calibrator
from xdr.models.train import LABEL_COLUMNS, prepare_matrix


def load_target_features(config: XdrConfig) -> pd.DataFrame:
    feature_dir = REPO_ROOT / config.paths.parquet / "features"
    frames = []
    for comp in config.data.competitions.test:
        path = feature_dir / f"{comp.competition_id}_{comp.season_id}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing target feature file {path}. Run `make features` first.")
        frames.append(pd.read_parquet(path))
    if not frames:
        raise ValueError("No target competitions configured under data.competitions.test.")
    return pd.concat(frames, ignore_index=True)


def target_split(game_ids: np.ndarray, recalibration_fraction: float, seed: int) -> dict[str, list[int]]:
    """Match-level split of the TARGET competition: a recalibration slice
    (held out, used only to fit the new calibrator) and an evaluation slice,
    disjoint from each other (SPEC.md §11.2: "held-out 20% of the target")."""
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(game_ids)
    n_recal = max(1, int(round(len(shuffled) * recalibration_fraction)))
    return {"recalibration": shuffled[:n_recal].tolist(), "evaluation": shuffled[n_recal:].tolist()}


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, game_ids: np.ndarray) -> dict:
    decomp = brier_decomposition(y_true, y_pred)
    eval_df = pd.DataFrame({"game_id": game_ids, "y_true": y_true})
    preds = pd.Series(y_pred, index=eval_df.index)

    def brier_of(d: pd.DataFrame) -> float:
        return float(((preds.loc[d.index] - d["y_true"]) ** 2).mean())

    point, lower, upper = match_level_bootstrap(eval_df, brier_of, match_col="game_id", n_draws=2000, seed=0)
    return {
        "n": int(len(y_true)),
        "brier": decomp.brier,
        "reliability": decomp.reliability,
        "resolution": decomp.resolution,
        "uncertainty": decomp.uncertainty,
        "ece": expected_calibration_error(y_true, y_pred),
        "brier_ci": {"point": point, "lower": lower, "upper": upper},
        "reliability_curve": reliability_curve(y_true, y_pred).to_dict(orient="records"),
    }


def run_transfer_for_label(
    label: str,
    target_df: pd.DataFrame,
    X_target: pd.DataFrame,
    run_meta: dict,
    config: XdrConfig,
    artifacts_dir: Path,
) -> dict:
    stem = label.replace("label_", "")
    booster = lgb.Booster(model_file=str(artifacts_dir / f"{stem}_lgbm.txt"))
    source_calibrator = joblib.load(artifacts_dir / f"{stem}_calibrator.joblib")
    source_full = json.loads((artifacts_dir / "report_test.json").read_text(encoding="utf-8"))[label]
    source_report = {"n": source_full["n"], **source_full["calibrated"]}

    game_ids = target_df["game_id"].unique()
    splits = target_split(game_ids, config.evaluation.recalibration_fraction, config.seed)
    recal_mask = target_df["game_id"].isin(splits["recalibration"]).values
    eval_mask = target_df["game_id"].isin(splits["evaluation"]).values

    raw_pred = booster.predict(X_target)
    y_true = target_df[label].astype(int).values

    before_pred = source_calibrator.predict(raw_pred[eval_mask])
    before = _evaluate(y_true[eval_mask], before_pred, target_df.loc[eval_mask, "game_id"].values)

    method = config.calibration.method
    n_recal = int(recal_mask.sum())
    if method == "isotonic" and n_recal < MIN_ROWS_FOR_ISOTONIC:
        print(f"  [{label}] target recalibration split has {n_recal} rows (< {MIN_ROWS_FOR_ISOTONIC}); using Platt.")
        method = "platt"
    target_calibrator = fit_calibrator(y_true[recal_mask], raw_pred[recal_mask], method)

    after_pred = target_calibrator.predict(raw_pred[eval_mask])
    after = _evaluate(y_true[eval_mask], after_pred, target_df.loc[eval_mask, "game_id"].values)

    ece_gap = before["ece"] - source_report["ece"]
    recovered = (before["ece"] - after["ece"]) / ece_gap if ece_gap > 0 else None

    return {
        "source": {
            "competitions": [c.name for c in config.data.competitions.train],
            **source_report,
        },
        "target_before": {
            "competitions": [c.name for c in config.data.competitions.test],
            **before,
        },
        "target_after": {
            "competitions": [c.name for c in config.data.competitions.test],
            "calibration_method": method,
            "n_recalibration_rows": n_recal,
            **after,
        },
        "ece_degradation_ratio": (before["ece"] / source_report["ece"]) if source_report["ece"] else None,
        "ece_recovered_fraction": recovered,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="transfer_euro24_to_weuro25.yaml")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    artifacts_dir = REPO_ROOT / config.paths.artifacts
    run_meta = json.loads((artifacts_dir / "run.json").read_text(encoding="utf-8"))

    target_df = load_target_features(config)
    X_target, _ = prepare_matrix(target_df, run_meta["feature_columns"])

    report = {"config_name": config.config_name, "run_id": run_meta["run_id"]}
    for label in LABEL_COLUMNS:
        print(f"Transfer study [{label}]...")
        result = run_transfer_for_label(label, target_df, X_target, run_meta, config, artifacts_dir)
        report[label] = result
        before, after = result["target_before"], result["target_after"]
        print(f"  source ece={result['source']['ece']:.5f}")
        print(f"  target before recal: ece={before['ece']:.5f} brier={before['brier']:.5f}")
        print(f"  target after recal:  ece={after['ece']:.5f} brier={after['brier']:.5f}")
        if result["ece_recovered_fraction"] is not None:
            print(
                f"  ECE degraded {result['ece_degradation_ratio']:.2f}x on transfer; "
                f"recalibration recovered {result['ece_recovered_fraction']:.1%} of the gap"
            )
        else:
            print("  no ECE degradation observed on transfer -- reporting as a valid finding (SPEC.md §1.3)")

    out_dir = REPO_ROOT / config.paths.runs / run_meta["run_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "transfer.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (artifacts_dir / "transfer.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'transfer.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
