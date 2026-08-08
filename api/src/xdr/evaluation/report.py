"""In-competition evaluation (SPEC.md §11.2 item 1): Brier decomposition, ECE,
and reliability curve on the held-out test matches, raw and calibrated side by
side, with a match-level bootstrap CI on the headline Brier number.

CLI: python -m xdr.evaluation.report --config base.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd

from xdr.config import XdrConfig, load_config
from xdr.evaluation.metrics import (
    brier_decomposition,
    expected_calibration_error,
    match_level_bootstrap,
    reliability_curve,
)
from xdr.models.train import LABEL_COLUMNS, load_features, prepare_matrix

REPO_ROOT = Path(__file__).resolve().parents[4]


def evaluate_split(
    df: pd.DataFrame, X: pd.DataFrame, mask: pd.Series, label: str, booster: lgb.Booster, calibrator
) -> dict:
    y_true = df.loc[mask, label].astype(int).values
    raw_pred = booster.predict(X[mask])
    calibrated_pred = calibrator.predict(raw_pred)

    eval_df = pd.DataFrame({"game_id": df.loc[mask, "game_id"].values, "y_true": y_true})

    def brier_of(d: pd.DataFrame, preds: pd.Series) -> float:
        return float(((preds.loc[d.index] - d["y_true"]) ** 2).mean())

    raw_series = pd.Series(raw_pred, index=eval_df.index)
    calibrated_series = pd.Series(calibrated_pred, index=eval_df.index)

    raw_decomp = brier_decomposition(y_true, raw_pred)
    calibrated_decomp = brier_decomposition(y_true, calibrated_pred)

    point, lower, upper = match_level_bootstrap(
        eval_df, lambda d: brier_of(d, calibrated_series), match_col="game_id", n_draws=2000, seed=0
    )

    return {
        "n": int(mask.sum()),
        "positive_rate": float(y_true.mean()),
        "raw": {
            "brier": raw_decomp.brier,
            "reliability": raw_decomp.reliability,
            "resolution": raw_decomp.resolution,
            "uncertainty": raw_decomp.uncertainty,
            "ece": expected_calibration_error(y_true, raw_pred),
        },
        "calibrated": {
            "brier": calibrated_decomp.brier,
            "reliability": calibrated_decomp.reliability,
            "resolution": calibrated_decomp.resolution,
            "uncertainty": calibrated_decomp.uncertainty,
            "ece": expected_calibration_error(y_true, calibrated_pred),
            "brier_ci": {"point": point, "lower": lower, "upper": upper},
        },
        "reliability_curve": reliability_curve(y_true, calibrated_pred).to_dict(orient="records"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="base.yaml")
    parser.add_argument("--split", default="test", choices=["validation", "calibration", "test"])
    args = parser.parse_args(argv)

    config: XdrConfig = load_config(args.config)
    artifacts_dir = REPO_ROOT / config.paths.artifacts
    run_meta = json.loads((artifacts_dir / "run.json").read_text(encoding="utf-8"))

    df = load_features(config)
    X, _ = prepare_matrix(df, run_meta["feature_columns"])
    mask = df["game_id"].isin(run_meta["splits"][args.split])

    report = {"split": args.split, "run_id": run_meta["run_id"]}
    for label in LABEL_COLUMNS:
        stem = label.replace("label_", "")
        booster = lgb.Booster(model_file=str(artifacts_dir / f"{stem}_lgbm.txt"))
        calibrator = joblib.load(artifacts_dir / f"{stem}_calibrator.joblib")
        report[label] = evaluate_split(df, X, mask, label, booster, calibrator)

    out_dir = REPO_ROOT / config.paths.runs / run_meta["run_id"]
    out_path = out_dir / f"report_{args.split}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (artifacts_dir / f"report_{args.split}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    for label in LABEL_COLUMNS:
        r = report[label]
        print(f"{label} [{args.split}, n={r['n']}, positive_rate={r['positive_rate']:.2%}]")
        print(f"  raw:        brier={r['raw']['brier']:.5f}  ece={r['raw']['ece']:.5f}")
        print(f"  calibrated: brier={r['calibrated']['brier']:.5f}  ece={r['calibrated']['ece']:.5f}")
        ci = r["calibrated"]["brier_ci"]
        print(f"  brier 95% CI (match-level bootstrap): [{ci['lower']:.5f}, {ci['upper']:.5f}]")

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
