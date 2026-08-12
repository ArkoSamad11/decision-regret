"""Isotonic calibration (SPEC.md §8.1, §5): fit on the calibration split held
out from both training and test, write a reliability report.

Isotonic regression makes no sigmoid-shape assumption, at the cost of needing
more data than Platt scaling. SPEC.md §5's fallback ("if the calibration
split is under ~2000 rows, use Platt instead") is checked and logged here
rather than assumed away.

CLI: python -m xdr.models.calibrate --config base.yaml
"""

from __future__ import annotations

import argparse
import json

import joblib
import lightgbm as lgb
from sklearn.calibration import _SigmoidCalibration
from sklearn.isotonic import IsotonicRegression

from xdr.config import REPO_ROOT, XdrConfig, load_config
from xdr.models.train import LABEL_COLUMNS, load_features, prepare_matrix

MIN_ROWS_FOR_ISOTONIC = 2000


def fit_calibrator(y_true, y_pred_raw, method: str):
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(y_pred_raw, y_true)
        return model
    model = _SigmoidCalibration()
    model.fit(y_pred_raw, y_true)
    return model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="base.yaml")
    args = parser.parse_args(argv)

    config: XdrConfig = load_config(args.config)
    artifacts_dir = REPO_ROOT / config.paths.artifacts
    run_meta = json.loads((artifacts_dir / "run.json").read_text(encoding="utf-8"))

    df = load_features(config)
    cols = run_meta["feature_columns"]
    X, _ = prepare_matrix(df, cols)

    calib_mask = df["game_id"].isin(run_meta["splits"]["calibration"])
    n_calib = int(calib_mask.sum())

    method = config.calibration.method
    if method == "isotonic" and n_calib < MIN_ROWS_FOR_ISOTONIC:
        print(
            f"Calibration split has {n_calib} rows (< {MIN_ROWS_FOR_ISOTONIC}); "
            "falling back to Platt scaling per SPEC.md §5."
        )
        method = "platt"

    calibrators = {}
    for label in LABEL_COLUMNS:
        booster = lgb.Booster(model_file=str(artifacts_dir / (label.replace("label_", "") + "_lgbm.txt")))
        raw_pred = booster.predict(X[calib_mask])
        y_true = df.loc[calib_mask, label].astype(int).values
        calibrators[label] = fit_calibrator(y_true, raw_pred, method)

    for label, calibrator in calibrators.items():
        name = label.replace("label_", "") + "_calibrator.joblib"
        joblib.dump(calibrator, artifacts_dir / name)

    run_meta["calibration_method"] = method
    run_meta["calibration_rows"] = n_calib
    (artifacts_dir / "run.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print(f"Fit {method} calibration on {n_calib} calibration-split rows for {list(calibrators)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
