"""Isotonic/Platt calibrator fitting (SPEC.md §5, §8.1)."""

import numpy as np

from xdr.models.calibrate import fit_calibrator


def test_isotonic_calibrator_improves_a_miscalibrated_model():
    rng = np.random.default_rng(0)
    n = 5000
    true_prob = rng.uniform(0, 1, n)
    y_true = (rng.uniform(0, 1, n) < true_prob).astype(int)
    # Systematically overconfident raw predictions.
    raw_pred = np.clip(true_prob * 1.5, 0, 1)

    calibrator = fit_calibrator(y_true, raw_pred, method="isotonic")
    calibrated = calibrator.predict(raw_pred)

    brier_raw = np.mean((raw_pred - y_true) ** 2)
    brier_calibrated = np.mean((calibrated - y_true) ** 2)
    assert brier_calibrated < brier_raw


def test_platt_calibrator_is_monotonic_in_raw_score():
    rng = np.random.default_rng(1)
    n = 2000
    raw_pred = rng.uniform(0, 1, n)
    y_true = (rng.uniform(0, 1, n) < raw_pred).astype(int)

    calibrator = fit_calibrator(y_true, raw_pred, method="platt")
    order = np.argsort(raw_pred)
    calibrated_sorted = calibrator.predict(raw_pred[order])
    assert np.all(np.diff(calibrated_sorted) >= -1e-9)
