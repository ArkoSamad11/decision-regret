"""SPEC.md §15: Brier identity holds; ECE monotone under inflation."""

import numpy as np
import pandas as pd
import pytest

from xdr.evaluation.metrics import (
    brier_decomposition,
    expected_calibration_error,
    match_level_bootstrap,
    reliability_curve,
)


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(42)
    n = 5000
    y_prob = rng.uniform(0, 1, n)
    y_true = (rng.uniform(0, 1, n) < y_prob).astype(float)  # well-calibrated by construction
    return y_true, y_prob


def test_brier_identity_holds(synthetic):
    y_true, y_prob = synthetic
    d = brier_decomposition(y_true, y_prob, n_bins=10)
    reconstructed = d.reliability - d.resolution + d.uncertainty
    assert d.brier == pytest.approx(reconstructed, abs=1e-3)


def test_perfectly_calibrated_predictions_have_near_zero_reliability():
    # Predictions that exactly equal the empirical rate per bin should have
    # reliability close to zero regardless of resolution.
    rng = np.random.default_rng(0)
    n = 20000
    y_prob = rng.choice([0.1, 0.3, 0.5, 0.7, 0.9], size=n)
    y_true = (rng.uniform(0, 1, n) < y_prob).astype(float)
    d = brier_decomposition(y_true, y_prob, n_bins=10)
    assert d.reliability < 0.001


def test_ece_monotone_under_inflation(synthetic):
    y_true, y_prob = synthetic
    ece_calibrated = expected_calibration_error(y_true, y_prob, n_bins=10)

    # Deliberately miscalibrate by pushing every prediction toward 1.0 --
    # this should only ever make the calibration gap worse, never better.
    inflated = np.clip(y_prob + 0.3, 0, 1)
    ece_inflated = expected_calibration_error(y_true, inflated, n_bins=10)

    assert ece_inflated > ece_calibrated


def test_reliability_curve_counts_sum_to_n(synthetic):
    y_true, y_prob = synthetic
    curve = reliability_curve(y_true, y_prob, n_bins=10)
    assert curve["count"].sum() == len(y_true)
    assert (curve["mean_predicted"] >= curve["bin_lower"]).all()
    assert (curve["mean_predicted"] <= curve["bin_upper"] + 1e-9).all()


def test_match_level_bootstrap_widens_interval_vs_action_level():
    # Build 20 "matches" of 50 highly-correlated actions each (same outcome
    # within a match) -- the true match-level uncertainty is about 20 draws,
    # not 1000. A match-level bootstrap CI should be wide; treating actions as
    # independent would understate it substantially.
    rng = np.random.default_rng(1)
    rows = []
    for game_id in range(20):
        match_rate = rng.uniform(0.1, 0.9)
        for _ in range(50):
            rows.append({"game_id": game_id, "y_true": rng.uniform() < match_rate, "y_prob": 0.5})
    df = pd.DataFrame(rows)
    df["y_true"] = df["y_true"].astype(float)

    def mean_outcome(d: pd.DataFrame) -> float:
        return float(d["y_true"].mean())

    point, lower, upper = match_level_bootstrap(df, mean_outcome, match_col="game_id", n_draws=500, seed=2)
    assert lower <= point <= upper
    assert (upper - lower) > 0.05  # wide: match-level resampling, not action-level
