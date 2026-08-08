"""Calibration metrics (SPEC.md §11.1). This is the defensible half of the
project: a model that ranks actions correctly can still emit meaningless
probability magnitudes, and these functions are how that gets caught instead
of assumed away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BrierDecomposition:
    """Murphy's three-term decomposition: brier = reliability - resolution + uncertainty."""

    brier: float
    reliability: float
    resolution: float
    uncertainty: float


def _bin_probabilities(y_prob: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-width bins over [0, 1], returned as an integer bin index per sample."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right-inclusive last edge so a prediction of exactly 1.0 lands in the last bin
    return np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, n_bins - 1)


def brier_decomposition(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> BrierDecomposition:
    """Count-weighted reliability/resolution over `n_bins` equal-width bins.

    Binning is an approximation to the exact (per-distinct-value) decomposition;
    with n_bins=10 over O(1e4-1e5) actions the approximation error against the
    directly-computed Brier score is negligible (`test_metrics.py` asserts the
    identity holds to a small tolerance, not exactly).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    if n == 0:
        raise ValueError("brier_decomposition requires at least one sample")

    brier = float(np.mean((y_prob - y_true) ** 2))
    obar = float(np.mean(y_true))
    uncertainty = obar * (1 - obar)

    bins = _bin_probabilities(y_prob, n_bins)
    reliability = 0.0
    resolution = 0.0
    for k in range(n_bins):
        mask = bins == k
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        p_k = float(y_prob[mask].mean())
        o_k = float(y_true[mask].mean())
        weight = n_k / n
        reliability += weight * (p_k - o_k) ** 2
        resolution += weight * (o_k - obar) ** 2

    return BrierDecomposition(
        brier=brier, reliability=reliability, resolution=resolution, uncertainty=uncertainty
    )


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Count-weighted mean absolute gap between predicted and observed rate per bin."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    bins = _bin_probabilities(y_prob, n_bins)
    ece = 0.0
    for k in range(n_bins):
        mask = bins == k
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        p_k = float(y_prob[mask].mean())
        o_k = float(y_true[mask].mean())
        ece += (n_k / n) * abs(p_k - o_k)
    return ece


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """One row per non-empty bin: bounds, mean predicted, mean observed, count.

    Marker area in the frontend reliability diagram is driven by `count`
    (SPEC.md §13: "equal-sized markers over unequal bins is the standard way
    this chart misleads").
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = _bin_probabilities(y_prob, n_bins)

    rows = []
    for k in range(n_bins):
        mask = bins == k
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        rows.append(
            {
                "bin_lower": float(edges[k]),
                "bin_upper": float(edges[k + 1]),
                "mean_predicted": float(y_prob[mask].mean()),
                "mean_observed": float(y_true[mask].mean()),
                "count": n_k,
            }
        )
    return pd.DataFrame(rows)


def match_level_bootstrap(
    df: pd.DataFrame,
    metric_fn,
    match_col: str = "game_id",
    n_draws: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap over match IDs, not actions (SPEC.md §11.1).

    Actions within a match are correlated (they share players, a scoreline,
    momentum); resampling actions directly understates the interval by
    treating correlated rows as independent draws. Resampling whole matches
    with replacement preserves that correlation structure.

    `metric_fn(df) -> float` is evaluated once on the true data for the point
    estimate and once per bootstrap draw. The resampled frame passed to
    `metric_fn` keeps the *original* row index (a match sampled twice appears
    with its index duplicated rather than renumbered), so callers that look
    predictions up by that index -- e.g. joining back to a Series computed
    once, outside the loop -- keep working under resampling.
    """
    matches = df[match_col].unique()
    n_matches = len(matches)
    rng = np.random.default_rng(seed)

    point_estimate = metric_fn(df)

    draws = np.empty(n_draws)
    grouped = {m: g for m, g in df.groupby(match_col)}
    for i in range(n_draws):
        sampled_matches = rng.choice(matches, size=n_matches, replace=True)
        resampled = pd.concat([grouped[m] for m in sampled_matches])
        draws[i] = metric_fn(resampled)

    alpha = (1 - ci) / 2
    lower = float(np.quantile(draws, alpha))
    upper = float(np.quantile(draws, 1 - alpha))
    return point_estimate, lower, upper
