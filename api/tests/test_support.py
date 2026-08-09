"""SPEC.md §9.3: the gate MUST fire on synthetic, clearly-OOD points. If it
doesn't, the mechanism isn't working and every downstream regret number is
unsupported extrapolation dressed up as a measurement.
"""

import numpy as np
import pandas as pd

from xdr.models.support import fit_support_index, numeric_feature_columns, support_score


def _cluster(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "start_x": rng.normal(50, 5, n),
            "start_y": rng.normal(34, 5, n),
            "dist_to_goal": rng.normal(30, 5, n),
        }
    )


def test_ood_point_falls_below_the_support_floor():
    X_train = _cluster(500, seed=0)
    X_val = _cluster(150, seed=1)
    support = fit_support_index(X_train, X_val, k=20)

    # A shot from inside the *defending* penalty area / a pass to a
    # coordinate far outside anything observed: wildly outside the training
    # cluster in every dimension at once.
    ood = pd.DataFrame({"start_x": [500.0], "start_y": [-200.0], "dist_to_goal": [900.0]})
    score = support_score(support, ood)[0]

    assert score < 0.35  # SPEC.md §9.2 default min_support floor
    assert score == 0.0  # farther than every validation reference point


def test_in_distribution_point_clears_the_floor():
    X_train = _cluster(500, seed=0)
    X_val = _cluster(150, seed=1)
    support = fit_support_index(X_train, X_val, k=20)

    typical = pd.DataFrame({"start_x": [50.0], "start_y": [34.0], "dist_to_goal": [30.0]})
    score = support_score(support, typical)[0]

    assert score >= 0.35


def test_numeric_feature_columns_excludes_categoricals():
    columns = ["start_x", "actiontype_a0", "endpolar_a0_dist", "bodypart_a0"]
    categorical = ["actiontype_a0", "bodypart_a0"]
    assert numeric_feature_columns(columns, categorical) == ["start_x", "endpolar_a0_dist"]
