"""k-NN support scoring (SPEC.md §9): the honest-extrapolation mechanism that
separates counterfactual scoring from confidently scoring fantasy.

SPEC.md §9.2's reference space is "action features standardized, plus the
frame embedding." The frame embedding half doesn't exist yet -- the DeepSets
encoder (M6) is deferred -- so this gate runs on the standardized
action-feature space alone. That is a real gap (a candidate that is bizarre
only in its player configuration, not its action geometry, won't be caught),
not a silent one: it is called out here and in docs/DECISIONS.md, and
`fit_support_index` takes whatever numeric columns it is given, so plugging
in a frame embedding later is additive, not a rewrite.

CLI: python -m xdr.models.support --config base.yaml
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from xdr.config import REPO_ROOT, XdrConfig, load_config
from xdr.models.train import load_features, prepare_matrix


@dataclass
class SupportIndex:
    scaler: StandardScaler
    index: NearestNeighbors
    k: int
    columns: list[str]
    # Sorted validation-set self-distances (mean distance to k nearest
    # training neighbours), i.e. the reference distribution a query's own
    # mean distance is percentile-ranked against (SPEC.md §9.2 step 2).
    reference_distances: np.ndarray


def save_support_index(support: SupportIndex, path: Path) -> None:
    """Persists fields as a plain dict, not the `SupportIndex` instance
    itself. Run as `python -m xdr.models.support`, this module executes as
    `__main__`, so a `SupportIndex` instance built during that run pickles
    with `__module__ == "__main__"` and fails to unpickle in every other
    entrypoint (store.py, options.py, tests) that imports the class properly
    as `xdr.models.support.SupportIndex`. A plain dict of primitives and
    third-party sklearn objects (which are always importable from their real
    installed location, regardless of how this script was invoked) sidesteps
    the problem instead of fighting pickle's module-identity check.
    """
    joblib.dump(
        {
            "scaler": support.scaler,
            "index": support.index,
            "k": support.k,
            "columns": support.columns,
            "reference_distances": support.reference_distances,
        },
        path,
    )


def load_support_index(path: Path) -> SupportIndex:
    return SupportIndex(**joblib.load(path))


def numeric_feature_columns(columns: list[str], categorical_columns: list[str]) -> list[str]:
    """SPEC.md §9.2's distance space must be pure-numeric; socceraction's raw
    actiontype/bodypart/result columns are native LightGBM categoricals
    (strings), excluded here in favour of their already-numeric `*_onehot`
    counterparts, which carry the same information.
    """
    excluded = set(categorical_columns)
    return [c for c in columns if c not in excluded]


def fit_support_index(X_train: pd.DataFrame, X_val: pd.DataFrame, k: int = 20) -> SupportIndex:
    """Fit on the training set; the reference distribution comes from the
    validation set, not training-set self-distances, which are biased low
    (SPEC.md §9.2 step 2: a training point's nearest neighbour is often
    itself at distance ~0).
    """
    columns = list(X_train.columns)
    scaler = StandardScaler()
    # Onehot columns may be bool dtype alongside float columns; a mixed-dtype
    # DataFrame.values would come back as an object array, which neither
    # StandardScaler nor NearestNeighbors accept as numeric input.
    train_scaled = scaler.fit_transform(X_train.astype(np.float64).values)
    index = NearestNeighbors(n_neighbors=k).fit(train_scaled)

    val_scaled = scaler.transform(X_val[columns].astype(np.float64).values)
    val_distances, _ = index.kneighbors(val_scaled)
    reference = np.sort(val_distances.mean(axis=1))

    return SupportIndex(scaler=scaler, index=index, k=k, columns=columns, reference_distances=reference)


def support_score(support: SupportIndex, X_query: pd.DataFrame) -> np.ndarray:
    """1.0 = as densely covered as the median validation point or more;
    0.0 = farther from training data than any validation point observed
    (SPEC.md §9.2 step 3: `support = 1 - percentile_rank`)."""
    scaled = support.scaler.transform(X_query[support.columns].astype(np.float64).values)
    distances, _ = support.index.kneighbors(scaled)
    mean_dist = distances.mean(axis=1)
    percentile_rank = np.searchsorted(support.reference_distances, mean_dist) / len(
        support.reference_distances
    )
    return 1.0 - np.clip(percentile_rank, 0.0, 1.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="base.yaml")
    args = parser.parse_args(argv)

    config: XdrConfig = load_config(args.config)
    artifacts_dir = REPO_ROOT / config.paths.artifacts
    run_meta = json.loads((artifacts_dir / "run.json").read_text(encoding="utf-8"))

    df = load_features(config)
    numeric_cols = numeric_feature_columns(run_meta["feature_columns"], run_meta["categorical_columns"])
    X, _ = prepare_matrix(df, numeric_cols)

    train_mask = df["game_id"].isin(run_meta["splits"]["train"])
    val_mask = df["game_id"].isin(run_meta["splits"]["validation"])

    support = fit_support_index(X[train_mask], X[val_mask], k=config.support.k_neighbors)
    save_support_index(support, artifacts_dir / "support_index.joblib")

    # Sanity diagnostic only, not the SPEC.md §9.3 acceptance check. Support
    # is `1 - percentile_rank` against the validation set's OWN distance
    # distribution, so for real test-split actions -- drawn from roughly the
    # same distribution as validation -- the percentile rank is close to
    # uniform on [0, 1] by construction, and the gated fraction here should
    # land close to `min_support` itself (~35%), not near 0%. A number far
    # above that (most of the test split gated) would suggest the test split
    # is itself unusually different from train/validation; a number far
    # below it would suggest a bug in the percentile-rank computation. The
    # SPEC's real acceptance target -- 15-50% of *enumerated counterfactual
    # candidates* gated -- is reported once M8 wiring produces candidates.
    test_mask = df["game_id"].isin(run_meta["splits"]["test"])
    test_scores = support_score(support, X[test_mask])
    gated_rate = float((test_scores < config.support.min_support).mean())
    print(
        f"Support index fit: k={support.k}, {len(support.columns)} numeric columns, "
        f"{len(support.reference_distances)} validation reference points."
    )
    print(
        f"Sanity check -- real test-split actions gated below floor: {gated_rate:.2%} "
        f"(expect roughly min_support={config.support.min_support:.0%}, by construction)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
