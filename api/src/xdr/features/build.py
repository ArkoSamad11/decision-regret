"""Orchestrates feature + label construction across every ingested match of a
competition and writes one feature Parquet file (SPEC.md §6.5, `features/`).

CLI: python -m xdr.features.build --config base.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from xdr.config import Competition, XdrConfig, load_config
from xdr.features.actions import build_action_features
from xdr.features.labels import build_labels

REPO_ROOT = Path(__file__).resolve().parents[4]


def load_competition_actions(comp: Competition, config: XdrConfig) -> pd.DataFrame:
    parquet_dir = REPO_ROOT / config.paths.parquet / f"{comp.competition_id}_{comp.season_id}"
    action_files = sorted(parquet_dir.glob("*_actions.parquet"))
    if not action_files:
        raise FileNotFoundError(
            f"No ingested actions found under {parquet_dir}. Run `make ingest` first."
        )
    frames = [pd.read_parquet(f) for f in action_files]
    actions = pd.concat(frames, ignore_index=False)
    actions["competition_id"] = comp.competition_id
    actions["season_id"] = comp.season_id
    return actions


def build_features_for_competition(comp: Competition, config: XdrConfig) -> pd.DataFrame:
    actions = load_competition_actions(comp, config)
    features = build_action_features(actions, action_window=config.features.action_window)
    labels = build_labels(actions, horizon=config.labels.horizon)

    merged = features.merge(labels, on=["game_id", "action_id"], how="left", validate="one_to_one")
    merged["competition_id"] = comp.competition_id
    merged["season_id"] = comp.season_id
    return merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="base.yaml")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    competitions = config.data.competitions.train + config.data.competitions.test
    out_dir = REPO_ROOT / config.paths.parquet / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    for comp in competitions:
        print(f"Building features for {comp.name} ({comp.competition_id}/{comp.season_id})...")
        merged = build_features_for_competition(comp, config)
        positive_rate = merged["label_scores"].mean()
        out_path = out_dir / f"{comp.competition_id}_{comp.season_id}.parquet"
        merged.to_parquet(out_path, index=False)
        print(
            f"  {len(merged)} rows, {merged.shape[1]} columns, "
            f"label_scores positive rate {positive_rate:.2%} -> {out_path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
