"""features/actions.py + features/build.py against a synthetic actions table
shaped like real ingest.py output (see conftest.make_synthetic_actions)."""

from __future__ import annotations

from xdr.config import Competition, load_config
from xdr.features.actions import DISPLAY_COLUMNS, build_action_features
from xdr.features.build import build_features_for_competition, load_competition_actions
from xdr.features.labels import build_labels


def test_build_action_features_shape_and_display_columns(synthetic_actions):
    features = build_action_features(synthetic_actions, action_window=3)
    assert len(features) == len(synthetic_actions)
    for col in DISPLAY_COLUMNS:
        assert f"display_{col}" in features.columns
    # Feature columns beyond ids/display exist (the VAEP block).
    non_meta_cols = [
        c
        for c in features.columns
        if c not in {"game_id", "action_id", "team_id", "period_id", "time_seconds"}
        and not c.startswith("display_")
    ]
    assert len(non_meta_cols) > 20


def test_build_action_features_history_does_not_cross_game_boundary(synthetic_actions):
    features = build_action_features(synthetic_actions, action_window=3)
    # The first action of game 2 should not see game 1's last action as its
    # "previous action" -- socceraction's gamestates groups by (game_id,
    # period_id), so its own start_x_a1 (previous action) should just repeat
    # its own start_x_a0 (fs.gamestates pads short history with the action
    # itself) rather than reflecting the tail of a different game.
    first_of_game2 = features[features.game_id == 2].iloc[0]
    game2_raw_first = synthetic_actions[synthetic_actions.game_id == 2].iloc[0]
    assert first_of_game2["start_x_a1"] == game2_raw_first["start_x"]


def test_build_labels_matches_shot_row(synthetic_actions):
    labels = build_labels(synthetic_actions, horizon=10)
    assert set(labels.columns) == {"game_id", "action_id", "label_scores", "label_concedes"}
    assert len(labels) == len(synthetic_actions)
    # The synthetic fixture plants exactly one guaranteed successful shot, as
    # the last action of game 1 -- it must label itself as a score.
    game1 = synthetic_actions[synthetic_actions.game_id == 1]
    shot_row = game1[game1.type_name == "shot"].iloc[-1]
    shot_label = labels[(labels.game_id == 1) & (labels.action_id == shot_row.action_id)].iloc[0]
    assert shot_label["label_scores"]


def test_build_features_for_competition_end_to_end(tmp_path, monkeypatch, synthetic_actions):
    comp = Competition(competition_id=999, season_id=1, name="Synthetic Cup")
    config = load_config("base.yaml")
    config = config.model_copy(update={"paths": config.paths.model_copy(update={"parquet": str(tmp_path)})})

    parquet_dir = tmp_path / "999_1"
    parquet_dir.mkdir(parents=True)
    for game_id, group in synthetic_actions.groupby("game_id"):
        group.to_parquet(parquet_dir / f"{game_id}_actions.parquet", index=False)

    loaded = load_competition_actions(comp, config)
    assert len(loaded) == len(synthetic_actions)

    merged = build_features_for_competition(comp, config)
    assert "label_scores" in merged.columns
    assert len(merged) == len(synthetic_actions)
