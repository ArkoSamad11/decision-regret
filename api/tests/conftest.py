"""Shared fixtures: a synthetic actions table shaped like real ingest.py
output (SPADL columns + the display_* columns xdr adds), used by feature,
build, and store tests so they don't depend on real StatsBomb data or a
network call."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_synthetic_actions(n_games: int = 2, n_per_game: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    action_types = ["pass", "dribble", "shot"]
    for game_id in range(1, n_games + 1):
        team_ids = [game_id * 10, game_id * 10 + 1]
        for i in range(n_per_game):
            is_shot = i == n_per_game - 1 and game_id == 1  # guarantee one goal-bearing action
            type_name = "shot" if is_shot else rng.choice(action_types)
            result_name = "success"
            rows.append(
                {
                    "game_id": game_id,
                    "action_id": i,
                    "original_event_id": f"evt-{game_id}-{i}",
                    "period_id": 1 if i < n_per_game // 2 else 2,
                    "time_seconds": float(i * 5 % 2700),
                    "team_id": team_ids[i % 2],
                    "player_id": 100 + (i % 2),
                    "start_x": float(rng.uniform(0, 105)),
                    "start_y": float(rng.uniform(0, 68)),
                    "end_x": float(rng.uniform(0, 105)),
                    "end_y": float(rng.uniform(0, 68)),
                    "type_id": 11 if is_shot else 0,
                    "type_name": type_name,
                    "result_id": 1,
                    "result_name": result_name,
                    "bodypart_id": 0,
                    "bodypart_name": "foot",
                    "player_name": f"Player {100 + (i % 2)}",
                    "team_name": f"Team {team_ids[i % 2]}",
                    "minute": i // 6,
                    "second": (i * 5) % 60,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_actions() -> pd.DataFrame:
    return make_synthetic_actions()
