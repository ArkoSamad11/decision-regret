"""serve/store.py read functions (SPEC.md §11.3), against a small
hand-built DuckDB matching the exact schema `build_database` writes.

`build_database` itself (the write path, including LightGBM scoring) is
exercised by the real Euro 2024 run produced by `make reproduce` -- see
docs/DECISIONS.md. This suite covers the read functions directly rather than
reconstructing a full train/calibrate pipeline on a handful of synthetic
matches, which is too small a sample for LightGBM's native Dataset
construction to behave reliably.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from xdr.serve import store


@pytest.fixture
def db_path(tmp_path):
    matches = pd.DataFrame(
        {
            "match_id": ["1", "2"],
            "competition_name": ["Synthetic Cup", "Synthetic Cup"],
            "home_team": ["A", "C"],
            "away_team": ["B", "D"],
            "match_date": ["2026-01-01", "2026-01-02"],
            "action_count": [3, 2],
            "mean_value": [0.5, 0.9],
        }
    )
    moments = pd.DataFrame(
        {
            "moment_id": ["1_0", "1_1", "1_2", "2_0", "2_1"],
            "match_id": ["1", "1", "1", "2", "2"],
            "minute": [1, 2, 3, 1, 2],
            "second": [0, 0, 0, 0, 0],
            "team": ["A", "A", "B", "C", "D"],
            "player_name": ["p1", "p2", "p3", "p4", "p5"],
            "ball_x": [10.0, 20.0, 30.0, 40.0, 50.0],
            "ball_y": [10.0, 20.0, 30.0, 40.0, 50.0],
            "chosen_type_name": ["pass", "shot", "carry", "pass", "shot"],
            "chosen_start_x": [10.0, 20.0, 30.0, 40.0, 50.0],
            "chosen_start_y": [10.0, 20.0, 30.0, 40.0, 50.0],
            "chosen_end_x": [15.0, 25.0, 35.0, 45.0, 55.0],
            "chosen_end_y": [15.0, 25.0, 35.0, 45.0, 55.0],
            "chosen_p_scores": [0.1, 0.9, 0.2, 0.1, 0.95],
            "chosen_p_concedes": [0.01, 0.01, 0.01, 0.01, 0.01],
            "chosen_value": [0.09, 0.89, 0.19, 0.09, 0.94],
            "value": [0.09, 0.89, 0.19, 0.09, 0.94],
        }
    )

    path = tmp_path / "xdr.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE matches AS SELECT * FROM matches")
    con.execute("CREATE TABLE moments AS SELECT * FROM moments")
    con.close()
    return path


def test_list_matches_ordered_by_mean_value_desc(db_path):
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        matches = store.list_matches(con)
        assert [m["match_id"] for m in matches] == ["2", "1"]
    finally:
        con.close()


def test_list_moments_filters_and_orders_by_value(db_path):
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        moments = store.list_moments(con, "1")
        assert [m["moment_id"] for m in moments] == ["1_1", "1_2", "1_0"]
        assert moments[0]["regret"] is None
        assert moments[0]["options"] == []
        assert moments[0]["chosen"]["value"] == pytest.approx(0.89)

        filtered = store.list_moments(con, "1", min_value=0.5)
        assert [m["moment_id"] for m in filtered] == ["1_1"]

        limited = store.list_moments(con, "1", limit=1)
        assert len(limited) == 1
    finally:
        con.close()


def test_get_moment_latest_is_global_max(db_path):
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        latest = store.get_moment(con, "latest")
        assert latest["moment_id"] == "2_1"  # value 0.94, highest across both matches
    finally:
        con.close()


def test_get_moment_by_id_and_missing(db_path):
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        found = store.get_moment(con, "1_0")
        assert found["player_name"] == "p1"

        assert store.get_moment(con, "not-a-real-id") is None
    finally:
        con.close()
