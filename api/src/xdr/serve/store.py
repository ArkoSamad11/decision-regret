"""DuckDB read layer (SPEC.md §11.3) -- plus the build step that populates it.

SPEC.md's file tree doesn't assign ownership of *writing* `artifacts/xdr.duckdb`
to any single module; `build_database` lives here, next to the read functions,
rather than as a new top-level file not in the spec (see docs/DECISIONS.md).

This pass has no counterfactual layer (M8) or support gate (M7), so every
`moments` row has `best = NULL`, `regret = NULL`, `options = []`,
`unscored_count = 0` by construction -- not omitted, not defaulted to zero
silently, but genuinely absent because nothing has computed them yet. The API
and frontend both render that as "gap unsupported", never as 0.0 (SPEC.md
§13).

`GET /matches` is specified as "ordered by total regret" (SPEC.md §12); with
no regret computed yet, matches are ordered by mean predicted action value
instead -- the closest honest analogue -- and `list_moments`/`get_moment`
rank the same way. `get_moment("latest")` resolves to the highest-value
scored moment, standing in for "highest regret" until M8.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import pandas as pd

from xdr.config import XdrConfig, load_config
from xdr.models.train import LABEL_COLUMNS, load_features, prepare_matrix

REPO_ROOT = Path(__file__).resolve().parents[4]


def _score_all(df: pd.DataFrame, X: pd.DataFrame, artifacts_dir: Path) -> pd.DataFrame:
    scored = df.copy()
    for label in LABEL_COLUMNS:
        stem = label.replace("label_", "")
        booster = lgb.Booster(model_file=str(artifacts_dir / f"{stem}_lgbm.txt"))
        calibrator = joblib.load(artifacts_dir / f"{stem}_calibrator.joblib")
        raw = booster.predict(X)
        scored[f"p_{stem}_raw"] = raw
        scored[f"p_{stem}"] = calibrator.predict(raw)
    scored["value"] = scored["p_scores"] - scored["p_concedes"]
    return scored


def build_database(config: XdrConfig) -> Path:
    artifacts_dir = REPO_ROOT / config.paths.artifacts
    run_meta = json.loads((artifacts_dir / "run.json").read_text(encoding="utf-8"))

    df = load_features(config)
    X, _ = prepare_matrix(df, run_meta["feature_columns"])
    scored = _score_all(df, X, artifacts_dir)

    competitions = config.data.competitions.train + config.data.competitions.test
    games = pd.concat(
        [
            pd.read_parquet(
                REPO_ROOT / config.paths.raw / f"{c.competition_id}_{c.season_id}" / "games.parquet"
            ).assign(competition_name=c.name)
            for c in competitions
        ],
        ignore_index=True,
    )

    # socceraction's game schema carries team ids, not names; recover names
    # from the same id -> name map ingest.py already derived from events.
    team_names = (
        scored.dropna(subset=["display_team_name"])
        .drop_duplicates("team_id")
        .set_index("team_id")["display_team_name"]
    )
    games["home_team"] = games["home_team_id"].map(team_names)
    games["away_team"] = games["away_team_id"].map(team_names)
    games = games.rename(columns={"game_id": "match_id", "game_date": "match_date"})[
        ["match_id", "match_date", "home_team", "away_team", "competition_name"]
    ]
    games["match_date"] = games["match_date"].astype(str)

    match_agg = (
        scored.groupby("game_id")
        .agg(action_count=("action_id", "count"), mean_value=("value", "mean"))
        .reset_index()
        .rename(columns={"game_id": "match_id"})
    )
    matches = games.merge(match_agg, on="match_id", how="inner").sort_values(
        "mean_value", ascending=False
    )

    moments = pd.DataFrame(
        {
            "moment_id": scored["game_id"].astype(str) + "_" + scored["action_id"].astype(str),
            "match_id": scored["game_id"].astype(str),
            "minute": scored["display_minute"],
            "second": scored["display_second"],
            "team": scored["display_team_name"],
            "player_name": scored["display_player_name"].fillna("(unknown)"),
            "ball_x": scored["display_start_x"],
            "ball_y": scored["display_start_y"],
            "chosen_type_name": scored["display_type_name"],
            "chosen_start_x": scored["display_start_x"],
            "chosen_start_y": scored["display_start_y"],
            "chosen_end_x": scored["display_end_x"],
            "chosen_end_y": scored["display_end_y"],
            "chosen_p_scores": scored["p_scores"],
            "chosen_p_concedes": scored["p_concedes"],
            "chosen_value": scored["value"],
            "value": scored["value"],
        }
    )

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    db_path = artifacts_dir / "xdr.duckdb"
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE matches AS SELECT * FROM matches")
    con.execute("CREATE TABLE moments AS SELECT * FROM moments")
    con.close()
    return db_path


def _row_to_moment(row: dict) -> dict:
    return {
        "moment_id": row["moment_id"],
        "match_id": row["match_id"],
        "minute": int(row["minute"]),
        "second": int(row["second"]),
        "team": row["team"],
        "player_name": row["player_name"],
        "ball_x": row["ball_x"],
        "ball_y": row["ball_y"],
        "chosen": {
            "type_name": row["chosen_type_name"],
            "start_x": row["chosen_start_x"],
            "start_y": row["chosen_start_y"],
            "end_x": row["chosen_end_x"],
            "end_y": row["chosen_end_y"],
            "p_scores": row["chosen_p_scores"],
            "p_concedes": row["chosen_p_concedes"],
            "value": row["chosen_value"],
        },
        "best": None,
        "regret": None,
        "options": [],
        "unscored_count": 0,
    }


def list_matches(con: duckdb.DuckDBPyConnection) -> list[dict]:
    rows = con.execute(
        "SELECT match_id, competition_name, home_team, away_team, match_date, "
        "action_count, mean_value FROM matches ORDER BY mean_value DESC"
    ).fetchall()
    cols = ["match_id", "competition_name", "home_team", "away_team", "match_date", "action_count", "mean_value"]
    return [
        {**dict(zip(cols, r)), "match_id": str(r[0])} for r in rows
    ]


def list_moments(
    con: duckdb.DuckDBPyConnection, match_id: str, min_value: float | None = None, limit: int = 100
) -> list[dict]:
    """`limit` defaults to 100: a match has ~2000 actions, and with no
    counterfactual layer yet to naturally narrow "moments" down to ones with
    an interesting declined alternative, every scored action would otherwise
    qualify. Ranking by value and truncating is a stand-in for that until M8;
    it is not a claim that only the top 100 actions "count".
    """
    query = "SELECT * FROM moments WHERE match_id = ?"
    params: list = [match_id]
    if min_value is not None:
        query += " AND value >= ?"
        params.append(min_value)
    query += " ORDER BY value DESC LIMIT ?"
    params.append(limit)
    result = con.execute(query, params)
    cols = [d[0] for d in result.description]
    return [_row_to_moment(dict(zip(cols, r))) for r in result.fetchall()]


def get_moment(con: duckdb.DuckDBPyConnection, moment_id: str) -> dict | None:
    if moment_id == "latest":
        row = con.execute("SELECT * FROM moments ORDER BY value DESC LIMIT 1").fetchone()
    else:
        row = con.execute("SELECT * FROM moments WHERE moment_id = ?", [moment_id]).fetchone()
    if row is None:
        return None
    cols = [d[0] for d in con.description]
    return _row_to_moment(dict(zip(cols, row)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="base.yaml")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    path = build_database(config)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
