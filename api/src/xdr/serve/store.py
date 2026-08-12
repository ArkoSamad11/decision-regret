"""DuckDB read layer (SPEC.md §11.3) -- plus the build step that populates it.

SPEC.md's file tree doesn't assign ownership of *writing* `artifacts/xdr.duckdb`
to any single module; `build_database` lives here, next to the read functions,
rather than as a new top-level file not in the spec (see docs/DECISIONS.md).

The counterfactual layer (M8) and support gate (M7) are now wired in, with
one deliberate scope cut: computing a full counterfactual set (enumeration +
scoring + support gating) for all ~170k ingested actions was not necessary to
ship a real dashboard, since `list_moments` only ever serves the top
`COUNTERFACTUAL_TOP_K` actions per match by value anyway. `add_counterfactuals`
computes real `best`/`regret`/`options`/`unscored_count` for exactly that
served subset; every other action keeps the honest `NULL`/`[]` this pass
shipped with before M8 landed -- still genuinely absent, not defaulted to
zero, just for a documented reason now (SPEC.md §13: never render a null
regret as `0.0000`).

`GET /matches` is specified as "ordered by total regret" (SPEC.md §12); with
regret only computed for a subset, matches are still ordered by mean
predicted action value -- the same honest analogue as before M8.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import pandas as pd

from xdr.config import REPO_ROOT, XdrConfig, load_config
from xdr.counterfactual.options import (
    compute_regret,
    enumerate_candidates,
    score_candidates,
    visible_teammate_points,
)
from xdr.models.support import load_support_index, numeric_feature_columns, support_score
from xdr.models.train import LABEL_COLUMNS, load_features, prepare_matrix

# Only these action types get a counterfactual set: "declined a pass/carry/
# shot instead" is a coherent framing for them; for a duel, foul, or
# interception it is not (SPEC.md §10 doesn't rule this out explicitly, but
# doesn't require it either -- documented as a scope decision, not an
# oversight, in docs/DECISIONS.md).
DECISION_TYPES = {"pass", "dribble", "shot"}

# Matches `list_moments`' default `limit`: every moment the dashboard serves
# under default query params gets real counterfactual data, not just a
# subset of a subset. Computing this for the full ~170k-action corpus was not
# necessary for a working dashboard and would have cost much more runtime for
# moments nothing ever displays (SPEC.md §11.2 item 4's full xDR distribution
# is future work, tracked in docs/DECISIONS.md).
COUNTERFACTUAL_TOP_K = 100


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


def _load_match_actions_frames(config: XdrConfig, competition_id: int, season_id: int, game_id: int):
    parquet_dir = REPO_ROOT / config.paths.parquet / f"{competition_id}_{season_id}"
    actions = pd.read_parquet(parquet_dir / f"{game_id}_actions.parquet")
    frames = pd.read_parquet(parquet_dir / f"{game_id}_frames.parquet")
    if not frames.empty:
        frames = frames.assign(points=frames["points"].apply(json.loads))
    actions = actions.sort_values(["period_id", "time_seconds"]).reset_index(drop=True)
    actions["period_pos"] = actions.groupby("period_id").cumcount()
    return actions, frames


def _history_rows(match_actions: pd.DataFrame, pos: int, action_window: int) -> list[dict]:
    """The `action_window` real actions preceding `pos`, clamped to the start
    of `pos`'s own period -- mirrors `socceraction.vaep.features.gamestates`'s
    `fillna(x.iloc[0])` padding exactly, so a counterfactual's history matches
    what the real feature table would have computed for the same decision
    point (SPEC.md §7.1: history never crosses a period boundary).
    """
    period_id = match_actions.at[pos, "period_id"]
    period_start = match_actions.index[match_actions["period_id"] == period_id].min()
    period_pos = match_actions.at[pos, "period_pos"]
    positions = [period_start + max(period_pos - i, 0) for i in range(1, action_window + 1)]
    return [match_actions.loc[p].to_dict() for p in positions]


def add_counterfactuals(
    scored: pd.DataFrame, X: pd.DataFrame, config: XdrConfig, run_meta: dict, artifacts_dir: Path
) -> pd.DataFrame:
    boosters = {
        "scores": lgb.Booster(model_file=str(artifacts_dir / "scores_lgbm.txt")),
        "concedes": lgb.Booster(model_file=str(artifacts_dir / "concedes_lgbm.txt")),
    }
    calibrators = {
        "scores": joblib.load(artifacts_dir / "scores_calibrator.joblib"),
        "concedes": joblib.load(artifacts_dir / "concedes_calibrator.joblib"),
    }
    support = load_support_index(artifacts_dir / "support_index.joblib")
    numeric_cols = numeric_feature_columns(run_meta["feature_columns"], run_meta["categorical_columns"])
    feature_cols = run_meta["feature_columns"]
    min_support = config.support.min_support
    action_window = config.features.action_window

    scored = scored.copy()
    best_columns = [
        "best_type_name", "best_end_x", "best_end_y", "best_p_scores", "best_p_concedes", "best_value", "regret"
    ]
    for col in best_columns:
        scored[col] = None
    scored["options_json"] = "[]"
    scored["unscored_count"] = 0
    scored["chosen_support"] = support_score(support, X[numeric_cols])

    decision_mask = scored["display_type_name"].isin(DECISION_TYPES)
    selected_idx: list = []
    for _, group in scored[decision_mask].groupby("game_id"):
        selected_idx.extend(group.nlargest(min(COUNTERFACTUAL_TOP_K, len(group)), "value").index.tolist())

    n_enumerated = 0
    n_gated = 0
    n_moments = 0

    for (competition_id, season_id, game_id), group in scored.loc[selected_idx].groupby(
        ["competition_id", "season_id", "game_id"]
    ):
        match_actions, match_frames = _load_match_actions_frames(
            config, int(competition_id), int(season_id), int(game_id)
        )
        frame_by_action = (
            match_frames.set_index("action_id")["points"] if not match_frames.empty else pd.Series(dtype=object)
        )
        pos_by_action_id = pd.Series(match_actions.index, index=match_actions["action_id"])

        for idx in group.index:
            action_id = int(scored.at[idx, "action_id"])
            pos = pos_by_action_id.get(action_id)
            if pos is None:
                continue
            period_id = match_actions.at[pos, "period_id"]
            period_max_time = float(match_actions.loc[match_actions.period_id == period_id, "time_seconds"].max())

            real_action = match_actions.loc[pos].to_dict()
            history = _history_rows(match_actions, pos, action_window)

            raw_frame = frame_by_action.get(action_id)
            visible = visible_teammate_points(raw_frame) if isinstance(raw_frame, list) else []

            candidates = enumerate_candidates(
                real_action["start_x"], real_action["start_y"], bool(real_action["attacks_right"]), visible
            )
            candidates_scored = score_candidates(
                candidates, real_action, history, period_max_time, feature_cols, numeric_cols,
                prepare_matrix, boosters, calibrators, support, min_support,
            )

            n_moments += 1
            n_enumerated += len(candidates_scored)
            n_gated += int((~candidates_scored["scored"]).sum())

            chosen_value = float(scored.at[idx, "value"])
            chosen_support = float(scored.at[idx, "chosen_support"])
            regret, best = compute_regret(chosen_value, chosen_support, candidates_scored, min_support)

            scored.at[idx, "unscored_count"] = int((~candidates_scored["scored"]).sum())
            scored.at[idx, "options_json"] = json.dumps(
                [
                    {
                        "type_name": r["kind"],
                        "end_x": float(r["end_x"]),
                        "end_y": float(r["end_y"]),
                        "value": None if pd.isna(r["value"]) else float(r["value"]),
                        "scored": bool(r["scored"]),
                    }
                    for _, r in candidates_scored.iterrows()
                ]
            )

            if regret is not None and best is not None:
                scored.at[idx, "regret"] = regret
                scored.at[idx, "best_type_name"] = best["kind"]
                scored.at[idx, "best_end_x"] = float(best["end_x"])
                scored.at[idx, "best_end_y"] = float(best["end_y"])
                scored.at[idx, "best_p_scores"] = float(best["p_scores"])
                scored.at[idx, "best_p_concedes"] = float(best["p_concedes"])
                scored.at[idx, "best_value"] = float(best["value"])

    if n_enumerated:
        print(
            f"Counterfactual layer: {n_moments} moments across {scored['game_id'].nunique()} matches, "
            f"{n_enumerated} candidates enumerated, {n_gated / n_enumerated:.1%} gated below support floor."
        )
    return scored


def build_database(config: XdrConfig) -> Path:
    artifacts_dir = REPO_ROOT / config.paths.artifacts
    run_meta = json.loads((artifacts_dir / "run.json").read_text(encoding="utf-8"))

    df = load_features(config)
    X, _ = prepare_matrix(df, run_meta["feature_columns"])
    scored = _score_all(df, X, artifacts_dir)
    scored = add_counterfactuals(scored, X, config, run_meta, artifacts_dir)

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
    # Not unused despite ruff's static analysis: DuckDB's `execute("... FROM
    # matches")` below resolves `matches` by variable name via frame
    # introspection (a "replacement scan"), not by explicit binding.
    matches = games.merge(match_agg, on="match_id", how="inner").sort_values(  # noqa: F841
        "mean_value", ascending=False
    )

    moments = pd.DataFrame(  # noqa: F841
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
            "best_type_name": scored["best_type_name"],
            "best_end_x": scored["best_end_x"],
            "best_end_y": scored["best_end_y"],
            "best_p_scores": scored["best_p_scores"],
            "best_p_concedes": scored["best_p_concedes"],
            "best_value": scored["best_value"],
            "regret": scored["regret"],
            "options_json": scored["options_json"],
            "unscored_count": scored["unscored_count"],
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
    has_best = row.get("best_type_name") is not None
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
        "best": (
            {
                "type_name": row["best_type_name"],
                "start_x": row["chosen_start_x"],
                "start_y": row["chosen_start_y"],
                "end_x": row["best_end_x"],
                "end_y": row["best_end_y"],
                "p_scores": row["best_p_scores"],
                "p_concedes": row["best_p_concedes"],
                "value": row["best_value"],
            }
            if has_best
            else None
        ),
        "regret": row.get("regret"),
        "options": json.loads(row["options_json"]) if row.get("options_json") else [],
        "unscored_count": int(row.get("unscored_count") or 0),
    }


def list_matches(con: duckdb.DuckDBPyConnection) -> list[dict]:
    rows = con.execute(
        "SELECT match_id, competition_name, home_team, away_team, match_date, "
        "action_count, mean_value FROM matches ORDER BY mean_value DESC"
    ).fetchall()
    cols = ["match_id", "competition_name", "home_team", "away_team", "match_date", "action_count", "mean_value"]
    return [
        {**dict(zip(cols, r, strict=True)), "match_id": str(r[0])} for r in rows
    ]


def list_moments(
    con: duckdb.DuckDBPyConnection, match_id: str, min_value: float | None = None, limit: int = 100
) -> list[dict]:
    """`limit` defaults to `COUNTERFACTUAL_TOP_K`: every moment served under
    default query params has a real, computed counterfactual set (see
    `add_counterfactuals`), not just a subset of what's served.
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
    return [_row_to_moment(dict(zip(cols, r, strict=True))) for r in result.fetchall()]


def get_moment(con: duckdb.DuckDBPyConnection, moment_id: str) -> dict | None:
    if moment_id == "latest":
        row = con.execute("SELECT * FROM moments ORDER BY value DESC LIMIT 1").fetchone()
    else:
        row = con.execute("SELECT * FROM moments WHERE moment_id = ?", [moment_id]).fetchone()
    if row is None:
        return None
    cols = [d[0] for d in con.description]
    return _row_to_moment(dict(zip(cols, row, strict=True)))


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
