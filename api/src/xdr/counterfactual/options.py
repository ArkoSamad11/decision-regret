"""Counterfactual enumeration and regret arithmetic (SPEC.md §10).

Three kinds of declined option, per SPEC.md §10:
  - a pass to each *visible* teammate (off-camera teammates are never
    enumerated -- a player cannot be charged with declining an option the
    360 data does not show was there),
  - a carry to each cell of a coarse pitch grid around the ball, and
  - a shot on goal.

Regret is clipped at zero (the model has no basis for claiming a player beat
every alternative it could enumerate) and withheld (`None`) whenever either
side of the subtraction -- the action taken, or the best scored alternative
-- is unsupported (SPEC.md §9.1/§10).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import socceraction.spadl.config as spadlcfg

from xdr.counterfactual.score import FOOT_BODYPART_ID, SUCCESS_RESULT_ID, build_candidate_features
from xdr.data.spadl import SPADL_X_MAX, SPADL_Y_MAX
from xdr.models.support import SupportIndex, support_score

PASS_TYPE_ID = spadlcfg.actiontypes.index("pass")
CARRY_TYPE_ID = spadlcfg.actiontypes.index("dribble")
SHOT_TYPE_ID = spadlcfg.actiontypes.index("shot")

TYPE_NAME_BY_ID = {PASS_TYPE_ID: "pass", CARRY_TYPE_ID: "dribble", SHOT_TYPE_ID: "shot"}

# Coarse carry grid: 8 compass directions at two distances from the ball --
# "coarse" per SPEC.md §10, not an attempt to enumerate every reachable point.
_CARRY_ANGLES = np.linspace(0, 2 * np.pi, 8, endpoint=False)
_CARRY_DISTANCES = (10.0, 20.0)


def visible_teammate_points(frame_points: list[dict]) -> list[dict]:
    """Freeze-frame points already carry `teammate`/`actor` flags; the actor
    itself (the player performing the action) is never a pass target."""
    return [p for p in frame_points if p.get("teammate") and not p.get("actor")]


def enumerate_candidates(
    start_x: float, start_y: float, attacks_right: bool, visible_teammates: list[dict]
) -> pd.DataFrame:
    rows = []
    for tm in visible_teammates:
        rows.append(
            {
                "kind": "pass",
                "end_x": tm["x"],
                "end_y": tm["y"],
                "type_id": PASS_TYPE_ID,
                "bodypart_id": FOOT_BODYPART_ID,
                "result_id": SUCCESS_RESULT_ID,
            }
        )

    for angle in _CARRY_ANGLES:
        for dist in _CARRY_DISTANCES:
            end_x = float(np.clip(start_x + dist * np.cos(angle), 0.0, SPADL_X_MAX))
            end_y = float(np.clip(start_y + dist * np.sin(angle), 0.0, SPADL_Y_MAX))
            rows.append(
                {
                    "kind": "carry",
                    "end_x": end_x,
                    "end_y": end_y,
                    "type_id": CARRY_TYPE_ID,
                    "bodypart_id": FOOT_BODYPART_ID,
                    "result_id": SUCCESS_RESULT_ID,
                }
            )

    goal_x = SPADL_X_MAX if attacks_right else 0.0
    rows.append(
        {
            "kind": "shot",
            "end_x": goal_x,
            "end_y": SPADL_Y_MAX / 2,
            "type_id": SHOT_TYPE_ID,
            "bodypart_id": FOOT_BODYPART_ID,
            "result_id": SUCCESS_RESULT_ID,
        }
    )
    return pd.DataFrame(rows)


def score_candidates(
    candidates: pd.DataFrame,
    real_action: dict,
    history: list[dict],
    period_max_time: float,
    feature_columns: list[str],
    numeric_columns: list[str],
    prepare_matrix,
    boosters: dict,
    calibrators: dict,
    support: SupportIndex,
    min_support: float,
) -> pd.DataFrame:
    """Feature-construct, predict, calibrate, and support-gate every
    candidate in one batch. `prepare_matrix` is injected (rather than
    imported) to avoid a circular import with `xdr.models.train`."""
    features = build_candidate_features(real_action, history, candidates, period_max_time)
    X, _ = prepare_matrix(features, feature_columns)

    p_scores = calibrators["scores"].predict(boosters["scores"].predict(X))
    p_concedes = calibrators["concedes"].predict(boosters["concedes"].predict(X))

    support_scores = support_score(support, X[numeric_columns])
    scored_mask = support_scores >= min_support

    result = candidates.reset_index(drop=True).copy()
    result["p_scores"] = p_scores
    result["p_concedes"] = p_concedes
    result["value"] = p_scores - p_concedes
    result["support"] = support_scores
    result["scored"] = scored_mask
    result.loc[~scored_mask, "value"] = np.nan
    return result


def compute_regret(
    chosen_value: float, chosen_support: float, scored_candidates: pd.DataFrame, min_support: float
) -> tuple[float | None, pd.Series | None]:
    """SPEC.md §10/§9.1: regret is `None` if the action taken is itself
    unsupported, or if every enumerated candidate was gated. Otherwise
    `max(0, best_value - chosen_value)` -- never negative."""
    if chosen_support < min_support:
        return None, None

    available = scored_candidates[scored_candidates["scored"]]
    if available.empty:
        return None, None

    best = available.loc[available["value"].idxmax()]
    regret = max(0.0, float(best["value"] - chosen_value))
    return regret, best
