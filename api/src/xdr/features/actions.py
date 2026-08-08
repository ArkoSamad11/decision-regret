"""Deterministic action features (SPEC.md §7.1), built from socceraction's
standard VAEP feature functions rather than reimplemented by hand -- this is
the canonical socceraction feature recipe (gamestates + the documented xfns
list), which is what makes the LightGBM baseline comparable to published VAEP
numbers (SPEC.md §5).
"""

from __future__ import annotations

import pandas as pd
import socceraction.vaep.features as fs

# Carried through for the API/dashboard to render a moment, never fed to the
# model -- train.py's ID_COLUMNS excludes every "display_"-prefixed column
# from the feature matrix. Team/player identity in particular must not be a
# model feature: VAEP-style value is meant to be judged independent of who
# performed the action.
DISPLAY_COLUMNS = [
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "type_name",
    "result_name",
    "player_name",
    "team_name",
    "minute",
    "second",
]

# The standard VAEP feature block. Functions that "look" single-action
# (e.g. actiontype_onehot) are actually wrapped with socceraction's @simple
# decorator and operate on the full gamestates list, producing one column
# group per history slot (_a0 = current action, _a1..: preceding actions).
XFNS = [
    fs.actiontype,
    fs.actiontype_onehot,
    fs.bodypart,
    fs.bodypart_onehot,
    fs.result,
    fs.result_onehot,
    fs.goalscore,
    fs.startlocation,
    fs.endlocation,
    fs.startpolar,
    fs.endpolar,
    fs.movement,
    fs.team,
    fs.time,
    fs.time_delta,
    fs.space_delta,
]


def _seconds_remaining_in_period(actions: pd.DataFrame) -> pd.DataFrame:
    """Not a standard socceraction feature: added period duration is a real
    match-clock quantity socceraction's `time` feature does not expose, and
    stoppage time varies per period, so we estimate period length from the
    latest observed `time_seconds` in that (game, period) rather than
    assuming a fixed 45/15 minutes.
    """
    period_end = actions.groupby(["game_id", "period_id"])["time_seconds"].transform("max")
    return pd.DataFrame(
        {"seconds_remaining_in_period": period_end - actions["time_seconds"]}, index=actions.index
    )


def build_action_features(actions: pd.DataFrame, action_window: int = 3) -> pd.DataFrame:
    """One row per action: the standard VAEP feature block for the action
    itself plus the same block for the preceding `action_window` actions.

    `actions` may span many matches -- `fs.gamestates` groups by
    (game_id, period_id) internally when building action history, so history
    never crosses a match or period boundary.
    """
    actions = actions.sort_values(["game_id", "period_id", "time_seconds"]).reset_index(drop=True)
    states = fs.gamestates(actions, nb_prev_actions=action_window)
    blocks = [fn(states) for fn in XFNS]
    blocks.append(_seconds_remaining_in_period(actions))
    features = pd.concat(blocks, axis=1)
    features.index = actions.index
    id_cols = ["game_id", "action_id", "team_id", "period_id", "time_seconds"]
    display = actions[DISPLAY_COLUMNS].add_prefix("display_")
    return actions[id_cols].join(display).join(features)
