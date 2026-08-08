"""VAEP labels (SPEC.md §7.3): did the acting team score / concede within the
next `horizon` actions.

`socceraction.vaep.labels.scores` / `.concedes` determine "within the next N
actions" with a plain positional `shift(-i)` over whatever dataframe they are
given -- they do not group by `game_id`. Call them on a multi-match
concatenation directly and the last `horizon` actions of every match but the
last would look ahead into the *next* match's opening actions. This module's
only job is making sure that never happens: `build_labels` always calls them
per match, one game's actions at a time, so lookahead truncates at that
game's final action exactly as SPEC.md §7.3 / `test_labels.py` require.
"""

from __future__ import annotations

import pandas as pd
import socceraction.vaep.labels as lb


def build_labels(actions: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
    """One row per action: `label_scores`, `label_concedes`.

    `actions` may span many matches; each match's actions are sliced out,
    sorted chronologically, and labelled independently before recombining.
    """
    actions = actions.sort_values(["game_id", "period_id", "time_seconds"]).reset_index(drop=True)

    parts = []
    for game_id, game_actions in actions.groupby("game_id", sort=False):
        scores = lb.scores(game_actions, nr_actions=horizon)
        concedes = lb.concedes(game_actions, nr_actions=horizon)
        part = pd.DataFrame(
            {
                "game_id": game_id,
                "action_id": game_actions["action_id"].values,
                "label_scores": scores["scores"].values,
                "label_concedes": concedes["concedes"].values,
            },
            index=game_actions.index,
        )
        parts.append(part)

    labels = pd.concat(parts).sort_index()
    return labels
