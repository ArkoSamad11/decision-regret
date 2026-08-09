"""Feature construction for hypothetical actions (SPEC.md §10). `options.py`
enumerates *what* a player could have done instead; this module builds the
feature row the trained model needs to score *how good* each hypothetical
would have been.

Reimplementing socceraction's feature functions by hand for a synthetic
action would risk silently diverging from how the real training features
were computed. Instead, this reuses the exact same `XFNS` list and gamestates
structure `xdr.features.actions.build_action_features` feeds them: a list of
per-history-slot DataFrames `[a0, a1, a2, a3]`. Only `a0` (the action itself)
differs across candidates; `a1`/`a2`/`a3` (the real actions immediately
preceding it) are broadcast unchanged to every candidate, because a
counterfactual changes what happens *now*, not what already happened.

SPEC.md §10's outcome-handling decision: this pass takes the explicitly
sanctioned default (b) -- assume every candidate succeeds (`result_id` =
`success`) rather than training a completion-probability model to marginalize
over. Every candidate's `value` is therefore an **upper bound** on its true
expected value, and regret computed from it is an upper bound on forgone
value, not a point estimate. This is stated here, in docs/DECISIONS.md, and
in the dashboard copy -- SPEC.md §10 is explicit that silently doing (b) and
describing it as the marginalized version (a) is not acceptable.
"""

from __future__ import annotations

import pandas as pd
import socceraction.spadl.config as spadlcfg

from xdr.features.actions import XFNS

SUCCESS_RESULT_ID = spadlcfg.results.index("success")
FOOT_BODYPART_ID = spadlcfg.bodyparts.index("foot")

CANDIDATE_OVERRIDE_COLUMNS = ["end_x", "end_y", "type_id", "bodypart_id", "result_id"]


def build_candidate_features(
    real_action: dict,
    history: list[dict],
    candidates: pd.DataFrame,
    period_max_time: float,
) -> pd.DataFrame:
    """One feature row per row of `candidates`.

    `real_action`: the SPADL action row (as a dict) the player actually
    performed -- its `game_id`/`period_id`/`time_seconds`/`team_id`/
    `player_id`/`start_x`/`start_y` define the decision point every candidate
    shares.
    `history`: the real actions immediately preceding it, oldest last removed
    -- `history[0]` is one action before `real_action`, `history[1]` two
    before, matching `action_window` slot order.
    `candidates`: one row per hypothetical action, with `end_x`, `end_y`,
    `type_id`, `bodypart_id`, `result_id` filled in; every other attribute
    (start location, timing, team/player identity) is copied from
    `real_action`.
    `period_max_time`: the real period's observed max `time_seconds`
    (SPEC.md §7.1's "seconds remaining in period" needs the whole match's
    context, which a single decision point does not have on its own).
    """
    n = len(candidates)
    a0 = pd.DataFrame([real_action] * n)
    for col in CANDIDATE_OVERRIDE_COLUMNS:
        a0[col] = candidates[col].values
    a0 = a0.reset_index(drop=True)

    states = [a0]
    for prev in history:
        states.append(pd.DataFrame([prev] * n).reset_index(drop=True))

    blocks = [fn(states) for fn in XFNS]
    blocks.append(
        pd.DataFrame(
            {"seconds_remaining_in_period": period_max_time - a0["time_seconds"]}, index=a0.index
        )
    )
    features = pd.concat(blocks, axis=1)
    features.index = a0.index
    return features
