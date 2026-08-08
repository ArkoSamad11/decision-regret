"""SPEC.md §7.3 / §15: labels must never look past the end of the match they
belong to. socceraction's `scores`/`concedes` shift positionally over
whatever frame they're handed with no awareness of match boundaries -- the
one property this test suite exists to pin down is that `xdr.features.labels`
never hands them more than one match at a time.
"""

import pandas as pd

from xdr.features.labels import build_labels

HORIZON = 10


def _action(game_id, period_id, time_seconds, action_id, team_id, type_name="pass", result_id=1):
    return {
        "game_id": game_id,
        "period_id": period_id,
        "time_seconds": time_seconds,
        "action_id": action_id,
        "team_id": team_id,
        "type_name": type_name,
        "result_id": result_id,
    }


def test_no_lookahead_across_match_boundary():
    # Game 1: five ordinary actions by team 100, no goal anywhere in the game.
    game1 = [_action(1, 1, t, t, team_id=100) for t in range(5)]
    # Game 2: opens with a goal scored by team 100 -- the same team id as
    # game 1, so a boundary leak would plausibly go unnoticed if not checked.
    game2 = [_action(2, 1, 0, 0, team_id=100, type_name="shot", result_id=1)]
    game2 += [_action(2, 1, t, t, team_id=100) for t in range(1, 5)]

    actions = pd.DataFrame(game1 + game2)
    labels = build_labels(actions, horizon=HORIZON)

    game1_labels = labels[labels.game_id == 1]
    assert not game1_labels["label_scores"].any(), (
        "game 1's actions must not see game 2's opening goal even though it "
        "falls within the horizon window positionally"
    )


def test_goal_at_last_action_labels_correctly_within_horizon():
    # A 15-action match where the very last action is a successful shot.
    n = 15
    actions_list = [_action(1, 1, t, t, team_id=100) for t in range(n)]
    actions_list[-1] = _action(1, 1, n - 1, n - 1, team_id=100, type_name="shot", result_id=1)
    actions = pd.DataFrame(actions_list)

    labels = build_labels(actions, horizon=HORIZON)
    labels = labels.sort_values("action_id").reset_index(drop=True)

    # Actions within `horizon` actions of the goal (inclusive) score True;
    # actions further back do not.
    for action_id in range(n):
        expected = action_id >= (n - 1) - (HORIZON - 1)
        assert labels.loc[action_id, "label_scores"] == expected, action_id


def test_truncation_at_match_end_does_not_crash_or_wrap():
    # A short match (fewer actions than the horizon) with no goal at all.
    # This must not raise, and every action's labels must be False.
    actions = pd.DataFrame([_action(1, 1, t, t, team_id=100) for t in range(4)])
    labels = build_labels(actions, horizon=HORIZON)
    assert not labels["label_scores"].any()
    assert not labels["label_concedes"].any()


def test_concedes_is_scoped_to_own_match_too():
    game1 = [_action(1, 1, t, t, team_id=100) for t in range(5)]
    # Game 2's opening goal is scored by the OPPONENT (team 200) against a
    # team also present as team 100 in game 1 -- concedes must not leak either.
    game2 = [_action(2, 1, 0, 0, team_id=200, type_name="shot", result_id=1)]
    game2 += [_action(2, 1, t, t, team_id=100) for t in range(1, 5)]

    actions = pd.DataFrame(game1 + game2)
    labels = build_labels(actions, horizon=HORIZON)

    game1_labels = labels[labels.game_id == 1]
    assert not game1_labels["label_concedes"].any()
