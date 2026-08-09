"""SPEC.md §10 / §15: regret never negative, null propagates, off-camera
teammates never enumerated as declined options."""

import pandas as pd
import pytest

from xdr.counterfactual.options import compute_regret, enumerate_candidates, visible_teammate_points


def test_off_camera_teammates_are_never_pass_targets():
    frame = [
        {"x": 40.0, "y": 30.0, "teammate": True, "actor": False},
        {"x": 60.0, "y": 40.0, "teammate": False, "actor": False},  # opponent
        {"x": 50.0, "y": 34.0, "teammate": True, "actor": True},  # the actor themself
    ]
    visible = visible_teammate_points(frame)
    assert visible == [{"x": 40.0, "y": 30.0, "teammate": True, "actor": False}]


def test_enumerate_candidates_covers_pass_carry_and_shot():
    teammates = [{"x": 40.0, "y": 30.0}, {"x": 60.0, "y": 40.0}]
    candidates = enumerate_candidates(50.0, 34.0, attacks_right=True, visible_teammates=teammates)

    assert (candidates["kind"] == "pass").sum() == len(teammates)
    assert (candidates["kind"] == "carry").sum() == 16  # 8 angles x 2 distances
    assert (candidates["kind"] == "shot").sum() == 1

    shot = candidates[candidates["kind"] == "shot"].iloc[0]
    assert shot["end_x"] == 105.0  # attacks_right -> shoots at x=105

    # Every candidate stays on the pitch.
    assert (candidates["end_x"] >= 0).all() and (candidates["end_x"] <= 105).all()
    assert (candidates["end_y"] >= 0).all() and (candidates["end_y"] <= 68).all()


def test_shot_targets_the_opposite_goal_when_attacking_left():
    candidates = enumerate_candidates(50.0, 34.0, attacks_right=False, visible_teammates=[])
    shot = candidates[candidates["kind"] == "shot"].iloc[0]
    assert shot["end_x"] == 0.0


def _scored(values):
    return pd.DataFrame({"value": values, "scored": [True] * len(values)})


def test_regret_is_never_negative():
    # The best enumerated alternative is worse than what the player actually
    # did -- regret must clip at zero, not go negative.
    candidates = _scored([0.01, 0.02, 0.015])
    regret, best = compute_regret(chosen_value=0.5, chosen_support=0.9, scored_candidates=candidates, min_support=0.35)
    assert regret == 0.0
    assert best is not None


def test_regret_is_null_when_chosen_action_is_unsupported():
    candidates = _scored([0.3, 0.4])
    regret, best = compute_regret(chosen_value=0.1, chosen_support=0.1, scored_candidates=candidates, min_support=0.35)
    assert regret is None
    assert best is None


def test_regret_is_null_when_every_candidate_is_gated():
    candidates = pd.DataFrame({"value": [None, None], "scored": [False, False]})
    regret, best = compute_regret(chosen_value=0.1, chosen_support=0.9, scored_candidates=candidates, min_support=0.35)
    assert regret is None
    assert best is None


def test_regret_ignores_gated_candidates_even_if_others_are_scored():
    candidates = pd.DataFrame({"value": [0.9, 0.05], "scored": [False, True]})
    regret, best = compute_regret(chosen_value=0.02, chosen_support=0.9, scored_candidates=candidates, min_support=0.35)
    # The 0.9-value candidate is gated (unsupported); only the 0.05 one counts.
    assert regret == pytest.approx(0.03)
    assert best["value"] == 0.05
