"""SPEC.md §11.2 item 2: the recalibration slice must be disjoint from the
evaluation slice, and both must be genuine subsets of the target competition's
own matches -- a transfer study that accidentally evaluates on part of its own
recalibration data would overstate how well recalibration worked."""

import numpy as np

from xdr.evaluation.transfer import target_split


def test_recalibration_and_evaluation_splits_are_disjoint():
    game_ids = np.arange(100)
    splits = target_split(game_ids, recalibration_fraction=0.2, seed=0)
    recal = set(splits["recalibration"])
    eval_ = set(splits["evaluation"])
    assert recal.isdisjoint(eval_)
    assert recal | eval_ == set(game_ids.tolist())


def test_recalibration_fraction_is_approximately_respected():
    game_ids = np.arange(1000)
    splits = target_split(game_ids, recalibration_fraction=0.2, seed=0)
    assert 190 <= len(splits["recalibration"]) <= 210


def test_at_least_one_match_in_recalibration_even_for_tiny_competitions():
    game_ids = np.arange(3)
    splits = target_split(game_ids, recalibration_fraction=0.2, seed=0)
    assert len(splits["recalibration"]) >= 1
