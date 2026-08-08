"""SPEC.md §3: the single most likely source of silent, plausible-looking wrong
output. Written before spadl.py's implementation, per SPEC.md §15.

Three systems are in play:
  StatsBomb raw: 120 x 80, origin top-left, y increases downward.
  SPADL:         105 x 68, origin bottom-left, y increases upward.

A round trip through both conversions must be idempotent, and a fixed landmark
(the penalty spot, 11m from the goal line, horizontally centred) must land in
the expected place in both systems.
"""

import pytest

from xdr.data.spadl import (
    SB_X_MAX,
    SB_Y_MAX,
    SPADL_X_MAX,
    SPADL_Y_MAX,
    sb_to_spadl,
    spadl_to_sb,
)


def test_penalty_spot_lands_correctly_in_both_systems():
    # StatsBomb's documented coordinate for the penalty spot is (108, 40) in
    # its 120x80 grid.
    sb_x, sb_y = 108.0, 40.0
    x, y = sb_to_spadl(sb_x, sb_y)

    # SPADL: 105m long, penalty spot 11m from the goal line -> x ~= 94.
    assert x == pytest.approx(94.46, abs=0.1)
    # Horizontally centred on a 68m-wide pitch -> y ~= 34.
    assert y == pytest.approx(34.04, abs=0.1)


def test_round_trip_is_idempotent():
    for sb_x, sb_y in [(0.05, 0.05), (120, 80), (60, 40), (13.4, 71.2), (108.0, 40.0)]:
        x, y = sb_to_spadl(sb_x, sb_y)
        back_x, back_y = spadl_to_sb(x, y)
        assert back_x == pytest.approx(sb_x, abs=1e-9)
        assert back_y == pytest.approx(sb_y, abs=1e-9)


def test_cell_offset_is_applied_not_a_plain_linear_scale():
    # A naive x * 105/120 scale (no half-cell correction) would put the
    # penalty spot at exactly x=94.5. The corrected, cell-aware formula must
    # differ from that naive value -- this is the exact silent bug SPEC.md §3
    # warns about, so assert the correction is actually present.
    naive_x = 108.0 * (SPADL_X_MAX / SB_X_MAX)
    x, _ = sb_to_spadl(108.0, 40.0)
    assert x != pytest.approx(naive_x, abs=1e-6)


def test_origin_corners_flip_correctly():
    # StatsBomb top-left cell (0.05, 0.05) is near SPADL bottom-left (0, SPADL_Y_MAX).
    x, y = sb_to_spadl(0.05, 0.05)
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(SPADL_Y_MAX, abs=1e-9)

    # StatsBomb bottom-right (SB_X_MAX, SB_Y_MAX) -> SPADL top-right (SPADL_X_MAX, 0).
    x, y = sb_to_spadl(SB_X_MAX, SB_Y_MAX)
    assert x == pytest.approx(SPADL_X_MAX, abs=0.2)
    assert y == pytest.approx(0.0, abs=0.2)


def test_scale_is_applied_independently_per_axis():
    # Half the raw width/height must map to (approximately, modulo the
    # half-cell offset) half the SPADL width/height.
    x, y = sb_to_spadl(SB_X_MAX / 2, SB_Y_MAX / 2)
    assert x == pytest.approx(SPADL_X_MAX / 2, abs=0.1)
    assert y == pytest.approx(SPADL_Y_MAX / 2, abs=0.1)


def test_attacking_direction_normalization_flips_second_half():
    from xdr.data.spadl import normalize_attacking_direction

    # A team attacking right-to-left (period 2, say) has its actions mirrored
    # so every team always appears to attack left-to-right.
    x, y = 20.0, 10.0
    flipped_x, flipped_y = normalize_attacking_direction(x, y, attacks_right=False)
    assert flipped_x == pytest.approx(SPADL_X_MAX - x)
    assert flipped_y == pytest.approx(SPADL_Y_MAX - y)

    # A team already attacking left-to-right is left untouched.
    same_x, same_y = normalize_attacking_direction(x, y, attacks_right=True)
    assert same_x == pytest.approx(x)
    assert same_y == pytest.approx(y)
