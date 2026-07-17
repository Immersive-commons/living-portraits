"""
test_crossframe.py -- the cross-panel walk effect (runtime/crossframe.py).

numpy-only under test (no pygame, no cv2, no disk). Covers the figure-layer
geometry (exit/enter shape + dtype matching the rig_frame contract, centroid
motion, fade), edge_for_move direction derivation, and the CrossFrameState phase
machine: canonical EXIT->GAP->ENTER->DONE ordering, the empty GAP on both panels,
progress ramps, and occupancy handoff on DONE.

Skipped wholesale (not errored) on a box without numpy.
"""
from __future__ import annotations

import math

import pytest

from conftest import HAVE_NUMPY

pytestmark = pytest.mark.skipif(not HAVE_NUMPY, reason="crossframe needs numpy")

import crossframe as cf  # noqa: E402
from crossframe import (  # noqa: E402
    CrossFrameState,
    EXIT_S, GAP_S, ENTER_S,
    PHASE_PRE, PHASE_EXIT, PHASE_GAP, PHASE_ENTER, PHASE_DONE,
    edge_for_move, enter_frame, exit_frame, empty_layer,
)

PANEL_A = (0, 0, 256, 256)
PANEL_B = (256, 0, 192, 192)


# --------------------------------------------------------------------------- #
# Layer shape / dtype: must match stage_render's rig_frame = (rgb(h,w,3), alpha(h,w))
# --------------------------------------------------------------------------- #
def test_exit_frame_layer_shape_dtype(crossframe_cutout):
    import numpy as np
    rgb, alpha = exit_frame(PANEL_A, crossframe_cutout, 0.0, "right")
    assert rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8
    assert alpha.shape == (256, 256) and alpha.dtype == np.uint8


def test_enter_frame_layer_sized_to_target_panel(crossframe_cutout):
    # Panel B is 192 -- the layer must be exactly the target panel's (h, w),
    # never the source cutout's native size.
    rgb_b, alpha_b = enter_frame(PANEL_B, crossframe_cutout, 1.0, "left")
    assert rgb_b.shape == (192, 192, 3)
    assert alpha_b.shape == (192, 192)


# --------------------------------------------------------------------------- #
# EXIT: centroid moves toward the chosen edge, alpha fades to ~0
# --------------------------------------------------------------------------- #
def test_exit_right_moves_centroid_right_then_clears(crossframe_cutout):
    cx0 = cf._alpha_centroid_x(exit_frame(PANEL_A, crossframe_cutout, 0.0, "right")[1])
    cx_mid = cf._alpha_centroid_x(exit_frame(PANEL_A, crossframe_cutout, 0.5, "right")[1])
    _, a_end = exit_frame(PANEL_A, crossframe_cutout, 1.0, "right")
    assert cx0 is not None and cx_mid is not None
    assert cx_mid > cx0 + 1.0, f"EXIT centroid did not move right: {cx0:.1f}->{cx_mid:.1f}"
    assert cf._opaque_fraction(a_end) < 0.01, "EXIT did not clear the frame by p=1"


def test_exit_right_is_monotonic_rightward(crossframe_cutout):
    prev = -1.0
    for p in (0.0, 0.2, 0.4, 0.6):
        c = cf._alpha_centroid_x(exit_frame(PANEL_A, crossframe_cutout, p, "right")[1])
        assert c is not None and c >= prev - 0.5, f"EXIT not monotonic at p={p}"
        prev = c


def test_exit_left_moves_centroid_left(crossframe_cutout):
    # The synth blob is left-of-centre, so sample early (p=0.2) before it clears.
    cxl0 = cf._alpha_centroid_x(exit_frame(PANEL_A, crossframe_cutout, 0.0, "left")[1])
    cxl_early = cf._alpha_centroid_x(exit_frame(PANEL_A, crossframe_cutout, 0.2, "left")[1])
    assert cxl0 is not None and cxl_early is not None
    assert cxl_early < cxl0 - 1.0, f"EXIT-left centroid did not move left: {cxl0:.1f}->{cxl_early:.1f}"
    assert cf._opaque_fraction(exit_frame(PANEL_A, crossframe_cutout, 1.0, "left")[1]) < 0.01


# --------------------------------------------------------------------------- #
# ENTER: starts off-frame, marches in, settles on the static-home centroid
# --------------------------------------------------------------------------- #
def test_enter_left_starts_offframe_and_marches_in(crossframe_cutout):
    _, ain0 = enter_frame(PANEL_A, crossframe_cutout, 0.0, "left")
    cx_mid = cf._alpha_centroid_x(enter_frame(PANEL_A, crossframe_cutout, 0.6, "left")[1])
    cx_late = cf._alpha_centroid_x(enter_frame(PANEL_A, crossframe_cutout, 1.0, "left")[1])
    assert cf._opaque_fraction(ain0) < 0.05, "ENTER should start off-frame (near-empty)"
    assert cx_mid is not None and cx_late is not None
    assert cx_late > cx_mid + 1.0, f"ENTER centroid did not advance inward: {cx_mid:.1f}->{cx_late:.1f}"


def test_enter_settles_to_static_home(crossframe_cutout):
    cx_late = cf._alpha_centroid_x(enter_frame(PANEL_A, crossframe_cutout, 1.0, "left")[1])
    home_ref = cf._prep_figure_rgba(crossframe_cutout, 256, 256)
    home_cx = cf._alpha_centroid_x(home_ref[..., 3])
    assert abs(cx_late - home_cx) < 2.0, \
        f"ENTER did not settle to home centroid: {cx_late:.1f} vs {home_cx:.1f}"


def test_fade_curves_endpoints():
    # EXIT opacity: 1.0 near the start, 0.0 at the end. ENTER: mirror.
    assert cf._fade_out_curve(0.0) == pytest.approx(1.0)
    assert cf._fade_out_curve(1.0) == pytest.approx(0.0)
    assert cf._fade_in_curve(0.0) == pytest.approx(0.0)
    assert cf._fade_in_curve(1.0) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# edge_for_move: geometry-derived + name fallback
# --------------------------------------------------------------------------- #
def test_edge_for_move_from_geometry():
    geom = {"A": PANEL_A, "B": PANEL_B}
    assert edge_for_move("A", "B", geom) == ("right", "left")
    assert edge_for_move("B", "A", geom) == ("left", "right")


def test_edge_for_move_name_fallback():
    assert edge_for_move("A", "B", None) == ("right", "left")
    assert edge_for_move("B", "A", None) == ("left", "right")


# --------------------------------------------------------------------------- #
# CrossFrameState: lifecycle + idempotency
# --------------------------------------------------------------------------- #
def test_begin_transition_activates():
    x = CrossFrameState()
    x.set_occupant("A", "phineas")
    x.begin_transition("phineas", "A", "B")
    assert x.active and not x.done
    assert x.moving_char == "phineas"


def test_rebegin_same_move_does_not_restart_clock():
    x = CrossFrameState()
    x.begin_transition("phineas", "A", "B")
    x.advance(0.30)
    e_before = x._elapsed
    x.begin_transition("phineas", "A", "B")  # same move
    assert x._elapsed == e_before, "re-begin restarted an in-flight transition"


def test_different_move_supersedes():
    x = CrossFrameState()
    x.begin_transition("phineas", "A", "B")
    x.advance(0.30)
    x.begin_transition("seraphina", "B", "A")  # different move
    assert x._elapsed == 0.0 and x.moving_char == "seraphina"


# --------------------------------------------------------------------------- #
# The phase machine: canonical ordering, empty GAP, progress ramps
# --------------------------------------------------------------------------- #
def _walk(x, dt=1.0 / 30.0):
    """Drive a transition to DONE, collecting the observable telemetry the player
    switches on. Returns a dict of sequences."""
    steps = int(math.ceil((EXIT_S + GAP_S + ENTER_S) / dt)) + 5
    seq, gap_ticks = [], 0
    exit_prog, enter_prog = [], []
    for _ in range(steps):
        ph = x.phase()
        if not seq or seq[-1] != ph:
            seq.append(ph)
        pa, ga = x.panel_state("A")
        pb, gb = x.panel_state("B")
        if ph == PHASE_EXIT:
            assert pa == PHASE_EXIT, f"from-panel not EXIT during EXIT: {pa}"
            assert not x.should_show_figure("B"), "to-panel showed figure during EXIT"
            exit_prog.append(ga)
        if ph == PHASE_GAP:
            gap_ticks += 1
            assert not x.should_show_figure("A"), "from-panel showed figure during GAP"
            assert not x.should_show_figure("B"), "to-panel showed figure during GAP"
        if ph == PHASE_ENTER:
            assert pb == PHASE_ENTER, f"to-panel not ENTER during ENTER: {pb}"
            assert not x.should_show_figure("A"), "from-panel showed figure during ENTER"
            enter_prog.append(gb)
        x.advance(dt)
    return {"seq": seq, "gap_ticks": gap_ticks,
            "exit_prog": exit_prog, "enter_prog": enter_prog}


def test_phase_order_exit_gap_enter_done():
    x = CrossFrameState()
    x.set_occupant("A", "phineas")
    x.set_occupant("B", "seraphina")
    x.begin_transition("phineas", "A", "B")
    tel = _walk(x)
    # PRE may or may not be observed depending on the first tick; filter it.
    seq_no_pre = [p for p in tel["seq"] if p != PHASE_PRE]
    assert seq_no_pre == [PHASE_EXIT, PHASE_GAP, PHASE_ENTER, PHASE_DONE], \
        f"phase order wrong: {tel['seq']}"


def test_gap_shows_no_figure_on_either_panel():
    x = CrossFrameState()
    x.begin_transition("phineas", "A", "B")
    tel = _walk(x)
    assert tel["gap_ticks"] >= 1, "GAP phase (empty frame) never observed"


def test_progress_ramps_rise():
    x = CrossFrameState()
    x.begin_transition("phineas", "A", "B")
    tel = _walk(x)
    assert tel["exit_prog"] and tel["exit_prog"][0] < 0.2
    assert max(tel["exit_prog"]) > 0.8
    assert tel["enter_prog"] and tel["enter_prog"][0] < 0.3
    assert max(tel["enter_prog"]) > 0.8


def test_pre_phase_before_first_advance():
    x = CrossFrameState()
    x.begin_transition("phineas", "A", "B")
    # elapsed == 0 right after begin -> PRE.
    assert x.phase() == PHASE_PRE


# --------------------------------------------------------------------------- #
# Occupancy handoff on DONE
# --------------------------------------------------------------------------- #
def test_occupancy_committed_on_done():
    x = CrossFrameState()
    x.set_occupant("A", "phineas")
    x.set_occupant("B", "seraphina")
    x.begin_transition("phineas", "A", "B")
    x.advance(EXIT_S + GAP_S + ENTER_S + 0.1)  # past total
    assert x.done
    assert x.occupant_of("A") is None, "from-panel A not cleared on DONE"
    assert x.occupant_of("B") == "phineas", "to-panel B did not receive the mover"


def test_uninvolved_panel_reads_done_zero():
    x = CrossFrameState()
    x.begin_transition("phineas", "A", "B")
    x.advance(EXIT_S + GAP_S + ENTER_S + 0.1)
    # after the walk, the (now idle) machine reads DONE/0 for any panel
    assert x.panel_state("A") == (PHASE_DONE, 0.0)


def test_idle_machine_reads_done_for_any_panel():
    x = CrossFrameState()  # never began a transition
    assert x.panel_state("A") == (PHASE_DONE, 0.0)
    assert x.panel_state("B") == (PHASE_DONE, 0.0)
    assert x.should_show_figure("A") is False


def test_reset_drops_inflight_keeps_committed_occupancy():
    x = CrossFrameState()
    x.set_occupant("A", "phineas")
    x.begin_transition("phineas", "A", "B")
    x.advance(0.2)
    x.reset()
    assert not x.active and x.done
    # committed occupancy is untouched by reset (only the in-flight walk is dropped)
    assert x.occupant_of("A") == "phineas"


# --------------------------------------------------------------------------- #
# B->A mirrored move
# --------------------------------------------------------------------------- #
def test_b_to_a_uses_mirrored_directions_and_enter_panel():
    geom = {"A": PANEL_A, "B": PANEL_B}
    assert edge_for_move("B", "A", geom) == ("left", "right")
    y = CrossFrameState()
    y.set_occupant("B", "seraphina")
    y.begin_transition("seraphina", "B", "A")
    # mid-ENTER, A (the destination) is the panel showing the figure
    y.advance(EXIT_S + GAP_S + ENTER_S * 0.5)
    assert y.should_show_figure("A") and not y.should_show_figure("B")


# --------------------------------------------------------------------------- #
# empty_layer helper
# --------------------------------------------------------------------------- #
def test_empty_layer_is_fully_transparent():
    import numpy as np
    rgb, alpha = empty_layer(PANEL_B)
    assert rgb.shape == (192, 192, 3) and alpha.shape == (192, 192)
    assert int(alpha.sum()) == 0, "empty_layer alpha is not fully transparent"
