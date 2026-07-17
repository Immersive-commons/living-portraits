"""
test_rig_loop.py -- the rig.py <-> stage_render.py bridge (runtime/rig_loop.py).

The bridge owns one Rig per slug (cached, build-once), pulls rig.frame(t), and
splits the RGBA tile into the (rgb(H,W,3), alpha(H,W)) uint8 TUPLE stage_render
destructures. It degrades to None for a missing/unbuildable rig (the static-cutout
fallback sentinel).

Two regimes, both honest about deps:
  * cv2 present  -> a real Rig builds from rig.py's synth fixtures; we assert the
    full tuple-shape / colour-split / build-once-cache / live-over-t contract.
  * cv2 absent   -> the bridge must still import and degrade every call to None.
Numpy-absent boxes skip the file entirely.
"""
from __future__ import annotations

import pytest

from conftest import HAVE_CV2, HAVE_NUMPY

pytestmark = pytest.mark.skipif(not HAVE_NUMPY, reason="rig_loop tests need numpy")

import rig_loop  # noqa: E402
from rig_loop import RigLoop, _slug_for_char  # noqa: E402


# --------------------------------------------------------------------------- #
# Degrade-to-None (works WITH or WITHOUT cv2; a missing slug always degrades)
# --------------------------------------------------------------------------- #
def test_missing_slug_returns_none(gen_dir):
    loop = RigLoop(gen_dir=gen_dir)
    assert loop.rig_frame_for("no_such_character", 0.0) is None


def test_missing_slug_miss_is_cached(gen_dir):
    loop = RigLoop(gen_dir=gen_dir)
    loop.get_rig("no_such_character")
    # the None is cached so we don't re-probe the decode every frame
    assert "no_such_character" in loop._rigs
    assert loop._rigs["no_such_character"] is None


def test_empty_slug_is_none(gen_dir):
    loop = RigLoop(gen_dir=gen_dir)
    assert loop.get_rig("") is None
    assert loop.rig_frame_for("", 0.0) is None


def test_rig_frame_for_beat_none_and_charless(gen_dir):
    loop = RigLoop(gen_dir=gen_dir)
    # None beat / char-less beat / empty-char beat are all the static-path sentinel.
    assert loop.rig_frame_for_beat(None, 0.0) is None
    assert loop.rig_frame_for_beat({"action": "idle"}, 0.0) is None
    assert loop.rig_frame_for_beat({"char": ""}, 0.0) is None


def test_slug_resolution_matches_stage_render():
    # The bridge keys the rig on the SAME slug as the static-cutout fallback.
    assert _slug_for_char("Synthhero") == "synthhero"
    assert _slug_for_char("Madame X") == "madame_x"
    assert _slug_for_char("") == ""


# --------------------------------------------------------------------------- #
# Real Rig path (needs cv2 -- rig.py decodes the cutout with it)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HAVE_CV2, reason="Rig.from_slug needs cv2 to decode the cutout")
def test_rig_builds_and_is_cached(written_rig):
    slug, gd = written_rig
    loop = RigLoop(gen_dir=gd)
    r1 = loop.get_rig(slug)
    r2 = loop.get_rig(slug)
    assert r1 is not None, "Rig failed to build from the synth fixture"
    assert r1 is r2, "get_rig rebuilt the Rig instead of caching it"


@pytest.mark.skipif(not HAVE_CV2, reason="needs cv2")
def test_rig_frame_tuple_shape_dtype(written_rig):
    import numpy as np
    slug, gd = written_rig
    loop = RigLoop(gen_dir=gd)
    rig = loop.get_rig(slug)
    H, W = rig.h, rig.w
    rf = loop.rig_frame_for(slug, 0.0)
    assert isinstance(rf, tuple) and len(rf) == 2
    rgb, alpha = rf
    # rgb: (H,W,3) uint8 -- what stage_render._cover_resize_rgb expects
    assert rgb.dtype == np.uint8 and rgb.shape == (H, W, 3)
    # alpha: (H,W) uint8, 2-D single channel -- what _fit_alpha indexes (NOT (H,W,1))
    assert alpha.dtype == np.uint8 and alpha.ndim == 2 and alpha.shape == (H, W)


@pytest.mark.skipif(not HAVE_CV2, reason="needs cv2")
def test_rig_frame_colour_split_no_swap(written_rig):
    """rgb must be frame[...,:3] and alpha frame[...,3] -- no cvtColor leaked into
    the bridge (both ends are RGB-ordered)."""
    import numpy as np
    slug, gd = written_rig
    loop = RigLoop(gen_dir=gd)
    rig = loop.get_rig(slug)
    probe_t = 7 / 24.0
    src = rig.frame(probe_t)
    rgb, alpha = loop.rig_frame_for(slug, probe_t)
    assert np.array_equal(rgb, src[..., :3]), "rgb is not frame[...,:3] (colour swap leaked)"
    assert np.array_equal(alpha, src[..., 3]), "alpha is not frame[...,3] (wrong channel index)"


@pytest.mark.skipif(not HAVE_CV2, reason="needs cv2")
def test_rig_frame_changes_over_time(written_rig):
    """The rig is live: rgb changes across t (idle sway / breathe / blink)."""
    import numpy as np
    slug, gd = written_rig
    loop = RigLoop(gen_dir=gd)
    frames = []
    for i in range(30):
        rgb, alpha = loop.rig_frame_for(slug, i / 24.0)
        frames.append(rgb.copy())
    base = frames[0].astype(np.int16)
    deltas = [int(np.abs(frames[i].astype(np.int16) - base).sum())
              for i in range(1, len(frames))]
    assert max(deltas) > 0, "rig frames never change over t -- animation static"


@pytest.mark.skipif(not HAVE_CV2, reason="needs cv2")
def test_rig_frame_for_beat_resolves_slug(written_rig):
    slug, gd = written_rig
    loop = RigLoop(gen_dir=gd)
    # capitalised display name -> lowercase slug, same as the static-cutout path
    rf = loop.rig_frame_for_beat({"char": "Synthhero", "action": "idle"}, 0.25)
    assert rf is not None, "rig_frame_for_beat('Synthhero') resolved no rig"


@pytest.mark.skipif(not HAVE_CV2, reason="needs cv2")
def test_rig_frame_composites_through_stage_render(written_rig):
    """End-to-end: the (rgb, alpha) tuple drops into stage_render.composite_art and
    composites (had_art True, panel-sized RGB) at both panel sizes."""
    import numpy as np
    import stage_render as sr
    slug, gd = written_rig
    loop = RigLoop(gen_dir=gd)
    renderer = sr.StageRenderer(gen_dir=gd)
    rf = loop.rig_frame_for(slug, 0.5)
    for pw, ph in ((256, 256), (192, 192)):
        art, had = sr.composite_art(renderer, slug, pw, ph, sr.PANEL_THEME["A"],
                                    rig_frame=rf)
        assert had is True, "stage_render reported no art for a live rig frame"
        assert art.shape == (ph, pw, 3) and art.dtype == np.uint8
        assert int(art.sum()) > 0, "stage_render composite is blank"


# --------------------------------------------------------------------------- #
# cv2-ABSENT regime: the bridge still imports and degrades to None
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(HAVE_CV2, reason="only meaningful when cv2 is genuinely absent")
def test_degrades_to_none_without_cv2(gen_dir):
    # On a box with no cv2 the Rig cannot build; every slug -> None.
    loop = RigLoop(gen_dir=gen_dir)
    assert loop.rig_frame_for("anything", 0.0) is None
    assert loop.rig_frame_for_beat({"char": "Anything"}, 0.0) is None
