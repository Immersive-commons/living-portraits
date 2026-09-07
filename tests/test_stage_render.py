"""
test_stage_render.py -- the per-panel portrait compositor (runtime/stage_render.py).

Exercises the numpy/cv2 art path headless (pygame is import-guarded and NOT
required; the render() path runs onto clip_player.DummySurface). Covers:
  * composite_art over real assets (had_art True, exact panel size, non-blank);
  * the no-art fallback (had_art False) for a missing slug AND a corrupt asset;
  * the image cache NOT growing on a repeat call at the same size;
  * the rig_frame override landing over the plate;
  * slugify;
  * the headless DummySurface composite being non-blank.

cv2-absent boxes verify only the numpy fallback-plate path (it's all that can run);
numpy-absent boxes skip the file.
"""
from __future__ import annotations

import pytest

from conftest import HAVE_CV2, HAVE_NUMPY

pytestmark = pytest.mark.skipif(not HAVE_NUMPY, reason="stage_render tests need numpy")

import stage_render as sr


# --------------------------------------------------------------------------- #
# slugify (pure, dep-free)
# --------------------------------------------------------------------------- #
def test_slugify():
    assert sr.slugify("Phineas") == "phineas"
    assert sr.slugify("Seraphina") == "seraphina"
    assert sr.slugify("Madame X") == "madame_x"
    assert sr.slugify("  Mixed-Case Name! ") == "mixed_case_name"
    assert sr.slugify("") == ""
    assert sr.slugify(None) == ""


# --------------------------------------------------------------------------- #
# Fallback plate: the no-art backdrop is non-blank + panel-sized (cv2-free)
# --------------------------------------------------------------------------- #
def test_fallback_plate_nonblank_panel_sized():
    plate = sr._fallback_plate(256, 256, sr.PANEL_THEME["A"])
    assert plate.shape == (256, 256, 3)
    assert int(plate.sum()) > 0, "fallback plate is blank"


def test_missing_slug_reports_no_art():
    # composite_art with a slug that has no assets must fall back and report False.
    art, had = sr.composite_art(None, "no_such_slug", 256, 256, sr.PANEL_THEME["A"])
    assert art.shape == (256, 256, 3)
    assert int(art.sum()) > 0, "fallback art is blank"
    assert had is False, "missing-art must report had_art=False"


# --------------------------------------------------------------------------- #
# Art-present path (needs cv2 to decode the synth PNGs)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HAVE_CV2, reason="composite_art decode path needs cv2")
def test_composite_art_present_panel_a(written_portrait):
    import numpy as np
    slug, gd = written_portrait
    renderer = sr.StageRenderer(gen_dir=gd)
    art, had = sr.composite_art(renderer, slug, 256, 256, sr.PANEL_THEME["A"])
    assert art.shape == (256, 256, 3) and art.dtype == np.uint8
    assert int(art.sum()) > 0, "panel A art is blank"
    assert had is True, "should report had_art (portrait + cutout present)"


@pytest.mark.skipif(not HAVE_CV2, reason="needs cv2")
def test_composite_art_panel_b_different_size(written_portrait):
    slug, gd = written_portrait
    renderer = sr.StageRenderer(gen_dir=gd)
    art_b, had_b = sr.composite_art(renderer, slug, 192, 192, sr.PANEL_THEME["B"])
    assert art_b.shape == (192, 192, 3)
    assert int(art_b.sum()) > 0 and had_b


@pytest.mark.skipif(not HAVE_CV2, reason="needs cv2")
def test_cutout_composite_changes_the_plate(written_portrait):
    import numpy as np
    slug, gd = written_portrait
    renderer = sr.StageRenderer(gen_dir=gd)
    art, _ = sr.composite_art(renderer, slug, 256, 256, sr.PANEL_THEME["A"])
    # the bare portrait plate (no cutout) must differ from the composited result
    portrait_plate = sr._cover_resize_rgb(
        sr._imread_rgb(gd / f"{slug}_portrait.png"), 256, 256)
    assert not np.array_equal(art, portrait_plate), \
        "cutout composite did not change the plate"


@pytest.mark.skipif(not HAVE_CV2, reason="needs cv2")
def test_corrupt_asset_falls_back_not_crash(written_portrait, gen_dir):
    # A garbage file under a slug must resolve to the fallback look, never raise.
    bad = gen_dir / "bad_portrait.png"
    bad.write_bytes(b"not a real png")
    renderer = sr.StageRenderer(gen_dir=gen_dir)
    art, had = sr.composite_art(renderer, "bad", 256, 256, sr.PANEL_THEME["A"])
    assert art.shape == (256, 256, 3) and int(art.sum()) > 0
    assert had is False, "corrupt asset must fall back (had_art=False)"


# --------------------------------------------------------------------------- #
# Cache: a repeat call at the SAME (path, size) must not grow the cache
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HAVE_CV2, reason="needs cv2")
def test_cache_does_not_grow_on_repeat(written_portrait):
    slug, gd = written_portrait
    renderer = sr.StageRenderer(gen_dir=gd)
    sr.composite_art(renderer, slug, 256, 256, sr.PANEL_THEME["A"])  # warm
    before = len(renderer._cache)
    sr.composite_art(renderer, slug, 256, 256, sr.PANEL_THEME["A"])  # repeat
    assert len(renderer._cache) == before, "cache grew on a repeat call (re-decode)"


@pytest.mark.skipif(not HAVE_CV2, reason="needs cv2")
def test_cache_negative_entry_for_missing_then_picks_up_new_file(written_portrait, gen_dir):
    """A missing asset caches a miss (no disk hit per frame); creating the file
    later (mtime now present) invalidates that negative entry."""
    import cv2
    import stage_render as sr2
    renderer = sr.StageRenderer(gen_dir=gen_dir)
    # first: a slug with no bg -> negative cache entry
    img = renderer._cached_scaled(gen_dir / "late_bg.png", 256, 256, "rgb_cover")
    assert img is None
    # now create the file; the mtime-keyed cache must re-read it
    portrait = sr2._synthetic_portrait(256)
    cv2.imwrite(str(gen_dir / "late_bg.png"), cv2.cvtColor(portrait, cv2.COLOR_RGB2BGR))
    img2 = renderer._cached_scaled(gen_dir / "late_bg.png", 256, 256, "rgb_cover")
    assert img2 is not None and img2.shape == (256, 256, 3)


# --------------------------------------------------------------------------- #
# rig_frame override lands over the plate
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HAVE_CV2, reason="needs cv2")
def test_rig_frame_lands_over_plate(written_portrait):
    import numpy as np
    slug, gd = written_portrait
    renderer = sr.StageRenderer(gen_dir=gd)
    rig_rgb = np.full((256, 256, 3), (90, 0, 0), np.uint8)
    rig_alpha = np.zeros((256, 256), np.uint8)
    rig_alpha[64:192, 64:192] = 255  # opaque centre block
    art, had = sr.composite_art(renderer, slug, 256, 256, sr.PANEL_THEME["A"],
                                rig_frame=(rig_rgb, rig_alpha))
    assert had and art.shape == (256, 256, 3)
    # the centre pixel now carries the rig's red, not the underlying plate
    assert art[128, 128][0] >= 80, "rig frame did not land over the plate"


def test_rig_frame_over_fallback_plate_no_assets():
    """rig_frame composites even with no on-disk assets (plate = tinted fallback),
    and the result reports had_art True because a figure layer was drawn. cv2-free."""
    import numpy as np
    rig_rgb = np.full((128, 128, 3), (0, 80, 0), np.uint8)
    rig_alpha = np.full((128, 128), 255, np.uint8)
    art, had = sr.composite_art(None, "", 128, 128, sr.PANEL_THEME["A"],
                                rig_frame=(rig_rgb, rig_alpha))
    assert art.shape == (128, 128, 3) and had is True


# --------------------------------------------------------------------------- #
# Headless render() onto DummySurface: non-blank, no pygame needed
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HAVE_CV2, reason="needs cv2")
def test_render_onto_dummy_surface_nonblank(gen_dir):
    # render() slugifies beat_panel['char'] to find assets, so write them under
    # slugify('Phineas')=='phineas' and pass char='Phineas' (the real-name path).
    import cv2
    slug = "phineas"
    portrait = sr._synthetic_portrait(512)
    cutout = sr._synthetic_cutout(512)
    cv2.imwrite(str(gen_dir / f"{slug}_portrait.png"),
                cv2.cvtColor(portrait, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(gen_dir / f"{slug}_cutout.png"),
                cv2.cvtColor(cutout, cv2.COLOR_RGBA2BGRA))

    from clip_player import DummySurface
    renderer = sr.StageRenderer(gen_dir=gen_dir)
    ds = DummySurface(256, 256)
    panel = {"rect": (0, 0, 256, 256), **sr.PANEL_THEME["A"]}
    beat_panel = {"char": "Phineas", "action": "address_house",
                  "aside": "The feed says the worm industrialized."}
    had = renderer.render(ds, "A", panel, beat_panel, frame=5, fonts=None)
    assert had, "render() over real art should report had_art"
    assert ds.buffer.shape == (256, 256, 3)
    assert int(ds.buffer.sum()) > 0, "DummySurface buffer is blank"
    assert ds.blits >= 1


def test_render_dummy_surface_waiting_state_nonblank():
    """render() with beat_panel=None (no beat yet) still fills the panel with the
    fallback art -- the show never regresses to blank. cv2-free (fallback plate)."""
    from clip_player import DummySurface
    renderer = sr.StageRenderer(gen_dir="/no/such/dir")
    ds = DummySurface(192, 192)
    panel = {"rect": (0, 0, 192, 192), **sr.PANEL_THEME["B"]}
    renderer.render(ds, "B", panel, None, frame=0, fonts=None)
    assert int(ds.buffer.sum()) > 0, "waiting-state panel is blank"
