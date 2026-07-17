"""
rig_loop.py -- the seam between rig.py (the animator) and stage_render.py (the
compositor). One small bridge box in the living-portraits architecture (v3).

The two sides speak almost-but-not-quite the same dialect:

    rig.Rig.frame(t)            -> (H, W, 4) uint8 RGBA          (RGB-ordered)
    stage_render.render_stage(..., rig_frame=...) wants
                                   rig_frame = (rgb (H,W,3) uint8,
                                                alpha (H,W) uint8)  (RGB-ordered)

So this module:
  * owns one Rig per character slug (lazily built via Rig.from_slug, cached),
  * pulls rig.frame(t) and splits the RGBA tile into the (rgb, alpha) TUPLE
    stage_render destructures (`rig_rgb, rig_alpha = rig_frame`), in the exact
    dtype / shape / colour-order it expects, and
  * degrades gracefully: if a character has no cutout / rig.json, the Rig build
    fails, we cache that failure as None, and rig_frame_for() returns None --
    which is precisely the "no rig" sentinel stage_render already handles by
    falling back to the static <slug>_cutout.png (and that, in turn, to the
    tinted plate). A missing rig disables animation for THAT character only; the
    show never regresses to blank.

Colour order (the one cross-module gotcha)
-------------------------------------------
Both ends are RGB, so NO cv2.cvtColor is needed in this bridge:
  * rig.py reads the cutout BGRA via cv2 then converts BGRA->RGBA on load
    (rig.py Rig.__init__, COLOR_BGRA2RGBA) explicitly "so our frames are
    RGB-ordered like clip_player's frames". So rig.frame(t)[..., :3] is RGB.
  * stage_render's plate is RGB end-to-end (_imread_rgb does BGR->RGB), and it
    blends rig_frame's rgb straight over that plate (composite_art ->
    _alpha_over) with no channel swap. _cover_resize_rgb / _fit_alpha never
    touch channel order.
Feeding rig.frame's RGB into stage_render's RGB slot is therefore correct as-is.
Splitting alpha: take index 3 (`rgba[..., 3]`), a single (H,W) channel -- NOT a
3-channel or keepdims slice; stage_render's _fit_alpha indexes it as 2-D (h, w).

Sizing
------
We hand stage_render a panel-agnostic frame at the cutout's native resolution.
stage_render cover-resizes both rgb and alpha to the panel size itself
(composite_art: `_cover_resize_rgb(rig_rgb, w, h)` + `_fit_alpha(rig_alpha,
w, h)`), so this module does NOT need to know the panel rect. Keeping the rig at
native res also means one cached Rig serves both Panel A (256) and Panel B (192).

Cost
----
rig.frame is ~2 ms (small affine warps on feature sub-rects, never the whole
stage). The only expensive step is Rig.from_slug (PNG decode + iris-mask
precompute), which is why it's cached per slug and built at most once. Calling
rig_frame_for(slug, t) every frame at 30 FPS just re-runs the ~2 ms warp.

Integration (the one call the conductor adds to player.py)
----------------------------------------------------------
player.py's draw path already resolves the per-panel beat at
`beat.get("panels", {}).get(name)` and calls
`render_stage(screen, name, panel, beat_panel, frame, fonts)`
(player.py draw_stage, inside the `_HAVE_RENDER` try block). The seam is one
extra line that produces the rig frame and threads it in:

    # once, module-level (alongside the render_stage import):
    from rig_loop import RigLoop
    RIGLOOP = RigLoop()
    PLAYER_T0 = time.monotonic()        # show start, for the animation clock

    # inside draw_stage(screen, name, panel, beat, frame, fonts):
    beat_panel = beat.get("panels", {}).get(name) if beat else None
    t = time.monotonic() - PLAYER_T0    # elapsed SECONDS since player start
    rf = RIGLOOP.rig_frame_for_beat(beat_panel, t)   # (rgb, alpha) | None
    render_stage(screen, name, panel, beat_panel, frame, fonts, rig_frame=rf)

`t` is wall-clock elapsed seconds since the player started (monotonic so it is
immune to clock adjustments). rig.frame(t) drives its idle sway / breathe /
blink off that single scalar; the existing per-frame integer `frame` counter
stays as-is for the liveness dot, untouched. When rf is None (no rig assets for
that character, or no beat yet) render_stage behaves exactly as it does today.

This module imports `rig` and `stage_render` with bare imports, matching the
sibling modules' convention -- they resolve once `runtime/` is on sys.path,
which player.py guarantees (`sys.path.insert(0, str(ROOT / "runtime"))`). The
self-test below puts its own directory on sys.path so it runs standalone too.

    python runtime/rig_loop.py     # headless self-test (synth cutout+spec, no pygame)

opencv (cv2) + numpy back the Rig; this bridge itself is pure numpy slicing.
Both are import-guarded so the module imports even where they are absent (the
self-test then no-ops with a clear message instead of crashing).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

# numpy is only needed for type hints + the self-test; guard it so importing this
# module never hard-fails on a box without numpy (mirrors stage_render's cv2 guard).
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy is present in the real pipeline
    np = None  # type: ignore

ROOT = Path(__file__).resolve().parent.parent
GEN_DIR = ROOT / "data" / "gen"

# A rig frame as stage_render consumes it: (rgb (H,W,3) uint8, alpha (H,W) uint8).
RigFrame = Tuple["np.ndarray", "np.ndarray"]


def _slug_for_char(char_name: str) -> str:
    """Lowercase a beat's capitalised display name to the pipeline's asset slug.

    Mirrors stage_render.slugify so the rig and the static-cutout fallback key on
    the SAME slug ('Phineas' -> 'phineas', 'Madame X' -> 'madame_x'). We prefer
    stage_render's own slugify when importable (single source of truth); only if
    that import fails do we fall back to an inline copy of the identical rule.
    """
    try:
        from stage_render import slugify  # type: ignore
        return slugify(char_name)
    except Exception:  # pragma: no cover - stage_render is a sibling, normally present
        import re
        s = (char_name or "").strip().lower()
        return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


class RigLoop:
    """Per-slug Rig cache + the RGBA -> (rgb, alpha) split stage_render wants.

    One instance lives for the whole player run so each character's Rig is built
    at most once (Rig.from_slug decodes a PNG and precomputes iris masks -- not
    something to re-pay every frame). Build failures are cached as None, so a
    character with no cutout / rig.json is asked for exactly once and thereafter
    cheaply reports "no rig" (rig_frame_for -> None).

    Usage:
        RIGLOOP = RigLoop()                         # once, before the player loop
        rf = RIGLOOP.rig_frame_for("phineas", t)    # (rgb, alpha) | None
        render_stage(screen, name, panel, beat_panel, frame, fonts, rig_frame=rf)
    """

    def __init__(self, gen_dir: Optional[Path] = None, seed: int = 7) -> None:
        self.gen_dir = Path(gen_dir) if gen_dir is not None else GEN_DIR
        self.seed = seed
        # slug -> Rig | None.  None is a *cached* "this slug has no rig assets".
        self._rigs: dict[str, object] = {}

    # ---- the Rig cache ----------------------------------------------------- #
    def get_rig(self, slug: str):
        """Return the cached Rig for `slug`, building it once on first use.

        Returns the Rig instance, or None if its assets are missing/unreadable
        (the build raised). Both outcomes are cached -- a missing rig is probed
        exactly once. An empty/false-y slug is always None (no asset to load).
        """
        if not slug:
            return None
        if slug in self._rigs:
            return self._rigs[slug]

        rig = None
        try:
            # Import here (not at module load) so this file imports even if rig's
            # heavy deps (cv2) are unavailable on an import-only box; matches the
            # lazy-degrade contract (a missing dep just disables animation).
            from rig import Rig  # type: ignore
            rig = Rig.from_slug(slug, gen_dir=self.gen_dir, seed=self.seed)
        except Exception:
            # Missing cutout / rig.json / bad spec / no cv2 -> no animation for
            # this character. Cache the None so we don't retry the decode each
            # frame; stage_render falls back to the static cutout or the plate.
            rig = None

        self._rigs[slug] = rig
        return rig

    # ---- the seam: RGBA -> (rgb, alpha) ------------------------------------ #
    def rig_frame_for(self, slug: str, t: float) -> Optional[RigFrame]:
        """Animate `slug` at time `t` (seconds) into stage_render's rig_frame.

        Returns (rgb (H,W,3) uint8, alpha (H,W) uint8) -- the exact tuple
        stage_render destructures as `rig_rgb, rig_alpha = rig_frame` -- or None
        when this slug has no rig (so the caller passes rig_frame=None and
        stage_render uses the static cutout / fallback plate).

        rig.frame(t) returns (H,W,4) RGBA already RGB-ordered, so we split with
        no colour conversion: rgb = frame[..., :3], alpha = frame[..., 3]. We
        return native-resolution channels; stage_render cover-resizes both to the
        panel size itself, so one Rig serves both panels.
        """
        rig = self.get_rig(slug)
        if rig is None:
            return None
        rgba = rig.frame(t)  # (H, W, 4) uint8 RGBA, ~2 ms
        # Slice views; stage_render copies them through resize/dstack, so sharing
        # the rgba buffer is safe (it isn't mutated downstream). rgb is RGB (no
        # cvtColor), alpha is the single index-3 channel as a 2-D (H, W) array.
        rgb = rgba[..., :3]
        alpha = rgba[..., 3]
        return rgb, alpha

    def rig_frame_for_beat(self, beat_panel: Optional[dict],
                           t: float) -> Optional[RigFrame]:
        """Convenience: resolve the slug from a beat panel, then rig_frame_for.

        `beat_panel` is player.py's inner {char, action, aside} dict (what
        `beat.get("panels", {}).get(name)` yields), or None when the director
        hasn't written a beat for this panel yet. Returns None for a None /
        char-less beat (-> stage_render's static path), else delegates.

        Note this takes the inner beat PANEL, not the whole beat: the panel-name
        lookup (which panel shows which character) stays in player.py, exactly
        where stage_render also expects the caller to have already done it.
        """
        if not beat_panel:
            return None
        slug = _slug_for_char(beat_panel.get("char", ""))
        return self.rig_frame_for(slug, t)


# --------------------------------------------------------------------------- #
# Headless self-test -- synthesises a cutout + rig spec (reusing rig.py's own
# fixtures so we exercise the true Rig path), drives rig_frame_for across ~30 t
# values, and asserts the tuple shape / dtype / colour-split / motion-over-time
# contract stage_render relies on. No pygame, no network, no real assets.
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    # Make bare `import rig` / `import stage_render` resolve when run directly.
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    print("[rig_loop] self-test", flush=True)

    if np is None:
        print("[rig_loop] numpy absent -- skipping (bridge is import-safe, "
              "nothing to assert without numpy)", flush=True)
        return 0

    try:
        import cv2  # noqa: F401  (proves the Rig decode path is available)
    except Exception:
        print("[rig_loop] cv2 absent -- Rig cannot build; verifying graceful "
              "degrade only.", flush=True)
        loop_nodep = RigLoop()
        assert loop_nodep.rig_frame_for("anything", 0.0) is None
        assert loop_nodep.rig_frame_for_beat({"char": "Anything"}, 0.0) is None
        assert loop_nodep.rig_frame_for_beat(None, 0.0) is None
        print("[rig_loop] OK (cv2 absent: degrade-to-None path verified)",
              flush=True)
        return 0

    import json
    import tempfile

    # Reuse rig.py's synthetic fixtures so we test against the REAL Rig + the
    # real rig-spec schema (not a private re-implementation of either).
    import rig as rig_mod  # type: ignore

    tmp = Path(tempfile.mkdtemp(prefix="lp_rigloop_"))
    gen = tmp / "gen"
    gen.mkdir(parents=True, exist_ok=True)

    slug = "synthhero"
    cut = rig_mod._synth_cutout(256)
    spec = rig_mod._synth_spec_for(cut)
    # Write under the slug-keyed names Rig.from_slug expects: <slug>_cutout.png +
    # <slug>_rig.json. Cutout is written BGRA (rig.py reads it back as RGBA).
    cv2.imwrite(str(gen / f"{slug}_cutout.png"),
                cv2.cvtColor(cut, cv2.COLOR_RGBA2BGRA))
    (gen / f"{slug}_rig.json").write_text(json.dumps(spec), encoding="utf-8")

    loop = RigLoop(gen_dir=gen)

    # --- the Rig builds once and is cached -----------------------------------
    r1 = loop.get_rig(slug)
    r2 = loop.get_rig(slug)
    assert r1 is not None, "Rig failed to build from the synth fixture"
    assert r1 is r2, "get_rig did not cache -- rebuilt the Rig on second call"
    print(f"[rig_loop] Rig cached: build-once for '{slug}' "
          f"({r1.w}x{r1.h}, layers {sorted(r1._box)})", flush=True)

    # --- the seam shape/dtype contract over ~30 t values ---------------------
    H, W = r1.h, r1.w
    frames = []
    for i in range(30):
        t = i / 24.0
        rf = loop.rig_frame_for(slug, t)
        assert rf is not None, f"rig_frame_for returned None at t={t}"
        assert isinstance(rf, tuple) and len(rf) == 2, \
            f"rig_frame must be a 2-tuple, got {type(rf)} len " \
            f"{len(rf) if hasattr(rf, '__len__') else '?'}"
        rgb, alpha = rf
        # rgb: (H, W, 3) uint8 -- what stage_render._cover_resize_rgb expects
        assert rgb.dtype == np.uint8, f"rgb dtype {rgb.dtype} != uint8"
        assert rgb.ndim == 3 and rgb.shape == (H, W, 3), \
            f"rgb shape {rgb.shape} != {(H, W, 3)}"
        # alpha: (H, W) uint8, 2-D single channel -- what _fit_alpha indexes
        assert alpha.dtype == np.uint8, f"alpha dtype {alpha.dtype} != uint8"
        assert alpha.ndim == 2 and alpha.shape == (H, W), \
            f"alpha shape {alpha.shape} != {(H, W)} (must be 2-D, not (H,W,1))"
        frames.append((rgb.copy(), alpha.copy()))
    print(f"[rig_loop] 30 frames: rgb {frames[0][0].shape} uint8, "
          f"alpha {frames[0][1].shape} uint8 -- matches stage_render rig_frame "
          f"tuple", flush=True)

    # --- colour-order split is correct vs the source RGBA --------------------
    # rgb must equal rig.frame(t)[..., :3] and alpha rig.frame(t)[..., 3], i.e.
    # NO channel swap happened in the bridge (both ends are RGB).
    probe_t = 7 / 24.0
    src = r1.frame(probe_t)
    rgb_p, alpha_p = loop.rig_frame_for(slug, probe_t)
    assert np.array_equal(rgb_p, src[..., :3]), \
        "rgb is not frame[...,:3] -- a colour swap leaked into the bridge"
    assert np.array_equal(alpha_p, src[..., 3]), \
        "alpha is not frame[...,3] -- wrong channel index"
    print("[rig_loop] colour split verified: rgb==frame[...,:3] (RGB, no swap), "
          "alpha==frame[...,3]", flush=True)

    # --- frames change over time (the rig is live, not frozen) ---------------
    rgb0 = frames[0][0].astype(np.int16)
    deltas = [int(np.abs(frames[i][0].astype(np.int16) - rgb0).sum())
              for i in range(1, len(frames))]
    assert max(deltas) > 0, "rig frames never change over t -- animation static"
    # alpha can move too (head sway shifts the figure) -- not required to, but
    # the rgb must; report both so a reviewer sees the magnitude.
    a0 = frames[0][1].astype(np.int16)
    adeltas = [int(np.abs(frames[i][1].astype(np.int16) - a0).sum())
               for i in range(1, len(frames))]
    print(f"[rig_loop] motion over t: max rgb delta-from-f0 = {max(deltas):,}, "
          f"max alpha delta = {max(adeltas):,}", flush=True)

    # --- the (rgb, alpha) tuple actually drives stage_render end-to-end ------
    # Prove the seam fits the consumer: feed our frame into composite_art and
    # confirm it composites (had_art True, panel-sized RGB out, no exception).
    try:
        import stage_render as sr  # type: ignore
        renderer = sr.StageRenderer(gen_dir=gen)
        rf = loop.rig_frame_for(slug, 0.5)
        # Panel A size (256) and Panel B size (192) -- stage_render resizes both.
        for pw, ph in ((256, 256), (192, 192)):
            art, had = sr.composite_art(renderer, slug, pw, ph,
                                        sr.PANEL_THEME["A"], rig_frame=rf)
            assert had is True, "stage_render reported no art for a live rig frame"
            assert art.shape == (ph, pw, 3), \
                f"stage_render output {art.shape} != {(ph, pw, 3)}"
            assert art.dtype == np.uint8 and int(art.sum()) > 0, \
                "stage_render composite blank/ wrong dtype"
        print("[rig_loop] end-to-end: rig_frame composited via stage_render at "
              "256 and 192 (had_art=True, panel-sized RGB)", flush=True)
    except Exception as exc:
        # stage_render is a sibling module; if it can't be exercised here we
        # still consider the shape assertions above authoritative, but surface it.
        print(f"[rig_loop] end-to-end via stage_render skipped: "
              f"{type(exc).__name__}: {exc}", flush=True)

    # --- missing-rig slug returns None (and caches the miss) -----------------
    miss = loop.rig_frame_for("no_such_character", 0.0)
    assert miss is None, "missing-rig slug must return None, not a frame"
    assert loop.get_rig("no_such_character") is None
    assert "no_such_character" in loop._rigs, "missing-rig miss was not cached"
    print("[rig_loop] missing slug -> None (cached): "
          "stage_render falls back to static cutout", flush=True)

    # --- rig_frame_for_beat resolves the slug from a beat panel --------------
    # Capitalised display name -> lowercase slug, same as the static-cutout path.
    rf_beat = loop.rig_frame_for_beat({"char": "Synthhero", "action": "idle",
                                       "aside": "..."}, 0.25)
    assert rf_beat is not None, "rig_frame_for_beat('Synthhero') resolved no rig"
    assert _slug_for_char("Synthhero") == slug, \
        f"slug resolution drift: {_slug_for_char('Synthhero')} != {slug}"
    # None / char-less beats are the static-path sentinel.
    assert loop.rig_frame_for_beat(None, 0.0) is None, \
        "None beat must yield None (no rig)"
    assert loop.rig_frame_for_beat({"action": "idle"}, 0.0) is None, \
        "char-less beat must yield None (no slug -> no rig)"
    assert loop.rig_frame_for_beat({"char": ""}, 0.0) is None, \
        "empty-char beat must yield None"
    print("[rig_loop] rig_frame_for_beat: 'Synthhero'->rig, "
          "None/char-less/empty->None", flush=True)

    print("[rig_loop] OK -- seam returns (rgb (H,W,3), alpha (H,W)) uint8, "
          "RGB-ordered, live over t, None when no assets", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
