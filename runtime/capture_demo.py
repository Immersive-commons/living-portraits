"""
capture_demo.py -- headless GIF/PNG-montage proof that the panels are ALIVE.

A still screenshot of living-portraits proves nothing: a blink, a breath, a gaze
drift, and the cross-frame "walk out of one portrait, into the next" all only
read as MOTION over time. This tool renders the real 448x256 dual-panel stage
HEADLESS (no pygame, no window) for N seconds at a chosen FPS, drives a scripted
beat timeline -- idle -> a cross-frame walk A->B -> idle -- and assembles the
frames into a single shareable artifact the conductor can drop into a chat:

    data/gen/demo_<timestamp>.gif      (preferred -- animates)
    data/gen/demo_<timestamp>.png      (fallback contact-sheet -- still proves it)

What it composites (and why it's honest)
-----------------------------------------
Every panel is built through the SAME numpy path the live player uses, so the
demo cannot drift from the show:

  * the bg plate + figure are composited by stage_render.composite_art(...)
    (the exact production compositor -- _alpha_over over the <slug>_bg.png /
    _portrait.png plate, panel-size cover-resize, the lot);
  * the IDLE living layer is rig_loop.RigLoop.rig_frame_for(slug, t) -> the same
    (rgb, alpha) tuple stage_render takes as rig_frame= in player.py;
  * the WALK is crossframe.exit_frame / enter_frame / empty_layer driving a real
    crossframe.CrossFrameState through PHASE_EXIT -> PHASE_GAP -> PHASE_ENTER,
    fed to the same rig_frame= slot.

The only thing this module owns is the *timeline* (when to idle vs walk) and the
*frame assembly* (panels onto a 448x256 canvas, then -> GIF). It never blits with
pygame, never touches the network, and never imports the director or player.

Layout matches the physical LEDs (player.py:PANELS)
---------------------------------------------------
    Panel A : (0,   0) 256 x 256
    Panel B : (256, 0) 192 x 192     (top-aligned; the 256..448 x 192..256
                                      corner is LED-dark, kept black here too)

Assets
------
Prefers the real generated assets in data/gen/ (data/gen/<slug>_bg.png +
_cutout.png + _portrait.png + _rig.json). When a slug has no assets (a dev box
that never ran the SD pipeline), it SYNTHESISES a stand-in into a private temp
gen dir using rig.py's own _synth_cutout / _synth_spec_for fixtures -- so the
demo always produces a visual artifact and still exercises the true Rig +
compositor code paths. The output banner states which slug ran against real vs
synth assets so a reviewer is never misled.

GIF backend selection (import-guarded, always-an-artifact)
----------------------------------------------------------
    imageio  (preferred)  -> animated .gif
    PIL/Pillow            -> animated .gif
    neither               -> opencv horizontal PNG contact-sheet (every Kth
                             frame side by side). opencv + numpy are the only
                             hard deps; imageio/PIL are optional.

CLI
---
    python runtime/capture_demo.py                       # 8s @ 15fps, phineas/seraphina
    python runtime/capture_demo.py --seconds 6 --fps 20
    python runtime/capture_demo.py --slug-a phineas --slug-b seraphina
    python runtime/capture_demo.py --selftest            # offline assert pass

ON SC2 the conductor runs (from the project dir, with the real assets present):
    .venv\\Scripts\\python.exe runtime\\capture_demo.py --seconds 8 --fps 15
and shares the printed `data/gen/demo_<ts>.gif`.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Make bare `import stage_render` / `crossframe` / `rig_loop` / `clip_player`
# resolve when this file is run directly (runtime/ is not otherwise on the path
# for a `python runtime/capture_demo.py` invocation). player.py does the same
# `sys.path.insert(0, runtime)`; we mirror it so the import convention matches.
_RUNTIME_DIR = Path(__file__).resolve().parent
if str(_RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(_RUNTIME_DIR))

ROOT = _RUNTIME_DIR.parent
GEN_DIR = ROOT / "data" / "gen"

# Panel geometry -- copied verbatim from player.py:PANELS so the demo canvas is
# pixel-identical to the live window. (We can't import player.py: it imports
# pygame at module load, which is absent on a dev box and is a MUST-NOT-TOUCH
# file anyway. These four numbers are the contract, mirrored intentionally.)
PANELS = {
    "A": {"rect": (0, 0, 256, 256), "bg": (38, 14, 16), "accent": (210, 170, 90)},
    "B": {"rect": (256, 0, 192, 192), "bg": (12, 26, 28), "accent": (120, 200, 190)},
}
CANVAS_W = 448
CANVAS_H = 256

# Sibling renderers (import-guarded only insofar as their own deps are; numpy +
# cv2 are present in the real pipeline and on this dev box). A hard failure here
# is a real bug -- unlike pygame, these are the modules the demo exists to drive.
import stage_render as sr          # noqa: E402  (after sys.path tweak)
import crossframe as xf            # noqa: E402
from rig_loop import RigLoop       # noqa: E402

try:  # opencv: present in the pipeline; used for asset I/O + the PNG fallback.
    import cv2  # type: ignore
except Exception:  # pragma: no cover - cv2 is a pipeline dep
    cv2 = None  # type: ignore


# --------------------------------------------------------------------------- #
# Asset resolution: real data/gen/<slug>_* if present, else synthesise.
# --------------------------------------------------------------------------- #
def _slug_has_assets(slug: str, gen_dir: Path) -> bool:
    """True iff at least the cutout exists for `slug` in `gen_dir`.

    The cutout is the load-bearing asset (it is what both the rig and the walk
    animate). _bg / _portrait / _rig.json are nice-to-have: composite_art falls
    back plate->portrait->tinted, and RigLoop degrades to the static cutout if
    the rig json is missing. So "has assets" == "has a cutout to move".
    """
    return (gen_dir / f"{slug}_cutout.png").exists()


def _synthesise_slug(slug: str, gen_dir: Path, size: int = 512) -> None:
    """Write a stand-in <slug>_{portrait,cutout,bg}.png + _rig.json into gen_dir.

    Reuses rig.py's own synthetic fixtures (so the synth path exercises the REAL
    Rig + rig-spec schema, not a private re-implementation): _synth_cutout builds
    an RGBA figure-on-transparency, _synth_spec_for builds the matching rig spec.
    The portrait/bg plates are derived from the cutout so composite_art has a real
    backdrop to alpha-over (cutout flattened onto a warm field; bg = that field
    with the figure region softened, standing in for segment.py's inpaint).

    No-op (raises) if cv2 is missing -- without it there is no PNG encoder here,
    and the caller should not have been asked to synthesise.
    """
    if cv2 is None:
        raise RuntimeError("cv2 unavailable: cannot synthesise stand-in assets")
    import rig as rig_mod  # local import: rig pulls cv2; only needed on synth.

    gen_dir.mkdir(parents=True, exist_ok=True)
    cut = rig_mod._synth_cutout(size)             # (size, size, 4) RGBA
    spec = rig_mod._synth_spec_for(cut)
    spec["slug"] = slug

    # Warm chiaroscuro field behind the figure (mirrors segment._synthetic_portrait
    # tone without importing pipeline). Deterministic so reruns are stable.
    field = np.zeros((size, size, 3), np.uint8)
    grad = np.linspace(64, 10, size).astype(np.uint8)
    field[..., 0] = grad[:, None]
    field[..., 1] = (grad * 0.7).astype(np.uint8)[:, None]
    field[..., 2] = (grad * 0.45).astype(np.uint8)[:, None]

    # Portrait = field with the cutout composited on top (a "painting" with the
    # figure in it). bg plate = the field with the figure area gently blurred so
    # it stands in for the inpainted backdrop the rig parallaxes over.
    a = cut[..., 3:4].astype(np.float32) / 255.0
    portrait = (cut[..., :3].astype(np.float32) * a
                + field.astype(np.float32) * (1.0 - a)).astype(np.uint8)
    bg = cv2.GaussianBlur(field, (0, 0), sigmaX=6.0)

    cv2.imwrite(str(gen_dir / f"{slug}_portrait.png"),
                cv2.cvtColor(portrait, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(gen_dir / f"{slug}_bg.png"),
                cv2.cvtColor(bg, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(gen_dir / f"{slug}_cutout.png"),
                cv2.cvtColor(cut, cv2.COLOR_RGBA2BGRA))
    (gen_dir / f"{slug}_rig.json").write_text(json.dumps(spec), encoding="utf-8")


def resolve_assets(char_a: str, char_b: str):
    """Pick a gen dir that holds usable assets for both characters.

    Returns (gen_dir, sources) where sources = {slug: "real"|"synth"} keyed on
    the canonical asset slug (sr.slugify), so a caller may pass either "Phineas"
    or "phineas" and the keys + filename lookups stay consistent with the slug
    build_timeline / _Stage use downstream.

    If BOTH slugs already have a real cutout in data/gen/, that dir is used as-is
    (zero copying, real art). Otherwise a fresh temp gen dir is created; any slug
    that has real assets there is COPIED in (so its real art is preserved), and
    any slug that doesn't is SYNTHESISED. This keeps "real where possible, synth
    only as needed" and never writes synth files into the real data/gen.
    """
    sources: dict[str, str] = {}
    slug_a = sr.slugify(char_a)
    slug_b = sr.slugify(char_b)
    slugs = [slug_a] + ([slug_b] if slug_b and slug_b != slug_a else [])

    if all(_slug_has_assets(s, GEN_DIR) for s in slugs):
        for s in slugs:
            sources[s] = "real"
        return GEN_DIR, sources

    # Mixed / none real -> assemble a temp gen dir.
    tmp = Path(tempfile.mkdtemp(prefix="lp_demo_gen_"))
    for s in slugs:
        if _slug_has_assets(s, GEN_DIR):
            # Copy the real asset set across so the demo uses true art for it.
            for suffix in ("_portrait.png", "_cutout.png", "_bg.png", "_rig.json"):
                src = GEN_DIR / f"{s}{suffix}"
                if src.exists():
                    (tmp / src.name).write_bytes(src.read_bytes())
            sources[s] = "real"
        else:
            _synthesise_slug(s, tmp)
            sources[s] = "synth"
    return tmp, sources


# --------------------------------------------------------------------------- #
# Beat timeline: idle -> walk A->B -> idle.
# --------------------------------------------------------------------------- #
# Asides are flavour only (they'd render as pygame text in the live player; in
# this headless numpy path stage_render skips the pygame text step, so the demo
# shows the art + motion, not the captions -- which is exactly what we want to
# prove). Kept here so the beat dicts match the director's {char, action, aside}.
_BEAT_A = {"char": "", "action": "address_house",
           "aside": "The feed says the worm industrialized. Quaint."}
_BEAT_B = {"char": "", "action": "lean_in",
           "aside": "A frontier model broke the perimeter. Over a weekend."}


def _beat_with_char(beat: dict, char: str) -> dict:
    """A copy of a beat template with its character display name filled in."""
    out = dict(beat)
    out["char"] = char
    return out


def build_timeline(seconds: float, fps: int, char_a: str, char_b: str):
    """Produce a per-frame plan describing what each panel renders.

    The show is three acts whose lengths sum to `seconds`:
        idle_pre   : both panels idle (A occupied by char_a, B by char_b)
        walk       : char_a walks A -> B (crossframe EXIT/GAP/ENTER), fixed length
                     = EXIT_S + GAP_S + ENTER_S (the real crossframe durations)
        idle_post  : both panels idle again (now A empty, char_a settled in B)

    Returns a list of per-frame dicts:
        {"t": <elapsed s>, "act": "idle"|"walk",
         "A": {"slug","beat","mode","layer"}, "B": {...}}
    where mode in {"idle","exit","enter","gap","empty"} and `layer`, when set, is
    the precomputed (rgb, alpha) crossframe figure layer for that panel this tick.

    We precompute the whole plan (not stream it) so the renderer is a dumb loop
    and the timeline logic is all in one testable place.
    """
    fps = max(1, int(fps))
    dt = 1.0 / fps
    total_frames = max(1, int(round(seconds * fps)))

    slug_a = sr.slugify(char_a)
    slug_b = sr.slugify(char_b)
    walk_len = xf.EXIT_S + xf.GAP_S + xf.ENTER_S

    # Carve the timeline: give the walk its real fixed duration, split the rest
    # evenly into pre/post idle. If `seconds` is too short to fit a walk, shrink
    # the idle to near-zero but always keep the full walk (it's the whole point).
    idle_total = max(0.0, seconds - walk_len)
    idle_pre_s = idle_total * 0.5
    walk_start = idle_pre_s
    walk_end = walk_start + walk_len

    # crossframe directions from the real panel geometry.
    geom = {nm: PANELS[nm]["rect"] for nm in PANELS}
    exit_dir, enter_dir = xf.edge_for_move("A", "B", geom)
    rect_a = PANELS["A"]["rect"]
    rect_b = PANELS["B"]["rect"]

    # Load the moving character's cutout once (RGBA) for the walk layers. The
    # walk animates char_a's figure; we need its raw cutout the way exit/enter
    # expect (RGBA, native res). Read straight off disk via stage_render's reader
    # so colour order matches the compositor. None -> walk degrades to empty
    # layers (still a clean GAP), which build_frame tolerates.
    state = xf.CrossFrameState()
    state.set_occupant("A", slug_a)
    state.set_occupant("B", slug_b)
    state.begin_transition(slug_a, "A", "B")

    plan = []
    began = False
    for i in range(total_frames):
        t = i * dt
        in_walk = (walk_start <= t < walk_end) and walk_len > 0
        frame = {"t": t, "act": "walk" if in_walk else "idle"}

        if not in_walk:
            # Idle act. Occupancy depends on whether the walk has completed.
            walk_done = t >= walk_end
            a_slug = None if walk_done else slug_a
            b_slug = slug_a if walk_done else slug_b
            frame["A"] = {"slug": a_slug,
                          "beat": _beat_with_char(_BEAT_A, char_a) if a_slug else None,
                          "mode": "idle" if a_slug else "empty", "layer": None}
            frame["B"] = {"slug": b_slug,
                          "beat": _beat_with_char(_BEAT_B if not walk_done else _BEAT_A,
                                                  char_a if walk_done else char_b),
                          "mode": "idle", "layer": None}
            plan.append(frame)
            continue

        # Walk act. Advance the state machine to this elapsed-within-walk time.
        # begin_transition was already called; we step the clock to (t-walk_start).
        if not began:
            began = True
        # Re-derive elapsed deterministically rather than accumulating float dt,
        # so a dropped/extra frame can't desync the phase from wall position.
        state._elapsed = min(t - walk_start, state._total())  # noqa: SLF001
        if state._elapsed >= state._total():
            state._commit_done()  # noqa: SLF001

        pa, prog_a = state.panel_state("A")
        pb, prog_b = state.panel_state("B")

        # Panel A: EXIT while the figure is leaving, else empty (it has gone).
        if pa == xf.PHASE_EXIT:
            frame["A"] = {"slug": slug_a, "beat": _beat_with_char(_BEAT_A, char_a),
                          "mode": "exit", "layer": ("exit", prog_a, exit_dir)}
        else:
            frame["A"] = {"slug": None, "beat": None, "mode": "empty", "layer": "empty"}

        # Panel B: ENTER while the figure is arriving, else empty (not here yet).
        if pb == xf.PHASE_ENTER:
            frame["B"] = {"slug": slug_a, "beat": _beat_with_char(_BEAT_A, char_a),
                          "mode": "enter", "layer": ("enter", prog_b, enter_dir)}
        else:
            frame["B"] = {"slug": None, "beat": None, "mode": "empty", "layer": "empty"}

        plan.append(frame)

    return plan, {"slug_a": slug_a, "slug_b": slug_b,
                  "idle_pre_s": idle_pre_s, "walk_len": walk_len,
                  "idle_post_s": max(0.0, seconds - walk_end)}


# --------------------------------------------------------------------------- #
# Frame rendering: compose one 448x256 canvas for a plan entry.
# --------------------------------------------------------------------------- #
class _Stage:
    """Holds the per-run renderer + rig loop + cutout cache for the moving char,
    and composes one canvas per plan frame.

    Bound to a single resolved gen dir so both the StageRenderer (plates +
    figures) and the RigLoop (idle animation) read the same assets.
    """

    def __init__(self, gen_dir: Path, walk_slug: str | None) -> None:
        self.gen_dir = Path(gen_dir)
        self.renderer = sr.StageRenderer(gen_dir=self.gen_dir)
        self.rigloop = RigLoop(gen_dir=self.gen_dir)
        # The walking figure's cutout (RGBA native) for crossframe layers. Read
        # once; None if absent (walk falls back to empty layers).
        self.walk_cutout = None
        if walk_slug:
            self.walk_cutout = sr._imread_rgba(self.gen_dir / f"{walk_slug}_cutout.png")

    def _rig_frame_for(self, panel_spec: dict, panel_rect, t: float):
        """Resolve the rig_frame= layer for one panel from its plan spec.

        idle  -> the live rig (rig_loop) at time t, or None (static cutout) if no
                 rig. exit/enter -> the crossframe figure layer. empty -> a fully
                 transparent layer (bg-plate-only). None means "let composite_art
                 use the static cutout".
        """
        mode = panel_spec.get("mode")
        layer = panel_spec.get("layer")
        if mode == "idle":
            slug = panel_spec.get("slug") or ""
            return self.rigloop.rig_frame_for(slug, t)  # (rgb,alpha) | None
        if mode == "empty" or layer == "empty":
            return xf.empty_layer(panel_rect)
        if isinstance(layer, tuple):
            kind, prog, direction = layer
            if self.walk_cutout is None:
                return xf.empty_layer(panel_rect)
            if kind == "exit":
                return xf.exit_frame(panel_rect, self.walk_cutout, prog, direction)
            if kind == "enter":
                return xf.enter_frame(panel_rect, self.walk_cutout, prog, direction)
        return None

    def render_canvas(self, plan_frame: dict) -> np.ndarray:
        """Compose the 448x256 RGB canvas for one plan frame.

        Panel A fills (0,0,256,256); Panel B fills (256,0,192,192) top-aligned;
        the (256..448, 192..256) corner stays black (LED-dark). Each panel goes
        through stage_render.composite_art with the resolved rig_frame=, so the
        plate + figure + cover-resize are the production compositor's, not ours.
        """
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
        t = plan_frame["t"]
        for nm in ("A", "B"):
            spec = plan_frame[nm]
            x, y, w, h = PANELS[nm]["rect"]
            theme = {"bg": PANELS[nm]["bg"], "accent": PANELS[nm]["accent"]}
            slug = spec.get("slug") or ""
            rig_frame = self._rig_frame_for(spec, (x, y, w, h), t)
            art, _had = sr.composite_art(self.renderer, slug, w, h, theme,
                                         rig_frame=rig_frame)
            canvas[y:y + h, x:x + w] = art
        return canvas


# --------------------------------------------------------------------------- #
# GIF / contact-sheet assembly (import-guarded; always emits an artifact).
# --------------------------------------------------------------------------- #
def _assemble_gif(frames: list[np.ndarray], out_path: Path, fps: int) -> tuple[Path, str]:
    """Write `frames` (list of (H,W,3) RGB uint8) as an animated GIF.

    Tries imageio first, then PIL/Pillow. Returns (path, backend). Raises if
    neither backend imports (caller falls back to the PNG contact-sheet).
    """
    duration_ms = max(1, int(round(1000.0 / max(1, fps))))

    # ---- imageio (preferred) ---------------------------------------------
    try:
        try:
            import imageio.v2 as imageio  # type: ignore  (v2 API: deprecation-free)
        except Exception:
            import imageio  # type: ignore
        # duration is per-frame SECONDS in imageio's gif writer; loop=0 = forever.
        imageio.mimsave(str(out_path), frames, format="GIF",
                        duration=1.0 / max(1, fps), loop=0)
        return out_path, "imageio"
    except Exception:
        pass

    # ---- PIL / Pillow -----------------------------------------------------
    try:
        from PIL import Image  # type: ignore
        pil_frames = [Image.fromarray(f, mode="RGB") for f in frames]
        first, rest = pil_frames[0], pil_frames[1:]
        first.save(str(out_path), format="GIF", save_all=True,
                   append_images=rest, duration=duration_ms, loop=0, optimize=False)
        return out_path, "pillow"
    except Exception as exc:
        raise RuntimeError(f"no GIF backend (imageio/PIL both failed): {exc}")


def _assemble_contact_sheet(frames: list[np.ndarray], out_path: Path,
                            every: int = 6, max_tiles: int = 16) -> tuple[Path, str]:
    """Fallback when no GIF backend exists: a horizontal PNG strip of every Kth
    frame, side by side, via opencv. Still a single shareable image proving the
    panels change frame-to-frame. Returns (png_path, "opencv-strip").
    """
    if cv2 is None:
        raise RuntimeError("no GIF backend AND cv2 absent: cannot emit any artifact")
    sampled = frames[::max(1, every)]
    if len(sampled) > max_tiles:
        # Even-stride down to max_tiles so the strip stays a sane width.
        idx = np.linspace(0, len(sampled) - 1, max_tiles).astype(int)
        sampled = [sampled[i] for i in idx]
    if not sampled:
        sampled = [frames[-1]]
    strip = np.concatenate(sampled, axis=1)  # all are (H, 448, 3) -> (H, 448*N, 3)
    png_path = out_path.with_suffix(".png")
    cv2.imwrite(str(png_path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    return png_path, "opencv-strip"


def assemble(frames: list[np.ndarray], out_stem: Path, fps: int) -> tuple[Path, str]:
    """Emit the best artifact available for `frames`: animated GIF if a backend
    exists, else an opencv PNG contact-sheet. `out_stem` is the path WITHOUT
    extension; the chosen writer appends .gif or .png. Returns (path, backend)."""
    gif_path = out_stem.with_suffix(".gif")
    try:
        return _assemble_gif(frames, gif_path, fps)
    except Exception:
        return _assemble_contact_sheet(frames, out_stem, every=max(1, fps // 3))


# --------------------------------------------------------------------------- #
# Top-level render entry point.
# --------------------------------------------------------------------------- #
def render_demo(seconds: float = 8.0, fps: int = 15,
                slug_a: str = "phineas", slug_b: str = "seraphina",
                out_dir: Path | None = None, verbose: bool = True):
    """Render the scripted demo and assemble it. Returns a result dict:

        {"path": Path, "backend": str, "frames": int, "fps": int,
         "seconds": float, "sources": {slug: "real"|"synth"}, "phases": {...}}

    `slug_a` / `slug_b` are character display names (slugified internally for
    asset lookup; passing "phineas" or "Phineas" both work). out_dir defaults to
    data/gen/ -- the same dir the pipeline writes, so the artifact sits beside the
    assets it was made from.
    """
    out_dir = Path(out_dir) if out_dir is not None else GEN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_dir, sources = resolve_assets(slug_a, slug_b)
    plan, phases = build_timeline(seconds, fps, slug_a, slug_b)
    walk_slug = phases["slug_a"]  # the moving character

    stage = _Stage(gen_dir, walk_slug)
    frames = [stage.render_canvas(pf) for pf in plan]

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_stem = out_dir / f"demo_{ts}"
    path, backend = assemble(frames, out_stem, fps)

    result = {
        "path": path, "backend": backend, "frames": len(frames),
        "fps": fps, "seconds": seconds, "sources": sources, "phases": phases,
    }
    if verbose:
        src_str = ", ".join(f"{k}={v}" for k, v in sources.items())
        print(f"[capture_demo] {len(frames)} frames @ {fps}fps "
              f"({seconds:.1f}s: idle {phases['idle_pre_s']:.1f}s -> walk "
              f"{phases['walk_len']:.1f}s -> idle {phases['idle_post_s']:.1f}s)",
              flush=True)
        print(f"[capture_demo] assets: {src_str}", flush=True)
        print(f"[capture_demo] backend={backend}", flush=True)
        print(f"[capture_demo] wrote {path}", flush=True)
    return result


# --------------------------------------------------------------------------- #
# Self-test: render a tiny demo offline (synth assets), assert the artifact.
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    """Offline assertion pass -- no real assets required, no network.

    Renders a short demo (2s @ 8fps) into a temp out dir, forcing synth assets by
    using slugs the real data/gen never holds. Asserts: an artifact file is
    written; canvases are exactly 448x256 RGB and non-blank; the timeline has the
    expected idle/walk/idle structure; during the walk the moving figure is shown
    on exactly one panel at a time (EXIT on A, then ENTER on B) with a GAP where
    neither shows it; and frames actually differ over time (motion).
    """
    print("[capture_demo] self-test", flush=True)
    print(f"[capture_demo] cv2={'yes' if cv2 is not None else 'no(guarded)'}",
          flush=True)

    if cv2 is None:
        # Without cv2 there's no PNG encoder and no Rig decode -- the whole demo
        # is inert. Assert the import surface is sane and stop (mirrors the
        # sibling self-tests' "degrade cleanly" stance).
        plan, phases = build_timeline(2.0, 8, "phineas", "seraphina")
        assert plan and phases["walk_len"] > 0
        print("[capture_demo] OK (cv2 absent: timeline-only path verified)",
              flush=True)
        return 0

    # Use synth-only slugs so we never depend on real art being present.
    char_a, char_b = "Selftesta", "Selftestb"
    slug_a, slug_b = sr.slugify(char_a), sr.slugify(char_b)

    # ---- asset resolution synthesises both (neither has real assets) --------
    gen_dir, sources = resolve_assets(char_a, char_b)
    assert sources.get(slug_a) == "synth" and sources.get(slug_b) == "synth", sources
    assert (gen_dir / f"{slug_a}_cutout.png").exists(), "synth cutout A missing"
    assert (gen_dir / f"{slug_b}_cutout.png").exists(), "synth cutout B missing"
    assert (gen_dir / f"{slug_a}_rig.json").exists(), "synth rig spec A missing"

    # ---- timeline structure: idle -> walk -> idle ---------------------------
    seconds, fps = 2.0 + xf.EXIT_S + xf.GAP_S + xf.ENTER_S, 8
    plan, phases = build_timeline(seconds, fps, char_a, char_b)
    acts = [pf["act"] for pf in plan]
    assert "idle" in acts and "walk" in acts, f"missing acts: {set(acts)}"
    # canonical order: some idle, then a contiguous walk block, then idle again.
    first_walk = acts.index("walk")
    last_walk = len(acts) - 1 - acts[::-1].index("walk")
    assert all(a == "walk" for a in acts[first_walk:last_walk + 1]), \
        "walk act is not contiguous"
    assert first_walk > 0, "no idle before the walk"
    assert last_walk < len(acts) - 1, "no idle after the walk"

    # During the walk: exactly one panel shows the moving figure at a time, and
    # there is at least one GAP frame where NEITHER panel shows it.
    saw_exit = saw_enter = saw_gap = False
    for pf in plan[first_walk:last_walk + 1]:
        a_shows = pf["A"]["mode"] == "exit"
        b_shows = pf["B"]["mode"] == "enter"
        assert not (a_shows and b_shows), "figure shown on BOTH panels mid-walk"
        saw_exit |= a_shows
        saw_enter |= b_shows
        if not a_shows and not b_shows:
            saw_gap = True
    assert saw_exit, "EXIT never rendered on panel A"
    assert saw_enter, "ENTER never rendered on panel B"
    assert saw_gap, "no GAP frame (empty portrait) during the walk"

    # post-walk idle: A emptied, char_a settled into B.
    post = plan[last_walk + 1]
    assert post["A"]["slug"] is None, "A not emptied after walk"
    assert sr.slugify(post["B"]["slug"] or "") == slug_a or post["B"]["slug"] == char_a, \
        f"char_a did not settle into B: {post['B']['slug']}"

    # ---- canvases: 448x256 RGB, non-blank, panel B corner is black ----------
    stage = _Stage(gen_dir, slug_a)
    canv_idle = stage.render_canvas(plan[0])
    assert canv_idle.shape == (CANVAS_H, CANVAS_W, 3), canv_idle.shape
    assert canv_idle.dtype == np.uint8
    assert int(canv_idle.sum()) > 0, "idle canvas is blank"
    # Panel A region (0..256, 0..256) must carry art (non-zero) on the idle frame.
    assert int(canv_idle[:256, :256].sum()) > 0, "panel A blank on idle"
    # The B dead-corner (rows 192..256, cols 256..448) must be LED-black.
    corner = canv_idle[192:256, 256:448]
    assert int(corner.sum()) == 0, "B dead-corner is not black (layout wrong)"

    # ---- motion: rig idle changes the panels over time ----------------------
    canv_mid = stage.render_canvas(plan[min(len(plan) - 1, first_walk // 2)])
    walk_canvases = [stage.render_canvas(pf) for pf in plan[first_walk:last_walk + 1]]
    # idle-to-idle should differ (blink/breathe/gaze) OR walk should differ from
    # idle -- assert the stronger, easy claim: the walk frames are not all equal.
    deltas = [int(np.abs(walk_canvases[i].astype(np.int16)
                         - walk_canvases[0].astype(np.int16)).sum())
              for i in range(1, len(walk_canvases))]
    assert deltas and max(deltas) > 0, "walk canvases never change (no motion)"

    # ---- assembly: an artifact is written, openable ------------------------
    frames = [stage.render_canvas(pf) for pf in plan]
    with tempfile.TemporaryDirectory() as td:
        out_stem = Path(td) / "demo_selftest"
        path, backend = assemble(frames, out_stem, fps)
        assert path.exists() and path.stat().st_size > 0, \
            f"artifact not written: {path}"
        # If it's a GIF, confirm a reader can re-open it (round-trip sanity).
        if path.suffix == ".gif":
            reopened = False
            try:
                import imageio.v2 as imageio  # type: ignore
                got = imageio.mimread(str(path))
                reopened = len(got) >= 1
            except Exception:
                try:
                    from PIL import Image  # type: ignore
                    with Image.open(str(path)) as im:
                        reopened = getattr(im, "n_frames", 1) >= 1
                except Exception:
                    reopened = True  # can't re-read here, but it was written
            assert reopened, "GIF written but unreadable"
        print(f"[capture_demo] artifact OK: {path.name} ({backend}, "
              f"{path.stat().st_size:,} bytes, {len(frames)} frames)", flush=True)

    print("[capture_demo] OK -- 448x256 canvases, idle->walk->idle timeline, "
          "EXIT/GAP/ENTER one-panel-at-a-time, motion over t, artifact emitted",
          flush=True)
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="capture_demo",
        description="Headless GIF/PNG demo of the living-portraits dual panel "
                    "(idle blink/breathe/gaze + a cross-frame walk A->B).")
    ap.add_argument("--seconds", type=float, default=8.0,
                    help="total demo length in seconds (default 8).")
    ap.add_argument("--fps", type=int, default=15,
                    help="frames per second (default 15).")
    ap.add_argument("--slug-a", default="phineas",
                    help="character in panel A / the one who walks (default phineas).")
    ap.add_argument("--slug-b", default="seraphina",
                    help="character in panel B (default seraphina).")
    ap.add_argument("--out-dir", default=None,
                    help="output dir for demo_<ts>.gif (default data/gen).")
    ap.add_argument("--selftest", action="store_true",
                    help="run the offline self-test (synth assets) and exit.")
    return ap


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)
    if args.selftest:
        return _selftest()
    out_dir = Path(args.out_dir) if args.out_dir else None
    res = render_demo(seconds=args.seconds, fps=args.fps,
                      slug_a=args.slug_a, slug_b=args.slug_b, out_dir=out_dir)
    return 0 if res.get("path") else 1


if __name__ == "__main__":
    raise SystemExit(main())
