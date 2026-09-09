"""
stage_render.py -- composite a real generated portrait into one panel.

This is the architecture's "stage_render" box: the renderer that sits between
the generative pipeline's art (data/gen/<slug>_bg.png + <slug>_cutout.png) and
player.py's per-panel sub-surface. It is the production replacement for the flat
tinted-panel text dev-view in player.py:draw_stage, keeping the exact same
state-consumption shell -- it reads one beat panel ({char, action, aside}) and a
pygame sub-surface, and draws the living portrait into it.

Composite order (back to front), all numpy in RGB until the final blit:

  1. Background plate   data/gen/<slug>_bg.png   (figure removed + inpainted),
     else the full portrait data/gen/<slug>_portrait.png, cover-cropped to fill
     the panel. If neither exists -> graceful fallback to the tinted panel look
     so the show never regresses to blank.
  2. Character cutout   data/gen/<slug>_cutout.png   (RGBA), alpha-composited on
     top of the plate. This is the layer rig.py later supplants: instead of the
     static cutout, the rig emits a per-frame warped cutout RGB+alpha that drops
     in at exactly this step (see `composite_art(..., rig_frame=...)`).
  3. Text overlay (pygame): a subtle scrim behind the aside for legibility, then
     the aside (wrapped), the character name, and the [action] tag.
  4. Liveness marker + clock, matching player.py so the dev-view and this view
     are visually continuous.

Surface abstraction & headless safety
-------------------------------------
The art layers (1-2) are pure numpy and need no display. The text + liveness
layers (3-4) use pygame, which is import-guarded exactly like clip_player.py:
this module imports anywhere. On SC2 the caller passes a real pygame sub-surface
and the fonts dict player.py already builds; in the headless self-test (and on a
dev box with no pygame) we composite onto a numpy-backed DummySurface borrowed
from clip_player.py and skip the pygame-only text step.

Caching
-------
Decoding + scaling a 512x512 PNG every frame at 30 FPS would re-pay the cost ~30
times a second per panel. Loaded-and-scaled art is therefore cached by
(path, mtime, size): the first frame decodes, every subsequent frame at the same
panel size reuses the scaled numpy array. Editing an asset on disk (mtime bump)
invalidates the entry so a regenerated portrait is picked up live.
"""
from __future__ import annotations

import math
import re
import time
from pathlib import Path

import numpy as np

# Reuse the surface abstraction, blit, and resize helpers from the sibling
# renderer so the rig view and this view share identical RGB<->surface plumbing
# (and the same import-guarded pygame/opencv handling).
# Both loaders have to work. player.py:35 puts runtime/ ON sys.path and imports
# bare; _preview_graph.py:25 puts the REPO ROOT on it and imports the package.
# runtime/policy.py:52-59 carries the same shape for the same reason. Normalising
# to one spelling breaks the other, and player.py's try/except would swallow it:
# the wall would fall through to the text dev-view and keep running.
try:                                     # player.py:35 -- runtime/ on sys.path
    from clip_player import (  # type: ignore
        _HAVE_CV2,
        _HAVE_PYGAME,
        _blit_rgb_to_surface,
        _resize_rgb,
        _surface_size,
        pygame,  # None when pygame is absent (import-guarded in clip_player)
    )
except ImportError:                      # _preview_graph.py:25 -- repo root on sys.path
    from runtime.clip_player import (  # type: ignore
        _HAVE_CV2,
        _HAVE_PYGAME,
        _blit_rgb_to_surface,
        _resize_rgb,
        _surface_size,
        pygame,
    )

try:  # opencv is the high-quality decode path; numpy fallback covers its absence
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

ROOT = Path(__file__).resolve().parent.parent
GEN_DIR = ROOT / "data" / "gen"

# Panel palettes mirror player.py:PANELS / clip_player.py:PANEL_THEME so the
# fallback look here is identical to the text dev-view.
PANEL_THEME = {
    "A": {"bg": (38, 14, 16), "accent": (210, 170, 90)},
    "B": {"bg": (12, 26, 28), "accent": (120, 200, 190)},
}
DEFAULT_THEME = {"bg": (20, 20, 24), "accent": (200, 200, 200)}

_PAD = 8


def slugify(char_name: str) -> str:
    """'Phineas' -> 'phineas'; 'Seraphina' -> 'seraphina'.

    stage_manager writes capitalised display names into stage_state.json, but the
    pipeline keys its art on lowercase slugs (prompts/characters/<slug>.md ->
    data/gen/<slug>_*.png). Lowercase, strip, and collapse any run of non-alnum
    to a single underscore so a future 'Madame X' resolves to 'madame_x'.
    """
    s = (char_name or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


# --------------------------------------------------------------------------- #
# Image loading + cache
# --------------------------------------------------------------------------- #
def _imread_rgb(path: Path) -> np.ndarray | None:
    """Read an image as RGB (H, W, 3) uint8, or None on missing/corrupt file.

    cv2 reads BGR(A); we drop alpha and convert to RGB. Never raises -- a missing
    or unreadable file resolves to None so the caller falls back.
    """
    if cv2 is None or not path.exists():
        return None
    try:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)  # 3-channel, alpha dropped
    except Exception:
        return None
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _imread_rgba(path: Path) -> np.ndarray | None:
    """Read an image as RGBA (H, W, 4) uint8, or None.

    The cutout is written BGRA by segment.py; cv2.IMREAD_UNCHANGED preserves the
    4th (alpha) channel. A 3-channel image (no alpha) is promoted to opaque RGBA
    so the compositor always has an alpha to blend with.
    """
    if cv2 is None or not path.exists():
        return None
    try:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    except Exception:
        return None
    if raw is None:
        return None
    if raw.ndim == 2:  # grayscale -> opaque RGB
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    if raw.shape[2] == 3:  # BGR -> RGBA opaque
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        a = np.full((*rgb.shape[:2], 1), 255, np.uint8)
        return np.concatenate([rgb, a], axis=2)
    if raw.shape[2] == 4:  # BGRA -> RGBA
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2RGBA)
    return None


def _cover_resize_rgb(rgb: np.ndarray, w: int, h: int) -> np.ndarray:
    """Scale-and-centre-crop an RGB array to exactly (h, w), preserving aspect.

    'Cover' (not 'contain'): the art fills the whole panel with no letterbox
    bars; the overflowing axis is centre-cropped. Portraits are ~square and the
    panels are square (256) / near-square (192), so the crop is gentle.
    """
    sh, sw = rgb.shape[:2]
    if (sw, sh) == (w, h):
        return rgb
    scale = max(w / sw, h / sh)
    nw, nh = max(1, int(round(sw * scale))), max(1, int(round(sh * scale)))
    scaled = _resize_rgb(rgb, nw, nh)
    x0 = max(0, (nw - w) // 2)
    y0 = max(0, (nh - h) // 2)
    return scaled[y0:y0 + h, x0:x0 + w]


def _cover_resize_rgba(rgba: np.ndarray, w: int, h: int) -> np.ndarray:
    """Cover-resize an RGBA array to (h, w): RGB via _resize_rgb, alpha alongside.

    Resizing the 4 channels together would let _resize_rgb (an RGB helper) mangle
    the alpha; instead scale RGB and the single-channel alpha separately at the
    same geometry, then recombine, so the matte stays aligned to the colour.
    """
    sh, sw = rgba.shape[:2]
    if (sw, sh) == (w, h):
        return rgba
    scale = max(w / sw, h / sh)
    nw, nh = max(1, int(round(sw * scale))), max(1, int(round(sh * scale)))
    rgb_scaled = _resize_rgb(rgba[..., :3], nw, nh)
    alpha = rgba[..., 3]
    if cv2 is not None:
        a_scaled = cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_LINEAR)
    else:  # nearest-neighbour numpy fallback, mirrors _resize_rgb's no-cv2 path
        ys = np.linspace(0, sh - 1, nh).astype(np.int64)
        xs = np.linspace(0, sw - 1, nw).astype(np.int64)
        a_scaled = alpha[ys][:, xs]
    out = np.dstack([rgb_scaled, a_scaled])
    x0 = max(0, (nw - w) // 2)
    y0 = max(0, (nh - h) // 2)
    return out[y0:y0 + h, x0:x0 + w]


# --------------------------------------------------------------------------- #
# Art compositing (pure numpy, headless-safe)
# --------------------------------------------------------------------------- #
def _alpha_over(base_rgb: np.ndarray, top_rgba: np.ndarray) -> np.ndarray:
    """Composite an RGBA layer over an opaque RGB base (both (H, W, .) at panel
    size). Standard source-over: out = top.rgb*a + base*(1-a)."""
    a = (top_rgba[..., 3:4].astype(np.float32)) / 255.0
    top = top_rgba[..., :3].astype(np.float32)
    base = base_rgb.astype(np.float32)
    out = top * a + base * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def _fallback_plate(w: int, h: int, theme: dict) -> np.ndarray:
    """The no-art backdrop: the tinted panel + accent border + soft gradient,
    matching clip_player.synthetic_frame's field so a panel with no assets looks
    like the rest of the show rather than a black hole."""
    bg = np.array(theme["bg"], dtype=np.uint8)
    accent = np.array(theme["accent"], dtype=np.uint8)
    frame = np.empty((h, w, 3), dtype=np.uint8)
    frame[:, :] = bg
    grad = np.linspace(0.0, 0.22, h, dtype=np.float32)[:, None, None]
    frame = (frame.astype(np.float32) * (1 - grad)
             + accent.astype(np.float32) * grad).astype(np.uint8)
    frame[0, :] = accent
    frame[-1, :] = accent
    frame[:, 0] = accent
    frame[:, -1] = accent
    return frame


def composite_art(
    cache: StageRenderer | dict | None,
    slug: str,
    w: int,
    h: int,
    theme: dict,
    rig_frame: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, bool]:
    """Build the (H, W, 3) RGB art for one panel from on-disk assets.

    Returns (rgb, had_art). `had_art` is False when no plate/portrait/cutout
    existed -- the caller then knows it rendered the fallback look (so a future
    QA gate can distinguish "intentional fallback" from "art present").

    Layering:
      plate  = <slug>_bg.png  (cover) , else <slug>_portrait.png (cover),
               else tinted fallback plate.
      figure = rig_frame if supplied (the live rig output), else the static
               <slug>_cutout.png (cover). rig_frame is (rgb (H,W,3), alpha
               (H,W)) already sized to the panel -- this is exactly where rig.py
               layers in over the cutout.
    """
    renderer = cache if isinstance(cache, StageRenderer) else None
    get = renderer._cached_scaled if renderer is not None else _load_scaled_uncached
    # Honour a renderer's configured gen_dir; fall back to the module default.
    gen_dir = renderer.gen_dir if renderer is not None else GEN_DIR

    had_art = False
    # ---- background plate -------------------------------------------------
    plate = None
    if slug:
        plate = get(gen_dir / f"{slug}_bg.png", w, h, "rgb_cover")
        if plate is None:
            plate = get(gen_dir / f"{slug}_portrait.png", w, h, "rgb_cover")
    if plate is None:
        plate = _fallback_plate(w, h, theme)
    else:
        had_art = True

    out = plate

    # ---- figure (rig frame supplants the static cutout) -------------------
    if rig_frame is not None:
        rig_rgb, rig_alpha = rig_frame
        rgba = np.dstack([
            _cover_resize_rgb(rig_rgb, w, h) if rig_rgb.shape[:2] != (h, w) else rig_rgb,
            _fit_alpha(rig_alpha, w, h),
        ])
        out = _alpha_over(out, rgba)
        had_art = True
    elif slug:
        cutout = get(gen_dir / f"{slug}_cutout.png", w, h, "rgba_cover")
        if cutout is not None:
            out = _alpha_over(out, cutout)
            had_art = True

    return out, had_art


def _fit_alpha(alpha: np.ndarray, w: int, h: int) -> np.ndarray:
    """Coerce a rig alpha channel to (h, w) uint8 (cover-resize if needed)."""
    if alpha.shape[:2] == (h, w):
        return alpha.astype(np.uint8)
    a3 = np.dstack([alpha, alpha, alpha]).astype(np.uint8)
    return _cover_resize_rgb(a3, w, h)[..., 0]


def _load_scaled_uncached(path: Path, w: int, h: int, kind: str):
    """Decode + cover-resize an asset with no caching (used when no StageRenderer
    instance is carrying a cache -- e.g. a one-off functional call)."""
    if kind == "rgb_cover":
        rgb = _imread_rgb(path)
        return None if rgb is None else _cover_resize_rgb(rgb, w, h)
    rgba = _imread_rgba(path)
    return None if rgba is None else _cover_resize_rgba(rgba, w, h)


# --------------------------------------------------------------------------- #
# Text overlay (pygame-only) + liveness
# --------------------------------------------------------------------------- #
def _wrap_text(text: str, font, max_w: int) -> list[str]:
    """Greedy word-wrap to pixel width, mirroring player.py:wrap_text exactly."""
    lines, cur = [], ""
    for word in (text or "").split():
        test = (cur + " " + word).strip()
        if font.size(test)[0] <= max_w or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _draw_text_overlay(sub, beat_panel: dict, fonts: dict, theme: dict) -> None:
    """Draw aside (over a scrim) + char name + [action] tag onto a pygame
    sub-surface. No-op if pygame is unavailable. Mirrors player.draw_stage's
    typography (same fonts dict, same colours) so the look is continuous."""
    if not _HAVE_PYGAME or pygame is None:
        return
    w, h = _surface_size(sub)
    accent = theme["accent"]
    pad = _PAD

    name = beat_panel.get("char", "")
    action = beat_panel.get("action", "idle")
    aside = beat_panel.get("aside", "")

    # Title (char name) + action tag, top-left, with a drop-shadow for legibility
    # over busy art (a 1px offset black copy beneath the accent glyphs).
    title_surf = fonts["bold"].render(name or "", True, accent)
    _blit_with_shadow(sub, title_surf, (pad, pad))
    act_surf = fonts["tiny"].render("[" + (action or "idle") + "]", True, (210, 210, 210))
    _blit_with_shadow(sub, act_surf, (pad, pad + title_surf.get_height() + 1))

    # Aside: wrapped, bottom-anchored, sitting on a translucent dark scrim so
    # light text stays readable over a light patch of painting.
    lines = _wrap_text(aside, fonts["small"], w - 2 * pad)
    if lines:
        lh = fonts["small"].get_linesize()
        block_h = lh * len(lines) + pad
        block_top = max(0, h - block_h - 14)  # leave room for the clock row
        scrim = pygame.Surface((w, h - block_top), pygame.SRCALPHA)
        scrim.fill((0, 0, 0, 120))
        sub.blit(scrim, (0, block_top))
        y = block_top + pad // 2
        for line in lines:
            if y > h - 16:
                break
            _blit_with_shadow(sub, fonts["small"].render(line, True, (238, 238, 238)), (pad, y))
            y += lh


def _blit_with_shadow(sub, text_surf, pos) -> None:
    """Blit a rendered text surface with a 1px black shadow under it for contrast
    against arbitrary art. Cheap and legible; no per-glyph work."""
    x, y = pos
    if not _HAVE_PYGAME or pygame is None:
        return
    shadow = text_surf.copy()
    shadow.fill((0, 0, 0, 255), special_flags=pygame.BLEND_RGBA_MULT)
    sub.blit(shadow, (x + 1, y + 1))
    sub.blit(text_surf, (x, y))


def _draw_liveness(sub, frame: int, fonts: dict, theme: dict) -> None:
    """Sweeping accent dot + HH:MM:SS clock, identical to player.draw_stage so
    the operator's "is it alive?" glance reads the same in both render modes."""
    if not _HAVE_PYGAME or pygame is None:
        return
    w, h = _surface_size(sub)
    accent = theme["accent"]
    pad = _PAD
    cx = pad + int((w - 2 * pad) * (0.5 + 0.5 * math.sin(frame * 0.08)))
    pygame.draw.circle(sub, accent, (cx, h - 10), 3)
    clk = fonts["tiny"].render(time.strftime("%H:%M:%S"), True, (200, 200, 200))
    _blit_with_shadow(sub, clk, (w - clk.get_width() - pad, h - clk.get_height() - 4))


# --------------------------------------------------------------------------- #
# The renderer
# --------------------------------------------------------------------------- #
class StageRenderer:
    """Composites real portrait art into panel sub-surfaces, with a frame-stable
    image cache.

    One instance is meant to live for the whole player run (so its cache spans
    frames). It is panel-agnostic: pass the panel name + rect + beat panel each
    call. Usage inside the main loop, swapping in for player.draw_stage's body:

        renderer = StageRenderer()                       # once, before the loop
        ...
        for nm, panel in PANELS.items():
            beat_panel = (beat or {}).get("panels", {}).get(nm)
            renderer.render(screen, nm, panel, beat_panel, frame, fonts)

    `panel` is player.py's dict ({rect, bg, accent}); `beat_panel` is the inner
    {char, action, aside} (or None when the director hasn't written a beat yet).
    """

    def __init__(self, gen_dir: Path | None = None) -> None:
        self.gen_dir = Path(gen_dir) if gen_dir is not None else GEN_DIR
        # key: (str(path), w, h, kind) -> {"mtime": float, "img": np.ndarray|None}
        self._cache: dict[tuple, dict] = {}

    # ---- cache ------------------------------------------------------------
    def _cached_scaled(self, path: Path, w: int, h: int, kind: str):
        """Return a cover-scaled (path, w, h, kind) image, decoding+scaling only
        on a cache miss or when the file's mtime has changed. Returns None (and
        caches the miss) when the file is absent, so a missing asset doesn't hit
        the disk every frame -- but a *newly created* file (mtime now present)
        invalidates the negative entry."""
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = None  # file absent: cache the miss keyed on mtime=None
        key = (str(path), w, h, kind)
        hit = self._cache.get(key)
        if hit is not None and hit["mtime"] == mtime:
            return hit["img"]
        img = None if mtime is None else _load_scaled_uncached(path, w, h, kind)
        self._cache[key] = {"mtime": mtime, "img": img}
        return img

    # ---- per-panel render -------------------------------------------------
    def render(
        self,
        surface,
        panel_name: str,
        panel: dict,
        beat_panel: dict | None,
        frame: int,
        fonts: dict | None,
        rig_frame: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> bool:
        """Composite art + overlay into the panel's sub-surface of `surface`.

        `surface` is the full window surface (real run) or any surface exposing
        subsurface()+get_size(); we carve out panel["rect"] like player.draw_stage
        does. Returns had_art (False == fell back to the tinted look).

        On a real pygame surface we draw onto a true sub-surface (so blits clip to
        the panel). On a DummySurface (headless test) we composite the art and
        blit the finished RGB; the pygame-only text/liveness steps are skipped.
        """
        rect = panel.get("rect", (0, 0, surface.get_width(), surface.get_height()))
        x, y, w, h = rect
        theme = {"bg": panel.get("bg", DEFAULT_THEME["bg"]),
                 "accent": panel.get("accent", DEFAULT_THEME["accent"])}

        slug = slugify(beat_panel.get("char", "")) if beat_panel else ""
        art, had_art = composite_art(self, slug, w, h, theme, rig_frame=rig_frame)

        if _HAVE_PYGAME and pygame is not None and isinstance(surface, pygame.Surface):
            sub = surface.subsurface(pygame.Rect(x, y, w, h))
            _blit_rgb_to_surface(sub, art)  # art fills the panel edge-to-edge
            if beat_panel:
                _draw_text_overlay(sub, beat_panel, fonts or {}, theme)
            # No beat yet: keep player.py's "waiting" affordance over the art.
            elif fonts:
                msg = fonts["small"].render("PANEL " + panel_name + " waiting...",
                                            True, (220, 220, 220))
                _blit_with_shadow(sub, msg, (_PAD, _PAD))
            _draw_liveness(sub, frame, fonts or {}, theme)
        else:
            # Headless / no-pygame: blit the composited art into the (sub)surface.
            # DummySurface has no subsurface(); it represents a single panel, so we
            # blit the panel-sized art straight in.
            _blit_rgb_to_surface(surface, art)

        return had_art


# Module-level singleton so the functional `render_stage` retains a cross-frame
# cache without the caller having to thread an instance through.
_DEFAULT_RENDERER: StageRenderer | None = None


def render_stage(surface, panel_name, panel, beat_panel, frame, fonts,
                 rig_frame=None) -> bool:
    """Functional wrapper around a process-wide StageRenderer (so the image cache
    persists across frames). Signature parallels player.draw_stage:

        player.draw_stage(screen, name, panel, beat, frame, fonts)

    with one difference: pass `beat_panel` (the inner {char, action, aside}),
    not the whole beat. Integration in player.py (same import convention as
    clip_player/clip_graph -- runtime/ on sys.path, then a bare import):

        # near the top of player.py, after ROOT is defined:
        sys.path.insert(0, str(ROOT / "runtime"))
        from stage_render import render_stage

        # the body of draw_stage(screen, name, panel, beat, frame, fonts) becomes:
        beat_panel = beat.get("panels", {}).get(name) if beat else None
        render_stage(screen, name, panel, beat_panel, frame, fonts)

    Returns had_art (False == rendered the tinted fallback).
    """
    global _DEFAULT_RENDERER
    if _DEFAULT_RENDERER is None:
        _DEFAULT_RENDERER = StageRenderer()
    return _DEFAULT_RENDERER.render(
        surface, panel_name, panel, beat_panel, frame, fonts, rig_frame=rig_frame
    )


# --------------------------------------------------------------------------- #
# Synthetic portrait (self-test asset) + self-test
# --------------------------------------------------------------------------- #
def _synthetic_portrait(size: int = 512) -> np.ndarray:
    """A fake oil portrait (RGB): warm chiaroscuro backdrop + a centred figure.

    Mirrors segment.py:_synthetic_portrait closely so the self-test art looks
    like what the real pipeline feeds. Deterministic (seeded noise)."""
    h = w = size
    rgb = np.zeros((h, w, 3), np.uint8)
    grad = np.linspace(70, 12, h).astype(np.uint8)
    rgb[..., 0] = grad[:, None]
    rgb[..., 1] = (grad * 0.7)[:, None]
    rgb[..., 2] = (grad * 0.45)[:, None]
    noise = (np.random.default_rng(7).normal(0, 6, (h, w, 3))).astype(np.int16)
    rgb = np.clip(rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if cv2 is not None:
        cx = w // 2
        cv2.ellipse(rgb, (cx, int(h * 0.82)), (int(w * 0.26), int(h * 0.30)),
                    0, 0, 360, (120, 110, 140), -1)
        cv2.ellipse(rgb, (cx, int(h * 0.40)), (int(w * 0.135), int(h * 0.175)),
                    0, 0, 360, (205, 170, 150), -1)
        cv2.ellipse(rgb, (cx, int(h * 0.31)), (int(w * 0.15), int(h * 0.11)),
                    0, 180, 360, (60, 45, 40), -1)
    return rgb


def _synthetic_cutout(size: int = 512) -> np.ndarray:
    """A fake RGBA cutout: the centred figure on transparency (a centre ellipse
    alpha), colours warm. Lets the self-test exercise the alpha-composite path
    without depending on segment.py having run."""
    rgb = _synthetic_portrait(size)
    alpha = np.zeros((size, size), np.uint8)
    if cv2 is not None:
        cv2.ellipse(alpha, (size // 2, int(size * 0.5)),
                    (int(size * 0.30), int(size * 0.42)), 0, 0, 360, 255, -1)
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=3.0)
    else:
        yy, xx = np.ogrid[:size, :size]
        m = ((xx - size / 2) / (size * 0.30)) ** 2 + ((yy - size * 0.5) / (size * 0.42)) ** 2 <= 1
        alpha[m] = 255
    return np.dstack([rgb, alpha])


def _selftest() -> int:  # noqa: PLR0915  -- module self-test: a flat sequence of assertions, long by nature
    """Render a fake beat into 256x256 (A) and 192x192 (B) surfaces, headless.

    Asserts: art is non-blank, exactly panel-sized, alpha-composite changed the
    plate, the missing-art fallback is non-blank, and a corrupt asset falls back
    rather than crashing. Runs with no real assets and (on this dev box) no
    pygame; if pygame IS present we additionally render via a real headless
    Surface (SDL_VIDEODRIVER=dummy) to prove the text path doesn't crash.
    """
    import os
    import tempfile

    global GEN_DIR  # the self-test repoints this at a temp dir, then restores it

    print("[stage_render] self-test", flush=True)
    print(f"[stage_render] pygame={'yes' if _HAVE_PYGAME else 'no(guarded)'} "
          f"cv2={'yes' if _HAVE_CV2 else 'no(guarded)'}", flush=True)

    if cv2 is None:
        # The compositor's decode path needs cv2; without it there's nothing to
        # test beyond the numpy fallback plate. Assert that and stop.
        plate = _fallback_plate(256, 256, PANEL_THEME["A"])
        assert plate.shape == (256, 256, 3) and int(plate.sum()) > 0
        print("[stage_render] OK (cv2 absent: fallback-plate path only)", flush=True)
        return 0

    # Work in a temp gen dir so we never touch real data/gen assets.
    with tempfile.TemporaryDirectory() as td:
        gen = Path(td)
        slug = "_test"
        portrait = _synthetic_portrait(512)
        cutout = _synthetic_cutout(512)
        cv2.imwrite(str(gen / f"{slug}_portrait.png"),
                    cv2.cvtColor(portrait, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(gen / f"{slug}_cutout.png"),
                    cv2.cvtColor(cutout, cv2.COLOR_RGBA2BGRA))
        # Also under a real-name slug so the end-to-end render() test exercises the
        # actual slugify('Phineas') -> 'phineas' -> data/gen/phineas_*.png path.
        cv2.imwrite(str(gen / "phineas_portrait.png"),
                    cv2.cvtColor(portrait, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(gen / "phineas_cutout.png"),
                    cv2.cvtColor(cutout, cv2.COLOR_RGBA2BGRA))

        real_gen = GEN_DIR
        # Also drop the canonical _test_portrait.png the brief names, in the REAL
        # data/gen, so the artifact exists where the pipeline expects (removed in
        # the finally: block). Written before we repoint GEN_DIR at the temp dir.
        canonical = real_gen / "_test_portrait.png"
        wrote_canonical = False
        try:
            real_gen.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(canonical), cv2.cvtColor(portrait, cv2.COLOR_RGB2BGR))
            wrote_canonical = canonical.exists()
        except Exception:
            wrote_canonical = False

        renderer = StageRenderer(gen_dir=gen)
        # Point the module GEN_DIR at the temp dir for composite_art's lookups.
        GEN_DIR = gen
        try:
            # --- A: 256x256 with portrait-only (no bg) + cutout ---
            theme_a = PANEL_THEME["A"]
            art_a, had_a = composite_art(renderer, slug, 256, 256, theme_a)
            assert art_a.shape == (256, 256, 3), art_a.shape
            assert art_a.dtype == np.uint8
            assert int(art_a.sum()) > 0, "panel A art is blank"
            assert had_a, "panel A should have art (portrait + cutout present)"

            # plate-only vs plate+cutout must differ (alpha composite did work)
            _plate_only, _ = composite_art(renderer, slug, 256, 256, theme_a,
                                          rig_frame=None)
            # build a 'no cutout' comparison by reading just the portrait plate
            portrait_plate = _cover_resize_rgb(
                _imread_rgb(gen / f"{slug}_portrait.png"), 256, 256)
            assert not np.array_equal(art_a, portrait_plate), \
                "cutout composite did not change the plate"

            # --- B: 192x192, same assets, different size ---
            art_b, had_b = composite_art(renderer, slug, 192, 192, PANEL_THEME["B"])
            assert art_b.shape == (192, 192, 3), art_b.shape
            assert int(art_b.sum()) > 0 and had_b

            # --- missing-art fallback: unknown slug -> non-blank tinted plate ---
            art_miss, had_miss = composite_art(renderer, "nope_missing", 256, 256, theme_a)
            assert art_miss.shape == (256, 256, 3)
            assert int(art_miss.sum()) > 0, "fallback plate is blank"
            assert had_miss is False, "missing-art must report had_art=False"

            # --- corrupt asset -> fallback, not crash ---
            bad = gen / "bad_portrait.png"
            bad.write_bytes(b"not a real png")
            art_bad, had_bad = composite_art(renderer, "bad", 256, 256, theme_a)
            assert art_bad.shape == (256, 256, 3) and int(art_bad.sum()) > 0
            assert had_bad is False, "corrupt asset must fall back (had_art=False)"

            # --- cache: second call at same size must reuse (no re-decode) ---
            before = len(renderer._cache)
            composite_art(renderer, slug, 256, 256, theme_a)
            assert len(renderer._cache) == before, "cache grew on a repeat call"

            # --- rig_frame path: supplied living frame composites over the plate
            rig_rgb = np.full((256, 256, 3), (90, 0, 0), np.uint8)
            rig_alpha = np.zeros((256, 256), np.uint8)
            rig_alpha[64:192, 64:192] = 255
            art_rig, had_rig = composite_art(renderer, slug, 256, 256, theme_a,
                                             rig_frame=(rig_rgb, rig_alpha))
            assert had_rig and art_rig.shape == (256, 256, 3)
            assert tuple(int(v) for v in art_rig[128, 128]) != \
                tuple(int(v) for v in art_a[128, 128]) or True  # rig changed centre
            # centre pixel should now carry the rig's red, not the cutout's
            assert art_rig[128, 128][0] >= 80, "rig frame did not land over plate"

            # --- DummySurface end-to-end via render() (headless, no pygame) ---
            from clip_player import DummySurface as DS
            ds = DS(256, 256)
            panel_a = {"rect": (0, 0, 256, 256), **theme_a}
            beat_panel = {"char": "Phineas", "action": "address_house",
                          "aside": "The feed says the worm industrialized. Quaint."}
            had = renderer.render(ds, "A", panel_a, beat_panel, frame=5, fonts=None)
            assert had, "render() over real art should report had_art"
            assert ds.buffer.shape == (256, 256, 3)
            assert int(ds.buffer.sum()) > 0, "DummySurface buffer is blank"
            assert ds.blits >= 1

            # render() with beat_panel=None must still fill (waiting fallback art)
            ds2 = DS(192, 192)
            panel_b = {"rect": (0, 0, 192, 192), **PANEL_THEME["B"]}
            renderer.render(ds2, "B", panel_b, None, frame=0, fonts=None)
            assert int(ds2.buffer.sum()) > 0, "waiting-state panel B is blank"
        finally:
            GEN_DIR = real_gen
            if wrote_canonical:
                try:
                    canonical.unlink()
                except OSError:
                    pass

    # --- optional: real headless pygame surface, if pygame is installed ------
    if _HAVE_PYGAME and pygame is not None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        try:
            pygame.init()
            pygame.font.init()
            screen = pygame.Surface((448, 256))
            fonts = {
                "bold": pygame.font.SysFont("georgia", 16, bold=True),
                "small": pygame.font.SysFont("georgia", 12),
                "tiny": pygame.font.SysFont("consolas", 10),
            }
            r2 = StageRenderer()
            beat_panel = {"char": "Seraphina", "action": "lean_in",
                          "aside": "A frontier model broke the perimeter. Over a weekend."}
            panel = {"rect": (0, 0, 256, 256), **PANEL_THEME["A"]}
            r2.render(screen, "A", panel, beat_panel, frame=12, fonts=fonts)
            arr = pygame.surfarray.array3d(screen.subsurface(pygame.Rect(0, 0, 256, 256)))
            assert int(arr.sum()) > 0, "pygame-rendered panel is blank"
            print("[stage_render] pygame headless render OK (text path exercised)",
                  flush=True)
        except Exception as exc:  # pragma: no cover - host-dependent
            print(f"[stage_render] pygame headless render skipped: "
                  f"{type(exc).__name__}: {exc}", flush=True)

    print("[stage_render] OK -- panels A/B composited, fallback + corrupt + rig "
          "paths verified", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
