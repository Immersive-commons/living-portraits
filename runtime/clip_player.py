"""
clip_player.py -- play a clip (or a path of clips) into one panel sub-surface.

This is the "rig + clip" render mode the architecture calls for: same
state-consumption shell as player.py's text dev-view, swapped renderer. A
ClipPlayer is bound to one panel's pygame.Rect (Panel A 256x256 @ (0,0), Panel B
192x192 @ (256,0)). Given a Clip -- or a sequence from ClipGraph.path() -- it
advances one frame per tick() and blits that frame into the panel, crossfading
across clip seams so transitions don't pop.

Two frame sources, chosen per clip:

  SYNTHETIC  (clip.asset is None)
      Procedural frames generated with numpy: a panel-tinted field, the pose
      label, and a moving element (a sweeping marker) so motion is visible.
      This is what runs *today*, before any real footage is baked -- it makes
      the whole player testable offline with zero art assets.

  ASSET-BACKED  (clip.asset set)
      Decoded with OpenCV: an .mp4 (cv2.VideoCapture) or a png-sequence
      directory (sorted frames via cv2.imread). Resized to the panel and
      colour-corrected BGR->RGB. This is the path the verified clip library
      feeds once the generative pipeline lands.

Surface abstraction
-------------------
The blit target is anything exposing a minimal pygame.Surface API
(get_size + blitting a frame). On SC2 that's a real pygame sub-surface. For the
headless self-test (and on this dev box, where pygame isn't installed) we use a
numpy-backed DummySurface -- no window, no pygame, SDL_VIDEODRIVER=dummy not even
required. pygame is import-guarded so this module imports anywhere; opencv only
imports if an asset-backed clip is actually played.

Frames are numpy uint8 arrays shaped (H, W, 3) in RGB. Blitting to a real
pygame surface goes through pygame.surfarray (no per-pixel Python loop).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable, Iterator

import numpy as np

from runtime.clip_graph import CLIPS_DIR, Clip

# ---- optional heavy deps, import-guarded ---------------------------------
# pygame is present on SC2's .venv but not on every dev box; opencv is only
# needed to decode real footage. Neither is required for the synthetic path or
# the headless self-test.
try:  # pragma: no cover - presence depends on host
    import pygame  # type: ignore

    _HAVE_PYGAME = True
except Exception:  # pragma: no cover
    pygame = None  # type: ignore
    _HAVE_PYGAME = False

try:  # pragma: no cover - presence depends on host
    import cv2  # type: ignore

    _HAVE_CV2 = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    _HAVE_CV2 = False


# Panel palettes mirror player.py:PANELS so the rig view matches the dev view.
PANEL_THEME = {
    "A": {"bg": (38, 14, 16), "accent": (210, 170, 90)},
    "B": {"bg": (12, 26, 28), "accent": (120, 200, 190)},
}
DEFAULT_THEME = {"bg": (20, 20, 24), "accent": (200, 200, 200)}


# --------------------------------------------------------------------------- #
# Surface abstraction
# --------------------------------------------------------------------------- #
class DummySurface:
    """A headless stand-in for a pygame sub-surface, backed by a numpy buffer.

    Exposes just enough of the pygame.Surface API for ClipPlayer: get_size(),
    get_width/height(), and a blit_frame() the player uses to push an RGB array.
    Lets the whole render path run with no window and no pygame -- the basis of
    the offline self-test. .buffer is the current panel contents (H, W, 3) RGB.
    """

    def __init__(self, width: int, height: int) -> None:
        self._w = int(width)
        self._h = int(height)
        self.buffer = np.zeros((self._h, self._w, 3), dtype=np.uint8)
        self.blits = 0  # count for the self-test to prove frames landed

    def get_size(self) -> tuple[int, int]:
        return (self._w, self._h)

    def get_width(self) -> int:
        return self._w

    def get_height(self) -> int:
        return self._h

    def blit_frame(self, frame: np.ndarray) -> None:
        """Copy an (H, W, 3) RGB array into the buffer, resizing if needed."""
        h, w = frame.shape[:2]
        if (w, h) != (self._w, self._h):
            frame = _resize_rgb(frame, self._w, self._h)
        self.buffer[:, :, :] = frame
        self.blits += 1


def _is_pygame_surface(surf) -> bool:
    return _HAVE_PYGAME and isinstance(surf, pygame.Surface)  # type: ignore[arg-type]


def _surface_size(surf) -> tuple[int, int]:
    # Both pygame.Surface and DummySurface expose get_size().
    return surf.get_size()


def _blit_rgb_to_surface(surf, frame: np.ndarray) -> None:
    """Blit an (H, W, 3) RGB uint8 array onto a pygame.Surface or DummySurface."""
    w, h = _surface_size(surf)
    fh, fw = frame.shape[:2]
    if (fw, fh) != (w, h):
        frame = _resize_rgb(frame, w, h)
    if _is_pygame_surface(surf):
        # pygame.surfarray wants (W, H, 3); our frames are (H, W, 3) -> swap axes 0,1.
        arr_wh = np.transpose(frame, (1, 0, 2))
        try:
            surf_arr = pygame.surfarray.pixels3d(surf)  # type: ignore[union-attr]
            surf_arr[:, :, :] = arr_wh
            del surf_arr  # release the surface lock
        except Exception:
            # Fallback for surfaces that don't support direct pixel access:
            # build a temp surface from the buffer and blit it.
            tmp = pygame.surfarray.make_surface(arr_wh)  # type: ignore[union-attr]
            surf.blit(tmp, (0, 0))
    else:
        surf.blit_frame(frame)


def _resize_rgb(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize an RGB array to (h, w). Uses cv2 when available, else nearest-neighbour numpy."""
    if _HAVE_CV2:
        return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)  # type: ignore[union-attr]
    fh, fw = frame.shape[:2]
    ys = (np.linspace(0, fh - 1, h)).astype(np.int64)
    xs = (np.linspace(0, fw - 1, w)).astype(np.int64)
    return frame[ys][:, xs]


# --------------------------------------------------------------------------- #
# Frame sources
# --------------------------------------------------------------------------- #
def _tiny_glyphs() -> dict[str, np.ndarray]:
    """A minimal 3x5 bitmap font (A-Z, 0-9, space) for pose labels.

    We can't rely on pygame.font in the synthetic path (pygame may be absent),
    so labels are drawn from this tiny ROM. Only the characters used by pose
    names + digits are defined; unknown chars render blank.
    """
    rows = {
        "A": ["010", "101", "111", "101", "101"],
        "B": ["110", "101", "110", "101", "110"],
        "C": ["011", "100", "100", "100", "011"],
        "D": ["110", "101", "101", "101", "110"],
        "E": ["111", "100", "110", "100", "111"],
        "F": ["111", "100", "110", "100", "100"],
        "G": ["011", "100", "101", "101", "011"],
        "H": ["101", "101", "111", "101", "101"],
        "I": ["111", "010", "010", "010", "111"],
        "J": ["001", "001", "001", "101", "010"],
        "K": ["101", "110", "100", "110", "101"],
        "L": ["100", "100", "100", "100", "111"],
        "M": ["101", "111", "111", "101", "101"],
        "N": ["101", "111", "111", "111", "101"],
        "O": ["010", "101", "101", "101", "010"],
        "P": ["110", "101", "110", "100", "100"],
        "Q": ["010", "101", "101", "110", "011"],
        "R": ["110", "101", "110", "101", "101"],
        "S": ["011", "100", "010", "001", "110"],
        "T": ["111", "010", "010", "010", "010"],
        "U": ["101", "101", "101", "101", "011"],
        "V": ["101", "101", "101", "010", "010"],
        "W": ["101", "101", "111", "111", "101"],
        "X": ["101", "101", "010", "101", "101"],
        "Y": ["101", "101", "010", "010", "010"],
        "Z": ["111", "001", "010", "100", "111"],
        "0": ["111", "101", "101", "101", "111"],
        "1": ["010", "110", "010", "010", "111"],
        "2": ["110", "001", "010", "100", "111"],
        "3": ["110", "001", "010", "001", "110"],
        "4": ["101", "101", "111", "001", "001"],
        "5": ["111", "100", "110", "001", "110"],
        "6": ["011", "100", "110", "101", "010"],
        "7": ["111", "001", "010", "010", "010"],
        "8": ["010", "101", "010", "101", "010"],
        "9": ["010", "101", "011", "001", "110"],
        "_": ["000", "000", "000", "000", "111"],
        " ": ["000", "000", "000", "000", "000"],
    }
    out: dict[str, np.ndarray] = {}
    for ch, pat in rows.items():
        out[ch] = np.array([[1 if c == "1" else 0 for c in row] for row in pat], dtype=np.uint8)
    return out


_GLYPHS = _tiny_glyphs()


def _draw_label(frame: np.ndarray, text: str, x: int, y: int, color, scale: int = 2) -> None:
    """Stamp `text` into `frame` (RGB, modified in place) using the tiny font."""
    color = np.array(color, dtype=np.uint8)
    cx = x
    for ch in text.upper():
        glyph = _GLYPHS.get(ch)
        if glyph is None:
            cx += (3 + 1) * scale
            continue
        gh, gw = glyph.shape
        for gy in range(gh):
            for gx in range(gw):
                if glyph[gy, gx]:
                    y0 = y + gy * scale
                    x0 = cx + gx * scale
                    y1 = min(y0 + scale, frame.shape[0])
                    x1 = min(x0 + scale, frame.shape[1])
                    if 0 <= y0 < frame.shape[0] and 0 <= x0 < frame.shape[1]:
                        frame[y0:y1, x0:x1] = color
        cx += (gw + 1) * scale


def synthetic_frame(
    width: int,
    height: int,
    pose: str,
    t: float,
    theme: dict,
) -> np.ndarray:
    """One procedural frame: tinted field + pose label + a sweeping marker.

    t in [0, 1) is the clip's normalized progress, which drives the moving
    element so the clip visibly animates and a seam-crossfade is observable.
    """
    bg = np.array(theme["bg"], dtype=np.uint8)
    accent = np.array(theme["accent"], dtype=np.uint8)
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:, :] = bg

    # A soft vertical gradient toward the accent so each frame isn't flat.
    grad = np.linspace(0.0, 0.22, height, dtype=np.float32)[:, None, None]
    frame = (frame.astype(np.float32) * (1 - grad) + accent.astype(np.float32) * grad).astype(np.uint8)

    # Border in accent (1px) so the panel edge is legible like the dev view.
    frame[0, :] = accent
    frame[-1, :] = accent
    frame[:, 0] = accent
    frame[:, -1] = accent

    # Pose label, top-left.
    _draw_label(frame, pose, 4, 4, accent, scale=max(1, width // 128))

    # Moving element: a marker sweeping left<->right near the bottom, plus a
    # small progress bar. Position is a function of t so it animates over the
    # clip and is continuous across a crossfade.
    pad = 6
    span = max(1, width - 2 * pad - 6)
    mx = pad + int(span * (0.5 + 0.5 * math.sin(t * 2 * math.pi)))
    my = height - 12
    r = max(2, width // 64)
    yy, xx = np.ogrid[:height, :width]
    mask = (xx - mx) ** 2 + (yy - my) ** 2 <= r * r
    frame[mask] = accent

    # progress bar along the very bottom interior row
    bar_w = int((width - 2 * pad) * t)
    frame[height - 4 : height - 2, pad : pad + bar_w] = accent

    return frame


def _iter_synthetic(clip: Clip, width: int, height: int, pose: str, theme: dict) -> Iterator[np.ndarray]:
    n = max(1, clip.frames)
    for i in range(n):
        t = i / n  # 0..1, exclusive of 1 so a loop's last frame != first
        yield synthetic_frame(width, height, pose, t, theme)


def _iter_asset(clip: Clip, width: int, height: int) -> Iterator[np.ndarray]:
    """Decode an asset-backed clip (mp4 or png-sequence dir) into RGB frames."""
    if not _HAVE_CV2:
        raise RuntimeError("opencv (cv2) not available; cannot decode asset-backed clip")
    assert clip.asset is not None
    path = Path(clip.asset)
    if not path.is_absolute():
        path = CLIPS_DIR / path

    if path.is_dir():
        # png-sequence: sorted image files
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in exts)
        for fp in files:
            bgr = cv2.imread(str(fp), cv2.IMREAD_COLOR)  # type: ignore[union-attr]
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # type: ignore[union-attr]
            yield _resize_rgb(rgb, width, height)
    else:
        # video file
        cap = cv2.VideoCapture(str(path))  # type: ignore[union-attr]
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # type: ignore[union-attr]
                yield _resize_rgb(rgb, width, height)
        finally:
            cap.release()


def clip_frames(clip: Clip, width: int, height: int, theme: dict) -> Iterator[np.ndarray]:
    """Frames for one clip from whichever source it declares.

    The pose label shown is the clip's destination pose (the rig is moving
    *toward* to_node), which reads naturally for both transitions and holds.
    """
    if clip.is_synthetic:
        yield from _iter_synthetic(clip, width, height, clip.to_node, theme)
    else:
        # Fall back to synthetic if decode yields nothing (missing/corrupt file),
        # so the show never goes black on a bad asset.
        produced = False
        try:
            for fr in _iter_asset(clip, width, height):
                produced = True
                yield fr
        except Exception:
            produced = False
        if not produced:
            yield from _iter_synthetic(clip, width, height, clip.to_node, theme)


def crossfade(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """Linear blend of two RGB frames; alpha=0 -> a, alpha=1 -> b."""
    if a.shape != b.shape:
        b = _resize_rgb(b, a.shape[1], a.shape[0])
    alpha = float(max(0.0, min(1.0, alpha)))
    if _HAVE_CV2:
        return cv2.addWeighted(a, 1 - alpha, b, alpha, 0.0)  # type: ignore[union-attr]
    return (a.astype(np.float32) * (1 - alpha) + b.astype(np.float32) * alpha).astype(np.uint8)


# --------------------------------------------------------------------------- #
# The player
# --------------------------------------------------------------------------- #
@dataclass
class _Active:
    """Bookkeeping for the clip currently playing."""

    clip: Clip
    frames: list[np.ndarray]
    idx: int = 0

    @property
    def done(self) -> bool:
        return self.idx >= len(self.frames)


class ClipPlayer:
    """Plays clips into one panel surface, one frame per tick(), crossfading seams.

    Usage (per panel, per frame of the main loop):

        cp = ClipPlayer(panel_name="A", rect=pygame.Rect(0, 0, 256, 256))
        cp.play_path(graph.path(cur_pose, beat_pose) or [graph.hold_clip(beat_pose)])
        ...
        cp.tick(panel_surface)   # advances one frame and blits

    When a clip ends, the next clip in the queued path begins; the first
    `crossfade_frames` of the new clip are blended over the last frame of the
    previous one. When the whole queue drains, the player holds the last frame
    (or loops it if the last clip is loopable -- the held-pose case).
    """

    def __init__(
        self,
        panel_name: str,
        rect=None,
        size: tuple[int, int] | None = None,
        crossfade_frames: int = 6,
    ) -> None:
        self.panel_name = panel_name
        self.theme = PANEL_THEME.get(panel_name, DEFAULT_THEME)
        # Size comes from rect (pygame.Rect) or an explicit (w, h). On SC2 the
        # caller passes the panel's pygame.Rect; the self-test passes size=.
        if rect is not None:
            self.width = int(getattr(rect, "width", rect[2]))
            self.height = int(getattr(rect, "height", rect[3]))
        elif size is not None:
            self.width, self.height = int(size[0]), int(size[1])
        else:
            raise ValueError("ClipPlayer needs a rect or a size")

        self.crossfade_frames = max(0, int(crossfade_frames))
        self._queue: list[Clip] = []
        self._active: _Active | None = None
        self._prev_last: np.ndarray | None = None  # last frame of the just-finished clip
        self._fade_i: int = 0  # frames into the current crossfade (0 = not fading)
        self._last_frame: np.ndarray | None = None  # what we most recently showed
        self.current_pose: str | None = None  # to_node of the clip in flight

    # ---- scheduling -------------------------------------------------------
    def play_clip(self, clip: Clip) -> None:
        """Play a single clip now, clearing any queued path."""
        self.play_path([clip])

    def play_path(self, clips: Iterable[Clip] | None) -> None:
        """Queue a sequence of clips (e.g. from ClipGraph.path()) and start it.

        None or [] means 'nothing to do' -- the player holds its last frame.
        """
        clips = [c for c in (clips or []) if c is not None]
        self._queue = list(clips)
        self._active = None
        self._fade_i = 0
        # keep _prev_last as the on-screen frame so the first new clip can fade in
        self._prev_last = self._last_frame
        self._advance_clip()

    def enqueue(self, clip: Clip) -> None:
        """Append a clip to the queue without disturbing what's playing."""
        self._queue.append(clip)

    def _advance_clip(self) -> None:
        """Pull the next clip off the queue and bake its frame list."""
        if not self._queue:
            self._active = None
            return
        clip = self._queue.pop(0)
        frames = list(clip_frames(clip, self.width, self.height, self.theme))
        if not frames:  # paranoia: never an empty clip
            frames = [synthetic_frame(self.width, self.height, clip.to_node, 0.0, self.theme)]
        self._active = _Active(clip=clip, frames=frames)
        self.current_pose = clip.to_node
        # start a crossfade from the previous clip's tail, if we have one
        self._fade_i = 0 if self._prev_last is None else 1

    # ---- per-frame --------------------------------------------------------
    def _next_frame(self) -> np.ndarray:
        """Compute the next frame to show (advancing internal state)."""
        if self._active is None:
            self._advance_clip()

        if self._active is None:
            # queue drained: hold (or loop a loopable tail) -- never go black
            if self._last_frame is not None:
                return self._last_frame
            return synthetic_frame(self.width, self.height, self.current_pose or "idle", 0.0, self.theme)

        active = self._active
        frame = active.frames[min(active.idx, len(active.frames) - 1)]

        # crossfade the opening frames of this clip over the previous tail
        if self._fade_i and self._prev_last is not None and self.crossfade_frames > 0:
            alpha = self._fade_i / (self.crossfade_frames + 1)
            frame = crossfade(self._prev_last, frame, alpha)
            self._fade_i += 1
            if self._fade_i > self.crossfade_frames:
                self._fade_i = 0
                self._prev_last = None

        active.idx += 1

        if active.done:
            # remember tail for the next clip's crossfade
            self._prev_last = active.frames[-1]
            if self._queue:
                self._advance_clip()
            elif active.clip.loopable:
                # held-pose loop: restart this same clip seamlessly
                active.idx = 0
                self._prev_last = None  # loopable => last flows into first, no fade
            else:
                # final transition with nothing queued: keep showing last frame
                self._active = active  # stays "done", _next_frame holds last
                active.idx = len(active.frames)  # clamp

        self._last_frame = frame
        return frame

    def tick(self, surface) -> np.ndarray:
        """Advance one frame and blit it into `surface`. Returns the RGB frame.

        `surface` is a pygame sub-surface (real run) or a DummySurface (test).
        """
        frame = self._next_frame()
        _blit_rgb_to_surface(surface, frame)
        return frame

    # ---- introspection ----------------------------------------------------
    @property
    def idle(self) -> bool:
        """True when nothing is queued and the active clip (if any) is finished
        and not looping -- the director may pick a new beat."""
        if self._queue:
            return False
        if self._active is None:
            return True
        if self._active.clip.loopable:
            return False  # a hold loop is 'busy' forever until replaced
        return self._active.done


# --------------------------------------------------------------------------- #
# Integration helper for player.py
# --------------------------------------------------------------------------- #
def beat_to_path(graph, current_pose: str | None, beat_pose: str):
    """Map a beat's `action` (target pose) to a clip path the player can run.

    - If current_pose is None (cold start) we just hold on the target pose.
    - Otherwise route current_pose -> beat_pose via the graph.
    - If already on the pose, or no route exists, fall back to the held loop
      for the target pose so the panel always has something to play.

    Returns a list[Clip] suitable for ClipPlayer.play_path().
    """
    target_hold = graph.hold_clip(beat_pose)
    fallback = [target_hold] if target_hold is not None else []

    if current_pose is None or current_pose == beat_pose:
        return fallback

    route = graph.path(current_pose, beat_pose)
    if route is None:
        return fallback
    if route == []:
        return fallback
    # after the transition, settle into the held loop on the target pose
    return route + ([target_hold] if target_hold is not None else [])


# --------------------------------------------------------------------------- #
# Headless self-test
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    """Build idle<->address_house<->leaving, find a path, run the player headless."""
    from clip_graph import ClipGraph

    # small graph with the three poses the brief names
    g = ClipGraph()
    g.add("idle", "address_house", frames=8, tags=["transition"])
    g.add("address_house", "idle", frames=8, tags=["transition"])
    g.add("address_house", "leaving", frames=10, tags=["transition"])
    g.add("idle", "idle", frames=12, loopable=True, tags=["hold"])
    g.add("leaving", "leaving", frames=12, loopable=True, tags=["hold"])

    route = g.path("idle", "leaving")
    assert route is not None, "idle->leaving must be reachable"
    assert [c.key() for c in route] == ["idle->address_house", "address_house->leaving"], \
        [c.key() for c in route]

    # Panel A geometry, headless dummy surface (no pygame, no window).
    surf = DummySurface(256, 256)
    cp = ClipPlayer("A", size=(256, 256), crossfade_frames=4)

    # use the integration helper exactly as player.py will
    path = beat_to_path(g, current_pose="idle", beat_pose="leaving")
    assert path and path[0].key() == "idle->address_house", [c.key() for c in path]
    cp.play_path(path)

    # run enough frames to cross both transition seams + into the hold loop
    total = sum(c.frames for c in path)
    seen_nonblack = 0
    last = None
    for _ in range(total + 20):
        fr = cp.tick(surf)
        assert fr.shape == (256, 256, 3), fr.shape
        assert fr.dtype == np.uint8
        if int(fr.sum()) > 0:
            seen_nonblack += 1
        last = fr

    assert surf.blits == total + 20, surf.blits
    assert seen_nonblack > 0, "every rendered frame was black -- synthetic source failed"
    # buffer holds the last shown frame
    assert np.array_equal(surf.buffer, last), "DummySurface buffer != last blitted frame"

    # asset-backed clip with a bogus path must fall back to synthetic, not crash
    g.add("ponder", "ponder", asset="does_not_exist.mp4", frames=6, loopable=True)
    cp2 = ClipPlayer("B", size=(192, 192))
    cp2.play_clip(g.get_clip("ponder", "ponder"))
    fr2 = cp2.tick(DummySurface(192, 192))
    assert fr2.shape == (192, 192, 3) and int(fr2.sum()) > 0, "asset fallback to synthetic failed"

    # crossfade blends
    a = np.zeros((4, 4, 3), np.uint8)
    b = np.full((4, 4, 3), 200, np.uint8)
    mid = crossfade(a, b, 0.5)
    assert 90 <= int(mid.mean()) <= 110, int(mid.mean())

    # a clip that DECLARES an asset (even a broken one) is NOT a 'missing' edge:
    # missing_edges() means 'never generated' (asset=None). Broken-decode is a
    # separate failure the player papers over by falling back to synthetic.
    assert ("ponder", "ponder") not in g.missing_edges(), \
        "asset-set edge wrongly reported as missing"
    # the never-generated transitions ARE missing
    assert ("address_house", "leaving") in g.missing_edges(), g.missing_edges()

    print(
        "clip_player OK |",
        f"pygame={'yes' if _HAVE_PYGAME else 'no(guarded)'}",
        f"cv2={'yes' if _HAVE_CV2 else 'no(guarded)'}",
        "| frames blitted:",
        surf.blits,
        "| route:",
        " -> ".join(c.key() for c in route),
    )


if __name__ == "__main__":
    _selftest()
