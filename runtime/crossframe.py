"""
crossframe.py -- the Harry-Potter "walk out of one frame, into the next" effect.

A character occupying Panel A can WALK OUT (slide + fade off A's right edge),
disappear for a beat, then APPEAR in Panel B (slide + fade in from B's left
edge). The two LED panels are physically separate surfaces (A 256x256 @ (0,0),
B 192x192 @ (256,0) -- see player.py:PANELS), so this is NOT one continuous
canvas with a figure gliding across a seam. It is a *coordinated* exit-then-
enter: the source panel plays an EXIT, both panels show an empty GAP beat, then
the destination panel plays an ENTER. The brief empty frame is what sells the
illusion that the same character left one portrait and arrived in another.

Where it sits in the render stack
---------------------------------
This module produces nothing but a figure layer. It returns an (rgb, alpha)
pair shaped *exactly* like the `rig_frame=` argument that
stage_render.composite_art already accepts:

    rig_frame = (rgb (H, W, 3) uint8, alpha (H, W) uint8)

(see stage_render.py: composite_art / _fit_alpha). So during a transition the
player calls exit_frame()/enter_frame() to get a translated+faded figure layer,
then hands it straight to the existing compositor as `rig_frame=`. The bg plate,
the alpha-over, the panel-size cover-resize, the text overlay, and the liveness
marker are all drawn by stage_render unchanged. crossframe never blits, never
touches pygame, and never reads disk: it is pure numpy and imports anywhere
(headless-safe, like clip_player / stage_render).

The figure the layer carries is the character's static cutout
(data/gen/<slug>_cutout.png, RGBA, figure on transparency -- see
pipeline/segment.py). The caller decodes it once (it already has the
StageRenderer image cache) and passes the RGBA in; this module only does the
geometry (translate toward/away from an edge) and the opacity ramp.

The director beat that triggers a transition
--------------------------------------------
The stage manager (director/stage_manager.py) writes data/stage_state.json. To
move a character between panels it adds a top-level `move` object to the beat:

    {
      "ts": "...",
      "scene": "...",
      "panels": { "A": {...}, "B": {...} },
      "move": { "char": "Phineas", "from": "A", "to": "B" },   <-- NEW, optional
      "bridge": null
    }

`move.char` is the display name (e.g. "Phineas"); `from`/`to` are panel names
("A"/"B"). The player slugifies the name (stage_render.slugify) to find the
cutout, exactly as it does for `panels[*].char`. When `move` is absent the
player renders normally (each panel performs its own beat). The director should
keep `panels` populated for the duration too: panels[from].char is the moving
character (shown only until EXIT completes), and panels[to].char becomes the
moving character (shown once ENTER completes) so the static beat that follows
the transition is already consistent.

Exact per-tick calls the player makes during a transition
---------------------------------------------------------
On the first tick after it sees a new `move`, the player does (once):

    xstate.begin_transition(slug, from_panel="A", to_panel="B")

Then every tick while `not xstate.done`:

    xstate.advance(dt)                       # dt = seconds since last tick (1/FPS)
    a_phase, a_prog = xstate.panel_state("A")
    b_phase, b_prog = xstate.panel_state("B")

and for EACH panel it decides what to render:

  * The panel currently in PHASE_EXIT  (the from-panel during EXIT):
        layer = exit_frame(panel_rect, cutout_rgba, a_prog, direction="right")
        render_stage(screen, "A", panelA, beatA, frame, fonts, rig_frame=layer)
    -> figure slides toward / off the panel's right edge, fading out.

  * The panel currently in PHASE_ENTER (the to-panel during ENTER):
        layer = enter_frame(panel_rect, cutout_rgba, b_prog, direction="left")
        render_stage(screen, "B", panelB, beatB, frame, fonts, rig_frame=layer)
    -> figure slides in from the panel's left edge, fading in.

  * A panel in PHASE_GAP, PHASE_PRE, or PHASE_DONE renders its *normal* content
    for that beat (no rig_frame override) -- EXCEPT the moving character must not
    appear in a panel it has logically left/not-yet-entered. The simplest correct
    rule, and the one the docstring above bakes into the beat, is:
        - during the whole transition, suppress the moving char on the to-panel
          until ENTER starts, and on the from-panel once EXIT finishes. The
          player does this by passing rig_frame=_EMPTY_LAYER(panel_rect) (a fully
          transparent layer) for that panel, which composites to "bg plate only,
          no figure" -- i.e. the empty frame.

`direction` is derived from the panel geometry by the player: a figure leaving
toward the destination exits the edge that faces it. For the canonical A(left)->
B(right) layout that is exit "right" / enter "left". A B->A move is exit "left" /
enter "right". `edge_for_move(from_panel, to_panel, panels_geometry)` is provided
so the player doesn't hardcode it.

Phase model
-----------
    PHASE_PRE   -- transition requested, not started (0-length unless paused)
    PHASE_EXIT  -- from-panel: figure slides off its facing edge, alpha 1->0
    PHASE_GAP   -- neither panel shows the figure (the empty beat)
    PHASE_ENTER -- to-panel: figure slides in from its facing edge, alpha 0->1
    PHASE_DONE  -- arrived; player resumes normal rendering

Durations are module constants (seconds): EXIT_S, GAP_S, ENTER_S.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------- #
# Tunables (seconds). The whole walk is EXIT_S + GAP_S + ENTER_S long. Kept
# short so a move reads as a deliberate beat, not a scene change; the GAP is the
# load-bearing "empty portrait" that makes the two separate panels feel linked.
# --------------------------------------------------------------------------- #
EXIT_S: float = 1.10
GAP_S: float = 0.45
ENTER_S: float = 1.10

# Phase tokens. Strings (not an Enum) so they serialise trivially into
# stage_state-style JSON and read plainly in logs / asserts.
PHASE_PRE = "pre"
PHASE_EXIT = "exit"
PHASE_GAP = "gap"
PHASE_ENTER = "enter"
PHASE_DONE = "done"

# How far past the panel edge the figure travels at progress=1.0, as a fraction
# of panel width. 1.0 would leave its trailing edge exactly on the border; a
# little extra (1.15) walks it fully clear so no sliver lingers when alpha hits 0.
_OVERSHOOT = 1.15


# --------------------------------------------------------------------------- #
# Easing
# --------------------------------------------------------------------------- #
def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


def _ease_in_out(t: float) -> float:
    """Smoothstep (3t^2 - 2t^3): eases the slide so the figure accelerates off
    and decelerates on rather than translating linearly (which reads robotic)."""
    t = _clamp01(t)
    return t * t * (3.0 - 2.0 * t)


# --------------------------------------------------------------------------- #
# Figure-layer geometry (pure numpy)
# --------------------------------------------------------------------------- #
def _blank_layer(w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """A fully transparent (rgb=0, alpha=0) layer at panel size -- composites to
    'bg plate only, no figure'. This is the empty-frame the GAP renders, and what
    the player passes for the moving character's vacated/not-yet-occupied panel."""
    rgb = np.zeros((h, w, 3), np.uint8)
    alpha = np.zeros((h, w), np.uint8)
    return rgb, alpha


def _empty_layer_for_rect(panel_rect) -> tuple[np.ndarray, np.ndarray]:
    """Public-ish helper mirroring _blank_layer but keyed on a panel rect, for the
    player's PHASE_GAP / suppressed-figure ticks. (rect = (x, y, w, h).)"""
    _, _, w, h = panel_rect
    return _blank_layer(int(w), int(h))


def _prep_figure_rgba(cutout_rgba: np.ndarray, w: int, h: int) -> np.ndarray:
    """Cover-resize an RGBA cutout to (h, w) so the figure fills the panel the way
    the static composite does. Pure numpy (nearest-neighbour) so this module never
    needs opencv: stage_render already does a high-quality cover-resize on the
    cutout for the *static* case; here the figure is mid-motion and being faded,
    so a light NN scale of the (already panel-ish) cutout is visually ample and
    keeps crossframe dependency-free.

    Accepts RGB (promoted to opaque) or RGBA; returns (h, w, 4) uint8.
    """
    arr = np.asarray(cutout_rgba)
    if arr.ndim == 2:  # single-channel matte -> treat as opaque gray
        arr = np.dstack([arr, arr, arr, np.full_like(arr, 255)])
    elif arr.shape[2] == 3:  # RGB -> opaque RGBA
        a = np.full((*arr.shape[:2], 1), 255, np.uint8)
        arr = np.concatenate([arr, a], axis=2)
    sh, sw = arr.shape[:2]
    if (sw, sh) == (w, h):
        return arr.astype(np.uint8, copy=False)
    # 'cover': scale by the larger ratio, then centre-crop the overflow axis.
    scale = max(w / sw, h / sh)
    nw, nh = max(1, int(round(sw * scale))), max(1, int(round(sh * scale)))
    ys = np.linspace(0, sh - 1, nh).astype(np.int64)
    xs = np.linspace(0, sw - 1, nw).astype(np.int64)
    scaled = arr[ys][:, xs]
    x0 = max(0, (nw - w) // 2)
    y0 = max(0, (nh - h) // 2)
    return scaled[y0:y0 + h, x0:x0 + w].astype(np.uint8, copy=False)


def _shift_x(rgba: np.ndarray, dx: int) -> np.ndarray:
    """Translate an RGBA figure layer horizontally by dx pixels (vacated columns
    become transparent). Positive dx pushes the figure right (toward the panel's
    right edge); negative pushes left. Vertical position is unchanged -- the walk
    is horizontal, panel-to-panel."""
    _h, w = rgba.shape[:2]
    out = np.zeros_like(rgba)
    if dx == 0:
        out[:] = rgba
        return out
    if dx > 0:
        if dx < w:
            out[:, dx:w, :] = rgba[:, 0:w - dx, :]
    else:
        d = -dx
        if d < w:
            out[:, 0:w - d, :] = rgba[:, d:w, :]
    return out


def _split_layer(rgba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split (h,w,4) RGBA into the (rgb (h,w,3), alpha (h,w)) pair the compositor
    consumes. alpha is the layer's matte; the caller's opacity ramp is already
    folded in by the time we split."""
    return rgba[..., :3].copy(), rgba[..., 3].copy()


def exit_frame(
    panel_rect,
    cutout_rgba: np.ndarray,
    progress: float,
    direction: str = "right",
) -> tuple[np.ndarray, np.ndarray]:
    """Figure layer for a character WALKING OUT of a panel.

    Returns (rgb (h,w,3), alpha (h,w)) -- the exact shape stage_render's
    `rig_frame=` takes -- of the cutout translated toward `direction`'s edge and
    faded out, as a function of `progress` in [0, 1] (0 = fully in place & opaque,
    1 = fully off-edge & transparent).

    panel_rect : (x, y, w, h) -- only w, h are used to size the layer; x, y are
                 accepted so the caller can pass PANELS["A"]["rect"] verbatim.
    direction  : "right" (default, exits the right edge -> moving to a panel on
                 the right) or "left" (exits the left edge).

    The slide distance is eased (smoothstep) and the alpha ramps to 0 a touch
    *ahead* of the slide so the figure has visibly dimmed before it clears frame
    (a hard pop at the edge would read as a cut, not a walk).
    """
    _, _, w, h = panel_rect
    w, h = int(w), int(h)
    p = _clamp01(progress)
    fig = _prep_figure_rgba(cutout_rgba, w, h)

    eased = _ease_in_out(p)
    travel = int(round(eased * _OVERSHOOT * w))
    dx = travel if direction == "right" else -travel
    shifted = _shift_x(fig, dx)

    # Opacity ramp: stay near-opaque through the first third, then fall to 0 by
    # p=1. Folded into the layer's alpha channel (multiplicative) so the matte
    # shape is preserved while the whole figure dims.
    opacity = _fade_out_curve(p)
    shifted[..., 3] = (shifted[..., 3].astype(np.float32) * opacity).astype(np.uint8)
    return _split_layer(shifted)


def enter_frame(
    panel_rect,
    cutout_rgba: np.ndarray,
    progress: float,
    direction: str = "left",
) -> tuple[np.ndarray, np.ndarray]:
    """Figure layer for a character WALKING IN to a panel.

    Mirror of exit_frame. Returns (rgb, alpha) of the cutout entering from
    `direction`'s edge and fading in, as a function of `progress` in [0, 1]
    (0 = fully off-edge & transparent, 1 = settled in place & opaque).

    direction : "left" (default, enters from the left edge -> arriving from a
                panel on the left) or "right" (enters from the right edge).
    """
    _, _, w, h = panel_rect
    w, h = int(w), int(h)
    p = _clamp01(progress)
    fig = _prep_figure_rgba(cutout_rgba, w, h)

    # At p=0 the figure is one overshoot off the entering edge; at p=1 it's home.
    eased = _ease_in_out(p)
    offset = int(round((1.0 - eased) * _OVERSHOOT * w))
    dx = -offset if direction == "left" else offset  # left-enter starts at -x
    shifted = _shift_x(fig, dx)

    opacity = _fade_in_curve(p)
    shifted[..., 3] = (shifted[..., 3].astype(np.float32) * opacity).astype(np.uint8)
    return _split_layer(shifted)


def _fade_out_curve(p: float) -> float:
    """Opacity for EXIT: ~1 until a third through, then smooth ramp to 0 at p=1."""
    if p <= 0.33:
        return 1.0
    return _clamp01(1.0 - (p - 0.33) / 0.67)


def _fade_in_curve(p: float) -> float:
    """Opacity for ENTER: 0 at p=0, smooth ramp, ~1 by two-thirds in (so the
    figure is solid as it settles, not still ghosting)."""
    if p >= 0.66:
        return 1.0
    return _clamp01(p / 0.66)


def edge_for_move(from_panel: str, to_panel: str, panels_geometry: dict | None = None):
    """Derive (exit_direction, enter_direction) from panel geometry.

    A figure leaving toward the destination exits the edge that faces it; it
    enters the destination from the edge that faces the source. With the default
    living-portraits layout (A left of B) an A->B move is ("right", "left") and a
    B->A move is ("left", "right").

    panels_geometry, if given, is {name: (x, y, w, h)} (e.g. built from
    player.PANELS); the facing edge is decided by comparing panel-centre x. If
    omitted, falls back to the canonical A-left/B-right convention by name.
    """
    if panels_geometry and from_panel in panels_geometry and to_panel in panels_geometry:
        fx, _, fw, _ = panels_geometry[from_panel]
        tx, _, tw, _ = panels_geometry[to_panel]
        from_cx = fx + fw / 2.0
        to_cx = tx + tw / 2.0
        if to_cx >= from_cx:           # destination is to the right
            return "right", "left"
        return "left", "right"         # destination is to the left
    # Name fallback: canonical layout has A on the left, B on the right.
    if (from_panel, to_panel) == ("B", "A"):
        return "left", "right"
    return "right", "left"


# --------------------------------------------------------------------------- #
# Transition state machine
# --------------------------------------------------------------------------- #
@dataclass
class CrossFrameState:
    """Tracks panel occupancy and drives one A<->B walk through its phases.

    Occupancy
    ---------
    `occupant` maps panel name -> character slug currently "living" in that panel
    (or None). The player seeds this from the beat (slugify(panels[X].char)) and
    this class updates it as a transition resolves: the moving slug leaves the
    from-panel when EXIT completes and takes up the to-panel when ENTER completes.
    A QA gate / the player can read `occupant` to know who belongs where.

    Driving a transition
    ---------------------
        x = CrossFrameState()
        x.set_occupant("A", "phineas")            # seed from the current beat
        ...
        x.begin_transition("phineas", "A", "B")   # once, when the move beat lands
        while not x.done:
            x.advance(dt)                          # dt seconds since last tick
            phase_a, prog_a = x.panel_state("A")
            phase_b, prog_b = x.panel_state("B")
            # ... render per the phase (see module docstring) ...

    `panel_state(name)` returns (phase, progress):
      - the from-panel reads (PHASE_EXIT, 0..1) during EXIT, then PHASE_DONE.
      - the to-panel reads (PHASE_ENTER, 0..1) during ENTER, PHASE_PRE/PHASE_GAP
        before that (figure suppressed), then PHASE_DONE.
      - a panel not involved in the move always reads (PHASE_DONE, 0.0) -> "render
        normally".
    """

    occupant: dict = field(default_factory=lambda: {"A": None, "B": None})

    # active-transition fields (None/idle when no walk in flight)
    _char: str | None = None
    _from: str | None = None
    _to: str | None = None
    _elapsed: float = 0.0
    _active: bool = False

    # ---- occupancy ----------------------------------------------------------
    def set_occupant(self, panel: str, slug: str | None) -> None:
        """Seed/override who lives in a panel (player calls this from the beat)."""
        self.occupant[panel] = slug

    def occupant_of(self, panel: str) -> str | None:
        return self.occupant.get(panel)

    # ---- lifecycle ----------------------------------------------------------
    def begin_transition(self, slug: str, from_panel: str, to_panel: str) -> None:
        """Start a walk of character `slug` from `from_panel` to `to_panel`.

        Idempotent for the in-flight move: re-calling with the same
        (slug, from, to) while active is a no-op (so the player can call it every
        tick it still sees the same `move` beat without restarting the walk).
        A *different* move supersedes the current one (begins fresh).
        """
        if (self._active and self._char == slug
                and self._from == from_panel and self._to == to_panel):
            return
        self._char = slug
        self._from = from_panel
        self._to = to_panel
        self._elapsed = 0.0
        self._active = True
        # The moving character is, from the first EXIT tick, no longer "settled"
        # anywhere; occupancy is re-established when ENTER completes. We leave the
        # from-panel occupant set until EXIT finishes so panel_state can suppress
        # correctly, but mark the to-panel as not-yet-occupied by this char.
        if self.occupant.get(from_panel) is None:
            self.occupant[from_panel] = slug  # tolerate an unseeded caller

    def advance(self, dt: float) -> None:
        """Advance the clock by `dt` seconds. Clamps at total duration and, on
        first reaching DONE, commits final occupancy (char now lives in to-panel,
        from-panel emptied). No-op when idle."""
        if not self._active:
            return
        self._elapsed += max(0.0, float(dt))
        if self._elapsed >= self._total():
            self._elapsed = self._total()
            self._commit_done()

    def reset(self) -> None:
        """Drop any in-flight transition (does not change committed occupancy)."""
        self._active = False
        self._char = None
        self._from = None
        self._to = None
        self._elapsed = 0.0

    # ---- queries ------------------------------------------------------------
    @property
    def active(self) -> bool:
        return self._active

    @property
    def done(self) -> bool:
        """True when no walk is in flight, or the in-flight one has elapsed."""
        return (not self._active) or (self._elapsed >= self._total())

    @property
    def moving_char(self) -> str | None:
        return self._char if self._active else None

    def phase(self) -> str:
        """The global phase of the in-flight walk (PHASE_DONE when idle)."""
        if not self._active:
            return PHASE_DONE
        e = self._elapsed
        if e <= 0.0:
            return PHASE_PRE
        if e < EXIT_S:
            return PHASE_EXIT
        if e < EXIT_S + GAP_S:
            return PHASE_GAP
        if e < self._total():
            return PHASE_ENTER
        return PHASE_DONE

    def panel_state(self, panel: str) -> tuple[str, float]:
        """(phase, progress) for one panel -- what the player switches on.

        Progress is 0..1 *within that panel's active phase* (EXIT for the
        from-panel, ENTER for the to-panel); 0.0 otherwise. A panel uninvolved in
        the current move -- or when idle -- reads (PHASE_DONE, 0.0).
        """
        if not self._active:
            return PHASE_DONE, 0.0
        g = self.phase()
        if panel == self._from:
            if g == PHASE_EXIT:
                return PHASE_EXIT, _clamp01(self._elapsed / EXIT_S)
            if g == PHASE_PRE:
                return PHASE_PRE, 0.0
            # GAP / ENTER / DONE: the figure has left this panel.
            return (PHASE_GAP if g in (PHASE_GAP, PHASE_ENTER) else PHASE_DONE), 0.0
        if panel == self._to:
            if g == PHASE_ENTER:
                t0 = EXIT_S + GAP_S
                return PHASE_ENTER, _clamp01((self._elapsed - t0) / ENTER_S)
            if g == PHASE_DONE:
                return PHASE_DONE, 0.0
            # PRE / EXIT / GAP: char hasn't arrived; suppress it here.
            return (PHASE_GAP if g in (PHASE_GAP, PHASE_EXIT) else PHASE_PRE), 0.0
        # Panel not in this move.
        return PHASE_DONE, 0.0

    def should_show_figure(self, panel: str) -> bool:
        """Convenience for the player: True iff THIS panel should render the moving
        figure layer this tick (EXIT on the from-panel, ENTER on the to-panel).
        When False during an active move, the player passes an empty layer for the
        moving character so the panel shows bg-plate-only (the empty frame)."""
        ph, _ = self.panel_state(panel)
        return ph in (PHASE_EXIT, PHASE_ENTER)

    # ---- internals ----------------------------------------------------------
    def _total(self) -> float:
        return EXIT_S + GAP_S + ENTER_S

    def _commit_done(self) -> None:
        """Move the character into the destination panel and clear the source."""
        if self._from is not None and self.occupant.get(self._from) == self._char:
            self.occupant[self._from] = None
        if self._to is not None:
            self.occupant[self._to] = self._char
        self._active = False


# Re-export the empty-layer helper under the name the docstring references, so a
# player can `from crossframe import empty_layer` for its GAP / suppressed ticks.
empty_layer = _empty_layer_for_rect


# --------------------------------------------------------------------------- #
# Headless self-test (numpy only -- no pygame, no opencv, no disk, no network)
# --------------------------------------------------------------------------- #
def _synth_cutout(size: int = 256) -> np.ndarray:
    """A fake RGBA cutout: an opaque blob LEFT-of-centre on transparency.

    Deliberately off-centre (centroid < width/2) so the self-test can assert the
    EXIT pushes the centroid rightward and the ENTER pulls it from the far left
    toward home -- a centred blob would make the direction checks ambiguous.
    """
    rgba = np.zeros((size, size, 4), np.uint8)
    cx, cy = int(size * 0.40), size // 2
    r = size // 6
    yy, xx = np.ogrid[:size, :size]
    blob = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    rgba[..., 0][blob] = 200          # warm fill so rgb is non-zero where opaque
    rgba[..., 1][blob] = 150
    rgba[..., 2][blob] = 120
    rgba[..., 3][blob] = 255
    return rgba


def _alpha_centroid_x(alpha: np.ndarray) -> float | None:
    """Mean x of the opaque mass (alpha>10), or None if the layer is empty."""
    m = alpha.astype(np.float32)
    total = float(m.sum())
    if total <= 0.0:
        return None
    xs = np.arange(alpha.shape[1], dtype=np.float32)[None, :]
    return float((m * xs).sum() / total)


def _opaque_fraction(alpha: np.ndarray) -> float:
    return float((alpha > 10).mean())


def _selftest() -> int:  # noqa: PLR0915  -- module self-test: a flat sequence of assertions, long by nature
    import math

    print("[crossframe] self-test", flush=True)
    panel_a = (0, 0, 256, 256)        # PANELS["A"]
    panel_b = (256, 0, 192, 192)      # PANELS["B"] -- different (near-square) size
    cut = _synth_cutout(256)

    # ---- layer shape + dtype matches the rig_frame contract -----------------
    rgb, alpha = exit_frame(panel_a, cut, 0.0, "right")
    assert rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8, (rgb.shape, rgb.dtype)
    assert alpha.shape == (256, 256) and alpha.dtype == np.uint8, (alpha.shape, alpha.dtype)
    # Panel B sizing: layer must be exactly the (h, w) of the *target* panel.
    rgb_b, alpha_b = enter_frame(panel_b, cut, 1.0, "left")
    assert rgb_b.shape == (192, 192, 3) and alpha_b.shape == (192, 192), \
        (rgb_b.shape, alpha_b.shape)

    # ---- EXIT (direction="right"): centroid moves RIGHT, alpha fades to 0 -----
    cx0 = _alpha_centroid_x(exit_frame(panel_a, cut, 0.0, "right")[1])
    cx_mid = _alpha_centroid_x(exit_frame(panel_a, cut, 0.5, "right")[1])
    _, a_end = exit_frame(panel_a, cut, 1.0, "right")
    assert cx0 is not None and cx_mid is not None, "EXIT figure vanished too early"
    assert cx_mid > cx0 + 1.0, f"EXIT centroid did not move right: {cx0:.1f}->{cx_mid:.1f}"
    assert _opaque_fraction(a_end) < 0.01, \
        f"EXIT did not clear the frame (opaque frac {_opaque_fraction(a_end):.3f})"
    # monotonic rightward through the slide (sample several progresses)
    prev = -1.0
    for p in (0.0, 0.2, 0.4, 0.6):
        c = _alpha_centroid_x(exit_frame(panel_a, cut, p, "right")[1])
        assert c is not None and c >= prev - 0.5, f"EXIT not monotonic at p={p}"
        prev = c

    # ---- EXIT direction="left": centroid moves LEFT instead ------------------
    # The synth blob sits left-of-centre, so sample early (p=0.2): by mid-slide a
    # left-exiting figure has already cleared the left edge (zero opaque mass).
    cxl0 = _alpha_centroid_x(exit_frame(panel_a, cut, 0.0, "left")[1])
    cxl_early = _alpha_centroid_x(exit_frame(panel_a, cut, 0.2, "left")[1])
    assert cxl0 is not None and cxl_early is not None, "EXIT-left figure vanished too early"
    assert cxl_early < cxl0 - 1.0, f"EXIT-left centroid did not move left: {cxl0:.1f}->{cxl_early:.1f}"
    # and by p=1 it has cleared the frame entirely
    assert _opaque_fraction(exit_frame(panel_a, cut, 1.0, "left")[1]) < 0.01, \
        "EXIT-left did not clear the frame"

    # ---- ENTER (direction="left"): figure comes IN from the left, centroid -> home
    # At p=0 the figure is a full overshoot off the left edge (clipped -> empty).
    # The left-of-centre blob only re-enters frame past the slide's midpoint, so
    # sample the inward march at p=0.6 vs p=1.0 (both have opaque mass).
    _, ain0 = enter_frame(panel_a, cut, 0.0, "left")
    cx_in_mid = _alpha_centroid_x(enter_frame(panel_a, cut, 0.6, "left")[1])
    cx_in_late = _alpha_centroid_x(enter_frame(panel_a, cut, 1.0, "left")[1])
    # at p=0 the figure is off-edge (mostly clipped) -> little/no opaque mass
    assert _opaque_fraction(ain0) < 0.05, "ENTER should start off-frame (near-empty)"
    assert cx_in_mid is not None and cx_in_late is not None, "ENTER figure never appeared"
    assert cx_in_late > cx_in_mid + 1.0, \
        f"ENTER centroid did not advance inward: {cx_in_mid:.1f}->{cx_in_late:.1f}"
    # at p=1 the entered figure equals the static (un-shifted, opaque) cutout
    home_ref = _prep_figure_rgba(cut, 256, 256)
    home_cx = _alpha_centroid_x(home_ref[..., 3])
    assert abs(cx_in_late - home_cx) < 2.0, \
        f"ENTER did not settle to home centroid: {cx_in_late:.1f} vs {home_cx:.1f}"

    # ---- edge_for_move: geometry + name fallback -----------------------------
    geom = {"A": panel_a, "B": panel_b}
    assert edge_for_move("A", "B", geom) == ("right", "left")
    assert edge_for_move("B", "A", geom) == ("left", "right")
    assert edge_for_move("A", "B", None) == ("right", "left")   # name fallback
    assert edge_for_move("B", "A", None) == ("left", "right")

    # ---- state machine: phase ordering A -> B --------------------------------
    x = CrossFrameState()
    x.set_occupant("A", "phineas")
    x.set_occupant("B", "seraphina")
    x.begin_transition("phineas", "A", "B")
    assert x.active and not x.done
    assert x.moving_char == "phineas"

    # idempotent re-begin of the same move must NOT restart the clock
    x.advance(0.30)
    e_before = x._elapsed
    x.begin_transition("phineas", "A", "B")   # same move again
    assert x._elapsed == e_before, "re-begin restarted an in-flight transition"

    # Walk the clock in small steps, recording the global phase sequence and
    # per-panel (phase, progress). Assert canonical ordering and that GAP shows
    # no figure on EITHER panel.
    dt = 1.0 / 30.0
    seq = []
    gap_ticks = 0
    enter_progress_seen = []
    exit_progress_seen = []
    steps = int(math.ceil((EXIT_S + GAP_S + ENTER_S) / dt)) + 5
    # reset and re-run cleanly (we consumed 0.30s above proving idempotency)
    x.reset()
    x.set_occupant("A", "phineas")
    x.set_occupant("B", "seraphina")
    x.begin_transition("phineas", "A", "B")
    for _ in range(steps):
        ph = x.phase()
        if not seq or seq[-1] != ph:
            seq.append(ph)
        pa, ga = x.panel_state("A")   # from-panel
        pb, gb = x.panel_state("B")   # to-panel
        if ph == PHASE_EXIT:
            assert pa == PHASE_EXIT, f"from-panel not EXIT during EXIT: {pa}"
            assert not x.should_show_figure("B"), "to-panel showed figure during EXIT"
            exit_progress_seen.append(ga)
        if ph == PHASE_GAP:
            gap_ticks += 1
            assert not x.should_show_figure("A"), "from-panel showed figure during GAP"
            assert not x.should_show_figure("B"), "to-panel showed figure during GAP"
        if ph == PHASE_ENTER:
            assert pb == PHASE_ENTER, f"to-panel not ENTER during ENTER: {pb}"
            assert not x.should_show_figure("A"), "from-panel showed figure during ENTER"
            enter_progress_seen.append(gb)
        x.advance(dt)

    # canonical phase order (PRE may or may not be observed depending on first tick)
    expected = [PHASE_EXIT, PHASE_GAP, PHASE_ENTER, PHASE_DONE]
    seq_no_pre = [p for p in seq if p != PHASE_PRE]
    assert seq_no_pre == expected, f"phase order wrong: {seq} (filtered {seq_no_pre})"
    assert gap_ticks >= 1, "GAP phase never observed (empty frame missing)"
    # EXIT progress should rise from ~0 toward ~1; ENTER likewise.
    assert exit_progress_seen and exit_progress_seen[0] < 0.2, "EXIT progress didn't start low"
    assert max(exit_progress_seen) > 0.8, "EXIT progress never approached 1"
    assert enter_progress_seen and enter_progress_seen[0] < 0.3, "ENTER progress didn't start low"
    assert max(enter_progress_seen) > 0.8, "ENTER progress never approached 1"

    # ---- occupancy committed on DONE -----------------------------------------
    assert x.done, "transition should be done after walking the full clock"
    assert x.occupant_of("A") is None, f"from-panel A not cleared: {x.occupant_of('A')}"
    assert x.occupant_of("B") == "phineas", f"to-panel B occupant wrong: {x.occupant_of('B')}"
    # a panel uninvolved (idle now) reads DONE/0
    assert x.panel_state("A") == (PHASE_DONE, 0.0)

    # ---- empty_layer helper composites to 'no figure' ------------------------
    erb, eab = empty_layer(panel_b)
    assert erb.shape == (192, 192, 3) and eab.shape == (192, 192)
    assert int(eab.sum()) == 0, "empty_layer alpha is not fully transparent"

    # ---- a B->A move uses the mirrored directions end-to-end -----------------
    y = CrossFrameState()
    y.set_occupant("B", "seraphina")
    y.begin_transition("seraphina", "B", "A")
    ed, en = edge_for_move("B", "A", geom)
    assert (ed, en) == ("left", "right")
    # drive to ENTER and confirm the to-panel (A) is the one showing the figure
    y.advance(EXIT_S + GAP_S + ENTER_S * 0.5)
    assert y.should_show_figure("A") and not y.should_show_figure("B"), \
        "B->A: A should be entering mid-ENTER"

    print("[crossframe] OK -- exit/enter layer shape + centroid motion + fade, "
          "phase order EXIT->GAP->ENTER->DONE, empty GAP on both panels, "
          "occupancy handoff, B->A mirror", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
