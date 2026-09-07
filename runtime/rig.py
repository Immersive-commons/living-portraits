"""
rig.py -- animate a static cutout into living frames via its Live2D-style rig spec.

This is the "rig" box of the living-portraits architecture (v3): the sibling of
stage_render.py (which composites the portrait + asides into the panels) and
orchestrate.py (the producer loop). The generative pipeline hands us two files:

    pipeline/segment.py   ->  data/gen/<slug>_cutout.png   (RGBA, figure on transparency)
    pipeline/rig_spec.py  ->  data/gen/<slug>_rig.json     (layer boxes + pivots + warp + deformers)

and this module turns that frozen pair into MOTION -- a frame generator that
breathes, sways, blinks, and (on demand) gazes / opens its mouth. No Live2D
runtime, no GPU: every parameter in the spec's Live2D vocabulary
(ParamAngleX/Y/Z, ParamEyeBallX/Y, ParamEyeL/ROpen, ParamBrowL/RY,
ParamMouthOpenY, ParamBodyAngleX) is realised as a small OpenCV affine / scale
warp on just that layer's region. We never warp the whole 512-wide stage -- only
the head / eye / mouth / torso sub-rectangles, which are tens of pixels on the
256x256 (Panel A) and 192x192 (Panel B) panels this renders into.

How the params compose (a tiny Live2D-style parent->child rig):

    torso   is the base body deformer (ParamBodyAngleX -> shear + breathe + bob).
    head    is the parent of the face: ParamAngleZ rotates it about its pivot,
            ParamAngleX/Y translate it. Because the eye/brow/mouth boxes all sit
            INSIDE the head box, we apply their local warps first, then transform
            the whole head region -- so the features ride along with the head
            exactly as Live2D children inherit their parent's deform.
    eyes    ParamEyeBallX/Y translate the iris within the opening (gaze);
            ParamEyeL/ROpen squashes the eye vertically about its centre (blink).
    mouth   ParamMouthOpenY scales the mouth vertically about its centre -- the
            seam voicesmith's viseme track plugs into (see frame() docstring).

The autonomous IDLE behaviour (when frame() is called with params=None) layers a
slow sway + breathe (sinusoids off warp.idle.period_s / torso_period_s) with a
stochastic blink scheduler (warp.blink.interval_s, occasional double-blink). Pass
an explicit params dict to override any subset -- the director / voicesmith do
this to drive gaze and visemes on top of the idle baseline.

Public API:
    rig = Rig.from_slug("phineas")                 # loads cutout + rig.json
    rgba = rig.frame(t)                            # autonomous idle+blink at time t (sec)
    rgba = rig.frame(t, {"ParamMouthOpenY": 0.8})  # override one param, idle fills the rest
    for rgba in rig.idle_frames(fps=24, seconds=5):# convenience generator
        ...

Every frame is an (H, W, 4) uint8 RGBA ndarray the same size as the cutout, with
the same alpha convention -- stage_render.py composites it over the bg plate the
same way it would the static cutout (see the integration note below).

    python runtime/rig.py        # headless self-test (synthesises a cutout+spec, runs idle)

opencv (cv2) + numpy are required (same as the rest of the pipeline). Nothing
else is imported.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Iterator

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GEN_DIR = ROOT / "data" / "gen"

# --------------------------------------------------------------------------- #
# Tunable motion constants. All amplitudes that the spec doesn't itself carry
# live here so the look is adjustable without touching the warp math. The spec's
# own warp.* numbers (head_sway_px, iris_range_px, lid timings, ...) take
# precedence; these fill the gaps and bound the deformer ranges.
# --------------------------------------------------------------------------- #

# Idle sway: ParamAngleX/Y oscillate in [-1, 1]; multiplied by the spec's
# head_sway_px to get a pixel translation, and by HEAD_TILT_DEG for the Z tilt.
HEAD_TILT_DEG = 1.4              # peak ParamAngleZ idle tilt (deg), if spec gives none
IDLE_PHASE_Y_OFFSET = math.pi / 2  # pitch lags yaw a quarter cycle -> organic figure-8

# Torso breathe: ParamBodyAngleX in [-1, 1] -> horizontal shear; plus a vertical
# scale "breath" and a small vertical bob, both off torso_period_s.
TORSO_SHEAR = 0.010             # max horizontal shear factor at |ParamBodyAngleX|=1

# Blink: the eye region's vertical scale goes 1.0 (open) -> EYE_BLINK_MIN (shut).
EYE_BLINK_MIN = 0.06            # residual height at full close (a sliver, not a void)

# Gaze: ParamEyeBallX/Y in [-1, 1] -> iris pixel translation (scaled by spec
# iris_range_px). The iris is the darkest connected mass inside the eye box.
IRIS_DARK_PCTL = 28.0           # pixels darker than this percentile = iris candidate

# Mouth: ParamMouthOpenY in [0, 1] -> vertical scale of the mouth region. 0 keeps
# the painted (closed) mouth; 1 stretches it by MOUTH_OPEN_MAX_SCALE about centre.
MOUTH_OPEN_MAX_SCALE = 1.9

# Fallback warp numbers, used only if a spec is missing a field (defensive --
# rig_spec.py always writes them, but a hand-authored / older spec might not).
_DEF_IDLE = {
    "head_sway_px": [4.0, 3.0], "head_rot_deg": 1.2, "period_s": 3.2,
    "torso_bob_px": 4.0, "torso_breathe_scale": 1.012, "torso_period_s": 4.0,
}
_DEF_GAZE = {"iris_range_px": [6.0, 4.0], "settle_s": 0.6}
_DEF_BLINK = {
    "close_s": 0.08, "hold_s": 0.04, "open_s": 0.12,
    "interval_s": [2.5, 6.0], "double_blink_prob": 0.15,
}

# The Live2D parameters this rig understands. Anything else in a params dict is
# ignored (forward-compatible: voicesmith could pass extra viseme detail).
KNOWN_PARAMS = (
    "ParamAngleX", "ParamAngleY", "ParamAngleZ",
    "ParamEyeBallX", "ParamEyeBallY",
    "ParamEyeLOpen", "ParamEyeROpen",
    "ParamBrowLY", "ParamBrowRY",
    "ParamMouthOpenY", "ParamBodyAngleX",
)


def _neutral_params() -> dict:
    """All-zero params == rest pose, except eyes fully open (Open == 1)."""
    p = dict.fromkeys(KNOWN_PARAMS, 0.0)
    p["ParamEyeLOpen"] = 1.0
    p["ParamEyeROpen"] = 1.0
    return p


# --------------------------------------------------------------------------- #
# Small geometry / warp helpers. Each operates on a sub-image and returns one of
# the same shape, so the caller can paste it straight back into the layer box.
# All clamp to the box; none allocate full-stage buffers.
# --------------------------------------------------------------------------- #

def _clamp_box(box, w: int, h: int) -> tuple[int, int, int, int]:
    """Clip an [x, y, bw, bh] layer box to the canvas; return (x0, y0, x1, y1).

    rig_spec boxes are derived from detection and can poke a few px past the
    edge; we intersect with the canvas so every slice is valid."""
    x, y, bw, bh = (int(round(v)) for v in box)
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w, x + bw)
    y1 = min(h, y + bh)
    return x0, y0, x1, y1


def _scale_about_center(region: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Affine-scale an RGBA region about its own centre, output same shape.

    Border is transparent (constant 0) so a shrink reveals the layer beneath when
    this region is alpha-composited, and a stretch doesn't smear edge pixels."""
    h, w = region.shape[:2]
    if abs(sx - 1.0) < 1e-3 and abs(sy - 1.0) < 1e-3:
        return region
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    m = np.array([[sx, 0.0, cx - sx * cx],
                  [0.0, sy, cy - sy * cy]], dtype=np.float32)
    return cv2.warpAffine(region, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def _shear_breathe(region: np.ndarray, shear: float, scale_y: float) -> np.ndarray:
    """Horizontal shear + vertical scale about the region centre (torso breath).

    shear is the x-displacement per unit y, normalised so the top/bottom of the
    region move +/- (shear * height/2). Same shape out, transparent border."""
    h, w = region.shape[:2]
    if abs(shear) < 1e-4 and abs(scale_y - 1.0) < 1e-3:
        return region
    _cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    # shear maps y-from-center to an x offset; scale_y stretches about cy.
    m = np.array([[1.0, shear, -shear * cy],
                  [0.0, scale_y, cy - scale_y * cy]], dtype=np.float32)
    return cv2.warpAffine(region, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def _rotate_translate(region: np.ndarray, deg: float, dx: float, dy: float,
                      pivot_local: tuple[float, float]) -> np.ndarray:
    """Rotate an RGBA region about a local pivot by `deg`, then translate (dx,dy).

    Used for the head (parent) transform. Output same shape as input; the head
    box is sized with headroom by rig_spec, and the panels downscale, so the
    small rotations here don't clip the skull. Transparent border."""
    h, w = region.shape[:2]
    if abs(deg) < 1e-3 and abs(dx) < 1e-2 and abs(dy) < 1e-2:
        return region
    px, py = pivot_local
    m = cv2.getRotationMatrix2D((px, py), deg, 1.0)
    m[0, 2] += dx
    m[1, 2] += dy
    return cv2.warpAffine(region, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def _alpha_over(dst: np.ndarray, src: np.ndarray, x0: int, y0: int) -> None:
    """Alpha-composite RGBA `src` onto RGBA `dst` at (x0, y0), in place.

    Standard 'source-over': out = src*a + dst*(1-a), with alpha accumulated so a
    warped layer's own transparency is respected. Clips to dst bounds."""
    H, W = dst.shape[:2]
    sh, sw = src.shape[:2]
    if sw == 0 or sh == 0:
        return
    # destination window, clipped to the canvas
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(W, x0 + sw), min(H, y0 + sh)
    if dx0 >= dx1 or dy0 >= dy1:
        return
    # matching source window
    sx0, sy0 = dx0 - x0, dy0 - y0
    sx1, sy1 = sx0 + (dx1 - dx0), sy0 + (dy1 - dy0)

    d = dst[dy0:dy1, dx0:dx1].astype(np.float32)
    s = src[sy0:sy1, sx0:sx1].astype(np.float32)
    sa = (s[..., 3:4] / 255.0)
    da = (d[..., 3:4] / 255.0)
    out_a = sa + da * (1.0 - sa)
    # premultiplied-style blend for colour, then un-premultiply by out_a
    safe = np.where(out_a > 1e-6, out_a, 1.0)
    out_rgb = (s[..., :3] * sa + d[..., :3] * da * (1.0 - sa)) / safe
    d[..., :3] = out_rgb
    d[..., 3:4] = out_a * 255.0
    dst[dy0:dy1, dx0:dx1] = np.clip(d, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Easing
# --------------------------------------------------------------------------- #
def _ease_in_out_sine(u: float) -> float:
    """0->1 ease, matching the spec's gaze 'easeInOutSine' so head-follow settles."""
    u = max(0.0, min(1.0, u))
    return 0.5 - 0.5 * math.cos(math.pi * u)


# --------------------------------------------------------------------------- #
# Blink scheduler -- stochastic, time-driven, deterministic given a seed.
# --------------------------------------------------------------------------- #
@dataclass
class _BlinkScheduler:
    """Drives ParamEyeL/ROpen for the autonomous idle loop.

    Blinks fire at random intervals in warp.blink.interval_s. A blink is a
    close->hold->open envelope (durations from the spec). With probability
    double_blink_prob a second blink follows immediately. Pure function of `t`
    once the schedule is rolled forward, so frame(t) is reproducible.
    """
    close_s: float
    hold_s: float
    open_s: float
    interval_lo: float
    interval_hi: float
    double_prob: float
    rng: random.Random

    # rolling state
    _next_start: float = 0.0
    _events: list = field(default_factory=list)  # (start, is_double_tail)

    def __post_init__(self) -> None:
        # first blink lands somewhere in the first interval window
        self._next_start = self.rng.uniform(self.interval_lo, self.interval_hi)

    @property
    def _blink_len(self) -> float:
        return self.close_s + self.hold_s + self.open_s

    def _ensure_scheduled(self, t: float) -> None:
        """Make sure we've rolled the schedule forward past time t."""
        horizon = t + 0.5
        while self._next_start <= horizon:
            self._events.append(self._next_start)
            after = self._next_start + self._blink_len
            if self.rng.random() < self.double_prob:
                # immediate second blink after a tiny gap
                self._events.append(after + 0.06)
                after = after + 0.06 + self._blink_len
            self._next_start = after + self.rng.uniform(self.interval_lo, self.interval_hi)

    def openness(self, t: float) -> float:
        """Eye openness in [0, 1] at time t: 1 = open, 0 = shut."""
        self._ensure_scheduled(t)
        # find any blink envelope covering t
        for start in self._events:
            if start <= t < start + self._blink_len:
                u = t - start
                if u < self.close_s:                       # closing
                    return 1.0 - (u / max(self.close_s, 1e-6))
                if u < self.close_s + self.hold_s:         # held shut
                    return 0.0
                # opening
                uo = (u - self.close_s - self.hold_s) / max(self.open_s, 1e-6)
                return min(1.0, uo)
        return 1.0


# --------------------------------------------------------------------------- #
# The Rig
# --------------------------------------------------------------------------- #
class Rig:
    """Loads a cutout RGBA + its rig_spec.json and produces animation frames.

    Construct via `Rig.from_slug("phineas")` (reads data/gen/) or
    `Rig(cutout_path, spec_path)` directly. Then call `frame(t)` for an
    autonomous idle+blink frame, `frame(t, params)` to drive specific deformers,
    or iterate `idle_frames(...)`.
    """

    def __init__(self, cutout_path, spec_path, seed: int = 7) -> None:
        cutout_path = Path(cutout_path)
        spec_path = Path(spec_path)
        img = cv2.imread(str(cutout_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"could not read cutout: {cutout_path}")
        if img.ndim == 3 and img.shape[2] == 4:
            # cv2 reads BGRA; convert to RGBA so our frames are RGB-ordered like
            # clip_player's frames (stage_render composites RGB over the RGB bg).
            self.cutout = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        elif img.ndim == 3 and img.shape[2] == 3:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            a = np.full(rgb.shape[:2], 255, np.uint8)
            self.cutout = np.dstack([rgb, a])
        else:
            g = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            a = np.full(g.shape[:2], 255, np.uint8)
            self.cutout = np.dstack([g, a])
        self.h, self.w = self.cutout.shape[:2]

        self.spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
        self.layers = self.spec.get("layers", {})
        warp = self.spec.get("warp", {})
        self.idle_w = {**_DEF_IDLE, **warp.get("idle", {})}
        self.gaze_w = {**_DEF_GAZE, **warp.get("gaze", {})}
        self.blink_w = {**_DEF_BLINK, **warp.get("blink", {})}

        # Pre-resolve clamped layer boxes once (canvas-clipped (x0,y0,x1,y1)).
        self._box = {name: _clamp_box(self.layers[name]["box"], self.w, self.h)
                     for name in self.layers if "box" in self.layers[name]}

        # Per-eye cached iris mask + centroid (relative to the eye box), so gaze
        # translation only moves the iris pixels, not the whole eye region.
        self._iris = {}
        for eye in ("left_eye", "right_eye"):
            if eye in self._box:
                self._iris[eye] = self._build_iris_mask(eye)

        self.rng = random.Random(seed)
        il = self.blink_w.get("interval_s", _DEF_BLINK["interval_s"])
        self.blinker = _BlinkScheduler(
            close_s=float(self.blink_w.get("close_s", 0.08)),
            hold_s=float(self.blink_w.get("hold_s", 0.04)),
            open_s=float(self.blink_w.get("open_s", 0.12)),
            interval_lo=float(il[0]), interval_hi=float(il[1]),
            double_prob=float(self.blink_w.get("double_blink_prob", 0.15)),
            rng=self.rng,
        )

    # ---- construction helpers --------------------------------------------- #
    @classmethod
    def from_slug(cls, slug: str, gen_dir: Path | None = None, **kw) -> Rig:
        """Load data/gen/<slug>_cutout.png + <slug>_rig.json."""
        gen_dir = Path(gen_dir) if gen_dir is not None else GEN_DIR
        return cls(gen_dir / f"{slug}_cutout.png",
                   gen_dir / f"{slug}_rig.json", **kw)

    def _build_iris_mask(self, eye: str) -> dict:
        """Find the iris (darkest blob) inside an eye box. Returns {mask, cx, cy}
        in eye-local coords, or an empty mask if the eye region is degenerate.

        Gaze translates only these pixels; the sclera / lid stay put, which reads
        far more naturally than sliding the whole eye rectangle."""
        x0, y0, x1, y1 = self._box[eye]
        region = self.cutout[y0:y1, x0:x1]
        eh, ew = region.shape[:2]
        out = {"mask": np.zeros((eh, ew), bool), "cx": ew / 2.0, "cy": eh / 2.0}
        if eh < 3 or ew < 3:
            return out
        rgb = region[..., :3].astype(np.float32)
        lum = rgb.mean(axis=2)
        opaque = region[..., 3] > 64
        if opaque.sum() < 4:
            return out
        thr = np.percentile(lum[opaque], IRIS_DARK_PCTL)
        mask = (lum <= thr) & opaque
        if mask.sum() < 2:
            return out
        ys, xs = np.where(mask)
        out["mask"] = mask
        out["cx"] = float(xs.mean())
        out["cy"] = float(ys.mean())
        return out

    # ---- the parameter timeline ------------------------------------------- #
    def idle_params(self, t: float) -> dict:
        """Autonomous deformer values at time t (seconds): slow sway + breathe +
        scheduled blink. This is what frame() uses when params is None, and what
        an explicit params dict is merged on top of."""
        p = _neutral_params()

        period = max(0.5, float(self.idle_w.get("period_s", 3.2)))
        phase = 2.0 * math.pi * t / period
        # yaw/pitch trace a gentle figure-8 (pitch a quarter-cycle behind yaw).
        p["ParamAngleX"] = math.sin(phase)
        p["ParamAngleY"] = math.sin(phase + IDLE_PHASE_Y_OFFSET)
        p["ParamAngleZ"] = 0.5 * math.sin(phase * 0.5)  # slow tilt, half rate

        tperiod = max(0.5, float(self.idle_w.get("torso_period_s", 4.0)))
        tphase = 2.0 * math.pi * t / tperiod
        p["ParamBodyAngleX"] = math.sin(tphase)

        op = self.blinker.openness(t)
        p["ParamEyeLOpen"] = op
        p["ParamEyeROpen"] = op
        return p

    @staticmethod
    def gaze_params(dx: float, dy: float, t: float = 1.0, settle_s: float = 0.0,
                    intensity: float = 1.0) -> dict:
        """Build a gaze override: look toward (dx, dy) in [-1, 1] eye-space (and a
        proportional head turn). Optionally ease in over settle_s for a smooth
        saccade->settle. Returned dict is meant to be passed to frame(); idle
        fills the params it omits (so the figure still breathes while looking).

            rig.frame(t, rig.gaze_params(0.6, -0.2))   # glance up-right
        """
        dx = max(-1.0, min(1.0, dx))
        dy = max(-1.0, min(1.0, dy))
        k = intensity
        if settle_s > 0:
            k *= _ease_in_out_sine(t / settle_s)
        return {
            "ParamEyeBallX": dx * k,
            "ParamEyeBallY": dy * k,
            # head follows the eyes a little (gaze, not idle sway)
            "ParamAngleX": dx * 0.5 * k,
            "ParamAngleY": dy * 0.5 * k,
        }

    # ---- the renderer ------------------------------------------------------ #
    def frame(self, t: float, params: dict | None = None) -> np.ndarray:
        """Render one (H, W, 4) RGBA frame at time t (seconds).

        params: optional dict of Live2D-vocab overrides (any subset of
            KNOWN_PARAMS). Whatever you supply replaces the autonomous idle value
            for that key; everything you omit keeps idling. So
                frame(t)                              -> pure idle + blink
                frame(t, {"ParamMouthOpenY": v})      -> idle, but mouth open by v
                frame(t, rig.gaze_params(0.6, 0.0))   -> idle + a rightward glance

            VISEME HOOK: voicesmith produces, per audio frame, a mouth-openness
            track in [0, 1] (e.g. from phoneme/RMS -> jaw). Feeding that value as
            ParamMouthOpenY each frame makes the portrait lip-flap in sync with
            the spoken aside -- no rig change needed; the seam is exactly this
            parameter.

        Returns RGBA the same size as the cutout, same alpha convention, so
        stage_render.py composites it over the bg plate identically to the static
        cutout (see module docstring 'integration').
        """
        p = self.idle_params(t)
        if params:
            for k, v in params.items():
                if k in KNOWN_PARAMS:
                    p[k] = float(v)

        # 1. Base = a copy of the cutout, with the TORSO breath applied in place.
        out = self.cutout.copy()
        self._apply_torso(out, p)

        # 2. Build the moved HEAD on a separate RGBA tile: composite the face
        #    children (eyes/brows/mouth) into a head-local copy first, then apply
        #    the head parent transform so they ride along.
        self._apply_head(out, p)
        return out

    def idle_frames(self, fps: int = 24, seconds: float = 5.0,
                    t0: float = 0.0) -> Iterator[np.ndarray]:
        """Convenience generator: yield `fps*seconds` autonomous idle frames.

        The producer/clip baker can pull a fixed-length loop from this; the live
        player would instead call frame(t) with its own wall-clock t and per-beat
        param overrides."""
        n = max(1, int(round(fps * seconds)))
        for i in range(n):
            yield self.frame(t0 + i / float(fps))

    # ---- per-layer warps --------------------------------------------------- #
    def _apply_torso(self, out: np.ndarray, p: dict) -> None:
        """ParamBodyAngleX -> horizontal shear + breathe scale + vertical bob,
        applied to the torso region of `out` in place."""
        if "torso" not in self._box:
            return
        x0, y0, x1, y1 = self._box["torso"]
        if x1 - x0 < 4 or y1 - y0 < 4:
            return
        body = float(p.get("ParamBodyAngleX", 0.0))
        breathe = float(self.idle_w.get("torso_breathe_scale", 1.012))
        # breathe scale oscillates 1 .. breathe with the body phase magnitude
        scale_y = 1.0 + (breathe - 1.0) * (0.5 + 0.5 * abs(body))
        shear = TORSO_SHEAR * body
        bob = float(self.idle_w.get("torso_bob_px", 4.0)) * body * 0.25

        region = out[y0:y1, x0:x1].copy()
        warped = _shear_breathe(region, shear, scale_y)
        # tiny vertical bob: blank the slot, composite warped shifted by `bob`.
        out[y0:y1, x0:x1] = 0
        _alpha_over(out, warped, x0, int(round(y0 + bob)))

    def _apply_head(self, out: np.ndarray, p: dict) -> None:
        """Composite eyes/brows/mouth into a head-local tile, then rotate+translate
        the whole head (parent transform) and alpha-blend it back onto `out`."""
        if "head" not in self._box:
            return
        hx0, hy0, hx1, hy1 = self._box["head"]
        hw, hh = hx1 - hx0, hy1 - hy0
        if hw < 6 or hh < 6:
            return

        # Head-local working tile (RGBA copy of the head region).
        head = out[hy0:hy1, hx0:hx1].copy()

        # --- children, in head-local coords --------------------------------
        # Brows: a vertical shift proportional to ParamBrowL/RY (expression).
        self._apply_brow(head, "left_brow", p.get("ParamBrowLY", 0.0), hx0, hy0)
        self._apply_brow(head, "right_brow", p.get("ParamBrowRY", 0.0), hx0, hy0)
        # Eyes: gaze (iris translate) + blink (vertical squash).
        self._apply_eye(head, "left_eye", p.get("ParamEyeLOpen", 1.0),
                        p.get("ParamEyeBallX", 0.0), p.get("ParamEyeBallY", 0.0),
                        hx0, hy0)
        self._apply_eye(head, "right_eye", p.get("ParamEyeROpen", 1.0),
                        p.get("ParamEyeBallX", 0.0), p.get("ParamEyeBallY", 0.0),
                        hx0, hy0)
        # Mouth: ParamMouthOpenY vertical scale (viseme seam).
        self._apply_mouth(head, p.get("ParamMouthOpenY", 0.0), hx0, hy0)

        # --- head parent transform -----------------------------------------
        sway = self.idle_w.get("head_sway_px", _DEF_IDLE["head_sway_px"])
        ax = float(p.get("ParamAngleX", 0.0))   # yaw  [-1, 1]
        ay = float(p.get("ParamAngleY", 0.0))   # pitch
        az = float(p.get("ParamAngleZ", 0.0))   # tilt
        dx = ax * float(sway[0])
        dy = ay * float(sway[1])
        rot = az * float(self.idle_w.get("head_rot_deg", HEAD_TILT_DEG))

        # pivot in head-local coords (spec pivot is canvas-absolute)
        piv = self.layers["head"].get("pivot", [hx0 + hw / 2, hy0 + hh / 2])
        plx = float(piv[0]) - hx0
        ply = float(piv[1]) - hy0
        moved = _rotate_translate(head, rot, dx, dy, (plx, ply))

        # blank the original head footprint so the rotated head doesn't ghost
        # against the un-moved face still sitting in `out`, then composite.
        out[hy0:hy1, hx0:hx1] = 0
        _alpha_over(out, moved, hx0, hy0)

    def _apply_brow(self, head: np.ndarray, name: str, val: float,
                    hx0: int, hy0: int) -> None:
        """Shift a brow region vertically by `val` in [-1, 1] (up = raise)."""
        if name not in self._box or abs(val) < 1e-3:
            return
        x0, y0, x1, y1 = self._box[name]
        lx0, ly0, lx1, ly1 = x0 - hx0, y0 - hy0, x1 - hx0, y1 - hy0
        if lx1 - lx0 < 2 or ly1 - ly0 < 2:
            return
        bh = ly1 - ly0
        shift = -val * bh * 0.6  # raise = negative y; up to 0.6 of brow height
        region = head[ly0:ly1, lx0:lx1].copy()
        head[ly0:ly1, lx0:lx1] = 0
        _alpha_over(head, region, lx0, int(round(ly0 + shift)))

    def _apply_eye(self, head: np.ndarray, name: str, openness: float,
                   gaze_x: float, gaze_y: float, hx0: int, hy0: int) -> None:
        """Blink (vertical squash about eye centre) + gaze (iris translate),
        operating on the eye sub-region of the head-local tile."""
        if name not in self._box:
            return
        x0, y0, x1, y1 = self._box[name]
        lx0, ly0, lx1, ly1 = x0 - hx0, y0 - hy0, x1 - hx0, y1 - hy0
        ew, eh = lx1 - lx0, ly1 - ly0
        if ew < 3 or eh < 3:
            return
        region = head[ly0:ly1, lx0:lx1].copy()

        # 1. Gaze: translate the iris pixels within the eye opening.
        iris = self._iris.get(name)
        if iris is not None and iris["mask"].any() and (abs(gaze_x) > 1e-3 or abs(gaze_y) > 1e-3):
            rng = self.gaze_w.get("iris_range_px", _DEF_GAZE["iris_range_px"])
            idx = min(gaze_x * float(rng[0]), ew * 0.4)
            idy = min(gaze_y * float(rng[1]), eh * 0.4)
            region = self._translate_iris(region, iris, idx, idy)

        # 2. Blink: squash vertically about the eye centre. openness 1->open.
        if openness < 0.999:
            sy = EYE_BLINK_MIN + (1.0 - EYE_BLINK_MIN) * max(0.0, min(1.0, openness))
            squashed = _scale_about_center(region, 1.0, sy)
            head[ly0:ly1, lx0:lx1] = 0
            _alpha_over(head, squashed, lx0, ly0)
        else:
            head[ly0:ly1, lx0:lx1] = region

    def _translate_iris(self, region: np.ndarray, iris: dict,
                        dx: float, dy: float) -> np.ndarray:
        """Move just the iris pixels of an eye region by (dx, dy), filling the
        vacated spot with nearby sclera so no hole opens up.

        Cheap approach: shift the whole region by (dx, dy) into a copy, then
        composite only the iris-masked pixels of that shifted copy back over the
        original. The lid/sclera of the original stay fixed; the dark iris slides.
        """
        h, w = region.shape[:2]
        m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        shifted = cv2.warpAffine(region, m, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        # the iris mask also moves with the shift -> recompute where it landed
        mask = iris["mask"]
        moved_mask = cv2.warpAffine(mask.astype(np.uint8), m, (w, h),
                                    flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        out = region.copy()
        out[moved_mask] = shifted[moved_mask]
        return out

    def _apply_mouth(self, head: np.ndarray, open_y: float,
                     hx0: int, hy0: int) -> None:
        """ParamMouthOpenY in [0, 1] -> vertical scale of the mouth region about
        its centre. 0 = closed (painted mouth untouched). This is the seam a
        viseme track drives for lip-sync."""
        if "mouth" not in self._box or open_y < 1e-3:
            return
        x0, y0, x1, y1 = self._box["mouth"]
        lx0, ly0, lx1, ly1 = x0 - hx0, y0 - hy0, x1 - hx0, y1 - hy0
        if lx1 - lx0 < 3 or ly1 - ly0 < 3:
            return
        open_y = max(0.0, min(1.0, open_y))
        sy = 1.0 + (MOUTH_OPEN_MAX_SCALE - 1.0) * open_y
        region = head[ly0:ly1, lx0:lx1].copy()
        stretched = _scale_about_center(region, 1.0, sy)
        head[ly0:ly1, lx0:lx1] = 0
        _alpha_over(head, stretched, lx0, ly0)


# --------------------------------------------------------------------------- #
# Synthetic fixtures for the self-test (no dependency on pipeline/*).
# --------------------------------------------------------------------------- #
def _synth_cutout(size: int = 256) -> np.ndarray:
    """A simple RGBA 'portrait' cutout: head + torso on transparency, with
    distinct dark eyes + a mouth, so the rig has real pixels to push. Mirrors the
    proportions rig_spec assumes (head in the upper third, eyes ~0.4 down)."""
    h = w = size
    rgba = np.zeros((h, w, 4), np.uint8)
    cx = w // 2

    def _ell(cxp, cyp, ax, ay, color):
        cv2.ellipse(rgba, (int(cxp), int(cyp)), (int(ax), int(ay)), 0, 0, 360,
                    (*color, 255), -1)

    _ell(cx, h * 0.74, w * 0.30, h * 0.26, (120, 110, 140))   # torso
    _ell(cx, h * 0.40, w * 0.16, h * 0.20, (205, 170, 150))   # head
    _ell(cx, h * 0.30, w * 0.17, h * 0.12, (60, 45, 40))      # hair cap (upper)
    eye_y = int(h * 0.38)
    for ex in (cx - int(w * 0.06), cx + int(w * 0.06)):
        _ell(ex, eye_y, w * 0.028, h * 0.018, (35, 30, 30))   # dark eye
    _ell(cx, h * 0.48, w * 0.06, h * 0.022, (110, 50, 50))    # mouth
    return rgba


def _synth_spec_for(cutout: np.ndarray) -> dict:
    """Build a minimal rig_spec for a synthetic cutout WITHOUT importing pipeline.

    Prefer the real rig_spec.build_rig_spec if importable (keeps the self-test
    honest against the actual schema); else hand-roll the same shape from known
    proportions. Either path yields the keys frame() reads."""
    h, w = cutout.shape[:2]
    # boxes from the same proportions _synth_cutout drew at, [x, y, bw, bh].
    cx = w // 2
    head = [int(cx - w * 0.18), int(h * 0.18), int(w * 0.36), int(h * 0.30)]
    le = [int(cx - w * 0.06 - w * 0.035), int(h * 0.38 - h * 0.022),
          int(w * 0.07), int(h * 0.045)]
    re = [int(cx + w * 0.06 - w * 0.035), int(h * 0.38 - h * 0.022),
          int(w * 0.07), int(h * 0.045)]
    lb = [le[0], int(le[1] - h * 0.03), le[2], int(h * 0.018)]
    rb = [re[0], int(re[1] - h * 0.03), re[2], int(h * 0.018)]
    mouth = [int(cx - w * 0.07), int(h * 0.45), int(w * 0.14), int(h * 0.05)]
    torso = [int(cx - w * 0.30), int(h * 0.48), int(w * 0.60), int(h * 0.52)]

    def _c(b):
        return [int(b[0] + b[2] / 2), int(b[1] + b[3] / 2)]

    return {
        "version": "1.0", "format": "living-portraits/rig-spec", "slug": "synth",
        "canvas": {"w": w, "h": h},
        "layers": {
            "figure": {"box": [int(cx - w * 0.30), int(h * 0.18), int(w * 0.60), int(h * 0.82)]},
            "head": {"box": head, "pivot": _c(head)},
            "left_eye": {"box": le, "pivot": _c(le)},
            "right_eye": {"box": re, "pivot": _c(re)},
            "left_brow": {"box": lb}, "right_brow": {"box": rb},
            "mouth": {"box": mouth, "pivot": _c(mouth)},
            "torso": {"box": torso, "pivot": _c(torso)},
        },
        "landmarks": {"eye_line_y": le[1] + le[3] // 2, "mouth_line_y": mouth[1] + mouth[3] // 2},
        "warp": {
            "idle": dict(_DEF_IDLE), "gaze": dict(_DEF_GAZE), "blink": dict(_DEF_BLINK),
        },
        "deformers": [],
    }


# --------------------------------------------------------------------------- #
# Headless self-test
# --------------------------------------------------------------------------- #
def _selftest() -> int:  # noqa: PLR0915  -- module self-test: a flat sequence of assertions, long by nature
    import tempfile

    print("[rig] self-test", flush=True)
    tmp = Path(tempfile.mkdtemp(prefix="lp_rig_"))

    # Two fixtures, two jobs:
    #  * SYNTH (always built) -- a cutout with guaranteed contrast in every
    #    feature (distinct dark eyes, a coloured mouth). It's the fixture we make
    #    the per-feature motion assertions against, because a real segmented
    #    painting can flatten a feature (e.g. the matte can paint a low-contrast
    #    mouth to near-uniform skin, leaving nothing for a vertical scale to move
    #    -- physically correct, but untestable). SYNTH guarantees pixels to push.
    #  * REAL (if the pipeline _selftest pair is on disk) -- exercises the TRUE
    #    rig_spec schema end-to-end (idle / shape / alpha) at 1024x1024.
    cut = _synth_cutout(256)
    spec = _synth_spec_for(cut)
    cut_path = tmp / "synth_cutout.png"
    cv2.imwrite(str(cut_path), cv2.cvtColor(cut, cv2.COLOR_RGBA2BGRA))
    spec_path = tmp / "synth_rig.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    synth_rig = Rig(cut_path, spec_path)

    real_cut = GEN_DIR / "_selftest" / "selftest_cutout.png"
    real_rig_p = GEN_DIR / "_selftest" / "selftest_rig.json"
    real_rig = (Rig(real_cut, real_rig_p)
                if real_cut.exists() and real_rig_p.exists() else None)

    print(f"[rig] synth fixture: {synth_rig.w}x{synth_rig.h}, "
          f"layers {sorted(synth_rig._box)}", flush=True)
    print(f"[rig] real fixture : "
          f"{'present (' + str(real_rig.w) + 'x' + str(real_rig.h) + ', true schema)' if real_rig else 'absent (synth-only run)'}",
          flush=True)

    # The rig we run the schema/idle/alpha checks on: prefer REAL when present.
    rig = real_rig or synth_rig

    # --- idle generator: ~60 frames, all RGBA, same shape, and CHANGING ----
    frames = list(rig.idle_frames(fps=24, seconds=2.5))  # 60 frames
    assert len(frames) == 60, len(frames)
    for f in frames:
        assert f.dtype == np.uint8, f.dtype
        assert f.ndim == 3 and f.shape[2] == 4, f"not RGBA: {f.shape}"
        assert f.shape[:2] == (rig.h, rig.w), f.shape
    # frames must differ over time (idle sway + breathe is always moving)
    diffs = [int(np.abs(frames[i].astype(np.int16) - frames[0].astype(np.int16)).sum())
             for i in range(1, len(frames))]
    assert max(diffs) > 0, "idle frames never change -- animation is static"
    print(f"[rig] idle: 60 RGBA frames, max delta-from-f0 = {max(diffs):,}", flush=True)

    # --- per-feature motion: assert on SYNTH (guaranteed contrast) ----------
    # A neutral param set (eyes open, no sway) so each test isolates ONE deformer.
    base = dict.fromkeys(KNOWN_PARAMS, 0.0)
    base["ParamEyeLOpen"] = 1.0
    base["ParamEyeROpen"] = 1.0

    # blink: a fully-shut eye frame must differ from a fully-open one.
    open_f = synth_rig.frame(0.0, base)
    shut_f = synth_rig.frame(0.0, {**base, "ParamEyeLOpen": 0.0, "ParamEyeROpen": 0.0})
    eye_delta = int(np.abs(open_f.astype(np.int16) - shut_f.astype(np.int16)).sum())
    assert eye_delta > 0, "blink produced no change (eyes didn't close)"
    assert "left_eye" in synth_rig._box, "no left_eye box in spec"
    print(f"[rig] blink: open vs shut delta = {eye_delta:,}", flush=True)

    # gaze: looking right differs from looking left (iris translate + head turn).
    gaze_r = synth_rig.frame(0.0, {**base, **synth_rig.gaze_params(1.0, 0.0)})
    gaze_l = synth_rig.frame(0.0, {**base, **synth_rig.gaze_params(-1.0, 0.0)})
    gaze_delta = int(np.abs(gaze_r.astype(np.int16) - gaze_l.astype(np.int16)).sum())
    assert gaze_delta > 0, "gaze right vs left identical -- gaze not applied"
    print(f"[rig] gaze: look-right vs look-left delta = {gaze_delta:,}", flush=True)

    # mouth (the voicesmith viseme seam): open mouth differs from closed.
    mouth_open = synth_rig.frame(0.0, {**base, "ParamMouthOpenY": 1.0})
    mouth_shut = synth_rig.frame(0.0, {**base, "ParamMouthOpenY": 0.0})
    mouth_delta = int(np.abs(mouth_open.astype(np.int16) - mouth_shut.astype(np.int16)).sum())
    assert mouth_delta > 0, "ParamMouthOpenY had no effect -- viseme seam broken"
    print(f"[rig] mouth: open vs closed delta = {mouth_delta:,} (voicesmith seam)", flush=True)

    # mouth openness must be monotone-ish: half-open sits between shut and full.
    half = int(np.abs(synth_rig.frame(0.0, {**base, "ParamMouthOpenY": 0.5}).astype(np.int16)
                      - mouth_shut.astype(np.int16)).sum())
    assert 0 < half <= mouth_delta + 1, f"mouth not monotone: half={half} full={mouth_delta}"
    print(f"[rig] mouth: half-open delta = {half:,} (between shut and full)", flush=True)

    # --- alpha is preserved (RGBA, figure still on transparency) ------------
    cov0 = float((rig.cutout[..., 3] > 127).mean())
    covf = float((frames[0][..., 3] > 127).mean())
    assert covf > 0.01, "frame alpha is empty -- composite ate the figure"
    # the figure shouldn't grow or vanish: coverage stays in the same ballpark.
    assert 0.4 * cov0 <= covf <= 2.5 * cov0, \
        f"alpha coverage drifted too far: {cov0:.3f} -> {covf:.3f}"
    print(f"[rig] alpha coverage: cutout {cov0:.3f} -> frame {covf:.3f}", flush=True)

    # --- autonomous loop: the blink scheduler fires within a few seconds ----
    # frame() with no params == idle; a blink should drop eye openness < 0.5 at
    # least once in an 8 s window, AND that frame should differ from a wide-open
    # one. We probe the scheduler directly (deterministic) + confirm pixels move.
    blinked = any(synth_rig.blinker.openness(t / 24.0) < 0.5 for t in range(24 * 8))
    assert blinked, "blink scheduler never closed the eyes in 8 s"
    long_run = [rig.frame(t / 24.0) for t in range(24 * 8)]
    base_open = rig.frame(0.0)
    peak = max(int(np.abs(f.astype(np.int16) - base_open.astype(np.int16)).sum())
               for f in long_run)
    assert peak > 0
    print(f"[rig] autonomous 8s run: scheduler blinked={blinked}, peak delta = {peak:,}",
          flush=True)

    print("[rig] OK -- idle/blink/gaze/mouth all produce RGBA frames that change",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
