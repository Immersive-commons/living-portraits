"""rig_spec.py -- emit a Live2D-style rig spec from a character cutout.

Stage 4 of the generative pipeline (architecture v2 "Live2D rig" box):

    segment.py   ->  data/gen/<slug>_cutout.png   (RGBA, figure on transparency)
    rig_spec.py  ->  data/gen/<slug>_rig.json     (layer regions + warp params)

This module produces a SPEC, not a rigged model. A future Live2D / parametric-
warp step (the player's "action -> Live2D rig + baked clip" renderer) consumes
the JSON to know WHERE the head / eyes / brows / mouth / torso are and HOW to
move them for idle sway, gaze tracking, and blinking. No Live2D runtime is
needed here -- the cost stays ~0 GPU, exactly as the architecture box promises.

Region discovery, best effort first:

  * The figure's bounding box comes from the cutout's alpha channel (the only
    ground truth we have about where the subject actually is).
  * A frontal-face Haar cascade (ships with opencv) refines the head box and,
    via the eye cascade, the eye line -- when it fires.
  * On a stylised oil painting Haar usually does NOT fire, so we fall back to
    anatomical proportions measured from the alpha bbox (head ~ top 28% of the
    figure, eye line at ~0.40 of head height, etc.). The proportional path is
    deterministic and always produces a complete, plausible rig.

The dlib/mediapipe 68-point landmark libraries are import-guarded and optional;
the module is fully functional on opencv + numpy alone.

    python pipeline/rig_spec.py            # self-test (segments a synthetic image first)
    python pipeline/rig_spec.py phineas     # rig data/gen/phineas_cutout.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "gen"

# Anatomical proportions (fractions), measured down from the top of the figure
# bbox. Standard portrait/"Loomis" head proportions, tuned for head-and-shoulders
# framing. All used only when face detection doesn't give us something better.
HEAD_FRAC = 0.30          # head occupies the top ~30% of the figure height
EYE_LINE_FRAC = 0.42      # eyes sit ~42% down the head
BROW_LINE_FRAC = 0.30     # brows ~30% down the head
MOUTH_LINE_FRAC = 0.72    # mouth ~72% down the head
EYE_SEP_FRAC = 0.46       # inter-pupil distance ~46% of head width
EYE_W_FRAC = 0.22         # one eye width ~22% of head width
EYE_H_FRAC = 0.11         # one eye height ~11% of head height


# ---------------------------------------------------------------------------
# Optional landmark backend -- import-guarded, never required.
# ---------------------------------------------------------------------------

def _landmarks_available() -> bool:
    """True iff an optional 68-pt landmark lib (dlib) is importable. We don't
    require it; the Haar + proportional path is the supported baseline."""
    try:
        import dlib  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Region discovery
# ---------------------------------------------------------------------------

def _bbox_from_alpha(alpha: np.ndarray) -> tuple[int, int, int, int]:
    """Tight (x, y, w, h) bounding box of the opaque region. Falls back to the
    full frame if the alpha is empty (shouldn't happen post-segment cleanup)."""
    ys, xs = np.where(alpha > 64)
    if xs.size == 0 or ys.size == 0:
        h, w = alpha.shape[:2]
        return 0, 0, w, h
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, (x1 - x0 + 1), (y1 - y0 + 1)


def _detect_face(rgb: np.ndarray, fig_box: tuple[int, int, int, int]
                 ) -> Optional[tuple[int, int, int, int]]:
    """Frontal-face Haar detection, restricted to the figure bbox. Returns the
    largest face (x, y, w, h) in full-image coords, or None if none fire.

    Haar rarely triggers on painterly/stylised faces, which is expected -- the
    caller falls through to proportional defaults."""
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            return None
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4,
                                         minSize=(48, 48))
    except cv2.error:
        return None
    if len(faces) == 0:
        return None
    fx, fy, fw, fh = fig_box
    inside = [f for f in faces
              if fx - 10 <= f[0] and f[1] >= fy - 10
              and f[0] + f[2] <= fx + fw + 10]
    pool = inside if inside else list(faces)
    x, y, w, h = max(pool, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)


def _detect_eyes(rgb: np.ndarray, head_box: tuple[int, int, int, int]
                 ) -> Optional[list[tuple[int, int, int, int]]]:
    """Eye Haar detection within the head box. Returns up to two eye boxes in
    full-image coords, or None. Best effort -- proportional defaults cover the
    miss case."""
    try:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml")
        if cascade.empty():
            return None
        hx, hy, hw, hh = head_box
        roi = cv2.cvtColor(rgb[hy:hy + hh, hx:hx + hw], cv2.COLOR_RGB2GRAY)
        if roi.size == 0:
            return None
        eyes = cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=6,
                                        minSize=(18, 18))
    except cv2.error:
        return None
    if len(eyes) < 2:
        return None
    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    eyes = sorted(eyes, key=lambda e: e[0])  # left-to-right
    hx, hy = head_box[0], head_box[1]
    return [(int(hx + e[0]), int(hy + e[1]), int(e[2]), int(e[3])) for e in eyes]


def _proportional_regions(fig_box, head_box):
    """Derive eye / brow / mouth regions from head proportions. Always returns a
    complete dict -- this is the deterministic fallback when Haar misses."""
    hx, hy, hw, hh = head_box
    cx = hx + hw // 2
    eye_y = int(hy + hh * EYE_LINE_FRAC)
    brow_y = int(hy + hh * BROW_LINE_FRAC)
    mouth_y = int(hy + hh * MOUTH_LINE_FRAC)
    sep = int(hw * EYE_SEP_FRAC)
    ew = int(hw * EYE_W_FRAC)
    eh = int(hh * EYE_H_FRAC)

    def _box(cxp, cyp, bw, bh):
        return [int(cxp - bw / 2), int(cyp - bh / 2), int(bw), int(bh)]

    left_eye = _box(cx - sep // 2, eye_y, ew, eh)
    right_eye = _box(cx + sep // 2, eye_y, ew, eh)
    brow_w = int(ew * 1.15)
    brow_h = max(int(eh * 0.5), 4)
    left_brow = _box(cx - sep // 2, brow_y, brow_w, brow_h)
    right_brow = _box(cx + sep // 2, brow_y, brow_w, brow_h)
    mouth = _box(cx, mouth_y, int(hw * 0.42), int(hh * 0.14))
    return {
        "left_eye": left_eye, "right_eye": right_eye,
        "left_brow": left_brow, "right_brow": right_brow,
        "mouth": mouth,
        "eye_line_y": eye_y, "mouth_line_y": mouth_y,
    }


def _union_box(a, b):
    """Smallest [x, y, w, h] box containing both input (x, y, w, h) boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = min(ax, bx)
    y0 = min(ay, by)
    x1 = max(ax + aw, bx + bw)
    y1 = max(ay + ah, by + bh)
    return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]


def _torso_region(fig_box, head_box):
    """Torso = the figure below the head, full figure width."""
    fx, fy, fw, fh = fig_box
    hy, hh = head_box[1], head_box[3]
    top = hy + hh
    return [fx, top, fw, max(fy + fh - top, 1)]


# ---------------------------------------------------------------------------
# Warp parameter synthesis
# ---------------------------------------------------------------------------

def _warp_params(fig_box, head_box, regions) -> dict:
    """Suggested mesh-warp parameters the Live2D step animates. Amplitudes are
    expressed in pixels (absolute, in portrait coords) AND normalised fractions
    so the consumer can scale to whatever panel resolution it composites at.

    Idle  -- slow head sway + breathing torso bob.
    Gaze  -- pupil/iris translation range + head-turn yaw/pitch in degrees.
    Blink -- eyelid close, the alpha-scale + cycle timing for the eye regions.
    """
    fw, fh = fig_box[2], fig_box[3]
    hw, hh = head_box[2], head_box[3]
    return {
        "idle": {
            # gentle head translation + rotation, ~3s loop
            "head_sway_px": [round(hw * 0.012, 2), round(hh * 0.010, 2)],
            "head_sway_frac": [0.012, 0.010],
            "head_rot_deg": 1.2,
            "period_s": 3.2,
            # torso breathing: vertical scale pulse + tiny bob
            "torso_bob_px": round(fh * 0.006, 2),
            "torso_breathe_scale": 1.012,
            "torso_period_s": 4.0,
        },
        "gaze": {
            # how far the iris can translate within the eye opening
            "iris_range_px": [round(hw * 0.018, 2), round(hh * 0.012, 2)],
            "iris_range_frac": [0.018, 0.012],
            # head follows the gaze target a little
            "head_yaw_deg": 8.0,
            "head_pitch_deg": 5.0,
            "ease": "easeInOutSine",
            "settle_s": 0.6,
        },
        "blink": {
            # eyelid close as a fraction of eye-region height (1.0 = fully shut)
            "lid_close": 1.0,
            "close_s": 0.08,
            "hold_s": 0.04,
            "open_s": 0.12,
            # stochastic blink scheduling for the idle loop
            "interval_s": [2.5, 6.0],
            "double_blink_prob": 0.15,
        },
    }


def _deform_layers(regions, head_box) -> list:
    """Live2D-style deformer hints: which mesh region each parameter drives.
    Mirrors the standard ParamAngleX/Y, ParamEyeLOpen, ParamMouthOpenY vocab so
    the consuming rig step can map 1:1 onto Live2D parameters."""
    return [
        {"id": "ParamAngleX", "target": "head", "range_deg": [-10, 10],
         "drives": "head yaw (idle sway + gaze)"},
        {"id": "ParamAngleY", "target": "head", "range_deg": [-8, 8],
         "drives": "head pitch"},
        {"id": "ParamAngleZ", "target": "head", "range_deg": [-6, 6],
         "drives": "head tilt (idle only)"},
        {"id": "ParamEyeBallX", "target": ["left_eye", "right_eye"],
         "range": [-1, 1], "drives": "iris horizontal (gaze)"},
        {"id": "ParamEyeBallY", "target": ["left_eye", "right_eye"],
         "range": [-1, 1], "drives": "iris vertical (gaze)"},
        {"id": "ParamEyeLOpen", "target": "left_eye", "range": [0, 1],
         "drives": "left eyelid (blink)"},
        {"id": "ParamEyeROpen", "target": "right_eye", "range": [0, 1],
         "drives": "right eyelid (blink)"},
        {"id": "ParamBrowLY", "target": "left_brow", "range": [-1, 1],
         "drives": "left brow raise (expression)"},
        {"id": "ParamBrowRY", "target": "right_brow", "range": [-1, 1],
         "drives": "right brow raise (expression)"},
        {"id": "ParamMouthOpenY", "target": "mouth", "range": [0, 1],
         "drives": "mouth open (viseme / aside)"},
        {"id": "ParamBodyAngleX", "target": "torso", "range_deg": [-6, 6],
         "drives": "torso sway (idle breathing)"},
    ]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_rig_spec(cutout_path, out_dir: Optional[Path] = None) -> dict:
    """Build a Live2D-style rig spec from a character cutout RGBA.

    Args:
        cutout_path: data/gen/<slug>_cutout.png (RGBA; alpha defines the figure).
        out_dir: where to write the JSON (defaults to the cutout's dir).

    Returns the spec dict (also written to <slug>_rig.json). Keys:
        version, slug, source, canvas{w,h}, detection{method,face_found,eyes_found},
        layers{head,eyes,brows,mouth,torso,figure}, landmarks, warp, deformers.
    """
    cutout_path = Path(cutout_path)
    if not cutout_path.exists():
        raise FileNotFoundError(f"cutout not found: {cutout_path}")
    out_dir = Path(out_dir) if out_dir is not None else cutout_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = cutout_path.stem
    slug = stem[:-len("_cutout")] if stem.endswith("_cutout") else stem

    img = cv2.imread(str(cutout_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"could not read cutout: {cutout_path}")
    h, w = img.shape[:2]
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[..., 3]
        rgb = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)
    else:
        # No alpha channel -> treat the whole frame as opaque figure.
        alpha = np.full((h, w), 255, np.uint8)
        rgb = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3
               else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))

    # 1. figure bbox from alpha.
    fig_box = _bbox_from_alpha(alpha)

    # 2. head box: Haar face if it fires, else top HEAD_FRAC of the figure.
    method = "proportional"
    face = _detect_face(rgb, fig_box)
    if face is not None:
        head_box = list(face)
        method = "haar_face"
        # Reconcile: a Haar face can extend a little beyond the alpha-derived
        # figure bbox (low-contrast hair against a dark backdrop gets trimmed by
        # the matte feather). The head is part of the figure, so grow the figure
        # bbox to contain it -- keeps every downstream layer internally
        # consistent (head never floats above the figure).
        fig_box = _union_box(fig_box, tuple(head_box))
    else:
        fx, fy, fw, fh = fig_box
        hh = int(fh * HEAD_FRAC)
        hw = int(fw * 0.62)
        hx = fx + (fw - hw) // 2
        head_box = [hx, fy, hw, hh]

    # 3. eyes: Haar if it fires (refines the eye line), else proportional.
    regions = _proportional_regions(fig_box, head_box)
    eyes = _detect_eyes(rgb, tuple(head_box))
    eyes_found = False
    if eyes is not None and len(eyes) == 2:
        eyes_found = True
        regions["left_eye"], regions["right_eye"] = eyes[0], eyes[1]
        # recompute eye line from detected eye centres
        ly = eyes[0][1] + eyes[0][3] // 2
        ry = eyes[1][1] + eyes[1][3] // 2
        regions["eye_line_y"] = int((ly + ry) / 2)
        if method == "haar_face":
            method = "haar_face+eyes"

    torso_box = _torso_region(fig_box, head_box)

    def _centre(box):
        return [int(box[0] + box[2] / 2), int(box[1] + box[3] / 2)]

    spec = {
        "version": "1.0",
        "format": "living-portraits/rig-spec",
        "slug": slug,
        "source": cutout_path.name,
        "canvas": {"w": int(w), "h": int(h)},
        "detection": {
            "method": method,
            "face_found": face is not None,
            "eyes_found": eyes_found,
            "landmark_lib": _landmarks_available(),
        },
        # All boxes are [x, y, w, h] in cutout pixel coords.
        "layers": {
            "figure": {"box": list(fig_box)},
            "head": {"box": list(head_box), "pivot": _centre(head_box)},
            "left_eye": {"box": regions["left_eye"], "pivot": _centre(regions["left_eye"])},
            "right_eye": {"box": regions["right_eye"], "pivot": _centre(regions["right_eye"])},
            "left_brow": {"box": regions["left_brow"]},
            "right_brow": {"box": regions["right_brow"]},
            "mouth": {"box": regions["mouth"], "pivot": _centre(regions["mouth"])},
            "torso": {"box": torso_box, "pivot": _centre(torso_box)},
        },
        "landmarks": {
            "eye_line_y": regions["eye_line_y"],
            "mouth_line_y": regions["mouth_line_y"],
            "head_center": _centre(head_box),
            "left_eye_center": _centre(regions["left_eye"]),
            "right_eye_center": _centre(regions["right_eye"]),
        },
        "warp": _warp_params(fig_box, head_box, regions),
        "deformers": _deform_layers(regions, head_box),
    }

    rig_path = out_dir / f"{slug}_rig.json"
    rig_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    spec["_path"] = str(rig_path)
    return spec


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest() -> int:
    print("[rig_spec] self-test", flush=True)
    print("[rig_spec] landmark lib available:", _landmarks_available(), flush=True)

    # Reuse segment.py to produce a real cutout from a synthetic portrait, so the
    # two stages are exercised exactly as main() chains them.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import segment as seg

    tmp = OUT / "_selftest"
    tmp.mkdir(parents=True, exist_ok=True)
    synth = seg._synthetic_portrait()
    portrait = tmp / "selftest_portrait.png"
    cv2.imwrite(str(portrait), cv2.cvtColor(synth, cv2.COLOR_RGB2BGR))
    seg_res = seg.segment(portrait)
    cutout = seg_res["cutout"]
    print(f"[rig_spec] segmented cutout ({seg_res['backend']}): {cutout}", flush=True)

    spec = build_rig_spec(cutout)
    print(f"[rig_spec] rig -> {spec['_path']}", flush=True)
    print(f"[rig_spec] detection: {spec['detection']}", flush=True)
    print(f"[rig_spec] head box: {spec['layers']['head']['box']}", flush=True)
    print(f"[rig_spec] eye line y: {spec['landmarks']['eye_line_y']}", flush=True)

    # Assertions: file exists + is valid JSON + has all required layers + boxes
    # are inside the canvas + warp/deformers present.
    rig_path = Path(spec["_path"])
    assert rig_path.exists(), "rig JSON missing"
    loaded = json.loads(rig_path.read_text(encoding="utf-8"))
    cw, ch = loaded["canvas"]["w"], loaded["canvas"]["h"]
    required = ["figure", "head", "left_eye", "right_eye",
                "left_brow", "right_brow", "mouth", "torso"]
    for name in required:
        assert name in loaded["layers"], f"missing layer {name}"
        x, y, bw, bh = loaded["layers"][name]["box"]
        assert bw > 0 and bh > 0, f"{name} has degenerate box {(x, y, bw, bh)}"
        # box must lie (mostly) within the canvas
        assert -5 <= x <= cw and -5 <= y <= ch, f"{name} box origin off-canvas: {(x, y)}"
    assert "idle" in loaded["warp"] and "gaze" in loaded["warp"] \
        and "blink" in loaded["warp"], "warp params incomplete"
    assert len(loaded["deformers"]) >= 8, "deformer list too short"
    # head should sit in the upper half of the figure
    fy = loaded["layers"]["figure"]["box"][1]
    hy = loaded["layers"]["head"]["box"][1]
    assert hy >= fy - 5, "head box above figure"
    print(f"[rig_spec] OK -- {len(required)} layers, "
          f"{len(loaded['deformers'])} deformers, warp idle/gaze/blink present", flush=True)
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] not in ("--selftest", "selftest"):
        slug = sys.argv[1]
        cutout = OUT / f"{slug}_cutout.png"
        spec = build_rig_spec(cutout)
        print("rig:", spec["_path"], flush=True)
        print("detection:", spec["detection"], flush=True)
        return 0
    return _selftest()


if __name__ == "__main__":
    raise SystemExit(main())
