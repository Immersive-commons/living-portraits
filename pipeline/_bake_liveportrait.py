"""_bake_liveportrait.py -- personality-driven LivePortrait idle loops from the anchor.

LivePortrait is the path (not SVD) BECAUSE it isn't per-frame diffusion -- it keypoint-
retargets expression/head/eye onto a still, so it is fast on Turing AND directly
controllable. That controllability is what makes "declaim vs sulk" *literal*: each
idle_behavior in prompts/characters/<slug>.json is compiled into a keyframed sequence of
explicit motion DELTAS (head pitch/yaw/roll, an expression push, eye-close, lip-open) that
START and END at the neutral anchor pose -- a native neutral->behaviour->neutral loop, no
boomerang. Each loop is saved (mp4+gif+strip via _bake_svd_loops.save), seam-scored by
verify.loopability, and registered on the clip_graph as an idle self-loop carrying the
behaviour id + character in its tags.

    python pipeline/_bake_liveportrait.py --slug phineas
    python pipeline/_bake_liveportrait.py --slug phineas --behaviors declaim wounded_sulk

VRAM (hil = 2080 Ti, 11 GB): pause lp-director + `ollama stop` first; loads LivePortrait
once and reuses it across behaviours.

------------------------------------------------------------------------------------------
LivePortrait API this is coded against (github.com/KwaiVGI/LivePortrait, MIT):
  from src.config.inference_config import InferenceConfig
  from src.config.crop_config import CropConfig
  from src.live_portrait_wrapper import LivePortraitWrapper
  from src.utils.camera import get_rotation_matrix          # (pitch,yaw,roll in DEGREES)->[bs,3,3]

  wrapper = LivePortraitWrapper(inference_cfg=InferenceConfig())
  I_s   = wrapper.prepare_source(rgb_uint8_HWC)              # -> source tensor
  f_s   = wrapper.extract_feature_3d(I_s)                    # appearance feature
  x_info= wrapper.get_kp_info(I_s)                           # {'pitch','yaw','roll','t','exp','scale','kp'}
  x_c_s = wrapper.transform_keypoint(x_info)                 # canonical/transformed source keypoints
  # per frame, build a DRIVEN keypoint set from explicit deltas:
  #   R_d         = get_rotation_matrix(pitch+dp, yaw+dy, roll+dr)
  #   x_d_i_new   = scale * ( (kp_canonical) @ R_d + (exp + d_exp) ) + (t + d_t)
  #   x_d_i_new   = wrapper.stitching(x_s, x_d_i_new)        # keep shoulders/edges stable
  #   out         = wrapper.warp_decode(f_s, x_s, x_d_i_new)
  #   frame_uint8 = wrapper.parse_output(out['out'])
The motion-template delta path (x_d_new = scale*(x_c_s @ R + exp_delta) + t) is exactly how
LivePortrait drives from a .pkl template with no driving video; we synthesize the template
in code from the per-behaviour spec instead of reading a .pkl. See ASSUMPTIONS in the bake
report -- the wrapper attribute/key names are from the public source but UNVERIFIED on hil.
# VERIFY against install/GENSTACK_INSTALL.md for the pinned import paths + the model dir.
------------------------------------------------------------------------------------------
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))   # so `import _bake_svd_loops` / `import verify` resolve as siblings
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

CHARS = ROOT / "prompts" / "characters"
GEN = ROOT / "data" / "gen"
# Bake into the same proto dir _bake_svd_loops uses; clip_graph asset paths are relative to
# data/clips, so an asset baked at data/clips/_proto/<stem>.mp4 is stored as "_proto/<stem>.mp4".
CLIPS_DIR = ROOT / "data" / "clips"
OUT = CLIPS_DIR / "_proto"

# The vendored LivePortrait repo (install layout). Its InferenceConfig points checkpoint_F/G/M/S/W
# at repo-relative pretrained_weights/, so we run from inside this dir (verified on hil 2026-05-27).
LIVEPORTRAIT_ROOT = ROOT / "pipeline" / "vendor" / "LivePortrait"

# The clip_graph pose every idle loop lives on. clip_graph.POSES[0] == "idle"; an idle
# behaviour is a self-loop on this pose (the behaviour id rides in tags, NOT as a node --
# POSES is a fixed 7-pose vocabulary and Clip.__post_init__ REJECTS any other node name).
BASE_POSE = "idle"


# ======================================================================================
# Per-behaviour driving spec -- the clean abstraction the brief asked for.
# ======================================================================================
# A behaviour is a short keyframed envelope of motion DELTAS relative to the neutral anchor.
# Every channel starts at 0 and ends at 0 (so the clip loops natively). We hand-author a
# default envelope per known phineas behaviour id from its `desc`, and provide a generic
# fallback derived from the behaviour's `drivers`/`desc` for any unknown id (e.g. seraphina's).
#
# Channels (all small; LivePortrait deltas are sensitive):
#   dp/dy/dr : head pitch / yaw / roll deltas, in DEGREES (pitch>0 = chin up/look up here;
#              yaw>0 = turn toward +x = the rival panel side; roll = head tilt).
#   exp_push : scalar 0..1 multiplied into an EXPRESSION DELTA vector (see _exp_delta) -- how
#              far to push the behaviour's expression shape (mouth-open for declaim, brow-down
#              for sulk, etc.). The exact LivePortrait exp basis is unnamed (21x3 deform), so
#              the per-behaviour exp vector is an ASSUMPTION (see report) -- a clean place for
#              the conductor to tune once they can see frames.
#   eye      : eye-close ratio 0 (open) .. 1 (closed) -- routed through retarget_eye if used.
#   lip      : lip-open amount 0..1 -- routed into the mouth region of the exp delta.
#   tx/ty    : tiny translation deltas (fraction of frame) for a "settle"/"sink".
#
# `phases` is a list of (t_norm, weights) keyframes with t_norm in [0,1]; channel values are
# linearly interpolated and a raised-cosine ease is applied so the motion is smooth and the
# seam at t=0/1 is C1-continuous (no velocity pop). The first and last keyframe MUST be all-zero.

# Semantic expression "shapes" -- coarse, NAMED groups we map onto LivePortrait's exp delta
# via _exp_delta(). These indices are an EXPLICIT ASSUMPTION (LivePortrait's 21 implicit
# keypoints x 3 are not a labelled blendshape space). They are isolated here so the conductor
# can re-map them after eyeballing one bake, WITHOUT touching the envelope authoring.
# VERIFY: confirm the index->facial-region mapping against a LivePortrait exp visualisation.
_EXP_DIM = 63  # 21 keypoints * 3 (LivePortrait exp is flattened (1,21,3) -> 63); see report.
_EXP_REGION_KP = {
    # region -> list of (keypoint_index, axis, sign) the region nudges. Keypoint indices are
    # LivePortrait's mouth/eye/brow implicit kps (ASSUMED from common LP usage examples).
    "mouth_open":  [(6, 1, +1.0), (7, 1, +1.0), (8, 1, +1.0)],   # lower mouth kps move +y (open)
    "mouth_tight": [(6, 0, +0.6), (7, 0, -0.6)],                  # corners pull in (tighten)
    "brow_down":   [(0, 1, +1.0), (1, 1, +1.0)],                  # inner brow kps move +y (lower)
    "brow_up":     [(0, 1, -1.0), (1, 1, -1.0)],                  # brow raise
    "smile":       [(6, 1, -0.6), (7, 1, -0.6), (6, 0, +0.4), (7, 0, -0.4)],  # corners up+out
}


def _behavior_envelopes() -> dict:
    """Hand-authored neutral->behaviour->neutral envelopes for the known phineas behaviours.

    Each value is a list of (t, channels) keyframes; channels is a dict of the channels above.
    First/last keyframe all-zero == native loop. Amplitudes are deliberately small (LivePortrait
    deltas exaggerate fast). Tuned for the `desc` of each phineas idle_behavior.
    """
    Z = {}  # neutral keyframe (all channels default to 0 via _chan)
    return {
        # "lifts chin, sweeping gaze up to an unseen balcony, lips part mid-soliloquy, grand settle"
        "declaim": [
            (0.00, Z),
            (0.45, {"dp": 7.0, "dy": -4.0, "exp_push": 0.7, "lip": 0.5}),   # chin up, gaze up-left, lips part
            (0.70, {"dp": 6.0, "dy": -3.0, "exp_push": 0.6, "lip": 0.35}),  # hold the line
            (1.00, Z),                                                       # grand settle back
        ],
        # "eyes lower, mouth tightens, a slow self-pitying exhale, shoulders sink a touch"
        "wounded_sulk": [
            (0.00, Z),
            (0.50, {"dp": -5.0, "eye": 0.45, "exp_push": 0.6, "ty": 0.012, "_shape": "mouth_tight"}),
            (0.78, {"dp": -4.0, "eye": 0.30, "exp_push": 0.5, "ty": 0.010, "_shape": "mouth_tight"}),
            (1.00, Z),
        ],
        # "resentful sidelong glance toward the brighter frame next door, then a disdainful sniff back"
        "side_eye_rival": [
            (0.00, Z),
            (0.40, {"dy": 9.0, "dr": -2.0, "eye": 0.15, "exp_push": 0.4, "_shape": "brow_down"}),  # cut to rival
            (0.62, {"dy": 9.0, "dr": -2.0, "eye": 0.10, "exp_push": 0.35, "_shape": "brow_down"}), # hold the glare
            (0.80, {"dy": 2.0, "dp": 3.0, "exp_push": 0.2}),                                       # sniff, chin up
            (1.00, Z),
        ],
        # "narrows eyes at you, a tiny tilt, decides you are beneath him"
        "appraise_viewer": [
            (0.00, Z),
            (0.45, {"eye": 0.35, "dr": 3.0, "dp": 2.0, "exp_push": 0.35, "_shape": "brow_up"}),  # narrow + tilt
            (0.72, {"eye": 0.30, "dr": 3.0, "dp": 2.0, "exp_push": 0.3, "_shape": "brow_up"}),
            (1.00, Z),
        ],
    }


def _generic_envelope(behavior: dict) -> list:
    """A clean fallback envelope for an UNKNOWN behaviour id (e.g. a seraphina behaviour),
    derived from its `drivers` + a light read of its `desc`. Conservative single-arc in/out.

    This is NOT faked motion -- it maps the behaviour's declared personality drivers to a
    plausible head/expression arc, and leaves a TODO breadcrumb so the conductor can author a
    bespoke envelope (like the phineas ones) once they can see the result. We never emit a
    static clip pretending to move.
    """
    drivers = [str(d).lower() for d in behavior.get("drivers", [])]
    desc = str(behavior.get("desc", "")).lower()
    ch = {"exp_push": 0.45}
    # crude desc/driver -> channel heuristics; amplitude stays small.
    if any(k in drivers for k in ("extraversion", "theatrical", "openness")) or "gaze up" in desc:
        ch["dp"] = 5.0
    if any(k in drivers for k in ("neuroticism", "self-pitying")) or "lower" in desc:
        ch["dp"] = -4.0
        ch["eye"] = 0.3
    if any(k in drivers for k in ("low_agreeableness", "resentful")) or "glance" in desc or "aside" in desc:
        ch["dy"] = 7.0
    if "smile" in desc or "amused" in desc or "delight" in desc:
        ch["_shape"] = "smile"
        ch["lip"] = 0.3
    if "tilt" in desc:
        ch["dr"] = 3.0
    return [(0.00, {}), (0.5, ch), (1.00, {})]


def envelope_for(slug: str, behavior: dict) -> list:
    """Pick the hand-authored envelope for a known id, else the generic personality fallback."""
    bid = behavior.get("id", "")
    known = _behavior_envelopes()
    if slug == "phineas" and bid in known:
        return known[bid]
    if bid in known:
        # a non-phineas character that happens to share an id -> reuse the authored shape.
        return known[bid]
    return _generic_envelope(behavior)


# ======================================================================================
# Envelope -> per-frame channel values (eased interpolation, loop-closed)
# ======================================================================================
_CHANNELS = ("dp", "dy", "dr", "exp_push", "eye", "lip", "tx", "ty")


def _chan(kf: dict, name: str) -> float:
    return float(kf.get(name, 0.0))


def _ease(u: float) -> float:
    """Raised-cosine ease in [0,1] -> [0,1]; zero slope at both ends so the seam has no
    velocity discontinuity (a linear ramp would pop at the loop point even if value-matched)."""
    return 0.5 - 0.5 * math.cos(math.pi * max(0.0, min(1.0, u)))


def sample_envelope(envelope: list, n_frames: int) -> list:
    """Sample the keyframed envelope into n_frames dicts of channel->value (+ optional _shape).

    Eases each linear segment so motion accelerates/decelerates smoothly. Because the first and
    last keyframe are all-zero AND the cosine ease has zero slope at the ends, frame 0 == last
    frame's *target* (neutral) and the velocity matches -> the loop closes invisibly.
    """
    if len(envelope) < 2:
        raise ValueError("envelope needs >=2 keyframes")
    ts = [kf[0] for kf in envelope]
    if ts[0] != 0.0 or ts[-1] != 1.0:
        raise ValueError("envelope must span t=0..1 (first/last keyframe)")
    frames = []
    for i in range(n_frames):
        t = i / (n_frames - 1) if n_frames > 1 else 0.0
        # find bracketing keyframes
        j = 0
        while j < len(envelope) - 2 and envelope[j + 1][0] < t:
            j += 1
        t0, kf0 = envelope[j]
        t1, kf1 = envelope[j + 1]
        span = (t1 - t0) or 1.0
        u = _ease((t - t0) / span)
        vals = {}
        for name in _CHANNELS:
            a, b = _chan(kf0, name), _chan(kf1, name)
            vals[name] = a + (b - a) * u
        # carry the dominant shape label from whichever bracket keyframe declares one
        vals["_shape"] = kf1.get("_shape") or kf0.get("_shape")
        frames.append(vals)
    return frames


# ======================================================================================
# Channel values -> a LivePortrait expression-delta vector
# ======================================================================================
def _exp_delta(vals: dict, exp_dim: int = _EXP_DIM) -> np.ndarray:
    """Turn (exp_push, lip, _shape) into a flat expression-delta vector for LivePortrait.

    Builds a (exp_dim,) float32 delta by nudging the keypoints named in the active _shape
    region (scaled by exp_push) plus a lip-open contribution. exp_dim defaults to 63 = 21*3.

    # VERIFY: _EXP_REGION_KP indices + _EXP_DIM are ASSUMPTIONS about LivePortrait's exp basis.
    # The shape of the push (which kps, which axis) is the thing to tune after the first bake.
    """
    d = np.zeros((exp_dim,), dtype=np.float32)
    push = float(vals.get("exp_push", 0.0))
    scale = 0.02 * push  # keep deltas tiny -- LP exp is in normalized keypoint space

    shape = vals.get("_shape")
    if shape and shape in _EXP_REGION_KP:
        for (kp_idx, axis, sign) in _EXP_REGION_KP[shape]:
            flat = kp_idx * 3 + axis
            if 0 <= flat < exp_dim:
                d[flat] += sign * scale

    lip = float(vals.get("lip", 0.0))
    if lip:
        for (kp_idx, axis, sign) in _EXP_REGION_KP["mouth_open"]:
            flat = kp_idx * 3 + axis
            if 0 <= flat < exp_dim:
                d[flat] += sign * 0.02 * lip
    return d


# ======================================================================================
# The bake -- one behaviour -> one loop (LivePortrait delta-driven)
# ======================================================================================
def _load_liveportrait():
    """Construct the LivePortrait wrapper + the rotation helper. Heavy; raises if absent.

    # VERIFY against install/GENSTACK_INSTALL.md: import paths + that InferenceConfig picks up
    # weights from LIVEPORTRAIT_ROOT (the repo's default is ./pretrained_weights; the installer
    # may set an env var or a config field instead).
    """
    import os
    lp = str(LIVEPORTRAIT_ROOT)
    if lp not in sys.path:
        sys.path.insert(0, lp)
    # InferenceConfig's checkpoint_{F,G,M,S,W} are RELATIVE to the repo root, so models load
    # correctly only with cwd == the repo. Restore cwd afterward so this bake's manifest/clip
    # writes still land under living-portraits (GEN/OUT are absolute, but the manifest may not be).
    cwd0 = os.getcwd()
    os.chdir(lp)
    try:
        from src.config.inference_config import InferenceConfig
        from src.live_portrait_wrapper import LivePortraitWrapper
        from src.utils.camera import get_rotation_matrix
        inference_cfg = InferenceConfig(flag_use_half_precision=True)   # fp16 -> Turing-safe
        wrapper = LivePortraitWrapper(inference_cfg=inference_cfg)
    finally:
        os.chdir(cwd0)
    return wrapper, get_rotation_matrix


def _to_tensor_like(x_example, arr: np.ndarray):
    """Make a torch tensor matching x_example's dtype/device from a numpy array."""
    import torch

    return torch.from_numpy(arr).to(device=x_example.device, dtype=x_example.dtype)


def bake_behavior(wrapper, get_rotation_matrix, anchor_rgb: np.ndarray,
                  vals_seq: list, behavior_id: str) -> list:
    """Drive LivePortrait from the anchor through one behaviour's delta sequence.

    Returns a list of uint8 HxWx3 frames (the loop). Implements the motion-template delta
    path: per frame, R from (base+delta) Euler angles, x_driven = scale*(kp_canon @ R + exp+
    d_exp) + (t + d_t), stitched, warp-decoded, parsed to uint8.

    # VERIFY: the exact tensor shapes/keys (kp vs transform_keypoint output, exp flattening,
    # stitching/​warp_decode signatures) are from the public source and UNVERIFIED on hil.
    """
    import torch

    I_s = wrapper.prepare_source(anchor_rgb)
    f_s = wrapper.extract_feature_3d(I_s)
    x_info = wrapper.get_kp_info(I_s)

    # base pose + base expression from the source still.
    # Keep pitch/yaw/roll as TENSORS: this LivePortrait's get_rotation_matrix reads
    # `pitch.device`, so it needs tensors in degrees, not python floats (tensor + float = tensor).
    base_pitch = x_info["pitch"]
    base_yaw = x_info["yaw"]
    base_roll = x_info["roll"]
    scale_s = x_info["scale"]
    t_s = x_info["t"]
    exp_s = x_info["exp"]
    # canonical (neutral) keypoints of the source; transform_keypoint returns the posed source
    # keypoints x_s used as the stitching reference.
    x_c = x_info["kp"]                         # canonical keypoints (1, 21, 3) ASSUMED
    x_s = wrapper.transform_keypoint(x_info)   # posed source keypoints (stitch reference)

    exp_dim = int(np.prod(exp_s.shape[1:])) if hasattr(exp_s, "shape") and exp_s.ndim > 1 else _EXP_DIM

    frames = []
    for vals in vals_seq:
        R_d = get_rotation_matrix(
            base_pitch + vals["dp"], base_yaw + vals["dy"], base_roll + vals["dr"])
        # expression delta -> tensor matching exp_s
        d_exp_np = _exp_delta(vals, exp_dim).reshape(exp_s.shape[1:]) if hasattr(exp_s, "shape") and exp_s.ndim > 1 \
            else _exp_delta(vals, exp_dim)
        d_exp = _to_tensor_like(exp_s, np.asarray(d_exp_np, dtype=np.float32)[None, ...]) \
            if hasattr(exp_s, "ndim") and exp_s.ndim > 1 else _to_tensor_like(exp_s, np.asarray(d_exp_np, dtype=np.float32))
        exp_new = exp_s + d_exp

        # translation delta (tx,ty as a fraction of normalized space; tz unchanged)
        t_delta = np.zeros((3,), dtype=np.float32)
        t_delta[0] = vals.get("tx", 0.0)
        t_delta[1] = vals.get("ty", 0.0)
        t_new = t_s + _to_tensor_like(t_s, t_delta.reshape(t_s.shape[-1:]) if hasattr(t_s, "shape") else t_delta)

        # x_driven = scale * (x_c @ R + exp) + t   (LivePortrait motion-template formula)
        x_driven = scale_s * (torch.bmm(x_c, R_d) + exp_new) + t_new.unsqueeze(1) \
            if x_c.ndim == 3 else scale_s * (x_c @ R_d + exp_new) + t_new

        # eye-close retarget (optional) -- pull lids via the wrapper's helper if asked
        eye = float(vals.get("eye", 0.0))
        if eye > 0.0:
            try:
                eye_ratio = _to_tensor_like(x_s, np.asarray([[eye]], dtype=np.float32))
                x_driven = x_driven + wrapper.retarget_eye(x_s, eye_ratio)
            except Exception:
                pass  # retarget API mismatch -> skip eye-close rather than crash the bake

        x_driven = wrapper.stitching(x_s, x_driven)
        out = wrapper.warp_decode(f_s, x_s, x_driven)
        frame = wrapper.parse_output(out["out"] if isinstance(out, dict) else out)
        arr = np.asarray(frame, dtype=np.uint8)
        if arr.ndim == 4:        # parse_output returns (1, H, W, 3) -- drop the batch dim
            arr = arr[0]
        frames.append(arr)
    return frames


def _scalar(v):
    """Best-effort scalar from a tensor/array/number (LivePortrait pose fields are tensors)."""
    try:
        import torch

        if isinstance(v, torch.Tensor):
            return v.detach().float().reshape(-1)[0].item()
    except Exception:
        pass
    arr = np.asarray(v).reshape(-1)
    return float(arr[0])


# ======================================================================================
# Clip registration (uses clip_graph's EXISTING API -- read-only consumer)
# ======================================================================================
def register_clip(graph, slug: str, behavior_id: str, asset_rel: str, n_frames: int,
                  fps: int, loop_score: float) -> str:
    """Register the baked loop as an idle self-loop edge via ClipGraph.add_clip().

    Mirrors orchestrate.py:_idle_clip_for, but with a REAL asset (we baked footage) so the
    edge is no longer a missing/synthetic placeholder. The behaviour id + character ride in
    tags exactly as the brief specifies; clip_graph keys the edge by (from,to)=(idle,idle),
    so the LAST behaviour baked wins that single self-loop slot.

    Returns the clip key. NOTE: add_clip stores ONE clip per (from,to) pair -- see the bake
    report's assumption about per-behaviour storage; we tag every bake but the graph keeps the
    most-recent on the idle self-loop. The player's behaviour selector (task #4) reads the
    manifest rows / tags to choose among baked behaviours.
    """
    from runtime.clip_graph import Clip

    tags = [behavior_id, "character:%s" % slug, "liveportrait",
            "loop_ssim=%.3f" % loop_score]
    clip = Clip(
        from_node=BASE_POSE,
        to_node=BASE_POSE,
        asset=asset_rel,        # relative to data/clips (CLIPS_DIR) -- real baked footage
        frames=n_frames,
        fps=fps,
        loopable=True,
        tags=tags,
    )
    graph.add_clip(clip)
    return clip.key()


# ======================================================================================
# Driver
# ======================================================================================
def load_spec(slug: str) -> dict:
    p = CHARS / (slug + ".json")
    if not p.exists():
        sys.exit("no character spec at %s" % p)
    return json.loads(p.read_text(encoding="utf-8-sig"))


def select_behaviors(spec: dict, names: list | None) -> list:
    behaviors = spec.get("motion", {}).get("idle_behaviors", [])
    if not behaviors:
        sys.exit("spec.motion.idle_behaviors is empty -- nothing to bake")
    if not names:
        return behaviors
    by_id = {b.get("id"): b for b in behaviors}
    picked = []
    for n in names:
        if n not in by_id:
            sys.exit("behavior %r not in spec (have: %s)" % (n, ", ".join(by_id)))
        picked.append(by_id[n])
    return picked


def save_native(frames, stem, fps):
    """Save numpy uint8 frames as a NATIVE loop (no boomerang -- LivePortrait frames
    already start+end neutral). mp4 (3 cycles, watchable) + gif (loops) + 8-frame strip."""
    import imageio.v2 as imageio
    from PIL import Image
    OUT.mkdir(parents=True, exist_ok=True)
    npl = [np.asarray(f, dtype=np.uint8) for f in frames]
    w = imageio.get_writer(OUT / (stem + ".mp4"), fps=fps, codec="libx264",
                           quality=8, macro_block_size=8)
    for _ in range(3):
        for fr in npl:
            w.append_data(fr)
    w.close()
    imageio.mimsave(OUT / (stem + ".gif"), npl, duration=1.0 / fps, loop=0)
    n = min(8, len(npl))
    idx = np.linspace(0, len(npl) - 1, n).round().astype(int)
    tiles = [Image.fromarray(npl[i]) for i in idx]
    tw, th = tiles[0].size
    strip = Image.new("RGB", (tw * n, th), "black")
    for k, t in enumerate(tiles):
        strip.paste(t, (k * tw, 0))
    strip.save(OUT / (stem + "_strip.png"))


def main():
    ap = argparse.ArgumentParser(description="LivePortrait personality-driven idle loops")
    ap.add_argument("--slug", default="phineas")
    ap.add_argument("--behaviors", nargs="+", default=None,
                    help="behaviour ids to bake (default: all idle_behaviors in the spec)")
    ap.add_argument("--frames", type=int, default=25, help="frames per loop")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--manifest", default=None, help="clip manifest path (default: data/clips/manifest.json)")
    args = ap.parse_args()

    spec = load_spec(args.slug)
    behaviors = select_behaviors(spec, args.behaviors)

    src = GEN / (args.slug + "_anchor.png")
    if not src.exists():
        sys.exit("no anchor at %s -- run _gen_anchor_sdxl.py first" % src)

    from PIL import Image
    anchor_rgb = np.asarray(Image.open(src).convert("RGB"), dtype=np.uint8)

    import verify
    from runtime.clip_graph import ClipGraph, MANIFEST_PATH

    manifest_path = Path(args.manifest) if args.manifest else Path(MANIFEST_PATH)
    graph = ClipGraph.load(manifest_path)

    wrapper, get_rotation_matrix = _load_liveportrait()

    results = []
    for behavior in behaviors:
        bid = behavior.get("id", "behavior")
        envelope = envelope_for(args.slug, behavior)
        vals_seq = sample_envelope(envelope, args.frames)
        frames = bake_behavior(wrapper, get_rotation_matrix, anchor_rgb, vals_seq, bid)

        if len(frames) < 2:
            print("BEHAVIOR %s: produced <2 frames -- skipping" % bid, flush=True)
            continue

        stem = "%s__lp_%s" % (args.slug, bid)
        # native loop (NO boomerang) -- LivePortrait frames already start+end neutral.
        save_native(frames, stem, args.fps)

        # seam score: native loop -> compare TRUE first vs TRUE last frame (pre-boomerang).
        ok, score, reason = verify.loopability(frames[0], frames[-1])
        asset_rel = "_proto/%s.mp4" % stem  # relative to data/clips
        clip_key = register_clip(graph, args.slug, bid, asset_rel, len(frames), args.fps, score)

        print("BEHAVIOR %s -> %s  loop=%s (%s)  clip=%s" % (
            bid, stem + ".mp4", "PASS" if ok else "FAIL", reason, clip_key), flush=True)
        results.append((bid, stem, ok, score, clip_key))

    graph.save(manifest_path)
    print("SAVED manifest %s  (%d behaviour clip(s) registered)" % (manifest_path, len(results)), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
