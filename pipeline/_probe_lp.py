"""Throwaway: introspect the REAL LivePortrait wrapper API on hil so the bake can be
reconciled against it (verifies gen-pipeline-coder's assumptions #1/#2/#8). No GPU:
imports the class + reads signatures; does NOT instantiate (which would load models)."""
import inspect
import os
import sys

LP = r"C:\living-portraits\pipeline\vendor\LivePortrait"
sys.path.insert(0, LP)
os.chdir(LP)

from src.live_portrait_wrapper import LivePortraitWrapper as W

print("WRAPPER PUBLIC METHODS:")
print(sorted(m for m in dir(W) if not m.startswith("_")))
print()
for m in ["prepare_source", "extract_feature_3d", "get_kp_info", "transform_keypoint",
          "stitching", "stitch", "warp_decode", "warping", "parse_output",
          "retarget_eye", "retarget_lip", "calc_driving_ratio"]:
    f = getattr(W, m, None)
    if f is None:
        print("  MISSING:", m)
        continue
    try:
        print("  SIG", m, inspect.signature(f))
    except (TypeError, ValueError) as e:
        print("  SIG", m, "(unintrospectable)", e)

print()
try:
    from src.utils.camera import get_rotation_matrix
    print("get_rotation_matrix", inspect.signature(get_rotation_matrix))
except Exception as e:
    print("camera import err", repr(e))

print()
try:
    from src.config.inference_config import InferenceConfig
    cfg = InferenceConfig()
    fields = [a for a in dir(cfg) if not a.startswith("_") and not callable(getattr(cfg, a))]
    print("INFERENCE_CFG FIELDS:", fields)
    # which field points at the weights tree?
    for a in fields:
        v = getattr(cfg, a)
        if isinstance(v, str) and ("weight" in a.lower() or "ckpt" in a.lower()
                                   or "checkpoint" in a.lower() or "model" in a.lower()
                                   or ".pth" in str(v) or "pretrained" in str(v).lower()):
            print("   weights-ish:", a, "=", v)
except Exception as e:
    print("InferenceConfig err", repr(e))
