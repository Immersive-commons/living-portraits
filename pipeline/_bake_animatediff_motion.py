"""_bake_animatediff_motion.py -- PROMPT-DRIVEN full-body motion loops (AnimateDiff).

Rayyan's directive: COMPLETE BODY MOVEMENT, prompted in detail via the character JSON
(not LivePortrait's face-only warp). Each motion.idle_behaviors entry carries an
`animation_prompt` (rich full-body action) + a 77-token-safe `lead` (the action verbs).
This bakes that into a full-frame AnimateDiff clip (SD1.5 + motion adapter), identity-
guided by the frameless anchor via IP-Adapter, looped via boomerang (the action plays
out then reverses back to the rest pose), saved + registered on the clip graph.

    python pipeline/_bake_animatediff_motion.py --slug phineas --behaviors declaim
    python pipeline/_bake_animatediff_motion.py --slug phineas               # all behaviours

Honest ceiling: AnimateDiff gives real gestures/lean/arm-sweeps but moderate magnitude;
cinematic large motion (fully standing, walking out) needs Ampere+ models. `motion_mag`
in the output reports how much actually moved so we can tune.
VRAM: pause lp-director + `ollama stop` first; uses model CPU offload.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHARS = ROOT / "prompts" / "characters"
GEN = ROOT / "data" / "gen"
OUT = ROOT / "data" / "clips" / "_proto"
SD15 = "stable-diffusion-v1-5/stable-diffusion-v1-5"
ADAPTER = "guoyww/animatediff-motion-adapter-v1-5-3"
BASE_POSE = "idle"

# A short painterly/frameless tail (kept brief so the action `lead` dominates the 77 tokens).
STYLE_TAG = "oil painting portrait, full figure fills the frame, plain dark background, painterly"
NEG = ("picture frame, gilded frame, framed painting, painting on a wall, gallery wall, "
       "border, static, still, frozen, motionless, blurry, deformed, extra limbs, "
       "duplicate, watermark, text")


def to_np(im):
    return np.asarray(im.convert("RGB")) if hasattr(im, "convert") else np.asarray(im)


def boomerang(fr):
    return list(fr) + list(fr)[-2:0:-1] if len(fr) >= 3 else list(fr)


def save_loop(frames, stem, fps):
    import imageio.v2 as imageio
    OUT.mkdir(parents=True, exist_ok=True)
    loop = boomerang(frames)
    npl = [to_np(f) for f in loop]
    w = imageio.get_writer(OUT / (stem + ".mp4"), fps=fps, codec="libx264",
                           quality=8, macro_block_size=8)
    for _ in range(2):
        for fr in npl:
            w.append_data(fr)
    w.close()
    imageio.mimsave(OUT / (stem + ".gif"), npl, duration=1.0 / fps, loop=0)
    n = min(8, len(frames))
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    tiles = []
    for i in idx:
        f = frames[i]
        tiles.append(f.convert("RGB") if hasattr(f, "convert") else Image.fromarray(to_np(f)))
    tw, th = tiles[0].size
    strip = Image.new("RGB", (tw * n, th), "black")
    for k, t in enumerate(tiles):
        strip.paste(t, (k * tw, 0))
    strip.save(OUT / (stem + "_strip.png"))


def load_spec(slug):
    p = CHARS / (slug + ".json")
    if not p.exists():
        sys.exit("no character spec at %s" % p)
    return json.loads(p.read_text(encoding="utf-8-sig"))


def build_pipe(ip_scale):
    from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
    adapter = MotionAdapter.from_pretrained(ADAPTER, torch_dtype=torch.float16)
    pipe = AnimateDiffPipeline.from_pretrained(
        SD15, motion_adapter=adapter, torch_dtype=torch.float16, safety_checker=None)
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, beta_schedule="linear", clip_sample=False,
        timestep_spacing="linspace", steps_offset=1)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    pipe.set_ip_adapter_scale(ip_scale)
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="phineas")
    ap.add_argument("--behaviors", nargs="+", default=None)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--ip-scale", type=float, default=0.55, dest="ip_scale",
                    help="IP-Adapter identity strength. LOWER = more motion freedom, more drift.")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    spec = load_spec(args.slug)
    motion = spec.get("motion", {})
    fps = int(motion.get("fps", 12))
    frames = int(motion.get("frames", 25))
    behaviors = motion.get("idle_behaviors", [])
    if args.behaviors:
        behaviors = [b for b in behaviors if b.get("id") in args.behaviors]
    if not behaviors:
        sys.exit("no matching idle_behaviors")

    src = GEN / (args.slug + "_anchor.png")
    if not src.exists():
        sys.exit("no anchor at %s -- run _gen_anchor_sdxl.py first" % src)
    anchor = Image.open(src).convert("RGB").resize((args.res, args.res), Image.LANCZOS)

    import verify
    from runtime.clip_graph import Clip, ClipGraph, MANIFEST_PATH
    manifest = Path(MANIFEST_PATH)
    graph = ClipGraph.load(manifest)

    if torch.cuda.is_available():
        f, t = torch.cuda.mem_get_info()
        print("vram_free_GB %.2f" % (f / 1e9), flush=True)
    pipe = build_pipe(args.ip_scale)

    for b in behaviors:
        bid = b["id"]
        lead = b.get("lead") or b.get("animation_prompt", "")
        prompt = lead + ", " + STYLE_TAG
        print("BEHAVIOR %s prompt=%r" % (bid, prompt), flush=True)
        t0 = time.time()
        out = pipe(prompt=prompt, negative_prompt=NEG, num_frames=frames,
                   height=args.res, width=args.res, guidance_scale=args.guidance,
                   num_inference_steps=args.steps, ip_adapter_image=anchor,
                   generator=torch.manual_seed(args.seed))
        fr = out.frames[0]
        stem = "%s__ad_%s" % (args.slug, bid)
        save_loop(fr, stem, fps)
        # how much actually moved: mean abs pixel delta, first vs middle frame
        a0 = to_np(fr[0]).astype("float32")
        amid = to_np(fr[len(fr) // 2]).astype("float32")
        motion_mag = float(np.abs(amid - a0).mean())
        loop = boomerang(fr)
        _, seam, _ = verify.loopability(to_np(loop[0]), to_np(loop[-1]))
        clip = Clip(from_node=BASE_POSE, to_node=BASE_POSE, asset="_proto/%s.mp4" % stem,
                    frames=len(fr), fps=fps, loopable=True,
                    tags=[bid, "character:%s" % args.slug, "animatediff"])
        graph.add_clip(clip)
        print("  -> %s  motion_mag=%.1f (higher=more body movement)  boomerang_seam=%.3f  %.0fs"
              % (stem, motion_mag, seam, time.time() - t0), flush=True)

    graph.save(manifest)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
