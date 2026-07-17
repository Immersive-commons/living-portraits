"""_bake_svd_loops.py -- SVD living-portrait loops from the FRAMELESS anchor.

The Turing-fast pivot after LTX FLF2V proved unusable on the 2080 Ti (~5h/clip).
Feeds data/gen/<slug>_anchor.png (frameless, edge-to-edge) to SVD image-to-video,
bakes ONE clip per --seed (each seed = a different subtle motion = the variety lever),
and closes each into a seamless loop via boomerang. Every loop starts on the anchor,
so they chain (the clip_graph idea). Loads SVD once, reuses it across seeds.

    python pipeline/_bake_svd_loops.py --slug phineas --seeds 1 2 3 --motion 80

VRAM: pause lp-director + `ollama stop` first; uses model CPU offload.
NOTE: boomerang loop = mirror motion (forward then reverse), not a generated return.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
GEN = ROOT / "data" / "gen"
OUT = ROOT / "data" / "clips" / "_proto"


def to_np(im):
    return np.asarray(im.convert("RGB"))


def boomerang(fr):
    return list(fr) + list(fr)[-2:0:-1] if len(fr) >= 3 else list(fr)


def save(frames, stem, fps):
    import imageio.v2 as imageio
    OUT.mkdir(parents=True, exist_ok=True)
    loop = boomerang(frames)
    npl = [to_np(f) for f in loop]
    w = imageio.get_writer(OUT / (stem + ".mp4"), fps=fps, codec="libx264",
                           quality=8, macro_block_size=8)
    for _ in range(3):
        for fr in npl:
            w.append_data(fr)
    w.close()
    imageio.mimsave(OUT / (stem + ".gif"), npl, duration=1.0 / fps, loop=0)
    n = min(8, len(frames))
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    tiles = [frames[i].convert("RGB") for i in idx]
    tw, th = tiles[0].size
    strip = Image.new("RGB", (tw * n, th), "black")
    for k, t in enumerate(tiles):
        strip.paste(t, (k * tw, 0))
    strip.save(OUT / (stem + "_strip.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="phineas")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--res", type=int, default=576)
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--motion", type=int, default=80, help="SVD motion_bucket_id (subtle~40, lively~127)")
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    src = GEN / (args.slug + "_anchor.png")
    if not src.exists():
        sys.exit("no anchor at %s -- run _gen_anchor.py first" % src)
    img = Image.open(src).convert("RGB").resize((args.res, args.res), Image.LANCZOS)

    from diffusers import StableVideoDiffusionPipeline
    repo = "stabilityai/stable-video-diffusion-img2vid-xt"
    try:
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            repo, torch_dtype=torch.float16, variant="fp16")
    except Exception:
        pipe = StableVideoDiffusionPipeline.from_pretrained(repo, torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()

    if torch.cuda.is_available():
        f, t = torch.cuda.mem_get_info()
        print("vram_free_GB %.2f" % (f / 1e9), flush=True)
    for seed in args.seeds:
        t0 = time.time()
        fr = pipe(img, height=args.res, width=args.res, decode_chunk_size=args.chunk,
                  num_frames=args.frames, motion_bucket_id=args.motion,
                  noise_aug_strength=0.02, fps=args.fps,
                  generator=torch.manual_seed(seed)).frames[0]
        stem = "%s__svdanchor_s%d" % (args.slug, seed)
        save(fr, stem, args.fps)
        print("seed %d: %d frames in %.0fs -> %s" % (seed, len(fr), time.time() - t0, stem), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
