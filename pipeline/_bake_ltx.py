"""_bake_ltx.py -- TRUE FLF2V loop via LTX-Video (anchor pinned at first AND last frame).

Unlike _bake.py's SVD/AnimateDiff (which need boomerang to fake a loop), LTX conditions
the SAME anchor at frame 0 AND the last frame, so the model GENERATES motion that departs
from and returns to the anchor -> a native seamless loop. Different --seed values give
different in-betweens from the same anchor: that's the "variety" lever, and because every
clip begins and ends on the anchor they chain forever (the clip_graph idea).

    python pipeline/_bake_ltx.py --slug phineas --seed 1
    python pipeline/_bake_ltx.py --slug phineas --seed 2 --vae-fp32   # if frames are black (fp16 VAE NaN on Turing)

VRAM: pause lp-director + `ollama stop` first. Uses model CPU offload (T5-XXL is large).
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
MODEL = "Lightricks/LTX-Video-0.9.7-dev"

PROMPT = ("oil painting of an old alchemist, alive and breathing, subtle slow head movement, "
          "occasional slow blink, gentle shift of weight, flickering candlelight on the face, "
          "painterly brushwork, cinemagraph, seamless loop")
NEG = ("static, frozen, still image, fast motion, jitter, flicker, morphing, distortion, "
       "picture frame, border, text, watermark, deformed, extra limbs, low quality")


def to_np(im):
    return np.asarray(im.convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="phineas")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--res", type=int, default=512)        # multiple of 32
    ap.add_argument("--frames", type=int, default=49)      # must be 8N+1
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--vae-fp32", action="store_true", dest="vae_fp32")
    args = ap.parse_args()

    src = GEN / (args.slug + "_anchor.png")
    if not src.exists():
        sys.exit("no anchor at %s -- run _gen_anchor.py first" % src)
    anchor = Image.open(src).convert("RGB").resize((args.res, args.res), Image.LANCZOS)

    from diffusers import LTXConditionPipeline
    from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
    pipe = LTXConditionPipeline.from_pretrained(MODEL, torch_dtype=torch.float16)
    if args.vae_fp32:
        pipe.vae = pipe.vae.to(torch.float32)
    pipe.enable_model_cpu_offload()
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass

    last = args.frames - 1
    conditions = [
        LTXVideoCondition(image=anchor, frame_index=0, strength=1.0),
        LTXVideoCondition(image=anchor, frame_index=last, strength=1.0),
    ]
    print("LTX FLF2V slug=%s seed=%d frames=%d (anchor pinned at 0 and %d)" % (
        args.slug, args.seed, args.frames, last), flush=True)
    if torch.cuda.is_available():
        f, t = torch.cuda.mem_get_info()
        print("vram_free_GB %.2f" % (f / 1e9), flush=True)

    t0 = time.time()
    out = pipe(conditions=conditions, prompt=PROMPT, negative_prompt=NEG,
               width=args.res, height=args.res, num_frames=args.frames,
               num_inference_steps=args.steps, guidance_scale=args.guidance,
               generator=torch.manual_seed(args.seed))
    frames = out.frames[0]
    print("baked %d frames in %.0fs" % (len(frames), time.time() - t0), flush=True)

    arr0 = to_np(frames[0]).astype("float32")
    print("frame0 mean=%.1f (near 0 => black/NaN; rerun with --vae-fp32)" % arr0.mean(), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = "%s__ltx_s%d" % (args.slug, args.seed)
    import imageio.v2 as imageio
    mp4 = OUT / (stem + ".mp4")
    w = imageio.get_writer(mp4, fps=args.fps, codec="libx264", quality=8, macro_block_size=8)
    for _ in range(3):                       # 3 loops so the seam is watchable
        for fr in frames:
            w.append_data(to_np(fr))
    w.close()
    gif = OUT / (stem + ".gif")
    imageio.mimsave(gif, [to_np(f) for f in frames], duration=1.0 / args.fps, loop=0)

    n = min(8, len(frames))
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    tiles = [frames[i].convert("RGB") for i in idx]
    tw, th = tiles[0].size
    strip = Image.new("RGB", (tw * n, th), "black")
    for k, t_ in enumerate(tiles):
        strip.paste(t_, (k * tw, 0))
    strip_path = OUT / (stem + "_strip.png")
    strip.save(strip_path)

    # FLF2V should loop natively (first ~ last by construction) -- no boomerang.
    try:
        import verify
        p, s, _ = verify.loopability(to_np(frames[0]), to_np(frames[-1]))
        print("native loop (FLF2V): %s SSIM=%.3f (gate>=%.2f)" % (
            "PASS" if p else "FAIL", s, verify.LOOP_THRESHOLD), flush=True)
    except Exception as e:
        print("score err", repr(e), flush=True)

    print("MP4", mp4, flush=True)
    print("GIF", gif, flush=True)
    print("STRIP", strip_path, flush=True)


if __name__ == "__main__":
    main()
