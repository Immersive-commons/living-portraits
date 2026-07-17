"""_bake.py -- img2vid loop prototype for the panels (run on hil's 2080 Ti).

Two backends, one harness, so we can eyeball them side by side:

    python pipeline/_bake.py svd          --slug phineas
    python pipeline/_bake.py animatediff  --slug phineas

Both take data/gen/<slug>_portrait.png and produce a SEAMLESS LOOP:
  * SVD          -- true image->video (animates the actual portrait pixels), text-free.
  * animatediff  -- SD1.5 + motion adapter + IP-Adapter (portrait-guided, same oil style).

Loop closure is BOOMERANG (forward + reverse): the wraparound frame is adjacent to
the first, so the seam never pops -- the model-agnostic way to get "first == last".
Wan FLF2V (native first-last conditioning) is NOT usable here: it needs Ampere+
(FlashAttention) and only ships as 14B. Boomerang gets the same loop on Turing.

Outputs (data/clips/_proto/):
    <slug>__<backend>.mp4     boomerang x3 (so the loop is visible on playback)
    <slug>__<backend>.gif     auto-loops in any previewer
    <slug>__<backend>_strip.png   8 sampled frames, horizontal -- to judge motion at a glance

Prints the loopability score from the REAL verify gate (pipeline/verify.py).

VRAM: pause lp-director + `ollama stop qwen3:8b` first (the director holds ~8GB).
This script uses model CPU offload, so peak VRAM stays modest, but the director
reloading qwen3 mid-bake can still OOM it.
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent          # pipeline/
ROOT = HERE.parent                                # living-portraits/
sys.path.insert(0, str(HERE))                     # so `import verify` resolves
GEN = ROOT / "data" / "gen"
OUT = ROOT / "data" / "clips" / "_proto"

STYLE = ("oil portrait painting, Old Masters style, dramatic chiaroscuro lighting, "
         "ornate gilded frame, museum quality, highly detailed face, painterly brushwork")
NEG = ("photograph, 3d render, cartoon, anime, modern clothing, text, watermark, "
       "deformed, extra fingers, blurry, low quality, flickering, morphing")
MOTION_HINT = "subtle idle motion, gentle breathing, slight head turn, alive, cinemagraph"


def theme_of(slug: str) -> str:
    p = ROOT / "prompts" / "characters" / (slug + ".md")
    if not p.exists():
        return ""
    m = re.search(r"Theme:\s*(.+)", p.read_text(encoding="utf-8-sig"))
    return m.group(1).strip() if m else ""


def load_portrait(slug: str, res: int) -> Image.Image:
    src = GEN / (slug + "_portrait.png")
    if not src.exists():
        sys.exit("no portrait at %s -- bake the still first (generate.py)" % src)
    return Image.open(src).convert("RGB").resize((res, res), Image.LANCZOS)


def boomerang(frames):
    """forward + reverse(without the endpoints) -> seamless ping-pong loop."""
    if len(frames) < 3:
        return frames
    return list(frames) + list(frames)[-2:0:-1]


def to_np(im):
    return np.asarray(im.convert("RGB"))


def save_outputs(frames, stem: str, fps: int):
    import imageio.v2 as imageio
    OUT.mkdir(parents=True, exist_ok=True)
    loop = boomerang(frames)
    np_loop = [to_np(f) for f in loop]

    mp4 = OUT / (stem + ".mp4")
    w = imageio.get_writer(mp4, fps=fps, codec="libx264", quality=8,
                           macro_block_size=8)
    for _ in range(3):                       # 3 cycles so the seam is watchable
        for fr in np_loop:
            w.append_data(fr)
    w.close()

    gif = OUT / (stem + ".gif")
    imageio.mimsave(gif, np_loop, duration=1.0 / fps, loop=0)

    # 8-frame horizontal contact sheet of the ORIGINAL (pre-boomerang) motion arc
    n = min(8, len(frames))
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    tiles = [frames[i].convert("RGB") for i in idx]
    tw, th = tiles[0].size
    strip = Image.new("RGB", (tw * n, th), "black")
    for k, t in enumerate(tiles):
        strip.paste(t, (k * tw, 0))
    strip_path = OUT / (stem + "_strip.png")
    strip.save(strip_path)
    return mp4, gif, strip_path


def score_loop(frames):
    """Score the ACTUAL output loop.

    The output is a boomerang, so its wraparound seam is boomerang[-1] -> boomerang[0]
    (== frames[1] -> frames[0], adjacent => smooth). We also report the NATIVE self-loop
    SSIM (frames[0] vs frames[-1]) purely as info: low here just means the raw model
    output doesn't loop on its own and needs the boomerang, which is expected for SVD.
    """
    try:
        import verify
        loop = boomerang(frames)
        p_out, s_out, _ = verify.loopability(to_np(loop[0]), to_np(loop[-1]))
        _, s_nat, _ = verify.loopability(to_np(frames[0]), to_np(frames[-1]))
        return ("loop seam (boomerang output): %s SSIM=%.3f (gate>=%.2f)  |  "
                "native self-loop SSIM=%.3f (low = needs boomerang, expected for SVD)" % (
                    "PASS" if p_out else "FAIL", s_out, verify.LOOP_THRESHOLD, s_nat))
    except Exception as e:
        return "loopability: (could not score: %r)" % e


# ---------------------------------------------------------------- backends
def bake_svd(slug, args):
    from diffusers import StableVideoDiffusionPipeline
    img = load_portrait(slug, args.res)
    repo = "stabilityai/stable-video-diffusion-img2vid-xt"
    try:
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            repo, torch_dtype=torch.float16, variant="fp16")
    except Exception:
        pipe = StableVideoDiffusionPipeline.from_pretrained(repo, torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
    gen = torch.manual_seed(args.seed)
    res = pipe(img, height=args.res, width=args.res,
               decode_chunk_size=args.chunk, num_frames=args.frames,
               motion_bucket_id=args.motion, noise_aug_strength=0.02,
               fps=args.fps, generator=gen)
    return res.frames[0]


def bake_animatediff(slug, args):
    from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
    portrait = load_portrait(slug, args.res)
    adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5-3", torch_dtype=torch.float16)
    pipe = AnimateDiffPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        motion_adapter=adapter, torch_dtype=torch.float16, safety_checker=None)
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, beta_schedule="linear", clip_sample=False,
        timestep_spacing="linspace", steps_offset=1)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                         weight_name="ip-adapter_sd15.bin")
    pipe.set_ip_adapter_scale(0.75)
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()
    prompt = STYLE + ", " + theme_of(slug) + ", a single dignified figure, " + MOTION_HINT
    gen = torch.manual_seed(args.seed)
    out = pipe(prompt=prompt, negative_prompt=NEG, num_frames=args.frames,
               height=args.res, width=args.res,
               guidance_scale=7.5, num_inference_steps=args.steps,
               ip_adapter_image=portrait, generator=gen)
    return out.frames[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backend", choices=["svd", "animatediff"])
    ap.add_argument("--slug", default="phineas")
    ap.add_argument("--res", type=int, default=576)
    ap.add_argument("--frames", type=int, default=0, help="0 = backend default")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--motion", type=int, default=60, help="SVD motion_bucket_id (subtle~40, lively~127)")
    ap.add_argument("--chunk", type=int, default=2, help="SVD decode_chunk_size (low VRAM)")
    ap.add_argument("--steps", type=int, default=25, help="AnimateDiff inference steps")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.frames == 0:
        args.frames = 25 if args.backend == "svd" else 16

    print("BACKEND %s | slug=%s res=%d frames=%d" % (args.backend, args.slug, args.res, args.frames), flush=True)
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print("vram_free_GB %.2f / %.2f" % (free / 1e9, total / 1e9), flush=True)
    t0 = time.time()
    frames = (bake_svd if args.backend == "svd" else bake_animatediff)(args.slug, args)
    print("baked %d frames in %.0fs" % (len(frames), time.time() - t0), flush=True)

    stem = "%s__%s" % (args.slug, args.backend)
    mp4, gif, strip = save_outputs(frames, stem, args.fps)
    print(score_loop(frames), flush=True)
    print("MP4  ", mp4, flush=True)
    print("GIF  ", gif, flush=True)
    print("STRIP", strip, flush=True)


if __name__ == "__main__":
    main()
