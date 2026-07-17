"""Pre-fetch the img2vid prototype models into the local HF cache (run on hil).

CPU-only load (no .to(cuda)) so it never touches VRAM -- the director can keep
running while this downloads. Each grab is fault-isolated: one repo failing does
not abort the others. Run detached; tail _dl.log for progress.
"""
import torch

def grab(label, fn):
    print("=== downloading:", label, flush=True)
    try:
        fn()
        print("OK:", label, flush=True)
    except Exception as e:
        print("FAIL:", label, repr(e), flush=True)

# SD1.5 oil-portrait base -- SAME id generate.py uses, so AnimateDiff frames match the portrait style.
def _sd15():
    from diffusers import StableDiffusionPipeline
    StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=torch.float16, use_safetensors=True, safety_checker=None)

def _adapter():
    from diffusers import MotionAdapter
    MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5-3", torch_dtype=torch.float16)

def _svd():
    from diffusers import StableVideoDiffusionPipeline
    try:
        StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16, variant="fp16")
    except Exception as e:
        print("  (fp16 variant failed: %r; retrying without variant)" % e, flush=True)
        StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16)

grab("SD1.5 base", _sd15)
grab("AnimateDiff motion-adapter v1-5-3", _adapter)
grab("SVD-XT", _svd)
print("PREDOWNLOAD DONE", flush=True)
