"""
generate.py -- offline portrait generation (free, local SD 1.5 on SC2).

Reads a character's Theme line from prompts/characters/<slug>.md, composes an
oil-portrait prompt, and renders a 1024x1024 painting to data/gen/<slug>_portrait.png.

VRAM note: SC2's RTX 2080 Ti has 11GB shared with qwen3 + the catalog, so we
use model CPU offload + attention slicing to keep SDXL's peak ~4GB. Slower, but
this is offline batch work -- latency does not matter. Pause lp-director during
a run if VRAM is tight.

    python pipeline/generate.py phineas
"""
import re
import sys
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

ROOT = Path(__file__).resolve().parent.parent
CHARS = ROOT / "prompts" / "characters"
OUT = ROOT / "data" / "gen"

STYLE = ("oil portrait painting, Old Masters style, dramatic chiaroscuro lighting, "
         "ornate gilded frame, museum quality, highly detailed face, painterly brushwork")
NEG = ("photograph, 3d render, cartoon, anime, modern clothing, text, watermark, "
       "deformed, extra fingers, blurry, low quality")


def theme_of(slug):
    txt = (CHARS / (slug + ".md")).read_text(encoding="utf-8-sig")
    m = re.search(r"Theme:\s*(.+)", txt)
    return m.group(1).strip() if m else ""


def main():
    slug = sys.argv[1] if len(sys.argv) > 1 else "phineas"
    theme = theme_of(slug)
    prompt = STYLE + ", " + theme + ", a single dignified figure facing forward"
    print("SLUG:", slug, flush=True)
    print("PROMPT:", prompt, flush=True)

    # SD 1.5 (~4GB) not SDXL (~13GB multi-file snapshot): the panels are 256x256
    # and 192x192, so SDXL's 1024 is wasteful, and the far smaller pull survives
    # SC2's flaky HuggingFace link. fp16 at runtime; no variant (avoids a missing-
    # variant failure); safety_checker off (skips a ~1.2GB download + black-image
    # false positives on dark portraits). 512 native -> downscaled to the panel.
    pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        use_safetensors=True,
        safety_checker=None,
    )
    pipe.enable_model_cpu_offload()   # keep peak VRAM low on the shared 2080 Ti
    pipe.enable_attention_slicing()

    image = pipe(
        prompt=prompt,
        negative_prompt=NEG,
        num_inference_steps=30,
        guidance_scale=7.0,
        height=512,
        width=512,
    ).images[0]

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / (slug + "_portrait.png")
    image.save(out)
    print("SAVED", out, flush=True)


if __name__ == "__main__":
    main()
