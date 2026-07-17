"""_gen_anchor.py -- FRAMELESS edge-to-edge painterly anchor for the LED panels.

The LED bezel IS the frame, so this generates the figure filling the panel: no
painted gilt frame, no wall, no picture-of-a-picture. Painterly (it "pretends" to
be a painting) but composed as a living person who fills the panel. Writes
data/gen/<slug>_anchor.png -- NON-destructive (leaves <slug>_portrait.png alone).

    python pipeline/_gen_anchor.py phineas [seed]
"""
import re
import sys
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

ROOT = Path(__file__).resolve().parent.parent
CHARS = ROOT / "prompts" / "characters"
OUT = ROOT / "data" / "gen"

# Painterly look, but FRAMELESS + edge-to-edge. The negatives actively suppress
# any frame / border / wall so SD can't reintroduce the picture-of-a-picture.
STYLE = ("oil painting portrait, Old Masters style, dramatic chiaroscuro lighting, "
         "head and shoulders filling the entire frame, tightly cropped, edge to edge, "
         "plain dark background, museum quality, highly detailed expressive face, "
         "painterly brushwork")
NEG = ("picture frame, gilded frame, ornate frame, gold frame, wooden frame, border, "
       "framed painting, painting hanging on a wall, gallery wall, matte border, "
       "photograph, 3d render, cartoon, anime, modern clothing, text, watermark, "
       "deformed, extra fingers, blurry, low quality")


def theme_of(slug: str) -> str:
    p = CHARS / (slug + ".md")
    if not p.exists():
        return ""
    m = re.search(r"Theme:\s*(.+)", p.read_text(encoding="utf-8-sig"))
    return m.group(1).strip() if m else ""


def main():
    slug = sys.argv[1] if len(sys.argv) > 1 else "phineas"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    prompt = (STYLE + ", " + theme_of(slug) +
              ", a single living figure, shoulders visible, looking toward the viewer")
    print("SLUG", slug, "SEED", seed, flush=True)
    print("PROMPT", prompt, flush=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=torch.float16, use_safetensors=True, safety_checker=None)
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()

    img = pipe(prompt=prompt, negative_prompt=NEG, num_inference_steps=30,
               guidance_scale=7.0, height=512, width=512,
               generator=torch.manual_seed(seed)).images[0]

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / (slug + "_anchor.png")
    img.save(out)
    print("SAVED", out, flush=True)


if __name__ == "__main__":
    main()
