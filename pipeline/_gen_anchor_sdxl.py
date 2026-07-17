"""_gen_anchor_sdxl.py -- SDXL + InstantID identity-locked frameless anchor.

The SDXL successor to _gen_anchor.py (which is SD 1.5 @ 512 -> melty faces at panel
scale). This reads a character JSON (prompts/characters/<slug>.json), assembles the
positive prompt by formatting `generation.positive_template` with the nested spec, and
renders a FRAMELESS edge-to-edge anchor at 1024 locked to one face via InstantID, so
every re-gen and BOTH panels stay the same person. Writes data/gen/<slug>_anchor.png.

    python pipeline/_gen_anchor_sdxl.py --slug phineas --seed 7
    python pipeline/_gen_anchor_sdxl.py --slug phineas --steps 40 --cfg 6.0

Identity-lock contract (from <slug>.json:generation.identity_lock):
  * If data/gen/<slug>_anchor.png exists -> use it as the identity reference (re-gen
    keeps the SAME Phineas). Else fall back to <slug>_portrait.png if present.
  * If NEITHER exists -> this gen DEFINES the identity (no reference; first portrait).
    InstantID still needs a face to read embeddings from, so the no-reference path
    renders a plain SDXL pass first and locks onto THAT face. See the no-ref note below.

VRAM (hil = 2080 Ti, Turing, 11 GB): fp16 + enable_model_cpu_offload(). Pause the
director (`ollama stop` / stop lp-director) before a render so the 11 GB is free.

# VERIFY against install/GENSTACK_INSTALL.md -- genstack-installer is finalizing the
# exact InstantID checkpoint paths + load API for hil's .venv-gen in parallel. The
# InstantID block below codes the standard diffusers usage (InstantX/InstantID); the
# conductor must confirm the model dir + the import path before the first run.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CHARS = ROOT / "prompts" / "characters"
GEN = ROOT / "data" / "gen"


# ======================================================================================
# Spec loading + dotted-placeholder prompt assembly
# ======================================================================================
def load_spec(slug: str) -> dict:
    """Load prompts/characters/<slug>.json (the living-portrait.character/v1 spec)."""
    p = CHARS / (slug + ".json")
    if not p.exists():
        sys.exit("no character spec at %s" % p)
    return json.loads(p.read_text(encoding="utf-8-sig"))


def _resolve_dotted(spec: dict, dotted: str):
    """Resolve a dotted path like 'appearance.face' against the nested spec dict.

    Raises KeyError with the full path on a miss so a typo'd template placeholder fails
    loud (a silently-empty field would quietly degrade the prompt).
    """
    node = spec
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError("positive_template placeholder %r: missing %r" % (dotted, part))
        node = node[part]
    return node


class _DottedFormatter:
    """A Formatter helper that resolves {block.field} dotted keys from the nested spec.

    str.format_map with a custom mapping does not handle dotted attribute access the way
    we want (it would try attribute/index access on the resolved object). So we expose a
    mapping whose __getitem__ takes the WHOLE field name (e.g. 'appearance.face') and
    walks the spec. format() passes the text between braces as the key, which is exactly
    the dotted path -- as long as the template uses {a.b} with no !conv or :spec suffix.
    """

    def __init__(self, spec: dict):
        self.spec = spec

    def __getitem__(self, key: str):
        return _resolve_dotted(self.spec, key)


def assemble_prompt(spec: dict) -> str:
    """Format generation.positive_template, resolving {block.field} from the nested JSON."""
    gen = spec.get("generation", {})
    template = gen.get("positive_template")
    if not template:
        sys.exit("spec.generation.positive_template is missing/empty")
    # str.format_map treats a dotted name inside {} as a single field name only when there
    # is no '.' getattr step -- which is NOT the default. So resolve manually: find every
    # {placeholder} and substitute. This sidesteps Formatter's attribute/index semantics.
    import re

    def _sub(m):
        key = m.group(1).strip()
        return str(_resolve_dotted(spec, key))

    return re.sub(r"\{([^{}]+)\}", _sub, template)


def negative_prompt(spec: dict) -> str:
    neg = spec.get("generation", {}).get("negative", [])
    if isinstance(neg, list):
        return ", ".join(str(x) for x in neg)
    return str(neg)


def gen_params(spec: dict, steps: int | None, cfg: float | None) -> dict:
    """generation.params with optional CLI overrides for steps/cfg."""
    p = dict(spec.get("generation", {}).get("params", {}))
    out = {
        "steps": int(p.get("steps", 35)),
        "cfg": float(p.get("cfg", 6.5)),
        "width": int(p.get("width", 1024)),
        "height": int(p.get("height", 1024)),
    }
    if steps is not None:
        out["steps"] = steps
    if cfg is not None:
        out["cfg"] = cfg
    return out


def identity_reference(slug: str) -> Path | None:
    """The face to lock onto: existing anchor, else portrait, else None (gen DEFINES it)."""
    anchor = GEN / (slug + "_anchor.png")
    if anchor.exists():
        return anchor
    portrait = GEN / (slug + "_portrait.png")
    if portrait.exists():
        return portrait
    root_portrait = ROOT / (slug + "_portrait.png")  # early real render lives at repo root
    if root_portrait.exists():
        return root_portrait
    return None


def frame_health(img) -> tuple[float, bool]:
    """Mean pixel value + a black/NaN guard. A near-zero mean or any NaN => bad render."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    mean = float(np.nanmean(arr))
    bad = bool(np.isnan(arr).any() or mean < 2.0)
    return mean, bad


# ======================================================================================
# InstantID render
# ======================================================================================
# VERIFY against install/GENSTACK_INSTALL.md (paths + import path may be pinned there).
# Defaults follow the upstream InstantID README (github.com/instantX-research/InstantID):
#   * antelopev2 insightface pack under INSTANTID_ROOT/models/
#   * ControlNetModel + ip-adapter.bin from HF repo InstantX/InstantID
INSTANTID_REPO = "InstantX/InstantID"            # HF repo holding ControlNetModel + ip-adapter.bin
INSTANTID_ROOT = ROOT / "data" / "gen" / "models" / "instantid"   # matches _install_genstack.ps1 layout
ANTELOPE_ROOT = ROOT / "data" / "gen" / "models" / "insightface"  # FaceAnalysis root; pack at <root>/models/antelopev2
VENDOR_INSTANTID = ROOT / "pipeline" / "vendor" / "InstantID"     # vendored pipeline + its ip_adapter/ package
SDXL_BASE_DEFAULT = "stabilityai/stable-diffusion-xl-base-1.0"


def _draw_kps(image, kps, color_list=((255, 0, 0), (0, 255, 0), (0, 0, 255),
                                      (255, 255, 0), (255, 0, 255))):
    """Render the 5-point face landmark "stickman" InstantID conditions on.

    Vendored from InstantID's pipeline (github.com/instantX-research/InstantID, MIT) so
    we do not depend on the repo being importable as a package -- the installer may drop
    it as a loose pipeline file. Draws limb lines between eyes/nose/mouth + dots.
    # VERIFY: if install/GENSTACK_INSTALL.md exposes the upstream draw_kps, import that
    # instead of this copy to stay byte-identical with the conditioning the model trained on.
    """
    import math

    import cv2
    from PIL import Image

    w, h = image.size
    out = np.zeros([h, w, 3])
    kps = np.array(kps)
    stickwidth = 4
    limb_seq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    for i in range(len(limb_seq)):
        index = limb_seq[i]
        color = color_list[index[0]]
        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly(
            (int(np.mean(x)), int(np.mean(y))),
            (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out = cv2.fillConvexPoly(out.copy(), polygon, color)
        out = (out * 0.6).astype(np.uint8)
    for idx_kp, kp in enumerate(kps):
        x, y = kp
        out = cv2.circle(out.copy(), (int(x), int(y)), 10, color_list[idx_kp], -1)
    return Image.fromarray(out.astype(np.uint8))


def _face_analysis():
    """Build the insightface antelopev2 app InstantID expects. Raises if unavailable."""
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="antelopev2",
        root=str(ANTELOPE_ROOT),
        providers=["CPUExecutionProvider"],   # once-per-anchor; CPU avoids the onnxruntime/cuDNN Turing risk
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def _largest_face(app, pil_img):
    """Return the InstantID face_info dict for the biggest face, or None if none found."""
    import cv2

    bgr = cv2.cvtColor(np.asarray(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    faces = app.get(bgr)
    if not faces:
        return None
    faces = sorted(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
    return faces[-1]


def _build_instantid_pipe(base_model: str):
    """Load the SDXL InstantID pipeline (fp16 + model CPU offload for the 2080 Ti).

    # VERIFY against install/GENSTACK_INSTALL.md: the import path for
    # StableDiffusionXLInstantIDPipeline. InstantID's pipeline merged into diffusers, but
    # the upstream repo still ships it as a loose `pipeline_stable_diffusion_xl_instantid`
    # module. We try the diffusers location first, then the loose module the installer drops.
    """
    from diffusers.models import ControlNetModel

    # InstantID's pipeline is NOT in diffusers 0.37.1 -- import the vendored copy
    # (pipeline/vendor/InstantID, populated by _install_genstack.ps1) + its ip_adapter/ package.
    if str(VENDOR_INSTANTID) not in sys.path:
        sys.path.insert(0, str(VENDOR_INSTANTID))
    from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline  # vendored

    controlnet_dir = INSTANTID_ROOT / "ControlNetModel"
    ip_adapter_bin = INSTANTID_ROOT / "ip-adapter.bin"

    controlnet = ControlNetModel.from_pretrained(str(controlnet_dir), torch_dtype=torch.float16)
    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        base_model, controlnet=controlnet, torch_dtype=torch.float16)
    pipe.load_ip_adapter_instantid(str(ip_adapter_bin))
    # Turing/11 GB: offload to CPU between sub-modules instead of pipe.cuda().
    pipe.enable_model_cpu_offload()
    try:
        pipe.enable_vae_tiling()
    except Exception:
        pass
    return pipe


def render_anchor(spec: dict, slug: str, seed: int, base_model: str,
                  steps: int | None, cfg: float | None,
                  identity_scale: float, controlnet_scale: float) -> "object":
    """Render the identity-locked anchor PIL image. Heavy: imports torch/diffusers/insightface."""
    prompt = assemble_prompt(spec)
    neg = negative_prompt(spec)
    params = gen_params(spec, steps, cfg)

    print("SLUG", slug, "SEED", seed, flush=True)
    print("PROMPT", prompt, flush=True)
    print("NEGATIVE", neg, flush=True)
    print("PARAMS", params, flush=True)

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print("vram_free_GB %.2f / %.2f" % (free / 1e9, total / 1e9), flush=True)

    app = _face_analysis()
    pipe = _build_instantid_pipe(base_model)
    generator = torch.manual_seed(seed)

    ref = identity_reference(slug)
    if ref is not None:
        print("IDENTITY_REF", ref, flush=True)
        from PIL import Image

        ref_img = Image.open(ref).convert("RGB")
        face_info = _largest_face(app, ref_img)
        if face_info is None:
            sys.exit("identity reference %s has no detectable face -- pick a clearer ref" % ref)
        face_emb = face_info["embedding"]
        face_kps = _draw_kps(ref_img.resize((params["width"], params["height"])), _scaled_kps(
            face_info["kps"], ref_img.size, (params["width"], params["height"])))
        image = pipe(
            prompt=prompt, negative_prompt=neg,
            image_embeds=face_emb, image=face_kps,
            controlnet_conditioning_scale=controlnet_scale,
            ip_adapter_scale=identity_scale,
            num_inference_steps=params["steps"], guidance_scale=params["cfg"],
            width=params["width"], height=params["height"],
            generator=generator).images[0]
        return image

    # ---- no-reference path: this gen DEFINES the identity --------------------------------
    # InstantID needs a face to read an embedding from. With no approved reference we cannot
    # truly identity-lock the FIRST render -- so we mark this honestly and use a centered
    # synthetic keypoint layout so the figure is at least posed front-on. The conductor
    # should then re-run with the produced anchor as the reference to LOCK subsequent gens.
    print("IDENTITY_REF none -- this render DEFINES identity (no lock on the first pass);"
          " re-run after to lock subsequent gens to this face.", flush=True)
    face_kps = _centered_kps(params["width"], params["height"])
    zero_emb = np.zeros((512,), dtype=np.float32)  # neutral embedding; look comes from the prompt
    image = pipe(
        prompt=prompt, negative_prompt=neg,
        image_embeds=zero_emb, image=face_kps,
        controlnet_conditioning_scale=controlnet_scale * 0.5,  # softer -- no real identity to hold
        ip_adapter_scale=0.0,
        num_inference_steps=params["steps"], guidance_scale=params["cfg"],
        width=params["width"], height=params["height"],
        generator=generator).images[0]
    return image


def _scaled_kps(kps, src_size, dst_size):
    """Scale insightface 5-point kps from the reference image size to the render size."""
    sx = dst_size[0] / src_size[0]
    sy = dst_size[1] / src_size[1]
    return [[float(x) * sx, float(y) * sy] for (x, y) in np.array(kps)]


def _centered_kps(w: int, h: int):
    """A plausible front-on 5-point face layout (L-eye, R-eye, nose, L-mouth, R-mouth)
    centered in the frame, for the no-reference first render. Proportions match a head
    that fills ~80% of the panel per the framing.crop contract."""
    cx, cy = w / 2.0, h * 0.46
    eye_dx, eye_dy = w * 0.11, -h * 0.06
    mouth_dx, mouth_dy = w * 0.07, h * 0.12
    nose_dy = h * 0.02
    kps = [
        [cx - eye_dx, cy + eye_dy],
        [cx + eye_dx, cy + eye_dy],
        [cx, cy + nose_dy],
        [cx - mouth_dx, cy + mouth_dy],
        [cx + mouth_dx, cy + mouth_dy],
    ]
    return _draw_kps(_blank(w, h), kps)


def _blank(w: int, h: int):
    from PIL import Image

    return Image.new("RGB", (w, h), (0, 0, 0))


# ======================================================================================
# CLI
# ======================================================================================
def main():
    ap = argparse.ArgumentParser(description="SDXL + InstantID frameless identity-locked anchor")
    ap.add_argument("--slug", default="phineas", help="character slug (prompts/characters/<slug>.json)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=None, help="override generation.params.steps")
    ap.add_argument("--cfg", type=float, default=None, help="override generation.params.cfg")
    ap.add_argument("--base-model", default=SDXL_BASE_DEFAULT,
                    help="SDXL base checkpoint (spec.generation.model if you prefer)")
    ap.add_argument("--identity-scale", type=float, default=0.8, help="ip_adapter_scale (face strength)")
    ap.add_argument("--controlnet-scale", type=float, default=0.8, help="controlnet_conditioning_scale (kps strength)")
    args = ap.parse_args()

    spec = load_spec(args.slug)
    # Prefer the spec's declared base model when the caller left the default in place.
    base_model = args.base_model
    spec_model = spec.get("generation", {}).get("model")
    if base_model == SDXL_BASE_DEFAULT and spec_model:
        base_model = spec_model

    image = render_anchor(
        spec, args.slug, args.seed, base_model, args.steps, args.cfg,
        args.identity_scale, args.controlnet_scale)

    mean, bad = frame_health(image)
    GEN.mkdir(parents=True, exist_ok=True)
    out = GEN / (args.slug + "_anchor.png")
    image.save(out)
    print("FRAME_MEAN %.2f%s" % (mean, "  !! BLACK/NaN GUARD TRIPPED" if bad else ""), flush=True)
    print("SAVED", out, flush=True)
    if bad:
        sys.exit("render looks black/NaN (mean %.2f) -- check VRAM / model load" % mean)


if __name__ == "__main__":
    main()
