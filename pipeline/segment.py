"""segment.py -- split a generated portrait into a character cutout + bg plate.

Stage 3 of the generative pipeline (architecture v2 "SAM segment" box):

    generate.py  ->  data/gen/<slug>_portrait.png   (1024x1024 PNG, the painting)
    segment.py   ->  data/gen/<slug>_cutout.png      (RGBA, figure on transparency)
                 ->  data/gen/<slug>_bg.png          (RGB, figure removed + filled)

The cutout's alpha is what the Live2D rig (rig_spec.py) and the player composite
over the background plate, so the figure can be parallax-shifted / parametrically
warped independently of its backdrop.

Two segmentation backends, chosen at runtime:

  * SAM 3.1 (preferred) -- the same model the projection-mapping rig uses
    (projects/sam-3). Prompt-grounded: we ask it for "the person / figure /
    portrait subject" and take the union mask. Heavy: needs torch + CUDA + the
    gated HF checkpoint, none of which are guaranteed on a given box. The import
    is therefore GUARDED -- if anything in the SAM stack is missing we silently
    fall back.

  * GrabCut (fallback) -- a deterministic, dependency-free OpenCV segmenter.
    Seeds a foreground rectangle from the centre of the frame (portraits put the
    figure dead-centre) and refines with cv2.grabCut. Always available wherever
    opencv + numpy are. A centre-ellipse alpha is the last-ditch backstop if even
    grabCut produces an empty mask.

The background plate is filled by inpainting the figure region (cv2.inpaint,
Telea) so the player never sees a hard hole where the character used to be.

    python pipeline/segment.py            # self-test on a synthetic portrait
    python pipeline/segment.py phineas     # segment data/gen/phineas_portrait.png
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "gen"

# The portrait dimensions generate.py emits. Used only for the synthetic
# self-test image and as a resize target for sanity; the real pipeline reads
# whatever generate.py wrote.
PORTRAIT_SIZE = 1024

# Text prompts handed to SAM, most-specific first. SAM grounds on the union of
# whatever it finds; "person" is the strong general anchor, the painterly synonyms
# help on stylised oil-portrait subjects that a literal "person" detector misses.
_SAM_PROMPTS = ("person", "the figure in the portrait", "face", "head and shoulders")


# ---------------------------------------------------------------------------
# SAM 3.1 backend -- import-guarded. Returns a (H, W) uint8 mask (255=fg) or None.
# ---------------------------------------------------------------------------

# Resolved once, lazily, by _load_sam(). Sentinel None = "not tried yet".
_SAM_STATE: dict = {"tried": False, "processor": None, "torch": None, "err": None}


def sam_available() -> bool:
    """True iff the SAM 3.1 image stack imports AND a checkpoint can be built.

    Cheap-ish: imports + (on first call) builds the model. Result memoised so
    repeated calls in a batch don't re-pay the load. Never raises -- a missing
    dep, missing weights, or absent CUDA all resolve to False.
    """
    return _load_sam() is not None


def _sam3_repo_on_path() -> None:
    """Best-effort: add the sibling sam-3 checkout to sys.path so its `sam3`
    package imports even when it was never pip-installed into this venv.

    projects/living-portraits and projects/sam-3 are siblings; the importable
    package lives at projects/sam-3/sam3 (i.e. `sam3/__init__.py` is under
    .../sam-3/sam3/sam3/, with the top of the package tree at .../sam-3/sam3)."""
    candidates = [
        ROOT.parent / "sam-3" / "sam3",   # projects/sam-3/sam3  (package root)
        ROOT.parent / "sam-3",            # projects/sam-3       (fallback layout)
    ]
    for c in candidates:
        if c.is_dir() and str(c) not in sys.path:
            sys.path.insert(0, str(c))


def _load_sam():
    """Lazily import + build the SAM 3.1 image processor. Memoised.

    Returns the Sam3Processor instance, or None if the stack is unavailable for
    any reason (missing torch/sam3, no checkpoint, no GPU, build error).
    """
    if _SAM_STATE["tried"]:
        return _SAM_STATE["processor"]
    _SAM_STATE["tried"] = True
    try:
        _sam3_repo_on_path()
        import torch  # noqa: F401 -- presence check + handed to _sam_segment
        from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
        from sam3.model.sam3_image_processor import Sam3Processor

        # Mirror server.py: prefer the sam3.1 checkpoint, fall back to whatever
        # build_sam3_image_model pulls by default. Both download from a gated HF
        # repo, so this raises (caught below) on a box without the weights/creds.
        try:
            ckpt = download_ckpt_from_hf(version="sam3.1")
            model = build_sam3_image_model(checkpoint_path=ckpt, load_from_HF=False)
        except Exception:
            # Default path -- build_sam3_image_model fetches sam3 weights itself.
            model = build_sam3_image_model()
        _SAM_STATE["processor"] = Sam3Processor(model)
        _SAM_STATE["torch"] = torch
    except Exception as e:  # ImportError, OSError (no creds), RuntimeError (no CUDA)...
        _SAM_STATE["err"] = f"{type(e).__name__}: {e}"
        _SAM_STATE["processor"] = None
    return _SAM_STATE["processor"]


def _squeeze_mask_union(masks, h: int, w: int) -> np.ndarray:
    """Collapse SAM's (N,H,W)/(N,1,H,W)/(H,W) logits-or-bool output into a single
    (H, W) uint8 union mask (255=fg). Mirrors server.py:_union_mask semantics so
    the in-process path matches the HTTP path the projection rig uses."""
    arr = np.asarray(masks)
    if arr.size == 0:
        return np.zeros((h, w), np.uint8)
    while arr.ndim > 3:
        squeezed = False
        for ax in range(arr.ndim):
            if arr.shape[ax] == 1 and ax not in (arr.ndim - 1, arr.ndim - 2):
                arr = np.squeeze(arr, axis=ax)
                squeezed = True
                break
        if not squeezed:
            break
    if arr.ndim == 2:
        out = (arr > 0.5).astype(np.uint8) * 255
    else:
        out = np.any(arr > 0.5, axis=0).astype(np.uint8) * 255
    if out.shape != (h, w):
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)
    return out


def _sam_segment(rgb: np.ndarray) -> Optional[np.ndarray]:
    """Run SAM 3.1 over the prompts and return a (H, W) uint8 fg mask, or None.

    `rgb` is an (H, W, 3) RGB uint8 array. Tries each prompt until one returns a
    non-trivial mask (>=0.5% of the frame), unioning nothing across prompts --
    the first confident hit wins, which keeps a stray "face" detection from
    eating the whole canvas."""
    processor = _load_sam()
    if processor is None:
        return None
    torch = _SAM_STATE["torch"]
    try:
        from PIL import Image
    except Exception:
        return None
    h, w = rgb.shape[:2]
    pil = Image.fromarray(rgb, mode="RGB")
    use_amp = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    for prompt in _SAM_PROMPTS:
        try:
            ctx_amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                       if use_amp else _nullctx())
            with torch.inference_mode(), ctx_amp:
                state = processor.set_image(pil)
                out = processor.set_text_prompt(state=state, prompt=prompt)
            masks = out["masks"]
            if torch.is_tensor(masks):
                masks = masks.float().cpu().numpy()
            mask = _squeeze_mask_union(masks, h, w)
            if (mask > 127).mean() >= 0.005:
                return mask
        except Exception:
            continue
    return None


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ---------------------------------------------------------------------------
# GrabCut fallback -- deterministic, opencv-only.
# ---------------------------------------------------------------------------

def _grabcut_mask(rgb: np.ndarray, iters: int = 5) -> np.ndarray:
    """Deterministic foreground mask via cv2.grabCut seeded from a centre rect.

    Portraits frame the figure centrally facing forward (generate.py prompts a
    "single dignified figure facing forward"), so a generous centred rectangle is
    a reliable GrabCut seed. Returns a (H, W) uint8 mask, 255=fg. Never raises;
    returns a centre-ellipse mask if grabCut degenerates to empty."""
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # Seed rect: inset ~12% each side -> central 76% of the frame.
    mx, my = int(w * 0.12), int(h * 0.08)
    rect = (mx, my, w - 2 * mx, h - my)  # figures usually run to the bottom edge
    gc_mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, gc_mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
        fg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0)
        fg = fg.astype(np.uint8)
    except cv2.error:
        fg = np.zeros((h, w), np.uint8)
    if (fg > 127).mean() < 0.01:
        fg = _center_ellipse_mask(h, w)
    return fg


def _center_ellipse_mask(h: int, w: int) -> np.ndarray:
    """Last-ditch alpha: a head-and-shoulders ellipse centred in the frame.
    Used only when both SAM and GrabCut fail to find a figure."""
    mask = np.zeros((h, w), np.uint8)
    cx, cy = w // 2, int(h * 0.46)
    ax, ay = int(w * 0.30), int(h * 0.42)
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    return mask


# ---------------------------------------------------------------------------
# Mask cleanup + compositing
# ---------------------------------------------------------------------------

def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """Keep the largest connected fg component, close holes, feather the edge.

    A portrait has exactly one subject; dropping all but the biggest blob removes
    speckle (SAM) or background islands (GrabCut). Morphological close fills the
    pinholes inside the figure; a light Gaussian blur on the binary edge yields a
    soft anti-aliased alpha so the cutout doesn't have a jagged matte line."""
    h, w = mask.shape[:2]
    binary = (mask > 127).astype(np.uint8)
    if binary.sum() == 0:
        return _center_ellipse_mask(h, w)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n > 1:
        # label 0 is background; pick the largest non-bg component by area
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        binary = (labels == biggest).astype(np.uint8)

    k = np.ones((7, 7), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    alpha = (binary * 255).astype(np.uint8)
    # Feather: blur then renormalise so the core stays fully opaque.
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=2.0)
    return alpha


def _make_cutout(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """RGBA cutout: original colours with `alpha` as the 4th channel."""
    h, w = rgb.shape[:2]
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = alpha
    return rgba


def _make_bg_plate(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Background plate: figure removed and inpainted so there's no hard hole.

    Dilate the figure mask a touch before inpainting so the soft feathered edge
    (which still carries some of the subject's colour) is also painted over.
    cv2.inpaint (Telea) extrapolates surrounding backdrop into the hole -- good
    enough for an out-of-focus painterly background that sits behind the rig."""
    hole = (alpha > 40).astype(np.uint8) * 255
    hole = cv2.dilate(hole, np.ones((9, 9), np.uint8), iterations=2)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    try:
        filled_bgr = cv2.inpaint(bgr, hole, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    except cv2.error:
        # Degenerate (e.g. whole frame masked) -> blur the original as a soft plate.
        filled_bgr = cv2.GaussianBlur(bgr, (0, 0), sigmaX=18.0)
    return cv2.cvtColor(filled_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def segment(portrait_path, out_dir: Optional[Path] = None) -> dict:
    """Segment a portrait PNG into a character cutout (RGBA) + a background plate.

    Args:
        portrait_path: path to data/gen/<slug>_portrait.png (any RGB/RGBA image).
        out_dir: where to write outputs (defaults to the portrait's own dir).

    Returns a dict:
        {"cutout": <Path>, "bg": <Path>, "mask": <np.ndarray uint8 HxW>,
         "backend": "sam" | "grabcut", "coverage": <float fg fraction>}

    The slug is derived from the portrait filename: "<slug>_portrait.png" ->
    "<slug>". Outputs are "<slug>_cutout.png" and "<slug>_bg.png".
    """
    portrait_path = Path(portrait_path)
    if not portrait_path.exists():
        raise FileNotFoundError(f"portrait not found: {portrait_path}")
    out_dir = Path(out_dir) if out_dir is not None else portrait_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = portrait_path.stem
    slug = stem[:-len("_portrait")] if stem.endswith("_portrait") else stem

    # Load as RGB (drop any alpha generate.py emitted).
    bgr = cv2.imread(str(portrait_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"could not read image: {portrait_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 1. segment -- SAM if available, else GrabCut.
    raw = _sam_segment(rgb)
    backend = "sam"
    if raw is None:
        raw = _grabcut_mask(rgb)
        backend = "grabcut"

    # 2. clean + feather.
    alpha = _clean_mask(raw)
    coverage = float((alpha > 127).mean())

    # 3. composite cutout + bg plate.
    cutout = _make_cutout(rgb, alpha)
    bg = _make_bg_plate(rgb, alpha)

    cutout_path = out_dir / f"{slug}_cutout.png"
    bg_path = out_dir / f"{slug}_bg.png"
    # RGBA cutout -> BGRA for cv2; bg RGB -> BGR.
    cv2.imwrite(str(cutout_path), cv2.cvtColor(cutout, cv2.COLOR_RGBA2BGRA))
    cv2.imwrite(str(bg_path), cv2.cvtColor(bg, cv2.COLOR_RGB2BGR))

    return {
        "cutout": cutout_path,
        "bg": bg_path,
        "mask": alpha,
        "backend": backend,
        "coverage": coverage,
    }


# ---------------------------------------------------------------------------
# Synthetic test image + self-test
# ---------------------------------------------------------------------------

def _synthetic_portrait(size: int = PORTRAIT_SIZE) -> np.ndarray:
    """Build a fake 'oil portrait': a warm gradient backdrop with a clearly
    foregrounded figure (head + torso) in the centre. Deterministic -- gives the
    GrabCut/ellipse fallback an unambiguous central subject to find offline."""
    h = w = size
    rgb = np.zeros((h, w, 3), np.uint8)
    # Backdrop: vertical chiaroscuro gradient, deep umber -> near-black.
    grad = np.linspace(70, 12, h).astype(np.uint8)
    rgb[..., 0] = grad[:, None]            # R
    rgb[..., 1] = (grad * 0.7)[:, None]    # G
    rgb[..., 2] = (grad * 0.45)[:, None]   # B
    # Subtle vignette / brush texture so it isn't a flat plane.
    noise = (np.random.default_rng(7).normal(0, 6, (h, w, 3))).astype(np.int16)
    rgb = np.clip(rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    cx = w // 2
    # Torso (trapezoid-ish ellipse, lower half), warm slate cloth.
    cv2.ellipse(rgb, (cx, int(h * 0.82)), (int(w * 0.26), int(h * 0.30)),
                0, 0, 360, (120, 110, 140), -1)
    # Head (skin oval, upper-centre).
    cv2.ellipse(rgb, (cx, int(h * 0.40)), (int(w * 0.135), int(h * 0.175)),
                0, 0, 360, (205, 170, 150), -1)
    # Hair cap.
    cv2.ellipse(rgb, (cx, int(h * 0.31)), (int(w * 0.15), int(h * 0.11)),
                0, 180, 360, (60, 45, 40), -1)
    # Eyes -- two small dark ovals (gives rig_spec's Haar/proportional path
    # something plausible to anchor on; Haar won't fire on a painting but the
    # proportional defaults will).
    eye_y = int(h * 0.38)
    for ex in (cx - int(w * 0.055), cx + int(w * 0.055)):
        cv2.ellipse(rgb, (ex, eye_y), (int(w * 0.022), int(h * 0.013)),
                    0, 0, 360, (40, 35, 35), -1)
    return rgb


def _selftest() -> int:
    print("[segment] self-test", flush=True)
    print("[segment] SAM available:", sam_available(),
          f"({_SAM_STATE['err']})" if _SAM_STATE.get("err") else "", flush=True)

    tmp = OUT / "_selftest"
    tmp.mkdir(parents=True, exist_ok=True)
    synth = _synthetic_portrait()
    portrait = tmp / "selftest_portrait.png"
    cv2.imwrite(str(portrait), cv2.cvtColor(synth, cv2.COLOR_RGB2BGR))
    print(f"[segment] wrote synthetic portrait {portrait} {synth.shape}", flush=True)

    res = segment(portrait)
    print(f"[segment] backend={res['backend']}  coverage={res['coverage']:.3f}", flush=True)
    print(f"[segment] cutout -> {res['cutout']}", flush=True)
    print(f"[segment] bg     -> {res['bg']}", flush=True)

    # Assertions: files exist, cutout is RGBA, bg is RGB, alpha is non-trivial.
    assert res["cutout"].exists(), "cutout PNG missing"
    assert res["bg"].exists(), "bg PNG missing"
    cut = cv2.imread(str(res["cutout"]), cv2.IMREAD_UNCHANGED)
    bg = cv2.imread(str(res["bg"]), cv2.IMREAD_UNCHANGED)
    assert cut is not None and cut.ndim == 3 and cut.shape[2] == 4, \
        f"cutout must be 4-channel RGBA, got {None if cut is None else cut.shape}"
    assert bg is not None and bg.ndim == 3 and bg.shape[2] == 3, \
        f"bg must be 3-channel RGB, got {None if bg is None else bg.shape}"
    assert cut.shape[:2] == synth.shape[:2], "cutout size != portrait size"
    assert bg.shape[:2] == synth.shape[:2], "bg size != portrait size"
    cov = float((cut[..., 3] > 127).mean())
    assert 0.02 < cov < 0.95, f"implausible figure coverage {cov:.3f}"
    print(f"[segment] OK -- cutout RGBA {cut.shape}, bg RGB {bg.shape}, "
          f"alpha coverage {cov:.3f}", flush=True)
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] not in ("--selftest", "selftest"):
        slug = sys.argv[1]
        portrait = OUT / f"{slug}_portrait.png"
        res = segment(portrait)
        print(f"backend={res['backend']}  coverage={res['coverage']:.3f}", flush=True)
        print("cutout:", res["cutout"], flush=True)
        print("bg:", res["bg"], flush=True)
        return 0
    return _selftest()


if __name__ == "__main__":
    raise SystemExit(main())
