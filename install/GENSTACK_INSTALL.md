# Generation-stack install plan — hil (`.venv-gen`)

**Target:** SDXL anchor + InstantID identity-lock + LivePortrait motion, on hil's existing
`.venv-gen`. Authored 2026-05-27 by `genstack-installer` (research-only; nothing executed on hil).

Companion script: [`_install_genstack.ps1`](./_install_genstack.ps1) — the exact, idempotent
commands the conductor runs on hil. This doc is the *why* + the verification steps; the script
is the *what*.

---

## 0. The target environment (given — do not re-probe to author this)

| Fact | Value | Consequence for this plan |
|---|---|---|
| GPU | RTX 2080 Ti = **Turing, sm_75** | **fp16 only.** No bf16. No FlashAttention / Ampere kernels. |
| VRAM | 11 GB | SDXL + InstantID (ControlNet + IP-Adapter) needs `enable_model_cpu_offload()` to fit. |
| OS | Windows 11 | onnxruntime DLL-resolution is the sharp edge (see §3). Bash paths differ. |
| Python | `C:\living-portraits\.venv-gen\Scripts\python.exe` | all installs target this interpreter explicitly. |
| uv | `C:\Users\Immersive Commons 1\.local\bin\uv.exe` (**path has spaces** — always quote) | `& $uv pip install --python <pyexe> ...` |
| torch | **2.5.1+cu121** (already installed) | bundles its own CUDA 12.1 + **cuDNN 9** runtime DLLs. Pins the onnxruntime-gpu choice. **Do NOT reinstall torch.** |
| diffusers | **0.37.1** | InstantID pipeline file imports are all stable here (verified §2). Has CVE-2026-44513 in the `custom_pipeline` auto-fetch path → we **vendor** instead (§2). |
| transformers | **5.9.0** | InstantID & LivePortrait `requirements.txt` pin transformers `4.38.0`. **Never `pip install -r requirements.txt`** — it would downgrade transformers and break diffusers 0.37.1. Install LivePortrait deps *selectively* (§4). |
| Already present | accelerate, insightface, opencv(-python), imageio(+imageio-ffmpeg), sentencepiece, tiktoken, safetensors | insightface present → we still need **onnxruntime-gpu** (its inference backend) and the **antelopev2** model pack. |
| Disk | 511 GB free | total new download ≈ **14.6 GB** (§5). Plenty. |
| HF cache | default (`~/.cache/huggingface`) | SDXL goes to the hub cache; InstantID + LivePortrait + antelopev2 go to explicit `data/gen/models/...` dirs (gitignored). |

---

## 1. SDXL base — the anchor model

**Repo (verified, HF API):** `stabilityai/stable-diffusion-xl-base-1.0`, **`variant="fp16"`**.

**fp16 on-disk (verified via `/api/models/...?blobs=true`, 2026-05-27):**

| File | Size |
|---|---|
| `unet/diffusion_pytorch_model.fp16.safetensors` | 5.14 GB |
| `text_encoder_2/model.fp16.safetensors` | 1.39 GB |
| `text_encoder/model.fp16.safetensors` | 246 MB |
| `vae/diffusion_pytorch_model.fp16.safetensors` | 167 MB |
| **fp16 weights total** | **≈ 7.11 GB** (+ small configs/tokenizers) |

We do **not** add the SDXL refiner — the spec's `params` (35 steps, base only) doesn't call for it,
and it would cost another ~6 GB of VRAM pressure we don't have.

**Load snippet (fits 11 GB via offload):**

```python
import torch
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
)
pipe.enable_model_cpu_offload()        # REQUIRED on 11 GB — keeps idle submodels on CPU
pipe.enable_vae_slicing()              # cheap VRAM win at 1024²
# NOTE: for the InstantID path we do NOT load this plain pipeline — InstantID wraps SDXL
#       with a ControlNet (see §2). This snippet is the smoke test / no-identity fallback only.
```

> **Turing note:** do **not** call `pipe.enable_xformers_memory_efficient_attention()` — xformers'
> efficient kernels are unreliable on sm_75 and we did not install xformers. diffusers' default
> attention (or `enable_attention_slicing()`) is correct for Turing.

---

## 2. InstantID — identity lock (the SAME face every re-gen, both panels)

### Decision: vendor the standalone pipeline, **not** the diffusers community auto-fetch

There are two ways to get `StableDiffusionXLInstantIDPipeline`:

1. **diffusers community pipeline** — `DiffusionPipeline.from_pretrained(..., custom_pipeline="instantid")`.
   This *auto-downloads remote code at runtime*. On **diffusers 0.37.1 this is exposed to
   CVE-2026-44513** (a `trust_remote_code` bypass in the `custom_pipeline` path; fixed in 0.38.0).
   Since the spec pins 0.37.1, runtime remote-fetch is the wrong call.
2. **Vendor the standalone repo** — clone `instantX-research/InstantID` once into
   `pipeline/vendor/InstantID/`, pin it, and import the pipeline from disk. No runtime remote
   fetch, fully reproducible, and it works on 0.37.1.

**We choose (2).** It is also *required*, not just preferred: the standalone pipeline file imports
`from ip_adapter import Resampler, IPAttnProcessor, AttnProcessor` — `ip_adapter/` is a **local
package inside the InstantID repo**, not a PyPI package. Vendoring the repo brings the pipeline file
*and* its `ip_adapter/` dependency together. (The community single-file pipeline inlines these,
which is why it looks simpler — but it's the remote-fetch path we're avoiding.)

**diffusers-0.37.1 compatibility (verified by reading the pipeline source):** the file imports only
stable diffusers symbols — `StableDiffusionXLControlNetPipeline`, `ControlNetModel`,
`MultiControlNetModel`, `PipelineImageInput`, `StableDiffusionXLPipelineOutput`,
`is_torch_version`, `is_xformers_available`. All present in 0.37.1. No cutting-edge API. ✅

> **Org-name correction:** the InstantID code repo is **`instantX-research/InstantID`** on GitHub
> (the old `InstantID/InstantID` link in the HF model card is stale). The *model weights* repo is
> **`InstantX/InstantID`** on Hugging Face. Two different orgs — don't conflate.

### Weights (verified, HF API `InstantX/InstantID?blobs=true`, 2026-05-27)

| File | Path under `checkpoints/` | Size |
|---|---|---|
| IdentityNet (ControlNet) | `ControlNetModel/diffusion_pytorch_model.safetensors` | **2.50 GB** |
| IdentityNet config | `ControlNetModel/config.json` | ~1 KB |
| Image-prompt adapter | `ip-adapter.bin` | **1.69 GB** |
| **InstantID weights total** | | **≈ 4.19 GB** |

Pull these with `hf_hub_download` (huggingface_hub is already in `.venv-gen`):

```python
from huggingface_hub import hf_hub_download
DST = "data/gen/models/instantid"   # gitignored
hf_hub_download("InstantX/InstantID", "ControlNetModel/config.json", local_dir=DST)
hf_hub_download("InstantX/InstantID", "ControlNetModel/diffusion_pytorch_model.safetensors", local_dir=DST)
hf_hub_download("InstantX/InstantID", "ip-adapter.bin", local_dir=DST)
```

### Face analysis: antelopev2 + onnxruntime — **the Turing sharp edge** (read §3 first)

InstantID's `FaceAnalysis(name='antelopev2')` runs **insightface on onnxruntime** to detect the face
and extract the 512-d embedding + 5-point keypoints that condition the ControlNet. This is the one
place the stack touches a GPU runtime *other* than torch, and it is the **single biggest risk** (§3).

**antelopev2 pack — robust HF mirror (avoid the flaky GitHub release):** the canonical source is
`https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip`, but that release
asset has a documented history of download failures. Use the HF mirror **`DIAMONIK7777/antelopev2`**
(verified: 5 onnx files, ~428 MB total):

| File | Size |
|---|---|
| `glintr100.onnx` (recognition / 512-d embedding) | 261 MB |
| `1k3d68.onnx` (3D landmarks) | 144 MB |
| `scrfd_10g_bnkps.onnx` (detector) | 16.9 MB |
| `2d106det.onnx` (2D landmarks) | 5.0 MB |
| `genderage.onnx` | 1.3 MB |
| **antelopev2 total** | **≈ 428 MB** |

**On-disk layout matters.** insightface's `FaceAnalysis(name='antelopev2', root=R)` looks for the five
files at **`R/models/antelopev2/*.onnx`**. The `DIAMONIK7777` repo is *flat* (no `models/antelopev2/`
prefix), so the script downloads into `data/gen/models/insightface/models/antelopev2/` explicitly and
passes `root="data/gen/models/insightface"`. Getting this path wrong → silent re-download attempt of
the GitHub zip → failure. The script pins the layout so insightface never reaches for the network.

### InstantID load snippet (vendored, CPU-safe face analysis, 11 GB offload)

```python
import sys, torch
sys.path.insert(0, "pipeline/vendor/InstantID")          # brings pipeline + ip_adapter
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps
from diffusers.models import ControlNetModel
from insightface.app import FaceAnalysis

# Face analysis — CPU provider is the SAFE default on Turing/Windows (runs once per anchor; §3).
app = FaceAnalysis(name="antelopev2", root="data/gen/models/insightface",
                   providers=["CPUExecutionProvider"])   # swap to CUDA only after the §6 probe passes
app.prepare(ctx_id=0, det_size=(640, 640))

controlnet = ControlNetModel.from_pretrained(
    "data/gen/models/instantid/ControlNetModel", torch_dtype=torch.float16)

pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet, torch_dtype=torch.float16)
pipe.load_ip_adapter_instantid("data/gen/models/instantid/ip-adapter.bin")
pipe.enable_model_cpu_offload()      # do NOT pipe.cuda() on 11 GB — offload instead
```

### Fallback if InstantID won't co-exist with 11 GB + offload

InstantID adds a 2.5 GB ControlNet **on top of** SDXL. If it OOMs even with offload, fall back to
**IP-Adapter-FaceID-plusv2 (SDXL)** — same "lock one face" goal, lighter (no ControlNet), native to
diffusers' `load_ip_adapter`:

| Asset | Repo / file | Size |
|---|---|---|
| FaceID-plusv2 adapter | `h94/IP-Adapter-FaceID` → `ip-adapter-faceid-plusv2_sdxl.bin` | ~1.1 GB |
| CLIP image encoder | `h94/IP-Adapter` → `models/image_encoder/` | ~2.5 GB |

```python
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, variant="fp16")
pipe.load_ip_adapter("h94/IP-Adapter-FaceID", subfolder=None,
                     weight_name="ip-adapter-faceid-plusv2_sdxl.bin", image_encoder_folder=None)
pipe.set_ip_adapter_scale(0.7)
pipe.enable_model_cpu_offload()
# still needs insightface antelopev2 to extract the FaceID embedding from the source face
```

This is a *fallback*, not the plan — InstantID gives a stronger identity lock + pose control, which is
what "same Phineas on both panels" wants. Script installs InstantID; the FaceID assets are listed so
the conductor can pivot in one command if §6's InstantID smoke OOMs.

---

## 3. THE BIGGEST TURING RISK — onnxruntime-gpu ↔ cuDNN, on Windows

**This is the one thing most likely to eat an afternoon.** insightface's face analysis is the only
GPU dependency outside torch, and onnxruntime loads its **own** CUDA + cuDNN DLLs — which must match
the runtime, on a Windows DLL search path that also contains torch's bundled CUDA 12.1 / cuDNN 9 DLLs.

The version coupling (verified across onnxruntime docs + multiple field reports):

| onnxruntime-gpu | CUDA | cuDNN |
|---|---|---|
| 1.17.x | 11.x | 8.x (wants `libcublas*11`) |
| 1.18.x | 12.x | **8.x** |
| 1.19.x / 1.20.x | 12.x | **9.x** |
| ≥ 1.21.0 | 12.x | 9.x + adds `onnxruntime.preload_dlls()` to fix Windows DLL clashes |

hil's torch is **cu121 → cuDNN 9**. Therefore:

- **Do NOT** install LivePortrait's pinned `onnxruntime-gpu==1.18.0` (that targets cuDNN **8** → it
  won't find cuDNN 9 and silently falls back to CPU, or errors).
- **Install `onnxruntime-gpu==1.20.1`** — the version field-verified working with torch 2.5.1+cu121
  on Windows (cuDNN 9). (1.21+ would also work and adds `preload_dlls`, but 1.20.1 is the known-good
  point; bump to 1.21.x only if the §6 probe still can't see the GPU.)

**The de-risking move: run face analysis on CPU.** The face pass runs **once per anchor image**
(seconds, not per-frame). The script's default is `providers=["CPUExecutionProvider"]`, which **removes
this entire risk class** — no GPU-DLL matching required, InstantID still works, anchors just take a few
extra seconds to detect the face. We attempt GPU only as an *optimization* after a probe confirms
`CUDAExecutionProvider` actually loads (§6). **If you change one thing under time pressure, keep face
analysis on CPU and move on.**

> Common silent-failure signature: `app.prepare()` prints `Applied providers: ['CPUExecutionProvider']`
> even though you asked for CUDA → onnxruntime couldn't load the CUDA EP (DLL mismatch). That's a
> warning, not a crash; on CPU it still produces correct embeddings, so InstantID still works.

---

## 4. LivePortrait — the motion engine (load-bearing)

**Repo (code):** `https://github.com/KwaiVGI/LivePortrait`.
**Weights (HF, verified):** `KlingTeam/LivePortrait` — the org was renamed; `KwaiVGI/LivePortrait`
on HF now 404s, `KlingTeam/LivePortrait` is the live weights repo (24 files, 2.14 GB total incl.
animals mode).

### Install LivePortrait deps SELECTIVELY — never `-r requirements.txt`

LivePortrait's `requirements.txt` does `-r requirements_base.txt` and then pins
`onnxruntime-gpu==1.18.0` + `transformers==4.38.0`. Both are **poison** for hil:

- `transformers==4.38.0` would **downgrade** hil's 5.9.0 and break diffusers 0.37.1.
- `onnxruntime-gpu==1.18.0` targets cuDNN 8 (wrong for cu121 / cuDNN 9 — see §3).

LivePortrait does **not** pin torch/torchvision (verified — neither appears in either requirements
file), so torch 2.5.1+cu121 is left intact. We install only the *new* runtime deps it actually needs,
letting torch/transformers/onnxruntime/insightface/opencv/imageio stay at hil's versions:

```
tyro pykalman lmdb rich ffmpeg-python  scikit-image  albumentations
```

(Already on hil and deliberately NOT reinstalled: numpy, opencv-python, scipy, imageio,
imageio-ffmpeg, onnx, pillow, matplotlib, tqdm, pyyaml. `gradio` is skipped — we drive LivePortrait
from Python, not its web UI.) Use `& $uv pip install <pkgs>` **without** `==` pins so uv resolves
against hil's existing torch/numpy and doesn't fight the lock.

> `tyro` is the one that bites if missing — LivePortrait's config/CLI is built on it, and `import`ing
> the inference module pulls it in. `pykalman` is used for driving-signal smoothing. `scikit-image`
> + `albumentations` are imported at module load. The rest are I/O niceties.

### Weights — humans-only subset (we don't need animals mode)

`huggingface-cli download KlingTeam/LivePortrait` pulls **2.14 GB** (incl. `liveportrait_animals/` +
`xpose.pth` 435 MB, which we don't use). To save the animals download, fetch only the human tree
(`liveportrait/*` + `insightface/*`) — **≈ 660 MB** of the 2.14 GB:

| File | Size |
|---|---|
| `liveportrait/base_models/spade_generator.pth` | 222 MB |
| `liveportrait/base_models/warping_module.pth` | 182 MB |
| `liveportrait/landmark.onnx` | 115 MB |
| `liveportrait/base_models/motion_extractor.pth` | 113 MB |
| `liveportrait/base_models/appearance_feature_extractor.pth` | 3.4 MB |
| `liveportrait/retargeting_models/stitching_retargeting_module.pth` | 2.4 MB |
| `insightface/models/buffalo_l/det_10g.onnx` | 16.9 MB |
| `insightface/models/buffalo_l/2d106det.onnx` | 5.0 MB |
| **humans subset** | **≈ 660 MB** |

> **LivePortrait ships its OWN insightface pack** (`buffalo_l`, inside the weights repo) — it does
> **not** use antelopev2. So buffalo_l (LivePortrait) and antelopev2 (InstantID) are two separate face
> packs in two separate locations. Don't try to share them. The script keeps LivePortrait's `buffalo_l`
> under its own `pretrained_weights/insightface/` exactly where its config expects it.

### How it's invoked from Python (not just the CLI)

LivePortrait is a cloned repo, not a pip package — so it runs from inside the repo dir. The
documented CLI is `python inference.py -s <source.jpg> -d <driving.mp4>`. For our bake pipeline
(task #3) it's driven in-process via its own classes:

```python
# cwd must be the LivePortrait repo root (it uses repo-relative config/weight paths)
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline

inference_cfg = InferenceConfig(flag_use_half_precision=True)   # fp16 → Turing-safe
crop_cfg = CropConfig()
args = ArgumentConfig(source="<anchor.png>", driving="<behavior_driving.mp4_or_.pkl>",
                      output_dir="data/gen/loops")
pipe = LivePortraitPipeline(inference_cfg=inference_cfg, crop_cfg=crop_cfg)
pipe.execute(args)
```

The exact wiring (anchor + behavior → seamless neutral→behaviour→neutral loop, register on
`runtime/clip_graph.py`) is **task #3's** job — this plan just guarantees the import works and the
weights resolve. The smoke test (§6) only does `import` + checkpoint-presence, no generation.

### Turing / fp16 / Windows gotchas for LivePortrait

- **fp16 is supported and is the right setting:** `InferenceConfig(flag_use_half_precision=True)`.
  LivePortrait is *not* a per-frame diffusion model (it's a keypoint-warp network), so it avoids the
  Ampere-kernel wall that sank Wan/LTX on this card. This is exactly why the spec picked it.
- **`flag_do_torch_compile` is Linux/Windows-optional and risky on Turing** — leave it OFF. It needs
  a working compiler toolchain and gains little on sm_75.
- **Animals mode needs X-Pose, which compiles a custom CUDA op** — we don't install it (humans only),
  sidestepping a Windows build that frequently fails.
- **FFmpeg must be on PATH** for video I/O. hil already has `imageio-ffmpeg`; the script verifies
  `imageio_ffmpeg.get_ffmpeg_exe()` resolves and notes that if LivePortrait's CLI wants a *system*
  ffmpeg, point it at that exe (LivePortrait reads `ffmpeg`/`ffprobe` from PATH for some ops).
- LivePortrait's own `requirements` cap onnxruntime at 1.18 for *their* CUDA-11.8 recipe — we
  deliberately diverge (§3). Its `landmark.onnx` + `buffalo_l` onnx run through the **same**
  onnxruntime we set up in §3, so the §6 CPU-vs-CUDA provider decision applies here too.

---

## 5. Total download budget

| Component | Size |
|---|---|
| SDXL base (fp16) | ≈ 7.11 GB |
| InstantID weights (ControlNet + ip-adapter) | ≈ 4.19 GB |
| antelopev2 (InstantID face analysis) | ≈ 0.43 GB |
| LivePortrait humans subset + buffalo_l | ≈ 0.66 GB |
| Python wheels (onnxruntime-gpu + LivePortrait deps) | ≈ 0.2 GB |
| **Total new on disk** | **≈ 14.6 GB** (of 511 GB free) |

If `huggingface-cli download KlingTeam/LivePortrait` is used whole (animals included) instead of the
humans subset, add ~1.48 GB → ≈ 16.1 GB. Still trivial against 511 GB.

---

## 6. Verification — import + presence only, NO full generation

Run **after** the install script, with the show paused (the director's qwen3 also wants the 11 GB —
see the spec's GPU note). Each check is independent; a failure localizes the problem.

1. **Env probe** (already exists): `& .\.venv-gen\Scripts\python.exe pipeline\_probe_caps.py`
   → expect `compute_capability (7, 5) Turing,fp16-only`, `cuda_avail True`, diffusers 0.37.1,
   transformers **still 5.9.0** (proves no downgrade).

2. **onnxruntime providers** (the §3 risk, checked directly):
   ```python
   import onnxruntime as ort
   print(ort.__version__)                 # expect 1.20.1
   print(ort.get_available_providers())   # CPUExecutionProvider always; CUDAExecutionProvider = bonus
   ```

3. **insightface + antelopev2 load** (CPU provider — the safe default):
   ```python
   from insightface.app import FaceAnalysis
   app = FaceAnalysis(name="antelopev2", root="data/gen/models/insightface",
                      providers=["CPUExecutionProvider"])
   app.prepare(ctx_id=0, det_size=(640, 640))   # must NOT try to download the GitHub zip
   ```
   If this reaches for the network → the `models/antelopev2/` path layout is wrong (§2).

4. **SDXL loads at fp16** (no generation):
   ```python
   import torch; from diffusers import StableDiffusionXLPipeline
   p = StableDiffusionXLPipeline.from_pretrained(
       "stabilityai/stable-diffusion-xl-base-1.0",
       torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
   print("SDXL ok")
   ```

5. **InstantID pipeline imports + ControlNet loads** (vendored path; no generation):
   ```python
   import sys, torch; sys.path.insert(0, "pipeline/vendor/InstantID")
   from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps
   from diffusers.models import ControlNetModel
   cn = ControlNetModel.from_pretrained("data/gen/models/instantid/ControlNetModel",
                                        torch_dtype=torch.float16)
   print("InstantID import + ControlNet ok")
   ```

6. **LivePortrait imports + checkpoints present** (no generation):
   ```python
   import os, sys
   LP = "pipeline/vendor/LivePortrait"; sys.path.insert(0, LP); os.chdir(LP)
   from src.live_portrait_pipeline import LivePortraitPipeline      # pulls tyro/pykalman/etc.
   need = ["pretrained_weights/liveportrait/base_models/spade_generator.pth",
           "pretrained_weights/liveportrait/base_models/warping_module.pth",
           "pretrained_weights/liveportrait/base_models/motion_extractor.pth",
           "pretrained_weights/liveportrait/landmark.onnx"]
   assert all(os.path.exists(p) for p in need), [p for p in need if not os.path.exists(p)]
   print("LivePortrait import + checkpoints ok")
   ```

All six printing OK = the stack is installed and the generation work (tasks #2-anchor / #3-bake) can
start. None of these generate an image or a frame — they only prove imports resolve and weights are on
disk, per the "import/smoke only" scope.

---

## 7. Open items / unverified claims (flagged honestly)

- **`onnxruntime-gpu==1.20.1` GPU acceleration on *this exact* box is not verified by me** — it's the
  field-verified known-good version for torch 2.5.1+cu121 / cuDNN 9 on Windows, but whether
  `CUDAExecutionProvider` actually loads on hil can only be confirmed by running §6 step 2 on hil.
  **Mitigation built into the plan:** default to `CPUExecutionProvider`, which needs no GPU DLLs and
  is plenty fast for a once-per-anchor face pass. The whole plan works on CPU face analysis.
- **InstantID + SDXL fitting in 11 GB with `enable_model_cpu_offload()` is expected, not measured.**
  InstantID's own demos cite ~11–16 GB; offload should bring it under 11, but if §6 step 5's *generation*
  (later, task #2-anchor) OOMs, the documented pivot is IP-Adapter-FaceID-plusv2 (§2 fallback).
- **LivePortrait in-process class names** (`src.live_portrait_pipeline.LivePortraitPipeline`,
  `ArgumentConfig`, etc.) are from the public repo layout; I read the README/CLI but did not execute the
  import. If the module path shifted in a newer commit, §6 step 6 will surface it — pin the repo to a
  known commit in the script to avoid drift.
- **antelopev2 mirror `DIAMONIK7777/antelopev2`** is a community re-host of the official insightface
  v0.7 release (file names + sizes match the canonical pack). It is not an Anthropic/insightface-official
  repo. The official GitHub zip is the alternative if the mirror is distrusted (script keeps both URLs).
- **diffusers CVE-2026-44513** (the reason we vendor InstantID rather than auto-fetch `custom_pipeline`)
  is reported fixed in 0.38.0; I did not independently audit 0.37.1's code path, but vendoring sidesteps
  it entirely regardless.
- **LivePortrait deps install without version pins** (uv resolves against hil's torch/numpy). I verified
  the *names* from `requirements_base.txt` but not that uv's resolver leaves every existing package
  untouched — `& $uv pip install` output should be eyeballed for any unexpected
  torch/transformers/numpy/onnxruntime change before declaring success.

---

### One-line summary for the conductor

Install **SDXL fp16 (7.1 GB)** + **vendored InstantID (4.2 GB weights + antelopev2 0.43 GB,
face-analysis on CPU)** + **LivePortrait humans weights (0.66 GB) with deps installed *selectively***
(never `-r requirements.txt`, never `onnxruntime-gpu==1.18.0`, never `transformers==4.38.0`). Use
**`onnxruntime-gpu==1.20.1`** to match torch-cu121's cuDNN 9. Biggest risk = onnxruntime↔cuDNN GPU
matching on Windows → neutralized by defaulting face analysis to CPU. ≈ 14.6 GB total.
