# _install_genstack.ps1 -- install the SDXL + InstantID + LivePortrait generation stack
# into hil's existing .venv-gen.  RUN ON hil (the RTX 2080 Ti box), NOT locally.
#
#   Plan + rationale + verification: install/GENSTACK_INSTALL.md   (read it first)
#
# Idempotent: every step is safe to re-run. uv pip is no-op if already satisfied;
# hf_hub_download / huggingface-cli skip files already in place; git clone is guarded
# by a dir-exists check. Detached-friendly: writes install_genstack.log, drops
# install_genstack.done at the end. Does NOT touch torch/transformers/insightface
# (already on hil) and never runs `-r requirements.txt` (would downgrade them).
#
#   Suggested invocation (detached, survives SSH drop), from C:\living-portraits:
#     powershell -ExecutionPolicy Bypass -File install\_install_genstack.ps1 *> install_genstack.log
#
# This script DOWNLOADS ~14.6 GB and installs wheels. It does NOT generate any image
# or video and does NOT start/stop the director or player. Pause the show yourself if
# you want the 11 GB free for the later anchor bake (see GENSTACK_INSTALL.md GPU note).

$ErrorActionPreference = "Continue"   # one failed download must not abort the rest

# --- pinned paths (hil) -- uv path HAS SPACES, must stay quoted -------------------
$Repo = "C:\living-portraits"
$Py   = "C:\living-portraits\.venv-gen\Scripts\python.exe"
$Uv   = "C:\Users\Immersive Commons 1\.local\bin\uv.exe"
Set-Location $Repo
Remove-Item install_genstack.done -ErrorAction SilentlyContinue

# Model + vendor destinations (under data/ + pipeline/vendor/ -- data/ is gitignored)
$ModelDir = "$Repo\data\gen\models"
$VendorDir = "$Repo\pipeline\vendor"
New-Item -ItemType Directory -Force -Path "$ModelDir\instantid" | Out-Null
New-Item -ItemType Directory -Force -Path "$ModelDir\insightface\models\antelopev2" | Out-Null
New-Item -ItemType Directory -Force -Path $VendorDir | Out-Null

function Section($m) { Write-Output ("`n=== " + $m + " ===") }

# Sanity: confirm the interpreter + uv exist before doing anything.
if (-not (Test-Path $Py)) { Write-Output "FATAL: .venv-gen python not found at $Py"; exit 1 }
if (-not (Test-Path $Uv)) { Write-Output "FATAL: uv not found at $Uv"; exit 1 }
Write-Output "python: $Py"
Write-Output "uv:     $Uv"


# =====================================================================================
# 1. PYTHON DEPS  (selective -- NEVER `-r requirements.txt`; see GENSTACK_INSTALL.md §3-4)
# =====================================================================================

# 1a. onnxruntime-gpu pinned to 1.20.1 -- matches torch 2.5.1+cu121's bundled cuDNN 9.
#     (LivePortrait's own pin of 1.18.0 targets cuDNN 8 -> WRONG for this box.)
#     If a previous attempt installed a different onnxruntime, force the right one.
Section "onnxruntime-gpu==1.20.1 (cuDNN 9, matches torch cu121)"
& $Uv pip install --python $Py "onnxruntime-gpu==1.20.1"

# 1b. LivePortrait's NEW runtime deps only -- NO version pins (let uv resolve against
#     hil's existing torch/numpy/opencv so nothing gets downgraded). torch, transformers,
#     numpy, opencv, scipy, imageio*, onnx, pillow are already present and left alone.
#     `tyro` is load-bearing (LP's config/CLI); pykalman smooths the driving signal.
Section "LivePortrait deps (selective, unpinned): tyro pykalman lmdb rich ffmpeg-python scikit-image albumentations"
& $Uv pip install --python $Py tyro pykalman lmdb rich ffmpeg-python scikit-image albumentations

# 1c. Belt-and-suspenders: confirm the install did NOT downgrade the load-bearing libs.
#     If any line below is unexpected (transformers != 5.9.x, torch != 2.5.1+cu121),
#     STOP and reconcile before continuing -- diffusers 0.37.1 will break otherwise.
Section "post-deps version guard (transformers must stay 5.9.x, torch 2.5.1+cu121)"
& $Py -c "import torch,transformers,diffusers,onnxruntime as ort; print('torch',torch.__version__);print('transformers',transformers.__version__);print('diffusers',diffusers.__version__);print('onnxruntime',ort.__version__);print('ort_providers',ort.get_available_providers())"


# =====================================================================================
# 2. SDXL BASE (fp16) -> HF hub cache    (~7.11 GB; idempotent -- cached files skip)
# =====================================================================================
Section "SDXL base 1.0 fp16 -> HF cache (~7.11 GB)"
& $Py -c @"
import torch
from diffusers import StableDiffusionXLPipeline
# Download-only: pull fp16 weights into the hub cache. No .to(cuda), no generation.
StableDiffusionXLPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0',
    torch_dtype=torch.float16, use_safetensors=True, variant='fp16')
print('SDXL fp16 cached OK')
"@


# =====================================================================================
# 3. INSTANTID WEIGHTS -> data/gen/models/instantid   (~4.19 GB; hf_hub_download skips cached)
# =====================================================================================
Section "InstantID weights (ControlNet 2.50 GB + ip-adapter 1.69 GB)"
& $Py -c @"
from huggingface_hub import hf_hub_download
DST = r'$ModelDir\instantid'
for fn in ['ControlNetModel/config.json',
           'ControlNetModel/diffusion_pytorch_model.safetensors',
           'ip-adapter.bin']:
    p = hf_hub_download('InstantX/InstantID', fn, local_dir=DST)
    print('have', p)
print('InstantID weights OK')
"@


# =====================================================================================
# 4. ANTELOPEV2 face pack -> data/gen/models/insightface/models/antelopev2/  (~0.43 GB)
#    insightface expects <root>/models/antelopev2/*.onnx . The DIAMONIK7777 mirror is
#    FLAT (no models/antelopev2/ prefix) so we drop the 5 files into that exact dir.
#    (Official-but-flaky alternative: github.com/deepinsight/insightface releases v0.7 antelopev2.zip)
# =====================================================================================
Section "antelopev2 face pack (HF mirror DIAMONIK7777/antelopev2 -> models/antelopev2/)"
& $Py -c @"
from huggingface_hub import hf_hub_download
DST = r'$ModelDir\insightface\models\antelopev2'   # the path insightface looks in
for fn in ['scrfd_10g_bnkps.onnx','glintr100.onnx','1k3d68.onnx','2d106det.onnx','genderage.onnx']:
    p = hf_hub_download('DIAMONIK7777/antelopev2', fn, local_dir=DST)
    print('have', p)
print('antelopev2 OK')
"@


# =====================================================================================
# 5. VENDOR InstantID pipeline code  (git clone, pinned)  -- brings pipeline + ip_adapter/
#    We import this from disk instead of diffusers custom_pipeline='instantid' (the
#    runtime remote-fetch path, exposed to CVE-2026-44513 on diffusers 0.37.1). See §2.
# =====================================================================================
Section "vendor instantX-research/InstantID -> pipeline/vendor/InstantID (pinned)"
$InstantIDDir = "$VendorDir\InstantID"
if (Test-Path "$InstantIDDir\pipeline_stable_diffusion_xl_instantid.py") {
    Write-Output "InstantID already vendored -- skip clone"
} else {
    git clone --depth 1 https://github.com/instantX-research/InstantID.git $InstantIDDir
    # Pin to the cloned HEAD so re-runs / task #3 import a stable revision.
    Push-Location $InstantIDDir; $sha = (git rev-parse HEAD); Pop-Location
    Write-Output "InstantID vendored at commit $sha (record this in task #3)"
}


# =====================================================================================
# 6. VENDOR + WEIGHTS for LivePortrait   (clone code, then humans-only weights ~0.66 GB)
#    Selective weight pull: humans tree (liveportrait/*) + its bundled buffalo_l insightface
#    pack. We SKIP liveportrait_animals/* and xpose.pth (435 MB, animals mode -- unused).
# =====================================================================================
Section "vendor KwaiVGI/LivePortrait code -> pipeline/vendor/LivePortrait"
$LPDir = "$VendorDir\LivePortrait"
if (Test-Path "$LPDir\inference.py") {
    Write-Output "LivePortrait already vendored -- skip clone"
} else {
    git clone --depth 1 https://github.com/KwaiVGI/LivePortrait.git $LPDir
    Push-Location $LPDir; $lpsha = (git rev-parse HEAD); Pop-Location
    Write-Output "LivePortrait vendored at commit $lpsha (record this in task #3)"
}

Section "LivePortrait humans weights -> pipeline/vendor/LivePortrait/pretrained_weights (~0.66 GB)"
# weights repo is KlingTeam/LivePortrait (KwaiVGI/LivePortrait 404s on HF -- org renamed).
# hf_hub_download per-file = humans-only; skips the 1.48 GB animals tree.
& $Py -c @"
from huggingface_hub import hf_hub_download
DST = r'$LPDir\pretrained_weights'
files = [
    'liveportrait/base_models/spade_generator.pth',
    'liveportrait/base_models/warping_module.pth',
    'liveportrait/base_models/motion_extractor.pth',
    'liveportrait/base_models/appearance_feature_extractor.pth',
    'liveportrait/retargeting_models/stitching_retargeting_module.pth',
    'liveportrait/landmark.onnx',
    'insightface/models/buffalo_l/det_10g.onnx',
    'insightface/models/buffalo_l/2d106det.onnx',
]
for fn in files:
    p = hf_hub_download('KlingTeam/LivePortrait', fn, local_dir=DST)
    print('have', p)
print('LivePortrait humans weights OK')
"@


# =====================================================================================
# 7. SMOKE VERIFICATION  (import + presence ONLY -- no generation; see GENSTACK_INSTALL.md §6)
#    Each check is fault-isolated so a single failure localizes the problem.
# =====================================================================================
Section "SMOKE: env + caps"
& $Py "$Repo\pipeline\_probe_caps.py"

Section "SMOKE: insightface + antelopev2 load (CPU provider -- the safe default)"
& $Py -c @"
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='antelopev2', root=r'$ModelDir\insightface',
                   providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640,640))   # must NOT reach for the network
print('antelopev2 load OK (providers=CPU)')
"@

Section "SMOKE: SDXL fp16 load (no generation)"
& $Py -c @"
import torch
from diffusers import StableDiffusionXLPipeline
StableDiffusionXLPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0',
    torch_dtype=torch.float16, use_safetensors=True, variant='fp16')
print('SDXL fp16 load OK')
"@

Section "SMOKE: InstantID pipeline import + ControlNet load (vendored)"
& $Py -c @"
import sys, torch
sys.path.insert(0, r'$InstantIDDir')
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps
from diffusers.models import ControlNetModel
ControlNetModel.from_pretrained(r'$ModelDir\instantid\ControlNetModel', torch_dtype=torch.float16)
print('InstantID import + ControlNet load OK')
"@

Section "SMOKE: LivePortrait import + checkpoint presence (no generation)"
& $Py -c @"
import os, sys
LP = r'$LPDir'
sys.path.insert(0, LP); os.chdir(LP)
from src.live_portrait_pipeline import LivePortraitPipeline   # pulls tyro/pykalman/scikit-image/etc.
need = [r'pretrained_weights\liveportrait\base_models\spade_generator.pth',
        r'pretrained_weights\liveportrait\base_models\warping_module.pth',
        r'pretrained_weights\liveportrait\base_models\motion_extractor.pth',
        r'pretrained_weights\liveportrait\landmark.onnx']
missing = [p for p in need if not os.path.exists(p)]
assert not missing, ('MISSING: ' + repr(missing))
print('LivePortrait import + checkpoints OK')
"@


# =====================================================================================
Section "DONE"
"DONE $(Get-Date -Format o)" | Out-File -Encoding ascii "$Repo\install_genstack.done"
Write-Output "gen-stack install complete. Review the SMOKE blocks above -- all should print '... OK'."
Write-Output "If any onnxruntime CUDAExecutionProvider line is absent, that is EXPECTED:"
Write-Output "  face analysis runs on CPU by default (once per anchor) -- see GENSTACK_INSTALL.md §3."
