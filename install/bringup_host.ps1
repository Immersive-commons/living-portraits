# bringup_host.ps1 -- GENERALIZED, host-adaptive bring-up for living-portraits.
#
# sc2_bringup.ps1 (the sibling) is SC2-specific: it assumes .venv-gen already
# exists with torch 2.5.1+cu121 baked in, and only adds the lighter producer
# deps on top. THIS script makes no such assumption. It DETECTS the host's GPU
# and picks one of two self-consistent paths, so the same file brings up either
# the generative backend OR a GPU-less display/re-rig host with no edits.
#
# WHY: the migration target hil08rd's GPU is UNKNOWN. A bring-up that hard-codes
# "torch+cu121 is already here" (sc2_bringup) or "this box has an RTX" would
# either crash or silently install a multi-GB CUDA stack onto a box that can't
# use it. This script asks the host what it is, then does the right thing.
#
# ------------------------------------------------------------------------------
# THE TWO PATHS
# ------------------------------------------------------------------------------
#   GPU PRESENT  -- the host is a self-contained PRODUCER (like SC2).
#     Builds .venv-gen from scratch, installs torch (cu121) + diffusers stack +
#     piper TTS + voices + insightface/onnxruntime + SAM-pip deps, checks for
#     ollama and pulls qwen3:8b (the director model) + the VL model (the verify
#     register gate), then runs the producer backfill (orchestrate --all). After
#     this the host can generate portraits, run the verify gate at full strength,
#     and serve the director -- it owes nothing to another machine.
#
#   GPU ABSENT   -- the host is a DISPLAY + RE-RIG node (pairs with a backend).
#     Builds the lean runtime .venv only: pygame + opencv + numpy +
#     imageio-ffmpeg + Pillow + piper-tts + voices. NO torch, NO diffusers, NO
#     SD weights. Generation and the qwen3 director come from the SC2 backend
#     (assets arrive via syncer's sync_assets.ps1; the director is reached by
#     pointing LP_OLLAMA at the backend's Ollama). Segmentation (GrabCut) and the
#     rig math are pure opencv+numpy, so this host CAN still segment + re-rig
#     synced cutouts locally -- it just can't paint new portraits or score the
#     ArcFace/VL gates.
#
# ------------------------------------------------------------------------------
# THE CONDUCTOR'S RUN COMMAND  (same shape as install/BRINGUP.md)
# ------------------------------------------------------------------------------
#   # 0. Dot-source the remote-PS helper (handles BOM + base64 EncodedCommand).
#   . C:\Users\jtole\Documents\2026\life\scripts\invoke-remote-ps.ps1
#
#   # 1. scp this script to the target host (pure ASCII, no BOM -- straight scp OK).
#   scp C:\Users\jtole\Documents\2026\life\projects\living-portraits\install\bringup_host.ps1 `
#       <user>@<host>:C:/Users/<user>/living-portraits/install/bringup_host.ps1
#
#   # 2. Run it on the host. It auto-detects GPU and streams every step's verdict.
#   Invoke-RemotePS -Host_ <user>@<host> -Script @'
#   & powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\Users\<user>\living-portraits\install\bringup_host.ps1
#   '@
#
# GPU-ABSENT hosts pair with two things the conductor wires SEPARATELY:
#   * ASSETS  -- syncer's sync_assets.ps1 mirrors SC2's data\gen + data\clips +
#                the clip manifest down to this host (this script does NOT sync).
#   * DIRECTOR -- set LP_OLLAMA on this host to the backend's Ollama endpoint
#                (e.g. setx LP_OLLAMA http://<producer-host>:11434) so the local
#                director loop talks to SC2's qwen3 instead of a missing local one.
#                (See the GPU-absent tail of this script -- it prints the exact line.)
#
# Override the GPU decision for testing with -ForceCpu / -ForceGpu. Pass
# -SkipBackfill to install deps without running the producer backfill (GPU path).
#
# ------------------------------------------------------------------------------
# DESIGN (same contract as sc2_bringup.ps1)
# ------------------------------------------------------------------------------
# Every step is wrapped so ONE failure does NOT abort the rest. Each step prints
# exactly one of OK / SKIP / FAIL so the conductor reading the transcript sees
# what landed without parsing stack traces. Idempotent: pip is a no-op when
# satisfied, voice download skips when the .onnx already exists, the producer
# backfill registers only NEW clean passes (never duplicates a clip row), and an
# already-pulled Ollama model is detected and skipped.
#
# Layout: drop at <project>\install\bringup_host.ps1. Run from anywhere -- it
# derives the project root as the parent of its folder.

param(
    [switch]$ForceCpu,       # treat the host as GPU-absent regardless of detection
    [switch]$ForceGpu,       # treat the host as GPU-present regardless of detection
    [switch]$SkipBackfill    # GPU path: install deps but skip the producer backfill
)

$ErrorActionPreference = 'Continue'   # keep going past a failed step; we wrap each one

# Resolve the project root from this script's location (install\ under project).
# No SC2 fallback here ON PURPOSE -- this script is host-generic, so a wrong-path
# guess would be misleading. If install\ is not under the project, that is a
# deploy bug the operator should see, not paper over.
$PROJECT = Split-Path -Parent $PSScriptRoot
if (-not $PROJECT -or -not (Test-Path $PROJECT)) {
    Write-Host "FATAL: could not resolve project root from script location ($PSScriptRoot)."
    Write-Host "       Expected this script at <project>\install\bringup_host.ps1."
    exit 1
}

# --- the two venvs (one per path) --------------------------------------------
# GPU path builds the heavy generative venv; GPU-absent builds the lean runtime
# venv. install_tasks.ps1 also targets .venv (runtime) -- we create the SAME
# .venv so the scheduled tasks it registers find their interpreter.
$VENV_GEN     = Join-Path $PROJECT '.venv-gen'
$PY_GEN       = Join-Path $VENV_GEN 'Scripts\python.exe'
$VENV_RUN     = Join-Path $PROJECT '.venv'
$PY_RUN       = Join-Path $VENV_RUN 'Scripts\python.exe'

# Where Piper drops its .onnx voices. tts.py reads PIPER_VOICE_DIR = data\tts\voices.
$VOICE_DIR    = Join-Path $PROJECT 'data\tts\voices'

# The two voices tts.py's VOICES map names (phineas=alan, seraphina=jenny_dioco).
$VOICE_ALAN   = 'en_GB-alan-medium'
$VOICE_JENNY  = 'en_GB-jenny_dioco-medium'

# The characters whose portraits the GPU-path backfill runs over. Matches
# prompts\characters\<slug>.md and data\gen\<slug>_portrait.png.
$SLUGS        = @('phineas', 'seraphina')

# Ollama models the GPU path ensures are pulled:
#   qwen3:8b     -- the director's model (director\stage_manager.py beats).
#   qwen2.5vl:7b -- verify.py's register gate (sends the portrait IMAGE to a VL
#                   model). Without it register SKIPs -> strict backfill registers
#                   nothing. Pulling it lets strict produce clean registrations.
$OLLAMA_TEXT_MODEL = 'qwen3:8b'
$OLLAMA_VL_MODEL   = 'qwen2.5vl:7b'

# --- generative pip stack (GPU path) -----------------------------------------
# torch is installed SEPARATELY from the cu121 index (see Step G1) -- it must NOT
# be in this list or pip would pull the default (often CPU) wheel. These are the
# deps that ride on top of torch.
$GEN_DIFFUSERS = @('diffusers', 'transformers', 'accelerate', 'safetensors', 'Pillow')
$GEN_TTS       = @('piper-tts')
$GEN_IDENTITY  = @('insightface', 'onnxruntime')
$GEN_SAM       = @('iopath', 'hydra-core', 'omegaconf')   # pip-able SAM deps; weights gated, fetched out-of-band

# --- runtime pip stack (GPU-absent path) -------------------------------------
# Mirrors what the runtime show actually imports: player.py -> pygame;
# runtime/* -> numpy, cv2 (opencv-python), imageio + imageio-ffmpeg, PIL
# (Pillow). Plus piper-tts + voices so spoken asides are audible on a display
# host. NO torch / diffusers / SD here -- those are the generative backend's job.
$RUN_CORE      = @('pygame', 'opencv-python', 'numpy', 'imageio', 'imageio-ffmpeg', 'Pillow')
$RUN_TTS       = @('piper-tts')

Write-Host "=================================================================="
Write-Host " living-portraits GENERALIZED host bring-up (GPU-adaptive)"
Write-Host "=================================================================="
Write-Host "Project root : $PROJECT"
Write-Host "Gen venv     : $VENV_GEN  (GPU path)"
Write-Host "Runtime venv : $VENV_RUN  (GPU-absent path)"
Write-Host "Voice dir    : $VOICE_DIR"
Write-Host "Characters   : $($SLUGS -join ', ')"
Write-Host ""


# ==============================================================================
# GPU DETECTION
# ==============================================================================
# Two independent signals, OR'd:
#   1. nvidia-smi on PATH and returning exit 0 -- the strongest signal (a working
#      NVIDIA driver stack, which is what torch+cu121 actually needs).
#   2. Win32_VideoController name matching NVIDIA -- catches a card present even
#      if nvidia-smi is not on PATH (driver installed, smi elsewhere). Weaker, but
#      a useful backstop.
# -ForceCpu / -ForceGpu short-circuit detection for testing or a known host.
function Test-Gpu {
    if ($ForceCpu) { Write-Host "GPU detect : FORCED CPU (-ForceCpu)"; return $false }
    if ($ForceGpu) { Write-Host "GPU detect : FORCED GPU (-ForceGpu)"; return $true }

    # Signal 1: nvidia-smi. Wrap in try/catch -- a missing exe throws under
    # ErrorActionPreference handling; a present-but-erroring smi just sets a
    # nonzero exit code we read from $LASTEXITCODE.
    $smiFound = $false
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi) {
        try {
            $null = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
            if ($LASTEXITCODE -eq 0) {
                $smiFound = $true
                Write-Host "GPU detect : nvidia-smi present and OK (exit 0)"
            } else {
                Write-Host "GPU detect : nvidia-smi present but returned exit $LASTEXITCODE"
            }
        } catch {
            Write-Host "GPU detect : nvidia-smi present but threw ($($_.Exception.Message))"
        }
    } else {
        Write-Host "GPU detect : nvidia-smi NOT on PATH"
    }

    # Signal 2: a video controller whose name says NVIDIA. CimInstance is the
    # modern replacement for the deprecated Get-WmiObject. Guarded -- on a host
    # where CIM is unavailable this just yields no match, not a crash.
    $nvController = $false
    try {
        $vc = Get-CimInstance Win32_VideoController -ErrorAction Stop |
              Where-Object { $_.Name -match 'NVIDIA' }
        if ($vc) {
            $nvController = $true
            Write-Host "GPU detect : Win32_VideoController matches NVIDIA -- $(@($vc.Name) -join ', ')"
        } else {
            Write-Host "GPU detect : no NVIDIA in Win32_VideoController names"
        }
    } catch {
        Write-Host "GPU detect : Win32_VideoController query failed ($($_.Exception.Message))"
    }

    return ($smiFound -or $nvController)
}

$HAS_GPU = Test-Gpu
Write-Host ""
Write-Host ("PATH SELECTED : {0}" -f $(if ($HAS_GPU) { 'GPU PRESENT  -> full generative producer (.venv-gen)' } else { 'GPU ABSENT   -> lean display/re-rig host (.venv)' }))
Write-Host ""


# ==============================================================================
# STEP PLUMBING  (same shape as sc2_bringup.ps1 so transcripts read alike)
# ==============================================================================
# Step result tally so the final summary is honest about what landed.
$RESULTS = [ordered]@{}

# The interpreter the steps below run through -- set per path once the venv is
# created. Steps use $script:PY so a venv re-creation mid-run is picked up.
$script:PY = $null

# Run one named step. The scriptblock RETURNs one of 'OK' / 'SKIP' / 'FAIL' (a
# thrown exception is caught and recorded as FAIL). One step's failure never
# aborts the others -- that is the whole point of this wrapper.
function Step($name, $block) {
    Write-Host "------------------------------------------------------------------"
    Write-Host "STEP: $name"
    Write-Host "------------------------------------------------------------------"
    $verdict = 'FAIL'
    try {
        $verdict = & $block
        if ($verdict -notin @('OK', 'SKIP', 'FAIL')) { $verdict = 'OK' }
    } catch {
        Write-Host "  exception: $($_.Exception.Message)"
        $verdict = 'FAIL'
    }
    $RESULTS[$name] = $verdict
    Write-Host "  => $verdict"
    Write-Host ""
}

# Run the active venv python and stream its output into the transcript. Returns
# the process exit code so a step can branch on success/failure.
function Invoke-Py([string[]]$pyArgs) {
    & $script:PY @pyArgs 2>&1 | ForEach-Object { Write-Host "    $_" }
    return $LASTEXITCODE
}

# pip install into the active venv. Returns the exit code. --disable-pip-version-
# check keeps the transcript clean; we do NOT pass --quiet so a real failure is visible.
function Invoke-Pip([string[]]$pkgs) {
    $pipArgs = @('-m', 'pip', 'install', '--disable-pip-version-check') + $pkgs
    return (Invoke-Py $pipArgs)
}

# Create a venv at $venvPath using the system python if its interpreter is not
# already present. Returns $true if a usable interpreter exists afterward. We try
# `py -3` first (the Windows launcher, most reliable on a fresh box), then a bare
# `python` on PATH. Idempotent -- an existing venv is reused, never rebuilt.
function Ensure-Venv($venvPath, $pyExe) {
    if (Test-Path $pyExe) {
        Write-Host "  venv already present: $venvPath"
        return $true
    }
    Write-Host "  creating venv: $venvPath"
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        & py -3 -m venv $venvPath 2>&1 | ForEach-Object { Write-Host "    $_" }
    } else {
        $sys = Get-Command python -ErrorAction SilentlyContinue
        if (-not $sys) {
            Write-Host "  no system python found (tried 'py -3' and 'python') -- cannot create venv"
            return $false
        }
        & python -m venv $venvPath 2>&1 | ForEach-Object { Write-Host "    $_" }
    }
    if (-not (Test-Path $pyExe)) {
        Write-Host "  venv creation did not produce $pyExe"
        return $false
    }
    # Upgrade pip in the new venv so wheels (esp. torch) resolve cleanly. Best-
    # effort -- a failed upgrade is non-fatal (old pip usually still installs).
    & $pyExe -m pip install --upgrade pip --disable-pip-version-check 2>&1 |
        ForEach-Object { Write-Host "    $_" }
    return $true
}

# Pull an Ollama model if it is not already present. Returns 'OK' / 'SKIP' /
# 'FAIL' so it can be a Step body directly. SKIP when the model already shows in
# `ollama list`; FAIL only when ollama is reachable but the pull errors. If
# ollama itself is missing, the caller decides (we return 'SKIP' with a note --
# the verify/director paths degrade open without it).
function Ensure-OllamaModel($model) {
    $ollama = Get-Command ollama -ErrorAction SilentlyContinue
    if (-not $ollama) {
        Write-Host "  ollama not on PATH -- skipping pull of $model"
        Write-Host "  (verify register + the director degrade open without it; install Ollama to enable)"
        return 'SKIP'
    }
    # `ollama list` prints a table; a present model's name (with or without the
    # :tag) appears as a row. Match the exact model string we asked for.
    $listing = ''
    try { $listing = (& ollama list 2>&1 | Out-String) } catch { $listing = '' }
    if ($listing -match [regex]::Escape($model)) {
        Write-Host "  $model already pulled (found in 'ollama list')"
        return 'SKIP'
    }
    Write-Host "  pulling $model (this can take a while on first fetch)..."
    & ollama pull $model 2>&1 | ForEach-Object { Write-Host "    $_" }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ollama pull $model returned exit $LASTEXITCODE"
        return 'FAIL'
    }
    return 'OK'
}

# Download the two Piper voices into $VOICE_DIR unless both .onnx already exist.
# Returns 'OK' / 'SKIP' / 'FAIL'. Shared by both paths (both want audible asides).
function Step-PiperVoices {
    New-Item -ItemType Directory -Force -Path $VOICE_DIR | Out-Null
    $alanOnnx  = Join-Path $VOICE_DIR ($VOICE_ALAN  + '.onnx')
    $jennyOnnx = Join-Path $VOICE_DIR ($VOICE_JENNY + '.onnx')
    if ((Test-Path $alanOnnx) -and (Test-Path $jennyOnnx)) {
        Write-Host "  both voices already present:"
        Write-Host "    $alanOnnx"
        Write-Host "    $jennyOnnx"
        return 'SKIP'
    }
    Write-Host "  downloading: $VOICE_ALAN $VOICE_JENNY -> $VOICE_DIR"
    $rc = Invoke-Py @('-m', 'piper.download_voices', $VOICE_ALAN, $VOICE_JENNY, '--data-dir', $VOICE_DIR)
    if ($rc -ne 0) {
        Write-Host "  download_voices returned exit $rc"
        return 'FAIL'
    }
    if ((Test-Path $alanOnnx) -and (Test-Path $jennyOnnx)) { return 'OK' }
    Write-Host "  WARNING: download exited 0 but expected .onnx files are missing"
    return 'FAIL'
}


# ==============================================================================
# ==========================  GPU-PRESENT PATH  ================================
# ==============================================================================
function Invoke-GpuPath {
    # ---- G0: create / reuse .venv-gen --------------------------------------
    Step 'gen venv (.venv-gen) create-or-reuse' {
        if (Ensure-Venv $VENV_GEN $PY_GEN) {
            $script:PY = $PY_GEN
            return 'OK'
        }
        return 'FAIL'
    }
    # If the venv could not be created there is no interpreter for any step below;
    # bail to the summary rather than FAIL every step identically.
    if (-not (Test-Path $PY_GEN)) {
        Write-Host "Gen venv interpreter missing after create step -- skipping remaining GPU steps."
        return
    }
    $script:PY = $PY_GEN

    # ---- G1: torch (cu121) -- the ONE special-index install -----------------
    # Installed from the cu121 wheel index, NOT plain PyPI, so the CUDA build
    # lands (matches SC2's torch 2.5.1+cu121). SKIP if torch already imports with
    # CUDA so a re-run does not re-pull multi-GB wheels.
    Step 'torch cu121 install' {
        $probe = 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 3)'
        $null = Invoke-Py @('-c', $probe)
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  torch already present with CUDA available -- skipping install"
            return 'SKIP'
        }
        Write-Host "  installing torch from the cu121 wheel index..."
        $rc = Invoke-Py @('-m', 'pip', 'install', '--disable-pip-version-check',
                          'torch', '--index-url', 'https://download.pytorch.org/whl/cu121')
        if ($rc -ne 0) {
            Write-Host "  torch cu121 install returned exit $rc"
            return 'FAIL'
        }
        # Confirm CUDA actually came up -- a CPU wheel sneaking in would defeat the
        # whole GPU path. A successful install with torch.cuda.is_available()==False
        # is reported as FAIL so the operator sees the driver/index mismatch.
        $null = Invoke-Py @('-c', $probe)
        if ($LASTEXITCODE -eq 0) { return 'OK' }
        Write-Host "  torch installed but torch.cuda.is_available() is False -- check the NVIDIA driver"
        return 'FAIL'
    }

    # ---- G2: diffusers stack ------------------------------------------------
    Step 'diffusers stack install' {
        $rc = Invoke-Pip $GEN_DIFFUSERS
        if ($rc -ne 0) { Write-Host "  pip install $($GEN_DIFFUSERS -join ' ') returned exit $rc"; return 'FAIL' }
        return 'OK'
    }

    # ---- G3: piper-tts + voices --------------------------------------------
    Step 'piper-tts install' {
        $rc = Invoke-Pip $GEN_TTS
        if ($rc -ne 0) { Write-Host "  pip install piper-tts returned exit $rc"; return 'FAIL' }
        return 'OK'
    }
    Step 'piper voices download' { Step-PiperVoices }

    # ---- G4: insightface + onnxruntime -------------------------------------
    Step 'insightface + onnxruntime install' {
        $rc = Invoke-Pip $GEN_IDENTITY
        if ($rc -ne 0) { Write-Host "  pip install $($GEN_IDENTITY -join ' ') returned exit $rc"; return 'FAIL' }
        return 'OK'
    }

    # ---- G5: SAM seg deps (non-fatal -- GrabCut fallback covers it) ---------
    Step 'SAM seg deps (iopath + friends)' {
        $rc = Invoke-Pip $GEN_SAM
        if ($rc -ne 0) {
            Write-Host "  pip install $($GEN_SAM -join ' ') returned exit $rc"
            Write-Host "  (non-fatal: segment.py falls back to GrabCut, which needs no SAM deps)"
            return 'SKIP'
        }
        return 'OK'
    }

    # ---- G6: Ollama models (director + verify VL) ---------------------------
    Step "ollama pull $OLLAMA_TEXT_MODEL (director model)" { Ensure-OllamaModel $OLLAMA_TEXT_MODEL }
    Step "ollama pull $OLLAMA_VL_MODEL (verify register gate)" { Ensure-OllamaModel $OLLAMA_VL_MODEL }

    # ---- G7: producer backfill (the point) ----------------------------------
    # Runs segment -> rig -> verify -> clip-row over each character's EXISTING
    # portrait. NO --allow-generate (portraits must already be on disk; the
    # conductor renders them separately -- the HEAVY SD 1.5 step). Strict mode.
    Step 'producer backfill (orchestrate --all, strict)' {
        if ($SkipBackfill) {
            Write-Host "  -SkipBackfill set -- deps installed, backfill not run"
            return 'SKIP'
        }
        $missing = @()
        foreach ($s in $SLUGS) {
            $p = Join-Path $PROJECT ("data\gen\{0}_portrait.png" -f $s)
            if (-not (Test-Path $p)) { $missing += $s }
        }
        if ($missing.Count -gt 0) {
            Write-Host "  portraits NOT found for: $($missing -join ', ')"
            Write-Host "  expected at data\gen\<slug>_portrait.png -- the conductor renders these"
            Write-Host "  separately (the HEAVY SD 1.5 step). Re-run this script after they exist,"
            Write-Host "  or render here now with: .venv-gen\Scripts\python.exe pipeline\orchestrate.py --all $($SLUGS -join ' ') --allow-generate"
            return 'FAIL'
        }
        Write-Host "  running: .venv-gen\Scripts\python.exe pipeline\orchestrate.py --all $($SLUGS -join ' ')"
        # Build the full arg array FIRST -- `Invoke-Py @(...) + $SLUGS` would bind
        # as (call) + array, not a single concatenated argument, so the slugs
        # would be lost. cwd = $PROJECT so orchestrate's ROOT + sibling sam-3
        # path resolution work as the module's self-test assumes.
        $orchArgs = @('pipeline\orchestrate.py', '--all') + $SLUGS
        Push-Location $PROJECT
        try { $rc = Invoke-Py $orchArgs } finally { Pop-Location }
        if ($rc -ne 0) { Write-Host "  orchestrate returned exit $rc"; return 'FAIL' }
        return 'OK'
    }
}


# ==============================================================================
# ==========================  GPU-ABSENT PATH  ================================
# ==============================================================================
function Invoke-CpuPath {
    # ---- C0: create / reuse .venv (runtime) ---------------------------------
    # Same .venv install_tasks.ps1 wires the player/director/watchdog tasks
    # against, so creating it here means those tasks find their interpreter.
    Step 'runtime venv (.venv) create-or-reuse' {
        if (Ensure-Venv $VENV_RUN $PY_RUN) {
            $script:PY = $PY_RUN
            return 'OK'
        }
        return 'FAIL'
    }
    if (-not (Test-Path $PY_RUN)) {
        Write-Host "Runtime venv interpreter missing after create step -- skipping remaining steps."
        return
    }
    $script:PY = $PY_RUN

    # ---- C1: runtime core (pygame/opencv/numpy/imageio-ffmpeg/Pillow) -------
    Step 'runtime core install (pygame/opencv/numpy/imageio/Pillow)' {
        $rc = Invoke-Pip $RUN_CORE
        if ($rc -ne 0) { Write-Host "  pip install $($RUN_CORE -join ' ') returned exit $rc"; return 'FAIL' }
        return 'OK'
    }

    # ---- C2: piper-tts + voices (audible asides on a display host) ----------
    Step 'piper-tts install' {
        $rc = Invoke-Pip $RUN_TTS
        if ($rc -ne 0) { Write-Host "  pip install piper-tts returned exit $rc"; return 'FAIL' }
        return 'OK'
    }
    Step 'piper voices download' { Step-PiperVoices }

    # ---- C3: explicitly NOT installing torch/SD -- print why ----------------
    # This is a step (not a comment) so it shows in the summary table: a reader
    # of the transcript sees the deliberate ABSENCE, not a silent gap.
    Step 'torch / SD deliberately SKIPPED (no GPU)' {
        Write-Host "  This host has no NVIDIA GPU -- torch + diffusers + SD 1.5 are NOT installed."
        Write-Host "  Portrait GENERATION and the qwen3 DIRECTOR come from the SC2 backend:"
        Write-Host "    * assets (data\gen, data\clips, manifest) arrive via syncer's sync_assets.ps1"
        Write-Host "    * point the local director at the backend's Ollama:"
        Write-Host "        setx LP_OLLAMA http://<producer-host>:11434"
        Write-Host "  Segmentation (GrabCut) + rig math are pure opencv+numpy, so THIS host can"
        Write-Host "  still segment + re-rig synced cutouts locally -- it just cannot paint new"
        Write-Host "  portraits or run the ArcFace/VL verify gates at full strength."
        return 'SKIP'
    }
}


# ==============================================================================
# DISPATCH
# ==============================================================================
if ($HAS_GPU) { Invoke-GpuPath } else { Invoke-CpuPath }


# ==============================================================================
# CAPABILITY SUMMARY  (advisory probe -- runs on whichever interpreter we built)
# ==============================================================================
# Prints AVAILABLE / MISSING per heavy dep so the conductor sees the host's real
# capability surface without trusting which path was claimed. Never fails the
# bring-up. Skipped only if no interpreter was created at all.
Step 'capability summary (import probe)' {
    if (-not $script:PY -or -not (Test-Path $script:PY)) {
        Write-Host "  no venv interpreter was created -- nothing to probe"
        return 'SKIP'
    }
    $probe = @'
import importlib
mods = [
    ("torch",        "gen: SD 1.5 + img2vid (GPU path only)"),
    ("diffusers",    "gen: SD pipeline (GPU path only)"),
    ("cv2",          "segment/rig/verify core (both paths)"),
    ("numpy",        "array math (both paths)"),
    ("pygame",       "player display surface (runtime)"),
    ("imageio",      "clip read/write (runtime)"),
    ("PIL",          "image io (runtime)"),
    ("piper",        "TTS engine for asides (both paths)"),
    ("insightface",  "identity ArcFace (GPU path) -- else verify degraded"),
    ("onnxruntime",  "insightface inference backend (GPU path)"),
    ("iopath",       "SAM-3 io (GPU path; weights still gated)"),
]
ok = []
for name, why in mods:
    try:
        importlib.import_module(name)
        print(f"    AVAILABLE  {name:13s} -- {why}")
        ok.append(name)
    except Exception as e:
        print(f"    MISSING    {name:13s} -- {why}  [{type(e).__name__}]")
# torch CUDA line (only meaningful if torch imported)
if "torch" in ok:
    try:
        import torch
        print(f"    torch.cuda.is_available() = {torch.cuda.is_available()}")
    except Exception as e:
        print(f"    torch present but cuda probe failed [{type(e).__name__}]")
'@
    Push-Location $PROJECT
    try { Invoke-Py @('-c', $probe) | Out-Null } finally { Pop-Location }
    return 'OK'
}


# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
Write-Host "=================================================================="
Write-Host " bring-up summary"
Write-Host "=================================================================="
Write-Host ("Path taken : {0}" -f $(if ($HAS_GPU) { 'GPU PRESENT (.venv-gen, full producer)' } else { 'GPU ABSENT (.venv, display/re-rig host)' }))
Write-Host ""
$anyFail = $false
foreach ($k in $RESULTS.Keys) {
    $v = $RESULTS[$k]
    if ($v -eq 'FAIL') { $anyFail = $true }
    "{0,-6} {1}" -f $v, $k | Write-Host
}
Write-Host ""
if ($anyFail) {
    Write-Host "One or more steps FAILED -- see the transcript above. The wrapper is idempotent:"
    Write-Host "fix the failing step's cause (e.g. install Python/Ollama, render a missing"
    Write-Host "portrait, check the NVIDIA driver) and re-run -- pip no-ops, voices skip-if-present,"
    Write-Host "the backfill never duplicates a clip row, and an already-pulled model is skipped."
} elseif ($HAS_GPU) {
    Write-Host "GPU path complete. This host is a self-contained producer: it can generate"
    Write-Host "portraits, run the verify gate at full strength (insightface + VL register), and"
    Write-Host "serve the director. A SKIP on SAM deps is harmless (GrabCut covers segmentation);"
    Write-Host "a '0 registered' backfill is expected only if the VL model is not yet pulled."
} else {
    Write-Host "GPU-absent path complete. This host is a display + re-rig node. Pair it with the"
    Write-Host "SC2 backend: run syncer's sync_assets.ps1 to mirror the rendered assets here, and"
    Write-Host "set LP_OLLAMA to the backend's Ollama (setx LP_OLLAMA http://<producer-host>:11434)"
    Write-Host "so the local director loop reaches SC2's qwen3. Then install_tasks.ps1 wires the"
    Write-Host "player/director/watchdog scheduled tasks against the .venv this script just built."
}
Write-Host "=================================================================="
