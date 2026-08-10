# sc2_bringup.ps1 -- one-shot FULL-LOCAL bring-up for living-portraits on
# supercommons2 (immer@100.123.185.12). Installs the producer-side deps that are
# NOT part of the runtime show (the .venv-gen heavy stack), seeds the two Piper
# voices, then runs the producer backfill so every rendered portrait lands in the
# clip library.
#
# This is the SECOND install script, run AFTER the portraits exist and AFTER
# install_tasks.ps1 has made the show durable. install_tasks.ps1 wires the runtime
# .venv (player + director + watchdog); THIS script wires the generative .venv-gen
# (piper TTS voices, insightface identity, SAM seg deps) and backfills the graph.
#
# DESIGN: every step is wrapped so ONE failure does not abort the rest. Each step
# prints exactly one of OK / SKIP / FAIL so the conductor reading the transcript
# can see what landed without parsing stack traces. Re-runnable: pip is idempotent,
# the voice download is skipped when the .onnx already exists, and the producer
# backfill registers only NEW clean passes (it never duplicates a clip row).
#
# WHY TWO VENVS: the runtime show (.venv) stays lean -- it only needs pygame/qwen
# client bits. The heavy generative stack (torch+cu121, diffusers, insightface,
# SAM) lives in .venv-gen so a runtime crash-loop never drags the multi-GB gen
# deps into the player's import path. torch 2.5.1+cu121 is ALREADY in .venv-gen;
# this script adds only the lighter producer deps on top of it.
#
# WHAT THIS DOES NOT DO: it does NOT render portraits (no --allow-generate). The
# conductor renders portraits separately (that is the HEAVY SD 1.5 step, guarded
# off here on purpose so a bring-up can never trigger a model download by surprise).
# This script assumes data\gen\<slug>_portrait.png already exists for each slug and
# only runs the segment -> rig -> verify -> clip-row chain over them.
#
# Layout: drop at C:\Users\immer\living-portraits\install\sc2_bringup.ps1 on SC2.
# Run it from anywhere -- it derives the project root as the parent of its folder.

$ErrorActionPreference = 'Continue'   # keep going past a failed step; we wrap each one

# Resolve the project root from this script's location (install\ under project),
# with a hard fallback to the canonical SC2 path the conductor deploys to.
$PROJECT = Split-Path -Parent $PSScriptRoot
if (-not $PROJECT -or -not (Test-Path $PROJECT)) { $PROJECT = 'C:\Users\immer\living-portraits' }

# The GENERATIVE venv (heavy stack: torch+cu121 already installed here). All pip
# installs and the producer backfill run through THIS interpreter, not the runtime
# .venv -- segment/rig/verify/generate all import from the gen stack.
$VENV_GEN   = Join-Path $PROJECT '.venv-gen'
$PY_GEN     = Join-Path $VENV_GEN 'Scripts\python.exe'

# Where Piper drops its .onnx voices. tts.py reads PIPER_VOICE_DIR = data\tts\voices.
$VOICE_DIR  = Join-Path $PROJECT 'data\tts\voices'

# The two voices tts.py's VOICES map names (phineas=alan, seraphina=jenny_dioco).
$VOICE_ALAN  = 'en_GB-alan-medium'
$VOICE_JENNY = 'en_GB-jenny_dioco-medium'

# The characters whose portraits the backfill runs over. Matches
# prompts\characters\<slug>.md and data\gen\<slug>_portrait.png.
$SLUGS = @('phineas', 'seraphina')

Write-Host "=================================================================="
Write-Host " living-portraits SC2 full-local bring-up"
Write-Host "=================================================================="
Write-Host "Project root : $PROJECT"
Write-Host "Gen venv     : $VENV_GEN"
Write-Host "Gen python   : $PY_GEN"
Write-Host "Voice dir    : $VOICE_DIR"
Write-Host "Characters   : $($SLUGS -join ', ')"
Write-Host ""

# Preflight: the gen interpreter MUST exist, or every step below would FAIL the
# same way. This is the one hard stop -- if .venv-gen is not on this host there is
# nothing to bring up. (install_tasks.ps1 guards .venv the same way for the show.)
if (-not (Test-Path $PY_GEN)) {
    Write-Host "FATAL: gen python not found at $PY_GEN"
    Write-Host "       Is .venv-gen created on this host? (expected $VENV_GEN)"
    Write-Host "       Nothing to bring up -- aborting before any step."
    exit 1
}

# Step result tally so the final summary is honest about what landed.
$RESULTS = [ordered]@{}

# Run one named step. The scriptblock should RETURN one of 'OK' / 'SKIP' / 'FAIL'
# (a thrown exception is caught and recorded as FAIL with the message). One step's
# failure never aborts the others -- that is the whole point of this wrapper.
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

# Run the gen-venv python and stream its output into the transcript. Returns the
# process exit code so a step can branch on success/failure. Args are passed as a
# single array so paths with no spaces stay intact (we have none with spaces here).
function Invoke-GenPy([string[]]$pyArgs) {
    & $PY_GEN @pyArgs 2>&1 | ForEach-Object { Write-Host "    $_" }
    return $LASTEXITCODE
}

# pip install into the gen venv. Returns the exit code. --disable-pip-version-check
# keeps the transcript clean; we do NOT pass --quiet so a real failure is visible.
function Invoke-GenPip([string[]]$pkgs) {
    $pipArgs = @('-m', 'pip', 'install', '--disable-pip-version-check') + $pkgs
    return (Invoke-GenPy $pipArgs)
}


# ============================== STEP 1: Piper TTS ==============================
# pip install piper-tts into .venv-gen, then download the two voices into
# data\tts\voices. SKIP the download per-voice when the .onnx is already present
# (idempotent -- a re-run does not re-fetch ~60-120MB voices). The pip install is
# itself idempotent (no-op when already satisfied).
Step 'piper-tts install' {
    $rc = Invoke-GenPip @('piper-tts')
    if ($rc -ne 0) {
        Write-Host "  pip install piper-tts returned exit $rc"
        return 'FAIL'
    }
    return 'OK'
}

Step 'piper voices download' {
    # Ensure the target dir exists (download_voices writes into --data-dir).
    New-Item -ItemType Directory -Force -Path $VOICE_DIR | Out-Null

    $alanOnnx  = Join-Path $VOICE_DIR ($VOICE_ALAN  + '.onnx')
    $jennyOnnx = Join-Path $VOICE_DIR ($VOICE_JENNY + '.onnx')
    if ((Test-Path $alanOnnx) -and (Test-Path $jennyOnnx)) {
        Write-Host "  both voices already present:"
        Write-Host "    $alanOnnx"
        Write-Host "    $jennyOnnx"
        return 'SKIP'
    }

    # piper.download_voices accepts multiple voice ids in one call. Pass the relative
    # data-dir (we set -WorkingDirectory to $PROJECT via the call below) -- actually
    # we pass the ABSOLUTE voice dir so cwd does not matter.
    Write-Host "  downloading: $VOICE_ALAN $VOICE_JENNY -> $VOICE_DIR"
    $rc = Invoke-GenPy @('-m', 'piper.download_voices', $VOICE_ALAN, $VOICE_JENNY, '--data-dir', $VOICE_DIR)
    if ($rc -ne 0) {
        Write-Host "  download_voices returned exit $rc"
        return 'FAIL'
    }
    # Confirm the .onnx files actually landed (download_voices can exit 0 yet a
    # voice id typo would leave nothing).
    if ((Test-Path $alanOnnx) -and (Test-Path $jennyOnnx)) { return 'OK' }
    Write-Host "  WARNING: download exited 0 but expected .onnx files are missing"
    return 'FAIL'
}


# ========================== STEP 2: identity (verify) =========================
# insightface + onnxruntime (CPU build is fine -- identity runs on a single still,
# not in the hot loop). verify.py import-guards insightface; with it present the
# identity check upgrades from the degraded hist+ORB fallback to real ArcFace
# embeddings. onnxruntime is insightface's inference backend.
Step 'insightface + onnxruntime install' {
    $rc = Invoke-GenPip @('insightface', 'onnxruntime')
    if ($rc -ne 0) {
        Write-Host "  pip install insightface onnxruntime returned exit $rc"
        return 'FAIL'
    }
    return 'OK'
}


# ============================ STEP 3: SAM seg deps ============================
# iopath (+ the pip-able SAM-3 deps) so segment.py can attempt the SAM 3.1 backend.
# The SAM WEIGHTS are gated on HuggingFace and are NOT fetched here -- segment.py
# import-guards the whole SAM stack and falls back to GrabCut (dependency-free,
# always available) if the weights/creds are absent. So a FAIL here is NOT fatal to
# the show: the backfill still segments via GrabCut. We therefore treat a pip
# failure as SKIP-with-warning, not a hard FAIL, because the fallback covers it.
Step 'SAM seg deps (iopath + friends)' {
    # iopath is the one reliably pip-able SAM dependency. hydra-core + omegaconf are
    # the config libs SAM-3's builder imports; they are pure-python and safe to add.
    # We do NOT pip the heavy/gated pieces (the model itself, CUDA kernels) -- torch
    # is already in .venv-gen and the weights are a separate gated download.
    $rc = Invoke-GenPip @('iopath', 'hydra-core', 'omegaconf')
    if ($rc -ne 0) {
        Write-Host "  pip install iopath hydra-core omegaconf returned exit $rc"
        Write-Host "  (non-fatal: segment.py falls back to GrabCut, which needs no SAM deps)"
        return 'SKIP'
    }
    return 'OK'
}


# ===================== STEP 4: producer backfill (the point) =====================
# Run the producer over both characters into the canonical clip library. NO
# --allow-generate: portraits must already be rendered (the conductor does that
# separately). This runs segment -> rig -> verify -> clip-row for each slug.
#
# Strict mode (the default -- we pass NO --lenient): a character registers only on
# a CLEAN pass (passed AND not degraded AND not skipped). Whether a given slug
# registers depends on the soft gates (insightface present? Ollama VL model pulled?
# canonical reference for identity?), which is exactly the honest behavior we want.
# A non-registering soft pass is parked, not an error -- the step is OK as long as
# the producer ran the chain without crashing.
Step 'producer backfill (orchestrate --all, strict)' {
    # Verify the portraits the backfill needs are actually on disk first, so the
    # transcript explains a "0 registered" result instead of it looking like a bug.
    $missing = @()
    foreach ($s in $SLUGS) {
        $p = Join-Path $PROJECT ("data\gen\{0}_portrait.png" -f $s)
        if (-not (Test-Path $p)) { $missing += $s }
    }
    if ($missing.Count -gt 0) {
        Write-Host "  portraits NOT found for: $($missing -join ', ')"
        Write-Host "  expected at data\gen\<slug>_portrait.png -- the conductor renders these"
        Write-Host "  separately (the HEAVY SD 1.5 step). Re-run this script after they exist."
        return 'FAIL'
    }

    Write-Host "  running: .venv-gen\Scripts\python.exe pipeline\orchestrate.py --all $($SLUGS -join ' ')"
    # cwd = $PROJECT so orchestrate's ROOT resolution and the sibling sam-3 path
    # work exactly as the module's own self-test assumes. Build the full arg array
    # FIRST -- `Invoke-GenPy @(...) + $SLUGS` would bind as (call) + array, not as a
    # single concatenated argument, so the slugs would be lost.
    $orchArgs = @('pipeline\orchestrate.py', '--all') + $SLUGS
    Push-Location $PROJECT
    try {
        $rc = Invoke-GenPy $orchArgs
    } finally {
        Pop-Location
    }
    if ($rc -ne 0) {
        Write-Host "  orchestrate returned exit $rc"
        return 'FAIL'
    }
    return 'OK'
}


# ====================== STEP 5: capability verify (advisory) ======================
# Print which heavy deps are now importable in .venv-gen and what the producer's
# remaining worklist looks like. Pure read -- never fails the bring-up (advisory).
Step 'capability verify + worklist' {
    Write-Host "  -- import probe in .venv-gen --"
    # One python -c that imports each dep and prints AVAILABLE / MISSING per line.
    # Single-quoted here-string so PowerShell does not touch the python source.
    $probe = @'
import importlib
mods = [
    ("torch",        "gen: SD 1.5 + img2vid"),
    ("cv2",          "segment/rig/verify core"),
    ("numpy",        "array math"),
    ("piper",        "TTS engine (asides)"),
    ("insightface",  "identity (ArcFace) -- else degraded"),
    ("onnxruntime",  "insightface inference backend"),
    ("iopath",       "SAM-3 io (weights still gated)"),
]
for name, why in mods:
    try:
        importlib.import_module(name)
        print(f"    AVAILABLE  {name:13s} -- {why}")
    except Exception as e:
        print(f"    MISSING    {name:13s} -- {why}  [{type(e).__name__}]")
'@
    Push-Location $PROJECT
    try {
        Invoke-GenPy @('-c', $probe) | Out-Null
    } finally {
        Pop-Location
    }

    Write-Host ""
    Write-Host "  -- producer worklist (transitions still owing real footage) --"
    Push-Location $PROJECT
    try {
        Invoke-GenPy @('pipeline\orchestrate.py', '--worklist') | Out-Null
    } finally {
        Pop-Location
    }
    return 'OK'
}


# ================================ FINAL SUMMARY ================================
Write-Host "=================================================================="
Write-Host " bring-up summary"
Write-Host "=================================================================="
$anyFail = $false
foreach ($k in $RESULTS.Keys) {
    $v = $RESULTS[$k]
    if ($v -eq 'FAIL') { $anyFail = $true }
    "{0,-6} {1}" -f $v, $k | Write-Host
}
Write-Host ""
if ($anyFail) {
    Write-Host "One or more steps FAILED -- see the transcript above. The show itself is"
    Write-Host "unaffected (it runs from .venv via install_tasks.ps1); these are producer-"
    Write-Host "side deps. Re-run this script after fixing the failing step -- it is"
    Write-Host "idempotent (pip no-ops, voices skip-if-present, backfill never duplicates)."
} else {
    Write-Host "All steps OK/SKIP. Producer-side bring-up complete. Note: a SKIP on the SAM"
    Write-Host "deps is expected and harmless (GrabCut fallback covers segmentation); a"
    Write-Host "'0 registered' backfill under strict mode is also expected until the Ollama"
    Write-Host "VL model is pulled (see BRINGUP.md)."
}
Write-Host "=================================================================="
