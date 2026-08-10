# deploy_host.ps1 -- one command to stand up the living-portraits RUNTIME show
# on ANY Windows tailnet host. Host-agnostic: the migration target is hil08rd,
# but nothing here is wired to a specific machine -- pass -Target and go.
#
# WHAT THIS DEPLOYS: the lean RUNTIME only (player + director + watchdog). It
# does NOT deploy the heavy generative stack (.venv-gen, SD 1.5, piper, SAM) --
# that is the producer side, handled separately by sc2_bringup.ps1 on the box
# that actually renders portraits. A pure runtime host just needs the code, a
# small runtime venv, the three scheduled tasks, and the rendered clips synced
# in out-of-band (the syncer handles assets; see MIGRATION.md).
#
# WHAT IT DOES (each step prints OK / FAIL; one failure does not abort the rest,
# except the two hard preflights that must pass before anything is sent):
#   1. PREFLIGHT  -- local tree + scp/ssh reachable + install_tasks.ps1 present.
#   2. STAGE      -- robocopy the runtime tree to a clean local temp dir,
#                    EXCLUDING .venv* / data / *.png / *.gif / *.log / *.mp4 /
#                    __pycache__, so the scp payload is exactly the runtime code.
#   3. COPY       -- scp -r the staged tree to <RemoteRoot> on the host, then
#                    regenerate run_player.bat as ASCII on the host with the
#                    correct RemoteRoot baked in (the committed .bat hardcodes
#                    the SC2 immer path, so it is rewritten per-host here).
#   4. VENV       -- create the runtime venv (py -3.12 -m venv .venv) on the host
#                    and pip install the runtime deps:
#                    pygame opencv-python-headless numpy imageio-ffmpeg Pillow.
#   5. TASKS      -- run install_tasks.ps1 on the host (via Invoke-RemotePS) to
#                    register lp-player / lp-director / lp-watchdog at-logon.
#                    install_tasks.ps1 self-resolves the project root from its
#                    own location, so it works on any host with no edits.
#   6. VERIFY     -- confirm pythonw.exe exists in the venv and the three tasks
#                    are registered; print a summary the conductor can read.
#
# IDEMPOTENT: re-runnable end to end. robocopy /MIR mirrors the staged tree
# (deletes stale files in the temp dir); the per-host RemoteRoot copy overwrites
# in place; venv creation SKIPs when .venv already exists; pip install is a
# no-op when satisfied; install_tasks.ps1 registers every task with -Force, so a
# second run updates in place and never duplicates a trigger.
#
# ============================== USAGE (conductor) ==============================
# Run from the life repo root AFTER dot-sourcing the remote-PS helper (this
# script CALLS Invoke-RemotePS, so that function must be in the session first):
#
#   . C:\Users\jtole\Documents\2026\life\scripts\invoke-remote-ps.ps1
#   .\projects\living-portraits\install\deploy_host.ps1 -Target immer@hil08rd
#
# Or from inside the project's install\ folder:
#
#   . C:\Users\jtole\Documents\2026\life\scripts\invoke-remote-ps.ps1
#   .\deploy_host.ps1 -Target immer@hil08rd
#
# Override the remote path (default is C:\Users\<user-from-Target>\living-portraits):
#
#   .\deploy_host.ps1 -Target immer@hil08rd -RemoteRoot D:\shows\living-portraits
#
# HARD RULES (match the sibling install scripts): ASCII only, straight quotes,
# '--' not em-dashes -- these files scp to a Windows-codepage host and smart
# punctuation breaks the PS parser. Do NOT render portraits here; do NOT touch
# .venv-gen. Runtime only.
# ==============================================================================

[CmdletBinding()]
param(
    # user@host of the deploy target, e.g. immer@hil08rd or immer@100.123.185.12.
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$Target,

    # Where the runtime lands on the host. Default is derived from the user in
    # -Target: C:\Users\<user>\living-portraits. Override for non-default layouts.
    [string]$RemoteRoot,

    # The local repo dir to deploy FROM. Default is the parent of this script's
    # folder (install\ lives under the project), matching install_tasks.ps1.
    [string]$LocalRoot
)

$ErrorActionPreference = 'Continue'   # keep going past a failed step; we wrap each

# ----------------------------------------------------------------------------
# Resolve LocalRoot (parent of install\, where this script lives) unless given.
# ----------------------------------------------------------------------------
if (-not $LocalRoot) { $LocalRoot = Split-Path -Parent $PSScriptRoot }

# ----------------------------------------------------------------------------
# Parse -Target into user + host. Accept "user@host" (preferred) or bare "host"
# (then RemoteRoot must be supplied, since we cannot derive C:\Users\<user>\).
# ----------------------------------------------------------------------------
if ($Target -match '^([^@]+)@(.+)$') {
    $RemoteUser = $matches[1]
    $RemoteHost = $matches[2]
} else {
    $RemoteUser = ''
    $RemoteHost = $Target
}

# ----------------------------------------------------------------------------
# Derive RemoteRoot from the user when not explicitly provided.
# ----------------------------------------------------------------------------
if (-not $RemoteRoot) {
    if (-not $RemoteUser) {
        Write-Host "FATAL: -Target had no 'user@' part and -RemoteRoot was not given."
        Write-Host "       Cannot derive C:\Users\<user>\living-portraits. Pass one of:"
        Write-Host "         -Target user@host        (derives the default root), or"
        Write-Host "         -RemoteRoot <path>       (explicit remote project root)."
        exit 1
    }
    $RemoteRoot = "C:\Users\$RemoteUser\living-portraits"
}

# Forward-slash form of RemoteRoot for scp destinations. scp on Windows is happy
# with C:/Users/... and it sidesteps backslash-escaping through the ssh layer.
$RemoteRootFwd = $RemoteRoot -replace '\\', '/'

# Where install_tasks.ps1 will land on the host (parent-of-folder = RemoteRoot).
$RemoteInstallTasks = (Join-Path $RemoteRoot 'install\install_tasks.ps1')
$RemoteVenvPythonw  = (Join-Path $RemoteRoot '.venv\Scripts\pythonw.exe')

# ----------------------------------------------------------------------------
# THE AUDITABLE PAYLOAD. These are the runtime items copied to the host. Kept as
# a flat, reviewable array on purpose: a reader can see exactly what ships and
# what does not. Dirs are copied recursively (minus the excludes below); files
# are copied as-is. EXCLUDES (applied during staging): .venv* / data / *.png /
# *.gif / *.log / *.mp4 / __pycache__.
# ----------------------------------------------------------------------------
$RUNTIME_DIRS = @(
    'runtime',      # clip graph / clip player / rig / stage_render -- player's deps
    'director',     # stage_manager.py (the qwen3 beat loop) + signals/feeds
    'prompts',      # _stage-directives.md + characters\*  (read by the director)
    'pipeline',     # segment/rig/verify/tts/generate/orchestrate (producer-side too)
    'install'       # install_tasks.ps1 + watchdog.ps1 + this script + the docs
)
$RUNTIME_FILES = @(
    'player.py'     # the frameless dual-panel player (lp-player runs this)
)
# run_player.bat is NOT scp'd -- it is regenerated ASCII on the host (step 3)
# with the correct RemoteRoot baked in, because the committed copy hardcodes the
# SC2 immer path. EXCLUDED from the show entirely: data\ (runtime state + clips),
# *.png / *.gif (portraits + demo media), *.log, .venv* (rebuilt per host),
# __pycache__, gallery.py + tests\ (dev-only; not part of the runtime show).

# robocopy exclusion lists (dir names + file globs).
$EXCLUDE_DIRS  = @('.venv', '.venv-gen', 'data', '__pycache__', '.git')
$EXCLUDE_FILES = @('*.png', '*.gif', '*.log', '*.mp4', '*.pyc')

# Staging dir: a clean temp tree we mirror the payload into, then scp -r. Named
# per-PID so concurrent deploys to different hosts never collide.
$StageRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("lp-deploy-{0}" -f $PID)

Write-Host "=================================================================="
Write-Host " living-portraits RUNTIME deploy-to-host"
Write-Host "=================================================================="
Write-Host "Target host  : $RemoteHost"
Write-Host "Remote user  : $(if ($RemoteUser) { $RemoteUser } else { '(none -- bare host)' })"
Write-Host "Remote root  : $RemoteRoot"
Write-Host "Local root   : $LocalRoot"
Write-Host "Stage dir    : $StageRoot"
Write-Host "Payload dirs : $($RUNTIME_DIRS -join ', ')"
Write-Host "Payload files: $($RUNTIME_FILES -join ', ')"
Write-Host "Excl. dirs   : $($EXCLUDE_DIRS -join ', ')"
Write-Host "Excl. files  : $($EXCLUDE_FILES -join ', ')"
Write-Host ""

# ----------------------------------------------------------------------------
# Pre-req: Invoke-RemotePS must already be dot-sourced into this session. We
# CALL it (steps 4-6), we do not re-implement it. Fail fast with the exact
# remedy if the conductor forgot to dot-source the helper.
# ----------------------------------------------------------------------------
if (-not (Get-Command Invoke-RemotePS -ErrorAction SilentlyContinue)) {
    Write-Host "FATAL: Invoke-RemotePS is not in this session."
    Write-Host "       Dot-source the remote-PS helper first, then re-run:"
    Write-Host "         . C:\Users\jtole\Documents\2026\life\scripts\invoke-remote-ps.ps1"
    exit 1
}

# Step-result tally so the final summary is honest about what landed.
$RESULTS = [ordered]@{}

# Run one named step. The scriptblock RETURNs 'OK' or 'FAIL' (a thrown exception
# is caught and recorded as FAIL with the message). One step's failure never
# aborts the others -- same wrapper shape as sc2_bringup.ps1's Step().
function Step($name, $block) {
    Write-Host "------------------------------------------------------------------"
    Write-Host "STEP: $name"
    Write-Host "------------------------------------------------------------------"
    $verdict = 'FAIL'
    try {
        $verdict = & $block
        if ($verdict -notin @('OK', 'FAIL')) { $verdict = 'OK' }
    } catch {
        Write-Host "  exception: $($_.Exception.Message)"
        $verdict = 'FAIL'
    }
    $RESULTS[$name] = $verdict
    Write-Host "  => $verdict"
    Write-Host ""
    return $verdict
}

# ============================ STEP 1: PREFLIGHT (local) =======================
# Hard stops: nothing is sent to the host until these pass. Verifies the local
# payload exists and ssh can reach the target. A FAIL here aborts the deploy --
# there is no point staging/copying if the source or the link is broken.
$pf = Step 'preflight (local tree + ssh reachability)' {
    $ok = $true

    # 1a. Every payload dir + file must exist locally.
    foreach ($d in $RUNTIME_DIRS) {
        $p = Join-Path $LocalRoot $d
        if (Test-Path -LiteralPath $p -PathType Container) {
            Write-Host "  OK   dir   $d"
        } else {
            Write-Host "  MISS dir   $d  (expected $p)"
            $ok = $false
        }
    }
    foreach ($f in $RUNTIME_FILES) {
        $p = Join-Path $LocalRoot $f
        if (Test-Path -LiteralPath $p -PathType Leaf) {
            Write-Host "  OK   file  $f"
        } else {
            Write-Host "  MISS file  $f  (expected $p)"
            $ok = $false
        }
    }

    # 1b. install_tasks.ps1 must be in the payload (step 5 runs it on the host).
    $itLocal = Join-Path $LocalRoot 'install\install_tasks.ps1'
    if (Test-Path -LiteralPath $itLocal -PathType Leaf) {
        Write-Host "  OK   install_tasks.ps1 present (will run on host post-copy)"
    } else {
        Write-Host "  MISS install_tasks.ps1 (expected $itLocal) -- cannot register tasks"
        $ok = $false
    }

    # 1c. ssh can reach the host (BatchMode so it never hangs on a prompt). A
    # cheap remote echo proves auth + reachability before we move bytes.
    Write-Host "  probing ssh to $RemoteHost ..."
    try {
        $probe = Invoke-RemotePS -Host_ $Target -Script 'Write-Output "reachable"' -ConnectTimeoutSec 10
        if ($probe -match 'reachable') {
            Write-Host "  OK   ssh reachable: $RemoteHost"
        } else {
            Write-Host "  WARN ssh returned unexpected output: $probe"
            $ok = $false
        }
    } catch {
        Write-Host "  FAIL ssh unreachable: $($_.Exception.Message)"
        $ok = $false
    }

    if ($ok) { return 'OK' } else { return 'FAIL' }
}
if ($pf -ne 'OK') {
    Write-Host "Preflight FAILED -- nothing was sent to $RemoteHost. Fix the items"
    Write-Host "marked MISS/FAIL above and re-run. (Local payload + ssh reachability"
    Write-Host "must both be clean before any copy.)"
    exit 1
}

# ============================ STEP 2: STAGE (local copy) ======================
# Mirror the runtime payload into a clean local temp tree with the excludes
# applied, so the scp -r in step 3 sends exactly the runtime code (no .venv,
# no data, no media, no caches). robocopy /MIR keeps the stage dir authoritative
# across re-runs; /XD + /XF carry the directory + file excludes.
$stg = Step 'stage runtime tree (robocopy, excludes applied)' {
    # Fresh stage dir each run (PID-scoped). Remove a stale one defensively.
    if (Test-Path -LiteralPath $StageRoot) {
        Remove-Item -LiteralPath $StageRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
    New-Item -ItemType Directory -Force -Path $StageRoot | Out-Null

    $anyFail = $false

    # robocopy exit codes 0-7 are SUCCESS (8+ is a real failure). We translate.
    function Invoke-Robo([string]$src, [string]$dst, [string[]]$roboArgs) {
        & robocopy $src $dst @roboArgs | Out-Null
        $rc = $LASTEXITCODE
        # robocopy: <8 == success (bits are "files copied / extra / mismatch"),
        # >=8 == at least one path failed.
        return ($rc -lt 8)
    }

    # Each payload DIR -> StageRoot\<dir>, recursive, with excludes.
    foreach ($d in $RUNTIME_DIRS) {
        $src = Join-Path $LocalRoot $d
        $dst = Join-Path $StageRoot $d
        $roboArgs = @('/MIR', '/NFL', '/NDL', '/NJH', '/NJS', '/NP', '/R:1', '/W:1')
        if ($EXCLUDE_DIRS.Count)  { $roboArgs += @('/XD') + $EXCLUDE_DIRS }
        if ($EXCLUDE_FILES.Count) { $roboArgs += @('/XF') + $EXCLUDE_FILES }
        if (Invoke-Robo $src $dst $roboArgs) {
            Write-Host "  staged dir   $d"
        } else {
            Write-Host "  FAIL staging dir $d (robocopy exit $LASTEXITCODE)"
            $anyFail = $true
        }
    }

    # Each payload FILE -> StageRoot\ (top level).
    foreach ($f in $RUNTIME_FILES) {
        $src = Join-Path $LocalRoot $f
        try {
            Copy-Item -LiteralPath $src -Destination $StageRoot -Force -ErrorAction Stop
            Write-Host "  staged file  $f"
        } catch {
            Write-Host "  FAIL staging file $f -- $($_.Exception.Message)"
            $anyFail = $true
        }
    }

    if ($anyFail) { return 'FAIL' } else { return 'OK' }
}
if ($stg -ne 'OK') {
    Write-Host "Staging FAILED -- not copying a partial tree. Fix the local copy"
    Write-Host "errors above and re-run."
    exit 1
}

# ============================ STEP 3: COPY (to host) ==========================
# scp -r the staged tree into RemoteRoot, then regenerate run_player.bat ASCII on
# the host with the right RemoteRoot. We ensure RemoteRoot exists first so scp -r
# lands children under it deterministically.
$null = Step 'copy runtime tree to host + regenerate run_player.bat' {
    $anyFail = $false

    # 3a. Ensure RemoteRoot exists on the host.
    try {
        $mk = "New-Item -ItemType Directory -Force -Path '$RemoteRoot' | Out-Null; Write-Output 'root ready'"
        $r = Invoke-RemotePS -Host_ $Target -Script $mk
        Write-Host "  remote root: $r ($RemoteRoot)"
    } catch {
        Write-Host "  FAIL could not create remote root $RemoteRoot -- $($_.Exception.Message)"
        return 'FAIL'   # without the root, every scp below fails the same way
    }

    # 3b. scp -r each staged item into RemoteRoot. We copy the staged children
    # (dirs + files) one by one so the destination layout is explicit and a
    # single failed dir is attributable in the transcript.
    $scpBase = @('-r', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10')

    foreach ($d in $RUNTIME_DIRS) {
        $localStaged = Join-Path $StageRoot $d
        $dest = "$($Target):$RemoteRootFwd/$d"
        & scp @scpBase '--' $localStaged $dest
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  FAIL scp dir $d (exit $LASTEXITCODE)"
            $anyFail = $true
        } else {
            Write-Host "  copied dir   $d -> $RemoteRoot\$d"
        }
    }
    foreach ($f in $RUNTIME_FILES) {
        $localStaged = Join-Path $StageRoot $f
        $dest = "$($Target):$RemoteRootFwd/$f"
        & scp '-o' 'BatchMode=yes' '-o' 'ConnectTimeout=10' '--' $localStaged $dest
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  FAIL scp file $f (exit $LASTEXITCODE)"
            $anyFail = $true
        } else {
            Write-Host "  copied file  $f -> $RemoteRoot\$f"
        }
    }

    # 3c. Regenerate run_player.bat ASCII on the host with the correct RemoteRoot.
    # The committed .bat hardcodes the SC2 immer path + uses python.exe; we write
    # a per-host copy that cd's into RemoteRoot and runs pythonw.exe (windowless,
    # matching how install_tasks.ps1 launches the player). Out-File ASCII so no
    # BOM/CRLF surprises on the Windows-codepage host.
    $batBody = @"
@echo off
cd /d $RemoteRoot
.venv\Scripts\pythonw.exe player.py
"@
    # Build the remote command that writes the .bat. Single here-string sent to
    # the host; we use Set-Content -Encoding ASCII there for a clean .bat.
    $remoteBatPath = (Join-Path $RemoteRoot 'run_player.bat')
    $batScript = @"
`$bat = @'
$batBody
'@
Set-Content -LiteralPath '$remoteBatPath' -Value `$bat -Encoding ASCII
Write-Output 'run_player.bat written'
"@
    try {
        $br = Invoke-RemotePS -Host_ $Target -Script $batScript
        Write-Host "  $br ($remoteBatPath, RemoteRoot baked in)"
    } catch {
        Write-Host "  FAIL regenerating run_player.bat -- $($_.Exception.Message)"
        $anyFail = $true
    }

    if ($anyFail) { return 'FAIL' } else { return 'OK' }
}

# ============================ STEP 4: VENV (on host) ==========================
# Create the runtime venv with py -3.12 and pip install the runtime deps. SKIP
# venv creation when .venv already exists (idempotent). pip install is a no-op
# when satisfied. This is the RUNTIME stack only -- pygame for the player, the
# cv2/numpy/imageio/Pillow set for the clip + frame plumbing the player imports.
$null = Step 'create runtime venv + pip install runtime deps' {
    # All of this runs on the host. Use the py launcher pinned to 3.12 so the
    # interpreter matches what install_tasks.ps1 expects at .venv\Scripts\.
    $venvScript = @"
`$ErrorActionPreference = 'Continue'
`$root  = '$RemoteRoot'
`$venv  = Join-Path `$root '.venv'
`$pyw   = Join-Path `$venv 'Scripts\pythonw.exe'
`$pip   = Join-Path `$venv 'Scripts\python.exe'

# 1. py -3.12 must be present on the host.
`$pyver = (& py -3.12 -c "import sys; print('.'.join(map(str, sys.version_info[:3])))" 2>&1)
if (`$LASTEXITCODE -ne 0) {
    Write-Output "PYLAUNCHER-FAIL: py -3.12 not available on this host (`$pyver)"
    exit 3
}
Write-Output "py -3.12 -> `$pyver"

# 2. Create the venv unless it already exists (idempotent).
if (Test-Path `$pyw) {
    Write-Output "venv exists -- skipping create (`$venv)"
} else {
    Write-Output "creating venv: py -3.12 -m venv `$venv"
    & py -3.12 -m venv `$venv
    if (`$LASTEXITCODE -ne 0) { Write-Output "VENV-CREATE-FAIL (exit `$LASTEXITCODE)"; exit 4 }
}

# 3. Upgrade pip then install the runtime deps. pip is idempotent (no-op when
#    satisfied). Explicit list -- the lean runtime stack only.
& `$pip -m pip install --upgrade pip --disable-pip-version-check
& `$pip -m pip install --disable-pip-version-check pygame opencv-python-headless numpy imageio-ffmpeg Pillow
if (`$LASTEXITCODE -ne 0) { Write-Output "PIP-INSTALL-FAIL (exit `$LASTEXITCODE)"; exit 5 }

# 4. Confirm pythonw landed.
if (Test-Path `$pyw) { Write-Output "VENV-OK: `$pyw" } else { Write-Output "VENV-MISSING-PYTHONW"; exit 6 }
"@
    try {
        $vr = Invoke-RemotePS -Host_ $Target -Script $venvScript
        $vr -split "`n" | ForEach-Object { if ($_.Trim()) { Write-Host "    $_" } }
        if ($vr -match 'VENV-OK:') { return 'OK' }
        Write-Host "  venv step did not report VENV-OK -- see lines above"
        return 'FAIL'
    } catch {
        Write-Host "  FAIL venv/pip on host -- $($_.Exception.Message)"
        return 'FAIL'
    }
}

# ============================ STEP 5: TASKS (on host) =========================
# Run install_tasks.ps1 on the host to register lp-player / lp-director /
# lp-watchdog at-logon. install_tasks.ps1 derives its own project root from its
# location ($PSScriptRoot's parent), so the copy we just landed at
# <RemoteRoot>\install\install_tasks.ps1 self-resolves to RemoteRoot with no
# edits. It is idempotent (-Force on every Register-ScheduledTask) and it
# smoke-runs the watchdog once at the end.
#
# NOTE: install_tasks.ps1 hardcodes RUN_USER='immer'. If the host's interactive
# user is NOT immer, the at-logon tasks would register against the wrong
# principal. We do not edit that file (owned elsewhere) -- this is flagged as a
# known gap in the deliverable. Pre-check + warn so the conductor sees it.
$null = Step 'register scheduled tasks (run install_tasks.ps1 on host)' {
    # Warn (do not fail) if the deploy user differs from immer, since
    # install_tasks.ps1 pins the principal to immer.
    if ($RemoteUser -and $RemoteUser -ne 'immer') {
        Write-Host "  WARN deploy user is '$RemoteUser' but install_tasks.ps1 pins"
        Write-Host "       RUN_USER='immer'. The at-logon tasks will register for"
        Write-Host "       'immer', not '$RemoteUser'. If '$RemoteUser' is the box's"
        Write-Host "       interactive user, edit RUN_USER in install_tasks.ps1 (owned"
        Write-Host "       by another task) or run as immer. (Known gap -- see summary.)"
    }

    $runScript = @"
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File '$RemoteInstallTasks'
"@
    try {
        $tr = Invoke-RemotePS -Host_ $Target -Script $runScript
        $tr -split "`n" | ForEach-Object { if ($_.Trim()) { Write-Host "    $_" } }
        # install_tasks.ps1 prints "[3/3] lp-watchdog registered" on success.
        if ($tr -match '\[3/3\]') { return 'OK' }
        Write-Host "  install_tasks.ps1 did not print the [3/3] line -- check output above"
        return 'FAIL'
    } catch {
        Write-Host "  FAIL running install_tasks.ps1 on host -- $($_.Exception.Message)"
        return 'FAIL'
    }
}

# ============================ STEP 6: VERIFY (on host) ========================
# Confirm the venv pythonw exists and the three tasks are registered. Pure read,
# never destructive. Prints a compact summary the conductor can eyeball.
$null = Step 'verify (pythonw present + tasks registered)' {
    $verifyScript = @"
`$root = '$RemoteRoot'
`$pyw  = Join-Path `$root '.venv\Scripts\pythonw.exe'
if (Test-Path `$pyw) { Write-Output "PYTHONW: present (`$pyw)" } else { Write-Output "PYTHONW: MISSING (`$pyw)" }

`$tasks = Get-ScheduledTask -TaskName lp-player, lp-director, lp-watchdog -ErrorAction SilentlyContinue
if (`$tasks) {
    foreach (`$t in `$tasks) {
        `$trig = (`$t.Triggers | ForEach-Object { `$_.CimClass.CimClassName -replace 'MSFT_Task','' -replace 'Trigger','' }) -join ','
        Write-Output ("TASK: {0,-12} state={1,-8} trigger={2,-10} runas={3}" -f `$t.TaskName, `$t.State, `$trig, `$t.Principal.UserId)
    }
} else {
    Write-Output "TASKS: none of lp-player/lp-director/lp-watchdog are registered"
}
"@
    try {
        $vr = Invoke-RemotePS -Host_ $Target -Script $verifyScript
        $vr -split "`n" | ForEach-Object { if ($_.Trim()) { Write-Host "    $_" } }
        $pyOk    = $vr -match 'PYTHONW: present'
        $taskCnt = ([regex]::Matches($vr, 'TASK:')).Count
        if ($pyOk -and $taskCnt -ge 3) { return 'OK' }
        Write-Host "  verify incomplete: pythonw present=$([bool]$pyOk), tasks found=$taskCnt (want 3)"
        return 'FAIL'
    } catch {
        Write-Host "  FAIL verify on host -- $($_.Exception.Message)"
        return 'FAIL'
    }
}

# ============================ CLEANUP + SUMMARY ==============================
# Drop the local stage dir (PID-scoped temp). Non-fatal if it lingers.
if (Test-Path -LiteralPath $StageRoot) {
    Remove-Item -LiteralPath $StageRoot -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "=================================================================="
Write-Host " deploy summary -- $Target ($RemoteRoot)"
Write-Host "=================================================================="
$anyFail = $false
foreach ($k in $RESULTS.Keys) {
    $v = $RESULTS[$k]
    if ($v -eq 'FAIL') { $anyFail = $true }
    "{0,-6} {1}" -f $v, $k | Write-Host
}
Write-Host ""
if ($anyFail) {
    Write-Host "One or more steps FAILED -- see the transcript above. Re-run this"
    Write-Host "script after fixing the failing step; it is idempotent (stage mirrors,"
    Write-Host "venv create skips if present, pip no-ops, tasks register with -Force)."
    exit 1
} else {
    Write-Host "All steps OK. The runtime show is deployed + durable on ${RemoteHost}:"
    Write-Host "  - code at $RemoteRoot (player.py / runtime / director / prompts / pipeline / install)"
    Write-Host "  - runtime venv at $RemoteRoot\.venv (pythonw present)"
    Write-Host "  - lp-player / lp-director / lp-watchdog registered at-logon"
    Write-Host ""
    Write-Host "REMAINING (not this script's job):"
    Write-Host "  - rendered clips + data\ are synced separately (the syncer / MIGRATION.md)."
    Write-Host "  - the director needs an Ollama qwen3 endpoint reachable from this host."
    Write-Host "  - if this host's interactive user is not 'immer', re-check the task"
    Write-Host "    principal (install_tasks.ps1 pins RUN_USER='immer')."
}
Write-Host "=================================================================="
