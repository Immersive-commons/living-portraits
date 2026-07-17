# sync_assets.ps1 -- copy the GPU-baked living-portraits assets from supercommons2
# (the GPU host) to a NO-GPU player host, so that host's player + director can
# perform the show without ever running the heavy generative stack.
#
# WHY THIS EXISTS
# ---------------
# supercommons2 (SC2, immer@<producer-host>) has the RTX 2080 Ti. It bakes every
# per-character portrait / cutout / bg-plate / rig-spec and runs qwen3 for the
# director. A second host with NO GPU (call it "hil") can still RUN the show --
# player.py + stage_manager.py are pure-CPU at runtime -- but it has no way to
# PRODUCE the assets. Two things must therefore reach hil from SC2:
#
#   1. THE ASSETS (this script): the per-slug files the runtime reads from
#      data\gen\ plus the clip-graph manifest. The exact set the player +
#      stage_render + rig_loop consume:
#         data\gen\<slug>_portrait.png   (generate.py  -- the painting / fallback plate)
#         data\gen\<slug>_cutout.png     (segment.py   -- RGBA figure, alpha-composited)
#         data\gen\<slug>_bg.png         (segment.py   -- figure removed + inpainted plate)
#         data\gen\<slug>_rig.json       (rig_spec.py  -- Live2D-style rig regions/warp)
#         data\clips\manifest.json       (clip_graph.py -- the KV-index of clip rows)
#      (The data\gen\_selftest\ subdir and any _test_* / bad_* debris are LOCAL
#      test output and are NOT synced -- only the real per-slug assets travel.)
#
#   2. THE DIRECTOR'S OLLAMA (a 2-line code edit, NOT done by this script -- see
#      "REMOTE DIRECTOR SWITCH" below). hil has no GPU and so no local qwen3; its
#      stage_manager must call SC2's Ollama over the tailnet instead of 127.0.0.1.
#
# TOPOLOGY: STAGE-VIA-LOCAL (default) vs DIRECT host-to-host
# ---------------------------------------------------------
# This script runs from the life-repo dev box (jtole), which holds the SSH keys
# to BOTH SC2 and the target. The default and most reliable path is therefore a
# TWO-HOP STAGE:
#
#       SC2 (data\gen + data\clips)
#         |  scp  pull          (leg 1: dev box pulls from SC2)
#         v
#       local staging dir  ($env:TEMP\lp-asset-sync\)
#         |  scp  push          (leg 2: dev box pushes to target)
#         v
#       TARGET host  data\gen + data\clips
#
# Both scp legs originate from the dev box, where key auth to each host is known
# to work (per memory: `ssh immer@<producer-host>` is key-auth; the target is
# whatever host the bring-up already brought onto the tailnet with the same key).
# We do NOT assume SC2 itself can SSH to the target -- that is a separate trust
# relationship the fleet does not generally grant, so host-to-host is OFF by
# default.
#
# -Direct attempts a single host-to-host hop instead (scp from SC2 straight to
# the target, driven over SSH-to-SC2). It is faster (no local round-trip, no
# temp space) BUT requires SC2 -> target SSH key trust to already exist. If that
# trust is absent, -Direct fails on the copy; fall back to the default stage path.
# See "KNOWN GAPS" at the foot of this file.
#
# USAGE
# -----
#   # Default: pull from SC2, stage locally, push to the target. Discovers the
#   # slugs from whatever portraits exist on SC2 (minus the _selftest dir).
#   powershell -NoProfile -ExecutionPolicy Bypass -File install\sync_assets.ps1 `
#       -Target immer@100.x.y.z
#
#   # Pin the slugs explicitly (skip remote discovery):
#   ... -File install\sync_assets.ps1 -Target user@hil -Slugs phineas,seraphina
#
#   # Override the SC2 source host / project paths if they ever move:
#   ... -Source immer@<producer-host> `
#       -SourceProject C:/Users/immer/living-portraits `
#       -TargetProject C:/Users/immer/living-portraits
#
#   # Attempt a direct SC2 -> target hop (needs SC2->target SSH trust):
#   ... -Target user@hil -Direct
#
#   # Preview only -- list what WOULD sync, copy nothing:
#   ... -Target user@hil -DryRun
#
# Idempotent: scp overwrites the destination files every run (last write wins),
# exactly matching the clip-graph's own "last write wins per edge" discipline.
# Re-running after a fresh bake on SC2 simply refreshes the target. Prints one
# line per file synced + a final tally so the conductor's transcript is honest
# about what landed.
#
# REMOTE DIRECTOR SWITCH (the 2-line edit -- DO THIS ON THE TARGET, not here)
# --------------------------------------------------------------------------
# hil has no GPU and therefore no local Ollama/qwen3. Its director must reach
# SC2's Ollama over the tailnet. director\stage_manager.py currently hardcodes:
#
#       OLLAMA = "http://127.0.0.1:11434/api/chat"
#
# Make that line env-overridable (a 2-line change the conductor applies to
# stage_manager.py -- this script does NOT edit it):
#
#       import os                                                        # near the other imports
#       OLLAMA = os.environ.get("LP_OLLAMA", "http://127.0.0.1:11434/api/chat")
#
# Then on the NO-GPU target, set the env var so its director calls SC2's Ollama
# (SC2's Ollama listens on 127.0.0.1:11434; reach it from another host at its
# tailnet IP <producer-host> -- confirmed by the supercommons2 memory note that
# the Windows-host Ollama is at <producer-host>:11434, NOT localhost, from off-box):
#
#       LP_OLLAMA = http://<producer-host>:11434/api/chat
#
# Set it where the director's scheduled task can see it (a Machine-scope env var,
# or inside the task's launcher), e.g.:
#
#       [Environment]::SetEnvironmentVariable(
#           'LP_OLLAMA', 'http://<producer-host>:11434/api/chat', 'Machine')
#
# With the env var unset (the SC2/full-local case) the const falls back to
# 127.0.0.1 and nothing changes. NOTE: for an off-box client to reach it, SC2's
# Ollama must be bound to listen beyond loopback (OLLAMA_HOST=0.0.0.0) and the
# tailnet ACL must permit hil -> SC2:11434. Both are SC2/ACL-side concerns the
# conductor verifies out-of-band; this script neither checks nor changes them.

[CmdletBinding()]
param(
    # The NO-GPU destination, in scp/ssh "user@host" form (e.g. immer@100.x.y.z).
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$Target,

    # The GPU source. Defaults to SC2 (the canonical baker).
    [string]$Source = 'immer@<producer-host>',

    # Project roots on each host (forward slashes -- scp/ssh remote paths).
    [string]$SourceProject = 'C:/Users/immer/living-portraits',
    [string]$TargetProject = 'C:/Users/immer/living-portraits',

    # Explicit slug list; when empty we discover slugs from SC2's data\gen.
    [string[]]$Slugs = @(),

    # Attempt a single host-to-host hop (SC2 -> target) instead of staging via
    # the local box. Requires SC2 -> target SSH key trust.
    [switch]$Direct,

    # List what would sync; copy nothing.
    [switch]$DryRun,

    # scp/ssh connect timeout (seconds). Matches invoke-remote-ps.ps1 default.
    [int]$ConnectTimeoutSec = 8,

    # Optional identity file handed to both scp and ssh.
    [string]$IdentityFile
)

$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Guards -- reject anything that could turn a host arg into an ssh/scp flag or
# inject shell metacharacters. Mirrors invoke-remote-ps.ps1's host validation so
# this standalone script is just as safe when the conductor passes -Target.
# ---------------------------------------------------------------------------
function Assert-SafeHost([string]$h, [string]$label) {
    if ($h -match '[\s;&|`<>$"' + "'" + ']') {
        throw "sync_assets: $label contains shell metacharacters; refusing: $h"
    }
    if ($h.StartsWith('-')) {
        throw "sync_assets: $label starts with '-' (could parse as an ssh/scp flag); refusing: $h"
    }
}
Assert-SafeHost $Source 'Source'
Assert-SafeHost $Target 'Target'

if ($IdentityFile -and -not (Test-Path -LiteralPath $IdentityFile)) {
    throw "sync_assets: IdentityFile not found: $IdentityFile"
}

# Common ssh/scp option block. BatchMode=yes so a missing key fails fast instead
# of hanging on a password prompt (this is non-interactive conductor tooling).
$commonOpts = @(
    '-o', 'BatchMode=yes',
    '-o', "ConnectTimeout=$ConnectTimeoutSec"
)
if ($IdentityFile) { $commonOpts += @('-i', $IdentityFile) }

Write-Host '=================================================================='
Write-Host ' living-portraits asset sync  (GPU host -> no-GPU host)'
Write-Host '=================================================================='
Write-Host "Source (GPU)   : $Source : $SourceProject"
Write-Host "Target (no-GPU): $Target : $TargetProject"
Write-Host ("Mode           : {0}{1}" -f ($(if ($Direct) { 'DIRECT host-to-host' } else { 'STAGE via local' })), ($(if ($DryRun) { '   [DRY RUN]' } else { '' })))
Write-Host ''

# ---------------------------------------------------------------------------
# Run ssh against the SOURCE, returning stdout as a single string. Used only for
# slug discovery (a read). Uses the base64 EncodedCommand pattern from
# feedback_ps_over_ssh_b64 so the remote PowerShell is never quote-mangled.
# ---------------------------------------------------------------------------
function Invoke-SourcePS([string]$psScript) {
    $bytes = [Text.Encoding]::Unicode.GetBytes($psScript)
    $b64 = [Convert]::ToBase64String($bytes)
    $remoteCmd = 'powershell.exe -NoProfile -EncodedCommand ' + $b64
    $sshArgs = $commonOpts + @('--', $Source, $remoteCmd)
    $global:LASTEXITCODE = 0
    $out = & ssh @sshArgs
    if ($LASTEXITCODE -ne 0) {
        throw "sync_assets: ssh to $Source failed (exit $LASTEXITCODE) during slug discovery"
    }
    return ($out -join "`n")
}

# ---------------------------------------------------------------------------
# Slug discovery -- when -Slugs is not given, list SC2's data\gen for
# *_portrait.png and derive the slug from each filename, EXCLUDING the _selftest
# subdir (and any _test_* / bad_* debris stage_render leaves in the real dir).
# Discovering from the portraits (not cutouts/rigs) anchors on the first stage's
# output -- a slug with a portrait but no cutout still surfaces, so a missing
# cutout shows up as a per-file FAIL rather than the slug vanishing silently.
# ---------------------------------------------------------------------------
function Get-SourceSlugs() {
    $genGlob = "$SourceProject/data/gen"
    # -Filter on the engine side keeps it fast; we still post-filter names in PS.
    # BaseName strips the extension; we then require the _portrait suffix and peel
    # it off. _selftest is a directory so Get-ChildItem -File never returns it,
    # but we belt-and-suspenders exclude any leading-underscore stem too.
    $probe = @"
`$ErrorActionPreference = 'Stop'
`$dir = '$genGlob'
if (-not (Test-Path -LiteralPath `$dir)) { Write-Output ''; exit 0 }
Get-ChildItem -LiteralPath `$dir -File -Filter '*_portrait.png' |
  ForEach-Object { `$_.BaseName } |
  Where-Object { `$_ -like '*_portrait' } |
  ForEach-Object { `$_.Substring(0, `$_.Length - '_portrait'.Length) } |
  Where-Object { `$_ -and -not `$_.StartsWith('_') } |
  Sort-Object -Unique
"@
    $raw = Invoke-SourcePS $probe
    return @($raw -split "`r?`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}

# Resolve the slug list (explicit beats discovery).
#
# Split each -Slugs element on commas / semicolons / whitespace. This matters
# because `powershell.exe -File ... -Slugs phineas,seraphina` (the form the
# conductor uses) binds the whole "phineas,seraphina" as a SINGLE string element
# -- array-comma-splitting only happens for in-process calls, not via -File. So
# we always re-split, which makes both `-Slugs phineas,seraphina` (one token)
# and `-Slugs phineas seraphina` (a real two-element array) resolve identically.
if ($Slugs.Count -gt 0) {
    $slugList = @(
        $Slugs |
          ForEach-Object { $_ -split '[,;\s]+' } |
          ForEach-Object { $_.Trim() } |
          Where-Object { $_ }
    )
    Write-Host "Slugs (explicit): $($slugList -join ', ')"
} else {
    Write-Host 'Slugs: discovering from SC2 data\gen (*_portrait.png, excluding _selftest)...'
    try {
        $slugList = Get-SourceSlugs
    } catch {
        Write-Host "  discovery FAILED: $($_.Exception.Message)"
        Write-Host '  Re-run with an explicit -Slugs list (e.g. -Slugs phineas,seraphina) to skip discovery.'
        exit 1
    }
    if (-not $slugList -or $slugList.Count -eq 0) {
        Write-Host "  no *_portrait.png found under $SourceProject/data/gen -- nothing to sync."
        Write-Host '  Has the GPU host baked any portraits yet? (pipeline/generate.py)'
        exit 1
    }
    Write-Host "  discovered: $($slugList -join ', ')"
}
Write-Host ''

# ---------------------------------------------------------------------------
# Build the relative-path worklist: per-slug gen files + the single clip manifest.
# Relative to the project root on each host; the source/target absolute paths are
# composed per leg below.
# ---------------------------------------------------------------------------
$relPaths = New-Object System.Collections.Generic.List[string]
foreach ($s in $slugList) {
    $relPaths.Add("data/gen/${s}_portrait.png")
    $relPaths.Add("data/gen/${s}_cutout.png")
    $relPaths.Add("data/gen/${s}_bg.png")
    $relPaths.Add("data/gen/${s}_rig.json")
}
# The clip-graph KV-index (one shared file, not per-slug).
$relPaths.Add('data/clips/manifest.json')

Write-Host ("Worklist: {0} file(s) ({1} slug(s) x 4 gen files + 1 manifest)" -f $relPaths.Count, $slugList.Count)
foreach ($r in $relPaths) { Write-Host "  - $r" }
Write-Host ''

if ($DryRun) {
    Write-Host '[DRY RUN] No files copied. Re-run without -DryRun to perform the sync.'
    Write-Host '=================================================================='
    exit 0
}

# Per-file result tally for the honest final summary.
$results = [ordered]@{}

# ---------------------------------------------------------------------------
# DIRECT host-to-host: one scp invocation per file, SC2 -> target, driven from
# the dev box. Requires SC2 -> target SSH key trust. scp's two-remote form copies
# straight between hosts without a local round-trip.
# ---------------------------------------------------------------------------
function Sync-Direct([string]$rel) {
    # Ensure the target's parent dir exists first (scp will not mkdir -p). We ssh
    # to the TARGET to create it, since the file is being written there.
    $parentRel = Split-Path -Parent ($rel -replace '/', '\')   # windows-style for the remote PS
    $parentFwd = ($parentRel -replace '\\', '/')
    $mkdir = "New-Item -ItemType Directory -Force -Path '$TargetProject/$parentFwd' | Out-Null"
    $b64 = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($mkdir))
    $mkArgs = $commonOpts + @('--', $Target, ('powershell.exe -NoProfile -EncodedCommand ' + $b64))
    $global:LASTEXITCODE = 0
    & ssh @mkArgs 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Host "    FAIL (mkdir on target) $rel"; return 'FAIL' }

    $src = "$($Source):$SourceProject/$rel"
    $dst = "$($Target):$TargetProject/$rel"
    $scpArgs = $commonOpts + @('-3', '--', $src, $dst)   # -3 routes the data through the local box's scp even in two-remote mode
    $global:LASTEXITCODE = 0
    & scp @scpArgs 2>&1 | ForEach-Object { Write-Host "      $_" }
    if ($LASTEXITCODE -ne 0) { Write-Host "    FAIL $rel"; return 'FAIL' }
    Write-Host "    OK   $rel"
    return 'OK'
}

# ---------------------------------------------------------------------------
# STAGE-VIA-LOCAL: pull each file SC2 -> local temp, then push local -> target.
# Both legs originate from the dev box (known-good key auth to each host).
# ---------------------------------------------------------------------------
$stageRoot = Join-Path $env:TEMP 'lp-asset-sync'

function Sync-Staged([string]$rel) {
    $localPath = Join-Path $stageRoot ($rel -replace '/', '\')
    $localDir = Split-Path -Parent $localPath
    New-Item -ItemType Directory -Force -Path $localDir | Out-Null

    # Leg 1: pull from SC2 into the local staging dir.
    $src = "$($Source):$SourceProject/$rel"
    $pullArgs = $commonOpts + @('--', $src, $localPath)
    $global:LASTEXITCODE = 0
    & scp @pullArgs 2>&1 | ForEach-Object { Write-Host "      pull: $_" }
    if ($LASTEXITCODE -ne 0) { Write-Host "    FAIL (pull from source) $rel"; return 'FAIL' }
    if (-not (Test-Path -LiteralPath $localPath)) {
        Write-Host "    FAIL (pull exited 0 but local file missing) $rel"; return 'FAIL'
    }

    # Ensure the target parent dir exists (scp will not create it).
    $parentFwd = (Split-Path -Parent $rel) -replace '\\', '/'
    $mkdir = "New-Item -ItemType Directory -Force -Path '$TargetProject/$parentFwd' | Out-Null"
    $b64 = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($mkdir))
    $mkArgs = $commonOpts + @('--', $Target, ('powershell.exe -NoProfile -EncodedCommand ' + $b64))
    $global:LASTEXITCODE = 0
    & ssh @mkArgs 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Host "    FAIL (mkdir on target) $rel"; return 'FAIL' }

    # Leg 2: push the staged file to the target (overwrite).
    $dst = "$($Target):$TargetProject/$rel"
    $pushArgs = $commonOpts + @('--', $localPath, $dst)
    $global:LASTEXITCODE = 0
    & scp @pushArgs 2>&1 | ForEach-Object { Write-Host "      push: $_" }
    if ($LASTEXITCODE -ne 0) { Write-Host "    FAIL (push to target) $rel"; return 'FAIL' }

    Write-Host "    OK   $rel"
    return 'OK'
}

# ---------------------------------------------------------------------------
# Run the worklist.
# ---------------------------------------------------------------------------
if (-not $Direct) {
    New-Item -ItemType Directory -Force -Path $stageRoot | Out-Null
    Write-Host "Staging dir: $stageRoot"
    Write-Host ''
}

Write-Host 'Syncing...'
foreach ($rel in $relPaths) {
    Write-Host "  $rel"
    if ($Direct) {
        $results[$rel] = (Sync-Direct $rel)
    } else {
        $results[$rel] = (Sync-Staged $rel)
    }
}
Write-Host ''

# ---------------------------------------------------------------------------
# Summary -- one line per file, then a tally. Exit non-zero if anything failed
# so the conductor's transcript (and any wrapping automation) sees the failure.
# ---------------------------------------------------------------------------
Write-Host '=================================================================='
Write-Host ' sync summary'
Write-Host '=================================================================='
$okCount = 0
$failCount = 0
foreach ($k in $results.Keys) {
    $v = $results[$k]
    if ($v -eq 'OK') { $okCount++ } else { $failCount++ }
    "{0,-6} {1}" -f $v, $k | Write-Host
}
Write-Host ''
Write-Host ("Done: {0} OK, {1} FAIL ({2} file(s) total)." -f $okCount, $failCount, $results.Count)

if ($failCount -gt 0) {
    Write-Host ''
    Write-Host 'One or more files FAILED. Common causes:'
    Write-Host '  - A slug has a portrait on SC2 but no cutout/bg/rig yet (bake not finished).'
    Write-Host '    -> Run the producer backfill on SC2 first (install\sc2_bringup.ps1), then re-sync.'
    Write-Host '  - data\clips\manifest.json does not exist on SC2 yet (no clips registered).'
    Write-Host '    -> Expected on a fresh box; re-sync after the first clip-row lands.'
    Write-Host '  - (-Direct only) SC2 cannot SSH to the target -- drop -Direct to stage via local.'
    exit 1
}

Write-Host ''
Write-Host 'All assets synced. Reminder: also apply the REMOTE DIRECTOR SWITCH on the'
Write-Host 'target so its stage_manager calls SC2''s Ollama (see this script''s header):'
Write-Host '  OLLAMA = os.environ.get("LP_OLLAMA", "http://127.0.0.1:11434/api/chat")'
Write-Host '  LP_OLLAMA = http://<producer-host>:11434/api/chat   (set on the target)'
Write-Host '=================================================================='

# ===========================================================================
# KNOWN GAPS / NOTES (read before relying on this in a new fleet config)
# ===========================================================================
# 1. DIRECT vs STAGED. Default is STAGED (two hops via the local box) because it
#    only needs the dev box's existing key auth to SC2 and to the target -- no
#    SC2->target trust. -Direct is faster but needs SC2->target SSH key trust;
#    if that is absent it FAILs on the copy and you should drop -Direct. (-3 on
#    the direct scp still routes bytes through the local box, but the control
#    connections are SC2<->local and target<->local, so SC2->target trust is
#    only needed for scp's non-`-3` path; we keep -3 to avoid requiring it where
#    possible, but older scp builds vary -- staged remains the safe default.)
# 2. NO DELETE / NO PRUNE. This is an overwrite-only push. If a slug is RETIRED
#    on SC2 (its portrait removed), this script will not delete the stale copy on
#    the target. Manual cleanup on the target if a character is dropped.
# 3. SLUG DISCOVERY anchors on *_portrait.png. A slug whose portrait exists but
#    whose cutout/bg/rig have not been baked yet will surface those missing files
#    as per-file FAILs (intentional -- it tells you the bake is incomplete rather
#    than silently shipping a half-set).
# 4. CLIP ASSETS. Only data\clips\manifest.json is synced. If/when real clip
#    media (mp4 / png-sequence dirs referenced by a manifest row's `asset` field)
#    is baked on SC2, this script does NOT yet pull those media files -- today the
#    library is synthetic-placeholder (asset=None) per clip_graph.py, so the
#    manifest alone is the whole library. Extend $relPaths to walk the manifest's
#    non-null `asset` paths when real footage starts landing.
# 5. The REMOTE DIRECTOR SWITCH is a code edit the CONDUCTOR makes to
#    stage_manager.py (2 lines, shown above + in install\SYNC.md). This script
#    deliberately does NOT touch stage_manager.py. It also does not verify SC2's
#    Ollama is reachable from the target (bind-beyond-loopback + tailnet ACL are
#    SC2/ACL-side concerns).
