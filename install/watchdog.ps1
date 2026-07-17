# watchdog.ps1 -- runs every 5 min via the 'lp-watchdog' scheduled task.
#
# Purpose: keep the living-portraits show alive on supercommons2. Two long-lived
# pythonw processes drive the show; neither serves HTTP, so this watchdog probes
# by PROCESS EXISTENCE (Win32_Process.CommandLine match), not by a /healthz port.
# If either process is gone, re-trigger its launch task (schtasks /Run). The
# launch tasks live in their own interactive session and land the player window
# back on console session 1.
#
# Processes kept alive:
#   lp-player    -- .venv\Scripts\pythonw.exe player.py
#                   (the borderless pinned dual-panel stage window)
#   lp-director  -- .venv\Scripts\pythonw.exe director\stage_manager.py --loop 45
#                   (the qwen3 beat loop that writes data\stage_state.json)
#
# Hard rules:
#   - Idempotent. Safe to run with a healthy show (no-op, no restarts).
#   - Never kills a LIVING process. A live process is healthy by definition here;
#     the director legitimately PAUSES its loop during portrait renders (frees
#     VRAM), so a frozen stage_state.json with a live director is NOT a fault.
#     The only restart trigger is process ABSENCE.
#   - Never blocks. CIM queries are local and fast; schtasks /Run returns at once.
#   - Logs to <project>\watchdog.log (rotated when >2MB).
#
# Director freshness (stage_state.json mtime) is logged as an ADVISORY signal
# only -- it does NOT trigger a restart, because a paused-for-render director
# rightfully stops writing. It is there so a human reading the log can tell a
# wedged loop (process alive, file frozen for hours) from a paused one.
#
# Layout: drop this file at C:\Users\immer\living-portraits\install\watchdog.ps1
# on supercommons2. install_tasks.ps1 registers the scheduled task that runs it.

$ErrorActionPreference = 'Continue'

# Project root = parent of this script's folder (install\ lives under the project).
$PROJECT  = Split-Path -Parent $PSScriptRoot
if (-not $PROJECT) { $PROJECT = 'C:\Users\immer\living-portraits' }
$LOG      = Join-Path $PROJECT 'watchdog.log'
$STATE    = Join-Path $PROJECT 'data\stage_state.json'

# Generous staleness threshold for the advisory director-freshness line. The
# director loops every 45s, but renders pause it for minutes at a time, so this
# is set well above the loop cadence -- it flags an HOURS-frozen file, not a
# normal render pause.
$STATE_STALE_MINUTES = 30

# Each managed process: a friendly name, the substring its CommandLine must
# contain, and the launch task to re-trigger when it is absent.
$TARGETS = @(
    @{ Name = 'lp-player';   Match = 'player.py';        LaunchTask = 'lp-player'   }
    @{ Name = 'lp-director'; Match = 'stage_manager.py'; LaunchTask = 'lp-director' }
)

function Log($msg) {
    $line = "{0} {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Add-Content -Path $LOG -Value $line -Encoding UTF8 -ErrorAction SilentlyContinue
    Write-Output $line
}

function Rotate-Log {
    if ((Test-Path $LOG) -and ((Get-Item $LOG).Length -gt 2MB)) {
        Move-Item -Path $LOG -Destination "$LOG.1" -Force -ErrorAction SilentlyContinue
    }
}

# Is a pythonw/python process whose CommandLine contains $Match currently alive?
# CIM Win32_Process exposes the full command line, which is how we tell the
# player apart from the director (both are the same .venv pythonw.exe).
function Test-ProcAlive($match) {
    try {
        $procs = Get-CimInstance Win32_Process `
            -Filter "Name = 'pythonw.exe' OR Name = 'python.exe'" `
            -ErrorAction Stop
        foreach ($p in $procs) {
            if ($p.CommandLine -and $p.CommandLine.Contains($match)) {
                return @{ alive = $true; pid = $p.ProcessId }
            }
        }
        return @{ alive = $false; pid = 0 }
    } catch {
        # If CIM itself errors we cannot prove the process is dead; report a
        # probe error so we do NOT issue a spurious restart on a transient WMI
        # hiccup.
        return @{ alive = $false; pid = 0; probe_error = $_.Exception.Message }
    }
}

# Re-trigger a launch task. We do NOT kill anything first: a process is only
# restarted when it is ABSENT, so there is no stale instance to clean up.
function Restart-Target($name, $launchTask) {
    Log "RESTART $name -- process absent, triggering task '$launchTask'"
    & schtasks /Run /TN $launchTask *>&1 | ForEach-Object { Log "  $_" }
    # Confirm the process comes back. Player paints immediately; the director
    # spawns just as fast (the first beat takes longer, but the process exists
    # right away). Poll process existence for up to 15s.
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Milliseconds 500
        $still = Test-ProcAlive $TARGETS_BY_NAME[$name].Match
        if ($still.alive) {
            Log ("  {0} back as PID {1} after {2}s" -f $name, $still.pid, [Math]::Round(0.5 * ($i + 1), 1))
            return $true
        }
    }
    Log "  WARNING: $name did not reappear within 15s (check $name task + the proc's own log)"
    return $false
}

# Advisory only: report how stale stage_state.json is. Never restarts on this.
function Report-DirectorFreshness {
    if (-not (Test-Path $STATE)) {
        Log "ADVISORY: stage_state.json not present yet (director may not have written a first beat)"
        return
    }
    $ageMin = ((Get-Date) - (Get-Item $STATE).LastWriteTime).TotalMinutes
    if ($ageMin -gt $STATE_STALE_MINUTES) {
        Log ("ADVISORY: stage_state.json is {0:N1} min stale (> {1} min). Director process is alive; likely a long render pause, but if this persists for hours the beat loop may be wedged -- inspect director.log." -f $ageMin, $STATE_STALE_MINUTES)
    }
}

# -------- main --------
Rotate-Log

# Build a name->target index so Restart-Target can re-probe by name.
$TARGETS_BY_NAME = @{}
foreach ($t in $TARGETS) { $TARGETS_BY_NAME[$t.Name] = $t }

$restarted = $false
$directorAlive = $false

foreach ($t in $TARGETS) {
    $r = Test-ProcAlive $t.Match
    if ($r.ContainsKey('probe_error')) {
        Log "PROBE-ERROR $($t.Name): $($r.probe_error) -- skipping restart this tick (cannot confirm death)"
        continue
    }
    if (-not $r.alive) {
        Restart-Target $t.Name $t.LaunchTask | Out-Null
        $restarted = $true
    } elseif ($t.Name -eq 'lp-director') {
        $directorAlive = $true
    }
}

# Director-freshness advisory only when the director is actually running (a
# just-restarted or dead director has nothing meaningful to say about mtime).
if ($directorAlive) { Report-DirectorFreshness }

# Healthy + nothing restarted -- stay quiet on a healthy box. Log a heartbeat
# only every 12th tick (about hourly) so the log proves the watchdog is running
# without one line every 5 minutes.
if (-not $restarted) {
    $tickFile = Join-Path $PROJECT '.watchdog-tick'
    $tick = 0
    if (Test-Path $tickFile) { $tick = [int](Get-Content $tickFile -ErrorAction SilentlyContinue) }
    $tick = ($tick + 1) % 12
    $tick | Set-Content $tickFile -Encoding ASCII
    if ($tick -eq 1) { Log "ok (hourly heartbeat) -- player + director both alive" }
}
