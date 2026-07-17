# stop_portraits.ps1 -- one-click STOP for the Living Portraits show on desktop-hil08rd.
#
# Double-click "Stop Living Portraits" on the Desktop. Takes the show DOWN and KEEPS it down.
# The self-heal (the lp-watchdog-preview task + heartbeat._ensure_player) both RESPECT a
# Disabled task, so the off-switch is a strict 3-phase teardown:
#   A) DISABLE both lp-mind + lp-preview FIRST -- with both off-switches set, neither the
#      watchdog nor a last heartbeat tick can repop a panel while we tear the run down. (The
#      old order disabled the player before the brain, leaving a window where the still-live
#      brain's _ensure_player could /Run the player right back.)
#   B) END the running instances (Stop-ScheduledTask, brain before player).
#   C) FORCE-KILL any pythonw that didn't wind down -- Stop-ScheduledTask only ends
#      task-tracked runs, so an orphaned player would otherwise keep painting the panels.
#
# Then it RE-QUERIES the truth (task State + both processes) and reports DOWN only when it's
# really down. It never prints "disabled" it didn't verify: a Disable can be silently refused
# if the task runs under a principal you can't edit without admin -- in that case it SAYS SO
# and tells you to Run as administrator, instead of lying while the watchdog repops the show.
#
# To bring it back: double-click "Start Living Portraits" (re-enables + starts everything).
# The stop PERSISTS across reboots until you hit Start -- that is the point. OllamaServe and
# lp-watchdog-preview are left running on purpose (both no-op on a disabled task).
param([switch]$NoPause)

$ErrorActionPreference = 'SilentlyContinue'
try { $Host.UI.RawUI.WindowTitle = "Living Portraits -- Stop" } catch {}
function Line($m, $c = 'Gray') { Write-Host $m -ForegroundColor $c }

Line "===============================================" Magenta
Line "   Living Portraits  --  stopping the show"        Magenta
Line "===============================================" Magenta
Line ""

# Brain BEFORE player everywhere below: lp-mind's heartbeat is what re-launches lp-preview.
# 'needle' = the substring that identifies THIS task's pythonw process by command line.
$tasks = @(
    @{ n = 'lp-mind';    needle = 'heartbeat.py';      d = "GLM heartbeat -- chooses each pose" },
    @{ n = 'lp-preview'; needle = '_preview_graph.py'; d = "the player -- paints the panels" }
)

# Only operate on tasks that actually exist on this host.
$present = @()
foreach ($t in $tasks) {
    if (Get-ScheduledTask -TaskName $t.n -ErrorAction SilentlyContinue) { $present += $t }
    else { Line ("  [skip] {0,-12} not installed on this PC" -f $t.n) DarkYellow }
}

# --- Phase A: DISABLE both off-switches first, and VERIFY each actually took. ---
# Check Settings.Enabled, NOT State: a task with a running instance reports State='Running'
# even after a successful Disable, so a State check cries a false "needs admin" on every
# normal stop. Settings.Enabled is the real off-switch flag, independent of a running instance.
$denied = @()
foreach ($t in $present) {
    Disable-ScheduledTask -TaskName $t.n -ErrorAction SilentlyContinue | Out-Null
    $enabled = (Get-ScheduledTask -TaskName $t.n -ErrorAction SilentlyContinue).Settings.Enabled
    if ($enabled -eq $false) {
        Line ("  [off ] {0,-12} disabled  ({1})" -f $t.n, $t.d) Yellow
    } else {
        $denied += $t.n
        Line ("  [WARN] {0,-12} could NOT disable -- needs admin (Run as administrator)" -f $t.n) Red
    }
}

# --- Phase B: END the currently-running task instances (brain before player). ---
foreach ($t in $present) { Stop-ScheduledTask -TaskName $t.n -ErrorAction SilentlyContinue | Out-Null }

# --- Phase C: FORCE-KILL anything still up. Safe BECAUSE we already disabled in Phase A,
# so neither the watchdog nor a (now-dead) brain will repop what we kill here. ---
Line ""
Line "Force-stopping any lingering processes..." Magenta
$pw = Get-CimInstance Win32_Process -Filter "Name='pythonw.exe'" -ErrorAction SilentlyContinue
$killed = 0
foreach ($t in $present) {
    foreach ($p in ($pw | Where-Object { $_.CommandLine -like "*$($t.needle)*" })) {
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
        Line ("  [kill] {0,-12} pid {1}" -f $t.n, $p.ProcessId) Yellow
        $killed++
    }
}
if ($killed -eq 0) { Line "  (nothing lingering)" DarkGray }
Start-Sleep -Seconds 2

# --- Verify the TRUTH -- report from a fresh query, never from what we hoped happened. ---
Line ""
Line "=== status ===" Magenta
foreach ($t in $tasks) {
    $s = (Get-ScheduledTask -TaskName $t.n -ErrorAction SilentlyContinue).State
    Line ("  {0,-14} {1}" -f $t.n, $s)
}
$pwNow  = Get-CimInstance Win32_Process -Filter "Name='pythonw.exe'" -ErrorAction SilentlyContinue
$player = $pwNow | Where-Object { $_.CommandLine -like "*_preview_graph.py*" }
$mind   = $pwNow | Where-Object { $_.CommandLine -like "*heartbeat.py*" }
Line ""
if ($denied.Count) {
    Line ("  Could NOT disable: {0}." -f ($denied -join ', ')) Red
    Line "  The show can pop back up within ~5 min. Right-click 'Stop Living Portraits'" Red
    Line "  -> Run as administrator to make the stop stick." Red
} elseif ($player -or $mind) {
    $still = @(); if ($mind) { $still += 'brain' }; if ($player) { $still += 'player' }
    Line ("  {0} still winding down -- tasks are disabled, so it will NOT restart." -f ($still -join ' + ')) Yellow
    Line "  Give it a few seconds; the panels go dark once it exits." Yellow
} else {
    Line "  Show is DOWN and will STAY down (self-heal paused)." Green
    Line "  Double-click 'Start Living Portraits' when you want it back." Cyan
}
Line ""
if (-not $NoPause) {
    Line "Press any key to close..."
    try { $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") } catch { Start-Sleep -Seconds 4 }
}
