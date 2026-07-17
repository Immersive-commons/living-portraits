# lp_watchdog_preview.ps1 -- scoped self-heal for the Living Portraits panels.
#
# WHY: the player (lp-preview) and brain (lp-mind) can exit CLEANLY (result 0) -- e.g.
# a fullscreen app grabbing the display sends pygame a QUIT. A clean exit is NOT a
# "failure", so the task's RestartOnFailure never fires and the show stays dark until a
# human restarts it (happened 2026-06-06: a game took the console, panels went dark for
# hours). This watchdog catches that: every few minutes it checks each task's PROCESS is
# alive and restarts the task if not.
#
# DELIBERATELY SCOPED: it touches ONLY lp-preview + lp-mind. It does NOT re-enable the
# parked lp-director / lp-player (the old Track-1 show) -- that is exactly why the original
# blanket lp-watchdog stays disabled. It also RESPECTS a deliberate disable: a task whose
# State is Disabled is left alone (that is how you turn the show off without the watchdog
# fighting you). To stop the watchdog itself: Disable-ScheduledTask lp-watchdog-preview.
#
# IDEMPOTENT: starts a task only when its process is absent; a healthy show is a no-op.
# Tasks are MultipleInstancesPolicy=IgnoreNew, so a Start during a start-race is ignored.
# Runs headless (S4U principal) so it never flashes a window onto the kiosk panels.

$ErrorActionPreference = 'SilentlyContinue'
$log = 'C:\living-portraits\_watchdog.log'

function Log($m) {
    "{0} {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Append -Encoding utf8
}

# task name -> the substring that identifies ITS pythonw process by command line
$watch = [ordered]@{
    'lp-preview' = '_preview_graph.py'   # the player (paints the panels) -- critical
    'lp-mind'    = 'heartbeat.py'         # the GLM brain (writes goals) -- degrades gracefully if down
}

$pythonw = Get-CimInstance Win32_Process -Filter "Name='pythonw.exe'" -ErrorAction SilentlyContinue

foreach ($task in $watch.Keys) {
    $needle = $watch[$task]
    $t = Get-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue
    if (-not $t) { continue }                       # task not installed on this host
    if ($t.State -eq 'Disabled') { continue }       # deliberately turned off -> leave it
    $alive = $pythonw | Where-Object { $_.CommandLine -like "*$needle*" }
    if (-not $alive) {
        Log "$task DOWN (no '$needle' process; task State=$($t.State)) -> Start-ScheduledTask"
        Start-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue
    }
}
