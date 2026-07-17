# start_portraits.ps1 -- one-click launcher for the Living Portraits show on desktop-hil08rd.
#
# Double-click "Start Living Portraits" on the Desktop. Brings the whole live stack back up:
#   OllamaServe          local qwen3 (Track-1 brain; idle unless the spoken asides are on)
#   lp-mind              the GLM heartbeat -- picks each character's next pose
#   lp-preview           the PLAYER that paints the two LED panels
#   lp-watchdog-preview  self-heal -- restarts the above if they exit cleanly
#
# Best-effort: enables a task if it was turned off, then starts it. No admin needed in the
# normal case (tasks are Ready); if a task was hard-Disabled, right-click -> Run as administrator.
param([switch]$NoPause)

$ErrorActionPreference = 'SilentlyContinue'
try { $Host.UI.RawUI.WindowTitle = "Living Portraits Launcher" } catch {}
function Line($m, $c = 'Gray') { Write-Host $m -ForegroundColor $c }

Line "===============================================" Cyan
Line "   Living Portraits  --  launching the show"      Cyan
Line "===============================================" Cyan
Line ""

$tasks = @(
    @{ n = 'OllamaServe';         d = "local qwen3 brain (idle unless asides are on)" },
    @{ n = 'lp-mind';             d = "GLM heartbeat -- chooses each pose" },
    @{ n = 'lp-preview';          d = "the player -- paints the panels" },
    @{ n = 'lp-watchdog-preview'; d = "self-heal -- restarts the show if it stops" }
)

foreach ($t in $tasks) {
    $task = Get-ScheduledTask -TaskName $t.n -ErrorAction SilentlyContinue
    if (-not $task) { Line ("  [skip ] {0,-20} not installed on this PC" -f $t.n) DarkYellow; continue }
    if ($task.State -eq 'Disabled') { Enable-ScheduledTask -TaskName $t.n | Out-Null }
    Start-ScheduledTask -TaskName $t.n
    Line ("  [start] {0,-20} {1}" -f $t.n, $t.d) Green
}

Line ""
Line "Waiting for the panels to come up..." Cyan
Start-Sleep -Seconds 7

Line ""
Line "=== status ===" Cyan
foreach ($t in $tasks) {
    $s = (Get-ScheduledTask -TaskName $t.n -ErrorAction SilentlyContinue).State
    Line ("  {0,-22} {1}" -f $t.n, $s)
}
$pv = Get-CimInstance Win32_Process -Filter "Name='pythonw.exe'" -ErrorAction SilentlyContinue |
      Where-Object { $_.CommandLine -like "*_preview_graph.py*" }
Line ""
if ($pv) {
    Line "  PANELS ARE PAINTING -- the portraits are live." Green
} else {
    Line "  Panels not detected yet. Wait a few seconds; if still blank, a fullscreen" Yellow
    Line "  app may be covering them (close it), or re-run this As Administrator."      Yellow
}
Line ""
Line "You can close this window." Cyan

if (-not $NoPause) {
    Line "Press any key to close..."
    try { $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") } catch { Start-Sleep -Seconds 4 }
}
