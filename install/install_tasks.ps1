# install_tasks.ps1 -- run ONCE on supercommons2 to make the living-portraits
# show durable. Re-runnable: every task is registered with -Force, so a second
# run updates in place and never duplicates triggers or tasks.
#
# Registers three scheduled tasks (all run-as the interactive 'immer' user):
#
#   lp-player    AT LOGON, INTERACTIVE  -- .venv\Scripts\pythonw.exe player.py
#                The borderless pinned dual-panel stage. MUST be interactive so
#                the window lands on console session 1 and the HWND_TOPMOST pin
#                in player.py has a desktop to attach to.
#   lp-director  AT LOGON, INTERACTIVE  -- .venv\Scripts\pythonw.exe director\stage_manager.py --loop 45
#                The qwen3 beat loop. Interactive too (shares the session, no
#                console window because it is pythonw).
#   lp-watchdog  EVERY 5 MIN            -- powershell.exe ... watchdog.ps1
#                Probes both processes by command line; restarts whichever is
#                absent. Does NOT need interactive -- it only probes + triggers.
#
# WHY AT-LOGON (vs the current /SC ONCE /IT + manual /Run): an at-logon trigger
# means the whole show comes back by itself after a reboot or a sign-out/in,
# with zero human action. The current ONCE tasks fire one time and then never
# again, so a reboot leaves the panels dark until someone manually re-runs them.
# pythonw.exe (not python.exe) keeps it windowless -- no stray console on the
# desktop that the LED card would grab.

$ErrorActionPreference = 'Stop'

# Resolve the project root from this script's location (install\ under project),
# with a hard fallback to the canonical SC2 path the conductor deploys to.
$PROJECT = Split-Path -Parent $PSScriptRoot
if (-not $PROJECT -or -not (Test-Path $PROJECT)) { $PROJECT = 'C:\Users\immer\living-portraits' }

$RUN_USER     = 'immer'
$PYTHONW      = Join-Path $PROJECT '.venv\Scripts\pythonw.exe'
$PLAYER_PY    = 'player.py'                          # relative to -WorkingDirectory
$DIRECTOR_PY  = 'director\stage_manager.py'          # relative to -WorkingDirectory
$WATCHDOG_PS1 = Join-Path $PSScriptRoot 'watchdog.ps1'

$PLAYER_TASK   = 'lp-player'
$DIRECTOR_TASK = 'lp-director'
$WATCH_TASK    = 'lp-watchdog'

Write-Host "Project root : $PROJECT"
Write-Host "pythonw      : $PYTHONW"
Write-Host "watchdog.ps1 : $WATCHDOG_PS1"
Write-Host ""

# Preflight: the interpreter and watchdog must actually exist where we point at
# them, or the tasks would register but fail silently at trigger time.
if (-not (Test-Path $PYTHONW)) {
    throw "pythonw.exe not found at $PYTHONW -- is the .venv created on this host? (expected $PROJECT\.venv)"
}
if (-not (Test-Path $WATCHDOG_PS1)) {
    throw "watchdog.ps1 not found at $WATCHDOG_PS1 -- scp the install\ folder to the host before running this."
}

# Interactive principal: LogonType Interactive == the schtasks '/IT' flag. This
# is what lands the task in the signed-in desktop session (console session 1).
# RunLevel Limited: the show does not need admin; the player just pins a window.
$interactivePrincipal = New-ScheduledTaskPrincipal `
    -UserId $RUN_USER -LogonType Interactive -RunLevel Limited

# At-logon trigger for the show processes (fires for the immer user at sign-in,
# which on this single-user box is effectively at boot once auto-logon settles).
$logonTrigger = New-ScheduledTaskTrigger -AtLogOn -User $RUN_USER

# Settings shared by the two show tasks: let them start if the box is on
# battery, allow start when available (covers a missed trigger after downtime),
# and -- crucially -- DO NOT impose an execution time limit (these are meant to
# run forever; the default 3-day kill would murder the show).
$showSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 0

# ---- lp-player (at logon, interactive) ----
$playerAction = New-ScheduledTaskAction `
    -Execute $PYTHONW -Argument $PLAYER_PY -WorkingDirectory $PROJECT
Register-ScheduledTask `
    -TaskName $PLAYER_TASK `
    -Action $playerAction `
    -Trigger $logonTrigger `
    -Principal $interactivePrincipal `
    -Settings $showSettings `
    -Force | Out-Null
Write-Host "[1/3] $PLAYER_TASK registered (at-logon, interactive) -- $PYTHONW $PLAYER_PY"

# ---- lp-director (at logon, interactive) ----
# Pass the script path and its flags as a single -Argument string so the loop
# interval survives intact.
$directorAction = New-ScheduledTaskAction `
    -Execute $PYTHONW -Argument "$DIRECTOR_PY --loop 45" -WorkingDirectory $PROJECT
Register-ScheduledTask `
    -TaskName $DIRECTOR_TASK `
    -Action $directorAction `
    -Trigger $logonTrigger `
    -Principal $interactivePrincipal `
    -Settings $showSettings `
    -Force | Out-Null
Write-Host "[2/3] $DIRECTOR_TASK registered (at-logon, interactive) -- $PYTHONW $DIRECTOR_PY --loop 45"

# ---- lp-watchdog (every 5 min) ----
# A one-time trigger 'now' with a 5-minute repetition interval and an indefinite
# duration is the durable way to express "every 5 minutes forever" with
# New-ScheduledTaskTrigger. Runs Limited (non-interactive is fine -- it only
# probes processes and triggers the launch tasks).
$watchAction = New-ScheduledTaskAction `
    -Execute 'powershell.exe' `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$WATCHDOG_PS1`"" `
    -WorkingDirectory $PROJECT
$watchTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date) `
    -RepetitionInterval (New-TimeSpan -Minutes 5)
$watchPrincipal = New-ScheduledTaskPrincipal `
    -UserId $RUN_USER -LogonType Interactive -RunLevel Limited
$watchSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 4)
Register-ScheduledTask `
    -TaskName $WATCH_TASK `
    -Action $watchAction `
    -Trigger $watchTrigger `
    -Principal $watchPrincipal `
    -Settings $watchSettings `
    -Force | Out-Null
Write-Host "[3/3] $WATCH_TASK registered (every 5 min) -- runs watchdog.ps1"

# Smoke-run the watchdog once. With the show already running this is a no-op
# (it will not restart a live process); if a process happens to be down it will
# be restored right here, which is exactly the behavior we want on first install.
Write-Host ""
Write-Host "Smoke-running watchdog once..."
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $WATCHDOG_PS1
Write-Host ""

# Final state print for all three tasks, including each task's trigger types so
# the operator can confirm the at-logon wiring stuck.
Write-Host "Final task states:"
Get-ScheduledTask -TaskName $PLAYER_TASK, $DIRECTOR_TASK, $WATCH_TASK |
    Select-Object TaskName, State,
        @{N='Triggers'; E={ ($_.Triggers | ForEach-Object { $_.CimClass.CimClassName -replace 'MSFT_Task','' -replace 'Trigger','' }) -join ',' }},
        @{N='RunAs';    E={ $_.Principal.UserId }},
        @{N='LogonType';E={ $_.Principal.LogonType }} |
    Format-Table -AutoSize
