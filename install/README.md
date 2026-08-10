# living-portraits -- self-heal watchdog (install/)

Makes the show **durable** on supercommons2 (`immer@100.123.185.12`): the two
pythonw processes that drive the panels come back by themselves after a crash,
a sign-out/in, or a reboot, and a 5-minute watchdog restarts whichever one dies
between reboots.

This folder is built LOCALLY and deployed BY THE CONDUCTOR. Do not run it from a
dev machine -- it registers scheduled tasks for the `immer` user on SC2.

## What it manages

| Task | Trigger | Command (run-as `immer`, interactive) |
|---|---|---|
| `lp-player` | **at logon** | `.venv\Scripts\pythonw.exe player.py` |
| `lp-director` | **at logon** | `.venv\Scripts\pythonw.exe director\stage_manager.py --loop 45` |
| `lp-watchdog` | **every 5 min** | `powershell.exe ... install\watchdog.ps1` |

Both show tasks run **pythonw** (no console window) and are **interactive**
(`LogonType Interactive` == schtasks `/IT`), so the borderless window lands on
console session 1 where player.py's `HWND_TOPMOST` pin has a desktop to attach
to. They have **no execution time limit** (`ExecutionTimeLimit = 0`) -- they are
meant to run forever; the Windows default 3-day kill would otherwise end the
show.

## How the watchdog decides to restart

living-portraits serves no HTTP, so the watchdog probes by **process existence**,
not by a `/healthz` port:

- It queries `Win32_Process` (CIM) for `pythonw.exe` / `python.exe` and matches
  each one's **command line** -- `player.py` for the player, `stage_manager.py`
  for the director. (Both are the same `.venv` interpreter; the command line is
  what tells them apart.)
- If a process is **absent**, the watchdog runs `schtasks /Run /TN <task>` to
  relaunch it, then polls for up to 15s to confirm it reappeared.
- If a process is **alive**, the watchdog leaves it completely alone. It never
  kills a living process. (The director legitimately **pauses its beat loop
  during portrait renders** to free VRAM, so a frozen `stage_state.json` with a
  live director is normal, not a fault.)
- `stage_state.json` mtime is logged as an **advisory** only (flags an
  hours-frozen file so a human can tell a wedged loop from a render pause). It
  never triggers a restart.

Everything is idempotent: running the watchdog against a healthy show is a no-op.

## Deploy (conductor runs these from the life repo root)

Two files go to the host: `watchdog.ps1` and `install_tasks.ps1`. They install
into `C:\Users\immer\living-portraits\install\` (the installer derives the
project root as the parent of its own folder).

```powershell
# 0. Dot-source the remote-PS helper (handles BOM + base64 EncodedCommand).
. C:\Users\jtole\Documents\2026\life\scripts\invoke-remote-ps.ps1

# 1. Make sure the install dir exists on SC2.
Invoke-RemotePS -Host_ immer@100.123.185.12 -Script @'
New-Item -ItemType Directory -Force -Path C:\Users\immer\living-portraits\install | Out-Null
"install dir ready"
'@

# 2. scp the two files (straight scp; they are pure ASCII, no BOM).
scp C:\Users\jtole\Documents\2026\life\projects\living-portraits\install\watchdog.ps1 `
    immer@100.123.185.12:C:/Users/immer/living-portraits/install/watchdog.ps1
scp C:\Users\jtole\Documents\2026\life\projects\living-portraits\install\install_tasks.ps1 `
    immer@100.123.185.12:C:/Users/immer/living-portraits/install/install_tasks.ps1

# 3. Run the installer on SC2. It registers all three tasks, smoke-runs the
#    watchdog once, and prints the final task table.
Invoke-RemotePS -Host_ immer@100.123.185.12 -Script @'
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\Users\immer\living-portraits\install\install_tasks.ps1
'@
```

`Send-And-RemotePS` (also in `invoke-remote-ps.ps1`) can collapse a copy + run
into one call per file if preferred; the explicit form above is clearer for a
two-file deploy.

### How at-logon differs from the current `/SC ONCE /IT` setup

The player and director are currently registered with `/SC ONCE /IT` and started
by hand with `schtasks /Run`. `ONCE` fires a single time and never again -- after
a reboot or sign-out the panels stay dark until a human re-runs them.
`install_tasks.ps1` re-registers both tasks (with `-Force`, so it replaces the
ONCE versions in place) using an **at-logon** trigger: the show now relaunches by
itself whenever `immer` signs in (effectively at boot on this auto-logon box). The
5-minute `lp-watchdog` covers the gap between reboots -- a mid-session crash is
healed within 5 minutes without waiting for the next logon.

## Verify (non-destructive)

```powershell
. C:\Users\jtole\Documents\2026\life\scripts\invoke-remote-ps.ps1
Invoke-RemotePS -Host_ immer@100.123.185.12 -Script @'
# 1. All three tasks exist with the right triggers + interactive principal.
Get-ScheduledTask -TaskName lp-player, lp-director, lp-watchdog |
  Select-Object TaskName, State,
    @{N="Trigger";  E={ ($_.Triggers   | ForEach-Object { $_.CimClass.CimClassName }) -join "," }},
    @{N="LogonType";E={ $_.Principal.LogonType }} | Format-Table -AutoSize

# 2. Both show processes are actually alive (matched by command line).
Get-CimInstance Win32_Process -Filter "Name = '"'"'pythonw.exe'"'"'" |
  Where-Object { $_.CommandLine -match "player\.py|stage_manager\.py" } |
  Select-Object ProcessId, CommandLine | Format-List

# 3. Watchdog has logged a clean run.
Get-Content C:\Users\immer\living-portraits\watchdog.log -Tail 10
'@
```

Expect: `lp-player` + `lp-director` show a `...LogonTrigger`, `LogonType =
Interactive`; `lp-watchdog` shows a `...TimeTrigger`; two pythonw processes
appear (one per script); and the log tail shows either an hourly heartbeat or a
RESTART line.

## Smoke test (destructive -- proves self-heal)

Kill the player and confirm the watchdog restores it within one 5-minute cycle.
The fastest proof is to kill it and then fire the watchdog manually (no need to
wait the full 5 minutes):

```powershell
. C:\Users\jtole\Documents\2026\life\scripts\invoke-remote-ps.ps1
Invoke-RemotePS -Host_ immer@100.123.185.12 -Script @'
# 1. Kill the player process (find it by command line, stop that PID).
$p = Get-CimInstance Win32_Process -Filter "Name = '"'"'pythonw.exe'"'"'" |
     Where-Object { $_.CommandLine -match "player\.py" } | Select-Object -First 1
if ($p) { Stop-Process -Id $p.ProcessId -Force; "killed player PID $($p.ProcessId)" } else { "player not running" }
Start-Sleep -Seconds 2

# 2. Confirm it is gone.
$gone = -not (Get-CimInstance Win32_Process -Filter "Name = '"'"'pythonw.exe'"'"'" |
        Where-Object { $_.CommandLine -match "player\.py" })
"player gone after kill: $gone"

# 3. Fire the watchdog now (the same thing the 5-min task does).
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\Users\immer\living-portraits\install\watchdog.ps1

# 4. Confirm the player is back.
Start-Sleep -Seconds 2
$back = [bool](Get-CimInstance Win32_Process -Filter "Name = '"'"'pythonw.exe'"'"'" |
        Where-Object { $_.CommandLine -match "player\.py" })
"player restored: $back"
'@
```

PASS = step 2 prints `gone after kill: True`, the watchdog logs a `RESTART
lp-player` line, and step 4 prints `player restored: True`. To prove the
unattended path instead of firing the watchdog by hand, just kill the player and
wait <= 5 minutes, then re-check step 4 -- the scheduled `lp-watchdog` tick will
have restored it.

To prove **reboot survival**: reboot SC2, let `immer` auto-logon settle, then run
the Verify block -- both pythonw processes should be present with no manual
`schtasks /Run`.

## Files

- `watchdog.ps1` -- the 5-minute probe + restart (process-existence; no port).
- `install_tasks.ps1` -- idempotent task registration (run once; re-runnable).
- `README.md` -- this file.

## Notes / pitfalls

- ASCII only, straight quotes, `--` not em-dashes throughout -- these files scp
  to a Windows-codepage host and any smart punctuation would break the PS parser.
- The watchdog writes `watchdog.log` and `.watchdog-tick` into the project root;
  both are covered by the project `.gitignore` (`*.log` + the dotfile is
  harmless).
- Do not `Start-Process` the show from an SSH session -- children die when SSH
  closes. The restart path is always `schtasks /Run`, whose tasks live in their
  own interactive session. (Same pitfall the `funneled-service` skill documents.)
- The watchdog will not start the show on a machine where nobody is signed in
  (an interactive task needs an interactive session). On SC2 with auto-logon this
  is a non-issue; if SC2 ever sits at the lock screen, the panels wait for logon.
