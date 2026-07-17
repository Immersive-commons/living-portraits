# sera_build.ps1 -- detached one-shot on SC2: render Seraphina (SD1.5) + segment/rig,
# then resume the director and restart the player so Panel B picks up her new assets.
# Run via an interactive scheduled task (lp-sera). Survives SSH drops; writes gen_sera.done.
$ErrorActionPreference = "Continue"
Set-Location C:\Users\immer\living-portraits
Remove-Item gen_sera.done -ErrorAction SilentlyContinue

Write-Output "=== pause director + unload qwen3 (free GPU for SD1.5) ==="
Get-CimInstance Win32_Process -Filter "Name='pythonw.exe'" |
    Where-Object { $_.CommandLine -like '*stage_manager.py*' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
try { ollama stop qwen3:8b } catch {}
Start-Sleep -Seconds 2

Write-Output "=== render seraphina portrait (SD1.5, .venv-gen) ==="
& .\.venv-gen\Scripts\python.exe pipeline\generate.py seraphina *> gen_sera.log

Write-Output "=== segment + rig spec (.venv / cv2) ==="
& .\.venv\Scripts\python.exe pipeline\orchestrate.py seraphina *>> gen_sera.log

Write-Output "=== resume director + restart player (pick up new assets past the negative cache) ==="
schtasks /Run /TN lp-director | Out-Null
Get-CimInstance Win32_Process -Filter "Name='pythonw.exe'" |
    Where-Object { $_.CommandLine -like '*player.py*' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Start-Sleep -Seconds 1
schtasks /Run /TN lp-player | Out-Null

"DONE" | Out-File -Encoding ascii gen_sera.done
Write-Output "=== seraphina assets in data/gen: ==="
Get-ChildItem data\gen -Filter "seraphina_*" | Select-Object Name, Length
