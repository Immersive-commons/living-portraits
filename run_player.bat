@echo off
REM Runs from wherever this file lives, so the same script works on any box. It used to
REM hardcode C:\Users\immer\living-portraits -- the pre-migration supercommons2 path -- which
REM has been wrong on the production host since 2026-05-26 and silently so, because the
REM scheduled task supplies its own WorkingDirectory and never invoked this file.
cd /d "%~dp0"
.venv\Scripts\python.exe player.py > player.log 2>&1
