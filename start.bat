@echo off
REM LoRA Dataset Studio - launch the UI (run setup.bat once first).
REM
REM MAINTAINER NOTE: this script must never exit without pausing. Double-clicked, a
REM bare `exit /b 1` closes the window instantly, so a crash looked to users like
REM "start.bat did nothing" — most often an ImportError after a `git pull` added a
REM dependency, which the message below names explicitly.
setlocal
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe goto :novenv

.venv\Scripts\python.exe app.py
if errorlevel 1 goto :appfailed
endlocal & exit /b 0

:novenv
echo [ERROR] No virtual environment found (.venv is missing).
echo         Run setup.bat first - it installs everything this needs.
goto :die

:appfailed
echo.
echo [ERROR] LoRA Dataset Studio exited with an error. The cause is in the output
echo         above - scroll up to read it.
echo.
echo   Most common cause: you just ran "git pull" and the new version needs a
echo   dependency you do not have yet. Fix it by re-running:
echo         setup.bat
echo.
echo   For a full check of this install (Python, dependencies, API keys, ComfyUI):
echo         .venv\Scripts\python.exe cli.py doctor
goto :die

:die
echo.
pause
endlocal & exit /b 1
