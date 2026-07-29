@echo off
REM Dataset Deviser - one-time setup (Windows). Safe to re-run: use it to
REM switch an existing install between the CPU-only and NVIDIA-GPU PyTorch builds,
REM to install dependencies added by a `git pull`, or to change an API key.
REM
REM MAINTAINER NOTE - two rules this script exists to obey:
REM  1) Never put an unescaped ( or ) inside an `if ... ( ... )` block. cmd ends the
REM     block at the first bare `)`, and the script dies with "X was unexpected at
REM     this time". That bug shipped once and closed the window on new users mid-way
REM     through setup. This file therefore uses `if not X goto :label` instead of
REM     parenthesized blocks almost everywhere.
REM  2) Never exit without pausing. Double-clicked, the window closes instantly and
REM     the user sees nothing at all. Every failure path goes through :die.
setlocal
cd /d "%~dp0"

set "EXITCODE=1"

echo === Dataset Deviser setup ===
echo.

REM --- Python present? ---
python --version >nul 2>&1
if errorlevel 1 goto :nopython

REM --- Python new enough? Ask Python itself rather than parsing the version in
REM     batch, which is where this kind of check usually goes wrong.
python -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 goto :oldpython
for /f "tokens=2" %%v in ('python --version 2^>^&1') do echo Found Python %%v
python -c "import sys; sys.exit(0 if sys.version_info[:2] <= (3, 13) else 1)" >nul 2>&1
if not errorlevel 1 goto :pyok
echo [note] That is newer than the versions this project is tested against (3.10-3.13).
echo        It usually works, but if pip cannot find a torch or onnxruntime wheel,
echo        install Python 3.13 instead, delete the .venv folder, and re-run setup.
echo.
:pyok

REM --- venv ---
if exist .venv\Scripts\python.exe goto :venvok
echo Creating virtual environment...
python -m venv .venv
if errorlevel 1 goto :venvfail
:venvok
set "PIP=.venv\Scripts\pip.exe"
set "PY=.venv\Scripts\python.exe"

REM --- choose the compute build (GPU-optimized vs CPU-only) ---
set "GPU_DEFAULT=2"
where nvidia-smi >nul 2>&1 && set "GPU_DEFAULT=1"
echo.
echo Choose your PyTorch / ONNX Runtime build:
echo    [1] NVIDIA GPU (CUDA) - fast local generation, captioning and isolation
echo    [2] CPU only          - no NVIDIA GPU; use the cloud options for heavy stages
if "%GPU_DEFAULT%"=="1" echo    (NVIDIA GPU detected - [1] recommended)
if "%GPU_DEFAULT%"=="2" echo    (No NVIDIA GPU detected - [2] recommended)
set "CHOICE="
set /p CHOICE="Enter 1 or 2 [default %GPU_DEFAULT%]: "
if not defined CHOICE set "CHOICE=%GPU_DEFAULT%"
set "WANT=cpu"
if "%CHOICE%"=="1" set "WANT=gpu"

REM --- what is installed now? (so a re-run can switch builds) ---
set "CURRENT=none"
%PY% -c "import torch,sys; sys.stdout.write('gpu' if torch.version.cuda else 'cpu')" > "%TEMP%\lds_torch.txt" 2>nul
if not exist "%TEMP%\lds_torch.txt" goto :nocurrent
set /p CURRENT=<"%TEMP%\lds_torch.txt"
del "%TEMP%\lds_torch.txt" >nul 2>&1
:nocurrent
if "%CURRENT%"=="" set "CURRENT=none"

if "%CURRENT%"=="%WANT%" goto :torchok
if "%CURRENT%"=="none" goto :torchinstall
echo Switching PyTorch from %CURRENT% to %WANT% - reinstalling...
%PIP% uninstall -y torch torchvision >nul 2>&1
:torchinstall
if "%WANT%"=="cpu" goto :torchcpu
echo Installing CUDA build of PyTorch...
%PIP% install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :pipfail
goto :torchok
:torchcpu
echo Installing CPU build of PyTorch.
echo NOTE: local captioning/isolation models are very slow on CPU.
echo       Cloud captioners (Gemini/Groq) work fine without a GPU.
%PIP% install torch torchvision
if errorlevel 1 goto :pipfail
:torchok

echo.
echo Installing dependencies...
%PIP% install -r requirements.txt
if errorlevel 1 goto :pipfail

REM --- ONNX Runtime for the WD/e621 taggers (3), matched to the chosen build ---
REM Kept out of requirements.txt so the CPU vs CUDA variant tracks your choice.
echo.
echo Installing ONNX Runtime (%WANT%) for the taggers...
%PIP% uninstall -y onnxruntime onnxruntime-gpu >nul 2>&1
if "%WANT%"=="cpu" goto :onnxcpu
%PIP% install onnxruntime-gpu
if errorlevel 1 echo [warn] onnxruntime-gpu failed - the taggers can still use the CPU build: pip install onnxruntime
goto :onnxdone
:onnxcpu
%PIP% install onnxruntime
if errorlevel 1 echo [warn] onnxruntime failed - the taggers need it; install it later with: pip install onnxruntime
:onnxdone

REM --- optional API keys -> .env ---
REM Handled by cli.py, NOT here: batch could not update a key that was already set
REM (so a typo was unfixable), echoed keys in the clear, and broke .env files that
REM had no trailing newline. One tested Python implementation instead.
echo.
%PY% cli.py keys --setup
if errorlevel 1 echo [warn] Key setup was interrupted - run "python cli.py keys" any time to set them.

REM --- final report: dependencies, keys, optional backends ---
echo.
echo === Checking the finished install ===
%PY% cli.py doctor
if errorlevel 1 goto :doctorfail

echo.
echo === Setup complete (%WANT% build). Run start.bat to launch. ===
echo     Re-run setup.bat any time to switch GPU/CPU builds or install new
echo     dependencies after a git pull. Change an API key with:
echo         .venv\Scripts\python.exe cli.py keys
set "EXITCODE=0"
goto :done

REM ---------------------------------------------------------------- failures
:nopython
echo [ERROR] Python was not found on PATH.
echo         Install Python 3.10-3.13 from https://www.python.org/downloads/ and
echo         tick "Add python.exe to PATH" in the installer, then re-run setup.bat.
echo         If typing "python" opens the Microsoft Store, that is a placeholder:
echo         install the real Python, or turn off the App Execution Alias for it.
goto :die

:oldpython
for /f "tokens=2" %%v in ('python --version 2^>^&1') do echo [ERROR] Found Python %%v - too old.
echo         This project needs Python 3.10 or newer (3.10-3.13 are tested).
echo         Install a newer Python, delete the .venv folder, then re-run setup.bat.
goto :die

:venvfail
echo [ERROR] Could not create the virtual environment in .venv
echo         Check you can write to this folder, that antivirus is not blocking it,
echo         and that "python -m venv" works. Delete any partial .venv and retry.
goto :die

:pipfail
echo [ERROR] A pip install failed - see the output above for the real reason.
echo         Common causes: no internet or a proxy blocking PyPI; a disk that is
echo         full; or no wheel for your Python version. Re-run setup.bat to retry.
goto :die

:doctorfail
echo [ERROR] The install check above found a problem. Fix what it lists, then
echo         re-run setup.bat. You can re-run the check on its own with:
echo             .venv\Scripts\python.exe cli.py doctor
goto :die

:die
echo.
echo Setup did NOT complete.
:done
echo.
REM Keeps the window open when this was launched by double-clicking, so the
REM message above is actually readable. This is the whole reason :die exists.
pause
endlocal & exit /b %EXITCODE%
