@echo off
REM setup.bat - One-click setup for AIFlux (run from project root)
REM Usage: open cmd.exe, cd /d E:\flux, run setup.bat

SETLOCAL ENABLEDELAYEDEXPANSION

REM Ensure script runs from its directory (project root)
cd /d "%~dp0"

echo ===========================================================
echo AIFlux One-Click Setup
echo Project root: %CD%
echo ===========================================================

REM If a .venv exists under scripts (common mistake), remove it to avoid confusion
if exist "%CD%\scripts\.venv" (
  echo Found scripts\.venv (likely created from running script from scripts folder). Removing to avoid confusion...
  rmdir /S /Q "%CD%\scripts\.venv"
)

REM Create virtual environment in project root if missing
if not exist "%CD%\.venv\Scripts\activate.bat" (
  echo Creating virtual environment in %CD%\.venv ...
  python -m venv .venv
  if errorlevel 1 (
    echo Failed to create virtualenv. Ensure Python 3.10+ is on PATH.
    pause
    exit /b 1
  )
) else (
  echo Virtual environment .venv already exists.
)

REM Activate venv (cmd)
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
  echo Failed to activate virtualenv. Aborting.
  pause
  exit /b 1
)

echo Upgrading pip, setuptools and wheel...
python -m pip install --upgrade pip setuptools wheel

REM Run environment diagnostic (creates logs\env_report.json)
if exist tools\env_check.py (
  echo Running environment diagnostic (tools\env_check.py)...
  python tools\env_check.py
) else (
  echo Warning: tools\env_check.py not found.
)

REM Attempt to find a local torch wheel in project root or wheels\ folder
set "TORCH_WHL="
for %%F in (torch-*.whl) do if not defined TORCH_WHL set "TORCH_WHL=%%~fF"
if not defined TORCH_WHL (
  if exist wheels (
    for %%F in (wheels\torch-*.whl) do if not defined TORCH_WHL set "TORCH_WHL=%%~fF"
  )
)

if defined TORCH_WHL (
  echo Found local torch wheel: %TORCH_WHL%
  echo Installing local torch wheel (this may take a few minutes)...
  pip install "%TORCH_WHL%"
  if errorlevel 1 (
    echo Local wheel installation failed. Will try index-based installation next.
    set "TORCH_WHL="
  )
) 

REM If no local wheel installed, install torch from official index (cu121 recommended)
if not defined TORCH_WHL (
  echo Installing PyTorch (recommended build: CUDA 12.1 / cu121)...
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  if errorlevel 1 (
    echo Automatic torch installation failed. Trying CPU-only wheel as fallback...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    if errorlevel 1 (
      echo Failed to install torch. Inspect logs\env_report.json and fix manually.
      pause
      exit /b 1
    )
  )
)

REM Install remaining Python requirements
echo Installing project Python requirements (may skip torch if already satisfied)...
python -m pip install -r requirements.txt || (
  echo Some packages failed to install. You can continue but optional packages (xformers, onnxruntime-gpu) may need manual installation.
)

REM Best-effort install onnxruntime-gpu
echo Attempting to install onnxruntime-gpu (or fallback to onnxruntime)...
python -m pip install onnxruntime-gpu || python -m pip install onnxruntime

REM Ensure models path exists and set environment variable for current user
if exist "E:\flux\model" (
  echo Setting AI_STUDIO_MODELS_PATH to E:\flux\model (permanent for this user)...
  setx AI_STUDIO_MODELS_PATH "E:\flux\model" > nul
) else if exist "%CD%\models" (
  echo Setting AI_STUDIO_MODELS_PATH to %CD%\models (permanent for this user)...
  setx AI_STUDIO_MODELS_PATH "%CD%\models" > nul
) else (
  echo No models folder found at E:\flux\model or %CD%\models. Please place your models in one of these locations or set AI_STUDIO_MODELS_PATH manually.
)

REM Final sanity check for torch+cuda
echo Verifying torch and GPU availability...
python - <<"PY"
import sys
try:
    import torch
    print('torch', torch.__version__)
    print('cuda_available', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('devices', torch.cuda.device_count())
except Exception as e:
    print('torch_not_ok', str(e))
PY

REM Launch application
echo Launching AIFlux (launcher.py)...
python launcher.py

ENDLOCAL
exit /b 0
