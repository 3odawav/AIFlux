@echo off
REM AIFlux - Full setup and run script for Windows (cmd.exe)
REM Place this file in the repo root and run it from cmd (not PowerShell):
REM   setup_all.bat

SETLOCAL ENABLEDELAYEDEXPANSION

echo ==================================================================
echo AIFlux - Automated Setup (Windows CMD)
echo ==================================================================

REM 1) Check Python version (must be >= 3.10)
echo [1/10] Checking Python version...
python -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}'); sys.exit(0 if (v.major>3 or (v.major==3 and v.minor>=10)) else 1)" > nul 2>&1
if errorlevel 1 (
  echo ERROR: Python 3.10+ is required and was not found on PATH.
  echo Please install Python 3.10 or 3.11 and re-run this script.
  pause
  exit /b 1
) else (
  for /f "delims=" %%p in ('python -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}')"') do set PYVER=%%p
  echo Found Python version %PYVER%
)

REM 2) Create virtual environment if it doesn't exist
if not exist ".venv\Scripts\activate.bat" (
  echo [2/10] Creating virtual environment (.venv)...
  python -m venv .venv
  if errorlevel 1 (
    echo Failed to create virtualenv. Aborting.
    pause
    exit /b 1
  )
) else (
  echo [2/10] Virtual environment already exists.
)

REM 3) Activate the virtualenv (cmd.exe activate)
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
  echo Failed to activate virtualenv. Make sure you're running this script from the repo root.
  pause
  exit /b 1
)

echo [3/10] Upgrading pip, setuptools, wheel...
python -m pip install --upgrade pip setuptools wheel

REM 4) Run environment diagnostic (tools/env_check.py)
if not exist tools\env_check.py (
  echo Missing tools\env_check.py - ensure you are in the repository root.
  pause
  exit /b 1
)

echo [4/10] Running environment diagnostic (tools/env_check.py)...
python tools\env_check.py
if errorlevel 1 (
  echo Diagnostic script failed (non-zero exit). Continue anyway? (Y/N)
  set /p cont=Continue? [Y/N]:
  if /i "%cont%" NEQ "Y" (
    echo Aborting.
    pause
    exit /b 1
  )
)

echo Diagnostic saved to logs\env_report.json

REM 5) Decide PyTorch wheel (auto-detect or override)
set TORCH_CHOICE=
for /f "delims=" %%A in ('python - <<"PY"
import json,sys
try:
    r=json.load(open('logs/env_report.json','r',encoding='utf-8'))
except Exception:
    print('NO_REPORT')
    sys.exit(0)
# If torch already installed and cuda available -> SKIP
if r.get('torch',{}).get('installed') and r.get('torch',{}).get('cuda_available'):
    print('SKIP')
    sys.exit(0)
# If nvidia-smi present try to choose cu121 by default (modern safe pick)
if r.get('nvidia-smi',{}).get('ok'):
    print('cu121')
else:
    print('cpu')
PY
') do set TORCH_CHOICE=%%A

if "%TORCH_CHOICE%"=="" set TORCH_CHOICE=cu121
if "%TORCH_CHOICE%"=="SKIP" (
  echo [5/10] PyTorch detected with CUDA available — skipping torch install.
) else (
  echo [5/10] Recommended torch build: %TORCH_CHOICE%
  echo Installing PyTorch (%TORCH_CHOICE%) - this may take several minutes...
  if "%TORCH_CHOICE%"=="cpu" (
    pip install "torch" "torchvision" "torchaudio" --index-url https://download.pytorch.org/whl/cpu
  ) else (
    pip install "torch" "torchvision" "torchaudio" --index-url https://download.pytorch.org/whl/%TORCH_CHOICE%
  )
  if errorlevel 1 (
    echo Failed to install torch using the automatic wheel. You can manually install a suitable wheel.
    echo See README and logs\env_report.json for guidance.
    pause
  ) else (
    echo PyTorch installed.
  )
)

REM 6) Install remaining Python requirements (requirements.txt excludes torch in our setup)
echo [6/10] Installing Python requirements (excluding torch)...
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo There were errors installing requirements. Some optional packages (xformers, onnxruntime-gpu) may fail to build on Windows.
  echo Continue and fix them later? (Y/N)
  set /p cont=Continue? [Y/N]:
  if /i "%cont%" NEQ "Y" (
    echo Aborting for manual fix.
    pause
    exit /b 1
  )
)

REM 7) Try to install onnxruntime-gpu (best-effort)
echo [7/10] Installing onnxruntime-gpu (best effort)...
python -m pip install onnxruntime-gpu || python -m pip install onnxruntime

REM 8) Check models folder and set environment variable
set MODELS_PATH=
for /f "delims=" %%M in ('python - <<"PY"
import os, json
p=None
# try env var or settings
env=os.environ.get('AI_STUDIO_MODELS_PATH')
if env:
    p=env
else:
    try:
        s=json.load(open(os.path.expanduser('~\\.ai_studio\\settings.json')))
        p=s.get('models_path')
    except Exception:
        pass
# Common default (user's provided path)
defaults=['E:\\flux\\model','E:\\flux','./models']
if p and os.path.exists(p):
    print(p)
    sys.exit(0)
for d in defaults:
    if os.path.exists(d):
        print(d)
        sys.exit(0)
print('NOT_FOUND')
PY
') do set MODELS_PATH=%%M

if "%MODELS_PATH%"=="NOT_FOUND" (
  echo WARNING: No models path detected automatically.
  echo Please copy your models into the repo's models\ folder or set environment variable AI_STUDIO_MODELS_PATH to point to E:\flux\model
  echo Example (current session): set AI_STUDIO_MODELS_PATH=E:\flux\model
  echo Example (permanent): setx AI_STUDIO_MODELS_PATH "E:\flux\model"
  pause
) else (
  echo [8/10] Detected models path: %MODELS_PATH%
  echo Setting AI_STUDIO_MODELS_PATH to %MODELS_PATH% (permanent for current user)...
  setx AI_STUDIO_MODELS_PATH "%MODELS_PATH%" > nul
)

REM 9) Final sanity checks
echo [9/10] Sanity checks: verifying torch and GPU availability...
python - <<"PY"
import sys
try:
    import torch
    print('torch',torch.__version__)
    print('cuda_available',torch.cuda.is_available())
    if torch.cuda.is_available():
        print('devices',torch.cuda.device_count())
except Exception as e:
    print('torch_not_ok',str(e))
PY

REM 10) Launch the application
echo [10/10] Launching AIFlux (launcher.py)...
python launcher.py

ENDLOCAL
exit /b 0
