<#
Windows setup script for AIFlux (AIStudio skeleton).

This script creates a Python virtual environment, upgrades pip, runs the environment diagnostic
(tools/env_check.py), and installs non-Torch Python dependencies. It does NOT automatically install
PyTorch because the correct wheel depends on your CUDA/toolkit; see instructions below.

Usage (PowerShell, run as regular user):
  1) Open PowerShell and cd to the repo root (where launcher.py sits).
  2) ./scripts/setup_windows.ps1

After the script completes you will find logs/env_report.json. Based on that report install PyTorch
with the recommended command in the README (examples included).
#>

param(
    [switch]$SkipDiagnostic
)

$ErrorActionPreference = 'Stop'

Write-Host "== AIFlux Windows Setup Script ==" -ForegroundColor Cyan

# 1) Create virtualenv
if (-Not (Test-Path -Path .venv)) {
    Write-Host "Creating virtual environment .venv..."
    python -m venv .venv
} else {
    Write-Host ".venv already exists. Skipping creation."
}

$VENV_PY = Join-Path -Path $PWD -ChildPath ".venv\Scripts\python.exe"
$VENV_PIP = Join-Path -Path $PWD -ChildPath ".venv\Scripts\pip.exe"

# 2) Upgrade pip and install build tools
Write-Host "Upgrading pip, setuptools and wheel..."
& $VENV_PY -m pip install --upgrade pip setuptools wheel

# 3) Run diagnostic (unless skipped)
if (-Not $SkipDiagnostic) {
    Write-Host "Running environment diagnostic (tools/env_check.py)..."
    & $VENV_PY tools/env_check.py
    Write-Host "Diagnostic saved to logs/env_report.json"
} else {
    Write-Host "Skipping diagnostic as requested"
}

# 4) Install non-Torch requirements (we avoid installing torch here because the right wheel depends on CUDA)
Write-Host "Installing non-Torch requirements from requirements.txt (will skip torch if present)..."

# Build a temporary requirements file that excludes torch/torchaudio/torchvision so user can install the correct torch wheel manually
$reqs = Get-Content requirements.txt | Where-Object { $_ -notmatch '^(torch|torchvision|torchaudio)' }
$reqsFile = Join-Path $env:TEMP 'aiflux_reqs_no_torch.txt'
$reqs | Set-Content -Path $reqsFile -Encoding UTF8

& $VENV_PIP install -r $reqsFile

Write-Host "Non-Torch dependencies installed."

Write-Host "Setup finished. Next steps:"
Write-Host "  - Inspect logs/env_report.json: cat logs\env_report.json" -ForegroundColor Yellow
Write-Host "  - Install the correct PyTorch wheel for your CUDA version (see README or run the commands in the project docs)." -ForegroundColor Yellow
Write-Host "  - After installing torch, run: python launcher.py" -ForegroundColor Green

