# Running AIFlux (Windows)

This document collects the exact commands needed to prepare and run the project on a Windows machine
with an NVIDIA RTX Quadro 5000 (16 GB) GPU. Follow the steps below carefully.

Prerequisites
- Git installed and available on PATH.
- Python 3.10 or 3.11 installed and available on PATH (you will create a venv for the project).
- NVIDIA drivers installed (you already have Driver 610.47; verified with nvidia-smi).

Step 1 — Clone the repo and checkout branch

1) Open PowerShell and run:

    git clone https://github.com/3odawav/AIFlux.git
    cd AIFlux
    git checkout ai-studio/init

Step 2 — Create virtual environment and run diagnostic

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python --version

Run the environment diagnostic (this script detects CUDA/toolkit, installed packages, and the models folder):

    python tools/env_check.py

This produces logs/env_report.json. Paste the file here or inspect it. Important fields:
- nvidia-smi output (GPU name, memory, driver)
- torch import status (if torch is already installed inside the venv)
- detected models path (should detect E:\flux\model or whatever you have)

Step 3 — Install PyTorch (choose correct wheel)

The correct torch wheel depends on the CUDA runtime your PyTorch will target. The env_check report contains
useful hints (look at 'nvidia-smi' and 'torch' sections). Use one of these commands inside the activated venv.

Common examples (run only one):

# If you want CUDA 12.1 build (recommended if your system supports CUDA 12.x):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# If you need CUDA 11.8 build:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# If you cannot install CUDA-compatible wheel, install CPU-only PyTorch:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

After installing, verify:
    python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"

Step 4 — Install remaining dependencies (we install everything except torch earlier; if you didn't, run now):

    pip install -r requirements.txt

If specific packages fail to build (xformers, onnxruntime-gpu), see Troubleshooting section below.

Step 5 — Configure models folder

Option A — Keep your existing models in place (recommended, you already have E:\flux\model)
- Set environment variable so AIFlux uses that folder (PowerShell for current session):

    $env:AI_STUDIO_MODELS_PATH = 'E:\flux\model'

- Make it permanent for your user (PowerShell):

    setx AI_STUDIO_MODELS_PATH "E:\flux\model"

Option B — Copy your models into the project (if you prefer local repository models):

    mkdir models
    robocopy /E "E:\flux\model" "<path-to-repo>\models"

AIFlux auto-detects models in models/ or the path set by AI_STUDIO_MODELS_PATH.
The models tree you provided should be recognized automatically (FLUX.1-dev, FLUX.1-Redux-dev, loras/ ).

Step 6 — Run the application

    python launcher.py

The application will open a PySide6-based UI. The first run will create %USERPROFILE%\.ai_studio\settings.json with the detected models path.

Troubleshooting — Common problems & fixes

1) torch import fails or CUDA not available
- Reinstall torch with the correct wheel as shown above. If the wheel for your exact CUDA is unavailable, try cu121 or cu118.

2) onnxruntime-gpu installation problems
- Try: pip install onnxruntime-gpu
- If it fails, specify a wheel: check https://github.com/microsoft/onnxruntime/releases and download the wheel matching your Python and CUDA versions.

3) xformers installation fails
- xformers wheels are not available for all Python + CUDA combos. You can skip xformers for now; the app will run slower.
- If you need xformers, find a pre-built wheel for your Python and CUDA (community builds exist) and install via:
    pip install <path-to-xformers-whl>

4) exiv2-python / exif metadata
- On Windows it is easiest to install ExifTool and modify the project to use it. To install ExifTool:
    - Download from https://exiftool.org/ and add exiftool.exe to PATH, or
    - (If you have Chocolatey) choco install exiftool

5) Permission / Driver Issues
- If nvidia-smi shows older drivers or CUDA unavailable, update NVIDIA drivers from https://www.nvidia.com/Download/index.aspx for Quadro RTX 5000.

Performance tips
- For VRAM-limited scenarios (16GB), use FP16/BF16 and model offload: set appropriate options in the UI (Performance page) or set:
    setx PYTORCH_CUDA_ALLOC_CONF "expandable_segments:True"
- Use smaller batch sizes and limit image size for memory-intensive runs (start with 1024x1024)


