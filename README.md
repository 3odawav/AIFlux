AI Studio Pro — Desktop Image Generation Studio (skeleton)

This repository is the AIStudio production-ready skeleton. It is plugin-based and modular,
designed to run local engines (FLUX, PuLID, InstantID, etc.) and preserve facial identity.

See docs/ and the README for usage.

Quick start (Linux/WSL/macOS):

1. python -m venv .venv
2. source .venv/bin/activate
3. pip install -r requirements.txt
4. Place your models under ./models/ (see config/models.json example)
5. python launcher.py

Notes:
- Requires NVIDIA GPU and correct torch+CUDA matching your drivers.
- This skeleton implements plugin hooks; add engines under engines/<engine_name>/.
