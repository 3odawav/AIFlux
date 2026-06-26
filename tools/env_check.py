#!/usr/bin/env python3
"""
Environment and Model Diagnostic Tool for AIStudio

Run this script before first launch to verify Python, CUDA, GPU, and required packages.
It also checks the detected models folder and validates presence of model files.

Usage:
    python tools/env_check.py

Outputs:
 - console readable summary
 - JSON report at ./logs/env_report.json

This script is defensive and will not modify your environment. It only reports problems and
suggests next steps.
"""
from __future__ import annotations
import sys
import os
import json
import shutil
import subprocess
import platform
from pathlib import Path
from typing import Dict, Any, List

ROOT = Path.cwd()
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = LOGS_DIR / "env_report.json"

REQUIREMENTS = [
    ("python", "."),
    ("torch", "torch"),
    ("diffusers", "diffusers"),
    ("transformers", "transformers"),
    ("optimum", "optimum"),
    ("optimum.quanto", "optimum.quanto"),
    ("xformers", "xformers"),
    ("accelerate", "accelerate"),
    ("safetensors", "safetensors"),
    ("numpy", "numpy"),
    ("Pillow", "PIL"),
    ("opencv-python", "cv2"),
    ("insightface", "insightface"),
    ("onnxruntime-gpu", "onnxruntime"),
    ("exiv2-python", "exiv2"),
    ("PySide6", "PySide6"),
]

MODEL_FILE_HINTS = ["safetensors", "model.safetensors", "model_index.json", "model-00001-of-", "config.json"]


def run_cmd(cmd: List[str]) -> Dict[str, Any]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return {"ok": True, "output": out.strip()}
    except Exception as e:
        return {"ok": False, "output": str(e)}


def check_python() -> Dict[str, Any]:
    info = {}
    info["sys.version"] = sys.version
    info["platform"] = platform.platform()
    info["executable"] = sys.executable
    try:
        major = sys.version_info.major
        minor = sys.version_info.minor
        info["ok"] = (major > 3) or (major == 3 and minor >= 10)
        info["required"] = "3.10+"
    except Exception:
        info["ok"] = False
        info["required"] = "3.10+"
    return info


def check_command_exists(cmd_name: str) -> bool:
    return shutil.which(cmd_name) is not None


def check_nvidia_smi() -> Dict[str, Any]:
    if not check_command_exists("nvidia-smi"):
        return {"ok": False, "reason": "nvidia-smi not found in PATH"}
    res = run_cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_capability", "--format=csv,noheader,nounits"]) 
    if not res["ok"]:
        # fallback to simple nvidia-smi
        res2 = run_cmd(["nvidia-smi"]) 
        return {"ok": True, "raw": res2.get("output")}
    lines = [l.strip() for l in res["output"].splitlines() if l.strip()]
    gpus = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"name": parts[0], "memory_total_mb": parts[1], "driver_version": parts[2], "compute": parts[3] if len(parts) > 3 else None})
    return {"ok": True, "gpus": gpus}


def check_torch() -> Dict[str, Any]:
    try:
        import torch
        info = {
            "installed": True,
            "torch_version": getattr(torch, "__version__", None),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version_build": torch.version.cuda if hasattr(torch, "version") else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        # attempt to fetch device properties if available
        if info["cuda_available"] and info["device_count"] > 0:
            try:
                devices = []
                for i in range(info["device_count"]):
                    props = torch.cuda.get_device_properties(i)
                    devices.append({"name": props.name, "total_memory_gb": round(props.total_memory/1024**3, 2)})
                info["devices"] = devices
            except Exception as e:
                info["devices_exception"] = str(e)
        return info
    except Exception as e:
        return {"installed": False, "error": str(e)}


def check_packages() -> Dict[str, Any]:
    results = {}
    for req_name, module_name in REQUIREMENTS:
        try:
            # special-case Python: already checked
            if req_name == "python":
                results[req_name] = {"installed": True}
                continue
            __import__(module_name)
            # try to get version
            ver = None
            try:
                import importlib.metadata as md
                ver = md.version(req_name)
            except Exception:
                try:
                    import pkg_resources
                    ver = pkg_resources.get_distribution(req_name).version
                except Exception:
                    ver = None
            results[req_name] = {"installed": True, "module": module_name, "version": ver}
        except Exception as e:
            results[req_name] = {"installed": False, "module": module_name, "error": str(e)}
    return results


def check_system_packages() -> Dict[str, Any]:
    checks = {}
    for cmd in ("nvidia-smi", "nvcc", "exiv2"):
        checks[cmd] = {"found": check_command_exists(cmd)}
        if checks[cmd]["found"]:
            checks[cmd]["output"] = run_cmd([cmd, "--version"]) if cmd == "nvcc" else run_cmd([cmd, "--help"]) if cmd=="exiv2" else run_cmd([cmd, "--version"]) if cmd=="nvidia-smi" else {}
    return checks


def check_models_folder(models_path: str | None) -> Dict[str, Any]:
    result = {"path": models_path, "exists": False, "folders": []}
    if not models_path:
        return result
    p = Path(models_path)
    if not p.exists():
        return result
    result["exists"] = True
    # list top-level directories and for each try to validate
    for item in sorted(p.iterdir()):
        if item.is_dir():
            folder_info = {"name": item.name, "path": str(item), "files": [], "hints": []}
            try:
                for f in item.iterdir():
                    folder_info["files"].append(f.name)
                    if any(h in f.name.lower() for h in ("safetensors", "model.safetensors", "model_index.json", "config.json", "diffusion_pytorch_model")):
                        folder_info["hints"].append(f.name)
            except Exception as e:
                folder_info["error_listing"] = str(e)
            result["folders"].append(folder_info)
    return result


def main():
    report: Dict[str, Any] = {}
    print("AIStudio Environment Check\n===========================")
    report["python"] = check_python()
    print(f"Python: {report['python']['ok']} - {report['python']['sys.version'].splitlines()[0]}")

    print("\nChecking nvidia-smi and GPUs...")
    nvsmi = check_nvidia_smi()
    report["nvidia-smi"] = nvsmi
    if nvsmi.get("ok"):
        gpus = nvsmi.get("gpus", [])
        if gpus:
            for g in gpus:
                print(f"GPU detected: {g.get('name')} - {g.get('memory_total_mb')} MB")
        else:
            print("nvidia-smi returned info but parsing produced no GPUs (raw output saved)")
    else:
        print("nvidia-smi not available or failed to run: ", nvsmi.get("reason") or nvsmi.get("output"))

    print("\nChecking PyTorch...")
    report["torch"] = check_torch()
    if report["torch"].get("installed"):
        print(f"Torch {report['torch'].get('torch_version')} - CUDA available: {report['torch'].get('cuda_available')}")
        if report['torch'].get('devices'):
            for d in report['torch']['devices']:
                print(f" - {d['name']} {d['total_memory_gb']} GB")
    else:
        print("PyTorch not installed or failed to import:", report['torch'].get("error"))

    print("\nChecking Python packages (this may take a few seconds)...")
    report["packages"] = check_packages()
    for pkg, info in report["packages"].items():
        status = "OK" if info.get("installed") else "MISSING"
        ver = info.get("version")
        print(f"{pkg:20} {status:10} {('v'+ver) if ver else ''}")

    print("\nChecking system-level commands...")
    report["system"] = check_system_packages()
    for cmd, info in report["system"].items():
        print(f"{cmd}: {'found' if info.get('found') else 'missing'}")

    print("\nChecking models folder as configured in settings...")
    # try to read settings
    models_path = None
    try:
        from config.settings import Settings
        s = Settings.load()
        models_path = s.models_path
    except Exception:
        # fallback to environment or common paths
        models_path = os.environ.get("AI_STUDIO_MODELS_PATH")
    report["models"] = check_models_folder(models_path)
    if report["models"].get("exists"):
        print(f"Models path: {report['models']['path']} — {len(report['models']['folders'])} folders found")
        for f in report['models']["folders"]:
            hints = f.get('hints', [])
            print(f" - {f['name']}: {len(f['files'])} files, hints: {', '.join(hints) if hints else 'none'}")
    else:
        print("No models path detected. Set AI_STUDIO_MODELS_PATH env or put models under ./models")

    # recommendations
    recs: List[str] = []
    if not report['python']['ok']:
        recs.append("Python 3.10 or newer is required.")
    if not report['nvidia-smi'].get('ok'):
        recs.append("nvidia-smi not found — ensure NVIDIA drivers are installed and nvidia-smi is on PATH.")
    if not report['torch'].get('installed'):
        recs.append("PyTorch not installed. Install the correct wheel for your CUDA version (see README).")
    else:
        if not report['torch'].get('cuda_available'):
            recs.append("PyTorch installed but CUDA not available. Ensure torch built with CUDA and drivers are compatible.")
    # check large packages
    missing_pkgs = [p for p,i in report['packages'].items() if not i.get('installed')]
    if missing_pkgs:
        recs.append("Missing Python packages: " + ", ".join(missing_pkgs))
    if not report['models'].get('exists'):
        recs.append("Models folder not found. Place your models under ./models or set AI_STUDIO_MODELS_PATH to your models folder.")

    report['recommendations'] = recs

    # Save report JSON
    try:
        with open(REPORT_PATH, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed report saved to: {REPORT_PATH}")
    except Exception as e:
        print("Failed to save report:", e)

    print("\nSummary recommendations:")
    for r in recs:
        print(" - ", r)

    print("\nIf you want, copy the file logs/env_report.json and paste it here and I will analyze and give step-by-step fixes.")


if __name__ == '__main__':
    main()
