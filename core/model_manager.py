from __future__ import annotations
import logging
from pathlib import Path
import json
from typing import Dict, List, Optional
from config.paths import MODELS_DIR, ENGINES_DIR
import os

logger = logging.getLogger("model-manager")

class ModelManager:
    """Discover models in models/ and map them to engine plugins.

    Enhanced discovery heuristics:
    - If folder contains transformer/ and vae/ and text_encoder/ -> classify as 'flux'
    - If folder name or contained files include 'lora' or a safetensors file with 'lora' -> 'lora'
    - If folder contains 'controlnet' or filename includes 'controlnet' -> 'controlnet'
    - If folder contains 'ipadapter' or 'ip_adapter' -> 'ipadapter'
    - If folder contains 'image_embedder' or 'image_encoder' and looks like redux -> 'redux'
    - Fallback to simple name heuristics or 'base'
    """

    def __init__(self, models_path: str | None = None):
        # Allow env override or passed path
        env_path = os.environ.get("AI_STUDIO_MODELS_PATH")
        if env_path:
            self.models_path = Path(models_path) if models_path else Path(env_path)
        else:
            self.models_path = Path(models_path) if models_path else MODELS_DIR

        self.models: Dict[str, Dict] = {}
        self.engines = {}
        self.discover_models()
        self.discover_engines()

    def discover_models(self) -> None:
        """Scan models dir and classify models by heuristics (folder contents)."""
        if not self.models_path.exists():
            self.models_path.mkdir(parents=True, exist_ok=True)
            logger.info("Models directory created: %s", self.models_path)
            return
        for item in sorted(self.models_path.iterdir()):
            if item.is_dir():
                model_id = item.name
                try:
                    model_type = self._classify(item)
                    self.models[model_id] = {"path": str(item.resolve()), "type": model_type}
                except Exception as e:
                    logger.warning("Failed to classify model folder %s: %s", item, e)
                    self.models[model_id] = {"path": str(item.resolve()), "type": "unknown"}
        logger.info("Discovered models: %s", list(self.models.keys()))

    def _classify(self, path: Path) -> str:
        name = path.name.lower()
        # Quick name-based hints
        if "lora" in name or any(f.lower().endswith('.safetensors') and 'lora' in f.lower() for f in [p.name for p in path.iterdir() if p.is_file()]):
            return "lora"
        if "controlnet" in name or any('controlnet' in f.lower() for f in [p.name for p in path.iterdir() if p.is_file()]):
            return "controlnet"
        if "ipadapter" in name or "ip_adapter" in name or any('ipadapter' in f.lower() or 'ip_adapter' in f.lower() for f in [p.name for p in path.iterdir() if p.is_file()]):
            return "ipadapter"
        # Check for FLUX model layout
        subdirs = {p.name.lower() for p in path.iterdir() if p.is_dir()}
        files = {p.name.lower() for p in path.iterdir() if p.is_file()}
        if {"transformer", "vae", "text_encoder"}.issubset(subdirs) or any('flux' in fn for fn in files):
            return "flux"
        # Redux / image embedder style
        if "image_embedder" in subdirs or "image_encoder" in subdirs or 'redux' in name:
            return "redux"
        # LoRA folders sometimes live under root loras/
        if path.name.lower() == 'loras' or any(f.endswith('.safetensors') for f in files):
            # if contains many safetensors, classify as collection (loras)
            safes = [f for f in files if f.endswith('.safetensors')]
            if safes:
                return "lora"
        # fallback name heuristics
        if "flux" in name or name.startswith('flux'):
            return "flux"
        if "redux" in name:
            return "redux"
        # default fallback
        return "base"

    def discover_engines(self) -> None:
        """Load engine plugins from engines/ directory by importing their module."""
        try:
            if not ENGINES_DIR.exists():
                ENGINES_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        for d in sorted(ENGINES_DIR.iterdir() if ENGINES_DIR.exists() else []):
            if d.is_dir():
                try:
                    module_name = f"engines.{d.name}"
                    module = __import__(module_name, fromlist=["*"])
                    if hasattr(module, "EnginePlugin"):
                        self.engines[d.name] = module.EnginePlugin()
                        logger.info("Loaded engine plugin: %s", d.name)
                except Exception as e:
                    logger.warning("Failed to load engine plugin %s: %s", d.name, e)

    def select_default(self) -> Optional[str]:
        # prefer flux models
        for k, v in self.models.items():
            if v.get("type") == "flux":
                return k
        # otherwise return first discovered
        if self.models:
            return next(iter(self.models.keys()))
        return None

    def get_engine_for_model(self, model_id: str):
        # map model type -> engine instance
        meta = self.models.get(model_id)
        if not meta:
            return None
        t = meta.get("type")
        # direct mapping
        if t in self.engines:
            return self.engines.get(t)
        # fallback: flux engine if available
        return self.engines.get("flux") or self.engines.get(t) or None
