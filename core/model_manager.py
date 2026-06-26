from __future__ import annotations
import logging
from pathlib import Path
import json
from typing import Dict, List, Optional
from config.paths import MODELS_DIR, ENGINES_DIR

logger = logging.getLogger("model-manager")

class ModelManager:
    """Discover models in models/ and map them to engine plugins."""

    def __init__(self, models_path: str | None = None):
        self.models_path = Path(models_path) if models_path else MODELS_DIR
        self.models: Dict[str, Dict] = {}
        self.engines = {}
        self.discover_models()
        self.discover_engines()

    def discover_models(self) -> None:
        """Scan models dir and classify models by heuristics (filename, folders)."""
        if not self.models_path.exists():
            self.models_path.mkdir(parents=True, exist_ok=True)
            return
        for item in self.models_path.iterdir():
            if item.is_dir():
                # simple heuristic based on folder name
                model_id = item.name
                self.models[model_id] = {"path": str(item.resolve()), "type": self._classify(item)}
        logger.info("Discovered models: %s", list(self.models.keys()))

    def _classify(self, path: Path) -> str:
        name = path.name.lower()
        if "lora" in name:
            return "lorA"
        if "controlnet" in name:
            return "controlnet"
        if "flux" in name:
            return "flux"
        # fallback
        return "base"

    def discover_engines(self) -> None:
        """Load engine plugins from engines/ directory by importing their module."""
        for d in ENGINES_DIR.iterdir():
            if d.is_dir():
                try:
                    module_name = f"engines.{d.name}"
                    module = __import__(module_name, fromlist=["*"])
                    if hasattr(module, "EnginePlugin"):
                        self.engines[d.name] = module.EnginePlugin()
                except Exception as e:
                    logger.warning("Failed to load engine plugin %s: %s", d.name, e)

    def select_default(self) -> Optional[str]:
        # basic selection (first flux if exists)
        for k,v in self.models.items():
            if v.get("type") == "flux":
                return k
        if self.models:
            return next(iter(self.models.keys()))
        return None

    def get_engine_for_model(self, model_id: str):
        # map model type -> engine instance
        meta = self.models.get(model_id)
        if not meta:
            return None
        t = meta.get("type")
        return self.engines.get(t) or self.engines.get("flux") or None
