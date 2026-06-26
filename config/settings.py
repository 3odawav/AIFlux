from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict

CONFIG_PATH = Path.home() / ".ai_studio" / "settings.json"

@dataclass
class Settings:
    theme: str = "dark"
    language: str = "en"
    window_position: Dict[str,int] | None = None
    last_used_model: str | None = None
    output_folder: str = "./outputs"
    models_path: str = "./models"
    cache_path: str = "./cache/cache.sqlite"

    @classmethod
    def load(cls) -> "Settings":
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return cls(**data)
        else:
            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            settings = cls()
            settings.save()
            return settings

    def save(self) -> None:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
