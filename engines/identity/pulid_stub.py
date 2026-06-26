"""
Stub for PuLID identity engine.
Real implementation should expose:
- extract_embedding(image) -> vector
- condition_latents(latents, embedding, strength) -> latents
"""

import logging
from typing import Any

logger = logging.getLogger("identity.pulid")

class EnginePlugin:
    def __init__(self):
        self.name = "pulid"
        self.loaded = False

    def load(self):
        logger.info("PuLID engine loaded (stub)")
        self.loaded = True

    def extract_embedding(self, image_path: str):
        # TODO: implement using real identity engine
        return b"fake_embedding"

    def apply_identity(self, latents, embedding, strength: float):
        # modify latents in-place or return new latents
        return latents
