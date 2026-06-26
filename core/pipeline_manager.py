from __future__ import annotations
import logging
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from core.queue_manager import JobQueueManager
from core.model_manager import ModelManager
from config.settings import Settings

logger = logging.getLogger("pipeline")

@dataclass
class GenerateRequest:
    prompt: str
    seed: Optional[int] = None
    steps: int = 28
    width: int = 1024
    height: int = 1024
    scheduler: Optional[str] = None
    base_model: Optional[str] = None
    identity_engine: Optional[str] = None
    init_image: Optional[str] = None
    strength: float = 0.8
    extra: Dict[str,Any] = field(default_factory=dict)

class PipelineManager:
    """Central manager that decides which engines/plugins to call."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_manager = ModelManager(settings.models_path)
        self.queue = JobQueueManager(self)
        logger.info("PipelineManager initialized")

    def generate(self, request: GenerateRequest) -> str:
        """Schedule a generation job and return job id."""
        job_id = self.queue.enqueue(request)
        logger.info("Job enqueued: %s", job_id)
        return job_id

    def _run_job(self, job_id: str, request: GenerateRequest) -> None:
        """
        Internal: pipeline orchestration.
        - Select base model via ModelManager if not provided
        - Select identity engine
        - Compose final pipeline and call engine plugin
        """
        logger.info("Running job %s", job_id)
        base_model = request.base_model or self.model_manager.select_default()
        identity_engine = request.identity_engine or "auto"
        # TODO: plugin resolution; for now call a default engine if available
        engine = self.model_manager.get_engine_for_model(base_model)
        if engine is None:
            logger.error("No engine available for model %s", base_model)
            return
        # engine.run(...) should be implemented by engine plugin
        engine.run(request)
