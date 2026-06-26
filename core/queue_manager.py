from __future__ import annotations
import uuid
import logging
from typing import Dict
from queue import Queue
from PySide6.QtCore import QObject, Signal
from concurrent.futures import ThreadPoolExecutor
from core.pipeline_manager import GenerateRequest, PipelineManager

logger = logging.getLogger("queue")

class JobSignals(QObject):
    progress = Signal(str, float)  # job_id, progress 0..1
    finished = Signal(str, dict)  # job_id, result

class JobQueueManager:
    """Simple queue manager using ThreadPoolExecutor and signals."""

    def __init__(self, pipeline: PipelineManager, max_workers: int = 1):
        self.pipeline = pipeline
        self.queue: Queue = Queue()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.signals = JobSignals()
        self._jobs: Dict[str, GenerateRequest] = {}

    def enqueue(self, request: GenerateRequest) -> str:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = request
        self.executor.submit(self._worker, job_id, request)
        return job_id

    def _worker(self, job_id: str, request: GenerateRequest) -> None:
        try:
            self.pipeline._run_job(job_id, request)
            # on finish emit signal (currently no result payload)
            self.signals.finished.emit(job_id, {"status": "ok"})
        except Exception as e:
            logger.exception("Job failed %s", job_id)
            self.signals.finished.emit(job_id, {"status": "error", "reason": str(e)})
