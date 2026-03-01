"""In-process queue and worker for async LoRA training jobs."""

from __future__ import annotations

import queue
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from entity_store import get_entity_metadata, update_entity_metadata
from lora_trainer import train_lora_for_entity


def _utc_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")


@dataclass(slots=True)
class TrainingJob:
    id: str
    entity_id: str
    trigger_word: str
    steps: int
    rank: int
    queued_at: str


class TrainingQueueManager:
    def __init__(self) -> None:
        self._queue: queue.Queue[TrainingJob] = queue.Queue()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="lora-training-worker",
            daemon=True,
        )
        self._worker.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=timeout)

    def enqueue(
        self,
        *,
        entity_id: str,
        trigger_word: str,
        steps: int = 500,
        rank: int = 8,
    ) -> dict[str, Any]:
        job = TrainingJob(
            id=f"job_{uuid.uuid4().hex[:12]}",
            entity_id=entity_id,
            trigger_word=trigger_word,
            steps=steps,
            rank=rank,
            queued_at=_utc_now(),
        )
        with self._lock:
            self._jobs[job.id] = {
                "id": job.id,
                "entity_id": job.entity_id,
                "status": "queued",
                "queued_at": job.queued_at,
                "started_at": None,
                "finished_at": None,
                "error": None,
            }

        self._queue.put(job)
        return {
            "job_id": job.id,
            "status": "queued",
        }

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            data = self._jobs.get(job_id)
            return dict(data) if data else None

    def _set_job_state(
        self,
        *,
        job_id: str,
        status: str,
        started_at: str | None = None,
        finished_at: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            if job_id not in self._jobs:
                return
            job = self._jobs[job_id]
            job["status"] = status
            if started_at is not None:
                job["started_at"] = started_at
            if finished_at is not None:
                job["finished_at"] = finished_at
            job["error"] = error

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            started_at = _utc_now()
            self._set_job_state(job_id=job.id, status="training", started_at=started_at, error=None)
            update_entity_metadata(
                job.entity_id,
                {
                    "status": "training",
                    "training_job_id": job.id,
                    "training_started_at": started_at,
                    "error": None,
                },
            )

            try:
                result = train_lora_for_entity(
                    entity_id=job.entity_id,
                    trigger_word=job.trigger_word,
                    steps=job.steps,
                    rank=job.rank,
                )
                raw = get_entity_metadata(job.entity_id) or {}
                versions = raw.get("versions", [])
                if not isinstance(versions, list):
                    versions = []
                version = result["version"]
                if version not in versions:
                    versions.append(version)

                finished_at = _utc_now()
                update_entity_metadata(
                    job.entity_id,
                    {
                        "status": "ready",
                        "versions": versions,
                        "active_version": version,
                        "training_finished_at": finished_at,
                        "training_result": result,
                        "error": None,
                    },
                )
                self._set_job_state(job_id=job.id, status="ready", finished_at=finished_at, error=None)
            except Exception as exc:
                finished_at = _utc_now()
                update_entity_metadata(
                    job.entity_id,
                    {
                        "status": "failed",
                        "training_finished_at": finished_at,
                        "error": str(exc),
                    },
                )
                self._set_job_state(job_id=job.id, status="failed", finished_at=finished_at, error=str(exc))
            finally:
                self._queue.task_done()


training_queue = TrainingQueueManager()
