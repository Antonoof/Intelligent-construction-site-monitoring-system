from __future__ import annotations

import logging
import threading
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

log = logging.getLogger(__name__)


@dataclass
class Task:
    id: str
    kind: str
    title: str
    status: str = "running"
    progress: float = 0.0
    message: str = ""
    result: Any = None
    error: str | None = None
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=400))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    finished_at: str | None = None

    def as_dict(self, include_logs: bool = True) -> dict:
        payload = {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "status": self.status,
            "progress": round(self.progress, 4),
            "message": self.message,
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "result": self.result,
        }
        if include_logs:
            payload["logs"] = list(self.logs)
        return payload


class TaskContext:
    def __init__(self, task: Task, lock: threading.Lock) -> None:
        self._task = task
        self._lock = lock

    def progress(self, message: str, value: float | None = None) -> None:
        with self._lock:
            self._task.message = message
            if value is not None:
                self._task.progress = max(0.0, min(1.0, float(value)))
            self._task.logs.append(message)

    def log(self, message: str) -> None:
        with self._lock:
            self._task.logs.append(message)


class TaskRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()

    def submit(self, kind: str, title: str, fn: Callable[[TaskContext], Any]) -> Task:
        task = Task(id=uuid.uuid4().hex[:12], kind=kind, title=title)
        with self._lock:
            self._tasks[task.id] = task
        context = TaskContext(task, self._lock)

        def runner() -> None:
            try:
                result = fn(context)
                with self._lock:
                    task.result = result
                    task.status = "done"
                    task.progress = 1.0
                    task.message = task.message or "Готово"
            except Exception as exc:  # noqa: BLE001 - surfaced to the UI
                log.exception("Задача %s (%s) упала", task.id, kind)
                with self._lock:
                    task.status = "error"
                    task.error = str(exc) or exc.__class__.__name__
                    task.logs.append(traceback.format_exc().strip().splitlines()[-1])
            finally:
                with self._lock:
                    task.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        threading.Thread(target=runner, name=f"task-{kind}", daemon=True).start()
        return task

    def get(self, task_id: str) -> Task | None:
        with self._lock:
            return self._tasks.get(task_id)

    def list(self, kind: str | None = None) -> list[dict]:
        with self._lock:
            tasks = list(self._tasks.values())
        if kind:
            tasks = [t for t in tasks if t.kind == kind]
        return [t.as_dict(include_logs=False) for t in sorted(tasks, key=lambda t: t.created_at, reverse=True)]


registry = TaskRegistry()
