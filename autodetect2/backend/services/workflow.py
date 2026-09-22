from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ..config import STATUS_ASSIGNED, STATUS_DONE, STATUS_NEW, STATUS_REVIEW
from .project import Project, utc_now

STATE_VERSION = 1

# Один файл на проект: индекс кадров (index.json) переписывается только при
# пересканировании источников, а статусы меняются каждые несколько секунд —
# держать их вместе значило бы переписывать мегабайты ради одной строки.
_LOCKS: dict[str, threading.Lock] = {}
_LOCK_GUARD = threading.Lock()


def _lock_for(project_id: str) -> threading.Lock:
    with _LOCK_GUARD:
        return _LOCKS.setdefault(project_id, threading.Lock())


@dataclass
class State:
    project: Project
    images: dict[str, dict] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return self.project.root / "state.json"

    # ------------------------------------------------------------- чтение
    def entry(self, name: str) -> dict:
        return self.images.get(name, {})

    def status(self, name: str) -> str:
        return self.images.get(name, {}).get("status", STATUS_NEW)

    def owner(self, name: str) -> str | None:
        return self.images.get(name, {}).get("owner")

    def assignment(self, name: str) -> str | None:
        return self.images.get(name, {}).get("assignment")

    def names_with(self, status: str) -> list[str]:
        return sorted(n for n, e in self.images.items() if e.get("status") == status)

    def names_of_assignment(self, assignment_id: str) -> list[str]:
        return sorted(n for n, e in self.images.items() if e.get("assignment") == assignment_id)

    def busy(self) -> set[str]:
        """Кадры, которые нельзя предлагать к разметке: уже отданы или закрыты."""
        return {n for n, e in self.images.items() if e.get("status") in (STATUS_ASSIGNED, STATUS_DONE)}

    def counts(self) -> dict[str, int]:
        out = {STATUS_NEW: 0, STATUS_ASSIGNED: 0, STATUS_DONE: 0, STATUS_REVIEW: 0}
        for entry in self.images.values():
            key = entry.get("status", STATUS_NEW)
            out[key] = out.get(key, 0) + 1
        total = len(self.project.samples())
        out[STATUS_NEW] = max(total - out[STATUS_ASSIGNED] - out[STATUS_DONE] - out[STATUS_REVIEW], 0)
        out["total"] = total
        return out

    # ------------------------------------------------------------- запись
    def set(self, names: Iterable[str], status: str, **fields) -> int:
        stamp = utc_now()
        touched = 0
        for name in names:
            entry = self.images.setdefault(name, {})
            entry["status"] = status
            entry["at"] = stamp
            for key, value in fields.items():
                if value is None:
                    entry.pop(key, None)
                else:
                    entry[key] = value
            touched += 1
        return touched

    def release(self, names: Iterable[str]) -> int:
        """Снимает занятость, не трогая уже закрытые кадры."""
        touched = 0
        for name in list(names):
            entry = self.images.get(name)
            if not entry or entry.get("status") == STATUS_DONE:
                continue
            self.images.pop(name, None)
            touched += 1
        return touched

    def flag(self, name: str, note: str | None) -> None:
        entry = self.images.setdefault(name, {})
        entry["status"] = STATUS_REVIEW
        entry["note"] = note or ""
        entry["at"] = utc_now()

    def save(self) -> None:
        payload = {"version": STATE_VERSION, "images": self.images, "updated_at": utc_now()}
        self.path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load(project: Project) -> State:
    path = project.root / "state.json"
    if not path.exists():
        return State(project=project)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return State(project=project)
    return State(project=project, images=dict(payload.get("images") or {}))


class transaction:
    """Читает состояние, отдаёт его блоку и пишет обратно под блокировкой.

    Занятость кадров правят сразу несколько мест — задания, разметка, импорт
    результатов, — и все они живут в потоках фоновых задач. Без общей блокировки
    два одновременных задания разошлись бы на одном и том же наборе кадров.
    """

    def __init__(self, project: Project) -> None:
        self.project = project
        self._lock = _lock_for(project.id)
        self.state: State | None = None

    def __enter__(self) -> State:
        self._lock.acquire()
        self.state = load(self.project)
        return self.state

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            if exc_type is None and self.state is not None:
                self.state.save()
        finally:
            self._lock.release()
        return False
