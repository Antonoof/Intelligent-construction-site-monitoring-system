"""Подключение к базе данных (SQLAlchemy 2.0).

PostgreSQL — основная СУБД (docker-compose), SQLite — для запуска на ноутбуке без Docker.
Схема создаётся при старте (`init_db`); для промышленной эксплуатации — миграции Alembic.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import JSON, create_engine, event
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings

# JSON-поля: JSONB в PostgreSQL (индексируемый), обычный JSON в SQLite
JSONType = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    pass


def make_engine(url: str | None = None):
    url = url or settings.db_url()
    kwargs = {"future": True, "pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _fk_on(dbapi_conn, _):  # внешние ключи и WAL для SQLite
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()
    return engine


engine = make_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


# столбцы, добавленные после первой версии схемы: create_all новые таблицы создаёт, а столбцы в существующие —
# нет, поэтому база, созданная прежней версией (том PostgreSQL на демо-ВМ), дополняется здесь
ADDED_COLUMNS = {"snapshots": {"frame_zone": "VARCHAR(32)"}}


def init_db(bind=None) -> None:
    from sqlalchemy import inspect, text

    from . import models  # noqa: F401  регистрация таблиц
    eng = bind or engine
    Base.metadata.create_all(eng)
    insp = inspect(eng)
    with eng.begin() as conn:
        for table, cols in ADDED_COLUMNS.items():
            have = {c["name"] for c in insp.get_columns(table)}
            for name, ddl in cols.items():
                if name not in have:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))


@contextmanager
def session_scope() -> Iterator[Session]:
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def get_session() -> Iterator[Session]:
    """Зависимость FastAPI: сессия на запрос."""
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()
