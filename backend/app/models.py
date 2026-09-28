"""Модель данных: техника, этапы работ, наблюдения, отклонения и нарушения.

Четыре группы таблиц:

1. Методика (справочники, загружаются из methodology/*.yaml, версионируются):
   equipment_classes, object_types, equipment_profiles, work_types, deviation_rules.
2. Проект: projects, zones, cameras, camera_zones, site_equipment,
   schedule_versions, schedule_tasks (этапы календарного графика с привязкой к виду работ).
3. Наблюдения: snapshots (снимок + качество + время), detections (рамки техники по зонам),
   snapshot_checks (объяснение сопоставления снимка с этапами по каждой зоне).
4. Выводы: deviations (автоматические отклонения), deviation_evidence (снимки-доказательства),
   deviation_reviews (вердикты инженера), violations (зарегистрированные нарушения).

Отклонение — вывод системы; нарушение — отклонение, подтверждённое инженером и поставленное
на контроль (номер, подрядчик, срок устранения).
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import (Boolean, Date, DateTime, Float, ForeignKey, Index, Integer, String, Text,
                        UniqueConstraint)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base, JSONType


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


# ============================================================ 1. Методика

class EquipmentClass(Base):
    __tablename__ = "equipment_classes"
    key: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    name_en: Mapped[str] = mapped_column(String(64), default="")
    in_tz: Mapped[bool] = mapped_column(Boolean, default=False)          # есть в перечне ТЗ
    site_wide: Mapped[bool] = mapped_column(Boolean, default=False)      # не привязывается к зоне
    models: Mapped[list] = mapped_column(JSONType, default=list)         # версии детектора
    color: Mapped[str] = mapped_column(String(9), default="#888888")


class ObjectType(Base):
    __tablename__ = "object_types"
    key: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(64))


class EquipmentProfile(Base):
    """Профиль техники: какая техника нужна этапу, какая парная, какая допустима."""
    __tablename__ = "equipment_profiles"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    observability: Mapped[str] = mapped_column(String(8))                # high|medium|low|none
    required: Mapped[list] = mapped_column(JSONType, default=list)       # [{role, any_of, min, window}]
    companions: Mapped[list] = mapped_column(JSONType, default=list)     # [{lead, partner, severity, message, window}]
    allowed: Mapped[list] = mapped_column(JSONType, default=list)
    not_detected: Mapped[list] = mapped_column(JSONType, default=list)
    is_composite: Mapped[bool] = mapped_column(Boolean, default=False)
    methodology_version: Mapped[str] = mapped_column(String(32))


class WorkType(Base):
    """Вид работ справочника организаторов (377 строк) с привязкой к профилю и стадии."""
    __tablename__ = "work_types"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)       # «12.3.7-1»
    code: Mapped[str | None] = mapped_column(String(32), nullable=True)  # код из справочника
    name: Mapped[str] = mapped_column(Text)
    level: Mapped[int] = mapped_column(Integer)
    parent_id: Mapped[str | None] = mapped_column(ForeignKey("work_types.id"), nullable=True)
    stage: Mapped[str | None] = mapped_column(String(4), nullable=True)  # S1..S5
    profile_id: Mapped[str] = mapped_column(ForeignKey("equipment_profiles.id"))
    observability: Mapped[str] = mapped_column(String(8))
    object_types: Mapped[list] = mapped_column(JSONType, default=list)
    expert_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    has_children: Mapped[bool] = mapped_column(Boolean, default=False)


class DeviationRule(Base):
    __tablename__ = "deviation_rules"
    code: Mapped[str] = mapped_column(String(32), primary_key=True)
    title: Mapped[str] = mapped_column(String(120))
    kind: Mapped[str] = mapped_column(String(16))                        # data|absence|unexpected|visual
    severity: Mapped[str] = mapped_column(String(10))
    confirmed_severity: Mapped[str] = mapped_column(String(10))
    message: Mapped[str] = mapped_column(Text)
    recommendation: Mapped[str] = mapped_column(Text)
    rules_version: Mapped[str] = mapped_column(String(32))


# ============================================================ 2. Проект, камеры, график

class Project(Base):
    __tablename__ = "projects"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    object_type: Mapped[str] = mapped_column(ForeignKey("object_types.key"), default="housing")
    address: Mapped[str] = mapped_column(String(300), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    timezone: Mapped[str] = mapped_column(String(40), default="Europe/Moscow")
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)

    zones: Mapped[list["Zone"]] = relationship(back_populates="project", cascade="all, delete-orphan",
                                               order_by="Zone.id")
    cameras: Mapped[list["Camera"]] = relationship(back_populates="project", cascade="all, delete-orphan",
                                                   order_by="Camera.id")


class Zone(Base):
    """Зона (захватка, участок) — единица сопоставления: этапы графика и техника на снимках."""
    __tablename__ = "zones"
    __table_args__ = (UniqueConstraint("project_id", "key"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    key: Mapped[str] = mapped_column(String(32))
    name: Mapped[str] = mapped_column(String(200))
    color: Mapped[str] = mapped_column(String(9), default="#2A78D6")
    project: Mapped[Project] = relationship(back_populates="zones")


class Camera(Base):
    __tablename__ = "cameras"
    __table_args__ = (UniqueConstraint("project_id", "key"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    key: Mapped[str] = mapped_column(String(32))
    name: Mapped[str] = mapped_column(String(200))
    source_url: Mapped[str] = mapped_column(String(500), default="")
    is_overview: Mapped[bool] = mapped_column(Boolean, default=False)    # общий план с высоты
    tiles: Mapped[int] = mapped_column(Integer, default=0)               # детекция по фрагментам 3×3
    default_zone_id: Mapped[int | None] = mapped_column(ForeignKey("zones.id"), nullable=True)
    note: Mapped[str] = mapped_column(Text, default="")
    project: Mapped[Project] = relationship(back_populates="cameras")
    zone_polygons: Mapped[list["CameraZone"]] = relationship(back_populates="camera", cascade="all, delete-orphan")


class CameraZone(Base):
    """Полигон зоны на кадре камеры в долях ширины и высоты (0..1) — не зависит от разрешения."""
    __tablename__ = "camera_zones"
    __table_args__ = (UniqueConstraint("camera_id", "zone_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"))
    zone_id: Mapped[int] = mapped_column(ForeignKey("zones.id", ondelete="CASCADE"))
    polygon: Mapped[list] = mapped_column(JSONType)
    camera: Mapped[Camera] = relationship(back_populates="zone_polygons")


class SiteEquipment(Base):
    """Заявленная общеплощадочная техника (башенный кран может не попадать в кадр зоны)."""
    __tablename__ = "site_equipment"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    equipment_class: Mapped[str] = mapped_column(ForeignKey("equipment_classes.key"))
    name: Mapped[str] = mapped_column(String(120))
    zone_keys: Mapped[list] = mapped_column(JSONType, default=list)      # пусто — все зоны
    active_from: Mapped[dt.date | None] = mapped_column(Date, nullable=True)
    active_to: Mapped[dt.date | None] = mapped_column(Date, nullable=True)


class ScheduleVersion(Base):
    __tablename__ = "schedule_versions"
    __table_args__ = (UniqueConstraint("project_id", "version"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    version: Mapped[int] = mapped_column(Integer)
    source_filename: Mapped[str] = mapped_column(String(300))
    is_baseline: Mapped[bool] = mapped_column(Boolean, default=False)
    rows_total: Mapped[int] = mapped_column(Integer, default=0)
    rows_matched: Mapped[int] = mapped_column(Integer, default=0)
    uploaded_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)


class ScheduleTask(Base):
    """Этап календарного графика. Вид работ справочника → профиль техники → проверки на снимках."""
    __tablename__ = "schedule_tasks"
    __table_args__ = (Index("ix_task_project_dates", "project_id", "start_date", "end_date"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    schedule_version_id: Mapped[int] = mapped_column(ForeignKey("schedule_versions.id", ondelete="CASCADE"))
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    row_no: Mapped[int] = mapped_column(Integer, default=0)
    wbs: Mapped[str] = mapped_column(String(64), default="")             # шифр/код в файле графика
    name: Mapped[str] = mapped_column(Text)
    work_type_id: Mapped[str | None] = mapped_column(ForeignKey("work_types.id"), nullable=True)
    match_method: Mapped[str] = mapped_column(String(16), default="none")  # code|name|manual|none
    match_score: Mapped[float] = mapped_column(Float, default=0.0)
    profile_id: Mapped[str | None] = mapped_column(ForeignKey("equipment_profiles.id"), nullable=True)
    zone_id: Mapped[int | None] = mapped_column(ForeignKey("zones.id"), nullable=True)  # NULL — вся площадка
    start_date: Mapped[dt.date] = mapped_column(Date)
    end_date: Mapped[dt.date] = mapped_column(Date)
    parent_task_id: Mapped[int | None] = mapped_column(ForeignKey("schedule_tasks.id"), nullable=True)
    is_summary: Mapped[bool] = mapped_column(Boolean, default=False)     # у этапа есть подэтапы в графике
    planned_equipment: Mapped[dict] = mapped_column(JSONType, default=dict)  # {класс: кол-во} из графика
    contractor: Mapped[str] = mapped_column(String(200), default="")


# ============================================================ 3. Наблюдения

class Snapshot(Base):
    __tablename__ = "snapshots"
    __table_args__ = (UniqueConstraint("project_id", "sha256"), Index("ix_snap_project_time", "project_id", "taken_at"))
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"))
    taken_at: Mapped[dt.datetime] = mapped_column(DateTime)
    time_source: Mapped[str] = mapped_column(String(16))                 # form|exif|filename|ocr|upload
    original_name: Mapped[str] = mapped_column(String(300), default="")
    file_path: Mapped[str] = mapped_column(String(500))
    sha256: Mapped[str] = mapped_column(String(64))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    brightness: Mapped[float] = mapped_column(Float)
    contrast: Mapped[float] = mapped_column(Float)
    sharpness: Mapped[float] = mapped_column(Float)
    quality_ok: Mapped[bool] = mapped_column(Boolean)
    quality_reason: Mapped[str] = mapped_column(String(200), default="")
    detector: Mapped[str] = mapped_column(String(120))                   # версия модели / demo
    assessment: Mapped[dict | None] = mapped_column(JSONType, nullable=True)  # стадия/готовность (ML-часть)
    processing_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    detections: Mapped[list["Detection"]] = relationship(back_populates="snapshot", cascade="all, delete-orphan",
                                                          order_by="Detection.id")


class Detection(Base):
    __tablename__ = "detections"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    snapshot_id: Mapped[int] = mapped_column(ForeignKey("snapshots.id", ondelete="CASCADE"), index=True)
    equipment_class: Mapped[str] = mapped_column(ForeignKey("equipment_classes.key"))
    confidence: Mapped[float] = mapped_column(Float)
    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    x2: Mapped[float] = mapped_column(Float)
    y2: Mapped[float] = mapped_column(Float)
    zone_id: Mapped[int | None] = mapped_column(ForeignKey("zones.id"), nullable=True)
    source: Mapped[str] = mapped_column(String(16), default="model")     # model|demo|manual
    is_rejected: Mapped[bool] = mapped_column(Boolean, default=False)    # инженер: ложная рамка
    snapshot: Mapped[Snapshot] = relationship(back_populates="detections")


class SnapshotCheck(Base):
    """Как снимок сопоставлен с графиком в зоне: этапы, ожидаемая и видимая техника, итог проверок."""
    __tablename__ = "snapshot_checks"
    __table_args__ = (UniqueConstraint("snapshot_id", "zone_key"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    snapshot_id: Mapped[int] = mapped_column(ForeignKey("snapshots.id", ondelete="CASCADE"), index=True)
    zone_key: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict] = mapped_column(JSONType)


# ============================================================ 4. Отклонения и нарушения

class Deviation(Base):
    __tablename__ = "deviations"
    __table_args__ = (UniqueConstraint("project_id", "key"), Index("ix_dev_project_day", "project_id", "day"))
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    key: Mapped[str] = mapped_column(String(200))                         # ключ дедупликации
    day: Mapped[dt.date] = mapped_column(Date)
    zone_id: Mapped[int | None] = mapped_column(ForeignKey("zones.id"), nullable=True)
    task_id: Mapped[int | None] = mapped_column(ForeignKey("schedule_tasks.id", ondelete="SET NULL"), nullable=True)
    rule_code: Mapped[str] = mapped_column(ForeignKey("deviation_rules.code"))
    equipment_class: Mapped[str | None] = mapped_column(String(32), nullable=True)
    severity: Mapped[str] = mapped_column(String(10))                     # critical|warning|info
    status: Mapped[str] = mapped_column(String(16))                       # preliminary|confirmed
    title: Mapped[str] = mapped_column(String(200))
    message: Mapped[str] = mapped_column(Text)
    recommendation: Mapped[str] = mapped_column(Text)
    metrics: Mapped[dict] = mapped_column(JSONType, default=dict)         # расчёт, объясняющий вывод
    first_seen: Mapped[dt.datetime] = mapped_column(DateTime)
    last_seen: Mapped[dt.datetime] = mapped_column(DateTime)
    snapshots_count: Mapped[int] = mapped_column(Integer, default=1)
    rules_version: Mapped[str] = mapped_column(String(32))
    review_status: Mapped[str | None] = mapped_column(String(16), nullable=True)  # accepted|rejected|force_majeure
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    evidence: Mapped[list["DeviationEvidence"]] = relationship(back_populates="deviation", cascade="all, delete-orphan",
                                                                order_by="DeviationEvidence.id")


class DeviationEvidence(Base):
    __tablename__ = "deviation_evidence"
    __table_args__ = (UniqueConstraint("deviation_id", "snapshot_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    deviation_id: Mapped[int] = mapped_column(ForeignKey("deviations.id", ondelete="CASCADE"), index=True)
    snapshot_id: Mapped[int] = mapped_column(ForeignKey("snapshots.id", ondelete="CASCADE"))
    detection_ids: Mapped[list] = mapped_column(JSONType, default=list)
    note: Mapped[str] = mapped_column(String(300), default="")
    deviation: Mapped[Deviation] = relationship(back_populates="evidence")


class DeviationReview(Base):
    __tablename__ = "deviation_reviews"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    deviation_id: Mapped[int] = mapped_column(ForeignKey("deviations.id", ondelete="CASCADE"), index=True)
    verdict: Mapped[str] = mapped_column(String(16))                      # accepted|rejected|force_majeure
    comment: Mapped[str] = mapped_column(Text, default="")
    reviewer: Mapped[str] = mapped_column(String(120), default="инженер")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)


class Violation(Base):
    """Нарушение: подтверждённое инженером отклонение, поставленное на контроль."""
    __tablename__ = "violations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    deviation_id: Mapped[int] = mapped_column(ForeignKey("deviations.id", ondelete="CASCADE"), unique=True)
    number: Mapped[str] = mapped_column(String(32))
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text)
    zone_id: Mapped[int | None] = mapped_column(ForeignKey("zones.id"), nullable=True)
    task_id: Mapped[int | None] = mapped_column(ForeignKey("schedule_tasks.id", ondelete="SET NULL"), nullable=True)
    contractor: Mapped[str] = mapped_column(String(200), default="")
    severity: Mapped[str] = mapped_column(String(10))
    status: Mapped[str] = mapped_column(String(10), default="open")      # open|closed
    registered_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    due_date: Mapped[dt.date | None] = mapped_column(Date, nullable=True)
    closed_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
