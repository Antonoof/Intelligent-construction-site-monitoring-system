from __future__ import annotations

from pydantic import BaseModel, Field


class SourceIn(BaseModel):
    path: str
    split: str = Field(default="train", pattern="^(train|val)$")


class ProjectIn(BaseModel):
    name: str
    sources: list[SourceIn]
    classes: list[str] | None = None


class OpenProjectIn(BaseModel):
    path: str


class PathIn(BaseModel):
    """A path on the machine running the app — submission, model or data.yaml."""

    path: str


class ClassesIn(BaseModel):
    names: list[str]


class SelectionIn(BaseModel):
    budget: int = Field(default=100, ge=1, le=100_000)
    clusters: int | None = Field(default=None, ge=1, le=2000)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    split: str | None = None
    weights: dict[str, float] | None = None
    scope: str = Field(default="all", pattern="^(all|unlabeled|labeled|issues)$")
    include_assigned: bool = False


class LabelBox(BaseModel):
    cls: int = 0
    x: float
    y: float
    w: float
    h: float


class LabelsIn(BaseModel):
    boxes: list[LabelBox]
    write_source: bool = False


class TrainConfigIn(BaseModel):
    config: dict


class BundleIn(BaseModel):
    config: dict | None = None


class TuneIn(BaseModel):
    models: list[str]
    k: int = Field(default=100, ge=1, le=5000)
    trials: int = Field(default=40, ge=1, le=1000)
    conf_range: tuple[float, float] = (0.005, 0.5)
    iou_range: tuple[float, float] = (0.3, 0.7)
    skip_range: tuple[float, float] = (0.001, 0.05)
    conf_type: str = Field(default="avg", pattern="^(avg|max|box_and_model_avg|absent_model_aware_avg)$")
    split: str | None = "val"
    device: str = "auto"


class InferBundleIn(BaseModel):
    include_weights: bool = False


class LabelsBatchIn(BaseModel):
    """Массовая правка: пометить пачку кадров пустыми, принять предразметку и т. п."""

    names: list[str]
    boxes: list[LabelBox] = []
    write_source: bool = False
    mark_done: bool = True


class FlagIn(BaseModel):
    status: str = Field(default="review", pattern="^(new|assigned|done|review)$")
    note: str | None = None


class QualityIn(BaseModel):
    check_classes: bool = True


class SplitIn(BaseModel):
    val_ratio: float | None = Field(default=0.2, ge=0.0, le=0.95)
    val_count: int | None = Field(default=None, ge=0)
    scope: str = Field(default="labeled", pattern="^(labeled|all)$")
    group_by: str = Field(default="series", pattern="^(none|series|folder)$")
    stratify: bool = True
    seed: int = 42


class ExportDatasetIn(BaseModel):
    link_mode: str = Field(default="hardlink", pattern="^(copy|hardlink|symlink)$")
    include_unlabeled: bool = False


class CandidatesIn(BaseModel):
    count: int = Field(default=200, ge=1, le=20_000)
    source: str = Field(default="unlabeled", pattern="^(unlabeled|issues|selection|all)$")
    split: str | None = None


class AssignmentIn(BaseModel):
    title: str = ""
    owner: str = ""
    note: str = ""
    images: list[str] | None = None
    count: int = Field(default=200, ge=1, le=20_000)
    source: str = Field(default="unlabeled", pattern="^(unlabeled|issues|selection|all)$")
    split: str | None = None


class AssignmentExportIn(BaseModel):
    mode: str = Field(default="manifest", pattern="^(manifest|pack)$")


class ImportIn(BaseModel):
    path: str
    assignment: str | None = None


class PrelabelIn(BaseModel):
    model: str
    conf: float = Field(default=0.25, ge=0.0, le=1.0)
    iou: float = Field(default=0.6, ge=0.05, le=0.95)
    imgsz: int = Field(default=960, ge=128, le=2048)
    device: str = "auto"
    scope: str = Field(default="unlabeled", pattern="^(all|unlabeled|labeled|issues|selection)$")
    split: str | None = None
    limit: int = Field(default=2000, ge=1, le=100_000)


class ApplyProposalsIn(BaseModel):
    names: list[str] | None = None
    conf: float = Field(default=0.5, ge=0.0, le=1.0)
    only_unlabeled: bool = True
    mark_done: bool = False
