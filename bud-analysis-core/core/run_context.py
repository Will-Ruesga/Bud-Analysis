"""Path resolver for one run.

A `RunContext` is built once from (date, cultivar, backbone_name, task)
and exposes every output path the project uses. No other module hardcodes
paths — they all flow through here.
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunContext:
    """Identifies one run dir and resolves every path inside it.

    Frozen so it can be hashed and shared safely. Path resolution is pure
    (no I/O); validation is the caller's job. `task` may be None (accessing
    `task_dir` then raises), though a prepared run always records its task.
    """

    date: str
    cultivar: str
    backbone_name: str
    task: str | None
    backbone_checkpoint: str | None = None
    output_dir: str | None = None

    def __post_init__(self):
        # frozen=True blocks normal attribute assignment; bypass it to set
        # up a per-instance cache for data_dir() / index().
        object.__setattr__(self, "_cache", {})

    # --- constructors ---

    @classmethod
    def from_info_json(cls, run_dir: str) -> "RunContext":
        """Build a fully-resolved context from a prepared run's manifest.

        `run_dir` is the directory `prepare` created (e.g.
        `output/<date>-<cultivar>-<backbone>`). Reads `prep/info.json` for the
        run's identity, backbone, and checkpoint, so downstream stages
        (`train`, `export`) need only the path — never the task config.
        `output_dir` is set to the run dir's parent so `root` resolves back to
        exactly `run_dir`.
        """
        run_dir = Path(run_dir).resolve()
        info = json.loads((run_dir / "prep" / "info.json").read_text())
        ctx = cls(
            date=info["date"],
            cultivar=info["cultivar"],
            backbone_name=info["backbone_name"],
            task=info["task"],
            backbone_checkpoint=info["backbone_checkpoint"],
            output_dir=str(run_dir.parent),
        )
        ctx._cache["info"] = info  # type: ignore[attr-defined]  # prime the cache; avoid a re-read
        return ctx

    # --- resolved paths ---

    @property
    def root(self) -> Path:
        base = Path(self.output_dir) if self.output_dir else Path("output")
        return base / f"{self.date}-{self.cultivar}-{self.backbone_name}"

    @property
    def prep_dir(self) -> Path:
        return self.root / "prep"

    @property
    def index_csv(self) -> Path:
        return self.prep_dir / "index.csv"

    @property
    def prep_info_json(self) -> Path:
        return self.prep_dir / "info.json"

    @property
    def prep_distribution_png(self) -> Path:
        return self.prep_dir / "distribution.png"

    @property
    def embeddings_dir(self) -> Path:
        return self.root / "emb"

    @property
    def embeddings_meta_json(self) -> Path:
        return self.embeddings_dir / "meta.json"

    @property
    def task_dir(self) -> Path:
        if self.task is None:
            raise ValueError(
                "task_dir requires task to be set; this RunContext was "
                "built task-agnostically (task=None)."
            )
        return self.root / f"{self.task}-results"

    @property
    def onnx_dir(self) -> Path:
        return self.root / "onnx"

    # --- helpers ---

    def variant_dir(self, variant: str) -> Path:
        """One compared variant's kept-winner dir under the current task.

        `variant` is the value of the study's compared dimension (e.g. the loss
        name `mse`/`huber`), so the dir is `<task>-results/<variant>/`.
        """
        return self.task_dir / variant

    def embedding_path(self, image_id: str) -> Path:
        return self.embeddings_dir / f"{image_id}.npy"

    def info(self) -> dict:
        """Read `prep/info.json` once and cache the full run manifest in-process.

        The manifest is the single source of truth for a prepared run: data
        path, backbone, splits, and the snapshotted training config. `train`,
        and `export` read everything they need from here.
        """
        cache = self._cache  # type: ignore[attr-defined]
        if "info" not in cache:
            cache["info"] = json.loads(self.prep_info_json.read_text())
        return cache["info"]

    def data_dir(self) -> Path:
        """Absolute dataset path recorded in the run manifest (`prep/info.json`)."""
        return Path(self.info()["data_dir"])

    @property
    def image_size(self) -> int:
        """The run's frozen input image size (from `prep/info.json`).

        Set by `prepare` from the backbone's default; read here so extraction and
        export use the run's own size, not the live `backbones._VARIANTS` constant.
        Runs prepared before this field existed lack it — fail loud (re-prepare).
        """
        info = self.info()
        if "image_size" not in info:
            raise ValueError(
                f"{self.prep_info_json} has no 'image_size' — this run predates the "
                "frozen-size field; re-run prepare to record it."
            )
        return info["image_size"]

    @property
    def name(self) -> str:
        """The run's label, used in the ONNX filename — the run's `cultivar`."""
        return self.cultivar

    def index(self):
        """Read index.csv lazily and cache in-process.

        Returns the snake_case DataFrame from `data.read_index`. Imported
        locally to avoid a `run_context ↔ data` import cycle (`data.run`
        takes a ctx).
        """
        cache = self._cache  # type: ignore[attr-defined]
        if "index" not in cache:
            from core import data

            cache["index"] = data.read_index(self.index_csv)
        return cache["index"]
