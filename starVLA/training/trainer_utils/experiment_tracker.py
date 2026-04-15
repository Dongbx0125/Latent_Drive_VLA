from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExperimentTracker:
    backend: str = "none"
    _client: Any = None
    _enabled: bool = False

    def init(self, cfg, output_dir: str | Path, run_group: str) -> None:
        trackers = getattr(cfg, "trackers", []) or []
        if isinstance(trackers, str):
            trackers = [t.strip() for t in trackers.split(",") if t.strip()]
        else:
            trackers = list(trackers)
        if not trackers:
            trackers = ["swanlab"]
        if "swanlab" not in trackers:
            return

        try:
            from starVLA.training.trainer_utils import wb_compat as swanlab
        except ImportError:
            logger.warning("SwanLab package not available, tracker disabled.")
            return

        swanlab.init(
            name=cfg.run_id,
            dir=str(Path(output_dir) / "swanlab"),
            project=getattr(cfg, "swanlab_project", None),
            entity=getattr(cfg, "swanlab_workspace", None),
            group=run_group,
        )
        self.backend = "swanlab"
        self._client = swanlab
        self._enabled = True

    def log(self, metrics: dict, step: int) -> None:
        if not self._enabled:
            return
        self._client.log(metrics, step=step)

    def finish(self) -> None:
        if not self._enabled:
            return
        self._client.finish()
