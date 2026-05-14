import gc
import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.fraud_detection.utils.logging import get_logger
from src.fraud_detection.utils.memory import get_available_memory_mb

logger = get_logger()


class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, stage: str) -> Path:
        return self.dir / f"{stage}.pkl"

    def exists(self, stage: str) -> bool:
        return self._path(stage).exists()

    def save(self, stage: str, data: Any, description: str = "") -> None:
        path = self._path(stage)
        with open(path, "wb") as f:
            pickle.dump({"data": data, "description": description, "ts": datetime.utcnow().isoformat()}, f)
        logger.info("[Checkpoint] Saved '%s' -> %s", stage, path)

    def load(self, stage: str) -> Any:
        path = self._path(stage)
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("[Checkpoint] Loaded '%s'", stage)
        return obj["data"]

    def delete(self, stage: str) -> None:
        p = self._path(stage)
        if p.exists():
            p.unlink()


class PipelineTracker:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stages: Dict[str, Dict] = {}
        self._active: Optional[str] = None
        self._start: float = 0.0

    def start_stage(self, name: str) -> None:
        self._active = name
        self._start = time.perf_counter()
        self.stages[name] = {"status": "running", "mem_start_mb": get_available_memory_mb()}
        logger.info("=" * 60)
        logger.info("STAGE START: %s", name)

    def end_stage(self, name: str, success: bool = True, metrics: Optional[Dict] = None) -> None:
        elapsed = time.perf_counter() - self._start
        entry = self.stages.get(name, {})
        entry.update({
            "status": "success" if success else "failed",
            "elapsed_s": round(elapsed, 2),
            "mem_end_mb": get_available_memory_mb(),
        })
        if metrics:
            entry["metrics"] = metrics
        self.stages[name] = entry
        logger.info("STAGE END: %s [%s] %.1fs", name, "OK" if success else "FAIL", elapsed)
        self._save_summary()

    def record_metric(self, key: str, value: Any) -> None:
        if self._active and self._active in self.stages:
            self.stages[self._active].setdefault("metrics", {})[key] = value

    def _save_summary(self) -> None:
        path = self.output_dir / "pipeline_summary.json"
        with open(path, "w") as f:
            json.dump(self.stages, f, indent=2, default=str)

    def get_summary(self) -> Dict:
        return self.stages
