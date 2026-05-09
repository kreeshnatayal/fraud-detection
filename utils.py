"""
Shared utilities: logging, memory management, disk usage, timing, checkpointing.
"""

import gc
import json
import logging
import os
import pickle
import shutil
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str = "fraud_detection") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger()


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def get_available_memory_mb() -> float:
    """Available system RAM in MB."""
    return psutil.virtual_memory().available / (1024 ** 2)


def get_memory_usage_pct() -> float:
    """Current system RAM usage as a percentage (0-100)."""
    return psutil.virtual_memory().percent


def get_disk_usage_mb(path: Optional[str] = None) -> tuple:
    """Return (used_mb, total_mb) for the given path (defaults to cwd)."""
    path = path or os.getcwd()
    try:
        usage = shutil.disk_usage(path)
        return usage.used / (1024 ** 2), usage.total / (1024 ** 2)
    except OSError:
        return 0.0, 0.0


@contextmanager
def memory_guard(threshold_pct: float = 85.0):
    """Context manager that forces gc.collect() when memory exceeds threshold."""
    yield
    if get_memory_usage_pct() > threshold_pct:
        gc.collect()
        logger.warning("Memory above %.0f%% — ran garbage collection.", threshold_pct)


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Downcast numeric columns to their smallest sufficient dtype."""
    start_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == object:
            df[col] = df[col].astype("category")
        elif pd.api.types.is_integer_dtype(dtype):
            col_min, col_max = df[col].min(), df[col].max()
            for t in [np.int8, np.int16, np.int32, np.int64]:
                if np.iinfo(t).min <= col_min and col_max <= np.iinfo(t).max:
                    df[col] = df[col].astype(t)
                    break
        elif pd.api.types.is_float_dtype(dtype):
            df[col] = pd.to_numeric(df[col], downcast="float")
    end_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    if verbose:
        logger.info("Memory reduced %.1f MB -> %.1f MB (%.0f%%)", start_mb, end_mb, 100 * (1 - end_mb / max(start_mb, 1)))
    return df


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def timer(label: str):
    """Decorator that logs elapsed time for a function call."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.info("%s ...", label)
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            logger.info("%s finished in %.1fs", label, time.perf_counter() - t0)
            return result
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC that returns 0.5 when only one class is present."""
    try:
        if len(np.unique(y_true)) < 2:
            return 0.5
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def safe_metric_str(value: Any, fmt: str = ".4f") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# Encoding / scaling
# ---------------------------------------------------------------------------

def safe_category_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode categoricals and StandardScale numerics. Returns a copy."""
    df = df.copy()
    dt_cols = df.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
    if dt_cols:
        logger.warning("Dropping datetime columns: %s", dt_cols)
        df = df.drop(columns=dt_cols)
    for col in df.select_dtypes(include=["object", "category", "string"]).columns:
        df[col] = df[col].fillna("missing").astype("category").cat.codes.astype("int32")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols].fillna(df[num_cols].median()))
    return df


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Saves and loads pipeline stage results to/from disk."""

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


# ---------------------------------------------------------------------------
# Pipeline tracker
# ---------------------------------------------------------------------------

class PipelineTracker:
    """Tracks stage timing, metrics, and memory snapshots."""

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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(transaction_csv: str, identity_csv: Optional[str] = None) -> pd.DataFrame:
    """Load and merge transaction + identity CSVs with memory optimisation."""
    logger.info("Loading transactions from %s", transaction_csv)
    df = pd.read_csv(transaction_csv, index_col=0)
    logger.info("Loaded %d transactions", len(df))

    if identity_csv and os.path.exists(identity_csv):
        logger.info("Loading identity from %s", identity_csv)
        df_id = pd.read_csv(identity_csv, index_col=0)
        df = df.merge(df_id, left_index=True, right_index=True, how="left")
        logger.info("Merged -> %d rows", len(df))
    else:
        logger.warning("Identity CSV not found; proceeding without it.")

    df = reduce_memory_usage(df)
    return df
