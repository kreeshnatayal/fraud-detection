import gc
import os
import shutil
from contextlib import contextmanager

import numpy as np
import pandas as pd
import psutil

from src.fraud_detection.utils.logging import get_logger

logger = get_logger()


def get_available_memory_mb() -> float:
    return psutil.virtual_memory().available / (1024 ** 2)


def get_memory_usage_pct() -> float:
    return psutil.virtual_memory().percent


def get_disk_usage_mb(path: str | None = None) -> tuple:
    path = path or os.getcwd()
    try:
        usage = shutil.disk_usage(path)
        return usage.used / (1024 ** 2), usage.total / (1024 ** 2)
    except OSError:
        return 0.0, 0.0


@contextmanager
def memory_guard(threshold_pct: float = 85.0):
    yield
    if get_memory_usage_pct() > threshold_pct:
        gc.collect()
        logger.warning("Memory above %.0f%% — ran garbage collection.", threshold_pct)


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
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
