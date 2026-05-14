import time
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from src.fraud_detection.utils.logging import get_logger

logger = get_logger()


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
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


def timer(label: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.info("%s ...", label)
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            logger.info("%s finished in %.1fs", label, time.perf_counter() - t0)
            return result
        return wrapper
    return decorator
