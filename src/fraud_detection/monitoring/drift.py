from typing import Any, Dict

import numpy as np
from scipy import stats

from src.fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    quantiles = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    quantiles = np.unique(quantiles)
    if len(quantiles) < 2:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=quantiles)
    cur_counts, _ = np.histogram(current, bins=quantiles)

    ref_pct = (ref_counts + eps) / (len(reference) + eps * n_bins)
    cur_pct = (cur_counts + eps) / (len(current) + eps * n_bins)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(psi, 4)


def psi_severity(psi: float) -> str:
    if psi < 0.10:
        return "STABLE"
    elif psi < 0.25:
        return "MONITOR"
    return "RETRAIN"


def ks_score_drift(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    stat, p_value = stats.ks_2samp(reference_scores, current_scores)
    drift_detected = p_value < alpha
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p_value), 6),
        "drift_detected": drift_detected,
        "severity": "DRIFT DETECTED" if drift_detected else "NO DRIFT",
        "alpha": alpha,
    }
