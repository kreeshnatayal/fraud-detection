from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

from src.fraud_detection.utils.logging import get_logger
from src.fraud_detection.utils.metrics import safe_divide, safe_roc_auc

logger = get_logger()


def calculate_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    label: str = "",
) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "auc": safe_roc_auc(y_true, y_score),
        "ap": float(average_precision_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "log_loss": float(log_loss(y_true, y_score)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "fpr": safe_divide(fp, fp + tn),
        "threshold": threshold,
    }
    if label:
        logger.info(
            "[%s] AUC=%.4f  AP=%.4f  F1=%.4f  Recall=%.4f  Precision=%.4f",
            label, metrics["auc"], metrics["ap"], metrics["f1"],
            metrics["recall"], metrics["precision"],
        )
    return metrics


def optimise_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fn_cost: float = 15.0,
    fp_cost: float = 1.0,
) -> float:
    thresholds = np.linspace(0.01, 0.99, 200)
    best_thresh, best_cost = 0.5, float("inf")
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        cost = fn_cost * fn + fp_cost * fp
        if cost < best_cost:
            best_cost = cost
            best_thresh = t
    logger.info("Optimal threshold: %.4f  (cost=%.1f)", best_thresh, best_cost)
    return best_thresh


def calculate_fraud_cost(
    y_true: np.ndarray,
    y_score: np.ndarray,
    amounts: np.ndarray | None,
    threshold: float,
    fn_cost_mult: float = 15.0,
    fp_cost_mult: float = 1.0,
) -> float:
    y_pred = (y_score >= threshold).astype(int)
    fn_mask = (y_true == 1) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    if amounts is not None:
        return float(fn_cost_mult * amounts[fn_mask].sum() + fp_cost_mult * amounts[fp_mask].sum())
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fn_cost_mult * fn + fp_cost_mult * fp)
