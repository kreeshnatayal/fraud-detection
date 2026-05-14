from typing import Any, Dict, Optional

import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix, roc_auc_score

from src.fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


def quant_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    amounts: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    avg_txn_value: float = 200.0,
    annual_volume: int = 500_000,
    investigation_cost: float = 5.0,
    fp_revenue_loss_rate: float = 0.85,
) -> Dict[str, Any]:
    auc = float(roc_auc_score(y_true, y_score))
    gini = round(2 * auc - 1, 4)

    fraud_scores = y_score[y_true == 1]
    legit_scores = y_score[y_true == 0]
    ks_stat, _ = stats.ks_2samp(fraud_scores, legit_scores)
    ks_stat = round(float(ks_stat), 4)

    n_bins = 10
    bins = np.percentile(y_score, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    if len(bins) < 2:
        iv = 0.0
    else:
        eps = 1e-6
        iv = 0.0
        for i in range(len(bins) - 1):
            mask = (y_score >= bins[i]) & (y_score < bins[i + 1])
            if mask.sum() == 0:
                continue
            good = (y_true[mask] == 0).sum() / max((y_true == 0).sum(), 1)
            bad = (y_true[mask] == 1).sum() / max((y_true == 1).sum(), 1)
            woe = np.log((bad + eps) / (good + eps))
            iv += (bad - good) * woe
        iv = round(float(iv), 4)

    iv_strength = (
        "Very Strong" if iv > 0.5 else
        "Strong" if iv > 0.3 else
        "Medium" if iv > 0.1 else
        "Weak"
    )

    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    if amounts is not None:
        fraud_saved = float(amounts[(y_true == 1) & (y_pred == 1)].sum())
        fraud_missed = float(amounts[(y_true == 1) & (y_pred == 0)].sum())
        fp_blocked_amt = float(amounts[(y_true == 0) & (y_pred == 1)].sum())
    else:
        fraud_saved = tp * avg_txn_value
        fraud_missed = fn * avg_txn_value
        fp_blocked_amt = fp * avg_txn_value

    investigation_cost_total = (tp + fp) * investigation_cost
    revenue_loss_from_fp = fp_blocked_amt * fp_revenue_loss_rate
    net_benefit = fraud_saved - investigation_cost_total - revenue_loss_from_fp

    n_samples = len(y_true)
    scale = annual_volume / max(n_samples, 1)
    annual_fraud_prevented = round(fraud_saved * scale, 0)
    annual_net_benefit = round(net_benefit * scale, 0)

    n_top = max(1, int(len(y_score) * 0.05))
    top_idx = np.argsort(y_score)[::-1][:n_top]
    lift_at_5pct = round(y_true[top_idx].mean() / max(y_true.mean(), 1e-6), 2)

    return {
        "auc": round(auc, 4),
        "threshold": round(threshold, 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "gini_coefficient": gini,
        "ks_statistic": ks_stat,
        "information_value": iv,
        "iv_strength": iv_strength,
        "lift_at_top_5pct": lift_at_5pct,
        "fraud_dollars_caught": round(fraud_saved, 2),
        "fraud_dollars_missed": round(fraud_missed, 2),
        "fp_revenue_blocked": round(fp_blocked_amt, 2),
        "investigation_cost_total": round(investigation_cost_total, 2),
        "net_benefit_usd": round(net_benefit, 2),
        "estimated_annual_fraud_prevented_usd": annual_fraud_prevented,
        "estimated_annual_net_benefit_usd": annual_net_benefit,
        "annual_volume_assumption": annual_volume,
    }
