"""
Generates realistic sample output files so the Streamlit dashboard works
as a live demo without needing to run the full ML pipeline.

Run once before launching the dashboard in demo mode:
    python generate_sample_data.py
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Realistic transaction predictions
# ---------------------------------------------------------------------------

N = 5_000
fraud_mask = rng.random(N) < 0.035          # ~3.5% fraud rate — realistic for IEEE dataset

# Base features that would come out of the pipeline
amounts = rng.lognormal(mean=4.5, sigma=1.2, size=N).clip(0.5, 15_000)
hours = rng.integers(0, 24, size=N)
days = rng.integers(0, 7, size=N)

# Fraud transactions tend to: be larger, happen at night, rapid sequences
fraud_amounts = rng.lognormal(mean=5.5, sigma=1.4, size=fraud_mask.sum()).clip(1, 20_000)
amounts[fraud_mask] = fraud_amounts

# Build fraud probability scores (correlated with actual label + noise)
base_prob = rng.beta(0.5, 9, size=N)        # mostly low, heavy right-tail skew
base_prob[fraud_mask] = rng.beta(5, 3, size=fraud_mask.sum())
noise = rng.normal(0, 0.05, size=N)
fraud_prob = np.clip(base_prob + noise, 0.001, 0.999)

threshold = 0.42
predicted = (fraud_prob >= threshold).astype(int)
actual = fraud_mask.astype(int)

# Extra numeric features (what would be in the val predictions file)
card_freq = rng.integers(1, 500, size=N).astype(float)
amount_log = np.log1p(amounts)
is_night = ((hours >= 22) | (hours <= 6)).astype(int)
is_weekend = (days >= 5).astype(int)
time_since_last = rng.exponential(scale=3600, size=N)
amount_zscore = rng.normal(0, 1, size=N)
amount_zscore[fraud_mask] += rng.normal(2.5, 1, size=fraud_mask.sum())
isolation_score = rng.normal(-0.1, 0.08, size=N)
isolation_score[fraud_mask] -= rng.exponential(0.15, size=fraud_mask.sum())

email_domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "protonmail.com", "other.net"]
email_freq = rng.integers(10, 5000, size=N).astype(float)

df_preds = pd.DataFrame({
    "fraud_probability": fraud_prob.round(6),
    "predicted_fraud": predicted,
    "actual_fraud": actual,
    "TransactionAmt": amounts.round(2),
    "DT_hour": hours,
    "DT_day": days,
    "is_night": is_night,
    "is_weekend": is_weekend,
    "time_since_last_txn": time_since_last.round(1),
    "amount_log": amount_log.round(4),
    "amount_zscore": amount_zscore.round(4),
    "card1_frequency": card_freq,
    "isolation_score": isolation_score.round(4),
    "P_emaildomain_freq": email_freq,
    "is_large_payment": (amounts > 1000).astype(int),
    "is_micropayment": (amounts < 1.0).astype(int),
    "is_suspicious_amount": (amounts > 5000).astype(int),
})

df_preds.to_csv(OUTPUT_DIR / "val_predictions.csv", index=False)
print(f"Saved val_predictions.csv  ({len(df_preds):,} rows, {actual.sum()} frauds)")


# ---------------------------------------------------------------------------
# 2. Evaluation metrics
# ---------------------------------------------------------------------------

try:
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    SKL_OK = True
except ImportError:
    SKL_OK = False

if SKL_OK:
    auc = float(roc_auc_score(actual, fraud_prob))
    ap = float(average_precision_score(actual, fraud_prob))
    f1 = float(f1_score(actual, predicted, zero_division=0))
    prec = float(precision_score(actual, predicted, zero_division=0))
    rec = float(recall_score(actual, predicted, zero_division=0))
    ll = float(log_loss(actual, fraud_prob))
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
else:
    # fallback approximations when sklearn not installed
    auc, ap, f1, prec, rec, ll = 0.9631, 0.8201, 0.7902, 0.7988, 0.7818, 0.2341
    tp = int((actual == 1).sum() * 0.78)
    fn = int((actual == 1).sum()) - tp
    fp = int((actual == 0).sum() * 0.02)
    tn = int((actual == 0).sum()) - fp

metrics = {
    "auc": round(auc, 4),
    "ap": round(ap, 4),
    "f1": round(f1, 4),
    "precision": round(prec, 4),
    "recall": round(rec, 4),
    "log_loss": round(ll, 4),
    "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    "fpr": round(fp / max(fp + tn, 1), 4),
    "threshold": threshold,
}

with open(OUTPUT_DIR / "evaluation_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved evaluation_metrics.json")
print(f"  AUC={auc:.4f}  F1={f1:.4f}  Recall={rec:.4f}  Precision={prec:.4f}")


# ---------------------------------------------------------------------------
# 3. Per-model comparison metrics
# ---------------------------------------------------------------------------

models = {
    "lgbm":     {"auc": 0.9612, "f1": 0.7841, "precision": 0.7923, "recall": 0.7762, "ap": 0.8134},
    "xgb":      {"auc": 0.9588, "f1": 0.7714, "precision": 0.7811, "recall": 0.7620, "ap": 0.8002},
    "catboost": {"auc": 0.9631, "f1": 0.7902, "precision": 0.7988, "recall": 0.7818, "ap": 0.8201},
    "rf":       {"auc": 0.9342, "f1": 0.7213, "precision": 0.7401, "recall": 0.7034, "ap": 0.7689},
    "lr":       {"auc": 0.8921, "f1": 0.6847, "precision": 0.7012, "recall": 0.6690, "ap": 0.7211},
    "stacking": {"auc": round(auc, 4), "f1": round(f1, 4), "precision": round(prec, 4), "recall": round(rec, 4), "ap": round(ap, 4)},
    "validation": metrics,
}

with open(OUTPUT_DIR / "model_metrics.json", "w") as f:
    json.dump(models, f, indent=2)
print("Saved model_metrics.json")


# ---------------------------------------------------------------------------
# 4. Pipeline execution summary
# ---------------------------------------------------------------------------

summary = {
    "data_loading": {
        "status": "success", "elapsed_s": 12.4,
        "mem_start_mb": 8200, "mem_end_mb": 6900,
        "metrics": {"rows": 590_540, "cols": 394},
    },
    "feature_engineering": {
        "status": "success", "elapsed_s": 87.3,
        "mem_start_mb": 6900, "mem_end_mb": 5100,
        "metrics": {"cols": 512},
    },
    "graph_features": {
        "status": "success", "elapsed_s": 143.2,
        "mem_start_mb": 5100, "mem_end_mb": 4200,
        "metrics": {"cols": 608},
    },
    "model_training": {
        "status": "success", "elapsed_s": 612.7,
        "mem_start_mb": 4200, "mem_end_mb": 3800,
        "metrics": {"auc": 0.9631, "f1": 0.7902, "recall": 0.7818, "precision": 0.7988},
    },
    "evaluation": {
        "status": "success", "elapsed_s": 8.1,
        "mem_start_mb": 3800, "mem_end_mb": 3750,
        "metrics": metrics,
    },
}

with open(OUTPUT_DIR / "pipeline_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Saved pipeline_summary.json")
print("\nDemo data ready. Launch the dashboard with:")
print("  streamlit run dashboard.py")
