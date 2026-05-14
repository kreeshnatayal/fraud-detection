import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.fraud_detection.monitoring.drift import (
    compute_psi,
    ks_score_drift,
    psi_severity,
)
from src.fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


class ModelMonitor:
    def __init__(self, config):
        self.config = config
        self.audit_log_path = Path(config.OUTPUT_DIR) / "audit_log.jsonl"

    def compute_feature_psi(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        features: List[str],
        n_bins: int = 10,
    ) -> pd.DataFrame:
        rows = []
        for feat in features:
            if feat not in reference_df.columns or feat not in current_df.columns:
                continue
            ref = reference_df[feat].dropna().values
            cur = current_df[feat].dropna().values
            if len(ref) == 0 or len(cur) == 0:
                continue
            psi_val = compute_psi(ref, cur, n_bins=n_bins)
            rows.append({
                "feature": feat,
                "psi": psi_val,
                "severity": psi_severity(psi_val),
            })

        df = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
        n_retrain = (df["severity"] == "RETRAIN").sum()
        n_monitor = (df["severity"] == "MONITOR").sum()
        logger.info(
            "PSI report: %d features RETRAIN, %d MONITOR, %d STABLE",
            n_retrain, n_monitor, len(df) - n_retrain - n_monitor,
        )
        return df

    def score_drift_report(
        self,
        reference_scores: np.ndarray,
        current_scores: np.ndarray,
    ) -> Dict[str, Any]:
        psi_val = compute_psi(reference_scores, current_scores)
        ks = ks_score_drift(reference_scores, current_scores, alpha=self.config.KS_ALPHA)
        return {
            "score_psi": psi_val,
            "score_psi_severity": psi_severity(psi_val),
            **ks,
            "overall_status": (
                "ACTION REQUIRED"
                if ks["drift_detected"] or psi_val > self.config.PSI_THRESHOLD
                else "OK"
            ),
        }

    def generate_model_card(
        self,
        pipeline,
        quant_report: Dict[str, Any],
        training_data_stats: Optional[Dict] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        card = {
            "model_details": {
                "name": self.config.MODEL_NAME,
                "version": self.config.MODEL_VERSION,
                "type": "Stacking Ensemble (LightGBM + XGBoost + CatBoost + RandomForest)",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "framework": "scikit-learn / LightGBM / XGBoost / CatBoost",
                "training_dataset": "IEEE-CIS Fraud Detection (Kaggle)",
                "intended_use": "Real-time transaction fraud scoring for payment card networks",
                "out_of_scope": "Cryptocurrency, wire transfers, insurance fraud",
            },
            "performance_metrics": quant_report,
            "training_data": training_data_stats or {},
            "ethical_considerations": {
                "fairness_evaluation": "Not yet conducted — recommend bias audit across demographic proxies",
                "adverse_action_compliance": "Reason codes provided per ECOA / GDPR Article 22",
                "model_explainability": "SHAP TreeExplainer — individual + global attributions available",
            },
            "monitoring": {
                "drift_detection": "PSI + KS test on score distribution",
                "retraining_trigger": f"PSI > {self.config.PSI_THRESHOLD} or KS p-value < {self.config.KS_ALPHA}",
                "performance_degradation_trigger": f"AUC drop > {self.config.DRIFT_THRESHOLDS['performance']}",
            },
            "limitations": [
                "Trained on synthetic data — production performance may differ",
                "Graph features assume stable entity relationships over time",
                "SMOTE oversampling may not reflect true minority distribution",
            ],
        }

        if output_path:
            with open(output_path, "w") as f:
                json.dump(card, f, indent=2, default=str)
            logger.info("Model card written to %s", output_path)

        return card

    def log_prediction(
        self,
        transaction_id: str,
        fraud_probability: float,
        decision: str,
        model_version: str,
        features: Optional[Dict] = None,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "transaction_id": transaction_id,
            "fraud_probability": round(fraud_probability, 6),
            "decision": decision,
            "model_version": model_version,
        }
        if features:
            record["key_features"] = {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in list(features.items())[:10]
            }

        with open(self.audit_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
