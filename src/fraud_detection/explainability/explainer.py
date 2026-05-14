import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.fraud_detection.utils import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


REASON_CATALOGUE: Dict[str, str] = {
    "amount_zscore":            "Transaction amount is unusually high relative to card history",
    "is_rapid_txn":             "Multiple transactions detected within a short time window",
    "time_since_last_txn":      "Unusually short time since previous transaction",
    "is_night":                 "Transaction occurred during high-risk overnight hours",
    "is_weekend":               "Transaction occurred on a weekend — elevated risk period",
    "is_suspicious_amount":     "Transaction amount exceeds high-risk threshold ($5,000)",
    "is_large_payment":         "Transaction amount is in the large-payment tier",
    "is_micropayment":          "Micro-payment pattern — possible account probing",
    "card1_frequency":          "Card has low prior transaction frequency",
    "card1_fraud_rate":         "High historical fraud rate associated with this card",
    "P_emaildomain_is_rare":    "Payer email domain rarely seen in transaction history",
    "card1_graph_pagerank":     "Card is a high-centrality node in the transaction graph",
    "card1_graph_degree":       "Card is connected to an unusually large number of entities",
    "isolation_score":          "Transaction is a statistical outlier (Isolation Forest signal)",
    "lof_score":                "Transaction is in a low-density region (Local Outlier Factor)",
    "is_mobile":                "Transaction initiated from a mobile device — higher risk channel",
    "P_emaildomain_is_free":    "Payer uses a free email provider (lower identity assurance)",
    "amount_percentile":        "Transaction amount is in the top percentile for this portfolio",
    "is_amount_outlier":        "Transaction amount deviates more than 3σ from card baseline",
    "rolling_txn_count_1d":     "Elevated transaction velocity in the past 24 hours",
    "rolling_txn_count_3d":     "Elevated transaction velocity in the past 3 days",
}

DEFAULT_REASON = "Statistical anomaly detected in transaction pattern"


class FraudExplainer:
    def __init__(
        self,
        pipeline,
        feature_names: List[str],
        background_data: Optional[pd.DataFrame] = None,
    ):
        self.pipeline = pipeline
        self.feature_names = feature_names
        self.background_data = background_data
        self._shap_explainers: Dict[str, Any] = {}
        self._build_explainers()

    def _build_explainers(self) -> None:
        try:
            import shap
        except ImportError:
            logger.warning("shap not installed — explanations will use fallback feature importance.")
            return

        for name, model in self.pipeline.models.items():
            try:
                explainer = shap.TreeExplainer(model)
                self._shap_explainers[name] = explainer
                logger.info("SHAP TreeExplainer built for model: %s", name)
            except Exception as exc:
                logger.warning("Could not build SHAP explainer for %s: %s", name, exc)

    def explain(
        self,
        X: pd.DataFrame,
        top_k: int = 5,
        avg_transaction_value: float = 200.0,
    ) -> Dict[str, Any]:
        X = X[self.feature_names].fillna(0)
        prob = float(self.pipeline.predict_proba(X)[0]) if len(X) == 1 else self.pipeline.predict_proba(X)

        if isinstance(prob, float):
            shap_vals, top_features = self._compute_shap(X)
            reason_codes, action_codes = self._generate_reason_codes(top_features)

            amt = float(X["TransactionAmt"].iloc[0]) if "TransactionAmt" in X.columns else avg_transaction_value
            expected_loss = round(prob * amt, 2)

            risk_tier = self._risk_tier(prob)
            decision = self._decision(prob, self.pipeline.threshold)

            return {
                "fraud_probability": round(prob, 4),
                "risk_tier": risk_tier,
                "decision": decision,
                "expected_loss_usd": expected_loss,
                "shap_values": shap_vals,
                "top_features": top_features[:top_k],
                "reason_codes": reason_codes[:top_k],
                "adverse_action_codes": action_codes[:top_k],
                "model_version": getattr(self.pipeline, "model_version", "v1.0.0"),
            }
        else:
            return {"fraud_probabilities": prob.tolist()}

    def _compute_shap(
        self, X: pd.DataFrame
    ) -> Tuple[Dict[str, float], List[Tuple[str, float]]]:
        if not self._shap_explainers:
            return self._fallback_importance(X)

        all_shap = []
        for name, explainer in self._shap_explainers.items():
            try:
                sv = explainer.shap_values(X)
                if isinstance(sv, list):
                    sv = sv[1]
                all_shap.append(sv[0])
            except Exception as exc:
                logger.warning("SHAP failed for %s: %s", name, exc)

        if not all_shap:
            return self._fallback_importance(X)

        mean_shap = np.mean(all_shap, axis=0)
        shap_dict = dict(zip(self.feature_names, mean_shap.tolist()))
        sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        return shap_dict, sorted_features

    def _fallback_importance(
        self, X: pd.DataFrame
    ) -> Tuple[Dict[str, float], List[Tuple[str, float]]]:
        vals = X.iloc[0].to_dict()
        sorted_feats = sorted(vals.items(), key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0, reverse=True)
        return vals, [(k, float(v)) for k, v in sorted_feats if isinstance(v, (int, float))]

    @staticmethod
    def _generate_reason_codes(
        top_features: List[Tuple[str, float]],
    ) -> Tuple[List[str], List[str]]:
        reasons, codes = [], []
        for feat, val in top_features:
            if val > 0:
                text = REASON_CATALOGUE.get(feat, DEFAULT_REASON)
                reasons.append(text)
                codes.append(feat.upper()[:20])
        return reasons, codes

    @staticmethod
    def _risk_tier(prob: float) -> str:
        if prob >= 0.80:
            return "CRITICAL"
        elif prob >= 0.50:
            return "HIGH"
        elif prob >= 0.20:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _decision(prob: float, threshold: float) -> str:
        if prob >= threshold * 1.5:
            return "BLOCK"
        elif prob >= threshold * 0.7:
            return "REVIEW"
        return "APPROVE"

    def global_importance(self, X: pd.DataFrame, n_samples: int = 500) -> pd.DataFrame:
        if not self._shap_explainers:
            logger.warning("No SHAP explainers available for global importance.")
            return pd.DataFrame()

        sample = X[self.feature_names].fillna(0).sample(n=min(n_samples, len(X)), random_state=42)

        all_shap = []
        for explainer in self._shap_explainers.values():
            try:
                sv = explainer.shap_values(sample)
                if isinstance(sv, list):
                    sv = sv[1]
                all_shap.append(np.abs(sv).mean(axis=0))
            except Exception:
                pass

        if not all_shap:
            return pd.DataFrame()

        mean_importance = np.mean(all_shap, axis=0)
        return (
            pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": mean_importance})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

    def calibration_report(
        self, y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10
    ) -> Dict[str, Any]:
        bins = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bins[:-1]
        bin_uppers = bins[1:]

        ece = 0.0
        reliability = []
        for lower, upper in zip(bin_lowers, bin_uppers):
            mask = (y_score >= lower) & (y_score < upper)
            if mask.sum() == 0:
                continue
            frac_pos = y_true[mask].mean()
            mean_prob = y_score[mask].mean()
            weight = mask.sum() / len(y_score)
            ece += weight * abs(frac_pos - mean_prob)
            reliability.append({
                "bin_lower": round(float(lower), 2),
                "bin_upper": round(float(upper), 2),
                "mean_predicted_prob": round(float(mean_prob), 4),
                "fraction_positives": round(float(frac_pos), 4),
                "count": int(mask.sum()),
            })

        return {
            "ece": round(float(ece), 4),
            "calibration_quality": "Well-calibrated" if ece < 0.05 else "Needs calibration",
            "n_bins": n_bins,
            "reliability_diagram": reliability,
        }
