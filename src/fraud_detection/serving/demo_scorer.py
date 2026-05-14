from typing import Any, Dict, List

import numpy as np

from src.fraud_detection.config import Config


def demo_score_transaction(
    amount: float,
    hour: int = 12,
    email_domain: str | None = None,
    device_info: str | None = None,
    rapid_sequence: bool = False,
    is_weekend: bool = False,
) -> Dict[str, Any]:
    score = 0.05
    reasons: List[str] = []

    if amount > 5000:
        score += 0.35
        reasons.append("Transaction amount exceeds high-risk threshold ($5,000)")
    elif amount > 1000:
        score += 0.12
        reasons.append("Transaction amount is in the large-payment tier")

    if hour >= 22 or hour <= 5:
        score += 0.22
        reasons.append("Transaction occurred during high-risk overnight hours")

    if rapid_sequence:
        score += 0.25
        reasons.append("Multiple transactions detected within a short time window")

    if email_domain in (None, "unknown"):
        score += 0.12
        reasons.append("Payer email domain rarely seen in transaction history")
    elif email_domain in ("gmail.com", "yahoo.com", "hotmail.com"):
        score += 0.03
        reasons.append("Payer uses a free email provider (lower identity assurance)")

    if device_info and any(k in device_info.lower() for k in ["android", "mobile", "ios"]):
        score += 0.06
        reasons.append("Transaction initiated from a mobile device — higher risk channel")

    if is_weekend:
        score += 0.04
        reasons.append("Transaction occurred on a weekend")

    score = min(round(score, 4), 0.99)
    tier = (
        "CRITICAL" if score >= 0.80 else
        "HIGH" if score >= 0.50 else
        "MEDIUM" if score >= 0.20 else "LOW"
    )
    decision = "BLOCK" if score >= 0.60 else "REVIEW" if score >= 0.30 else "PASS"

    if not reasons:
        reasons.append("Transaction pattern is within normal bounds")

    return {
        "fraud_probability": score,
        "risk_tier": tier,
        "decision": decision,
        "expected_loss_usd": round(score * amount, 2),
        "reason_codes": reasons,
        "model_version": f"{Config.MODEL_VERSION}-demo",
    }
