"""
FastAPI application for fraud scoring.
Run: uvicorn src.fraud_detection.serving.api:app
"""
from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import RedirectResponse
    from pydantic import BaseModel, Field, field_validator, ConfigDict
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from src.fraud_detection.config import Config
from src.fraud_detection.utils import get_logger
from src.fraud_detection.serving.demo_scorer import demo_score_transaction

logger = get_logger(__name__)

_PIPELINE = None
_EXPLAINER = None
_QUANT_METRICS: Dict = {}
_MODEL_CARD: Dict = {}


def _load_artifacts() -> None:
    global _PIPELINE, _EXPLAINER, _QUANT_METRICS, _MODEL_CARD

    pipeline_path = Path(Config.OUTPUT_DIR) / "pipeline.pkl"
    metrics_path = Path(Config.OUTPUT_DIR) / "evaluation_metrics.json"
    model_card_path = Path(Config.OUTPUT_DIR) / "model_card.json"

    if pipeline_path.exists():
        try:
            from src.fraud_detection.models.trained_pipeline import TrainedPipeline
            _PIPELINE = TrainedPipeline.load(str(pipeline_path))
            logger.info("Pipeline loaded from %s", pipeline_path)

            from src.fraud_detection.explainability import FraudExplainer
            _EXPLAINER = FraudExplainer(_PIPELINE, _PIPELINE.feature_names)
            logger.info("Explainer initialised.")
        except Exception as exc:
            logger.warning("Could not load pipeline: %s — running in demo mode.", exc)
    else:
        logger.warning("No pipeline.pkl found at %s — running in demo mode.", pipeline_path)

    if metrics_path.exists():
        with open(metrics_path) as f:
            _QUANT_METRICS = json.load(f)

    if model_card_path.exists():
        with open(model_card_path) as f:
            _MODEL_CARD = json.load(f)


class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., description="Unique transaction identifier")
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    card_id: Optional[str] = Field(None, description="Anonymised card identifier (card1)")
    timestamp: Optional[int] = Field(None, description="Unix timestamp (TransactionDT)")
    email_domain: Optional[str] = Field(None, description="Payer email domain")
    device_info: Optional[str] = Field(None, description="Device information string")
    addr1: Optional[float] = Field(None, description="Billing address region code")
    addr2: Optional[float] = Field(None, description="Billing country code")

    @classmethod
    @field_validator("amount")
    def amount_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("amount must be positive")
        return v

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "transaction_id": "TXN-2024-001",
            "amount": 349.99,
            "card_id": "card_abc123",
            "timestamp": 86400,
            "email_domain": "gmail.com",
            "device_info": "Android 11",
            "addr1": 315.0,
            "addr2": 87.0,
        }
    })


class PredictRequest(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    hour: Optional[int] = Field(None, ge=0, le=23, description="Hour of day (0-23)")
    card1: Optional[int] = Field(None, description="Card identifier (anonymised)")
    email_domain: Optional[str] = Field(None, description="Payer email domain")
    device_info: Optional[str] = Field(None, description="Device type / OS string")
    is_weekend: Optional[bool] = Field(None, description="Whether transaction is on a weekend")
    rapid_sequence: Optional[bool] = Field(False, description="Transaction within 5 min of last")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "amount": 150.0,
            "hour": 2,
            "card1": 12345,
            "email_domain": "gmail.com",
            "device_info": "Android Mobile",
            "is_weekend": False,
            "rapid_sequence": False,
        }
    })


class PredictResponse(BaseModel):
    fraud_probability: float
    decision: str
    risk_tier: str
    expected_loss_usd: float
    reason: List[str]
    model_version: str
    latency_ms: float


class ScoreResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    risk_tier: str
    decision: str
    expected_loss_usd: float
    reason_codes: List[str]
    model_version: str
    latency_ms: float


class BatchRequest(BaseModel):
    transactions: List[TransactionRequest] = Field(..., max_length=1000)


class BatchScoreResponse(BaseModel):
    results: List[ScoreResponse]
    total_expected_loss_usd: float
    high_risk_count: int
    processing_time_ms: float


def _request_to_features(txn: TransactionRequest, feature_names: List[str]) -> pd.DataFrame:
    raw: Dict[str, Any] = {
        "TransactionAmt": txn.amount,
        "card1": txn.card_id or "unknown",
        "TransactionDT": txn.timestamp or 86400,
        "P_emaildomain": txn.email_domain or "unknown",
        "DeviceInfo": txn.device_info or "unknown",
        "addr1": txn.addr1 or 0.0,
        "addr2": txn.addr2 or 0.0,
    }

    amount = txn.amount
    raw["amount_log"] = float(np.log1p(amount))
    raw["amount_sqrt"] = float(np.sqrt(max(amount, 0)))
    raw["is_micropayment"] = int(amount < 1.0)
    raw["is_large_payment"] = int(amount > 1_000)
    raw["is_suspicious_amount"] = int(amount > 5_000)
    raw["amount_percentile"] = 0.5

    if txn.timestamp is not None:
        dt = txn.timestamp
        hour = (dt // 3600) % 24
        day = (dt // (3600 * 24)) % 7
        raw["DT_hour"] = hour
        raw["DT_day"] = day
        raw["DT_hour_sin"] = float(np.sin(2 * np.pi * hour / 24))
        raw["DT_hour_cos"] = float(np.cos(2 * np.pi * hour / 24))
        raw["DT_day_sin"] = float(np.sin(2 * np.pi * day / 7))
        raw["DT_day_cos"] = float(np.cos(2 * np.pi * day / 7))
        raw["is_night"] = int(hour >= 22 or hour <= 6)
        raw["is_weekend"] = int(day >= 5)

    if txn.email_domain:
        FREE_DOMAINS = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com"}
        raw["P_emaildomain_is_free"] = int(txn.email_domain in FREE_DOMAINS)

    if txn.device_info:
        raw["is_mobile"] = int(any(kw in txn.device_info.lower() for kw in ["android", "ios", "mobile"]))

    row = pd.DataFrame([raw])
    for col in feature_names:
        if col not in row.columns:
            row[col] = 0.0
    return row[feature_names].fillna(0)


def _demo_score(txn: TransactionRequest) -> Dict[str, Any]:
    hour = ((txn.timestamp or 86400) // 3600) % 24 if txn.timestamp else 12
    return demo_score_transaction(
        amount=txn.amount,
        hour=hour,
        email_domain=txn.email_domain,
        device_info=txn.device_info,
        rapid_sequence=False,
        is_weekend=False,
    )


def _demo_predict(req: PredictRequest) -> Dict[str, Any]:
    return demo_score_transaction(
        amount=req.amount,
        hour=req.hour or 12,
        email_domain=req.email_domain,
        device_info=req.device_info,
        rapid_sequence=req.rapid_sequence or False,
        is_weekend=req.is_weekend or False,
    )


if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Fraud Risk Intelligence API",
        description="Real-time transaction fraud scoring with SHAP-based explanations.",
        version=Config.MODEL_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @asynccontextmanager
    async def lifespan_event(app: FastAPI):
        _load_artifacts()
        logger.info("Fraud Risk API started — version %s", Config.MODEL_VERSION)
        yield

    app.router.lifespan_context = lifespan_event

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/docs")

    @app.get("/health", tags=["System"])
    def health():
        return {
            "status": "ok",
            "model_loaded": _PIPELINE is not None,
            "model_version": Config.MODEL_VERSION,
            "timestamp": time.time(),
        }

    @app.get("/model/info", tags=["System"])
    def model_info():
        return {
            "model_name": Config.MODEL_NAME,
            "version": Config.MODEL_VERSION,
            "champion_model": Config.CHAMPION_MODEL,
            "features": _PIPELINE.feature_names if _PIPELINE else [],
            "threshold": _PIPELINE.threshold if _PIPELINE else 0.5,
            "model_card": _MODEL_CARD or {"note": "Model card not generated yet."},
        }

    @app.get("/metrics", tags=["System"])
    def get_metrics():
        return _QUANT_METRICS or {"note": "No metrics available. Run the training pipeline first."}

    @app.post("/predict", response_model=PredictResponse, tags=["Scoring"])
    def predict(req: PredictRequest):
        t0 = time.perf_counter()
        result = _demo_predict(req)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return PredictResponse(
            fraud_probability=result["fraud_probability"],
            decision=result["decision"],
            risk_tier=result["risk_tier"],
            expected_loss_usd=result["expected_loss_usd"],
            reason=result["reason_codes"],
            model_version=result["model_version"],
            latency_ms=latency_ms,
        )

    @app.post("/score", response_model=ScoreResponse, tags=["Scoring"])
    def score_transaction(txn: TransactionRequest):
        t0 = time.perf_counter()

        if _PIPELINE is None or _EXPLAINER is None:
            result = _demo_score(txn)
        else:
            try:
                X = _request_to_features(txn, _PIPELINE.feature_names)
                result = _EXPLAINER.explain(X, top_k=3, avg_transaction_value=txn.amount)
            except Exception as exc:
                logger.error("Scoring error: %s", exc)
                raise HTTPException(status_code=500, detail=f"Scoring failed: {exc}")

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        if _PIPELINE:
            from src.fraud_detection.monitoring import ModelMonitor
            monitor = ModelMonitor(Config)
            monitor.log_prediction(
                transaction_id=txn.transaction_id,
                fraud_probability=result["fraud_probability"],
                decision=result["decision"],
                model_version=result["model_version"],
            )

        return ScoreResponse(
            transaction_id=txn.transaction_id,
            fraud_probability=result["fraud_probability"],
            risk_tier=result["risk_tier"],
            decision=result["decision"],
            expected_loss_usd=result["expected_loss_usd"],
            reason_codes=result.get("reason_codes", []),
            model_version=result["model_version"],
            latency_ms=latency_ms,
        )

    @app.post("/score/batch", response_model=BatchScoreResponse, tags=["Scoring"])
    def score_batch(batch: BatchRequest):
        t0 = time.perf_counter()
        results = []

        for txn in batch.transactions:
            if _PIPELINE is None or _EXPLAINER is None:
                r = _demo_score(txn)
            else:
                try:
                    X = _request_to_features(txn, _PIPELINE.feature_names)
                    r = _EXPLAINER.explain(X, top_k=3, avg_transaction_value=txn.amount)
                except Exception:
                    r = _demo_score(txn)

            results.append(ScoreResponse(
                transaction_id=txn.transaction_id,
                fraud_probability=r["fraud_probability"],
                risk_tier=r["risk_tier"],
                decision=r["decision"],
                expected_loss_usd=r["expected_loss_usd"],
                reason_codes=r.get("reason_codes", []),
                model_version=r["model_version"],
                latency_ms=0.0,
            ))

        total_ms = round((time.perf_counter() - t0) * 1000, 2)
        total_loss = round(sum(r.expected_loss_usd for r in results), 2)
        high_risk = sum(1 for r in results if r.risk_tier in ("HIGH", "CRITICAL"))

        return BatchScoreResponse(
            results=results,
            total_expected_loss_usd=total_loss,
            high_risk_count=high_risk,
            processing_time_ms=total_ms,
        )


if __name__ == "__main__":
    if not FASTAPI_AVAILABLE:
        print("Install FastAPI first: pip install fastapi uvicorn")
    else:
        import uvicorn
        uvicorn.run(
            "src.fraud_detection.serving.api:app",
            host=Config.API_HOST,
            port=Config.API_PORT,
            log_level="info",
        )
