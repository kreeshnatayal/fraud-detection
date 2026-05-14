from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.fraud_detection.config import Config
from src.fraud_detection.features.graph import (
    attach_embeddings,
    build_transaction_graph,
    extract_graph_features,
    generate_node2vec_embeddings,
)
from src.fraud_detection.features.pipeline import run_feature_engineering
from src.fraud_detection.models.trained_pipeline import TrainedPipeline
from src.fraud_detection.models.evaluate import calculate_metrics
from src.fraud_detection.models.selection import select_features
from src.fraud_detection.models.train import train_full_pipeline
from src.fraud_detection.utils import (
    CheckpointManager,
    PipelineTracker,
    get_logger,
    load_data,
    safe_category_encode,
)

logger = get_logger(__name__)


def detect_drift(
    reference_metrics: Dict,
    current_metrics: Dict,
    thresholds: Dict,
) -> Dict:
    alerts = {}
    for key in ["auc", "f1", "recall", "precision"]:
        ref = reference_metrics.get(key, 0)
        cur = current_metrics.get(key, 0)
        drop = ref - cur
        if drop > thresholds.get("performance", 0.02):
            alerts[key] = {"reference": ref, "current": cur, "drop": drop}
            logger.warning("DRIFT ALERT: %s dropped %.4f (%.4f -> %.4f)", key, drop, ref, cur)
    return alerts


class FraudDetectionPipeline:
    def __init__(self, config: Config = None):
        self.cfg = config or Config()
        self.cfg.ensure_dirs()
        self.tracker = PipelineTracker(self.cfg.OUTPUT_DIR)
        self.ckpt = CheckpointManager(self.cfg.CHECKPOINT_DIR)

    def _stage_load(self) -> pd.DataFrame:
        stage = "data_loading"
        if self.cfg.RESUME_FROM_CHECKPOINT and self.ckpt.exists(stage):
            logger.info("[Stage 1] Loading from checkpoint …")
            return self.ckpt.load(stage)

        self.tracker.start_stage(stage)
        try:
            df = load_data(self.cfg.TRANSACTION_CSV, self.cfg.IDENTITY_CSV)
            self.ckpt.save(stage, df, "raw merged dataframe")
            self.tracker.end_stage(stage, success=True, metrics={"rows": len(df), "cols": df.shape[1]})
            return df
        except Exception as exc:
            self.tracker.end_stage(stage, success=False)
            raise RuntimeError(f"Data loading failed: {exc}") from exc

    def _stage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        stage = "feature_engineering"
        if self.cfg.RESUME_FROM_CHECKPOINT and self.ckpt.exists(stage):
            logger.info("[Stage 2] Loading from checkpoint …")
            return self.ckpt.load(stage)

        self.tracker.start_stage(stage)
        try:
            df = run_feature_engineering(df, self.cfg)
            self.ckpt.save(stage, df, "feature-engineered dataframe")
            self.tracker.end_stage(stage, success=True, metrics={"cols": df.shape[1]})
            return df
        except Exception as exc:
            self.tracker.end_stage(stage, success=False)
            raise RuntimeError(f"Feature engineering failed: {exc}") from exc

    def _stage_graph(self, df: pd.DataFrame) -> pd.DataFrame:
        stage = "graph_features"
        if self.cfg.RESUME_FROM_CHECKPOINT and self.ckpt.exists(stage):
            logger.info("[Stage 3] Loading from checkpoint …")
            return self.ckpt.load(stage)

        self.tracker.start_stage(stage)
        try:
            G = build_transaction_graph(df, edge_attrs=self.cfg.EDGE_ATTRS)
            if G is not None:
                df = extract_graph_features(df, G, edge_attrs=self.cfg.EDGE_ATTRS)
                embeddings = generate_node2vec_embeddings(
                    G, embed_dim=self.cfg.EMBED_DIM,
                    walk_length=self.cfg.WALK_LENGTH,
                    epochs=self.cfg.EPOCHS,
                )
                for col in [c for c in self.cfg.EDGE_ATTRS if c in df.columns]:
                    df = attach_embeddings(df, embeddings, col, embed_dim=self.cfg.EMBED_DIM)
                del G, embeddings
                gc.collect()
            self.ckpt.save(stage, df, "graph-enriched dataframe")
            self.tracker.end_stage(stage, success=True, metrics={"cols": df.shape[1]})
            return df
        except Exception as exc:
            logger.warning("Graph stage failed (%s); continuing without graph features.", exc)
            self.tracker.end_stage(stage, success=False)
            return df

    def _stage_prepare(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        if "isFraud" not in df.columns:
            raise ValueError("Target column 'isFraud' not found in dataframe.")

        y = df["isFraud"].astype(int)
        X = df.drop(columns=["isFraud"], errors="ignore")
        X = safe_category_encode(X)
        X = X.fillna(0)

        selected = select_features(X, y, max_features=self.cfg.MAX_FEATURES, random_state=self.cfg.RANDOM_STATE)
        X = X[selected]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=1 - self.cfg.TRAIN_TEST_SPLIT_RATIO,
            random_state=self.cfg.RANDOM_STATE,
            stratify=y,
        )
        logger.info(
            "Split: train=%d (fraud=%.2f%%)  val=%d (fraud=%.2f%%)",
            len(y_train), 100 * y_train.mean(),
            len(y_val), 100 * y_val.mean(),
        )
        return X_train, y_train, X_val, y_val

    def _stage_train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> TrainedPipeline:
        stage = "model_training"
        if self.cfg.RESUME_FROM_CHECKPOINT and self.ckpt.exists(stage):
            logger.info("[Stage 5] Loading from checkpoint …")
            return self.ckpt.load(stage)

        self.tracker.start_stage(stage)
        try:
            trained = train_full_pipeline(X_train, y_train, X_val, y_val, self.cfg)
            self.ckpt.save(stage, trained, "trained pipeline object")
            val_metrics = trained.metrics.get("validation", {})
            self.tracker.end_stage(stage, success=True, metrics=val_metrics)
            return trained
        except Exception as exc:
            self.tracker.end_stage(stage, success=False)
            raise RuntimeError(f"Model training failed: {exc}") from exc

    def _stage_evaluate(
        self,
        trained: TrainedPipeline,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        amounts: Optional[pd.Series] = None,
    ) -> Dict:
        self.tracker.start_stage("evaluation")
        try:
            val_proba = trained.predict_proba(X_val)
            metrics = calculate_metrics(y_val.values, val_proba, trained.threshold, label="final_eval")

            out = Path(self.cfg.OUTPUT_DIR) / "evaluation_metrics.json"
            with open(out, "w") as f:
                json.dump(metrics, f, indent=2, default=str)

            preds_df = X_val.copy()
            preds_df["fraud_probability"] = val_proba
            preds_df["predicted_fraud"] = trained.predict(X_val)
            preds_df["actual_fraud"] = y_val.values
            preds_df.to_csv(Path(self.cfg.OUTPUT_DIR) / "val_predictions.csv", index=False)

            all_metrics = {k: v for k, v in trained.metrics.items()}
            with open(Path(self.cfg.OUTPUT_DIR) / "model_metrics.json", "w") as f:
                json.dump(all_metrics, f, indent=2, default=str)

            self.tracker.end_stage("evaluation", success=True, metrics=metrics)
            return metrics
        except Exception as exc:
            self.tracker.end_stage("evaluation", success=False)
            raise RuntimeError(f"Evaluation failed: {exc}") from exc

    def run(self) -> Dict:
        logger.info("=" * 70)
        logger.info("FRAUD DETECTION PIPELINE — START")
        logger.info("=" * 70)

        df = self._stage_load()
        df = self._stage_features(df)
        df = self._stage_graph(df)

        amounts = df["TransactionAmt"].values if "TransactionAmt" in df.columns else None
        X_train, y_train, X_val, y_val = self._stage_prepare(df)
        del df
        gc.collect()

        trained = self._stage_train(X_train, y_train, X_val, y_val)

        amt_series = pd.Series(amounts[y_val.index]) if amounts is not None else None
        metrics = self._stage_evaluate(trained, X_val, y_val, amounts=amt_series)

        model_path = Path(self.cfg.OUTPUT_DIR) / "trained_pipeline.pkl"
        trained.save(str(model_path))

        summary = self.tracker.get_summary()
        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETE — Validation AUC: %.4f", metrics.get("auc", 0))
        logger.info("=" * 70)
        return {"metrics": metrics, "summary": summary, "model_path": str(model_path)}
