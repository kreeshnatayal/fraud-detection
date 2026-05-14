from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


class TrainedPipeline:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.oof_predictions: Dict[str, np.ndarray] = {}
        self.metrics: Dict[str, Dict] = {}
        self.feature_names: List[str] = []
        self.threshold: float = 0.5
        self.meta_model: Optional[Any] = None
        self.calibrator: Optional[Any] = None  # isotonic calibration layer

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.feature_names].fillna(0)
        preds = np.column_stack([m.predict_proba(X)[:, 1] for m in self.models.values()])
        if self.meta_model is not None:
            raw = self.meta_model.predict_proba(preds)[:, 1]
        else:
            raw = preds.mean(axis=1)
        if self.calibrator is not None:
            return np.clip(self.calibrator.predict(raw), 0.0, 1.0)
        return raw

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Pipeline saved to %s", path)

    @staticmethod
    def load(path: str) -> "TrainedPipeline":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
