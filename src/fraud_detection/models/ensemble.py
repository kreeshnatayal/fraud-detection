from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.fraud_detection.models.evaluate import calculate_metrics
from src.fraud_detection.models.train import train_model_cv
from src.fraud_detection.utils.logging import get_logger

logger = get_logger()


def train_voting_ensemble(
    base_oof: Dict[str, np.ndarray],
    y: pd.Series,
) -> np.ndarray:
    stack = np.column_stack(list(base_oof.values()))
    preds = stack.mean(axis=1)
    calculate_metrics(y.values, preds, label="voting_ensemble")
    return preds
