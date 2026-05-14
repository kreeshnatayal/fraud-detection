from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif

from src.fraud_detection.utils.logging import get_logger

logger = get_logger()


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    max_features: int = 100,
    random_state: int = 42,
) -> List[str]:
    logger.info("Selecting features (max=%d) from %d candidates …", max_features, X.shape[1])

    vt = VarianceThreshold(threshold=0.0)
    vt.fit(X.fillna(0))
    cols = X.columns[vt.get_support()].tolist()
    X_filtered = X[cols].fillna(0)

    if len(cols) <= max_features:
        logger.info("After variance filter: %d features (below cap).", len(cols))
        return cols

    mi = mutual_info_classif(X_filtered, y, random_state=random_state)
    scores = pd.Series(mi, index=cols).sort_values(ascending=False)
    selected = scores.head(max_features).index.tolist()
    logger.info("Selected %d features.", len(selected))
    return selected
