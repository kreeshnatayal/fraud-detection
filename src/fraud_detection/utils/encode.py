import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.fraud_detection.utils.logging import get_logger

logger = get_logger()


def safe_category_encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt_cols = df.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
    if dt_cols:
        logger.warning("Dropping datetime columns: %s", dt_cols)
        df = df.drop(columns=dt_cols)
    for col in df.select_dtypes(include=["object", "category", "string"]).columns:
        df[col] = df[col].fillna("missing").astype("category").cat.codes.astype("int32")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols].fillna(df[num_cols].median()))
    return df
