from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from src.fraud_detection.utils.logging import get_logger

logger = get_logger()


def add_rolling_features(df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
    if "TransactionDT" not in df.columns or "card1" not in df.columns:
        return df
    windows = windows or [1, 3, 7, 14]
    df = df.copy().sort_values(["card1", "TransactionDT"])

    for w in windows:
        if "TransactionAmt" in df.columns:
            df[f"rolling_amt_mean_{w}d"] = (
                df.groupby("card1")["TransactionAmt"]
                .transform(lambda x: x.rolling(w, min_periods=1).mean())
                .astype(np.float32)
            )
        df[f"rolling_txn_count_{w}d"] = (
            df.groupby("card1")["TransactionDT"]
            .transform(lambda x: x.rolling(w, min_periods=1).count())
            .astype(np.float32)
        )

    logger.info("Rolling features added.")
    return df


def add_lag_features(df: pd.DataFrame, lags: List[int] = None) -> pd.DataFrame:
    if "TransactionDT" not in df.columns or "card1" not in df.columns:
        return df
    lags = lags or [1, 3, 7]
    df = df.copy().sort_values(["card1", "TransactionDT"])

    for lag in lags:
        if "TransactionAmt" in df.columns:
            df[f"lag_amt_{lag}"] = (
                df.groupby("card1")["TransactionAmt"].shift(lag).astype(np.float32)
            )
        if "isFraud" in df.columns:
            df[f"lag_fraud_{lag}"] = (
                df.groupby("card1")["isFraud"].shift(lag).astype(np.float32)
            )

    logger.info("Lag features added.")
    return df


def add_anomaly_features(df: pd.DataFrame, sample_size: int = 10_000) -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.difference(
        ["isFraud", "TransactionID"]
    ).tolist()
    if len(num_cols) < 2:
        return df

    subset = df[num_cols].fillna(0)
    n = min(sample_size, len(df))

    try:
        iso = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
        iso.fit(subset.sample(n=n, random_state=42))
        df["isolation_score"] = iso.score_samples(subset).astype(np.float32)
    except Exception as exc:
        logger.warning("IsolationForest failed: %s", exc)

    try:
        lof = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1)
        lof.fit(subset.sample(n=n, random_state=42))
        df["lof_score"] = lof.score_samples(subset).astype(np.float32)
    except Exception as exc:
        logger.warning("LocalOutlierFactor failed: %s", exc)

    logger.info("Anomaly features added.")
    return df
