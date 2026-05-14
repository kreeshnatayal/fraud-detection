import numpy as np
import pandas as pd

from src.fraud_detection.utils.logging import get_logger

logger = get_logger()


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if "TransactionDT" not in df.columns:
        return df
    df = df.copy()
    dt = df["TransactionDT"]
    df["DT_hour"] = (dt // 3600) % 24
    df["DT_day"] = (dt // (3600 * 24)) % 7
    df["DT_week"] = (dt // (3600 * 24 * 7)) % 52
    df["DT_hour_sin"] = np.sin(2 * np.pi * df["DT_hour"] / 24)
    df["DT_hour_cos"] = np.cos(2 * np.pi * df["DT_hour"] / 24)
    df["DT_day_sin"] = np.sin(2 * np.pi * df["DT_day"] / 7)
    df["DT_day_cos"] = np.cos(2 * np.pi * df["DT_day"] / 7)
    df["is_night"] = ((df["DT_hour"] >= 22) | (df["DT_hour"] <= 6)).astype(np.int8)
    df["is_weekend"] = (df["DT_day"] >= 5).astype(np.int8)

    if "card1" in df.columns:
        df = df.sort_values(["card1", "TransactionDT"])
        prev_ts = df.groupby("card1")["TransactionDT"].shift(1)
        df["time_since_last_txn"] = (dt - prev_ts).fillna(1_000_000)
        df["time_since_last_txn_log"] = np.log1p(df["time_since_last_txn"])
        df["is_rapid_txn"] = (df["time_since_last_txn"] < 300).astype(np.int8)

    logger.info("Time features added.")
    return df
