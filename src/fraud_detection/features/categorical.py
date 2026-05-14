from typing import List, Optional

import numpy as np
import pandas as pd

from src.fraud_detection.utils.logging import get_logger

logger = get_logger()


def add_entity_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    entity_cols = [c for c in ["card1", "card2", "addr1", "addr2"] if c in df.columns]

    for col in entity_cols:
        freq = df[col].map(df[col].value_counts())
        df[f"{col}_frequency"] = freq.fillna(0).astype(np.float32)

        if "TransactionAmt" in df.columns:
            stats = df.groupby(col)["TransactionAmt"].agg(["mean", "std", "count"])
            stats.columns = [f"{col}_amt_mean", f"{col}_amt_std", f"{col}_txn_count"]
            df = df.join(stats, on=col)

        if "isFraud" in df.columns:
            fraud_rate = df.groupby(col)["isFraud"].mean().rename(f"{col}_fraud_rate")
            df = df.join(fraud_rate, on=col)

    logger.info("Entity features added.")
    return df


def add_frequency_encoding(df: pd.DataFrame, rare_threshold: int = 3) -> pd.DataFrame:
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        freq = df[col].value_counts()
        df[f"{col}_freq"] = df[col].map(freq).fillna(0).astype(np.float32)
        df[f"{col}_is_rare"] = (df[f"{col}_freq"] < rare_threshold).astype(np.int8)
    logger.info("Frequency encoding added.")
    return df


def add_target_encoding(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    if "isFraud" not in df.columns:
        return df
    df = df.copy()
    cols = cols or [c for c in ["card1", "card2", "addr1", "P_emaildomain", "DeviceInfo"] if c in df.columns]
    global_mean = df["isFraud"].mean()
    for col in cols:
        means = df.groupby(col)["isFraud"].mean()
        df[f"{col}_target_enc"] = df[col].map(means).fillna(global_mean).astype(np.float32)
    logger.info("Target encoding added.")
    return df


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for email_col in ["P_emaildomain", "R_emaildomain"]:
        if email_col not in df.columns:
            continue
        df[f"{email_col}_domain"] = df[email_col].str.split(".").str[-1].fillna("unknown")
        df[f"{email_col}_is_free"] = df[email_col].isin(
            ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com"]
        ).astype(np.int8)

    if "DeviceInfo" in df.columns:
        df["device_os"] = df["DeviceInfo"].str.split(" ").str[0].fillna("unknown")
        df["is_mobile"] = df["DeviceInfo"].str.lower().str.contains("android|ios|mobile", na=False).astype(np.int8)

    logger.info("Domain features added.")
    return df
