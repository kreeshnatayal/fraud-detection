import numpy as np
import pandas as pd

from src.fraud_detection.utils.logging import get_logger

logger = get_logger()


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    if "TransactionAmt" not in df.columns:
        return df
    df = df.copy()
    amt = df["TransactionAmt"]
    df["amount_log"] = np.log1p(amt)
    df["amount_sqrt"] = np.sqrt(amt.clip(lower=0))
    df["is_micropayment"] = (amt < 1.0).astype(np.int8)
    df["is_large_payment"] = (amt > 1_000).astype(np.int8)
    df["is_suspicious_amount"] = (amt > 5_000).astype(np.int8)
    df["amount_percentile"] = amt.rank(pct=True)

    if "card1" in df.columns:
        card_stats = df.groupby("card1")["TransactionAmt"].agg(["mean", "std"]).rename(
            columns={"mean": "_card_amt_mean", "std": "_card_amt_std"}
        )
        df = df.join(card_stats, on="card1")
        std = df["_card_amt_std"].fillna(1).replace(0, 1)
        df["amount_zscore"] = (amt - df["_card_amt_mean"].fillna(amt)) / std
        df["is_amount_outlier"] = (df["amount_zscore"].abs() > 3).astype(np.int8)
        df.drop(columns=["_card_amt_mean", "_card_amt_std"], inplace=True)

    logger.info("Amount features added.")
    return df
