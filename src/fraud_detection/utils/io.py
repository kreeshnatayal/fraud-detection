import os

import pandas as pd

from src.fraud_detection.utils.logging import get_logger
from src.fraud_detection.utils.memory import reduce_memory_usage

logger = get_logger()


def load_data(transaction_csv: str, identity_csv: str | None = None) -> pd.DataFrame:
    logger.info("Loading transactions from %s", transaction_csv)
    df = pd.read_csv(transaction_csv, index_col=0)
    logger.info("Loaded %d transactions", len(df))

    if identity_csv and os.path.exists(identity_csv):
        logger.info("Loading identity from %s", identity_csv)
        df_id = pd.read_csv(identity_csv, index_col=0)
        df = df.merge(df_id, left_index=True, right_index=True, how="left")
        logger.info("Merged -> %d rows", len(df))
    else:
        logger.warning("Identity CSV not found; proceeding without it.")

    df = reduce_memory_usage(df)
    return df
