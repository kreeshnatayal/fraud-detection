import gc

from src.fraud_detection.features.temporal import add_time_features
from src.fraud_detection.features.amount import add_amount_features
from src.fraud_detection.features.categorical import (
    add_domain_features,
    add_entity_features,
    add_frequency_encoding,
    add_target_encoding,
)
from src.fraud_detection.features.anomaly import add_rolling_features, add_lag_features, add_anomaly_features
from src.fraud_detection.utils.logging import get_logger
from src.fraud_detection.utils.memory import memory_guard

logger = get_logger()


def run_feature_engineering(df, config):
    steps = [
        ("time", lambda d: add_time_features(d)),
        ("amount", lambda d: add_amount_features(d)),
        ("entity", lambda d: add_entity_features(d)),
        ("frequency", lambda d: add_frequency_encoding(d, rare_threshold=config.RARE_THRESHOLD)),
        ("target_enc", lambda d: add_target_encoding(d)),
        ("domain", lambda d: add_domain_features(d)),
        ("rolling", lambda d: add_rolling_features(d, windows=config.ROLLING_WINDOWS)),
        ("lag", lambda d: add_lag_features(d, lags=config.LAG_PERIODS)),
        ("anomaly", lambda d: add_anomaly_features(d, sample_size=config.SAMPLE_SIZE)),
    ]

    for name, fn in steps:
        logger.info("Feature step: %s", name)
        with memory_guard(threshold_pct=config.CRITICAL_MEMORY_THRESHOLD * 100):
            df = fn(df)
        gc.collect()

    return df
