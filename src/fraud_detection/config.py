import os
from pathlib import Path

DRIFT_THRESHOLDS = {
    "statistical": 0.05,
    "performance": 0.02,
    "feature": 0.1,
    "temporal": 0.15,
}


class Config:
    OUTPUT_DIR: str = os.environ.get("FRAUD_OUTPUT_DIR", str(Path(__file__).parent.parent.parent / "output"))
    CHECKPOINT_DIR: str = os.environ.get("FRAUD_CHECKPOINT_DIR", str(Path(__file__).parent.parent.parent / "checkpoints"))

    TRANSACTION_CSV: str = os.environ.get(
        "FRAUD_TRANSACTION_CSV",
        str(Path(__file__).parent.parent.parent / "train_transaction.csv"),
    )
    IDENTITY_CSV: str = os.environ.get(
        "FRAUD_IDENTITY_CSV",
        str(Path(__file__).parent.parent.parent / "train_identity.csv"),
    )
    RESUME_FROM_CHECKPOINT: bool = True

    EMBED_DIM: int = 32
    WALK_LENGTH: int = 80
    EPOCHS: int = 25
    TIME_DECAY_BETA: float = 1e-5
    EDGE_ATTRS: list = ["card1", "card2", "addr1", "P_emaildomain", "DeviceInfo"]

    GPU_AVAILABLE: bool = False
    BATCH_SIZE: int = 1024
    NUM_WORKERS: int = 4

    RARE_THRESHOLD: int = 3
    ROLLING_WINDOWS: list = [1, 3, 7, 14]
    LAG_PERIODS: list = [1, 3, 7]

    MAX_FEATURES: int = 150
    SAMPLE_SIZE: int = 15_000
    MEMORY_THRESHOLD: float = 70.0
    CRITICAL_MEMORY_THRESHOLD: float = 0.85

    FN_COST_MULTIPLIER: float = 15.0
    FP_COST_MULTIPLIER: float = 1.0
    MIN_RECALL: float = 0.90
    MAX_FPR: float = 0.03

    RANDOM_STATE: int = 42
    TRAIN_TEST_SPLIT_RATIO: float = 0.8
    VALIDATION_SPLIT_RATIO: float = 0.2
    TEMPORAL_SPLIT_DAYS: int = 30
    EARLY_STOPPING_PATIENCE: int = 15
    MAX_EPOCHS: int = 150
    LEARNING_RATE_SCHEDULE: bool = True
    BATCH_SIZE_TRAINING: int = 2048
    BATCH_SIZE_PREDICTION: int = 4096
    CROSS_VALIDATION_FOLDS: int = 5
    HYPERPARAMETER_TRIALS: int = 25
    ENSEMBLE_METHODS: list = ["stacking", "voting", "blending"]

    DRIFT_THRESHOLDS: dict = DRIFT_THRESHOLDS
    PSI_THRESHOLD: float = 0.25
    KS_ALPHA: float = 0.05
    MONITORING_WINDOW_DAYS: int = 7

    AVG_TRANSACTION_VALUE: float = 200.0
    ANNUAL_TRANSACTION_VOLUME: int = 500_000
    INVESTIGATION_COST_PER_ALERT: float = 5.0
    FP_REVENUE_LOSS_RATE: float = 0.85
    FRAUD_CHARGEBACK_RATE: float = 1.0

    MODEL_VERSION: str = "v1.0.0"
    MODEL_NAME: str = "FraudRiskIntelligence"
    CHAMPION_MODEL: str = "stacking"

    API_HOST: str = os.environ.get("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.environ.get("API_PORT", "8000"))
    API_WORKERS: int = int(os.environ.get("API_WORKERS", "2"))
    SCORE_LATENCY_BUDGET_MS: float = 100.0

    @classmethod
    def ensure_dirs(cls) -> None:
        Path(cls.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
