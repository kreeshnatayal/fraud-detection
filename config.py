"""
Centralized configuration for the fraud detection pipeline.
All constants and tuneable parameters live here — never scattered in code.
"""

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Drift thresholds
# ---------------------------------------------------------------------------
DRIFT_THRESHOLDS = {
    "statistical": 0.05,
    "performance": 0.02,
    "feature": 0.1,
    "temporal": 0.15,
}


class Config:
    # -----------------------------------------------------------------------
    # Paths  (override via env vars for portability)
    # -----------------------------------------------------------------------
    OUTPUT_DIR: str = os.environ.get("FRAUD_OUTPUT_DIR", str(Path(__file__).parent / "output"))
    CHECKPOINT_DIR: str = os.environ.get("FRAUD_CHECKPOINT_DIR", str(Path(__file__).parent / "checkpoints"))

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    TRANSACTION_CSV: str = os.environ.get(
        "FRAUD_TRANSACTION_CSV",
        str(Path(__file__).parent / "train_transaction.csv"),
    )
    IDENTITY_CSV: str = os.environ.get(
        "FRAUD_IDENTITY_CSV",
        str(Path(__file__).parent / "train_identity.csv"),
    )
    RESUME_FROM_CHECKPOINT: bool = True

    # -----------------------------------------------------------------------
    # Graph / embedding settings
    # -----------------------------------------------------------------------
    EMBED_DIM: int = 32
    WALK_LENGTH: int = 80
    EPOCHS: int = 25
    TIME_DECAY_BETA: float = 1e-5
    EDGE_ATTRS: list = ["card1", "card2", "addr1", "P_emaildomain", "DeviceInfo"]

    # -----------------------------------------------------------------------
    # Hardware
    # -----------------------------------------------------------------------
    GPU_AVAILABLE: bool = False          # set True if CUDA is present
    BATCH_SIZE: int = 1024
    NUM_WORKERS: int = 4

    # -----------------------------------------------------------------------
    # Feature engineering
    # -----------------------------------------------------------------------
    RARE_THRESHOLD: int = 3
    ROLLING_WINDOWS: list = [1, 3, 7, 14]
    LAG_PERIODS: list = [1, 3, 7]

    # -----------------------------------------------------------------------
    # Memory management
    # -----------------------------------------------------------------------
    MAX_FEATURES: int = 100
    SAMPLE_SIZE: int = 15_000
    MEMORY_THRESHOLD: float = 70.0          # % — warn level
    CRITICAL_MEMORY_THRESHOLD: float = 0.85 # fraction — aggressive cleanup

    # -----------------------------------------------------------------------
    # Fraud cost parameters
    # -----------------------------------------------------------------------
    FN_COST_MULTIPLIER: float = 15.0
    FP_COST_MULTIPLIER: float = 1.0
    MIN_RECALL: float = 0.90
    MAX_FPR: float = 0.03

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Monitoring / drift
    # -----------------------------------------------------------------------
    DRIFT_THRESHOLDS: dict = DRIFT_THRESHOLDS

    @classmethod
    def ensure_dirs(cls) -> None:
        """Create output / checkpoint directories if they don't exist."""
        Path(cls.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
