import argparse
import sys
from pathlib import Path

from src.fraud_detection.config import Config
from src.fraud_detection.pipeline import FraudDetectionPipeline
from src.fraud_detection.utils import get_logger

logger = get_logger("main")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fraud Detection Pipeline")
    p.add_argument("--transaction", default=None, help="Path to train_transaction.csv")
    p.add_argument("--identity", default=None, help="Path to train_identity.csv")
    p.add_argument("--output", default=None, help="Output directory")
    p.add_argument("--no-resume", action="store_true", help="Ignore checkpoints and start fresh")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.transaction:
        Config.TRANSACTION_CSV = args.transaction
    if args.identity:
        Config.IDENTITY_CSV = args.identity
    if args.output:
        Config.OUTPUT_DIR = args.output
        Config.CHECKPOINT_DIR = str(Path(args.output) / "checkpoints")
    if args.no_resume:
        Config.RESUME_FROM_CHECKPOINT = False

    if not Path(Config.TRANSACTION_CSV).exists():
        logger.error("Transaction CSV not found: %s", Config.TRANSACTION_CSV)
        logger.error("Set via --transaction flag or FRAUD_TRANSACTION_CSV environment variable.")
        sys.exit(1)

    logger.info("Transaction CSV : %s", Config.TRANSACTION_CSV)
    logger.info("Identity CSV    : %s", Config.IDENTITY_CSV)
    logger.info("Output dir      : %s", Config.OUTPUT_DIR)
    logger.info("Resume          : %s", Config.RESUME_FROM_CHECKPOINT)

    pipeline = FraudDetectionPipeline(config=Config)
    result = pipeline.run()

    logger.info("")
    logger.info("Results:")
    metrics = result.get("metrics", {})
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info("  %-20s %.4f", k, v)
        else:
            logger.info("  %-20s %s", k, v)

    logger.info("Model saved to: %s", result.get("model_path", ""))
    logger.info("Launch dashboard: streamlit run dashboard/dashboard.py -- --output %s", Config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
