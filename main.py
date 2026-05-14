"""
Entry point: adds src/ to path and delegates to the package.
Usage:
    python main.py
    python -m src.fraud_detection.main
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.fraud_detection.main import main

if __name__ == "__main__":
    main()
