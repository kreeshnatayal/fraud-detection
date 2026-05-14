import numpy as np
import pandas as pd
import pytest

from src.fraud_detection.models.evaluate import calculate_metrics, optimise_threshold


class TestCalculateMetrics:
    def test_basic_metrics(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.9, 0.8])
        metrics = calculate_metrics(y_true, y_score, threshold=0.5)
        assert metrics["auc"] == 1.0
        assert metrics["tp"] == 2
        assert metrics["fn"] == 0
        assert metrics["fp"] == 0

    def test_imperfect_model(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.6, 0.7, 0.8, 0.2, 0.3, 0.4])
        metrics = calculate_metrics(y_true, y_score, threshold=0.5)
        assert metrics["tp"] == 0
        assert metrics["fp"] == 3


class TestOptimiseThreshold:
    def test_returns_float(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.9, 0.8])
        thresh = optimise_threshold(y_true, y_score)
        assert 0.0 < thresh < 1.0
