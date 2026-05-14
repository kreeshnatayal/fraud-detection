import numpy as np
import pandas as pd
import pytest

from src.fraud_detection.utils.metrics import safe_roc_auc, safe_divide


class TestSafeRocAuc:
    def test_perfect_separation(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.9, 0.8])
        auc = safe_roc_auc(y_true, y_score)
        assert auc == 1.0

    def test_single_class(self):
        y_true = np.array([0, 0, 0])
        y_score = np.array([0.1, 0.2, 0.3])
        assert safe_roc_auc(y_true, y_score) == 0.5


class TestSafeDivide:
    def test_normal_division(self):
        assert safe_divide(10, 2) == 5.0

    def test_division_by_zero(self):
        assert safe_divide(10, 0) == 0.0

    def test_custom_default(self):
        assert safe_divide(10, 0, default=-1.0) == -1.0
