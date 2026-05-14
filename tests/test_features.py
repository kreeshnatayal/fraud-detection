import numpy as np
import pandas as pd
import pytest

from src.fraud_detection.features.temporal import add_time_features
from src.fraud_detection.features.amount import add_amount_features


class TestTimeFeatures:
    def test_hour_and_day_extracted(self):
        df = pd.DataFrame({"TransactionDT": [0, 3600, 86400, 86400 * 3 + 7200]})
        result = add_time_features(df)
        assert "DT_hour" in result.columns
        assert "DT_day" in result.columns
        assert "is_night" in result.columns
        assert result["DT_hour"].iloc[0] == 0
        assert result["DT_day"].iloc[0] == 0

    def test_no_op_if_no_transactiondt(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = add_time_features(df)
        assert result is df


class TestAmountFeatures:
    def test_amount_transforms(self):
        df = pd.DataFrame({"TransactionAmt": [0.5, 100, 5000, 20000]})
        result = add_amount_features(df)
        assert "amount_log" in result.columns
        assert "is_micropayment" in result.columns
        assert "is_large_payment" in result.columns
        assert "is_suspicious_amount" in result.columns
        assert result["is_micropayment"].iloc[0] == 1
        assert result["is_large_payment"].iloc[1] == 0
        assert result["is_suspicious_amount"].iloc[2] == 0
        assert result["is_suspicious_amount"].iloc[3] == 1
