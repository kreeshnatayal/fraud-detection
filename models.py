"""
Model training, evaluation, threshold optimisation, and ensemble methods.
"""

from __future__ import annotations

import gc
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

from utils import get_logger, safe_divide, safe_roc_auc

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Optional heavy dependencies
# ---------------------------------------------------------------------------

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBMClassifier = None
    LGBM_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGBClassifier = None
    XGB_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CatBoostClassifier = None
    CATBOOST_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    max_features: int = 100,
    random_state: int = 42,
) -> List[str]:
    """Select top features by mutual information score."""
    logger.info("Selecting features (max=%d) from %d candidates …", max_features, X.shape[1])

    # Drop zero-variance features first
    vt = VarianceThreshold(threshold=0.0)
    vt.fit(X.fillna(0))
    cols = X.columns[vt.get_support()].tolist()
    X_filtered = X[cols].fillna(0)

    if len(cols) <= max_features:
        logger.info("After variance filter: %d features (below cap).", len(cols))
        return cols

    mi = mutual_info_classif(X_filtered, y, random_state=random_state)
    scores = pd.Series(mi, index=cols).sort_values(ascending=False)
    selected = scores.head(max_features).index.tolist()
    logger.info("Selected %d features.", len(selected))
    return selected


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    label: str = "",
) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "auc": safe_roc_auc(y_true, y_score),
        "ap": float(average_precision_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "log_loss": float(log_loss(y_true, y_score)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "fpr": safe_divide(fp, fp + tn),
        "threshold": threshold,
    }
    if label:
        logger.info(
            "[%s] AUC=%.4f  AP=%.4f  F1=%.4f  Recall=%.4f  Precision=%.4f",
            label, metrics["auc"], metrics["ap"], metrics["f1"],
            metrics["recall"], metrics["precision"],
        )
    return metrics


def optimise_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fn_cost: float = 15.0,
    fp_cost: float = 1.0,
) -> float:
    """Find threshold that minimises (fn_cost * FN + fp_cost * FP)."""
    thresholds = np.linspace(0.01, 0.99, 200)
    best_thresh, best_cost = 0.5, float("inf")
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        cost = fn_cost * fn + fp_cost * fp
        if cost < best_cost:
            best_cost = cost
            best_thresh = t
    logger.info("Optimal threshold: %.4f  (cost=%.1f)", best_thresh, best_cost)
    return best_thresh


def calculate_fraud_cost(
    y_true: np.ndarray,
    y_score: np.ndarray,
    amounts: Optional[np.ndarray],
    threshold: float,
    fn_cost_mult: float = 15.0,
    fp_cost_mult: float = 1.0,
) -> float:
    y_pred = (y_score >= threshold).astype(int)
    fn_mask = (y_true == 1) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    if amounts is not None:
        return float(fn_cost_mult * amounts[fn_mask].sum() + fp_cost_mult * amounts[fp_mask].sum())
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fn_cost_mult * fn + fp_cost_mult * fp)


# ---------------------------------------------------------------------------
# Individual model builders
# ---------------------------------------------------------------------------

def _build_base_models(config) -> Dict[str, Any]:
    models: Dict[str, Any] = {}

    if LGBM_AVAILABLE:
        models["lgbm"] = LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            random_state=config.RANDOM_STATE, n_jobs=-1, verbose=-1,
        )
    if XGB_AVAILABLE:
        models["xgb"] = XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            random_state=config.RANDOM_STATE, n_jobs=-1,
            eval_metric="auc", verbosity=0,
        )
    if CATBOOST_AVAILABLE:
        models["catboost"] = CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6,
            random_state=config.RANDOM_STATE, verbose=0,
        )

    models["rf"] = RandomForestClassifier(
        n_estimators=200, random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    models["lr"] = LogisticRegression(
        max_iter=1000, random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    return models


# ---------------------------------------------------------------------------
# Training a single model with cross-validation
# ---------------------------------------------------------------------------

def train_model_cv(
    name: str,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5,
    random_state: int = 42,
) -> Tuple[Any, np.ndarray, Dict]:
    """Train a model with stratified k-fold CV; return (fitted_model, oof_preds, metrics)."""
    logger.info("Training %s …", name)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof = np.zeros(len(y))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        if SMOTE_AVAILABLE:
            try:
                sm = SMOTE(random_state=random_state, k_neighbors=min(5, y_tr.sum() - 1))
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
            except Exception:
                pass

        try:
            if hasattr(model, "fit") and "eval_set" in model.fit.__code__.co_varnames:
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            else:
                model.fit(X_tr, y_tr)
        except TypeError:
            model.fit(X_tr, y_tr)

        oof[val_idx] = model.predict_proba(X_val)[:, 1]
        logger.info("  Fold %d/%d  AUC=%.4f", fold, n_folds, safe_roc_auc(y_val.values, oof[val_idx]))

    model.fit(X, y)
    metrics = calculate_metrics(y.values, oof, label=name)
    return model, oof, metrics


# ---------------------------------------------------------------------------
# Ensemble methods
# ---------------------------------------------------------------------------

def train_voting_ensemble(
    base_oof: Dict[str, np.ndarray],
    y: pd.Series,
) -> np.ndarray:
    """Average OOF predictions from all base models."""
    stack = np.column_stack(list(base_oof.values()))
    preds = stack.mean(axis=1)
    m = calculate_metrics(y.values, preds, label="voting_ensemble")
    return preds


def train_stacking_ensemble(
    base_oof: Dict[str, np.ndarray],
    y: pd.Series,
    n_folds: int = 5,
    random_state: int = 42,
) -> Tuple[Any, np.ndarray, Dict]:
    """Train a logistic regression meta-learner on stacked OOF predictions."""
    X_meta = pd.DataFrame(base_oof)
    meta = LogisticRegression(max_iter=500, random_state=random_state)
    meta, oof, metrics = train_model_cv(
        "stacking_meta", meta, X_meta, y, n_folds=n_folds, random_state=random_state
    )
    return meta, oof, metrics


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------

class TrainedPipeline:
    """Container for trained models and metadata."""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.oof_predictions: Dict[str, np.ndarray] = {}
        self.metrics: Dict[str, Dict] = {}
        self.feature_names: List[str] = []
        self.threshold: float = 0.5
        self.meta_model: Optional[Any] = None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.feature_names].fillna(0)
        preds = np.column_stack([m.predict_proba(X)[:, 1] for m in self.models.values()])
        if self.meta_model is not None:
            return self.meta_model.predict_proba(preds)[:, 1]
        return preds.mean(axis=1)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Pipeline saved to %s", path)

    @staticmethod
    def load(path: str) -> "TrainedPipeline":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


def train_full_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config,
) -> TrainedPipeline:
    result = TrainedPipeline()
    result.feature_names = X_train.columns.tolist()

    base_models = _build_base_models(config)

    for name, model in base_models.items():
        with gc.collect.__class__():
            pass
        fitted, oof, metrics = train_model_cv(
            name, model, X_train, y_train,
            n_folds=config.CROSS_VALIDATION_FOLDS,
            random_state=config.RANDOM_STATE,
        )
        result.models[name] = fitted
        result.oof_predictions[name] = oof
        result.metrics[name] = metrics
        gc.collect()

    # Stacking
    if len(result.oof_predictions) > 1:
        meta, meta_oof, meta_metrics = train_stacking_ensemble(
            result.oof_predictions, y_train,
            n_folds=config.CROSS_VALIDATION_FOLDS,
            random_state=config.RANDOM_STATE,
        )
        result.meta_model = meta
        result.metrics["stacking"] = meta_metrics

    # Threshold optimisation on validation set
    val_proba = result.predict_proba(X_val)
    result.threshold = optimise_threshold(
        y_val.values, val_proba,
        fn_cost=config.FN_COST_MULTIPLIER,
        fp_cost=config.FP_COST_MULTIPLIER,
    )
    result.metrics["validation"] = calculate_metrics(y_val.values, val_proba, result.threshold, label="val_final")

    return result
