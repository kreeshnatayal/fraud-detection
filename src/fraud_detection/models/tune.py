"""
Bayesian hyperparameter optimisation via Optuna.

Each tuner runs a quick 3-fold CV (instead of the full 5-fold) to keep
wall-clock time reasonable.  The returned dict of best params is merged
into the model constructor in train.py.
"""
from __future__ import annotations

import warnings
from typing import Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def _cv_auc(model_cls, params: dict, X: pd.DataFrame, y: pd.Series,
            n_folds: int = 3, random_state: int = 42) -> float:
    from sklearn.model_selection import StratifiedKFold
    from src.fraud_detection.utils.metrics import safe_roc_auc

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    aucs = []
    for tr_idx, val_idx in skf.split(X, y):
        m = model_cls(**params)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        preds = m.predict_proba(X.iloc[val_idx])[:, 1]
        aucs.append(safe_roc_auc(y.iloc[val_idx].values, preds))
    return float(np.mean(aucs))


def tune_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    pos_weight: float,
    n_trials: int = 25,
    random_state: int = 42,
) -> Dict:
    """Return best LightGBM hyperparameters found by Optuna."""
    if not OPTUNA_AVAILABLE:
        return {}

    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        return {}

    def objective(trial: "optuna.Trial") -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 800),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            num_leaves=trial.suggest_int("num_leaves", 20, 127),
            max_depth=trial.suggest_int("max_depth", 4, 10),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 80),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            scale_pos_weight=pos_weight,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        return _cv_auc(LGBMClassifier, params, X, y, random_state=random_state)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    pos_weight: float,
    n_trials: int = 25,
    random_state: int = 42,
) -> Dict:
    """Return best XGBoost hyperparameters found by Optuna."""
    if not OPTUNA_AVAILABLE:
        return {}

    try:
        from xgboost import XGBClassifier
    except ImportError:
        return {}

    def objective(trial: "optuna.Trial") -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 800),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            scale_pos_weight=pos_weight,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="auc",
            verbosity=0,
        )
        return _cv_auc(XGBClassifier, params, X, y, random_state=random_state)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params
