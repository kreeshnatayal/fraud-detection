import gc
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.fraud_detection.models.evaluate import calculate_metrics, optimise_threshold
from src.fraud_detection.models.trained_pipeline import TrainedPipeline
from src.fraud_detection.utils.logging import get_logger
from src.fraud_detection.utils.metrics import safe_roc_auc

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


try:
    from lightgbm import LGBMClassifier
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBMClassifier = None
    lgb = None
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
    # BorderlineSMOTE focuses on the decision boundary — better than SMOTE for
    # fraud where the class boundary is complex and non-uniform.
    from imblearn.over_sampling import BorderlineSMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    BorderlineSMOTE = None
    SMOTE_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError:
    RandomForestClassifier = None

try:
    from sklearn.isotonic import IsotonicRegression
    ISOTONIC_AVAILABLE = True
except ImportError:
    IsotonicRegression = None
    ISOTONIC_AVAILABLE = False


def _build_base_models(config, pos_weight: float = 1.0,
                       lgbm_params: Optional[Dict] = None,
                       xgb_params: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Build base models with class-imbalance weighting via scale_pos_weight.
    pos_weight = n_negatives / n_positives; mirrors FN_COST_MULTIPLIER at
    gradient level so the loss function itself penalises missed frauds.
    """
    models: Dict[str, Any] = {}

    if LGBM_AVAILABLE:
        lgbm_base = dict(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            scale_pos_weight=pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        if lgbm_params:
            lgbm_base.update(lgbm_params)
        models["lgbm"] = LGBMClassifier(**lgbm_base)

    if XGB_AVAILABLE:
        xgb_base = dict(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            # early_stopping_rounds triggers only when eval_set is provided in fit()
            early_stopping_rounds=config.EARLY_STOPPING_PATIENCE,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            eval_metric="auc",
            verbosity=0,
        )
        if xgb_params:
            xgb_base.update(xgb_params)
        models["xgb"] = XGBClassifier(**xgb_base)

    if CATBOOST_AVAILABLE:
        models["catboost"] = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            # CatBoost class_weights: index 0 = negative, index 1 = positive
            class_weights=[1.0, pos_weight],
            early_stopping_rounds=config.EARLY_STOPPING_PATIENCE,
            random_state=config.RANDOM_STATE,
            verbose=0,
        )

    if RandomForestClassifier is not None:
        models["rf"] = RandomForestClassifier(
            n_estimators=200,
            class_weight={0: 1.0, 1: pos_weight},
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )

    return models


def _fit_with_early_stopping(model, X_tr, y_tr, X_val, y_val, patience: int) -> None:
    """Fit a model with early stopping where the model type supports it."""
    model_cls = type(model).__name__

    if LGBM_AVAILABLE and "LGBM" in model_cls:
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=patience, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
    elif XGB_AVAILABLE and "XGB" in model_cls:
        # early_stopping_rounds already set in constructor; triggers via eval_set
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    elif CATBOOST_AVAILABLE and "CatBoost" in model_cls:
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
    else:
        try:
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        except TypeError:
            model.fit(X_tr, y_tr)


def train_model_cv(
    name: str,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5,
    random_state: int = 42,
    patience: int = 15,
) -> Tuple[Any, np.ndarray, Dict]:
    logger.info("Training %s …", name)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof = np.zeros(len(y))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        if SMOTE_AVAILABLE:
            try:
                # BorderlineSMOTE generates synthetic samples near the decision
                # boundary, producing more discriminative training examples than
                # vanilla SMOTE for fraud's complex class border.
                sm = BorderlineSMOTE(
                    random_state=random_state,
                    k_neighbors=min(5, int(y_tr.sum()) - 1),
                    kind="borderline-1",
                )
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
            except Exception:
                pass

        _fit_with_early_stopping(model, X_tr, y_tr, X_val, y_val, patience)
        oof[val_idx] = model.predict_proba(X_val)[:, 1]
        logger.info("  Fold %d/%d  AUC=%.4f", fold, n_folds,
                    safe_roc_auc(y_val.values, oof[val_idx]))

    # Final refit on full training data (no early stopping — no eval_set passed)
    model.fit(X, y)
    metrics = calculate_metrics(y.values, oof, label=name)
    return model, oof, metrics


def train_stacking_ensemble(
    base_oof: Dict[str, np.ndarray],
    y: pd.Series,
    n_folds: int = 5,
    random_state: int = 42,
) -> Tuple[Any, np.ndarray, Dict]:
    """
    Train a LightGBM meta-learner on OOF predictions from base models.
    LightGBM (vs. Logistic Regression) can learn non-linear interactions
    between base model scores, e.g. when CatBoost and XGBoost disagree.
    """
    logger.info("Training LightGBM stacking ensemble …")
    X_meta = pd.DataFrame(base_oof)

    if LGBM_AVAILABLE:
        # Few leaves to prevent overfitting on the small meta-feature space
        meta = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=7,
            min_child_samples=20,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        logger.warning("LightGBM unavailable — falling back to Logistic Regression meta-learner")
        meta = LogisticRegression(max_iter=500, random_state=random_state)

    meta, oof, metrics = train_model_cv(
        "stacking_meta", meta, X_meta, y, n_folds=n_folds, random_state=random_state
    )
    return meta, oof, metrics


def train_full_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config,
) -> TrainedPipeline:
    result = TrainedPipeline()
    result.feature_names = X_train.columns.tolist()

    # ── Class-imbalance weight ────────────────────────────────────────────
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    pos_weight = n_neg / max(n_pos, 1)
    logger.info("Class ratio neg/pos = %.1f  →  scale_pos_weight=%.2f", pos_weight, pos_weight)

    # ── Optional Bayesian hyperparameter tuning (Optuna) ─────────────────
    lgbm_params: Optional[Dict] = None
    xgb_params:  Optional[Dict] = None
    if getattr(config, "HYPERPARAMETER_TRIALS", 0) > 0:
        try:
            from src.fraud_detection.models.tune import tune_lgbm, tune_xgb
            logger.info("Running Optuna HPO (%d trials per model) …", config.HYPERPARAMETER_TRIALS)
            lgbm_params = tune_lgbm(
                X_train, y_train, pos_weight,
                n_trials=config.HYPERPARAMETER_TRIALS,
                random_state=config.RANDOM_STATE,
            )
            xgb_params = tune_xgb(
                X_train, y_train, pos_weight,
                n_trials=config.HYPERPARAMETER_TRIALS,
                random_state=config.RANDOM_STATE,
            )
            logger.info("Optuna best LGBM params: %s", lgbm_params)
            logger.info("Optuna best XGB params:  %s", xgb_params)
        except Exception as exc:
            logger.warning("Optuna tuning skipped (%s). Using defaults.", exc)

    # ── Train base models ─────────────────────────────────────────────────
    base_models = _build_base_models(config, pos_weight=pos_weight,
                                     lgbm_params=lgbm_params, xgb_params=xgb_params)

    for name, model in base_models.items():
        fitted, oof, metrics = train_model_cv(
            name, model, X_train, y_train,
            n_folds=config.CROSS_VALIDATION_FOLDS,
            random_state=config.RANDOM_STATE,
            patience=config.EARLY_STOPPING_PATIENCE,
        )
        result.models[name] = fitted
        result.oof_predictions[name] = oof
        result.metrics[name] = metrics
        gc.collect()

    # ── Stacking meta-learner ─────────────────────────────────────────────
    if len(result.oof_predictions) > 1:
        meta, meta_oof, meta_metrics = train_stacking_ensemble(
            result.oof_predictions, y_train,
            n_folds=config.CROSS_VALIDATION_FOLDS,
            random_state=config.RANDOM_STATE,
        )
        result.meta_model = meta
        result.metrics["stacking"] = meta_metrics

        # ── Isotonic calibration ──────────────────────────────────────────
        # Fit a calibrator on the OOF meta-predictions (already out-of-fold,
        # so no data leakage) to align raw probabilities with true fraud rates.
        if ISOTONIC_AVAILABLE:
            try:
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(meta_oof, y_train.values)
                result.calibrator = calibrator
                logger.info("Isotonic calibration fitted on %d OOF meta-predictions", len(meta_oof))
            except Exception as exc:
                logger.warning("Calibration skipped: %s", exc)

    # ── Threshold optimisation & final validation metrics ─────────────────
    val_proba = result.predict_proba(X_val)
    result.threshold = optimise_threshold(
        y_val.values, val_proba,
        fn_cost=config.FN_COST_MULTIPLIER,
        fp_cost=config.FP_COST_MULTIPLIER,
    )
    result.metrics["validation"] = calculate_metrics(
        y_val.values, val_proba, result.threshold, label="val_final"
    )

    return result
