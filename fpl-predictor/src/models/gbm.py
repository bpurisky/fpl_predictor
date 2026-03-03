"""
src/models/gbm.py
------------------
Gradient Boosting Model for FPL points prediction.

Design decisions:
  - HistGradientBoostingRegressor (sklearn) for native NaN support —
    critical because us_* Understat features are NaN for unmapped players
  - Absolute error loss: more robust to the heavy-tailed FPL point
    distribution (hauls skew the distribution; MAE loss downweights them)
  - Isotonic regression calibration on residuals: adjusts for systematic
    over/under-prediction by position or price tier
  - Quantile variants (P10, P50, P90) for risk-aware selection
  - Position-stratified evaluation: GKP/DEF/MID/FWD have very different
    point distributions — overall MAE masks position-level errors

INTERFACE:
  FPLGBMModel wraps sklearn estimators with:
    fit(train_df, feature_cols)
    predict(df, feature_cols) → np.ndarray
    predict_quantiles(df, feature_cols) → dict[str, np.ndarray]
    evaluate(val_df, feature_cols) → dict of metrics
    save(path) / load(path)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)

TARGET_COL = "total_points"

# Quantile levels for risk-aware predictions
QUANTILE_LEVELS = {"p10": 0.10, "p50": 0.50, "p90": 0.90}

# Position labels for stratified eval
POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


# ---------------------------------------------------------------------------
# Core model class
# ---------------------------------------------------------------------------


class FPLGBMModel:
    """
    Gradient Boosting regressor for FPL points prediction.

    Parameters
    ----------
    learning_rate : float
    max_iter : int — number of boosting rounds
    max_depth : int | None
    min_samples_leaf : int
    l2_regularization : float
    max_features : float — fraction of features per split (like colsample)
    random_state : int
    calibrate : bool — fit isotonic regression on val residuals after training
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        max_iter: int = 500,
        max_depth: int | None = 6,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.1,
        max_features: float = 0.8,
        random_state: int = 42,
        calibrate: bool = True,
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 20,
        tol: float = 1e-4,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_features = max_features
        self.random_state = random_state
        self.calibrate = calibrate
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol

        self._model: HistGradientBoostingRegressor | None = None
        self._calibrator: IsotonicRegression | None = None
        self._quantile_models: dict[str, HistGradientBoostingRegressor] = {}
        self._feature_cols: list[str] = []
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_estimator(
        self,
        loss: str = "absolute_error",
        quantile: float | None = None,
    ) -> HistGradientBoostingRegressor:
        """Build a HistGradientBoostingRegressor with current config."""
        kwargs: dict[str, Any] = dict(
            loss=loss,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            max_features=self.max_features,
            random_state=self.random_state,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction if self.early_stopping else None,
            n_iter_no_change=self.n_iter_no_change if self.early_stopping else None,
            tol=self.tol,
        )
        if loss == "quantile" and quantile is not None:
            kwargs["quantile"] = quantile

        return HistGradientBoostingRegressor(**kwargs)

    def _get_X_y(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Extract feature matrix and target from DataFrame."""
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns in DataFrame: {missing}")
        X = df[feature_cols].copy()
        y = df[TARGET_COL].copy()
        return X, y

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: list[str],
        calibration_df: pd.DataFrame | None = None,
    ) -> "FPLGBMModel":
        """
        Fit the main point-prediction model.

        Parameters
        ----------
        train_df : training data (rows where gw < val_start_gw)
        feature_cols : list of feature column names from build_dataset
        calibration_df : optional held-out set for isotonic calibration.
                         If None and calibrate=True, uses internal
                         early-stopping validation split instead.
                         Best practice: pass your val_df here.

        Returns self for chaining.
        """
        self._feature_cols = feature_cols
        X_train, y_train = self._get_X_y(train_df, feature_cols)

        logger.info(
            "Fitting GBM: %d rows × %d features, loss=absolute_error",
            len(X_train), len(feature_cols),
        )

        self._model = self._make_estimator(loss="absolute_error")
        self._model.fit(X_train, y_train)

        n_iters = getattr(self._model, "n_iter_", self.max_iter)
        logger.info("GBM fit complete. Iterations used: %d", n_iters)

        # Isotonic calibration on residuals
        if self.calibrate and calibration_df is not None:
            self._fit_calibrator(calibration_df, feature_cols)
        elif self.calibrate:
            logger.warning(
                "calibrate=True but no calibration_df provided. "
                "Pass val_df to fit() for proper calibration."
            )

        self._is_fitted = True
        return self

    def fit_quantiles(
        self,
        train_df: pd.DataFrame,
        feature_cols: list[str],
        quantile_levels: dict[str, float] = QUANTILE_LEVELS,
    ) -> "FPLGBMModel":
        """
        Fit separate quantile regression models (P10, P50, P90).

        Call after fit() — uses the same feature_cols.
        Quantile models are used for risk-aware player selection:
          - High P90 with low P10 = boom-or-bust (risky captain)
          - High P10 = floor player (reliable)

        Parameters
        ----------
        train_df : same training data used in fit()
        feature_cols : same feature columns
        quantile_levels : dict mapping label → quantile e.g. {'p10': 0.10}
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before fit_quantiles()")

        X_train, y_train = self._get_X_y(train_df, feature_cols)

        for label, q in quantile_levels.items():
            logger.info("Fitting quantile model: %s (q=%.2f)", label, q)
            qmodel = self._make_estimator(loss="quantile", quantile=q)
            qmodel.fit(X_train, y_train)
            self._quantile_models[label] = qmodel
            logger.info("Quantile %s fit complete", label)

        return self

    def _fit_calibrator(
        self,
        calibration_df: pd.DataFrame,
        feature_cols: list[str],
    ) -> None:
        """
        Fit isotonic regression on (raw_prediction, actual) pairs.

        Isotonic regression learns a monotone mapping from raw predictions
        to calibrated predictions, correcting systematic bias without
        disrupting rank ordering.
        """
        X_cal, y_cal = self._get_X_y(calibration_df, feature_cols)
        raw_preds = self._model.predict(X_cal)

        self._calibrator = IsotonicRegression(out_of_bounds="clip")
        self._calibrator.fit(raw_preds, y_cal)

        cal_preds = self._calibrator.predict(raw_preds)
        raw_mae = np.mean(np.abs(raw_preds - y_cal))
        cal_mae = np.mean(np.abs(cal_preds - y_cal))

        logger.info(
            "Calibration: raw MAE=%.4f → calibrated MAE=%.4f (Δ=%.4f)",
            raw_mae, cal_mae, raw_mae - cal_mae,
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        calibrated: bool = True,
    ) -> np.ndarray:
        """
        Predict expected points. Returns 1D np.ndarray.

        Parameters
        ----------
        df : DataFrame with feature columns
        feature_cols : if None, uses columns from fit()
        calibrated : apply isotonic calibration if available
        """
        self._check_fitted()
        feature_cols = feature_cols or self._feature_cols
        X, _ = self._get_X_y(df, feature_cols)

        raw = self._model.predict(X)

        if calibrated and self._calibrator is not None:
            return self._calibrator.predict(raw)
        return raw

    def predict_quantiles(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Predict P10, P50, P90 quantiles.

        Returns dict mapping label → np.ndarray of predictions.
        Empty dict if fit_quantiles() was not called.
        """
        self._check_fitted()
        if not self._quantile_models:
            logger.warning("No quantile models fitted. Call fit_quantiles() first.")
            return {}

        feature_cols = feature_cols or self._feature_cols
        X, _ = self._get_X_y(df, feature_cols)

        return {
            label: model.predict(X)
            for label, model in self._quantile_models.items()
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        val_df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        calibrated: bool = True,
    ) -> dict[str, Any]:
        """
        Compute evaluation metrics on val_df.

        Returns dict with:
            mae          : overall mean absolute error
            rmse         : root mean squared error
            mae_by_pos   : MAE broken down by position (GKP/DEF/MID/FWD)
            top_k_recall : fraction of actual top-K scorers captured in
                           predicted top-K (K=15 and K=30)
            n_iter       : boosting rounds used
        """
        self._check_fitted()
        feature_cols = feature_cols or self._feature_cols

        preds = self.predict(val_df, feature_cols, calibrated=calibrated)
        actual = val_df[TARGET_COL].values

        mae = float(np.mean(np.abs(preds - actual)))
        rmse = float(np.sqrt(np.mean((preds - actual) ** 2)))

        metrics: dict[str, Any] = {
            "mae": mae,
            "rmse": rmse,
            "n_samples": len(val_df),
            "n_iter": getattr(self._model, "n_iter_", self.max_iter),
            "calibrated": calibrated and self._calibrator is not None,
        }

        # Position-stratified MAE
        if "position_id" in val_df.columns:
            mae_by_pos: dict[str, float] = {}
            for pos_id, pos_name in POSITION_MAP.items():
                mask = val_df["position_id"] == pos_id
                if mask.sum() > 0:
                    pos_mae = float(np.mean(np.abs(preds[mask] - actual[mask])))
                    mae_by_pos[pos_name] = pos_mae
            metrics["mae_by_pos"] = mae_by_pos

        # Top-K recall: of actual top scorers, how many did we predict?
        for k in [15, 30]:
            if len(actual) >= k:
                actual_top = set(np.argsort(actual)[-k:])
                pred_top = set(np.argsort(preds)[-k:])
                recall = len(actual_top & pred_top) / k
                metrics[f"top_{k}_recall"] = round(recall, 4)

        return metrics

    def evaluate_by_gw(
        self,
        val_df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        calibrated: bool = True,
    ) -> pd.DataFrame:
        """
        Compute MAE per gameweek in val_df.

        Useful for spotting performance degradation late in season
        or after major player injuries.

        Returns DataFrame with columns: gw, mae, n_players
        """
        self._check_fitted()
        feature_cols = feature_cols or self._feature_cols

        preds = self.predict(val_df, feature_cols, calibrated=calibrated)
        df = val_df.copy()
        df["_pred"] = preds
        df["_abs_err"] = np.abs(preds - df[TARGET_COL].values)

        gw_metrics = (
            df.groupby("gw")
            .agg(mae=("_abs_err", "mean"), n_players=("_abs_err", "count"))
            .reset_index()
        )
        return gw_metrics

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(
        self,
        feature_cols: list[str] | None = None,
        top_n: int = 30,
    ) -> pd.DataFrame:
        """
        Return feature importances from the main model.

        HistGradientBoostingRegressor uses split-gain importance.
        Returns DataFrame sorted by importance descending.
        """
        self._check_fitted()
        feature_cols = feature_cols or self._feature_cols

        importances = self._model.feature_importances_
        df = pd.DataFrame({
            "feature": feature_cols,
            "importance": importances,
        }).sort_values("importance", ascending=False).head(top_n)

        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Pickle the full model object to disk."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
        logger.info("GBM model saved to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> "FPLGBMModel":
        """Load a saved model from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No model file at {path}")
        with path.open("rb") as f:
            obj = pickle.load(f)
        logger.info("GBM model loaded from %s", path)
        return obj

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        cal = "calibrated" if self._calibrator else "uncalibrated"
        q = list(self._quantile_models.keys()) or "none"
        return (
            f"FPLGBMModel({status}, {cal}, "
            f"lr={self.learning_rate}, max_iter={self.max_iter}, "
            f"quantiles={q})"
        )
