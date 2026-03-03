"""
src/trainers/gbm_trainer.py
----------------------------
Training orchestrator for FPLGBMModel.

Responsibilities:
  - Load processed train/val parquet files
  - Fit main model + quantile models
  - Evaluate on val set (overall + per-GW + per-position)
  - Log config + metrics to a JSON experiment record
  - Save model artifact with versioned filename
  - Print a ranked prediction table for the next GW

This is the entry point called by scripts/train_gbm.py.
It is NOT a hyperparameter search — that belongs in a separate
experiment script. This trainer runs one config cleanly.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.gbm import FPLGBMModel, TARGET_COL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_MODEL_DIR = Path("models/artifacts")
DEFAULT_EXPERIMENT_DIR = Path("models/experiments")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_splits(
    processed_dir: Path,
    version: str = "v1",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Load train and val DataFrames from processed parquet files.

    Returns (train_df, val_df, feature_cols).
    feature_cols are inferred as all non-ID, non-target columns.
    """
    train_path = processed_dir / f"train_{version}.parquet"
    val_path = processed_dir / f"val_{version}.parquet"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Train file not found: {train_path}\n"
            "Run build_dataset() with processed_dir set first."
        )
    if not val_path.exists():
        raise FileNotFoundError(f"Val file not found: {val_path}")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    # Infer feature cols: everything except ID cols and target
    non_feature = {
        "player_id", "gw", "player_name", "team_id",
        "position_id", "kickoff_time", TARGET_COL,
    }
    feature_cols = [c for c in train_df.columns if c not in non_feature]

    logger.info(
        "Loaded train: %d rows, val: %d rows, %d features",
        len(train_df), len(val_df), len(feature_cols),
    )
    return train_df, val_df, feature_cols


# ---------------------------------------------------------------------------
# Experiment logging
# ---------------------------------------------------------------------------


def _build_experiment_record(
    config: dict,
    metrics: dict,
    gw_metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    model_path: Path,
    train_gws: tuple[int, int],
    val_gws: tuple[int, int],
    elapsed_seconds: float,
) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "config": config,
        "train_gw_range": list(train_gws),
        "val_gw_range": list(val_gws),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "metrics": metrics,
        "gw_metrics": gw_metrics.to_dict(orient="records"),
        "top_features": feature_importance.head(20).to_dict(orient="records"),
    }


def save_experiment(record: dict, experiment_dir: Path, version: str) -> Path:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = experiment_dir / f"gbm_{version}_{ts}.json"
    with path.open("w") as f:
        json.dump(record, f, indent=2, default=str)
    logger.info("Experiment record saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------


def _print_metrics(metrics: dict) -> None:
    print("\n" + "=" * 55)
    print("  GBM VALIDATION METRICS")
    print("=" * 55)
    print(f"  MAE (overall):   {metrics['mae']:.4f}")
    print(f"  RMSE:            {metrics['rmse']:.4f}")
    print(f"  Iterations used: {metrics.get('n_iter', '?')}")
    print(f"  Calibrated:      {metrics.get('calibrated', False)}")

    if "mae_by_pos" in metrics:
        print("\n  MAE by position:")
        for pos, mae in metrics["mae_by_pos"].items():
            print(f"    {pos:<5} {mae:.4f}")

    if "top_15_recall" in metrics:
        print(f"\n  Top-15 recall:   {metrics['top_15_recall']:.1%}")
    if "top_30_recall" in metrics:
        print(f"  Top-30 recall:   {metrics['top_30_recall']:.1%}")
    print("=" * 55 + "\n")


def _print_gw_metrics(gw_df: pd.DataFrame) -> None:
    print("  MAE per gameweek (validation):")
    print(f"  {'GW':<6} {'MAE':<10} {'N Players'}")
    print("  " + "-" * 30)
    for _, row in gw_df.iterrows():
        print(f"  {int(row['gw']):<6} {row['mae']:<10.4f} {int(row['n_players'])}")
    print()


def print_ranked_predictions(
    pred_df: pd.DataFrame,
    top_n: int = 30,
) -> None:
    """
    Print ranked prediction table for the next GW.

    pred_df must have columns: player_name, position_id, cost,
    predicted_points, and optionally p10, p50, p90.
    """
    cols = ["player_name", "position_id", "cost", "predicted_points"]
    optional = ["p10", "p50", "p90", "value_per_m"]
    cols += [c for c in optional if c in pred_df.columns]

    top = pred_df.sort_values("predicted_points", ascending=False).head(top_n)

    pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    top = top.copy()
    top["pos"] = top["position_id"].map(pos_map)

    print("\n" + "=" * 70)
    print(f"  TOP {top_n} PREDICTED PLAYERS")
    print("=" * 70)
    header = f"  {'#':<4} {'Player':<25} {'Pos':<5} {'£':<6} {'xPts':<7}"
    if "p10" in top.columns:
        header += f" {'P10':<6} {'P90':<6}"
    if "value_per_m" in top.columns:
        header += f" {'Val/£':<6}"
    print(header)
    print("  " + "-" * 66)

    for rank, (_, row) in enumerate(top.iterrows(), 1):
        line = (
            f"  {rank:<4} {row['player_name']:<25} "
            f"{row['pos']:<5} {row['cost']:<6.1f} "
            f"{row['predicted_points']:<7.2f}"
        )
        if "p10" in row:
            line += f" {row['p10']:<6.2f} {row['p90']:<6.2f}"
        if "value_per_m" in row:
            line += f" {row['value_per_m']:<6.3f}"
        print(line)
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_gbm(
    config: dict,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    model_dir: Path = DEFAULT_MODEL_DIR,
    experiment_dir: Path = DEFAULT_EXPERIMENT_DIR,
    version: str = "v1",
    fit_quantiles: bool = True,
    verbose: bool = True,
) -> tuple[FPLGBMModel, dict]:
    """
    Full GBM training run.

    Parameters
    ----------
    config : dict from config/gbm.yaml (see below for expected keys)
    processed_dir : location of train/val parquet files
    model_dir : where to save model artifact
    experiment_dir : where to save experiment JSON
    version : dataset version tag
    fit_quantiles : whether to also fit P10/P50/P90 models
    verbose : print metrics and ranked table

    Returns
    -------
    (fitted_model, metrics_dict)

    Expected config keys (all optional, defaults used if missing):
        gbm.learning_rate, gbm.max_iter, gbm.max_depth,
        gbm.min_samples_leaf, gbm.l2_regularization,
        gbm.max_features, gbm.random_state, gbm.calibrate,
        gbm.early_stopping, gbm.n_iter_no_change
    """
    t0 = time.monotonic()
    gbm_cfg = config.get("gbm", {})

    # 1. Load data
    train_df, val_df, feature_cols = load_splits(processed_dir, version)

    train_gws = (int(train_df["gw"].min()), int(train_df["gw"].max()))
    val_gws = (int(val_df["gw"].min()), int(val_df["gw"].max()))

    # 2. Build model
    model = FPLGBMModel(
        learning_rate=gbm_cfg.get("learning_rate", 0.05),
        max_iter=gbm_cfg.get("max_iter", 500),
        max_depth=gbm_cfg.get("max_depth", 6),
        min_samples_leaf=gbm_cfg.get("min_samples_leaf", 20),
        l2_regularization=gbm_cfg.get("l2_regularization", 0.1),
        max_features=gbm_cfg.get("max_features", 0.8),
        random_state=gbm_cfg.get("random_state", 42),
        calibrate=gbm_cfg.get("calibrate", True),
        early_stopping=gbm_cfg.get("early_stopping", True),
        n_iter_no_change=gbm_cfg.get("n_iter_no_change", 20),
    )

    # 3. Fit — pass val_df as calibration set
    model.fit(train_df, feature_cols, calibration_df=val_df)

    # 4. Fit quantile models
    if fit_quantiles:
        model.fit_quantiles(train_df, feature_cols)

    # 5. Evaluate
    metrics = model.evaluate(val_df, feature_cols, calibrated=True)
    gw_metrics = model.evaluate_by_gw(val_df, feature_cols, calibrated=True)
    feat_imp = model.feature_importance(feature_cols)

    elapsed = time.monotonic() - t0

    if verbose:
        _print_metrics(metrics)
        _print_gw_metrics(gw_metrics)
        print(f"  Top 10 features:")
        for _, row in feat_imp.head(10).iterrows():
            print(f"    {row['feature']:<45} {row['importance']:.4f}")
        print()

    # 6. Save model
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_path = Path(model_dir) / f"gbm_{version}_{ts}.pkl"
    model.save(model_path)

    # 7. Save experiment record
    record = _build_experiment_record(
        config=config,
        metrics=metrics,
        gw_metrics=gw_metrics,
        feature_importance=feat_imp,
        model_path=model_path,
        train_gws=train_gws,
        val_gws=val_gws,
        elapsed_seconds=elapsed,
    )
    save_experiment(record, Path(experiment_dir), version)

    logger.info(
        "Training complete in %.1fs | val MAE=%.4f",
        elapsed, metrics["mae"],
    )

    return model, metrics


# ---------------------------------------------------------------------------
# Prediction helper (for weekly inference after training)
# ---------------------------------------------------------------------------


def predict_next_gw(
    model: FPLGBMModel,
    prediction_features: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Run inference on the upcoming GW feature DataFrame.

    Parameters
    ----------
    model : fitted FPLGBMModel
    prediction_features : output of build_dataset.build_prediction_features()
    feature_cols : same list used during training

    Returns
    -------
    DataFrame with player metadata + predicted_points + quantiles + value_per_m
    Sorted by predicted_points descending.
    """
    preds = model.predict(prediction_features, feature_cols, calibrated=True)
    quantiles = model.predict_quantiles(prediction_features, feature_cols)

    out = prediction_features[
        [c for c in ["player_id", "player_name", "team_id", "position_id",
                      "gw", "cost", "fixture_difficulty"]
         if c in prediction_features.columns]
    ].copy()

    out["predicted_points"] = preds

    for label, q_preds in quantiles.items():
        out[label] = q_preds

    # Value metric: predicted points per £m
    if "cost" in out.columns:
        out["value_per_m"] = np.where(
            out["cost"] > 0,
            out["predicted_points"] / out["cost"],
            np.nan,
        )

    # Spread (risk metric): P90 - P10
    if "p10" in out.columns and "p90" in out.columns:
        out["spread"] = out["p90"] - out["p10"]

    return out.sort_values("predicted_points", ascending=False).reset_index(drop=True)
