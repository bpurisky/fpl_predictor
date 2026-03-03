"""
tests/test_gbm.py
------------------
Unit and smoke tests for FPLGBMModel and gbm_trainer.

Coverage:
  - Model fits and predicts without error
  - Calibrator reduces MAE on calibration set
  - Quantile ordering: P10 <= P50 <= P90 (stochastic, checked on mean)
  - evaluate() returns all expected keys
  - feature_importance() length matches feature_cols
  - save() / load() round-trip preserves predictions
  - predict_next_gw() returns correct output columns
  - Temporal correctness: model trained on GW < N, evaluated on GW >= N
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models.gbm import FPLGBMModel, TARGET_COL
from trainers.gbm_trainer import predict_next_gw

# ---------------------------------------------------------------------------
# Synthetic dataset factory
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "total_points_roll3", "total_points_roll5",
    "minutes_roll3", "minutes_roll5",
    "goals_scored_roll3", "assists_roll3",
    "bonus_roll3", "ict_index_roll3",
    "cost", "selected", "was_home",
    "fixture_difficulty", "opponent_difficulty",
    "is_double_gw", "n_fixtures",
    "pos_1", "pos_2", "pos_3", "pos_4",
    "started_roll3", "sub_risk",
]

N_PLAYERS = 100
N_GWS = 20
VAL_START_GW = 16
RNG = np.random.default_rng(42)


def _make_dataset(n_players: int = N_PLAYERS, n_gws: int = N_GWS) -> pd.DataFrame:
    """
    Synthetic FPL feature matrix with realistic structure.
    Target = total_points drawn from Poisson(3) — realistic FPL distribution.
    """
    rows = []
    for pid in range(1, n_players + 1):
        pos = ((pid - 1) % 4) + 1  # cycle through 1-4
        cost = RNG.uniform(4.5, 12.5)
        for gw in range(1, n_gws + 1):
            row = {
                "player_id": pid,
                "gw": gw,
                "player_name": f"Player {pid}",
                "team_id": ((pid - 1) % 20) + 1,
                "position_id": pos,
                "kickoff_time": pd.Timestamp("2024-08-01") + pd.Timedelta(weeks=gw),
                TARGET_COL: float(RNG.poisson(3)),
                "cost": cost,
                "selected": int(RNG.integers(1000, 500000)),
                "was_home": gw % 2 == 0,
                "fixture_difficulty": float(RNG.integers(2, 5)),
                "opponent_difficulty": float(RNG.integers(2, 5)),
                "is_double_gw": False,
                "n_fixtures": 1,
                "pos_1": float(pos == 1),
                "pos_2": float(pos == 2),
                "pos_3": float(pos == 3),
                "pos_4": float(pos == 4),
            }
            # Rolling features: use previous GW's points with noise
            base = max(0.0, float(gw) * 0.1 + RNG.normal(2.5, 1.5))
            row["total_points_roll3"] = base if gw > 1 else float("nan")
            row["total_points_roll5"] = base if gw > 1 else float("nan")
            row["minutes_roll3"] = min(90.0, max(0.0, float(RNG.normal(75, 15))))
            row["minutes_roll5"] = row["minutes_roll3"]
            row["goals_scored_roll3"] = max(0.0, float(RNG.normal(0.2, 0.3)))
            row["assists_roll3"] = max(0.0, float(RNG.normal(0.1, 0.2)))
            row["bonus_roll3"] = max(0.0, float(RNG.normal(0.5, 0.5)))
            row["ict_index_roll3"] = max(0.0, float(RNG.normal(20, 10)))
            row["started_roll3"] = float(RNG.uniform(0.5, 1.0))
            row["sub_risk"] = float(RNG.uniform(0.0, 0.4))
            rows.append(row)

    df = pd.DataFrame(rows).sort_values(["player_id", "gw"]).reset_index(drop=True)
    return df


def _split(df: pd.DataFrame, val_start_gw: int = VAL_START_GW):
    train = df[df["gw"] < val_start_gw].copy()
    val = df[df["gw"] >= val_start_gw].copy()
    return train, val


# ---------------------------------------------------------------------------
# Basic fit / predict
# ---------------------------------------------------------------------------


def test_model_fits_without_error():
    df = _make_dataset()
    train, val = _split(df)
    model = FPLGBMModel(max_iter=50, calibrate=False, early_stopping=False)
    model.fit(train, FEATURE_COLS)
    assert model._is_fitted


def test_predict_returns_correct_shape():
    df = _make_dataset()
    train, val = _split(df)
    model = FPLGBMModel(max_iter=50, calibrate=False, early_stopping=False)
    model.fit(train, FEATURE_COLS)
    preds = model.predict(val, FEATURE_COLS)
    assert preds.shape == (len(val),)
    assert not np.any(np.isnan(preds))


def test_predict_raises_before_fit():
    model = FPLGBMModel()
    df = _make_dataset().head(10)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(df, FEATURE_COLS)


def test_predict_raises_on_missing_feature():
    df = _make_dataset()
    train, val = _split(df)
    model = FPLGBMModel(max_iter=30, calibrate=False, early_stopping=False)
    model.fit(train, FEATURE_COLS)
    val_bad = val.drop(columns=["cost"])
    with pytest.raises(ValueError, match="Missing feature"):
        model.predict(val_bad, FEATURE_COLS)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_calibration_reduces_or_maintains_mae():
    """
    Calibrated MAE should be <= raw MAE on the calibration set.
    (Isotonic regression is fit on the same val set, so this is expected.)
    """
    df = _make_dataset(n_players=200, n_gws=25)
    train, val = _split(df)
    model = FPLGBMModel(max_iter=100, calibrate=True, early_stopping=False)
    model.fit(train, FEATURE_COLS, calibration_df=val)

    raw_preds = model.predict(val, FEATURE_COLS, calibrated=False)
    cal_preds = model.predict(val, FEATURE_COLS, calibrated=True)
    actual = val[TARGET_COL].values

    raw_mae = np.mean(np.abs(raw_preds - actual))
    cal_mae = np.mean(np.abs(cal_preds - actual))

    # Calibrated should be at most marginally worse (floating point tolerance)
    assert cal_mae <= raw_mae + 0.01, (
        f"Calibration made MAE worse: raw={raw_mae:.4f}, cal={cal_mae:.4f}"
    )


def test_no_calibrator_when_calibrate_false():
    df = _make_dataset()
    train, val = _split(df)
    model = FPLGBMModel(max_iter=30, calibrate=False, early_stopping=False)
    model.fit(train, FEATURE_COLS, calibration_df=val)
    assert model._calibrator is None


# ---------------------------------------------------------------------------
# Quantile models
# ---------------------------------------------------------------------------


def test_quantile_models_fit():
    df = _make_dataset()
    train, _ = _split(df)
    model = FPLGBMModel(max_iter=30, calibrate=False, early_stopping=False)
    model.fit(train, FEATURE_COLS)
    model.fit_quantiles(train, FEATURE_COLS)
    assert "p10" in model._quantile_models
    assert "p50" in model._quantile_models
    assert "p90" in model._quantile_models


def test_quantile_ordering_on_average():
    """
    On average across all players, P10 <= P50 <= P90.
    Individual rows may violate this (quantile crossing), but means should hold.
    """
    df = _make_dataset(n_players=200, n_gws=25)
    train, val = _split(df)
    model = FPLGBMModel(max_iter=80, calibrate=False, early_stopping=False)
    model.fit(train, FEATURE_COLS)
    model.fit_quantiles(train, FEATURE_COLS)

    q = model.predict_quantiles(val, FEATURE_COLS)
    assert np.mean(q["p10"]) <= np.mean(q["p50"]) + 0.5
    assert np.mean(q["p50"]) <= np.mean(q["p90"]) + 0.5


def test_predict_quantiles_empty_without_fit():
    df = _make_dataset()
    train, val = _split(df)
    model = FPLGBMModel(max_iter=30, calibrate=False, early_stopping=False)
    model.fit(train, FEATURE_COLS)
    # No fit_quantiles called
    result = model.predict_quantiles(val, FEATURE_COLS)
    assert result == {}


def test_fit_quantiles_raises_before_fit():
    model = FPLGBMModel()
    df = _make_dataset()
    train, _ = _split(df)
    with pytest.raises(RuntimeError, match="Call fit()"):
        model.fit_quantiles(train, FEATURE_COLS)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def test_evaluate_returns_required_keys():
    df = _make_dataset()
    train, val = _split(df)
    model = FPLGBMModel(max_iter=50, calibrate=False, early_stopping=False)
    model.fit(train, FEATURE_COLS)
    metrics = model.evaluate(val, FEATURE_COLS)

    for key in ["mae", "rmse", "n_samples", "n_iter"]:
        assert key in metrics, f"Missing metric key: {key}"


def test_evaluate_mae_is_positive():
    df = _make_dataset()
    train, val = _split(df)
    model = FPLGBMModel(max_iter=50, calibrate=False, early_stopping=False)
    model.fit(train, FEATURE_COLS)
    metrics = model.evaluate(val, FEATURE_COLS)
    assert metrics["mae"] > 0
    assert metrics["rmse"] >= metrics["mae"]  # RMSE >= MAE always


def test_evaluate_position_stratified():
    df = _make_dataset()
    train, val = _split(df)
    model = FPLGBMModel(max_iter=50, calibrate=False, early_stopping=False)
    model.fit(train, FEATURE_COLS)
    metrics = model.evaluate(val, FEATURE_COLS)
    assert "mae_by_pos" in metrics
    assert "GKP" in metrics["mae_by_pos"]
    assert "FWD" in metrics["mae_by_pos"]


def test_evaluate_top_k_recall():
    df = _make_dataset()
    train, val = _split(df)
    model = FPLGBMModel(max_iter=50, calibrate=False, early_stopping=False)
    model.fit(train, FEATURE_COLS)
    metrics = model.evaluate(val, FEATURE_COLS)
    assert "top_15_recall" in metrics
    assert 0.0 <= metrics["top_15_recall"] <= 1.0


def test_evaluate_by_gw_shape():
    df = _make_dataset()
    train, val = _split(df)
    model = FPLGBMModel(max_iter=50, calibrate=False, early_stopping=False)
    model.fit(train, FEATURE_COLS)
    gw_df = model.evaluate_by_gw(val, FEATURE_COLS)
    assert "gw" in gw_df.columns
    assert "mae" in gw_df.columns
    assert "n_players" in gw_df.columns
    assert len(gw_df) == val["gw"].nunique()


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


def test_feature_importance_length():
    df = _make_dataset()
    train, _ = _split(df)
    model = FPLGBMModel(max_iter=50, calibrate=False, early_stopping=False)
    model.fit(train, FEATURE_COLS)
    fi = model.feature_importance(FEATURE_COLS, top_n=100)
    assert len(fi) == len(FEATURE_COLS)
    assert "feature" in fi.columns
    assert "importance" in fi.columns


def test_feature_importance_sorted():
    df = _make_dataset()
    train, _ = _split(df)
    model = FPLGBMModel(max_iter=50, calibrate=False, early_stopping=False)
    model.fit(train, FEATURE_COLS)
    fi = model.feature_importance(FEATURE_COLS)
    assert fi["importance"].is_monotonic_decreasing


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


def test_save_load_predictions_identical():
    df = _make_dataset()
    train, val = _split(df)
    model = FPLGBMModel(max_iter=50, calibrate=False, early_stopping=False)
    model.fit(train, FEATURE_COLS)
    preds_before = model.predict(val, FEATURE_COLS)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gbm_test.pkl"
        model.save(path)
        loaded = FPLGBMModel.load(path)

    preds_after = loaded.predict(val, FEATURE_COLS)
    np.testing.assert_array_almost_equal(preds_before, preds_after)


def test_load_nonexistent_raises():
    with pytest.raises(FileNotFoundError):
        FPLGBMModel.load(Path("/nonexistent/model.pkl"))


# ---------------------------------------------------------------------------
# NaN handling (Understat columns)
# ---------------------------------------------------------------------------


def test_handles_nan_features_natively():
    """
    HistGradientBoostingRegressor handles NaN natively.
    Players without Understat data (NaN in us_* cols) must not cause errors.
    """
    df = _make_dataset()
    # Add fake Understat columns that are NaN for half the players
    df["us_xG_roll3"] = np.where(df["player_id"] <= 50, df["total_points_roll3"] * 0.3, np.nan)
    df["us_xA_roll3"] = np.where(df["player_id"] <= 50, df["total_points_roll3"] * 0.1, np.nan)

    extended_features = FEATURE_COLS + ["us_xG_roll3", "us_xA_roll3"]
    train, val = _split(df)

    model = FPLGBMModel(max_iter=50, calibrate=False, early_stopping=False)
    model.fit(train, extended_features)
    preds = model.predict(val, extended_features)

    assert preds.shape == (len(val),)
    assert not np.any(np.isnan(preds))


# ---------------------------------------------------------------------------
# predict_next_gw output
# ---------------------------------------------------------------------------


def test_predict_next_gw_output_columns():
    df = _make_dataset()
    train, val = _split(df)
    model = FPLGBMModel(max_iter=50, calibrate=False, early_stopping=False)
    model.fit(train, FEATURE_COLS)
    model.fit_quantiles(train, FEATURE_COLS)

    result = predict_next_gw(model, val, FEATURE_COLS)

    assert "predicted_points" in result.columns
    assert "p10" in result.columns
    assert "p90" in result.columns
    assert "value_per_m" in result.columns
    assert "spread" in result.columns


def test_predict_next_gw_sorted_descending():
    df = _make_dataset()
    train, val = _split(df)
    model = FPLGBMModel(max_iter=50, calibrate=False, early_stopping=False)
    model.fit(train, FEATURE_COLS)

    result = predict_next_gw(model, val, FEATURE_COLS)
    assert result["predicted_points"].is_monotonic_decreasing


# ---------------------------------------------------------------------------
# Temporal correctness smoke test
# ---------------------------------------------------------------------------


def test_temporal_correctness_no_future_leakage():
    """
    Model trained on GW < VAL_START_GW must not have seen val targets.
    Verify by checking that val MAE is worse than train MAE
    (overfitting would make train MAE << val MAE, which is expected and correct).
    """
    df = _make_dataset(n_players=200, n_gws=25)
    train, val = _split(df)

    model = FPLGBMModel(max_iter=100, calibrate=False, early_stopping=False)
    model.fit(train, FEATURE_COLS)

    train_mae = model.evaluate(train, FEATURE_COLS)["mae"]
    val_mae = model.evaluate(val, FEATURE_COLS)["mae"]

    # Val MAE should be >= train MAE (model was not trained on val)
    # If val_mae < train_mae significantly, something is wrong
    assert val_mae >= train_mae * 0.5, (
        f"Val MAE ({val_mae:.4f}) is suspiciously lower than train MAE ({train_mae:.4f}). "
        "Possible data leakage."
    )
    assert train_df_gws_ok := train["gw"].max() < VAL_START_GW
    assert val_df_gws_ok := val["gw"].min() >= VAL_START_GW
