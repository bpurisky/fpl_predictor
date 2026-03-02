"""
tests/test_build_dataset.py
---------------------------
Unit tests for build_dataset.py.

Key coverage:
  - Rolling features respect shift(1) — no leakage
  - Temporal split is strictly ordered
  - Fixture merge correctness
  - Prediction features use only past data
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.build_dataset import (
    add_minutes_features,
    add_rolling_features,
    build_feature_matrix,
    build_fixture_df,
    build_history_df,
    merge_fixture_context,
    temporal_split,
)

# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------


def _make_history_df(n_players: int = 2, n_gws: int = 8) -> pd.DataFrame:
    """Synthetic sorted (player_id, gw) DataFrame."""
    rows = []
    for pid in range(1, n_players + 1):
        for gw in range(1, n_gws + 1):
            rows.append({
                "player_id": pid,
                "gw": gw,
                "total_points": float(gw + pid),
                "minutes": 90.0,
                "goals_scored": 0.0,
                "assists": 0.0,
                "clean_sheets": 0.0,
                "goals_conceded": 0.0,
                "bonus": 0.0,
                "bps": 0.0,
                "influence": 1.0,
                "creativity": 1.0,
                "threat": 1.0,
                "ict_index": 3.0,
                "expected_goals": 0.1,
                "expected_assists": 0.05,
                "expected_goal_involvements": 0.15,
                "cost": 5.0,
                "value": 50.0,
                "selected": 10000,
                "team_id": pid,
                "position_id": 3,
                "was_home": gw % 2 == 0,
                "player_name": f"Player {pid}",
                "kickoff_time": pd.Timestamp("2024-08-01", tz="UTC") + pd.Timedelta(weeks=gw),
            })
    return pd.DataFrame(rows).sort_values(["player_id", "gw"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# build_history_df
# ---------------------------------------------------------------------------


def test_build_history_df_shape():
    elements = [
        {"id": 1, "first_name": "A", "second_name": "B",
         "element_type": 3, "team": 1, "now_cost": 50},
    ]
    summaries = {
        1: {
            "history": [
                {"element": 1, "round": 1, "total_points": 6, "minutes": 90,
                 "kickoff_time": "2024-08-01T15:00:00Z", "was_home": True,
                 "opponent_team": 2, "goals_scored": 1, "assists": 0,
                 "clean_sheets": 0, "goals_conceded": 0, "bonus": 0,
                 "bps": 20, "influence": "5.0", "creativity": "3.0",
                 "threat": "4.0", "ict_index": "12.0", "value": 50,
                 "selected": 5000, "transfers_in": 100, "transfers_out": 50,
                 "yellow_cards": 0, "red_cards": 0, "saves": 0},
            ],
            "history_past": [],
            "fixtures": [],
        }
    }
    df = build_history_df(elements, summaries)
    assert len(df) == 1
    assert "player_id" in df.columns
    assert "gw" in df.columns
    assert "total_points" in df.columns
    assert "cost" in df.columns


def test_build_history_df_empty_history_warns():
    elements = [{"id": 99, "first_name": "X", "second_name": "Y",
                 "element_type": 1, "team": 1, "now_cost": 40}]
    summaries = {99: {"history": [], "history_past": [], "fixtures": []}}
    with pytest.raises(ValueError, match="No history rows"):
        build_history_df(elements, summaries)


# ---------------------------------------------------------------------------
# Rolling features — TEMPORAL LEAKAGE TESTS
# ---------------------------------------------------------------------------


def test_rolling_features_no_leakage_first_row():
    """
    First row for each player must have NaN rolling features
    (shift(1) means there's no past data for GW 1).
    """
    df = _make_history_df(n_players=1, n_gws=6)
    result = add_rolling_features(df, windows=[3], ewm_spans=[])

    first_row = result[result["gw"] == 1].iloc[0]
    # With shift(1) and min_periods=1, GW1 rolled value uses nothing → NaN
    # (rolling on shifted series: row 0 shifted = NaN, rolling(min_periods=1) = NaN)
    assert pd.isna(first_row["total_points_roll3"])


def test_rolling_features_second_row_uses_only_first():
    """
    GW 2's roll3 must equal GW 1's value (only one past observation).
    """
    df = _make_history_df(n_players=1, n_gws=6)
    result = add_rolling_features(df, windows=[3], ewm_spans=[])

    gw1_points = df[df["gw"] == 1]["total_points"].iloc[0]
    gw2_roll = result[result["gw"] == 2]["total_points_roll3"].iloc[0]

    assert abs(gw2_roll - gw1_points) < 1e-9, (
        f"GW2 roll3 ({gw2_roll}) should equal GW1 total_points ({gw1_points})"
    )


def test_rolling_features_value_never_equals_current_gw_target():
    """
    The rolling feature at row t must NOT equal the target at row t
    (which would indicate leakage if point values are monotonically distinct).
    """
    df = _make_history_df(n_players=1, n_gws=10)
    result = add_rolling_features(df, windows=[3], ewm_spans=[])

    # For GW 5 onwards, if roll3 == current total_points, that's suspicious
    # (We constructed points as gw + pid so they're all unique and increasing)
    for _, row in result[result["gw"] >= 5].iterrows():
        assert row["total_points_roll3"] != row["total_points"], (
            f"GW {row['gw']}: roll3 equals current points — possible leakage"
        )


def test_rolling_features_players_are_independent():
    """Rolling window must not bleed across player groups."""
    df = _make_history_df(n_players=2, n_gws=5)
    result = add_rolling_features(df, windows=[5], ewm_spans=[])

    p1 = result[result["player_id"] == 1]["total_points_roll5"].values
    p2 = result[result["player_id"] == 2]["total_points_roll5"].values

    # If player windows bled, values would be mixed
    # Check that player 2's GW2 roll doesn't include player 1's data
    # Player 2 GW2 should use only Player 2 GW1 data
    p2_gw2 = result[(result["player_id"] == 2) & (result["gw"] == 2)]["total_points_roll5"].iloc[0]
    expected = df[(df["player_id"] == 2) & (df["gw"] == 1)]["total_points"].iloc[0]
    assert abs(p2_gw2 - expected) < 1e-9


# ---------------------------------------------------------------------------
# Minutes features
# ---------------------------------------------------------------------------


def test_minutes_features_added():
    df = _make_history_df(n_players=1, n_gws=6)
    result = add_minutes_features(df)
    assert "started" in result.columns
    assert "started_roll3" in result.columns
    assert "sub_risk" in result.columns


def test_started_flag():
    df = _make_history_df(n_players=1, n_gws=3)
    df.loc[df["gw"] == 2, "minutes"] = 30.0  # not started GW 2
    result = add_minutes_features(df)
    assert result[result["gw"] == 1]["started"].iloc[0] == 1.0
    assert result[result["gw"] == 2]["started"].iloc[0] == 0.0


# ---------------------------------------------------------------------------
# Fixture DataFrame
# ---------------------------------------------------------------------------


def test_build_fixture_df_basic():
    fixtures = [
        {"id": 1, "event": 1, "team_h": 1, "team_a": 2,
         "team_h_difficulty": 3, "team_a_difficulty": 2, "finished": True},
    ]
    teams = [{"id": 1, "short_name": "ARS"}, {"id": 2, "short_name": "CHE"}]
    df = build_fixture_df(fixtures, teams)
    assert len(df) == 2  # one row per team
    assert set(df["team_id"]) == {1, 2}


def test_build_fixture_df_double_gw():
    fixtures = [
        {"id": 1, "event": 25, "team_h": 1, "team_a": 2,
         "team_h_difficulty": 3, "team_a_difficulty": 2, "finished": False},
        {"id": 2, "event": 25, "team_h": 3, "team_a": 1,
         "team_h_difficulty": 4, "team_a_difficulty": 5, "finished": False},
    ]
    teams = [{"id": i, "short_name": f"T{i}"} for i in range(1, 4)]
    df = build_fixture_df(fixtures, teams)
    team1 = df[df["team_id"] == 1]
    assert team1["is_double_gw"].all()
    assert team1["n_fixtures"].iloc[0] == 2


def test_build_fixture_df_blank_gw():
    """A team with no fixture in a GW should simply not appear."""
    fixtures = [
        {"id": 1, "event": 10, "team_h": 1, "team_a": 2,
         "team_h_difficulty": 3, "team_a_difficulty": 2, "finished": False},
    ]
    teams = [{"id": i, "short_name": f"T{i}"} for i in [1, 2, 3]]
    df = build_fixture_df(fixtures, teams)
    assert 3 not in df["team_id"].values  # team 3 has blank GW 10


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------


def test_temporal_split_no_overlap():
    df = _make_history_df(n_players=2, n_gws=10)
    train, val = temporal_split(df, val_start_gw=7)
    assert train["gw"].max() < 7
    assert val["gw"].min() >= 7
    assert len(set(train["gw"]) & set(val["gw"])) == 0


def test_temporal_split_ordering():
    df = _make_history_df(n_players=1, n_gws=10)
    train, val = temporal_split(df, val_start_gw=6)
    # Train must be GWs 1-5, val GWs 6-10
    assert set(train["gw"]) == {1, 2, 3, 4, 5}
    assert set(val["gw"]) == {6, 7, 8, 9, 10}


def test_temporal_split_empty_train_raises():
    df = _make_history_df(n_players=1, n_gws=5)
    with pytest.raises(ValueError, match="Train set is empty"):
        temporal_split(df, val_start_gw=1)


def test_temporal_split_empty_val_raises():
    df = _make_history_df(n_players=1, n_gws=5)
    with pytest.raises(ValueError, match="Val set is empty"):
        temporal_split(df, val_start_gw=99)


# ---------------------------------------------------------------------------
# build_feature_matrix
# ---------------------------------------------------------------------------


def test_feature_matrix_drops_nan_target():
    df = _make_history_df(n_players=1, n_gws=5)
    df = add_rolling_features(df, windows=[3], ewm_spans=[3])
    df = add_minutes_features(df)
    # Inject NaN target
    df.loc[df["gw"] == 3, "total_points"] = np.nan
    df["fixture_difficulty"] = 3.0
    df["opponent_difficulty"] = 2.0
    df["is_double_gw"] = False
    df["n_fixtures"] = 1

    feat_df, feat_cols = build_feature_matrix(df)
    assert 3 not in feat_df["gw"].values, "Row with NaN target should be dropped"


def test_feature_matrix_discovers_rolling_cols():
    df = _make_history_df(n_players=1, n_gws=8)
    df = add_rolling_features(df, windows=[3, 5], ewm_spans=[3])
    df = add_minutes_features(df)
    df["fixture_difficulty"] = 3.0
    df["opponent_difficulty"] = 2.0
    df["is_double_gw"] = False
    df["n_fixtures"] = 1

    _, feat_cols = build_feature_matrix(df)
    roll_cols = [c for c in feat_cols if "_roll" in c or "_ewm" in c]
    assert len(roll_cols) > 0, "Rolling/EWM columns should be discovered"
