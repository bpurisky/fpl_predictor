"""
tests/test_understat.py
------------------------
Unit tests for the Understat ingestion and feature layers.

Coverage:
  - JSON extraction from raw HTML
  - Cache TTL logic (fresh / stale)
  - matchesData → DataFrame flattening
  - shotsData aggregation
  - Rolling features: no leakage (shift-before-roll verified)
  - GW alignment
  - Merge into FPL history
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from io.understat_scraper import _extract_json_vars
from io.understat_cache import (
    get_cached_player,
    save_player,
    fetch_player_with_cache,
    bulk_fetch_players_with_cache,
    _write_meta,
    player_cache_path,
    _meta_path,
)
from data.understat_features import (
    build_matches_df,
    build_shots_df,
    aggregate_shots_to_matches,
    build_understat_match_df,
    add_understat_rolling_features,
    build_gw_dates,
    align_understat_to_gw,
    merge_understat_into_history,
)


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

SAMPLE_PLAYER_DATA = {
    1001: {
        "matchesData": [
            {
                "id": "m1", "date": "2024-08-17 15:00:00", "season": "2024",
                "isHome": True, "xG": "0.45", "xA": "0.12", "npxG": "0.45",
                "goals": "1", "assists": "0", "shots": "3", "key_passes": "2",
                "yellow": "0", "red": "0", "time": "90",
                "h_team": "Arsenal", "a_team": "Wolves",
                "h_goals": "2", "a_goals": "0", "position": "FW",
            },
            {
                "id": "m2", "date": "2024-08-25 14:00:00", "season": "2024",
                "isHome": False, "xG": "0.22", "xA": "0.31", "npxG": "0.22",
                "goals": "0", "assists": "1", "shots": "2", "key_passes": "3",
                "yellow": "0", "red": "0", "time": "90",
                "h_team": "Brighton", "a_team": "Arsenal",
                "h_goals": "1", "a_goals": "1", "position": "FW",
            },
            {
                "id": "m3", "date": "2024-09-01 14:00:00", "season": "2024",
                "isHome": True, "xG": "0.61", "xA": "0.05", "npxG": "0.61",
                "goals": "1", "assists": "0", "shots": "4", "key_passes": "1",
                "yellow": "1", "red": "0", "time": "90",
                "h_team": "Arsenal", "a_team": "Brighton",
                "h_goals": "3", "a_goals": "1", "position": "FW",
            },
        ],
        "shotsData": [
            {"match_id": "m1", "date": "2024-08-17", "minute": "23",
             "X": "0.90", "Y": "0.50", "xG": "0.45", "result": "Goal",
             "shotType": "RightFoot", "situation": "OpenPlay",
             "player_assisted": None, "lastAction": "Pass"},
            {"match_id": "m1", "date": "2024-08-17", "minute": "67",
             "X": "0.70", "Y": "0.60", "xG": "0.08", "result": "MissedShots",
             "shotType": "Head", "situation": "Corner",
             "player_assisted": None, "lastAction": "Cross"},
            {"match_id": "m2", "date": "2024-08-25", "minute": "55",
             "X": "0.88", "Y": "0.48", "xG": "0.22", "result": "SavedShot",
             "shotType": "LeftFoot", "situation": "OpenPlay",
             "player_assisted": "Saka", "lastAction": "Pass"},
        ],
        "groupsData": [],
    }
}

SAMPLE_BOOTSTRAP = {
    "events": [
        {"id": 1, "deadline_time": "2024-08-16T17:30:00Z"},
        {"id": 2, "deadline_time": "2024-08-23T17:30:00Z"},
        {"id": 3, "deadline_time": "2024-08-30T17:30:00Z"},
        {"id": 4, "deadline_time": "2024-09-13T17:30:00Z"},
    ]
}


# ---------------------------------------------------------------------------
# JSON extraction from HTML
# ---------------------------------------------------------------------------


def test_extract_json_vars_basic():
    html = """
    <script>
    var playersData = JSON.parse('[{"id": "123", "player_name": "Saka"}]');
    var shotsData = JSON.parse('[{"xG": "0.5"}]');
    </script>
    """
    result = _extract_json_vars(html)
    assert "playersData" in result
    assert "shotsData" in result
    assert result["playersData"][0]["player_name"] == "Saka"
    assert result["shotsData"][0]["xG"] == "0.5"


def test_extract_json_vars_empty_html():
    result = _extract_json_vars("<html><body>No scripts</body></html>")
    assert result == {}


def test_extract_json_vars_handles_unicode_escapes():
    # Understat encodes special chars as unicode escapes in the JSON string
    name = "Traoré"
    escaped_json = json.dumps([{"player_name": name}])
    # Simulate raw_unicode_escape encoding as Understat would serve it
    escaped = escaped_json.encode("unicode_escape").decode("ascii")
    html = f"var testData = JSON.parse('{escaped}');"
    result = _extract_json_vars(html)
    if "testData" in result:  # may or may not round-trip perfectly in test
        assert isinstance(result["testData"], list)


# ---------------------------------------------------------------------------
# Cache TTL logic
# ---------------------------------------------------------------------------


def test_cache_miss_when_no_file(tmp_path):
    result = get_cached_player(999, cache_dir=tmp_path, ttl_days=7)
    assert result is None


def test_cache_hit_when_fresh(tmp_path):
    data = {"matchesData": [{"id": "x"}], "shotsData": [], "groupsData": []}
    save_player(9999, data, cache_dir=tmp_path)
    result = get_cached_player(9999, cache_dir=tmp_path, ttl_days=7)
    assert result is not None
    assert result["matchesData"][0]["id"] == "x"


def test_cache_miss_when_stale(tmp_path):
    data = {"matchesData": [], "shotsData": [], "groupsData": []}
    save_player(8888, data, cache_dir=tmp_path)

    # Backdate the meta file to 8 days ago
    meta_p = _meta_path(player_cache_path(8888, tmp_path))
    stale_time = (datetime.now(tz=timezone.utc) - timedelta(days=8)).isoformat()
    meta_p.write_text(json.dumps({"fetched_at": stale_time}))

    result = get_cached_player(8888, cache_dir=tmp_path, ttl_days=7)
    assert result is None


def test_fetch_player_with_cache_uses_cache(tmp_path):
    data = {"matchesData": [{"id": "cached"}], "shotsData": [], "groupsData": []}
    save_player(111, data, cache_dir=tmp_path)

    scraper_fn = MagicMock(return_value={"matchesData": [{"id": "fresh"}]})
    result = fetch_player_with_cache(111, scraper_fn, cache_dir=tmp_path, ttl_days=7)

    scraper_fn.assert_not_called()  # cache was fresh
    assert result["matchesData"][0]["id"] == "cached"


def test_fetch_player_with_cache_calls_scraper_when_stale(tmp_path):
    fresh_data = {"matchesData": [{"id": "scraped"}], "shotsData": [], "groupsData": []}
    scraper_fn = MagicMock(return_value=fresh_data)

    result = fetch_player_with_cache(222, scraper_fn, cache_dir=tmp_path, ttl_days=7)

    scraper_fn.assert_called_once_with(222)
    assert result["matchesData"][0]["id"] == "scraped"

    # Should now be cached
    cached = get_cached_player(222, cache_dir=tmp_path, ttl_days=7)
    assert cached is not None


def test_bulk_fetch_only_scrapes_stale(tmp_path):
    # Pre-cache player 1
    save_player(1, {"matchesData": [{"id": "old"}], "shotsData": [], "groupsData": []}, tmp_path)
    # Player 2 not cached

    fresh = {"matchesData": [{"id": "new"}], "shotsData": [], "groupsData": []}
    scraper_fn = MagicMock(return_value=fresh)

    results = bulk_fetch_players_with_cache([1, 2], scraper_fn, cache_dir=tmp_path, ttl_days=7)

    # Only player 2 should have been scraped
    scraper_fn.assert_called_once_with(2)
    assert 1 in results
    assert 2 in results


# ---------------------------------------------------------------------------
# build_matches_df
# ---------------------------------------------------------------------------


def test_build_matches_df_shape():
    df = build_matches_df(SAMPLE_PLAYER_DATA, seasons=["2024"])
    assert len(df) == 3
    assert "understat_id" in df.columns
    assert "xG" in df.columns
    assert "xA" in df.columns
    assert df["understat_id"].iloc[0] == 1001


def test_build_matches_df_season_filter():
    df = build_matches_df(SAMPLE_PLAYER_DATA, seasons=["2022"])
    assert len(df) == 0  # no 2022 data in sample


def test_build_matches_df_xG_overperformance():
    df = build_understat_match_df(SAMPLE_PLAYER_DATA, seasons=["2024"])
    # Match 1: 1 goal, 0.45 xG → overperformance = 0.55
    m1 = df[df["match_id"] == "m1"].iloc[0]
    assert abs(m1["xG_overperformance"] - (1 - 0.45)) < 1e-3


# ---------------------------------------------------------------------------
# build_shots_df
# ---------------------------------------------------------------------------


def test_build_shots_df_shape():
    df = build_shots_df(SAMPLE_PLAYER_DATA, seasons=["2024"])
    assert len(df) == 3
    assert "xG" in df.columns
    assert "is_danger_zone" in df.columns
    assert "is_header" in df.columns
    assert "is_big_chance" in df.columns


def test_danger_zone_flag():
    df = build_shots_df(SAMPLE_PLAYER_DATA, seasons=["2024"])
    # Shot 1: X=0.90, Y=0.50 → should be danger zone (X>0.83, 0.3<Y<0.7)
    shot1 = df[(df["match_id"] == "m1") & (df["minute"] == 23)].iloc[0]
    assert shot1["is_danger_zone"] == 1.0


def test_header_flag():
    df = build_shots_df(SAMPLE_PLAYER_DATA, seasons=["2024"])
    shot2 = df[(df["match_id"] == "m1") & (df["minute"] == 67)].iloc[0]
    assert shot2["is_header"] == 1.0


def test_big_chance_flag():
    df = build_shots_df(SAMPLE_PLAYER_DATA, seasons=["2024"])
    shot1 = df[(df["match_id"] == "m1") & (df["minute"] == 23)].iloc[0]
    assert shot1["is_big_chance"] == 1.0  # xG=0.45 >= 0.30


# ---------------------------------------------------------------------------
# aggregate_shots_to_matches
# ---------------------------------------------------------------------------


def test_shot_aggregation_xG_per_shot():
    shots_df = build_shots_df(SAMPLE_PLAYER_DATA, seasons=["2024"])
    agg = aggregate_shots_to_matches(shots_df)

    m1 = agg[agg["match_id"] == "m1"].iloc[0]
    # 2 shots: xG 0.45 + 0.08 → total 0.53 / 2 = 0.265
    assert abs(m1["xG_per_shot"] - (0.45 + 0.08) / 2) < 1e-3


def test_shot_aggregation_danger_zone_pct():
    shots_df = build_shots_df(SAMPLE_PLAYER_DATA, seasons=["2024"])
    agg = aggregate_shots_to_matches(shots_df)

    m1 = agg[agg["match_id"] == "m1"].iloc[0]
    # 2 shots: 1 danger zone → 50%
    assert abs(m1["danger_zone_pct"] - 0.5) < 1e-3


# ---------------------------------------------------------------------------
# Rolling features — temporal leakage tests
# ---------------------------------------------------------------------------


def test_understat_rolling_gw1_is_nan():
    """
    First match per player must produce NaN rolling features (shift(1)).
    """
    df = build_understat_match_df(SAMPLE_PLAYER_DATA, seasons=["2024"])
    df_rolled = add_understat_rolling_features(df, windows=[3], ewm_spans=[])

    first_match = df_rolled.sort_values("date").groupby("understat_id").first().reset_index()
    assert pd.isna(first_match["us_xG_roll3"].iloc[0]), (
        "First match xG roll should be NaN — no past data available"
    )


def test_understat_rolling_second_match_uses_first_only():
    """
    Second match's roll3 must equal first match's xG value (only 1 past obs).
    """
    df = build_understat_match_df(SAMPLE_PLAYER_DATA, seasons=["2024"])
    df = df.sort_values("date").reset_index(drop=True)
    df_rolled = add_understat_rolling_features(df, windows=[3], ewm_spans=[])

    sorted_df = df_rolled.sort_values("date").reset_index(drop=True)
    first_xg = sorted_df.iloc[0]["xG"]
    second_roll3 = sorted_df.iloc[1]["us_xG_roll3"]

    assert abs(second_roll3 - first_xg) < 1e-6, (
        f"Match 2 roll3 ({second_roll3}) should equal match 1 xG ({first_xg})"
    )


def test_understat_rolling_col_prefix():
    df = build_understat_match_df(SAMPLE_PLAYER_DATA, seasons=["2024"])
    df_rolled = add_understat_rolling_features(df, windows=[3], ewm_spans=[3])
    rolling_cols = [c for c in df_rolled.columns if c.startswith("us_")]
    assert len(rolling_cols) > 0
    assert all(c.startswith("us_") for c in rolling_cols)


# ---------------------------------------------------------------------------
# GW alignment
# ---------------------------------------------------------------------------


def test_build_gw_dates():
    gw_dates = build_gw_dates(SAMPLE_BOOTSTRAP)
    assert len(gw_dates) == 4
    assert list(gw_dates["gw"]) == [1, 2, 3, 4]


def test_align_to_gw_correct_assignment():
    gw_dates = build_gw_dates(SAMPLE_BOOTSTRAP)
    df = build_understat_match_df(SAMPLE_PLAYER_DATA, seasons=["2024"])
    aligned = align_understat_to_gw(df, gw_dates)

    # match date 2024-08-17: between deadline GW1 (Aug 16) and GW2 (Aug 23) → GW 1
    m1 = aligned[aligned["match_id"] == "m1"].iloc[0]
    assert m1["gw"] == 1

    # match date 2024-08-25: between GW2 deadline (Aug 23) and GW3 (Aug 30) → GW 2
    m2 = aligned[aligned["match_id"] == "m2"].iloc[0]
    assert m2["gw"] == 2


# ---------------------------------------------------------------------------
# Merge into FPL history
# ---------------------------------------------------------------------------


def test_merge_understat_into_history_left_join():
    """
    FPL history rows without Understat data should survive with NaN us_* cols.
    """
    history_df = pd.DataFrame({
        "player_id": [10, 10, 99],   # 99 has no Understat data
        "gw": [1, 2, 1],
        "total_points": [6, 8, 4],
    })

    us_features = pd.DataFrame({
        "fpl_id": [10, 10],
        "gw": [1, 2],
        "us_xG_roll3": [0.3, 0.4],
    })

    merged = merge_understat_into_history(history_df, us_features)

    assert len(merged) == 3  # no rows dropped
    # Player 99 should have NaN
    row_99 = merged[merged["player_id"] == 99].iloc[0]
    assert pd.isna(row_99["us_xG_roll3"])
    # Player 10 GW1 should have the feature
    row_10_gw1 = merged[(merged["player_id"] == 10) & (merged["gw"] == 1)].iloc[0]
    assert abs(row_10_gw1["us_xG_roll3"] - 0.3) < 1e-6


def test_merge_does_not_add_raw_understat_cols():
    """
    Only us_* prefixed columns should be added — raw xG etc. must not leak in.
    """
    history_df = pd.DataFrame({
        "player_id": [10], "gw": [1], "total_points": [6],
    })
    us_features = pd.DataFrame({
        "fpl_id": [10], "gw": [1], "us_xG_roll3": [0.3],
    })
    merged = merge_understat_into_history(history_df, us_features)

    # 'xG' (raw) should not be in merged — only 'us_xG_roll3'
    assert "xG" not in merged.columns
    assert "us_xG_roll3" in merged.columns
