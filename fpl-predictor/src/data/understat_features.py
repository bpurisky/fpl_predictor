"""
src/data/understat_features.py
-------------------------------
Transforms raw Understat scrape data into match-level feature rows
that can be merged with the FPL history DataFrame in build_dataset.py.

WHAT THIS MODULE PRODUCES (per player per match):
  From matchesData:
    - xG, xA, shots, key_passes per match
    - npxG (non-penalty xG)
    - xG overperformance (goals - xG) — rolling trend

  From shotsData (aggregated to match level):
    - xG per shot (shot quality metric)
    - shot_zone_danger: proportion of shots from high-danger zones
    - headed_shot_pct: proportion of shots that were headers
    - open_play_pct: proportion from open play vs set pieces
    - xA per key pass (pass quality)
    - big_chance_count (shots with xG > 0.3)

TEMPORAL SAFETY:
  All features are built from historical match data only.
  Rolling features use shift(1) before windowing — same pattern
  as build_dataset.py. Target match data is never included in features.

MERGE KEY:
  (fpl_id, gw) via the player mapping and date-to-GW alignment.
  The merge happens in build_dataset.py — this module only produces
  the Understat-side feature DataFrame.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Understat season start year → FPL season label
UNDERSTAT_SEASON_TO_FPL = {
    "2022": "2022-23",
    "2023": "2023-24",
    "2024": "2024-25",
}

# High-danger zone: shots from central area within 18 yards
# Understat coordinates: X (0–1, goal=1), Y (0–1, centre=0.5)
# High danger ≈ X > 0.83 and 0.3 < Y < 0.7
DANGER_X_THRESHOLD = 0.83
DANGER_Y_LOW = 0.30
DANGER_Y_HIGH = 0.70

# Big chance threshold
BIG_CHANCE_XG = 0.30

# Rolling windows for Understat features
ROLL_WINDOWS = [3, 5]
EWM_SPANS = [3, 5]


# ---------------------------------------------------------------------------
# Step 1: Flatten matchesData → match-level DataFrame
# ---------------------------------------------------------------------------


def build_matches_df(
    player_data: dict[int, dict],
    seasons: list[str] | None = None,
) -> pd.DataFrame:
    """
    Flatten Understat matchesData for all players into a long-form DataFrame.

    Parameters
    ----------
    player_data : dict mapping understat_id → {matchesData, shotsData, ...}
    seasons : list of Understat season years to include e.g. ['2022','2023','2024']
              If None, includes all seasons found.

    Returns
    -------
    DataFrame with columns:
        understat_id, date, season, is_home,
        xG, xA, shots, key_passes, npxG, goals, assists,
        h_team, a_team, h_goals, a_goals, match_id
    Sorted by (understat_id, date).
    """
    season_filter = set(seasons) if seasons else None
    rows: list[dict] = []

    for uid, data in player_data.items():
        matches = data.get("matchesData", [])
        if not matches:
            continue

        for m in matches:
            season = str(m.get("season", ""))
            if season_filter and season not in season_filter:
                continue

            rows.append({
                "understat_id": int(uid),
                "match_id": m.get("id"),
                "date": m.get("date"),
                "season": season,
                "is_home": m.get("isHome", False),
                "xG": _float(m.get("xG")),
                "xA": _float(m.get("xA")),
                "npxG": _float(m.get("npxG", m.get("xG"))),  # fallback to xG
                "goals": _int(m.get("goals")),
                "assists": _int(m.get("assists")),
                "shots": _int(m.get("shots")),
                "key_passes": _int(m.get("key_passes")),
                "yellow_cards": _int(m.get("yellow")),
                "red_cards": _int(m.get("red")),
                "minutes": _int(m.get("time")),
                "h_team": m.get("h_team", ""),
                "a_team": m.get("a_team", ""),
                "h_goals": _int(m.get("h_goals")),
                "a_goals": _int(m.get("a_goals")),
                "position": m.get("position", ""),
            })

    if not rows:
        raise ValueError("No match rows found. Check player_data and seasons filter.")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.sort_values(["understat_id", "date"]).reset_index(drop=True)

    logger.info(
        "Understat matches: %d rows, %d players, seasons %s",
        len(df),
        df["understat_id"].nunique(),
        sorted(df["season"].unique()),
    )
    return df


# ---------------------------------------------------------------------------
# Step 2: Flatten shotsData → shot-level DataFrame
# ---------------------------------------------------------------------------


def build_shots_df(
    player_data: dict[int, dict],
    seasons: list[str] | None = None,
) -> pd.DataFrame:
    """
    Flatten Understat shotsData for all players.

    Returns
    -------
    DataFrame with columns:
        understat_id, match_id, season, date, minute,
        X, Y, xG, result, type (foot/head), situation,
        player_assisted, lastAction
    Sorted by (understat_id, date, minute).
    """
    season_filter = set(seasons) if seasons else None
    rows: list[dict] = []

    for uid, data in player_data.items():
        shots = data.get("shotsData", [])
        if not shots:
            continue

        # Build match_id → season lookup (shots often lack their own season field)
        match_season_map: dict[str, str] = {
            str(m.get("id", "")): str(m.get("season", ""))
            for m in data.get("matchesData", [])
        }

        for s in shots:
            season = str(s.get("season", "")) or match_season_map.get(
                str(s.get("match_id", "")), ""
            )
            if season_filter and season not in season_filter:
                continue

            rows.append({
                "understat_id": int(uid),
                "match_id": s.get("match_id"),
                "season": season,
                "date": s.get("date"),
                "minute": _int(s.get("minute")),
                "X": _float(s.get("X")),            # pitch x-coord (0–1)
                "Y": _float(s.get("Y")),            # pitch y-coord (0–1)
                "xG": _float(s.get("xG")),
                "result": s.get("result", ""),      # Goal/SavedShot/MissedShots/BlockedShot
                "shot_type": s.get("shotType", s.get("type", "")),  # LeftFoot/RightFoot/Head
                "situation": s.get("situation", ""),  # OpenPlay/SetPiece/Corner/DirectFreekick
                "player_assisted": s.get("player_assisted"),
                "last_action": s.get("lastAction", ""),
            })

    if not rows:
        logger.warning("No shot rows found.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.sort_values(["understat_id", "date", "minute"]).reset_index(drop=True)

    # Derived shot features
    df["is_danger_zone"] = (
        (df["X"] > DANGER_X_THRESHOLD) &
        (df["Y"] > DANGER_Y_LOW) &
        (df["Y"] < DANGER_Y_HIGH)
    ).astype(float)
    df["is_header"] = (df["shot_type"].str.lower() == "head").astype(float)
    df["is_open_play"] = (df["situation"].str.lower() == "openplay").astype(float)
    df["is_big_chance"] = (df["xG"] >= BIG_CHANCE_XG).astype(float)

    logger.info(
        "Understat shots: %d rows, %d players",
        len(df),
        df["understat_id"].nunique(),
    )
    return df


# ---------------------------------------------------------------------------
# Step 3: Aggregate shots → match-level shot quality features
# ---------------------------------------------------------------------------


def aggregate_shots_to_matches(shots_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate shot-level data to (understat_id, match_id) level.

    Returns per-match shot quality features:
        xG_per_shot    : average xG per attempt (shot quality)
        danger_zone_pct: % shots from high-danger zones
        header_pct     : % shots that were headers
        open_play_pct  : % shots from open play
        big_chances    : count of shots xG >= 0.30
        total_shots    : total shots in match (from shot records)
    """
    if shots_df.empty:
        return pd.DataFrame()

    agg = (
        shots_df
        .groupby(["understat_id", "match_id"], as_index=False)
        .agg(
            shot_xG_total=("xG", "sum"),
            shot_count=("xG", "count"),
            danger_zone_shots=("is_danger_zone", "sum"),
            header_shots=("is_header", "sum"),
            open_play_shots=("is_open_play", "sum"),
            big_chances=("is_big_chance", "sum"),
        )
    )

    # Derived ratios (safe divide)
    def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
        return a.where(b > 0, np.nan) / b.where(b > 0, np.nan)

    agg["xG_per_shot"] = _safe_div(agg["shot_xG_total"], agg["shot_count"])
    agg["danger_zone_pct"] = _safe_div(agg["danger_zone_shots"], agg["shot_count"])
    agg["header_pct"] = _safe_div(agg["header_shots"], agg["shot_count"])
    agg["open_play_pct"] = _safe_div(agg["open_play_shots"], agg["shot_count"])

    return agg[[
        "understat_id", "match_id",
        "xG_per_shot", "danger_zone_pct", "header_pct",
        "open_play_pct", "big_chances",
    ]]


# ---------------------------------------------------------------------------
# Step 4: Combine match + shot aggregates
# ---------------------------------------------------------------------------


def build_understat_match_df(
    player_data: dict[int, dict],
    seasons: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build the full Understat per-match feature DataFrame.

    Combines matchesData (xG, xA, shots, key_passes, npxG)
    with shot-level aggregates (xG per shot, danger zone %, etc.).

    Returns
    -------
    DataFrame sorted by (understat_id, date) with columns:
        understat_id, match_id, date, season, is_home,
        xG, xA, npxG, goals, assists, shots, key_passes,
        xG_per_shot, danger_zone_pct, header_pct,
        open_play_pct, big_chances,
        xG_overperformance (goals - xG)
    """
    matches_df = build_matches_df(player_data, seasons)
    shots_df = build_shots_df(player_data, seasons)

    if not shots_df.empty:
        shot_agg = aggregate_shots_to_matches(shots_df)
        merged = matches_df.merge(
            shot_agg, on=["understat_id", "match_id"], how="left"
        )
    else:
        logger.warning("No shot data available — shot quality features will be NaN")
        merged = matches_df.copy()
        for col in ["xG_per_shot", "danger_zone_pct", "header_pct",
                    "open_play_pct", "big_chances"]:
            merged[col] = np.nan

    # Derived
    merged["xG_overperformance"] = merged["goals"] - merged["xG"]

    logger.info("Understat match+shot combined: %d rows", len(merged))
    return merged


# ---------------------------------------------------------------------------
# Step 5: Add rolling features (same shift(1) pattern as build_dataset.py)
# ---------------------------------------------------------------------------

UNDERSTAT_ROLL_COLS = [
    "xG", "xA", "npxG", "shots", "key_passes",
    "xG_per_shot", "danger_zone_pct", "big_chances",
    "open_play_pct", "xG_overperformance",
]


def add_understat_rolling_features(
    df: pd.DataFrame,
    windows: list[int] = ROLL_WINDOWS,
    ewm_spans: list[int] = EWM_SPANS,
    roll_cols: list[str] = UNDERSTAT_ROLL_COLS,
) -> pd.DataFrame:
    """
    Add rolling and EWM features to Understat match DataFrame.

    Uses shift(1) before every rolling call — identical pattern to
    build_dataset.add_rolling_features(). No target-GW leakage.

    df must be sorted by (understat_id, date) before calling.
    """
    df = df.copy()
    available = [c for c in roll_cols if c in df.columns]

    def _roll(series: pd.Series, w: int) -> pd.Series:
        return series.groupby(df["understat_id"]).transform(
            lambda g: g.shift(1).rolling(w, min_periods=1).mean()
        )

    def _ewm(series: pd.Series, span: int) -> pd.Series:
        return series.groupby(df["understat_id"]).transform(
            lambda g: g.shift(1).ewm(span=span, min_periods=1).mean()
        )

    for col in available:
        for w in windows:
            df[f"us_{col}_roll{w}"] = _roll(df[col], w)
        for span in ewm_spans:
            df[f"us_{col}_ewm{span}"] = _ewm(df[col], span)

    return df


# ---------------------------------------------------------------------------
# Step 6: Load player mapping and produce merge-ready DataFrame
# ---------------------------------------------------------------------------


def load_player_mapping(
    mapping_path: Path | str = "data/understat_player_mapping.csv",
) -> pd.DataFrame:
    """
    Load the manually verified FPL ↔ Understat player ID mapping.

    Raises FileNotFoundError if mapping doesn't exist yet.
    Raises ValueError if mapping has unverified rows.
    """
    path = Path(mapping_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Player mapping not found: {path}\n"
            "Run: python scripts/generate_player_mapping.py\n"
            "Then review and save as data/understat_player_mapping.csv"
        )

    df = pd.read_csv(path)

    required_cols = {"fpl_id", "understat_id", "verified"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Mapping CSV missing columns: {missing}")

    # Drop players with no Understat match (understat_id is blank/NaN)
    valid = df[df["understat_id"].notna() & (df["understat_id"] != "")]
    valid = valid.copy()
    valid["understat_id"] = valid["understat_id"].astype(int)
    valid["fpl_id"] = valid["fpl_id"].astype(int)

    unverified = (~valid["verified"].astype(bool)).sum()
    if unverified > 0:
        logger.warning(
            "%d player mappings are unverified. Review data/understat_player_mapping.csv",
            unverified,
        )

    no_match = len(df) - len(valid)
    logger.info(
        "Player mapping: %d verified, %d no Understat match, %d unverified",
        valid["verified"].astype(bool).sum(),
        no_match,
        unverified,
    )

    return valid[["fpl_id", "understat_id"]].reset_index(drop=True)


def align_understat_to_gw(
    understat_df: pd.DataFrame,
    gw_dates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a 'gw' column to understat_df by matching match date to FPL gameweek.

    FPL gameweeks don't have clean date boundaries (some GWs span Tue–Mon).
    We assign each Understat match to the GW whose deadline it falls within.

    Parameters
    ----------
    understat_df : match-level DataFrame with 'date' column (UTC)
    gw_dates : DataFrame with columns [gw, deadline_time (UTC)]
               Built from bootstrap['events'] — each event has a deadline_time

    Returns
    -------
    understat_df with 'gw' column added.
    Rows with no matching GW get gw=NaN and are dropped with a warning.
    """
    gw_dates = gw_dates.sort_values("deadline_time").reset_index(drop=True)

    def _find_gw(match_date: pd.Timestamp) -> int | None:
        if pd.isna(match_date):
            return None
        # GW n contains matches between deadline[n-1] and deadline[n]
        idx = gw_dates["deadline_time"].searchsorted(match_date, side="right") - 1
        if idx < 0 or idx >= len(gw_dates):
            return None
        return int(gw_dates.iloc[idx]["gw"])

    understat_df = understat_df.copy()
    understat_df["gw"] = understat_df["date"].apply(_find_gw)

    unmatched = understat_df["gw"].isna().sum()
    if unmatched > 0:
        logger.warning(
            "%d Understat rows could not be aligned to a GW (pre/post season dates)",
            unmatched,
        )
        understat_df = understat_df.dropna(subset=["gw"])

    understat_df["gw"] = understat_df["gw"].astype(int)
    return understat_df


def build_gw_dates(bootstrap: dict) -> pd.DataFrame:
    """
    Extract GW deadline times from bootstrap['events'].

    Returns DataFrame with [gw, deadline_time].
    """
    rows = []
    for event in bootstrap["events"]:
        if event.get("deadline_time"):
            rows.append({
                "gw": event["id"],
                "deadline_time": pd.to_datetime(event["deadline_time"], utc=True),
            })
    return pd.DataFrame(rows).sort_values("gw").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Top-level: produce merge-ready Understat feature DataFrame
# ---------------------------------------------------------------------------


def build_understat_features(
    player_data: dict[int, dict],
    mapping_path: Path | str,
    bootstrap: dict,
    seasons: list[str] | None = None,
    windows: list[int] = ROLL_WINDOWS,
    ewm_spans: list[int] = EWM_SPANS,
) -> pd.DataFrame:
    """
    Full Understat feature pipeline.

    Parameters
    ----------
    player_data : dict understat_id → raw scrape data
    mapping_path : path to verified FPL ↔ Understat mapping CSV
    bootstrap : raw FPL bootstrap-static response (for GW dates)
    seasons : Understat season years to include e.g. ['2022','2023','2024']
    windows : rolling window sizes
    ewm_spans : EWM span values

    Returns
    -------
    DataFrame keyed on (fpl_id, gw) with Understat rolling features.
    Ready to left-join onto the FPL history DataFrame from build_dataset.py.

    All column names are prefixed with 'us_' to avoid collisions.
    """
    # 1. Build combined match + shot DataFrame
    match_df = build_understat_match_df(player_data, seasons)

    # 2. Add rolling features (temporally safe)
    match_df = add_understat_rolling_features(match_df, windows, ewm_spans)

    # 3. Align to FPL gameweeks
    gw_dates = build_gw_dates(bootstrap)
    match_df = align_understat_to_gw(match_df, gw_dates)

    # 4. Join FPL IDs via mapping
    mapping = load_player_mapping(mapping_path)
    match_df = match_df.merge(mapping, on="understat_id", how="inner")

    # 5. Select only rolling/derived features for the merge
    #    (raw match stats are excluded — they'd leak if merged at the target GW)
    rolling_cols = [
        c for c in match_df.columns
        if c.startswith("us_")
    ]
    id_cols = ["fpl_id", "understat_id", "gw", "season", "date"]
    id_cols = [c for c in id_cols if c in match_df.columns]

    out = match_df[id_cols + rolling_cols].copy()

    # 6. For players with double GWs in understat: keep latest match row per (fpl_id, gw)
    out = out.sort_values("date").drop_duplicates(
        subset=["fpl_id", "gw"], keep="last"
    )

    logger.info(
        "Understat features ready: %d rows, %d players, %d feature cols",
        len(out),
        out["fpl_id"].nunique(),
        len(rolling_cols),
    )

    return out.reset_index(drop=True)


def merge_understat_into_history(
    history_df: pd.DataFrame,
    understat_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join Understat features onto the FPL history DataFrame.

    Join key: (fpl_id=player_id, gw).

    Players without Understat data (e.g. unmapped, or pre-2022) will
    have NaN for all us_* columns. GBM handles NaN natively; Transformer
    will need masking.
    """
    # Rename fpl_id → player_id for join
    us = understat_features.rename(columns={"fpl_id": "player_id"})

    # Only bring rolling feature cols (prefix us_) plus the join keys
    us_cols = ["player_id", "gw"] + [c for c in us.columns if c.startswith("us_")]
    us = us[us_cols]

    merged = history_df.merge(us, on=["player_id", "gw"], how="left")

    us_feature_cols = [c for c in merged.columns if c.startswith("us_")]
    coverage = merged[us_feature_cols[0]].notna().mean() if us_feature_cols else 0
    logger.info(
        "Understat merge: coverage %.1f%% of history rows have us_* features",
        coverage * 100,
    )

    return merged


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _float(val: Any) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


def _int(val: Any) -> int:
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return 0
