"""
src/data/build_dataset.py
--------------------------
Transforms raw FPL API data into ML-ready feature matrices.

TEMPORAL CORRECTNESS RULES (enforced throughout):
  - Features for GW t may only use match data from GWs < t
  - Rolling windows are computed on strictly past rows (shift before rolling)
  - No target-GW statistics are included in features
  - Train/val split is time-based — never shuffled
  - Upcoming fixture features use only pre-kickoff public data (difficulty, home/away)

OUTPUT SCHEMA (one row per player per gameweek):
  player_id, gw, [rolling features], [fixture features], target_points
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

# Raw history columns we keep from element-summary
HISTORY_COLS = [
    "element",          # player_id
    "round",            # gameweek number
    "kickoff_time",
    "was_home",
    "opponent_team",
    "total_points",
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "value",            # price * 10 at time of match
    "selected",         # ownership count
    "transfers_in",
    "transfers_out",
    "yellow_cards",
    "red_cards",
    "saves",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
]

# Columns that exist from GW ~2024+ (xG data); older seasons may lack them
OPTIONAL_COLS = {
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
}

# Rolling window sizes
ROLL_WINDOWS = [3, 5]

# EWM span (half-life in matches)
EWM_SPANS = [3, 5]

# Columns to apply rolling/ewm to
ROLL_TARGETS = [
    "total_points",
    "minutes",
    "goals_scored",
    "assists",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "clean_sheets",
    "goals_conceded",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
]


# ---------------------------------------------------------------------------
# Step 1: Flatten raw API data → long-form history DataFrame
# ---------------------------------------------------------------------------


def build_history_df(
    elements: list[dict],
    element_summaries: dict[int, dict],
) -> pd.DataFrame:
    """
    Flatten per-player element summaries into a single long-form DataFrame.

    Parameters
    ----------
    elements : list of player dicts from bootstrap['elements']
    element_summaries : dict mapping player_id → element-summary API response

    Returns
    -------
    pd.DataFrame with columns from HISTORY_COLS plus:
        position_id, team_id, player_name, cost (in £m)
    Sorted by (player_id, gw) ascending.
    """
    # Build player metadata lookup
    player_meta = {
        el["id"]: {
            "player_name": f"{el.get('first_name', '')} {el.get('second_name', '')}".strip(),
            "position_id": el["element_type"],   # 1=GKP, 2=DEF, 3=MID, 4=FWD
            "team_id": el["team"],
            "now_cost": el["now_cost"],           # current price * 10
        }
        for el in elements
    }

    rows: list[dict] = []
    missing_players: list[int] = []

    for pid, summary in element_summaries.items():
        history = summary.get("history", [])
        if not history:
            missing_players.append(pid)
            continue

        meta = player_meta.get(pid, {})

        for match in history:
            row: dict[str, Any] = {}

            # Core history fields
            for col in HISTORY_COLS:
                if col in match:
                    row[col] = match[col]
                elif col in OPTIONAL_COLS:
                    row[col] = np.nan
                # else: silently skip unknown fields

            # Rename 'element' → 'player_id', 'round' → 'gw'
            row["player_id"] = match.get("element", pid)
            row["gw"] = match.get("round")

            # Attach metadata
            row.update(meta)

            rows.append(row)

    if missing_players:
        logger.warning(
            "%d players had no match history: %s",
            len(missing_players),
            missing_players[:10],
        )

    if not rows:
        raise ValueError("No history rows found. Check element_summaries input.")

    df = pd.DataFrame(rows)

    # Drop raw 'element' and 'round' columns (renamed above)
    df = df.drop(columns=["element", "round"], errors="ignore")

    # Parse kickoff_time
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True, errors="coerce")

    # Convert numeric cols
    numeric_cols = [
        c for c in df.columns
        if c not in ("player_id", "gw", "player_name", "kickoff_time", "was_home")
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Derive cost in £m
    df["cost"] = df["value"] / 10.0

    df = df.sort_values(["player_id", "gw"]).reset_index(drop=True)

    logger.info(
        "History DataFrame: %d rows, %d players, GWs %d–%d",
        len(df),
        df["player_id"].nunique(),
        df["gw"].min(),
        df["gw"].max(),
    )

    return df


# ---------------------------------------------------------------------------
# Step 2: Build fixture context DataFrame
# ---------------------------------------------------------------------------


def build_fixture_df(
    fixtures: list[dict],
    teams: list[dict],
) -> pd.DataFrame:
    """
    Build a (team_id, gw) → fixture context DataFrame.

    Each team gets one row per gameweek they play in, with:
        team_id, gw, opponent_id, was_home,
        fixture_difficulty, opponent_difficulty

    For double gameweeks: a team may appear twice for the same gw.
    For blank gameweeks: a team will be absent.

    TEMPORAL SAFETY: all fixture difficulty data is pre-kickoff public info.
    """
    # Build team lookup: id → short_name
    team_lookup = {t["id"]: t["short_name"] for t in teams}

    rows: list[dict] = []
    for f in fixtures:
        gw = f.get("event")
        if gw is None:
            continue  # postponed / unscheduled

        team_h = f["team_h"]
        team_a = f["team_a"]

        rows.append({
            "gw": gw,
            "team_id": team_h,
            "opponent_id": team_a,
            "was_home": True,
            "fixture_difficulty": f.get("team_h_difficulty", np.nan),
            "opponent_difficulty": f.get("team_a_difficulty", np.nan),
            "fixture_id": f["id"],
            "finished": f.get("finished", False),
        })
        rows.append({
            "gw": gw,
            "team_id": team_a,
            "opponent_id": team_h,
            "was_home": False,
            "fixture_difficulty": f.get("team_a_difficulty", np.nan),
            "opponent_difficulty": f.get("team_h_difficulty", np.nan),
            "fixture_id": f["id"],
            "finished": f.get("finished", False),
        })

    df = pd.DataFrame(rows)
    df["gw"] = df["gw"].astype(int)

    # For double GWs: aggregate difficulty as mean, flag double gw
    agg = (
        df.groupby(["team_id", "gw"])
        .agg(
            fixture_difficulty=("fixture_difficulty", "mean"),
            opponent_difficulty=("opponent_difficulty", "mean"),
            is_home=("was_home", "any"),          # True if any leg at home
            is_double_gw=("fixture_id", lambda x: len(x) > 1),
            n_fixtures=("fixture_id", "count"),
        )
        .reset_index()
    )

    logger.info(
        "Fixture DataFrame: %d (team, gw) rows, %d double GWs",
        len(agg),
        agg["is_double_gw"].sum(),
    )

    return agg


# ---------------------------------------------------------------------------
# Step 3: Rolling / EWM features — TEMPORALLY SAFE
# ---------------------------------------------------------------------------


def add_rolling_features(
    df: pd.DataFrame,
    windows: list[int] = ROLL_WINDOWS,
    ewm_spans: list[int] = EWM_SPANS,
    roll_cols: list[str] = ROLL_TARGETS,
) -> pd.DataFrame:
    """
    Add rolling mean and EWM features per player.

    CRITICAL: uses shift(1) before every rolling call so that the
    feature at row t uses only rows < t (no look-ahead).

    Parameters
    ----------
    df : sorted by (player_id, gw) ascending — must be pre-sorted
    windows : list of rolling window sizes
    ewm_spans : list of EWM half-life spans
    roll_cols : columns to roll over

    Returns
    -------
    df with new columns added in-place copy.
    """
    df = df.copy()

    # Only roll columns that actually exist in df
    available_roll_cols = [c for c in roll_cols if c in df.columns]
    if len(available_roll_cols) < len(roll_cols):
        missing = set(roll_cols) - set(available_roll_cols)
        logger.debug("Rolling skipped for missing columns: %s", missing)

    def _roll_col(series: pd.Series, w: int) -> pd.Series:
        """Shift-then-roll per player group, preserving original index."""
        return (
            series
            .groupby(df["player_id"])
            .transform(lambda g: g.shift(1).rolling(w, min_periods=1).mean())
        )

    def _ewm_col(series: pd.Series, span: int) -> pd.Series:
        return (
            series
            .groupby(df["player_id"])
            .transform(lambda g: g.shift(1).ewm(span=span, min_periods=1).mean())
        )

    for col in available_roll_cols:
        for w in windows:
            df[f"{col}_roll{w}"] = _roll_col(df[col], w)
        for span in ewm_spans:
            df[f"{col}_ewm{span}"] = _ewm_col(df[col], span)

    logger.debug(
        "Rolling features added: %d new columns",
        len(windows) * len(available_roll_cols) + len(ewm_spans) * len(available_roll_cols),
    )

    return df


def add_minutes_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive minutes-related engineered features.

    - started: played >= 45 minutes
    - minutes_pct: minutes / 90
    - started_roll3/5: consistency of starting
    - sub_risk: proportion of recent games with < 45 min
    """
    df = df.copy()

    df["started"] = (df["minutes"] >= 45).astype(float)
    df["minutes_pct"] = df["minutes"] / 90.0

    def _roll_started(series: pd.Series, w: int) -> pd.Series:
        return series.groupby(df["player_id"]).transform(
            lambda g: g.shift(1).rolling(w, min_periods=1).mean()
        )

    for w in [3, 5]:
        df[f"started_roll{w}"] = _roll_started(df["started"], w)
        df[f"minutes_pct_roll{w}"] = _roll_started(df["minutes_pct"], w)

    df["sub_risk"] = df["started"].groupby(df["player_id"]).transform(
        lambda g: g.shift(1).rolling(5, min_periods=1).apply(lambda x: (x < 1).mean())
    )

    return df


# ---------------------------------------------------------------------------
# Step 4: Merge player history with fixture context
# ---------------------------------------------------------------------------


def merge_fixture_context(
    history_df: pd.DataFrame,
    fixture_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join player history with fixture context on (team_id, gw).

    Players in blank gameweeks won't appear in history_df anyway
    (no match = no history row), so this is safe.

    TEMPORAL SAFETY: fixture_difficulty is pre-kickoff public info.
    """
    merged = history_df.merge(
        fixture_df[["team_id", "gw", "fixture_difficulty", "opponent_difficulty",
                    "is_double_gw", "n_fixtures"]],
        on=["team_id", "gw"],
        how="left",
    )

    n_unmatched = merged["fixture_difficulty"].isna().sum()
    if n_unmatched > 0:
        logger.warning(
            "%d rows missing fixture context after merge (e.g. blank GWs or data gaps)",
            n_unmatched,
        )

    return merged


# ---------------------------------------------------------------------------
# Step 5: Add position encoding and team strength
# ---------------------------------------------------------------------------


def add_position_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode position_id (1=GKP, 2=DEF, 3=MID, 4=FWD)."""
    df = df.copy()
    pos_dummies = pd.get_dummies(
        df["position_id"], prefix="pos", drop_first=False
    ).astype(float)
    # Ensure all 4 positions present even if some missing in data
    for p in [1, 2, 3, 4]:
        col = f"pos_{p}"
        if col not in pos_dummies.columns:
            pos_dummies[col] = 0.0
    df = pd.concat([df, pos_dummies], axis=1)
    return df


def add_team_strength(
    df: pd.DataFrame,
    teams: list[dict],
) -> pd.DataFrame:
    """
    Attach FPL's built-in team strength scores.

    strength_overall_home/away, strength_attack_home/away,
    strength_defence_home/away — these are pre-season public ratings.
    """
    df = df.copy()
    strength_cols = [
        "strength_overall_home",
        "strength_overall_away",
        "strength_attack_home",
        "strength_attack_away",
        "strength_defence_home",
        "strength_defence_away",
    ]
    team_strength = pd.DataFrame([
        {"team_id": t["id"], **{c: t.get(c, np.nan) for c in strength_cols}}
        for t in teams
    ])
    df = df.merge(team_strength, on="team_id", how="left")
    return df


# ---------------------------------------------------------------------------
# Step 6: Assemble final feature matrix
# ---------------------------------------------------------------------------


FEATURE_COLS_BASE = [
    # Rolling features are added dynamically
    # Static/contextual
    "cost",
    "selected",
    "was_home",
    "fixture_difficulty",
    "opponent_difficulty",
    "is_double_gw",
    "n_fixtures",
    "pos_1", "pos_2", "pos_3", "pos_4",
    "strength_overall_home", "strength_overall_away",
    "strength_attack_home", "strength_attack_away",
    "strength_defence_home", "strength_defence_away",
    # Minutes reliability
    "started_roll3", "started_roll5",
    "minutes_pct_roll3", "minutes_pct_roll5",
    "sub_risk",
]

TARGET_COL = "total_points"
ID_COLS = ["player_id", "gw", "player_name", "team_id", "position_id", "kickoff_time"]


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Select feature columns and return (feature_df, feature_col_names).

    Automatically discovers all rolling/EWM columns added by previous steps.
    Drops rows where target is NaN.
    """
    # Discover rolling/EWM columns
    roll_ewm_cols = [
        c for c in df.columns
        if any(c.endswith(f"_roll{w}") for w in ROLL_WINDOWS)
        or any(c.endswith(f"_ewm{s}") for s in EWM_SPANS)
    ]

    all_feature_cols = FEATURE_COLS_BASE + roll_ewm_cols

    # Only keep columns that exist
    feature_cols = [c for c in all_feature_cols if c in df.columns]

    # Keep ID cols + features + target
    keep = list(dict.fromkeys(ID_COLS + feature_cols + [TARGET_COL]))
    keep = [c for c in keep if c in df.columns]

    out = df[keep].copy()

    # Drop rows without a target (shouldn't happen for historical data)
    before = len(out)
    out = out.dropna(subset=[TARGET_COL])
    if len(out) < before:
        logger.warning("Dropped %d rows with missing target", before - len(out))

    logger.info(
        "Feature matrix: %d rows × %d features",
        len(out),
        len(feature_cols),
    )

    return out, feature_cols


# ---------------------------------------------------------------------------
# Step 7: Temporal train/val split
# ---------------------------------------------------------------------------


def temporal_split(
    df: pd.DataFrame,
    val_start_gw: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train (gw < val_start_gw) and val (gw >= val_start_gw).

    NEVER shuffle. This is a strict temporal split.

    Parameters
    ----------
    df : feature matrix with 'gw' column
    val_start_gw : first GW to include in validation set

    Returns
    -------
    (train_df, val_df)
    """
    train = df[df["gw"] < val_start_gw].copy()
    val = df[df["gw"] >= val_start_gw].copy()

    logger.info(
        "Temporal split at GW %d → train: %d rows (GW %d–%d), val: %d rows (GW %d–%d)",
        val_start_gw,
        len(train), train["gw"].min(), train["gw"].max(),
        len(val), val["gw"].min(), val["gw"].max(),
    )

    if len(train) == 0:
        raise ValueError(f"Train set is empty. val_start_gw={val_start_gw} may be too early.")
    if len(val) == 0:
        raise ValueError(f"Val set is empty. val_start_gw={val_start_gw} may be too late.")

    return train, val


# ---------------------------------------------------------------------------
# Step 8: Build upcoming GW prediction features (inference mode)
# ---------------------------------------------------------------------------


def build_prediction_features(
    history_df: pd.DataFrame,
    fixture_df: pd.DataFrame,
    target_gw: int,
    teams: list[dict],
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Build feature rows for the upcoming (target) gameweek.

    For each player who has a fixture in target_gw:
      - Rolling features computed from matches BEFORE target_gw
      - Fixture context from the upcoming fixture record

    TEMPORAL SAFETY: uses only history up to gw < target_gw.

    Returns
    -------
    DataFrame with same feature_cols as training data, no target column.
    """
    # Use only past history to compute rolling features
    past = history_df[history_df["gw"] < target_gw].copy()
    past = add_rolling_features(past)
    past = add_minutes_features(past)

    # Get most recent row per player (latest state of rolling features)
    latest = (
        past.sort_values("gw")
        .groupby("player_id")
        .last()
        .reset_index()
    )

    # Get upcoming fixture context for target_gw
    upcoming_fixtures = fixture_df[fixture_df["gw"] == target_gw].copy()

    # Join: player → team → fixture
    pred = latest.merge(
        upcoming_fixtures[["team_id", "gw", "fixture_difficulty",
                            "opponent_difficulty", "is_double_gw", "n_fixtures"]],
        on="team_id",
        how="inner",  # only players with a fixture this GW
        suffixes=("", "_upcoming"),
    )
    pred["gw"] = target_gw

    # Override fixture columns with upcoming values
    for col in ["fixture_difficulty", "opponent_difficulty", "is_double_gw", "n_fixtures"]:
        if f"{col}_upcoming" in pred.columns:
            pred[col] = pred[f"{col}_upcoming"]
            pred = pred.drop(columns=[f"{col}_upcoming"])

    # Add team strength
    pred = add_team_strength(pred, teams)
    pred = add_position_dummies(pred)

    # Select only the feature columns used during training
    missing_cols = [c for c in feature_cols if c not in pred.columns]
    if missing_cols:
        logger.warning("Filling %d missing feature cols with NaN: %s", len(missing_cols), missing_cols)
        for c in missing_cols:
            pred[c] = np.nan

    id_cols_present = [c for c in ID_COLS if c in pred.columns]
    out = pred[id_cols_present + feature_cols].copy()

    logger.info(
        "Prediction features for GW %d: %d players",
        target_gw,
        len(out),
    )

    return out


# ---------------------------------------------------------------------------
# Top-level pipeline entry point
# ---------------------------------------------------------------------------


def build_dataset(
    bootstrap: dict,
    fixtures: list[dict],
    element_summaries: dict[int, dict],
    val_start_gw: int,
    processed_dir: Path | None = None,
    version: str = "v1",
) -> dict:
    """
    Full dataset build pipeline.

    Parameters
    ----------
    bootstrap : raw bootstrap-static API response
    fixtures : raw fixtures API response
    element_summaries : dict mapping player_id → element-summary response
    val_start_gw : first gameweek to use as validation
    processed_dir : if provided, saves train/val parquet files here
    version : version tag for saved files

    Returns
    -------
    dict with keys:
        train_df, val_df, feature_cols,
        history_df, fixture_df
    """
    elements = bootstrap["elements"]
    teams = bootstrap["teams"]

    logger.info("=== Dataset Build Started ===")

    # 1. Flatten history
    history_df = build_history_df(elements, element_summaries)

    # 2. Fixture context
    fixture_df = build_fixture_df(fixtures, teams)

    # 3. Rolling features
    history_df = add_rolling_features(history_df)
    history_df = add_minutes_features(history_df)

    # 4. Merge fixture context
    history_df = merge_fixture_context(history_df, fixture_df)

    # 5. Position + team strength
    history_df = add_position_dummies(history_df)
    history_df = add_team_strength(history_df, teams)

    # 6. Feature matrix
    feature_matrix, feature_cols = build_feature_matrix(history_df)

    # 7. Temporal split
    train_df, val_df = temporal_split(feature_matrix, val_start_gw)

    # 8. Optionally save
    if processed_dir is not None:
        processed_dir = Path(processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(processed_dir / f"train_{version}.parquet", index=False)
        val_df.to_parquet(processed_dir / f"val_{version}.parquet", index=False)
        logger.info("Saved train/val parquet to %s", processed_dir)

    logger.info("=== Dataset Build Complete ===")

    return {
        "train_df": train_df,
        "val_df": val_df,
        "feature_cols": feature_cols,
        "history_df": history_df,
        "fixture_df": fixture_df,
    }
