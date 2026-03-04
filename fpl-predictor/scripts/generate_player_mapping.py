"""
scripts/generate_player_mapping.py
------------------------------------
One-time script to generate data/understat_player_mapping.csv.

This script:
  1. Loads FPL player list from bootstrap cache
  2. Loads Understat player lists for configured seasons
  3. Fuzzy-matches names to produce candidate pairs
  4. Writes a CSV with all candidates for manual review

After running, you MUST manually review the CSV and:
  - Fix incorrect matches (set understat_id to blank or correct ID)
  - Mark ambiguous players (e.g. two players called "Andreas")
  - Add players the fuzzy matcher missed entirely

The reviewed CSV is then used as the ground-truth mapping.

USAGE:
    python scripts/generate_player_mapping.py
    # → writes data/understat_player_mapping_candidates.csv
    # → review and save as data/understat_player_mapping.csv

MAPPING CSV SCHEMA:
    fpl_id          : int  (FPL element ID)
    fpl_name        : str  (FPL display name)
    fpl_team        : str  (FPL team name)
    understat_id    : int  (Understat player ID, blank if unknown)
    understat_name  : str  (Understat display name)
    understat_team  : str  (Understat team name)
    match_score     : float (fuzzy match score 0–100, for review aid)
    verified        : bool (set True after manual review)
    notes           : str  (e.g. "name change", "dual nationality name")
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
import pandas as pd
import rapidfuzz


_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src" / "io"))
sys.path.insert(0, str(_ROOT / "src" / "data"))
sys.path.insert(0, str(_ROOT / "src"))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
UNDERSTAT_DIR = RAW_DIR / "understat"
MAPPING_OUTPUT = Path("data/understat_player_mapping_candidates.csv")
BOOTSTRAP_CACHE = RAW_DIR / "bootstrap_latest.json"

# Seasons to pull Understat players from
TARGET_SEASONS = ["2022", "2023", "2024"]


def load_fpl_players() -> pd.DataFrame:
    """Load FPL player list from bootstrap cache."""
    if not BOOTSTRAP_CACHE.exists():
        raise FileNotFoundError(
            "Bootstrap cache not found. Run: python scripts/ingest.py --bootstrap-only"
        )
    bootstrap = json.loads(BOOTSTRAP_CACHE.read_text(encoding="utf-8"))

    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}

    rows = []
    for el in bootstrap["elements"]:
        rows.append({
            "fpl_id": el["id"],
            "fpl_name": f"{el.get('first_name', '')} {el.get('second_name', '')}".strip(),
            "fpl_team": teams.get(el["team"], "Unknown"),
            "position_id": el["element_type"],
        })
    return pd.DataFrame(rows)


from understatapi import UnderstatClient

def load_understat_players(seasons: list[str]) -> pd.DataFrame:
    from understat_scraper import scrape_league_players
    from understat_cache import fetch_league_with_cache

    all_players = []
    for season_year in seasons:
        logger.info("Loading Understat season %s...", season_year)
        players = fetch_league_with_cache(
            season_year,
            scraper_fn=scrape_league_players,
            cache_dir=UNDERSTAT_DIR,
        )
        for p in players:
            all_players.append({
                "understat_id": int(p["id"]),
                "understat_name": p.get("player_name", ""),
                "understat_team": p.get("team_title", ""),
                "season": season_year,
            })

    df = pd.DataFrame(all_players)
    df = df.sort_values("season", ascending=False).drop_duplicates("understat_id")
    logger.info("Understat: %d unique players across %s", len(df), seasons)
    return df


def fuzzy_match(
    fpl_df: pd.DataFrame,
    understat_df: pd.DataFrame,
    score_cutoff: float = 70.0,
) -> pd.DataFrame:
    """
    Produce candidate matches using rapidfuzz token_sort_ratio.

    For each FPL player, finds the best-matching Understat name.
    Returns a DataFrame of all candidates above score_cutoff.

    NOTE: This output is for HUMAN REVIEW only. Do not use programmatically.
    """
    try:
        from rapidfuzz import process, fuzz
    except ImportError:
        logger.error(
            "rapidfuzz not installed. Run: pip install rapidfuzz"
        )
        raise

    understat_names = understat_df["understat_name"].tolist()
    understat_index = understat_df.reset_index(drop=True)

    rows = []
    for _, fpl_row in fpl_df.iterrows():
        result = process.extractOne(
            fpl_row["fpl_name"],
            understat_names,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=score_cutoff,
        )
        if result is None:
            rows.append({
                "fpl_id": fpl_row["fpl_id"],
                "fpl_name": fpl_row["fpl_name"],
                "fpl_team": fpl_row["fpl_team"],
                "understat_id": "",
                "understat_name": "",
                "understat_team": "",
                "match_score": 0.0,
                "verified": False,
                "notes": "NO_MATCH",
            })
        else:
            matched_name, score, idx = result
            understat_row = understat_index.iloc[idx]
            rows.append({
                "fpl_id": fpl_row["fpl_id"],
                "fpl_name": fpl_row["fpl_name"],
                "fpl_team": fpl_row["fpl_team"],
                "understat_id": understat_row["understat_id"],
                "understat_name": matched_name,
                "understat_team": understat_row["understat_team"],
                "match_score": round(score, 1),
                "verified": score >= 95.0,  # auto-verify near-perfect matches
                "notes": "",
            })

    df = pd.DataFrame(rows)
    df = df.sort_values("match_score", ascending=False)
    return df


def main() -> None:
    logger.info("Loading FPL players...")
    fpl_df = load_fpl_players()
    logger.info("FPL: %d players", len(fpl_df))

    logger.info("Loading Understat players for seasons %s...", TARGET_SEASONS)
    understat_df = load_understat_players(TARGET_SEASONS)

    logger.info("Fuzzy matching...")
    candidates = fuzzy_match(fpl_df, understat_df)

    MAPPING_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(MAPPING_OUTPUT, index=False)

    # Summary
    auto_verified = candidates["verified"].sum()
    no_match = (candidates["match_score"] == 0).sum()
    needs_review = len(candidates) - auto_verified - no_match

    logger.info("=" * 50)
    logger.info("Candidate file written to: %s", MAPPING_OUTPUT)
    logger.info("  Auto-verified (score >= 95): %d", auto_verified)
    logger.info("  Needs manual review:         %d", needs_review)
    logger.info("  No match found:              %d", no_match)
    logger.info("=" * 50)
    logger.info(
        "Next step: open %s, review all rows where verified=False,\n"
        "           correct understat_id where wrong, then save as:\n"
        "           data/understat_player_mapping.csv",
        MAPPING_OUTPUT,
    )


if __name__ == "__main__":
    main()
