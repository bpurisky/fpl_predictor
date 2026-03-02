"""
scripts/ingest.py
-----------------
CLI entrypoint for running FPL data ingestion.

Usage:
    python scripts/ingest.py                        # full ingestion
    python scripts/ingest.py --bootstrap-only       # just bootstrap + fixtures
    python scripts/ingest.py --player-id 123        # single player (debug)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from io.fpl_api import (
    fetch_bootstrap,
    fetch_fixtures,
    fetch_player_summary,
    run_full_ingestion,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FPL Data Ingestion")
    parser.add_argument(
        "--bootstrap-only",
        action="store_true",
        help="Only fetch bootstrap + fixtures (no per-player summaries)",
    )
    parser.add_argument(
        "--player-id",
        type=int,
        default=None,
        help="Fetch a single player summary by FPL element ID (debug mode)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help="Directory to save raw JSON files",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.5,
        help="Seconds to pause between player requests",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.player_id is not None:
        logger.info("Debug mode: fetching single player %d", args.player_id)
        data = fetch_player_summary(args.player_id, raw_dir=args.raw_dir)
        history_len = len(data.get("history", []))
        upcoming_len = len(data.get("fixtures", []))
        logger.info(
            "Player %d: %d history rows, %d upcoming fixtures",
            args.player_id,
            history_len,
            upcoming_len,
        )
        return

    if args.bootstrap_only:
        logger.info("Bootstrap-only mode")
        fetch_bootstrap(raw_dir=args.raw_dir)
        fetch_fixtures(raw_dir=args.raw_dir)
        return

    run_full_ingestion(
        raw_dir=args.raw_dir,
        rate_limit_pause=args.rate_limit,
    )


if __name__ == "__main__":
    main()
