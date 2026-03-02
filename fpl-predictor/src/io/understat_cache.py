"""
src/io/understat_cache.py
--------------------------
Disk cache layer for Understat scraped data.

Cache strategy:
  - Each player's data is stored as raw/understat/player_{id}.json
  - A .meta.json sidecar records the fetch timestamp
  - Default TTL: 7 days for completed seasons, 1 day for current season
  - League player lists (for ID discovery) are cached per season year

This means:
  - Historical seasons (2022-23, 2023-24) are scraped ONCE and never re-fetched
    unless you explicitly force refresh (rare: Understat may backfill xG occasionally)
  - Current season (2024-25) is re-fetched weekly to pick up new matches

Directory structure:
  data/raw/understat/
    league_2022.json / league_2022.meta.json
    league_2023.json / league_2023.meta.json
    league_2024.json / league_2024.meta.json
    player_12345.json / player_12345.meta.json
    ...
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_UNDERSTAT_DIR = Path("data/raw/understat")

# Default TTL values
TTL_HISTORICAL_DAYS = 365   # completed seasons: almost never change
TTL_CURRENT_DAYS = 1        # current season: refresh daily
TTL_LEAGUE_DAYS = 7         # player ID lists: refresh weekly


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _meta_path(base: Path) -> Path:
    return base.parent / (base.stem + ".meta.json")


def _read_meta(base: Path) -> datetime | None:
    """Return fetch timestamp from sidecar, or None if missing/corrupt."""
    mp = _meta_path(base)
    if not mp.exists():
        return None
    try:
        meta = json.loads(mp.read_text())
        return datetime.fromisoformat(meta["fetched_at"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


def _write_meta(base: Path) -> None:
    mp = _meta_path(base)
    mp.write_text(json.dumps({"fetched_at": _utc_now().isoformat()}))


def _is_fresh(base: Path, ttl_days: float) -> bool:
    fetched_at = _read_meta(base)
    if fetched_at is None:
        return False
    age_days = (_utc_now() - fetched_at).total_seconds() / 86400
    return age_days < ttl_days


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# League cache
# ---------------------------------------------------------------------------


def league_cache_path(season_year: str, cache_dir: Path) -> Path:
    return cache_dir / f"league_{season_year}.json"


def get_cached_league(
    season_year: str,
    cache_dir: Path = DEFAULT_UNDERSTAT_DIR,
    ttl_days: float = TTL_LEAGUE_DAYS,
) -> list[dict] | None:
    """Return cached league player list if fresh, else None."""
    path = league_cache_path(season_year, cache_dir)
    if path.exists() and _is_fresh(path, ttl_days):
        logger.debug("Cache hit: league %s", season_year)
        return _read_json(path)
    return None


def save_league(
    season_year: str,
    data: list[dict],
    cache_dir: Path = DEFAULT_UNDERSTAT_DIR,
) -> None:
    path = league_cache_path(season_year, cache_dir)
    _write_json(path, data)
    _write_meta(path)
    logger.info("Cached league %s (%d players)", season_year, len(data))


# ---------------------------------------------------------------------------
# Player cache
# ---------------------------------------------------------------------------


def player_cache_path(understat_id: int, cache_dir: Path) -> Path:
    return cache_dir / f"player_{understat_id}.json"


def get_cached_player(
    understat_id: int,
    cache_dir: Path = DEFAULT_UNDERSTAT_DIR,
    ttl_days: float = TTL_CURRENT_DAYS,
) -> dict | None:
    """Return cached player data if fresh, else None."""
    path = player_cache_path(understat_id, cache_dir)
    if path.exists() and _is_fresh(path, ttl_days):
        logger.debug("Cache hit: player %d", understat_id)
        return _read_json(path)
    return None


def save_player(
    understat_id: int,
    data: dict,
    cache_dir: Path = DEFAULT_UNDERSTAT_DIR,
) -> None:
    path = player_cache_path(understat_id, cache_dir)
    _write_json(path, data)
    _write_meta(path)
    logger.debug("Cached player %d", understat_id)


def invalidate_player(
    understat_id: int,
    cache_dir: Path = DEFAULT_UNDERSTAT_DIR,
) -> None:
    """Force re-fetch on next access by deleting meta file."""
    mp = _meta_path(player_cache_path(understat_id, cache_dir))
    if mp.exists():
        mp.unlink()
        logger.info("Invalidated cache for player %d", understat_id)


def list_cached_player_ids(cache_dir: Path = DEFAULT_UNDERSTAT_DIR) -> list[int]:
    """Return list of player IDs that have a cache file (fresh or stale)."""
    return [
        int(p.stem.split("_")[1])
        for p in cache_dir.glob("player_*.json")
        if not p.stem.endswith(".meta")
    ]


# ---------------------------------------------------------------------------
# Orchestrated fetch-or-cache
# ---------------------------------------------------------------------------


def fetch_league_with_cache(
    season_year: str,
    scraper_fn,
    cache_dir: Path = DEFAULT_UNDERSTAT_DIR,
    ttl_days: float = TTL_LEAGUE_DAYS,
    force_refresh: bool = False,
) -> list[dict]:
    """
    Return league player list, using cache if fresh.

    Parameters
    ----------
    season_year : e.g. '2024'
    scraper_fn : callable(season_year) → list[dict]
                 Should be understat_scraper.scrape_league_players
    cache_dir : where to store cache files
    ttl_days : cache TTL
    force_refresh : ignore cache and scrape fresh

    Returns list of player dicts from Understat league page.
    """
    if not force_refresh:
        cached = get_cached_league(season_year, cache_dir, ttl_days)
        if cached is not None:
            return cached

    logger.info("Scraping Understat league page for season %s...", season_year)
    data = scraper_fn(season_year)
    save_league(season_year, data, cache_dir)
    return data


def fetch_player_with_cache(
    understat_id: int,
    scraper_fn,
    cache_dir: Path = DEFAULT_UNDERSTAT_DIR,
    ttl_days: float = TTL_CURRENT_DAYS,
    force_refresh: bool = False,
) -> dict:
    """
    Return player match/shot data, using cache if fresh.

    Parameters
    ----------
    understat_id : Understat numeric player ID
    scraper_fn : callable(understat_id) → dict
                 Should be understat_scraper.scrape_player_matches
    cache_dir : where to store cache files
    ttl_days : cache TTL in days
    force_refresh : ignore cache

    Returns dict with matchesData, shotsData, groupsData.
    """
    if not force_refresh:
        cached = get_cached_player(understat_id, cache_dir, ttl_days)
        if cached is not None:
            return cached

    data = scraper_fn(understat_id)
    save_player(understat_id, data, cache_dir)
    return data


def bulk_fetch_players_with_cache(
    understat_ids: list[int],
    scraper_fn,
    cache_dir: Path = DEFAULT_UNDERSTAT_DIR,
    ttl_days: float = TTL_CURRENT_DAYS,
    force_refresh: bool = False,
    log_every: int = 25,
) -> dict[int, dict]:
    """
    Fetch multiple players, respecting cache TTL.

    Only hits the network for players whose cache is stale or missing.
    This is the main entry point for weekly ingestion runs.

    Returns dict mapping understat_id → player data dict.
    """
    results: dict[int, dict] = {}
    to_scrape: list[int] = []

    # Partition into cache-hits and network-needed
    for uid in understat_ids:
        if not force_refresh:
            cached = get_cached_player(uid, cache_dir, ttl_days)
            if cached is not None:
                results[uid] = cached
                continue
        to_scrape.append(uid)

    logger.info(
        "Understat bulk fetch: %d cache hits, %d to scrape",
        len(results),
        len(to_scrape),
    )

    for i, uid in enumerate(to_scrape):
        if i > 0 and i % log_every == 0:
            logger.info("Scraping progress: %d / %d", i, len(to_scrape))
        try:
            data = scraper_fn(uid)
            save_player(uid, data, cache_dir)
            results[uid] = data
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch/cache player %d: %s", uid, exc)

    logger.info(
        "Understat bulk fetch complete: %d / %d available",
        len(results),
        len(understat_ids),
    )
    return results
