"""
src/io/understat_scraper.py
----------------------------
Scrapes Understat player and league data by parsing embedded JSON
from HTML <script> tags.

Understat has no public API. All data is injected into the page as
JSON strings inside <script> blocks via this pattern:

    var matchesData = JSON.parse('...')
    var shotsData   = JSON.parse('...')
    var playersData = JSON.parse('...')

We extract these with regex, unescape them, and parse to Python dicts.

TEMPORAL SAFETY:
  All Understat data is historical (post-match). We only use it for
  features at GW t built from matches < t. No future data is exposed.

SCRAPING ETHICS:
  - Rate-limited (configurable, default 2s between requests)
  - Retries with exponential backoff
  - Respects HTTP errors (no hammering on 429/5xx)
  - Caches aggressively to minimize repeat requests
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://understat.com"

# Understat seasons are referenced by the start year
SEASON_MAP = {
    "2022-23": "2022",
    "2023-24": "2023",
    "2024-25": "2024",
}

# Regex to extract embedded JSON blobs
# Understat uses: var <name> = JSON.parse('<escaped_json>')
_JSON_SCRIPT_RE = re.compile(
    r"var\s+(\w+)\s*=\s*JSON\.parse\('(.+?)'\)",
    re.DOTALL,
)

# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------


def _build_session(
    max_retries: int = 5,
    backoff_factor: float = 2.0,
) -> requests.Session:
    retry = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-GB,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })
    return session


# ---------------------------------------------------------------------------
# Core HTML + JSON extraction
# ---------------------------------------------------------------------------


def _fetch_html(
    session: requests.Session,
    url: str,
    timeout: int = 30,
    rate_limit_pause: float = 2.0,
) -> str:
    """
    GET url, return HTML text.
    Raises requests.HTTPError on 4xx/5xx after retries.
    """
    time.sleep(rate_limit_pause)
    logger.debug("GET %s", url)
    resp = session.get(url, timeout=timeout)
    if resp.status_code != 200:
        logger.error("HTTP %d for %s", resp.status_code, url)
        resp.raise_for_status()
    return resp.text


def _extract_json_vars(html: str) -> dict[str, Any]:
    """
    Parse all JSON.parse('...') variable assignments from Understat HTML.
    Uses BeautifulSoup to find script tags, then extracts by string indexing.
    """
    from bs4 import BeautifulSoup
    
    result: dict[str, Any] = {}
    soup = BeautifulSoup(html, "html.parser")
    scripts = soup.find_all("script")
    
    for script in scripts:
        if not script.string:
            continue
        text = script.string.strip()
        if "JSON.parse('" not in text:
            continue
        # Extract variable name
        if not text.startswith("var "):
            continue
        var_name = text.split(" ")[1]
        try:
            ind_start = text.index("('") + 2
            ind_end = text.index("')")
            json_str = text[ind_start:ind_end]
            decoded = json_str.encode("utf-8").decode("unicode_escape")
            result[var_name] = json.loads(decoded)
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.warning("Failed to decode var '%s': %s", var_name, exc)
    
    return result


# ---------------------------------------------------------------------------
# League-level scraping (player list + season IDs)
# ---------------------------------------------------------------------------


def scrape_league_players(
    season_year: str,
    session: requests.Session | None = None,
    rate_limit_pause: float = 2.0,
) -> list[dict[str, Any]]:
    """
    Scrape the EPL league page for a given season year (e.g. '2024').

    Returns list of player dicts with fields:
        id, player_name, team_title, games, time, goals, assists,
        shots, xG, xA, npg, npxG, xGChain, xGBuildup

    This is the master list used to discover Understat player IDs.
    """
    session = session or _build_session()
    url = f"{BASE_URL}/league/EPL/{season_year}"
    html = _fetch_html(session, url, rate_limit_pause=rate_limit_pause)
    vars_ = _extract_json_vars(html)

    players = vars_.get("playersData")
    if not players:
        raise ValueError(
            f"playersData not found in EPL/{season_year} page. "
            "Understat may have changed its page structure."
        )

    logger.info(
        "League %s: found %d players in playersData",
        season_year,
        len(players),
    )
    return players


# ---------------------------------------------------------------------------
# Player-level scraping (match history + shot data)
# ---------------------------------------------------------------------------


def scrape_player_matches(
    understat_id: int | str,
    session: requests.Session | None = None,
    rate_limit_pause: float = 2.0,
) -> dict[str, Any]:
    """
    Scrape a player's Understat page and return all embedded data.

    Returns dict with keys (all present if player page loaded correctly):
        matchesData  – list of match-level aggregates
        shotsData    – list of individual shot records
        groupsData   – season group summaries
        (possibly others depending on Understat version)

    matchesData fields per match:
        id, isHome, date, season, goals, xG, assists, xA,
        shots, key_passes, yellow, red, time, roster_id, h_team, a_team,
        h_goals, a_goals, position, positionOrder

    shotsData fields per shot:
        id, minute, result, X, Y, xG, player, h_a,
        player_assisted, lastAction, type (foot/head),
        situation (open play / corner / set piece / direct free kick),
        season, match_id, h_team, a_team, h_goals, a_goals, date
    """
    session = session or _build_session()
    url = f"{BASE_URL}/player/{understat_id}"
    html = _fetch_html(session, url, rate_limit_pause=rate_limit_pause)
    vars_ = _extract_json_vars(html)

    if not vars_:
        raise ValueError(
            f"No JSON vars found for player {understat_id}. "
            f"Page may have failed to load or structure changed."
        )

    matches = vars_.get("matchesData", [])
    shots = vars_.get("shotsData", [])

    logger.debug(
        "Player %s: %d matches, %d shots",
        understat_id,
        len(matches),
        len(shots),
    )

    return {
        "matchesData": matches,
        "shotsData": shots,
        "groupsData": vars_.get("groupsData", []),
    }


# ---------------------------------------------------------------------------
# Bulk player scraping
# ---------------------------------------------------------------------------


def scrape_all_players(
    understat_ids: list[int | str],
    session: requests.Session | None = None,
    rate_limit_pause: float = 2.0,
    log_every: int = 25,
) -> dict[int, dict[str, Any]]:
    """
    Scrape match + shot data for a list of Understat player IDs.

    Returns dict mapping understat_id → scrape result.
    Failed players are logged and skipped (not raised).
    """
    session = session or _build_session()
    results: dict[int, dict] = {}
    failed: list[int] = []

    for i, uid in enumerate(understat_ids):
        if i > 0 and i % log_every == 0:
            logger.info("Understat scrape progress: %d / %d", i, len(understat_ids))
        try:
            results[int(uid)] = scrape_player_matches(
                uid,
                session=session,
                rate_limit_pause=rate_limit_pause,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed player %s: %s", uid, exc)
            failed.append(int(uid))

    if failed:
        logger.warning("Failed Understat IDs (%d): %s", len(failed), failed[:20])

    logger.info(
        "Understat bulk scrape: %d / %d succeeded",
        len(results),
        len(understat_ids),
    )
    return results
