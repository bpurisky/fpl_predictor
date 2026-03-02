"""
fpl_api.py
----------
Production-grade FPL API client.

Responsibilities:
  - Fetch raw data from official FPL endpoints
  - Persist raw JSON to disk with versioned filenames
  - Retry / rate-limit safely
  - Never transform or aggregate data (that belongs in build_dataset.py)

Endpoints covered:
  - /bootstrap-static/   → players, teams, gameweeks
  - /fixtures/           → all fixture metadata
  - /element-summary/{id}/  → per-player match history + upcoming fixtures
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://fantasy.premierleague.com/api"

ENDPOINTS = {
    "bootstrap": "/bootstrap-static/",
    "fixtures": "/fixtures/",
    "element_summary": "/element-summary/{player_id}/",
}

DEFAULT_RAW_DIR = Path("data/raw")

# ---------------------------------------------------------------------------
# HTTP Session
# ---------------------------------------------------------------------------


def _build_session(
    total_retries: int = 5,
    backoff_factor: float = 1.0,
    status_forcelist: tuple[int, ...] = (429, 500, 502, 503, 504),
) -> requests.Session:
    """Return a requests.Session with retry logic and a browser-like UA."""
    session = requests.Session()
    retry = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (compatible; FPL-Predictor/1.0; "
                "+https://github.com/your-repo)"
            )
        }
    )
    return session


# ---------------------------------------------------------------------------
# Low-level fetch
# ---------------------------------------------------------------------------


def _fetch(
    session: requests.Session,
    url: str,
    timeout: int = 30,
    rate_limit_pause: float = 0.5,
) -> dict[str, Any]:
    """
    GET *url* and return parsed JSON.

    Raises:
        requests.HTTPError  – on 4xx/5xx after retries
        ValueError          – if response is not valid JSON
    """
    logger.debug("GET %s", url)
    time.sleep(rate_limit_pause)  # polite pause between requests
    response = session.get(url, timeout=timeout)

    if response.status_code != 200:
        logger.error("HTTP %s for %s", response.status_code, url)
        response.raise_for_status()

    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise ValueError(f"Non-JSON response from {url}: {response.text[:200]}") from exc


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _save_raw(data: dict | list, name: str, raw_dir: Path) -> Path:
    """
    Persist *data* as JSON to *raw_dir*/<name>_<timestamp>.json.
    Also writes a 'latest' symlink for convenience.

    Returns the path of the written file.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    ts = _timestamp()
    versioned_path = raw_dir / f"{name}_{ts}.json"

    with versioned_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    latest_path = raw_dir / f"{name}_latest.json"
    # Overwrite latest (plain copy, not symlink, for cross-platform compat)
    with latest_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("Saved %s → %s", name, versioned_path)
    return versioned_path


def load_latest(name: str, raw_dir: Path = DEFAULT_RAW_DIR) -> dict | list:
    """Load the most recently saved raw file for *name*."""
    latest_path = raw_dir / f"{name}_latest.json"
    if not latest_path.exists():
        raise FileNotFoundError(
            f"No cached data for '{name}'. Run fetch first. Expected: {latest_path}"
        )
    with latest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Public fetch functions
# ---------------------------------------------------------------------------


def fetch_bootstrap(
    session: requests.Session | None = None,
    raw_dir: Path = DEFAULT_RAW_DIR,
    save: bool = True,
) -> dict[str, Any]:
    """
    Fetch /bootstrap-static/.

    Returns the full response dict containing:
      - 'elements'   : list of player dicts
      - 'teams'      : list of team dicts
      - 'events'     : list of gameweek dicts
      - 'element_types' : position metadata
    """
    session = session or _build_session()
    url = BASE_URL + ENDPOINTS["bootstrap"]
    data = _fetch(session, url)

    if save:
        _save_raw(data, "bootstrap", raw_dir)

    logger.info(
        "Bootstrap: %d players, %d teams, %d gameweeks",
        len(data.get("elements", [])),
        len(data.get("teams", [])),
        len(data.get("events", [])),
    )
    return data


def fetch_fixtures(
    session: requests.Session | None = None,
    raw_dir: Path = DEFAULT_RAW_DIR,
    save: bool = True,
) -> list[dict[str, Any]]:
    """
    Fetch /fixtures/.

    Returns list of fixture dicts. Key fields:
      - id, code, event (GW number), team_h, team_a
      - team_h_difficulty, team_a_difficulty
      - finished, kickoff_time, started
      - team_h_score, team_a_score (None if not played)
    """
    session = session or _build_session()
    url = BASE_URL + ENDPOINTS["fixtures"]
    data = _fetch(session, url)

    if save:
        _save_raw(data, "fixtures", raw_dir)

    played = sum(1 for f in data if f.get("finished"))
    logger.info("Fixtures: %d total, %d finished", len(data), played)
    return data


def fetch_player_summary(
    player_id: int,
    session: requests.Session | None = None,
    raw_dir: Path = DEFAULT_RAW_DIR,
    save: bool = True,
) -> dict[str, Any]:
    """
    Fetch /element-summary/{player_id}/.

    Returns dict with:
      - 'history'          : list of past match stats for this season
      - 'history_past'     : list of prior season summaries
      - 'fixtures'         : list of upcoming fixtures

    History fields of interest:
      total_points, minutes, goals_scored, assists, clean_sheets,
      goals_conceded, bonus, bps, influence, creativity, threat,
      ict_index, value (price * 10), selected, transfers_in/out,
      round (GW), was_home, opponent_team, kickoff_time
    """
    session = session or _build_session()
    url = (BASE_URL + ENDPOINTS["element_summary"]).format(player_id=player_id)
    data = _fetch(session, url)

    if save:
        _save_raw(data, f"player_{player_id}", raw_dir)

    return data


def fetch_all_players(
    player_ids: list[int],
    session: requests.Session | None = None,
    raw_dir: Path = DEFAULT_RAW_DIR,
    save: bool = True,
    rate_limit_pause: float = 0.5,
    log_every: int = 50,
) -> dict[int, dict[str, Any]]:
    """
    Fetch element-summary for every player in *player_ids*.

    Returns dict mapping player_id → raw API response.

    Args:
        player_ids       : list of FPL element IDs
        session          : shared session (created if None)
        raw_dir          : where to persist raw JSON per player
        save             : whether to persist each player's data
        rate_limit_pause : seconds to sleep between requests
        log_every        : log progress every N players
    """
    session = session or _build_session()
    results: dict[int, dict] = {}
    failed: list[int] = []

    for i, pid in enumerate(player_ids):
        if i > 0 and i % log_every == 0:
            logger.info("Player fetch progress: %d / %d", i, len(player_ids))
        try:
            results[pid] = fetch_player_summary(
                pid,
                session=session,
                raw_dir=raw_dir,
                save=save,
            )
            time.sleep(rate_limit_pause)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch player %d: %s", pid, exc)
            failed.append(pid)

    if failed:
        logger.warning("Failed player IDs (%d): %s", len(failed), failed)

    logger.info("Fetched %d / %d players successfully", len(results), len(player_ids))
    return results


# ---------------------------------------------------------------------------
# Convenience: full ingestion run
# ---------------------------------------------------------------------------


def run_full_ingestion(
    raw_dir: Path = DEFAULT_RAW_DIR,
    save: bool = True,
    rate_limit_pause: float = 0.5,
) -> dict[str, Any]:
    """
    Run a complete data pull:
      1. bootstrap-static  → players, teams, gameweeks
      2. fixtures
      3. element-summary for every active player

    Returns dict with keys: 'bootstrap', 'fixtures', 'players'

    This is the entry point for weekly retraining pipelines.
    """
    session = _build_session()
    logger.info("=== FPL Full Ingestion Started ===")

    bootstrap = fetch_bootstrap(session=session, raw_dir=raw_dir, save=save)
    fixtures = fetch_fixtures(session=session, raw_dir=raw_dir, save=save)

    # Only fetch players who are active (not unavailable codes 'u' which are
    # removed from the game entirely). We still fetch injured/suspended players
    # because we need their history.
    player_ids = [
        el["id"]
        for el in bootstrap["elements"]
        if el.get("status") != "u"  # 'u' = permanently unavailable
    ]
    logger.info("Fetching summaries for %d active players...", len(player_ids))

    players = fetch_all_players(
        player_ids,
        session=session,
        raw_dir=raw_dir,
        save=save,
        rate_limit_pause=rate_limit_pause,
    )

    logger.info("=== FPL Full Ingestion Complete ===")
    return {
        "bootstrap": bootstrap,
        "fixtures": fixtures,
        "players": players,
    }
