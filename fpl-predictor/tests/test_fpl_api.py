"""
tests/test_fpl_api.py
---------------------
Unit tests for data ingestion layer.
Uses mocked HTTP responses — no real API calls.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from io.fpl_api import (
    _fetch,
    _save_raw,
    fetch_bootstrap,
    fetch_fixtures,
    fetch_player_summary,
    load_latest,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_BOOTSTRAP = {
    "elements": [{"id": 1, "status": "a"}, {"id": 2, "status": "u"}],
    "teams": [{"id": 1, "name": "Arsenal"}],
    "events": [{"id": 1, "finished": True}, {"id": 2, "finished": False}],
}

MOCK_FIXTURES = [
    {
        "id": 1,
        "event": 1,
        "team_h": 1,
        "team_a": 2,
        "finished": True,
        "team_h_difficulty": 2,
        "team_a_difficulty": 4,
    }
]

MOCK_PLAYER_SUMMARY = {
    "history": [
        {
            "round": 1,
            "total_points": 9,
            "minutes": 90,
            "goals_scored": 1,
            "assists": 0,
        }
    ],
    "history_past": [],
    "fixtures": [{"event": 2, "team_h": 1, "team_a": 3}],
}


def _mock_response(data: dict | list, status_code: int = 200) -> MagicMock:
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = data
    mock.raise_for_status = MagicMock()
    return mock


# ---------------------------------------------------------------------------
# _fetch tests
# ---------------------------------------------------------------------------


def test_fetch_returns_parsed_json():
    session = MagicMock()
    session.get.return_value = _mock_response({"key": "value"})
    result = _fetch(session, "https://example.com", rate_limit_pause=0)
    assert result == {"key": "value"}


def test_fetch_raises_on_bad_status():
    session = MagicMock()
    mock_resp = _mock_response({}, status_code=500)
    mock_resp.raise_for_status.side_effect = Exception("HTTP 500")
    session.get.return_value = mock_resp
    with pytest.raises(Exception, match="HTTP 500"):
        _fetch(session, "https://example.com", rate_limit_pause=0)


def test_fetch_raises_on_invalid_json():
    session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.side_effect = json.JSONDecodeError("err", "", 0)
    mock_resp.text = "not json"
    session.get.return_value = mock_resp
    with pytest.raises(ValueError, match="Non-JSON"):
        _fetch(session, "https://example.com", rate_limit_pause=0)


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


def test_save_and_load_latest(tmp_path):
    data = {"test": 123}
    _save_raw(data, "bootstrap", tmp_path)
    loaded = load_latest("bootstrap", raw_dir=tmp_path)
    assert loaded == data


def test_load_latest_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_latest("nonexistent", raw_dir=tmp_path)


def test_save_creates_versioned_and_latest(tmp_path):
    _save_raw({"a": 1}, "fixtures", tmp_path)
    files = list(tmp_path.glob("fixtures_*.json"))
    assert any("latest" in f.name for f in files)
    assert any("latest" not in f.name for f in files)


# ---------------------------------------------------------------------------
# fetch_bootstrap tests
# ---------------------------------------------------------------------------


@patch("io.fpl_api._fetch")
def test_fetch_bootstrap_returns_data(mock_fetch, tmp_path):
    mock_fetch.return_value = MOCK_BOOTSTRAP
    result = fetch_bootstrap(raw_dir=tmp_path, save=True)
    assert "elements" in result
    assert len(result["elements"]) == 2


@patch("io.fpl_api._fetch")
def test_fetch_bootstrap_saves_raw(mock_fetch, tmp_path):
    mock_fetch.return_value = MOCK_BOOTSTRAP
    fetch_bootstrap(raw_dir=tmp_path, save=True)
    assert (tmp_path / "bootstrap_latest.json").exists()


@patch("io.fpl_api._fetch")
def test_fetch_bootstrap_no_save(mock_fetch, tmp_path):
    mock_fetch.return_value = MOCK_BOOTSTRAP
    fetch_bootstrap(raw_dir=tmp_path, save=False)
    assert not (tmp_path / "bootstrap_latest.json").exists()


# ---------------------------------------------------------------------------
# fetch_fixtures tests
# ---------------------------------------------------------------------------


@patch("io.fpl_api._fetch")
def test_fetch_fixtures_returns_list(mock_fetch, tmp_path):
    mock_fetch.return_value = MOCK_FIXTURES
    result = fetch_fixtures(raw_dir=tmp_path, save=False)
    assert isinstance(result, list)
    assert result[0]["event"] == 1


# ---------------------------------------------------------------------------
# fetch_player_summary tests
# ---------------------------------------------------------------------------


@patch("io.fpl_api._fetch")
def test_fetch_player_summary_structure(mock_fetch, tmp_path):
    mock_fetch.return_value = MOCK_PLAYER_SUMMARY
    result = fetch_player_summary(42, raw_dir=tmp_path, save=False)
    assert "history" in result
    assert "fixtures" in result
    assert result["history"][0]["total_points"] == 9


@patch("io.fpl_api._fetch")
def test_fetch_player_summary_saves_with_id(mock_fetch, tmp_path):
    mock_fetch.return_value = MOCK_PLAYER_SUMMARY
    fetch_player_summary(99, raw_dir=tmp_path, save=True)
    assert (tmp_path / "player_99_latest.json").exists()


# ---------------------------------------------------------------------------
# run_full_ingestion — active player filtering
# ---------------------------------------------------------------------------


@patch("io.fpl_api.fetch_all_players")
@patch("io.fpl_api.fetch_fixtures")
@patch("io.fpl_api.fetch_bootstrap")
def test_run_full_ingestion_filters_unavailable(
    mock_bootstrap, mock_fixtures, mock_all_players, tmp_path
):
    """Players with status='u' should be excluded from element-summary fetch."""
    mock_bootstrap.return_value = MOCK_BOOTSTRAP
    mock_fixtures.return_value = MOCK_FIXTURES
    mock_all_players.return_value = {1: MOCK_PLAYER_SUMMARY}

    from io.fpl_api import run_full_ingestion

    result = run_full_ingestion(raw_dir=tmp_path, save=False)

    called_ids = mock_all_players.call_args[0][0]
    assert 1 in called_ids    # status='a' → included
    assert 2 not in called_ids  # status='u' → excluded
    assert "bootstrap" in result
    assert "fixtures" in result
    assert "players" in result
