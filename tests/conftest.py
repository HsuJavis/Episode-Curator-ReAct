import json
import os
import tempfile
from pathlib import Path

import pytest


def _resolve_api_key():
    """Resolve API key: param → env var → OAuth credentials."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    creds_path = Path.home() / ".claude" / ".credentials.json"
    if creds_path.exists():
        creds = json.loads(creds_path.read_text())
        oauth = creds.get("claudeAiOauth", {})
        token = oauth.get("accessToken")
        if token:
            return token
    pytest.skip("No API key or OAuth token available")


@pytest.fixture
def api_key():
    """Provide a valid API key for tests (from env or OAuth)."""
    return _resolve_api_key()


@pytest.fixture
def tmp_storage(tmp_path):
    """Provide a temporary directory for EpisodeStore."""
    return str(tmp_path / "episode_store")
