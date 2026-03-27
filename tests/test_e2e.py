"""E2E Playwright browser tests with real Anthropic API."""

import json
import os
import subprocess
import sys
import tempfile
import time

import pytest

# Check if playwright is available
try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not HAS_PLAYWRIGHT, reason="playwright not installed"),
]

PORT = 18765


@pytest.fixture(scope="module")
def server():
    """Launch web_app.py as a subprocess for E2E testing."""
    storage_dir = tempfile.mkdtemp(prefix="e2e_store_")
    env = os.environ.copy()
    env["EPISODE_STORE_DIR"] = storage_dir

    # Resolve API key for subprocess
    from tests.conftest import _resolve_api_key
    try:
        api_key = _resolve_api_key()
        env["ANTHROPIC_API_KEY"] = api_key
    except Exception:
        pytest.skip("No API key available for E2E tests")

    proc = subprocess.Popen(
        [sys.executable, "web_app.py", str(PORT)],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to start
    import urllib.request
    for _ in range(30):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/")
            break
        except Exception:
            time.sleep(0.5)
    else:
        proc.kill()
        pytest.fail("Server failed to start")

    yield {"port": PORT, "storage_dir": storage_dir, "process": proc}

    proc.terminate()
    proc.wait(timeout=5)


@pytest.fixture(scope="module")
def browser_context():
    """Create a Playwright browser context."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        yield context
        context.close()
        browser.close()


class TestE2EBrowser:
    """E2E: Playwright browser tests against real web app + real LLM."""

    @pytest.mark.llm
    def test_page_loads(self, server, browser_context):
        """Home page should load with input and button."""
        page = browser_context.new_page()
        page.goto(f"http://127.0.0.1:{server['port']}/")
        assert page.title() == "Episode Curator ReAct Agent"
        assert page.locator("#input").is_visible()
        assert page.locator("#send").is_visible()
        page.close()

    @pytest.mark.llm
    def test_send_message_gets_response(self, server, browser_context):
        """Sending a message via UI should display LLM response."""
        page = browser_context.new_page()
        page.goto(f"http://127.0.0.1:{server['port']}/")

        # Type and send
        page.fill("#input", "What is 1+1? Reply with just the number.")
        page.click("#send")

        # Wait for response (up to 30s for real API)
        page.wait_for_function(
            "document.getElementById('response').textContent !== 'Thinking...' && "
            "document.getElementById('response').textContent !== ''",
            timeout=30000,
        )
        response_text = page.text_content("#response")
        assert "2" in response_text
        page.close()

    @pytest.mark.llm
    def test_episodes_endpoint(self, server, browser_context):
        """After conversation, /episodes should return data."""
        page = browser_context.new_page()
        resp = page.request.get(f"http://127.0.0.1:{server['port']}/episodes")
        data = resp.json()
        assert "episodes" in data
        # Episodes may or may not exist depending on whether compression triggered
        assert isinstance(data["episodes"], dict)
        page.close()

    @pytest.mark.llm
    def test_facts_endpoint(self, server, browser_context):
        """After conversation, /facts should return data."""
        page = browser_context.new_page()
        resp = page.request.get(f"http://127.0.0.1:{server['port']}/facts")
        data = resp.json()
        assert "facts" in data
        assert isinstance(data["facts"], list)
        page.close()

    @pytest.mark.llm
    def test_chat_api_directly(self, server, browser_context):
        """POST /chat should return LLM response."""
        page = browser_context.new_page()
        resp = page.request.post(
            f"http://127.0.0.1:{server['port']}/chat",
            data=json.dumps({"message": "Say hello in one word."}),
            headers={"Content-Type": "application/json"},
        )
        data = resp.json()
        assert "response" in data
        assert len(data["response"]) > 0
        page.close()
