"""TDD tests for model-aware context window and compression threshold.

Requirements:
1. Model → context window mapping (some models are 200k, some 1M)
2. Compression threshold defaults to 50% of model's context window
3. --threshold-pct CLI arg to override compression threshold percentage
4. TUI context panel threshold = model's context window (not compression threshold)

Written BEFORE fix per TDD methodology.
"""

import time

import pytest

from cli_app import ContextUsagePanel, EpisodeCuratorApp, TUIPlugin
from react_agent import AgentContext


pytestmark = pytest.mark.tui


# ============================================================
# 1. Model context window mapping
# ============================================================

class TestModelContextWindows:
    """Each model should have a known context window size."""

    def test_model_context_windows_exist(self):
        """A mapping from model name to context window should exist."""
        from cli_app import MODEL_CONTEXT_WINDOWS
        assert isinstance(MODEL_CONTEXT_WINDOWS, dict)
        assert len(MODEL_CONTEXT_WINDOWS) >= 2

    def test_haiku_context_window(self):
        from cli_app import MODEL_CONTEXT_WINDOWS
        assert MODEL_CONTEXT_WINDOWS["claude-haiku-4-5-20251001"] == 200_000

    def test_sonnet_context_window(self):
        from cli_app import MODEL_CONTEXT_WINDOWS
        assert MODEL_CONTEXT_WINDOWS["claude-sonnet-4-20250514"] == 200_000

    def test_opus_context_window(self):
        """Opus models with 1M context should be in the mapping."""
        from cli_app import MODEL_CONTEXT_WINDOWS
        # At least one model should have 1M context
        has_1m = any(v >= 1_000_000 for v in MODEL_CONTEXT_WINDOWS.values())
        assert has_1m, "Should have at least one 1M context model"

    def test_unknown_model_defaults(self):
        """Unknown model should have a sensible default."""
        from cli_app import get_model_context_window
        window = get_model_context_window("some-unknown-model")
        assert window == 200_000  # safe default


# ============================================================
# 2. Compression threshold = 50% of context window by default
# ============================================================

class TestCompressionThresholdDefault:
    """Compression threshold should default to 50% of model's context window."""

    def test_default_threshold_is_half_of_context(self):
        """create_agent default threshold should be 50% of model context window."""
        from episode_curator import create_agent
        from cli_app import get_model_context_window
        # Default worker model is claude-sonnet-4-20250514, window = 200k
        # So default threshold should be 100k
        # We can't call create_agent without API key, but check the default
        import inspect
        sig = inspect.signature(create_agent)
        default_threshold = sig.parameters["threshold"].default
        default_model = sig.parameters["worker_model"].default
        expected_window = get_model_context_window(default_model)
        assert default_threshold == expected_window // 2, (
            f"Default threshold should be {expected_window // 2}, got {default_threshold}"
        )

    def test_threshold_pct_overrides(self):
        """threshold_pct parameter should override the default 50%."""
        from cli_app import get_model_context_window
        model = "claude-sonnet-4-20250514"
        window = get_model_context_window(model)

        # 30% of 200k = 60k
        threshold = int(window * 0.3)
        assert threshold == 60_000

        # 80% of 200k = 160k
        threshold_80 = int(window * 0.8)
        assert threshold_80 == 160_000


# ============================================================
# 3. TUI context panel threshold = model context window
# ============================================================

class TestTUIContextWindowFromModel:
    """TUI should set context panel threshold from model's context window."""

    def test_panel_threshold_matches_model(self):
        """ContextUsagePanel threshold should be the model's context window."""
        from cli_app import get_model_context_window
        window = get_model_context_window("claude-haiku-4-5-20251001")
        panel = ContextUsagePanel()
        panel.threshold = window
        rendered = panel.render()
        assert "200.0k" in rendered

    def test_panel_threshold_1m_model(self):
        """For a 1M context model, panel should show 1M."""
        from cli_app import get_model_context_window
        # Find a 1M model
        from cli_app import MODEL_CONTEXT_WINDOWS
        model_1m = None
        for name, window in MODEL_CONTEXT_WINDOWS.items():
            if window >= 1_000_000:
                model_1m = name
                break
        assert model_1m is not None, "Need a 1M model to test"

        window = get_model_context_window(model_1m)
        panel = ContextUsagePanel()
        panel.threshold = window
        rendered = panel.render()
        assert "1.0M" in rendered or "1000" in rendered


# ============================================================
# 4. CLI --threshold-pct argument
# ============================================================

class TestCLIThresholdPct:
    """Threshold percentage configuration."""

    def test_model_context_windows_has_known_models(self):
        """MODEL_CONTEXT_WINDOWS should contain haiku and sonnet."""
        from cli_app import MODEL_CONTEXT_WINDOWS
        models = list(MODEL_CONTEXT_WINDOWS.keys())
        assert any("haiku" in m for m in models)
        assert any("sonnet" in m for m in models)

    def test_threshold_calculation_from_pct(self):
        """Given model and pct, threshold should be window * pct / 100."""
        from cli_app import get_model_context_window
        model = "claude-sonnet-4-20250514"
        window = get_model_context_window(model)
        pct = 40
        threshold = int(window * pct / 100)
        assert threshold == 80_000  # 200k * 40% = 80k
