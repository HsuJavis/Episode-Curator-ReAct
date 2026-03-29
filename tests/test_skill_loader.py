"""TDD tests for SkillLoader — written BEFORE implementation."""

from pathlib import Path

import pytest

from skill_loader import SkillManager, SkillLoaderPlugin
from react_agent import AgentContext, SkillPluginManager

FIXTURES_DIR = str(Path(__file__).parent / "fixtures" / "skills")


# ============================================================
# Frontmatter Parsing
# ============================================================

class TestFrontmatterParsing:
    def test_parse_yaml_frontmatter(self):
        text = "---\nname: commit\ndescription: A skill\n---\n# Body"
        meta, body = SkillManager._parse_frontmatter(text)
        assert isinstance(meta, dict)
        assert "name" in meta

    def test_parse_name_and_description(self):
        text = "---\nname: commit\ndescription: Helps with commits\n---\nBody"
        meta, _ = SkillManager._parse_frontmatter(text)
        assert meta["name"] == "commit"
        assert meta["description"] == "Helps with commits"

    def test_parse_body_content(self):
        text = "---\nname: x\n---\n# Heading\nParagraph"
        _, body = SkillManager._parse_frontmatter(text)
        assert "# Heading" in body
        assert "Paragraph" in body

    def test_missing_frontmatter_uses_directory_name(self):
        text = "Just content, no frontmatter."
        meta, body = SkillManager._parse_frontmatter(text)
        assert meta == {}
        assert "Just content" in body

    def test_empty_skill_md(self):
        meta, body = SkillManager._parse_frontmatter("")
        assert meta == {}
        assert body == ""


# ============================================================
# SkillManager Discovery
# ============================================================

class TestSkillManager:
    def test_discover_skills_from_directory(self):
        mgr = SkillManager(skills_dir=FIXTURES_DIR)
        skills = mgr.list_skills()
        names = {s["name"] for s in skills}
        assert "commit" in names
        assert "review-pr" in names

    def test_no_skills_directory_returns_empty(self, tmp_path):
        mgr = SkillManager(skills_dir=str(tmp_path / "nonexistent"))
        skills = mgr.list_skills()
        assert skills == []

    def test_multiple_skills_discovered(self):
        mgr = SkillManager(skills_dir=FIXTURES_DIR)
        skills = mgr.list_skills()
        assert len(skills) >= 2

    def test_skill_descriptions_collected(self):
        mgr = SkillManager(skills_dir=FIXTURES_DIR)
        skills = mgr.list_skills()
        for skill in skills:
            assert "name" in skill
            assert "description" in skill


# ============================================================
# Skill Body Loading
# ============================================================

class TestSkillBodyLoading:
    def test_load_skill_body(self):
        mgr = SkillManager(skills_dir=FIXTURES_DIR)
        body = mgr.load_skill_body("commit")
        assert body is not None
        assert "# Commit Skill" in body

    def test_load_nonexistent_skill(self):
        mgr = SkillManager(skills_dir=FIXTURES_DIR)
        body = mgr.load_skill_body("nonexistent-skill")
        assert body is None or body == ""


# ============================================================
# SkillLoaderPlugin
# ============================================================

class TestSkillLoaderPlugin:
    def test_plugin_name(self):
        mgr = SkillManager(skills_dir=FIXTURES_DIR)
        plugin = SkillLoaderPlugin(mgr)
        assert plugin.name == "skills"

    def test_on_agent_start_injects_descriptions(self):
        mgr = SkillManager(skills_dir=FIXTURES_DIR)
        plugin = SkillLoaderPlugin(mgr)
        ctx = AgentContext(user_query="test")
        plugin.on_agent_start(ctx)
        extra = ctx.metadata.get("system_prompt_extra", "")
        assert "Available skills:" in extra
        assert "commit" in extra
        assert "review-pr" in extra

    def test_registers_with_plugin_manager(self):
        mgr = SkillManager(skills_dir=FIXTURES_DIR)
        plugin = SkillLoaderPlugin(mgr)
        pm = SkillPluginManager()
        pm.register(plugin)  # Should not raise

    def test_no_tools(self):
        mgr = SkillManager(skills_dir=FIXTURES_DIR)
        plugin = SkillLoaderPlugin(mgr)
        assert plugin.get_tools() == []
