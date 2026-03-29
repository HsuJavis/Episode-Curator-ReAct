# skill_loader.py — Skill Loader Plugin (discovers SKILL.md files)

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from react_agent import AgentContext, SkillPlugin


class SkillManager:
    """Discovers and loads skills from a directory of SKILL.md files."""

    def __init__(self, skills_dir: str = "skills/"):
        self._skills_dir = Path(skills_dir)
        self._skills: list[dict] = []
        self._discover()

    def _discover(self) -> None:
        """Scan skills_dir for subdirectories containing SKILL.md."""
        self._skills = []
        if not self._skills_dir.is_dir():
            return
        for child in sorted(self._skills_dir.iterdir()):
            if not child.is_dir():
                continue
            skill_file = child / "SKILL.md"
            if not skill_file.exists():
                continue
            text = skill_file.read_text(encoding="utf-8")
            meta, body = self._parse_frontmatter(text)
            name = meta.get("name", child.name)
            description = meta.get("description", "")
            self._skills.append({
                "name": name,
                "description": description,
                "dir": str(child),
            })

    @staticmethod
    def _parse_frontmatter(text: str) -> tuple[dict, str]:
        """Parse YAML-like frontmatter delimited by --- lines.

        Returns (metadata_dict, body_text). Uses regex, no pyyaml dependency.
        """
        if not text:
            return {}, ""

        match = re.match(r"^---\n(.*?)\n---\n?(.*)", text, re.DOTALL)
        if not match:
            return {}, text

        frontmatter_raw = match.group(1)
        body = match.group(2)

        meta: dict[str, str] = {}
        for line in frontmatter_raw.splitlines():
            # Split on first ": " to get key-value
            sep_idx = line.find(": ")
            if sep_idx != -1:
                key = line[:sep_idx].strip()
                value = line[sep_idx + 2:].strip()
                meta[key] = value

        return meta, body

    def list_skills(self) -> list[dict]:
        """Return list of discovered skills with name, description, dir."""
        return list(self._skills)

    def load_skill_body(self, name: str) -> Optional[str]:
        """Read the full SKILL.md body (below frontmatter) for a skill."""
        for skill in self._skills:
            if skill["name"] == name:
                skill_file = Path(skill["dir"]) / "SKILL.md"
                if skill_file.exists():
                    text = skill_file.read_text(encoding="utf-8")
                    _, body = self._parse_frontmatter(text)
                    return body
        return None


class SkillLoaderPlugin(SkillPlugin):
    """Plugin that injects a skill catalog into the agent's system prompt."""

    def __init__(self, manager: SkillManager):
        self._manager = manager

    @property
    def name(self) -> str:
        return "skills"

    def get_tools(self) -> list[dict]:
        return []

    def on_agent_start(self, ctx: AgentContext) -> None:
        skills = self._manager.list_skills()
        if not skills:
            return

        lines = ["Available skills:"]
        for s in skills:
            desc = s.get("description", "")
            lines.append(f"- {s['name']}: {desc}")

        catalog = "\n".join(lines)
        existing = ctx.metadata.get("system_prompt_extra", "")
        if existing:
            ctx.metadata["system_prompt_extra"] = f"{existing}\n\n{catalog}"
        else:
            ctx.metadata["system_prompt_extra"] = catalog
