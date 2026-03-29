# tool_registry.py — Dynamic tool loading plugin (open/close book)

from __future__ import annotations

from typing import Any, Optional

from react_agent import AgentContext, SkillPlugin, SkillPluginManager


class ToolRegistryPlugin(SkillPlugin):
    """Provides load_tools/unload_tools and injects deferred tool catalog into system prompt."""

    def __init__(self, manager: SkillPluginManager):
        self._manager = manager

    @property
    def name(self) -> str:
        return "tool_registry"

    def get_tools(self) -> list[dict]:
        return [
            {
                "name": "load_tools",
                "description": (
                    "Load tool schemas so they become available for use. "
                    "Call this with the names of tools you need from the catalog."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of tool names to load",
                        },
                    },
                    "required": ["names"],
                },
            },
            {
                "name": "unload_tools",
                "description": (
                    "Unload tool schemas to free up context space. "
                    "The tools become unavailable until loaded again."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of tool names to unload",
                        },
                    },
                    "required": ["names"],
                },
            },
        ]

    def execute_tool(self, name: str, tool_input: dict) -> Any:
        names = tool_input.get("names", [])
        if name == "load_tools":
            return self._manager.load_tools(names)
        elif name == "unload_tools":
            return self._manager.unload_tools(names)
        raise ValueError(f"Unknown tool: {name}")

    def on_agent_start(self, ctx: AgentContext) -> None:
        """Inject deferred tool catalog into system prompt."""
        catalog = self._manager.get_tool_catalog()
        deferred = [t for t in catalog if not t["loaded"] and t["name"] not in ("load_tools", "unload_tools")]
        if not deferred:
            return

        lines = [
            "Deferred tools (use load_tools to activate before use, unload_tools to free context):",
        ]
        for t in deferred:
            lines.append(f"- {t['name']}: {t['description']}")

        catalog_text = "\n".join(lines)
        existing = ctx.metadata.get("system_prompt_extra", "")
        if existing:
            ctx.metadata["system_prompt_extra"] = f"{existing}\n\n{catalog_text}"
        else:
            ctx.metadata["system_prompt_extra"] = catalog_text
