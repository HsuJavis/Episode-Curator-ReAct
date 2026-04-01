# tool_registry.py — Dynamic tool loading plugin (open/close book)

from __future__ import annotations

from typing import Any, Optional

from react_agent import AgentContext, SkillPlugin, SkillPluginManager

# Minimum content length to trigger compression on close-book
_COMPRESS_THRESHOLD = 200


class ToolRegistryPlugin(SkillPlugin):
    """Provides load_tools/unload_tools and injects deferred tool catalog into system prompt.

    Close-book behavior: when unload_tools is called, tool_result content for
    the unloaded tools is compressed in ctx.messages — keeping metadata summary
    but discarding the bulk content to free context window space.
    """

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

    def after_action(self, ctx: AgentContext, tool_call: dict, result: Any) -> Optional[Any]:
        """Close-book / re-open-book: compress or restore tool_result content."""
        name = tool_call.get("name")
        tool_names = set(tool_call.get("input", {}).get("names", []))
        if not tool_names:
            return None
        if name == "unload_tools":
            self._compress_tool_history(ctx, tool_names)
        elif name == "load_tools":
            self._expand_tool_history(ctx, tool_names)
        return None

    @staticmethod
    def _build_tool_use_id_map(messages: list[dict]) -> dict[str, str]:
        """Build a map of tool_use_id → tool_name from assistant messages."""
        id_to_name: dict[str, str] = {}
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                block_type = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
                if block_type == "tool_use":
                    tu_id = getattr(block, "id", None) or (block.get("id") if isinstance(block, dict) else None)
                    tu_name = getattr(block, "name", None) or (block.get("name") if isinstance(block, dict) else None)
                    if tu_id and tu_name:
                        id_to_name[tu_id] = tu_name
        return id_to_name

    def _compress_tool_history(self, ctx: AgentContext, tool_names: set[str]) -> None:
        """Scan ctx.messages and compress tool_result content for the given tools."""
        id_to_name = self._build_tool_use_id_map(ctx.messages)

        # Compress matching tool_result blocks in user messages
        for msg in ctx.messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                tu_id = block.get("tool_use_id", "")
                tool_name = id_to_name.get(tu_id)
                if tool_name not in tool_names:
                    continue
                original = block.get("content", "")
                if not isinstance(original, str) or len(original) <= _COMPRESS_THRESHOLD:
                    continue
                # Save original for re-expand
                store = ctx.metadata.setdefault("_compressed_results", {})
                store[tu_id] = original
                # Compress: keep preview + metadata tag
                preview = original[:100].rstrip()
                block["content"] = (
                    f"[{tool_name} result compressed] {preview}... "
                    f"({len(original)} chars — use load_tools to re-expand)"
                )

    def _expand_tool_history(self, ctx: AgentContext, tool_names: set[str]) -> None:
        """Restore previously compressed tool_result content (re-open book)."""
        store: dict = ctx.metadata.get("_compressed_results", {})
        if not store:
            return

        id_to_name = self._build_tool_use_id_map(ctx.messages)

        # Restore matching tool_result blocks
        for msg in ctx.messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                tu_id = block.get("tool_use_id", "")
                tool_name = id_to_name.get(tu_id)
                if tool_name not in tool_names:
                    continue
                if tu_id in store:
                    block["content"] = store.pop(tu_id)

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
        ctx.metadata.setdefault("_system_extra_parts", {})["tool_catalog"] = catalog_text
