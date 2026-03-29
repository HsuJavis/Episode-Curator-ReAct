# system_tools.py — System Tools Plugin (read/write/grep/search/bash/web_search)

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from react_agent import SkillPlugin

MAX_OUTPUT = 100_000  # 100KB cap for bash output


class SystemToolsPlugin(SkillPlugin):
    """Provides filesystem, shell, and search tools for the ReAct Agent."""

    TOOLS = [
        {
            "name": "read",
            "description": "Read a file's contents. Returns numbered lines.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to the file"},
                    "offset": {"type": "integer", "description": "Start line (1-indexed, default 1)"},
                    "limit": {"type": "integer", "description": "Max lines to return (default 2000)"},
                },
                "required": ["file_path"],
            },
        },
        {
            "name": "write",
            "description": "Write content to a file. Creates intermediate directories.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to the file"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["file_path", "content"],
            },
        },
        {
            "name": "grep",
            "description": "Search file contents with a regex pattern.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "File or directory to search in"},
                    "include": {"type": "string", "description": "Glob to filter files (e.g. '*.py')"},
                },
                "required": ["pattern", "path"],
            },
        },
        {
            "name": "search",
            "description": "Find files matching a glob pattern.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern (e.g. '*.py', '**/*.md')"},
                    "path": {"type": "string", "description": "Directory to search in"},
                },
                "required": ["pattern"],
            },
        },
        {
            "name": "bash",
            "description": "Execute a shell command and return output.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)"},
                },
                "required": ["command"],
            },
        },
        {
            "name": "web_search",
            "description": "Search the web for information.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    ]

    @property
    def name(self) -> str:
        return "system_tools"

    def is_deferred(self) -> bool:
        return True

    def get_tools(self) -> list[dict]:
        return list(self.TOOLS)

    def execute_tool(self, name: str, tool_input: dict) -> Any:
        method = getattr(self, f"_tool_{name}", None)
        if method is None:
            raise ValueError(f"Unknown tool: {name}")
        return method(tool_input)

    # ── read ──

    def _tool_read(self, inp: dict) -> str:
        file_path = inp.get("file_path", "")
        offset = inp.get("offset", 1)
        limit = inp.get("limit", 2000)

        p = Path(file_path)
        if not p.exists():
            return f"Error: File not found: {file_path}"
        if p.is_dir():
            return f"Error: Path is a directory, not a file: {file_path}"

        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"Error: Cannot read binary file: {file_path}"
        except PermissionError:
            return f"Error: Permission denied: {file_path}"

        lines = text.splitlines()
        start = max(0, offset - 1)
        end = start + limit
        selected = lines[start:end]

        numbered = []
        for i, line in enumerate(selected, start=start + 1):
            numbered.append(f"{i}\t{line}")
        return "\n".join(numbered)

    # ── write ──

    def _tool_write(self, inp: dict) -> str:
        file_path = inp.get("file_path", "")
        content = inp.get("content", "")

        p = Path(file_path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} bytes to {file_path}"
        except PermissionError:
            return f"Error: Permission denied: {file_path}"
        except OSError as e:
            return f"Error: {e}"

    # ── grep ──

    def _tool_grep(self, inp: dict) -> str:
        pattern = inp.get("pattern", "")
        path = inp.get("path", ".")
        include = inp.get("include")

        p = Path(path)
        if not p.exists():
            return f"Error: Path not found: {path}"

        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Error: Invalid regex: {e}"

        matches = []
        files = []
        if p.is_file():
            files = [p]
        elif p.is_dir():
            if include:
                files = sorted(p.rglob(include))
            else:
                files = sorted(f for f in p.rglob("*") if f.is_file())

        for fp in files[:1000]:  # Cap file count
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore")
            except (PermissionError, OSError):
                continue
            for line_no, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    rel = fp.relative_to(p) if p.is_dir() else fp.name
                    matches.append(f"{rel}:{line_no}: {line}")
                    if len(matches) >= 500:
                        break
            if len(matches) >= 500:
                break

        if not matches:
            return "No matches found."
        return "\n".join(matches)

    # ── search ──

    def _tool_search(self, inp: dict) -> str:
        pattern = inp.get("pattern", "*")
        path = inp.get("path", ".")

        p = Path(path)
        if not p.exists():
            return f"Error: Path not found: {path}"

        results = sorted(p.glob(pattern))
        if not results:
            return "No matches found."
        lines = [str(r) for r in results[:500]]
        return "\n".join(lines)

    # ── bash ──

    def _tool_bash(self, inp: dict) -> str:
        command = inp.get("command", "")
        timeout = inp.get("timeout", 30)

        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = proc.stdout
            if proc.returncode != 0:
                output = proc.stdout + proc.stderr
                if not output.strip():
                    output = f"Error: Command exited with code {proc.returncode}"
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout}s"
        except Exception as e:
            return f"Error: {e}"

        if len(output) > MAX_OUTPUT:
            output = output[:MAX_OUTPUT] + f"\n... (truncated, {len(output)} total bytes)"
        return output

    # ── web_search ──

    def _tool_web_search(self, inp: dict) -> str:
        query = inp.get("query", "").strip()
        if not query:
            return "Error: Empty search query"

        # Stub implementation — returns structured placeholder
        return json.dumps({
            "results": [
                {
                    "title": f"Search result for: {query}",
                    "snippet": f"This is a placeholder result for the query '{query}'. "
                               "Connect a real search provider for actual results.",
                    "url": "https://example.com",
                },
            ],
        }, ensure_ascii=False)
