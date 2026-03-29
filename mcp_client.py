# mcp_client.py — MCP Client support for the ReAct Agent

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Any

from react_agent import SkillPlugin


class MCPManager:
    """Manages MCP server subprocesses and JSON-RPC communication."""

    def __init__(self, config_path: str | None = None):
        self.server_configs: dict[str, dict] = {}
        self.servers: dict[str, subprocess.Popen] = {}
        self._request_id = 0
        self._locks: dict[str, threading.Lock] = {}

        if config_path is not None:
            self._load_config(config_path)

    def _load_config(self, config_path: str) -> None:
        p = Path(config_path)
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        raw_servers = data.get("mcpServers", {})
        for name, cfg in raw_servers.items():
            self.server_configs[name] = self._expand_env(cfg)

    @staticmethod
    def _expand_env(cfg: dict) -> dict:
        """Expand ${VAR} references in command and args."""
        result = dict(cfg)
        if "command" in result:
            result["command"] = MCPManager._expand_vars(result["command"])
        if "args" in result:
            result["args"] = [MCPManager._expand_vars(a) for a in result["args"]]
        return result

    @staticmethod
    def _expand_vars(s: str) -> str:
        def replacer(m):
            var_name = m.group(1)
            return os.environ.get(var_name, m.group(0))
        return re.sub(r"\$\{(\w+)\}", replacer, s)

    def start_all(self) -> None:
        """Start all configured servers."""
        for name, cfg in self.server_configs.items():
            self.start_server(name, cfg)

    def start_server(self, name: str, config: dict) -> bool:
        """Launch an MCP server subprocess and send initialize."""
        command = config.get("command", "")
        args = config.get("args", [])
        env = {**os.environ, **config.get("env", {})}

        try:
            proc = subprocess.Popen(
                [command] + args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
        except (OSError, FileNotFoundError):
            return False

        self.servers[name] = proc
        self._locks[name] = threading.Lock()

        # Send initialize
        resp = self._send_request(name, "initialize", {})
        if resp is None:
            proc.terminate()
            del self.servers[name]
            del self._locks[name]
            return False

        return True

    def _send_request(self, server_name: str, method: str, params: dict) -> dict | None:
        """Send a JSON-RPC request and read the response."""
        proc = self.servers.get(server_name)
        if proc is None or proc.poll() is not None:
            return None

        lock = self._locks[server_name]
        with lock:
            self._request_id += 1
            request = {
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": method,
                "params": params,
            }
            try:
                line = json.dumps(request) + "\n"
                proc.stdin.write(line.encode("utf-8"))
                proc.stdin.flush()
                resp_line = proc.stdout.readline()
                if not resp_line:
                    return None
                return json.loads(resp_line.decode("utf-8"))
            except (OSError, json.JSONDecodeError, BrokenPipeError):
                return None

    def discover_tools(self, server_name: str) -> list[dict]:
        """Call tools/list and return Anthropic-format tool definitions."""
        resp = self._send_request(server_name, "tools/list", {})
        if resp is None:
            return []

        result = resp.get("result", {})
        raw_tools = result.get("tools", [])

        anthropic_tools = []
        for tool in raw_tools:
            converted = {
                "name": f"mcp__{server_name}__{tool['name']}",
                "description": tool.get("description", ""),
                "input_schema": tool.get("inputSchema", {"type": "object", "properties": {}}),
            }
            anthropic_tools.append(converted)

        return anthropic_tools

    def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> str:
        """Call a tool on a server and return the text result."""
        resp = self._send_request(server_name, "tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })
        if resp is None:
            return "Error: Server not responding"

        if "error" in resp:
            return f"Error: {resp['error'].get('message', 'Unknown error')}"

        result = resp.get("result", {})
        content = result.get("content", [])
        texts = []
        for block in content:
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "\n".join(texts) if texts else "No output"

    def shutdown(self) -> None:
        """Terminate all server subprocesses."""
        for name, proc in self.servers.items():
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    proc.kill()
                except OSError:
                    pass
        self.servers.clear()
        self._locks.clear()


class MCPPlugin(SkillPlugin):
    """SkillPlugin that exposes MCP server tools to the ReAct Agent."""

    def __init__(self, manager: MCPManager):
        self._manager = manager
        self._tools: list[dict] = []
        self._tool_map: dict[str, tuple[str, str]] = {}  # prefixed_name -> (server, tool)

        # Discover tools from all running servers
        for server_name in manager.servers:
            tools = manager.discover_tools(server_name)
            for tool in tools:
                prefixed = tool["name"]
                # Parse: mcp__{server}__{tool}
                parts = prefixed.split("__", 2)
                if len(parts) == 3:
                    original_tool_name = parts[2]
                    self._tool_map[prefixed] = (server_name, original_tool_name)
                self._tools.append(tool)

    @property
    def name(self) -> str:
        return "mcp"

    def is_deferred(self) -> bool:
        return True


    def get_tools(self) -> list[dict]:
        return list(self._tools)

    def execute_tool(self, name: str, tool_input: dict) -> Any:
        if name not in self._tool_map:
            raise ValueError(f"Unknown MCP tool: {name}")
        server_name, tool_name = self._tool_map[name]
        return self._manager.call_tool(server_name, tool_name, tool_input)
