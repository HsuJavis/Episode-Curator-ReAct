# system_tools.py — System Tools Plugin
# read/write/grep/search/bash/web_search/web_fetch/task_*/process_*/system_info/execute_skill

from __future__ import annotations

import json
import os
import platform
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from react_agent import SkillPlugin

MAX_OUTPUT = 100_000  # 100KB cap for bash output
MAX_FETCH = 500_000   # 500KB cap for web_fetch


class SystemToolsPlugin(SkillPlugin):
    """Provides filesystem, shell, search, task, process, and system tools."""

    def __init__(self):
        self._tasks: dict[str, dict] = {}      # task_id → task info
        self._processes: dict[int, dict] = {}   # pid → process info
        self._task_counter = 0
        self._lock = threading.Lock()
        self._skills_dir: str = str(Path("skills").resolve())  # absolute path

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
        # ── web_fetch ──
        {
            "name": "web_fetch",
            "description": "Fetch content from a URL via HTTP GET.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 15)"},
                    "max_size": {"type": "integer", "description": "Max response bytes (default 500000)"},
                },
                "required": ["url"],
            },
        },
        # ── task management ──
        {
            "name": "task_create",
            "description": "Create a background task (runs a shell command asynchronously).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"},
                    "name": {"type": "string", "description": "Human-readable task name"},
                },
                "required": ["command"],
            },
        },
        {
            "name": "task_get",
            "description": "Get info about a background task by ID.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID"},
                },
                "required": ["task_id"],
            },
        },
        {
            "name": "task_list",
            "description": "List all background tasks.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "task_update",
            "description": "Update a background task's metadata (e.g. name).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID"},
                    "name": {"type": "string", "description": "New name"},
                },
                "required": ["task_id"],
            },
        },
        {
            "name": "task_stop",
            "description": "Stop a running background task.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID"},
                },
                "required": ["task_id"],
            },
        },
        {
            "name": "task_output",
            "description": "Get stdout/stderr output of a background task.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID"},
                },
                "required": ["task_id"],
            },
        },
        {
            "name": "task_delete",
            "description": "Delete a background task and its output.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID"},
                },
                "required": ["task_id"],
            },
        },
        # ── process management ──
        {
            "name": "process_spawn",
            "description": "Spawn a long-running subprocess.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"},
                    "name": {"type": "string", "description": "Human-readable process name"},
                },
                "required": ["command"],
            },
        },
        {
            "name": "process_kill",
            "description": "Kill a subprocess by PID.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pid": {"type": "integer", "description": "Process ID to kill"},
                },
                "required": ["pid"],
            },
        },
        {
            "name": "process_list",
            "description": "List all tracked subprocesses.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "process_status",
            "description": "Get status of a subprocess by PID.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pid": {"type": "integer", "description": "Process ID"},
                },
                "required": ["pid"],
            },
        },
        # ── system info ──
        {
            "name": "system_info",
            "description": "Get current system information (OS, CPU, memory, Python version, etc.).",
            "input_schema": {"type": "object", "properties": {}},
        },
        # ── execute skill ──
        {
            "name": "execute_skill",
            "description": "Load and return the content of a SKILL.md skill by name.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string", "description": "Name of the skill to execute"},
                },
                "required": ["skill_name"],
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

    # ── web_fetch ──

    def _tool_web_fetch(self, inp: dict) -> str:
        url = inp.get("url", "").strip()
        if not url:
            return "Error: Empty URL"
        timeout = inp.get("timeout", 15)
        max_size = inp.get("max_size", MAX_FETCH)

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ReActAgent/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read(max_size + 1)
                text = data.decode("utf-8", errors="replace")
                if len(data) > max_size:
                    text = text[:max_size] + f"\n... (truncated at {max_size} bytes)"
                return text
        except urllib.error.HTTPError as e:
            return f"Error: HTTP {e.code} — {e.reason}"
        except urllib.error.URLError as e:
            return f"Error: {e.reason}"
        except TimeoutError:
            return f"Error: Request timed out after {timeout}s"
        except Exception as e:
            return f"Error: {e}"

    # ── task_create ──

    def _tool_task_create(self, inp: dict) -> str:
        command = inp.get("command", "")
        name = inp.get("name", "")

        with self._lock:
            self._task_counter += 1
            task_id = f"task_{self._task_counter}"

        try:
            proc = subprocess.Popen(
                command, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

        with self._lock:
            self._tasks[task_id] = {
                "task_id": task_id,
                "name": name or command[:50],
                "command": command,
                "pid": proc.pid,
                "process": proc,
                "status": "running",
                "created_at": time.time(),
            }

        return json.dumps({"task_id": task_id, "pid": proc.pid})

    # ── task_get ──

    def _tool_task_get(self, inp: dict) -> str:
        task_id = inp.get("task_id", "")
        with self._lock:
            task = self._tasks.get(task_id)
        if not task:
            return f"Error: Task not found: {task_id}"

        self._refresh_task_status(task)
        return json.dumps({
            "task_id": task["task_id"],
            "name": task["name"],
            "command": task["command"],
            "pid": task["pid"],
            "status": task["status"],
            "created_at": task["created_at"],
        })

    # ── task_list ──

    def _tool_task_list(self, inp: dict) -> str:
        with self._lock:
            tasks = list(self._tasks.values())
        for t in tasks:
            self._refresh_task_status(t)
        return json.dumps({
            "tasks": [
                {
                    "task_id": t["task_id"],
                    "name": t["name"],
                    "status": t["status"],
                    "pid": t["pid"],
                }
                for t in tasks
            ]
        })

    # ── task_update ──

    def _tool_task_update(self, inp: dict) -> str:
        task_id = inp.get("task_id", "")
        with self._lock:
            task = self._tasks.get(task_id)
        if not task:
            return f"Error: Task not found: {task_id}"

        if "name" in inp:
            task["name"] = inp["name"]
        return json.dumps({"updated": True, "task_id": task_id})

    # ── task_stop ──

    def _tool_task_stop(self, inp: dict) -> str:
        task_id = inp.get("task_id", "")
        with self._lock:
            task = self._tasks.get(task_id)
        if not task:
            return f"Error: Task not found: {task_id}"

        proc: subprocess.Popen = task.get("process")
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        task["status"] = "stopped"
        return json.dumps({"stopped": True, "task_id": task_id})

    # ── task_output ──

    def _tool_task_output(self, inp: dict) -> str:
        task_id = inp.get("task_id", "")
        with self._lock:
            task = self._tasks.get(task_id)
        if not task:
            return f"Error: Task not found: {task_id}"

        proc: subprocess.Popen = task.get("process")
        if not proc or not proc.stdout:
            return ""

        self._refresh_task_status(task)
        try:
            # Read available output (non-blocking for completed tasks)
            if proc.poll() is not None:
                output = proc.stdout.read()
            else:
                # For running tasks, read what's available
                import select
                output = ""
                while select.select([proc.stdout], [], [], 0.1)[0]:
                    chunk = proc.stdout.readline()
                    if not chunk:
                        break
                    output += chunk
        except Exception:
            output = ""

        if len(output) > MAX_OUTPUT:
            output = output[:MAX_OUTPUT] + f"\n... (truncated)"
        return output

    # ── task_delete ──

    def _tool_task_delete(self, inp: dict) -> str:
        task_id = inp.get("task_id", "")
        with self._lock:
            task = self._tasks.get(task_id)
        if not task:
            return f"Error: Task not found: {task_id}"

        # Stop if running
        proc: subprocess.Popen = task.get("process")
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()

        with self._lock:
            del self._tasks[task_id]
        return json.dumps({"deleted": True, "task_id": task_id})

    # ── process_spawn ──

    def _tool_process_spawn(self, inp: dict) -> str:
        command = inp.get("command", "")
        name = inp.get("name", "")

        try:
            proc = subprocess.Popen(
                command, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, preexec_fn=os.setsid if os.name != "nt" else None,
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

        with self._lock:
            self._processes[proc.pid] = {
                "pid": proc.pid,
                "name": name or command[:50],
                "command": command,
                "process": proc,
                "started_at": time.time(),
            }
        return json.dumps({"pid": proc.pid, "name": name or command[:50]})

    # ── process_kill ──

    def _tool_process_kill(self, inp: dict) -> str:
        pid = inp.get("pid", 0)

        with self._lock:
            info = self._processes.get(pid)

        if info:
            proc: subprocess.Popen = info["process"]
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
            except Exception:
                pass
            return json.dumps({"killed": True, "pid": pid})

        # Try OS-level kill for untracked PIDs
        try:
            os.kill(pid, signal.SIGTERM)
            return json.dumps({"killed": True, "pid": pid})
        except ProcessLookupError:
            return f"Error: Process not found: {pid}"
        except PermissionError:
            return f"Error: Permission denied to kill PID {pid}"

    # ── process_list ──

    def _tool_process_list(self, inp: dict) -> str:
        with self._lock:
            procs = list(self._processes.values())

        result = []
        for info in procs:
            proc: subprocess.Popen = info["process"]
            status = "running" if proc.poll() is None else "terminated"
            result.append({
                "pid": info["pid"],
                "name": info["name"],
                "status": status,
                "started_at": info["started_at"],
            })
        return json.dumps({"processes": result})

    # ── process_status ──

    def _tool_process_status(self, inp: dict) -> str:
        pid = inp.get("pid", 0)
        with self._lock:
            info = self._processes.get(pid)

        if not info:
            return json.dumps({"pid": pid, "status": "not_found"})

        proc: subprocess.Popen = info["process"]
        rc = proc.poll()
        status = "running" if rc is None else "terminated"
        result = {
            "pid": pid,
            "name": info["name"],
            "status": status,
            "started_at": info["started_at"],
        }
        if rc is not None:
            result["return_code"] = rc
        return json.dumps(result)

    # ── system_info ──

    def _tool_system_info(self, inp: dict) -> str:
        info: dict[str, Any] = {
            "os": os.name,
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "cpu_count": os.cpu_count(),
            "cwd": os.getcwd(),
            "user": os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
            "hostname": platform.node(),
            "arch": platform.machine(),
        }

        # Memory info (platform-specific, best-effort)
        try:
            if sys.platform == "darwin":
                out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
                info["memory"] = f"{int(out) // (1024**3)} GB"
            elif sys.platform == "linux":
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal"):
                            kb = int(line.split()[1])
                            info["memory"] = f"{kb // (1024**2)} GB"
                            break
            else:
                info["memory"] = "unknown"
        except Exception:
            info["memory"] = "unknown"

        return json.dumps(info, ensure_ascii=False)

    # ── execute_skill ──

    def _tool_execute_skill(self, inp: dict) -> str:
        skill_name = inp.get("skill_name", "").strip()
        if not skill_name:
            return "Error: Empty skill name"

        skills_dir = Path(self._skills_dir)
        skill_path = skills_dir / skill_name / "SKILL.md"
        if not skill_path.exists():
            return f"Error: Skill not found: {skill_name}"

        try:
            content = skill_path.read_text(encoding="utf-8")
            # Strip YAML frontmatter — return body only
            import re as _re
            m = _re.match(r"^---\n.*?\n---\n?", content, _re.DOTALL)
            if m:
                body = content[m.end():]
            else:
                body = content
            return body.strip()
        except Exception as e:
            return f"Error: {e}"

    # ── helpers ──

    def _refresh_task_status(self, task: dict) -> None:
        proc: subprocess.Popen | None = task.get("process")
        if proc and proc.poll() is not None and task["status"] == "running":
            task["status"] = "completed"
