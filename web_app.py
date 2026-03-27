"""Minimal web app for E2E testing — uses Python built-in http.server only."""

import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

from episode_curator import create_agent, EpisodeStore

# Global state
_agent = None
_storage_dir = None
_history = []


def get_agent():
    global _agent, _storage_dir
    if _agent is None:
        _storage_dir = os.environ.get("EPISODE_STORE_DIR", "/tmp/episode_store_e2e")
        _agent = create_agent(
            worker_model="claude-haiku-4-5-20251001",
            curator_model="claude-haiku-4-5-20251001",
            max_iterations=5,
            storage_dir=_storage_dir,
        )
    return _agent


HTML_PAGE = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Episode Curator ReAct Agent</title></head>
<body>
  <h1>Episode Curator ReAct Agent</h1>
  <div id="chat"></div>
  <input type="text" id="input" placeholder="Ask a question..." style="width:60%">
  <button id="send" onclick="sendMessage()">Send</button>
  <div id="response"></div>
  <script>
    async function sendMessage() {
      const input = document.getElementById('input');
      const msg = input.value.trim();
      if (!msg) return;
      document.getElementById('response').textContent = 'Thinking...';
      input.value = '';
      try {
        const res = await fetch('/chat', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({message: msg})
        });
        const data = await res.json();
        document.getElementById('response').textContent = data.response || data.error;
      } catch(e) {
        document.getElementById('response').textContent = 'Error: ' + e.message;
      }
    }
    document.getElementById('input').addEventListener('keypress', function(e) {
      if (e.key === 'Enter') sendMessage();
    });
  </script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, "text/html", HTML_PAGE)
        elif path == "/episodes":
            store = EpisodeStore(_storage_dir or "/tmp/episode_store_e2e")
            self._send_json(200, {"episodes": store._index})
        elif path == "/facts":
            store = EpisodeStore(_storage_dir or "/tmp/episode_store_e2e")
            self._send_json(200, {"facts": store.get_facts()})
        else:
            self._send(404, "text/plain", "Not Found")

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/chat":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            message = body.get("message", "")
            if not message:
                self._send_json(400, {"error": "No message provided"})
                return
            try:
                global _history
                agent = get_agent()
                answer = agent.run(message, list(_history))
                _history.append({"role": "user", "content": message})
                _history.append({"role": "assistant", "content": answer})
                self._send_json(200, {"response": answer})
            except Exception as e:
                self._send_json(500, {"error": str(e)})
        else:
            self._send(404, "text/plain", "Not Found")

    def _send(self, code, content_type, body):
        self.send_response(code)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body.encode("utf-8") if isinstance(body, str) else body)

    def _send_json(self, code, data):
        self._send(code, "application/json", json.dumps(data, ensure_ascii=False))

    def log_message(self, format, *args):
        pass  # Suppress request logging


def run_server(port=8765):
    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"Server running on http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    run_server(port)
