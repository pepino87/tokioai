"""
DomiAI ◆ MatiAI — Web Panel
Panel web local: chat con Tokio + agentes de tareas en paralelo.

Uso:
  python -m tokioai_cli.web_panel            # arranca en http://127.0.0.1:8777
  python -m tokioai_cli.web_panel --no-browser
  TOKIOAI_PANEL_PORT=9000 python -m tokioai_cli.web_panel

Desde el CLI: /panel  (y /panel de nuevo para detener)
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

# Importar interactive ejecuta _load_dotenv() (misma config del CLI) y trae ops.
from tokioai_cli.interactive import _load_dotenv  # noqa: F401

MAX_AGENTS = 12
MAX_LOG = 300


# ═══════════════════════════════════════════════════════
# Agentes
# ═══════════════════════════════════════════════════════

class Agent:
    """Un agente = persona + modelo + tarea, ejecutada en su propio hilo."""

    def __init__(self, name: str, persona: str, model: str, task: str, program: str = "auto"):
        self.id = uuid.uuid4().hex[:8]
        self.name = (name or "").strip() or f"Agente {self.id[:4]}"
        self.persona = (persona or "").strip()
        self.model_alias = (model or "default").strip() or "default"
        self.task = (task or "").strip()
        self.program = (program or "auto").strip() or "auto"
        self.status = "queued"  # queued | running | done | error | stopped
        self.result = ""
        self.error = ""
        self.log: List[Dict[str, Any]] = []
        self.created_ts = time.time()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

    def _append(self, kind: str, text: str):
        with self._lock:
            self.log.append({"t": time.time(), "kind": kind, "text": text})
            self.log = self.log[-MAX_LOG:]

    def _run(self):
        self.status = "running"
        self._append("notice", f"▶ Arrancando ({self.model_alias})…")
        try:
            from tokioai_cli.ops import TokioOps, resolve_model
            if self.model_alias == "default":
                ops = TokioOps()
            else:
                ops = TokioOps(model=resolve_model(self.model_alias))
            if self.persona:
                ops.custom_persona = self.persona
            # Programa: "auto" = autodetección por dominio; nombre explícito = reglas fijas
            if self.program != "auto":
                ops.current_program = self.program
                self._append("notice", f"Programa fijado: {self.program}")

            def on_text(t):
                t = (t or "").strip()
                if t:
                    self._append("notice", t)

            def on_tool_start(n, a):
                self._append("tool", f"🔧 {n}({json.dumps(a, default=str)[:140]})")

            def on_tool_end(n, r):
                self._append("tool", f"✔ {n} → {str(r or '')[:180]}")

            out = ops.chat(
                self.task,
                on_text=on_text,
                on_tool_start=on_tool_start,
                on_tool_end=on_tool_end,
            )
            self.result = (out or "").strip()
            if self._stop.is_set():
                self.status = "stopped"
                self._append("notice", "⏹ Detenido.")
            else:
                self.status = "done"
                self._append("notice", "✔ Tarea finalizada.")
        except Exception as e:
            self.status = "error"
            self.error = str(e)
            self._append("error", f"✖ Error: {e}")

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"agent-{self.id}")
        self._thread.start()

    def stop(self):
        self._stop.set()
        with self._lock:
            if self.status in ("queued", "running"):
                self.status = "stopped"
        self._append("notice", "⏹ Detención solicitada.")

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "name": self.name,
                "persona": self.persona,
                "model": self.model_alias,
                "program": self.program,
                "task": self.task,
                "status": self.status,
                "result": self.result,
                "error": self.error,
                "log": self.log[-80:],
                "created": self.created_ts,
            }


# ═══════════════════════════════════════════════════════
# Chat principal (Tokio)
# ═══════════════════════════════════════════════════════

class ChatSession:
    """Conversación con Tokio: un TokioOps persistente + historial para la UI."""

    def __init__(self):
        self.entries: List[Dict[str, Any]] = []
        self.status = "idle"  # idle | thinking
        self.live: Dict[str, Any] = {"text": "", "notes": []}
        self.model_label = "default"
        self.ops = None
        self._lock = threading.RLock()

    def _ensure_ops(self):
        if self.ops is None:
            from tokioai_cli.ops import TokioOps
            self.ops = TokioOps()
            self.model_label = getattr(self.ops, "model", "default")

    def send(self, message: str) -> bool:
        with self._lock:
            if self.status == "thinking":
                return False
            self.status = "thinking"
            self.live = {"text": "", "notes": []}
            self.entries.append({"kind": "user", "text": message, "ts": time.time()})

        def on_token(t):
            with self._lock:
                self.live["text"] += t

        def on_text(t):
            t = (t or "").strip()
            if t:
                with self._lock:
                    self.live["notes"].append(t)

        def on_tool_start(n, a):
            with self._lock:
                self.live["notes"].append(f"🔧 {n}({json.dumps(a, default=str)[:120]})")

        def on_tool_end(n, r):
            with self._lock:
                self.live["notes"].append(f"✔ {n} → {str(r or '')[:160]}")

        final = ""
        error = None
        try:
            self._ensure_ops()
            final = self.ops.chat(
                message,
                stream=True,
                on_token=on_token,
                on_text=on_text,
                on_tool_start=on_tool_start,
                on_tool_end=on_tool_end,
            )
        except Exception as e:
            error = str(e)

        with self._lock:
            if error:
                self.entries.append({"kind": "error", "text": f"Error: {error}", "ts": time.time()})
            else:
                text = (self.live["text"] or (final or "")).strip()
                self.entries.append({
                    "kind": "assistant",
                    "text": text,
                    "notes": self.live["notes"],
                    "ts": time.time(),
                })
            self.live = {"text": "", "notes": []}
            self.status = "idle"
        return True

    def clear(self):
        with self._lock:
            self.entries = []
            self.live = {"text": "", "notes": []}
            self.ops = None
            self.model_label = "default"

    def state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "status": self.status,
                "model": self.model_label,
                "entries": self.entries[-80:],
                "live": dict(self.live),
            }


# ═══════════════════════════════════════════════════════
# Servidor
# ═══════════════════════════════════════════════════════

class _Handler(BaseHTTPRequestHandler):
    panel: "PanelServer" = None
    server_version = "DomiMatiPanel/1.0"

    def log_message(self, *args):  # silenciar logs de acceso
        pass

    def _json(self, obj: Any, status: int = 200):
        body = json.dumps(obj, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Optional[Dict]:
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
            if length <= 0 or length > 1_000_000:
                return None
            return json.loads(self.rfile.read(length).decode())
        except Exception:
            return None

    def _serve_html(self):
        html_path = Path(__file__).parent / "web_panel.html"
        body = html_path.read_text(encoding="utf-8") if html_path.exists() else "Panel no encontrado"
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body.encode())))
        self.end_headers()
        self.wfile.write(body.encode())

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._serve_html()
        if self.path == "/api/state":
            return self._json(self.panel.state())
        if self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/api/chat":
            data = self._read_json() or {}
            msg = (data.get("message") or "").strip()
            if not msg:
                return self._json({"ok": False, "error": "Mensaje vacío"}, 400)
            ok = self.panel.chat_send(msg)
            if not ok:
                return self._json({"ok": False, "error": "Tokio aún está pensando…"}, 409)
            return self._json({"ok": True})
        if self.path == "/api/chat/clear":
            self.panel.chat.clear()
            return self._json({"ok": True})
        if self.path == "/api/agents":
            data = self._read_json() or {}
            agent, error = self.panel.agent_create(
                data.get("name", ""), data.get("persona", ""),
                data.get("model", "default"), data.get("task", ""),
                data.get("program", "auto"),
            )
            if error:
                return self._json({"ok": False, "error": error}, 400)
            return self._json({"ok": True, "agent": agent.to_dict()})
        if self.path.startswith("/api/agents/") and self.path.endswith("/stop"):
            aid = self.path[len("/api/agents/"):-len("/stop")]
            if self.panel.agent_stop(aid):
                return self._json({"ok": True})
            return self._json({"ok": False, "error": "Agente no encontrado"}, 404)
        self._json({"error": "not found"}, 404)

    def do_DELETE(self):
        if self.path.startswith("/api/agents/"):
            aid = self.path[len("/api/agents/"):]
            if self.panel.agent_remove(aid):
                return self._json({"ok": True})
            return self._json({"ok": False, "error": "Agente no encontrado"}, 404)
        self._json({"error": "not found"}, 404)


class PanelServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8777):
        self.host = host
        self.port = port
        self.chat = ChatSession()
        self.agents: Dict[str, Agent] = {}
        self._lock = threading.RLock()
        self._httpd = None
        self._running = False
        self.started_ts = time.time()
        self.url = f"http://{host}:{port}/"

    # ── Chat ──
    def chat_send(self, message: str) -> bool:
        with self._lock:
            if self.chat.status == "thinking":
                return False
            threading.Thread(target=self.chat.send, args=(message,), daemon=True,
                             name="chat-turn").start()
            return True

    # ── Agentes ──
    def agent_create(self, name, persona, model, task, program="auto") -> tuple:
        with self._lock:
            if not (task or "").strip():
                return None, "La tarea está vacía"
            active = [a for a in self.agents.values() if a.status in ("queued", "running")]
            if len(self.agents) >= MAX_AGENTS or len(active) >= MAX_AGENTS:
                return None, f"Máximo {MAX_AGENTS} agentes"
            agent = Agent(name, persona, model, task, program)
            self.agents[agent.id] = agent
        agent.start()
        return agent, None

    def agent_stop(self, aid: str) -> bool:
        with self._lock:
            agent = self.agents.get(aid)
        if not agent:
            return False
        agent.stop()
        return True

    def agent_remove(self, aid: str) -> bool:
        with self._lock:
            agent = self.agents.get(aid)
            if not agent:
                return False
            agent.stop()
            del self.agents[aid]
        return True

    # ── Estado global ──
    def state(self) -> Dict[str, Any]:
        with self._lock:
            agents = [a.to_dict() for a in self.agents.values()]
        try:
            from tokioai_cli.ops import list_aliases, _load_programs
            aliases = list_aliases()
            programs = [
                {"name": p["name"], "domains": p.get("domains", [])}
                for p in _load_programs()
            ]
        except Exception:
            aliases = {}
            programs = []
        return {
            "chat": self.chat.state(),
            "agents": agents,
            "aliases": aliases,
            "programs": programs,
            "server": {
                "uptime_s": int(time.time() - self.started_ts),
                "max_agents": MAX_AGENTS,
            },
        }

    # ── Ciclo de vida ──
    def serve(self):
        _Handler.panel = self
        try:
            self._httpd = ThreadingHTTPServer((self.host, self.port), _Handler)
        except OSError as e:
            print(f"[PANEL] ✖ No se pudo abrir {self.url} — {e}")
            self._running = False
            return
        self._httpd.daemon_threads = True
        self._running = True
        print(f"[PANEL] ▶ DomiAI ◆ MatiAI — {self.url}")
        try:
            self._httpd.serve_forever()
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            self._running = False

    def shutdown(self):
        if self._httpd:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:
                pass
        self._running = False


# ═══════════════════════════════════════════════════════
# Singleton (para /panel en el CLI)
# ═══════════════════════════════════════════════════════

PANEL_SERVER: Optional[PanelServer] = None


def start_panel(host: str = None, port: int = None) -> str:
    global PANEL_SERVER
    if PANEL_SERVER is not None and PANEL_SERVER._running:
        return PANEL_SERVER.url
    h = host or os.getenv("TOKIOAI_PANEL_HOST", "127.0.0.1")
    p = int(port or os.getenv("TOKIOAI_PANEL_PORT", "8777"))
    PANEL_SERVER = PanelServer(host=h, port=p)
    threading.Thread(target=PANEL_SERVER.serve, daemon=True, name="panel-server").start()
    time.sleep(0.3)
    if not PANEL_SERVER._running:
        PANEL_SERVER = None
        raise RuntimeError(f"No se pudo iniciar el panel en {h}:{p} (¿puerto en uso?)")
    return PANEL_SERVER.url


def stop_panel():
    global PANEL_SERVER
    if PANEL_SERVER is not None:
        PANEL_SERVER.shutdown()
        PANEL_SERVER = None


# ═══════════════════════════════════════════════════════
# Entrada principal
# ═══════════════════════════════════════════════════════

def main(argv: Optional[List[str]] = None):
    import sys
    args = list(argv) if argv is not None else sys.argv[1:]
    host = os.getenv("TOKIOAI_PANEL_HOST", "127.0.0.1")
    port = int(os.getenv("TOKIOAI_PANEL_PORT", "8777"))
    open_browser = "--no-browser" not in args
    if "--port" in args:
        port = int(args[args.index("--port") + 1])
    if "--host" in args:
        host = args[args.index("--host") + 1]

    print("\033[1m\033[91m  DomiAI\033[0m\033[1m\033[95m ◆ \033[0m\033[1m\033[94mMatiAI\033[0m \033[90m— Panel (chat + agentes en paralelo)\033[0m")
    try:
        url = start_panel(host=host, port=port)
    except RuntimeError as e:
        print(f"\033[91m  ✖ {e}\033[0m")
        sys.exit(1)
    print(f"  → {url}")
    print("  Ctrl+C para salir")
    if open_browser:
        def _open():
            try:
                webbrowser.open(url)
            except Exception:
                pass
        threading.Timer(0.8, _open).start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_panel()
        print("\n  ⏹ Panel detenido.")


if __name__ == "__main__":
    main()
