#!/usr/bin/env python3
"""
WSGI entry point for Gunicorn/Koyeb — QTCL Server v6 + MCP 2025-06-18
=======================================================================
KEY: /health returns 200 in <100ms. Server loads in background.

FIXES v5.1  (2026-05-17)
─────────────────────────
  [v5.1] BUG FIX: _load_done.set() was never called on success path in
          _load_server(). The Event was only set after all retries failed,
          meaning _wait_for_app() always busy-polled via timeout instead of
          waking immediately when _full_app became ready. Every request that
          arrived during cold-start waited the FULL timeout window (8–30s)
          even if the server loaded in 4s. Fix: call _load_done.set()
          immediately after _full_app = full_app in the success path.

FIXES v5.0  (2026-05-17)
─────────────────────────
  [v5.0] _load_server retries 3× with 10s delay on import failure —
          transient Neon/QRNG cold-start crashes no longer brick the server.
  [v4.0] /rpc POST startup: waits up to 10s instead of hard 503.
  [v4.0] /rpc GET  startup: waits up to 30s.
  [v4.0] JSON-RPC error id mirrors request id for ALL routes.
  [v4.0] Content-Length added to every 503/200 response.
  [v3.1] /mcp POST startup: waits up to 8s instead of hard 503.
"""

import logging, os, sys, time, threading, threading as _flask_threading, json
from io import BytesIO

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_HYP_DIR   = os.path.join(_REPO_ROOT, "hlwe")
if _HYP_DIR not in sys.path:
    sys.path.insert(0, _HYP_DIR)

_STARTUP = time.time()

# ── Tunable timeouts ──────────────────────────────────────────────────────────
# RPC_POST: raised to 60s — Neon DB cold-start + server.py import on Koyeb free
# tier regularly takes 20-45s. 10s was too short and caused the perpetual 503 loop.
_TIMEOUT_RPC_POST  = float(os.environ.get("WSGI_RPC_INIT_TIMEOUT",  "60"))
_TIMEOUT_MCP_POST  = float(os.environ.get("WSGI_MCP_INIT_TIMEOUT",  "30"))
_TIMEOUT_SSE       = float(os.environ.get("WSGI_SSE_INIT_TIMEOUT",  "60"))
_POLL_INTERVAL     = 0.20

# ── Instant health app — zero heavy imports ────────────────────────────────────
from flask import Flask as _FlaskHealth
_health_app = _FlaskHealth("__health__")

@_health_app.route("/health")
def _health_instant():
    return "", 200

@_health_app.route("/mcp/health")
def _mcp_health_instant():
    elapsed = time.time() - _STARTUP
    ready   = _full_app is not None
    return _health_app.response_class(
        response=json.dumps({
            "status":   "ok" if ready else "starting",
            "server":   "qtcl-blockchain",
            "version":  "6.0.0",
            "protocol": "2025-06-18",
            "uptime_s": round(elapsed, 1),
            "ready":    ready,
        }),
        status=200,
        mimetype="application/json",
    )

print(f"[WSGI] ✅ /health + /mcp/health ready at {time.time() - _STARTUP:.2f}s", flush=True)

# ── Load full server in background ────────────────────────────────────────────
_full_app  = None
_load_done = threading.Event()

_LOAD_MAX_RETRIES = 3
_LOAD_RETRY_DELAY = 10.0

# Phase-1 timeout: how long to wait for server.py to execute `app = Flask(__name__)`
# and fire _QTCL_APP_EVENT.  On Koyeb this takes 2-8s (imports up to that point).
_TIMEOUT_PHASE1    = float(os.environ.get("WSGI_PHASE1_TIMEOUT", "30"))
# Phase-2: full module completion (oracle/lattice/DB background threads spin up).
# _full_app is ALREADY set after phase-1; this just logs when module finishes.
_TIMEOUT_PHASE2    = float(os.environ.get("WSGI_PHASE2_TIMEOUT", "120"))

def _load_server():
    """TWO-PHASE SERVER LOADER (v6.0)
    ─────────────────────────────────
    Phase 1: Start importing server.py in a thread. Poll sys.modules['server']
             for _QTCL_APP_EVENT — fires immediately after `app = Flask(__name__)`
             at line ~654 of server.py. Set _full_app as soon as that event fires.
             /rpc routes start serving within 5-8s of cold start.

    Phase 2: Full module import completes in background (oracle/lattice/DB init).
             No action needed — _full_app is already live. Log completion only.

    ROOT CAUSE OF OLD BUG: `from server import app` blocks the ENTIRE importing
    thread until ALL ~12,000 lines of server.py finish executing (45s on Koyeb).
    _full_app stayed None for that entire window → every /rpc POST 503'd.
    """
    global _full_app

    for attempt in range(1, _LOAD_MAX_RETRIES + 1):
        try:
            print(f"[WSGI] Phase-1 server load attempt {attempt}/{_LOAD_MAX_RETRIES} "
                  f"(MCP 2025-06-18 + JSON-RPC 2.0) — waiting for Flask app sentinel...",
                  flush=True)

            # ── Phase 1: import server in a child thread; poll for _QTCL_APP_EVENT ──
            import_done   = _flask_threading.Event()
            import_error  = [None]

            def _do_import():
                try:
                    import importlib
                    if attempt > 1 and "server" in sys.modules:
                        del sys.modules["server"]
                    import server as _srv_mod  # noqa: F401  (side-effects matter)
                except Exception as _ie:
                    import_error[0] = _ie
                finally:
                    import_done.set()

            _import_thread = _flask_threading.Thread(target=_do_import, daemon=True,
                                                     name=f"ServerImport-{attempt}")
            _import_thread.start()

            # ── Poll sys.modules['server'] for _QTCL_APP_EVENT + app ──────────────
            _phase1_deadline = time.monotonic() + _TIMEOUT_PHASE1
            _app_grabbed = False
            while time.monotonic() < _phase1_deadline:
                _srv = sys.modules.get("server")
                if _srv is not None:
                    _evt = getattr(_srv, "_QTCL_APP_EVENT", None)
                    _candidate_app = getattr(_srv, "app", None)
                    if _candidate_app is not None:
                        # Event may not exist in old server.py — accept app presence alone
                        if _evt is None or _evt.is_set():
                            _full_app = _candidate_app
                            _load_done.set()
                            _app_grabbed = True
                            _elapsed = time.time() - _STARTUP
                            print(f"[WSGI] ✅ Phase-1 complete: Flask app live at {_elapsed:.1f}s "
                                  f"(attempt {attempt}) — /rpc now serving", flush=True)
                            break
                # Check if import thread crashed before app was set
                if import_done.is_set() and import_error[0] is not None:
                    raise import_error[0]
                time.sleep(0.10)

            if not _app_grabbed:
                # Phase-1 timeout: check if import finished with error
                if import_error[0] is not None:
                    raise import_error[0]
                # Import is still running but app hasn't appeared — wait for full import
                # then try to grab app from module
                import_done.wait(timeout=_TIMEOUT_PHASE2)
                _srv = sys.modules.get("server")
                if _srv is not None and getattr(_srv, "app", None) is not None:
                    _full_app = _srv.app
                    _load_done.set()
                    print(f"[WSGI] ✅ Phase-1 fallback: app grabbed after full import at "
                          f"{time.time() - _STARTUP:.1f}s", flush=True)
                    _app_grabbed = True
                elif import_error[0] is not None:
                    raise import_error[0]
                else:
                    raise RuntimeError(
                        f"server.py imported but `app` not found in sys.modules['server'] "
                        f"after {_TIMEOUT_PHASE1 + _TIMEOUT_PHASE2:.0f}s"
                    )

            # ── Phase 2: wait for full module completion (non-blocking for routes) ──
            def _await_phase2():
                import_done.wait(timeout=_TIMEOUT_PHASE2)
                if import_error[0]:
                    print(f"[WSGI] ⚠️  Phase-2 server.py background init error (app already live): "
                          f"{import_error[0]}", flush=True)
                else:
                    print(f"[WSGI] ✅ Phase-2 complete: server.py fully initialized at "
                          f"{time.time() - _STARTUP:.1f}s", flush=True)
                print("[WSGI] ✅ Endpoints live: /rpc (JSON-RPC 2.0) | /mcp (MCP 2025-06-18) "
                      "| /mcp/health", flush=True)

            _flask_threading.Thread(target=_await_phase2, daemon=True,
                                    name="Phase2Watcher").start()
            return  # success — _full_app is set, routes are live

        except Exception as e:
            import traceback
            print(f"[WSGI] ❌ Server load attempt {attempt}/{_LOAD_MAX_RETRIES} failed: {e}",
                  flush=True)
            traceback.print_exc()
            if attempt < _LOAD_MAX_RETRIES:
                print(f"[WSGI] ⏳ Retrying in {_LOAD_RETRY_DELAY:.0f}s...", flush=True)
                time.sleep(_LOAD_RETRY_DELAY)

    print("[WSGI] ❌❌❌ All server load attempts exhausted — server will not start. "
          "Check logs above for the root cause.", flush=True)
    _load_done.set()  # unblock waiters so they can 503 cleanly

_thread = threading.Thread(target=_load_server, daemon=True)
_thread.start()

# ── CORS headers ───────────────────────────────────────────────────────────────
_MCP_CORS = [
    ("Access-Control-Allow-Origin",   "*"),
    ("Access-Control-Allow-Methods",  "GET, POST, OPTIONS, DELETE"),
    ("Access-Control-Allow-Headers",  "Content-Type, Accept, Mcp-Session-Id, Authorization, Last-Event-ID"),
    ("Access-Control-Expose-Headers", "Mcp-Session-Id"),
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def _jsonrpc_503(start_response, req_id, msg: str, extra_headers: list = None) -> list:
    body = json.dumps({
        "jsonrpc": "2.0",
        "error":   {"code": -32000, "message": msg},
        "id":      req_id,
    }).encode()
    headers = [
        ("Content-Type",   "application/json"),
        ("Content-Length", str(len(body))),
        ("Retry-After",    "3"),
    ] + (extra_headers or [])
    start_response("503 Service Unavailable", headers)
    return [body]


def _parse_request_id(environ: dict):
    try:
        length = int(environ.get("CONTENT_LENGTH") or 0)
        if length <= 0 or length > 65_536:
            return None
        wsgi_input = environ["wsgi.input"]
        body_bytes  = wsgi_input.read(length)
        environ["wsgi.input"] = BytesIO(body_bytes)
        payload = json.loads(body_bytes.decode("utf-8", errors="replace"))
        return payload.get("id")
    except Exception:
        return None


def _wait_for_app(timeout: float) -> bool:
    """
    Wait for _full_app to become non-None.
    With the v5.1 fix, _load_done fires immediately on success so this
    returns in ~4-7s on Koyeb instead of spinning the full timeout window.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _full_app is not None:
            return True
        remaining = deadline - time.monotonic()
        _load_done.wait(timeout=min(_POLL_INTERVAL, remaining))
    return _full_app is not None


# ── WSGI Application ──────────────────────────────────────────────────────────
def application(environ, start_response):
    path   = environ.get("PATH_INFO", "/")
    method = environ.get("REQUEST_METHOD", "GET")

    if path in ("/health", "/health/"):
        return _health_app(environ, start_response)

    if path in ("/mcp/health", "/mcp/health/"):
        return _health_app(environ, start_response)

    if method == "OPTIONS":
        start_response("204 No Content", _MCP_CORS)
        return [b""]

    if path == "/mcp" or path.startswith("/mcp/"):

        if method == "GET":
            _wait_for_app(_TIMEOUT_SSE)
            if _full_app:
                return _full_app(environ, start_response)
            start_response("503 Service Unavailable", [
                ("Content-Type", "text/plain"), ("Retry-After", "5"),
            ] + _MCP_CORS)
            return [b"MCP server starting, retry in 5s"]

        if method == "POST":
            if _full_app:
                return _full_app(environ, start_response)
            req_id = _parse_request_id(environ)
            _wait_for_app(_TIMEOUT_MCP_POST)
            if _full_app:
                return _full_app(environ, start_response)
            return _jsonrpc_503(
                start_response, req_id,
                "MCP server initializing, retry in 3s",
                _MCP_CORS,
            )

        if _full_app:
            return _full_app(environ, start_response)
        start_response("503 Service Unavailable", [
            ("Content-Type", "application/json"), ("Retry-After", "3"),
        ] + _MCP_CORS)
        return [b'{"jsonrpc":"2.0","error":{"code":-32000,"message":"Server initializing"},"id":null}']

    if path in ("/rpc", "/rpc/") or path.startswith("/rpc/"):

        if method == "POST":
            if _full_app:
                return _full_app(environ, start_response)
            req_id = _parse_request_id(environ)
            logger.info(
                f"[WSGI] /rpc POST during startup — waiting up to {_TIMEOUT_RPC_POST:.0f}s "
                f"for server (id={req_id!r})"
            )
            _wait_for_app(_TIMEOUT_RPC_POST)
            if _full_app:
                return _full_app(environ, start_response)
            return _jsonrpc_503(
                start_response, req_id,
                f"Server initializing — retry in 3s "
                f"(waited {_TIMEOUT_RPC_POST:.0f}s, boot still in progress)",
            )

        if method == "GET":
            if _full_app:
                return _full_app(environ, start_response)
            _wait_for_app(_TIMEOUT_RPC_POST)
            if _full_app:
                return _full_app(environ, start_response)
            req_id = _parse_request_id(environ)
            return _jsonrpc_503(start_response, req_id, "Server initializing, retry in 3s")

        if _full_app:
            return _full_app(environ, start_response)
        start_response("503 Service Unavailable", [
            ("Content-Type", "application/json"), ("Retry-After", "3"),
        ])
        return [b'{"jsonrpc":"2.0","error":{"code":-32000,"message":"Server initializing"},"id":null}']

    _wait_for_app(_TIMEOUT_SSE)
    if _full_app:
        return _full_app(environ, start_response)
    start_response("503 Service Unavailable", [("Content-Type", "text/plain")])
    return [b"Server starting, retry in a few seconds..."]


app = application

print(f"[WSGI] ✅ WSGI v6.0 ready at {time.time() - _STARTUP:.2f}s", flush=True)
print(
    "[WSGI] Routes: /health=instant | /mcp/health=instant "
    f"| /mcp POST waits {_TIMEOUT_MCP_POST:.0f}s "
    f"| /rpc POST waits {_TIMEOUT_RPC_POST:.0f}s "
    f"| /rpc GET waits {_TIMEOUT_SSE:.0f}s",
    flush=True,
)
print(
    f"[WSGI] FIX v6.0: TWO-PHASE LOAD — _full_app set immediately when Flask app "
    f"is created (phase-1, ~5s), not after full server.py init (phase-2, ~45s). "
    f"Phase-1 timeout={_TIMEOUT_PHASE1:.0f}s via WSGI_PHASE1_TIMEOUT env var.",
    flush=True,
)
print(
    f"[WSGI] FIX v5.1: _load_done.set() called on SUCCESS — waiters wake immediately.",
    flush=True,
)
print(
    f"[WSGI] FIX v5.0: _load_server retries {_LOAD_MAX_RETRIES}× with "
    f"{_LOAD_RETRY_DELAY:.0f}s delay — transient import crashes no longer brick the server.",
    flush=True,
)
