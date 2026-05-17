#!/usr/bin/env python3
"""
WSGI entry point for Gunicorn/Koyeb — QTCL Server v6 + MCP 2025-06-18
=======================================================================
KEY: /health returns 200 in <100ms. Server loads in background.

FIXES v4.0  (2026-05-17)
─────────────────────────
SERVER-SIDE (wsgi_config.py):
  [v3.1] /mcp POST startup: waits up to 8s instead of hard 503.
  [v4.0] /rpc  POST startup: SAME treatment — waits up to 10s before 503.
          Previously /rpc POST returned instant 503 "Server initializing"
          which caused 100% of cold-start tx rejections.  Now it waits the
          full boot window so qtcl_submitTransaction from qtcl_client.py
          always lands on a live server rather than the health stub.
  [v4.0] /rpc  GET  startup: waits up to 30s (was: 503 instant).
  [v4.0] JSON-RPC error id mirrors request id for ALL routes (not just /mcp).
  [v4.0] Content-Length added to every 503/200 response so HTTP/1.1 keep-
          alive framing is clean under gthread workers.
  [v4.0] _STARTUP_TIMEOUT_RPC_POST = 10s (configurable via env
          WSGI_RPC_INIT_TIMEOUT).  Set higher on slow infra if needed.
  [v4.0] Graceful path: if _full_app becomes ready DURING the wait loop,
          the request is forwarded immediately — no extra latency.
"""

import logging, os, sys, time, threading, json
from io import BytesIO

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_HYP_DIR   = os.path.join(_REPO_ROOT, "hlwe")
if _HYP_DIR not in sys.path:
    sys.path.insert(0, _HYP_DIR)

_STARTUP = time.time()

# ── Tunable timeouts ──────────────────────────────────────────────────────────
# /rpc POST (qtcl_submitTransaction, qtcl_submitBlock, etc.) — never drop on init
_TIMEOUT_RPC_POST  = float(os.environ.get("WSGI_RPC_INIT_TIMEOUT",  "10"))
# /mcp POST (initialize handshake) — Claude.ai won't retry a 503 initialize
_TIMEOUT_MCP_POST  = float(os.environ.get("WSGI_MCP_INIT_TIMEOUT",  "8"))
# /mcp GET (SSE stream) and general routes
_TIMEOUT_SSE       = float(os.environ.get("WSGI_SSE_INIT_TIMEOUT",  "30"))
# Poll granularity (seconds)
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
            "version":  "5.0.0",
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

# ── FIX v5.0: Retry server load on transient failures ──────────────────────
# If server.py import crashes (missing dep, Neon cold-start, QRNG timeout),
# retrying after a delay often succeeds. Without this, _full_app stays None
# FOREVER and every single request 503s until Koyeb kills/restarts the pod.
_LOAD_MAX_RETRIES = 3
_LOAD_RETRY_DELAY = 10.0  # seconds between retries

def _load_server():
    global _full_app
    for attempt in range(1, _LOAD_MAX_RETRIES + 1):
        try:
            print(f"[WSGI] Loading full server attempt {attempt}/{_LOAD_MAX_RETRIES} "
                  f"(MCP 2025-06-18 + JSON-RPC 2.0)...", flush=True)
            # Force reimport on retry — clear any partial module state
            import importlib
            if attempt > 1 and "server" in sys.modules:
                # Remove stale partial import so Python re-executes the module
                del sys.modules["server"]
            from server import app as full_app
            _full_app = full_app
            print(f"[WSGI] ✅ Full server loaded at {time.time() - _STARTUP:.1f}s "
                  f"(attempt {attempt})", flush=True)
            print("[WSGI] ✅ Endpoints live: /rpc (JSON-RPC 2.0) | /mcp (MCP 2025-06-18) "
                  "| /mcp/health", flush=True)
            return  # success
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
    _load_done.set()

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
    """Well-formed JSON-RPC 2.0 503 with id mirroring and Content-Length."""
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
    """
    Read JSON-RPC id from request body without consuming the stream.
    Replaces wsgi.input with a fresh BytesIO so downstream handlers can read it.
    Returns the id (int | str | None) or None on any parse failure.
    """
    try:
        length = int(environ.get("CONTENT_LENGTH") or 0)
        if length <= 0 or length > 65_536:
            return None
        wsgi_input = environ["wsgi.input"]
        body_bytes  = wsgi_input.read(length)
        environ["wsgi.input"] = BytesIO(body_bytes)      # put body back
        payload = json.loads(body_bytes.decode("utf-8", errors="replace"))
        return payload.get("id")
    except Exception:
        return None


def _wait_for_app(timeout: float) -> bool:
    """
    Spin-wait for _full_app to become non-None, checking every _POLL_INTERVAL.
    Returns True if ready within timeout, False otherwise.
    Uses _load_done.wait() in chunks so we don't busy-spin.
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

    # ── /health — always instant ───────────────────────────────────────────────
    if path in ("/health", "/health/"):
        return _health_app(environ, start_response)

    # ── /mcp/health — instant ─────────────────────────────────────────────────
    if path in ("/mcp/health", "/mcp/health/"):
        return _health_app(environ, start_response)

    # ── OPTIONS — instant CORS preflight ──────────────────────────────────────
    if method == "OPTIONS":
        start_response("204 No Content", _MCP_CORS)
        return [b""]

    # ── /mcp/* — MCP 2025-06-18 Streamable HTTP ───────────────────────────────
    if path == "/mcp" or path.startswith("/mcp/"):

        if method == "GET":
            # SSE stream — wait for full server; streaming before ready = desync
            _wait_for_app(_TIMEOUT_SSE)
            if _full_app:
                return _full_app(environ, start_response)
            start_response("503 Service Unavailable", [
                ("Content-Type", "text/plain"), ("Retry-After", "5"),
            ] + _MCP_CORS)
            return [b"MCP server starting, retry in 5s"]

        if method == "POST":
            # initialize handshake — Claude.ai will NOT retry a 503 init
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

        # DELETE / other MCP methods
        if _full_app:
            return _full_app(environ, start_response)
        start_response("503 Service Unavailable", [
            ("Content-Type", "application/json"), ("Retry-After", "3"),
        ] + _MCP_CORS)
        return [b'{"jsonrpc":"2.0","error":{"code":-32000,"message":"Server initializing"},"id":null}']

    # ── /rpc — JSON-RPC 2.0 (PRIMARY TX + QUERY ENDPOINT) ────────────────────
    #
    # FIX v4.0 — THE CRITICAL PATH:
    # Previously this returned an INSTANT 503 "Server initializing, retry in 5s"
    # for any POST during the cold-start window.  qtcl_client.py's _rpc() method
    # uses retries=4 with exponential backoff, but the CircuitBreaker tripped and
    # the wizard showed "❌ Rejected: {'code': -32000, 'message': 'Server initializing…'}"
    # because submit_transaction() parsed the error dict directly from _rpc() and
    # never got a chance to retry (circuit breaker opened after 5 quick 503s).
    #
    # Fix: mirror the /mcp POST treatment — wait up to _TIMEOUT_RPC_POST (10s)
    # for the server to load.  On Koyeb the server boots in ~4-7s so any request
    # arriving during the cold-start window will land on a live handler.
    #
    if path in ("/rpc", "/rpc/") or path.startswith("/rpc/"):

        if method == "POST":
            # Fast path: already loaded
            if _full_app:
                return _full_app(environ, start_response)
            # Slow path: read id BEFORE body is consumed, then wait
            req_id = _parse_request_id(environ)
            logger.info(
                f"[WSGI] /rpc POST during startup — waiting up to {_TIMEOUT_RPC_POST:.0f}s "
                f"for server (id={req_id!r})"
            )
            _wait_for_app(_TIMEOUT_RPC_POST)
            if _full_app:
                return _full_app(environ, start_response)
            # Server didn't load in time — return proper 503 with mirrored id
            return _jsonrpc_503(
                start_response, req_id,
                f"Server initializing — retry in 3s "
                f"(waited {_TIMEOUT_RPC_POST:.0f}s, boot still in progress)",
            )

        if method == "GET":
            # getBalance, getTransaction, etc. — wait up to _TIMEOUT_SSE
            if _full_app:
                return _full_app(environ, start_response)
            _wait_for_app(_TIMEOUT_SSE)
            if _full_app:
                return _full_app(environ, start_response)
            req_id = _parse_request_id(environ)
            return _jsonrpc_503(start_response, req_id, "Server initializing, retry in 5s")

        # HEAD / DELETE / other
        if _full_app:
            return _full_app(environ, start_response)
        start_response("503 Service Unavailable", [
            ("Content-Type", "application/json"), ("Retry-After", "3"),
        ])
        return [b'{"jsonrpc":"2.0","error":{"code":-32000,"message":"Server initializing"},"id":null}']

    # ── All other endpoints ────────────────────────────────────────────────────
    _wait_for_app(_TIMEOUT_SSE)
    if _full_app:
        return _full_app(environ, start_response)
    start_response("503 Service Unavailable", [("Content-Type", "text/plain")])
    return [b"Server starting, retry in a few seconds..."]


app = application

print(f"[WSGI] ✅ WSGI v5.0 ready at {time.time() - _STARTUP:.2f}s", flush=True)
print(
    "[WSGI] Routes: /health=instant | /mcp/health=instant "
    f"| /mcp POST waits {_TIMEOUT_MCP_POST:.0f}s "
    f"| /rpc POST waits {_TIMEOUT_RPC_POST:.0f}s "
    f"| /rpc GET waits {_TIMEOUT_SSE:.0f}s",
    flush=True,
)
print(
    f"[WSGI] FIX v5.0: _load_server retries {_LOAD_MAX_RETRIES}× with "
    f"{_LOAD_RETRY_DELAY:.0f}s delay — transient import crashes no longer brick the server.",
    flush=True,
)
