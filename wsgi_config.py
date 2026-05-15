#!/usr/bin/env python3
"""
WSGI entry point for Gunicorn/Koyeb — QTCL Server v6 + MCP 2025-06-18
=======================================================================
KEY: /health returns 200 in <100ms. Server loads in background.

FIXES v3.1:
  - /mcp POST during startup: waits up to 8s in small increments instead of
    returning an immediate 503 that drops the initialize handshake permanently.
    Claude.ai connector does NOT retry 503 on initialize — it marks the tool dead.
  - JSON-RPC error id field: was null; now mirrors request id when parseable,
    preserving JSON-RPC 2.0 spec (id must match request or be null only if
    request id was unparseable). Correct id prevents "orphaned response" drop.
  - /mcp GET (SSE stream): always wait for full load — streaming a 503 mid-SSE
    causes protocol desync.
  - Added /mcp/health as instant route (doesn't need full server).
  - MCP CORS headers aligned with mcp_flask_adapter (Last-Event-ID included).
"""

import logging, os, sys, time, threading, json
from io import BytesIO

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_HYP_DIR   = os.path.join(_REPO_ROOT, "hlwe")
if _HYP_DIR not in sys.path:
    sys.path.insert(0, _HYP_DIR)

_STARTUP = time.time()

# ── Instant health app — zero heavy imports ────────────────────────────────────
from flask import Flask as _FlaskHealth
_health_app = _FlaskHealth("__health__")

@_health_app.route("/health")
def _health_instant():
    return "", 200

@_health_app.route("/mcp/health")
def _mcp_health_instant():
    # FIX: /mcp/health must be fast so Claude.ai connector polling doesn't time out
    # during server startup. Return a minimal valid health response.
    elapsed = time.time() - _STARTUP
    ready   = _full_app is not None
    return _health_app.response_class(
        response=json.dumps({
            "status":   "ok" if ready else "starting",
            "server":   "qtcl-blockchain",
            "version":  "3.1.0",
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

def _load_server():
    global _full_app
    try:
        print("[WSGI] Loading full server (MCP 2025-06-18 + JSON-RPC 2.0)...", flush=True)
        from server import app as full_app
        _full_app = full_app
        print(f"[WSGI] ✅ Full server loaded at {time.time() - _STARTUP:.1f}s", flush=True)
        print("[WSGI] ✅ MCP endpoints: /mcp (POST/GET), /mcp/health", flush=True)
    except Exception as e:
        print(f"[WSGI] ❌ Server load failed: {e}", flush=True)
    finally:
        _load_done.set()

_thread = threading.Thread(target=_load_server, daemon=True)
_thread.start()

# ── CORS headers (aligned with mcp_flask_adapter) ─────────────────────────────
_MCP_CORS = [
    ("Access-Control-Allow-Origin",   "*"),
    ("Access-Control-Allow-Methods",  "GET, POST, OPTIONS, DELETE"),
    ("Access-Control-Allow-Headers",  "Content-Type, Accept, Mcp-Session-Id, Authorization, Last-Event-ID"),
    ("Access-Control-Expose-Headers", "Mcp-Session-Id"),
]

def _jsonrpc_503(start_response: any, req_id: any, msg: str) -> list:
    """Return a well-formed JSON-RPC 2.0 503 with correct id."""
    body = json.dumps({
        "jsonrpc": "2.0",
        "error":   {"code": -32000, "message": msg},
        "id":      req_id,  # FIX: must mirror request id, not always null
    }).encode()
    start_response("503 Service Unavailable", [
        ("Content-Type",  "application/json"),
        ("Retry-After",   "3"),
    ] + _MCP_CORS)
    return [body]

def _parse_request_id(environ: dict) -> any:
    """Try to read the JSON-RPC id from the request body without consuming it."""
    try:
        length  = int(environ.get("CONTENT_LENGTH") or 0)
        if length <= 0 or length > 65536:
            return None
        wsgi_input = environ["wsgi.input"]
        body_bytes  = wsgi_input.read(length)
        # Put body back so downstream handler can read it
        environ["wsgi.input"] = BytesIO(body_bytes)
        payload = json.loads(body_bytes.decode("utf-8", errors="replace"))
        return payload.get("id")  # may be int, str, or None
    except Exception:
        return None

# ── WSGI Application ──────────────────────────────────────────────────────────
def application(environ, start_response):
    path   = environ.get("PATH_INFO", "/")
    method = environ.get("REQUEST_METHOD", "GET")

    # /health — always instant
    if path in ("/health", "/health/"):
        return _health_app(environ, start_response)

    # /mcp/health — instant (handled by _health_app above)
    if path in ("/mcp/health", "/mcp/health/"):
        return _health_app(environ, start_response)

    # OPTIONS — always instant (CORS preflight never needs full server)
    if method == "OPTIONS":
        start_response("204 No Content", _MCP_CORS)
        return [b""]

    # /mcp/* — MCP 2025-06-18 Streamable HTTP
    if path == "/mcp" or path.startswith("/mcp/"):

        # GET (SSE stream) — wait up to 30s; streaming before ready causes desync
        if method == "GET":
            _load_done.wait(timeout=30)
            if _full_app:
                return _full_app(environ, start_response)
            start_response("503 Service Unavailable", [
                ("Content-Type", "text/plain"), ("Retry-After", "5"),
            ] + _MCP_CORS)
            return [b"MCP server starting, retry in 5s"]

        # POST — initialize handshake MUST succeed; Claude.ai won't retry a failed init.
        # FIX: wait up to 8s in 250ms steps before giving up with 503.
        if method == "POST":
            if _full_app:
                return _full_app(environ, start_response)
            # Try to read request id before body is consumed downstream
            req_id = _parse_request_id(environ)
            deadline = time.time() + 8.0
            while time.time() < deadline:
                if _full_app:
                    return _full_app(environ, start_response)
                time.sleep(0.25)
            # Still not ready — return proper 503 with mirrored id
            return _jsonrpc_503(start_response, req_id, "MCP server initializing, retry in 3s")

        # Other methods (DELETE for session termination)
        if _full_app:
            return _full_app(environ, start_response)
        start_response("503 Service Unavailable", [
            ("Content-Type", "application/json"), ("Retry-After", "3"),
        ] + _MCP_CORS)
        return [b'{"jsonrpc":"2.0","error":{"code":-32000,"message":"Server initializing"},"id":null}']

    # /rpc POST — never fake it; always process
    if method == "POST" and path in ("/rpc", "/rpc/"):
        if _full_app:
            return _full_app(environ, start_response)
        req_id = _parse_request_id(environ)
        start_response("503 Service Unavailable", [
            ("Content-Type", "application/json"), ("Retry-After", "5"),
        ])
        return [json.dumps({"jsonrpc": "2.0",
                            "error": {"code": -32000, "message": "Server initializing, retry in 5s"},
                            "id": req_id}).encode()]

    # /rpc GET — return 503 immediately if not ready
    if method == "GET" and (path == "/rpc" or path.startswith("/rpc/")):
        if _full_app:
            return _full_app(environ, start_response)
        start_response("503 Service Unavailable", [
            ("Content-Type", "application/json"), ("Retry-After", "5"),
        ])
        return [b'{"jsonrpc":"2.0","error":{"code":-32000,"message":"Server initializing"},"id":null}']

    # All other endpoints — wait up to 30s
    _load_done.wait(timeout=30)
    if _full_app:
        return _full_app(environ, start_response)
    start_response("503 Service Unavailable", [("Content-Type", "text/plain")])
    return [b"Server starting, retry in a few seconds..."]


app = application

print(f"[WSGI] ✅ WSGI ready at {time.time() - _STARTUP:.2f}s", flush=True)
print("[WSGI] Routes: /health=instant | /mcp/health=instant | /mcp=MCP-2025-06-18 | /rpc=JSON-RPC-2.0",
      flush=True)
print("[WSGI] FIX: /mcp POST waits up to 8s during startup (initialize handshake protection)",
      flush=True)
