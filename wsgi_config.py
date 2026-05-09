#!/usr/bin/env python3
"""
WSGI entry point for Gunicorn/Koyeb — QTCL Server v6 + MCP 2025-06-18
=======================================================================

KEY: /health returns 200 in <100ms. Server loads in background.
All other endpoints wait for server to load (max 30s).

MCP endpoints (/mcp/*) are fully integrated into server.py as pure
Flask routes — no proxy, no ASGI bridge, no port 8001.
"""

import logging
import os
import sys
import time
import threading

logger = logging.getLogger(__name__)

# Add hlwe to path
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_HYP_DIR = os.path.join(_REPO_ROOT, "hlwe")
if _HYP_DIR not in sys.path:
    sys.path.insert(0, _HYP_DIR)

_STARTUP = time.time()

# ═══ INSTANT HEALTH APP - no heavy imports ═══
from flask import Flask as _FlaskHealth

_health_app = _FlaskHealth("__health__")


@_health_app.route("/health")
def health():
    return "", 200


print(f"[WSGI] ✅ /health ready at {time.time() - _STARTUP:.2f}s", flush=True)

# ═══ LOAD SERVER IN BACKGROUND ═══
_full_app = None
_load_done = threading.Event()


def _load_server():
    global _full_app
    try:
        print("[WSGI] Loading full server module (MCP 2025-06-18 + JSON-RPC 2.0)...", flush=True)
        from server import app as full_app
        _full_app = full_app
        print(f"[WSGI] ✅ Full server loaded at {time.time() - _STARTUP:.1f}s", flush=True)
        print("[WSGI] ✅ MCP endpoints: /mcp (POST/GET), /mcp/sse, /mcp/health", flush=True)
    except Exception as e:
        print(f"[WSGI] ❌ Server load failed: {e}", flush=True)
    finally:
        _load_done.set()


_thread = threading.Thread(target=_load_server, daemon=True)
_thread.start()


# ═══ WSGI APP ═══
def application(environ, start_response):
    path   = environ.get("PATH_INFO", "/")
    method = environ.get("REQUEST_METHOD", "GET")

    # ── /health — always instant ──────────────────────────────────────────────
    if path in ("/health", "/health/"):
        return _health_app(environ, start_response)

    # ── /mcp/* — MCP 2025-06-18 Streamable HTTP ───────────────────────────────
    # These are pure Flask routes in server.py — route immediately when ready.
    # OPTIONS (CORS preflight) gets instant 204 even before full load.
    if path == "/mcp" or path.startswith("/mcp/"):
        if method == "OPTIONS":
            # CORS preflight never needs full server — return 204 immediately
            headers = [
                ("Content-Type", "text/plain"),
                ("Access-Control-Allow-Origin", "*"),
                ("Access-Control-Allow-Methods", "GET, POST, OPTIONS, DELETE"),
                ("Access-Control-Allow-Headers", "Content-Type, Accept, Mcp-Session-Id, Authorization, Last-Event-ID"),
                ("Access-Control-Expose-Headers", "Mcp-Session-Id"),
            ]
            start_response("204 No Content", headers)
            return [b""]

        if _full_app:
            return _full_app(environ, start_response)

        # Not yet ready — return proper JSON-RPC/MCP error
        start_response("503 Service Unavailable", [
            ("Content-Type", "application/json"),
            ("Retry-After", "3"),
            ("Access-Control-Allow-Origin", "*"),
        ])
        return [b'{"jsonrpc":"2.0","error":{"code":-32000,"message":"MCP server initializing, retry in 3s"},"id":null}']

    # ── /rpc POST — never fake it, always process ─────────────────────────────
    if method == "POST" and path in ("/rpc", "/rpc/"):
        if _full_app:
            return _full_app(environ, start_response)
        start_response("503 Service Unavailable", [
            ("Content-Type", "application/json"),
            ("Retry-After", "5"),
        ])
        return [b'{"jsonrpc":"2.0","error":{"code":-32000,"message":"Server initializing, retry in 5s"},"id":null}']

    # ── /rpc GET — return 503 immediately if not ready ────────────────────────
    if method == "GET" and (path == "/rpc" or path.startswith("/rpc/")):
        if _full_app:
            return _full_app(environ, start_response)
        start_response("503 Service Unavailable", [
            ("Content-Type", "application/json"),
            ("Retry-After", "5"),
        ])
        return [b'{"jsonrpc":"2.0","error":{"code":-32000,"message":"Server initializing, retry in 5s"},"id":null}']

    # ── All other endpoints — wait up to 30s for full load ───────────────────
    _load_done.wait(timeout=30)

    if _full_app:
        return _full_app(environ, start_response)

    start_response("503 Service Unavailable", [("Content-Type", "text/plain")])
    return [b"Server starting, retry in a few seconds..."]


app = application

print(f"[WSGI] ✅ WSGI ready at {time.time() - _STARTUP:.2f}s", flush=True)
print("[WSGI] Routes: /health=instant | /mcp=MCP-2025-06-18 | /rpc=JSON-RPC-2.0", flush=True)
