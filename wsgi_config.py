#!/usr/bin/env python3
"""
WSGI entry point for Gunicorn/Koyeb.

KEY: /health returns 200 in <100ms. Server loads in background.
All other endpoints wait for server to load (max 30s).
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
from flask import Flask

_health_app = Flask("__health__")


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
        print("[WSGI] Loading full server module...", flush=True)
        from server import app as full_app

        _full_app = full_app
        print(
            f"[WSGI] ✅ Full server loaded at {time.time() - _STARTUP:.1f}s", flush=True
        )
    except Exception as e:
        print(f"[WSGI] ❌ Server load failed: {e}", flush=True)
    finally:
        _load_done.set()


# Start loading immediately but don't block
_thread = threading.Thread(target=_load_server, daemon=True)
_thread.start()


# ═══ WSGI APP ═══
def application(environ, start_response):
    path = environ.get("PATH_INFO", "/")
    method = environ.get("REQUEST_METHOD", "GET")

    # Health check always instant
    if path in ("/health", "/health/"):
        return _health_app(environ, start_response)

    # POST to /rpc - ACTUALLY PROCESS IT (was returning fake health for Checkly - breaking block submission!)
    if method == "POST" and path == "/rpc":
        # Pass through to real server to process the RPC call
        logger.warning(f"[WSGI] POST /rpc - forwarding to full app for processing")
        if _full_app:
            return _full_app(environ, start_response)

    # GET /rpc - if server not ready, return 503 immediately (don't block client)
    if method == "GET" and (path == "/rpc" or path.startswith("/rpc/")):
        if _full_app:
            return _full_app(environ, start_response)
        # Server not ready - tell client to retry quickly
        start_response(
            "503 Service Unavailable",
            [("Content-Type", "application/json"), ("Retry-After", "5")],
        )
        return [
            b'{"jsonrpc":"2.0","error":{"code":-32000,"message":"Server initializing, retry in 5s"},"id":null}'
        ]

    # Standard timeout for other endpoints
    _load_done.wait(timeout=30)

    if _full_app:
        return _full_app(environ, start_response)

    # Not ready
    start_response("503 Service Unavailable", [("Content-Type", "text/plain")])
    return [b"Server starting, retry in a few seconds..."]


app = application

print(f"[WSGI] ✅ WSGI ready at {time.time() - _STARTUP:.2f}s", flush=True)
print("[WSGI] /health instant, /rpc waits for full load", flush=True)
