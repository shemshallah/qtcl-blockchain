#!/usr/bin/env python3
"""
WSGI entry point for Gunicorn/Koyeb.

KEY: /health returns 200 in <100ms. Server loads in background.
All other endpoints wait for server to load (max 30s).
"""

import os
import sys
import time
import threading

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

    # Health check always instant
    if path in ("/health", "/health/"):
        return _health_app(environ, start_response)

    # Wait for full app (with timeout)
    _load_done.wait(timeout=30)

    if _full_app:
        return _full_app(environ, start_response)

    # Not ready - log details to identify source of requests
    method = environ.get("REQUEST_METHOD", "UNKNOWN")
    user_agent = environ.get("HTTP_USER_AGENT", "unknown")
    remote_addr = environ.get("REMOTE_ADDR", "unknown")
    print(
        f"[WSGI-503] {method} {path} from {remote_addr} | UA: {user_agent[:50]}...",
        flush=True,
    )
    start_response("503 Service Unavailable", [("Content-Type", "text/plain")])
    return [b"Server starting, retry in a few seconds..."]


app = application

print(f"[WSGI] ✅ WSGI ready at {time.time() - _STARTUP:.2f}s", flush=True)
print("[WSGI] /health instant, /rpc waits for full load", flush=True)
