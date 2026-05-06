#!/usr/bin/env python3
"""
Gunicorn Configuration for QTCL MCP Server v3.0
================================================

Optimized for MCP streamable HTTP with the official MCP Python SDK.
Uses gevent or gthread worker class for async handling.

Usage:
    gunicorn -c gunicorn_mcp.conf.py mcp_server:main
    
Or with Flask adapter:
    gunicorn -c gunicorn_mcp.conf.py 'mcp_flask_adapter:app'
"""

import multiprocessing
import os

# ── Server Socket ──────────────────────────────────────────
bind = os.environ.get("GUNICORN_BIND", "0.0.0.0:8000")

# ── Worker Processes ─────────────────────────────────────────
# For MCP streamable HTTP, gthread or gevent is recommended over sync workers.
# gthread handles keep-alive connections better for SSE streams.
workers = int(os.environ.get("GUNICORN_WORKERS", multiprocessing.cpu_count() * 2 + 1))
worker_class = os.environ.get("GUNICORN_WORKER_CLASS", "gthread")
threads = int(os.environ.get("GUNICORN_THREADS", 4))
worker_connections = int(os.environ.get("GUNICORN_WORKER_CONNECTIONS", 1000))

# ── Timeouts ─────────────────────────────────────────────────
timeout = int(os.environ.get("GUNICORN_TIMEOUT", 120))
keepalive = int(os.environ.get("GUNICORN_KEEPALIVE", 5))
graceful_timeout = int(os.environ.get("GUNICORN_GRACEFUL_TIMEOUT", 30))

# ── Request Limits ───────────────────────────────────────────
max_requests = int(os.environ.get("GUNICORN_MAX_REQUESTS", 10000))
max_requests_jitter = int(os.environ.get("GUNICORN_MAX_REQUESTS_JITTER", 1000))

# ── Logging ──────────────────────────────────────────────────
accesslog = os.environ.get("GUNICORN_ACCESS_LOG", "-")
errorlog = os.environ.get("GUNICORN_ERROR_LOG", "-")
loglevel = os.environ.get("GUNICORN_LOG_LEVEL", "info")
access_log_format = os.environ.get(
    "GUNICORN_ACCESS_FORMAT",
    '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s µs'
)

# ── Process Naming ─────────────────────────────────────────
proc_name = "qtcl_mcp"

# ── Server Mechanics ─────────────────────────────────────────
daemon = False
preload_app = False  # Must be False for MCP stdio/subprocess transports

# ── SSL (handled by reverse proxy / Koyeb in production) ──────
# forwarded_allow_ips = '*'
# secure_scheme_headers = {
#     'X-FORWARDED-PROTOCOL': 'ssl',
#     'X-FORWARDED-PROTO': 'https',
#     'X-FORWARDED-SSL': 'on'
# }

# ── Hooks ────────────────────────────────────────────────────
def on_starting(server):
    print("[Gunicorn] QTCL MCP Server starting...")

def on_reload(server):
    print("[Gunicorn] QTCL MCP Server reloading...")

def when_ready(server):
    print(f"[Gunicorn] QTCL MCP Server ready on {bind}")
    print(f"[Gunicorn] Workers: {workers} ({worker_class}, {threads} threads)")
    print(f"[Gunicorn] Protocol: MCP 2025-06-18 (streamable-http)")

def worker_int(worker):
    print(f"[Gunicorn] Worker {worker.pid} interrupted")

def on_exit(server):
    print("[Gunicorn] QTCL MCP Server exiting...")
