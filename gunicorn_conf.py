#!/usr/bin/env python3
"""
Gunicorn Configuration for QTCL Server v5.0
============================================
Unified config (replaces gunicorn_conf.py + gunicorn_mcp_conf.py).
Optimized for MCP 2025-06-18 Streamable HTTP + JSON-RPC 2.0 on Koyeb free tier.

FIXES v5.0:
  - UNIFIED: single config file — Procfile references gunicorn_conf.py
  - preload_app=False: CRITICAL — wsgi_config.py uses a background thread
    to load server.py.  With preload_app=True the master forks workers before
    the thread finishes; the thread dies in children and _full_app stays None
    FOREVER.  Each worker must run its own _load_server thread.
  - workers: env-controlled, default 2 (Koyeb 512MB = ~120MB per worker)
  - keepalive=65: exceeds Koyeb LB idle timeout (60s)
  - worker_class=gthread: correct for MCP streamable-http with SSE streams
  - timeout=120: long enough for slow oracle DB queries / signAndSubmitTx

Usage:
    gunicorn -c gunicorn_conf.py wsgi_config:app
"""
import multiprocessing, os

# ── Server Socket ──────────────────────────────────────────────────────────────
bind = os.environ.get("GUNICORN_BIND", "0.0.0.0:8000")

# ── Worker Processes ───────────────────────────────────────────────────────────
# Koyeb free tier = 512MB RAM. cpu_count()*2+1 on 2-core = 5 workers → OOM.
# Default 2 workers; set GUNICORN_WORKERS env var to override.
# Each gthread worker uses ~80-120MB for QTCL server stack.
workers           = int(os.environ.get("GUNICORN_WORKERS", "2"))
worker_class      = os.environ.get("GUNICORN_WORKER_CLASS", "gthread")
threads           = int(os.environ.get("GUNICORN_THREADS", "4"))
worker_connections= int(os.environ.get("GUNICORN_WORKER_CONNECTIONS", "200"))

# ── Timeouts ───────────────────────────────────────────────────────────────────
timeout           = int(os.environ.get("GUNICORN_TIMEOUT", "120"))
# keepalive must exceed Koyeb LB idle timeout (60s). Set to 65.
keepalive         = int(os.environ.get("GUNICORN_KEEPALIVE", "65"))
graceful_timeout  = int(os.environ.get("GUNICORN_GRACEFUL_TIMEOUT", "30"))

# ── Request Limits ─────────────────────────────────────────────────────────────
max_requests       = int(os.environ.get("GUNICORN_MAX_REQUESTS", "10000"))
max_requests_jitter= int(os.environ.get("GUNICORN_MAX_REQUESTS_JITTER", "1000"))

# ── Logging ────────────────────────────────────────────────────────────────────
accesslog          = os.environ.get("GUNICORN_ACCESS_LOG", "-")
errorlog           = os.environ.get("GUNICORN_ERROR_LOG", "-")
loglevel           = os.environ.get("GUNICORN_LOG_LEVEL", "info")
access_log_format  = os.environ.get(
    "GUNICORN_ACCESS_FORMAT",
    '%(h)s - [%(t)s] "%(r)s" %(s)s %(b)s %(D)s "%(a)s"'
)

# ── Process ────────────────────────────────────────────────────────────────────
proc_name  = "qtcl_mcp"
daemon     = False

# ══ CRITICAL ══════════════════════════════════════════════════════════════════
# preload_app MUST be False.
#
# wsgi_config.py loads server.py in a background thread (_load_server).
# With preload_app=True:
#   1. Master process starts, runs wsgi_config.py module code
#   2. _load_server thread starts in master
#   3. Master forks workers — daemon threads do NOT survive fork
#   4. Workers get _full_app = None, _load_done Event never fires
#   5. Every request 503s FOREVER
#
# With preload_app=False:
#   1. Each worker runs wsgi_config.py independently
#   2. Each worker starts its own _load_server thread
#   3. Each worker's _full_app gets set when server.py finishes importing
#   4. Everything works
# ══════════════════════════════════════════════════════════════════════════════
preload_app = False

# ── Forwarded Headers (Koyeb reverse proxy) ────────────────────────────────────
forwarded_allow_ips = "*"
secure_scheme_headers = {
    "X-FORWARDED-PROTO": "https",
    "X-FORWARDED-SSL":   "on",
}

# ── Hooks ──────────────────────────────────────────────────────────────────────
def on_starting(server):
    print("[Gunicorn] QTCL Server v5.0 starting...")

def on_reload(server):
    print("[Gunicorn] QTCL Server reloading...")

def when_ready(server):
    print(f"[Gunicorn] Ready on {bind}")
    print(f"[Gunicorn] Workers: {workers} × {worker_class} ({threads} threads each)")
    print(f"[Gunicorn] Protocol: MCP 2025-06-18 (streamable-http) + JSON-RPC 2.0")
    print(f"[Gunicorn] Keepalive: {keepalive}s (Koyeb LB timeout 60s)")
    print(f"[Gunicorn] preload_app: {preload_app} (MUST be False for wsgi background loading)")

def worker_int(worker):
    print(f"[Gunicorn] Worker {worker.pid} interrupted")

def on_exit(server):
    print("[Gunicorn] QTCL Server exiting...")
