#!/usr/bin/env python3
"""
Gunicorn Configuration for QTCL MCP Server v3.1
================================================
Optimized for MCP 2025-06-18 Streamable HTTP on Koyeb free tier.

FIXES v3.1:
  - workers: was cpu_count*2+1 (OOM on Koyeb 512MB); now env-controlled, default 2
  - preload_app=True: safe with gthread, but sessions are in-memory per-worker —
    stateless_http=True in mcp_server.py means no shared session state needed
  - keepalive=65: Koyeb's load balancer idle timeout is 60s; must exceed it
  - worker_class=gthread: correct for MCP streamable-http with SSE streams
  - timeout=120: long enough for slow oracle DB queries

Usage:
    gunicorn -c gunicorn_mcp_conf.py wsgi_config:app
"""
import multiprocessing, os

# ── Server Socket ──────────────────────────────────────────────────────────────
bind = os.environ.get("GUNICORN_BIND", "0.0.0.0:8000")

# ── Worker Processes ───────────────────────────────────────────────────────────
# FIX: Koyeb free tier = 512MB RAM. cpu_count()*2+1 on 2-core = 5 workers → OOM.
# Default 2 workers; set GUNICORN_WORKERS env var to override.
# Each gthread worker uses ~80-120MB for QTCL server stack.
workers           = int(os.environ.get("GUNICORN_WORKERS", "2"))
worker_class      = os.environ.get("GUNICORN_WORKER_CLASS", "gthread")
threads           = int(os.environ.get("GUNICORN_THREADS", "4"))
worker_connections= int(os.environ.get("GUNICORN_WORKER_CONNECTIONS", "200"))

# ── Timeouts ───────────────────────────────────────────────────────────────────
timeout           = int(os.environ.get("GUNICORN_TIMEOUT", "120"))
# FIX: keepalive must exceed Koyeb LB idle timeout (60s). Set to 65.
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
    '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s µs'
)

# ── Process ────────────────────────────────────────────────────────────────────
proc_name  = "qtcl_mcp"
daemon     = False
# FIX: preload_app=True — safe because stateless_http=True means no shared session
# state between workers. All worker state is reconstructed from request headers.
preload_app = True

# ── Forwarded Headers (Koyeb reverse proxy) ────────────────────────────────────
forwarded_allow_ips = "*"
secure_scheme_headers = {
    "X-FORWARDED-PROTO": "https",
    "X-FORWARDED-SSL":   "on",
}

# ── Hooks ──────────────────────────────────────────────────────────────────────
def on_starting(server):
    print("[Gunicorn] QTCL MCP Server v3.1 starting...")

def on_reload(server):
    print("[Gunicorn] QTCL MCP Server reloading...")

def when_ready(server):
    print(f"[Gunicorn] Ready on {bind}")
    print(f"[Gunicorn] Workers: {workers} × {worker_class} ({threads} threads each)")
    print(f"[Gunicorn] Protocol: MCP 2025-06-18 (streamable-http)")
    print(f"[Gunicorn] Keepalive: {keepalive}s (Koyeb LB timeout 60s)")

def worker_int(worker):
    print(f"[Gunicorn] Worker {worker.pid} interrupted")

def on_exit(server):
    print("[Gunicorn] QTCL MCP Server exiting...")
