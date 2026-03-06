"""
gunicorn.conf.py — QTCL Blockchain Production Configuration
════════════════════════════════════════════════════════════
Auto-loaded by gunicorn at startup (before CLI args in some versions,
alongside them in others — worker_class here is the canonical source).

WHY gthread WORKER:
  The sync worker processes one request at a time. An SSE connection
  (/api/events) holds the socket open indefinitely. With sync, it
  captures the single worker permanently — every subsequent API call
  (mempool poll, W-state, peer registration) queues behind it and hits
  the 10s read timeout. gthread gives each request its own OS thread;
  SSE and API calls run concurrently with zero blocking.

WHY NOT gevent:
  Would require monkey-patching psycopg2 and all stdlib. ThreadedConnectionPool
  required. Adds fragility. gthread needs zero changes to existing code.
"""

import os
import multiprocessing

# ── Binding ────────────────────────────────────────────────────────────────────
bind        = f"0.0.0.0:{os.environ.get('PORT', '8000')}"

# ── Worker model ───────────────────────────────────────────────────────────────
# gthread: 1 worker process, N threads. SSE + API calls are fully concurrent.
# sync:    1 worker process, 1 thread. SSE locks out every other request. BROKEN.
worker_class   = "gthread"
workers        = 1          # Single process — shared in-memory state (SSE queues, mempool)
threads        = 32         # 32 concurrent requests: SSE subscribers + API calls + heartbeats

# ── Timeouts ───────────────────────────────────────────────────────────────────
timeout        = 0          # SSE workers must NEVER time out — 0 = infinite
graceful_timeout = 30       # Graceful shutdown window
keepalive      = 75         # TCP keep-alive seconds (> Koyeb's 60s idle timeout)

# ── Logging ────────────────────────────────────────────────────────────────────
loglevel       = "info"
accesslog      = "-"        # stdout
errorlog       = "-"        # stdout
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# ── Performance ────────────────────────────────────────────────────────────────
preload_app    = True       # Load app once in master — workers fork, saving memory
max_requests   = 0          # Don't recycle workers (would drop SSE connections)
max_requests_jitter = 0

# ── Connection limits ──────────────────────────────────────────────────────────
backlog        = 512
