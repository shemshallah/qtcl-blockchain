"""
gunicorn.conf.py — QTCL Blockchain Enterprise Production Configuration
═══════════════════════════════════════════════════════════════════════
Auto-loaded by gunicorn at startup. Overrides dashboard/CLI worker settings.

WORKER MODEL — gthread:
  SSE (/api/events) holds a connection open indefinitely. sync worker = SSE
  captures the only thread → every API call times out. gthread gives each
  request an OS thread; SSE + API calls are fully concurrent.

SCALE ARCHITECTURE:
  • Single Koyeb instance:  1 worker × 64 threads = 64 concurrent requests
  • Horizontal scale:       Multiple Koyeb instances, each 1 worker × 64 threads
  • Cross-instance events:  PostgreSQL LISTEN/NOTIFY (built into _SSEBroadcaster)
    Any instance publishes a TX → PG NOTIFY → all instances fan out to their
    local SSE subscribers. Zero Redis. Zero message broker.
  • Database:               Supabase pooler handles connection multiplexing.
    Each worker holds 1 persistent LISTEN conn + borrows from ThreadedConnectionPool.

SCALING ROADMAP:
  1. Now:     1 worker, 64 threads, PG NOTIFY cross-instance SSE  ← THIS CONFIG
  2. Phase 2: Increase Koyeb instance count (horizontal) — works automatically
  3. Phase 3: workers = 2-4 per instance if CPU-bound (PG NOTIFY handles SSE fanout)
"""

import os
import multiprocessing

# ── Binding ────────────────────────────────────────────────────────────────────
bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"

# ── Worker model ───────────────────────────────────────────────────────────────
worker_class   = "gthread"
workers        = 1          # 1 per Koyeb instance — shared in-memory state safe
                            # Scale horizontally via multiple Koyeb instances
threads        = 64         # 64 concurrent: SSE subscribers + API + heartbeats

# ── Timeouts ───────────────────────────────────────────────────────────────────
timeout        = 0          # SSE workers must NEVER time out — 0 = infinite
graceful_timeout = 30
keepalive      = 75         # > Koyeb's 60s idle TCP timeout

# ── Lifecycle ──────────────────────────────────────────────────────────────────
preload_app    = False      # False = each worker initializes independently
                            # Required for PG LISTEN (each worker needs own conn)
max_requests   = 10000      # Recycle worker after 10k requests (memory leak guard)
max_requests_jitter = 1000  # Spread recycling to avoid thundering herd

# ── Logging ────────────────────────────────────────────────────────────────────
loglevel             = "info"
accesslog            = "-"
errorlog             = "-"
access_log_format    = '%(h)s %(l)s %(t)s "%(r)s" %(s)s %(b)s "%(a)s"'

# ── Connection ─────────────────────────────────────────────────────────────────
backlog          = 2048
worker_connections = 1000   # per worker (gthread ignores this but good to declare)
