"""
gunicorn.conf.py — QTCL Blockchain Enterprise Production Configuration
═══════════════════════════════════════════════════════════════════════
Auto-loaded by gunicorn at startup. Overrides dashboard/CLI worker settings.

WORKER MODEL — gthread:
  SSE (/api/events, /api/snapshot/sse) now uses a C ring buffer — each SSE
  handler spins at 250 ms polling the ring, releasing the GIL on each sleep.
  A single gthread handles hundreds of SSE connections concurrently.
  64 threads → 64 concurrent blocking operations in flight at once.

SCALE ARCHITECTURE:
  • Single Koyeb instance:  1 worker × 64 threads = 64 concurrent requests
  • Horizontal scale:       Multiple Koyeb instances, each 1 worker × 64 threads
  • Cross-instance mempool: pg_notify('qtcl_mempool') — _PGListenerThread in each
    worker receives TX payloads from all other workers and inserts into local heap.
    Every worker's in-memory mempool is consistent within ~1–5 ms.
  • Cross-instance SSE:     pg_notify('qtcl_sse_events') — _SSEBroadcaster listener
    fans notifications into the C ring buffer; all SSE clients on any worker receive
    every event.
  • Connection budget:      pool max=40 query conns + 1 NOTIFY conn (_PGNotifier) +
    1 LISTEN conn (_PGListenerThread) + 1 LISTEN conn (_SSEBroadcaster) = 43 max
    per worker.  Supabase pooler comfortably handles this.
  • DB_POOL_MAX env var:     set to 40 (default) — raise if adding more workers.

SCALING ROADMAP:
  1. Now:     1 worker, 64 threads, PG NOTIFY cross-worker  ← THIS CONFIG
  2. Phase 2: Increase Koyeb instance count (horizontal) — works automatically
  3. Phase 3: workers = 2-4 per instance if CPU-bound; DB_POOL_MAX stays 40
             per worker — Supabase pooler multiplexes all of them.
"""

import os
import multiprocessing

# ── Binding ────────────────────────────────────────────────────────────────────
bind = f"0.0.0.0:{os.environ.get('FLASK_INTERNAL_PORT', '8000')}"

# ── Worker model ───────────────────────────────────────────────────────────────
worker_class   = "gthread"
workers        = 1          # 1 per Koyeb instance — shared in-memory state safe
threads        = 64         # 64 gthreads: SSE ring-pollers + API handlers + oracles
                            # Each SSE conn sleeps 250 ms / iteration (releases GIL)
                            # → 64 threads handles ~200+ SSE clients + full API load

# ── Timeouts ───────────────────────────────────────────────────────────────────
timeout        = 0          # SSE workers must NEVER time out — 0 = infinite
graceful_timeout = 30
keepalive      = 75         # > Koyeb's 60s idle TCP timeout

# ── Lifecycle ──────────────────────────────────────────────────────────────────
preload_app    = False      # False = each worker initializes independently
                            # Required: each worker needs its own PG LISTEN conn
max_requests   = 10000      # Recycle worker after 10k requests (memory leak guard)
max_requests_jitter = 1000  # Spread recycling to avoid thundering herd

# ── Logging ────────────────────────────────────────────────────────────────────
loglevel             = "info"
accesslog            = "-"
errorlog             = "-"
access_log_format    = '%(h)s %(l)s %(t)s "%(r)s" %(s)s %(b)s "%(a)s"'

# ── Connection ─────────────────────────────────────────────────────────────────
backlog          = 2048
worker_connections = 1000

# ── Hooks ──────────────────────────────────────────────────────────────────────

def post_fork(server, worker):
    """
    Called inside the new worker process after fork.
    Each worker must re-open its own PG connections — psycopg2 connections
    are NOT fork-safe and must never be shared across fork.

    The Mempool singleton is lazy-initialised on first get_mempool() call,
    which starts _PGListenerThread and _PGNotifier in the worker process.
    The _SSEBroadcaster starts its own _listen_loop thread.
    Nothing to do here explicitly — lazy init handles everything.
    """
    import logging
    log = logging.getLogger('gunicorn.error')
    log.info(f"[GUNICORN] Worker pid={worker.pid} forked — PG conns will init on first request")

def worker_exit(server, worker):
    """Close DB pool cleanly on worker shutdown."""
    try:
        import server as _srv
        _srv.db_pool.close_all()
    except Exception:
        pass

# ═════════════════════════════════════════════════════════════════════════════════
# 5-ORACLE CLUSTER WORKER CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════════

ORACLE_WORKER_CONFIG = {
    'oracle_1': {'port': 5000, 'workers': 4, 'threads': 2, 'timeout': 15},
    'oracle_2': {'port': 5001, 'workers': 4, 'threads': 2, 'timeout': 15},
    'oracle_3': {'port': 5002, 'workers': 2, 'threads': 2, 'timeout': 10},
    'oracle_4': {'port': 5003, 'workers': 2, 'threads': 2, 'timeout': 10},
    'oracle_5': {'port': 5004, 'workers': 2, 'threads': 2, 'timeout': 10}
}

TOTAL_ORACLE_WORKERS = 14
TOTAL_ORACLE_THREADS = 28

CONSENSUS_TIMEOUT = 10
CONSENSUS_BROADCAST_INTERVAL = 18
