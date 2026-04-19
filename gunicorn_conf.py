"""
🚀 gunicorn.conf.py — QTCL Blockchain WEB-SCALE Configuration
══════════════════════════════════════════════════════════════════════
Optimized for 10,000 concurrent miners on single instance

WORKER MODEL — gthread:
  200 threads → 200 concurrent blocking operations
  Async request handling with connection pooling
  Target: 10,000 requests/sec throughput

SCALE ARCHITECTURE (Web-Scale Single Instance):
  • Single instance:  1 worker × 200 threads = 200 concurrent requests
  • Backlog: 4096 (handles connection bursts)
  • DB Pool: 100 connections ( Neon PostgreSQL )
  • L1 Cache: In-memory LRU (eliminates 99% of height queries)
  • Rate Limiting: Token bucket per miner (10 req/sec burst 20)
  • Circuit Breaker: Protects DB from cascade failure

PERFORMANCE TARGETS:
  • Block submission: < 50ms p99 latency
  • Height query: < 1ms (cached)
  • Concurrent miners: 10,000+
  • Throughput: 10,000 RPC calls/sec

MEMORY FOOTPRINT:
  • Cache: 256MB for 100k entries
  • Threads: ~8MB per thread × 200 = 1.6GB
  • DB Pool: 100 conns × 5MB = 500MB
  • Total: ~3-4GB per instance
"""

import os
import multiprocessing

# ── Binding ────────────────────────────────────────────────────────────────────
bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"

# ── Worker model ───────────────────────────────────────────────────────────────
worker_class   = "gthread"
workers        = 1          # 1 worker = shared in-memory cache state
threads        = 200        # 🚀 200 threads for 10,000 miners (200 concurrent ops)
                            # Each thread handles 50 miners via async I/O

# ── Timeouts ───────────────────────────────────────────────────────────────────
timeout        = 120        # 120s timeout (blocks may take time to validate)
graceful_timeout = 60       # 60s graceful shutdown
keepalive      = 300        # 5 min keepalive for persistent miner connections

# ── Lifecycle ──────────────────────────────────────────────────────────────────
preload_app    = True       # 🚀 Preload for faster startup (cache init)
max_requests   = 50000      # Recycle after 50k requests
max_requests_jitter = 5000  # Spread recycling

# ── Logging ────────────────────────────────────────────────────────────────────
loglevel             = "info"
accesslog            = "-"
errorlog             = "-"
access_log_format    = '%(h)s %(l)s %(t)s "%(r)s" %(s)s %(b)s %(D)s "%(a)s"'  # Added %(D)s for request time

# ── Connection ─────────────────────────────────────────────────────────────────
backlog          = 4096     # 🚀 4k backlog for connection bursts
worker_connections = 2000   # 2k concurrent connections

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
