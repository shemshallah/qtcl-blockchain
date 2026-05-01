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
worker_class = "gthread"
workers = 1  # 1 worker = shared in-memory cache state
threads = 200  # 🚀 200 threads for 10,000 miners (200 concurrent ops)
# Each thread handles 50 miners via async I/O

# ── Timeouts ───────────────────────────────────────────────────────────────────
timeout = 120  # 120s timeout (blocks may take time to validate)
graceful_timeout = 60  # 60s graceful shutdown
keepalive = 300  # 5 min keepalive for persistent miner connections

# ── Lifecycle ──────────────────────────────────────────────────────────────────
preload_app = False  # Disable preload - let wsgi_config handle lazy loading
max_requests = 50000  # Recycle after 50k requests
max_requests_jitter = 5000  # Spread recycling

# ── Logging ────────────────────────────────────────────────────────────────────
# Access log: 200s hidden by default. Set LOG_200=1 env var to restore.
# Oracle/lattice metrics (WARNING level) always visible.
loglevel = "warning"
accesslog = "-"
errorlog = "-"
access_log_format = '%(h)s %(l)s %(t)s "%(r)s" %(s)s %(b)s %(D)s "%(a)s"'

import logging as _logging
import os as _os


class _ErrorOnlyAccessFilter(_logging.Filter):
    """Pass only 4xx/5xx. Drop 2xx/3xx. Disable with LOG_200=1."""
    def filter(self, record: _logging.LogRecord) -> bool:
        if _os.environ.get("LOG_200"):
            return True
        try:
            status = int(record.getMessage().rsplit('"', 1)[-1].strip().split()[0])
            return status >= 400
        except (ValueError, IndexError):
            return True


def _apply_filters():
    """
    Force gunicorn.access to WARNING + attach drop-200s filter.
    Must set .setLevel() — without it Gunicorn leaves the logger at NOTSET
    which inherits root DEBUG and every record reaches handlers regardless
    of any filter attached.
    """
    if not _os.environ.get("LOG_200"):
        _al = _logging.getLogger("gunicorn.access")
        _al.setLevel(_logging.WARNING)
        if not any(isinstance(f, _ErrorOnlyAccessFilter) for f in _al.filters):
            _al.addFilter(_ErrorOnlyAccessFilter())
    _logging.getLogger("werkzeug").setLevel(_logging.ERROR)
    # Oracle/lattice/quantum: ensure WARNING+ is visible
    for _ns in ("oracle", "lattice_controller", "globals", "qrng_ensemble"):
        _lg = _logging.getLogger(_ns)
        if _lg.level == _logging.NOTSET or _lg.level > _logging.WARNING:
            _lg.setLevel(_logging.WARNING)


# ── Connection ─────────────────────────────────────────────────────────────────
backlog = 4096  # 🚀 4k backlog for connection bursts
worker_connections = 2000  # 2k concurrent connections

# ── Hooks ──────────────────────────────────────────────────────────────────────


def on_starting(server):
    """Wire access log filter in master before any worker forks."""
    _apply_filters()
    server.log.info(
        "[GUNICORN] 200s suppressed (LOG_200=1 to restore) | "
        "oracle/lattice metrics at WARNING+ always visible"
    )


def post_fork(server, worker):
    """Re-apply filter in each worker process (filters don't cross fork)."""
    _apply_filters()
    import logging
    logging.getLogger("gunicorn.error").info(
        f"[GUNICORN] Worker {worker.pid} ready — access log: errors only"
    )


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
    "oracle_1": {"port": 5000, "workers": 4, "threads": 2, "timeout": 15},
    "oracle_2": {"port": 5001, "workers": 4, "threads": 2, "timeout": 15},
    "oracle_3": {"port": 5002, "workers": 2, "threads": 2, "timeout": 10},
    "oracle_4": {"port": 5003, "workers": 2, "threads": 2, "timeout": 10},
    "oracle_5": {"port": 5004, "workers": 2, "threads": 2, "timeout": 10},
}

TOTAL_ORACLE_WORKERS = 14
TOTAL_ORACLE_THREADS = 28

CONSENSUS_TIMEOUT = 10
CONSENSUS_BROADCAST_INTERVAL = 18
