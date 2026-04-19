"""
Gunicorn configuration for SSE server — async worker pool for unlimited concurrent streams.

Uses eventlet green threads instead of OS threads. Each green thread is lightweight and
non-blocking, so thousands of concurrent SSE connections consume no real threads.
"""

import os
import multiprocessing

# ───────────────────────────────────────────────────────────────────────────────
# Worker class: eventlet (async green threads, not gthread synchronous threads)
# ───────────────────────────────────────────────────────────────────────────────
worker_class = "eventlet"

# Number of worker processes (eventlet handles concurrency within each worker)
# More workers = more resilience to worker crashes, not for concurrency
workers = max(2, multiprocessing.cpu_count() // 2)

# eventlet doesn't use threads — it uses green threads
threads = 1

# ───────────────────────────────────────────────────────────────────────────────
# Timeout: Long-lived SSE streams need a long timeout
# ───────────────────────────────────────────────────────────────────────────────
timeout = 600  # 10 minutes — SSE heartbeats prevent timeout anyway

# ───────────────────────────────────────────────────────────────────────────────
# Binding
# ───────────────────────────────────────────────────────────────────────────────
bind = f"0.0.0.0:{os.environ.get('SSE_PORT', 8001)}"

# ───────────────────────────────────────────────────────────────────────────────
# Logging
# ───────────────────────────────────────────────────────────────────────────────
loglevel = "info"
accesslog = "-"  # stdout
errorlog = "-"   # stderr

# ───────────────────────────────────────────────────────────────────────────────
# Connection handling
# ───────────────────────────────────────────────────────────────────────────────
keepalive = 300  # 5 minute keep-alive for long SSE streams
backlog = 2048   # Connection queue depth
worker_connections = 10000  # Max concurrent connections per worker (eventlet supports many)

# ───────────────────────────────────────────────────────────────────────────────
# Process lifecycle
# ───────────────────────────────────────────────────────────────────────────────
max_requests = 100000  # Restart worker after N requests (prevent memory leaks)
max_requests_jitter = 10000  # Randomize restarts to avoid all workers restarting simultaneously
graceful_timeout = 30  # Grace period for request cleanup on shutdown
