#!/usr/bin/env python3
"""
Gunicorn configuration for QTCL Quantum Lattice Blockchain
with dedicated gthread pool for QuantumLatticeController

Usage:
    gunicorn -c gunicorn_lattice_config.py wsgi_lattice_loader:application
"""

import os
import multiprocessing

# Bind to all interfaces on port 8000
bind = "0.0.0.0:8000"

# Use gthread worker class for threaded concurrency
worker_class = "gthread"

# 1 worker process (master)
workers = 1

# 9 gthread threads per worker:
#   - 1 main (orchestrator)
#   - 5 dedicated oracle threads (PRIMARY_LATTICE, SECONDARY_LATTICE, VALIDATION, ARBITER, METRICS_OBSERVER)
#   - 3 spare for concurrent web requests
threads = 9

# Request timeout
timeout = 120

# Max requests before worker restart
max_requests = 10000
max_requests_jitter = 500

# Graceful timeout for worker shutdown
graceful_timeout = 30

# Keep alive
keepalive = 5

# Access log
accesslog = "-"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Error log
errorlog = "-"
loglevel = "info"

# Logging
capture_output = False
enable_stdio_inheritance = True

# Preload app (forces lattice_controller init before forking)
preload_app = True

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (optional - enable if behind reverse proxy with HTTPS)
# keyfile = None
# certfile = None
# ca_certs = None

# Process naming
proc_name = "qtcl-lattice-node"


def on_starting(server):
    """Called just before the master process is initialized."""
    print("\n" + "="*80)
    print("GUNICORN LATTICE CONFIG — Starting QuantumLatticeController node")
    print("="*80)
    print(f"Workers: {workers}")
    print(f"Worker class: {worker_class}")
    print(f"Threads per worker: {threads}")
    print(f"Timeout: {timeout}s")
    print("="*80 + "\n")


def on_exit(server):
    """Called just before exiting gunicorn."""
    print("\n[GUNICORN] Shutting down QuantumLatticeController...")
