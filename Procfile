# QUANTUM TEMPORAL COHERENCE LEDGER — Procfile
# ═══════════════════════════════════════════════════════════════════════════════
#
# SCALING ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
# This blockchain's load profile is unusual: each Gunicorn worker boots a full
# 106,496-qubit lattice, a 57-neuron adaptive controller, a non-Markovian noise
# bath, W-state managers, and an Aer quantum circuit simulator. That stack uses
# ~400-600MB RAM per worker and takes ~2-3s to initialize.
#
# Scaling strategy:
#   • Vertical first — more threads per worker, not more workers.
#     Workers = RAM cost. Threads = (almost) free within a worker.
#   • gthread worker class lets each worker serve N concurrent requests
#     without blocking the process, which is essential when some endpoints
#     (block-create, quantum-circuit, pq-key-gen) are CPU/I/O heavy.
#   • DO NOT use --preload. The quantum heartbeat is a daemon thread auto-
#     started at import time. fork() after preload kills threads in workers —
#     your heartbeat, neural refresh, and W-state maintainer would be dead.
#     Each worker MUST import fresh so its own thread group starts.
#   • DO NOT add more process types for heartbeat / oracle / scheduler / defi.
#     All of those are daemon threads already running inside each web worker
#     (registered as HEARTBEAT listeners). Separate processes would duplicate
#     state and fight the workers over DB connections.
#
# CONNECTION POOL MATH (Supabase/Postgres default max_connections = 60):
#   Workers × (threads + parallel_batch_workers) ≤ max_connections
#   2 workers × (4 threads + 3 batch workers) = 14 connections — safe.
#   Scale to 4 workers only if you upgrade to pgBouncer or a higher tier.
#
# KOYEB NOTES:
#   Koyeb runs exactly one process type per service deployment. Use the `web`
#   type for your main deployment. The `release` command runs once per deploy
#   before traffic is switched over. Other types only run if you create
#   additional services in your Koyeb dashboard pointing at the same repo.
#
# ═══════════════════════════════════════════════════════════════════════════════

# ── PRIMARY WEB SERVICE ───────────────────────────────────────────────────────
# -w 2            2 workers. Each holds the full quantum stack in memory.
#                 Raise to 4 only when you have >2GB RAM headroom AND
#                 a connection pooler in front of Postgres.
# gthread         Thread-based workers. Non-blocking: while one thread awaits a
#                 DB write, the other 3 keep serving requests. Essential here
#                 because block-create and quantum-circuit calls are slow.
# --threads 4     4 threads per worker = 8 concurrent request handlers total.
#                 Each thread shares the worker's quantum state (safe — all
#                 mutable state is RLock-protected inside the quantum classes).
# --timeout 120   Quantum circuit simulation + PQ key generation can take 30-60s
#                 under load. 120s gives headroom without hanging forever.
# --graceful-timeout 30
#                 On SIGTERM (deploy/scale event), give in-flight block
#                 finality and TX signing requests 30s to complete cleanly.
# --keep-alive 5  Blockchain clients (wallets, dApps) make burst requests.
#                 Keep-alive reduces TLS handshake overhead for sequential calls.
# --max-requests 1000
#                 Recycle workers after 1000 requests to prevent slow memory
#                 growth from quantum state accumulation over many cycles.
# --max-requests-jitter 100
#                 Stagger worker recycling so both workers don't restart at once,
#                 which would drop all in-flight requests simultaneously.
web: gunicorn \
  -w 2 \
  --worker-class gthread \
  --threads 4 \
  --timeout 120 \
  --graceful-timeout 30 \
  --keep-alive 5 \
  --max-requests 1000 \
  --max-requests-jitter 100 \
  --access-logfile - \
  --error-logfile - \
  -b 0.0.0.0:${PORT:-5000} \
  wsgi_config:application

# ── RELEASE PHASE ────────────────────────────────────────────────────────────
# Runs once before each deployment, before traffic is switched over.
# db_builder_v2.py creates/migrates all tables and seeds the genesis block.
# If this fails, the deployment is aborted — protects against schema drift.
release: python db_builder_v2.py
