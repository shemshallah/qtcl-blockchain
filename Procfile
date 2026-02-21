# ─── HANG FIX (Procfile) ─────────────────────────────────────────────────────
# ORIGINAL: `release: python db_builder_v2.py` ran the full 8-phase pipeline:
#   • 3 external QRNG API calls (ANU / Random.org / LFDR) — 10s timeout × retries = 90s+
#   • 106,496 pseudoqubit inserts via {8,3} tessellation — several minutes of CPU+DB
#   • Full ANALYZE + validation suite on ~80 tables — additional minutes
# All this runs BEFORE gunicorn ever binds to the port → Koyeb sees no port → FREEZE.
#
# FIX: Pass --schema-only to main():
#   Runs only CREATE TABLE IF NOT EXISTS + genesis upsert + PQ column migration.
#   Idempotent. Completes in <30 seconds even on a cold database.
#
# DO NOT remove --schema-only without understanding the cost above.
# Full bootstrap (`python db_builder_v2.py` with no args) is correct for manual
# one-time setup from a shell — it must NEVER be the Koyeb release command.
# ─────────────────────────────────────────────────────────────────────────────
release: python db_builder_v2.py --schema-only
web: gunicorn -w 2 --worker-class gthread --threads 4 --timeout 120 --graceful-timeout 30 --keep-alive 5 --max-requests 1000 --max-requests-jitter 100 --access-logfile - --error-logfile - -b 0.0.0.0:${PORT:-5000} wsgi_config:application
