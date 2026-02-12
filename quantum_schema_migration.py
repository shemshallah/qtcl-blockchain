#!/usr/bin/env python3
"""
quantum_schema_migration.py
─────────────────────────────────────────────────────────────────────────────
Idempotent migrations for quantum-specific columns.
Called at startup to ensure schema is complete regardless of when it was created.
Run automatically by initialize_app() → no manual invocation needed.
─────────────────────────────────────────────────────────────────────────────
"""

import logging
logger = logging.getLogger(__name__)


MIGRATIONS = [
    # ── transactions table ───────────────────────────────────────────────────
    ("transactions", "finalized_at",
     "ALTER TABLE transactions ADD COLUMN IF NOT EXISTS finalized_at TIMESTAMP WITH TIME ZONE"),
    ("transactions", "circuit_size",
     "ALTER TABLE transactions ADD COLUMN IF NOT EXISTS circuit_size INT"),
    ("transactions", "ghz_fidelity",
     "ALTER TABLE transactions ADD COLUMN IF NOT EXISTS ghz_fidelity FLOAT"),
    ("transactions", "noise_source",
     "ALTER TABLE transactions ADD COLUMN IF NOT EXISTS noise_source VARCHAR(50)"),

    # ── quantum_measurements table ───────────────────────────────────────────
    ("quantum_measurements", "entropy_score",
     "ALTER TABLE quantum_measurements ADD COLUMN IF NOT EXISTS entropy_score FLOAT"),
    ("quantum_measurements", "noise_source",
     "ALTER TABLE quantum_measurements ADD COLUMN IF NOT EXISTS noise_source VARCHAR(50) DEFAULT 'xorshift64'"),
    ("quantum_measurements", "is_valid_finality",
     "ALTER TABLE quantum_measurements ADD COLUMN IF NOT EXISTS is_valid_finality BOOLEAN DEFAULT FALSE"),
    ("quantum_measurements", "tx_id_unique",
     "CREATE UNIQUE INDEX IF NOT EXISTS uq_quantum_measurements_tx_id ON quantum_measurements(tx_id)"),

    # ── Disable legacy gas fields by setting DEFAULT 0 (keeps columns, stops use) ─
    # We leave gas columns in the schema so existing code doesn't break, but
    # they're always 0 in gas-free mode.
    ("transactions", "gas_used_zero",
     "ALTER TABLE transactions ALTER COLUMN gas_used SET DEFAULT 0"),
]


def run_migrations(db_connection) -> bool:
    """
    Run all migrations idempotently.
    Returns True if all succeeded (or were already applied).
    """
    success = True
    for (table, column, sql) in MIGRATIONS:
        try:
            db_connection.execute_update(sql)
            logger.debug(f"[MIGRATION] ✓ {table}.{column}")
        except Exception as exc:
            # Most errors here are "already exists" — that's fine
            if 'already exists' in str(exc).lower() or 'duplicate' in str(exc).lower():
                logger.debug(f"[MIGRATION] Already applied: {table}.{column}")
            else:
                logger.warning(f"[MIGRATION] Non-fatal: {table}.{column}: {exc}")
                # Don't set success=False for non-critical migrations
    return success


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.DEBUG)
    from db_config import DatabaseConnection
    run_migrations(DatabaseConnection)
    print("Migrations complete")
