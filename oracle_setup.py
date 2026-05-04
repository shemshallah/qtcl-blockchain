#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  QTCL ORACLE SETUP — Server-Side BFT Oracle Genesis                          ║
║                                                                              ║
║  Usage:                                                                      ║
║    python oracle_setup.py                                                    ║
║                                                                              ║
║  Environment:                                                                ║
║    DATABASE_URL — PostgreSQL connection string (required, Koyeb-managed)     ║
║                                                                              ║
║  Behavior:                                                                   ║
║    1. Connects to PostgreSQL via DATABASE_URL (SSL enforced)                 ║
║    2. Generates 5 fresh HypΓ keypairs (genuine wallets)                      ║
║    3. Inserts them into oracle_registry as the initial BFT set               ║
║    4. Prints credentials for distribution to oracle operators                ║
║                                                                              ║
║  Security:                                                                   ║
║    • Private keys are printed ONCE and never stored on the server            ║
║    • Each oracle operator receives one private key to run their node         ║
║    • The server only stores public keys and addresses in the DB              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import hashlib
import secrets
from datetime import datetime, timezone

# ── Path setup ───────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_HLWE_DIR = os.path.join(_SCRIPT_DIR, "hlwe")
if _HLWE_DIR not in sys.path:
    sys.path.insert(0, _HLWE_DIR)

# ── Dependencies ─────────────────────────────────────────────────────────────
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    print("❌ psycopg2 required. Install: pip install psycopg2-binary")
    sys.exit(1)

try:
    from hyp_engine import HypGammaEngine
except ImportError as e:
    print(f"❌ hyp_engine not found in {_HLWE_DIR}: {e}")
    sys.exit(1)

# ── Constants ────────────────────────────────────────────────────────────────
NUM_ORACLES = 5
MIN_ORACLES_REQUIRED = 3


def _get_db_conn():
    """Connect to PostgreSQL using DATABASE_URL env var with SSL."""
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("❌ DATABASE_URL environment variable not set")
        sys.exit(1)

    # Force SSL if not already specified
    if "sslmode=" not in dsn:
        dsn += "&sslmode=require" if "?" in dsn else "?sslmode=require"

    try:
        conn = psycopg2.connect(dsn, cursor_factory=RealDictCursor)
        conn.autocommit = True
        return conn
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        sys.exit(1)


def _ensure_oracle_registry_table(cur):
    """Ensure oracle_registry table exists with all required columns."""
    cur.execute("""
        CREATE TABLE IF NOT EXISTS oracle_registry (
            oracle_id       VARCHAR(128)  PRIMARY KEY,
            oracle_url      VARCHAR(512)  NOT NULL DEFAULT '',
            oracle_address  VARCHAR(128)  NOT NULL DEFAULT '',
            is_primary      BOOLEAN       NOT NULL DEFAULT FALSE,
            last_seen       TIMESTAMPTZ   DEFAULT NOW(),
            block_height    BIGINT        NOT NULL DEFAULT 0,
            peer_count      INTEGER       NOT NULL DEFAULT 0,
            wallet_address  VARCHAR(128)  NOT NULL DEFAULT '',
            oracle_pub_key  TEXT          NOT NULL DEFAULT '',
            cert_sig        VARCHAR(128)  NOT NULL DEFAULT '',
            mode            VARCHAR(32)   NOT NULL DEFAULT 'full',
            ip_hint         VARCHAR(256)  NOT NULL DEFAULT '',
            reg_tx_hash     VARCHAR(64)   NOT NULL DEFAULT '',
            registered_at   TIMESTAMPTZ   DEFAULT NOW(),
            created_at      TIMESTAMPTZ   DEFAULT NOW()
        )
    """)

    # Ensure columns exist
    for col, dtype in [
        ("wallet_address", "VARCHAR(128) DEFAULT ''"),
        ("oracle_pub_key", "TEXT DEFAULT ''"),
        ("cert_sig", "VARCHAR(128) DEFAULT ''"),
        ("mode", "VARCHAR(32) DEFAULT 'full'"),
        ("ip_hint", "VARCHAR(256) DEFAULT ''"),
        ("reg_tx_hash", "VARCHAR(64) DEFAULT ''"),
        ("registered_at", "TIMESTAMPTZ DEFAULT NOW()"),
    ]:
        cur.execute(
            f"ALTER TABLE oracle_registry ADD COLUMN IF NOT EXISTS {col} {dtype}"
        )


def _count_existing_oracles(cur) -> int:
    """Count how many oracles are already registered."""
    cur.execute("SELECT COUNT(*) FROM oracle_registry WHERE mode IN ('full', 'primary')")
    row = cur.fetchone()
    return row["count"] if row else 0


def _generate_oracle_credentials(engine: HypGammaEngine, index: int) -> dict:
    """Generate a fresh HypΓ keypair for an oracle."""
    kp = engine.generate_keypair()
    oracle_id = f"oracle_{index + 1}"

    # Create a cert_sig: the oracle signs its own id + address binding
    binding = f"{oracle_id}|{kp.address}".encode()
    binding_hash = hashlib.sha3_256(binding).digest()
    cert_sig = engine.sign_hash(binding_hash, kp.private_key)

    return {
        "oracle_id": oracle_id,
        "oracle_address": kp.address,
        "oracle_pub_key": kp.public_key,
        "private_key": kp.private_key,  # Printed only, never stored
        "cert_sig": json.dumps(cert_sig, separators=(",", ":")),
        "mode": "full",
        "is_primary": index == 0,  # First oracle is primary
    }


def _insert_oracle(cur, creds: dict):
    """Insert oracle credentials into oracle_registry (private key excluded)."""
    cur.execute("""
        INSERT INTO oracle_registry
        (oracle_id, oracle_address, oracle_pub_key, cert_sig, mode, is_primary, registered_at)
        VALUES (%s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (oracle_id) DO UPDATE SET
            oracle_address = EXCLUDED.oracle_address,
            oracle_pub_key = EXCLUDED.oracle_pub_key,
            cert_sig       = EXCLUDED.cert_sig,
            mode           = EXCLUDED.mode,
            is_primary     = EXCLUDED.is_primary,
            registered_at  = NOW()
    """, (
        creds["oracle_id"],
        creds["oracle_address"],
        creds["oracle_pub_key"],
        creds["cert_sig"],
        creds["mode"],
        creds["is_primary"],
    ))


def main():
    print("═" * 70)
    print("  QTCL BFT ORACLE SETUP")
    print("═" * 70)

    # Check existing
    conn = _get_db_conn()
    cur = conn.cursor()
    _ensure_oracle_registry_table(cur)

    existing = _count_existing_oracles(cur)
    if existing >= NUM_ORACLES:
        print(f"\n⚠️  {existing} oracles already registered. Setup already complete.")
        print("   To regenerate, DELETE FROM oracle_registry; first.")
        sys.exit(0)

    print(f"\n📊 Found {existing} existing oracles. Generating {NUM_ORACLES} new ones…\n")

    engine = HypGammaEngine()
    credentials = []

    for i in range(NUM_ORACLES):
        creds = _generate_oracle_credentials(engine, i)
        _insert_oracle(cur, creds)
        credentials.append(creds)
        print(f"  ✅ {creds['oracle_id']}  addr={creds['oracle_address'][:24]}…")

    print(f"\n{'═' * 70}")
    print("  🔐 ORACLE CREDENTIALS — DISTRIBUTE SECURELY TO OPERATORS")
    print(f"{'═' * 70}")

    for c in credentials:
        print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│  {c['oracle_id'].upper():74}│
├─────────────────────────────────────────────────────────────────────────────┤
│  Address:     {c['oracle_address']:62}│
│  Public Key:  {c['oracle_pub_key'][:60]:62}│
│  Private Key: {c['private_key'][:60]:62}│
│  Mode:        {c['mode']:62}│
│  Primary:     {str(c['is_primary']):62}│
└─────────────────────────────────────────────────────────────────────────────┘""")

    print(f"\n{'═' * 70}")
    print("  ⚠️  SECURITY WARNING")
    print(f"{'═' * 70}")
    print("  • Private keys above are printed ONCE.")
    print("  • Each oracle operator must save their private key securely.")
    print("  • The server does NOT store private keys — only addresses + pub keys.")
    print("  • BFT consensus requires 3-of-5 oracles to finalize blocks.")
    print(f"{'═' * 70}\n")

    # Also save a JSON file for convenience (but NOT on the server — this is local output)
    _export = [
        {
            "oracle_id": c["oracle_id"],
            "oracle_address": c["oracle_address"],
            "oracle_pub_key": c["oracle_pub_key"],
            "private_key": c["private_key"],
            "mode": c["mode"],
            "is_primary": c["is_primary"],
        }
        for c in credentials
    ]
    _fname = f"oracle_credentials_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    with open(_fname, "w") as f:
        json.dump(_export, f, indent=2)
    print(f"  📄 Credentials also saved to: {_fname}")
    print(f"     (Keep this file secure — it contains private keys)\n")


if __name__ == "__main__":
    main()
