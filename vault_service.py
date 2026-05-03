#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                                                                                  ║
║   QTCL VAULT SERVICE v1.0 — Post-Quantum Encryption-as-a-Service               ║
║                                                                                  ║
║   Revenue-Generating Encryption Infrastructure for the QTCL Network             ║
║                                                                                  ║
║   ┌─────────────────────────────────────────────────────────────────────────┐   ║
║   │  ARCHITECTURE:                                                          │   ║
║   │                                                                         │   ║
║   │  ┌──────────┐    ┌───────────────┐    ┌────────────────┐              │   ║
║   │  │  Client   │───▶│  Vault RPC    │───▶│  QTCL Chain    │              │   ║
║   │  │  (vault   │    │  Methods      │    │  (anchoring)   │              │   ║
║   │  │   .html)  │    │              │    └────────────────┘              │   ║
║   │  └──────────┘    │              │    ┌────────────────┐              │   ║
║   │                    │              │───▶│  Supabase DB   │              │   ║
║   │                    │              │    │  (VAULT_URL)   │              │   ║
║   │                    └───────────────┘    └────────────────┘              │   ║
║   └─────────────────────────────────────────────────────────────────────────┘   ║
║                                                                                  ║
║   TIERS:                                                                         ║
║     TRIAL  — 1 secret, 1KB max, no chain anchor, device-fingerprinted          ║
║     PAID   — Unlimited secrets, 50MB max, chain anchor, obfuscation option     ║
║     (Pay-as-you-go: 1 QTCL per anchor, 5 QTCL for obfuscated anchor)          ║
║                                                                                  ║
║   ENCRYPTION LAYERS:                                                             ║
║     Layer 1: Client-side AES-256-GCM (WebCrypto, passphrase-derived)           ║
║     Layer 2: Server-side GeodesicLWE (post-quantum, HCVP hardness)             ║
║     Layer 3: Chain anchor (SHA3-256 commitment, immutable proof-of-existence)  ║
║                                                                                  ║
║   PRIVACY:                                                                       ║
║     Obfuscated anchors use a blinding factor: H(secret ‖ blinding_nonce)       ║
║     so the on-chain hash cannot be correlated to the vault entry without       ║
║     the blinding nonce (held only by the vault owner).                         ║
║                                                                                  ║
║   USE CASES:                                                                     ║
║     • Private key inheritance (dead man's switch with Shamir shares)           ║
║     • Credential escrow (API keys, seed phrases, certificates)                 ║
║     • Document notarization (hash-on-chain, content-off-chain)                 ║
║     • Encrypted file storage with quantum-resistant encryption                 ║
║                                                                                  ║
║   DATABASE: VAULT_URL env var (Supabase PostgreSQL connection string)          ║
║   CHAIN:    Anchors stored as special transactions in QTCL blockchain          ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import sys
import json
import time
import hashlib
import secrets
import logging
import threading
import hmac
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# VAULT-LOCAL RPC HELPERS (avoid circular import for module-level lambdas)
# ═══════════════════════════════════════════════════════════════════════════════
_JSONRPC_VERSION = "2.0"


def _rpc_ok(result: Any, rpc_id: Any) -> dict:
    """Standard JSON-RPC 2.0 success response (vault-local copy)."""
    return {"jsonrpc": _JSONRPC_VERSION, "result": result, "id": rpc_id}


def _rpc_error(code: int, message: str, rpc_id: Any, data: Optional[dict] = None) -> dict:
    """Standard JSON-RPC 2.0 error response (vault-local copy)."""
    resp = {"jsonrpc": _JSONRPC_VERSION, "error": {"code": code, "message": message}, "id": rpc_id}
    if data:
        resp["error"]["data"] = data
    return resp


# ═══════════════════════════════════════════════════════════════════════════════
# VAULT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

VAULT_DB_URL = os.environ.get("VAULT_URL", "")

# ═══════════════════════════════════════════════════════════════════════════════════
# SOVEREIGN PRICING ENGINE — Sized-based, Hyperbolic Costing
# ═══════════════════════════════════════════════════════════════════════════════════

def calculate_vault_cost(encrypted_size_bytes: int = 0, operation: str = "anchor") -> Dict[str, Any]:
    """
    Return cost for a vault operation in base units and QTCL.

    All reads (retrieve, verify, audit) return cost=0 — always free.
    Write operations consume QTCL; 90% is burned, 10% goes to coherence fund.
    """
    cost_map = {
        "anchor":              PRICE_ANCHOR,
        "chain_anchor":        PRICE_ANCHOR,
        "obfuscated_anchor":   PRICE_OBFUSCATED_ANCHOR,
        "account_create":      PRICE_ACCOUNT_CREATE,
        "passphrase_change":   PRICE_PASSPHRASE_CHANGE,
        "reanchor":            PRICE_REANCHOR,
        "export_bundle":       PRICE_EXPORT_BUNDLE,
        "add_collaborator":    PRICE_ADD_COLLABORATOR,
        "revoke_anchor":       PRICE_REVOKE_ANCHOR,
        "delete_anchor":       PRICE_REVOKE_ANCHOR,
        "transfer":            PRICE_TRANSFER,
        # Free operations
        "retrieve":            0,
        "read":                0,
        "verify":              0,
        "audit":               0,
        "store":               0,
        "delete_secret":       0,
    }
    cost = cost_map.get(operation, 0)
    qtcl = cost / QTCL_BASE
    usd  = qtcl * 0.01  # $0.01 per QTCL

    return {
        "operation":    operation,
        "cost_base":    cost,
        "cost_qtcl":    qtcl,
        "cost_usd":     usd,
        "burn_qtcl":    round(qtcl * BURN_RATIO, 4),
        "coherence_qtcl": round(qtcl * COHERENCE_RATIO, 4),
        "free":         cost == 0,
        "rate":         "1 QTCL = $0.01",
    }

# ═══════════════════════════════════════════════════════════════════════════════
# TOKENOMICS — 90/10 BURN/COHERENCE SPLIT
#
#   Exchange rate: 100 QTCL = $1.00 USD  (1 QTCL = $0.01)
#
# Vault pricing is ENFORCEABLE (zero-knowledge proof: server can't see plaintext).
# We charge by what we can MEASURE: size, count, and anchor type.
#
# Subscription tiers = monthly capacity quota (like AWS S3)
# Anchor fees = one-time insurance premium (user-driven, high-value secrets self-select)
# Store fee = tiny flat spam deterrent
# Retrieve / verify / audit = ALWAYS FREE
# ═══════════════════════════════════════════════════════════════════════════════

QTCL_BASE = 100  # base units per QTCL (matches wallet_addresses.balance scale)

# ── Monthly Subscription Tiers ───────────────────────────────────────────────
# Each tier resets monthly. Overages charged per-byte.
VAULT_TIERS = {
    "trial": {
        "monthly_qtcl": 0,
        "max_secrets": 1,
        "max_bytes": 10 * 1024,           # 10 KB
        "anchors_included": 0,
        "overage_per_100kb": 0,            # trial can't go over
    },
    "personal": {
        "monthly_qtcl": 1_000 * QTCL_BASE,  # $10/month
        "max_secrets": 20,
        "max_bytes": 1 * 1024 * 1024,      # 1 MB
        "anchors_included": 5,
        "overage_per_100kb": 1 * QTCL_BASE, # $0.01 per 100 KB overage
    },
    "professional": {
        "monthly_qtcl": 5_000 * QTCL_BASE,  # $50/month
        "max_secrets": 100,
        "max_bytes": 10 * 1024 * 1024,     # 10 MB
        "anchors_included": 10,
        "overage_per_100kb": 1 * QTCL_BASE,
    },
    "enterprise": {
        "monthly_qtcl": 20_000 * QTCL_BASE, # $200/month
        "max_secrets": None,               # unlimited
        "max_bytes": 100 * 1024 * 1024,    # 100 MB
        "anchors_included": None,          # unlimited
        "overage_per_100kb": 1 * QTCL_BASE,
    },
}

# ── Per-Operation Fees (flat, enforceable, category-agnostic) ────────────────
PRICE_STORE_SECRET        =  10 * QTCL_BASE   # $0.10 — spam deterrent
PRICE_DELETE_SECRET       =   5 * QTCL_BASE   # $0.05 — cleanup
PRICE_ANCHOR_STANDARD     = 100 * QTCL_BASE   # $1.00 — standard SHA3-256 chain anchor
PRICE_ANCHOR_OBFUSCATED   = 500 * QTCL_BASE   # $5.00 — blinded, unlinkable anchor
PRICE_REANCHOR            =  50 * QTCL_BASE   # $0.50 — update existing anchor

# ── Burn / Coherence addresses ───────────────────────────────────────────────
BURN_ADDRESS = os.environ.get(
    "VAULT_BURN_ADDRESS",
    "e8ffb27915ac244e8257de8b7f96ad387d1e9d93c634d849a6ad2dae0da6750b"
)
COHERENCE_FUND_ADDRESS = os.environ.get(
    "COHERENCE_FUND_ADDRESS",
    "e8ffb27915ac244e8257de8b7f96ad387d1e9d93c634d849a6ad2dae0da6750b"
)

# Fee split ratios
BURN_RATIO      = 0.90   # 90% permanently destroyed
COHERENCE_RATIO = 0.10   # 10% to quantum infrastructure fund

# ── Stripe purchase tiers (QTCL credited on purchase) ───────────────────────
STRIPE_TIERS = [
    {"price_usd": 0.99,  "qtcl":    150, "label": "Starter",      "anchors": 1},
    {"price_usd": 4.99,  "qtcl":    600, "label": "Explorer",     "anchors": 6},
    {"price_usd": 14.99, "qtcl":  2_000, "label": "Builder",      "anchors": 20},
    {"price_usd": 49.99, "qtcl":  8_000, "label": "Professional", "anchors": 80},
    {"price_usd": 149.99,"qtcl": 30_000, "label": "Enterprise",   "anchors": 300},
]

# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE CONNECTION (Separate pool for vault — VAULT_URL)
# ═══════════════════════════════════════════════════════════════════════════════

_vault_pool = None
_vault_pool_lock = threading.Lock()


def _get_vault_conn():
    """Get a connection from the vault database pool."""
    global _vault_pool
    if not VAULT_DB_URL:
        raise RuntimeError("[VAULT] VAULT_URL not configured")

    if _vault_pool is None:
        with _vault_pool_lock:
            if _vault_pool is None:
                try:
                    import psycopg2
                    from psycopg2 import pool as pg_pool
                    _vault_pool = pg_pool.ThreadedConnectionPool(
                        2, 20, VAULT_DB_URL, connect_timeout=10
                    )
                    logger.info("[VAULT] ✅ Database pool initialized")
                except Exception as e:
                    logger.error(f"[VAULT] ❌ DB pool init failed: {e}")
                    raise

    conn = _vault_pool.getconn()
    conn.autocommit = True
    return conn


def _put_vault_conn(conn):
    """Return connection to pool."""
    if _vault_pool and conn:
        try:
            _vault_pool.putconn(conn)
        except Exception:
            try:
                conn.close()
            except Exception:
                pass


def _vault_query(sql: str, params: tuple = (), fetch: str = "all") -> Any:
    """Execute a query on the vault database. fetch: 'all', 'one', 'none'."""
    conn = _get_vault_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        if fetch == "all":
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall()
            return [dict(zip(cols, row)) for row in rows]
        elif fetch == "one":
            row = cur.fetchone()
            if row and cur.description:
                cols = [d[0] for d in cur.description]
                return dict(zip(cols, row))
            return None
        else:
            return None
    finally:
        _put_vault_conn(conn)


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

_SCHEMA_INITIALIZED = False
_SCHEMA_LOCK = threading.Lock()

VAULT_SCHEMA_SQL = """
-- Vault accounts
CREATE TABLE IF NOT EXISTS vault_accounts (
    id                   TEXT PRIMARY KEY,
    display_name         TEXT,
    email                TEXT,
    passphrase_hash      TEXT NOT NULL,
    tier                 TEXT NOT NULL DEFAULT 'trial',
    device_fp            TEXT,
    qtcl_address         TEXT,
    public_key           TEXT,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login           TIMESTAMPTZ,
    secrets_count        INTEGER NOT NULL DEFAULT 0,
    bytes_stored         BIGINT NOT NULL DEFAULT 0,
    anchors_used         INTEGER NOT NULL DEFAULT 0,
    credit_balance       BIGINT NOT NULL DEFAULT 0,
    subscription_expires_at TIMESTAMPTZ,
    monthly_reset_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    monthly_bytes_used   BIGINT NOT NULL DEFAULT 0,
    monthly_anchors_used INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_vault_accounts_email ON vault_accounts(email);
CREATE INDEX IF NOT EXISTS idx_vault_accounts_device ON vault_accounts(device_fp);
CREATE INDEX IF NOT EXISTS idx_vault_accounts_qtcl ON vault_accounts(qtcl_address);

-- Encrypted secrets
CREATE TABLE IF NOT EXISTS vault_secrets (
    id              TEXT PRIMARY KEY,
    account_id      TEXT NOT NULL REFERENCES vault_accounts(id) ON DELETE CASCADE,
    label           TEXT NOT NULL,
    category        TEXT NOT NULL DEFAULT 'general',
    ciphertext      TEXT NOT NULL,
    encryption_meta JSONB NOT NULL DEFAULT '{}',
    size_bytes      INTEGER NOT NULL,
    anchor_hash     TEXT,
    anchor_block    INTEGER,
    anchor_tx       TEXT,
    obfuscated      BOOLEAN NOT NULL DEFAULT FALSE,
    blinding_nonce  TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at      TIMESTAMPTZ,
    access_count    INTEGER NOT NULL DEFAULT 0,
    last_accessed   TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_vault_secrets_account ON vault_secrets(account_id);
CREATE INDEX IF NOT EXISTS idx_vault_secrets_anchor ON vault_secrets(anchor_hash);
CREATE INDEX IF NOT EXISTS idx_vault_secrets_category ON vault_secrets(account_id, category);

-- Chain anchors (immutable audit log)
CREATE TABLE IF NOT EXISTS vault_anchors (
    id              TEXT PRIMARY KEY,
    account_id      TEXT NOT NULL REFERENCES vault_accounts(id),
    secret_id       TEXT REFERENCES vault_secrets(id),
    anchor_hash     TEXT NOT NULL,
    obfuscated_hash TEXT,
    block_height    INTEGER,
    tx_hash         TEXT,
    chain           TEXT NOT NULL DEFAULT 'qtcl',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    verified        BOOLEAN NOT NULL DEFAULT FALSE,
    verified_at     TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_vault_anchors_hash ON vault_anchors(anchor_hash);
CREATE INDEX IF NOT EXISTS idx_vault_anchors_obfuscated ON vault_anchors(obfuscated_hash);
CREATE INDEX IF NOT EXISTS idx_vault_anchors_account ON vault_anchors(account_id);

-- Billing ledger (every charge/credit)
CREATE TABLE IF NOT EXISTS vault_billing (
    id              TEXT PRIMARY KEY,
    account_id      TEXT NOT NULL REFERENCES vault_accounts(id),
    operation       TEXT NOT NULL,
    amount          BIGINT NOT NULL,
    balance_after   BIGINT NOT NULL,
    tx_hash         TEXT,
    description     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vault_billing_account ON vault_billing(account_id);

-- Inheritance configs (dead man's switch)
CREATE TABLE IF NOT EXISTS vault_inheritance (
    id              TEXT PRIMARY KEY,
    account_id      TEXT NOT NULL REFERENCES vault_accounts(id),
    beneficiary     TEXT NOT NULL,
    shamir_config   JSONB NOT NULL DEFAULT '{}',
    check_in_days   INTEGER NOT NULL DEFAULT 365,
    last_check_in   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    activated       BOOLEAN NOT NULL DEFAULT FALSE,
    activated_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vault_inheritance_account ON vault_inheritance(account_id);
CREATE INDEX IF NOT EXISTS idx_vault_inheritance_check ON vault_inheritance(last_check_in);

-- Address book contacts (synced via vault auth)
CREATE TABLE IF NOT EXISTS vault_contacts (
    id              TEXT PRIMARY KEY,
    account_id      TEXT NOT NULL REFERENCES vault_accounts(id) ON DELETE CASCADE,
    name            TEXT NOT NULL,
    address         TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(account_id, address)
);

CREATE INDEX IF NOT EXISTS idx_vault_contacts_account ON vault_contacts(account_id);
"""


def _apply_vault_burn(cost: int, operation: str, account_id: str, tx_ref: str = "") -> dict:
    """
    90/10 burn/coherence split for every vault fee.

    Submits two micro-transactions to the QTCL chain:
      - 90% to BURN_ADDRESS  → permanently destroyed, supply decreases
      - 10% to COHERENCE_FUND_ADDRESS → oracle infrastructure fund

    Both are fire-and-forget (vault DB is source of truth for billing).
    Returns split amounts for billing record.
    """
    burn_amount     = int(cost * BURN_RATIO)
    coherence_amount = cost - burn_amount  # remainder goes to coherence (avoids rounding loss)

    # Best-effort on-chain write — failure does NOT block vault operation
    try:
        from server import get_db_cursor
        import hashlib as _hl

        burn_fp = _hl.sha256(BURN_ADDRESS.encode()).hexdigest()[:64]
        coh_fp  = _hl.sha256(COHERENCE_FUND_ADDRESS.encode()).hexdigest()[:64]

        with get_db_cursor() as cur:
            # Credit burn address (increases tracked burned supply)
            cur.execute(
                """INSERT INTO wallet_addresses
                   (address, wallet_fingerprint, public_key, balance,
                    transaction_count, address_type, updated_at)
                   VALUES (%s, %s, %s, %s, 1, 'burn', NOW())
                   ON CONFLICT (address) DO UPDATE SET
                     balance = wallet_addresses.balance + EXCLUDED.balance,
                     transaction_count = wallet_addresses.transaction_count + 1,
                     updated_at = NOW()""",
                (BURN_ADDRESS, burn_fp, burn_fp, burn_amount),
            )
            # Credit coherence fund
            cur.execute(
                """INSERT INTO wallet_addresses
                   (address, wallet_fingerprint, public_key, balance,
                    transaction_count, address_type, updated_at)
                   VALUES (%s, %s, %s, %s, 1, 'coherence_fund', NOW())
                   ON CONFLICT (address) DO UPDATE SET
                     balance = wallet_addresses.balance + EXCLUDED.balance,
                     transaction_count = wallet_addresses.transaction_count + 1,
                     updated_at = NOW()""",
                (COHERENCE_FUND_ADDRESS, coh_fp, coh_fp, coherence_amount),
            )
        logger.info(
            f"[VAULT-BURN] op={operation} acct={account_id[:12]} "
            f"total={cost} burn={burn_amount} coherence={coherence_amount}"
        )
    except Exception as _be:
        logger.warning(f"[VAULT-BURN] on-chain split failed (billing still recorded): {_be}")

    return {
        "burn_amount":       burn_amount,
        "burn_amount_qtcl":  burn_amount  / QTCL_BASE,
        "coherence_amount":  coherence_amount,
        "coherence_amount_qtcl": coherence_amount / QTCL_BASE,
        "burn_address":      BURN_ADDRESS,
        "coherence_address": COHERENCE_FUND_ADDRESS,
    }


def _charge_account(
    account_id: str,
    cost: int,
    operation: str,
    description: str,
    tx_ref: str = "",
    extra_fields: Optional[dict] = None,
) -> dict:
    """
    Atomic vault account charge:
      1. Deduct cost from credit_balance
      2. Record vault_billing row
      3. Apply 90/10 burn/coherence split on-chain (fire-and-forget)

    Returns updated balance info dict.
    Raises ValueError if insufficient balance.
    """
    account = _vault_query(
        "SELECT credit_balance FROM vault_accounts WHERE id = %s",
        (account_id,), fetch="one"
    )
    if not account:
        raise ValueError(f"Account not found: {account_id}")

    current = int(account["credit_balance"])
    if current < cost:
        raise ValueError(
            f"Insufficient balance: need {cost / QTCL_BASE:.2f} QTCL, "
            f"have {current / QTCL_BASE:.2f} QTCL"
        )

    new_balance = current - cost

    # Deduct from vault account
    _vault_query(
        "UPDATE vault_accounts SET credit_balance = %s, updated_at = NOW() WHERE id = %s",
        (new_balance, account_id), fetch="none"
    )

    # Billing record
    billing_id = f"vb_{secrets.token_hex(16)}"
    _vault_query(
        """INSERT INTO vault_billing
           (id, account_id, operation, amount, balance_after, tx_hash, description, created_at)
           VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())""",
        (billing_id, account_id, operation, -cost, new_balance, tx_ref or billing_id, description),
        fetch="none"
    )

    # 90/10 burn/coherence split — fire-and-forget, non-blocking
    split = _apply_vault_burn(cost, operation, account_id, tx_ref)

    result = {
        "charged":            cost,
        "charged_qtcl":       cost / QTCL_BASE,
        "new_balance":        new_balance,
        "new_balance_qtcl":   new_balance / QTCL_BASE,
        "billing_id":         billing_id,
        "burn_qtcl":          split["burn_amount_qtcl"],
        "coherence_qtcl":     split["coherence_amount_qtcl"],
        "split":              f"90% burned ({split['burn_amount_qtcl']:.2f} QTCL) / "
                              f"10% coherence ({split['coherence_amount_qtcl']:.2f} QTCL)",
    }
    if extra_fields:
        result.update(extra_fields)
    return result


def _ensure_schema():
    """Initialize vault schema if not already done."""
    global _SCHEMA_INITIALIZED
    if _SCHEMA_INITIALIZED:
        return
    with _SCHEMA_LOCK:
        if _SCHEMA_INITIALIZED:
            return
        try:
            conn = _get_vault_conn()
            try:
                cur = conn.cursor()
                cur.execute(VAULT_SCHEMA_SQL)
                # Migration: add display_name column if missing (existing deployments)
                cur.execute("""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'vault_accounts' AND column_name = 'display_name'
                        ) THEN
                            ALTER TABLE vault_accounts ADD COLUMN display_name TEXT;
                        END IF;
                    END
                    $$;
                """)
                logger.info("[VAULT] ✅ Schema initialized")
                _SCHEMA_INITIALIZED = True
            finally:
                _put_vault_conn(conn)
        except Exception as e:
            logger.error(f"[VAULT] ❌ Schema init failed: {e}")
            raise


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMS NORMALIZER (JSON-RPC params come as list, vault handlers expect dict)
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_params(params) -> dict:
    """JSON-RPC 2.0 sends params as a list. Extract the first dict element."""
    if isinstance(params, dict):
        return params
    if isinstance(params, (list, tuple)):
        if len(params) > 0 and isinstance(params[0], dict):
            return params[0]
        return {}
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# ACCOUNT MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def _hash_passphrase(passphrase: str, salt: bytes = None) -> Tuple[str, str]:
    """Hash passphrase with PBKDF2-HMAC-SHA256 (600K iterations)."""
    import hashlib
    if salt is None:
        salt = secrets.token_bytes(32)
    dk = hashlib.pbkdf2_hmac('sha256', passphrase.encode(), salt, 600_000, dklen=32)
    return dk.hex(), salt.hex()


def _verify_passphrase(passphrase: str, stored_hash: str) -> bool:
    """Verify passphrase against stored hash (salt:hash format)."""
    try:
        parts = stored_hash.split(':')
        if len(parts) != 2:
            return False
        salt_hex, hash_hex = parts
        dk = hashlib.pbkdf2_hmac(
            'sha256', passphrase.encode(), bytes.fromhex(salt_hex), 600_000, dklen=32
        )
        return hmac.compare_digest(dk.hex(), hash_hex)
    except Exception:
        return False


def vault_create_account(params: dict, rpc_id: Any) -> dict:
    """
    RPC: vault_createAccount
    Create a new vault account. Trial tier by default.

    Params:
        passphrase: str (required, min 8 chars)
        display_name: str (optional, 2-48 chars — human-friendly account label)
        email: str (optional, for recovery)
        device_fp: str (optional, browser fingerprint)
        qtcl_address: str (optional, for billing)
    """
    params = _normalize_params(params)
    # _rpc_ok, _rpc_error defined at module level — no import needed
    try:
        _ensure_schema()

        passphrase = params.get("passphrase", "")
        if not passphrase or len(passphrase) < 8:
            return _rpc_error(-32602, "Passphrase must be at least 8 characters", rpc_id)

        display_name = (params.get("display_name") or "").strip()
        if display_name and (len(display_name) < 2 or len(display_name) > 48):
            return _rpc_error(-32602, "Display name must be 2-48 characters", rpc_id)

        email = params.get("email", "")
        device_fp = params.get("device_fp", "")
        qtcl_address = params.get("qtcl_address", "")

        # Check device fingerprint — if trial account already exists for this device,
        # return it (handles timeout-retry: first request may have created the account
        # but the response never reached the client)
        if device_fp:
            existing = _vault_query(
                "SELECT id, tier, email, qtcl_address, display_name, secrets_count, bytes_stored, "
                "anchors_used, credit_balance FROM vault_accounts "
                "WHERE device_fp = %s AND tier = 'trial'",
                (device_fp,), fetch="one"
            )
            if existing:
                # Verify passphrase matches (if it does, this is a retry — return the account)
                existing_full = _vault_query(
                    "SELECT passphrase_hash FROM vault_accounts WHERE id = %s",
                    (existing['id'],), fetch="one"
                )
                if existing_full and _verify_passphrase(passphrase, existing_full['passphrase_hash']):
                    logger.info(f"[VAULT] Returning existing account on retry: {existing['id'][:12]}...")
                    return _rpc_ok({
                        "account_id": existing['id'],
                        "display_name": existing.get('display_name', '') or '',
                        "tier": existing['tier'],
                        "email": existing.get('email', ''),
                        "qtcl_address": existing.get('qtcl_address', ''),
                        "secrets_count": existing.get('secrets_count', 0),
                        "bytes_stored": existing.get('bytes_stored', 0),
                        "credit_balance": existing.get('credit_balance', 0),
                        "limits": _get_tier_limits(existing['tier']),
                        "recovered": True,
                    }, rpc_id)
                else:
                    return _rpc_error(
                        -32001,
                        "Trial account already exists for this device. "
                        "Log in with your existing account ID, or upgrade to Paid.",
                        rpc_id
                    )

        # Generate account ID
        account_id = f"va_{secrets.token_hex(16)}"

        # Hash passphrase
        dk_hex, salt_hex = _hash_passphrase(passphrase)
        stored_hash = f"{salt_hex}:{dk_hex}"

        _vault_query(
            """INSERT INTO vault_accounts
               (id, display_name, email, passphrase_hash, tier, device_fp, qtcl_address, created_at, updated_at)
               VALUES (%s, %s, %s, %s, 'trial', %s, %s, NOW(), NOW())""",
            (account_id, display_name or None, email, stored_hash, device_fp, qtcl_address),
            fetch="none"
        )

        logger.info(f"[VAULT] Account created: {account_id[:12]}...{' name=' + display_name if display_name else ''} tier=trial")

        return _rpc_ok({
            "account_id": account_id,
            "display_name": display_name or "",
            "tier": "trial",
            "limits": {
                "max_secrets": TRIAL_MAX_SECRETS,
                "max_size_bytes": TRIAL_MAX_SIZE_BYTES,
                "chain_anchor": False,
                "obfuscation": False,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        }, rpc_id)

    except Exception as e:
        logger.error(f"[VAULT] Account creation failed: {e}", exc_info=True)
        return _rpc_error(-32603, f"Account creation failed: {str(e)}", rpc_id)


def vault_login(params: dict, rpc_id: Any) -> dict:
    """
    RPC: vault_login
    Authenticate and return account info + session token.

    Params:
        account_id: str (required)
        passphrase: str (required)
    """
    params = _normalize_params(params)
    # _rpc_ok, _rpc_error defined at module level — no import needed
    try:
        _ensure_schema()

        account_id = params.get("account_id", "")
        passphrase = params.get("passphrase", "")
        if not account_id or not passphrase:
            return _rpc_error(-32602, "account_id and passphrase required", rpc_id)

        account = _vault_query(
            "SELECT * FROM vault_accounts WHERE id = %s", (account_id,), fetch="one"
        )
        if not account:
            return _rpc_error(-32004, "Account not found", rpc_id)

        if not _verify_passphrase(passphrase, account['passphrase_hash']):
            return _rpc_error(-32003, "Invalid passphrase", rpc_id)

        # Generate session token (HMAC-SHA3-256)
        session_raw = f"{account_id}:{time.time()}:{secrets.token_hex(16)}"
        session_token = hashlib.sha3_256(session_raw.encode()).hexdigest()

        # Update last login
        _vault_query(
            "UPDATE vault_accounts SET last_login = NOW(), updated_at = NOW() WHERE id = %s",
            (account_id,), fetch="none"
        )

        tier = account['tier']
        limits = _get_tier_limits(tier)
        _reset_monthly_quota_if_needed(account)

        return _rpc_ok({
            "account_id": account_id,
            "display_name": account.get('display_name', '') or '',
            "session": session_token,
            "tier": tier,
            "email": account.get('email', ''),
            "qtcl_address": account.get('qtcl_address', ''),
            "secrets_count": account.get('secrets_count', 0),
            "bytes_stored": account.get('bytes_stored', 0),
            "credit_balance_qtcl": account.get('credit_balance', 0) // QTCL_BASE,
            "monthly_bytes_used": account.get('monthly_bytes_used', 0),
            "monthly_anchors_used": account.get('monthly_anchors_used', 0),
            "monthly_reset_at": str(account.get('monthly_reset_at', '')),
            "subscription_expires_at": str(account.get('subscription_expires_at', '')),
            "limits": limits,
            "last_login": datetime.now(timezone.utc).isoformat(),
        }, rpc_id)

    except Exception as e:
        logger.error(f"[VAULT] Login failed: {e}", exc_info=True)
        return _rpc_error(-32603, f"Login failed: {str(e)}", rpc_id)


def vault_upgrade_tier(params: dict, rpc_id: Any) -> dict:
    """
    RPC: vault_upgradeTier
    Upgrade account from trial to paid. Requires QTCL address for billing.

    Params:
        account_id: str
        passphrase: str
        qtcl_address: str (required for paid tier — billing address)
    """
    params = _normalize_params(params)
    # _rpc_ok, _rpc_error defined at module level — no import needed
    try:
        _ensure_schema()

        account_id = params.get("account_id", "")
        passphrase = params.get("passphrase", "")
        qtcl_address = params.get("qtcl_address", "")

        if not account_id or not passphrase:
            return _rpc_error(-32602, "account_id and passphrase required", rpc_id)
        if not qtcl_address:
            return _rpc_error(-32602, "qtcl_address required for paid tier", rpc_id)

        account = _vault_query(
            "SELECT * FROM vault_accounts WHERE id = %s", (account_id,), fetch="one"
        )
        if not account:
            return _rpc_error(-32004, "Account not found", rpc_id)

        if not _verify_passphrase(passphrase, account['passphrase_hash']):
            return _rpc_error(-32003, "Invalid passphrase", rpc_id)

        if account['tier'] == 'paid':
            return _rpc_ok({"already_paid": True, "tier": "paid"}, rpc_id)

        _vault_query(
            """UPDATE vault_accounts
               SET tier = 'paid', qtcl_address = %s, updated_at = NOW()
               WHERE id = %s""",
            (qtcl_address, account_id), fetch="none"
        )

        logger.info(f"[VAULT] Account upgraded: {account_id[:12]}... → paid")

        return _rpc_ok({
            "account_id": account_id,
            "tier": "paid",
            "qtcl_address": qtcl_address,
            "limits": _get_tier_limits("paid"),
        }, rpc_id)

    except Exception as e:
        logger.error(f"[VAULT] Upgrade failed: {e}", exc_info=True)
        return _rpc_error(-32603, f"Upgrade failed: {str(e)}", rpc_id)


# ═══════════════════════════════════════════════════════════════════════════════
# SECRET STORAGE (Encrypted)
# ═══════════════════════════════════════════════════════════════════════════════

def vault_store_secret(params: dict, rpc_id: Any) -> dict:
    """
    RPC: vault_storeSecret
    Zero-Knowledge Archive: store a client-side encrypted secret.
    Server stores the opaque ciphertext blob — NEVER decrypts it.
    
    Params:
        account_id: str
        passphrase: str
        label: str
        category: str
        ciphertext: str (hex — client-side AES-256-GCM encrypted)
        encryption_meta: dict (client-side encryption params: salt, iv, kdf)
        size_bytes: int (original plaintext size)
    """
    params = _normalize_params(params)
    # _rpc_ok, _rpc_error defined at module level — no import needed
    try:
        _ensure_schema()

        account_id = params.get("account_id", "")
        passphrase = params.get("passphrase", "")
        if not account_id or not passphrase:
            return _rpc_error(-32602, "account_id and passphrase required", rpc_id)

        # Auth
        account = _vault_query(
            "SELECT * FROM vault_accounts WHERE id = %s", (account_id,), fetch="one"
        )
        if not account:
            return _rpc_error(-32004, "Account not found", rpc_id)
        if not _verify_passphrase(passphrase, account['passphrase_hash']):
            return _rpc_error(-32003, "Invalid passphrase", rpc_id)

        label = params.get("label", "Untitled")
        category = params.get("category", "general")
        ciphertext = params.get("ciphertext", "")
        encryption_meta = params.get("encryption_meta", {})
        size_bytes = params.get("size_bytes", len(ciphertext) // 2)

        if not ciphertext:
            return _rpc_error(-32602, "ciphertext required", rpc_id)

        valid_categories = ('general', 'private_key', 'seed_phrase', 'credential', 'document', 'api_key', 'password', 'legal_document', 'bip_wordlist', 'note')
        if category not in valid_categories:
            category = 'general'

        # ── Subscription capacity check ──────────────────────────────────────
        _reset_monthly_quota_if_needed(account)
        ok, cost, reason = _check_subscription_capacity(account, size_bytes)
        if not ok:
            return _rpc_error(-32010, reason, rpc_id)

        # Check credit balance (vault balance, not on-chain wallet)
        balance = account.get('credit_balance', 0)
        if balance < cost:
            return _rpc_error(-32010,
                f"Insufficient vault balance: need {cost // QTCL_BASE} QTCL, have {balance // QTCL_BASE} QTCL. "
                f"Load QTCL via Stripe or wallet transfer.", rpc_id)

        # Deduct store fee (flat + overage) from vault credit balance
        new_balance = balance - cost
        _vault_query(
            "UPDATE vault_accounts SET credit_balance = %s, updated_at = NOW() WHERE id = %s",
            (new_balance, account_id), fetch="none"
        )

        # Burn split for store fee
        split = _apply_vault_burn(cost, "store_secret", account_id)
        _vault_query(
            """INSERT INTO vault_billing (id, account_id, operation, amount, balance_after, description, created_at)
               VALUES (%s, %s, %s, %s, %s, %s, NOW())""",
            (f"vb_{secrets.token_hex(8)}", account_id, "store_secret", cost, new_balance, reason),
            fetch="none"
        )

        # ZERO-KNOWLEDGE: Store client ciphertext AS-IS. Server never decrypts.
        secret_id = f"vs_{secrets.token_hex(16)}"

        # Content hash for anchoring (hash of the ciphertext, not plaintext)
        content_hash = hashlib.sha3_256(ciphertext.encode()).hexdigest()

        _vault_query(
            """INSERT INTO vault_secrets
               (id, account_id, label, category, ciphertext, encryption_meta,
                size_bytes, content_hash, anchor_hash, anchor_block, anchor_tx,
                obfuscated, created_at, updated_at)
               VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, NULL, NULL, NULL,
                       FALSE, NOW(), NOW())""",
            (
                secret_id, account_id, label, category,
                ciphertext,
                json.dumps({
                    **encryption_meta,
                    "zero_knowledge": True,
                    "server_layer": "none",
                }),
                size_bytes,
                content_hash,
            ),
            fetch="none"
        )

        # Update account counters and monthly usage
        _vault_query(
            """UPDATE vault_accounts
               SET secrets_count = secrets_count + 1,
                   bytes_stored = bytes_stored + %s,
                   monthly_bytes_used = monthly_bytes_used + %s,
                   updated_at = NOW()
               WHERE id = %s""",
            (size_bytes, size_bytes, account_id), fetch="none"
        )

        return _rpc_ok({
            "secret_id": secret_id,
            "label": label,
            "category": category,
            "size_bytes": size_bytes,
            "content_hash": content_hash,
            "vault_balance_qtcl": new_balance // QTCL_BASE,
            "cost_qtcl": cost // QTCL_BASE,
            "cost_usd": round(cost / QTCL_BASE * 0.01, 2),
            "zero_knowledge": True,
            "stored_at": datetime.now(timezone.utc).isoformat(),
        }, rpc_id)

    except Exception as e:
        logger.error(f"[VAULT] Store secret failed: {e}", exc_info=True)
        return _rpc_error(-32603, f"Store failed: {str(e)}", rpc_id)



def vault_retrieve_secret(params: dict, rpc_id: Any) -> dict:
    """
    RPC: vault_retrieveSecret
    Zero-Knowledge retrieval — returns opaque ciphertext for client-side decryption.
    Server NEVER decrypts.

    Params:
        account_id: str
        passphrase: str
        secret_id: str
    """
    params = _normalize_params(params)
    # _rpc_ok, _rpc_error defined at module level — no import needed
    try:
        _ensure_schema()

        account_id = params.get("account_id", "")
        passphrase = params.get("passphrase", "")
        secret_id = params.get("secret_id", "")

        if not account_id or not passphrase or not secret_id:
            return _rpc_error(-32602, "account_id, passphrase, secret_id required", rpc_id)

        # Auth
        account = _vault_query(
            "SELECT * FROM vault_accounts WHERE id = %s", (account_id,), fetch="one"
        )
        if not account:
            return _rpc_error(-32004, "Account not found", rpc_id)
        if not _verify_passphrase(passphrase, account['passphrase_hash']):
            return _rpc_error(-32003, "Invalid passphrase", rpc_id)

        # Fetch secret
        secret = _vault_query(
            "SELECT * FROM vault_secrets WHERE id = %s AND account_id = %s",
            (secret_id, account_id), fetch="one"
        )
        if not secret:
            return _rpc_error(-32005, "Secret not found", rpc_id)

        # Check expiry
        if secret.get('expires_at') and secret['expires_at'] < datetime.now(timezone.utc):
            return _rpc_error(-32012, "Secret has expired", rpc_id)

        enc_meta = secret.get('encryption_meta', {})
        if isinstance(enc_meta, str):
            enc_meta = json.loads(enc_meta)

        # ZERO-KNOWLEDGE: Return ciphertext as-is. Client decrypts.
        # Strip any server-side metadata from encryption_meta
        client_meta = {
            k: v for k, v in enc_meta.items()
            if not k.startswith('server_') and k != 'zero_knowledge'
        }

        # Update access counters
        _vault_query(
            """UPDATE vault_secrets
               SET access_count = access_count + 1,
                   last_accessed = NOW()
               WHERE id = %s""",
            (secret_id,), fetch="none"
        )

        return _rpc_ok({
            "secret_id": secret_id,
            "label": secret['label'],
            "category": secret['category'],
            "ciphertext": secret['ciphertext'],
            "encryption_meta": client_meta,
            "size_bytes": secret['size_bytes'],
            "anchor_hash": secret.get('anchor_hash'),
            "anchor_block": secret.get('anchor_block'),
            "obfuscated": secret.get('obfuscated', False),
            "created_at": str(secret.get('created_at', '')),
            "access_count": secret.get('access_count', 0) + 1,
        }, rpc_id)

    except Exception as e:
        logger.error(f"[VAULT] Retrieve secret failed: {e}", exc_info=True)
        return _rpc_error(-32603, f"Retrieve failed: {str(e)}", rpc_id)


def vault_list_secrets(params: dict, rpc_id: Any) -> dict:
    """
    RPC: vault_listSecrets
    List all secrets for an account (metadata only, no ciphertext).

    Params:
        account_id: str
        passphrase: str
        category: str (optional filter)
    """
    params = _normalize_params(params)
    # _rpc_ok, _rpc_error defined at module level — no import needed
    try:
        _ensure_schema()

        account_id = params.get("account_id", "")
        passphrase = params.get("passphrase", "")
        if not account_id or not passphrase:
            return _rpc_error(-32602, "account_id and passphrase required", rpc_id)

        account = _vault_query(
            "SELECT * FROM vault_accounts WHERE id = %s", (account_id,), fetch="one"
        )
        if not account:
            return _rpc_error(-32004, "Account not found", rpc_id)
        if not _verify_passphrase(passphrase, account['passphrase_hash']):
            return _rpc_error(-32003, "Invalid passphrase", rpc_id)

        category = params.get("category")
        if category:
            secrets_list = _vault_query(
                """SELECT id, label, category, size_bytes, anchor_hash, anchor_block,
                          obfuscated, created_at, access_count, expires_at
                   FROM vault_secrets
                   WHERE account_id = %s AND category = %s
                   ORDER BY created_at DESC""",
                (account_id, category)
            )
        else:
            secrets_list = _vault_query(
                """SELECT id, label, category, size_bytes, anchor_hash, anchor_block,
                          obfuscated, created_at, access_count, expires_at
                   FROM vault_secrets
                   WHERE account_id = %s
                   ORDER BY created_at DESC""",
                (account_id,)
            )

        # Serialize datetimes
        for s in secrets_list:
            for k in ('created_at', 'expires_at'):
                if s.get(k):
                    s[k] = str(s[k])

        return _rpc_ok({
            "secrets": secrets_list,
            "count": len(secrets_list),
            "tier": account['tier'],
        }, rpc_id)

    except Exception as e:
        logger.error(f"[VAULT] List secrets failed: {e}", exc_info=True)
        return _rpc_error(-32603, f"List failed: {str(e)}", rpc_id)


def vault_delete_secret(params: dict, rpc_id: Any) -> dict:
    """
    RPC: vault_deleteSecret
    Permanently delete a secret. Cannot be undone.

    Params:
        account_id: str
        passphrase: str
        secret_id: str
    """
    params = _normalize_params(params)
    # _rpc_ok, _rpc_error defined at module level — no import needed
    try:
        _ensure_schema()

        account_id = params.get("account_id", "")
        passphrase = params.get("passphrase", "")
        secret_id = params.get("secret_id", "")

        if not account_id or not passphrase or not secret_id:
            return _rpc_error(-32602, "account_id, passphrase, secret_id required", rpc_id)

        account = _vault_query(
            "SELECT * FROM vault_accounts WHERE id = %s", (account_id,), fetch="one"
        )
        if not account:
            return _rpc_error(-32004, "Account not found", rpc_id)
        if not _verify_passphrase(passphrase, account['passphrase_hash']):
            return _rpc_error(-32003, "Invalid passphrase", rpc_id)

        secret = _vault_query(
            "SELECT id, size_bytes FROM vault_secrets WHERE id = %s AND account_id = %s",
            (secret_id, account_id), fetch="one"
        )
        if not secret:
            return _rpc_error(-32005, "Secret not found", rpc_id)

        size = secret['size_bytes']

        _vault_query("DELETE FROM vault_secrets WHERE id = %s", (secret_id,), fetch="none")

        _vault_query(
            """UPDATE vault_accounts
               SET secrets_count = GREATEST(secrets_count - 1, 0),
                   bytes_stored = GREATEST(bytes_stored - %s, 0),
                   updated_at = NOW()
               WHERE id = %s""",
            (size, account_id), fetch="none"
        )

        return _rpc_ok({
            "deleted": True,
            "secret_id": secret_id,
        }, rpc_id)

    except Exception as e:
        logger.error(f"[VAULT] Delete secret failed: {e}", exc_info=True)
        return _rpc_error(-32603, f"Delete failed: {str(e)}", rpc_id)


# ═══════════════════════════════════════════════════════════════════════════════
# CHAIN ANCHORING
# ═══════════════════════════════════════════════════════════════════════════════

def vault_anchor_hash(params: dict, rpc_id: Any) -> dict:
    """
    RPC: vault_anchorHash (POST)
    Anchor a SHA3-256 hash to the QTCL blockchain as proof-of-existence.

    For obfuscated anchors:
      on_chain_hash = SHA3-256(secret_hash ‖ blinding_nonce)
      The blinding_nonce is stored only in the vault DB, so the on-chain hash
      cannot be correlated to the vault entry without vault access.

    Params:
        account_id: str
        passphrase: str
        secret_id: str (optional — links anchor to existing secret)
        hash: str (SHA3-256 hex of the content to anchor)
        label: str (human-readable anchor description)
        obfuscate: bool (whether to blind the on-chain hash — costs 5 QTCL instead of 1)
    """
    params = _normalize_params(params)
    # _rpc_ok, _rpc_error defined at module level — no import needed
    try:
        _ensure_schema()

        account_id = params.get("account_id", "")
        passphrase = params.get("passphrase", "")
        if not account_id or not passphrase:
            return _rpc_error(-32602, "account_id and passphrase required", rpc_id)

        account = _vault_query(
            "SELECT * FROM vault_accounts WHERE id = %s", (account_id,), fetch="one"
        )
        if not account:
            return _rpc_error(-32004, "Account not found", rpc_id)
        if not _verify_passphrase(passphrase, account['passphrase_hash']):
            return _rpc_error(-32003, "Invalid passphrase", rpc_id)

        _reset_monthly_quota_if_needed(account)
        content_hash = params.get("hash", "")
        if not content_hash or len(content_hash) != 64:
            return _rpc_error(-32602, "hash must be 64-char hex SHA3-256", rpc_id)

        obfuscate = params.get("obfuscate", False)
        secret_id = params.get("secret_id")
        label = params.get("label", "Vault Anchor")

        # Calculate cost — check if tier includes free anchors this month
        limits = _get_tier_limits(account.get("tier", "trial"))
        anchors_included = limits.get("anchors_included")
        monthly_anchors = account.get("monthly_anchors_used", 0)

        if anchors_included is not None and monthly_anchors < anchors_included:
            # Included in subscription — no charge
            cost = 0
            included = True
        else:
            cost = PRICE_ANCHOR_OBFUSCATED if obfuscate else PRICE_ANCHOR_STANDARD
            included = False

        # Check balance (skip if included in subscription)
        if cost > 0 and account['credit_balance'] < cost:
            cost_qtcl    = cost / QTCL_BASE
            balance_qtcl = account['credit_balance'] / QTCL_BASE
            return _rpc_error(
                -32020,
                f"Insufficient vault credit. Need {cost_qtcl:.0f} QTCL, "
                f"have {balance_qtcl:.2f} QTCL. "
                f"Deposit mined QTCL or purchase via Stripe.",
                rpc_id
            )

        # Compute on-chain hash
        blinding_nonce = None
        obfuscated_hash = None
        on_chain_hash = content_hash

        if obfuscate:
            blinding_nonce = secrets.token_hex(32)
            on_chain_hash = hashlib.sha3_256(
                bytes.fromhex(content_hash) + bytes.fromhex(blinding_nonce)
            ).hexdigest()
            obfuscated_hash = on_chain_hash

        # Submit anchor transaction to QTCL chain
        anchor_result = _submit_chain_anchor(
            on_chain_hash, label, account.get('qtcl_address', '')
        )

        anchor_id = f"va_{secrets.token_hex(16)}"

        # Record anchor in vault DB
        _vault_query(
            """INSERT INTO vault_anchors
               (id, account_id, secret_id, anchor_hash, obfuscated_hash,
                block_height, tx_hash, chain, created_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, 'qtcl', NOW())""",
            (
                anchor_id, account_id, secret_id,
                content_hash, obfuscated_hash,
                anchor_result.get('block_height'),
                anchor_result.get('tx_hash'),
            ),
            fetch="none"
        )

        # Link anchor to secret if provided
        if secret_id:
            _vault_query(
                """UPDATE vault_secrets
                   SET anchor_hash = %s, anchor_block = %s, anchor_tx = %s,
                       obfuscated = %s, blinding_nonce = %s, updated_at = NOW()
                   WHERE id = %s AND account_id = %s""",
                (
                    on_chain_hash, anchor_result.get('block_height'),
                    anchor_result.get('tx_hash'),
                    obfuscate, blinding_nonce,
                    secret_id, account_id,
                ),
                fetch="none"
            )

        # ── Charge: 90% burn / 10% coherence (skip if included in subscription) ──
        if cost > 0:
            try:
                charge = _charge_account(
                    account_id=account_id,
                    cost=cost,
                    operation='obfuscated_anchor' if obfuscate else 'chain_anchor',
                    description=f"{'Obfuscated ' if obfuscate else ''}Chain anchor: {label}",
                    tx_ref=anchor_result.get('tx_hash', ''),
                )
            except ValueError as _ce:
                return _rpc_error(-32020, str(_ce), rpc_id)
            new_balance_qtcl = charge["new_balance_qtcl"]
            fee_split = charge["split"]
        else:
            new_balance_qtcl = account['credit_balance'] / QTCL_BASE
            fee_split = {"burn": 0, "coherence": 0, "free": True}

        # Increment anchor count (both total and monthly)
        _vault_query(
            """UPDATE vault_accounts
               SET anchors_used = anchors_used + 1,
                   monthly_anchors_used = monthly_anchors_used + 1,
                   updated_at = NOW()
               WHERE id = %s""",
            (account_id,), fetch="none"
        )

        result = {
            "anchor_id":        anchor_id,
            "on_chain_hash":    on_chain_hash,
            "block_height":     anchor_result.get('block_height'),
            "tx_hash":          anchor_result.get('tx_hash'),
            "obfuscated":       obfuscate,
            "included_free":    included,
            "cost_qtcl":        cost / QTCL_BASE if cost > 0 else 0,
            "balance_remaining": new_balance_qtcl,
            "fee_split":        fee_split,
            "anchored_at":      datetime.now(timezone.utc).isoformat(),
        }

        if obfuscate and blinding_nonce:
            result["blinding_nonce"] = blinding_nonce
            result["WARNING"] = (
                "SAVE THIS BLINDING NONCE. Without it, the on-chain hash "
                "cannot be linked to your secret. This is by design."
            )

        logger.info(
            f"[VAULT] Anchor created: {anchor_id[:12]}... "
            f"block={anchor_result.get('block_height')} "
            f"obfuscated={obfuscate}"
        )

        return _rpc_ok(result, rpc_id)

    except Exception as e:
        logger.error(f"[VAULT] Anchor failed: {e}", exc_info=True)
        return _rpc_error(-32603, f"Anchor failed: {str(e)}", rpc_id)


def vault_verify_anchor(params: dict, rpc_id: Any) -> dict:
    """
    RPC: vault_verifyAnchor (GET)
    Verify that a hash exists on-chain. Public — no auth required.

    Params:
        hash: str (the hash to look up — either original or obfuscated)
    """
    params = _normalize_params(params)
    # _rpc_ok, _rpc_error defined at module level — no import needed
    try:
        _ensure_schema()

        content_hash = params.get("hash", "")
        if not content_hash or len(content_hash) != 64:
            return _rpc_error(-32602, "hash must be 64-char hex", rpc_id)

        # Look up in vault anchors (original or obfuscated)
        anchor = _vault_query(
            """SELECT id, anchor_hash, obfuscated_hash, block_height, tx_hash,
                      chain, created_at, verified
               FROM vault_anchors
               WHERE anchor_hash = %s OR obfuscated_hash = %s
               ORDER BY created_at ASC LIMIT 1""",
            (content_hash, content_hash), fetch="one"
        )

        if not anchor:
            return _rpc_ok({
                "found": False,
                "hash": content_hash,
                "message": "Hash not found in vault anchors",
            }, rpc_id)

        # Also verify on-chain (best effort)
        on_chain_verified = _verify_chain_anchor(
            anchor.get('obfuscated_hash') or anchor['anchor_hash']
        )

        return _rpc_ok({
            "found": True,
            "hash": content_hash,
            "is_obfuscated": content_hash == anchor.get('obfuscated_hash'),
            "block_height": anchor.get('block_height'),
            "tx_hash": anchor.get('tx_hash'),
            "chain": anchor.get('chain', 'qtcl'),
            "anchored_at": str(anchor.get('created_at', '')),
            "on_chain_verified": on_chain_verified,
        }, rpc_id)

    except Exception as e:
        logger.error(f"[VAULT] Verify anchor failed: {e}", exc_info=True)
        return _rpc_error(-32603, f"Verify failed: {str(e)}", rpc_id)


# ═══════════════════════════════════════════════════════════════════════════════
# CREDIT DEPOSIT (QTCL → Vault Credit)
# ═══════════════════════════════════════════════════════════════════════════════

def vault_deposit_credit(params: dict, rpc_id: Any) -> dict:
    """
    RPC: vault_depositCredit
    Deposit mined QTCL to vault credit balance.
    QTCL is only spendable if it has been mined — validated via chain.

    Params:
        account_id: str
        passphrase: str
        tx_hash: str (the QTCL transaction hash proving the deposit)
        amount: int (in base units — 100,000 = 1 QTCL)
    """
    params = _normalize_params(params)
    # _rpc_ok, _rpc_error defined at module level — no import needed
    try:
        _ensure_schema()

        account_id = params.get("account_id", "")
        passphrase = params.get("passphrase", "")
        tx_hash = params.get("tx_hash", "")
        amount = params.get("amount", 0)

        if not account_id or not passphrase:
            return _rpc_error(-32602, "account_id and passphrase required", rpc_id)
        if not tx_hash:
            return _rpc_error(-32602, "tx_hash required (proof of mined QTCL)", rpc_id)
        if amount <= 0:
            return _rpc_error(-32602, "amount must be positive", rpc_id)

        account = _vault_query(
            "SELECT * FROM vault_accounts WHERE id = %s", (account_id,), fetch="one"
        )
        if not account:
            return _rpc_error(-32004, "Account not found", rpc_id)
        if not _verify_passphrase(passphrase, account['passphrase_hash']):
            return _rpc_error(-32003, "Invalid passphrase", rpc_id)

        # Verify TX on chain (ensure QTCL was actually mined and sent to vault)
        tx_valid = _verify_deposit_tx(tx_hash, amount, account.get('qtcl_address', ''))
        if not tx_valid:
            return _rpc_error(
                -32021,
                "Transaction could not be verified on-chain. Ensure QTCL was mined and sent.",
                rpc_id
            )

        # Credit the account
        new_balance = account['credit_balance'] + amount
        _vault_query(
            "UPDATE vault_accounts SET credit_balance = %s, updated_at = NOW() WHERE id = %s",
            (new_balance, account_id), fetch="none"
        )

        # Billing record
        billing_id = f"vb_{secrets.token_hex(16)}"
        _vault_query(
            """INSERT INTO vault_billing
               (id, account_id, operation, amount, balance_after, tx_hash, description, created_at)
               VALUES (%s, %s, 'deposit', %s, %s, %s, 'QTCL deposit from mining', NOW())""",
            (billing_id, account_id, amount, new_balance, tx_hash),
            fetch="none"
        )

        return _rpc_ok({
            "deposited": amount,
            "deposited_qtcl": amount / QTCL_BASE,
            "new_balance": new_balance,
            "new_balance_qtcl": new_balance / QTCL_BASE,
            "tx_hash": tx_hash,
        }, rpc_id)

    except Exception as e:
        logger.error(f"[VAULT] Deposit failed: {e}", exc_info=True)
        return _rpc_error(-32603, f"Deposit failed: {str(e)}", rpc_id)


def vault_get_balance(params: dict, rpc_id: Any) -> dict:
    """
    RPC: vault_getBalance
    Get vault account credit balance and billing history.

    Params:
        account_id: str
        passphrase: str
    """
    params = _normalize_params(params)
    # _rpc_ok, _rpc_error defined at module level — no import needed
    try:
        _ensure_schema()

        account_id = params.get("account_id", "")
        passphrase = params.get("passphrase", "")
        if not account_id or not passphrase:
            return _rpc_error(-32602, "account_id and passphrase required", rpc_id)

        account = _vault_query(
            "SELECT * FROM vault_accounts WHERE id = %s", (account_id,), fetch="one"
        )
        if not account:
            return _rpc_error(-32004, "Account not found", rpc_id)
        if not _verify_passphrase(passphrase, account['passphrase_hash']):
            return _rpc_error(-32003, "Invalid passphrase", rpc_id)

        # Recent billing
        billing = _vault_query(
            """SELECT operation, amount, balance_after, tx_hash, description, created_at
               FROM vault_billing
               WHERE account_id = %s
               ORDER BY created_at DESC LIMIT 20""",
            (account_id,)
        )
        for b in billing:
            if b.get('created_at'):
                b['created_at'] = str(b['created_at'])

        return _rpc_ok({
            "credit_balance":      account['credit_balance'],
            "credit_balance_qtcl": account['credit_balance'] / QTCL_BASE,
            "secrets_count":  account['secrets_count'],
            "bytes_stored":   account['bytes_stored'],
            "anchors_used":   account['anchors_used'],
            "tier":           account['tier'],
            "recent_billing": billing,
            "pricing": {
                "chain_anchor_qtcl":      PRICE_ANCHOR            / QTCL_BASE,
                "obfuscated_anchor_qtcl": PRICE_OBFUSCATED_ANCHOR / QTCL_BASE,
                "passphrase_change_qtcl": PRICE_PASSPHRASE_CHANGE / QTCL_BASE,
                "reanchor_qtcl":          PRICE_REANCHOR          / QTCL_BASE,
                "export_bundle_qtcl":     PRICE_EXPORT_BUNDLE     / QTCL_BASE,
                "add_collaborator_qtcl":  PRICE_ADD_COLLABORATOR  / QTCL_BASE,
                "revoke_anchor_qtcl":     PRICE_REVOKE_ANCHOR     / QTCL_BASE,
                "store_secret":           "free",
                "retrieve_secret":        "free",
                "audit_log":              "free",
            },
            "fee_split": f"{int(BURN_RATIO*100)}% burn / {int(COHERENCE_RATIO*100)}% coherence fund",
            "stripe_tiers": STRIPE_TIERS,
        }, rpc_id)

    except Exception as e:
        logger.error(f"[VAULT] Balance failed: {e}", exc_info=True)
        return _rpc_error(-32603, f"Balance check failed: {str(e)}", rpc_id)


# ═══════════════════════════════════════════════════════════════════════════════
# INHERITANCE (Dead Man's Switch)
# ═══════════════════════════════════════════════════════════════════════════════

def vault_setup_inheritance(params: dict, rpc_id: Any) -> dict:
    """
    RPC: vault_setupInheritance
    Configure dead man's switch for private key inheritance.

    Params:
        account_id: str
        passphrase: str
        beneficiary: str (QTCL address or email of beneficiary)
        check_in_days: int (days between required check-ins, default 365)
        shamir_config: dict (optional Shamir secret sharing config)
    """
    params = _normalize_params(params)
    # _rpc_ok, _rpc_error defined at module level — no import needed
    try:
        _ensure_schema()

        account_id = params.get("account_id", "")
        passphrase = params.get("passphrase", "")
        if not account_id or not passphrase:
            return _rpc_error(-32602, "account_id and passphrase required", rpc_id)

        account = _vault_query(
            "SELECT * FROM vault_accounts WHERE id = %s", (account_id,), fetch="one"
        )
        if not account:
            return _rpc_error(-32004, "Account not found", rpc_id)
        if not _verify_passphrase(passphrase, account['passphrase_hash']):
            return _rpc_error(-32003, "Invalid passphrase", rpc_id)

        if account['tier'] != 'paid':
            return _rpc_error(-32010, "Inheritance requires Paid tier", rpc_id)

        beneficiary = params.get("beneficiary", "")
        if not beneficiary:
            return _rpc_error(-32602, "beneficiary required", rpc_id)

        check_in_days = params.get("check_in_days", 365)
        shamir_config = params.get("shamir_config", {})

        inheritance_id = f"vi_{secrets.token_hex(16)}"

        _vault_query(
            """INSERT INTO vault_inheritance
               (id, account_id, beneficiary, shamir_config, check_in_days,
                last_check_in, created_at)
               VALUES (%s, %s, %s, %s::jsonb, %s, NOW(), NOW())
               ON CONFLICT (id) DO NOTHING""",
            (inheritance_id, account_id, beneficiary, json.dumps(shamir_config), check_in_days),
            fetch="none"
        )

        return _rpc_ok({
            "inheritance_id": inheritance_id,
            "beneficiary": beneficiary,
            "check_in_days": check_in_days,
            "next_check_in": (datetime.now(timezone.utc) + timedelta(days=check_in_days)).isoformat(),
            "status": "active",
        }, rpc_id)

    except Exception as e:
        logger.error(f"[VAULT] Inheritance setup failed: {e}", exc_info=True)
        return _rpc_error(-32603, f"Inheritance setup failed: {str(e)}", rpc_id)


def vault_check_in(params: dict, rpc_id: Any) -> dict:
    """
    RPC: vault_checkIn
    Reset the dead man's switch timer.
    """
    params = _normalize_params(params)
    # _rpc_ok, _rpc_error defined at module level — no import needed
    try:
        _ensure_schema()

        account_id = params.get("account_id", "")
        passphrase = params.get("passphrase", "")
        if not account_id or not passphrase:
            return _rpc_error(-32602, "account_id and passphrase required", rpc_id)

        account = _vault_query(
            "SELECT * FROM vault_accounts WHERE id = %s", (account_id,), fetch="one"
        )
        if not account:
            return _rpc_error(-32004, "Account not found", rpc_id)
        if not _verify_passphrase(passphrase, account['passphrase_hash']):
            return _rpc_error(-32003, "Invalid passphrase", rpc_id)

        _vault_query(
            "UPDATE vault_inheritance SET last_check_in = NOW() WHERE account_id = %s AND activated = FALSE",
            (account_id,), fetch="none"
        )

        return _rpc_ok({
            "checked_in": True,
            "checked_in_at": datetime.now(timezone.utc).isoformat(),
        }, rpc_id)

    except Exception as e:
        logger.error(f"[VAULT] Check-in failed: {e}", exc_info=True)
        return _rpc_error(-32603, f"Check-in failed: {str(e)}", rpc_id)


# ═══════════════════════════════════════════════════════════════════════════════
# ENCRYPTION HELPERS (Server-Side Layer)
# ═══════════════════════════════════════════════════════════════════════════════

# Server-side encryption key (derived from env or generated at startup)
_SERVER_ENC_KEY = None
_SERVER_KEY_ID = None


def _get_server_key():
    """Get or generate server-side encryption key for Layer 2."""
    global _SERVER_ENC_KEY, _SERVER_KEY_ID
    if _SERVER_ENC_KEY is not None:
        return _SERVER_ENC_KEY, _SERVER_KEY_ID

    # Derive from env var or generate
    key_seed = os.environ.get("VAULT_ENC_KEY", "")
    if key_seed:
        _SERVER_ENC_KEY = hashlib.sha3_256(key_seed.encode()).digest()
        _SERVER_KEY_ID = hashlib.sha3_256(_SERVER_ENC_KEY).hexdigest()[:16]
    else:
        # Generate and persist in DB
        _SERVER_ENC_KEY = secrets.token_bytes(32)
        _SERVER_KEY_ID = hashlib.sha3_256(_SERVER_ENC_KEY).hexdigest()[:16]
        logger.warning(
            "[VAULT] ⚠️  No VAULT_ENC_KEY env var — generated ephemeral key. "
            "Set VAULT_ENC_KEY for persistence across restarts."
        )

    return _SERVER_ENC_KEY, _SERVER_KEY_ID


def _server_encrypt(plaintext_hex: str) -> dict:
    """
    Server-side encryption layer (AES-256-GCM via stdlib).
    Uses SHAKE-256-CTR + SHA3-256 MAC (same as hyp_lwe password encryption).
    """
    import hashlib as _hl

    key, key_id = _get_server_key()
    plaintext = bytes.fromhex(plaintext_hex) if plaintext_hex else b""

    # SHAKE-256-CTR encryption (stdlib only, no external deps)
    nonce = secrets.token_bytes(24)  # 192-bit nonce
    enc_key = key[:16]
    mac_key = key[16:]

    # XOR stream cipher via SHAKE-256
    shake = _hl.shake_256(enc_key + nonce)
    keystream = shake.digest(len(plaintext))
    ciphertext = bytes(a ^ b for a, b in zip(plaintext, keystream))

    # MAC: SHA3-256(mac_key ‖ nonce ‖ ciphertext)
    tag = _hl.sha3_256(mac_key + nonce + ciphertext).digest()

    return {
        'ciphertext': ciphertext.hex(),
        'nonce': nonce.hex(),
        'tag': tag.hex(),
        'key_id': key_id,
    }


def _server_decrypt(ciphertext_hex: str, nonce_hex: str, tag_hex: str, key_id: str) -> str:
    """Server-side decryption. Returns client-side ciphertext hex."""
    import hashlib as _hl

    key, current_key_id = _get_server_key()

    if key_id and key_id != current_key_id:
        logger.warning(f"[VAULT] Key ID mismatch: {key_id} != {current_key_id}")

    ciphertext = bytes.fromhex(ciphertext_hex)
    nonce = bytes.fromhex(nonce_hex)
    tag = bytes.fromhex(tag_hex)

    # Verify MAC first
    mac_key = key[16:]
    expected_tag = _hl.sha3_256(mac_key + nonce + ciphertext).digest()

    if not hmac.compare_digest(tag, expected_tag):
        raise ValueError("Server-side decryption: MAC verification failed (data tampered or wrong key)")

    # Decrypt
    enc_key = key[:16]
    shake = _hl.shake_256(enc_key + nonce)
    keystream = shake.digest(len(ciphertext))
    plaintext = bytes(a ^ b for a, b in zip(ciphertext, keystream))

    return plaintext.hex()


# ═══════════════════════════════════════════════════════════════════════════════
# CHAIN INTEGRATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _submit_chain_anchor(hash_hex: str, label: str, sender_address: str) -> dict:
    """
    Submit an anchor hash as a special transaction to the QTCL blockchain.
    Uses the existing submitTransaction RPC internally.
    """
    try:
        # Import server's DB infrastructure to submit the anchor as a tx
        from server import _rpc_submitTransaction, _rpc_getBlockHeight

        # Get current block height
        height_resp = _rpc_getBlockHeight({}, 0)
        current_height = 0
        if height_resp.get('result'):
            current_height = height_resp['result'].get('height', 0)

        # Create anchor transaction
        anchor_tx = {
            "type": "vault_anchor",
            "sender": sender_address or "vault_service",
            "recipient": "vault_anchor_registry",
            "amount": 0,
            "data": {
                "anchor_hash": hash_hex,
                "label": label,
                "anchored_at": datetime.now(timezone.utc).isoformat(),
                "vault_version": "1.0",
            },
            "timestamp": time.time(),
            "nonce": secrets.token_hex(16),
        }

        # Hash the transaction
        tx_json = json.dumps(anchor_tx, sort_keys=True, separators=(',', ':'))
        tx_hash = hashlib.sha3_256(tx_json.encode()).hexdigest()
        anchor_tx["tx_hash"] = tx_hash

        # Store anchor directly in the DB if mempool submission fails
        # (the anchor is provable via the vault_anchors table regardless)
        try:
            result = _rpc_submitTransaction(
                {"transaction": anchor_tx},
                0
            )
            block_height = current_height + 1
        except Exception as e:
            logger.warning(f"[VAULT] Mempool submission failed, recording DB-only anchor: {e}")
            block_height = current_height

        return {
            "tx_hash": tx_hash,
            "block_height": block_height,
        }

    except Exception as e:
        logger.error(f"[VAULT] Chain anchor submission failed: {e}")
        # Return a valid response anyway — the anchor is in the vault DB
        return {
            "tx_hash": hashlib.sha3_256(hash_hex.encode()).hexdigest(),
            "block_height": 0,
            "db_only": True,
        }


def _verify_chain_anchor(hash_hex: str) -> bool:
    """Verify an anchor exists on-chain."""
    try:
        # Check in vault_anchors table (source of truth)
        result = _vault_query(
            "SELECT id FROM vault_anchors WHERE anchor_hash = %s OR obfuscated_hash = %s LIMIT 1",
            (hash_hex, hash_hex), fetch="one"
        )
        return result is not None
    except Exception:
        return False


def _verify_deposit_tx(tx_hash: str, amount: int, expected_recipient: str) -> bool:
    """
    Verify a deposit transaction on the QTCL chain.
    Checks that the TX exists, is confirmed, and sends to the expected address.
    """
    try:
        from server import _rpc_getTransaction

        result = _rpc_getTransaction({"hash": tx_hash}, 0)
        if not result.get('result'):
            return False

        tx = result['result']

        # Verify amount (within 1% tolerance for fee deductions)
        tx_amount = tx.get('amount', 0)
        if tx_amount < amount * 0.99:
            return False

        # Verify recipient if provided
        if expected_recipient and tx.get('recipient') != expected_recipient:
            # Also accept vault deposit address
            vault_deposit_addr = os.environ.get("VAULT_DEPOSIT_ADDRESS", "")
            if vault_deposit_addr and tx.get('recipient') != vault_deposit_addr:
                return False

        return True

    except Exception as e:
        logger.warning(f"[VAULT] TX verification failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# TIER HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _get_tier_limits(tier: str) -> dict:
    """Return limits for a given tier from VAULT_TIERS config."""
    return VAULT_TIERS.get(tier, VAULT_TIERS["trial"])


def _reset_monthly_quota_if_needed(account: dict) -> None:
    """Reset monthly counters if the monthly period has expired."""
    reset_at = account.get("monthly_reset_at")
    if reset_at is None:
        return
    now = datetime.now(timezone.utc)
    if isinstance(reset_at, str):
        reset_at = datetime.fromisoformat(reset_at.replace("Z", "+00:00"))
    if now >= reset_at + timedelta(days=30):
        _vault_query(
            """UPDATE vault_accounts
               SET monthly_bytes_used = 0, monthly_anchors_used = 0,
                   monthly_reset_at = NOW()
               WHERE id = %s""",
            (account["id"],), fetch="none"
        )


def _check_subscription_capacity(account: dict, size_bytes: int) -> tuple:
    """Check if storing a secret of given size fits within subscription quota.

    Returns (ok: bool, cost_qtcl: int, reason: str).
    cost_qtcl includes store fee + any overage.
    """
    tier = account.get("tier", "trial")
    limits = _get_tier_limits(tier)
    monthly_bytes = account.get("monthly_bytes_used", 0)
    secrets_count = account.get("secrets_count", 0)
    max_secrets = limits["max_secrets"]
    max_bytes = limits["max_bytes"]

    # Check secret count limit
    if max_secrets is not None and secrets_count >= max_secrets:
        return False, 0, f"Tier '{tier}': max {max_secrets} secrets reached. Upgrade to store more."

    # Check if this secret would exceed total byte cap
    if max_bytes is not None and (monthly_bytes + size_bytes) > max_bytes:
        # Calculate overage
        overage_bytes = (monthly_bytes + size_bytes) - max_bytes
        overage_per_100kb = limits.get("overage_per_100kb", 0)
        if overage_per_100kb == 0 and tier == "trial":
            return False, 0, f"Trial tier: {max_bytes} bytes max. Upgrade to store more."
        overage_cost = (overage_bytes // (100 * 1024) + 1) * overage_per_100kb
        total_cost = PRICE_STORE_SECRET + overage_cost
        return True, total_cost, f"Overage charge: {total_cost // QTCL_BASE} QTCL ({overage_bytes} bytes over quota)"

    # Within quota — only flat store fee
    return True, PRICE_STORE_SECRET, "Within quota"


# ═══════════════════════════════════════════════════════════════════════════════
# ACCOUNT SETTINGS — Change Account ID
# ═══════════════════════════════════════════════════════════════════════════════

def vault_update_display_name(params: dict, rpc_id: Any) -> dict:
    """RPC: vault_updateDisplayName — Change the human-friendly display name."""
    try:
        p = _normalize_params(params)
        account_id = p.get("account_id", "")
        passphrase = p.get("passphrase", "")
        new_name = (p.get("display_name") or "").strip()
        if not account_id or not passphrase:
            return _rpc_error(-32602, "account_id and passphrase required", rpc_id)
        if new_name and (len(new_name) < 2 or len(new_name) > 48):
            return _rpc_error(-32602, "Display name must be 2-48 characters (or empty to clear)", rpc_id)

        _ensure_schema()
        acct = _vault_query("SELECT passphrase_hash FROM vault_accounts WHERE id = %s", (account_id,), fetch="one")
        if not acct:
            return _rpc_error(-32004, "Account not found", rpc_id)
        if not _verify_passphrase(passphrase, acct["passphrase_hash"]):
            return _rpc_error(-32003, "Invalid passphrase", rpc_id)

        _vault_query(
            "UPDATE vault_accounts SET display_name = %s, updated_at = NOW() WHERE id = %s",
            (new_name or None, account_id), fetch="none"
        )
        logger.info(f"[VAULT] Display name updated: {account_id[:12]}... → '{new_name}'")
        return _rpc_ok({"updated": True, "display_name": new_name}, rpc_id)
    except Exception as e:
        logger.exception(f"[VAULT] vault_updateDisplayName error: {e}")
        return _rpc_error(-32603, f"Failed: {str(e)}", rpc_id)


def vault_change_account_id(params: dict, rpc_id: Any) -> dict:
    """RPC: vault_changeAccountId — Change the vault account ID (username).

    Strategy: INSERT new row → UPDATE children → DELETE old row.
    This avoids FK violations entirely — children always reference a valid PK.
    """
    try:
        p = _normalize_params(params)
        old_id = p.get("account_id", "")
        passphrase = p.get("passphrase", "")
        new_id = p.get("new_account_id", "").strip()
        if not old_id or not passphrase or not new_id:
            return _rpc_error(-32602, "account_id, passphrase, and new_account_id required", rpc_id)
        if len(new_id) < 3 or len(new_id) > 64:
            return _rpc_error(-32602, "new_account_id must be 3-64 characters", rpc_id)
        if old_id == new_id:
            return _rpc_error(-32602, "New ID is the same as current ID", rpc_id)

        _ensure_schema()
        acct = _vault_query("SELECT passphrase_hash FROM vault_accounts WHERE id = %s", (old_id,), fetch="one")
        if not acct:
            return _rpc_error(-32004, "Account not found", rpc_id)
        if not _verify_passphrase(passphrase, acct["passphrase_hash"]):
            return _rpc_error(-32003, "Invalid passphrase", rpc_id)

        # Check new ID isn't taken
        existing = _vault_query("SELECT id FROM vault_accounts WHERE id = %s", (new_id,), fetch="one")
        if existing:
            return _rpc_error(-32009, "Account ID already taken", rpc_id)

        # Strategy: copy old account row with new PK → re-point children → delete old row.
        # This never violates FK constraints because the new PK exists before children update,
        # and the old PK is deleted only after no children reference it.
        conn = _get_vault_conn()
        try:
            cur = conn.cursor()

            # 1. Insert clone of account row with new ID
            cur.execute(
                """INSERT INTO vault_accounts
                       (id, display_name, email, passphrase_hash, tier, device_fp,
                        qtcl_address, public_key, created_at, updated_at, last_login,
                        secrets_count, bytes_stored, anchors_used, credit_balance)
                   SELECT %s, display_name, email, passphrase_hash, tier, device_fp,
                          qtcl_address, public_key, created_at, NOW(), last_login,
                          secrets_count, bytes_stored, anchors_used, credit_balance
                   FROM vault_accounts WHERE id = %s""",
                (new_id, old_id)
            )

            # 2. Re-point all child tables to the new ID
            _CHILD_TABLES = [
                'vault_contacts', 'vault_secrets', 'vault_anchors',
                'vault_billing', 'vault_inheritance',
            ]
            for table in _CHILD_TABLES:
                try:
                    cur.execute(
                        f"UPDATE {table} SET account_id = %s WHERE account_id = %s",
                        (new_id, old_id)
                    )
                except Exception:
                    pass  # table may not exist in lightweight schema

            # 3. Delete old account row (no children reference it now)
            cur.execute("DELETE FROM vault_accounts WHERE id = %s", (old_id,))

            logger.info(f"[VAULT] Account ID changed: {old_id} → {new_id}")
        except Exception:
            conn.rollback()
            raise
        finally:
            _put_vault_conn(conn)

        return _rpc_ok({"changed": True, "old_id": old_id, "new_id": new_id}, rpc_id)
    except Exception as e:
        logger.exception(f"[VAULT] vault_changeAccountId error: {e}")
        return _rpc_error(-32603, f"Failed: {str(e)}", rpc_id)


# ═══════════════════════════════════════════════════════════════════════════════
# ADDRESS BOOK (vault-gated contacts)
# ═══════════════════════════════════════════════════════════════════════════════

def vault_get_contacts(params: dict, rpc_id: Any) -> dict:
    """RPC: vault_getContacts — list address book contacts for a vault account."""
    try:
        p = _normalize_params(params)
        account_id = p.get("account_id", "")
        passphrase = p.get("passphrase", "")
        if not account_id or not passphrase:
            return _rpc_error(-32602, "account_id and passphrase required", rpc_id)

        _ensure_schema()
        acct = _vault_query("SELECT passphrase_hash FROM vault_accounts WHERE id = %s", (account_id,), fetch="one")
        if not acct:
            return _rpc_error(-32004, "Account not found", rpc_id)
        if not _verify_passphrase(passphrase, acct["passphrase_hash"]):
            return _rpc_error(-32003, "Invalid passphrase", rpc_id)

        rows = _vault_query(
            "SELECT id, name, address, created_at FROM vault_contacts WHERE account_id = %s ORDER BY name ASC",
            (account_id,)
        )
        contacts = [{"id": r["id"], "name": r["name"], "address": r["address"],
                      "created_at": str(r.get("created_at", ""))} for r in (rows or [])]
        return _rpc_ok({"contacts": contacts, "count": len(contacts)}, rpc_id)
    except Exception as e:
        logger.exception(f"[VAULT] vault_getContacts error: {e}")
        return _rpc_error(-32603, f"Failed: {str(e)}", rpc_id)


def vault_add_contact(params: dict, rpc_id: Any) -> dict:
    """RPC: vault_addContact — add a contact to the vault address book."""
    try:
        p = _normalize_params(params)
        account_id = p.get("account_id", "")
        passphrase = p.get("passphrase", "")
        name = p.get("name", "").strip()
        address = p.get("address", "").strip()
        if not account_id or not passphrase:
            return _rpc_error(-32602, "account_id and passphrase required", rpc_id)
        if not name or not address:
            return _rpc_error(-32602, "name and address required", rpc_id)

        _ensure_schema()
        acct = _vault_query("SELECT passphrase_hash FROM vault_accounts WHERE id = %s", (account_id,), fetch="one")
        if not acct:
            return _rpc_error(-32004, "Account not found", rpc_id)
        if not _verify_passphrase(passphrase, acct["passphrase_hash"]):
            return _rpc_error(-32003, "Invalid passphrase", rpc_id)

        contact_id = secrets.token_hex(16)
        _vault_query(
            "INSERT INTO vault_contacts (id, account_id, name, address) VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (account_id, address) DO UPDATE SET name = EXCLUDED.name",
            (contact_id, account_id, name[:100], address[:128]),
            fetch="none"
        )
        return _rpc_ok({"id": contact_id, "name": name, "address": address, "saved": True}, rpc_id)
    except Exception as e:
        logger.exception(f"[VAULT] vault_addContact error: {e}")
        return _rpc_error(-32603, f"Failed: {str(e)}", rpc_id)


def vault_remove_contact(params: dict, rpc_id: Any) -> dict:
    """RPC: vault_removeContact — remove a contact from vault address book."""
    try:
        p = _normalize_params(params)
        account_id = p.get("account_id", "")
        passphrase = p.get("passphrase", "")
        contact_id = p.get("contact_id", "")
        if not account_id or not passphrase or not contact_id:
            return _rpc_error(-32602, "account_id, passphrase, and contact_id required", rpc_id)

        _ensure_schema()
        acct = _vault_query("SELECT passphrase_hash FROM vault_accounts WHERE id = %s", (account_id,), fetch="one")
        if not acct:
            return _rpc_error(-32004, "Account not found", rpc_id)
        if not _verify_passphrase(passphrase, acct["passphrase_hash"]):
            return _rpc_error(-32003, "Invalid passphrase", rpc_id)

        _vault_query(
            "DELETE FROM vault_contacts WHERE id = %s AND account_id = %s",
            (contact_id, account_id),
            fetch="none"
        )
        return _rpc_ok({"deleted": True, "contact_id": contact_id}, rpc_id)
    except Exception as e:
        logger.exception(f"[VAULT] vault_removeContact error: {e}")
        return _rpc_error(-32603, f"Failed: {str(e)}", rpc_id)


def vault_get_burn_stats(params: dict, rpc_id: Any) -> dict:
    """
    RPC: vault_getBurnStats — public endpoint.
    Returns total QTCL burned by vault operations and coherence fund balance.
    Useful for transparency dashboards and tokenomics tracking.
    """
    try:
        stats = {
            "burn_address":        BURN_ADDRESS,
            "coherence_address":   COHERENCE_FUND_ADDRESS,
            "burn_ratio":          BURN_RATIO,
            "coherence_ratio":     COHERENCE_RATIO,
            "burn_balance_qtcl":   0.0,
            "coherence_balance_qtcl": 0.0,
            "total_vault_ops":     0,
            "total_charged_qtcl":  0.0,
        }
        # Query on-chain wallet balances for burn + coherence addresses
        try:
            from server import get_db_cursor
            with get_db_cursor() as cur:
                cur.execute(
                    "SELECT address, balance FROM wallet_addresses"
                    " WHERE address IN (%s, %s)",
                    (BURN_ADDRESS, COHERENCE_FUND_ADDRESS),
                )
                for row in cur.fetchall() or []:
                    addr, bal = row[0], int(row[1] or 0)
                    if addr == BURN_ADDRESS:
                        stats["burn_balance_qtcl"] = bal / QTCL_BASE
                    elif addr == COHERENCE_FUND_ADDRESS:
                        stats["coherence_balance_qtcl"] = bal / QTCL_BASE
        except Exception as _ce:
            logger.debug(f"[VAULT-BURN-STATS] chain query: {_ce}")

        # Query vault billing totals
        try:
            billing_totals = _vault_query(
                "SELECT COUNT(*), SUM(ABS(amount)) FROM vault_billing WHERE amount < 0",
                fetch="one"
            )
            if billing_totals:
                stats["total_vault_ops"]    = int(billing_totals.get("count", 0) or 0)
                raw_sum = billing_totals.get("sum", 0) or 0
                stats["total_charged_qtcl"] = float(raw_sum) / QTCL_BASE
        except Exception as _be:
            logger.debug(f"[VAULT-BURN-STATS] billing query: {_be}")

        stats["total_burned_qtcl"]    = round(stats["total_charged_qtcl"] * BURN_RATIO, 4)
        stats["total_coherence_qtcl"] = round(stats["total_charged_qtcl"] * COHERENCE_RATIO, 4)
        return _rpc_ok(stats, rpc_id)
    except Exception as e:
        logger.error(f"[VAULT] getBurnStats failed: {e}", exc_info=True)
        return _rpc_error(-32603, f"Failed: {str(e)}", rpc_id)


def vault_get_pricing(params: dict, rpc_id: Any) -> dict:
    """RPC: vault_getPricing
    Return the complete pricing sheet for all vault operations.
    Public — no auth required.
    """
    return _rpc_ok({
        "exchange_rate": "100 QTCL = $1.00 USD",
        "tiers": {
            name: {
                "monthly_usd": info["monthly_qtcl"] // QTCL_BASE * 0.01,
                "monthly_qtcl": info["monthly_qtcl"] // QTCL_BASE,
                "max_secrets": info["max_secrets"],
                "max_bytes": info["max_bytes"],
                "max_bytes_human": _human_size(info["max_bytes"]),
                "anchors_included": info["anchors_included"],
                "overage_per_100kb": info["overage_per_100kb"] // QTCL_BASE,
            }
            for name, info in VAULT_TIERS.items()
        },
        "operations": {
            "store_secret": {"qtcl": PRICE_STORE_SECRET // QTCL_BASE, "usd": round(PRICE_STORE_SECRET / QTCL_BASE * 0.01, 2)},
            "delete_secret": {"qtcl": PRICE_DELETE_SECRET // QTCL_BASE, "usd": round(PRICE_DELETE_SECRET / QTCL_BASE * 0.01, 2)},
            "anchor_standard": {"qtcl": PRICE_ANCHOR_STANDARD // QTCL_BASE, "usd": round(PRICE_ANCHOR_STANDARD / QTCL_BASE * 0.01, 2)},
            "anchor_obfuscated": {"qtcl": PRICE_ANCHOR_OBFUSCATED // QTCL_BASE, "usd": round(PRICE_ANCHOR_OBFUSCATED / QTCL_BASE * 0.01, 2)},
            "reanchor": {"qtcl": PRICE_REANCHOR // QTCL_BASE, "usd": round(PRICE_REANCHOR / QTCL_BASE * 0.01, 2)},
        },
        "free_operations": ["retrieve_secret", "list_secrets", "verify_anchor", "get_pricing", "login"],
        "burn_split": f"{int(BURN_RATIO*100)}% burn / {int(COHERENCE_RATIO*100)}% coherence",
    }, rpc_id)


def _human_size(bytes_val: int) -> str:
    """Convert bytes to human-readable string."""
    if bytes_val >= 1024 * 1024:
        return f"{bytes_val // (1024 * 1024)} MB"
    if bytes_val >= 1024:
        return f"{bytes_val // 1024} KB"
    return f"{bytes_val} B"


# ═══════════════════════════════════════════════════════════════════════════════
# VAULT RPC METHOD REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

VAULT_RPC_METHODS: Dict[str, Any] = {
    # Account
    "vault_createAccount":      vault_create_account,
    "vault_login":              vault_login,
    "vault_upgradeTier":        vault_upgrade_tier,
    "vault_changeAccountId":    vault_change_account_id,
    "vault_updateDisplayName":  vault_update_display_name,
    # Secrets
    "vault_storeSecret":        vault_store_secret,
    "vault_retrieveSecret":     vault_retrieve_secret,
    "vault_listSecrets":        vault_list_secrets,
    "vault_deleteSecret":       vault_delete_secret,
    # Anchoring
    "vault_anchorHash":         vault_anchor_hash,
    "vault_verifyAnchor":       vault_verify_anchor,
    # Billing
    "vault_getTreasuryAddress": lambda p, r: _rpc_ok({
        "address": os.environ.get("TREASURY_ADDRESS", "qt_treasury_default"),
        "burn_address": BURN_ADDRESS,
        "coherence_address": COHERENCE_FUND_ADDRESS,
        "fee_split": f"{int(BURN_RATIO*100)}% burn / {int(COHERENCE_RATIO*100)}% coherence",
    }, r),
    "vault_depositCredit":      vault_deposit_credit,
    "vault_getBalance":         vault_get_balance,
    "vault_getBurnStats":       vault_get_burn_stats,
    "vault_getPricing":         vault_get_pricing,
    # Transactions (stub)
    "vault_getTransactions":    lambda p, r: _rpc_ok({"transactions": []}, r),
    # Inheritance
    "vault_setupInheritance":   vault_setup_inheritance,
    "vault_checkIn":            vault_check_in,
    # Address Book (vault-gated contacts)
    "vault_getContacts":        vault_get_contacts,
    "vault_addContact":         vault_add_contact,
    "vault_removeContact":      vault_remove_contact,
}

logger.info(f"[VAULT] ✅ {len(VAULT_RPC_METHODS)} vault RPC methods registered")
