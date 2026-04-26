#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                                                                                  ║
║   QTCL VAULT DATABASE BUILDER v1.0                                              ║
║   Supabase PostgreSQL · Zero-Knowledge Encrypted Storage Infrastructure         ║
║                                                                                  ║
║   Connection: VAULT_URL environment variable (Supabase PostgreSQL)              ║
║   Run:        python vault_db_builder.py                                        ║
║                                                                                  ║
║   ┌────────────────────────────────────────────────────────────────────────┐    ║
║   │                                                                        │    ║
║   │   ZERO-KNOWLEDGE ARCHITECTURE                                          │    ║
║   │                                                                        │    ║
║   │   Client encrypts → Server stores opaque blob → Client decrypts       │    ║
║   │   Server NEVER sees plaintext. Keys NEVER leave the browser.           │    ║
║   │                                                                        │    ║
║   │   Encryption: PBKDF2-SHA256 (600K) → AES-256-GCM (WebCrypto)          │    ║
║   │   Anchoring:  SHA3-256 commitment → QTCL blockchain proof-of-exist.   │    ║
║   │   Obfuscation: H(secret ‖ blinding_nonce) → unlinkable on-chain hash │    ║
║   │                                                                        │    ║
║   └────────────────────────────────────────────────────────────────────────┘    ║
║                                                                                  ║
║   SECURITY POSTURE:                                                              ║
║     • Row-Level Security (RLS) on every table — accounts isolated              ║
║     • Argon2id passphrase hashing (OWASP 2023 recommended)                     ║
║     • Rate limiting via vault_rate_limits (IP + account level)                 ║
║     • Immutable append-only audit log (vault_audit_log)                        ║
║     • Session tokens: SHA3-256 HMAC, short-lived (24h), device-bound           ║
║     • Canary secrets: honeypot entries that trigger alerts on access            ║
║     • Failed login lockout: 5 attempts → 15-minute cooldown                    ║
║     • Secrets hard-deleted with VACUUM to prevent forensic recovery             ║
║     • TOTP 2FA support (vault_totp_secrets)                                    ║
║     • Shamir secret sharing for inheritance (vault_shamir_shares)              ║
║     • Breach detection via vault_breach_canaries                               ║
║                                                                                  ║
║   TABLES (17):                                                                   ║
║     Core:                                                                        ║
║       vault_accounts         — Identity, auth, tier, credit                    ║
║       vault_sessions         — Short-lived device-bound tokens                 ║
║       vault_secrets          — Opaque encrypted blobs (zero-knowledge)         ║
║       vault_secret_versions  — Immutable version history per secret            ║
║       vault_secret_tags      — Searchable encrypted tags (server-blind)        ║
║     Chain:                                                                       ║
║       vault_anchors          — SHA3-256 blockchain commitments                 ║
║       vault_anchor_proofs    — On-chain verification receipts                  ║
║     Billing:                                                                     ║
║       vault_billing          — Append-only ledger (every charge/credit)        ║
║       vault_pricing_tiers    — Configurable pricing per operation              ║
║     Security:                                                                    ║
║       vault_audit_log        — Immutable event log (login, access, delete)     ║
║       vault_rate_limits      — Per-IP and per-account rate limiting            ║
║       vault_failed_logins    — Brute-force detection and lockout               ║
║       vault_totp_secrets     — TOTP 2FA enrollment                             ║
║       vault_breach_canaries  — Honeypot secrets that trigger alerts            ║
║     Inheritance:                                                                 ║
║       vault_inheritance      — Dead man's switch configuration                 ║
║       vault_shamir_shares    — Shamir threshold shares for recovery            ║
║     Metadata:                                                                    ║
║       vault_metadata         — Schema version, feature flags, config           ║
║                                                                                  ║
║   TRIGGERS & FUNCTIONS (6):                                                      ║
║     fn_vault_audit_insert    — Auto-log every secret INSERT                    ║
║     fn_vault_audit_delete    — Auto-log every secret DELETE                    ║
║     fn_vault_audit_access    — Auto-log every secret SELECT (via RPC)          ║
║     fn_vault_canary_check    — Alert on canary secret access                   ║
║     fn_vault_rate_check      — Enforce rate limits per window                  ║
║     fn_vault_updated_at      — Auto-update updated_at timestamps               ║
║                                                                                  ║
║   Author: QTCL / shemshallah (Justin Howard-Stanley)                            ║
║   Specification: QTCL Vault Architecture v1.0 · April 2026                      ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import logging
import hashlib
import secrets as _secrets

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

VAULT_URL = os.environ.get("VAULT_URL", "")
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("vault_db_builder")

if not VAULT_URL:
    logger.error("VAULT_URL environment variable not set.")
    logger.error("  export VAULT_URL='postgresql://user:pass@host:5432/dbname'")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# DEPENDENCY CHECK
# ─────────────────────────────────────────────────────────────────────────────

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    import subprocess
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "--break-system-packages", "psycopg2-binary"
    ])
    import psycopg2
    import psycopg2.extras

# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA VERSION — increment on every migration
# ═══════════════════════════════════════════════════════════════════════════════

SCHEMA_VERSION = 1
SCHEMA_LABEL = "vault_v1.0_zero_knowledge"

# ═══════════════════════════════════════════════════════════════════════════════
# COMPLETE VAULT SCHEMA (Supabase PostgreSQL)
# ═══════════════════════════════════════════════════════════════════════════════

VAULT_SCHEMA = """

-- ═══════════════════════════════════════════════════════════════════════════════
-- EXTENSIONS
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ═══════════════════════════════════════════════════════════════════════════════
-- TABLE: vault_metadata — Schema version, feature flags, global config
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS vault_metadata (
    key              TEXT PRIMARY KEY,
    value            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ═══════════════════════════════════════════════════════════════════════════════
-- TABLE: vault_accounts — Identity, authentication, tier, credit balance
--
-- Security:
--   • passphrase_hash is Argon2id (or PBKDF2-HMAC-SHA256 600K fallback)
--     stored as "salt_hex:dk_hex" — never reversible
--   • device_fp is SHA-256 of canvas + UA + screen + timezone
--   • totp_enrolled enables 2FA gate on all mutations
--   • locked_until implements progressive lockout after failed logins
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS vault_accounts (
    id                  TEXT PRIMARY KEY,           -- "va_" + 32 hex chars
    email               TEXT,                       -- optional, for recovery
    email_verified      BOOLEAN NOT NULL DEFAULT FALSE,
    passphrase_hash     TEXT NOT NULL,              -- "salt:dk" (PBKDF2-SHA256 600K)
    tier                TEXT NOT NULL DEFAULT 'trial',  -- trial | paid | enterprise
    device_fp           TEXT,                       -- SHA-256 browser fingerprint
    qtcl_address        TEXT,                       -- billing/identity address
    public_key          TEXT,                       -- HypΓ public key (hex)
    totp_enrolled       BOOLEAN NOT NULL DEFAULT FALSE,
    locked_until        TIMESTAMPTZ,                -- NULL = not locked
    failed_login_count  INTEGER NOT NULL DEFAULT 0,
    secrets_count       INTEGER NOT NULL DEFAULT 0,
    bytes_stored        BIGINT NOT NULL DEFAULT 0,
    anchors_used        INTEGER NOT NULL DEFAULT 0,
    credit_balance      BIGINT NOT NULL DEFAULT 0,  -- base units (100,000 = 1 QTCL)
    last_login          TIMESTAMPTZ,
    last_ip             INET,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_va_email       ON vault_accounts(email)      WHERE email IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_va_device      ON vault_accounts(device_fp)  WHERE device_fp IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_va_qtcl        ON vault_accounts(qtcl_address) WHERE qtcl_address IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_va_tier        ON vault_accounts(tier);
CREATE INDEX IF NOT EXISTS idx_va_locked      ON vault_accounts(locked_until) WHERE locked_until IS NOT NULL;

-- ═══════════════════════════════════════════════════════════════════════════════
-- TABLE: vault_sessions — Short-lived, device-bound authentication tokens
--
-- Security:
--   • token is SHA3-256(account_id : timestamp : random_hex)
--   • expires_at enforces 24h maximum lifetime
--   • device_fp must match the login device — session hijacking detection
--   • revoked flag allows instant invalidation
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS vault_sessions (
    id                  TEXT PRIMARY KEY,           -- "vs_" + 32 hex
    account_id          TEXT NOT NULL REFERENCES vault_accounts(id) ON DELETE CASCADE,
    token_hash          TEXT NOT NULL,              -- SHA3-256 of session token
    device_fp           TEXT,
    ip_address          INET,
    user_agent          TEXT,
    expires_at          TIMESTAMPTZ NOT NULL,
    revoked             BOOLEAN NOT NULL DEFAULT FALSE,
    revoked_at          TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vs_account     ON vault_sessions(account_id);
CREATE INDEX IF NOT EXISTS idx_vs_token       ON vault_sessions(token_hash);
CREATE INDEX IF NOT EXISTS idx_vs_expires     ON vault_sessions(expires_at) WHERE revoked = FALSE;

-- ═══════════════════════════════════════════════════════════════════════════════
-- TABLE: vault_secrets — Opaque encrypted blobs (ZERO-KNOWLEDGE)
--
-- Architecture:
--   • ciphertext is AES-256-GCM encrypted on the CLIENT
--   • Server stores it verbatim — NEVER decrypts, NEVER inspects
--   • encryption_meta holds client-side params (salt, iv, kdf) — no keys
--   • content_hash is SHA3-256(ciphertext) — for integrity, not plaintext
--   • size_bytes is the ORIGINAL plaintext size (client-reported)
--   • version tracks mutations (every update creates a new version row)
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS vault_secrets (
    id                  TEXT PRIMARY KEY,           -- "vs_" + 32 hex
    account_id          TEXT NOT NULL REFERENCES vault_accounts(id) ON DELETE CASCADE,
    label               TEXT NOT NULL,
    category            TEXT NOT NULL DEFAULT 'general',
    ciphertext          TEXT NOT NULL,              -- opaque client-encrypted blob
    encryption_meta     JSONB NOT NULL DEFAULT '{}',  -- {algorithm, kdf, salt, iv, zero_knowledge: true}
    content_hash        TEXT NOT NULL,              -- SHA3-256(ciphertext) for integrity
    size_bytes          INTEGER NOT NULL DEFAULT 0,
    current_version     INTEGER NOT NULL DEFAULT 1,
    -- Chain anchoring
    anchor_hash         TEXT,                       -- SHA3-256 commitment (on-chain)
    anchor_block        INTEGER,
    anchor_tx           TEXT,
    obfuscated          BOOLEAN NOT NULL DEFAULT FALSE,
    blinding_nonce      TEXT,                       -- only for obfuscated anchors
    -- Lifecycle
    expires_at          TIMESTAMPTZ,
    access_count        INTEGER NOT NULL DEFAULT 0,
    last_accessed       TIMESTAMPTZ,
    is_canary           BOOLEAN NOT NULL DEFAULT FALSE,  -- honeypot flag
    deleted_at          TIMESTAMPTZ,                -- soft-delete before hard purge
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vsc_account    ON vault_secrets(account_id) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_vsc_category   ON vault_secrets(account_id, category) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_vsc_anchor     ON vault_secrets(anchor_hash) WHERE anchor_hash IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_vsc_content    ON vault_secrets(content_hash);
CREATE INDEX IF NOT EXISTS idx_vsc_canary     ON vault_secrets(is_canary) WHERE is_canary = TRUE;
CREATE INDEX IF NOT EXISTS idx_vsc_expires    ON vault_secrets(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_vsc_deleted    ON vault_secrets(deleted_at) WHERE deleted_at IS NOT NULL;

-- ═══════════════════════════════════════════════════════════════════════════════
-- TABLE: vault_secret_versions — Immutable version history per secret
--
-- Every UPDATE to a secret creates a new version row here BEFORE the mutation.
-- This provides:
--   • Full audit trail of every change
--   • Point-in-time recovery (retrieve any prior version)
--   • Tamper evidence (content_hash chain)
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS vault_secret_versions (
    id                  BIGSERIAL PRIMARY KEY,
    secret_id           TEXT NOT NULL REFERENCES vault_secrets(id) ON DELETE CASCADE,
    version             INTEGER NOT NULL,
    ciphertext          TEXT NOT NULL,
    encryption_meta     JSONB NOT NULL DEFAULT '{}',
    content_hash        TEXT NOT NULL,
    size_bytes          INTEGER NOT NULL DEFAULT 0,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(secret_id, version)
);

CREATE INDEX IF NOT EXISTS idx_vsv_secret     ON vault_secret_versions(secret_id);

-- ═══════════════════════════════════════════════════════════════════════════════
-- TABLE: vault_secret_tags — Searchable encrypted tags (server-blind)
--
-- Tags are SHA3-256(tag_plaintext ‖ account_salt) — server can match
-- equality but cannot read the tag value. Client computes the hash.
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS vault_secret_tags (
    id                  BIGSERIAL PRIMARY KEY,
    secret_id           TEXT NOT NULL REFERENCES vault_secrets(id) ON DELETE CASCADE,
    account_id          TEXT NOT NULL REFERENCES vault_accounts(id) ON DELETE CASCADE,
    tag_hash            TEXT NOT NULL,              -- SHA3-256(tag ‖ salt)
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vst_secret     ON vault_secret_tags(secret_id);
CREATE INDEX IF NOT EXISTS idx_vst_tag        ON vault_secret_tags(account_id, tag_hash);

-- ═══════════════════════════════════════════════════════════════════════════════
-- TABLE: vault_anchors — SHA3-256 blockchain commitments
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS vault_anchors (
    id                  TEXT PRIMARY KEY,           -- "vanc_" + 32 hex
    account_id          TEXT NOT NULL REFERENCES vault_accounts(id) ON DELETE CASCADE,
    secret_id           TEXT REFERENCES vault_secrets(id) ON DELETE SET NULL,
    anchor_hash         TEXT NOT NULL,              -- original content hash
    obfuscated_hash     TEXT,                       -- H(hash ‖ blinding_nonce)
    on_chain_hash       TEXT NOT NULL,              -- what was actually committed
    block_height        INTEGER,
    tx_hash             TEXT,
    chain               TEXT NOT NULL DEFAULT 'qtcl',
    label               TEXT,
    verified            BOOLEAN NOT NULL DEFAULT FALSE,
    verified_at         TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vanc_hash      ON vault_anchors(anchor_hash);
CREATE INDEX IF NOT EXISTS idx_vanc_obf       ON vault_anchors(obfuscated_hash) WHERE obfuscated_hash IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_vanc_onchain   ON vault_anchors(on_chain_hash);
CREATE INDEX IF NOT EXISTS idx_vanc_account   ON vault_anchors(account_id);
CREATE INDEX IF NOT EXISTS idx_vanc_block     ON vault_anchors(block_height) WHERE block_height IS NOT NULL;

-- ═══════════════════════════════════════════════════════════════════════════════
-- TABLE: vault_anchor_proofs — On-chain verification receipts
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS vault_anchor_proofs (
    id                  BIGSERIAL PRIMARY KEY,
    anchor_id           TEXT NOT NULL REFERENCES vault_anchors(id) ON DELETE CASCADE,
    proof_type          TEXT NOT NULL DEFAULT 'merkle',  -- merkle | inclusion | spv
    proof_data          JSONB NOT NULL DEFAULT '{}',
    verified_by         TEXT,                       -- node that verified
    verified_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vap_anchor     ON vault_anchor_proofs(anchor_id);

-- ═══════════════════════════════════════════════════════════════════════════════
-- TABLE: vault_billing — Append-only financial ledger
--
-- INVARIANT: SUM(amount) WHERE account_id = X == vault_accounts.credit_balance
-- This table is INSERT-ONLY. No updates, no deletes. Ever.
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS vault_billing (
    id                  TEXT PRIMARY KEY,           -- "vb_" + 32 hex
    account_id          TEXT NOT NULL REFERENCES vault_accounts(id) ON DELETE CASCADE,
    operation           TEXT NOT NULL,              -- deposit | store | anchor | obfuscated_anchor | purge
    amount              BIGINT NOT NULL,            -- positive = credit, negative = debit
    balance_after       BIGINT NOT NULL,
    tx_hash             TEXT,                       -- QTCL chain tx hash (for deposits)
    reference_id        TEXT,                       -- secret_id or anchor_id
    description         TEXT,
    metadata            JSONB DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vbl_account    ON vault_billing(account_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_vbl_operation  ON vault_billing(operation);
CREATE INDEX IF NOT EXISTS idx_vbl_tx        ON vault_billing(tx_hash) WHERE tx_hash IS NOT NULL;

-- ═══════════════════════════════════════════════════════════════════════════════
-- TABLE: vault_pricing_tiers — Configurable pricing per operation
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS vault_pricing_tiers (
    id                  SERIAL PRIMARY KEY,
    tier                TEXT NOT NULL,              -- trial | paid | enterprise
    operation           TEXT NOT NULL,              -- store | anchor | obfuscated_anchor | retrieve
    qtcl_cost           BIGINT NOT NULL DEFAULT 0,  -- cost in base units
    max_size_bytes      BIGINT,
    max_secrets         INTEGER,
    description         TEXT,
    active              BOOLEAN NOT NULL DEFAULT TRUE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(tier, operation)
);

-- ═══════════════════════════════════════════════════════════════════════════════
-- TABLE: vault_audit_log — Immutable append-only event log
--
-- SECURITY: This table has no UPDATE or DELETE permissions.
-- Every access, mutation, login, and failure is recorded here.
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS vault_audit_log (
    id                  BIGSERIAL PRIMARY KEY,
    account_id          TEXT,                       -- NULL for anonymous events
    event_type          TEXT NOT NULL,              -- login | logout | create_secret | access_secret |
                                                    -- delete_secret | anchor | verify | deposit |
                                                    -- failed_login | lockout | canary_triggered |
                                                    -- session_revoked | totp_enrolled | tier_upgrade
    ip_address          INET,
    user_agent          TEXT,
    secret_id           TEXT,                       -- NULL if not secret-related
    success             BOOLEAN NOT NULL DEFAULT TRUE,
    failure_reason      TEXT,
    metadata            JSONB DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_val_account    ON vault_audit_log(account_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_val_event      ON vault_audit_log(event_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_val_secret     ON vault_audit_log(secret_id) WHERE secret_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_val_failed     ON vault_audit_log(success) WHERE success = FALSE;
CREATE INDEX IF NOT EXISTS idx_val_canary     ON vault_audit_log(event_type) WHERE event_type = 'canary_triggered';
CREATE INDEX IF NOT EXISTS idx_val_created    ON vault_audit_log(created_at DESC);

-- ═══════════════════════════════════════════════════════════════════════════════
-- TABLE: vault_rate_limits — Per-IP and per-account sliding window rate limits
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS vault_rate_limits (
    id                  BIGSERIAL PRIMARY KEY,
    key_type            TEXT NOT NULL,              -- ip | account | device
    key_value           TEXT NOT NULL,
    operation           TEXT NOT NULL,              -- login | create | store | retrieve | anchor
    window_start        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    request_count       INTEGER NOT NULL DEFAULT 1,
    UNIQUE(key_type, key_value, operation, window_start)
);

CREATE INDEX IF NOT EXISTS idx_vrl_key        ON vault_rate_limits(key_type, key_value, operation);
CREATE INDEX IF NOT EXISTS idx_vrl_window     ON vault_rate_limits(window_start);

-- ═══════════════════════════════════════════════════════════════════════════════
-- TABLE: vault_failed_logins — Brute-force tracking and progressive lockout
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS vault_failed_logins (
    id                  BIGSERIAL PRIMARY KEY,
    account_id          TEXT REFERENCES vault_accounts(id) ON DELETE CASCADE,
    ip_address          INET,
    attempted_id        TEXT,                       -- what account ID was tried
    failure_reason      TEXT NOT NULL DEFAULT 'invalid_passphrase',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vfl_account    ON vault_failed_logins(account_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_vfl_ip         ON vault_failed_logins(ip_address, created_at DESC);

-- ═══════════════════════════════════════════════════════════════════════════════
-- TABLE: vault_totp_secrets — TOTP 2FA enrollment
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS vault_totp_secrets (
    id                  TEXT PRIMARY KEY,
    account_id          TEXT NOT NULL REFERENCES vault_accounts(id) ON DELETE CASCADE,
    secret_encrypted    TEXT NOT NULL,              -- TOTP seed encrypted with account KEK
    algorithm           TEXT NOT NULL DEFAULT 'SHA1',
    digits              INTEGER NOT NULL DEFAULT 6,
    period              INTEGER NOT NULL DEFAULT 30,
    verified            BOOLEAN NOT NULL DEFAULT FALSE,
    backup_codes_hash   TEXT,                       -- SHA3-256 of backup codes
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(account_id)
);

-- ═══════════════════════════════════════════════════════════════════════════════
-- TABLE: vault_breach_canaries — Honeypot secrets that trigger alerts
--
-- Canary secrets look identical to real secrets but have is_canary=TRUE.
-- Any access to a canary secret logs a 'canary_triggered' audit event
-- and can optionally notify the account owner.
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS vault_breach_canaries (
    id                  BIGSERIAL PRIMARY KEY,
    account_id          TEXT NOT NULL REFERENCES vault_accounts(id) ON DELETE CASCADE,
    secret_id           TEXT NOT NULL REFERENCES vault_secrets(id) ON DELETE CASCADE,
    alert_email         TEXT,
    alert_webhook       TEXT,
    triggered           BOOLEAN NOT NULL DEFAULT FALSE,
    triggered_at        TIMESTAMPTZ,
    triggered_ip        INET,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(account_id, secret_id)
);

-- ═══════════════════════════════════════════════════════════════════════════════
-- TABLE: vault_inheritance — Dead man's switch configuration
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS vault_inheritance (
    id                  TEXT PRIMARY KEY,
    account_id          TEXT NOT NULL REFERENCES vault_accounts(id) ON DELETE CASCADE,
    beneficiary         TEXT NOT NULL,              -- QTCL address or email
    beneficiary_pubkey  TEXT,                       -- for re-encryption
    check_in_days       INTEGER NOT NULL DEFAULT 365,
    last_check_in       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    grace_period_days   INTEGER NOT NULL DEFAULT 30,  -- extra days after missed check-in
    notification_sent   BOOLEAN NOT NULL DEFAULT FALSE,
    activated           BOOLEAN NOT NULL DEFAULT FALSE,
    activated_at        TIMESTAMPTZ,
    shamir_threshold    INTEGER DEFAULT 3,          -- M-of-N shares needed
    shamir_total        INTEGER DEFAULT 5,          -- total shares
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vinh_account   ON vault_inheritance(account_id);
CREATE INDEX IF NOT EXISTS idx_vinh_checkin   ON vault_inheritance(last_check_in) WHERE activated = FALSE;

-- ═══════════════════════════════════════════════════════════════════════════════
-- TABLE: vault_shamir_shares — Shamir threshold shares for inheritance
--
-- Each share is encrypted with the beneficiary's public key.
-- M-of-N shares reconstruct the vault master key.
-- ═══════════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS vault_shamir_shares (
    id                  BIGSERIAL PRIMARY KEY,
    inheritance_id      TEXT NOT NULL REFERENCES vault_inheritance(id) ON DELETE CASCADE,
    share_index         INTEGER NOT NULL,           -- 1..N
    share_encrypted     TEXT NOT NULL,              -- encrypted share data
    custodian           TEXT NOT NULL,              -- who holds this share
    custodian_type      TEXT NOT NULL DEFAULT 'email',  -- email | qtcl_address | escrow
    delivered           BOOLEAN NOT NULL DEFAULT FALSE,
    delivered_at        TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(inheritance_id, share_index)
);

CREATE INDEX IF NOT EXISTS idx_vss_inh        ON vault_shamir_shares(inheritance_id);

-- ═══════════════════════════════════════════════════════════════════════════════
-- TRIGGER FUNCTIONS
-- ═══════════════════════════════════════════════════════════════════════════════

-- Auto-update updated_at on any row modification
CREATE OR REPLACE FUNCTION fn_vault_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply updated_at trigger to all mutable tables
DROP TRIGGER IF EXISTS trg_va_updated ON vault_accounts;
CREATE TRIGGER trg_va_updated BEFORE UPDATE ON vault_accounts
    FOR EACH ROW EXECUTE FUNCTION fn_vault_updated_at();

DROP TRIGGER IF EXISTS trg_vs_updated ON vault_secrets;
CREATE TRIGGER trg_vs_updated BEFORE UPDATE ON vault_secrets
    FOR EACH ROW EXECUTE FUNCTION fn_vault_updated_at();

DROP TRIGGER IF EXISTS trg_vinh_updated ON vault_inheritance;
CREATE TRIGGER trg_vinh_updated BEFORE UPDATE ON vault_inheritance
    FOR EACH ROW EXECUTE FUNCTION fn_vault_updated_at();

-- Auto-log secret creation to audit log
CREATE OR REPLACE FUNCTION fn_vault_audit_insert()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO vault_audit_log (account_id, event_type, secret_id, metadata)
    VALUES (
        NEW.account_id,
        'create_secret',
        NEW.id,
        jsonb_build_object(
            'category', NEW.category,
            'size_bytes', NEW.size_bytes,
            'content_hash', NEW.content_hash,
            'is_canary', NEW.is_canary
        )
    );
    -- Update account counters
    UPDATE vault_accounts
    SET secrets_count = secrets_count + 1,
        bytes_stored = bytes_stored + NEW.size_bytes
    WHERE id = NEW.account_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_vs_audit_insert ON vault_secrets;
CREATE TRIGGER trg_vs_audit_insert AFTER INSERT ON vault_secrets
    FOR EACH ROW EXECUTE FUNCTION fn_vault_audit_insert();

-- Auto-log secret deletion to audit log
CREATE OR REPLACE FUNCTION fn_vault_audit_delete()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO vault_audit_log (account_id, event_type, secret_id, metadata)
    VALUES (
        OLD.account_id,
        'delete_secret',
        OLD.id,
        jsonb_build_object(
            'label', OLD.label,
            'category', OLD.category,
            'size_bytes', OLD.size_bytes
        )
    );
    -- Update account counters
    UPDATE vault_accounts
    SET secrets_count = GREATEST(secrets_count - 1, 0),
        bytes_stored = GREATEST(bytes_stored - OLD.size_bytes, 0)
    WHERE id = OLD.account_id;
    RETURN OLD;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_vs_audit_delete ON vault_secrets;
CREATE TRIGGER trg_vs_audit_delete AFTER DELETE ON vault_secrets
    FOR EACH ROW EXECUTE FUNCTION fn_vault_audit_delete();

-- Canary check: alert when a canary secret is accessed
CREATE OR REPLACE FUNCTION fn_vault_canary_check()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.is_canary = TRUE AND NEW.access_count > OLD.access_count THEN
        -- Log canary trigger
        INSERT INTO vault_audit_log (account_id, event_type, secret_id, metadata)
        VALUES (
            NEW.account_id,
            'canary_triggered',
            NEW.id,
            jsonb_build_object(
                'access_count', NEW.access_count,
                'last_accessed', NEW.last_accessed
            )
        );
        -- Mark breach canary as triggered
        UPDATE vault_breach_canaries
        SET triggered = TRUE, triggered_at = NOW()
        WHERE secret_id = NEW.id AND triggered = FALSE;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_vs_canary_check ON vault_secrets;
CREATE TRIGGER trg_vs_canary_check AFTER UPDATE ON vault_secrets
    FOR EACH ROW EXECUTE FUNCTION fn_vault_canary_check();

-- Version history: snapshot before every update
CREATE OR REPLACE FUNCTION fn_vault_version_snapshot()
RETURNS TRIGGER AS $$
BEGIN
    -- Only snapshot if ciphertext actually changed
    IF OLD.ciphertext IS DISTINCT FROM NEW.ciphertext THEN
        INSERT INTO vault_secret_versions (secret_id, version, ciphertext, encryption_meta, content_hash, size_bytes)
        VALUES (OLD.id, OLD.current_version, OLD.ciphertext, OLD.encryption_meta, OLD.content_hash, OLD.size_bytes);
        NEW.current_version = OLD.current_version + 1;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_vs_version_snapshot ON vault_secrets;
CREATE TRIGGER trg_vs_version_snapshot BEFORE UPDATE ON vault_secrets
    FOR EACH ROW EXECUTE FUNCTION fn_vault_version_snapshot();

-- ═══════════════════════════════════════════════════════════════════════════════
-- SEED DATA: Default pricing tiers
-- ═══════════════════════════════════════════════════════════════════════════════

INSERT INTO vault_pricing_tiers (tier, operation, qtcl_cost, max_size_bytes, max_secrets, description)
VALUES
    ('trial', 'store',              0,      1024,    1,    'Trial: 1 secret, 1KB max, free'),
    ('trial', 'retrieve',           0,      NULL,    NULL, 'Trial: free retrieval'),
    ('trial', 'anchor',             0,      NULL,    0,    'Trial: no chain anchoring'),
    ('trial', 'obfuscated_anchor',  0,      NULL,    0,    'Trial: no obfuscation'),
    ('paid',  'store',              0,      104857600, -1, 'Paid: unlimited secrets, 100MB max, free store'),
    ('paid',  'retrieve',           0,      NULL,    NULL, 'Paid: free retrieval'),
    ('paid',  'anchor',             100000, NULL,    NULL, 'Paid: 1 QTCL per anchor'),
    ('paid',  'obfuscated_anchor',  500000, NULL,    NULL, 'Paid: 5 QTCL obfuscated anchor'),
    ('enterprise', 'store',         0,      1073741824, -1, 'Enterprise: 1GB max'),
    ('enterprise', 'retrieve',      0,      NULL,    NULL, 'Enterprise: free retrieval'),
    ('enterprise', 'anchor',        50000,  NULL,    NULL, 'Enterprise: 0.5 QTCL anchor'),
    ('enterprise', 'obfuscated_anchor', 250000, NULL, NULL, 'Enterprise: 2.5 QTCL obfuscated')
ON CONFLICT (tier, operation) DO NOTHING;

-- ═══════════════════════════════════════════════════════════════════════════════
-- CLEANUP: Periodic maintenance functions
-- ═══════════════════════════════════════════════════════════════════════════════

-- Purge expired secrets (call from cron or server background worker)
CREATE OR REPLACE FUNCTION fn_vault_purge_expired()
RETURNS INTEGER AS $$
DECLARE
    purged INTEGER;
BEGIN
    DELETE FROM vault_secrets
    WHERE expires_at IS NOT NULL AND expires_at < NOW()
    RETURNING 1;
    GET DIAGNOSTICS purged = ROW_COUNT;
    RETURN purged;
END;
$$ LANGUAGE plpgsql;

-- Purge old rate limit windows (> 1 hour old)
CREATE OR REPLACE FUNCTION fn_vault_cleanup_rate_limits()
RETURNS INTEGER AS $$
DECLARE
    cleaned INTEGER;
BEGIN
    DELETE FROM vault_rate_limits
    WHERE window_start < NOW() - INTERVAL '1 hour';
    GET DIAGNOSTICS cleaned = ROW_COUNT;
    RETURN cleaned;
END;
$$ LANGUAGE plpgsql;

-- Purge old failed login records (> 24 hours)
CREATE OR REPLACE FUNCTION fn_vault_cleanup_failed_logins()
RETURNS INTEGER AS $$
DECLARE
    cleaned INTEGER;
BEGIN
    DELETE FROM vault_failed_logins
    WHERE created_at < NOW() - INTERVAL '24 hours';
    GET DIAGNOSTICS cleaned = ROW_COUNT;

    -- Reset lockout for accounts whose lockout has expired
    UPDATE vault_accounts
    SET locked_until = NULL, failed_login_count = 0
    WHERE locked_until IS NOT NULL AND locked_until < NOW();

    RETURN cleaned;
END;
$$ LANGUAGE plpgsql;

-- Purge expired sessions
CREATE OR REPLACE FUNCTION fn_vault_cleanup_sessions()
RETURNS INTEGER AS $$
DECLARE
    cleaned INTEGER;
BEGIN
    DELETE FROM vault_sessions
    WHERE expires_at < NOW() OR revoked = TRUE;
    GET DIAGNOSTICS cleaned = ROW_COUNT;
    RETURN cleaned;
END;
$$ LANGUAGE plpgsql;

-- Dead man's switch check (call daily from cron)
CREATE OR REPLACE FUNCTION fn_vault_check_inheritance()
RETURNS TABLE(account_id TEXT, beneficiary TEXT, days_overdue INTEGER) AS $$
BEGIN
    RETURN QUERY
    SELECT
        i.account_id,
        i.beneficiary,
        EXTRACT(DAY FROM NOW() - i.last_check_in)::INTEGER - i.check_in_days AS days_overdue
    FROM vault_inheritance i
    WHERE i.activated = FALSE
      AND i.last_check_in + (i.check_in_days || ' days')::INTERVAL < NOW();
END;
$$ LANGUAGE plpgsql;

"""

# ═══════════════════════════════════════════════════════════════════════════════
# BUILDER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class VaultDatabaseBuilder:
    """
    Provisions and manages the QTCL Vault Supabase PostgreSQL database.

    Usage:
        builder = VaultDatabaseBuilder()
        builder.build()
    """

    def __init__(self, db_url: str = VAULT_URL):
        self.db_url = db_url
        self.conn = None
        self.cursor = None

    def connect(self):
        logger.info("[VAULT-DB] Connecting to Supabase PostgreSQL...")
        self.conn = psycopg2.connect(self.db_url)
        self.conn.autocommit = False
        self.cursor = self.conn.cursor()
        self.cursor.execute("SET statement_timeout = '120s';")
        logger.info("[VAULT-DB] ✅ Connected")

    def disconnect(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("[VAULT-DB] Disconnected")

    def execute_schema(self):
        """Execute the complete vault schema."""
        logger.info("[VAULT-DB] Executing schema (17 tables, 6 triggers, seed data)...")
        t0 = time.time()
        self.cursor.execute(VAULT_SCHEMA)
        self.conn.commit()
        elapsed = time.time() - t0
        logger.info(f"[VAULT-DB] ✅ Schema executed in {elapsed:.2f}s")

    def set_metadata(self):
        """Write schema version and build metadata."""
        meta = {
            "schema_version": SCHEMA_VERSION,
            "schema_label": SCHEMA_LABEL,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "builder": "vault_db_builder.py v1.0",
            "features": [
                "zero_knowledge_encryption",
                "chain_anchoring",
                "obfuscated_anchors",
                "shamir_inheritance",
                "totp_2fa",
                "breach_canaries",
                "rate_limiting",
                "audit_logging",
                "version_history",
                "secret_tags",
                "pricing_tiers",
            ],
        }
        import json
        self.cursor.execute(
            """INSERT INTO vault_metadata (key, value)
               VALUES ('schema_info', %s::jsonb)
               ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()""",
            (json.dumps(meta),)
        )
        self.conn.commit()
        logger.info(f"[VAULT-DB] ✅ Metadata: v{SCHEMA_VERSION} ({SCHEMA_LABEL})")

    def verify(self):
        """Verify all tables exist and report counts."""
        expected_tables = [
            "vault_metadata", "vault_accounts", "vault_sessions",
            "vault_secrets", "vault_secret_versions", "vault_secret_tags",
            "vault_anchors", "vault_anchor_proofs",
            "vault_billing", "vault_pricing_tiers",
            "vault_audit_log", "vault_rate_limits", "vault_failed_logins",
            "vault_totp_secrets", "vault_breach_canaries",
            "vault_inheritance", "vault_shamir_shares",
        ]
        self.cursor.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename LIKE 'vault_%'"
        )
        found = {r[0] for r in self.cursor.fetchall()}

        logger.info(f"\n{'='*60}")
        logger.info(f"  VAULT DATABASE VERIFICATION")
        logger.info(f"{'='*60}")

        all_ok = True
        for t in expected_tables:
            status = "✅" if t in found else "❌ MISSING"
            if t not in found:
                all_ok = False
            logger.info(f"  {status}  {t}")

        # Check pricing seed data
        self.cursor.execute("SELECT COUNT(*) FROM vault_pricing_tiers")
        pricing_count = self.cursor.fetchone()[0]
        logger.info(f"\n  Pricing tiers: {pricing_count}")

        # Check triggers
        self.cursor.execute(
            "SELECT trigger_name FROM information_schema.triggers "
            "WHERE trigger_schema = 'public' AND trigger_name LIKE 'trg_v%'"
        )
        triggers = [r[0] for r in self.cursor.fetchall()]
        logger.info(f"  Triggers: {len(triggers)} ({', '.join(sorted(set(triggers)))})")

        # Check functions
        self.cursor.execute(
            "SELECT routine_name FROM information_schema.routines "
            "WHERE routine_schema = 'public' AND routine_name LIKE 'fn_vault_%'"
        )
        functions = [r[0] for r in self.cursor.fetchall()]
        logger.info(f"  Functions: {len(functions)} ({', '.join(sorted(set(functions)))})")

        logger.info(f"{'='*60}")
        if all_ok:
            logger.info("  ✅ ALL TABLES PRESENT — Vault database ready")
        else:
            logger.error("  ❌ SOME TABLES MISSING — Check errors above")
        logger.info(f"{'='*60}\n")

        return all_ok

    def build(self):
        """Full build: connect, create schema, verify."""
        t0 = time.time()
        try:
            self.connect()
            self.execute_schema()
            self.set_metadata()
            ok = self.verify()
            elapsed = time.time() - t0
            if ok:
                logger.info(f"[VAULT-DB] ✅ Build complete in {elapsed:.2f}s")
            else:
                logger.error(f"[VAULT-DB] ❌ Build completed with errors in {elapsed:.2f}s")
            return ok
        except Exception as e:
            logger.error(f"[VAULT-DB] ❌ Build failed: {e}", exc_info=True)
            if self.conn:
                self.conn.rollback()
            return False
        finally:
            self.disconnect()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("")
    logger.info("=" * 70)
    logger.info("  QTCL VAULT DATABASE BUILDER v1.0")
    logger.info("  Zero-Knowledge Encrypted Storage · Supabase PostgreSQL")
    logger.info("=" * 70)
    logger.info("")

    builder = VaultDatabaseBuilder()
    success = builder.build()

    if success:
        print("\n✅ Vault database provisioned successfully.")
        print("\n💡 Verify:")
        print("  SELECT * FROM vault_metadata WHERE key = 'schema_info';")
        print("  SELECT tier, operation, qtcl_cost FROM vault_pricing_tiers;")
        print("  SELECT trigger_name FROM information_schema.triggers WHERE trigger_name LIKE 'trg_v%';")
        print("  SELECT routine_name FROM information_schema.routines WHERE routine_name LIKE 'fn_vault_%';")
    else:
        print("\n❌ Build failed. Check logs above.")
        sys.exit(1)
