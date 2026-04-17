-- ════════════════════════════════════════════════════════════════════════════════
-- QTCL MASTER SCHEMA PATCH — PRODUCTION CONSOLIDATED
-- ════════════════════════════════════════════════════════════════════════════════
-- Target: Supabase / PostgreSQL 15+
-- Purpose: Consolidates all QTCL migrations into one idempotent script.
-- Fixes: missing node_id in peer_registry, mermin columns in blocks,
--        and all P2P v2 / Quantum W-State fields.
-- ════════════════════════════════════════════════════════════════════════════════

BEGIN;

-- ─────────────────────────────────────────────────────────────────────────────
-- 0. EXTENSIONS & SCHEMA VERSIONING
-- ─────────────────────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS schema_migrations (
    version     TEXT        PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    description TEXT
);

INSERT INTO schema_migrations (version, description)
VALUES ('master_mega_20260404', 'Consolidated production-grade master schema')
ON CONFLICT (version) DO NOTHING;

-- ─────────────────────────────────────────────────────────────────────────────
-- 1. CORE BLOCKCHAIN: blocks TABLE
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS blocks (
    height             BIGINT PRIMARY KEY,
    block_number       BIGINT,
    block_hash         TEXT UNIQUE NOT NULL,
    previous_hash      TEXT,
    timestamp          BIGINT,
    oracle_w_state_hash TEXT,
    validator_public_key TEXT,
    nonce              BIGINT,
    difficulty         BIGINT,
    entropy_score      FLOAT,
    transactions_root  TEXT,
    created_at         TIMESTAMPTZ DEFAULT NOW()
);

-- Ensure all enhanced columns exist (idempotent ALTERs)
ALTER TABLE blocks ADD COLUMN IF NOT EXISTS pq_curr BIGINT NOT NULL DEFAULT 0;
ALTER TABLE blocks ADD COLUMN IF NOT EXISTS pq_last BIGINT NOT NULL DEFAULT 0;
ALTER TABLE blocks ADD COLUMN IF NOT EXISTS pq0 BIGINT DEFAULT 0;
ALTER TABLE blocks ADD COLUMN IF NOT EXISTS hyp_triangle_area NUMERIC(18,9) DEFAULT 0;
ALTER TABLE blocks ADD COLUMN IF NOT EXISTS hyp_dist_0c NUMERIC(18,9) DEFAULT 0;
ALTER TABLE blocks ADD COLUMN IF NOT EXISTS hyp_dist_cl NUMERIC(18,9) DEFAULT 0;
ALTER TABLE blocks ADD COLUMN IF NOT EXISTS hyp_dist_0l NUMERIC(18,9) DEFAULT 0;
ALTER TABLE blocks ADD COLUMN IF NOT EXISTS oracle_quorum_hash VARCHAR(64) DEFAULT NULL;
ALTER TABLE blocks ADD COLUMN IF NOT EXISTS peer_measurement_count INTEGER DEFAULT 1;
ALTER TABLE blocks ADD COLUMN IF NOT EXISTS consensus_agreement NUMERIC(5,4) DEFAULT 0;
ALTER TABLE blocks ADD COLUMN IF NOT EXISTS local_dm_hex VARCHAR(128) DEFAULT NULL;
ALTER TABLE blocks ADD COLUMN IF NOT EXISTS local_measurement_sig VARCHAR(64) DEFAULT NULL;
ALTER TABLE blocks ADD COLUMN IF NOT EXISTS oracle_consensus_reached BOOLEAN DEFAULT FALSE;

-- ⚛️ CRITICAL FIX: Mermin columns causing RPC errors
ALTER TABLE blocks ADD COLUMN IF NOT EXISTS mermin_value FLOAT DEFAULT 0.0;
ALTER TABLE blocks ADD COLUMN IF NOT EXISTS mermin_violated BOOLEAN DEFAULT FALSE;
ALTER TABLE blocks ADD COLUMN IF NOT EXISTS quantum_validation_status VARCHAR(50) DEFAULT 'unvalidated';
ALTER TABLE blocks ADD COLUMN IF NOT EXISTS quantum_measurements_count INTEGER DEFAULT 0;
ALTER TABLE blocks ADD COLUMN IF NOT EXISTS validated_at TIMESTAMP WITH TIME ZONE;
ALTER TABLE blocks ADD COLUMN IF NOT EXISTS validation_entropy_avg NUMERIC(5,4);

CREATE INDEX IF NOT EXISTS idx_blocks_hash ON blocks (block_hash);
CREATE INDEX IF NOT EXISTS idx_blocks_quorum_hash ON blocks (oracle_quorum_hash) WHERE oracle_quorum_hash IS NOT NULL;

-- ─────────────────────────────────────────────────────────────────────────────
-- 2. CORE BLOCKCHAIN: transactions TABLE
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS transactions (
    tx_hash           TEXT PRIMARY KEY,
    from_address      TEXT,
    to_address        TEXT,
    amount            NUMERIC(30,0) DEFAULT 0,
    nonce             BIGINT        DEFAULT 0,
    height            INTEGER,
    block_hash        TEXT,
    transaction_index INTEGER       DEFAULT 0,
    tx_type           TEXT          DEFAULT 'transfer',
    status            TEXT          DEFAULT 'pending',
    created_at        TIMESTAMPTZ   DEFAULT NOW(),
    updated_at        TIMESTAMPTZ   DEFAULT NOW()
);

ALTER TABLE transactions ADD COLUMN IF NOT EXISTS quantum_state_hash TEXT;
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS commitment_hash TEXT;
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS metadata JSONB;
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS finalized_at TIMESTAMP WITH TIME ZONE;
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS circuit_size INT;
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS circuit_depth INT;
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS ghz_fidelity FLOAT;
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS noise_source VARCHAR(50);
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS validator_agreement FLOAT DEFAULT 0.0;
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS dominant_bitstring VARCHAR(255);
CREATE INDEX IF NOT EXISTS idx_tx_height ON transactions (height DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_tx_block_hash ON transactions (block_hash);
CREATE INDEX IF NOT EXISTS idx_tx_type ON transactions (tx_type);

-- ─────────────────────────────────────────────────────────────────────────────
-- 3. NETWORK: peer_registry TABLE
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS peer_registry (
    node_id       TEXT PRIMARY KEY,
    external_addr TEXT NOT NULL,
    pubkey_hash   TEXT NOT NULL DEFAULT '',
    chain_height  BIGINT      DEFAULT 0,
    last_seen     TIMESTAMPTZ DEFAULT NOW(),
    first_seen    TIMESTAMPTZ DEFAULT NOW(),
    capabilities  JSONB       DEFAULT '[]',
    ban_score     INTEGER     DEFAULT 0,
    caller_ip     TEXT        DEFAULT '',
    mac_address   TEXT        DEFAULT '',
    device_id     TEXT        DEFAULT '',
    fingerprint   TEXT        DEFAULT ''
);

-- Ensure node_id and others exist if table was legacy
DO $$
BEGIN
    -- 1. Aggressive schema cleanup: if peer_id exists, we migrate and rebuild
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='peer_registry' AND column_name='peer_id') THEN
        -- Create a temporary table to hold data
        CREATE TEMP TABLE peer_registry_backup AS SELECT * FROM peer_registry;
        
        -- Drop the old table completely to clear all constraints
        DROP TABLE peer_registry CASCADE;
        
        -- Recreate correctly
        CREATE TABLE peer_registry (
            node_id       TEXT PRIMARY KEY,
            external_addr TEXT NOT NULL,
            pubkey_hash   TEXT NOT NULL DEFAULT '',
            chain_height  BIGINT      DEFAULT 0,
            last_seen     TIMESTAMPTZ DEFAULT NOW(),
            first_seen    TIMESTAMPTZ DEFAULT NOW(),
            capabilities  JSONB       DEFAULT '[]',
            ban_score     INTEGER     DEFAULT 0,
            caller_ip     TEXT        DEFAULT '',
            mac_address   TEXT        DEFAULT '',
            device_id     TEXT        DEFAULT '',
            fingerprint   TEXT        DEFAULT ''
        );
        
        -- Restore data (mapping peer_id to node_id)
        INSERT INTO peer_registry (node_id, external_addr, pubkey_hash, chain_height, last_seen, first_seen)
        SELECT peer_id, COALESCE(ip_address, '') || ':' || COALESCE(port::text, '9091'), public_key, block_height, last_seen, created_at
        FROM peer_registry_backup;
        
        RAISE NOTICE 'peer_registry migrated and rebuilt successfully';
    ELSE
        -- Just ensure columns exist if already in node_id format
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='peer_registry' AND column_name='node_id') THEN
            ALTER TABLE peer_registry ADD COLUMN node_id TEXT;
        END IF;
    END IF;

    -- Ensure node_id is the primary key (in case it was added later)
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid = 'peer_registry'::regclass AND contype = 'p') THEN
        BEGIN
            ALTER TABLE peer_registry ADD PRIMARY KEY (node_id);
        EXCEPTION WHEN OTHERS THEN 
            -- If primary key already exists on another column, just add unique constraint
            IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'peer_registry_node_id_key') THEN
                ALTER TABLE peer_registry ADD CONSTRAINT peer_registry_node_id_key UNIQUE (node_id);
            END IF;
        END;
    END IF;
END $$;

ALTER TABLE peer_registry ADD COLUMN IF NOT EXISTS external_addr TEXT;
ALTER TABLE peer_registry ADD COLUMN IF NOT EXISTS pubkey_hash TEXT DEFAULT '';
ALTER TABLE peer_registry ADD COLUMN IF NOT EXISTS chain_height BIGINT DEFAULT 0;
ALTER TABLE peer_registry ADD COLUMN IF NOT EXISTS last_seen TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE peer_registry ADD COLUMN IF NOT EXISTS first_seen TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE peer_registry ADD COLUMN IF NOT EXISTS capabilities JSONB DEFAULT '[]';
ALTER TABLE peer_registry ADD COLUMN IF NOT EXISTS ban_score INTEGER DEFAULT 0;
ALTER TABLE peer_registry ADD COLUMN IF NOT EXISTS caller_ip TEXT DEFAULT '';
ALTER TABLE peer_registry ADD COLUMN IF NOT EXISTS mac_address TEXT DEFAULT '';
ALTER TABLE peer_registry ADD COLUMN IF NOT EXISTS device_id TEXT DEFAULT '';
ALTER TABLE peer_registry ADD COLUMN IF NOT EXISTS fingerprint TEXT DEFAULT '';

-- 3.1 DEVICE FINGERPRINTING TABLE
CREATE TABLE IF NOT EXISTS peer_devices (
    fingerprint    TEXT PRIMARY KEY,
    node_id        TEXT NOT NULL,
    last_caller_ip TEXT,
    mac_address    TEXT,
    device_id      TEXT,
    first_seen     TIMESTAMPTZ DEFAULT NOW(),
    last_seen      TIMESTAMPTZ DEFAULT NOW(),
    trust_score    FLOAT DEFAULT 1.0
);
CREATE INDEX IF NOT EXISTS idx_peer_devices_node ON peer_devices(node_id);
CREATE INDEX IF NOT EXISTS idx_peer_devices_ip ON peer_devices(last_caller_ip);


-- ─────────────────────────────────────────────────────────────────────────────
-- 4. QUANTUM ANALYTICS: quantum_metrics & measurements
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS quantum_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    engine VARCHAR(50) DEFAULT 'QTCL-QE v8.0',
    w_state_coherence_avg FLOAT DEFAULT 0.0,
    w_state_fidelity_avg FLOAT DEFAULT 0.0,
    w_state_entanglement FLOAT DEFAULT 0.0,
    noise_kappa FLOAT DEFAULT 0.08,
    bell_s_chsh_mean FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS quantum_measurements (
    id BIGSERIAL PRIMARY KEY,
    tx_id VARCHAR(255) UNIQUE NOT NULL,
    circuit_name VARCHAR(255),
    num_qubits INT DEFAULT 8,
    measurement_result_json JSONB,
    validator_consensus_json JSONB,
    entropy_percent FLOAT,
    ghz_fidelity FLOAT,
    validator_agreement_score FLOAT,
    block_hash VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_valid_finality BOOLEAN DEFAULT FALSE,
    noise_source VARCHAR(50) DEFAULT 'xorshift64'
);

CREATE TABLE IF NOT EXISTS oracle_measurements (
  id               BIGSERIAL PRIMARY KEY,
  cycle            BIGINT NOT NULL,
  timestamp_ns     BIGINT NOT NULL,
  lattice_fidelity DOUBLE PRECISION,
  pq_curr          BIGINT DEFAULT 0,
  pq_last          BIGINT DEFAULT 0,
  mermin_violation DOUBLE PRECISION,
  mermin_angles    JSONB,
  dm_hex           TEXT,
  per_node_json    JSONB
);

-- ─────────────────────────────────────────────────────────────────────────────
-- 5. AUTH & USER MANAGEMENT
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(255) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash TEXT,
    password_method VARCHAR(50) DEFAULT 'hlwe',
    role VARCHAR(50) DEFAULT 'user',
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    access_token TEXT NOT NULL,
    refresh_token TEXT NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS auth_events (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT,
    event_type VARCHAR(50) NOT NULL,
    success BOOLEAN DEFAULT FALSE,
    details TEXT,
    ip_address VARCHAR(45),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────────────────────
-- 6. P2P v2 ADVANCED TABLES
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS p2p_peers (
    node_id_hex         VARCHAR(32)   PRIMARY KEY,
    host                VARCHAR(253)  NOT NULL,
    port                INTEGER       NOT NULL,
    services            SMALLINT      NOT NULL DEFAULT 1,
    chain_height        BIGINT        NOT NULL DEFAULT 0,
    last_fidelity       NUMERIC(8,6)  NOT NULL DEFAULT 0,
    last_seen_at        TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    ban_score           SMALLINT      NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS wstate_measurements (
    id                  BIGSERIAL     PRIMARY KEY,
    node_id_hex         VARCHAR(32)   NOT NULL,
    chain_height        BIGINT        NOT NULL,
    w_fidelity          NUMERIC(8,6)  NOT NULL DEFAULT 0,
    coherence           NUMERIC(8,6)  NOT NULL DEFAULT 0,
    purity              NUMERIC(8,6)  NOT NULL DEFAULT 0,
    auth_tag_hex        VARCHAR(64)   NOT NULL,
    received_at         TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS wstate_consensus_log (
    chain_height            BIGINT        PRIMARY KEY,
    block_hash              VARCHAR(64)   NOT NULL,
    median_fidelity         NUMERIC(8,6)  NOT NULL DEFAULT 0,
    quorum_hash             VARCHAR(64)   NOT NULL,
    peer_count              INTEGER       NOT NULL DEFAULT 1,
    agreement_score         NUMERIC(5,4)  NOT NULL DEFAULT 0,
    participant_node_ids    TEXT[],
    consensus_computed_at   TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────────────────────
-- 7. STORED PROCEDURES & TRIGGERS
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply to relevant tables
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'tr_blocks_updated') THEN
        CREATE TRIGGER tr_blocks_updated BEFORE UPDATE ON blocks FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;
EXCEPTION WHEN undefined_column THEN
    -- Handle case where updated_at doesn't exist yet
END $$;

-- ─────────────────────────────────────────────────────────────────────────────
-- 8. VIEWS
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW v_network_health AS
SELECT
    COUNT(*)                                        AS total_known_peers,
    COUNT(*) FILTER (WHERE last_seen > NOW() - INTERVAL '10 minutes')
                                                    AS peers_active_10m,
    MAX(chain_height)                               AS max_peer_height,
    ROUND(AVG(chain_height)::NUMERIC, 2)            AS avg_height
FROM peer_registry;

CREATE OR REPLACE VIEW v_device_stats AS
SELECT
    COUNT(DISTINCT fingerprint) AS total_unique_devices,
    COUNT(DISTINCT last_caller_ip) AS total_nat_groups,
    AVG(trust_score) AS avg_device_trust
FROM peer_devices;

-- ─────────────────────────────────────────────────────────────────────────────
-- 9. WALLET & BALANCES
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS wallet_addresses (
    address           TEXT PRIMARY KEY,
    wallet_fingerprint TEXT,
    public_key        TEXT,
    balance           NUMERIC(30,0) DEFAULT 0,
    transaction_count INTEGER DEFAULT 0,
    address_type      VARCHAR(20) DEFAULT 'user',
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    updated_at        TIMESTAMPTZ DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────────────────────
-- 10. SYSTEM STATE & ORACLE REGISTRY
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chain_state (
    state_id         INTEGER PRIMARY KEY,
    chain_height     BIGINT      DEFAULT 0,
    head_block_hash  TEXT        DEFAULT '',
    latest_coherence NUMERIC(5,4) DEFAULT 0.9,
    updated_at       TIMESTAMPTZ DEFAULT NOW()
);

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
);

-- Ensure oracle_registry has all columns
ALTER TABLE oracle_registry ADD COLUMN IF NOT EXISTS wallet_address VARCHAR(128) DEFAULT '';
ALTER TABLE oracle_registry ADD COLUMN IF NOT EXISTS oracle_pub_key TEXT DEFAULT '';
ALTER TABLE oracle_registry ADD COLUMN IF NOT EXISTS cert_sig VARCHAR(128) DEFAULT '';
ALTER TABLE oracle_registry ADD COLUMN IF NOT EXISTS mode VARCHAR(32) DEFAULT 'full';
ALTER TABLE oracle_registry ADD COLUMN IF NOT EXISTS ip_hint VARCHAR(256) DEFAULT '';
ALTER TABLE oracle_registry ADD COLUMN IF NOT EXISTS reg_tx_hash VARCHAR(64) DEFAULT '';
ALTER TABLE oracle_registry ADD COLUMN IF NOT EXISTS registered_at TIMESTAMPTZ DEFAULT NOW();

COMMIT;

-- ════════════════════════════════════════════════════════════════════════════════
-- END OF MASTER PATCH
-- ════════════════════════════════════════════════════════════════════════════════
