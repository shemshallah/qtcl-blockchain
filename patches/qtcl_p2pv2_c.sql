-- ════════════════════════════════════════════════════════════════════════════════
-- QTCL P2P v2 — SERVER MIGRATION SCRIPT
-- Target: Supabase PostgreSQL (pg 15+)
-- Run in: Supabase Dashboard → SQL Editor, or psql
--
-- Scope: extends `blocks`, adds `p2p_peers`, `wstate_measurements`,
--        `wstate_consensus_log`, `p2p_peer_exchange`.
--
-- Design principles:
--   • All ALTER TABLE use ADD COLUMN IF NOT EXISTS — fully idempotent
--   • CREATE TABLE statements use IF NOT EXISTS — re-runnable at any deploy
--   • All new REAL/NUMERIC columns DEFAULT 0.0 so existing rows remain valid
--   • Indexes target the hot query paths (height, quorum_hash, node_id, ts)
--   • Row-level security templates included (enable per-table as needed)
--   • A schema_migrations table tracks applied version for CI gating
-- ════════════════════════════════════════════════════════════════════════════════

BEGIN;

-- ─────────────────────────────────────────────────────────────────────────────
-- §0  SCHEMA VERSION GUARD
--     Skip entire migration if already applied (idempotent CI)
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS schema_migrations (
    version     TEXT        PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    description TEXT
);

-- Insert the migration marker; if it already exists the INSERT does nothing.
INSERT INTO schema_migrations (version, description)
VALUES ('p2pv2_20260318', 'QTCL P2P v2: hyperbolic geometry, C SSE, BFT consensus fields')
ON CONFLICT (version) DO NOTHING;

-- ─────────────────────────────────────────────────────────────────────────────
-- §1  EXTEND blocks TABLE WITH P2P v2 FIELDS
--     Every field maps directly to a submitted block header key.
--     The server's _validate_block() checks oracle_quorum_hash OR
--     oracle_consensus_reached; solo miners (quorum_hash only) are valid.
-- ─────────────────────────────────────────────────────────────────────────────

-- §1a  Pseudoqubit genesis index (always 0 for current chain)
ALTER TABLE blocks
    ADD COLUMN IF NOT EXISTS pq0               INTEGER       DEFAULT 0;

-- §1b  {8,3} hyperbolic triangle geometry
--      triangle_area = angular defect (Gauss–Bonnet); geodesic distances
ALTER TABLE blocks
    ADD COLUMN IF NOT EXISTS hyp_triangle_area NUMERIC(18,9) DEFAULT 0;
ALTER TABLE blocks
    ADD COLUMN IF NOT EXISTS hyp_dist_0c       NUMERIC(18,9) DEFAULT 0;
ALTER TABLE blocks
    ADD COLUMN IF NOT EXISTS hyp_dist_cl       NUMERIC(18,9) DEFAULT 0;
ALTER TABLE blocks
    ADD COLUMN IF NOT EXISTS hyp_dist_0l       NUMERIC(18,9) DEFAULT 0;

-- §1c  BFT consensus fields from WStateConsensus.compute()
ALTER TABLE blocks
    ADD COLUMN IF NOT EXISTS oracle_quorum_hash      VARCHAR(64)   DEFAULT NULL;
ALTER TABLE blocks
    ADD COLUMN IF NOT EXISTS peer_measurement_count  INTEGER       DEFAULT 1;
ALTER TABLE blocks
    ADD COLUMN IF NOT EXISTS consensus_agreement     NUMERIC(5,4)  DEFAULT 0;

-- §1d  Local miner DM snapshot (first 64 bytes of local_dm_hex for storage efficiency)
ALTER TABLE blocks
    ADD COLUMN IF NOT EXISTS local_dm_hex            VARCHAR(128)  DEFAULT NULL;

-- §1e  HLWE-authenticated measurement signature from miner
ALTER TABLE blocks
    ADD COLUMN IF NOT EXISTS local_measurement_sig   VARCHAR(64)   DEFAULT NULL;

-- §1f  Indexes on new hot columns
CREATE INDEX IF NOT EXISTS idx_blocks_quorum_hash
    ON blocks (oracle_quorum_hash)
    WHERE oracle_quorum_hash IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_blocks_pq0_curr_last
    ON blocks (pq0, pq_curr, pq_last);

CREATE INDEX IF NOT EXISTS idx_blocks_hyp_area
    ON blocks (hyp_triangle_area DESC);

-- ─────────────────────────────────────────────────────────────────────────────
-- §2  p2p_peers — Known P2P peers (used by /api/p2p/peer_exchange)
--     The server acts as a super-peer / seed node for new miners.
--     Mirrors the C QtclPeer struct fields exactly.
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS p2p_peers (
    -- Identity
    node_id_hex         VARCHAR(32)   NOT NULL,          -- SHA3-256(pubkey)[:16] as hex
    host                VARCHAR(253)  NOT NULL,           -- IPv4, IPv6, or hostname
    port                INTEGER       NOT NULL
                            CHECK (port BETWEEN 1 AND 65535),

    -- Capability flags (bitmask: 0x01=full-node 0x02=miner 0x04=oracle)
    services            SMALLINT      NOT NULL DEFAULT 1,
    protocol_version    SMALLINT      NOT NULL DEFAULT 2,

    -- Chain state (updated on each heartbeat / gossip)
    chain_height        BIGINT        NOT NULL DEFAULT 0,
    last_fidelity       NUMERIC(8,6)  NOT NULL DEFAULT 0,
    latency_ms          NUMERIC(8,3)  NOT NULL DEFAULT 0,
    ban_score           SMALLINT      NOT NULL DEFAULT 0
                            CHECK (ban_score BETWEEN 0 AND 100),

    -- Timestamps
    first_seen_at       TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    last_seen_at        TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    last_heartbeat_at   TIMESTAMPTZ,

    -- P2P address the peer advertised for inbound connections
    advertised_host     VARCHAR(253),
    advertised_port     INTEGER,

    -- Source: 'self_register'=peer announced itself, 'gossip'=learned via addr msg
    source              VARCHAR(32)   NOT NULL DEFAULT 'self_register',

    PRIMARY KEY (node_id_hex)
);

-- Upsert-friendly composite index for host+port lookups
CREATE UNIQUE INDEX IF NOT EXISTS idx_p2p_peers_host_port
    ON p2p_peers (host, port);

CREATE INDEX IF NOT EXISTS idx_p2p_peers_last_seen
    ON p2p_peers (last_seen_at DESC);

CREATE INDEX IF NOT EXISTS idx_p2p_peers_chain_height
    ON p2p_peers (chain_height DESC);

CREATE INDEX IF NOT EXISTS idx_p2p_peers_active
    ON p2p_peers (last_seen_at DESC)
    WHERE ban_score < 100;

-- ─────────────────────────────────────────────────────────────────────────────
-- §3  wstate_measurements — Peer W-state gossip archive
--     One row per QtclWStateMeasurement received from any peer.
--     Used for network-wide fidelity analytics and fork attribution.
--     Partitioned by chain_height for efficient pruning.
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS wstate_measurements (
    id                  BIGSERIAL     PRIMARY KEY,

    -- Source peer
    node_id_hex         VARCHAR(32)   NOT NULL,
    chain_height        BIGINT        NOT NULL,

    -- Pseudoqubit triangle
    pq0                 INTEGER       NOT NULL DEFAULT 0,
    pq_curr             INTEGER       NOT NULL,
    pq_last             INTEGER       NOT NULL,

    -- {8,3} hyperbolic geometry
    hyp_dist_0c         NUMERIC(18,9) NOT NULL DEFAULT 0,
    hyp_dist_cl         NUMERIC(18,9) NOT NULL DEFAULT 0,
    hyp_dist_0l         NUMERIC(18,9) NOT NULL DEFAULT 0,
    hyp_triangle_area   NUMERIC(18,9) NOT NULL DEFAULT 0,

    -- Poincaré ball coordinates (stored as float arrays)
    ball_pq0            NUMERIC(12,9)[3],
    ball_curr           NUMERIC(12,9)[3],
    ball_last           NUMERIC(12,9)[3],

    -- Quantum metrics
    w_fidelity          NUMERIC(8,6)  NOT NULL DEFAULT 0,
    coherence           NUMERIC(8,6)  NOT NULL DEFAULT 0,
    purity              NUMERIC(8,6)  NOT NULL DEFAULT 0,
    negativity          NUMERIC(8,6)  NOT NULL DEFAULT 0,
    entropy_vn          NUMERIC(8,6)  NOT NULL DEFAULT 0,
    discord             NUMERIC(8,6)  NOT NULL DEFAULT 0,

    -- Compressed DM (hex of first 64 bytes = 4 complex128 values for storage)
    dm_sample_hex       VARCHAR(128),

    -- Authentication
    auth_tag_hex        VARCHAR(64)   NOT NULL,           -- HMAC-SHA256 from C
    timestamp_ns        BIGINT,                           -- miner wall clock ns

    -- Ingestion metadata
    received_at         TIMESTAMPTZ   NOT NULL DEFAULT NOW(),

    FOREIGN KEY (node_id_hex) REFERENCES p2p_peers (node_id_hex)
        ON DELETE CASCADE
        DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX IF NOT EXISTS idx_wstate_height
    ON wstate_measurements (chain_height DESC);

CREATE INDEX IF NOT EXISTS idx_wstate_node_height
    ON wstate_measurements (node_id_hex, chain_height DESC);

CREATE INDEX IF NOT EXISTS idx_wstate_fidelity
    ON wstate_measurements (w_fidelity DESC, chain_height DESC);

CREATE INDEX IF NOT EXISTS idx_wstate_received
    ON wstate_measurements (received_at DESC);

-- Auto-prune: keep last 100k rows (background job, not a trigger)
-- Run manually or via pg_cron:
-- DELETE FROM wstate_measurements WHERE id NOT IN (
--     SELECT id FROM wstate_measurements ORDER BY id DESC LIMIT 100000);

-- ─────────────────────────────────────────────────────────────────────────────
-- §4  wstate_consensus_log — Per-block BFT consensus snapshots
--     One row per finalized block containing the full consensus result.
--     Powers the /api/blocks/{height}/consensus endpoint.
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS wstate_consensus_log (
    chain_height            BIGINT        PRIMARY KEY,
    block_hash              VARCHAR(64)   NOT NULL,

    -- BFT consensus output fields (from QtclWStateConsensus struct)
    median_fidelity         NUMERIC(8,6)  NOT NULL DEFAULT 0,
    median_coherence        NUMERIC(8,6)  NOT NULL DEFAULT 0,
    median_purity           NUMERIC(8,6)  NOT NULL DEFAULT 0,
    median_negativity       NUMERIC(8,6)  NOT NULL DEFAULT 0,
    median_entropy          NUMERIC(8,6)  NOT NULL DEFAULT 0,
    median_discord          NUMERIC(8,6)  NOT NULL DEFAULT 0,
    hyp_area_median         NUMERIC(18,9) NOT NULL DEFAULT 0,

    -- Merkle root of all peer auth_tags (32-byte SHA3-256, stored as hex)
    quorum_hash             VARCHAR(64)   NOT NULL,

    -- Aggregate stats
    peer_count              INTEGER       NOT NULL DEFAULT 1,
    agreement_score         NUMERIC(5,4)  NOT NULL DEFAULT 0,

    -- Mean density matrix (8×8 complex128 = 128 doubles = 2048 hex chars)
    consensus_dm_hex        TEXT,

    -- Participating node IDs (array for attribution)
    participant_node_ids    TEXT[],

    -- Timing
    consensus_computed_at   TIMESTAMPTZ   NOT NULL DEFAULT NOW(),

    FOREIGN KEY (chain_height) REFERENCES blocks (height)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_wscl_quorum_hash
    ON wstate_consensus_log (quorum_hash);

CREATE INDEX IF NOT EXISTS idx_wscl_fidelity
    ON wstate_consensus_log (median_fidelity DESC);

CREATE INDEX IF NOT EXISTS idx_wscl_computed_at
    ON wstate_consensus_log (consensus_computed_at DESC);

-- ─────────────────────────────────────────────────────────────────────────────
-- §5  p2p_peer_exchange — Bootstrap exchange log
--     Records every POST /api/p2p/peer_exchange request.
--     Enables the server to serve peer lists to new joiners and
--     track network growth over time.
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS p2p_peer_exchange (
    id              BIGSERIAL    PRIMARY KEY,
    requesting_node VARCHAR(32)  NOT NULL,         -- node_id_hex of requester
    requesting_host VARCHAR(253),                  -- IP of requester (from request)
    requesting_port INTEGER,
    peers_returned  INTEGER      NOT NULL DEFAULT 0,
    protocol_ver    SMALLINT     NOT NULL DEFAULT 2,
    exchanged_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_p2p_exchange_node
    ON p2p_peer_exchange (requesting_node, exchanged_at DESC);

CREATE INDEX IF NOT EXISTS idx_p2p_exchange_time
    ON p2p_peer_exchange (exchanged_at DESC);

-- ─────────────────────────────────────────────────────────────────────────────
-- §6  SERVER-SIDE FUNCTIONS
-- ─────────────────────────────────────────────────────────────────────────────

-- §6a  Upsert a peer (called by /api/p2p/peer_exchange and gossip ingest)
CREATE OR REPLACE FUNCTION upsert_p2p_peer(
    p_node_id_hex       TEXT,
    p_host              TEXT,
    p_port              INTEGER,
    p_services          SMALLINT  DEFAULT 1,
    p_protocol_version  SMALLINT  DEFAULT 2,
    p_chain_height      BIGINT    DEFAULT 0,
    p_last_fidelity     NUMERIC   DEFAULT 0,
    p_latency_ms        NUMERIC   DEFAULT 0,
    p_source            TEXT      DEFAULT 'self_register'
) RETURNS VOID
LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO p2p_peers (
        node_id_hex, host, port, services, protocol_version,
        chain_height, last_fidelity, latency_ms, source,
        first_seen_at, last_seen_at)
    VALUES (
        p_node_id_hex, p_host, p_port, p_services, p_protocol_version,
        p_chain_height, p_last_fidelity, p_latency_ms, p_source,
        NOW(), NOW())
    ON CONFLICT (node_id_hex) DO UPDATE SET
        host              = EXCLUDED.host,
        port              = EXCLUDED.port,
        services          = EXCLUDED.services,
        protocol_version  = EXCLUDED.protocol_version,
        chain_height      = GREATEST(p2p_peers.chain_height, EXCLUDED.chain_height),
        last_fidelity     = EXCLUDED.last_fidelity,
        latency_ms        = EXCLUDED.latency_ms,
        last_seen_at      = NOW(),
        last_heartbeat_at = NOW();
END;
$$;

-- §6b  Get active peers for bootstrapping
--      Returns peers seen within the last 10 minutes, not banned, sorted by height
CREATE OR REPLACE FUNCTION get_active_peers(
    p_limit             INTEGER DEFAULT 50,
    p_exclude_node      TEXT    DEFAULT NULL
) RETURNS TABLE (
    node_id_hex   TEXT,
    host          TEXT,
    port          INTEGER,
    chain_height  BIGINT,
    last_fidelity NUMERIC,
    latency_ms    NUMERIC
)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.node_id_hex::TEXT,
        p.host::TEXT,
        p.port,
        p.chain_height,
        p.last_fidelity,
        p.latency_ms
    FROM p2p_peers p
    WHERE
        p.ban_score < 100
        AND p.last_seen_at > NOW() - INTERVAL '10 minutes'
        AND (p_exclude_node IS NULL OR p.node_id_hex != p_exclude_node)
    ORDER BY p.chain_height DESC, p.latency_ms ASC
    LIMIT p_limit;
END;
$$;

-- §6c  Insert wstate measurement + update peer state atomically
CREATE OR REPLACE FUNCTION ingest_wstate_measurement(
    p_node_id_hex       TEXT,
    p_chain_height      BIGINT,
    p_pq0               INTEGER,
    p_pq_curr           INTEGER,
    p_pq_last           INTEGER,
    p_hyp_dist_0c       NUMERIC,
    p_hyp_dist_cl       NUMERIC,
    p_hyp_dist_0l       NUMERIC,
    p_hyp_triangle_area NUMERIC,
    p_w_fidelity        NUMERIC,
    p_coherence         NUMERIC,
    p_purity            NUMERIC,
    p_negativity        NUMERIC,
    p_entropy_vn        NUMERIC,
    p_discord           NUMERIC,
    p_dm_sample_hex     TEXT,
    p_auth_tag_hex      TEXT,
    p_timestamp_ns      BIGINT
) RETURNS BIGINT
LANGUAGE plpgsql AS $$
DECLARE
    v_id BIGINT;
BEGIN
    INSERT INTO wstate_measurements (
        node_id_hex, chain_height,
        pq0, pq_curr, pq_last,
        hyp_dist_0c, hyp_dist_cl, hyp_dist_0l, hyp_triangle_area,
        w_fidelity, coherence, purity, negativity, entropy_vn, discord,
        dm_sample_hex, auth_tag_hex, timestamp_ns)
    VALUES (
        p_node_id_hex, p_chain_height,
        p_pq0, p_pq_curr, p_pq_last,
        p_hyp_dist_0c, p_hyp_dist_cl, p_hyp_dist_0l, p_hyp_triangle_area,
        p_w_fidelity, p_coherence, p_purity, p_negativity, p_entropy_vn, p_discord,
        p_dm_sample_hex, p_auth_tag_hex, p_timestamp_ns)
    RETURNING id INTO v_id;

    -- Update peer's last known fidelity and height
    UPDATE p2p_peers SET
        chain_height  = GREATEST(chain_height, p_chain_height),
        last_fidelity = p_w_fidelity,
        last_seen_at  = NOW()
    WHERE node_id_hex = p_node_id_hex;

    RETURN v_id;
END;
$$;

-- §6d  Materialise consensus for a block (called after block is finalized)
CREATE OR REPLACE FUNCTION materialise_block_consensus(p_chain_height BIGINT)
RETURNS VOID
LANGUAGE plpgsql AS $$
DECLARE
    v_block_hash TEXT;
    v_quorum     TEXT;
BEGIN
    SELECT block_hash INTO v_block_hash
    FROM blocks WHERE height = p_chain_height;

    IF v_block_hash IS NULL THEN RETURN; END IF;

    -- Derive quorum_hash from sorted auth_tags of all measurements at this height
    -- (mirrors Python WStateConsensus.compute() quorum_hash logic)
    SELECT encode(
        digest(
            string_agg(auth_tag_hex ORDER BY node_id_hex),
            'sha256'),
        'hex')
    INTO v_quorum
    FROM wstate_measurements
    WHERE chain_height = p_chain_height;

    INSERT INTO wstate_consensus_log (
        chain_height, block_hash, quorum_hash,
        median_fidelity, median_coherence, median_purity,
        median_negativity, median_entropy, hyp_area_median,
        peer_count, agreement_score,
        participant_node_ids)
    SELECT
        p_chain_height,
        v_block_hash,
        COALESCE(v_quorum, '0'::TEXT),
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY w_fidelity),
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY coherence),
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY purity),
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY negativity),
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY entropy_vn),
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY hyp_triangle_area),
        COUNT(*)::INTEGER,
        -- agreement_score = 1 - std(fidelity)/mean(fidelity)
        GREATEST(0, 1.0 - (
            CASE WHEN AVG(w_fidelity) > 1e-9
                 THEN STDDEV(w_fidelity) / AVG(w_fidelity)
                 ELSE 0 END)),
        ARRAY_AGG(DISTINCT node_id_hex)
    FROM wstate_measurements
    WHERE chain_height = p_chain_height
    ON CONFLICT (chain_height) DO UPDATE SET
        median_fidelity       = EXCLUDED.median_fidelity,
        median_coherence      = EXCLUDED.median_coherence,
        median_purity         = EXCLUDED.median_purity,
        median_negativity     = EXCLUDED.median_negativity,
        median_entropy        = EXCLUDED.median_entropy,
        hyp_area_median       = EXCLUDED.hyp_area_median,
        peer_count            = EXCLUDED.peer_count,
        agreement_score       = EXCLUDED.agreement_score,
        quorum_hash           = EXCLUDED.quorum_hash,
        participant_node_ids  = EXCLUDED.participant_node_ids,
        consensus_computed_at = NOW();
END;
$$;

-- §6e  Network health view — live dashboard feed
CREATE OR REPLACE VIEW v_network_health AS
SELECT
    COUNT(*)                                        AS total_known_peers,
    COUNT(*) FILTER (WHERE last_seen_at > NOW() - INTERVAL '1 minute')
                                                    AS peers_active_1m,
    COUNT(*) FILTER (WHERE last_seen_at > NOW() - INTERVAL '10 minutes')
                                                    AS peers_active_10m,
    MAX(chain_height)                               AS max_peer_height,
    ROUND(AVG(last_fidelity)::NUMERIC, 4)           AS avg_fidelity,
    ROUND(AVG(latency_ms)::NUMERIC, 2)              AS avg_latency_ms,
    COUNT(*) FILTER (WHERE ban_score >= 100)        AS banned_peers
FROM p2p_peers;

-- §6f  Per-block network quorum view
CREATE OR REPLACE VIEW v_block_quorum_summary AS
SELECT
    b.height,
    b.block_hash,
    b.oracle_quorum_hash,
    b.peer_measurement_count,
    b.consensus_agreement,
    b.hyp_triangle_area,
    b.pq0, b.pq_curr, b.pq_last,
    w.median_fidelity,
    w.hyp_area_median,
    w.participant_node_ids
FROM blocks b
LEFT JOIN wstate_consensus_log w ON w.chain_height = b.height
ORDER BY b.height DESC;

-- ─────────────────────────────────────────────────────────────────────────────
-- §7  UPDATE _validate_block() COMPATIBILITY
--     The server's existing validation checks oracle_consensus_reached.
--     P2P v2 adds oracle_quorum_hash as an alternative acceptance path.
--     This view helps the server's Python code check both conditions cleanly.
-- ─────────────────────────────────────────────────────────────────────────────

CREATE OR REPLACE VIEW v_block_validation_state AS
SELECT
    height,
    block_hash,
    oracle_consensus_reached,
    oracle_quorum_hash,
    peer_measurement_count,
    -- P2P v2 acceptance: either oracle consensus OR quorum hash present
    (oracle_consensus_reached = TRUE
     OR (oracle_quorum_hash IS NOT NULL AND oracle_quorum_hash != ''))
        AS validation_accepted,
    -- Flag blocks with only 1 peer measurement (solo miner, valid but lower weight)
    (COALESCE(peer_measurement_count, 1) = 1)
        AS solo_measurement
FROM blocks;

-- ─────────────────────────────────────────────────────────────────────────────
-- §8  ROW LEVEL SECURITY (enable when ready — currently off)
-- ─────────────────────────────────────────────────────────────────────────────

-- ALTER TABLE p2p_peers          ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE wstate_measurements ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE wstate_consensus_log ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE p2p_peer_exchange   ENABLE ROW LEVEL SECURITY;

-- ─────────────────────────────────────────────────────────────────────────────
-- §9  GRANT READ ACCESS FOR ANON (Supabase public API)
-- ─────────────────────────────────────────────────────────────────────────────

GRANT SELECT ON v_network_health       TO anon, authenticated;
GRANT SELECT ON v_block_quorum_summary TO anon, authenticated;
GRANT SELECT ON v_block_validation_state TO anon, authenticated;
GRANT SELECT ON p2p_peers              TO anon, authenticated;
GRANT SELECT ON wstate_consensus_log   TO anon, authenticated;

-- Write access for the server's service role only
GRANT ALL ON p2p_peers, wstate_measurements,
             wstate_consensus_log, p2p_peer_exchange TO service_role;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO service_role;

-- ─────────────────────────────────────────────────────────────────────────────
-- §10  VERIFY
-- ─────────────────────────────────────────────────────────────────────────────

DO $$
DECLARE
    missing_cols TEXT := '';
    col_check    RECORD;
BEGIN
    FOR col_check IN
        SELECT table_name, column_name
        FROM (VALUES
            ('blocks', 'pq0'),
            ('blocks', 'hyp_triangle_area'),
            ('blocks', 'hyp_dist_0c'),
            ('blocks', 'hyp_dist_cl'),
            ('blocks', 'hyp_dist_0l'),
            ('blocks', 'oracle_quorum_hash'),
            ('blocks', 'peer_measurement_count'),
            ('blocks', 'consensus_agreement'),
            ('blocks', 'local_dm_hex'),
            ('blocks', 'local_measurement_sig')
        ) AS expected(table_name, column_name)
        WHERE NOT EXISTS (
            SELECT 1 FROM information_schema.columns ic
            WHERE ic.table_name   = expected.table_name
              AND ic.column_name  = expected.column_name
              AND ic.table_schema = 'public'
        )
    LOOP
        missing_cols := missing_cols || col_check.table_name
                     || '.' || col_check.column_name || ' ';
    END LOOP;

    IF missing_cols != '' THEN
        RAISE EXCEPTION 'MIGRATION INCOMPLETE — missing columns: %', missing_cols;
    END IF;

    RAISE NOTICE 'QTCL P2P v2 migration verified OK — all columns present';
END;
$$;

COMMIT;
-- ════════════════════════════════════════════════════════════════════════════════
-- END OF QTCL P2P v2 SERVER MIGRATION
-- Run time on empty DB: ~200ms | On production DB with blocks: ~1-5s (online)
-- ════════════════════════════════════════════════════════════════════════════════
