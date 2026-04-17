# QTCL Database Builder V8.2.0 - Technical Cheatsheet
## Complete Reference Guide for Developers and Administrators

---

## Table of Contents
1. [Database Modes](#database-modes)
2. [Complete Table Schema (69+ Tables)](#complete-table-schema)
3. [RLS Policies (100+ Policies)](#rls-policies)
4. [Trigger Functions](#trigger-functions)
5. [Triggers](#triggers)
6. [Roles and Permissions](#roles-and-permissions)
7. [CLI Commands](#cli-commands)
8. [Python Classes and Methods](#python-classes-and-methods)
9. [Environment Variables](#environment-variables)
10. [Security Audit](#security-audit)
11. [Troubleshooting](#troubleshooting)

---

## Database Modes

### Koyeb Mode (Production Server)
```bash
# Auto-detected via environment variables
export KOYEB=true
export KOYEB_APP_NAME="qtcl-blockchain"
export KOYEB_SERVICE_NAME="main"
export KOYEB_REGION="us-west-2"
export DATABASE_URL="postgresql://..."
export RLS_PASSWORD="your_secure_password"
```

**Features:**
- ✅ 69+ tables with RLS enabled
- ✅ 100+ RLS policies
- ✅ 5 password-protected roles
- ✅ 7 PostgreSQL trigger functions
- ✅ 9 triggers
- ✅ Maximum security hardening

### Client Mode (Local SQLite)
```bash
# Auto-detected when DATABASE_URL is not set
# Database location: <repo_root>/data/qtcl.db
```

**Features:**
- ✅ 4 SQLite-compatible triggers
- ✅ File permissions (0o600)
- ✅ RLS_PASSWORD fetched from Koyeb server
- ❌ No RLS (SQLite limitation)

---

## Complete Table Schema

### 1. Financial Tables (5 tables)

#### wallet_addresses
```sql
CREATE TABLE wallet_addresses (
    address VARCHAR(255) PRIMARY KEY,
    wallet_fingerprint VARCHAR(64) NOT NULL,
    derivation_path VARCHAR(100),
    account_index INT,
    change_index INT,
    address_index INT,
    public_key VARCHAR(255) NOT NULL,
    address_type VARCHAR(50) DEFAULT 'receiving',
    is_watching_only BOOLEAN DEFAULT FALSE,
    is_cold_storage BOOLEAN DEFAULT FALSE,
    balance NUMERIC(30, 0) DEFAULT 0,
    balance_updated_at TIMESTAMP WITH TIME ZONE,
    balance_at_height BIGINT,
    first_used_at TIMESTAMP WITH TIME ZONE,
    last_used_at TIMESTAMP WITH TIME ZONE,
    transaction_count INT DEFAULT 0,
    label VARCHAR(255),
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(wallet_fingerprint, derivation_path)
);
```
**RLS Policies:** 7 (wallet_owner_select, wallet_owner_update, wallet_miner_type, wallet_oracle_type, wallet_treasury_type, wallet_admin_all, wallet_readonly_select)

#### address_balance_history
```sql
CREATE TABLE address_balance_history (
    id BIGSERIAL PRIMARY KEY,
    address VARCHAR(255) NOT NULL,
    block_height BIGINT NOT NULL,
    block_hash VARCHAR(255),
    balance NUMERIC(30, 0) NOT NULL,
    delta NUMERIC(30, 0),
    snapshot_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(address, block_height)
);
```
**RLS Policies:** 5 (balance_history_owner, balance_history_admin, balance_history_miner, balance_history_oracle, balance_history_readonly)
**Triggers:** trg_balance_history (AFTER UPDATE OF balance ON wallet_addresses)

#### address_transactions
```sql
CREATE TABLE address_transactions (
    id BIGSERIAL PRIMARY KEY,
    address VARCHAR(255) NOT NULL,
    tx_hash VARCHAR(255) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    from_address VARCHAR(255),
    to_address VARCHAR(255),
    amount NUMERIC(30, 0),
    block_height BIGINT,
    block_hash VARCHAR(255),
    block_timestamp BIGINT,
    tx_status VARCHAR(50) DEFAULT 'pending',
    notes TEXT,
    label VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(address, tx_hash)
);
```
**RLS Policies:** 4 (addr_tx_owner, addr_tx_admin, addr_tx_oracle_all, addr_tx_readonly)

#### address_utxos
```sql
CREATE TABLE address_utxos (
    utxo_id BIGSERIAL PRIMARY KEY,
    address VARCHAR(255) NOT NULL,
    tx_hash VARCHAR(255) NOT NULL,
    output_index INT NOT NULL,
    amount NUMERIC(30, 0) NOT NULL,
    spent BOOLEAN DEFAULT FALSE,
    spent_at_height BIGINT,
    spent_in_tx_hash VARCHAR(255),
    created_at_height BIGINT,
    created_at_timestamp BIGINT
);
```
**RLS Policies:** 3 (utxo_owner, utxo_admin, utxo_unspent_public)

#### address_labels
```sql
CREATE TABLE address_labels (
    label_id BIGSERIAL PRIMARY KEY,
    address VARCHAR(255) NOT NULL,
    label VARCHAR(255) NOT NULL,
    description TEXT,
    label_type VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 3 (addrlabel_owner, addrlabel_public_read, addrlabel_admin_all)

---

### 2. Blockchain Tables (6 tables)

#### blocks
```sql
CREATE TABLE blocks (
    height                     BIGINT PRIMARY KEY,
    block_hash                 VARCHAR(255) UNIQUE NOT NULL,
    parent_hash                VARCHAR(255) NOT NULL,
    merkle_root                VARCHAR(255),
    timestamp                  BIGINT NOT NULL,
    tx_count                   INT DEFAULT 0,
    coherence_snapshot         NUMERIC(5,4) DEFAULT 1.0,
    fidelity_snapshot          NUMERIC(5,4) DEFAULT 1.0,
    w_state_hash               VARCHAR(255),
    hyp_witness                VARCHAR(255),
    miner_address              VARCHAR(255),
    difficulty                 INT DEFAULT 6,
    nonce                      BIGINT DEFAULT 0,
    pq_curr                    INTEGER DEFAULT 1,
    pq_last                    INTEGER DEFAULT 0,
    oracle_w_state_hash        VARCHAR(255),
    finalized                  BOOLEAN DEFAULT TRUE,
    finalized_at               BIGINT DEFAULT 0,
    created_at                 TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**Indexes:** idx_blocks_hash, idx_blocks_parent, idx_blocks_timestamp
**RLS Policies:** 6 (blocks_public_read, blocks_miner_insert, blocks_miner_update, blocks_oracle_finalize, blocks_oracle_update, blocks_admin_all)
**Triggers:** trg_blocks_reward, trg_sync_peers, trg_sync_oracles, trg_audit_blocks

#### block_headers_cache
```sql
CREATE TABLE block_headers_cache (
    height BIGINT PRIMARY KEY,
    block_hash VARCHAR(255) UNIQUE NOT NULL,
    previous_hash VARCHAR(255) NOT NULL,
    state_root VARCHAR(255),
    transactions_root VARCHAR(255),
    timestamp BIGINT NOT NULL,
    difficulty NUMERIC(20, 10),
    nonce VARCHAR(255),
    quantum_proof VARCHAR(255),
    quantum_state_hash VARCHAR(255),
    temporal_coherence NUMERIC(5, 4),
    pq_signature TEXT,
    pq_key_fingerprint VARCHAR(255),
    received_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 2 (blockcache_public_read, blockcache_admin_all)

#### chain_reorganizations
```sql
CREATE TABLE chain_reorganizations (
    reorg_id BIGSERIAL PRIMARY KEY,
    timestamp BIGINT NOT NULL,
    reorg_depth INT NOT NULL,
    old_head_height BIGINT,
    new_head_height BIGINT,
    old_head_hash VARCHAR(255),
    new_head_hash VARCHAR(255),
    reorg_point_hash VARCHAR(255),
    transactions_reverted INT,
    transactions_reinserted INT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 3 (reorg_public_read, reorg_oracle_insert, reorg_admin_all)

#### orphan_blocks
```sql
CREATE TABLE orphan_blocks (
    block_hash VARCHAR(255) PRIMARY KEY,
    parent_hash VARCHAR(255) NOT NULL,
    block_height BIGINT,
    timestamp BIGINT,
    block_data_compressed BYTEA,
    block_size_bytes INT,
    received_from_peer VARCHAR(255),
    received_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    resolution_status VARCHAR(50) DEFAULT 'awaiting_parent',
    resolution_attempt_count INT DEFAULT 0
);
```
**RLS Policies:** 2 (orphan_public_read, orphan_oracle_insert)

#### state_root_updates
```sql
CREATE TABLE state_root_updates (
    update_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    new_state_root VARCHAR(255) NOT NULL,
    previous_state_root VARCHAR(255),
    timestamp BIGINT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(block_height)
);
```
**RLS Policies:** 2 (stateroot_public_read, stateroot_oracle_insert)

#### finality_records
```sql
CREATE TABLE finality_records (
    finality_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL UNIQUE,
    block_hash VARCHAR(255) NOT NULL,
    finalized BOOLEAN DEFAULT FALSE,
    finalized_at TIMESTAMP WITH TIME ZONE,
    finality_epoch BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 2 (finality_public_read, finality_oracle_insert)

---

### 3. Transaction Tables (4 tables)

#### transactions
```sql
CREATE TABLE transactions (
    id BIGSERIAL PRIMARY KEY,
    tx_hash VARCHAR(255) UNIQUE NOT NULL,
    from_address VARCHAR(255) NOT NULL,
    to_address VARCHAR(255) NOT NULL,
    amount NUMERIC(30, 0) NOT NULL,
    nonce BIGINT,
    height BIGINT,
    block_hash VARCHAR(255),
    transaction_index INT,
    tx_type VARCHAR(50) DEFAULT 'transfer',
    status VARCHAR(50) DEFAULT 'pending',
    pq_signature TEXT,
    pq_signer_key_fp VARCHAR(255),
    pq_verified BOOLEAN DEFAULT FALSE,
    pq_verified_at TIMESTAMP WITH TIME ZONE,
    quantum_state_hash VARCHAR(255),
    commitment_hash VARCHAR(255),
    entropy_score NUMERIC(5, 4),
    input_data TEXT,
    metadata JSONB,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    finalized_at TIMESTAMP WITH TIME ZONE
);
```
**RLS Policies:** 6 (tx_participant_select, tx_coinbase_miner, tx_oracle_all, tx_admin_all, tx_readonly, tx_miner_submit)
**Triggers:** trg_tx_validate, trg_audit_tx

#### transaction_inputs
```sql
CREATE TABLE transaction_inputs (
    input_id BIGSERIAL PRIMARY KEY,
    tx_id BIGINT NOT NULL,
    previous_tx_hash VARCHAR(255),
    previous_output_index INT,
    script_sig TEXT,
    script_pubkey TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 3 (txin_public_read, txin_miner_insert, txin_admin_all)

#### transaction_outputs
```sql
CREATE TABLE transaction_outputs (
    output_id BIGSERIAL PRIMARY KEY,
    tx_id BIGINT NOT NULL,
    output_index INT NOT NULL,
    address VARCHAR(255) NOT NULL,
    amount NUMERIC(30, 0) NOT NULL,
    script_pubkey TEXT,
    spent BOOLEAN DEFAULT FALSE,
    spent_in_tx_id BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(tx_id, output_index)
);
```
**RLS Policies:** 3 (txout_public_read, txout_miner_insert, txout_admin_all)

#### transaction_receipts
```sql
CREATE TABLE transaction_receipts (
    receipt_id BIGSERIAL PRIMARY KEY,
    tx_id BIGINT NOT NULL,
    height BIGINT,
    status INT,
    logs_json JSONB,
    bloom_filter TEXT,
    quantum_proof TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 3 (txreceipt_public_read, txreceipt_oracle_insert, txreceipt_admin_all)

---

### 4. Oracle Tables (9 tables)

#### oracle_registry
```sql
CREATE TABLE oracle_registry (
    oracle_id       VARCHAR(128)  PRIMARY KEY,
    oracle_url      VARCHAR(512)  NOT NULL DEFAULT '',
    oracle_address  VARCHAR(128)  NOT NULL DEFAULT '',
    is_primary      BOOLEAN       NOT NULL DEFAULT FALSE,
    last_seen       BIGINT        NOT NULL DEFAULT 0,
    block_height    BIGINT        NOT NULL DEFAULT 0,
    peer_count      INTEGER       NOT NULL DEFAULT 0,
    gossip_url      JSONB         NOT NULL DEFAULT '{}'::JSONB,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    wallet_address  VARCHAR(128)  NOT NULL DEFAULT '',
    oracle_pub_key  TEXT          NOT NULL DEFAULT '',
    cert_sig        VARCHAR(128)  NOT NULL DEFAULT '',
    cert_auth_tag   VARCHAR(128)  NOT NULL DEFAULT '',
    mode            VARCHAR(32)   NOT NULL DEFAULT 'full',
    ip_hint         VARCHAR(256)  NOT NULL DEFAULT '',
    reg_tx_hash     VARCHAR(64)   NOT NULL DEFAULT '',
    registered_at   BIGINT        NOT NULL DEFAULT 0
);
```
**Indexes:** idx_oracle_registry_last_seen, idx_oracle_registry_primary, idx_oracle_registry_wallet, idx_oracle_registry_reg_tx, idx_oracle_registry_registered_at
**RLS Policies:** 10 (oracle_self_select, oracle_self_update, oracle_primary_all, oracle_secondary_all, oracle_validation_select, oracle_public_read, oracle_quorum_read, oracle_wallet_link, oracle_admin_all, oracle_treasury_read)

#### oracle_coherence_metrics
```sql
CREATE TABLE oracle_coherence_metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    system_coherence_measure NUMERIC(5, 4),
    lattice_coherence_score NUMERIC(5, 4),
    tessellation_synchronization_quality NUMERIC(5, 4),
    pseudoqubit_coherence_array JSONB,
    min_coherence NUMERIC(5, 4),
    max_coherence NUMERIC(5, 4),
    avg_coherence NUMERIC(5, 4),
    phase_drift_radians NUMERIC(200, 150),
    phase_correction_applied BOOLEAN,
    validator_agreement_score NUMERIC(5, 4),
    network_partition_detected BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 5 (coherence_oracle_insert, coherence_oracle_select, coherence_miner_select, coherence_public, coherence_admin_all)

#### oracle_consensus_state
```sql
CREATE TABLE oracle_consensus_state (
    consensus_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    oracle_consensus_reached BOOLEAN DEFAULT FALSE,
    validator_agreement_count INT,
    total_validators INT,
    consensus_threshold NUMERIC(5, 4),
    w_state_hash_agreement BOOLEAN,
    density_matrix_hash_agreement BOOLEAN,
    entropy_hash_agreement BOOLEAN,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(block_height)
);
```
**RLS Policies:** 4 (consensus_public_read, consensus_oracle_insert, consensus_oracle_update, consensus_admin_all)

#### oracle_density_matrix_stream
```sql
CREATE TABLE oracle_density_matrix_stream (
    stream_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    density_matrix_json JSONB NOT NULL,
    density_matrix_hash VARCHAR(255) UNIQUE NOT NULL,
    trace_value NUMERIC(5, 4),
    purity NUMERIC(5, 4),
    von_neumann_entropy NUMERIC(5, 4),
    eigenvalues JSONB,
    live_metrics_json JSONB,
    sensor_timestamps JSONB,
    update_sequence_number BIGINT,
    time_since_last_update_ms NUMERIC(15, 2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 4 (dmstream_oracle_insert, dmstream_oracle_select, dmstream_miner_select, dmstream_admin_all)

#### oracle_distribution_log
```sql
CREATE TABLE oracle_distribution_log (
    log_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    peer_id VARCHAR(255) NOT NULL,
    data_type VARCHAR(50),
    data_hash VARCHAR(255),
    distribution_successful BOOLEAN DEFAULT TRUE,
    distribution_latency_ms NUMERIC(15, 2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 3 (distlog_oracle_insert, distlog_oracle_select, distlog_admin_all)

#### oracle_entanglement_records
```sql
CREATE TABLE oracle_entanglement_records (
    entanglement_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    peer_id VARCHAR(255) NOT NULL,
    peer_public_key VARCHAR(255),
    entanglement_type VARCHAR(50),
    entanglement_measure NUMERIC(5, 4),
    bell_parameter NUMERIC(5, 4),
    oracle_entanglement_measure NUMERIC(5, 4),
    entanglement_match_score NUMERIC(5, 4),
    in_sync_with_oracle BOOLEAN DEFAULT FALSE,
    verification_proof TEXT,
    verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 4 (entangle_oracle_insert, entangle_oracle_select, entangle_miner_select, entangle_admin_all)

#### oracle_entropy_feeds
```sql
CREATE TABLE oracle_entropy_feeds (
    feed_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    anu_qrng_entropy TEXT,
    random_org_entropy TEXT,
    qbick_entropy TEXT,
    outshift_entropy TEXT,
    hotbits_entropy TEXT,
    xor_combined_seed TEXT,
    entropy_hash VARCHAR(255) UNIQUE NOT NULL,
    min_entropy_estimate NUMERIC(5, 4),
    shannon_entropy_estimate NUMERIC(5, 4),
    source_agreement_score NUMERIC(5, 4),
    distributed_to_peers INT DEFAULT 0,
    distribution_complete BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 4 (entropy_oracle_insert, entropy_oracle_select, entropy_miner_select, entropy_admin_all)

#### oracle_pq0_state
```sql
CREATE TABLE oracle_pq0_state (
    state_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    oracle_pq_id BIGINT,
    oracle_position_x NUMERIC(200, 150),
    oracle_position_y NUMERIC(200, 150),
    pq_inverse_virtual_id BIGINT,
    pq_virtual_id BIGINT,
    quantum_state_json JSONB,
    phase_theta NUMERIC(200, 150),
    coherence_measure NUMERIC(5, 4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 4 (pq0_oracle_insert, pq0_oracle_select, pq0_miner_select, pq0_admin_all)

#### oracle_w_state_snapshots
```sql
CREATE TABLE oracle_w_state_snapshots (
    snapshot_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    block_hash VARCHAR(255) NOT NULL,
    timestamp BIGINT NOT NULL,
    w_state_serialized TEXT NOT NULL,
    w_state_hash VARCHAR(255) UNIQUE NOT NULL,
    entanglement_measure NUMERIC(5, 4),
    coherence_time_us NUMERIC(15, 2),
    fidelity_estimate NUMERIC(5, 4),
    quantum_proof_data TEXT,
    quantum_proof_hash VARCHAR(255),
    shannon_entropy NUMERIC(5, 4),
    entropy_source_quality JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(block_height, block_hash)
);
```
**RLS Policies:** 5 (wstate_oracle_insert, wstate_oracle_select, wstate_miner_select, wstate_public, wstate_admin_all)
**Triggers:** trg_w_state_consensus

---

### 5. Peer/Network Tables (6 tables)

#### peer_registry
```sql
CREATE TABLE peer_registry (
    peer_id VARCHAR(255) PRIMARY KEY,
    public_key VARCHAR(255) UNIQUE NOT NULL,
    ip_address VARCHAR(45),
    port INTEGER,
    peer_type VARCHAR(50) DEFAULT 'full',
    capabilities TEXT[],
    block_height BIGINT DEFAULT 0,
    chain_head_hash VARCHAR(255),
    network_version VARCHAR(50),
    reputation_score NUMERIC(10, 4) DEFAULT 1.0,
    blocks_validated INT DEFAULT 0,
    blocks_rejected INT DEFAULT 0,
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_handshake TIMESTAMP WITH TIME ZONE,
    connection_attempts INT DEFAULT 0,
    failed_attempts INT DEFAULT 0,
    oracle_entanglement_ready BOOLEAN DEFAULT FALSE,
    oracle_density_matrix_version BIGINT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 6 (peer_public_read, peer_self_update, peer_self_insert, peer_admin_all, peer_oracle_manage, peer_readonly)

#### peer_connections
```sql
CREATE TABLE peer_connections (
    connection_id BIGSERIAL PRIMARY KEY,
    peer_id VARCHAR(255) NOT NULL,
    connection_state VARCHAR(50) DEFAULT 'disconnected',
    established_at TIMESTAMP WITH TIME ZONE,
    disconnected_at TIMESTAMP WITH TIME ZONE,
    latency_ms NUMERIC(10, 2),
    bandwidth_in_kbps NUMERIC(15, 2),
    bandwidth_out_kbps NUMERIC(15, 2),
    packet_loss_rate NUMERIC(5, 4),
    blocks_sync_height BIGINT,
    last_message_at TIMESTAMP WITH TIME ZONE,
    messages_sent INT DEFAULT 0,
    messages_received INT DEFAULT 0,
    bytes_sent BIGINT DEFAULT 0,
    bytes_received BIGINT DEFAULT 0,
    oracle_state_shared BOOLEAN DEFAULT FALSE,
    density_matrix_shared BOOLEAN DEFAULT FALSE,
    entropy_shared BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 4 (conn_public_read, conn_oracle_insert, conn_oracle_update, conn_admin_all)

#### peer_reputation
```sql
CREATE TABLE peer_reputation (
    reputation_id BIGSERIAL PRIMARY KEY,
    peer_id VARCHAR(255) NOT NULL,
    timestamp BIGINT NOT NULL,
    score NUMERIC(10, 4) NOT NULL,
    factors JSONB,
    event_type VARCHAR(50),
    event_description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 3 (rep_public_read, rep_oracle_update, rep_admin_all)

#### network_events
```sql
CREATE TABLE network_events (
    event_id BIGSERIAL PRIMARY KEY,
    timestamp BIGINT NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    event_description TEXT,
    affected_peers INT,
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 3 (netevent_public_read, netevent_oracle_insert, netevent_admin_all)

#### network_partition_events
```sql
CREATE TABLE network_partition_events (
    event_id BIGSERIAL PRIMARY KEY,
    timestamp BIGINT NOT NULL,
    partition_detected BOOLEAN DEFAULT TRUE,
    peers_in_partition_1 INT,
    peers_in_partition_2 INT,
    partition_healed BOOLEAN DEFAULT FALSE,
    healing_timestamp BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 3 (part_public_read, part_oracle_insert, part_admin_all)

#### network_bandwidth_usage
```sql
CREATE TABLE network_bandwidth_usage (
    usage_id BIGSERIAL PRIMARY KEY,
    peer_id VARCHAR(255),
    timestamp BIGINT NOT NULL,
    bandwidth_in_kbps NUMERIC(15, 2),
    bandwidth_out_kbps NUMERIC(15, 2),
    total_bandwidth_kbps NUMERIC(15, 2),
    bytes_in INT,
    bytes_out INT,
    congestion_level NUMERIC(5, 4),
    packet_loss_rate NUMERIC(5, 4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 3 (bw_public_read, bw_oracle_insert, bw_admin_all)

---

### 6. Quantum Tables (15 tables)

#### pseudoqubits
```sql
CREATE TABLE pseudoqubits (
    pq_id BIGINT PRIMARY KEY,
    triangle_id BIGINT NOT NULL,
    x NUMERIC(200, 150) NOT NULL,
    y NUMERIC(200, 150) NOT NULL,
    placement_type VARCHAR(50) NOT NULL,
    phase_theta NUMERIC(200, 150) DEFAULT 0,
    coherence_measure NUMERIC(5, 4) DEFAULT 0.99,
    coherence_time_us NUMERIC(15, 2) DEFAULT 100000,
    entanglement_with_oracle NUMERIC(5, 4) DEFAULT 0,
    entanglement_measure_neighbors JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_measured_at TIMESTAMP WITH TIME ZONE
);
```
**RLS Policies:** 3 (pq_public_read, pq_oracle_insert, pq_admin_all)

#### hyperbolic_triangles
```sql
CREATE TABLE hyperbolic_triangles (
    triangle_id BIGINT PRIMARY KEY,
    depth INT NOT NULL,
    parent_id BIGINT,
    v0_x NUMERIC(250, 210) NOT NULL,
    v0_y NUMERIC(250, 210) NOT NULL,
    v0_name TEXT,
    v1_x NUMERIC(250, 210) NOT NULL,
    v1_y NUMERIC(250, 210) NOT NULL,
    v1_name TEXT,
    v2_x NUMERIC(250, 210) NOT NULL,
    v2_y NUMERIC(250, 210) NOT NULL,
    v2_name TEXT,
    area NUMERIC(250, 210),
    perimeter NUMERIC(250, 210),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 3 (tri_public_read, tri_oracle_insert, tri_admin_all)

#### quantum_coherence_snapshots
```sql
CREATE TABLE quantum_coherence_snapshots (
    snapshot_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    global_coherence NUMERIC(5, 4) NOT NULL,
    average_coherence NUMERIC(5, 4),
    min_coherence NUMERIC(5, 4),
    max_coherence NUMERIC(5, 4),
    coherence_histogram JSONB,
    phase_drift_radians NUMERIC(200, 150),
    phase_correction_applied BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 3 (cohere_public_read, cohere_oracle_insert, cohere_admin_all)

#### quantum_density_matrix_global
```sql
CREATE TABLE quantum_density_matrix_global (
    state_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    density_matrix_json JSONB NOT NULL,
    density_matrix_hash VARCHAR(255) UNIQUE NOT NULL,
    trace_value NUMERIC(5, 4),
    purity NUMERIC(5, 4),
    von_neumann_entropy NUMERIC(5, 4),
    eigenvalues JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 3 (dm_global_public_read, dm_global_oracle_insert, dm_global_admin_all)

#### quantum_circuit_execution
```sql
CREATE TABLE quantum_circuit_execution (
    circuit_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    circuit_depth INT,
    circuit_size INT,
    num_qubits INT,
    num_gates INT,
    execution_successful BOOLEAN DEFAULT TRUE,
    execution_time_ms NUMERIC(15, 2),
    ghz_fidelity NUMERIC(5, 4),
    w_state_fidelity NUMERIC(5, 4),
    circuit_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 3 (circuit_public_read, circuit_oracle_insert, circuit_admin_all)

#### quantum_measurements
```sql
CREATE TABLE quantum_measurements (
    measurement_id BIGSERIAL PRIMARY KEY,
    pq_id BIGINT NOT NULL,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    outcome INT CHECK (outcome IN (0, 1)),
    basis VARCHAR(10),
    expectation_value NUMERIC(5, 4),
    variance NUMERIC(5, 4),
    post_measurement_state JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 3 (measure_public_read, measure_oracle_insert, measure_admin_all)

#### w_state_snapshots
```sql
CREATE TABLE w_state_snapshots (
    snapshot_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    w_state_serialized TEXT NOT NULL,
    w_state_hash VARCHAR(255) UNIQUE NOT NULL,
    entanglement_measure NUMERIC(5, 4),
    coherence_time_us NUMERIC(15, 2),
    fidelity_estimate NUMERIC(5, 4),
    pq_addresses TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(block_height)
);
```
**RLS Policies:** 3 (wstate_snap_public_read, wstate_snap_oracle_insert, wstate_snap_admin_all)

#### w_state_validator_states
```sql
CREATE TABLE w_state_validator_states (
    state_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    validator_public_key VARCHAR(255) NOT NULL,
    timestamp BIGINT NOT NULL,
    w_state_serialized TEXT,
    w_state_hash VARCHAR(255),
    coherence_with_oracle NUMERIC(5, 4),
    phase_alignment_radians NUMERIC(200, 150),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 3 (wstate_val_public_read, wstate_val_oracle_insert, wstate_val_admin_all)

#### entanglement_records
```sql
CREATE TABLE entanglement_records (
    entanglement_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    peer_id VARCHAR(255) NOT NULL,
    peer_public_key VARCHAR(255),
    entanglement_type VARCHAR(50),
    entanglement_measure NUMERIC(5, 4),
    bell_parameter NUMERIC(5, 4),
    oracle_entanglement_measure NUMERIC(5, 4),
    entanglement_match_score NUMERIC(5, 4),
    in_sync_with_oracle BOOLEAN DEFAULT FALSE,
    local_w_state_hash VARCHAR(255),
    local_density_matrix_hash VARCHAR(255),
    verification_proof TEXT,
    verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 3 (entangle_rec_public_read, entangle_rec_oracle_insert, entangle_rec_admin_all)

#### quantum_error_correction
```sql
CREATE TABLE quantum_error_correction (
    correction_id BIGSERIAL PRIMARY KEY,
    pq_id BIGINT NOT NULL,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    error_detected BOOLEAN NOT NULL,
    error_type VARCHAR(50),
    error_location_code VARCHAR(255),
    correction_applied BOOLEAN DEFAULT FALSE,
    correction_method VARCHAR(50),
    correction_strength NUMERIC(5, 4),
    correction_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 3 (qec_public_read, qec_oracle_insert, qec_admin_all)

#### quantum_lattice_metadata
```sql
CREATE TABLE quantum_lattice_metadata (
    metadata_id BIGSERIAL PRIMARY KEY,
    tessellation_depth INT NOT NULL,
    total_triangles BIGINT NOT NULL,
    total_pseudoqubits BIGINT NOT NULL,
    precision_bits INT DEFAULT 150,
    hyperbolicity_constant NUMERIC(5, 4) DEFAULT -1.0,
    poincare_radius NUMERIC(5, 4) DEFAULT 1.0,
    status VARCHAR(50) DEFAULT 'constructing',
    construction_started_at TIMESTAMP WITH TIME ZONE,
    construction_completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 3 (lattice_meta_public_read, lattice_meta_oracle_insert, lattice_meta_admin_all)

#### quantum_phase_evolution
```sql
CREATE TABLE quantum_phase_evolution (
    phase_id BIGSERIAL PRIMARY KEY,
    pq_id BIGINT NOT NULL,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    phase_theta NUMERIC(200, 150) NOT NULL,
    phase_derivative NUMERIC(200, 150),
    coherence_measure NUMERIC(5, 4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 3 (phase_public_read, phase_oracle_insert, phase_admin_all)

#### quantum_shadow_tomography
```sql
CREATE TABLE quantum_shadow_tomography (
    shadow_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    shadow_snapshots JSONB NOT NULL,
    shadow_measurement_bases JSONB,
    reconstruction_fidelity NUMERIC(5, 4),
    num_snapshots INT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 3 (shadow_public_read, shadow_oracle_insert, shadow_admin_all)

#### quantum_supremacy_proofs
```sql
CREATE TABLE quantum_supremacy_proofs (
    proof_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    circuit_depth INT,
    success_probability NUMERIC(5, 4),
    classical_simulation_complexity VARCHAR(255),
    quantum_result_hash VARCHAR(255),
    classical_hardness_assumption TEXT,
    verified BOOLEAN DEFAULT FALSE,
    verification_method VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 3 (supremacy_public_read, supremacy_oracle_insert, supremacy_admin_all)

#### pq_sequential
```sql
CREATE TABLE pq_sequential (
    pq_id BIGINT PRIMARY KEY,
    next_pq_id BIGINT,
    prev_pq_id BIGINT,
    sequence_order BIGINT UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**Indexes:** idx_pq_next, idx_pq_prev, idx_sequence_order
**RLS Policies:** 3 (pqseq_public_read, pqseq_oracle_insert, pqseq_admin_all)

---

### 7. Client Sync Tables (4 tables)

#### client_block_sync
```sql
CREATE TABLE client_block_sync (
    sync_id BIGSERIAL PRIMARY KEY,
    peer_id VARCHAR(255) NOT NULL,
    blocks_downloaded INT,
    blocks_requested INT,
    blocks_total INT,
    sync_started_at TIMESTAMP WITH TIME ZONE,
    sync_completed_at TIMESTAMP WITH TIME ZONE,
    sync_status VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 2 (blocksync_own, blocksync_admin)

#### client_oracle_sync
```sql
CREATE TABLE client_oracle_sync (
    sync_id BIGSERIAL PRIMARY KEY,
    peer_id VARCHAR(255) NOT NULL UNIQUE,
    block_height_local BIGINT,
    block_height_oracle BIGINT,
    w_state_hash_local VARCHAR(255),
    w_state_hash_oracle VARCHAR(255),
    density_matrix_hash_local VARCHAR(255),
    density_matrix_hash_oracle VARCHAR(255),
    density_matrix_sync_status VARCHAR(50) DEFAULT 'pending',
    entropy_hash_local VARCHAR(255),
    entropy_hash_oracle VARCHAR(255),
    coherence_measure_local NUMERIC(5, 4),
    coherence_measure_oracle NUMERIC(5, 4),
    coherence_aligned BOOLEAN DEFAULT FALSE,
    lattice_sync_quality NUMERIC(5, 4),
    tessellation_in_sync BOOLEAN DEFAULT FALSE,
    last_lattice_update TIMESTAMP WITH TIME ZONE,
    sync_status VARCHAR(50) DEFAULT 'initializing',
    sync_confidence NUMERIC(5, 4) DEFAULT 0.0,
    last_sync_attempt TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_successful_sync TIMESTAMP WITH TIME ZONE,
    sync_error_message TEXT,
    sync_attempt_count INT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 2 (oraclesync_own, oraclesync_admin)

#### client_network_metrics
```sql
CREATE TABLE client_network_metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    peer_id VARCHAR(255) NOT NULL,
    timestamp BIGINT NOT NULL,
    latency_ms NUMERIC(10, 2),
    bandwidth_in_kbps NUMERIC(15, 2),
    bandwidth_out_kbps NUMERIC(15, 2),
    packet_loss_rate NUMERIC(5, 4),
    blocks_per_second NUMERIC(10, 2),
    avg_sync_time_ms NUMERIC(15, 2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 2 (netmetrics_own, netmetrics_admin)

#### client_sync_events
```sql
CREATE TABLE client_sync_events (
    event_id BIGSERIAL PRIMARY KEY,
    peer_id VARCHAR(255) NOT NULL,
    timestamp BIGINT NOT NULL,
    event_type VARCHAR(50),
    event_description TEXT,
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 2 (syncevents_own, syncevents_admin)

---

### 8. Security Tables (7 tables)

#### wallet_encrypted_seeds
```sql
CREATE TABLE wallet_encrypted_seeds (
    seed_id              BIGSERIAL PRIMARY KEY,
    wallet_fingerprint   VARCHAR(64)   NOT NULL UNIQUE,
    kdf_type             VARCHAR(16)   NOT NULL DEFAULT 'argon2id',
    kdf_salt_b64         TEXT          NOT NULL,
    argon2_m_cost        INTEGER       DEFAULT 65536,
    argon2_t_cost        INTEGER       DEFAULT 3,
    argon2_p_cost        INTEGER       DEFAULT 4,
    scrypt_n             INTEGER,
    scrypt_r             INTEGER,
    scrypt_p             INTEGER,
    dek_nonce_b64        TEXT          NOT NULL,
    wrapped_dek_b64      TEXT          NOT NULL,
    seed_nonce_b64       TEXT          NOT NULL,
    seed_ciphertext_b64  TEXT          NOT NULL,
    bip32_xpub           TEXT,
    derivation_scheme    VARCHAR(32)   DEFAULT 'BIP44',
    coin_type            INTEGER       DEFAULT 60,
    mnemonic_word_count  SMALLINT      DEFAULT 24,
    is_passphrase_protected BOOLEAN    DEFAULT FALSE,
    device_bound         BOOLEAN       DEFAULT FALSE,
    last_decrypted_at    TIMESTAMP WITH TIME ZONE,
    decrypt_count        INTEGER       DEFAULT 0,
    created_at           TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at           TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 3 (seeds_owner_only, seeds_admin_emergency, seeds_no_delete)

#### encrypted_private_keys
```sql
CREATE TABLE encrypted_private_keys (
    key_id BIGSERIAL PRIMARY KEY,
    address VARCHAR(255) NOT NULL UNIQUE,
    algorithm VARCHAR(50) DEFAULT 'AES-256-GCM',
    kdf_algorithm VARCHAR(50) DEFAULT 'PBKDF2-SHA3-512',
    kdf_iterations INT DEFAULT 16384,
    nonce_hex VARCHAR(255) NOT NULL,
    salt_hex VARCHAR(255) NOT NULL,
    ciphertext_hex TEXT NOT NULL,
    auth_tag_hex VARCHAR(255),
    key_fingerprint VARCHAR(255),
    derivation_path VARCHAR(100),
    is_locked BOOLEAN DEFAULT FALSE,
    lock_reason TEXT,
    last_used_for_signing TIMESTAMP WITH TIME ZONE,
    signing_count INT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 3 (keys_owner_only, keys_admin_emergency, keys_no_delete)

#### key_audit_log
```sql
CREATE TABLE key_audit_log (
    audit_id             BIGSERIAL PRIMARY KEY,
    event_type           VARCHAR(64)  NOT NULL,
    wallet_fingerprint   VARCHAR(64),
    address              VARCHAR(255),
    kms_key_id           BIGINT,
    actor_peer_id        VARCHAR(255),
    tx_hash              VARCHAR(255),
    block_height         BIGINT,
    success              BOOLEAN      NOT NULL DEFAULT TRUE,
    failure_reason       TEXT,
    duration_ms          NUMERIC(10,2),
    created_at           TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**Indexes:** idx_key_audit_wallet, idx_key_audit_event, idx_key_audit_fail
**RLS Policies:** 4 (audit_user_own, audit_admin_all, audit_readonly, audit_no_modify)

#### nonce_ledger
```sql
CREATE TABLE nonce_ledger (
    nonce_id             BIGSERIAL PRIMARY KEY,
    nonce_hex            VARCHAR(128) NOT NULL UNIQUE,
    address              VARCHAR(255) NOT NULL,
    used_in_type         VARCHAR(50)  NOT NULL,
    used_in_hash         VARCHAR(255),
    created_at           TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at           TIMESTAMP WITH TIME ZONE
);
```
**Indexes:** idx_nonce_address, idx_nonce_expiry
**RLS Policies:** 3 (nonce_owner, nonce_admin_all, nonce_oracle_insert)

#### wallet_key_rotation_history
```sql
CREATE TABLE wallet_key_rotation_history (
    rotation_id BIGSERIAL PRIMARY KEY,
    wallet_fingerprint VARCHAR(64) NOT NULL,
    old_key_id VARCHAR(255),
    new_key_id VARCHAR(255),
    rotation_reason TEXT,
    rotation_timestamp TIMESTAMP WITH TIME ZONE,
    ratchet_material TEXT,
    next_rotation_material TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 3 (rotation_owner, rotation_admin_all, rotation_no_delete)

#### wallet_seed_backup_status
```sql
CREATE TABLE wallet_seed_backup_status (
    backup_id BIGSERIAL PRIMARY KEY,
    wallet_fingerprint VARCHAR(64) NOT NULL UNIQUE,
    seed_phrase_backed_up BOOLEAN DEFAULT FALSE,
    backup_confirmed_at TIMESTAMP WITH TIME ZONE,
    seed_hint VARCHAR(50),
    seed_hash VARCHAR(255),
    backup_required BOOLEAN DEFAULT TRUE,
    days_since_creation_without_backup INT,
    email_notifications_sent INT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 3 (backup_owner, backup_admin_all, backup_readonly)

#### audit_logs
```sql
CREATE TABLE audit_logs (
    log_id BIGSERIAL PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    actor_peer_id VARCHAR(255),
    action VARCHAR(255),
    resource_type VARCHAR(100),
    resource_id VARCHAR(255),
    changes JSONB,
    result VARCHAR(50),
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 3 (audit_logs_admin, audit_logs_readonly, audit_logs_no_modify)

---

### 9. System Tables (7 tables)

#### system_metrics
```sql
CREATE TABLE system_metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    timestamp BIGINT NOT NULL,
    db_size_mb NUMERIC(15, 2),
    active_connections INT,
    active_peers INT,
    total_peers INT,
    avg_latency_ms NUMERIC(10, 2),
    blocks_per_minute NUMERIC(10, 2),
    transactions_per_second NUMERIC(10, 2),
    avg_coherence NUMERIC(5, 4),
    oracle_sync_quality NUMERIC(5, 4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 2 (sysmetrics_public_read, sysmetrics_admin_all)

#### database_metadata
```sql
CREATE TABLE database_metadata (
    metadata_id BIGSERIAL PRIMARY KEY,
    schema_version VARCHAR(50),
    build_timestamp TIMESTAMP WITH TIME ZONE,
    build_info JSONB,
    tables_created INT,
    indexes_created INT,
    constraints_created INT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 2 (dbmeta_public_read, dbmeta_admin_all)

#### consensus_events
```sql
CREATE TABLE consensus_events (
    event_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT,
    timestamp BIGINT NOT NULL,
    event_type VARCHAR(100),
    event_description TEXT,
    severity VARCHAR(20),
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 2 (consevent_public_read, consevent_oracle_insert)

#### entropy_quality_log
```sql
CREATE TABLE entropy_quality_log (
    log_id BIGSERIAL PRIMARY KEY,
    timestamp BIGINT NOT NULL,
    anu_qrng_quality NUMERIC(5, 4),
    random_org_quality NUMERIC(5, 4),
    qbick_quality NUMERIC(5, 4),
    outshift_quality NUMERIC(5, 4),
    hotbits_quality NUMERIC(5, 4),
    ensemble_min_entropy NUMERIC(5, 4),
    ensemble_shannon_entropy NUMERIC(5, 4),
    passed_diehard BOOLEAN,
    passed_nist BOOLEAN,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 2 (entropylog_public_read, entropylog_oracle_insert)

#### lattice_sync_state
```sql
CREATE TABLE lattice_sync_state (
    state_id BIGSERIAL PRIMARY KEY,
    peer_id VARCHAR(255) NOT NULL,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    hyperbolic_coordinates_synced JSONB,
    poincare_disk_coverage NUMERIC(5, 4),
    vertex_synchronization_count INT,
    edge_synchronization_count INT,
    pseudoqubit_positions_hash VARCHAR(255),
    pseudoqubit_lattice_sync_quality NUMERIC(5, 4),
    lattice_coherence_measure NUMERIC(5, 4),
    critical_points_coherence JSONB,
    geodesic_paths_synchronized BOOLEAN DEFAULT FALSE,
    updates_since_last_sync INT DEFAULT 0,
    bytes_synchronized BIGINT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 2 (latticesync_public_read, latticesync_oracle_insert)

#### merkle_proofs
```sql
CREATE TABLE merkle_proofs (
    proof_id BIGSERIAL PRIMARY KEY,
    transaction_hash VARCHAR(255) NOT NULL,
    height BIGINT,
    block_hash VARCHAR(255) NOT NULL,
    proof_path TEXT NOT NULL,
    proof_index INT NOT NULL,
    verified BOOLEAN DEFAULT FALSE,
    verified_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(transaction_hash, block_hash)
);
```
**RLS Policies:** 2 (merkle_public_read, merkle_admin_all)

#### merkle_proofs (duplicate entry removed - already defined above)

---

### 10. Validator Tables (4 tables)

#### validators
```sql
CREATE TABLE validators (
    validator_id BIGSERIAL PRIMARY KEY,
    public_key VARCHAR(255) UNIQUE NOT NULL,
    peer_id VARCHAR(255),
    stake NUMERIC(30, 0) DEFAULT 0,
    commission_rate NUMERIC(5, 4),
    slashing_rate NUMERIC(5, 4) DEFAULT 0.0,
    blocks_proposed INT DEFAULT 0,
    blocks_validated INT DEFAULT 0,
    blocks_missed INT DEFAULT 0,
    status VARCHAR(50) DEFAULT 'active',
    is_slashed BOOLEAN DEFAULT FALSE,
    slashed_at TIMESTAMP WITH TIME ZONE,
    oracle_participation_score NUMERIC(5, 4) DEFAULT 0.0,
    w_state_sync_quality NUMERIC(5, 4) DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 3 (validators_public_read, validators_oracle_insert, validators_admin_all)

#### validator_stakes
```sql
CREATE TABLE validator_stakes (
    stake_id BIGSERIAL PRIMARY KEY,
    validator_id BIGINT NOT NULL,
    amount NUMERIC(30, 0) NOT NULL,
    staker_address VARCHAR(255),
    active BOOLEAN DEFAULT TRUE,
    delegated BOOLEAN DEFAULT FALSE,
    stake_at_height BIGINT,
    unstake_at_height BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 3 (stakes_owner, stakes_public_read, stakes_admin_all)

#### epochs
```sql
CREATE TABLE epochs (
    epoch_id BIGSERIAL PRIMARY KEY,
    epoch_number BIGINT UNIQUE NOT NULL,
    start_block_height BIGINT NOT NULL,
    end_block_height BIGINT,
    validator_count INT,
    total_stake NUMERIC(30, 0),
    finalized BOOLEAN DEFAULT FALSE,
    finalized_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
**RLS Policies:** 3 (epochs_public_read, epochs_oracle_insert, epochs_admin_all)

#### epoch_validators
```sql
CREATE TABLE epoch_validators (
    membership_id BIGSERIAL PRIMARY KEY,
    epoch_id BIGINT NOT NULL,
    validator_id BIGINT NOT NULL,
    stake NUMERIC(30, 0) NOT NULL,
    blocks_proposed INT DEFAULT 0,
    blocks_attested INT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(epoch_id, validator_id)
);
```
**RLS Policies:** 3 (epochval_public_read, epochval_oracle_insert, epochval_admin_all)

---

## RLS Policies Quick Reference

### Policy Categories Summary

| Category | Tables | Total Policies |
|----------|--------|----------------|
| Financial | 5 | 22 |
| Blockchain | 6 | 20 |
| Transaction | 4 | 12 |
| Oracle | 9 | 38 |
| Peer/Network | 6 | 22 |
| Quantum | 15 | 45 |
| Client Sync | 4 | 8 |
| Security | 7 | 23 |
| System | 7 | 14 |
| Validator | 4 | 12 |
| **TOTAL** | **69+** | **216+** |

### Common Policy Patterns

**Owner-Only Access:**
```sql
CREATE POLICY table_owner ON table_name
    FOR ALL TO qtcl_miner, qtcl_oracle, qtcl_treasury
    USING (column = current_setting('app.context_var', true));
```

**Public Read:**
```sql
CREATE POLICY table_public ON table_name
    FOR SELECT TO PUBLIC
    USING (true);
```

**Admin Full Access:**
```sql
CREATE POLICY table_admin ON table_name
    FOR ALL TO qtcl_admin
    USING (true);
```

**No Delete (Immutable):**
```sql
CREATE POLICY table_no_delete ON table_name
    FOR DELETE TO PUBLIC
    USING (false);
```

---

## Trigger Functions

### 1. fn_balance_history()
```sql
CREATE OR REPLACE FUNCTION fn_balance_history()
RETURNS TRIGGER AS $$
DECLARE
    _block_height BIGINT;
    _block_hash VARCHAR(255);
    _delta NUMERIC(30,0);
BEGIN
    SELECT COALESCE(MAX(height), 0) INTO _block_height FROM blocks;
    SELECT block_hash INTO _block_hash FROM blocks WHERE height = _block_height;
    _delta := NEW.balance - COALESCE(OLD.balance, 0);
    
    IF _delta != 0 OR OLD IS NULL THEN
        INSERT INTO address_balance_history (
            address, block_height, block_hash, balance, delta, snapshot_timestamp
        ) VALUES (
            NEW.address, _block_height, _block_hash, NEW.balance, _delta, NOW()
        );
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
```
**Purpose:** Records every balance change in address_balance_history

### 2. fn_distribute_block_rewards()
```sql
CREATE OR REPLACE FUNCTION fn_distribute_block_rewards()
RETURNS TRIGGER AS $$
DECLARE
    _miner_reward NUMERIC(30,0) := 5000000000;
    _treasury_reward NUMERIC(30,0) := 1000000000;
    _treasury_address VARCHAR(255) := 'qtcl1treasury0000000000000000000000000000';
BEGIN
    IF NEW.miner_address IS NOT NULL AND NEW.miner_address != '' THEN
        INSERT INTO wallet_addresses (address, balance, address_type, balance_at_height, ...)
        VALUES (NEW.miner_address, _miner_reward, 'miner', NEW.height, ...)
        ON CONFLICT (address) DO UPDATE
        SET balance = wallet_addresses.balance + _miner_reward,
            balance_at_height = NEW.height, updated_at = NOW();
    END IF;
    
    INSERT INTO wallet_addresses (address, balance, address_type, balance_at_height, ...)
    VALUES (_treasury_address, _treasury_reward, 'treasury', NEW.height, ...)
    ON CONFLICT (address) DO UPDATE
    SET balance = wallet_addresses.balance + _treasury_reward,
        balance_at_height = NEW.height, updated_at = NOW();
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
```
**Purpose:** Auto-credits miner and treasury on new block

### 3. fn_validate_transaction()
```sql
CREATE OR REPLACE FUNCTION fn_validate_transaction()
RETURNS TRIGGER AS $$
DECLARE
    _sender_balance NUMERIC(30,0);
    _total_cost NUMERIC(30,0);
BEGIN
    IF NEW.tx_type = 'coinbase' THEN
        RETURN NEW;
    END IF;
    
    SELECT COALESCE(balance, 0) INTO _sender_balance
    FROM wallet_addresses WHERE address = NEW.from_address;
    
    _total_cost := NEW.amount + COALESCE(NEW.fee, 0);
    
    IF _sender_balance < _total_cost THEN
        RAISE EXCEPTION 'Insufficient balance: sender % has % but needs %',
            NEW.from_address, _sender_balance, _total_cost;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
```
**Purpose:** Validates sender has sufficient balance before allowing transaction

### 4. fn_sync_peer_heights()
```sql
CREATE OR REPLACE FUNCTION fn_sync_peer_heights()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE peer_registry
    SET block_height = NEW.height,
        chain_head_hash = NEW.block_hash,
        updated_at = NOW()
    WHERE connection_state = 'connected';
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
```
**Purpose:** Updates all peers' block_height when new block is added

### 5. fn_sync_oracle_heights()
```sql
CREATE OR REPLACE FUNCTION fn_sync_oracle_heights()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE oracle_registry
    SET block_height = NEW.height,
        last_seen = EXTRACT(EPOCH FROM NOW())::BIGINT
    WHERE last_seen > 0;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
```
**Purpose:** Updates all oracles' block_height when new block is added

### 6. fn_check_w_state_consensus()
```sql
CREATE OR REPLACE FUNCTION fn_check_w_state_consensus()
RETURNS TRIGGER AS $$
DECLARE
    _snapshot_count INTEGER;
    _agreement_hash VARCHAR(255);
    _matching_count INTEGER;
BEGIN
    SELECT COUNT(*), mode() WITHIN GROUP (ORDER BY w_state_hash)
    INTO _snapshot_count, _agreement_hash
    FROM oracle_w_state_snapshots
    WHERE block_height = NEW.block_height;
    
    SELECT COUNT(*) INTO _matching_count
    FROM oracle_w_state_snapshots
    WHERE block_height = NEW.block_height AND w_state_hash = _agreement_hash;
    
    IF _snapshot_count >= 3 AND _matching_count >= 3 THEN
        INSERT INTO oracle_consensus_state (...)
        VALUES (NEW.block_height, ..., TRUE, _matching_count, _snapshot_count, ...)
        ON CONFLICT (block_height) DO UPDATE SET ...;
        
        UPDATE blocks SET finalized = TRUE, finalized_at = ...
        WHERE height = NEW.block_height AND finalized = FALSE;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
```
**Purpose:** Detects when enough oracles agree on W-state to finalize block

### 7. fn_audit_log()
```sql
CREATE OR REPLACE FUNCTION fn_audit_log()
RETURNS TRIGGER AS $$
DECLARE
    _event_type VARCHAR(100);
    _actor_peer_id VARCHAR(255);
    _action VARCHAR(255);
    _changes JSONB;
BEGIN
    _event_type := TG_TABLE_NAME || '_' || TG_OP;
    _actor_peer_id := current_setting('app.peer_id', true);
    _action := TG_OP;
    
    IF TG_OP = 'INSERT' THEN
        _changes := jsonb_build_object('new', row_to_json(NEW));
    ELSIF TG_OP = 'UPDATE' THEN
        _changes := jsonb_build_object('old', row_to_json(OLD), 'new', row_to_json(NEW));
    ELSIF TG_OP = 'DELETE' THEN
        _changes := jsonb_build_object('old', row_to_json(OLD));
    END IF;
    
    INSERT INTO audit_logs (event_type, actor_peer_id, action, ...)
    VALUES (_event_type, _actor_peer_id, _action, ...);
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;
```
**Purpose:** Comprehensive audit logging for all significant operations

---

## Triggers

### PostgreSQL Triggers (9 total)

| Trigger | Table | Event | Function |
|---------|-------|-------|----------|
| trg_balance_history | wallet_addresses | AFTER UPDATE OF balance | fn_balance_history() |
| trg_blocks_reward | blocks | AFTER INSERT | fn_distribute_block_rewards() |
| trg_tx_validate | transactions | BEFORE INSERT | fn_validate_transaction() |
| trg_sync_peers | blocks | AFTER INSERT | fn_sync_peer_heights() |
| trg_sync_oracles | blocks | AFTER INSERT | fn_sync_oracle_heights() |
| trg_w_state_consensus | oracle_w_state_snapshots | AFTER INSERT | fn_check_w_state_consensus() |
| trg_audit_wallet | wallet_addresses | AFTER INSERT/UPDATE/DELETE | fn_audit_log() |
| trg_audit_blocks | blocks | AFTER INSERT/UPDATE/DELETE | fn_audit_log() |
| trg_audit_tx | transactions | AFTER INSERT/UPDATE/DELETE | fn_audit_log() |

### SQLite Triggers (4 total)

| Trigger | Table | Event | Purpose |
|---------|-------|-------|---------|
| trg_balance_history | wallet_addresses | AFTER UPDATE OF balance | Record balance changes |
| trg_blocks_reward | blocks | AFTER INSERT | Credit miner rewards |
| trg_tx_validate | transactions | BEFORE INSERT | Validate sender balance |
| trg_audit_wallet | wallet_addresses | AFTER INSERT/UPDATE/DELETE | Audit logging |

---

## Roles and Permissions

### Role Definitions

| Role | Password | Purpose |
|------|----------|---------|
| qtcl_miner | `miner_password` (hardcoded) | Mining operations, block submission |
| qtcl_oracle | `RLS_PASSWORD` | Oracle consensus, validation |
| qtcl_treasury | `RLS_PASSWORD` | Treasury operations |
| qtcl_admin | `RLS_PASSWORD` | Full administrative access |
| qtcl_readonly | `RLS_PASSWORD` | Read-only access for monitoring |

### Permission Matrix

| Table | Miner | Oracle | Treasury | Admin | Readonly |
|-------|-------|--------|----------|-------|----------|
| blocks | INSERT/UPDATE | UPDATE | - | ALL | SELECT |
| transactions | INSERT/SELECT | ALL | - | ALL | SELECT |
| wallet_addresses | SELECT/UPDATE | SELECT/UPDATE | SELECT/UPDATE | ALL | SELECT |
| oracle_registry | SELECT | ALL | SELECT | ALL | SELECT |
| peer_registry | INSERT/UPDATE | ALL | - | ALL | SELECT |
| wallet_encrypted_seeds | - | - | - | SELECT | - |
| encrypted_private_keys | - | - | - | SELECT | - |
| audit_logs | - | - | - | ALL | SELECT |

### Dangerous Operations Revoked

```sql
-- DELETE revoked from non-admin roles on critical tables
REVOKE DELETE ON wallet_encrypted_seeds FROM qtcl_miner, qtcl_oracle, qtcl_treasury, qtcl_readonly;
REVOKE DELETE ON encrypted_private_keys FROM qtcl_miner, qtcl_oracle, qtcl_treasury, qtcl_readonly;
REVOKE DELETE ON key_audit_log FROM qtcl_miner, qtcl_oracle, qtcl_treasury, qtcl_readonly;
REVOKE DELETE ON audit_logs FROM qtcl_miner, qtcl_oracle, qtcl_treasury, qtcl_readonly;
REVOKE DELETE ON wallet_key_rotation_history FROM qtcl_miner, qtcl_oracle, qtcl_treasury, qtcl_readonly;
REVOKE DELETE ON address_balance_history FROM qtcl_miner, qtcl_oracle, qtcl_treasury, qtcl_readonly;

-- TRUNCATE revoked from all non-admin roles
REVOKE TRUNCATE ON ALL TABLES IN SCHEMA public FROM qtcl_miner, qtcl_oracle, qtcl_treasury, qtcl_readonly;
```

---

## CLI Commands

### Command Reference

```bash
# Full comprehensive setup (RECOMMENDED)
python qtcl_db_builder.py --comprehensive

# Security-only setup
python qtcl_db_builder.py --security-setup

# Apply only RLS policies
python qtcl_db_builder.py --apply-rls

# Create database roles only
python qtcl_db_builder.py --create-roles [--password <password>]

# Apply triggers only
python qtcl_db_builder.py --apply-triggers

# Run security audit
python qtcl_db_builder.py --security-audit

# Show database status
python qtcl_db_builder.py --status

# Sync client database from master
python qtcl_db_builder.py --sync-from-master

# Rebuild database (DESTRUCTIVE)
python qtcl_db_builder.py --rebuild --force

# Set tessellation depth
python qtcl_db_builder.py --comprehensive --tessellation-depth 6
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | (none - SQLite mode) |
| `RLS_PASSWORD` | Master password for roles | (none - fetched from server) |
| `KOYEB` | Force Koyeb mode | `false` |
| `KOYEB_APP_NAME` | Koyeb app name | (auto-detected) |
| `KOYEB_SERVICE_NAME` | Koyeb service name | (auto-detected) |
| `KOYEB_REGION` | Koyeb region | (auto-detected) |
| `FORCE_KOYEB_MODE` | Force Koyeb mode for testing | `false` |
| `ENTROPY_SERVER` | Server URL for client sync | `https://qtcl-blockchain.koyeb.app` |

---

## Python Classes and Methods

### QTCLSecurityManager

```python
class QTCLSecurityManager:
    def __init__(self, db_url: str = _DB_URL, db_mode: str = _DB_MODE)
    def connect() -> None
    def close() -> None
    def apply_rls_policies(category: str = 'all') -> None
    def enable_rls_on_all_tables() -> None
    def create_roles(password: str = None) -> None
    def grant_permissions() -> None
    def revoke_dangerous_permissions() -> None
    def create_trigger_functions() -> None
    def apply_triggers() -> None
    def comprehensive_security_setup(password: str = None) -> None
    def security_audit() -> Dict[str, Any]
```

### QTCLDatabaseSync

```python
class QTCLDatabaseSync:
    def __init__(self, master_url: str = None, local_db_path: str = None)
    def sync_from_master(start_height: int = 0) -> Dict[str, Any]
    def verify_sync_integrity() -> bool
```

### QuantumTemporalCoherenceLedgerServer

```python
class QuantumTemporalCoherenceLedgerServer:
    def __init__(self, db_url: str = _DB_URL, db_mode: str = _DB_MODE, tessellation_depth: int = 5)
    def connect() -> None
    def drop_all_tables() -> None
    def create_schema() -> None
    def populate_tessellation() -> None
    def rebuild_complete() -> None
    def close() -> None
```

---

## Security Audit

### Running an Audit

```bash
python qtcl_db_builder.py --security-audit
```

### Expected Output (Koyeb Mode)

```json
{
  "mode": "postgres",
  "timestamp": "2026-04-17T11:30:00+00:00",
  "rls_enabled_tables": 69,
  "total_policies": 216,
  "roles": [
    "qtcl_miner",
    "qtcl_oracle", 
    "qtcl_treasury",
    "qtcl_admin",
    "qtcl_readonly"
  ],
  "trigger_count": 9
}
```

### Expected Output (Client Mode)

```json
{
  "mode": "sqlite",
  "timestamp": "2026-04-17T11:30:00+00:00",
  "tables": 69,
  "triggers": [
    "trg_balance_history",
    "trg_blocks_reward",
    "trg_tx_validate",
    "trg_audit_wallet"
  ],
  "file_mode": "600",
  "file_owner": 1000
}
```

### Manual Verification SQL

```sql
-- Check RLS enabled tables
SELECT COUNT(*) FROM pg_tables 
WHERE schemaname = 'public' AND rowsecurity = TRUE;

-- Check RLS policies
SELECT schemaname, tablename, policyname, roles, cmd 
FROM pg_policies 
WHERE schemaname = 'public'
ORDER BY tablename, policyname;

-- Check roles
SELECT rolname, rolpassword IS NOT NULL as has_password 
FROM pg_roles 
WHERE rolname LIKE 'qtcl_%';

-- Check triggers
SELECT tgname, tgrelid::regclass AS table_name 
FROM pg_trigger 
WHERE tgname LIKE 'trg_%' 
AND tgname NOT LIKE 'RI_%';

-- Check trigger functions
SELECT proname 
FROM pg_proc 
WHERE proname LIKE 'fn_%';
```

---

## Troubleshooting

### Issue: RLS_PASSWORD not found
**Solution:**
```bash
# Option 1: Set environment variable
export RLS_PASSWORD="your_secure_password"

# Option 2: Client mode - password will be fetched from server
# Ensure ENTROPY_SERVER is set correctly
export ENTROPY_SERVER="https://qtcl-blockchain.koyeb.app"
```

### Issue: Cannot create roles (permission denied)
**Solution:**
```bash
# Must run as PostgreSQL superuser or user with CREATEROLE privilege
psql -U postgres -c "GRANT CREATEROLE TO your_user;"
```

### Issue: Triggers not firing
**Solution:**
```sql
-- Check if triggers are enabled
SELECT tgname, tgenabled 
FROM pg_trigger 
WHERE tgname LIKE 'trg_%';

-- Enable if disabled
ALTER TABLE table_name ENABLE TRIGGER trigger_name;
```

### Issue: RLS policies blocking legitimate queries
**Solution:**
```sql
-- Set application context
SET LOCAL app.current_address = 'your_address';
SET LOCAL app.peer_id = 'your_peer_id';
SET LOCAL app.wallet_fingerprint = 'your_fingerprint';

-- Or disable RLS for admin (use with caution)
ALTER TABLE table_name DISABLE ROW LEVEL SECURITY;
```

### Issue: SQLite file permissions too open
**Solution:**
```bash
# Fix file permissions
chmod 600 /path/to/qtcl.db
chmod 600 /path/to/.rls_password
```

### Issue: Mode detection failing
**Solution:**
```bash
# Force Koyeb mode
export FORCE_KOYEB_MODE=true

# Or explicitly set DATABASE_URL
export DATABASE_URL="postgresql://user:pass@host/db?sslmode=require"
```

---

## Quick Reference Card

### Essential Commands

```bash
# 1. Full setup on Koyeb
export RLS_PASSWORD="secure_password"
export DATABASE_URL="postgresql://..."
python qtcl_db_builder.py --comprehensive

# 2. Client setup (SQLite)
python qtcl_db_builder.py --comprehensive

# 3. Security audit
python qtcl_db_builder.py --security-audit

# 4. Apply only triggers
python qtcl_db_builder.py --apply-triggers

# 5. Create roles with custom password
python qtcl_db_builder.py --create-roles --password "custom_password"
```

### Key Files

| File | Purpose |
|------|---------|
| `qtcl_db_builder.py` | Main database builder (4,586 lines) |
| `qtcl.db` | SQLite database (client mode) |
| `.rls_password` | Cached RLS_PASSWORD (0o600) |
| `IMPLEMENTATION_SUMMARY.md` | Implementation overview |

### Password Policy

```
┌─────────────────┬─────────────────────────────────────────┐
│ Role            │ Password Source                         │
├─────────────────┼─────────────────────────────────────────┤
│ qtcl_miner      │ Hardcoded: 'miner_password'             │
│ qtcl_oracle     │ RLS_PASSWORD env / fetch from server    │
│ qtcl_treasury   │ RLS_PASSWORD env / fetch from server    │
│ qtcl_admin      │ RLS_PASSWORD env / fetch from server    │
│ qtcl_readonly   │ RLS_PASSWORD env / fetch from server    │
└─────────────────┴─────────────────────────────────────────┘
```

### Mode Comparison

```
┌──────────────────────┬────────────────────┬──────────────────┐
│ Feature              │ Koyeb (PostgreSQL) │ Client (SQLite)  │
├──────────────────────┼────────────────────┼──────────────────┤
│ Tables               │ 69+                │ 69+              │
│ RLS Policies         │ 216+               │ ❌ Not supported │
│ Roles                │ 5 password roles   │ ❌ Not supported │
│ Trigger Functions    │ 7 PL/pgSQL         │ ❌ Not supported │
│ Triggers             │ 9 triggers         │ 4 triggers       │
│ Password Source      │ Environment        │ Server fetch     │
│ Encryption           │ SSL/TLS            │ File permissions │
│ Sync                 │ Master             │ Slave            │
└──────────────────────┴────────────────────┴──────────────────┘
```

---

## Support and Documentation

### Documentation Sources
- `docs/MASSIVE_SECURITY_BRAINSTORM.md` - Sync & Architecture
- `docs/MAXIMUM_SECURITY_IMPLEMENTATION.md` - RLS Policies
- `docs/COMPREHENSIVE_BUILDER_COMPLETE.md` - Setup Guide
- `docs/TRIGGER_BRAINSTORM.md` - Trigger Functions
- `docs/RLS_SETUP_GUIDE.md` - RLS Configuration

### Total Implementation
- **Documentation:** 2,978 lines
- **Code Added:** 2,363 lines
- **Final File:** 4,586 lines
- **Tables:** 69+
- **RLS Policies:** 216+
- **Trigger Functions:** 7
- **Triggers:** 9 (PostgreSQL), 4 (SQLite)
- **Roles:** 5

---

*Generated: 2026-04-17*  
*Version: 8.2.0*  
*Status: Production Ready*
