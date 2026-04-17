# QTCL DATABASE TRIGGER BRAINSTORM - MASSIVE LINKAGE ANALYSIS

## Executive Summary: 69 Tables, 200+ Potential Trigger Relationships

This document explores how to link the entire QTCL database together via triggers for automatic maintenance, data integrity, audit trails, and blockchain consistency.

---

## Table Categories (69 Total Tables)

### 1. Blockchain Core (6 tables)
- `blocks` - Block headers and metadata
- `transactions` - Transaction records
- `transaction_inputs` - Tx inputs
- `transaction_outputs` - Tx outputs (UTXO model)
- `transaction_receipts` - Execution receipts
- `orphan_blocks` - Blocks awaiting parent

### 2. Wallet & Financial (7 tables)
- `wallet_addresses` - Wallet balances and metadata
- `address_balance_history` - Audit trail (NEW)
- `address_transactions` - Per-address tx history
- `address_utxos` - UTXO tracking
- `address_labels` - Human-readable labels
- `wallet_encrypted_seeds` - Key vault
- `encrypted_private_keys` - Private key storage

### 3. Oracle & Consensus (12 tables)
- `oracle_registry` - Oracle node registry
- `oracle_coherence_metrics` - Quantum metrics
- `oracle_consensus_state` - Consensus decisions
- `oracle_density_matrix_stream` - DM snapshots
- `oracle_distribution_log` - Data distribution
- `oracle_entanglement_records` - Entanglement proofs
- `oracle_entropy_feeds` - QRNG entropy
- `oracle_pq0_state` - PQ0 state tracking
- `oracle_w_state_snapshots` - W-state data
- `quantum_coherence_snapshots` - Coherence data
- `quantum_measurements` - Measurement results
- `w_state_snapshots` - W-state consensus

### 4. Peer & Network (8 tables)
- `peer_registry` - Peer identities
- `peer_connections` - Connection state
- `peer_reputation` - Reputation scores
- `network_events` - Network events
- `network_partition_events` - Partition detection
- `network_bandwidth_usage` - Bandwidth tracking
- `client_block_sync` - Client sync state
- `client_oracle_sync` - Oracle sync state

### 5. Geometry & Lattice (5 tables)
- `hyperbolic_triangles` - Tessellation geometry
- `pseudoqubits` - Virtual qubits
- `pq_sequential` - Sequential ordering
- `quantum_lattice_metadata` - Lattice metadata
- `lattice_sync_state` - Sync tracking

### 6. Validation & Staking (4 tables)
- `validators` - Validator registry
- `validator_stakes` - Stake records
- `epochs` - Epoch boundaries
- `epoch_validators` - Epoch assignments

### 7. Security & Audit (6 tables)
- `audit_logs` - Audit trail
- `key_audit_log` - Key operation audit
- `nonce_ledger` - Nonce tracking
- `wallet_key_rotation_history` - Key rotation
- `wallet_seed_backup_status` - Backup tracking
- `chain_reorganizations` - Reorg history

### 8. System & Metrics (6 tables)
- `system_metrics` - System health
- `database_metadata` - Schema info
- `consensus_events` - Consensus log
- `entropy_quality_log` - Entropy metrics
- `merkle_proofs` - Merkle tree data
- `block_headers_cache` - Block cache

### 9. Quantum Operations (8 tables)
- `quantum_circuit_execution` - Circuit runs
- `quantum_density_matrix_global` - Global DM
- `quantum_error_correction` - QEC data
- `quantum_phase_evolution` - Phase tracking
- `quantum_shadow_tomography` - Shadow data
- `quantum_supremacy_proofs` - Supremacy evidence
- `entanglement_records` - Entanglement log
- `w_state_validator_states` - Per-validator W-state

### 10. Client/Local State (7 tables)
- `client_network_metrics` - Client network
- `client_sync_events` - Sync events
- `finality_records` - Finality tracking
- `state_root_updates` - State root changes
- `quantum_measurements` - Local measurements
- `address_balance_history` - Local balance audit
- `schema_migrations` - Schema versions

---

## TRIGGER ARCHITECTURE PATTERNS

### Pattern 1: Audit Trail Cascade
**Concept**: Every significant change creates audit records

```sql
-- Master pattern: Any table INSERT/UPDATE/DELETE → audit_logs
CREATE TRIGGER trg_blocks_audit
AFTER INSERT OR UPDATE OR DELETE ON blocks
FOR EACH ROW EXECUTE FUNCTION fn_audit_log('blocks');

-- Specialized: wallet_addresses.balance change → address_balance_history
CREATE TRIGGER trg_balance_audit
AFTER UPDATE OF balance ON wallet_addresses
FOR EACH ROW EXECUTE FUNCTION fn_record_balance_history();
```

### Pattern 2: Derived Data Maintenance
**Concept**: One table change auto-updates calculated/derived data

```sql
-- blocks.new_block → Update chain state
-- blocks.new_block → Update all peers' block_height
-- blocks.new_block → Check for chain reorganization
-- blocks.new_block → Finalize older blocks
```

### Pattern 3: Referential Integrity (Soft FK)
**Concept**: SQLite-friendly enforcement without rigid FK constraints

```sql
-- transactions.to_address → Must exist in wallet_addresses
CREATE TRIGGER trg_tx_validate_address
BEFORE INSERT ON transactions
FOR EACH ROW EXECUTE FUNCTION fn_validate_address_exists();

-- oracle_registry.oracle_id → Must be unique and valid
CREATE TRIGGER trg_oracle_validate_id
BEFORE INSERT ON oracle_registry
FOR EACH ROW EXECUTE FUNCTION fn_validate_oracle_id();
```

### Pattern 4: Automated Calculations
**Concept**: Auto-compute and cache derived values

```sql
-- oracle_density_matrix_stream.metrics → Auto-compute purity, entropy
CREATE TRIGGER trg_dm_auto_compute
BEFORE INSERT ON oracle_density_matrix_stream
FOR EACH ROW EXECUTE FUNCTION fn_compute_dm_metrics();

-- blocks.difficulty, nonce → Auto-verify proof of work
CREATE TRIGGER trg_blocks_verify_pow
BEFORE INSERT ON blocks
FOR EACH ROW EXECUTE FUNCTION fn_verify_proof_of_work();
```

### Pattern 5: Cross-Table Synchronization
**Concept**: Keep related tables in sync

```sql
-- peer_registry.last_seen → Update peer_connections if exists
-- wallet_addresses.balance → Sync to address_utxos (if UTXO model)
-- oracle_registry.is_primary → Ensure only one primary
CREATE TRIGGER trg_single_primary_oracle
BEFORE UPDATE OF is_primary ON oracle_registry
FOR EACH ROW WHEN (NEW.is_primary = TRUE)
EXECUTE FUNCTION fn_ensure_single_primary();
```

### Pattern 6: Temporal Triggers
**Concept**: Time-based state transitions

```sql
-- orphan_blocks.expires_at → Auto-cleanup expired orphans
CREATE TRIGGER trg_cleanup_expired_orphans
AFTER INSERT ON orphan_blocks
FOR EACH ROW EXECUTE FUNCTION fn_schedule_orphan_cleanup();

-- finality_records.finality_time → Auto-finalize blocks
CREATE TRIGGER trg_auto_finalize
AFTER INSERT ON finality_records
FOR EACH ROW EXECUTE FUNCTION fn_finalize_blocks();
```

### Pattern 7: Consensus Triggers
**Concept**: Blockchain-specific consensus logic

```sql
-- oracle_consensus_state.consensus_reached → Update all oracles
-- oracle_w_state_snapshots.new_snapshot → Check consensus
-- blocks.height → Trigger difficulty retarget
CREATE TRIGGER trg_difficulty_retarget
AFTER INSERT ON blocks
FOR EACH ROW WHEN (NEW.height % 2016 = 0)
EXECUTE FUNCTION fn_retarget_difficulty();
```

### Pattern 8: RLS Enforcement Triggers
**Concept**: Security and access control

```sql
-- wallet_addresses.access → Log to key_audit_log
CREATE TRIGGER trg_wallet_access_log
AFTER SELECT ON wallet_addresses
FOR EACH ROW EXECUTE FUNCTION fn_log_access('wallet_addresses');

-- peer_registry.reputation_score → Update if suspicious activity
```

---

## MASSIVE TRIGGER LINKAGE MAP

### LEVEL 1: Core Blockchain Triggers

```
blocks (central table)
  ├── INSERT → transactions (cascade tx records)
  ├── INSERT → wallet_addresses (credit miner reward)
  ├── INSERT → address_balance_history (audit miner reward)
  ├── INSERT → oracle_registry (update all oracles' block_height)
  ├── INSERT → peer_registry (update all peers' block_height)
  ├── INSERT → chain_reorganizations (if parent missing)
  ├── INSERT → difficulty_retarget (if retarget interval)
  ├── INSERT → finality_records (schedule finality)
  ├── UPDATE → blocks (validate state transitions)
  └── DELETE → orphan_blocks (if not in main chain)

transactions
  ├── INSERT → wallet_addresses (update balances)
  ├── INSERT → address_transactions (per-address history)
  ├── INSERT → address_balance_history (audit)
  ├── INSERT → transaction_receipts (execution result)
  ├── INSERT → audit_logs (record tx)
  ├── UPDATE → transaction_inputs (mark spent)
  ├── UPDATE → address_utxos (mark spent)
  └── DELETE → (prevent - tx immutable)
```

### LEVEL 2: Wallet & Financial Triggers

```
wallet_addresses
  ├── INSERT → audit_logs (new wallet created)
  ├── UPDATE balance → address_balance_history (audit trail)
  ├── UPDATE balance → wallet_encrypted_seeds (update height)
  ├── UPDATE → key_audit_log (if key changes)
  ├── UPDATE → system_metrics (total supply calculation)
  └── DELETE → (prevent if balance > 0)

address_balance_history
  ├── INSERT → system_metrics (update circulating supply)
  ├── INSERT → address_transactions (cross-reference)
  └── (immutable - no UPDATE/DELETE)

address_utxos
  ├── INSERT → wallet_addresses (add balance)
  ├── UPDATE spent → wallet_addresses (subtract balance)
  ├── UPDATE spent → address_balance_history (record spend)
  └── DELETE → (prevent if unspent)
```

### LEVEL 3: Oracle Consensus Triggers

```
oracle_registry
  ├── INSERT → oracle_coherence_metrics (init metrics)
  ├── UPDATE is_primary → (ensure only 1 primary - complex logic)
  ├── UPDATE last_seen → oracle_distribution_log (heartbeat)
  ├── UPDATE block_height → oracle_consensus_state (check sync)
  └── DELETE → (cascade cleanup to all oracle_* tables)

oracle_w_state_snapshots
  ├── INSERT → oracle_coherence_metrics (compute coherence)
  ├── INSERT → oracle_consensus_state (check agreement)
  ├── INSERT → quantum_coherence_snapshots (archive)
  ├── INSERT → w_state_snapshots (global consensus)
  └── UPDATE → (prevent - snapshots immutable)

oracle_density_matrix_stream
  ├── INSERT → quantum_density_matrix_global (update global)
  ├── INSERT → quantum_coherence_snapshots (compute metrics)
  ├── UPDATE → oracle_entanglement_records (recalculate)
  └── (stream data - may have cleanup trigger)

oracle_consensus_state
  ├── INSERT/UPDATE consensus_reached → blocks (finalize)
  ├── INSERT/UPDATE → network_events (broadcast consensus)
  ├── INSERT/UPDATE → validators (update scores)
  └── UPDATE → client_oracle_sync (notify clients)
```

### LEVEL 4: Peer Network Triggers

```
peer_registry
  ├── INSERT → peer_connections (init connection state)
  ├── INSERT → peer_reputation (init reputation)
  ├── UPDATE last_seen → network_events (heartbeat event)
  ├── UPDATE block_height → client_block_sync (notify clients)
  ├── UPDATE reputation_score → (threshold checks)
  └── DELETE → peer_connections (cleanup), peer_reputation (archive)

peer_connections
  ├── INSERT → network_bandwidth_usage (init bandwidth)
  ├── UPDATE latency_ms → peer_registry (update score)
  ├── UPDATE disconnected_at → network_events (disconnect)
  └── DELETE → (archive to log)

network_partition_events
  ├── INSERT → network_events (partition detected)
  ├── UPDATE partition_healed → (notify all peers)
  └── → client_sync_events (notify clients of partition)
```

### LEVEL 5: Quantum & Geometry Triggers

```
hyperbolic_triangles
  ├── INSERT → pseudoqubits (auto-generate qubits)
  ├── UPDATE → pq_sequential (update ordering)
  └── DELETE → pseudoqubits (cascade)

pseudoqubits
  ├── INSERT → quantum_measurements (init measurement slot)
  ├── UPDATE coherence_measure → quantum_coherence_snapshots
  ├── UPDATE phase_theta → quantum_phase_evolution (log change)
  └── DELETE → (cleanup measurements)

quantum_measurements
  ├── INSERT → quantum_coherence_snapshots (compute)
  ├── INSERT → quantum_shadow_tomography (if shadow tomography)
  ├── INSERT → quantum_supremacy_proofs (if supremacy)
  └── UPDATE → oracle_coherence_metrics (update oracle)
```

### LEVEL 6: Validation & Staking Triggers

```
validators
  ├── INSERT → validator_stakes (init stake record)
  ├── UPDATE blocks_proposed → epochs (update epoch stats)
  ├── UPDATE blocks_missed → (penalty check)
  ├── UPDATE is_slashed → validator_stakes (freeze)
  └── DELETE → validator_stakes (return stakes)

validator_stakes
  ├── INSERT → validators (update total stake)
  ├── UPDATE active → (consensus weight update)
  └── UPDATE unstake_at_height → schedule_unstake

epochs
  ├── INSERT → epoch_validators (assign validators)
  ├── INSERT → difficulty_retarget (if epoch boundary)
  └── UPDATE → (trigger validator rotation)
```

### LEVEL 7: Security & Audit Triggers

```
audit_logs
  ├── INSERT → (no triggers - terminal table)
  └── (immutable - no UPDATE/DELETE)

key_audit_log
  ├── INSERT → system_metrics (security event count)
  ├── INSERT → (alert if suspicious pattern)
  └── (immutable)

wallet_encrypted_seeds
  ├── INSERT → key_audit_log (seed created)
  ├── UPDATE → key_audit_log (key rotation)
  └── DELETE → (prevent - require rotation instead)

encrypted_private_keys
  ├── INSERT → key_audit_log (key stored)
  └── DELETE → (prevent)
```

### LEVEL 8: System & Maintenance Triggers

```
system_metrics
  ├── INSERT → database_metadata (version tracking)
  └── UPDATE → (alert if thresholds exceeded)

database_metadata
  └── (reference table - minimal triggers)

schema_migrations
  ├── INSERT → audit_logs (schema change)
  └── (immutable)
```

---

## PRIORITY TRIGGER IMPLEMENTATIONS

### TIER 1: Critical Blockchain Integrity (Must Have)

1. **Balance History Trigger** ✅ DONE
   - wallet_addresses.balance UPDATE → address_balance_history
   - Files: `qtcl_client.py`, `qtcl_db_builder.py`

2. **Block Validation Trigger**
   - blocks.BEFORE INSERT → Validate parent_hash exists
   - blocks.BEFORE INSERT → Verify proof of work
   - blocks.BEFORE INSERT → Check difficulty

3. **Reward Distribution Trigger**
   - blocks.AFTER INSERT → Credit miner wallet_addresses
   - blocks.AFTER INSERT → Credit treasury wallet_addresses
   - blocks.AFTER INSERT → Record in address_balance_history

4. **Transaction Validation Trigger**
   - transactions.BEFORE INSERT → Validate from_address has balance
   - transactions.BEFORE INSERT → Check nonce sequence
   - transactions.AFTER INSERT → Update from/to wallet_addresses balance

### TIER 2: Data Consistency (High Priority)

5. **Peer Height Sync Trigger**
   - blocks.AFTER INSERT → Update all peer_registry.block_height
   - peer_registry.UPDATE last_seen → Update peer_connections

6. **Oracle Sync Trigger**
   - blocks.AFTER INSERT → Update oracle_registry.block_height
   - oracle_w_state_snapshots.AFTER INSERT → Update oracle_consensus_state

7. **UTXO Maintenance Trigger**
   - transaction_outputs.AFTER INSERT → Add to address_utxos
   - transaction_inputs.AFTER UPDATE → Mark address_utxos.spent
   - transaction_inputs.AFTER UPDATE → Subtract wallet_addresses.balance

8. **Chain Reorganization Trigger**
   - blocks.BEFORE INSERT IF parent_hash not in blocks → Insert to orphan_blocks
   - orphan_blocks.AFTER UPDATE (parent found) → Move to blocks
   - blocks.DELETE (chain reorg) → Cascade balance adjustments

### TIER 3: Audit & Security (Medium Priority)

9. **Comprehensive Audit Trail**
   - All financial tables → audit_logs
   - All security tables → key_audit_log
   - All consensus tables → consensus_events

10. **RLS Enforcement Trigger**
    - wallet_addresses.BEFORE SELECT → Log access to key_audit_log
    - peer_registry.BEFORE UPDATE → Validate peer identity

11. **Suspicious Activity Detection**
    - wallet_addresses.BEFORE UPDATE balance → Check for overflow/underflow
    - transactions.BEFORE INSERT → Check for double-spend patterns

### TIER 4: Performance & Maintenance (Low Priority)

12. **Auto-Cleanup Triggers**
    - orphan_blocks.AFTER INSERT → Schedule cleanup after expires_at
    - audit_logs.AFTER INSERT → Archive if older than retention
    - system_metrics.AFTER INSERT → Aggregate older data

13. **Cache Maintenance Triggers**
    - blocks.AFTER INSERT → Update block_headers_cache
    - transactions.AFTER INSERT → Update merkle_proofs

14. **Statistics Triggers**
    - pseudoqubits.AFTER INSERT → Update quantum_lattice_metadata
    - oracle_entropy_feeds.AFTER INSERT → Update entropy_quality_log

---

## COMPLEX TRIGGER SCENARIOS

### Scenario 1: New Block Mined (The Big One)

```sql
-- When a miner submits a new block:

-- 1. blocks.BEFORE INSERT:
--    - Validate parent_hash exists (or insert to orphan_blocks)
--    - Verify proof of work (hash < target)
--    - Check difficulty matches retarget
--    - Validate timestamp > parent.timestamp

-- 2. blocks.AFTER INSERT:
--    a. Credit Rewards:
--       - INSERT/UPDATE wallet_addresses (miner_address, +reward)
--       - INSERT/UPDATE wallet_addresses (treasury_address, +treasury_reward)
--       - INSERT address_balance_history (miner, +reward, block_height)
--       - INSERT address_balance_history (treasury, +treasury_reward, block_height)
    
--    b. Process Transactions:
--       - For each tx in block:
--         * UPDATE wallet_addresses (from_address, -amount-fee)
--         * UPDATE wallet_addresses (to_address, +amount)
--         * INSERT address_transactions (from_addr, 'out')
--         * INSERT address_transactions (to_addr, 'in')
--         * INSERT address_balance_history (from_addr, -amount-fee)
--         * INSERT address_balance_history (to_addr, +amount)
--         * UPDATE address_utxos (mark spent)
--         * INSERT transaction_receipts
    
--    c. Update Chain State:
--       - UPDATE oracle_registry SET block_height = NEW.height (all oracles)
--       - UPDATE peer_registry SET block_height = NEW.height (all peers)
--       - UPDATE validators SET blocks_proposed += 1 (for block proposer)
--       - INSERT chain_reorganizations (if reorg detected)
    
--    d. Check Consensus:
--       - Check if we have 5 oracle_w_state_snapshots for this height
--       - If yes, INSERT oracle_consensus_state
--       - If consensus reached, UPDATE blocks SET finalized = TRUE
    
--    e. Update Metrics:
--       - UPDATE system_metrics (total_blocks += 1)
--       - INSERT quantum_coherence_snapshots (if quantum data)
    
--    f. Audit:
--       - INSERT audit_logs ('block_accepted', block_hash, miner)
```

### Scenario 2: Oracle W-State Submission

```sql
-- When an oracle submits a W-state snapshot:

-- 1. oracle_w_state_snapshots.BEFORE INSERT:
--    - Verify oracle_id exists in oracle_registry
--    - Verify block_hash matches blocks at block_height
--    - Validate w_state_hash format

-- 2. oracle_w_state_snapshots.AFTER INSERT:
--    a. Update Oracle:
--       - UPDATE oracle_registry SET last_seen = NOW() WHERE oracle_id = NEW.oracle_id
--       - UPDATE oracle_registry SET block_height = NEW.block_height
    
--    b. Compute Consensus:
--       - Count oracle_w_state_snapshots for this block_height
--       - If count >= 3 (majority of 5):
--         * Compute agreement hash
--         * INSERT oracle_consensus_state
--         * If agreement > threshold, UPDATE blocks SET finalized = TRUE
    
--    c. Update Metrics:
--       - INSERT oracle_coherence_metrics (compute from w_state)
--       - UPDATE quantum_coherence_snapshots
    
--    d. Distribute:
--       - INSERT oracle_distribution_log (record distribution)
--       - Update client_oracle_sync for all clients
```

### Scenario 3: Chain Reorganization

```sql
-- When a longer chain is detected:

-- 1. Find common ancestor
-- 2. For each block being orphaned (from tip to ancestor):
--    a. blocks.UPDATE (move to orphan_blocks or mark orphaned)
--    b. For each transaction in orphaned block:
--       - UPDATE transactions SET status = 'reverted', height = NULL
--       - UPDATE wallet_addresses (reverse balance changes)
--       - INSERT address_balance_history (record reversal)
--       - UPDATE address_utxos (unmark spent)
--    c. INSERT chain_reorganizations (log the reorg)
--    d. UPDATE finality_records (if finality changed)

-- 3. For each block in new chain (from ancestor to new tip):
--    a. blocks.INSERT (if not exists)
--    b. Apply transactions (same as Scenario 1)
--    c. UPDATE chain_reorganizations (mark reinserted)

-- 4. Notify:
--    - INSERT network_events ('reorganization', depth, common_ancestor)
--    - UPDATE all peers and clients via sync tables
```

### Scenario 4: Validator Stake/Unstake

```sql
-- When validator stakes:

-- 1. validator_stakes.BEFORE INSERT:
--    - Verify validator_id exists in validators
--    - Verify amount >= minimum_stake
--    - Check wallet_addresses balance has sufficient funds

-- 2. validator_stakes.AFTER INSERT:
--    a. Update Validator:
--       - UPDATE validators SET stake = stake + NEW.amount
--       - UPDATE validators SET status = 'active' if first stake
    
--    b. Update Wallet:
--       - UPDATE wallet_addresses (staking_address, -amount)
--       - INSERT address_balance_history (staking_address, -amount, 'stake')
    
--    c. Update Epoch:
--       - If stake_at_height is in future epoch:
--         * INSERT epoch_validators (schedule for epoch)
    
--    d. Audit:
--       - INSERT audit_logs ('stake', validator_id, amount)

-- When validator unstakes (at unstake_at_height):
-- 3. Scheduled Trigger (runs at each new block):
--    - SELECT * FROM validator_stakes WHERE unstake_at_height = NEW.height
--    - For each:
--      * UPDATE validator_stakes SET active = FALSE
--      * UPDATE validators SET stake = stake - amount
--      * UPDATE wallet_addresses (staking_address, +amount)
--      * INSERT address_balance_history (staking_address, +amount, 'unstake')
```

---

## TRIGGER IMPLEMENTATION STRATEGY

### Phase 1: Core (Week 1)
1. Balance history trigger ✅ DONE
2. Block validation triggers
3. Reward distribution triggers
4. Transaction validation triggers

### Phase 2: Consensus (Week 2)
5. Oracle W-state consensus triggers
6. Block finalization triggers
7. Difficulty retarget triggers

### Phase 3: Network (Week 3)
8. Peer height sync triggers
9. Chain reorganization triggers
10. Orphan block management

### Phase 4: Audit (Week 4)
11. Comprehensive audit logging
12. Security event detection
13. RLS enforcement triggers

### Phase 5: Optimization (Ongoing)
14. Cache maintenance
15. Auto-cleanup
16. Statistics aggregation

---

## TECHNICAL NOTES

### SQLite vs PostgreSQL Trigger Differences

**SQLite Limitations:**
- No BEFORE/AFTER DELETE triggers with OLD reference in some cases
- Limited trigger recursion depth
- No INSTEAD OF triggers on tables
- Cannot modify the same table in trigger that fired it (recursion protection)

**PostgreSQL Advantages:**
- Full BEFORE/AFTER/INSTEAD OF support
- Rich PL/pgSQL for complex logic
- Exception handling in triggers
- Can call functions, procedures

**Cross-Platform Strategy:**
- Keep trigger logic simple
- Use functions for complex operations
- Document platform-specific code

### Performance Considerations

1. **Trigger Overhead**
   - Each trigger adds latency to INSERT/UPDATE/DELETE
   - Batch operations become slower
   - Solution: Disable triggers for bulk loads, re-enable after

2. **Cascade Effects**
   - One INSERT → multiple triggers → multiple INSERTs
   - Can create exponential growth
   - Solution: Careful trigger design, avoid deep cascades

3. **Locking**
   - Triggers hold locks on affected tables
   - Long-running triggers block other transactions
   - Solution: Keep triggers fast, defer heavy work to async jobs

4. **Recursion Prevention**
   - Trigger A updates table B
   - Trigger B on table B updates table A
   - Infinite recursion!
   - Solution: Session variables to detect recursion, or careful design

### Best Practices

1. **Idempotency**
   - Triggers should be safe to run multiple times
   - Use IF NOT EXISTS, UPSERT, etc.

2. **Defensive Programming**
   - Always check if row exists before updating
   - Handle NULLs gracefully
   - Use transactions for multi-table updates

3. **Logging**
   - Log trigger execution for debugging
   - Use different log levels for different operations

4. **Testing**
   - Test triggers with edge cases (NULLs, empty tables)
   - Test cascade scenarios
   - Performance test with large datasets

---

## IMPLEMENTATION EXAMPLES

### Example 1: Block Reward Distribution Trigger (PostgreSQL)

```sql
CREATE OR REPLACE FUNCTION fn_distribute_block_rewards()
RETURNS TRIGGER AS $$
DECLARE
    _miner_reward NUMERIC;
    _treasury_reward NUMERIC;
    _treasury_address VARCHAR;
BEGIN
    -- Get reward amounts
    SELECT miner_reward, treasury_reward, treasury_address
    INTO _miner_reward, _treasury_reward, _treasury_address
    FROM get_block_rewards(NEW.height);
    
    -- Credit miner
    INSERT INTO wallet_addresses (address, balance, address_type, balance_at_height)
    VALUES (NEW.miner_address, _miner_reward, 'miner', NEW.height)
    ON CONFLICT (address) DO UPDATE
    SET balance = wallet_addresses.balance + _miner_reward,
        balance_at_height = NEW.height,
        updated_at = NOW();
    
    -- Record in history
    INSERT INTO address_balance_history 
    (address, block_height, block_hash, balance, delta, snapshot_timestamp)
    VALUES (NEW.miner_address, NEW.height, NEW.block_hash, 
            COALESCE((SELECT balance FROM wallet_addresses WHERE address = NEW.miner_address), 0),
            _miner_reward, NOW());
    
    -- Credit treasury
    INSERT INTO wallet_addresses (address, balance, address_type, balance_at_height)
    VALUES (_treasury_address, _treasury_reward, 'treasury', NEW.height)
    ON CONFLICT (address) DO UPDATE
    SET balance = wallet_addresses.balance + _treasury_reward,
        balance_at_height = NEW.height,
        updated_at = NOW();
    
    -- Record in history
    INSERT INTO address_balance_history 
    (address, block_height, block_hash, balance, delta, snapshot_timestamp)
    VALUES (_treasury_address, NEW.height, NEW.block_hash,
            COALESCE((SELECT balance FROM wallet_addresses WHERE address = _treasury_address), 0),
            _treasury_reward, NOW());
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_blocks_reward_distribution
AFTER INSERT ON blocks
FOR EACH ROW
EXECUTE FUNCTION fn_distribute_block_rewards();
```

### Example 2: Transaction Validation Trigger (SQLite)

```sql
CREATE TRIGGER trg_tx_validate_sender_balance
BEFORE INSERT ON transactions
BEGIN
    SELECT CASE
        WHEN (
            SELECT COALESCE(balance, 0) 
            FROM wallet_addresses 
            WHERE address = NEW.from_address
        ) < NEW.amount + COALESCE(NEW.fee, 0)
        THEN RAISE(ABORT, 'Insufficient balance')
    END;
END;
```

### Example 3: Oracle Consensus Trigger

```sql
CREATE OR REPLACE FUNCTION fn_check_oracle_consensus()
RETURNS TRIGGER AS $$
DECLARE
    _snapshot_count INTEGER;
    _agreement_hash VARCHAR;
BEGIN
    -- Count snapshots for this block
    SELECT COUNT(*), mode() WITHIN GROUP (ORDER BY w_state_hash)
    INTO _snapshot_count, _agreement_hash
    FROM oracle_w_state_snapshots
    WHERE block_height = NEW.block_height;
    
    -- If we have majority (3 of 5)
    IF _snapshot_count >= 3 THEN
        -- Insert consensus state
        INSERT INTO oracle_consensus_state
        (block_height, oracle_consensus_reached, validator_agreement_count, 
         w_state_hash_agreement, consensus_threshold)
        VALUES (NEW.block_height, TRUE, _snapshot_count, TRUE, 0.6)
        ON CONFLICT (block_height) DO UPDATE
        SET oracle_consensus_reached = TRUE,
            validator_agreement_count = _snapshot_count,
            w_state_hash_agreement = TRUE;
        
        -- Finalize block if not already
        UPDATE blocks SET finalized = TRUE
        WHERE height = NEW.block_height AND finalized = FALSE;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_oracle_consensus_check
AFTER INSERT ON oracle_w_state_snapshots
FOR EACH ROW
EXECUTE FUNCTION fn_check_oracle_consensus();
```

---

## CONCLUSION

### What We Can Link (High Value)
1. ✅ Balance history audit trail (DONE)
2. ✅ Block reward distribution (NEXT)
3. ✅ Transaction validation (NEXT)
4. ✅ Oracle consensus (NEXT)
5. ✅ Chain reorganization handling (NEXT)
6. ✅ Peer synchronization (MEDIUM)

### What We Should Link (Medium Value)
7. Comprehensive audit logging
8. UTXO maintenance
9. Statistics aggregation
10. Cache maintenance

### What We Could Link (Lower Priority)
11. Auto-cleanup
12. Performance monitoring
13. Suspicious activity detection

### Recommended Implementation
Start with **Tier 1 triggers** (5 triggers) for core blockchain integrity, then add **Tier 2** (4 triggers) for data consistency. This gives us 9 critical triggers that link the database together properly.

**Total trigger count estimate**: 50-100 triggers for complete linkage of all 69 tables.

---

*Document generated: 2026-04-17*
*Version: 8.0.1*
*Tables analyzed: 69*
*Potential trigger relationships: 200+*
