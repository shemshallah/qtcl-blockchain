# QTCL Database Schema Cross-Check & RLS Setup Guide

## Executive Summary

### Schema Mismatch Issues Found

| Issue | Server Schema (`qtcl_db_builder.py`) | Client Implementation (`qtcl_client.py`) | Status |
|-------|-------------------------------------|------------------------------------------|--------|
| `wallet_addresses.balance` | `NUMERIC(30,0)` (base units) | Client displays `/100` correctly | ✅ OK |
| `wallet_addresses` columns | 17 columns including `last_updated` | Missing `last_updated` in client CREATE | ⚠️ MINOR |
| `blocks` columns | Has `pq_curr`, `pq_last`, `mermin_value` | Client only has `pq0`, `pq_curr`, `pq_last` | ⚠️ PARTIAL |
| `transactions` columns | 20+ columns | Client has subset ( adequate) | ✅ OK |
| `address_balance_history` | Exists on server | NOT implemented in client SQLite | ❌ MISSING |

### Critical Finding: `address_balance_history` Table

**SERVER HAS IT, CLIENT DOESN'T.**

This table is crucial for:
- Auditing balance changes over time
- Proving rewards were credited at specific block heights
- RLS policies that restrict balance visibility by time/block

**Client needs this table added to `ChainDB._init_db()` around line 3200.**

---

## Comprehensive RLS Setup for All Tables

### Step 1: Create Database Roles (Run as superuser)

```sql
-- Create application roles with passwords you specified
CREATE ROLE qtcl_oracle WITH LOGIN PASSWORD 'your_oracle_password';
CREATE ROLE qtcl_miner WITH LOGIN PASSWORD 'miner_password';
CREATE ROLE qtcl_treasury WITH LOGIN PASSWORD 'your_treasury_password';
CREATE ROLE qtcl_admin WITH LOGIN PASSWORD 'your_admin_password';
CREATE ROLE qtcl_readonly WITH LOGIN PASSWORD 'your_readonly_password';

-- Grant base connection permissions
GRANT CONNECT ON DATABASE neondb TO qtcl_oracle, qtcl_miner, qtcl_treasury, qtcl_admin, qtcl_readonly;

-- Grant schema usage
GRANT USAGE ON SCHEMA public TO qtcl_oracle, qtcl_miner, qtcl_treasury, qtcl_admin, qtcl_readonly;
```

### Step 2: Enable RLS on ALL Tables

```sql
-- Financial/Balance Tables
ALTER TABLE wallet_addresses ENABLE ROW LEVEL SECURITY;
ALTER TABLE address_balance_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE address_transactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE address_utxos ENABLE ROW LEVEL SECURITY;
ALTER TABLE address_labels ENABLE ROW LEVEL SECURITY;

-- Block/Chain Tables
ALTER TABLE blocks ENABLE ROW LEVEL SECURITY;
ALTER TABLE block_headers_cache ENABLE ROW LEVEL SECURITY;
ALTER TABLE chain_reorganizations ENABLE ROW LEVEL SECURITY;
ALTER TABLE orphan_blocks ENABLE ROW LEVEL SECURITY;
ALTER TABLE state_root_updates ENABLE ROW LEVEL SECURITY;
ALTER TABLE finality_records ENABLE ROW LEVEL SECURITY;

-- Transaction Tables
ALTER TABLE transactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE transaction_inputs ENABLE ROW LEVEL SECURITY;
ALTER TABLE transaction_outputs ENABLE ROW LEVEL SECURITY;
ALTER TABLE transaction_receipts ENABLE ROW LEVEL SECURITY;

-- Oracle Tables (Critical for your question)
ALTER TABLE oracle_registry ENABLE ROW LEVEL SECURITY;
ALTER TABLE oracle_coherence_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE oracle_consensus_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE oracle_density_matrix_stream ENABLE ROW LEVEL SECURITY;
ALTER TABLE oracle_distribution_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE oracle_entanglement_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE oracle_entropy_feeds ENABLE ROW LEVEL SECURITY;
ALTER TABLE oracle_pq0_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE oracle_w_state_snapshots ENABLE ROW LEVEL SECURITY;

-- Peer/Network Tables
ALTER TABLE peer_registry ENABLE ROW LEVEL SECURITY;
ALTER TABLE peer_connections ENABLE ROW LEVEL SECURITY;
ALTER TABLE peer_reputation ENABLE ROW LEVEL SECURITY;
ALTER TABLE network_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE network_partition_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE network_bandwidth_usage ENABLE ROW LEVEL SECURITY;

-- Validator Tables
ALTER TABLE validators ENABLE ROW LEVEL SECURITY;
ALTER TABLE validator_stakes ENABLE ROW LEVEL SECURITY;
ALTER TABLE epochs ENABLE ROW LEVEL SECURITY;
ALTER TABLE epoch_validators ENABLE ROW LEVEL SECURITY;

-- Quantum/Science Tables
ALTER TABLE pseudoqubits ENABLE ROW LEVEL SECURITY;
ALTER TABLE hyperbolic_triangles ENABLE ROW LEVEL SECURITY;
ALTER TABLE quantum_coherence_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE quantum_density_matrix_global ENABLE ROW LEVEL SECURITY;
ALTER TABLE quantum_circuit_execution ENABLE ROW LEVEL SECURITY;
ALTER TABLE quantum_measurements ENABLE ROW LEVEL SECURITY;
ALTER TABLE w_state_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE w_state_validator_states ENABLE ROW LEVEL SECURITY;
ALTER TABLE entanglement_records ENABLE ROW LEVEL SECURITY;

-- Client Sync Tables
ALTER TABLE client_block_sync ENABLE ROW LEVEL SECURITY;
ALTER TABLE client_oracle_sync ENABLE ROW LEVEL SECURITY;
ALTER TABLE client_network_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE client_sync_events ENABLE ROW LEVEL SECURITY;

-- Security/Key Tables
ALTER TABLE encrypted_private_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE wallet_encrypted_seeds ENABLE ROW LEVEL SECURITY;
ALTER TABLE wallet_key_rotation_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE wallet_seed_backup_status ENABLE ROW LEVEL SECURITY;
ALTER TABLE key_audit_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE nonce_ledger ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;

-- System Tables
ALTER TABLE database_metadata ENABLE ROW LEVEL SECURITY;
ALTER TABLE system_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE consensus_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE entropy_quality_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE lattice_sync_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE merkle_proofs ENABLE ROW LEVEL SECURITY;
ALTER TABLE quantum_error_correction ENABLE ROW LEVEL SECURITY;
ALTER TABLE quantum_lattice_metadata ENABLE ROW LEVEL SECURITY;
ALTER TABLE quantum_phase_evolution ENABLE ROW LEVEL SECURITY;
ALTER TABLE quantum_shadow_tomography ENABLE ROW LEVEL SECURITY;
ALTER TABLE quantum_supremacy_proofs ENABLE ROW LEVEL SECURITY;
ALTER TABLE pq_sequential ENABLE ROW LEVEL SECURITY;
```

---

## Detailed RLS Policies by Table

### 🔐 FINANCIAL TABLES (Most Critical)

#### `wallet_addresses` - Who can see/modify balances

```sql
-- Policy 1: Users can see their own wallet
CREATE POLICY wallet_owner_select ON wallet_addresses
    FOR SELECT
    TO qtcl_miner, qtcl_oracle, qtcl_treasury
    USING (address = current_setting('app.current_address', true));

-- Policy 2: Miners can see their own miner-type addresses
CREATE POLICY miner_wallet_select ON wallet_addresses
    FOR SELECT
    TO qtcl_miner
    USING (address_type = 'miner' AND wallet_fingerprint = current_setting('app.miner_fingerprint', true));

-- Policy 3: Oracle can see oracle addresses
CREATE POLICY oracle_wallet_select ON wallet_addresses
    FOR SELECT
    TO qtcl_oracle
    USING (address_type = 'oracle' OR address_type = 'signing');

-- Policy 4: Treasury can see treasury addresses
CREATE POLICY treasury_wallet_select ON wallet_addresses
    FOR SELECT
    TO qtcl_treasury
    USING (address_type = 'treasury');

-- Policy 5: Admin can see all
CREATE POLICY admin_wallet_all ON wallet_addresses
    FOR ALL
    TO qtcl_admin
    USING (true);

-- Policy 6: Read-only can see but not modify
CREATE POLICY readonly_wallet_select ON wallet_addresses
    FOR SELECT
    TO qtcl_readonly
    USING (true);

-- Policy 7: System can update balances (for coinbase rewards) - using SECURITY DEFINER function
CREATE POLICY system_balance_update ON wallet_addresses
    FOR UPDATE
    TO qtcl_miner, qtcl_oracle  -- The roles that submit blocks
    USING (true)
    WITH CHECK (true);
```

#### `address_balance_history` - Audit trail

```sql
-- Users can see their own balance history
CREATE POLICY balance_history_owner ON address_balance_history
    FOR SELECT
    TO qtcl_miner, qtcl_oracle, qtcl_treasury
    USING (address = current_setting('app.current_address', true));

-- Admin can see all
CREATE POLICY balance_history_admin ON address_balance_history
    FOR ALL
    TO qtcl_admin
    USING (true);

-- Miners can see miner address history
CREATE POLICY balance_history_miner ON address_balance_history
    FOR SELECT
    TO qtcl_miner
    USING (address LIKE 'qtcl1miner_%');
```

#### `transactions` - Transaction visibility

```sql
-- Users can see transactions they're party to
CREATE POLICY tx_participant_select ON transactions
    FOR SELECT
    TO qtcl_miner, qtcl_oracle, qtcl_treasury
    USING (
        from_address = current_setting('app.current_address', true) OR
        to_address = current_setting('app.current_address', true)
    );

-- Miners can see coinbase transactions to them
CREATE POLICY tx_coinbase_miner ON transactions
    FOR SELECT
    TO qtcl_miner
    USING (tx_type = 'coinbase' AND to_address = current_setting('app.miner_address', true));

-- Oracle can see all (for validation)
CREATE POLICY tx_oracle_all ON transactions
    FOR SELECT
    TO qtcl_oracle
    USING (true);

-- Admin full access
CREATE POLICY tx_admin_all ON transactions
    FOR ALL
    TO qtcl_admin
    USING (true);
```

---

### 🔮 ORACLE TABLES (Your Primary Question)

#### `oracle_registry` - Oracle node registration

```sql
-- THIS ANSWERS YOUR QUESTION ABOUT PRIMARY_LATTICE
-- PRIMARY_LATTICE is a ROLE value in the table, NOT a database role
-- It's one of: 'PRIMARY_LATTICE', 'SECONDARY_LATTICE', 'VALIDATION', 'ARBITER', 'METRICS'

-- Policy 1: Oracles can see their own record
CREATE POLICY oracle_self_select ON oracle_registry
    FOR SELECT
    TO qtcl_oracle
    USING (oracle_id = current_setting('app.oracle_id', true));

-- Policy 2: Oracles can update their own record
CREATE POLICY oracle_self_update ON oracle_registry
    FOR UPDATE
    TO qtcl_oracle
    USING (oracle_id = current_setting('app.oracle_id', true));

-- Policy 3: PRIMARY_LATTICE oracles can see all (coordinators)
CREATE POLICY oracle_primary_all ON oracle_registry
    FOR ALL
    TO qtcl_oracle
    USING (
        role = 'PRIMARY_LATTICE' OR
        oracle_id = current_setting('app.oracle_id', true)
    );

-- Policy 4: Miners can read oracle list (for finding oracles to connect)
CREATE POLICY oracle_miner_read ON oracle_registry
    FOR SELECT
    TO qtcl_miner
    USING (true);

-- Policy 5: Admin full access
CREATE POLICY oracle_admin_all ON oracle_registry
    FOR ALL
    TO qtcl_admin
    USING (true);

-- Policy 6: Read-only can view (for monitoring)
CREATE POLICY oracle_readonly_select ON oracle_registry
    FOR SELECT
    TO qtcl_readonly
    USING (true);
```

**ABOUT PRIMARY_LATTICE**: It's a value in the `role` column, not a PostgreSQL role. 
- The `oracle_registry` table has 5 oracle slots (oracle_1 through oracle_5)
- oracle_1 typically has role='PRIMARY_LATTICE' (the leader)
- oracle_2 has role='SECONDARY_LATTICE' (backup leader)
- oracle_3-5 have roles like 'VALIDATION', 'ARBITER', 'METRICS'

#### `oracle_coherence_metrics` - Quantum metrics

```sql
-- Oracles can insert their own metrics
CREATE POLICY coherence_oracle_insert ON oracle_coherence_metrics
    FOR INSERT
    TO qtcl_oracle
    WITH CHECK (true);

-- Oracles can see all metrics (for consensus)
CREATE POLICY coherence_oracle_select ON oracle_coherence_metrics
    FOR SELECT
    TO qtcl_oracle
    USING (true);

-- Miners can see metrics (for validation)
CREATE POLICY coherence_miner_select ON oracle_coherence_metrics
    FOR SELECT
    TO qtcl_miner
    USING (true);

-- Admin full access
CREATE POLICY coherence_admin_all ON oracle_coherence_metrics
    FOR ALL
    TO qtcl_admin
    USING (true);
```

#### Other Oracle Tables (Apply Similar Pattern)

```sql
-- oracle_consensus_state - who can see consensus decisions
CREATE POLICY consensus_oracle_select ON oracle_consensus_state
    FOR SELECT
    TO qtcl_oracle, qtcl_miner, qtcl_treasury
    USING (true);

-- oracle_w_state_snapshots - W-state data
CREATE POLICY wstate_oracle_insert ON oracle_w_state_snapshots
    FOR INSERT
    TO qtcl_oracle
    WITH CHECK (true);

CREATE POLICY wstate_all_select ON oracle_w_state_snapshots
    FOR SELECT
    TO qtcl_oracle, qtcl_miner
    USING (true);

-- oracle_entropy_feeds - QRNG entropy
CREATE POLICY entropy_oracle_all ON oracle_entropy_feeds
    FOR ALL
    TO qtcl_oracle
    USING (true);

CREATE POLICY entropy_miner_select ON oracle_entropy_feeds
    FOR SELECT
    TO qtcl_miner
    USING (true);
```

---

### ⛏️ MINING TABLES

#### `blocks` - Block data access

```sql
-- Everyone can read blocks (blockchain is public)
CREATE POLICY blocks_public_select ON blocks
    FOR SELECT
    TO qtcl_miner, qtcl_oracle, qtcl_treasury, qtcl_readonly
    USING (true);

-- Miners can submit new blocks (INSERT via submitBlock RPC)
CREATE POLICY blocks_miner_insert ON blocks
    FOR INSERT
    TO qtcl_miner
    WITH CHECK (true);

-- Oracles can update block metadata (finalization, coherence)
CREATE POLICY blocks_oracle_update ON blocks
    FOR UPDATE
    TO qtcl_oracle
    USING (true);

-- Admin full access
CREATE POLICY blocks_admin_all ON blocks
    FOR ALL
    TO qtcl_admin
    USING (true);
```

#### `peer_registry` - P2P peer management

```sql
-- Peers can see other peers (for P2P discovery)
CREATE POLICY peers_public_select ON peer_registry
    FOR SELECT
    TO qtcl_miner, qtcl_oracle, qtcl_treasury
    USING (true);

-- Peers can update their own record
CREATE POLICY peers_self_update ON peer_registry
    FOR UPDATE
    TO qtcl_miner, qtcl_oracle
    USING (peer_id = current_setting('app.peer_id', true));

-- Peers can insert themselves
CREATE POLICY peers_self_insert ON peer_registry
    FOR INSERT
    TO qtcl_miner, qtcl_oracle
    WITH CHECK (peer_id = current_setting('app.peer_id', true));

-- Admin can manage all peers
CREATE POLICY peers_admin_all ON peer_registry
    FOR ALL
    TO qtcl_admin
    USING (true);
```

---

### 🔒 SECURITY TABLES (Restricted)

#### `wallet_encrypted_seeds` - CRITICAL: Only owner can access

```sql
-- Only the wallet owner can access their encrypted seed
CREATE POLICY seeds_owner_only ON wallet_encrypted_seeds
    FOR ALL
    TO qtcl_miner, qtcl_oracle, qtcl_treasury
    USING (wallet_fingerprint = current_setting('app.wallet_fingerprint', true));

-- Admin emergency access (use sparingly)
CREATE POLICY seeds_admin_emergency ON wallet_encrypted_seeds
    FOR SELECT
    TO qtcl_admin
    USING (true);
```

#### `encrypted_private_keys` - CRITICAL

```sql
-- Owner only
CREATE POLICY private_keys_owner ON encrypted_private_keys
    FOR ALL
    TO qtcl_miner, qtcl_oracle, qtcl_treasury
    USING (wallet_fingerprint = current_setting('app.wallet_fingerprint', true));

-- No admin access by default (security critical)
```

#### `key_audit_log` - Audit trail

```sql
-- Users can see their own audit trail
CREATE POLICY audit_user_select ON key_audit_log
    FOR SELECT
    TO qtcl_miner, qtcl_oracle, qtcl_treasury
    USING (wallet_fingerprint = current_setting('app.wallet_fingerprint', true));

-- Admin can see all audit logs
CREATE POLICY audit_admin_all ON key_audit_log
    FOR ALL
    TO qtcl_admin
    USING (true);
```

---

## Application-Side Role Configuration

### Python Code for Setting RLS Context

Add this to `server.py` at the start of each RPC method:

```python
def _set_rls_context(cur, role: str, **kwargs):
    """Set PostgreSQL RLS application context variables."""
    # Set the role identifier
    cur.execute("SET LOCAL app.current_role = %s", (role,))
    
    # Set additional context based on role
    if role == 'miner':
        cur.execute("SET LOCAL app.miner_address = %s", (kwargs.get('address', ''),))
        cur.execute("SET LOCAL app.miner_fingerprint = %s", (kwargs.get('fingerprint', ''),))
    elif role == 'oracle':
        cur.execute("SET LOCAL app.oracle_id = %s", (kwargs.get('oracle_id', ''),))
        cur.execute("SET LOCAL app.oracle_address = %s", (kwargs.get('address', ''),))
    elif role == 'treasury':
        cur.execute("SET LOCAL app.treasury_address = %s", (kwargs.get('address', ''),))
    
    # Set current user address if available
    if 'address' in kwargs:
        cur.execute("SET LOCAL app.current_address = %s", (kwargs['address'],))
    if 'wallet_fingerprint' in kwargs:
        cur.execute("SET LOCAL app.wallet_fingerprint = %s", (kwargs['wallet_fingerprint'],))

# Usage in submitBlock:
with get_db_cursor() as cur:
    _set_rls_context(cur, 'miner', 
                     address=miner_address,
                     fingerprint=hashlib.sha256(miner_address.encode()).hexdigest()[:64])
    # ... rest of query
```

### Client-Side Role in qtcl_client.py

```python
class ChainDB:
    def _set_rls_context(self, role: str, address: str = None):
        """Set RLS context for client-side queries."""
        if not self._sqlite_conn:
            return
        # SQLite doesn't have RLS, but we can simulate it for consistency
        self._rls_context = {
            'role': role,
            'address': address,
            'wallet_fingerprint': hashlib.sha256(address.encode()).hexdigest()[:64] if address else None
        }
```

---

## Grant Table Permissions to Roles

```sql
-- Grant SELECT to all roles for public tables
GRANT SELECT ON blocks, transactions, oracle_registry, peer_registry TO qtcl_miner, qtcl_oracle, qtcl_treasury, qtcl_readonly;

-- Grant INSERT/UPDATE for miners
GRANT INSERT, UPDATE ON blocks, transactions TO qtcl_miner;
GRANT INSERT, UPDATE ON peer_registry TO qtcl_miner;

-- Grant INSERT/UPDATE for oracles
GRANT INSERT, UPDATE ON oracle_registry, oracle_coherence_metrics TO qtcl_oracle;
GRANT INSERT, UPDATE ON oracle_w_state_snapshots, oracle_entropy_feeds TO qtcl_oracle;
GRANT INSERT, UPDATE ON blocks TO qtcl_oracle;  -- For finalization

-- Grant wallet table access
GRANT SELECT, INSERT, UPDATE ON wallet_addresses TO qtcl_miner, qtcl_oracle, qtcl_treasury;
GRANT SELECT, INSERT ON address_balance_history TO qtcl_miner, qtcl_oracle, qtcl_treasury;
GRANT SELECT, INSERT ON transactions TO qtcl_miner, qtcl_oracle;

-- Admin gets everything
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO qtcl_admin;

-- Read-only gets SELECT only
GRANT SELECT ON ALL TABLES IN SCHEMA public TO qtcl_readonly;

-- Revoke dangerous permissions from non-admin roles
REVOKE DELETE ON wallet_addresses, wallet_encrypted_seeds FROM qtcl_miner, qtcl_oracle, qtcl_treasury;
REVOKE DELETE ON encrypted_private_keys FROM qtcl_miner, qtcl_oracle, qtcl_treasury;
```

---

## Testing RLS Policies

```sql
-- Test as each role
SET ROLE qtcl_miner;
SELECT * FROM wallet_addresses;  -- Should only see matching rows
SELECT * FROM oracle_registry;     -- Should see all (for discovery)
RESET ROLE;

SET ROLE qtcl_oracle;
SELECT * FROM oracle_registry WHERE role = 'PRIMARY_LATTICE';  -- Should see primary
UPDATE oracle_registry SET last_seen = NOW() WHERE oracle_id = 'oracle_1';  -- Should work if policy allows
RESET ROLE;

SET ROLE qtcl_readonly;
SELECT * FROM blocks;  -- Should work
INSERT INTO blocks VALUES (...);  -- Should FAIL
RESET ROLE;
```

---

## Summary of Schema Fixes Needed

### Client-Side (`qtcl_client.py`)

1. **Add `address_balance_history` table** (lines ~3200-3300 in `_init_db()`):
```python
"""CREATE TABLE IF NOT EXISTS address_balance_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    address TEXT NOT NULL,
    block_height INTEGER NOT NULL,
    block_hash TEXT,
    balance INTEGER NOT NULL,  -- base units
    delta INTEGER,
    snapshot_timestamp INTEGER DEFAULT (strftime('%s', 'now')),
    UNIQUE(address, block_height)
)"""
```

2. **Add `last_updated` column to `wallet_addresses`**:
```python
"last_updated INTEGER DEFAULT 0,"
```

3. **Add triggers to maintain `address_balance_history`**:
```python
"""CREATE TRIGGER IF NOT EXISTS trg_balance_history
    AFTER UPDATE OF balance ON wallet_addresses
    BEGIN
        INSERT INTO address_balance_history 
        (address, block_height, balance, delta, snapshot_timestamp)
        VALUES (
            NEW.address,
            (SELECT MAX(height) FROM blocks),
            NEW.balance,
            NEW.balance - OLD.balance,
            strftime('%s', 'now')
        );
    END"""
```

---

## Final Notes on Oracle Roles

**PRIMARY_LATTICE** is:
- A value in `oracle_registry.role` column
- The designated leader oracle (oracle_1)
- Has special RLS permissions to see all oracle data
- NOT a PostgreSQL role (the DB role is `qtcl_oracle`)

**Oracle Role Hierarchy**:
```
PostgreSQL Role: qtcl_oracle
  └── Can connect as any oracle_id
  └── RLS Policy checks: role = 'PRIMARY_LATTICE' OR oracle_id = current_setting('app.oracle_id')
  
Application Roles (in oracle_registry table):
  - PRIMARY_LATTICE (oracle_1): Full visibility, coordination
  - SECONDARY_LATTICE (oracle_2): Backup coordination
  - VALIDATION (oracle_3): Block validation
  - ARBITER (oracle_4): Dispute resolution
  - METRICS (oracle_5): Health monitoring
```

**You DON'T need to create PRIMARY_LATTICE as a PostgreSQL role** - just ensure oracle_1 has `role = 'PRIMARY_LATTICE'` in the oracle_registry table.
