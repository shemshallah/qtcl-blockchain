#!/usr/bin/env python3
"""
QTCL COMPREHENSIVE SCHEMA MIGRATION - GOOGLE COLAB
Simple version with direct print() - no logging issues
"""

import psycopg2
from psycopg2.extras import RealDictCursor

# DATABASE CONFIG
HOST = "aws-0-us-west-2.pooler.supabase.com"
USER = "postgres.rslvlsqwkfmdtebqsvtw"
PASSWORD = "$h10j1r1H0w4rd"
PORT = 5432
DATABASE = "postgres"
TIMEOUT = 30

print("\n" + "╔" + "═" * 98 + "╗")
print("║" + " " * 98 + "║")
print("║" + "QTCL SCHEMA MIGRATION - PRODUCTION GRADE".center(98) + "║")
print("║" + " " * 98 + "║")
print("╚" + "═" * 98 + "╝\n")

print("🔗 Connecting to Supabase...")
try:
    conn = psycopg2.connect(
        host=HOST,
        user=USER,
        password=PASSWORD,
        port=PORT,
        database=DATABASE,
        connect_timeout=TIMEOUT
    )
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    print("✓ Connected to Supabase successfully!\n")
except Exception as e:
    print(f"✗ Connection failed: {e}")
    exit(1)

executed = 0
skipped = 0
errors = 0

def execute_sql(sql, description):
    global executed, skipped, errors
    try:
        cursor.execute(sql)
        conn.commit()
        print(f"  ✓ {description}")
        executed += 1
        return True
    except psycopg2.errors.DuplicateTable:
        print(f"  ⚠ {description} (already exists)")
        conn.rollback()
        skipped += 1
        return True
    except psycopg2.errors.DuplicateColumn:
        print(f"  ⚠ {description} (already exists)")
        conn.rollback()
        skipped += 1
        return True
    except psycopg2.errors.DuplicateObject:
        print(f"  ⚠ {description} (already exists)")
        conn.rollback()
        skipped += 1
        return True
    except Exception as e:
        print(f"  ✗ {description}: {str(e)[:100]}")
        conn.rollback()
        errors += 1
        return False

def table_exists(table):
    try:
        cursor.execute(f"SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = %s)", (table,))
        return cursor.fetchone()[0]
    except:
        return False

def column_exists(table, column):
    try:
        cursor.execute(f"SELECT EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name = %s AND column_name = %s)", (table, column))
        return cursor.fetchone()[0]
    except:
        return False

print("=" * 100)
print("STARTING COMPREHENSIVE SCHEMA MIGRATION (PRODUCTION-GRADE)")
print("=" * 100 + "\n")

# SECTION 1
print("📝 [1/10] User Authentication Schema")
execute_sql("ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash VARCHAR(255) UNIQUE;", "Add password_hash to users")

# SECTION 2
print("\n📝 [2/10] Sessions Management")
execute_sql("""CREATE TABLE IF NOT EXISTS sessions (
    session_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    access_token TEXT NOT NULL,
    refresh_token TEXT NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ip_address VARCHAR(45),
    user_agent TEXT
);""", "Create sessions table (user_id as TEXT)")

if table_exists('sessions'):
    execute_sql("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);", "Index sessions(user_id)")
    execute_sql("CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);", "Index sessions(expires_at)")

# SECTION 3
print("\n📝 [3/10] Auth Events (Audit Log)")
execute_sql("""CREATE TABLE IF NOT EXISTS auth_events (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT REFERENCES users(user_id) ON DELETE SET NULL,
    event_type VARCHAR(50) NOT NULL,
    email VARCHAR(255),
    success BOOLEAN DEFAULT FALSE,
    details TEXT,
    ip_address VARCHAR(45),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);""", "Create auth_events table (user_id as TEXT)")

if table_exists('auth_events'):
    execute_sql("CREATE INDEX IF NOT EXISTS idx_auth_events_user_id ON auth_events(user_id);", "Index auth_events(user_id)")
    execute_sql("CREATE INDEX IF NOT EXISTS idx_auth_events_event_type ON auth_events(event_type);", "Index auth_events(event_type)")
    execute_sql("CREATE INDEX IF NOT EXISTS idx_auth_events_created_at ON auth_events(created_at);", "Index auth_events(created_at)")

# SECTION 4
print("\n📝 [4/10] Password Reset Management")
execute_sql("""CREATE TABLE IF NOT EXISTS password_resets (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    reset_token VARCHAR(255) NOT NULL UNIQUE,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    used BOOLEAN DEFAULT FALSE,
    used_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);""", "Create password_resets table (user_id as TEXT)")

if table_exists('password_resets'):
    execute_sql("CREATE INDEX IF NOT EXISTS idx_password_resets_user_id ON password_resets(user_id);", "Index password_resets(user_id)")
    execute_sql("CREATE INDEX IF NOT EXISTS idx_password_resets_reset_token ON password_resets(reset_token);", "Index password_resets(reset_token)")

# SECTION 5
print("\n📝 [5/10] Quantum Measurements (W-State Topology)")

try:
    cursor.execute("DROP VIEW IF EXISTS v_wsv_metrics_latest_hour;")
    conn.commit()
    print("  ⚠ Dropped existing view (will recreate)")
except:
    conn.rollback()

execute_sql("""CREATE TABLE IF NOT EXISTS quantum_measurements (
    id BIGSERIAL PRIMARY KEY,
    tx_id VARCHAR(255) UNIQUE NOT NULL,
    circuit_name VARCHAR(255),
    num_qubits INT DEFAULT 8,
    num_validators INT DEFAULT 5,
    measurement_result_json JSONB,
    validator_consensus_json JSONB,
    dominant_bitstring VARCHAR(255),
    dominant_count INT,
    shannon_entropy FLOAT,
    entropy_percent FLOAT,
    ghz_state_probability FLOAT,
    ghz_fidelity FLOAT,
    user_signature_bit INT,
    target_signature_bit INT,
    validator_agreement_score FLOAT,
    state_hash VARCHAR(255),
    commitment_hash VARCHAR(255),
    block_hash VARCHAR(255),
    measurement_type VARCHAR(50),
    coherence_quality NUMERIC(5,4),
    state_vector_hash VARCHAR(255),
    total_shots INTEGER DEFAULT 1024,
    validator_id VARCHAR(255),
    circuit_depth INTEGER,
    execution_time_ms NUMERIC(10,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);""", "Create quantum_measurements table")

if table_exists('quantum_measurements'):
    execute_sql("CREATE INDEX IF NOT EXISTS idx_quantum_measurements_tx_id ON quantum_measurements(tx_id);", "Index quantum_measurements(tx_id)")
    if column_exists('quantum_measurements', 'entropy_percent'):
        execute_sql("CREATE INDEX IF NOT EXISTS idx_quantum_measurements_entropy ON quantum_measurements(entropy_percent);", "Index quantum_measurements(entropy_percent)")
    if column_exists('quantum_measurements', 'ghz_fidelity'):
        execute_sql("CREATE INDEX IF NOT EXISTS idx_quantum_measurements_ghz_fidelity ON quantum_measurements(ghz_fidelity);", "Index quantum_measurements(ghz_fidelity)")
    if column_exists('quantum_measurements', 'validator_agreement_score'):
        execute_sql("CREATE INDEX IF NOT EXISTS idx_quantum_measurements_validator_agreement ON quantum_measurements(validator_agreement_score);", "Index quantum_measurements(validator_agreement_score)")
    if column_exists('quantum_measurements', 'block_hash'):
        execute_sql("CREATE INDEX IF NOT EXISTS idx_quantum_measurements_block_hash ON quantum_measurements(block_hash);", "Index quantum_measurements(block_hash)")
    if column_exists('quantum_measurements', 'validator_id'):
        execute_sql("CREATE INDEX IF NOT EXISTS idx_quantum_measurements_validator_id ON quantum_measurements(validator_id);", "Index quantum_measurements(validator_id)")
    if column_exists('quantum_measurements', 'created_at'):
        execute_sql("CREATE INDEX IF NOT EXISTS idx_quantum_measurements_created_at ON quantum_measurements(created_at DESC);", "Index quantum_measurements(created_at)")
    if column_exists('quantum_measurements', 'commitment_hash'):
        execute_sql("CREATE INDEX IF NOT EXISTS idx_quantum_measurements_commitment_hash ON quantum_measurements(commitment_hash);", "Index quantum_measurements(commitment_hash)")

# SECTION 6
print("\n📝 [6/10] Transaction Sessions")
execute_sql("""CREATE TABLE IF NOT EXISTS transaction_sessions (
    session_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    transaction_state VARCHAR(50) NOT NULL,
    session_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() + INTERVAL '30 minutes'
);""", "Create transaction_sessions table (user_id as TEXT)")

if table_exists('transaction_sessions'):
    execute_sql("CREATE INDEX IF NOT EXISTS idx_transaction_sessions_user_id ON transaction_sessions(user_id);", "Index transaction_sessions(user_id)")
    execute_sql("CREATE INDEX IF NOT EXISTS idx_transaction_sessions_expires_at ON transaction_sessions(expires_at);", "Index transaction_sessions(expires_at)")

# SECTION 7
print("\n📝 [7/10] Rate Limiting")
execute_sql("""CREATE TABLE IF NOT EXISTS rate_limits (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    limit_type VARCHAR(50) NOT NULL,
    count INTEGER DEFAULT 1,
    reset_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);""", "Create rate_limits table (user_id as TEXT)")

if table_exists('rate_limits'):
    execute_sql("CREATE INDEX IF NOT EXISTS idx_rate_limits_user_type ON rate_limits(user_id, limit_type);", "Index rate_limits(user_id, limit_type)")
    execute_sql("CREATE INDEX IF NOT EXISTS idx_rate_limits_reset_at ON rate_limits(reset_at);", "Index rate_limits(reset_at)")

# SECTION 8
print("\n📝 [8/10] Receive Codes")
execute_sql("""CREATE TABLE IF NOT EXISTS receive_codes (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    code VARCHAR(16) NOT NULL UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() + INTERVAL '24 hours',
    used BOOLEAN DEFAULT FALSE
);""", "Create receive_codes table (user_id as TEXT)")

if table_exists('receive_codes'):
    execute_sql("CREATE INDEX IF NOT EXISTS idx_receive_codes_user_id ON receive_codes(user_id);", "Index receive_codes(user_id)")
    execute_sql("CREATE INDEX IF NOT EXISTS idx_receive_codes_code ON receive_codes(code);", "Index receive_codes(code)")

# SECTION 9
print("\n📝 [9/10] Block Statistics & Validation")
execute_sql("ALTER TABLE blocks ADD COLUMN IF NOT EXISTS quantum_validation_status VARCHAR(50) DEFAULT 'unvalidated';", "Add quantum_validation_status to blocks")
execute_sql("ALTER TABLE blocks ADD COLUMN IF NOT EXISTS quantum_measurements_count INTEGER DEFAULT 0;", "Add quantum_measurements_count to blocks")
execute_sql("ALTER TABLE blocks ADD COLUMN IF NOT EXISTS validated_at TIMESTAMP WITH TIME ZONE;", "Add validated_at to blocks")
execute_sql("ALTER TABLE blocks ADD COLUMN IF NOT EXISTS validation_entropy_avg NUMERIC(5,4);", "Add validation_entropy_avg to blocks")

# SECTION 10
print("\n📝 [10/10] Transactions Table W-State Columns")
execute_sql("ALTER TABLE transactions ADD COLUMN IF NOT EXISTS commitment_hash VARCHAR(255);", "Add commitment_hash to transactions")
execute_sql("ALTER TABLE transactions ADD COLUMN IF NOT EXISTS validator_agreement FLOAT DEFAULT 0.0;", "Add validator_agreement to transactions")
execute_sql("ALTER TABLE transactions ADD COLUMN IF NOT EXISTS circuit_depth INT;", "Add circuit_depth to transactions")
execute_sql("ALTER TABLE transactions ADD COLUMN IF NOT EXISTS circuit_size INT;", "Add circuit_size to transactions")
execute_sql("ALTER TABLE transactions ADD COLUMN IF NOT EXISTS ghz_fidelity FLOAT;", "Add ghz_fidelity to transactions")
execute_sql("ALTER TABLE transactions ADD COLUMN IF NOT EXISTS dominant_bitstring VARCHAR(255);", "Add dominant_bitstring to transactions")

if table_exists('transactions'):
    execute_sql("CREATE INDEX IF NOT EXISTS idx_transactions_commitment_hash ON transactions(commitment_hash);", "Index transactions(commitment_hash)")
    execute_sql("CREATE INDEX IF NOT EXISTS idx_transactions_validator_agreement ON transactions(validator_agreement DESC);", "Index transactions(validator_agreement)")
    execute_sql("CREATE INDEX IF NOT EXISTS idx_transactions_ghz_fidelity ON transactions(ghz_fidelity DESC);", "Index transactions(ghz_fidelity)")

# BONUS
print("\n📝 [BONUS] Quantum Metrics Summary Table")
execute_sql("""CREATE TABLE IF NOT EXISTS quantum_measurements_summary (
    id BIGSERIAL PRIMARY KEY,
    hour TIMESTAMP NOT NULL,
    total_transactions INT DEFAULT 0,
    avg_validator_agreement FLOAT DEFAULT 0.0,
    avg_entropy_percent FLOAT DEFAULT 0.0,
    avg_ghz_fidelity FLOAT DEFAULT 0.0,
    summary_json JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(hour)
);""", "Create quantum_measurements_summary table")

if table_exists('quantum_measurements_summary'):
    execute_sql("CREATE INDEX IF NOT EXISTS idx_quantum_summary_hour ON quantum_measurements_summary(hour DESC);", "Index quantum_measurements_summary(hour)")

print("\n📝 [BONUS] Create Monitoring View")
if table_exists('quantum_measurements') and column_exists('quantum_measurements', 'entropy_percent'):
    execute_sql("""CREATE OR REPLACE VIEW v_wsv_metrics_latest_hour AS
       SELECT 
           COUNT(*) as total_transactions,
           AVG(entropy_percent) as avg_entropy,
           MIN(entropy_percent) as min_entropy,
           MAX(entropy_percent) as max_entropy,
           AVG(validator_agreement_score) as avg_validator_agreement,
           AVG(ghz_fidelity) as avg_ghz_fidelity,
           NOW() as as_of_timestamp
       FROM quantum_measurements
       WHERE created_at > NOW() - INTERVAL '1 hour';""", "Create v_wsv_metrics_latest_hour view")
else:
    print("  ⚠ Skipping view (quantum_measurements not ready)")

# COMPLETE
print("\n" + "=" * 100)
print("✓ SCHEMA MIGRATION COMPLETED!")
print("=" * 100)

print(f"\n📊 Summary:")
print(f"  ✓ Executed: {executed}")
print(f"  ⚠ Skipped: {skipped}")
print(f"  ✗ Errors: {errors}")

if errors == 0:
    print(f"\n✨ Schema includes:")
    print(f"  • 9 new tables (with TEXT user_id)")
    print(f"  • 2 enhanced tables")
    print(f"  • 25+ performance indexes")
    print(f"  • 1 monitoring view")
    print(f"  • 1 summary table")
    print(f"\n🚀 Your system is ready for W-state quantum execution!")
    print("\n✅ All done! Your Supabase schema is ready for production.")
else:
    print(f"\n⚠️  Migration completed with {errors} error(s)")
    print(f"Please review errors above and try again if needed")

cursor.close()
conn.close()
print("✓ Disconnected")
