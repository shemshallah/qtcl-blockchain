#!/usr/bin/env python3
"""
init_schema.py
Create required database tables if they don't exist
Run this once after deploying to Koyeb
"""

import os
import sys
import logging
import psycopg2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get database credentials from environment
SUPABASE_HOST = os.getenv('SUPABASE_HOST')
SUPABASE_USER = os.getenv('SUPABASE_USER')
SUPABASE_PASSWORD = os.getenv('SUPABASE_PASSWORD')
SUPABASE_PORT = int(os.getenv('SUPABASE_PORT', '5432'))
SUPABASE_DB = os.getenv('SUPABASE_DB')

if not all([SUPABASE_HOST, SUPABASE_USER, SUPABASE_PASSWORD]):
    print("ERROR: Set environment variables first")
    sys.exit(1)

# Connect to database
conn = psycopg2.connect(
    host=SUPABASE_HOST,
    user=SUPABASE_USER,
    password=SUPABASE_PASSWORD,
    port=SUPABASE_PORT,
    database=SUPABASE_DB
)

cur = conn.cursor()

# Create tables
tables = [
    """
    CREATE TABLE IF NOT EXISTS transactions (
        tx_id VARCHAR(255) PRIMARY KEY,
        from_user_id VARCHAR(255) NOT NULL,
        to_user_id VARCHAR(255) NOT NULL,
        amount NUMERIC(20, 8) NOT NULL,
        tx_type VARCHAR(50) NOT NULL,
        status VARCHAR(50) DEFAULT 'pending',
        quantum_state_hash VARCHAR(255),
        entropy_score NUMERIC(5, 2),
        execution_time_ms NUMERIC(10, 2),
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW(),
        metadata JSONB DEFAULT '{}'::jsonb
    )
    """,
    
    """
    CREATE TABLE IF NOT EXISTS blocks (
        block_number BIGINT PRIMARY KEY,
        block_hash VARCHAR(255) NOT NULL,
        parent_hash VARCHAR(255),
        timestamp TIMESTAMP DEFAULT NOW(),
        miner_id VARCHAR(255),
        difficulty BIGINT,
        total_difficulty BIGINT,
        gas_used BIGINT,
        gas_limit BIGINT,
        transaction_count INT,
        merkle_root VARCHAR(255),
        state_root VARCHAR(255),
        receipts_root VARCHAR(255),
        entropy_score NUMERIC(5, 2)
    )
    """,
    
    """
    CREATE TABLE IF NOT EXISTS users (
        id VARCHAR(255) PRIMARY KEY,
        username VARCHAR(255) UNIQUE,
        email VARCHAR(255),
        balance NUMERIC(20, 8) DEFAULT 0,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    )
    """,
    
    """
    CREATE TABLE IF NOT EXISTS pseudoqubits (
        id INT PRIMARY KEY,
        qubit_state VARCHAR(255),
        fidelity NUMERIC(5, 4),
        coherence NUMERIC(5, 4),
        purity NUMERIC(5, 4),
        entropy NUMERIC(5, 4),
        concurrence NUMERIC(5, 4),
        last_measurement_at TIMESTAMP DEFAULT NOW(),
        created_at TIMESTAMP DEFAULT NOW()
    )
    """,
    
    """
    CREATE TABLE IF NOT EXISTS quantum_measurements (
        id SERIAL PRIMARY KEY,
        tx_id VARCHAR(255) REFERENCES transactions(tx_id),
        bitstring_counts JSONB,
        entropy_percent NUMERIC(5, 2),
        dominant_states VARCHAR(255)[],
        measured_at TIMESTAMP DEFAULT NOW()
    )
    """,
]

for i, sql in enumerate(tables, 1):
    try:
        cur.execute(sql)
        conn.commit()
        logger.info(f"✓ Table {i}/{len(tables)} created/verified")
    except Exception as e:
        logger.error(f"✗ Table {i} error: {e}")

# Create indexes for performance
indexes = [
    "CREATE INDEX IF NOT EXISTS idx_tx_status ON transactions(status)",
    "CREATE INDEX IF NOT EXISTS idx_tx_created ON transactions(created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_tx_user ON transactions(from_user_id, to_user_id)",
    "CREATE INDEX IF NOT EXISTS idx_blocks_number ON blocks(block_number DESC)",
]

for i, sql in enumerate(indexes, 1):
    try:
        cur.execute(sql)
        conn.commit()
        logger.info(f"✓ Index {i}/{len(indexes)} created")
    except Exception as e:
        logger.error(f"✗ Index {i} error: {e}")

cur.close()
conn.close()

logger.info("✓ Schema initialization complete")
print("\nDatabase is ready. You can now deploy to Koyeb.")
