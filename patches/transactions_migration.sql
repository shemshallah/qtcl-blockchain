-- ═══════════════════════════════════════════════════════════════════════════
-- QTCL transactions table migration
-- Run in Supabase SQL editor: https://supabase.com/dashboard → SQL Editor
-- Safe to run multiple times — all statements are idempotent.
-- ═══════════════════════════════════════════════════════════════════════════

-- Create table if it doesn't exist at all
CREATE TABLE IF NOT EXISTS transactions (
    tx_hash           TEXT          PRIMARY KEY,
    from_address      TEXT,
    to_address        TEXT,
    amount            NUMERIC(30,0) DEFAULT 0,
    nonce             BIGINT        DEFAULT 0,
    height            INTEGER,
    block_hash        TEXT,
    transaction_index INTEGER       DEFAULT 0,
    tx_type           TEXT          DEFAULT 'transfer',
    status            TEXT          DEFAULT 'pending',
    quantum_state_hash TEXT,
    commitment_hash   TEXT,
    metadata          JSONB,
    created_at        TIMESTAMPTZ   DEFAULT NOW(),
    updated_at        TIMESTAMPTZ   DEFAULT NOW()
);

-- Add missing columns (idempotent — safe if already exist)
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS height             INTEGER;
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS block_hash         TEXT;
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS transaction_index  INTEGER     DEFAULT 0;
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS tx_type            TEXT        DEFAULT 'transfer';
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS quantum_state_hash TEXT;
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS commitment_hash    TEXT;
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS metadata           JSONB;
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS updated_at         TIMESTAMPTZ DEFAULT NOW();

-- Indexes for /api/transactions explorer queries
CREATE INDEX IF NOT EXISTS idx_tx_height      ON transactions (height DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_tx_type        ON transactions (tx_type);
CREATE INDEX IF NOT EXISTS idx_tx_from        ON transactions (from_address);
CREATE INDEX IF NOT EXISTS idx_tx_to          ON transactions (to_address);
CREATE INDEX IF NOT EXISTS idx_tx_status      ON transactions (status);
CREATE INDEX IF NOT EXISTS idx_tx_block_hash  ON transactions (block_hash);
CREATE INDEX IF NOT EXISTS idx_tx_created     ON transactions (created_at DESC);

-- Backfill tx_type for any existing rows missing it
-- Coinbase sender is '0' * 64 (server constant COINBASE_ADDRESS)
UPDATE transactions
SET tx_type = 'coinbase'
WHERE tx_type IS NULL
  AND from_address = '0000000000000000000000000000000000000000000000000000000000000000';

UPDATE transactions
SET tx_type = 'transfer'
WHERE tx_type IS NULL;

-- Confirm: show row counts by type
SELECT tx_type, status, COUNT(*) as count, MAX(height) as max_height
FROM transactions
GROUP BY tx_type, status
ORDER BY tx_type, status;
