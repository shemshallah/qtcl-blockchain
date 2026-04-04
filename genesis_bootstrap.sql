-- ╔══════════════════════════════════════════════════════════════════════════════════╗
-- ║  QTCL Genesis Block Bootstrap SQL                                                ║
-- ║                                                                                  ║
--  Run this ONCE on a fresh/empty Supabase database to seed the genesis block.       ║
--  Without this, miners see: "tip_hash is null at h=0 — refusing to mine"            ║
--                                                                                  ║
--  Usage:                                                                            ║
--    1. Supabase Dashboard → SQL Editor → paste entire file → Run                    ║
--    2. Or via CLI: psql "$POOLER_URL" -f genesis_bootstrap.sql                     ║
-- ╚══════════════════════════════════════════════════════════════════════════════════╝

-- ── Genesis constants ─────────────────────────────────────────────────────────────
-- Hash: SHA3-256 of canonical genesis block dict (sort_keys, compact separators)
-- Timestamp: 1700000000 (Nov 14 2023 — fixed epoch, deterministic)
-- Coinbase: 64 hex zeros (unspendable null address)
-- Difficulty: 4 leading hex zeros

-- ── 0. Fix existing genesis if it was created with wrong difficulty ───────────────
-- If a genesis block already exists but has difficulty=1 (old bug), patch it.
UPDATE blocks SET
    difficulty = 4,
    block_hash = '033df1b9c511248905e837af683e8192f4987167531327ab48a7559eed0b4edc'
WHERE height = 0 AND difficulty != 4;

-- ── 1. Insert genesis block ──────────────────────────────────────────────────────
-- Matches server.py INSERT INTO blocks (height, block_number, block_hash, ...)
-- and the canonical hash computed by _ensure_genesis_block_in_db() in server_backup.py

INSERT INTO blocks (
    height,
    block_number,
    block_hash,
    previous_hash,
    timestamp,
    oracle_w_state_hash,
    validator_public_key,
    nonce,
    difficulty,
    entropy_score,
    transactions_root
) VALUES (
    0,                                                    -- height
    0,                                                    -- block_number
    '033df1b9c511248905e837af683e8192f4987167531327ab48a7559eed0b4edc',  -- block_hash (SHA3-256 canonical, difficulty=4)
    '0000000000000000000000000000000000000000000000000000000000000000',  -- previous_hash (null parent)
    1700000000,                                           -- timestamp_s
    '0000000000000000000000000000000000000000000000000000000000000000',  -- oracle_w_state_hash
    '0000000000000000000000000000000000000000000000000000000000000000',  -- validator_public_key (null)
    0,                                                    -- nonce
    4,                                                    -- difficulty (4 leading hex zeros)
    0.9,                                                  -- entropy_score
    '0000000000000000000000000000000000000000000000000000000000000000'   -- transactions_root (merkle)
) ON CONFLICT (height) DO NOTHING;

-- ── 2. Update chain_state to point at genesis ────────────────────────────────────
-- Matches server.py INSERT INTO chain_state (...)

INSERT INTO chain_state (
    state_id,
    chain_height,
    head_block_hash,
    latest_coherence,
    updated_at
) VALUES (
    1,
    0,
    '033df1b9c511248905e837af683e8192f4987167531327ab48a7559eed0b4edc',
    0.9,
    NOW()
) ON CONFLICT (state_id) DO UPDATE SET
    chain_height     = EXCLUDED.chain_height,
    head_block_hash  = EXCLUDED.head_block_hash,
    latest_coherence = EXCLUDED.latest_coherence,
    updated_at       = NOW();

-- ── 3. Verify ────────────────────────────────────────────────────────────────────
-- After running, you should see:
--   blocks:        1 row (height=0, difficulty=4)
--   chain_state:   1 row (chain_height=0)

SELECT height, block_hash, difficulty FROM blocks WHERE height = 0;
SELECT chain_height, head_block_hash FROM chain_state WHERE state_id = 1;
