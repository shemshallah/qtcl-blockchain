-- Quick fix: patch existing genesis block difficulty from 1 → 4
-- Run this if you already ran the old genesis_bootstrap.sql
-- This updates difficulty AND the block_hash (which changes with difficulty)

UPDATE blocks SET
    difficulty = 4,
    block_hash = '033df1b9c511248905e837af683e8192f4987167531327ab48a7559eed0b4edc'
WHERE height = 0;

UPDATE chain_state SET
    head_block_hash = '033df1b9c511248905e837af683e8192f4987167531327ab48a7559eed0b4edc'
WHERE state_id = 1;

-- Verify
SELECT height, difficulty, block_hash FROM blocks WHERE height = 0;
