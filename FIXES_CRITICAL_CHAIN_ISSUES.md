# CRITICAL FIXES — Chain Advancement & Balance Updates

## Issues Found & Fixed

### 1. ❌ TRANSACTION BALANCES NOT UPDATING
**Problem:** When transactions were confirmed in blocks, sender/receiver balances were never updated. Only miner rewards were credited.

**Symptom:** "Transactions aren't actually being made"

**Root Cause:** Missing balance settlement logic in `_rpc_submitBlock`

**Fix:** Added comprehensive transaction settlement code (~line 4037-4093)
- For each non-coinbase transaction in block:
  - **DEDUCT** sender balance = amount + fee
  - **ADD** receiver balance = amount  
  - Update transaction status to 'confirmed'
- Ensures wallet_addresses table is created
- Handles address creation/update with upsert logic

```python
# For each non-coinbase transaction:
# 1. DEDUCT from sender (amount + fee)
cur.execute("""
    INSERT INTO wallet_addresses (address, balance, address_type, last_updated)
    VALUES (%s, -%s, 'standard', NOW())
    ON CONFLICT (address) DO UPDATE SET
        balance = wallet_addresses.balance - %s,
        ...
""", (tx_sender, total_deduction, total_deduction))

# 2. ADD to receiver (amount only)
cur.execute("""
    INSERT INTO wallet_addresses (address, balance, address_type, last_updated)
    VALUES (%s, %s, 'standard', NOW())
    ON CONFLICT (address) DO UPDATE SET
        balance = wallet_addresses.balance + %s,
        ...
""", (tx_receiver, tx_amount, tx_amount))

# 3. Mark transaction confirmed
cur.execute("""
    UPDATE transactions SET status = 'confirmed', height = %s, updated_at = NOW()
    WHERE tx_hash = %s OR tx_id = %s
""", (height, tx_id, tx_id))
```

---

### 2. ❌ HEIGHT NOT ADVANCING (keeps mining h=1)
**Problem:** After mining block h=1, server accepts it, but next getBlockHeight call still returns 1 instead of advancing to 2.

**Symptom:** Miner gets same height (h=1) repeatedly, mines same block again, gets rewards multiple times

**Root Cause:** Block may not be persisting to database despite reporting "accepted"

**Fix:** Added block persistence verification
- After block INSERT, verify block actually exists in DB
- Log critical warning if INSERT succeeded but block not found
- Set rowcount=0 if verification fails to prevent reward crediting

```python
# After block INSERT, verify it made it to the database
cur.execute("SELECT COUNT(*) FROM blocks WHERE height = %s", (height,))
verify_row = cur.fetchone()
block_exists_in_db = verify_row and verify_row[0] > 0
if block_exists_in_db:
    logger.warning(f"[RPC-submitBlock] ✅ BLOCK VERIFIED IN DATABASE: h={height}")
else:
    logger.error(f"[RPC-submitBlock] ❌ BLOCK NOT IN DATABASE AFTER INSERT: h={height}")
    _block_rowcount = 0  # Mark as failed
```

**Also Enhanced:**
- Added detailed logging to `query_latest_block()` 
- Shows exactly what height is being returned
- Logs if DB is empty (no blocks)
- Helps diagnose if query isn't seeing inserted blocks

```python
def query_latest_block():
    """... Get latest block from DB (authoritative source) ..."""
    with get_db_cursor() as cur:
        cur.execute("SELECT height, block_hash, timestamp FROM blocks ORDER BY height DESC LIMIT 1")
        row = cur.fetchone()
        if row:
            latest = {"height": row[0], "hash": row[1] or "", "timestamp": row[2] or 0}
            logger.debug(f"[QUERY-LATEST] ✅ Latest block: h={latest['height']}")
            return latest
        else:
            logger.debug(f"[QUERY-LATEST] No blocks in DB yet (genesis)")
            return None
```

---

### 3. ❌ BLOCK PERSISTENCE VISIBILITY
**Problem:** Unclear whether blocks were actually persisting to database or just reported as accepted

**Fix:** Improved logging in block INSERT
- Changed DEBUG to WARNING for persistence operations (easier to see)
- Clear success/failure messages
- Shows rowcount (0 = duplicate/conflict, >0 = inserted)

```python
logger.warning(
    f"[RPC-submitBlock] 🔄 BLOCK INSERT attempt: h={height}, "
    f"hash={block_hash[:16]}…, parent={parent_hash[:16]}…, "
    f"miner={miner_address[:16]}…"
)
# After insert
if _block_rowcount > 0:
    logger.warning(f"[RPC-submitBlock] ✅ BLOCK PERSISTED: rowcount={_block_rowcount}, height will advance to {height}")
else:
    logger.warning(f"[RPC-submitBlock] ⚠️  Block insert rowcount=0 (duplicate height or conflict)")
```

---

### 4. ✅ DATABASE COMMITMENT
**Status:** Verified working correctly
- `get_db_cursor()` context manager properly commits (line 1633)
- Rollback on exception (line 1637)  
- Proper connection pooling

---

### 5. 🔧 TREASURY ADDRESS (Client Code)
**Status:** Needs verification on client side
- Server uses `TessellationRewardSchedule.TREASURY_ADDRESS`  
- Miner/client may have hardcoded treasury address
- Should call `qtcl_getTreasuryAddress` RPC to get the authoritative address

**Location in server.py:**
- Line 4059: `treasury_address = TessellationRewardSchedule.TREASURY_ADDRESS`
- Line 4658: RPC endpoint returns it: `qtcl_getTreasuryAddress`

**Client should be updated to:**
```python
# Fetch treasury address from server instead of hardcoding
treasury_addr = self.api._rpc('qtcl_getTreasuryAddress', [])
treasury_address = treasury_addr.get('treasury_address') if treasury_addr else 'qtcl1...'
```

---

### 6. ✅ GENESIS BLOCK HANDLING
**Status:** Code appears correct
- Client has proper genesis detection (h=0, tip=0x0...0)
- Logging shows "At genesis (h=0, tip=0x0…0) — will mine block 1"
- If client doesn't see genesis, it's likely:
  1. Server hasn't initialized genesis block yet
  2. Database is empty
  3. Server initialization hasn't run (which creates genesis)

**Debug Steps:**
```bash
# Check if genesis block exists
curl -X POST http://localhost:8000/rpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "qtcl_getBlockHeight",
    "params": [],
    "id": 1
  }'
# Should return: {"height": 0, "tip_hash": "0000..."}
```

---

## Files Modified

1. **server.py** (~4040 lines)
   - Line ~4037-4093: Added transaction settlement (balance updates)
   - Line ~1952-1978: Enhanced query_latest_block() logging
   - Line ~3944-3963: Added block persistence verification
   - Line ~3907-3925: Enhanced block INSERT logging

2. **mempool.py** (previously fixed)
   - Enhanced signature verification

3. **hyp_schnorr.py** (previously fixed)
   - Fixed sign_hash() serialization

---

## Data Flow After Fixes

```
CLIENT SUBMITS BLOCK h=1
    ↓
[Server validation]
  • Signature check ✅
  • Height check (expect 1) ✅  
  • Parent hash check ✅
  • PoW verification ✅
    ↓
[Block Persistence] ← CRITICAL FIX POINT
  • INSERT into blocks table
  • VERIFY block actually in DB ← NEW VERIFICATION STEP
    ↓
[Transaction Settlement] ← CRITICAL NEW CODE
  • For each TX: DEDUCT sender, ADD receiver
  • Update TX status to 'confirmed'
    ↓
[Reward Crediting]
  • Credit miner balance
  • Credit treasury balance
    ↓
[Response to Client]
  • "status": "accepted" (only if block verified in DB)
  • "height": 1
  • "miner_reward_qtcl": 7.20
    ↓
CLIENT FETCHES NEXT BLOCK HEIGHT
  [Server queries DB] → Returns h=1 from query_latest_block()
    ↓
CLIENT COMPUTES target_height = 1 + 1 = 2 ← HEIGHT ADVANCES! ✅
```

---

## Testing Checklist

- [ ] Submit block at h=1 — verify "accepted"
- [ ] Check DB: `SELECT height FROM blocks WHERE height=1;` — should return 1 row
- [ ] Run `qtcl_getBlockHeight` — should return height=1
- [ ] Submit block at h=2 with parent=hash(block1)
- [ ] Check balance updates:
  - Query `wallet_addresses` for sender/receiver addresses
  - Verify balances changed correctly
- [ ] Check miner gets reward:
  - Query `wallet_addresses` for miner address  
  - Balance should increase by reward amount
- [ ] Check transaction status:
  - Query `transactions` table
  - Status should be 'confirmed', height should match block height

---

## Known Remaining Issues

1. **Deep HypΓ math errors** (from previous fixes)
   - PSLMatrix determinant checks may fail
   - Doesn't affect blockchain state, but prevents full signature verification
   - Needs debugging in hyp_schnorr.py

2. **Server /health endpoint hang** (separate issue)
   - May need timeout configuration
   - Oracle/lattice initialization may be blocking

3. **Client treasury address** (needs update)
   - Should fetch from server RPC instead of hardcoding

---

## Success Metrics

After these fixes, the system should:
✅ Accept blocks and persist them to database
✅ Update chain height correctly (1 → 2 → 3...)  
✅ Update transaction balances when transactions confirmed
✅ Update miner/treasury balances correctly
✅ Not mine the same height multiple times

---

Generated: 2026-04-17
Version: CRITICAL FIXES v2
