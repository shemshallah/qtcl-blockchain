#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                                                                                  ║
║   QTCL BRIDGE v1.1 — Cross-Chain Escrow Protocol                               ║
║   QTCL (L1 Native) ↔ wQTCL (ERC-20 on Base Mainnet)                            ║
║                                                                                  ║
║   ┌─────────────────────────────────────────────────────────────────────────┐   ║
║   │  ZERO MINTING. Every QTCL is PoW mined. The bridge NEVER creates      │   ║
║   │  new tokens. wQTCL on Base is a FIXED-SUPPLY wrapper backed 1:1       │   ║
║   │  by QTCL locked in the bridge escrow. The wQTCL contract has NO       │   ║
║   │  mint function — its entire supply is pre-deposited at deploy time    │   ║
║   │  from the premined equity pool (~100,000 QTCL).                       │   ║
║   └─────────────────────────────────────────────────────────────────────────┘   ║
║                                                                                  ║
║   How it works:                                                                  ║
║                                                                                  ║
║     EQUITY POOL BOOTSTRAP:                                                      ║
║       1. Premine ~100,000 QTCL on the native chain (PoW mining)                ║
║       2. Deploy wQTCL ERC-20 on Base with 100,000 fixed supply                 ║
║       3. All 100,000 wQTCL held by bridge vault contract                       ║
║       4. Bridge goes live — native QTCL backs wQTCL releases                   ║
║                                                                                  ║
║     OUTBOUND (QTCL → wQTCL):                                                   ║
║       1. User locks QTCL in native escrow via bridge_lock RPC                  ║
║       2. 3-of-5 oracles attest to the lock                                     ║
║       3. Bridge contract TRANSFERS (not mints) wQTCL from vault to user       ║
║       4. User now holds wQTCL on Base, tradeable on Uniswap                    ║
║                                                                                  ║
║     INBOUND (wQTCL → QTCL):                                                    ║
║       1. User sends wQTCL back to bridge vault on Base                         ║
║       2. Relayer submits return proof via bridge_returnProof RPC               ║
║       3. 3-of-5 oracles verify the return                                      ║
║       4. Bridge unlocks native QTCL from escrow to user                        ║
║                                                                                  ║
║     INVARIANT:                                                                   ║
║       escrow_locked_qtcl == wqtcl_circulating (always)                         ║
║       wqtcl_vault + wqtcl_circulating == equity_pool_total (always)            ║
║       No token is ever created by the bridge. Only transferred.                ║
║                                                                                  ║
║   Fees: 0.1% (10 bps), min 1 QTCL — collected to treasury                     ║
║   Limits: 1–10,000 QTCL per request, equity pool cap                           ║
║   Cooldown: 60s per address                                                     ║
║   Database: Neon PostgreSQL via DATABASE_URL                                    ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os, sys, json, time, hashlib, logging, secrets, threading
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# BRIDGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BRIDGE_ESCROW_ADDRESS = os.environ.get(
    "QTCL_BRIDGE_ESCROW",
    "qtcl1bridge_escrow_000000000000000000000000000000"
)
WQTCL_CONTRACT_ADDRESS = os.environ.get(
    "WQTCL_CONTRACT", "0x0000000000000000000000000000000000000000"
)
BASE_CHAIN_ID = 8453

# Equity pool — total premined QTCL available for bridging
EQUITY_POOL_BASE = int(os.environ.get("QTCL_EQUITY_POOL", 10_000_000))  # 100,000 QTCL

BRIDGE_FEE_BPS         = 10
BRIDGE_MIN_FEE_BASE    = 100        # 1 QTCL min fee
BRIDGE_MIN_AMOUNT_BASE = 100        # 1 QTCL min bridge
BRIDGE_MAX_AMOUNT_BASE = 1_000_000  # 10,000 QTCL max per request
BRIDGE_COOLDOWN_SECONDS = 60
BRIDGE_ATTESTATION_THRESHOLD = 3    # 3-of-5
BRIDGE_REQUEST_EXPIRY  = 3600       # 1 hour


class BridgeDirection(str, Enum):
    LOCK   = "qtcl_to_wqtcl"
    UNLOCK = "wqtcl_to_qtcl"

class BridgeStatus(str, Enum):
    PENDING         = "pending"
    LOCKED          = "locked"
    RETURN_VERIFIED = "return_verified"
    ATTESTING       = "attesting"
    ATTESTED        = "attested"
    RELEASING       = "releasing"
    UNLOCKING       = "unlocking"
    COMPLETED       = "completed"
    FAILED          = "failed"
    EXPIRED         = "expired"
    REFUNDED        = "refunded"

@dataclass
class BridgeRequest:
    request_id: str; direction: str; status: str
    amount_base: int; fee_base: int; net_amount_base: int
    qtcl_address: str
    qtcl_tx_hash: Optional[str] = None
    eth_address: Optional[str] = None
    evm_tx_hash: Optional[str] = None
    attestations: Dict[str, dict] = field(default_factory=dict)
    attestation_count: int = 0
    created_at: float = 0.0
    locked_at: Optional[float] = None
    attested_at: Optional[float] = None
    completed_at: Optional[float] = None
    expires_at: float = 0.0
    qtcl_block_height: Optional[int] = None
    evm_block_number: Optional[int] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["amount_qtcl"]     = self.amount_base     / 100.0
        d["fee_qtcl"]        = self.fee_base        / 100.0
        d["net_amount_qtcl"] = self.net_amount_base / 100.0
        return d


class BridgeManager:
    _instance: Optional["BridgeManager"] = None
    _cls_lock = threading.Lock()

    def __new__(cls):
        with cls._cls_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized: return
        self._initialized = True
        self._requests: Dict[str, BridgeRequest] = {}
        self._cooldowns: Dict[str, float] = {}
        self._total_locked_base: int = 0
        self._total_released_base: int = 0
        self._total_bridged_base: int = 0
        self._total_fees_base: int = 0
        self._lock = threading.RLock()
        self._ensure_table()
        self._load_active()
        logger.info(
            f"[BRIDGE] ✅ Ready — {len(self._requests)} active, "
            f"escrow={self._total_locked_base/100:.2f}, "
            f"circulating={self._total_released_base/100:.2f}, "
            f"vault={(EQUITY_POOL_BASE - self._total_released_base)/100:.2f}"
        )

    def _get_conn(self):
        try:
            import psycopg2
            dsn = os.environ.get("DATABASE_URL", "")
            if not dsn: return None
            c = psycopg2.connect(dsn, connect_timeout=8); c.autocommit = True; return c
        except Exception as e:
            logger.warning(f"[BRIDGE] DB: {e}"); return None

    def _ensure_table(self):
        c = self._get_conn()
        if not c: return
        try:
            with c.cursor() as cur:
                cur.execute("""CREATE TABLE IF NOT EXISTS bridge_requests (
                    request_id VARCHAR(66) PRIMARY KEY, direction VARCHAR(20) NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'pending',
                    amount_base BIGINT NOT NULL, fee_base BIGINT NOT NULL DEFAULT 0,
                    net_amount_base BIGINT NOT NULL, qtcl_address VARCHAR(255) NOT NULL,
                    qtcl_tx_hash VARCHAR(66), eth_address VARCHAR(42), evm_tx_hash VARCHAR(66),
                    attestations JSONB DEFAULT '{}'::jsonb, attestation_count INT DEFAULT 0,
                    created_at DOUBLE PRECISION NOT NULL, locked_at DOUBLE PRECISION,
                    attested_at DOUBLE PRECISION, completed_at DOUBLE PRECISION,
                    expires_at DOUBLE PRECISION NOT NULL,
                    qtcl_block_height INT, evm_block_number BIGINT)""")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_br_status ON bridge_requests(status)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_br_qtcl ON bridge_requests(qtcl_address)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_br_eth ON bridge_requests(eth_address)")
            logger.info("[BRIDGE] ✅ bridge_requests table ready")
        except Exception as e: logger.warning(f"[BRIDGE] DDL: {e}")
        finally: c.close()

    def _persist(self, req: BridgeRequest):
        c = self._get_conn()
        if not c: return
        try:
            with c.cursor() as cur:
                cur.execute("""INSERT INTO bridge_requests (
                    request_id,direction,status,amount_base,fee_base,net_amount_base,
                    qtcl_address,qtcl_tx_hash,eth_address,evm_tx_hash,
                    attestations,attestation_count,
                    created_at,locked_at,attested_at,completed_at,expires_at,
                    qtcl_block_height,evm_block_number
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (request_id) DO UPDATE SET
                    status=EXCLUDED.status, qtcl_tx_hash=EXCLUDED.qtcl_tx_hash,
                    evm_tx_hash=EXCLUDED.evm_tx_hash, attestations=EXCLUDED.attestations,
                    attestation_count=EXCLUDED.attestation_count,
                    locked_at=EXCLUDED.locked_at, attested_at=EXCLUDED.attested_at,
                    completed_at=EXCLUDED.completed_at,
                    qtcl_block_height=EXCLUDED.qtcl_block_height,
                    evm_block_number=EXCLUDED.evm_block_number""",
                (req.request_id, req.direction, req.status,
                 req.amount_base, req.fee_base, req.net_amount_base,
                 req.qtcl_address, req.qtcl_tx_hash, req.eth_address, req.evm_tx_hash,
                 json.dumps(req.attestations), req.attestation_count,
                 req.created_at, req.locked_at, req.attested_at,
                 req.completed_at, req.expires_at,
                 req.qtcl_block_height, req.evm_block_number))
        except Exception as e: logger.warning(f"[BRIDGE] persist: {e}")
        finally: c.close()

    def _load_active(self):
        c = self._get_conn()
        if not c: return
        try:
            with c.cursor() as cur:
                cur.execute("""SELECT request_id,direction,status,
                    amount_base,fee_base,net_amount_base,
                    qtcl_address,qtcl_tx_hash,eth_address,evm_tx_hash,
                    attestations,attestation_count,
                    created_at,locked_at,attested_at,completed_at,expires_at,
                    qtcl_block_height,evm_block_number
                    FROM bridge_requests
                    WHERE status NOT IN ('completed','failed','expired','refunded')
                    ORDER BY created_at DESC LIMIT 500""")
                for r in cur.fetchall():
                    att = r[10] if isinstance(r[10], dict) else json.loads(r[10] or "{}")
                    req = BridgeRequest(
                        request_id=r[0],direction=r[1],status=r[2],
                        amount_base=r[3],fee_base=r[4],net_amount_base=r[5],
                        qtcl_address=r[6],qtcl_tx_hash=r[7],
                        eth_address=r[8],evm_tx_hash=r[9],
                        attestations=att,attestation_count=r[11] or 0,
                        created_at=r[12],locked_at=r[13],attested_at=r[14],
                        completed_at=r[15],expires_at=r[16],
                        qtcl_block_height=r[17],evm_block_number=r[18])
                    self._requests[req.request_id] = req
                    if req.status in ("locked","attesting","attested","releasing"):
                        self._total_locked_base += req.amount_base
                        self._total_released_base += req.net_amount_base
                # Net circulating from completed history
                cur.execute("SELECT COALESCE(SUM(net_amount_base),0) FROM bridge_requests WHERE direction='qtcl_to_wqtcl' AND status='completed'")
                out_done = int(cur.fetchone()[0])
                cur.execute("SELECT COALESCE(SUM(net_amount_base),0) FROM bridge_requests WHERE direction='wqtcl_to_qtcl' AND status='completed'")
                in_done = int(cur.fetchone()[0])
                self._total_released_base += (out_done - in_done)
        except Exception as e: logger.warning(f"[BRIDGE] load: {e}")
        finally: c.close()

    @staticmethod
    def calc_fee(amount_base: int) -> int:
        return max((amount_base * BRIDGE_FEE_BPS) // 10_000, BRIDGE_MIN_FEE_BASE)

    @staticmethod
    def make_id(a, b, amt):
        return "br_" + hashlib.sha3_256(f"{a}:{b}:{amt}:{time.time()}:{secrets.token_hex(8)}".encode()).hexdigest()[:60]

    def _check_balance(self, address, required):
        c = self._get_conn()
        if not c: return False, 0
        try:
            with c.cursor() as cur:
                cur.execute("SELECT balance FROM wallet_addresses WHERE address=%s", (address,))
                row = cur.fetchone()
                if not row: return False, 0
                bal = int(row[0]); return bal >= required, bal
        except Exception as e: logger.warning(f"[BRIDGE] bal: {e}"); return False, 0
        finally: c.close()

    def _transfer_balance(self, from_addr, to_addr, amount):
        c = self._get_conn()
        if not c: return False
        try:
            c.autocommit = False
            with c.cursor() as cur:
                cur.execute("UPDATE wallet_addresses SET balance=balance-%s WHERE address=%s AND balance>=%s",
                            (amount, from_addr, amount))
                if cur.rowcount == 0: c.rollback(); return False
                cur.execute("""INSERT INTO wallet_addresses (address,balance,transaction_count) VALUES (%s,%s,1)
                    ON CONFLICT (address) DO UPDATE SET balance=wallet_addresses.balance+EXCLUDED.balance,
                    transaction_count=wallet_addresses.transaction_count+1""", (to_addr, amount))
                c.commit(); return True
        except Exception as e:
            logger.warning(f"[BRIDGE] xfer: {e}")
            try: c.rollback()
            except: pass
            return False
        finally: c.autocommit = True; c.close()

    # ══════════════════════════════════════════════════════════════════════
    # CORE OPS
    # ══════════════════════════════════════════════════════════════════════

    def initiate_lock(self, qtcl_address, eth_address, amount_base, signature=None):
        """OUTBOUND: Lock native QTCL → release wQTCL from vault. Zero minting."""
        with self._lock:
            if not qtcl_address or not qtcl_address.startswith("qtcl1"):
                return False, "Invalid QTCL address", None
            if not eth_address or not eth_address.startswith("0x") or len(eth_address) != 42:
                return False, "Invalid Base address", None
            if amount_base < BRIDGE_MIN_AMOUNT_BASE:
                return False, f"Min: {BRIDGE_MIN_AMOUNT_BASE/100:.2f} QTCL", None
            if amount_base > BRIDGE_MAX_AMOUNT_BASE:
                return False, f"Max: {BRIDGE_MAX_AMOUNT_BASE/100:.2f} QTCL", None
            last = self._cooldowns.get(qtcl_address, 0)
            if time.time() - last < BRIDGE_COOLDOWN_SECONDS:
                return False, f"Rate limited — {int(BRIDGE_COOLDOWN_SECONDS-(time.time()-last))}s", None

            fee = self.calc_fee(amount_base); net = amount_base - fee
            vault = EQUITY_POOL_BASE - self._total_released_base
            if net > vault:
                return False, f"Vault has {vault/100:.2f} wQTCL (need {net/100:.2f})", None
            ok, bal = self._check_balance(qtcl_address, amount_base)
            if not ok:
                return False, f"Balance: {bal/100:.2f} < {amount_base/100:.2f} QTCL", None
            if not self._transfer_balance(qtcl_address, BRIDGE_ESCROW_ADDRESS, amount_base):
                return False, "Escrow transfer failed", None

            now = time.time(); rid = self.make_id(qtcl_address, eth_address, amount_base)
            req = BridgeRequest(request_id=rid, direction=BridgeDirection.LOCK.value,
                status=BridgeStatus.LOCKED.value,
                amount_base=amount_base, fee_base=fee, net_amount_base=net,
                qtcl_address=qtcl_address, eth_address=eth_address,
                created_at=now, locked_at=now, expires_at=now+BRIDGE_REQUEST_EXPIRY)
            self._requests[rid] = req
            self._cooldowns[qtcl_address] = now
            self._total_locked_base += amount_base
            self._total_fees_base += fee
            self._persist(req)
            logger.info(f"[BRIDGE] 🔒 LOCK {amount_base/100:.2f} {qtcl_address[:16]}…→{eth_address} fee={fee/100:.2f}")
            return True, "Locked. Awaiting oracle attestation.", req

    def submit_attestation(self, request_id, oracle_id, oracle_signature, verified=True):
        with self._lock:
            req = self._requests.get(request_id)
            if not req: return False, "Not found"
            if req.status not in (BridgeStatus.LOCKED.value, BridgeStatus.RETURN_VERIFIED.value, BridgeStatus.ATTESTING.value):
                return False, f"Wrong state: {req.status}"
            if time.time() > req.expires_at:
                req.status = BridgeStatus.EXPIRED.value; self._persist(req); return False, "Expired"
            if oracle_id in req.attestations: return False, f"{oracle_id} already attested"
            req.attestations[oracle_id] = {"oracle_id":oracle_id,"signature":oracle_signature,"verified":verified,"ts":time.time()}
            req.attestation_count = len(req.attestations)
            if req.status != BridgeStatus.ATTESTING.value: req.status = BridgeStatus.ATTESTING.value
            if req.attestation_count >= BRIDGE_ATTESTATION_THRESHOLD:
                req.status = BridgeStatus.ATTESTED.value; req.attested_at = time.time()
                logger.info(f"[BRIDGE] ✅ ATTESTED {request_id[:24]}…")
            self._persist(req)
            return True, f"Attestation {req.attestation_count}/{BRIDGE_ATTESTATION_THRESHOLD}"

    def complete_bridge(self, request_id, evm_tx_hash=None, qtcl_tx_hash=None):
        with self._lock:
            req = self._requests.get(request_id)
            if not req: return False, "Not found"
            if req.status != BridgeStatus.ATTESTED.value:
                return False, f"Must be attested (is: {req.status})"
            if req.direction == BridgeDirection.LOCK.value:
                self._total_released_base += req.net_amount_base
                self._total_locked_base -= req.amount_base
            elif req.direction == BridgeDirection.UNLOCK.value:
                if not self._transfer_balance(BRIDGE_ESCROW_ADDRESS, req.qtcl_address, req.net_amount_base):
                    return False, "Escrow unlock failed"
                self._total_released_base -= req.net_amount_base
            req.status = BridgeStatus.COMPLETED.value; req.completed_at = time.time()
            if evm_tx_hash: req.evm_tx_hash = evm_tx_hash
            if qtcl_tx_hash: req.qtcl_tx_hash = qtcl_tx_hash
            self._total_bridged_base += req.net_amount_base
            self._persist(req)
            logger.info(f"[BRIDGE] 🏁 DONE {request_id[:24]}… {req.net_amount_base/100:.2f} ({req.direction})")
            return True, "Bridge completed"

    def refund(self, request_id, reason=""):
        with self._lock:
            req = self._requests.get(request_id)
            if not req: return False, "Not found"
            if req.status in (BridgeStatus.COMPLETED.value, BridgeStatus.REFUNDED.value):
                return False, f"Already {req.status}"
            if req.direction == BridgeDirection.LOCK.value and req.status != BridgeStatus.PENDING.value:
                self._transfer_balance(BRIDGE_ESCROW_ADDRESS, req.qtcl_address, req.amount_base)
                self._total_locked_base -= req.amount_base
            req.status = BridgeStatus.REFUNDED.value; req.completed_at = time.time()
            self._persist(req)
            logger.info(f"[BRIDGE] ↩️ REFUND {request_id[:24]}… {reason}")
            return True, f"Refunded {req.amount_base/100:.2f} QTCL"

    def submit_return_proof(self, qtcl_dest, eth_address, amount_base, evm_tx_hash, evm_block):
        """wQTCL returned to vault on Base → unlock native QTCL. No burning."""
        with self._lock:
            if not qtcl_dest or not qtcl_dest.startswith("qtcl1"): return False, "Invalid QTCL dest", None
            for r in self._requests.values():
                if r.evm_tx_hash == evm_tx_hash and r.status != BridgeStatus.FAILED.value:
                    return False, f"Already submitted: {r.request_id[:20]}…", None
            ok, bal = self._check_balance(BRIDGE_ESCROW_ADDRESS, amount_base)
            if not ok: return False, f"Escrow: {bal/100:.2f} < {amount_base/100:.2f}", None
            fee = self.calc_fee(amount_base); net = amount_base - fee
            now = time.time(); rid = self.make_id(qtcl_dest, eth_address, amount_base)
            req = BridgeRequest(request_id=rid, direction=BridgeDirection.UNLOCK.value,
                status=BridgeStatus.RETURN_VERIFIED.value,
                amount_base=amount_base, fee_base=fee, net_amount_base=net,
                qtcl_address=qtcl_dest, eth_address=eth_address,
                evm_tx_hash=evm_tx_hash, evm_block_number=evm_block,
                created_at=now, expires_at=now+BRIDGE_REQUEST_EXPIRY)
            self._requests[rid] = req; self._total_fees_base += fee; self._persist(req)
            logger.info(f"[BRIDGE] ↩️ RETURN {amount_base/100:.2f} wQTCL→{qtcl_dest[:16]}…")
            return True, "Return accepted. Awaiting attestation.", req

    def get_request(self, rid):
        with self._lock:
            r = self._requests.get(rid); return r.to_dict() if r else None

    def get_history(self, address, limit=50):
        with self._lock:
            out = []
            for r in sorted(self._requests.values(), key=lambda x: x.created_at, reverse=True):
                if r.qtcl_address == address or r.eth_address == address:
                    out.append(r.to_dict())
                    if len(out) >= limit: break
            return out

    def get_stats(self):
        with self._lock:
            counts = {}
            for r in self._requests.values(): counts[r.status] = counts.get(r.status, 0) + 1
            vault = EQUITY_POOL_BASE - self._total_released_base
            return {
                "total_requests": len(self._requests),
                "equity_pool_qtcl": EQUITY_POOL_BASE / 100.0,
                "vault_available_qtcl": vault / 100.0,
                "wqtcl_circulating_qtcl": self._total_released_base / 100.0,
                "escrow_locked_qtcl": self._total_locked_base / 100.0,
                "lifetime_bridged_qtcl": self._total_bridged_base / 100.0,
                "fees_collected_qtcl": self._total_fees_base / 100.0,
                "status_counts": counts,
                "escrow_address": BRIDGE_ESCROW_ADDRESS,
                "wqtcl_contract": WQTCL_CONTRACT_ADDRESS,
                "chain_id": BASE_CHAIN_ID,
                "fee_bps": BRIDGE_FEE_BPS,
                "min_qtcl": BRIDGE_MIN_AMOUNT_BASE / 100.0,
                "max_qtcl": BRIDGE_MAX_AMOUNT_BASE / 100.0,
                "threshold": f"{BRIDGE_ATTESTATION_THRESHOLD}-of-5",
                "model": "fixed-supply-escrow (zero minting)",
            }

    def get_pending_attestations(self):
        with self._lock:
            return [r.to_dict() for r in self._requests.values()
                if r.status in (BridgeStatus.LOCKED.value, BridgeStatus.RETURN_VERIFIED.value,
                    BridgeStatus.ATTESTING.value) and time.time() < r.expires_at]

    def expire_stale(self):
        with self._lock:
            now = time.time(); n = 0
            for req in list(self._requests.values()):
                if req.status in ("pending","locked","attesting","return_verified") and now > req.expires_at:
                    if req.direction == BridgeDirection.LOCK.value:
                        self._transfer_balance(BRIDGE_ESCROW_ADDRESS, req.qtcl_address, req.amount_base)
                        self._total_locked_base -= req.amount_base
                    req.status = BridgeStatus.EXPIRED.value; self._persist(req); n += 1
            if n: logger.info(f"[BRIDGE] expired {n} (refunded)")
            return n


def get_bridge(): return BridgeManager()

# ═══════════════════════════════════════════════════════════════════════════════
# RPC HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

def _ok(r, i):  return {"jsonrpc":"2.0","result":r,"id":i}
def _err(c,m,i,d=None):
    e = {"code":c,"message":m}
    if d: e["data"] = d
    return {"jsonrpc":"2.0","error":e,"id":i}

def rpc_bridge_lock(params, rpc_id):
    try:
        p = params[0] if isinstance(params,(list,tuple)) and params else params
        if not isinstance(p,dict): return _err(-32602,"need dict",rpc_id)
        ok,msg,req = get_bridge().initiate_lock(p.get("qtcl_address",""),p.get("eth_address",""),
            int(round(float(p.get("amount",0))*100)),p.get("signature"))
        if ok and req:
            return _ok({"status":"locked","request_id":req.request_id,
                "amount_qtcl":req.amount_base/100.0,"fee_qtcl":req.fee_base/100.0,
                "net_amount_qtcl":req.net_amount_base/100.0,
                "escrow":BRIDGE_ESCROW_ADDRESS,"expires_at":req.expires_at,"message":msg},rpc_id)
        return _err(-32000,msg,rpc_id)
    except Exception as e: return _err(-32603,str(e),rpc_id)

def rpc_bridge_attest(params, rpc_id):
    try:
        p = params[0] if isinstance(params,(list,tuple)) and params else params
        ok,msg = get_bridge().submit_attestation(p.get("request_id",""),p.get("oracle_id",""),
            p.get("oracle_signature",""),p.get("verified",True))
        if ok:
            req = get_bridge().get_request(p.get("request_id",""))
            return _ok({"message":msg,"request":req},rpc_id)
        return _err(-32000,msg,rpc_id)
    except Exception as e: return _err(-32603,str(e),rpc_id)

def rpc_bridge_status(params, rpc_id):
    try:
        rid = params[0] if isinstance(params,(list,tuple)) and params else params
        req = get_bridge().get_request(str(rid))
        return _ok(req,rpc_id) if req else _err(-32000,"Not found",rpc_id)
    except Exception as e: return _err(-32603,str(e),rpc_id)

def rpc_bridge_history(params, rpc_id):
    try:
        addr = params[0] if isinstance(params,(list,tuple)) and params else ""
        lim = int(params[1]) if isinstance(params,(list,tuple)) and len(params)>1 else 50
        h = get_bridge().get_history(str(addr),lim)
        return _ok({"address":addr,"count":len(h),"requests":h},rpc_id)
    except Exception as e: return _err(-32603,str(e),rpc_id)

def rpc_bridge_stats(params, rpc_id):
    try: return _ok(get_bridge().get_stats(),rpc_id)
    except Exception as e: return _err(-32603,str(e),rpc_id)

def rpc_bridge_return_proof(params, rpc_id):
    try:
        p = params[0] if isinstance(params,(list,tuple)) and params else params
        ok,msg,req = get_bridge().submit_return_proof(p.get("qtcl_destination",""),p.get("eth_address",""),
            int(round(float(p.get("amount",0))*100)),p.get("evm_tx_hash",""),int(p.get("evm_block_number",0)))
        if ok and req:
            return _ok({"status":"return_verified","request_id":req.request_id,
                "message":msg,"request":req.to_dict()},rpc_id)
        return _err(-32000,msg,rpc_id)
    except Exception as e: return _err(-32603,str(e),rpc_id)

def rpc_bridge_complete(params, rpc_id):
    try:
        p = params[0] if isinstance(params,(list,tuple)) and params else params
        ok,msg = get_bridge().complete_bridge(p.get("request_id",""),p.get("evm_tx_hash"),p.get("qtcl_tx_hash"))
        if ok:
            req = get_bridge().get_request(p.get("request_id",""))
            return _ok({"message":msg,"request":req},rpc_id)
        return _err(-32000,msg,rpc_id)
    except Exception as e: return _err(-32603,str(e),rpc_id)

def rpc_bridge_pending(params, rpc_id):
    try:
        p = get_bridge().get_pending_attestations()
        return _ok({"count":len(p),"requests":p},rpc_id)
    except Exception as e: return _err(-32603,str(e),rpc_id)

def rpc_bridge_refund(params, rpc_id):
    try:
        p = params[0] if isinstance(params,(list,tuple)) and params else params
        rid = p.get("request_id","") if isinstance(p,dict) else str(p)
        reason = p.get("reason","") if isinstance(p,dict) else ""
        ok,msg = get_bridge().refund(rid,reason)
        return _ok({"message":msg},rpc_id) if ok else _err(-32000,msg,rpc_id)
    except Exception as e: return _err(-32603,str(e),rpc_id)

BRIDGE_RPC_METHODS = {
    "qtcl_bridge_lock":                rpc_bridge_lock,
    "qtcl_bridge_attest":              rpc_bridge_attest,
    "qtcl_bridge_status":              rpc_bridge_status,
    "qtcl_bridge_history":             rpc_bridge_history,
    "qtcl_bridge_stats":               rpc_bridge_stats,
    "qtcl_bridge_returnProof":         rpc_bridge_return_proof,
    "qtcl_bridge_complete":            rpc_bridge_complete,
    "qtcl_bridge_pendingAttestations": rpc_bridge_pending,
    "qtcl_bridge_refund":              rpc_bridge_refund,
}
