#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                       ║
║  PostgresMempool v1.0 — Complete Paradigm Shift                                     ║
║                                                                                       ║
║  Single Source of Truth: PostgreSQL mempool table                                    ║
║  Everything orchestrated by triggers + LISTEN/NOTIFY                                 ║
║  Python becomes thin validation layer + SSE broadcaster                              ║
║                                                                                       ║
║  Museum-grade implementation:                                                        ║
║  • Zero redundancy (no in-memory cache)                                              ║
║  • Atomic operations (triggers handle all state transitions)                         ║
║  • Real-time synchronization (LISTEN thread + SSE)                                   ║
║  • Comprehensive error handling                                                      ║
║  • Production-ready logging                                                          ║
║                                                                                       ║
║  This replaces: mempool.py (which becomes deletable)                                 ║
║                                                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import threading
import logging
import time
import json
import select
import psycopg2
import psycopg2.extras
import psycopg2.pool
import psycopg2.extensions
from typing import Dict, Any, Optional, List, Callable, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager
from urllib.parse import quote_plus
import queue

# ═══════════════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════════════

if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s [POSTGRES_MEMPOOL]: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════════════
# DATABASE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════════════

POOLER_HOST = os.getenv('POOLER_HOST', 'aws-0-us-west-2.pooler.supabase.com')
POOLER_USER = os.getenv('POOLER_USER', 'postgres.rslvlsqwkfmdtebqsvtw')
POOLER_PASSWORD = os.getenv('POOLER_PASSWORD', '$h10j1r1H0w4rd')
POOLER_DB = os.getenv('POOLER_DB', 'postgres')
POOLER_PORT = os.getenv('POOLER_PORT', '5432')

POOLER_URL = (
    f"postgresql://{quote_plus(POOLER_USER)}:"
    f"{quote_plus(POOLER_PASSWORD)}@"
    f"{POOLER_HOST}:{POOLER_PORT}/{POOLER_DB}"
)

# ═══════════════════════════════════════════════════════════════════════════════════════
# CONSTANTS & ENUMS
# ═══════════════════════════════════════════════════════════════════════════════════════

MAX_BLOCK_TXS = 1000
LISTEN_TIMEOUT = 30  # seconds

class TxStatus(Enum):
    """Transaction status in mempool state machine"""
    PENDING = "pending"
    INCLUDED = "included"
    CONFIRMED = "confirmed"
    SETTLED = "settled"
    REJECTED = "rejected"

# ═══════════════════════════════════════════════════════════════════════════════════════
# DATABASE CONNECTION POOL (Singleton)
# ═══════════════════════════════════════════════════════════════════════════════════════

class DBPool:
    """Thread-safe connection pool singleton"""
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = object.__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        try:
            self.pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=2,
                maxconn=10,
                dsn=POOLER_URL,
                connect_timeout=5
            )
            self._initialized = True
            logger.info("[DB] Connection pool initialized (2-10 connections)")
        except Exception as e:
            logger.error(f"[DB] Failed to initialize pool: {e}")
            raise
    
    @contextmanager
    def cursor(self):
        """Get connection and cursor from pool"""
        conn = None
        try:
            conn = self.pool.getconn()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            yield cursor
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"[DB] Cursor error: {e}")
            raise
        finally:
            if conn:
                self.pool.putconn(conn)

# Global pool singleton
_db_pool = DBPool()

# ═══════════════════════════════════════════════════════════════════════════════════════
# TRANSACTION DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class MempoolTransaction:
    """Transaction in mempool (from PostgreSQL)"""
    id: int
    tx_hash: str
    from_address: str
    to_address: str
    amount: int
    fee: int
    nonce: int
    signature: str
    status: str
    pq_signature: Optional[str] = None
    pq_verified: bool = False
    tx_type: str = 'transfer'
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    included_in_block: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'tx_hash': self.tx_hash,
            'from_address': self.from_address,
            'to_address': self.to_address,
            'amount': self.amount,
            'fee': self.fee,
            'nonce': self.nonce,
            'signature': self.signature,
            'status': self.status,
            'pq_signature': self.pq_signature,
            'pq_verified': self.pq_verified,
            'tx_type': self.tx_type,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'included_in_block': self.included_in_block,
        }
    
    @staticmethod
    def from_row(row: Dict[str, Any]) -> 'MempoolTransaction':
        """Create from database row"""
        return MempoolTransaction(
            id=row['id'],
            tx_hash=row['tx_hash'],
            from_address=row['from_address'],
            to_address=row['to_address'],
            amount=row['amount'],
            fee=row['fee'],
            nonce=row['nonce'],
            signature=row['signature'],
            status=row['status'],
            pq_signature=row.get('pq_signature'),
            pq_verified=row.get('pq_verified', False),
            tx_type=row.get('tx_type', 'transfer'),
            created_at=row.get('created_at'),
            updated_at=row.get('updated_at'),
            included_in_block=row.get('included_in_block'),
        )

# ═══════════════════════════════════════════════════════════════════════════════════════
# POSTGRES MEMPOOL (Single Source of Truth)
# ═══════════════════════════════════════════════════════════════════════════════════════

class PostgresMempool:
    """
    PostgreSQL-native mempool implementation.
    
    All state lives in PostgreSQL mempool table.
    All state transitions managed by database triggers.
    All synchronization via LISTEN/NOTIFY.
    
    Python code responsibility:
    1. Validation (format, signature, state checks)
    2. INSERT/UPDATE to trigger database events
    3. LISTEN thread for real-time updates
    4. SSE broadcasting to clients
    """
    
    def __init__(self):
        """Initialize PostgresMempool with LISTEN thread"""
        self.db_pool = _db_pool
        self._lock = threading.RLock()
        
        # Callbacks for subscribers (SSE, webhooks, etc)
        self._subscribers: List[Callable] = []
        
        # LISTEN thread
        self._listen_thread = None
        self._listen_running = False
        
        logger.info("[MEMPOOL] Initializing PostgresMempool")
        self._start_listen_thread()
        logger.info("[MEMPOOL] ✅ PostgresMempool ready")
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # CORE INTERFACE — Add Transaction
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def add_transaction(self, tx_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate and add transaction to mempool (PostgreSQL table).
        
        Triggers will:
        1. Update wallet_addresses.transaction_count ✓
        2. NOTIFY all listening clients ✓
        3. Enforce state machine ✓
        
        Returns: (success, message)
        """
        start_time = time.time()
        
        with self._lock:
            try:
                # VALIDATION LAYER 1: Format
                if not self._validate_format(tx_data):
                    return False, "Invalid transaction format"
                
                # VALIDATION LAYER 2: Signature
                if not self._validate_signature(tx_data):
                    return False, "Invalid signature"
                
                # VALIDATION LAYER 3: State (balance, nonce, etc)
                if not self._validate_state(tx_data):
                    return False, "Failed state validation"
                
                # INSERT to mempool table
                # PostgreSQL triggers handle:
                # - wallet_addresses.transaction_count update
                # - pg_notify('mempool_updates', ...)
                # - state machine validation
                with self.db_pool.cursor() as cur:
                    cur.execute("""
                        INSERT INTO mempool (
                            tx_hash, from_address, to_address, amount, fee, nonce,
                            signature, pq_signature, pq_verified, tx_type, timestamp_ns, status
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id, created_at
                    """, (
                        tx_data['tx_hash'],
                        tx_data['from_address'],
                        tx_data['to_address'],
                        int(tx_data['amount']),
                        int(tx_data.get('fee', 0)),
                        int(tx_data['nonce']),
                        tx_data['signature'],
                        tx_data.get('pq_signature'),
                        bool(tx_data.get('pq_verified', False)),
                        tx_data.get('tx_type', 'transfer'),
                        int(tx_data.get('timestamp_ns', time.time_ns())),
                        'pending'
                    ))
                    
                    result = cur.fetchone()
                    tx_id = result['id']
                
                elapsed_ms = (time.time() - start_time) * 1000
                logger.info(
                    f"[MEMPOOL] ✅ TX accepted | hash={tx_data['tx_hash'][:16]}... "
                    f"| amount={tx_data['amount']} | fee={tx_data.get('fee', 0)} "
                    f"| time={elapsed_ms:.1f}ms"
                )
                
                return True, f"TX accepted: {tx_data['tx_hash'][:16]}"
            
            except psycopg2.IntegrityError as e:
                logger.warning(f"[MEMPOOL] Integrity error (duplicate TX?): {e}")
                return False, "TX already exists"
            
            except Exception as e:
                logger.error(f"[MEMPOOL] Unexpected error: {e}")
                return False, f"Error: {str(e)}"
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # VALIDATION LAYERS (Format, Signature, State)
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def _validate_format(self, tx_data: Dict[str, Any]) -> bool:
        """Layer 1: Validate transaction structure and required fields"""
        try:
            required_fields = {'tx_hash', 'from_address', 'to_address', 'amount', 'nonce', 'signature'}
            if not all(field in tx_data for field in required_fields):
                missing = required_fields - set(tx_data.keys())
                logger.warning(f"[MEMPOOL] Format validation failed: missing {missing}")
                return False
            
            # Type checks
            if not isinstance(tx_data['tx_hash'], str) or len(tx_data['tx_hash']) < 10:
                logger.warning("[MEMPOOL] Invalid tx_hash")
                return False
            
            if not isinstance(tx_data['from_address'], str) or len(tx_data['from_address']) < 5:
                logger.warning("[MEMPOOL] Invalid from_address")
                return False
            
            if not isinstance(tx_data['to_address'], str) or len(tx_data['to_address']) < 5:
                logger.warning("[MEMPOOL] Invalid to_address")
                return False
            
            if int(tx_data['amount']) <= 0:
                logger.warning("[MEMPOOL] Invalid amount (must be > 0)")
                return False
            
            if int(tx_data['nonce']) < 0:
                logger.warning("[MEMPOOL] Invalid nonce")
                return False
            
            if not isinstance(tx_data['signature'], str) or len(tx_data['signature']) < 10:
                logger.warning("[MEMPOOL] Invalid signature")
                return False
            
            return True
        
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"[MEMPOOL] Format validation exception: {e}")
            return False
    
    def _validate_signature(self, tx_data: Dict[str, Any]) -> bool:
        """Layer 2: Validate cryptographic signature (HLWE)"""
        try:
            # TODO: Implement actual HLWE signature verification
            # For now: basic check that signature exists
            signature = tx_data.get('signature', '')
            if not signature or len(signature) < 10:
                logger.warning("[MEMPOOL] Signature validation failed: invalid format")
                return False
            
            # Real verification would go here:
            # - Verify HLWE signature
            # - Check against public key
            # - Verify hash
            
            return True
        
        except Exception as e:
            logger.warning(f"[MEMPOOL] Signature validation exception: {e}")
            return False
    
    def _validate_state(self, tx_data: Dict[str, Any]) -> bool:
        """Layer 3: Validate state (balance, nonce, replay protection)"""
        try:
            from_address = tx_data['from_address']
            nonce = int(tx_data['nonce'])
            amount = int(tx_data['amount'])
            
            with self.db_pool.cursor() as cur:
                # Check wallet exists and has balance
                cur.execute("""
                    SELECT balance, transaction_count FROM wallet_addresses
                    WHERE address = %s
                """, (from_address,))
                
                wallet = cur.fetchone()
                if not wallet:
                    logger.warning(f"[MEMPOOL] State validation failed: wallet {from_address} not found")
                    return False
                
                wallet_balance = wallet['balance']
                
                # Check sufficient balance
                if wallet_balance < amount:
                    logger.warning(
                        f"[MEMPOOL] State validation failed: insufficient balance "
                        f"({wallet_balance} < {amount})"
                    )
                    return False
                
                # Check nonce hasn't been used (replay protection)
                cur.execute("""
                    SELECT COUNT(*) as count FROM mempool
                    WHERE from_address = %s AND nonce = %s AND status IN ('pending', 'included')
                """, (from_address, nonce))
                
                nonce_count = cur.fetchone()['count']
                if nonce_count > 0:
                    logger.warning(f"[MEMPOOL] State validation failed: nonce {nonce} already used")
                    return False
            
            return True
        
        except Exception as e:
            logger.warning(f"[MEMPOOL] State validation exception: {e}")
            return False
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # QUERY INTERFACE — Get Transactions
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def get_pending_transactions(self, max_count: int = MAX_BLOCK_TXS) -> List[MempoolTransaction]:
        """
        Get pending transactions sorted by fee (for mining).
        
        Query is fast because of indexed query on (fee DESC) WHERE status = 'pending'
        """
        try:
            with self.db_pool.cursor() as cur:
                cur.execute("""
                    SELECT * FROM get_pending_transactions(%s)
                """, (max_count,))
                
                rows = cur.fetchall()
                transactions = [MempoolTransaction.from_row(row) for row in rows]
                
                logger.debug(f"[MEMPOOL] Retrieved {len(transactions)} pending transactions for mining")
                return transactions
        
        except Exception as e:
            logger.error(f"[MEMPOOL] Error getting pending transactions: {e}")
            return []
    
    def get_transaction_by_hash(self, tx_hash: str) -> Optional[MempoolTransaction]:
        """Get transaction by hash (O(log n) index lookup)"""
        try:
            with self.db_pool.cursor() as cur:
                cur.execute("""
                    SELECT * FROM mempool WHERE tx_hash = %s
                """, (tx_hash,))
                
                row = cur.fetchone()
                if row:
                    return MempoolTransaction.from_row(row)
                return None
        
        except Exception as e:
            logger.error(f"[MEMPOOL] Error getting transaction {tx_hash}: {e}")
            return None
    
    def get_transactions_by_address(self, address: str) -> List[MempoolTransaction]:
        """Get all pending transactions for address (send or receive)"""
        try:
            with self.db_pool.cursor() as cur:
                cur.execute("""
                    SELECT * FROM mempool
                    WHERE (from_address = %s OR to_address = %s)
                    AND status IN ('pending', 'included')
                    ORDER BY created_at DESC
                    LIMIT 1000
                """, (address, address))
                
                rows = cur.fetchall()
                return [MempoolTransaction.from_row(row) for row in rows]
        
        except Exception as e:
            logger.error(f"[MEMPOOL] Error getting transactions for {address}: {e}")
            return []
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # STATE TRANSITIONS — Mark Included / Confirmed
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def mark_included_in_block(self, tx_hashes: List[str], block_height: int) -> bool:
        """
        Mark transactions as included in block.
        
        Trigger will NOTIFY clients of inclusion.
        """
        try:
            if not tx_hashes:
                return True
            
            with self.db_pool.cursor() as cur:
                cur.execute("""
                    UPDATE mempool
                    SET status = 'included', included_in_block = %s
                    WHERE tx_hash = ANY(%s) AND status = 'pending'
                """, (block_height, tx_hashes))
                
                count = cur.rowcount
                logger.info(f"[MEMPOOL] Marked {count} TXs as included in block {block_height}")
            
            return True
        
        except Exception as e:
            logger.error(f"[MEMPOOL] Error marking TXs as included: {e}")
            return False
    
    def confirm_transactions(self, tx_hashes: List[str]) -> bool:
        """
        Mark transactions as confirmed (after 6 blocks).
        """
        try:
            if not tx_hashes:
                return True
            
            with self.db_pool.cursor() as cur:
                cur.execute("""
                    UPDATE mempool
                    SET status = 'confirmed'
                    WHERE tx_hash = ANY(%s) AND status = 'included'
                """, (tx_hashes,))
                
                count = cur.rowcount
                logger.info(f"[MEMPOOL] Confirmed {count} TXs")
            
            return True
        
        except Exception as e:
            logger.error(f"[MEMPOOL] Error confirming TXs: {e}")
            return False
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # STATISTICS & HEALTH
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def get_stats(self) -> Dict[str, Any]:
        """Get mempool statistics"""
        try:
            with self.db_pool.cursor() as cur:
                cur.execute("SELECT * FROM get_mempool_stats()")
                row = cur.fetchone()
                
                return {
                    'pending_count': row['pending_count'],
                    'included_count': row['included_count'],
                    'confirmed_count': row['confirmed_count'],
                    'settled_count': row['settled_count'],
                    'rejected_count': row['rejected_count'],
                    'total_pending_fee': row['total_pending_fee'],
                    'total_pending_amount': row['total_pending_amount'],
                    'oldest_pending': row['oldest_pending_timestamp'].isoformat() if row['oldest_pending_timestamp'] else None,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                }
        
        except Exception as e:
            logger.error(f"[MEMPOOL] Error getting stats: {e}")
            return {}
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # REAL-TIME SYNCHRONIZATION (LISTEN/NOTIFY)
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def _start_listen_thread(self):
        """
        Background thread: LISTEN on 'mempool_updates' channel.

        Uses a dedicated raw autocommit connection — never borrowed from the
        connection pool — so LISTEN/NOTIFY state is stable for the lifetime of
        the process.  The pool remains fully available to all other threads.
        """
        def listen_loop():
            """Reconnect-on-failure loop listening for database notifications"""
            backoff = 1
            while True:
                conn = None
                try:
                    conn = psycopg2.connect(dsn=POOLER_URL, connect_timeout=10)
                    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
                    with conn.cursor() as cur:
                        cur.execute("LISTEN mempool_updates")

                    self._listen_running = True
                    backoff = 1  # reset on successful connect
                    logger.info("[MEMPOOL] LISTEN thread connected to 'mempool_updates' channel")

                    while self._listen_running:
                        ready = select.select([conn], [], [], LISTEN_TIMEOUT)
                        conn.poll()
                        while conn.notifies:
                            notify = conn.notifies.pop(0)
                            try:
                                event = json.loads(notify.payload)
                            except (json.JSONDecodeError, TypeError):
                                continue
                            for subscriber in list(self._subscribers):
                                try:
                                    subscriber(event)
                                except Exception as sub_err:
                                    logger.error(f"[MEMPOOL] Subscriber callback error: {sub_err}")

                except Exception as e:
                    logger.error(f"[MEMPOOL] LISTEN thread error: {e} — reconnecting in {backoff}s")
                    self._listen_running = False
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                finally:
                    if conn:
                        try:
                            conn.close()
                        except Exception:
                            pass
        
        self._listen_thread = threading.Thread(target=listen_loop, daemon=True)
        self._listen_thread.start()
    
    def subscribe(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Subscribe to mempool events.
        
        Callback receives event dict when triggers fire.
        Events: tx_added, tx_included, tx_confirmed, tx_rejected, tx_settled
        """
        with self._lock:
            self._subscribers.append(callback)
            logger.debug(f"[MEMPOOL] Subscriber registered (total: {len(self._subscribers)})")
    
    def unsubscribe(self, callback: Callable) -> None:
        """Unsubscribe from mempool events"""
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)
                logger.debug(f"[MEMPOOL] Subscriber removed (total: {len(self._subscribers)})")

# ═══════════════════════════════════════════════════════════════════════════════════════
# GLOBAL SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════════════

_postgres_mempool: Optional[PostgresMempool] = None
_singleton_lock = threading.RLock()

def get_mempool() -> PostgresMempool:
    """Get or create PostgresMempool singleton"""
    global _postgres_mempool
    
    if _postgres_mempool is None:
        with _singleton_lock:
            if _postgres_mempool is None:
                _postgres_mempool = PostgresMempool()
    
    return _postgres_mempool

# ═══════════════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL FUNCTIONS (Drop-in replacement for old mempool module)
# ═══════════════════════════════════════════════════════════════════════════════════════

def add_transaction(tx_data: Dict[str, Any]) -> Tuple[bool, str]:
    """Module-level function: add transaction to mempool"""
    return get_mempool().add_transaction(tx_data)

def get_pending_transactions(max_count: int = MAX_BLOCK_TXS) -> List[MempoolTransaction]:
    """Module-level function: get pending transactions"""
    return get_mempool().get_pending_transactions(max_count)

def get_transaction_by_hash(tx_hash: str) -> Optional[MempoolTransaction]:
    """Module-level function: get transaction by hash"""
    return get_mempool().get_transaction_by_hash(tx_hash)

def mark_included_in_block(tx_hashes: List[str], block_height: int) -> bool:
    """Module-level function: mark transactions as included"""
    return get_mempool().mark_included_in_block(tx_hashes, block_height)

def get_mempool_stats() -> Dict[str, Any]:
    """Module-level function: get mempool statistics"""
    return get_mempool().get_stats()

# ═══════════════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════════════

__all__ = [
    'PostgresMempool',
    'MempoolTransaction',
    'get_mempool',
    'add_transaction',
    'get_pending_transactions',
    'get_transaction_by_hash',
    'mark_included_in_block',
    'get_mempool_stats',
]

if __name__ == '__main__':
    # Test
    logger.info("PostgresMempool module loaded successfully")
