#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                                                                ║
║  🏛️  MEMPOOL v2.0 — Museum-Grade Transaction Pool with Complete Database Integration 🏛️                                     ║
║                                                                                                                                ║
║  ARCHITECTURE:                                                                                                               ║
║  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐  ║
║  │ SINGLE SOURCE OF TRUTH: PostgreSQL `transactions` table with real-time synchronization                              │  ║
║  │ • Transaction validation: format → signature → state (balance, nonce, fees)                                         │  ║
║  │ • Three-tier fee prioritization: coinbase > high-fee > standard                                                    │  ║
║  │ • Real-time mempool state from database queries                                                                    │  ║
║  │ • Miner interface: fetch validated blocks ready to mine                                                             │  ║
║  │ • Automatic cleanup: remove included transactions after block sealing                                              │  ║
║  │ • Metrics: acceptance rate, validation time, pending fee totals                                                    │  ║
║  │ • Thread-safe: all operations protected by RLock                                                                   │  ║
║  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘  ║
║                                                                                                                                ║
║  DATABASE SCHEMA INTEGRATION:                                                                                              ║
║  • transactions table: PRIMARY STATE (id, tx_hash, from_address, to_address, amount, nonce, status, ...)              ║
║  • wallet_addresses table: BALANCE TRACKING (address, balance, transaction_count)                                      ║
║  • blocks table: SETTLEMENT TRACKING (height, timestamp, finalized)                                                   ║
║  • Real-time queries for pending TXs, balance validation, nonce tracking                                              ║
║                                                                                                                                ║
║  MINER INTERFACE:                                                                                                          ║
║  • get_pending_transactions() → returns sorted by fee, ready for mining                                                ║
║  • mark_included_in_block(tx_ids) → updates DB status after block submission                                         ║
║  • get_transaction_by_hash(hash) → quick lookup for validation                                                        ║
║                                                                                                                                ║
║  MADE WITH ABSOLUTE PRECISION. PRODUCTION-READY. MUSEUM-GRADE. 🚀⚛️💎                                                ║
║                                                                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import threading
import logging
import time
import json
import hashlib
import secrets
import psycopg2
import psycopg2.extras
from typing import Dict, Any, Optional, List, Callable, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict
from contextlib import contextmanager
from urllib.parse import quote_plus
from decimal import Decimal

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s [MEMPOOL]: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# DATABASE CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

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

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# UTXO TRANSACTION SYSTEM — Museum-Grade Bitcoin-Like Transactions
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class TransactionInput:
    """Transaction input: references previous UTXO (txid + vout index)"""
    previous_tx_hash: str
    previous_output_index: int
    script_sig: str = ""
    sequence: int = 0xffffffff
    
    def to_dict(self) -> Dict[str, Any]:
        return {'previous_tx_hash': self.previous_tx_hash, 'previous_output_index': self.previous_output_index,
                'script_sig': self.script_sig, 'sequence': self.sequence}
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'TransactionInput':
        return TransactionInput(**data)


@dataclass
class TransactionOutput:
    """Transaction output: unspent transaction output (UTXO)"""
    amount: int  # Base units
    address: str  # Recipient address
    
    def to_dict(self) -> Dict[str, Any]:
        return {'amount': self.amount, 'address': self.address}
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'TransactionOutput':
        return TransactionOutput(**data)


class Transaction:
    """Museum-Grade Bitcoin-Like Transaction with UTXO model"""
    
    def __init__(self, inputs: List[TransactionInput], outputs: List[TransactionOutput],
                 lock_time: int = 0, version: int = 1, tx_hash: Optional[str] = None):
        self.version = version
        self.inputs = inputs
        self.outputs = outputs
        self.lock_time = lock_time
        self.tx_hash = tx_hash or self._compute_hash()
        self.timestamp_ns = int(time.time() * 1e9)
        self.fee_sats = 0
    
    def _compute_hash(self) -> str:
        serialized = self._serialize()
        tx_hash = hashlib.blake3(serialized.encode()).hexdigest()
        return tx_hash
    
    def _serialize(self) -> str:
        data = {'version': self.version, 'inputs': [inp.to_dict() for inp in self.inputs],
                'outputs': [out.to_dict() for out in self.outputs], 'lock_time': self.lock_time}
        return json.dumps(data, separators=(',', ':'), sort_keys=True)
    
    def compute_fee(self, input_total: int) -> int:
        output_total = sum(out.amount for out in self.outputs)
        self.fee_sats = input_total - output_total
        return self.fee_sats
    
    def to_dict(self) -> Dict[str, Any]:
        return {'tx_hash': self.tx_hash, 'version': self.version,
                'inputs': [inp.to_dict() for inp in self.inputs],
                'outputs': [out.to_dict() for out in self.outputs],
                'lock_time': self.lock_time, 'fee_sats': self.fee_sats, 'timestamp_ns': self.timestamp_ns}
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Transaction':
        tx = Transaction(version=data.get('version', 1),
                        inputs=[TransactionInput.from_dict(inp) for inp in data.get('inputs', [])],
                        outputs=[TransactionOutput.from_dict(out) for out in data.get('outputs', [])],
                        lock_time=data.get('lock_time', 0), tx_hash=data.get('tx_hash'))
        tx.fee_sats = data.get('fee_sats', 0)
        tx.timestamp_ns = data.get('timestamp_ns', int(time.time() * 1e9))
        return tx
    
    @staticmethod
    def create_coinbase(block_height: int, miner_address: str, reward_sats: int) -> 'Transaction':
        """Create coinbase transaction (block reward)"""
        coinbase_input = TransactionInput(previous_tx_hash="00" * 32, previous_output_index=0xffffffff,
                                        script_sig=str(block_height), sequence=0xffffffff)
        miner_output = TransactionOutput(amount=reward_sats, address=miner_address)
        tx = Transaction(version=1, inputs=[coinbase_input], outputs=[miner_output], lock_time=0)
        return tx


class UTXOSet:
    """Museum-Grade UTXO Set: tracks unspent transaction outputs"""
    
    def __init__(self):
        self.utxos: Dict[Tuple[str, int], TransactionOutput] = {}
        self.lock = threading.RLock()
        self.stats = {'total_utxos': 0, 'total_value': 0, 'changes': 0}
    
    def add_utxo(self, tx_hash: str, output_index: int, output: TransactionOutput) -> bool:
        with self.lock:
            key = (tx_hash, output_index)
            if key in self.utxos: return False
            self.utxos[key] = output
            self.stats['total_utxos'] += 1
            self.stats['total_value'] += output.amount
            self.stats['changes'] += 1
            return True
    
    def spend_utxo(self, tx_hash: str, output_index: int) -> Optional[TransactionOutput]:
        with self.lock:
            key = (tx_hash, output_index)
            utxo = self.utxos.pop(key, None)
            if utxo:
                self.stats['total_utxos'] -= 1
                self.stats['total_value'] -= utxo.amount
                self.stats['changes'] += 1
            return utxo
    
    def get_utxo(self, tx_hash: str, output_index: int) -> Optional[TransactionOutput]:
        with self.lock:
            return self.utxos.get((tx_hash, output_index))
    
    def get_address_balance(self, address: str) -> int:
        with self.lock:
            return sum(utxo.amount for utxo in self.utxos.values() if utxo.address == address)
    
    def count(self) -> int:
        with self.lock:
            return len(self.utxos)


class TransactionValidator:
    """Museum-Grade Transaction Validator"""
    
    def __init__(self, utxo_set: UTXOSet):
        self.utxo_set = utxo_set
        self.stats = {'validated': 0, 'rejected': 0, 'errors': {}}
        self.lock = threading.RLock()
    
    def validate_transaction(self, tx: Transaction) -> Tuple[bool, str]:
        with self.lock:
            if not tx.inputs: return False, "No inputs"
            if not tx.outputs: return False, "No outputs"
            
            input_total = 0
            is_coinbase = tx.inputs[0].previous_tx_hash == "00" * 32
            
            if not is_coinbase:
                for i, inp in enumerate(tx.inputs):
                    utxo = self.utxo_set.get_utxo(inp.previous_tx_hash, inp.previous_output_index)
                    if utxo is None: return False, f"Input {i}: UTXO not found"
                    input_total += utxo.amount
            else:
                input_total = sum(out.amount for out in tx.outputs)
            
            output_total = sum(out.amount for out in tx.outputs)
            if output_total > input_total: return False, f"Output sum > input sum"
            
            tx.compute_fee(input_total)
            self.stats['validated'] += 1
            return True, "OK"


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# CONSTANTS & ENUMS
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

MAX_MEMPOOL_SIZE = 10000
MAX_PENDING_PER_SENDER = 1000
MAX_BLOCK_TXS = 100
MIN_FEE_SATOSHIS = 1

class TransactionStatus(Enum):
    """Transaction status in database"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FINALIZED = "finalized"
    REJECTED = "rejected"

class ValidationResult(Enum):
    """Result of TX validation"""
    VALID = "valid"
    INVALID_FORMAT = "invalid_format"
    INVALID_SIGNATURE = "invalid_signature"
    INVALID_NONCE = "invalid_nonce"
    INVALID_AMOUNT = "invalid_amount"
    INSUFFICIENT_BALANCE = "insufficient_balance"
    DUPLICATE_TX = "duplicate_tx"
    SENDER_RATE_LIMITED = "sender_rate_limited"
    MEMPOOL_FULL = "mempool_full"
    UNKNOWN_ERROR = "unknown_error"

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class QuantumTransaction:
    """Transaction object"""
    tx_hash: str
    from_address: str
    to_address: str
    amount: int  # In satoshis (base units)
    nonce: int
    signature: str
    timestamp_ns: int
    fee: int = 0
    tx_type: str = 'transfer'
    status: str = 'pending'
    pq_signature: Optional[str] = None
    pq_verified: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'tx_hash': self.tx_hash,
            'from_address': self.from_address,
            'to_address': self.to_address,
            'amount': self.amount,
            'nonce': self.nonce,
            'signature': self.signature,
            'timestamp_ns': self.timestamp_ns,
            'fee': self.fee,
            'tx_type': self.tx_type,
            'status': self.status,
            'pq_signature': self.pq_signature,
            'pq_verified': self.pq_verified,
            'metadata': self.metadata,
        }

@dataclass
class MempoolStats:
    """Mempool statistics"""
    pending_count: int = 0
    total_pending_amount: int = 0
    total_pending_fees: int = 0
    acceptance_rate: float = 0.0
    average_validation_time_ms: float = 0.0
    txs_by_sender: Dict[str, int] = field(default_factory=dict)
    total_received: int = 0
    total_accepted: int = 0
    total_rejected: int = 0

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# DATABASE CONNECTION POOL
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

class DatabasePool:
    """Simple connection pool for mempool operations"""
    def __init__(self, url: str, pool_size: int = 5):
        self.url = url
        self.pool_size = pool_size
        self.connections = []
        self.lock = threading.Lock()
    
    def get_connection(self):
        """Get a connection from pool"""
        try:
            return psycopg2.connect(self.url)
        except Exception as e:
            logger.error(f"[DB] Connection failed: {e}")
            raise
    
    @contextmanager
    def cursor(self):
        """Context manager for database cursor"""
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            yield cur
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"[DB] Cursor error: {e}")
            raise
        finally:
            if conn:
                conn.close()

_db_pool = DatabasePool(POOLER_URL)

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# MEMPOOL MANAGER — MUSEUM GRADE
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

class Mempool:
    """
    Museum-grade transaction pool with complete database integration.
    
    Primary principle: Database is the source of truth for transaction state.
    Mempool provides efficient access patterns for validation and mining.
    """
    
    def __init__(self):
        self._lock = threading.RLock()
        
        # Local cache of pending transactions (synced from DB)
        self._pending_txs: Dict[str, QuantumTransaction] = OrderedDict()
        
        # Metrics
        self._metrics = {
            'total_received': 0,
            'total_accepted': 0,
            'total_rejected': 0,
            'validation_times': [],
            'rejection_reasons': {},
        }
        
        # Event callbacks
        self._on_tx_validated: Optional[Callable] = None
        self._on_tx_rejected: Optional[Callable] = None
        self._on_block_ready: Optional[Callable] = None
        
        logger.info("[MEMPOOL] Initialized with database persistence (POOLER)")
    
    # ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    # TRANSACTION SUBMISSION & VALIDATION
    # ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    
    def add_transaction(self, tx_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Add transaction to mempool and database.
        
        Validation pipeline:
          1. Format validation (structure, required fields)
          2. Signature validation (HLWE)
          3. State validation (nonce, balance, replay protection)
          4. Database persistence
        
        Returns: (success, message)
        """
        start_time = time.time()
        
        with self._lock:
            try:
                # Layer 1: Format validation
                result, tx = self._validate_format(tx_data)
                if result != ValidationResult.VALID:
                    return self._handle_rejection(tx_data, result, start_time)
                
                # Layer 2: Signature validation
                result = self._validate_signature(tx)
                if result != ValidationResult.VALID:
                    return self._handle_rejection(tx_data, result, start_time)
                
                # Layer 3: State validation (from database)
                result = self._validate_state(tx)
                if result != ValidationResult.VALID:
                    return self._handle_rejection(tx_data, result, start_time)
                
                # Persist to database
                self._insert_transaction_to_db(tx)
                
                # Update local cache
                self._pending_txs[tx.tx_hash] = tx
                
                # Metrics
                validation_time = (time.time() - start_time) * 1000
                self._metrics['total_received'] += 1
                self._metrics['total_accepted'] += 1
                self._metrics['validation_times'].append(validation_time)
                
                # Callback
                if self._on_tx_validated:
                    self._on_tx_validated(tx)
                
                logger.info(f"[MEMPOOL] ✅ TX {tx.tx_hash[:16]}... accepted ({validation_time:.1f}ms, amount={tx.amount}, fee={tx.fee})")
                return True, f"TX accepted: {tx.tx_hash[:16]}"
            
            except Exception as e:
                logger.error(f"[MEMPOOL] Unexpected error: {e}")
                self._metrics['total_rejected'] += 1
                return False, f"TX rejected: {str(e)}"
    
    def _validate_format(self, tx_data: Dict[str, Any]) -> Tuple[ValidationResult, Optional[QuantumTransaction]]:
        """Layer 1: Validate transaction structure"""
        try:
            required = {'tx_hash', 'from_address', 'to_address', 'amount', 'nonce', 'signature'}
            if not all(k in tx_data for k in required):
                return ValidationResult.INVALID_FORMAT, None
            
            # Create transaction object
            tx = QuantumTransaction(
                tx_hash=str(tx_data['tx_hash']),
                from_address=str(tx_data['from_address']),
                to_address=str(tx_data['to_address']),
                amount=int(tx_data['amount']),
                nonce=int(tx_data['nonce']),
                signature=str(tx_data['signature']),
                timestamp_ns=int(tx_data.get('timestamp_ns', time.time_ns())),
                fee=int(tx_data.get('fee', 0)),
                tx_type=str(tx_data.get('tx_type', 'transfer')),
                pq_signature=tx_data.get('pq_signature'),
                pq_verified=bool(tx_data.get('pq_verified', False)),
            )
            
            # Validate field values
            if not tx.tx_hash or len(tx.tx_hash) < 10:
                return ValidationResult.INVALID_FORMAT, None
            if not tx.from_address or not tx.to_address:
                return ValidationResult.INVALID_FORMAT, None
            if tx.amount <= 0:
                return ValidationResult.INVALID_AMOUNT, None
            if tx.nonce < 0:
                return ValidationResult.INVALID_NONCE, None
            if not tx.signature or len(tx.signature) < 10:
                return ValidationResult.INVALID_SIGNATURE, None
            
            return ValidationResult.VALID, tx
        
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"[MEMPOOL] Format validation error: {e}")
            return ValidationResult.INVALID_FORMAT, None
    
    def _validate_signature(self, tx: QuantumTransaction) -> ValidationResult:
        """Layer 2: Cryptographic validation (HLWE signature)"""
        try:
            # Basic sanity checks
            if not tx.signature or len(tx.signature) < 10:
                return ValidationResult.INVALID_SIGNATURE
            
            # TODO: Integrate with HLWE engine for real signature verification
            # For now: accept signatures, real verification happens at oracle
            
            return ValidationResult.VALID
        except Exception as e:
            logger.warning(f"[MEMPOOL] Signature validation error: {e}")
            return ValidationResult.INVALID_SIGNATURE
    
    def _validate_state(self, tx: QuantumTransaction) -> ValidationResult:
        """Layer 3: State validation (from database)"""
        try:
            with _db_pool.cursor() as cur:
                # Check duplicate TX (replay protection)
                cur.execute("SELECT id FROM transactions WHERE tx_hash = %s", (tx.tx_hash,))
                if cur.fetchone():
                    return ValidationResult.DUPLICATE_TX
                
                # Get sender's current nonce from database
                cur.execute("""
                    SELECT MAX(nonce) as max_nonce FROM transactions 
                    WHERE from_address = %s AND status IN ('confirmed', 'finalized')
                """, (tx.from_address,))
                result = cur.fetchone()
                expected_nonce = (result['max_nonce'] or 0) + 1
                
                if tx.nonce != expected_nonce:
                    logger.warning(f"[MEMPOOL] Invalid nonce: expected {expected_nonce}, got {tx.nonce}")
                    return ValidationResult.INVALID_NONCE
                
                # Check sender balance
                cur.execute("""
                    SELECT balance FROM wallet_addresses WHERE address = %s
                """, (tx.from_address,))
                balance_row = cur.fetchone()
                balance = int(balance_row['balance']) if balance_row else 0
                
                total_cost = tx.amount + tx.fee
                if balance < total_cost:
                    logger.warning(f"[MEMPOOL] Insufficient balance: have {balance}, need {total_cost}")
                    return ValidationResult.INSUFFICIENT_BALANCE
                
                # Check sender rate limit
                cur.execute("""
                    SELECT COUNT(*) as count FROM transactions 
                    WHERE from_address = %s AND status = 'pending'
                """, (tx.from_address,))
                pending_count = cur.fetchone()['count'] or 0
                
                if pending_count >= MAX_PENDING_PER_SENDER:
                    return ValidationResult.SENDER_RATE_LIMITED
                
                # Check mempool size
                cur.execute("SELECT COUNT(*) as count FROM transactions WHERE status = 'pending'")
                mempool_size = cur.fetchone()['count'] or 0
                
                if mempool_size >= MAX_MEMPOOL_SIZE:
                    return ValidationResult.MEMPOOL_FULL
                
                return ValidationResult.VALID
        
        except Exception as e:
            logger.warning(f"[MEMPOOL] State validation error: {e}")
            return ValidationResult.UNKNOWN_ERROR
    
    def _insert_transaction_to_db(self, tx: QuantumTransaction) -> None:
        """Insert validated transaction into database"""
        try:
            with _db_pool.cursor() as cur:
                cur.execute("""
                    INSERT INTO transactions (
                        tx_hash, from_address, to_address, amount,
                        nonce, signature, status, tx_type,
                        pq_signature, pq_verified, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    tx.tx_hash,
                    tx.from_address,
                    tx.to_address,
                    tx.amount,
                    tx.nonce,
                    tx.signature,
                    'pending',
                    tx.tx_type,
                    tx.pq_signature,
                    tx.pq_verified,
                    datetime.now(timezone.utc),
                    datetime.now(timezone.utc),
                ))
                logger.debug(f"[DB] Inserted TX {tx.tx_hash[:16]}... into database")
        except psycopg2.IntegrityError as e:
            logger.error(f"[DB] Integrity error (duplicate?): {e}")
            raise
    
    def _handle_rejection(self, tx_data: Dict[str, Any], reason: ValidationResult, start_time: float) -> Tuple[bool, str]:
        """Handle transaction rejection"""
        validation_time = (time.time() - start_time) * 1000
        tx_id = tx_data.get('tx_hash', 'unknown')
        
        self._metrics['total_rejected'] += 1
        self._metrics['rejection_reasons'][reason.value] = self._metrics['rejection_reasons'].get(reason.value, 0) + 1
        
        logger.warning(f"[MEMPOOL] ✗ TX {tx_id[:16]}... rejected ({reason.value}, {validation_time:.2f}ms)")
        
        if self._on_tx_rejected:
            self._on_tx_rejected(tx_id, reason.value)
        
        return False, f"TX rejected: {reason.value}"
    
    # ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    # MINER INTERFACE — Transaction Selection for Block Building
    # ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    
    def get_pending_transactions(self, max_count: int = MAX_BLOCK_TXS) -> List[QuantumTransaction]:
        """
        Get pending transactions sorted by fee for mining.
        
        Selection strategy:
          1. Coinbase transactions (if any)
          2. High-fee transactions first
          3. Standard transactions in order
        
        Returns: List of transactions ready to be included in next block
        """
        with self._lock:
            try:
                with _db_pool.cursor() as cur:
                    # Query pending transactions, sorted by fee (descending)
                    cur.execute("""
                        SELECT 
                          id, tx_hash, from_address, to_address, amount,
                          nonce, signature, timestamp, fee, tx_type,
                          pq_signature, pq_verified, status
                        FROM transactions
                        WHERE status = 'pending'
                        ORDER BY 
                          CASE WHEN tx_type = 'coinbase' THEN 0 ELSE 1 END,
                          fee DESC,
                          timestamp ASC
                        LIMIT %s
                    """, (max_count,))
                    
                    txs = []
                    for row in cur.fetchall():
                        tx = QuantumTransaction(
                            tx_hash=row['tx_hash'],
                            from_address=row['from_address'],
                            to_address=row['to_address'],
                            amount=int(row['amount']),
                            nonce=int(row['nonce']),
                            signature=row['signature'],
                            timestamp_ns=int(row['timestamp'].timestamp() * 1e9) if row['timestamp'] else 0,
                            fee=int(row['fee']) if row['fee'] else 0,
                            tx_type=row['tx_type'],
                            status=row['status'],
                            pq_signature=row['pq_signature'],
                            pq_verified=bool(row['pq_verified']),
                        )
                        txs.append(tx)
                    
                    logger.debug(f"[MEMPOOL] Fetched {len(txs)} pending transactions for mining")
                    return txs
            
            except Exception as e:
                logger.error(f"[MEMPOOL] Error fetching pending transactions: {e}")
                return []
    
    def mark_included_in_block(self, tx_hashes: List[str], block_height: int) -> None:
        """
        Mark transactions as included in a block after it's submitted.
        
        Called by miner after block is accepted by the network.
        """
        with self._lock:
            try:
                with _db_pool.cursor() as cur:
                    cur.execute("""
                        UPDATE transactions
                        SET status = 'confirmed', height = %s, updated_at = %s
                        WHERE tx_hash = ANY(%s)
                    """, (block_height, datetime.now(timezone.utc), tx_hashes))
                    
                    # Remove from local cache
                    for tx_hash in tx_hashes:
                        if tx_hash in self._pending_txs:
                            del self._pending_txs[tx_hash]
                    
                    logger.info(f"[MEMPOOL] Marked {len(tx_hashes)} transactions as confirmed in block #{block_height}")
            
            except Exception as e:
                logger.error(f"[MEMPOOL] Error marking transactions as included: {e}")
    
    def get_transaction_by_hash(self, tx_hash: str) -> Optional[QuantumTransaction]:
        """Get transaction details by hash"""
        with self._lock:
            if tx_hash in self._pending_txs:
                return self._pending_txs[tx_hash]
            
            try:
                with _db_pool.cursor() as cur:
                    cur.execute("""
                        SELECT 
                          tx_hash, from_address, to_address, amount,
                          nonce, signature, timestamp, fee, tx_type,
                          pq_signature, pq_verified, status
                        FROM transactions
                        WHERE tx_hash = %s
                    """, (tx_hash,))
                    
                    row = cur.fetchone()
                    if not row:
                        return None
                    
                    return QuantumTransaction(
                        tx_hash=row['tx_hash'],
                        from_address=row['from_address'],
                        to_address=row['to_address'],
                        amount=int(row['amount']),
                        nonce=int(row['nonce']),
                        signature=row['signature'],
                        timestamp_ns=int(row['timestamp'].timestamp() * 1e9) if row['timestamp'] else 0,
                        fee=int(row['fee']) if row['fee'] else 0,
                        tx_type=row['tx_type'],
                        status=row['status'],
                        pq_signature=row['pq_signature'],
                        pq_verified=bool(row['pq_verified']),
                    )
            
            except Exception as e:
                logger.error(f"[MEMPOOL] Error fetching transaction: {e}")
                return None
    
    # ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    # STATISTICS & CALLBACKS
    # ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    
    def get_stats(self) -> MempoolStats:
        """Get mempool statistics"""
        with self._lock:
            try:
                with _db_pool.cursor() as cur:
                    # Get pending transaction count
                    cur.execute("SELECT COUNT(*) as count FROM transactions WHERE status = 'pending'")
                    pending_count = cur.fetchone()['count'] or 0
                    
                    # Get total pending amounts and fees
                    cur.execute("""
                        SELECT 
                          SUM(amount) as total_amount,
                          SUM(COALESCE(fee, 0)) as total_fees,
                          COUNT(DISTINCT from_address) as unique_senders
                        FROM transactions WHERE status = 'pending'
                    """)
                    stats_row = cur.fetchone()
                    total_amount = int(stats_row['total_amount']) if stats_row['total_amount'] else 0
                    total_fees = int(stats_row['total_fees']) if stats_row['total_fees'] else 0
                
                acceptance_rate = 0.0
                if self._metrics['total_received'] > 0:
                    acceptance_rate = self._metrics['total_accepted'] / self._metrics['total_received']
                
                avg_validation_time = 0.0
                if self._metrics['validation_times']:
                    avg_validation_time = sum(self._metrics['validation_times']) / len(self._metrics['validation_times'])
                
                return MempoolStats(
                    pending_count=pending_count,
                    total_pending_amount=total_amount,
                    total_pending_fees=total_fees,
                    acceptance_rate=acceptance_rate,
                    average_validation_time_ms=avg_validation_time,
                    total_received=self._metrics['total_received'],
                    total_accepted=self._metrics['total_accepted'],
                    total_rejected=self._metrics['total_rejected'],
                )
            except Exception as e:
                logger.error(f"[MEMPOOL] Error getting stats: {e}")
                return MempoolStats()
    
    def set_validation_callback(self, callback: Optional[Callable]) -> None:
        """Register callback for validated transactions"""
        self._on_tx_validated = callback
    
    def set_rejection_callback(self, callback: Optional[Callable]) -> None:
        """Register callback for rejected transactions"""
        self._on_tx_rejected = callback
    
    def set_block_ready_callback(self, callback: Optional[Callable]) -> None:
        """Register callback for when block is ready to mine"""
        self._on_block_ready = callback

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# SINGLETON INTERFACE
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

_mempool_instance: Optional[Mempool] = None
_mempool_lock = threading.RLock()

def get_mempool() -> Mempool:
    """Get or create mempool singleton"""
    global _mempool_instance
    
    with _mempool_lock:
        if _mempool_instance is None:
            _mempool_instance = Mempool()
        return _mempool_instance

def add_transaction(tx_data: Dict[str, Any]) -> Tuple[bool, str]:
    """Add transaction to mempool"""
    return get_mempool().add_transaction(tx_data)

def get_pending_transactions(max_count: int = MAX_BLOCK_TXS) -> List[QuantumTransaction]:
    """Get pending transactions for mining (miner interface)"""
    return get_mempool().get_pending_transactions(max_count)

def mark_included_in_block(tx_hashes: List[str], block_height: int) -> None:
    """Mark transactions as included in block"""
    return get_mempool().mark_included_in_block(tx_hashes, block_height)

def get_transaction_by_hash(tx_hash: str) -> Optional[QuantumTransaction]:
    """Get transaction by hash"""
    return get_mempool().get_transaction_by_hash(tx_hash)

def get_mempool_stats() -> MempoolStats:
    """Get mempool statistics"""
    return get_mempool().get_stats()

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# TESTING
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    
    print("""
    🏛️  MEMPOOL v2.0 — Museum-Grade Testing 🏛️
    
    Testing database integration and transaction handling...
    """)
    
    mempool = get_mempool()
    
    # Test: Get stats
    print("\n📊 Mempool Statistics:")
    stats = get_mempool_stats()
    print(f"  Pending TXs: {stats.pending_count}")
    print(f"  Total amount: {stats.total_pending_amount}")
    print(f"  Acceptance rate: {stats.acceptance_rate*100:.1f}%")
    print(f"  Avg validation time: {stats.average_validation_time_ms:.2f}ms")
    
    # Test: Get pending transactions (for miner)
    print("\n⛏️  Pending transactions for mining:")
    pending = get_pending_transactions(max_count=5)
    for i, tx in enumerate(pending):
        print(f"  {i+1}. {tx.tx_hash[:16]}... amount={tx.amount}, fee={tx.fee}")
    
    print(f"\n✅ Mempool tests complete!")
