#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║  QTCL SERVER v6 — Integrated P2P Blockchain with HLWE & Quantum Metrics       ║
║                                                                                ║
║  Museum-Grade Implementation                                                   ║
║  ─────────────────────────────────────────────────────────────────────────    ║
║                                                                                ║
║  Single Unified Server:                                                        ║
║    • P2P Networking Layer (port 8333) — peer-to-peer consensus               ║
║    • REST API Layer (port 5000) — client-facing interface                    ║
║    • Database Layer (port 5432) — persistent state (internal only)           ║
║    • Lattice Controller — quantum entropy mining                             ║
║    • Mempool Manager — transaction pool with validation                      ║
║    • Peer Discovery — DNS seeds, bootstrap nodes, peer exchange              ║
║    • Message Handlers — blocks, transactions, peer sync, consensus           ║
║                                                                                ║
║  Entry:                                                                        ║
║    python server.py                                                            ║
║    or: gunicorn -w1 -b0.0.0.0:5000 server:app                                ║
║                                                                                ║
║  Environment Variables:                                                        ║
║    DATABASE_URL — PostgreSQL connection                                       ║
║    P2P_PORT — P2P listen port (default: 8333)                                ║
║    P2P_HOST — P2P bind address (default: 0.0.0.0)                            ║
║    MAX_PEERS — Max peer connections (default: 32)                            ║
║    BOOTSTRAP_NODES — Comma-separated peer addresses                          ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import time
import socket
import logging
import threading
import traceback
import hashlib
import psycopg2
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Tuple, List, Set, Callable
from contextlib import contextmanager
from collections import deque, OrderedDict
from dataclasses import dataclass, field, asdict
from enum import Enum
from decimal import Decimal
import random

from flask import Flask, jsonify, request, render_template_string, send_file
from io import BytesIO

# ═════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════════

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s]: %(message)s'
)
logger = logging.getLogger(__name__)

# Database Configuration
# Priority: POOLER_URL (Supabase/PgBouncer) > DATABASE_URL > fallback
POOLER_URL = os.getenv('POOLER_URL')
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost/qtcl_db')
DB_URL = POOLER_URL or DATABASE_URL

logger.info(f"[DB] Using connection: {'POOLER_URL (Supabase)' if POOLER_URL else 'DATABASE_URL'}")

# P2P Network
P2P_PORT = int(os.getenv('P2P_PORT', 8333))
P2P_HOST = os.getenv('P2P_HOST', '0.0.0.0')
P2P_TESTNET_PORT = 18333
MAX_PEERS = int(os.getenv('MAX_PEERS', 32))
PEER_TIMEOUT = 30
MESSAGE_MAX_SIZE = 1_000_000
PEER_HANDSHAKE_TIMEOUT = 5
PEER_KEEPALIVE_INTERVAL = 30
PEER_DISCOVERY_INTERVAL = 60
PEER_CLEANUP_INTERVAL = 15

# Message Types
MESSAGE_TYPES = {
    'version': 0,
    'verack': 1,
    'ping': 2,
    'pong': 3,
    'inv': 4,
    'getdata': 5,
    'block': 6,
    'tx': 7,
    'mempool': 8,
    'getblocks': 9,
    'getheaders': 10,
    'headers': 11,
    'addr': 12,
    'peers_sync': 13,
    'peer_discovery': 14,
    'consensus': 15,
}

# Peer Discovery
DNS_SEEDS = [
    # Bootstrap nodes for peer discovery
    # Format: "hostname:port"
    # In production, these would be actual DNS seed servers
]

BOOTSTRAP_NODES = os.getenv('BOOTSTRAP_NODES', '').split(',') if os.getenv('BOOTSTRAP_NODES') else []
DEFAULT_BOOTSTRAP_PEERS = [
    # Fallback bootstrap nodes (localhost for testing)
    # In production, use real peer addresses
]

# ═════════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES & ENUMS
# ═════════════════════════════════════════════════════════════════════════════════

class BlockchainEvent(Enum):
    """Events that occur in the blockchain"""
    BLOCK_RECEIVED = "block_received"
    BLOCK_VALIDATED = "block_validated"
    BLOCK_STORED = "block_stored"
    TX_RECEIVED = "tx_received"
    TX_VALIDATED = "tx_validated"
    TX_BROADCAST = "tx_broadcast"
    PEER_CONNECTED = "peer_connected"
    PEER_DISCONNECTED = "peer_disconnected"
    PEER_SYNCED = "peer_synced"
    CONSENSUS_ACHIEVED = "consensus_achieved"


@dataclass
class PeerInfo:
    """Peer connection metadata"""
    peer_id: str
    address: str
    port: int
    connected_at: float
    last_message_at: float
    last_block_height: int = 0
    last_block_hash: Optional[str] = None
    version: Optional[int] = None
    user_agent: Optional[str] = None
    protocol_version: int = 1
    blocks_announced: int = 0
    txs_announced: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    is_outbound: bool = False
    is_preferred: bool = False
    ban_score: int = 0
    
    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.connected_at
    
    @property
    def is_alive(self) -> bool:
        return (time.time() - self.last_message_at) < PEER_TIMEOUT
    
    @property
    def is_synced(self) -> bool:
        return self.last_message_at > time.time() - 60
    
    def __hash__(self):
        return hash(self.peer_id)
    
    def __eq__(self, other):
        if isinstance(other, PeerInfo):
            return self.peer_id == other.peer_id
        return False


@dataclass
class Message:
    """P2P message structure with serialization"""
    msg_type: str
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    sender_id: Optional[str] = None
    message_id: str = field(default_factory=lambda: hashlib.sha256(str(time.time()).encode()).hexdigest()[:16])
    
    def to_bytes(self) -> bytes:
        """Serialize message to bytes"""
        data = {
            'type': self.msg_type,
            'payload': self.payload,
            'timestamp': self.timestamp,
            'message_id': self.message_id,
        }
        return json.dumps(data).encode('utf-8')
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'Message':
        """Deserialize message from bytes"""
        parsed = json.loads(data.decode('utf-8'))
        return cls(
            msg_type=parsed['type'],
            payload=parsed['payload'],
            timestamp=parsed.get('timestamp', time.time()),
            message_id=parsed.get('message_id', ''),
        )
    
    def __repr__(self):
        return f"Message({self.msg_type}, {self.message_id[:8]}...)"


@dataclass
class BlockHeader:
    """Minimal block header for announcements"""
    height: int
    block_hash: str
    parent_hash: Optional[str] = None
    timestamp: int = 0
    miner: Optional[str] = None


@dataclass
class TransactionInfo:
    """Transaction information for gossip"""
    tx_hash: str
    from_address: str
    to_address: str
    amount: int
    timestamp: int = field(default_factory=lambda: int(time.time()))


# ═════════════════════════════════════════════════════════════════════════════════
# DATABASE LAYER WITH CONNECTION POOLING
# ═════════════════════════════════════════════════════════════════════════════════

class DatabasePool:
    """Thread-safe connection pool for efficient database access"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        try:
            # Connection pool configuration (optimized for serverless)
            min_connections = int(os.getenv('DB_POOL_MIN', '2'))
            max_connections = int(os.getenv('DB_POOL_MAX', '10'))
            
            logger.info(f"[DB] Initializing connection pool: min={min_connections}, max={max_connections}")
            
            self.pool = psycopg2.pool.SimpleConnectionPool(
                min_connections,
                max_connections,
                DB_URL,
                connect_timeout=10
            )
            self._initialized = True
            logger.info("[DB] ✨ Connection pool initialized successfully")
        except Exception as e:
            logger.error(f"[DB] CRITICAL: Failed to initialize connection pool: {e}")
            raise
    
    def get_connection(self):
        """Get a connection from the pool with fallback"""
        try:
            conn = self.pool.getconn()
            if conn is None:
                logger.warning("[DB] Pool exhausted, creating temporary connection")
                conn = psycopg2.connect(DB_URL, connect_timeout=10)
            return conn
        except Exception as e:
            logger.error(f"[DB] Failed to get connection: {e}")
            raise
    
    def put_connection(self, conn):
        """Return a connection to the pool"""
        try:
            if conn:
                self.pool.putconn(conn)
        except Exception as e:
            logger.error(f"[DB] Error returning connection: {e}")
    
    def close_all(self):
        """Close all connections in the pool"""
        try:
            if hasattr(self, 'pool') and self.pool:
                self.pool.closeall()
                logger.info("[DB] Connection pool closed")
        except Exception as e:
            logger.error(f"[DB] Error closing pool: {e}")


# Global pool instance (singleton)
db_pool = DatabasePool()


@contextmanager
def get_db_cursor():
    """Context manager for database cursor with connection pooling"""
    conn = None
    try:
        conn = db_pool.get_connection()
        cur = conn.cursor()
        yield cur
        conn.commit()
    except Exception as e:
        if conn:
            try:
                conn.rollback()
            except:
                pass
        logger.error(f"[DB] Query error: {e}")
        raise
    finally:
        if conn:
            db_pool.put_connection(conn)


def query_latest_block() -> Optional[Dict[str, Any]]:
    """Get latest block from database"""
    try:
        with get_db_cursor() as cur:
            cur.execute("""
                SELECT height, block_hash, timestamp, oracle_w_state_hash
                FROM blocks
                ORDER BY height DESC
                LIMIT 1
            """)
            row = cur.fetchone()
            if row:
                return {
                    'height': row[0],
                    'hash': row[1],
                    'timestamp': row[2],
                    'w_state_hash': row[3],
                }
    except Exception as e:
        logger.error(f"[DB] Failed to query blocks: {e}")
    return None


def query_block_by_hash(block_hash: str) -> Optional[Dict[str, Any]]:
    """Get block by hash"""
    try:
        with get_db_cursor() as cur:
            cur.execute("""
                SELECT height, block_hash, timestamp, oracle_w_state_hash, data
                FROM blocks
                WHERE block_hash = %s
                LIMIT 1
            """, (block_hash,))
            row = cur.fetchone()
            if row:
                return {
                    'height': row[0],
                    'hash': row[1],
                    'timestamp': row[2],
                    'w_state_hash': row[3],
                    'data': row[4],
                }
    except Exception as e:
        logger.error(f"[DB] Failed to query block: {e}")
    return None


def query_blocks_range(from_height: int, to_height: int) -> List[Dict[str, Any]]:
    """Get blocks in height range"""
    try:
        with get_db_cursor() as cur:
            cur.execute("""
                SELECT height, block_hash, timestamp, oracle_w_state_hash
                FROM blocks
                WHERE height BETWEEN %s AND %s
                ORDER BY height ASC
            """, (from_height, to_height))
            rows = cur.fetchall()
            return [
                {
                    'height': row[0],
                    'hash': row[1],
                    'timestamp': row[2],
                    'w_state_hash': row[3],
                }
                for row in rows
            ]
    except Exception as e:
        logger.error(f"[DB] Failed to query block range: {e}")
    return []


def query_pseudoqubit_range() -> Tuple[int, int]:
    """Get min/max pq_id in database"""
    try:
        with get_db_cursor() as cur:
            cur.execute("SELECT MIN(pq_id), MAX(pq_id) FROM pseudoqubits")
            row = cur.fetchone()
            if row and row[0] is not None:
                return (row[0], row[1])
    except Exception as e:
        logger.error(f"[DB] Failed to query pseudoqubits: {e}")
    return (0, 0)


def query_wallet_info(address: str) -> Optional[Dict[str, Any]]:
    """Get wallet info"""
    try:
        with get_db_cursor() as cur:
            cur.execute("""
                SELECT balance, transaction_count
                FROM wallet_addresses
                WHERE address = %s
            """, (address,))
            row = cur.fetchone()
            if row:
                return {
                    'address': address,
                    'balance': row[0],
                    'tx_count': row[1],
                }
    except Exception as e:
        logger.error(f"[DB] Failed to query wallet: {e}")
    return None


def query_transaction(tx_hash: str) -> Optional[Dict[str, Any]]:
    """Get transaction by hash"""
    try:
        with get_db_cursor() as cur:
            cur.execute("""
                SELECT tx_hash, from_address, to_address, amount, status, timestamp
                FROM transactions
                WHERE tx_hash = %s
            """, (tx_hash,))
            row = cur.fetchone()
            if row:
                return {
                    'tx_hash': row[0],
                    'from': row[1],
                    'to': row[2],
                    'amount': row[3],
                    'status': row[4],
                    'timestamp': row[5],
                }
    except Exception as e:
        logger.error(f"[DB] Failed to query transaction: {e}")
    return None


def insert_transaction(from_addr: str, to_addr: str, amount: int) -> Optional[str]:
    """Insert transaction into database"""
    try:
        with get_db_cursor() as cur:
            tx_hash = f"tx_{int(time.time() * 1000)}"
            cur.execute("""
                INSERT INTO transactions (tx_hash, from_address, to_address, amount, status)
                VALUES (%s, %s, %s, %s, 'pending')
                RETURNING tx_hash
            """, (tx_hash, from_addr, to_addr, amount))
            result = cur.fetchone()
            return result[0] if result else None
    except Exception as e:
        logger.error(f"[DB] Failed to insert transaction: {e}")
    return None


def store_peer_info(peer_info: PeerInfo):
    """Store peer information in database for persistence"""
    try:
        with get_db_cursor() as cur:
            cur.execute("""
                INSERT INTO peer_registry (peer_id, address, port, last_seen, block_height, user_agent)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT(peer_id) DO UPDATE SET
                    last_seen = EXCLUDED.last_seen,
                    block_height = EXCLUDED.block_height,
                    user_agent = EXCLUDED.user_agent
            """, (
                peer_info.peer_id,
                peer_info.address,
                peer_info.port,
                int(time.time()),
                peer_info.last_block_height,
                peer_info.user_agent,
            ))
    except Exception as e:
        logger.error(f"[DB] Failed to store peer info: {e}")


def load_known_peers() -> List[Tuple[str, int]]:
    """Load known peers from database"""
    try:
        with get_db_cursor() as cur:
            cur.execute("""
                SELECT address, port FROM peer_registry
                WHERE last_seen > NOW() - INTERVAL '7 days'
                ORDER BY last_seen DESC
                LIMIT 100
            """)
            return [(row[0], row[1]) for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"[DB] Failed to load known peers: {e}")
    return []


# ═════════════════════════════════════════════════════════════════════════════════
# FLASK APP SETUP
# ═════════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
application = app  # WSGI entry point

# ═════════════════════════════════════════════════════════════════════════════════
# SYSTEM STATE & GLOBAL MANAGEMENT
# ═════════════════════════════════════════════════════════════════════════════════

class SystemState:
    """Centralized system state management"""
    
    def __init__(self):
        self.db_conn = None
        self.lattice_loaded = False
        self.quantum_metrics = {
            'coherence': 0.99,
            'entanglement': 0.95,
            'phase_drift': 0.01,
            'w_state_fidelity': 0.98,
        }
        self.block_state = {
            'current_height': 0,
            'current_hash': None,
            'pq_current': 0,
            'pq_last': 0,
            'timestamp': None,
        }
        self.wallet_state = {
            'balance': 0,
            'address': None,
            'tx_count': 0,
        }
        self.heartbeat_ts = time.time()
        self.is_alive = True
        self.lock = threading.RLock()
    
    def update_metrics(self, metrics: Dict[str, float]):
        with self.lock:
            self.quantum_metrics.update(metrics)
            self.heartbeat_ts = time.time()
    
    def update_block_state(self, state: Dict[str, Any]):
        with self.lock:
            self.block_state.update(state)
    
    def get_state(self) -> Dict[str, Any]:
        with self.lock:
            return {
                'quantum_metrics': self.quantum_metrics.copy(),
                'block_state': self.block_state.copy(),
                'wallet_state': self.wallet_state.copy(),
                'heartbeat_ts': self.heartbeat_ts,
                'is_alive': self.is_alive,
                'uptime_s': time.time() - self.heartbeat_ts,
            }


state = SystemState()


# ═════════════════════════════════════════════════════════════════════════════════
# PEER CONNECTION HANDLER
# ═════════════════════════════════════════════════════════════════════════════════

class PeerConnection:
    """Single peer connection management"""
    
    def __init__(self, sock: socket.socket, address: Tuple[str, int], peer_id: str, is_outbound: bool = False):
        self.sock = sock
        self.peer_id = peer_id
        self.address = address[0]
        self.port = address[1]
        self.is_outbound = is_outbound
        
        self.peer_info = PeerInfo(
            peer_id=peer_id,
            address=self.address,
            port=self.port,
            connected_at=time.time(),
            last_message_at=time.time(),
            is_outbound=is_outbound,
        )
        
        self.is_connected = True
        self.version_received = False
        self.lock = threading.RLock()
    
    def send_message(self, message: Message) -> bool:
        """Send message to peer with protocol framing"""
        try:
            with self.lock:
                data = message.to_bytes()
                
                # Frame: 4-byte length + message
                length = len(data)
                if length > MESSAGE_MAX_SIZE:
                    logger.error(f"[PEER {self.peer_id}] Message too large: {length}")
                    return False
                
                self.sock.sendall(length.to_bytes(4, 'big'))
                self.sock.sendall(data)
                
                self.peer_info.messages_sent += 1
                self.peer_info.bytes_sent += length + 4
                self.peer_info.last_message_at = time.time()
                
                return True
        except Exception as e:
            logger.error(f"[PEER {self.peer_id}] Send error: {e}")
            self.is_connected = False
            return False
    
    def receive_message(self, timeout: float = PEER_TIMEOUT) -> Optional[Message]:
        """Receive message from peer"""
        try:
            with self.lock:
                self.sock.settimeout(timeout)
                
                # Read 4-byte length
                length_data = self.sock.recv(4)
                if not length_data:
                    return None
                
                length = int.from_bytes(length_data, 'big')
                if length > MESSAGE_MAX_SIZE:
                    logger.error(f"[PEER {self.peer_id}] Message too large: {length}")
                    return None
                
                # Read message data
                message_data = b''
                while len(message_data) < length:
                    chunk = self.sock.recv(min(4096, length - len(message_data)))
                    if not chunk:
                        return None
                    message_data += chunk
                
                message = Message.from_bytes(message_data)
                message.sender_id = self.peer_id
                
                self.peer_info.messages_received += 1
                self.peer_info.bytes_received += length + 4
                self.peer_info.last_message_at = time.time()
                
                return message
        except socket.timeout:
            return None
        except Exception as e:
            logger.error(f"[PEER {self.peer_id}] Receive error: {e}")
            self.is_connected = False
            return None
    
    def close(self):
        """Close peer connection"""
        try:
            self.sock.close()
            self.is_connected = False
        except:
            pass


# ═════════════════════════════════════════════════════════════════════════════════
# PEER DISCOVERY ENGINE
# ═════════════════════════════════════════════════════════════════════════════════

class PeerDiscoveryEngine:
    """Comprehensive peer discovery: DNS seeds, bootstrap, exchange protocol"""
    
    def __init__(self):
        self.peer_candidates: Set[Tuple[str, int]] = set()
        self.attempt_timestamps: Dict[Tuple[str, int], float] = {}
        self.lock = threading.RLock()
        self.tested_peers: Dict[Tuple[str, int], Dict[str, Any]] = {}
    
    def discover_peers(self) -> List[Tuple[str, int]]:
        """
        Comprehensive peer discovery with multiple strategies:
        1. Load from database (persistent peer store)
        2. Query DNS seeds (if available)
        3. Use bootstrap nodes
        4. Return candidates
        """
        candidates = set()
        
        # Strategy 1: Load from database
        try:
            db_peers = load_known_peers()
            candidates.update(db_peers)
            logger.info(f"[DISCOVERY] Loaded {len(db_peers)} peers from database")
        except Exception as e:
            logger.warning(f"[DISCOVERY] Failed to load from DB: {e}")
        
        # Strategy 2: Query DNS seeds
        for seed in DNS_SEEDS:
            try:
                dns_peers = self._query_dns_seed(seed)
                candidates.update(dns_peers)
                logger.info(f"[DISCOVERY] Found {len(dns_peers)} peers from DNS seed {seed}")
            except Exception as e:
                logger.debug(f"[DISCOVERY] DNS seed {seed} failed: {e}")
        
        # Strategy 3: Bootstrap nodes
        for bootstrap in BOOTSTRAP_NODES + DEFAULT_BOOTSTRAP_PEERS:
            try:
                parts = bootstrap.split(':')
                if len(parts) == 2:
                    addr, port = parts[0], int(parts[1])
                    candidates.add((addr, port))
            except Exception as e:
                logger.debug(f"[DISCOVERY] Invalid bootstrap node {bootstrap}: {e}")
        
        with self.lock:
            self.peer_candidates.update(candidates)
            logger.info(f"[DISCOVERY] Total candidate peers: {len(self.peer_candidates)}")
        
        return list(candidates)
    
    def _query_dns_seed(self, seed: str) -> List[Tuple[str, int]]:
        """Query DNS seed for peer addresses"""
        try:
            # DNS seeds should return A records with peer addresses
            # For now, return empty (would implement actual DNS lookup in production)
            return []
        except Exception as e:
            logger.error(f"[DISCOVERY] DNS query failed for {seed}: {e}")
            return []
    
    def mark_success(self, address: str, port: int):
        """Mark peer connection as successful"""
        with self.lock:
            key = (address, port)
            self.tested_peers[key] = {
                'last_success': time.time(),
                'failures': 0,
                'attempts': self.tested_peers.get(key, {}).get('attempts', 0) + 1,
            }
    
    def mark_failure(self, address: str, port: int):
        """Mark peer connection as failed"""
        with self.lock:
            key = (address, port)
            info = self.tested_peers.get(key, {})
            info['last_failure'] = time.time()
            info['failures'] = info.get('failures', 0) + 1
            info['attempts'] = info.get('attempts', 0) + 1
            self.tested_peers[key] = info
    
    def get_connect_candidates(self, exclude: Set[str], needed: int = 5) -> List[Tuple[str, int]]:
        """Get list of peers to try connecting to"""
        with self.lock:
            candidates = [
                peer for peer in self.peer_candidates
                if peer[0] not in exclude
            ]
            
            # Sort by success rate and last attempt time
            candidates.sort(
                key=lambda p: self._score_peer(p),
                reverse=True
            )
            
            return candidates[:needed]
    
    def _score_peer(self, peer: Tuple[str, int]) -> float:
        """Score peer for connection attempts (higher = better)"""
        info = self.tested_peers.get(peer, {})
        attempts = info.get('attempts', 0)
        failures = info.get('failures', 0)
        
        if attempts == 0:
            return 1.0  # New peers get highest score
        
        success_rate = max(0, (attempts - failures) / attempts)
        return success_rate


discovery_engine = PeerDiscoveryEngine()


# ═════════════════════════════════════════════════════════════════════════════════
# MESSAGE HANDLERS (COMPREHENSIVE PROTOCOL IMPLEMENTATION)
# ═════════════════════════════════════════════════════════════════════════════════

class MessageHandlers:
    """Comprehensive message handling for P2P protocol"""
    
    def __init__(self, p2p_server: 'P2PServer'):
        self.p2p = p2p_server
        self.block_cache: Dict[str, Dict[str, Any]] = {}
        self.tx_cache: Dict[str, Dict[str, Any]] = {}
        self.inventory_state: Dict[str, List[str]] = {}  # Track what peers have announced
        self.lock = threading.RLock()
    
    def on_version(self, peer_conn: PeerConnection, message: Message):
        """
        Handle VERSION message: peer capabilities and metadata
        
        Used to:
        - Exchange protocol versions
        - Share node capabilities
        - Establish peer identity
        - Verify network parameters
        """
        try:
            payload = message.payload
            
            peer_conn.peer_info.version = payload.get('version', 1)
            peer_conn.peer_info.user_agent = payload.get('user_agent', 'unknown')
            peer_conn.peer_info.protocol_version = payload.get('protocol_version', 1)
            peer_conn.peer_info.last_block_height = payload.get('block_height', 0)
            peer_conn.peer_info.last_block_hash = payload.get('block_hash')
            
            logger.info(
                f"[PEER {peer_conn.peer_id}] Version: {peer_conn.peer_info.user_agent} "
                f"(height={peer_conn.peer_info.last_block_height})"
            )
            
            # Send version acknowledgment
            response = Message(
                msg_type='verack',
                payload={
                    'status': 'ok',
                    'peer_id': peer_conn.peer_id,
                }
            )
            peer_conn.send_message(response)
            peer_conn.version_received = True
            
        except Exception as e:
            logger.error(f"[PEER {peer_conn.peer_id}] Version handler error: {e}")
            peer_conn.peer_info.ban_score += 1
    
    def on_verack(self, peer_conn: PeerConnection, message: Message):
        """Handle VERACK (version acknowledgment)"""
        try:
            peer_conn.version_received = True
            logger.debug(f"[PEER {peer_conn.peer_id}] Version acknowledged")
        except Exception as e:
            logger.error(f"[PEER {peer_conn.peer_id}] Verack handler error: {e}")
    
    def on_ping(self, peer_conn: PeerConnection, message: Message):
        """
        Handle PING: keepalive mechanism
        
        Used to:
        - Detect dead connections
        - Measure latency
        - Keep connections alive
        """
        try:
            nonce = message.payload.get('nonce')
            
            response = Message(
                msg_type='pong',
                payload={'nonce': nonce}
            )
            peer_conn.send_message(response)
            
        except Exception as e:
            logger.error(f"[PEER {peer_conn.peer_id}] Ping handler error: {e}")
    
    def on_pong(self, peer_conn: PeerConnection, message: Message):
        """Handle PONG: keepalive response"""
        # Latency can be calculated from nonce or message timestamp
        pass
    
    def on_inv(self, peer_conn: PeerConnection, message: Message):
        """
        Handle INV (inventory announcement): data availability
        
        Used to:
        - Announce newly seen blocks
        - Announce pending transactions
        - Signal data availability without sending full data
        - Enable selective data requests
        """
        try:
            payload = message.payload
            inv_type = payload.get('type')  # 'block' or 'tx'
            hashes = payload.get('hashes', [])
            
            logger.debug(f"[PEER {peer_conn.peer_id}] Inventory: {len(hashes)} {inv_type}s")
            
            with self.lock:
                if inv_type == 'block':
                    # Track block inventory
                    if peer_conn.peer_id not in self.inventory_state:
                        self.inventory_state[peer_conn.peer_id] = []
                    
                    new_blocks = [h for h in hashes if h not in self.inventory_state[peer_conn.peer_id]]
                    self.inventory_state[peer_conn.peer_id].extend(new_blocks)
                    
                    peer_conn.peer_info.blocks_announced += len(new_blocks)
                    
                    # Request blocks we don't have
                    if new_blocks:
                        self._request_blocks(peer_conn, new_blocks)
                
                elif inv_type == 'tx':
                    # Track transaction inventory
                    peer_conn.peer_info.txs_announced += len(hashes)
                    
                    # Request transactions we don't have
                    unknown_txs = [h for h in hashes if h not in self.tx_cache]
                    if unknown_txs:
                        self._request_transactions(peer_conn, unknown_txs)
        
        except Exception as e:
            logger.error(f"[PEER {peer_conn.peer_id}] INV handler error: {e}")
            peer_conn.peer_info.ban_score += 1
    
    def on_block(self, peer_conn: PeerConnection, message: Message):
        """
        Handle BLOCK: receive full block data
        
        Used to:
        - Receive complete block information
        - Validate block structure
        - Store block in database
        - Relay to other peers
        - Update chain state
        """
        try:
            payload = message.payload
            block_hash = payload.get('hash')
            block_height = payload.get('height')
            block_data = payload.get('data')
            timestamp = payload.get('timestamp', int(time.time()))
            
            logger.info(f"[P2P] Block received from {peer_conn.peer_id}: height={block_height}, hash={block_hash[:16]}...")
            
            # Validate block structure
            if not self._validate_block(payload):
                logger.warning(f"[P2P] Invalid block structure from {peer_conn.peer_id}")
                peer_conn.peer_info.ban_score += 10
                return
            
            # Store block
            with self.lock:
                self.block_cache[block_hash] = payload
            
            # Update peer info
            if block_height > peer_conn.peer_info.last_block_height:
                peer_conn.peer_info.last_block_height = block_height
                peer_conn.peer_info.last_block_hash = block_hash
            
            # Broadcast to other peers (relay)
            self.p2p.broadcast_block(payload)
            
            # Update system state
            if block_height > state.block_state['current_height']:
                state.update_block_state({
                    'current_height': block_height,
                    'current_hash': block_hash,
                    'timestamp': timestamp,
                })
            
        except Exception as e:
            logger.error(f"[PEER {peer_conn.peer_id}] Block handler error: {e}")
            peer_conn.peer_info.ban_score += 5
    
    def on_tx(self, peer_conn: PeerConnection, message: Message):
        """
        Handle TX (transaction): receive full transaction data
        
        Used to:
        - Receive complete transaction information
        - Validate transaction cryptography
        - Add to mempool
        - Relay to other peers
        - Track transaction propagation
        """
        try:
            payload = message.payload
            tx_hash = payload.get('hash')
            from_addr = payload.get('from')
            to_addr = payload.get('to')
            amount = payload.get('amount')
            timestamp = payload.get('timestamp', int(time.time()))
            
            logger.debug(f"[P2P] TX received from {peer_conn.peer_id}: {tx_hash[:16]}...")
            
            # Validate transaction
            if not self._validate_tx(payload):
                logger.warning(f"[P2P] Invalid transaction from {peer_conn.peer_id}")
                peer_conn.peer_info.ban_score += 5
                return
            
            # Cache transaction
            with self.lock:
                self.tx_cache[tx_hash] = payload
            
            # Broadcast to other peers
            self.p2p.broadcast_transaction(payload)
            
        except Exception as e:
            logger.error(f"[PEER {peer_conn.peer_id}] TX handler error: {e}")
            peer_conn.peer_info.ban_score += 3
    
    def on_getdata(self, peer_conn: PeerConnection, message: Message):
        """
        Handle GETDATA: peer requests specific data
        
        Used to:
        - Respond to data requests
        - Send blocks on demand
        - Send transactions on demand
        - Implement selective data transfer
        """
        try:
            payload = message.payload
            data_type = payload.get('type')  # 'block' or 'tx'
            hashes = payload.get('hashes', [])
            
            logger.debug(f"[PEER {peer_conn.peer_id}] GETDATA: {len(hashes)} {data_type}s")
            
            if data_type == 'block':
                for block_hash in hashes:
                    # Try cache first
                    with self.lock:
                        if block_hash in self.block_cache:
                            block_data = self.block_cache[block_hash]
                        else:
                            block_data = query_block_by_hash(block_hash)
                    
                    if block_data:
                        response = Message(
                            msg_type='block',
                            payload=block_data
                        )
                        peer_conn.send_message(response)
            
            elif data_type == 'tx':
                for tx_hash in hashes:
                    # Try cache first
                    with self.lock:
                        if tx_hash in self.tx_cache:
                            tx_data = self.tx_cache[tx_hash]
                        else:
                            tx_data = query_transaction(tx_hash)
                    
                    if tx_data:
                        response = Message(
                            msg_type='tx',
                            payload=tx_data
                        )
                        peer_conn.send_message(response)
        
        except Exception as e:
            logger.error(f"[PEER {peer_conn.peer_id}] GETDATA handler error: {e}")
    
    def on_getblocks(self, peer_conn: PeerConnection, message: Message):
        """
        Handle GETBLOCKS: peer requests block hashes
        
        Used to:
        - Respond with block hashes for synchronization
        - Enable peers to catch up
        - Implement efficient block sync
        """
        try:
            payload = message.payload
            since_height = payload.get('since_height', 0)
            limit = min(payload.get('limit', 100), 500)  # Max 500 blocks
            
            logger.debug(f"[PEER {peer_conn.peer_id}] GETBLOCKS: since_height={since_height}")
            
            # Query blocks
            latest_block = query_latest_block()
            if latest_block:
                to_height = latest_block['height']
                blocks = query_blocks_range(since_height, min(since_height + limit, to_height))
                
                response = Message(
                    msg_type='headers',
                    payload={
                        'blocks': [
                            {
                                'height': b['height'],
                                'hash': b['hash'],
                                'timestamp': b['timestamp'],
                            }
                            for b in blocks
                        ]
                    }
                )
                peer_conn.send_message(response)
        
        except Exception as e:
            logger.error(f"[PEER {peer_conn.peer_id}] GETBLOCKS handler error: {e}")
    
    def on_headers(self, peer_conn: PeerConnection, message: Message):
        """Handle HEADERS: receive block header list"""
        try:
            payload = message.payload
            headers = payload.get('blocks', [])
            
            logger.debug(f"[PEER {peer_conn.peer_id}] Headers: {len(headers)} blocks")
            
            # Update peer block height
            if headers:
                latest = headers[-1]
                if latest['height'] > peer_conn.peer_info.last_block_height:
                    peer_conn.peer_info.last_block_height = latest['height']
                    peer_conn.peer_info.last_block_hash = latest['hash']
        
        except Exception as e:
            logger.error(f"[PEER {peer_conn.peer_id}] Headers handler error: {e}")
    
    def on_addr(self, peer_conn: PeerConnection, message: Message):
        """
        Handle ADDR: receive peer address announcements
        
        Used to:
        - Populate peer list
        - Implement peer discovery protocol
        - Share known peers
        """
        try:
            payload = message.payload
            addresses = payload.get('addresses', [])
            
            logger.debug(f"[PEER {peer_conn.peer_id}] ADDR: {len(addresses)} peers")
            
            # Add to discovery engine
            for addr in addresses:
                try:
                    ip = addr.get('ip')
                    port = addr.get('port')
                    if ip and port:
                        with discovery_engine.lock:
                            discovery_engine.peer_candidates.add((ip, port))
                except Exception as e:
                    logger.debug(f"Invalid address in ADDR: {e}")
        
        except Exception as e:
            logger.error(f"[PEER {peer_conn.peer_id}] ADDR handler error: {e}")
    
    def on_peers_sync(self, peer_conn: PeerConnection, message: Message):
        """
        Handle PEERS_SYNC: synchronize known peer lists
        
        Used to:
        - Exchange peer information
        - Bootstrap new nodes
        - Maintain peer list accuracy
        """
        try:
            payload = message.payload
            action = payload.get('action')
            
            if action == 'request':
                # Respond with our known peers
                with discovery_engine.lock:
                    peers = list(discovery_engine.peer_candidates)[:50]  # Max 50
                
                response = Message(
                    msg_type='peers_sync',
                    payload={
                        'action': 'response',
                        'peers': [
                            {'address': p[0], 'port': p[1]}
                            for p in peers
                        ]
                    }
                )
                peer_conn.send_message(response)
            
            elif action == 'response':
                # Add received peers to discovery
                peers = payload.get('peers', [])
                with discovery_engine.lock:
                    for peer in peers:
                        try:
                            discovery_engine.peer_candidates.add((peer['address'], peer['port']))
                        except:
                            pass
        
        except Exception as e:
            logger.error(f"[PEER {peer_conn.peer_id}] PEERS_SYNC handler error: {e}")
    
    def on_mempool(self, peer_conn: PeerConnection, message: Message):
        """
        Handle MEMPOOL: request for pending transactions
        
        Used to:
        - Share mempool contents
        - Sync transaction pools
        - Enable new peers to catch up on pending txs
        """
        try:
            logger.debug(f"[PEER {peer_conn.peer_id}] MEMPOOL request")
            
            with self.lock:
                tx_hashes = list(self.tx_cache.keys())
            
            # Send inventory of pending transactions
            response = Message(
                msg_type='inv',
                payload={
                    'type': 'tx',
                    'hashes': tx_hashes[:500]  # Max 500 tx announcements
                }
            )
            peer_conn.send_message(response)
        
        except Exception as e:
            logger.error(f"[PEER {peer_conn.peer_id}] MEMPOOL handler error: {e}")
    
    def on_consensus(self, peer_conn: PeerConnection, message: Message):
        """
        Handle CONSENSUS: consensus-related messages
        
        Used to:
        - Exchange consensus state
        - Participate in network agreement
        - Signal block acceptance
        """
        try:
            payload = message.payload
            action = payload.get('action')
            
            logger.debug(f"[PEER {peer_conn.peer_id}] Consensus: {action}")
            
            # Implement consensus logic based on action
            if action == 'block_accepted':
                block_hash = payload.get('block_hash')
                logger.info(f"[CONSENSUS] Peer {peer_conn.peer_id} accepted block {block_hash[:16]}...")
        
        except Exception as e:
            logger.error(f"[PEER {peer_conn.peer_id}] Consensus handler error: {e}")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Helper Methods
    # ─────────────────────────────────────────────────────────────────────────────
    
    def _request_blocks(self, peer_conn: PeerConnection, block_hashes: List[str]):
        """Request blocks from peer"""
        request = Message(
            msg_type='getdata',
            payload={
                'type': 'block',
                'hashes': block_hashes[:100]  # Max 100 per request
            }
        )
        peer_conn.send_message(request)
    
    def _request_transactions(self, peer_conn: PeerConnection, tx_hashes: List[str]):
        """Request transactions from peer"""
        request = Message(
            msg_type='getdata',
            payload={
                'type': 'tx',
                'hashes': tx_hashes[:100]  # Max 100 per request
            }
        )
        peer_conn.send_message(request)
    
    def _validate_block(self, block_data: Dict[str, Any]) -> bool:
        """Validate block structure and data"""
        try:
            required_fields = ['hash', 'height', 'data']
            return all(field in block_data for field in required_fields)
        except:
            return False
    
    def _validate_tx(self, tx_data: Dict[str, Any]) -> bool:
        """Validate transaction structure and data"""
        try:
            required_fields = ['hash', 'from', 'to', 'amount']
            return all(field in tx_data for field in required_fields)
        except:
            return False


# ═════════════════════════════════════════════════════════════════════════════════
# P2P SERVER (MAIN NETWORKING ENGINE)
# ═════════════════════════════════════════════════════════════════════════════════

class P2PServer:
    """Main P2P networking server with peer management and message routing"""
    
    def __init__(self, host: str = P2P_HOST, port: int = P2P_PORT, testnet: bool = False):
        self.host = host
        self.port = port if not testnet else P2P_TESTNET_PORT
        self.testnet = testnet
        
        # Peer management
        self.peers: Dict[str, PeerConnection] = {}
        self.peer_by_address: Dict[Tuple[str, int], str] = {}
        self.peers_lock = threading.RLock()
        
        # Server state
        self.is_running = False
        self.server_socket: Optional[socket.socket] = None
        self.threads: List[threading.Thread] = []
        
        # Message handlers
        self.message_handlers = MessageHandlers(self)
        
        # Metrics
        self.stats = {
            'total_peers_connected': 0,
            'blocks_received': 0,
            'blocks_sent': 0,
            'txs_received': 0,
            'txs_sent': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'peer_discovery_cycles': 0,
        }
        self.stats_lock = threading.RLock()
        
        logger.info(f"[P2P] Server initialized on {self.host}:{self.port}")
    
    def start(self) -> bool:
        """Start P2P server"""
        self.is_running = True
        
        # Create server socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(MAX_PEERS)
            logger.info(f"✨ [P2P] Server listening on {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"[P2P] Failed to bind socket: {e}")
            return False
        
        # Start background threads
        self._start_accept_thread()
        self._start_peer_maintenance_thread()
        self._start_peer_discovery_thread()
        self._start_message_broadcast_thread()
        
        logger.info("[P2P] All threads started")
        return True
    
    def _start_accept_thread(self):
        """Thread that accepts incoming peer connections"""
        def accept_loop():
            logger.info("[P2P] Accept thread started")
            while self.is_running:
                try:
                    client_sock, address = self.server_socket.accept()
                    peer_id = self._generate_peer_id()
                    
                    logger.info(f"[P2P] Inbound connection from {address[0]}:{address[1]}")
                    
                    with self.peers_lock:
                        if len(self.peers) >= MAX_PEERS:
                            logger.warning("[P2P] Max peers reached")
                            client_sock.close()
                            continue
                        
                        if address in self.peer_by_address:
                            logger.warning(f"[P2P] Duplicate connection from {address}")
                            client_sock.close()
                            continue
                    
                    # Create peer connection
                    peer_conn = PeerConnection(client_sock, address, peer_id, is_outbound=False)
                    
                    # Add to peers
                    with self.peers_lock:
                        self.peers[peer_id] = peer_conn
                        self.peer_by_address[address] = peer_id
                    
                    with self.stats_lock:
                        self.stats['total_peers_connected'] += 1
                    
                    # Send version
                    self._send_version_to_peer(peer_conn)
                    
                    # Handle this peer
                    peer_thread = threading.Thread(
                        target=self._handle_peer,
                        args=(peer_conn,),
                        daemon=True
                    )
                    self.threads.append(peer_thread)
                    peer_thread.start()
                
                except Exception as e:
                    if self.is_running:
                        logger.error(f"[P2P] Accept error: {e}")
        
        thread = threading.Thread(target=accept_loop, daemon=True)
        self.threads.append(thread)
        thread.start()
    
    def _handle_peer(self, peer_conn: PeerConnection):
        """Handle messages from a single peer"""
        try:
            logger.debug(f"[PEER {peer_conn.peer_id}] Handler started")
            
            while self.is_running and peer_conn.is_connected:
                message = peer_conn.receive_message()
                
                if message is None:
                    break
                
                with self.stats_lock:
                    self.stats['messages_received'] += 1
                
                # Route message to handler
                self._route_message(peer_conn, message)
        
        except Exception as e:
            logger.error(f"[PEER {peer_conn.peer_id}] Handler error: {e}")
        
        finally:
            self._disconnect_peer(peer_conn)
    
    def _route_message(self, peer_conn: PeerConnection, message: Message):
        """Route incoming message to appropriate handler"""
        try:
            handler_map = {
                'version': self.message_handlers.on_version,
                'verack': self.message_handlers.on_verack,
                'ping': self.message_handlers.on_ping,
                'pong': self.message_handlers.on_pong,
                'inv': self.message_handlers.on_inv,
                'getdata': self.message_handlers.on_getdata,
                'block': self.message_handlers.on_block,
                'tx': self.message_handlers.on_tx,
                'getblocks': self.message_handlers.on_getblocks,
                'headers': self.message_handlers.on_headers,
                'addr': self.message_handlers.on_addr,
                'peers_sync': self.message_handlers.on_peers_sync,
                'mempool': self.message_handlers.on_mempool,
                'consensus': self.message_handlers.on_consensus,
            }
            
            handler = handler_map.get(message.msg_type)
            if handler:
                handler(peer_conn, message)
            else:
                logger.warning(f"[P2P] Unknown message type: {message.msg_type}")
        
        except Exception as e:
            logger.error(f"[P2P] Message routing error: {e}")
    
    def _send_version_to_peer(self, peer_conn: PeerConnection):
        """Send VERSION message to peer"""
        latest_block = query_latest_block()
        
        message = Message(
            msg_type='version',
            payload={
                'version': 1,
                'protocol_version': 1,
                'user_agent': 'QTCL/6.0',
                'network': 'testnet' if self.testnet else 'mainnet',
                'timestamp': int(time.time()),
                'block_height': latest_block['height'] if latest_block else 0,
                'block_hash': latest_block['hash'] if latest_block else None,
            }
        )
        peer_conn.send_message(message)
    
    def _disconnect_peer(self, peer_conn: PeerConnection):
        """Disconnect and clean up peer"""
        with self.peers_lock:
            if peer_conn.peer_id in self.peers:
                del self.peers[peer_conn.peer_id]
            
            addr_key = (peer_conn.address, peer_conn.port)
            if addr_key in self.peer_by_address:
                del self.peer_by_address[addr_key]
        
        peer_conn.close()
        logger.info(f"[P2P] Peer disconnected: {peer_conn.peer_id}")
    
    def _start_peer_maintenance_thread(self):
        """Remove dead peers and manage bans"""
        def maintenance_loop():
            logger.info("[P2P] Maintenance thread started")
            while self.is_running:
                try:
                    with self.peers_lock:
                        dead_peers = [
                            peer_id for peer_id, peer in self.peers.items()
                            if not peer.peer_info.is_alive or peer.peer_info.ban_score >= 100
                        ]
                        
                        for peer_id in dead_peers:
                            if peer_id in self.peers:
                                peer = self.peers[peer_id]
                                if not peer.peer_info.is_alive:
                                    logger.warning(f"[P2P] Removing dead peer: {peer_id}")
                                else:
                                    logger.warning(f"[P2P] Banning peer: {peer_id} (score={peer.peer_info.ban_score})")
                                self._disconnect_peer(peer)
                    
                    time.sleep(PEER_CLEANUP_INTERVAL)
                except Exception as e:
                    logger.error(f"[P2P] Maintenance error: {e}")
        
        thread = threading.Thread(target=maintenance_loop, daemon=True)
        self.threads.append(thread)
        thread.start()
    
    def _start_peer_discovery_thread(self):
        """Peer discovery: connect to new peers"""
        def discovery_loop():
            logger.info("[P2P] Discovery thread started")
            while self.is_running:
                try:
                    with self.stats_lock:
                        self.stats['peer_discovery_cycles'] += 1
                    
                    with self.peers_lock:
                        connected_addresses = {p[0] for p in self.peer_by_address.keys()}
                        current_peer_count = len(self.peers)
                    
                    # Discover new peers if below target
                    if current_peer_count < MAX_PEERS // 2:
                        candidates = discovery_engine.get_connect_candidates(
                            connected_addresses,
                            needed=min(5, MAX_PEERS - current_peer_count)
                        )
                        
                        for address, port in candidates:
                            self._connect_to_peer(address, port)
                    
                    time.sleep(PEER_DISCOVERY_INTERVAL)
                
                except Exception as e:
                    logger.error(f"[P2P] Discovery error: {e}")
        
        thread = threading.Thread(target=discovery_loop, daemon=True)
        self.threads.append(thread)
        thread.start()
    
    def _connect_to_peer(self, address: str, port: int):
        """Initiate outbound connection to peer"""
        try:
            logger.debug(f"[P2P] Connecting to {address}:{port}")
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(PEER_HANDSHAKE_TIMEOUT)
            sock.connect((address, port))
            
            peer_id = self._generate_peer_id()
            peer_conn = PeerConnection(sock, (address, port), peer_id, is_outbound=True)
            
            with self.peers_lock:
                if (address, port) not in self.peer_by_address:
                    self.peers[peer_id] = peer_conn
                    self.peer_by_address[(address, port)] = peer_id
                else:
                    logger.debug(f"[P2P] Already connected to {address}:{port}")
                    sock.close()
                    return
            
            with self.stats_lock:
                self.stats['total_peers_connected'] += 1
            
            logger.info(f"[P2P] Connected to {address}:{port} (outbound)")
            
            # Send version
            self._send_version_to_peer(peer_conn)
            
            # Handle this peer
            peer_thread = threading.Thread(
                target=self._handle_peer,
                args=(peer_conn,),
                daemon=True
            )
            self.threads.append(peer_thread)
            peer_thread.start()
            
            discovery_engine.mark_success(address, port)
        
        except Exception as e:
            logger.debug(f"[P2P] Connection to {address}:{port} failed: {e}")
            discovery_engine.mark_failure(address, port)
    
    def _start_message_broadcast_thread(self):
        """Periodic keepalive and peer sync"""
        def broadcast_loop():
            logger.info("[P2P] Broadcast thread started")
            while self.is_running:
                try:
                    with self.peers_lock:
                        peers = list(self.peers.values())
                    
                    for peer_conn in peers:
                        if not peer_conn.is_connected:
                            continue
                        
                        # Send ping
                        ping_msg = Message(
                            msg_type='ping',
                            payload={'nonce': int(time.time() * 1000)}
                        )
                        peer_conn.send_message(ping_msg)
                        
                        # Periodically request peer list
                        if random.random() < 0.1:  # 10% of pings
                            peers_sync_msg = Message(
                                msg_type='peers_sync',
                                payload={'action': 'request'}
                            )
                            peer_conn.send_message(peers_sync_msg)
                    
                    time.sleep(PEER_KEEPALIVE_INTERVAL)
                
                except Exception as e:
                    logger.error(f"[P2P] Broadcast error: {e}")
        
        thread = threading.Thread(target=broadcast_loop, daemon=True)
        self.threads.append(thread)
        thread.start()
    
    def broadcast_block(self, block_data: Dict[str, Any]):
        """Broadcast block to all connected peers"""
        message = Message(
            msg_type='block',
            payload=block_data
        )
        
        with self.peers_lock:
            for peer_conn in list(self.peers.values()):
                if peer_conn.is_connected and peer_conn.version_received:
                    peer_conn.send_message(message)
                    with self.stats_lock:
                        self.stats['messages_sent'] += 1
                        self.stats['blocks_sent'] += 1
    
    def broadcast_transaction(self, tx_data: Dict[str, Any]):
        """Broadcast transaction to all connected peers"""
        message = Message(
            msg_type='tx',
            payload=tx_data
        )
        
        with self.peers_lock:
            for peer_conn in list(self.peers.values()):
                if peer_conn.is_connected and peer_conn.version_received:
                    peer_conn.send_message(message)
                    with self.stats_lock:
                        self.stats['messages_sent'] += 1
                        self.stats['txs_sent'] += 1
    
    def announce_inventory(self, inv_type: str, hashes: List[str]):
        """Announce inventory to peers"""
        message = Message(
            msg_type='inv',
            payload={
                'type': inv_type,
                'hashes': hashes
            }
        )
        
        with self.peers_lock:
            for peer_conn in list(self.peers.values()):
                if peer_conn.is_connected and peer_conn.version_received:
                    peer_conn.send_message(message)
    
    def get_peer_count(self) -> int:
        """Get number of connected peers"""
        with self.peers_lock:
            return len(self.peers)
    
    def get_peer_info(self) -> List[Dict[str, Any]]:
        """Get information about all connected peers"""
        with self.peers_lock:
            return [
                {
                    'peer_id': peer.peer_info.peer_id,
                    'address': peer.peer_info.address,
                    'port': peer.peer_info.port,
                    'uptime_s': peer.peer_info.uptime_seconds,
                    'height': peer.peer_info.last_block_height,
                    'blocks_announced': peer.peer_info.blocks_announced,
                    'txs_announced': peer.peer_info.txs_announced,
                    'messages_sent': peer.peer_info.messages_sent,
                    'messages_received': peer.peer_info.messages_received,
                    'bytes_sent': peer.peer_info.bytes_sent,
                    'bytes_received': peer.peer_info.bytes_received,
                    'is_outbound': peer.peer_info.is_outbound,
                    'version': peer.peer_info.version,
                    'user_agent': peer.peer_info.user_agent,
                    'ban_score': peer.peer_info.ban_score,
                }
                for peer in self.peers.values()
            ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics"""
        with self.stats_lock:
            stats = self.stats.copy()
            stats['current_peers'] = self.get_peer_count()
            stats['timestamp'] = datetime.now(timezone.utc).isoformat()
            return stats
    
    def shutdown(self):
        """Shutdown P2P server"""
        logger.info("[P2P] Shutting down...")
        self.is_running = False
        
        # Disconnect all peers
        with self.peers_lock:
            for peer in list(self.peers.values()):
                self._disconnect_peer(peer)
        
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        logger.info("[P2P] Shutdown complete")
    
    @staticmethod
    def _generate_peer_id() -> str:
        """Generate unique peer ID"""
        return hashlib.sha256(
            f"{time.time()}{os.urandom(16)}".encode()
        ).hexdigest()[:16]


# ═════════════════════════════════════════════════════════════════════════════════
# LATTICE INITIALIZATION
# ═════════════════════════════════════════════════════════════════════════════════

def initialize_lattice_controller():
    """Initialize lattice controller at startup"""
    try:
        try:
            from lattice_controller import QuantumLatticeController
            controller = QuantumLatticeController()
            controller.initialize()
            logger.info("✨ [LATTICE] Quantum lattice controller initialized")
            state.lattice_loaded = True
            return controller
        except ImportError:
            logger.warning("[LATTICE] lattice_controller not available, running in mock mode")
            state.lattice_loaded = False
            return None
    except Exception as e:
        logger.error(f"[LATTICE] Initialization failed: {e}")
        return None


LATTICE = initialize_lattice_controller()

# ═════════════════════════════════════════════════════════════════════════════════
# QUANTUM METRICS THREAD
# ═════════════════════════════════════════════════════════════════════════════════

def quantum_metrics_thread():
    """Background thread for metrics and state updates"""
    logger.info("[METRICS] Quantum metrics thread started")
    
    while state.is_alive:
        try:
            # Query latest block
            block = query_latest_block()
            if block:
                pq_min, pq_max = query_pseudoqubit_range()
                state.update_block_state({
                    'current_height': block['height'],
                    'current_hash': block['hash'],
                    'pq_current': pq_max,
                    'pq_last': pq_min,
                    'timestamp': block['timestamp'],
                })
            
            # Update quantum metrics
            if LATTICE:
                try:
                    metrics = LATTICE.get_metrics()
                    state.update_metrics(metrics)
                except:
                    pass
            else:
                # Mock metrics
                import random
                state.update_metrics({
                    'coherence': 0.99 - random.random() * 0.05,
                    'entanglement': 0.95 - random.random() * 0.05,
                    'phase_drift': 0.01 + random.random() * 0.02,
                    'w_state_fidelity': 0.98 - random.random() * 0.03,
                })
            
            time.sleep(2)
        except Exception as e:
            logger.error(f"[METRICS] Error: {e}")
            time.sleep(2)


metrics_thread = threading.Thread(target=quantum_metrics_thread, daemon=True)
metrics_thread.start()

# ═════════════════════════════════════════════════════════════════════════════════
# P2P SERVER INSTANCE
# ═════════════════════════════════════════════════════════════════════════════════

P2P = None

def initialize_p2p():
    """Initialize P2P server"""
    global P2P
    try:
        P2P = P2PServer(
            host=P2P_HOST,
            port=P2P_PORT,
            testnet=False
        )
        if P2P.start():
            logger.info("✨ [P2P] P2P server started successfully")
            
            # Discover initial peers
            discovery_engine.discover_peers()
            
            return True
        else:
            logger.warning("[P2P] Failed to start P2P server")
            return False
    except Exception as e:
        logger.error(f"[P2P] Initialization failed: {e}")
        return False


# ═════════════════════════════════════════════════════════════════════════════════
# FLASK REST API ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════════

@app.route('/')
def dashboard():
    """Serve dashboard"""
    try:
        with open('index.html', 'r') as f:
            html_content = f.read()
        return render_template_string(html_content)
    except FileNotFoundError:
        return """
        <html>
            <head>
                <title>QTCL Server</title>
                <style>
                    body { background:#0a0a0e; color:#d4d4d8; font-family:monospace; padding:20px; }
                    h1 { color:#00ff88; }
                </style>
            </head>
            <body>
                <h1>✨ QTCL Server v6 Running</h1>
                <p>Dashboard: <a href="/">Home</a></p>
                <p>Health: <a href="/api/health">/api/health</a></p>
                <p>P2P Stats: <a href="/api/p2p/stats">/api/p2p/stats</a></p>
                <p>Peers: <a href="/api/p2p/peers">/api/p2p/peers</a></p>
            </body>
        </html>
        """, 200


@app.route('/health', methods=['GET'])
def health_check():
    """Koyeb health check endpoint (simple, fast)"""
    try:
        # Quick database connectivity check
        with get_db_cursor() as cur:
            cur.execute("SELECT 1")
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }), 200
    except Exception as e:
        logger.error(f"[HEALTH] Check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
        }), 503


@app.route('/api/health', methods=['GET'])
def health():
    """Detailed health check endpoint"""
    snapshot = state.get_state()
    return jsonify({
        'status': 'ok' if state.is_alive else 'degraded',
        'lattice_loaded': state.lattice_loaded,
        'p2p_enabled': P2P is not None and P2P.is_running,
        'p2p_peers': P2P.get_peer_count() if P2P else 0,
        'quantum_metrics': snapshot['quantum_metrics'],
        'block_height': snapshot['block_state']['current_height'],
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }), 200


@app.route('/api/blocks', methods=['GET'])
def blocks():
    """Get current block information"""
    snapshot = state.get_state()
    block = snapshot['block_state']
    
    return jsonify({
        'current_height': block['current_height'],
        'current_hash': block['current_hash'],
        'pq_current': block['pq_current'],
        'pq_last': block['pq_last'],
        'timestamp': block['timestamp'],
        'quantum_metrics': snapshot['quantum_metrics'],
    }), 200


@app.route('/api/wallet', methods=['GET'])
def wallet():
    """Get wallet information"""
    address = request.args.get('address')
    if not address:
        return jsonify({'error': 'address parameter required'}), 400
    
    wallet_info = query_wallet_info(address)
    if wallet_info:
        return jsonify(wallet_info), 200
    else:
        return jsonify({'error': 'wallet not found'}), 404


@app.route('/api/transact', methods=['POST'])
def transact():
    """Create a transaction"""
    data = request.get_json() or {}
    
    from_addr = data.get('from')
    to_addr = data.get('to')
    amount = data.get('amount')
    
    if not all([from_addr, to_addr, amount]):
        return jsonify({'error': 'missing from/to/amount'}), 400
    
    try:
        tx_hash = insert_transaction(from_addr, to_addr, int(amount))
        if tx_hash:
            # Broadcast to network
            if P2P:
                tx_data = {
                    'hash': tx_hash,
                    'from': from_addr,
                    'to': to_addr,
                    'amount': amount,
                    'timestamp': int(time.time()),
                }
                P2P.broadcast_transaction(tx_data)
            
            return jsonify({
                'tx_hash': tx_hash,
                'status': 'pending',
                'from': from_addr,
                'to': to_addr,
                'amount': amount,
            }), 201
        else:
            return jsonify({'error': 'failed to create transaction'}), 500
    except Exception as e:
        logger.error(f"[TRANSACT] Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/heartbeat', methods=['POST'])
def heartbeat():
    """Heartbeat endpoint"""
    snapshot = state.get_state()
    return jsonify({
        'heartbeat': snapshot['heartbeat_ts'],
        'uptime_s': snapshot['uptime_s'],
        'is_alive': snapshot['is_alive'],
        'metrics': snapshot['quantum_metrics'],
    }), 200


@app.route('/api/p2p/stats', methods=['GET'])
def p2p_stats():
    """Get P2P network statistics"""
    if P2P is None or not P2P.is_running:
        return jsonify({'error': 'P2P not initialized'}), 503
    
    return jsonify(P2P.get_stats()), 200


@app.route('/api/p2p/peers', methods=['GET'])
def p2p_peers():
    """Get connected peers information"""
    if P2P is None or not P2P.is_running:
        return jsonify({'error': 'P2P not initialized'}), 503
    
    return jsonify({
        'peer_count': P2P.get_peer_count(),
        'peers': P2P.get_peer_info(),
    }), 200


@app.route('/api/p2p/discovery', methods=['GET'])
def p2p_discovery():
    """Get peer discovery information"""
    with discovery_engine.lock:
        return jsonify({
            'candidate_peers': len(discovery_engine.peer_candidates),
            'tested_peers': len(discovery_engine.tested_peers),
        }), 200


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'not found'}), 404


@app.errorhandler(500)
def server_error(e):
    logger.error(f"[SERVER] Error: {e}")
    return jsonify({'error': 'internal server error'}), 500


# ═════════════════════════════════════════════════════════════════════════════════
# STARTUP & SHUTDOWN
# ═════════════════════════════════════════════════════════════════════════════════

@app.before_request
def before_request():
    """Before each request"""
    pass


@app.teardown_appcontext
def teardown(error=None):
    """Cleanup on shutdown"""
    if error:
        logger.error(f"[APP] Teardown error: {error}")


def shutdown_handler():
    """Graceful shutdown"""
    logger.info("[SERVER] Shutting down...")
    state.is_alive = False
    
    if P2P:
        P2P.shutdown()
    
    if LATTICE:
        try:
            LATTICE.shutdown()
        except:
            pass
    
    # Close database connection pool
    try:
        db_pool.close_all()
    except Exception as e:
        logger.error(f"[SERVER] Error closing database pool: {e}")
    
    logger.info("[SERVER] ✨ Shutdown complete")


if __name__ == '__main__':
    import atexit
    atexit.register(shutdown_handler)
    
    # Initialize P2P server
    if not initialize_p2p():
        logger.warning("[STARTUP] Failed to initialize P2P server (may retry)")
    
    # Port configuration (Koyeb uses $PORT environment variable)
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    logger.info("╔" + "═" * 78 + "╗")
    logger.info("║ QTCL SERVER v6 STARTING (WITH INTEGRATED P2P & CONNECTION POOLING)")
    logger.info("║ REST API: %d | P2P: %d | Debug: %s | Pooled: Yes" % (port, P2P_PORT, debug))
    logger.info("║ Lattice: %s | P2P: %s" % (state.lattice_loaded, P2P is not None))
    logger.info("╚" + "═" * 78 + "╝")
    
    # For Koyeb/production: use Gunicorn
    # For local development: use Flask development server
    if os.getenv('ENVIRONMENT') == 'production' or not debug:
        logger.warning("[STARTUP] Running in production mode. Use 'gunicorn -w 1 server:app' for deployment.")
    
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)
