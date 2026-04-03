#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║  QTCL SERVER v6 — Integrated P2P Blockchain with HLWE & Quantum Metrics       ║
║                                                                                ║
║  Museum-Grade Implementation — Pure JSON-RPC on Port 8000                          ║
║  ─────────────────────────────────────────────────────────────────────────    ║
║                                                                                ║
║  Single Unified Server (Port 8000 Internal):                                   ║
║    • REST/JSON-RPC API Layer (port 8000)                      ║
║    • Database Layer (internal) — persistent state (PostgreSQL)              ║
║    • Lattice Controller — quantum entropy mining                             ║
║    • Mempool Manager — transaction pool with validation                      ║
║    • Peer Discovery — DNS seeds, bootstrap nodes, peer exchange              ║
║    • Message Handlers — blocks, transactions, peer sync, consensus           ║
║                                                                                ║
║  Entry:                                                                        ║
║    python server.py                                                            ║
║    or: gunicorn -w1 -b0.0.0.0:$PORT server:app                                ║
║                                                                                ║
║  Environment Variables:                                                        ║
║    DATABASE_URL — PostgreSQL connection                                       ║
║    PORT — Listen port (default: 8000)                                     ║
║    FLASK_HOST — HTTP bind address (default: 0.0.0.0)                         ║
║    ORACLE_HTTP_URL — HTTP oracle endpoint for RPC calls                     ║
║    MAX_PEERS — Max peer connections (default: 32)                            ║
║    BOOTSTRAP_NODES — Comma-separated peer addresses                          ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import time
_SERVER_START_TIME = time.time()   # set once at module import — never drifts
import socket
import struct
import hashlib
import secrets
import logging
import threading
from typing import Dict, Any, Optional, List, Tuple, Set, Callable, Union, Deque
from collections import deque, OrderedDict
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════════════
# ENTERPRISE GRADE INITIALIZATION: QUANTUM ENTROPY + HLWE CRYPTOGRAPHY
# ═══════════════════════════════════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)

# ═══ ENTERPRISE METRICS THROTTLING ═══
_METRICS_SAMPLE_ORACLE = 50
_METRICS_SAMPLE_SHARD = 100
_ORACLE_CYCLE_COUNTERS = {}
_SHARD_CYCLE_COUNTERS = {}

def _should_log_oracle(oracle_id: str) -> bool:
    """Check if oracle measurement should be logged (sample-based)."""
    counter = _ORACLE_CYCLE_COUNTERS.get(oracle_id, 0)
    _ORACLE_CYCLE_COUNTERS[oracle_id] = counter + 1
    return (counter % _METRICS_SAMPLE_ORACLE) == 0

def _should_log_shard(shard_id: int) -> bool:
    """Check if shard cycle should be logged (sample-based)."""
    counter = _SHARD_CYCLE_COUNTERS.get(shard_id, 0)
    _SHARD_CYCLE_COUNTERS[shard_id] = counter + 1
    return (counter % _METRICS_SAMPLE_SHARD) == 0

# ═══ SNAPSHOT BROADCAST THROTTLING ═══
_verbose_p2p_logging = False
_last_snapshot_log_time = 0
_snapshot_log_interval = 10

# ═══ DENSITY MATRIX SNAPSHOT RING BUFFER (1000 snapshots, LRU eviction) ═══
_DM_SNAPSHOT_RING = deque(maxlen=1000)
_DM_SNAPSHOT_LOCK = threading.RLock()

# ═══ CLIENT TRIPARTITE ORACLE POOL ═══════════════════════════════════════════
# Receives fused DMs pushed by trusted client oracle nodes (qtcl_pushOracleDM).
# Keyed by oracle_addr → {dm_re, dm_im, fidelity, ts, node_ip, oracle_type}
# Pool is Hermitian-averaged every push into _client_consensus_dm which then
# enriches the server's own 5-oracle snapshot on /rpc/oracle/snapshot.
_CLIENT_DM_POOL: Dict[str, dict] = {}
_CLIENT_DM_POOL_LOCK = threading.RLock()
_CLIENT_POOL_MAX    = 64          # cap pool size — evict oldest on overflow
_CLIENT_DM_STALE_S  = 120.0      # drop frames older than 2 min from consensus
_client_consensus_dm_re: list = [0.0] * 64
_client_consensus_dm_im: list = [0.0] * 64
_client_consensus_fid:   float = 0.0
_client_pool_count:      int   = 0


def _recompute_client_consensus() -> None:
    """
    Hermitian-mean all fresh client DMs in _CLIENT_DM_POOL into
    _client_consensus_dm_re/_im and _client_consensus_fid.
    Called under _CLIENT_DM_POOL_LOCK — must not re-acquire it.
    ❤️  I love you — every client is a node in the lattice
    """
    global _client_consensus_dm_re, _client_consensus_dm_im
    global _client_consensus_fid, _client_pool_count

    now = time.time()
    fresh = [
        v for v in _CLIENT_DM_POOL.values()
        if (now - v['ts']) < _CLIENT_DM_STALE_S and v.get('fidelity', 0.0) > 0.0
    ]
    _client_pool_count = len(fresh)
    if not fresh:
        return

    total_w = sum(max(v['fidelity'], 1e-6) for v in fresh)
    re_acc  = [0.0] * 64
    im_acc  = [0.0] * 64
    fid_acc = 0.0
    for v in fresh:
        w = v['fidelity'] / total_w
        for i in range(64):
            re_acc[i] += w * v['dm_re'][i]
            im_acc[i] += w * v['dm_im'][i]
        fid_acc += w * v['fidelity']

    tr = sum(re_acc[i * 8 + i] for i in range(8))
    if tr > 1e-12:
        re_acc = [x / tr for x in re_acc]
        im_acc = [x / tr for x in im_acc]

    _client_consensus_dm_re = re_acc
    _client_consensus_dm_im = im_acc
    _client_consensus_fid   = fid_acc

# ═══ RPC INFRASTRUCTURE (JSON-RPC 2.0) ═══
_JSONRPC_VERSION = "2.0"

def _rpc_ok(result: Any, rpc_id: Any) -> dict:
    """Standard JSON-RPC 2.0 success response."""
    return {"jsonrpc": _JSONRPC_VERSION, "result": result, "id": rpc_id}

def _rpc_error(code: int, message: str, rpc_id: Any, data: Optional[dict] = None) -> dict:
    """Standard JSON-RPC 2.0 error response."""
    resp = {"jsonrpc": _JSONRPC_VERSION, "error": {"code": code, "message": message}, "id": rpc_id}
    if data:
        resp["error"]["data"] = data
    return resp

def _dispatch(body_bytes: bytes) -> Tuple[dict, int]:
    """JSON-RPC 2.0 dispatcher (parse, call, return). Handles batches."""
    try:
        body = json.loads(body_bytes.decode('utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return _rpc_error(-32700, f"Parse error: {str(e)}", None), 400
    
    if isinstance(body, list):
        if not body:
            return _rpc_error(-32600, "Invalid Request: empty batch", None), 400
        responses = [r for req in body if (r := _dispatch_single(req)) is not None]
        return responses if responses else None, 200
    
    return _dispatch_single(body), 200

def _dispatch_single(req: dict) -> Optional[dict]:
    """Dispatch single JSON-RPC 2.0 request."""
    if not isinstance(req, dict):
        return _rpc_error(-32600, "Invalid Request: not an object", None)
    
    jsonrpc, method, params, rpc_id = req.get("jsonrpc"), req.get("method"), req.get("params", []), req.get("id")
    
    if jsonrpc != _JSONRPC_VERSION:
        return _rpc_error(-32600, f"Invalid jsonrpc: {jsonrpc}", rpc_id)
    if not isinstance(method, str):
        return _rpc_error(-32600, "Invalid Request: method not a string", rpc_id)
    if method not in _RPC_METHODS:
        return _rpc_error(-32601, f"Method not found: {method}", rpc_id)
    
    try:
        return _RPC_METHODS[method](params, rpc_id)
    except Exception as e:
        logger.exception(f"[RPC] {method} raised: {e}")
        return _rpc_error(-32603, f"Internal error: {str(e)}", rpc_id)

_RPC_METHOD_META: Dict[str, dict] = {}

# ═══ LAZY INITIALIZATION (deferred until first use) ═══
# This allows Flask to bind port 8000 before heavy crypto/quantum init
QRNG_ENSEMBLE = None
HLWE_ENGINE = None
_QRNG_INIT_LOCK = threading.Lock()
_HLWE_INIT_LOCK = threading.Lock()

def _init_qrng_ensemble():
    """Lazy init QRNG_ENSEMBLE on first demand."""
    global QRNG_ENSEMBLE
    if QRNG_ENSEMBLE is not None:
        return QRNG_ENSEMBLE
    with _QRNG_INIT_LOCK:
        if QRNG_ENSEMBLE is not None:  # double-check
            return QRNG_ENSEMBLE
        try:
            from qrng_ensemble import get_qrng_ensemble
            QRNG_ENSEMBLE = get_qrng_ensemble()
            logger.info("[INIT-QRNG] ✅ Quantum RNG Ensemble initialized on first use")
            return QRNG_ENSEMBLE
        except Exception as e:
            logger.critical(f"[INIT-QRNG] ❌ FATAL: Cannot initialize QRNG_ENSEMBLE: {e}")
            raise RuntimeError(f"[INIT-QRNG] Cannot initialize Quantum RNG. Error: {e}")

def _init_hlwe_engine():
    """Lazy init HLWE_ENGINE on first demand."""
    global HLWE_ENGINE
    if HLWE_ENGINE is not None:
        return HLWE_ENGINE
    with _HLWE_INIT_LOCK:
        if HLWE_ENGINE is not None:  # double-check
            return HLWE_ENGINE
        try:
            from hlwe_engine import HLWEEngine
            HLWE_ENGINE = HLWEEngine()
            logger.info("[INIT-HLWE] ✅ HLWE Post-Quantum Cryptography initialized on first use")
            return HLWE_ENGINE
        except Exception as e:
            logger.critical(f"[INIT-HLWE] ❌ FATAL: Cannot initialize HLWE_ENGINE: {e}")
            raise RuntimeError(f"[INIT-HLWE] Cannot initialize HLWE cryptography. Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 5-ORACLE BYZANTINE CONSENSUS INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════════════

import traceback
from datetime import datetime, timezone, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import deque, OrderedDict
from contextlib import contextmanager

# ═════════════════════════════════════════════════════════════════════════════════════════
# EARLY LOGGER SETUP (before DHT/other classes)
# ═════════════════════════════════════════════════════════════════════════════════════════

if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════════════════
# DISTRIBUTED HASH TABLE (DHT) — KADEMLIA-BASED PEER DISCOVERY
# ═════════════════════════════════════════════════════════════════════════════════════════
# Museum-Grade DHT for decentralized peer discovery and state storage
# Implements XOR distance metric, k-buckets routing table, and peer queries

class DHTNode:
    """Museum-Grade DHT Node - Kademlia peer discovery"""
    
    def __init__(self, node_id: Optional[str] = None, address: str = "unknown", port: int = 9091):
        """
        Initialize DHT node.
        
        Args:
            node_id: 160-bit hex identifier (SHA1 of pubkey), or auto-generated
            address: Network address (IP or hostname)
            port: Listen port
        """
        if node_id is None:
            # Generate from address hash
            node_id = hashlib.sha1(f"{address}:{port}:{secrets.token_hex(16)}".encode()).hexdigest()
        
        self.node_id = node_id
        self.node_id_int = int(node_id, 16)
        self.address = address
        self.port = port
        self.last_seen = time.time()
        self.failed_pings = 0
        self.rpc_version = "1.0"
    
    def distance_to(self, other_id: str) -> int:
        """Calculate XOR distance to another node (Kademlia metric)"""
        other_int = int(other_id, 16)
        return self.node_id_int ^ other_int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "address": self.address,
            "port": self.port,
            "last_seen": self.last_seen,
            "failed_pings": self.failed_pings,
        }
    
    def is_alive(self, timeout_sec: int = 300) -> bool:
        """Check if node is considered alive (seen within timeout)"""
        return (time.time() - self.last_seen) < timeout_sec


class DHTRoutingTable:
    """Museum-Grade Kademlia routing table with k-buckets"""
    
    def __init__(self, local_node_id: str, k: int = 20):
        """
        Initialize routing table.
        
        Args:
            local_node_id: Local node's 160-bit hex ID
            k: Bucket size (default 20 for Kademlia)
        """
        self.local_node_id = local_node_id
        self.local_node_id_int = int(local_node_id, 16)
        self.k = k
        self.buckets: Dict[int, List[DHTNode]] = {}
        self.lock = threading.RLock()
        self.bucket_refreshes: Dict[int, float] = {}
    
    def _get_bucket_index(self, node_id: str) -> int:
        """Get bucket index (0-159) based on XOR distance"""
        other_int = int(node_id, 16)
        xor_distance = self.local_node_id_int ^ other_int
        if xor_distance == 0:
            return 0
        return xor_distance.bit_length() - 1
    
    def add_node(self, node: DHTNode) -> bool:
        """Add node to routing table, return True if added/updated"""
        with self.lock:
            bucket_idx = self._get_bucket_index(node.node_id)
            if bucket_idx not in self.buckets:
                self.buckets[bucket_idx] = []
            
            bucket = self.buckets[bucket_idx]
            
            # Check if already exists
            for existing in bucket:
                if existing.node_id == node.node_id:
                    existing.last_seen = time.time()
                    existing.failed_pings = 0
                    logger.debug(f"[DHT] ✓ Node updated: {node.node_id[:16]}…")
                    return True
            
            # Add new node if bucket not full
            if len(bucket) < self.k:
                bucket.append(node)
                logger.info(f"[DHT] ✅ Node added: {node.address}:{node.port} | {node.node_id[:16]}…")
                return True
            else:
                logger.debug(f"[DHT] ⚠️  Bucket {bucket_idx} full, cannot add {node.node_id[:16]}…")
                return False
    
    def get_closest_nodes(self, target_id: str, count: int = 20) -> List[DHTNode]:
        """Get k closest nodes to target ID"""
        with self.lock:
            all_nodes = []
            for bucket in self.buckets.values():
                all_nodes.extend(bucket)
            
            # Sort by XOR distance
            target_int = int(target_id, 16)
            all_nodes.sort(key=lambda n: n.node_id_int ^ target_int)
            return all_nodes[:count]
    
    def mark_node_failed(self, node_id: str) -> bool:
        """Mark node as failed, return True if removed"""
        with self.lock:
            bucket_idx = self._get_bucket_index(node_id)
            if bucket_idx not in self.buckets:
                return False
            
            bucket = self.buckets[bucket_idx]
            for node in bucket:
                if node.node_id == node_id:
                    node.failed_pings += 1
                    if node.failed_pings >= 3:
                        bucket.remove(node)
                        logger.warning(f"[DHT] ❌ Node removed (failed pings): {node_id[:16]}…")
                        return True
                    return False
            return False
    
    def get_all_nodes(self) -> List[DHTNode]:
        """Get all nodes in routing table"""
        with self.lock:
            return [node for bucket in self.buckets.values() for node in bucket]
    
    def count_peers(self) -> int:
        """Count total peers in routing table"""
        with self.lock:
            return sum(len(bucket) for bucket in self.buckets.values())


class DHTManager:
    """Museum-Grade DHT Manager - coordinates peer discovery and state storage"""
    
    def __init__(self, local_address: str = "localhost", local_port: int = 9091):
        self.local_node = DHTNode(address=local_address, port=local_port)
        self.routing_table = DHTRoutingTable(self.local_node.node_id)
        self.state_store: Dict[str, Dict[str, Any]] = {}  # key → {data, timestamp}
        self.store_lock = threading.RLock()
        self.lookup_cache: Dict[str, List[DHTNode]] = {}
        logger.info(f"[DHT] ✅ Manager initialized | node_id={self.local_node.node_id[:16]}… | {local_address}:{local_port}")
    
    def store_state(self, key: str, value: Dict[str, Any]) -> bool:
        """Store (key, value) pair in DHT"""
        with self.store_lock:
            self.state_store[key] = {
                "data": value,
                "timestamp": time.time(),
                "replicas": [self.local_node.node_id],
            }
            logger.debug(f"[DHT] 💾 State stored: {key[:32]}…")
            return True
    
    def retrieve_state(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve (key, value) from DHT"""
        with self.store_lock:
            if key in self.state_store:
                return self.state_store[key]["data"]
            return None
    
    def find_node(self, target_id: str) -> List[DHTNode]:
        """Find nodes closest to target ID"""
        closest = self.routing_table.get_closest_nodes(target_id, count=20)
        self.lookup_cache[target_id] = closest
        logger.info(f"[DHT] 🔍 Lookup: target={target_id[:16]}… | found {len(closest)} nodes")
        return closest
    
    def find_value(self, key: str) -> Optional[Dict[str, Any]]:
        """Find value for key, return stored value or None"""
        result = self.retrieve_state(key)
        if result:
            logger.debug(f"[DHT] ✓ Value found locally: {key[:32]}…")
            return result
        # In real system: query nodes in routing table
        return None


# ═════════════════════════════════════════════════════════════════════════════════════════
# REMAINING IMPORTS
# ═════════════════════════════════════════════════════════════════════════════════════════

from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor  # H2: Thread pooling for DoS prevention
import random  # required by P2P broadcast loop
try:
    import psycopg2
    import psycopg2.pool as psycopg2_pool_mod
except ImportError:
    psycopg2 = None  # type: ignore
    psycopg2_pool_mod = None

from flask import Flask, jsonify, request, render_template_string, send_file, Response
app = Flask(__name__)
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from io import BytesIO
import msgpack
import base64
import queue as _queue_mod
import uuid

# ═════════════════════════════════════════════════════════════════════════════════════════
# RPC SNAPSHOT DISTRIBUTION (replaces SSE)
# ═════════════════════════════════════════════════════════════════════════════════════════
# ENTROPY POOL INTEGRATION
# ═════════════════════════════════════════════════════════════════════════════════

try:
    from globals import (
        initialize_block_field_entropy,
        set_current_block_field,
        get_block_field_entropy,
        initialize_system as init_entropy_system,
        TessellationRewardSchedule,
    )
    ENTROPY_AVAILABLE = True
except ImportError:
    ENTROPY_AVAILABLE = False
    TessellationRewardSchedule = None
    logger.warning("[ENTROPY] Block field entropy not available - will use fallback")

# ═════════════════════════════════════════════════════════════════════════════════
# ORACLE & W-STATE INTEGRATION  (deferred — must not block gunicorn startup)
# ─────────────────────────────────────────────────────────────────────────────
# oracle.py calls the QRNG ensemble at module-level; each QRNG source that is
# unreachable on Koyeb takes 6-10 s to time out.  Importing oracle synchronously
# here would block gunicorn for 16-28 s, causing Koyeb's health-check to
# accumulate failures and fire SIGTERM before the worker ever binds port 8000.
#
# Solution: import oracle in a daemon thread; all code that uses ORACLE already
# guards with `if ORACLE_AVAILABLE` / `if ORACLE is not None`.
# ═════════════════════════════════════════════════════════════════════════════════

ORACLE_AVAILABLE = False
ORACLE = None
ORACLE_W_STATE_MANAGER = None
LATTICE = None
_ORACLE_INIT_EVENT = threading.Event()   # set once oracle is ready (or failed)
_LATTICE_INIT_EVENT = threading.Event()  # set once lattice is ready (or failed)

def _deferred_lattice_init() -> None:
    """Import and initialise lattice_controller.py in a background thread.
    
    QuantumLatticeController initializes the spatial-temporal field, quantum execution engine,
    W-state constructor, and non-Markovian noise bath.  This runs in a daemon thread to let
    gunicorn bind port 8000 immediately; lattice becomes available within ~2-5s.
    
    CRITICAL: Also starts the oracle measurement stream AFTER wiring lattice.
    """
    global LATTICE
    try:
        from lattice_controller import QuantumLatticeController
        LATTICE = QuantumLatticeController()
        logger.info("[LATTICE-INIT] ✅ QuantumLatticeController instantiated")
        LATTICE.start()
        logger.info("[LATTICE-INIT] ✅ Lattice daemon started — spatial-temporal field active, coherence maintenance loop running")
        
        # ── WIRE LATTICE INTO ORACLE ──────────────────────────────────────────────
        from globals import set_lattice
        set_lattice(LATTICE)
        logger.info("[LATTICE-INIT] ✅ Lattice registered with oracle — ready for measurement")
        
        # ── NOW START ORACLE MEASUREMENT STREAM (after lattice is wired) ──────────
        global ORACLE_W_STATE_MANAGER
        if ORACLE_W_STATE_MANAGER is not None:
            _ok = ORACLE_W_STATE_MANAGER.start()
            if _ok:
                logger.info("[LATTICE-INIT] ✅ Oracle measurement stream started — 5-node snapshot acquisition active")
            else:
                logger.error("[LATTICE-INIT] ❌ Oracle measurement stream failed to start")
        else:
            logger.warning("[LATTICE-INIT] ⚠️  ORACLE_W_STATE_MANAGER not ready yet")
    except ImportError as _ie:
        logger.error(f"[LATTICE-INIT] ❌ QuantumLatticeController import failed: {_ie}")
    except Exception as _ex:
        logger.error(f"[LATTICE-INIT] ❌ Lattice deferred init error: {_ex}", exc_info=True)
    finally:
        _LATTICE_INIT_EVENT.set()  # unblock oracle sync daemon even if lattice failed

threading.Thread(
    target=_deferred_lattice_init,
    daemon=True,
    name="LatticeDeferred",
).start()
logger.info("[LATTICE] 🔄 Lattice init deferred to background thread — gunicorn will serve /health immediately")

def _deferred_oracle_init() -> None:
    """Import and initialise oracle.py in a background thread.

    oracle.py spends 16-28 s at module-level waiting for QRNG network sources
    to respond (or time out).  Running this in a daemon thread lets gunicorn
    bind port 8000 and start answering /health checks in < 2 s.
    
    NOTE: Do NOT start the measurement stream here — wait for LATTICE initialization.
    """
    global ORACLE, ORACLE_W_STATE_MANAGER, ORACLE_AVAILABLE
    try:
        from oracle import ORACLE as _o, ORACLE_W_STATE_MANAGER as _owsm
        ORACLE = _o
        ORACLE_W_STATE_MANAGER = _owsm
        ORACLE_AVAILABLE = True
        logger.info("[ORACLE] ✅ Oracle engine initialised (deferred background thread)")
        # ⚠️  WAIT for LATTICE before starting measurement stream
        # _deferred_lattice_init will call ORACLE_W_STATE_MANAGER.start() after set_lattice()
    except ImportError as _ie:
        logger.warning(f"[ORACLE] ⚠️  Oracle not available (ImportError): {_ie}")
    except Exception as _ex:
        logger.error(f"[ORACLE] ❌ Oracle deferred init error: {_ex}", exc_info=True)
    finally:
        _ORACLE_INIT_EVENT.set()   # unblock _wsgi_startup heavy-init thread

threading.Thread(
    target=_deferred_oracle_init,
    daemon=True,
    name="OracleDeferred",
).start()
logger.info("[ORACLE] 🔄 Oracle init deferred to background thread — gunicorn will serve /health immediately")

# ═════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════════

# Database Configuration
# Supabase provides individual pooler connection variables OR a full URL# Try full URL first, then build from components

POOLER_URL = os.getenv('POOLER_URL')
_USE_HTTP_DB = os.getenv('USE_HTTP_DB', '0') == '1'  # PythonAnywhere: route SQL over HTTPS PostgREST

if not POOLER_URL:
    POOLER_HOST     = os.getenv('POOLER_HOST')
    POOLER_USER     = os.getenv('POOLER_USER')
    POOLER_PASSWORD = os.getenv('POOLER_PASSWORD')
    POOLER_DB       = os.getenv('POOLER_DB', 'postgres')
    POOLER_PORT     = os.getenv('POOLER_PORT', '6543')

    if POOLER_HOST and POOLER_USER and POOLER_PASSWORD:
        POOLER_URL = f"postgresql://{POOLER_USER}:{POOLER_PASSWORD}@{POOLER_HOST}:{POOLER_PORT}/{POOLER_DB}"
        logger.info("[DB] Built POOLER_URL from POOLER_* environment variables")
    elif _USE_HTTP_DB:
        # On PythonAnywhere, raw TCP to Supabase is blocked — we use HTTPS PostgREST RPC instead.
        # POOLER_URL is not needed; set a sentinel so module-level code that references it doesn't crash.
        POOLER_URL = 'postgresql://http-mode-no-tcp-needed/postgres'
        logger.info("[DB] USE_HTTP_DB=1 — POOLER_URL not required (all SQL routes via HTTPS PostgREST)")
    else:
        logger.error("[DB] ❌ CRITICAL: Supabase connection not configured!")
        logger.error("[DB] Set one of:")
        logger.error("[DB]   1. POOLER_URL=postgresql://...")
        logger.error("[DB]   2. POOLER_HOST, POOLER_USER, POOLER_PASSWORD, POOLER_DB, POOLER_PORT")
        logger.error("[DB]   3. USE_HTTP_DB=1 + SUPABASE_URL + SUPABASE_SERVICE_KEY  (PythonAnywhere)")
        raise ValueError("Supabase pooler connection variables not set")

DB_URL = POOLER_URL

# PATCH-10: Auto-correct session-mode port 5432 → transaction-mode port 6543.
# Supabase session mode (5432) limits concurrent clients to pool_size — under
# a retry storm this immediately hits MaxClientsInSessionMode.
# Transaction mode (6543) is stateless per-query and has no per-client limit.
# If POOLER_URL or POOLER_PORT was set to 5432, correct it silently.
if DB_URL and ':5432/' in DB_URL and not _USE_HTTP_DB:
    DB_URL = DB_URL.replace(':5432/', ':6543/')
    logger.warning("[DB] ⚡ PATCH-10: Auto-corrected port 5432→6543 (transaction mode) — "
                   "set POOLER_PORT=6543 in Koyeb env to suppress this warning")
if not _USE_HTTP_DB:
    logger.info(f"[DB] ✨ Using Supabase Pooler: {os.getenv('POOLER_HOST') or 'configured'}")
else:
    logger.info(f"[DB] ✨ HTTP-DB mode: {os.getenv('SUPABASE_URL', '(SUPABASE_URL not set)')}")

# ═══════════════════════════════════════════════════════════════════════════════
# TX QUERY WORKER — dedicated direct connection on port 6543 (transaction mode)
# ═══════════════════════════════════════════════════════════════════════════════
# /api/transactions runs heavyweight COUNT + page queries that can take 1-3s.
# Running them through the shared 10-connection pool starves background threads
# (oracle sync, lattice, P2P heartbeats) and causes cascading timeouts.
#
# This worker owns a single private psycopg2 connection to port 6543 — completely
# independent of DatabasePool. It processes one query at a time from _TX_JOB_Q.
# The Flask handler submits a job dict and blocks on a per-job result queue with
# a hard 9s timeout — if the worker is busy or the DB is slow the route returns
# a fast 503 so the client retries rather than holding a gthread indefinitely.
# ───────────────────────────────────────────────────────────────────────────────

import queue as _queue_mod2

_TX_JOB_Q: '_queue_mod2.Queue' = _queue_mod2.Queue(maxsize=8)

def _build_tx_dsn() -> str:
    """Always return a DSN pointed at port 6543 (Supabase transaction mode)."""
    dsn = DB_URL or ''
    if not dsn or _USE_HTTP_DB:
        return ''
    # Force transaction-mode port
    if ':5432/' in dsn:
        dsn = dsn.replace(':5432/', ':6543/')
    if ':6543/' not in dsn and dsn.startswith('postgresql://'):
        # No port in URL — inject 6543 before the DB path
        import re as _re
        dsn = _re.sub(r'(@[^/]+)(/.+)', r'\g<1>:6543\g<2>', dsn)
    return dsn

def _tx_worker_thread():
    """Dedicated TX query thread — owns one private psycopg2 connection."""
    import psycopg2 as _pg
    _tx_log = logging.getLogger('tx_worker')
    dsn = _build_tx_dsn()
    if not dsn:
        _tx_log.warning("[TX-WORKER] No DSN — thread idle (USE_HTTP_DB mode)")
        while True:
            try:
                job = _TX_JOB_Q.get(timeout=5)
                job['result_q'].put({'error': 'TX worker unavailable (HTTP-DB mode)'})
            except _queue_mod2.Empty:
                pass
        return

    conn = None

    def _connect():
        nonlocal conn
        try:
            if conn:
                try: conn.close()
                except Exception: pass
            conn = _pg.connect(dsn, connect_timeout=10,
                               options='-c statement_timeout=9000')
            conn.autocommit = True
            _tx_log.info("[TX-WORKER] ✅ Connected to Supabase :6543 (transaction mode)")
        except Exception as _ce:
            conn = None
            _tx_log.error(f"[TX-WORKER] Connect failed: {_ce}")

    _connect()

    while True:
        try:
            job = _TX_JOB_Q.get(timeout=30)
        except _queue_mod2.Empty:
            # Keepalive ping on idle
            if conn:
                try: conn.cursor().execute("SELECT 1")
                except Exception: _connect()
            continue

        result_q = job.get('result_q')
        try:
            # Reconnect if connection dropped
            if conn is None or conn.closed:
                _connect()
            if conn is None:
                if result_q: result_q.put({'error': 'DB connection unavailable'})
                continue

            cur = conn.cursor()
            queries = job['queries']  # list of (sql, params) tuples
            results = []
            for sql, params in queries:
                cur.execute(sql, params)
                results.append(cur.fetchall())
            cur.close()
            if result_q: result_q.put({'results': results})

        except _pg.OperationalError as _oe:
            _tx_log.warning(f"[TX-WORKER] OperationalError — reconnecting: {_oe}")
            _connect()
            if result_q: result_q.put({'error': str(_oe)})
        except Exception as _e:
            _tx_log.error(f"[TX-WORKER] Query error: {_e}")
            if result_q: result_q.put({'error': str(_e)})


def _tx_query(queries: list, timeout: float = 9.0) -> dict:
    """Submit queries to the TX worker and wait for results.

    Args:
        queries: list of (sql_string, params_tuple) to execute in sequence.
        timeout: max seconds to wait before returning {'error': 'timeout'}.
    Returns:
        {'results': [[rows], [rows], ...]} or {'error': str}.
    """
    rq: '_queue_mod2.Queue' = _queue_mod2.Queue(maxsize=1)
    job = {'queries': queries, 'result_q': rq}
    try:
        _TX_JOB_Q.put_nowait(job)
    except _queue_mod2.Full:
        return {'error': 'TX worker busy — retry in a moment'}
    try:
        return rq.get(timeout=timeout)
    except _queue_mod2.Empty:
        return {'error': 'DB query timed out — retry in a moment'}


# Launch TX worker daemon at module load
_tx_worker_daemon = threading.Thread(
    target=_tx_worker_thread, daemon=True, name="TxQueryWorker")
_tx_worker_daemon.start()
logger.info("[TX-WORKER] Dedicated transaction query thread started (port 6543)")

# ── Oracle identity — unique per deployed instance ────────────────────────────
# Set ORACLE_ID in env to distinguish instances:
#   primary   → Koyeb main       (ORACLE_ID=koyeb-primary)
#   secondary → PythonAnywhere   (ORACLE_ID=pa-secondary)
#   tertiary  → Koyeb account 2  (ORACLE_ID=koyeb-tertiary)
# All instances share the same Supabase DB — they are peers, not replicas.
ORACLE_ID   = os.getenv('ORACLE_ID',   'koyeb-primary')
ORACLE_ROLE = os.getenv('ORACLE_ROLE', 'primary')
# Peer oracle URLs — other oracle instances this one will cross-register with
_peer_oracle_env = os.getenv('BOOTSTRAP_NODES', '')
PEER_ORACLE_URLS = [u.strip() for u in _peer_oracle_env.split(',') if u.strip()] if _peer_oracle_env else []

# ═════════════════════════════════════════════════════════════════════════════════
# ORACLE ADDRESS LOOKUP: Per-oracle HLWE addresses from registry
# ═════════════════════════════════════════════════════════════════════════════════

def get_oracle_address(oracle_id: str, fallback: str = '') -> str:
    """Fetch oracle HLWE address from oracle_registry by oracle_id.
    
    Each oracle node has a unique registered address for auditability.
    oracle_id format: 'oracle_1', 'oracle_2', ... 'oracle_5'
    
    Fallback: if DB unavailable, returns fallback string.
    """
    try:
        if not db_ready():
            return fallback
        
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT oracle_address FROM oracle_registry WHERE oracle_id = %s",
                (oracle_id,)
            )
            result = cursor.fetchone()
            cursor.close()
            conn.commit()
            return result[0] if result else fallback
        finally:
            if db_pool.use_pooling and db_pool.pool:
                db_pool.pool.putconn(conn)
            else:
                conn.close()
    except Exception as e:
        logger.debug(f"[ORACLE-ADDRESS] Lookup failed for {oracle_id}: {e}")
        return fallback

def get_consensus_oracle_address() -> str:
    """
    Compute consensus oracle address (XOR of all 5 oracle addresses).
    Used for transactions that require all-oracle sign-off.
    """
    try:
        if not db_ready():
            return 'qtcl1consensus_all_oracles_xor'
        
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT oracle_address FROM oracle_registry WHERE role IN "
                "('PRIMARY_LATTICE', 'SECONDARY_LATTICE', 'VALIDATION', 'ARBITER', 'METRICS') "
                "ORDER BY oracle_id"
            )
            addresses = [row[0] for row in cursor.fetchall()]
            cursor.close()
            conn.commit()
            
            if len(addresses) != 5:
                logger.warning(f"[ORACLE-ADDRESS] Expected 5 oracles, got {len(addresses)}")
            
            # XOR all addresses together for deterministic consensus address
            import hashlib
            consensus_seed = '|'.join(addresses).encode()
            consensus_hash = hashlib.sha256(consensus_seed).hexdigest()[:24]
            return f"qtcl1consensus_{consensus_hash}"
        finally:
            if db_pool.use_pooling and db_pool.pool:
                db_pool.pool.putconn(conn)
            else:
                conn.close()
    except Exception as e:
        logger.debug(f"[ORACLE-ADDRESS] Consensus lookup failed: {e}")
        return 'qtcl1consensus_all_oracles_xor'


logger.info(f"[ORACLE] 🌐 Identity: id={ORACLE_ID} role={ORACLE_ROLE} peers={len(PEER_ORACLE_URLS)}")

# P2P raw-TCP port — separate from HTTP/gunicorn.
# Koyeb: set P2P_PORT=9091 env var (HTTP service on 9091, routes /api/*).
# Gunicorn binds PORT (typically 8000). P2P binds P2P_PORT (9091).
# They MUST be different ports; using PORT here caused the 8000→8001 fallback bug.
P2P_PORT = int(os.getenv('P2P_PORT', 9091))
P2P_HOST = os.getenv('P2P_HOST', '0.0.0.0')
P2P_TESTNET_PORT = P2P_PORT + 10000
MAX_PEERS = int(os.getenv('MAX_PEERS', 32))
PEER_TIMEOUT = 30
MESSAGE_MAX_SIZE = 1_000_000
PEER_HANDSHAKE_TIMEOUT = 5
PEER_KEEPALIVE_INTERVAL = 30


# ── Block policy ──────────────────────────────────────────────────────────────
# Max USER transactions per block (coinbase not counted).
# Matches miner's MAX_BLOCK_TX — must be kept in sync.
MAX_BLOCK_TX_SERVER = 100
# Coinbase null address — 64 hex zeros, provably unspendable
COINBASE_NULL_ADDRESS = '0' * 64
PEER_DISCOVERY_INTERVAL = 60
PEER_CLEANUP_INTERVAL = 15

# Message Types
MESSAGE_TYPES = {    'version': 0,
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


# ═════════════════════════════════════════════════════════════════════════════════
# DATABASE LAYER WITH CONNECTION POOLING
# ═════════════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════════════
# SUPABASE HTTP DATABASE ADAPTER (inline — PythonAnywhere TCP-blocked environments)
# ═════════════════════════════════════════════════════════════════════════════════
# When USE_HTTP_DB=1, every cursor.execute() routes through PostgREST RPC over HTTPS
# instead of a raw psycopg2 TCP connection.  The two Supabase RPC functions required:
#   exec_sql_select(query text) → jsonb   (SELECT / WITH / EXPLAIN / SHOW / VALUES)
#   exec_sql_write(query text)  → jsonb   (INSERT / UPDATE / DELETE / DO)
# Run the migration SQL in Supabase Dashboard → SQL Editor once to create them.
# Env vars needed:  SUPABASE_URL, SUPABASE_SERVICE_KEY  (service_role JWT)
# ─────────────────────────────────────────────────────────────────────────────────

import re as _re, decimal as _decimal

try:
    import requests as _http_requests; _HTTP_BACKEND = 'requests'
except ImportError:
    import urllib.request as _urllib_req, urllib.error as _urllib_err; _HTTP_BACKEND = 'urllib'

# True when `requests` lib is available (used by _fetch_peer and cross-oracle helpers)
_HAS_REQUESTS: bool = (_HTTP_BACKEND == 'requests')

def _http_json_serial(o):
    if isinstance(o, (datetime, )): return o.isoformat()
    if isinstance(o, _decimal.Decimal): return float(o)
    if isinstance(o, (bytes, bytearray)): return o.hex()
    raise TypeError(f"not serialisable: {type(o)}")

def _http_post_json(url, headers, payload, timeout=30, retries=3):
    """POST JSON; retry on 5xx/network with exponential backoff. Returns parsed body."""
    import json as _json
    raw = _json.dumps(payload, default=_http_json_serial).encode()
    hdrs = {**headers, 'Content-Type': 'application/json'}
    last = None
    for attempt in range(retries):
        if attempt: time.sleep(min(0.5 * 2**attempt, 8))
        try:
            if _HTTP_BACKEND == 'requests':
                r = _http_requests.post(url, data=raw, headers=hdrs, timeout=timeout)
                status, text = r.status_code, r.text
            else:
                req = _urllib_req.Request(url, data=raw, headers=hdrs, method='POST')
                try:
                    with _urllib_req.urlopen(req, timeout=timeout) as r:
                        status, text = r.status, r.read().decode()
                except _urllib_err.HTTPError as e:
                    status, text = e.code, e.read().decode()
            if status < 500:
                if status >= 400:
                    import json as _j
                    try: detail = _j.loads(text).get('message') or text
                    except Exception: detail = text
                    raise RuntimeError(f"Supabase RPC HTTP {status}: {detail}")
                import json as _j; return _j.loads(text)
            last = RuntimeError(f"Supabase RPC HTTP {status}: {text}")
        except (OSError, TimeoutError) as e:
            last = e; logger.warning(f"[SUPHTTP] attempt {attempt+1}/{retries}: {e}")
    raise last or RuntimeError("_http_post_json exhausted retries")

# Singleton HTTP client config
_SUPHTTP_CFG: Dict[str, Any] = {}
_SUPHTTP_LOCK = threading.Lock()

def _suphttp_cfg():
    """Lazily populate and return the HTTP client config dict."""
    with _SUPHTTP_LOCK:
        if _SUPHTTP_CFG: return _SUPHTTP_CFG
        url = os.getenv('SUPABASE_URL', '').rstrip('/')
        key = os.getenv('SUPABASE_SERVICE_KEY') or os.getenv('SUPABASE_ANON_KEY', '')
        if not url: raise EnvironmentError("SUPABASE_URL env var not set (required for USE_HTTP_DB=1)")
        if not key: raise EnvironmentError("SUPABASE_SERVICE_KEY env var not set (required for USE_HTTP_DB=1)")
        _SUPHTTP_CFG.update({
            'url': url, 'timeout': int(os.getenv('SUPABASE_HTTP_TIMEOUT', '30')),
            'retries': int(os.getenv('SUPABASE_HTTP_RETRIES', '3')),
            'headers': {'apikey': key, 'Authorization': f'Bearer {key}', 'Prefer': 'return=representation'},
        })
        logger.info(f"[SUPHTTP] ✓ client configured → {url}/rest/v1/rpc/exec_sql_*")
        return _SUPHTTP_CFG

_PARAM_RE = _re.compile(r'%\((\w+)\)s|%s')
_SELECT_FIRST = frozenset({'select','with','explain','show','table','values','fetch'})
_WRITE_FIRST  = frozenset({'insert','update','delete','do','call','perform'})
_COMMENT_STRIP = _re.compile(r'^(?:\s|--[^\n]*\n|/\*.*?\*/)*', _re.DOTALL)

def _escape_sql_literal(v):
    """Convert Python value → safe PostgreSQL literal (dollar-quoting / type-aware)."""
    if v is None: return 'NULL'
    if isinstance(v, bool): return 'TRUE' if v else 'FALSE'
    if isinstance(v, int): return str(v)
    if isinstance(v, float):
        if v != v: return 'NULL'
        if v == float('inf'): return "'Infinity'::float8"
        if v == float('-inf'): return "'-Infinity'::float8"
        return repr(v)
    if isinstance(v, _decimal.Decimal): return str(v)
    if isinstance(v, (bytes, bytearray)): return f"decode('{v.hex()}','hex')"
    if isinstance(v, datetime): 
        return f"'{v.isoformat()}'::timestamptz" if v.tzinfo else f"'{v.isoformat()}'::timestamp"
    if isinstance(v, (list, tuple)): return f"ARRAY[{','.join(_escape_sql_literal(x) for x in v)}]"
    if isinstance(v, dict):
        import json as _j; return f"'{_j.dumps(v, default=_http_json_serial)}'::jsonb"
    s = str(v); tag = '$qtcl$'
    if tag in s:
        return "E'" + s.replace('\\','\\\\').replace("'","\\'") + "'"
    return f"{tag}{s}{tag}"

def _substitute_params(sql, params):
    if not params: return sql
    if isinstance(params, dict):
        return _PARAM_RE.sub(lambda m: _escape_sql_literal(params[m.group(1)]) if m.group(1) else (_ for _ in ()).throw(ValueError("mixed placeholders")), sql)
    it = iter(params)
    return _PARAM_RE.sub(lambda m: _escape_sql_literal(next(it)), sql)

def _classify_sql(sql):
    first = _COMMENT_STRIP.sub('', sql).lstrip().split()
    kw = first[0].lower().rstrip(';') if first else ''
    if kw in _SELECT_FIRST: return 'select'
    if kw in _WRITE_FIRST:  return 'write'
    return 'select'  # unknown → try as select

def _suphttp_exec_select(sql):
    cfg = _suphttp_cfg()
    raw = _http_post_json(f"{cfg['url']}/rest/v1/rpc/exec_sql_select", cfg['headers'],
                          {'query': sql}, cfg['timeout'], cfg['retries'])
    # PostgREST wraps JSONB RPC result: [{exec_sql_select: [...]}, ...] or [[...]] or [...]
    if isinstance(raw, list) and raw:
        inner = raw[0]
        if isinstance(inner, dict):
            vals = list(inner.values())
            if len(vals) == 1 and isinstance(vals[0], list): return vals[0]
            return [inner]
        if isinstance(inner, list): return inner
        return raw
    if isinstance(raw, dict):
        vals = list(raw.values())
        if len(vals) == 1 and isinstance(vals[0], list): return vals[0]
        return [raw]
    return []

def _suphttp_exec_write(sql):
    cfg = _suphttp_cfg()
    raw = _http_post_json(f"{cfg['url']}/rest/v1/rpc/exec_sql_write", cfg['headers'],
                          {'query': sql}, cfg['timeout'], cfg['retries'])
    if isinstance(raw, list) and raw: raw = raw[0]
    if isinstance(raw, dict):
        inner = raw.get('exec_sql_write') or raw
        if isinstance(inner, dict): return int(inner.get('affected_rows', 0))
    return 0

class _SupHTTPCursor:
    """psycopg2-compatible cursor backed by Supabase PostgREST HTTPS RPC."""
    def __init__(self):
        self._rows: List[tuple] = []; self._pos = 0; self._rowcount = -1
        self._description = None; self.closed = False
    @property
    def rowcount(self): return self._rowcount
    @property
    def description(self): return self._description
    def mogrify(self, sql, params=None): return _substitute_params(sql, params)
    def execute(self, sql, params=None):
        if self.closed: raise RuntimeError("cursor closed")
        final = _substitute_params(sql, params)
        logger.debug(f"[SUPHTTP] execute: {final[:100]}{'...' if len(final)>100 else ''}")
        if _classify_sql(final) == 'select':
            rows_dicts = _suphttp_exec_select(final)
            if not rows_dicts: self._rows=[]; self._pos=0; self._rowcount=0; self._description=None; return
            keys = list(rows_dicts[0].keys())
            self._description = [(k,None,None,None,None,None,None) for k in keys]
            self._rows = [tuple(r.get(k) for k in keys) for r in rows_dicts]
            self._pos = 0; self._rowcount = len(self._rows)
        else:
            self._rowcount = _suphttp_exec_write(final)
            self._rows=[]; self._pos=0; self._description=None
    def executemany(self, sql, seq):
        for p in seq: self.execute(sql, p)
    def fetchone(self):
        if self._pos < len(self._rows):
            row = self._rows[self._pos]; self._pos += 1; return row
        return None
    def fetchall(self):
        rows = self._rows[self._pos:]; self._pos = len(self._rows); return rows
    def fetchmany(self, size=1):
        rows = self._rows[self._pos:self._pos+size]; self._pos += len(rows); return rows
    def __iter__(self):
        while self._pos < len(self._rows): yield self.fetchone()
    def close(self): self.closed = True; self._rows = []
    def __enter__(self): return self
    def __exit__(self, *_): self.close()

class _SupHTTPConn:
    """psycopg2-compatible connection backed by Supabase PostgREST HTTPS RPC.
    commit()/rollback() are no-ops — PostgREST RPC is auto-committed per call.
    .closed mirrors psycopg2 int semantics: 0=open, 1=closed, 2=lost."""
    def __init__(self): self.closed = 0; self.autocommit = True   # 0 = open (psycopg2 int semantics)
    def cursor(self, *_, **__):
        if self.closed: raise RuntimeError("connection closed")
        return _SupHTTPCursor()
    def commit(self): pass   # no-op: stateless HTTPS
    def rollback(self):
        logger.debug("[SUPHTTP] rollback() — HTTP connections are auto-committed; no-op")
    def close(self): self.closed = 1
    def set_session(self, *_, **__): pass
    def set_isolation_level(self, *_, **__): pass
    def __enter__(self): return self
    def __exit__(self, exc_type, *_):
        if not exc_type: self.commit()
        else: self.rollback()
        return False

class _SupHTTPPool:
    """Thread-safe free-list pool of _SupHTTPConn objects."""
    def __init__(self, minconn=1, maxconn=20):
        self._max = maxconn; self._lock = threading.Lock()
        self._free: List[_SupHTTPConn] = []; self._in_use: List[_SupHTTPConn] = []
        self.closed = False
    def getconn(self, key=None):
        if self.closed: raise RuntimeError("HTTP pool closed")
        with self._lock:
            while self._free:
                c = self._free.pop()
                if not c.closed: self._in_use.append(c); return c
            if len(self._in_use) < self._max:
                c = _SupHTTPConn(); self._in_use.append(c); return c
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            time.sleep(0.05)
            with self._lock:
                while self._free:
                    c = self._free.pop()
                    if not c.closed: self._in_use.append(c); return c
        raise RuntimeError("[SUPHTTP] Pool exhausted after 30s")
    def putconn(self, conn, close=False, key=None):
        if conn is None: return
        with self._lock:
            if conn in self._in_use: self._in_use.remove(conn)
            if close or conn.closed or self.closed: conn.close()
            else: conn.closed = False; self._free.append(conn)
    def closeall(self):
        with self._lock:
            self.closed = True
            for c in self._free + self._in_use:
                try: c.close()
                except Exception: pass
            self._free.clear(); self._in_use.clear()

def _suphttp_test_connection() -> bool:
    try:
        rows = _suphttp_exec_select("SELECT 1 AS ping, NOW() AS ts")
        ok = bool(rows and rows[0].get('ping') == 1)
        if ok: logger.info(f"[SUPHTTP] ✓ connection test passed — server ts={rows[0].get('ts')}")
        else:  logger.warning(f"[SUPHTTP] ⚠ unexpected test response: {rows}")
        return ok
    except Exception as e:
        logger.error(f"[SUPHTTP] ✗ connection test FAILED: {e}"); return False

# ═════════════════════════════════════════════════════════════════════════════════
# DATABASE POOL
# ═════════════════════════════════════════════════════════════════════════════════

class DatabasePool:
    """Thread-safe connection pool.  Transparently switches between:
       • psycopg2 TCP pool (Koyeb / any server with direct Supabase TCP access)
       • _SupHTTPPool  HTTP pool (PythonAnywhere where outbound TCP 5432/6543 is blocked)
    Controlled by USE_HTTP_DB=1 environment variable."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
                    cls._instance.pool = None
                    cls._instance.use_pooling = True
                    cls._instance._http_mode = False
                    cls._instance._next_retry_at = 0.0   # PATCH-9: retry backoff timestamp
                    cls._instance._retry_interval = 5.0  # PATCH-9: seconds between init attempts
        return cls._instance

    def _initialize_pool(self):
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return

            # ── PATCH-9: Retry backoff gate ───────────────────────────────────
            # Every failed init leaves _initialized=False.  Without this gate,
            # each db_ready() / get_db_connection() call in every background
            # thread immediately retries → thundering herd of psycopg2
            # ThreadedConnectionPool() calls each opening min_connections TCP
            # sessions against Supabase → MaxClientsInSessionMode exhaustion.
            # Gate enforces a minimum _retry_interval between attempts,
            # doubling on each failure up to 60 s.
            _now = time.monotonic()
            if _now < self._next_retry_at:
                return   # too soon — let the backoff expire
            # ─────────────────────────────────────────────────────────────────
            # ── HTTP mode (PythonAnywhere) ────────────────────────────────────
            if _USE_HTTP_DB:
                try:
                    _suphttp_cfg()   # validate SUPABASE_URL / SUPABASE_SERVICE_KEY present
                    if not _suphttp_test_connection():
                        logger.error("[DB] ❌ Supabase HTTP connection test failed — "
                                     "check SUPABASE_URL and SUPABASE_SERVICE_KEY")
                        # Don't mark initialized so it retries on next request
                        return
                    self.pool = _SupHTTPPool(
                        minconn=int(os.getenv('DB_POOL_MIN', '1')),
                        maxconn=int(os.getenv('DB_POOL_MAX', '20')),
                    )
                    self._initialized = True
                    self.use_pooling  = True
                    self._http_mode   = True
                    logger.info("[DB] ✨ Connected to Supabase via HTTPS PostgREST RPC (HTTP-DB mode)")
                except EnvironmentError as e:
                    logger.error(f"[DB] ❌ HTTP-DB config error: {e}")
                    self._initialized    = False
                    self._retry_interval = min(self._retry_interval * 2, 60.0)
                    self._next_retry_at  = time.monotonic() + self._retry_interval
                except Exception as e:
                    logger.error(f"[DB] ❌ HTTP-DB init error: {e}")
                    self._initialized    = False
                    self._retry_interval = min(self._retry_interval * 2, 60.0)
                    self._next_retry_at  = time.monotonic() + self._retry_interval
                return

            # ── Native psycopg2 TCP mode (Koyeb / self-hosted) ───────────────
            try:
                from psycopg2 import pool as psycopg2_pool
                # min=1: open only ONE connection on pool creation — avoids
                # exhausting Supabase session-mode slots during retry storms.
                min_connections = 1
                max_connections = int(os.getenv('DB_POOL_MAX', '10'))
                logger.info(f"[DB] Initializing app-level pooling: min={min_connections}, max={max_connections}")
                logger.info(f"[DB] Connecting to Supabase pooler (aws-0-us-west-2.pooler.supabase.com)")
                self.pool = psycopg2_pool.ThreadedConnectionPool(
                    min_connections, max_connections, DB_URL, connect_timeout=10)
                self._initialized = True
                self.use_pooling  = True
                self._next_retry_at   = 0.0         # reset backoff on success
                self._retry_interval  = 5.0
                logger.info("[DB] ✨ Connected to Supabase pooler successfully")
            except (ImportError, AttributeError):
                logger.info("[DB] App-level pooling unavailable, using direct connections")
                logger.info("[DB] ✨ Connected to Supabase pooler (direct mode)")
                self._initialized = True
                self.use_pooling  = False
                self.pool         = None
                self._next_retry_at  = 0.0
                self._retry_interval = 5.0
            except (psycopg2.OperationalError if psycopg2 else Exception) as e:
                logger.error(f"[DB] ❌ Cannot connect to Supabase pooler: {e}")
                logger.error("[DB] Check POOLER_* environment variables are set correctly")
                self._initialized = False
                self.use_pooling  = False
                # Advance backoff: double interval up to 60 s
                self._retry_interval = min(self._retry_interval * 2, 60.0)
                self._next_retry_at  = time.monotonic() + self._retry_interval
                logger.warning(f"[DB] ⏳ Next init retry in {self._retry_interval:.0f}s")
            except Exception as e:
                logger.error(f"[DB] Error initializing pool: {e}")
                self._initialized = True
                self.use_pooling  = False
                self.pool         = None
                self._next_retry_at  = 0.0
                self._retry_interval = 5.0

    def get_connection(self):
        if not self._initialized:
            self._initialize_pool()
        try:
            if self._http_mode and self.pool:
                return self.pool.getconn()
            if self.use_pooling and self.pool:
                conn = self.pool.getconn()
                if conn is None:
                    logger.debug("[DB] Pool exhausted, creating direct connection via pooler")
                    conn = psycopg2.connect(DB_URL, connect_timeout=10)
                return conn
            return psycopg2.connect(DB_URL, connect_timeout=10)
        except psycopg2.OperationalError as e:
            logger.error(f"[DB] ❌ Cannot connect to Supabase pooler: {e}")
            logger.error(f"[DB] Check POOLER_URL: {DB_URL[:50]}...")
            raise
        except Exception as e:
            logger.error(f"[DB] Connection error: {e}")
            raise

    def put_connection(self, conn):
        try:
            if self._http_mode and self.pool and conn:
                self.pool.putconn(conn)
            elif self.use_pooling and self.pool and conn:
                self.pool.putconn(conn)
            elif conn:
                conn.close()
        except Exception as e:
            logger.debug(f"[DB] Error handling connection return: {e}")

    def close_all(self):
        try:
            if self.pool:
                self.pool.closeall()
                logger.info("[DB] Connection pool closed")
        except Exception as e:
            logger.debug(f"[DB] Error closing pool: {e}")


# Global pool instance (singleton, lazy-initialized)
db_pool = DatabasePool()


# ─── PATCH-2: db_ready() ─────────────────────────────────────────────────────
# Called at ~line 459/483 inside get_oracle_address() / get_consensus_oracle_address()
# but was NEVER DEFINED anywhere — NameError on every call, silently swallowed
# by those functions' broad except blocks → silent fallback values forever.
def db_ready() -> bool:
    """Return True if the DB pool is usable; triggers lazy init if needed."""
    try:
        if not db_pool._initialized:
            db_pool._initialize_pool()
        return db_pool._initialized
    except Exception as _e:
        logger.debug(f"[DB] db_ready() check failed: {_e}")
        return False


# ─── PATCH-3: get_db_connection() ────────────────────────────────────────────
# Called at ~line 462/486 inside get_oracle_address() / get_consensus_oracle_address()
# but was NEVER DEFINED anywhere — same silent-NameError failure path as above.
# Caller owns the connection: must call db_pool.put_connection(conn) when done.
def get_db_connection():
    """Return a raw psycopg2 connection from the pool (lazy init on first call)."""
    if not db_pool._initialized:
        db_pool._initialize_pool()
    return db_pool.get_connection()


# ═══════════════════════════════════════════════════════════════════════════════
# CHAIN QUERY FUNCTIONS (Supabase PostgreSQL only — source of truth)
# Clients maintain their own SQLite mirrors, synced via P2P broadcasts
# ═══════════════════════════════════════════════════════════════════════════════

def query_latest_block() -> Optional[Dict[str, Any]]:
    """Get latest block from Supabase PostgreSQL (authoritative source)."""
    try:
        with get_db_cursor() as cur:
            cur.execute("SELECT height, block_hash, timestamp FROM blocks ORDER BY height DESC LIMIT 1")
            row = cur.fetchone()
            if row:
                return {"height": row[0], "hash": row[1] or "", "timestamp": row[2] or 0}
    except Exception as e:
        logger.debug(f"[QUERY-LATEST] PG error: {e}")
    return None

def query_block_by_height(height: int) -> Optional[Dict[str, Any]]:
    """Get block by height from Supabase PostgreSQL (authoritative source)."""
    try:
        with get_db_cursor() as cur:
            cur.execute("SELECT * FROM blocks WHERE height = %s", (height,))
            row = cur.fetchone()
            if row:
                cols = [desc[0] for desc in cur.description]
                return dict(zip(cols, row))
    except Exception as e:
        logger.debug(f"[QUERY-BLOCK] PG error: {e}")
    return None

def query_block_by_hash(block_hash: str) -> Optional[Dict[str, Any]]:
    """Get block by hash from Supabase PostgreSQL (authoritative source)."""
    if not block_hash:
        return None
    try:
        with get_db_cursor() as cur:
            cur.execute("SELECT * FROM blocks WHERE block_hash = %s LIMIT 1", (block_hash,))
            row = cur.fetchone()
            if row:
                cols = [desc[0] for desc in cur.description]
                return dict(zip(cols, row))
    except Exception as e:
        logger.debug(f"[QUERY-BLOCK-HASH] PG error: {e}")
    return None


@contextmanager
def get_db_cursor():
    """Context manager for database cursor with connection pooling.
    
    ⚛️  CRITICAL: Return connections to pool, never close them directly.
    Closing breaks the pool. Must use db_pool.putconn() to return.
    """
    conn=None
    try:
        conn=db_pool.get_connection()
        cur=conn.cursor()
        yield cur
        conn.commit()
    except Exception as e:
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        logger.debug(f"[DB-CURSOR] Error: {e}")
        raise
    finally:
        if conn:
            try:
                if db_pool.use_pooling and db_pool.pool:
                    db_pool.pool.putconn(conn)
                else:
                    conn.close()
            except Exception as e:
                logger.debug(f"[DB-CURSOR] putconn error: {e}")
                try:
                    conn.close()
                except Exception:
                    pass

_dht_manager: Optional[DHTManager] = None
_dht_lock = threading.RLock()

def get_dht_manager() -> DHTManager:
    """Get or create global DHT manager. Uses P2P_PORT (9091) — not gunicorn HTTP PORT."""
    global _dht_manager
    if _dht_manager is None:
        # Public hostname so remote peers can reach this node.
        # Falls back to 0.0.0.0 for local/dev use.
        host = (os.getenv('KOYEB_PUBLIC_DOMAIN') or
                os.getenv('RAILWAY_PUBLIC_DOMAIN') or
                os.getenv('FLASK_HOST') or '0.0.0.0')
        port = P2P_PORT  # 9091 — never gunicorn's HTTP port
        _dht_manager = DHTManager(local_address=host, local_port=port)
    return _dht_manager


# ═════════════════════════════════════════════════════════════════════════════════════════
# RPC SNAPSHOT DISTRIBUTION (JSON polling — no SSE)
# ═════════════════════════════════════════════════════════════════════════════════════════

# RPC snapshot cache + event log (no SSE infrastructure)
_rpc_event_log: Deque = Deque(maxlen=1000)           # Ring buffer of recent RPC events
_rpc_event_lock = threading.RLock()                  # Guards _rpc_event_log writes
_latest_snapshot: Optional[dict] = None              # Last cached snapshot (poll endpoint)
_latest_snapshot_ts: int = 0                         # Timestamp of latest snapshot
_snapshot_lock = threading.RLock()                   # Guards _latest_snapshot updates

def _cache_snapshot(snapshot: dict) -> None:
    """Cache snapshot for RPC polling (no SSE push)."""
    global _latest_snapshot
    with _snapshot_lock:
        _latest_snapshot = snapshot

def _log_rpc_event(event_type: str, data: Any) -> None:
    """Log event for /api/events RPC polling endpoint."""
    with _rpc_event_lock:
        _rpc_event_log.append({'ts': time.time(), 'type': event_type, 'data': data})




# Application startup flag
_APP_READY=False

def _set_app_ready():
    global _APP_READY
    _APP_READY=True
    logger.info("[APP] ✅ Application ready for Koyeb health checks")

# ═════════════════════════════════════════════════════════════════════════════════════════
# API Endpoints
# ═════════════════════════════════════════════════════════════════════════════════════════

class SafeFieldConverter:
    """⚛️  Autonomous diagnostic field converter with fallback recovery."""
    
    _errors = {}  # Track which fields fail across requests for autonomous healing
    
    @staticmethod
    def safe_int(value, field_name='unknown', default=0):
        """Convert to int with diagnostic logging."""
        try:
            if value is None: return default
            return int(value)
        except (ValueError, TypeError) as e:
            SafeFieldConverter._errors[f'int_{field_name}'] = str(e)
            logger.warning(f"[CONVERTER] int({field_name})={value} failed: {e}")
            return default
    
    @staticmethod
    def safe_float(value, field_name='unknown', default=0.0):
        """Convert to float with diagnostic logging."""
        try:
            if value is None: return default
            return float(value)
        except (ValueError, TypeError) as e:
            SafeFieldConverter._errors[f'float_{field_name}'] = str(e)
            logger.warning(f"[CONVERTER] float({field_name})={value} failed: {e}")
            return default
    
    @staticmethod
    def safe_str(value, field_name='unknown', default=''):
        """Convert to str with diagnostic logging."""
        try:
            if value is None: return default
            return str(value)
        except (ValueError, TypeError) as e:
            SafeFieldConverter._errors[f'str_{field_name}'] = str(e)
            logger.warning(f"[CONVERTER] str({field_name})={value} failed: {e}")
            return default
    
    @staticmethod
    def safe_bool(value, field_name='unknown', default=False):
        """Convert to bool with diagnostic logging."""
        try:
            if value is None: return default
            return bool(value)
        except (ValueError, TypeError) as e:
            SafeFieldConverter._errors[f'bool_{field_name}'] = str(e)
            logger.warning(f"[CONVERTER] bool({field_name})={value} failed: {e}")
            return default
    
    @staticmethod
    def get_error_report():
        """Return accumulated conversion errors for autonomous healing."""
        return dict(SafeFieldConverter._errors)
    
    @staticmethod
    def clear_errors():
        """Clear error history for next diagnostic cycle."""
        SafeFieldConverter._errors.clear()


class SnapshotAutonomousHealer:
    """⚛️  Autonomous diagnostic & healing loop for snapshot failures."""
    
    _last_valid_snapshot = {}  # Cache last valid state for fallback
    _healing_cycles = 0
    _healing_lock = threading.Lock()
    
    @staticmethod
    def build_from_row(row, diag_label='oracle_snapshot_json'):
        """Build snapshot with autonomous error detection & recovery."""
        if not row:
            return None, {'error': 'row_is_none', 'ready': False}
        
        diag = {'cycles': 0, 'errors': [], 'recovered': []}
        SafeFieldConverter.clear_errors()
        
        try:
            # ⚛️  Phase 1: Parse oracle measurements (most failure-prone)
            oracles = []
            try:
                if isinstance(row[18], list):
                    oracles = row[18]
                elif isinstance(row[18], str):
                    oracles = json.loads(row[18])
                else:
                    oracles = []
            except (json.JSONDecodeError, TypeError) as e:
                diag['errors'].append(f'oracle_measurements: {str(e)[:50]}')
                logger.warning(f"[HEALER] Oracle measurements parse failed: {e}")
                oracles = []
            
            # ⚛️  Phase 2: Parse mermin result (nullable)
            mermin_result = None
            if row[10] is not None:
                try:
                    m_val = SafeFieldConverter.safe_float(row[10], 'mermin_M')
                    mermin_result = {
                        'M_value': m_val,
                        'M': m_val,
                        'is_quantum': SafeFieldConverter.safe_bool(row[11], 'mermin_is_quantum'),
                        'verdict': SafeFieldConverter.safe_str(row[12], 'mermin_verdict')
                    }
                except Exception as e:
                    diag['errors'].append(f'mermin_result: {str(e)[:50]}')
                    logger.warning(f"[HEALER] Mermin result construction failed: {e}")
                    mermin_result = None
            
            # ⚛️  Phase 3: Safe numeric conversions with field-level diagnostics
            ts_ns = SafeFieldConverter.safe_int(row[1], 'timestamp_ns')
            chirp = SafeFieldConverter.safe_int(row[2], 'chirp_number')
            lat_f = SafeFieldConverter.safe_float(row[3], 'lattice_fidelity')
            lat_c = SafeFieldConverter.safe_float(row[4], 'lattice_coherence')
            lat_cy = SafeFieldConverter.safe_int(row[5], 'lattice_cycle')
            lat_s8 = SafeFieldConverter.safe_int(row[6], 'lattice_sigma_mod8')
            cons_f = SafeFieldConverter.safe_float(row[7], 'consensus_fidelity')
            cons_c = SafeFieldConverter.safe_float(row[8], 'consensus_coherence')
            cons_p = SafeFieldConverter.safe_float(row[9], 'consensus_purity')
            pq0_o = SafeFieldConverter.safe_float(row[13], 'pq0_oracle_fidelity')
            pq0_i = SafeFieldConverter.safe_float(row[14], 'pq0_IV_fidelity')
            pq0_v = SafeFieldConverter.safe_float(row[15], 'pq0_V_fidelity')
            pq_c = SafeFieldConverter.safe_int(row[16], 'pq_curr')
            pq_l = SafeFieldConverter.safe_int(row[17], 'pq_last')
            phase = SafeFieldConverter.safe_str(row[19], 'phase_name')
            
            # ⚛️  Collect conversion errors for autonomous healing
            conv_errors = SafeFieldConverter.get_error_report()
            if conv_errors:
                diag['errors'].extend([f'{k}' for k in conv_errors.keys()])
                logger.warning(f"[HEALER] Conversion errors detected: {len(conv_errors)}")
            
            # ⚛️  Phase 4: Construct snapshot with all safe values
            snapshot = {
                'timestamp_ns': ts_ns,
                'chirp_number': chirp,
                'lattice_quantum': {
                    'fidelity': lat_f,
                    'coherence': lat_c,
                    'cycle_count': lat_cy,
                    'lattice_sigma_mod8': lat_s8,
                    'phase_name': phase,
                    'lattice_status': 'online'
                },
                'consensus': {
                    'w_state_fidelity': cons_f,
                    'coherence': cons_c,
                    'purity': cons_p
                },
                'mermin_test': mermin_result,
                'bell_test': mermin_result,
                'pq0_components': {
                    'pq0_oracle_fidelity': pq0_o,
                    'pq0_IV_fidelity': pq0_i,
                    'pq0_V_fidelity': pq0_v
                },
                'pq_curr': pq_c,
                'pq_last': pq_l,
                'oracle_measurements': oracles,
                'fidelity': cons_f,
                'coherence': cons_c,
                'lattice_cycle': lat_cy,
                'source': 'supabase_snapshot_healed',
                'ready': True,
                '_diagnostics': {
                    'errors': diag['errors'],
                    'recovered_with_defaults': bool(conv_errors),
                    'conversion_errors': conv_errors
                }
            }
            
            # ⚛️  Cache this as last valid state for future fallback
            with SnapshotAutonomousHealer._healing_lock:
                SnapshotAutonomousHealer._last_valid_snapshot = snapshot.copy()
                SnapshotAutonomousHealer._healing_cycles += 1
            
            if not diag['errors']:
                logger.debug(f"[HEALER] Snapshot built clean (cycle {SnapshotAutonomousHealer._healing_cycles})")
            else:
                logger.info(f"[HEALER] Snapshot built with {len(diag['errors'])} recovered fields (cycle {SnapshotAutonomousHealer._healing_cycles})")
            
            return snapshot, diag
        
        except Exception as e:
            # ⚛️  Catastrophic failure — fall back to cached state
            logger.error(f"[HEALER] Snapshot construction catastrophically failed: {e}")
            with SnapshotAutonomousHealer._healing_lock:
                if SnapshotAutonomousHealer._last_valid_snapshot:
                    logger.warning(f"[HEALER] Falling back to last valid cached snapshot")
                    return SnapshotAutonomousHealer._last_valid_snapshot.copy(), {
                        'error': 'catastrophic_fallback',
                        'fallback_source': 'cache',
                        'ready': True
                    }
            return None, {'error': 'catastrophic_failure', 'details': str(e), 'ready': False}




# ══════════════════════════════════════════════════════════════════════════════
# JSON-RPC 2.0 FLASK ROUTES
# ══════════════════════════════════════════════════════════════════════════════
def _get_canonical_node() -> Optional[dict]:
    """Fallback: fetch canonical node state from module or globals (in-memory)."""
    try:
        import globals as _g
        gn = getattr(_g, "get_canonical_node", None)
        if callable(gn):
            return gn()
    except Exception:
        pass
    
    # Last resort: check module-level state
    try:
        _srv = sys.modules[__name__].__dict__
        cn = _srv.get("_canonical_node") or _srv.get("canonical_node")
        return cn if isinstance(cn, dict) else None
    except Exception:
        return None


def _rpc_getBlockHeight(params: Any, rpc_id: Any) -> dict:
    """qtcl_getBlockHeight — current chain tip height.
    
    DB-AUTHORITATIVE: reads from PostgreSQL blocks table via query_latest_block().
    In-memory _get_canonical_node() is stale across gunicorn workers.
    Falls back to in-memory only if DB query fails.
    """
    try:
        logger.debug(f"[RPC-METHOD] qtcl_getBlockHeight called with params={params}, id={rpc_id}")
        
        # PRIMARY: DB is authoritative (survives worker restarts, multi-worker consistent)
        db_tip = query_latest_block()
        if db_tip is not None:
            height   = int(db_tip['height'])
            tip_hash = str(db_tip.get('hash', ''))
            result = {
                "height":   height,
                "tip_hash": tip_hash,
                "ts":       time.time(),
            }
            logger.debug(f"[RPC-METHOD] qtcl_getBlockHeight (DB) success: height={height}")
            return _rpc_ok(result, rpc_id)
        
        # FALLBACK: in-memory canonical node (cold start before first block)
        node = _get_canonical_node()
        if node is not None:
            height   = node.get("block_height", 0)
            tip_hash = node.get("tip_hash", "")
            result = {
                "height":   height,
                "tip_hash": tip_hash,
                "ts":       time.time(),
            }
            logger.debug(f"[RPC-METHOD] qtcl_getBlockHeight (in-memory) success: height={height}")
            return _rpc_ok(result, rpc_id)
        
        # Genesis state — no blocks yet, height=0
        result = {
            "height":   0,
            "tip_hash": "0" * 64,
            "ts":       time.time(),
        }
        logger.debug(f"[RPC-METHOD] qtcl_getBlockHeight: no blocks yet, returning genesis")
        return _rpc_ok(result, rpc_id)
    except Exception as e:
        logger.exception(f"[RPC-METHOD] qtcl_getBlockHeight exception: {e}")
        return _rpc_error(-32603, f"Internal error: {str(e)}", rpc_id, {"exception": str(e).__class__.__name__})


def _rpc_getBalance(params: Any, rpc_id: Any) -> dict:
    """qtcl_getBalance — address QTCL balance via direct DB query."""
    try:
        if not isinstance(params, (list, dict)):
            return _rpc_error(-32602, "params must be list or object", rpc_id)
        address = (params[0] if isinstance(params, list) else params.get("address", "")) if params else ""
        if not address:
            return _rpc_error(-32602, "address required", rpc_id)
        try:
            with get_db_cursor() as cur:
                cur.execute(
                    "SELECT balance, transaction_count FROM wallet_addresses WHERE address = %s",
                    (address,)
                )
                row = cur.fetchone()
                if row:
                    wallet = {
                        'address': address,
                        'balance': row[0],
                        'tx_count': row[1],
                    }
                else:
                    wallet = None
        except Exception as _wqe:
            logger.debug(f"[RPC] query_wallet_info DB error: {_wqe}")
            wallet = None
        if wallet is None:
            # Address not yet in DB — return 0 balance, not an error
            result = {
                "address": address,
                "balance": 0.0,
                "symbol":  "QTCL",
            }
        else:
            raw_balance = int(wallet.get('balance') or 0)
            result = {
                "address": address,
                "balance": raw_balance / 100.0,
                "symbol":  "QTCL",
            }
        logger.debug(f"[RPC-METHOD] qtcl_getBalance success: address={address}, balance={result['balance']}")
        return _rpc_ok(result, rpc_id)
    except Exception as e:
        logger.exception(f"[RPC-METHOD] qtcl_getBalance outer exception: {e}")
        return _rpc_error(-32603, f"Internal error: {str(e)}", rpc_id)


def _rpc_getTransaction(params: Any, rpc_id: Any) -> dict:
    """qtcl_getTransaction — tx details by hash."""
    try:
        logger.debug(f"[RPC-METHOD] qtcl_getTransaction called with params={params}, id={rpc_id}")
        tx_hash = (params[0] if isinstance(params, list) else params.get("tx_hash", "")) if params else ""
        if not tx_hash:
            logger.debug(f"[RPC-METHOD] qtcl_getTransaction: tx_hash missing or empty")
            return _rpc_error(-32602, "tx_hash required", rpc_id)
        try:
            from globals import get_blockchain
            bc = get_blockchain()
            if bc is None:
                logger.warning(f"[RPC-METHOD] qtcl_getTransaction: blockchain not initialized")
                return _rpc_error(-32003, "Blockchain not synced", rpc_id)
            tx = bc.get_transaction(tx_hash)
            if tx is None:
                logger.debug(f"[RPC-METHOD] qtcl_getTransaction: tx not found (hash={tx_hash})")
                return _rpc_error(-32000, "Transaction not found", rpc_id, {"tx_hash": tx_hash})
            logger.debug(f"[RPC-METHOD] qtcl_getTransaction success: tx_hash={tx_hash}")
            return _rpc_ok(tx, rpc_id)
        except Exception as be:
            logger.exception(f"[RPC-METHOD] qtcl_getTransaction: blockchain error: {be}")
            return _rpc_error(-32603, f"TX lookup failed: {str(be)}", rpc_id, {"exception": str(be).__class__.__name__})
    except Exception as e:
        logger.exception(f"[RPC-METHOD] qtcl_getTransaction outer exception: {e}")
        return _rpc_error(-32603, f"Internal error: {str(e)}", rpc_id, {"exception": str(e).__class__.__name__})


def _rpc_getBlock(params: Any, rpc_id: Any) -> dict:
    """qtcl_getBlock — block by height or hash.

    DB-AUTHORITATIVE: queries PostgreSQL blocks table directly.
    params: [height] (list) or {height: int} or {hash: str}
    Returns full block header + transaction list for chain sync.
    """
    try:
        height = None
        block_hash = None
        if isinstance(params, list) and len(params) >= 1:
            height = int(params[0])
        elif isinstance(params, dict):
            height = params.get("height")
            block_hash = params.get("hash")
            if height is not None:
                height = int(height)

        def _query_block_at_height(h: int) -> Optional[dict]:
            """Full block query from Supabase PostgreSQL (authoritative source)."""
            try:
                with get_db_cursor() as cur:
                    cur.execute("""
                        SELECT height, block_hash, timestamp, oracle_w_state_hash,
                               previous_hash, validator_public_key, nonce, difficulty,
                               entropy_score, transactions_root, pq_curr, pq_last
                        FROM blocks WHERE height = %s LIMIT 1
                    """, (h,))
                    row = cur.fetchone()
                    if not row:
                        return None
                    block = {
                        'height':           row[0],
                        'block_height':     row[0],
                        'block_hash':       row[1],
                        'hash':             row[1],
                        'parent_hash':      row[4] or ('0' * 64),
                        'previous_hash':    row[4] or ('0' * 64),
                        'merkle_root':      row[9] or ('0' * 64),
                        'timestamp_s':      int(row[2]) if row[2] else 0,
                        'timestamp':        int(row[2]) if row[2] else 0,
                        'difficulty_bits':  int(float(row[7])) if row[7] else 4,
                        'difficulty':       int(float(row[7])) if row[7] else 4,
                        'nonce':            int(row[6]) if row[6] else 0,
                        'miner_address':    row[5] or '',
                        'w_state_fidelity': float(row[8]) if row[8] is not None else 0.0,
                        'w_entropy_hash':   row[3] or '',
                        'pq_curr':          int(row[10]) if row[10] is not None else h,
                        'pq_last':          int(row[11]) if row[11] is not None else max(0, h - 1),
                    }
                    # Fetch transactions for this block
                    cur.execute("""
                        SELECT tx_hash, from_address, to_address, amount,
                               transaction_index, tx_type, status,
                               quantum_state_hash, metadata
                        FROM transactions
                        WHERE height = %s
                        ORDER BY transaction_index ASC
                    """, (h,))
                    tx_rows = cur.fetchall()
                    txs = []
                    for tr in tx_rows:
                        txs.append({
                            'tx_id':        tr[0],
                            'from_addr':    tr[1] or '',
                            'to_addr':      tr[2] or '',
                            'amount':       int(tr[3]) if tr[3] is not None else 0,
                            'tx_index':     int(tr[4]) if tr[4] is not None else 0,
                            'tx_type':      tr[5] or 'transfer',
                            'status':       tr[6] or 'confirmed',
                            'w_proof':      tr[7] or '',
                            'metadata':     tr[8] if tr[8] else None,
                        })
                    block['transactions'] = txs
                    block['tx_count'] = len(txs)
                    return block
            except Exception as e:
                logger.exception(f"[RPC] _query_block_at_height({h}): {e}")
                return None

        block = None
        if height is not None:
            block = _query_block_at_height(height)
        elif block_hash:
            row = query_block_by_hash(block_hash)
            if row:
                block = _query_block_at_height(row['height'])

        if block is None:
            return _rpc_error(-32000, "Block not found", rpc_id)

        return _rpc_ok(block, rpc_id)

    except Exception as e:
        logger.exception(f"[RPC] _rpc_getBlock exception: {e}")
        return _rpc_error(-32603, f"Internal error: {str(e)}", rpc_id)


def _rpc_getBlockRange(params: Any, rpc_id: Any) -> dict:
    """qtcl_getBlockRange — batch fetch blocks by height range.

    params: [from_height, to_height] — inclusive, max 100 blocks per call.
    Returns list of block headers (no transactions for efficiency).
    Used by miners for Initial Block Download (IBD) chain sync.
    """
    try:
        if not isinstance(params, (list, tuple)) or len(params) < 2:
            return _rpc_error(-32602, "params: [from_height, to_height]", rpc_id)
        from_h = int(params[0])
        to_h = int(params[1])
        # Cap at 100 blocks per request
        if to_h - from_h > 99:
            to_h = from_h + 99
        if from_h < 0:
            from_h = 0

        with get_db_cursor() as cur:
            cur.execute("""
                SELECT height, block_hash, timestamp, oracle_w_state_hash,
                       previous_hash, validator_public_key, nonce, difficulty,
                       entropy_score, transactions_root, pq_curr, pq_last
                FROM blocks
                WHERE height BETWEEN %s AND %s
                ORDER BY height ASC
            """, (from_h, to_h))
            rows = cur.fetchall()

        blocks = []
        for row in rows:
            blocks.append({
                'height':           row[0],
                'block_height':     row[0],
                'block_hash':       row[1],
                'hash':             row[1],
                'parent_hash':      row[4] or ('0' * 64),
                'previous_hash':    row[4] or ('0' * 64),
                'merkle_root':      row[9] or ('0' * 64),
                'timestamp_s':      int(row[2]) if row[2] else 0,
                'timestamp':        int(row[2]) if row[2] else 0,
                'difficulty_bits':  int(float(row[7])) if row[7] else 4,
                'difficulty':       int(float(row[7])) if row[7] else 4,
                'nonce':            int(row[6]) if row[6] else 0,
                'miner_address':    row[5] or '',
                'w_state_fidelity': float(row[8]) if row[8] is not None else 0.0,
                'w_entropy_hash':   row[3] or '',
                'pq_curr':          int(row[10]) if row[10] is not None else row[0],
                'pq_last':          int(row[11]) if row[11] is not None else max(0, row[0] - 1),
            })

        return _rpc_ok({
            'blocks': blocks,
            'count':  len(blocks),
            'from':   from_h,
            'to':     to_h,
        }, rpc_id)

    except Exception as e:
        logger.exception(f"[RPC] _rpc_getBlockRange exception: {e}")
        return _rpc_error(-32603, f"Internal error: {str(e)}", rpc_id)


# ═══ RPC TIMEOUT PROTECTION ═══
_RPC_TIMEOUT_SEC = 5.0

def _call_with_timeout(func, timeout_sec=_RPC_TIMEOUT_SEC, default=None):
    """Call function with timeout using threading (non-blocking for RPC safety)."""
    import queue as _q
    result_q = _q.Queue()
    
    def _target():
        try:
            result_q.put(("ok", func()))
        except Exception as e:
            result_q.put(("error", e))
    
    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec)
    
    try:
        status, value = result_q.get_nowait()
        return value if status == "ok" else default
    except:
        return default


def _rpc_getQuantumMetrics(params: Any, rpc_id: Any) -> dict:
    """qtcl_getQuantumMetrics — live W-state oracle + lattice metrics + density matrix snapshot.
    
    All reads protected with 5s timeouts to prevent RPC hangs.
    """
    try:
        logger.debug(f"[RPC-METHOD] qtcl_getQuantumMetrics called with params={params}, id={rpc_id}")
        result: dict = {"oracle_available": ORACLE_AVAILABLE, "ts": time.time()}
        logger.debug(f"[RPC-METHOD] qtcl_getQuantumMetrics: oracle_available={ORACLE_AVAILABLE}")

        if ORACLE_AVAILABLE and ORACLE is not None:
            try:
                logger.debug(f"[RPC-METHOD] qtcl_getQuantumMetrics: fetching W-state snapshot (timeout=5s)")
                w_snap = _call_with_timeout(
                    lambda: ORACLE_W_STATE_MANAGER.get_latest_snapshot() if ORACLE_W_STATE_MANAGER else None,
                    timeout_sec=5.0
                )
                if w_snap:
                    result["w_state"] = {
                        "purity":     getattr(w_snap, "purity",     None),
                        "entropy":    getattr(w_snap, "entropy",    None),
                        "coherence":  getattr(w_snap, "coherence",  None),
                        "fidelity":   getattr(w_snap, "fidelity",   None),
                        "snapshot_id": getattr(w_snap, "snapshot_id", None),
                    }
                    logger.debug(f"[RPC-METHOD] qtcl_getQuantumMetrics: W-state snapshot obtained")
                else:
                    logger.warning(f"[RPC-METHOD] qtcl_getQuantumMetrics: W-state snapshot is None")
            except Exception as we:
                logger.exception(f"[RPC-METHOD] qtcl_getQuantumMetrics: W-state error: {we}")
                result["w_state_error"] = str(we)

        if LATTICE is not None:
            try:
                logger.debug(f"[RPC-METHOD] qtcl_getQuantumMetrics: fetching lattice state (timeout=5s)")
                # Safe access to LATTICE methods with fallback
                lm = _call_with_timeout(
                    lambda: LATTICE.get_metrics() if hasattr(LATTICE, 'get_metrics') and callable(LATTICE.get_metrics) else {},
                    timeout_sec=5.0,
                    default={}
                ) or {}
                ls = _call_with_timeout(
                    lambda: LATTICE.get_stats() if hasattr(LATTICE, 'get_stats') and callable(LATTICE.get_stats) else {},
                    timeout_sec=5.0,
                    default={}
                ) or {}
                
                # Fallback: use LATTICE attributes directly
                if not lm:
                    lm = {
                        "avg_fidelity_100": getattr(LATTICE, 'avg_fidelity_100', 0.0),
                        "avg_coherence_100": getattr(LATTICE, 'avg_coherence_100', 0.0),
                    }
                if not ls:
                    ls = {
                        "fidelity": getattr(LATTICE, 'fidelity', 0.0),
                        "coherence": getattr(LATTICE, 'coherence', 0.0),
                        "w_state_strength": getattr(LATTICE, 'w_state_strength', 0.0),
                        "cycle": getattr(LATTICE, 'cycle', 0),
                    }
                
                result["lattice"] = {
                    "fidelity":         lm.get("avg_fidelity_100", ls.get("fidelity", 0.0)),
                    "coherence":        lm.get("avg_coherence_100", ls.get("coherence", 0.0)),
                    "w_state_strength": ls.get("w_state_strength", 0.0),
                    "cycle":            ls.get("cycle", 0),
                    "avg_fidelity_100": lm.get("avg_fidelity_100", 0.0),
                    "avg_coherence_100": lm.get("avg_coherence_100", 0.0),
                }
                logger.debug(f"[RPC-METHOD] qtcl_getQuantumMetrics: lattice metrics obtained")
            except Exception as le:
                logger.exception(f"[RPC-METHOD] qtcl_getQuantumMetrics: lattice error: {le}")
                result["lattice_error"] = str(le)

        # ── WIRE DENSITY_MATRIX_HEX ──────────────────────────────────────────────
        # Extract DM from oracle snapshot if available
        try:
            if ORACLE_W_STATE_MANAGER is not None:
                w_snap = _call_with_timeout(
                    lambda: ORACLE_W_STATE_MANAGER.get_latest_snapshot() if ORACLE_W_STATE_MANAGER else None,
                    timeout_sec=2.0
                )
                if w_snap and hasattr(w_snap, 'density_matrix_hex'):
                    dm_hex = getattr(w_snap, 'density_matrix_hex', '')
                    if dm_hex:
                        result["density_matrix_hex"] = dm_hex
                        logger.debug(f"[RPC-METHOD] qtcl_getQuantumMetrics: density_matrix_hex from oracle ({len(dm_hex)} chars)")
                elif w_snap and hasattr(w_snap, 'density_matrix'):
                    # Extract DM as hex if it's raw bytes
                    dm = getattr(w_snap, 'density_matrix', b'')
                    if dm:
                        dm_hex = dm.hex() if isinstance(dm, bytes) else str(dm)
                        result["density_matrix_hex"] = dm_hex
                        logger.debug(f"[RPC-METHOD] qtcl_getQuantumMetrics: density_matrix extracted ({len(dm_hex)} chars)")
        except Exception as dme:
            logger.debug(f"[RPC-METHOD] qtcl_getQuantumMetrics: density_matrix extraction (non-fatal): {dme}")

        # ── INJECT block_height from DB so client oracle display shows correct chain tip ──
        try:
            _db_tip = query_latest_block()
            _bh = int(_db_tip['height']) if _db_tip else 0
        except Exception:
            _bh = 0
        result['block_height'] = _bh
        result['height']       = _bh

        # ── Inject client tripartite pool consensus fields ─────────────────
        try:
            with _CLIENT_DM_POOL_LOCK:
                result['client_fused_fidelity'] = round(_client_consensus_fid, 6)
                result['client_oracle_count']   = _client_pool_count
                if _client_pool_count > 0 and any(v != 0.0 for v in _client_consensus_dm_re):
                    import struct as _qms
                    result['client_consensus_dm_hex'] = b''.join(
                        _qms.pack('>dd', _client_consensus_dm_re[i], _client_consensus_dm_im[i])
                        for i in range(64)
                    ).hex()
        except Exception as _ce:
            logger.debug(f"[RPC-METHOD] client pool inject: {_ce}")

        logger.debug(f"[RPC-METHOD] qtcl_getQuantumMetrics success  block_height={_bh}")
        return _rpc_ok(result, rpc_id)
    except Exception as e:
        logger.exception(f"[RPC-METHOD] qtcl_getQuantumMetrics outer exception: {e}")
        return _rpc_error(-32603, f"Quantum metrics failed: {str(e)}", rpc_id, {"exception": str(e).__class__.__name__})


def _rpc_getPythPrice(params: Any, rpc_id: Any) -> dict:
    """
    qtcl_getPythPrice — atomic Pyth Network price snapshot.

    Params:
      list:   ["BTC", "ETH"]          — symbol list
      object: {"symbols": ["BTC"]}    — named
      null:   all feeds

    Returns:
      snapshot_id, fetch_time_ns, hermes_ok, hlwe_sig, feeds{price,conf,age_seconds,...}
    """
    try:
        logger.debug(f"[RPC-METHOD] qtcl_getPythPrice called with params={params}, id={rpc_id}")
        symbols: Optional[list] = None
        if isinstance(params, list):
            if params and isinstance(params[0], list):
                symbols = params[0]
            elif params and isinstance(params[0], str):
                symbols = params
        elif isinstance(params, dict):
            symbols = params.get("symbols")
        logger.debug(f"[RPC-METHOD] qtcl_getPythPrice: extracted symbols={symbols}")

        po = _get_pyth()
        if po is None:
            logger.warning(f"[RPC-METHOD] qtcl_getPythPrice: Pyth oracle not initialized")
            return _rpc_error(-32002, "Pyth oracle not initialized", rpc_id)

        try:
            logger.debug(f"[RPC-METHOD] qtcl_getPythPrice: fetching snapshot for symbols={symbols}")
            snap = po.get_snapshot(symbols)
            logger.debug(f"[RPC-METHOD] qtcl_getPythPrice: snapshot obtained successfully")
            return _rpc_ok(snap.to_dict(), rpc_id)
        except Exception as pe:
            logger.exception(f"[RPC-METHOD] qtcl_getPythPrice: Pyth fetch error: {pe}")
            return _rpc_error(-32002, f"Pyth fetch failed: {str(pe)}", rpc_id, {"exception": str(pe).__class__.__name__})
    except Exception as e:
        logger.exception(f"[RPC-METHOD] qtcl_getPythPrice outer exception: {e}")
        return _rpc_error(-32603, f"Internal error: {str(e)}", rpc_id, {"exception": str(e).__class__.__name__})


def _rpc_getMempoolStats(params: Any, rpc_id: Any) -> dict:
    """qtcl_getMempoolStats — mempool depth and fee percentiles."""
    try:
        logger.debug(f"[RPC-METHOD] qtcl_getMempoolStats called with params={params}, id={rpc_id}")
        # Walk resolution chain: module-level MEMPOOL → globals.get_mempool() → mempool module singleton
        mp = None
        _srv_globals = sys.modules[__name__].__dict__
        mp = _srv_globals.get("MEMPOOL") or _srv_globals.get("_MEMPOOL")
        if mp is None:
            try:
                import globals as _g
                _gf = getattr(_g, "get_mempool", None)
                if callable(_gf): mp = _gf()
            except Exception: pass
        if mp is None:
            try:
                import mempool as _mp_mod
                mp = getattr(_mp_mod, "MEMPOOL", None) or getattr(_mp_mod, "_MEMPOOL_INSTANCE", None)
            except Exception: pass
        if mp is None:
            logger.warning("[RPC-METHOD] qtcl_getMempoolStats: mempool not available")
            return _rpc_ok({"depth": 0, "pending": 0, "note": "mempool initializing"}, rpc_id)
        try:
            stats = mp.get_stats() if hasattr(mp, "get_stats") else {"depth": getattr(mp, "size", lambda: 0)()}
            logger.debug(f"[RPC-METHOD] qtcl_getMempoolStats success")
            return _rpc_ok(stats, rpc_id)
        except Exception as me:
            logger.exception(f"[RPC-METHOD] qtcl_getMempoolStats: get_stats error: {me}")
            return _rpc_error(-32603, f"Mempool stats failed: {str(me)}", rpc_id, {"exception": type(me).__name__})
    except Exception as e:
        logger.exception(f"[RPC-METHOD] qtcl_getMempoolStats outer exception: {e}")
        return _rpc_error(-32603, f"Internal error: {str(e)}", rpc_id, {"exception": type(e).__name__})


def _rpc_getPeers(params: Any, rpc_id: Any) -> dict:
    """qtcl_getPeers — active P2P peer list from peer_registry (5-min freshness window)."""
    try:
        limit = 50
        if isinstance(params, list) and params:
            try: limit = int(params[0])
            except (ValueError, TypeError): limit = 50
        elif isinstance(params, dict):
            try: limit = int(params.get("limit", 50))
            except (ValueError, TypeError): limit = 50
        limit = min(max(int(limit), 1), 200)
        peers: list = []
        # Primary: query peer_registry in Supabase — 5-min freshness window
        try:
            with get_db_cursor() as cur:
                cur.execute("""
                    SELECT node_id, external_addr, pubkey_hash, chain_height,
                           last_seen, capabilities, ban_score, mac_address, device_id
                    FROM   peer_registry
                    WHERE  last_seen > NOW() - INTERVAL '5 minutes'
                      AND  ban_score < 100
                    ORDER  BY chain_height DESC, last_seen DESC
                    LIMIT  %s
                """, (limit,))
                rows = cur.fetchall()
                if rows:
                    cols = [d[0] for d in cur.description]
                    for row in rows:
                        r = dict(zip(cols, row))
                        r["last_seen"] = r["last_seen"].timestamp() if hasattr(r.get("last_seen"), "timestamp") else r.get("last_seen")
                        peers.append(r)
        except Exception as _dbe:
            logger.debug(f"[RPC-METHOD] qtcl_getPeers DB query: {_dbe}")
        # Fallback: module-level LIVE_PEERS dict (in-process registrations since last restart)
        if not peers:
            peers = [v for v in list(_LIVE_PEERS_CACHE.values())[:limit]]
        logger.debug(f"[RPC-METHOD] qtcl_getPeers: returning {len(peers)} peers")
        return _rpc_ok({"peers": peers, "count": len(peers)}, rpc_id)
    except Exception as e:
        logger.exception(f"[RPC-METHOD] qtcl_getPeers outer exception: {e}")
        return _rpc_error(-32603, f"Internal error: {str(e)}", rpc_id, {"exception": type(e).__name__})


def _rpc_getPeersByNatGroup(params: Any, rpc_id: Any) -> dict:
    try:
        if isinstance(params, list):
            params = params[0] if params else {}
        if not isinstance(params, dict):
            return _rpc_error(-32602, "Invalid params: object expected", rpc_id)
        caller_ip = str(params.get("caller_ip") or "").strip()
        my_mac = str(params.get("mac_address") or "").strip().lower()
        if not caller_ip:
            return _rpc_error(-32602, "caller_ip required", rpc_id)
        peers = []
        try:
            with get_db_cursor() as cur:
                # Ensure table exists — first boot before any registerPeer call
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS peer_registry (
                        node_id       TEXT        PRIMARY KEY,
                        external_addr TEXT        NOT NULL,
                        pubkey_hash   TEXT        NOT NULL DEFAULT '',
                        chain_height  BIGINT      DEFAULT 0,
                        last_seen     TIMESTAMPTZ DEFAULT NOW(),
                        first_seen    TIMESTAMPTZ DEFAULT NOW(),
                        capabilities  JSONB       DEFAULT '[]',
                        ban_score     INTEGER     DEFAULT 0,
                        caller_ip     TEXT        DEFAULT '',
                        mac_address   TEXT        DEFAULT '',
                        device_id     TEXT        DEFAULT ''
                    )
                """)
                cur.execute("""
                    SELECT node_id, external_addr, pubkey_hash, chain_height,
                           last_seen, capabilities, ban_score, mac_address, device_id, caller_ip
                    FROM   peer_registry
                    WHERE  caller_ip = %s
                      AND  last_seen > NOW() - INTERVAL '5 minutes'
                      AND  ban_score < 100
                    ORDER  BY chain_height DESC, last_seen DESC
                    LIMIT  50
                """, (caller_ip,))
                rows = cur.fetchall()
                if rows:
                    cols = [d[0] for d in cur.description]
                    for row in rows:
                        r = dict(zip(cols, row))
                        _ls = r.get("last_seen")
                        r["last_seen"] = _ls.timestamp() if hasattr(_ls, "timestamp") else (float(_ls) if _ls else 0.0)
                        if r.get("mac_address", "").lower() != my_mac:
                            peers.append(r)
        except Exception as _dbe:
            logger.debug(f"[RPC-METHOD] qtcl_getPeersByNatGroup DB query: {_dbe}")
        if not peers:
            with _LIVE_PEERS_LOCK:
                for nid, p in _LIVE_PEERS_CACHE.items():
                    if p.get("caller_ip") == caller_ip and p.get("mac_address", "").lower() != my_mac:
                        _pc = dict(p)
                        # Normalise last_seen to float timestamp for consistent client parsing
                        _ls = _pc.get("last_seen", 0)
                        if hasattr(_ls, "timestamp"):
                            _pc["last_seen"] = _ls.timestamp()
                        elif not isinstance(_ls, (int, float)):
                            _pc["last_seen"] = 0.0
                        peers.append(_pc)
        return _rpc_ok({"peers": peers, "count": len(peers), "nat_group": caller_ip}, rpc_id)
    except Exception as e:
        logger.exception(f"[RPC-METHOD] qtcl_getPeersByNatGroup exception: {e}")
        return _rpc_error(-32603, f"Internal error: {str(e)}", rpc_id, {"exception": type(e).__name__})


# In-process peer cache (survives between requests, cleared on restart — DB is authoritative)
_LIVE_PEERS_CACHE: Dict[str, dict] = {}
_LIVE_PEERS_LOCK  = threading.Lock()


def _rpc_registerPeer(params: Any, rpc_id: Any) -> dict:
    """qtcl_registerPeer — miner announces itself to Koyeb bootstrap registry.

    Params (dict):
        external_addr  str   "ip:port" of miner's P2P listener (required)
        node_id        str   64-hex SHA-256(hlwe_pubkey) (required)
        pubkey         str   base64 HLWE public key
        chain_height   int   miner's current chain height
    """
    try:
        if isinstance(params, list):
            params = params[0] if params else {}
        if not isinstance(params, dict):
            return _rpc_error(-32602, "Invalid params: object expected", rpc_id)

        external_addr = str(params.get("external_addr") or "").strip()
        node_id       = str(params.get("node_id") or "").strip().lower()
        pubkey_b64    = str(params.get("pubkey") or "").strip()
        chain_height  = int(params.get("chain_height") or 0)
        mac_address   = str(params.get("mac_address") or "").strip().lower()
        device_id     = str(params.get("device_id") or "").strip().lower()

        if not external_addr:
            return _rpc_error(-32602, "external_addr required", rpc_id)
        if not node_id or len(node_id) != 64 or not all(c in "0123456789abcdef" for c in node_id):
            return _rpc_error(-32602, "node_id must be 64 lowercase hex chars (SHA-256 of pubkey)", rpc_id)

        # Derive caller IP from Flask request context (STUN: what address does Koyeb see?)
        try:
            caller_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "").split(",")[0].strip()
        except Exception:
            caller_ip = ""

        pubkey_hash = hashlib.sha256(pubkey_b64.encode()).hexdigest()[:32] if pubkey_b64 else node_id[:32]

        # Upsert into peer_registry — CREATE TABLE IF NOT EXISTS guard in case migration pending
        try:
            with get_db_cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS peer_registry (
                        node_id       TEXT        PRIMARY KEY,
                        external_addr TEXT        NOT NULL,
                        pubkey_hash   TEXT        NOT NULL DEFAULT '',
                        chain_height  BIGINT      DEFAULT 0,
                        last_seen     TIMESTAMPTZ DEFAULT NOW(),
                        first_seen    TIMESTAMPTZ DEFAULT NOW(),
                        capabilities  JSONB       DEFAULT '[]',
                        ban_score     INTEGER     DEFAULT 0,
                        caller_ip     TEXT        DEFAULT '',
                        mac_address   TEXT        DEFAULT '',
                        device_id     TEXT        DEFAULT ''
                    )
                """)
                try:
                    cur.execute("ALTER TABLE peer_registry ADD COLUMN IF NOT EXISTS mac_address TEXT DEFAULT ''")
                    cur.execute("ALTER TABLE peer_registry ADD COLUMN IF NOT EXISTS device_id TEXT DEFAULT ''")
                except Exception:
                    pass
                cur.execute("""
                    INSERT INTO peer_registry
                        (node_id, external_addr, pubkey_hash, chain_height, last_seen, caller_ip, mac_address, device_id)
                    VALUES (%s, %s, %s, %s, NOW(), %s, %s, %s)
                    ON CONFLICT (node_id) DO UPDATE SET
                        external_addr = EXCLUDED.external_addr,
                        pubkey_hash   = EXCLUDED.pubkey_hash,
                        chain_height  = EXCLUDED.chain_height,
                        last_seen     = NOW(),
                        caller_ip     = EXCLUDED.caller_ip,
                        mac_address   = EXCLUDED.mac_address,
                        device_id     = EXCLUDED.device_id
                """, (node_id, external_addr, pubkey_hash, chain_height, caller_ip, mac_address, device_id))
        except Exception as _dbe:
            # Non-fatal: fall through to in-process cache so peer can still be served
            logger.warning(f"[RPC-METHOD] qtcl_registerPeer DB upsert failed: {_dbe}")

        # Always update in-process cache for immediate availability
        with _LIVE_PEERS_LOCK:
            _LIVE_PEERS_CACHE[node_id] = {
                "node_id":       node_id,
                "external_addr": external_addr,
                "pubkey_hash":   pubkey_hash,
                "chain_height":  chain_height,
                "last_seen":     time.time(),
                "caller_ip":     caller_ip,
                "mac_address":   mac_address,
                "device_id":     device_id,
                "ban_score":     0,
            }
        logger.info(f"[P2P] ✅ Peer registered: node={node_id[:16]}… addr={external_addr} h={chain_height}")
        return _rpc_ok({"registered": True, "node_id": node_id, "external_addr": external_addr, "caller_ip": caller_ip}, rpc_id)
    except Exception as e:
        logger.exception(f"[RPC-METHOD] qtcl_registerPeer exception: {e}")
        return _rpc_error(-32603, f"Internal error: {str(e)}", rpc_id, {"exception": type(e).__name__})


def _rpc_getMyAddr(params: Any, rpc_id: Any) -> dict:
    """qtcl_getMyAddr — STUN: return the caller's observed source IP so miners can discover their external addr.

    Returns:
        external_addr  str   "observed_ip:suggested_port"
        ip             str   raw observed source IP
        port           int   suggested P2P port (from P2P_PORT env or 9091)
    """
    try:
        try:
            forwarded = request.headers.get("X-Forwarded-For", "")
            observed_ip = forwarded.split(",")[0].strip() if forwarded else request.remote_addr or "unknown"
        except Exception:
            observed_ip = "unknown"
        p2p_port = int(os.environ.get("P2P_PORT", "9091"))
        return _rpc_ok({
            "ip":            observed_ip,
            "port":          p2p_port,
            "external_addr": f"{observed_ip}:{p2p_port}",
        }, rpc_id)
    except Exception as e:
        logger.exception(f"[RPC-METHOD] qtcl_getMyAddr exception: {e}")
        return _rpc_error(-32603, f"Internal error: {str(e)}", rpc_id, {"exception": type(e).__name__})


def _rpc_getHealth(params: Any, rpc_id: Any) -> dict:
    """qtcl_getHealth — full system health vector."""
    try:
        logger.debug(f"[RPC-METHOD] qtcl_getHealth called with params={params}, id={rpc_id}")
        po = _get_pyth()
        logger.debug(f"[RPC-METHOD] qtcl_getHealth: oracle_ready={ORACLE_AVAILABLE}, lattice_ready={LATTICE is not None}, pyth_ready={po is not None}")
        result = {
            "status":           "ok",
            "ts":               time.time(),
            "uptime_s":         round(time.time() - _SERVER_START_TIME, 1),
            "oracle_ready":     ORACLE_AVAILABLE,
            "lattice_ready":    LATTICE is not None,
            "pyth_ready":       po is not None,
            "pyth_stats":       po.stats() if po else {},
            "jsonrpc_version":  _JSONRPC_VERSION,
            "qtcl_server":      "v6",
        }
        logger.debug(f"[RPC-METHOD] qtcl_getHealth success")
        return _rpc_ok(result, rpc_id)
    except Exception as e:
        logger.exception(f"[RPC-METHOD] qtcl_getHealth exception: {e}")
        return _rpc_error(-32603, f"Health check failed: {str(e)}", rpc_id, {"exception": str(e).__class__.__name__})


def _rpc_getOracleRegistry(params: Any, rpc_id: Any) -> dict:
    """qtcl_getOracleRegistry — paginated on-chain oracle registry.
    Params (object or positional list):
      mode           string   filter by mode: full|light|archive|deregistered (default: all)
      confirmed_only bool     only oracles with on-chain reg_tx_hash (default: false)
      limit          int      max records (default 100, max 500)
      offset         int      pagination offset (default 0)
    Returns: {oracles[], total, confirmed_count, limit, offset}
    """
    try:
        logger.debug(f"[RPC-METHOD] qtcl_getOracleRegistry called with params={params}, id={rpc_id}")
        p = params if isinstance(params, dict) else (params[0] if isinstance(params, list) and params and isinstance(params[0], dict) else {})
        mode_filter    = str(p.get('mode', ''))
        confirmed_only = bool(p.get('confirmed_only', False))
        limit          = min(int(p.get('limit',  100)), 500)
        offset         = int(p.get('offset', 0))
        logger.debug(f"[RPC-METHOD] qtcl_getOracleRegistry: mode={mode_filter}, confirmed_only={confirmed_only}, limit={limit}, offset={offset}")
        try:
            _lazy_ensure_oracle_registry()
            where_clauses: list = []
            qparams:       list = []
            if mode_filter:
                where_clauses.append("mode = %s"); qparams.append(mode_filter)
            if confirmed_only:
                where_clauses.append("reg_tx_hash != '' AND reg_tx_hash != 'gossip_pending'")
            where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
            logger.debug(f"[RPC-METHOD] qtcl_getOracleRegistry: executing query with where_sql={where_sql}")
            with get_db_cursor() as cur:
                cur.execute(f"""
                    SELECT oracle_id, oracle_url, oracle_address, is_primary,
                           last_seen, block_height, peer_count,
                           wallet_address, oracle_pub_key, cert_sig,
                           mode, ip_hint, reg_tx_hash, registered_at, created_at
                    FROM   oracle_registry {where_sql}
                    ORDER  BY registered_at DESC, last_seen DESC
                    LIMIT  %s OFFSET %s
                """, qparams + [limit, offset])
                rows = cur.fetchall()
                cur.execute(f"SELECT COUNT(*) FROM oracle_registry {where_sql}", qparams)
                total = cur.fetchone()[0]
                logger.debug(f"[RPC-METHOD] qtcl_getOracleRegistry: fetched {len(rows)} rows, total={total}")
            oracles = [{
                'oracle_id'     : r[0],  'oracle_url'    : r[1],
                'oracle_address': r[2],  'is_primary'    : r[3],
                'last_seen'     : _iso(r[4]),  'block_height'  : r[5],
                'peer_count'    : r[6],  'wallet_address': r[7],
                'oracle_pub_key': r[8],  'cert_sig'      : r[9],
                'mode'          : r[10], 'ip_hint'       : r[11],
                'reg_tx_hash'   : r[12], 'registered_at' : _iso(r[13]),
                'created_at'    : _iso(r[14]),
                'on_chain'      : bool(r[12] and r[12] not in ('', 'gossip_pending')),
            } for r in rows]
            result = {
                'oracles'        : oracles,
                'total'          : total,
                'confirmed_count': sum(1 for o in oracles if o['on_chain']),
                'limit'          : limit,
                'offset'         : offset,
            }
            logger.debug(f"[RPC-METHOD] qtcl_getOracleRegistry success: {len(oracles)} oracles returned")
            return _rpc_ok(result, rpc_id)
        except Exception as re:
            logger.exception(f"[RPC-METHOD] qtcl_getOracleRegistry: registry error: {re}")
            return _rpc_error(-32603, f"Oracle registry query failed: {str(re)}", rpc_id, {"exception": str(re).__class__.__name__})
    except Exception as e:
        logger.exception(f"[RPC-METHOD] qtcl_getOracleRegistry outer exception: {e}")
        return _rpc_error(-32603, f"Internal error: {str(e)}", rpc_id, {"exception": str(e).__class__.__name__})


def _rpc_getOracleRecord(params: Any, rpc_id: Any) -> dict:
    """qtcl_getOracleRecord — single oracle record by oracle_addr or oracle_id.
    Params: [oracle_addr] or {oracle_addr: string}
    Returns: full oracle_registry row or {registered: false} if unknown.
    """
    oracle_addr = ''
    if isinstance(params, list) and params:
        oracle_addr = str(params[0])
    elif isinstance(params, dict):
        oracle_addr = str(params.get('oracle_addr', params.get('address', '')))
    if not oracle_addr:
        return _rpc_error(-32602, "oracle_addr required", rpc_id)
    try:
        _lazy_ensure_oracle_registry()
        with get_db_cursor() as cur:
            cur.execute("""
                SELECT oracle_id, oracle_url, oracle_address, is_primary,
                       last_seen, block_height, peer_count,
                       wallet_address, oracle_pub_key, cert_sig, cert_auth_tag,
                       mode, ip_hint, reg_tx_hash, registered_at, created_at
                FROM   oracle_registry
                WHERE  oracle_id = %s OR oracle_address = %s
                LIMIT  1
            """, (oracle_addr, oracle_addr))
            r = cur.fetchone()
        if not r:
            return _rpc_ok({'registered': False, 'oracle_addr': oracle_addr}, rpc_id)
        on_chain = bool(r[13] and r[13] not in ('', 'gossip_pending'))
        return _rpc_ok({
            'registered'    : True,
            'on_chain'      : on_chain,
            'oracle_id'     : r[0],  'oracle_url'    : r[1],
            'oracle_address': r[2],  'is_primary'    : r[3],
            'last_seen'     : _iso(r[4]),  'block_height'  : r[5],
            'peer_count'    : r[6],  'wallet_address': r[7],
            'oracle_pub_key': r[8],  'cert_sig'      : r[9],
            'cert_auth_tag' : r[10], 'mode'          : r[11],
            'ip_hint'       : r[12], 'reg_tx_hash'   : r[13],
            'registered_at' : _iso(r[14]), 'created_at': _iso(r[15]),
        }, rpc_id)
    except Exception as e:
        return _rpc_error(-32603, f"Oracle record lookup failed: {e}", rpc_id)


def _rpc_submitOracleReg(params: Any, rpc_id: Any) -> dict:
    """qtcl_submitOracleReg — build and submit an oracle_reg TX through the mempool.
    Params (object):
      wallet_address  string  required — HLWE wallet signing the TX
      oracle_addr     string  required — oracle identity address
      oracle_pub      string  recommended — oracle HLWE public key hex
      cert_sig        string  optional — pre-computed cert sig (server computes if omitted)
      cert_auth_tag   string  optional
      mode            string  optional — full|light|archive (default: full)
      ip_hint         string  optional — advertised host:port
      action          string  optional — register|deregister (default: register)
      nonce           int     optional
      timestamp_ns    int     optional
      signature       object  required for mempool — HLWE sig over tx_hash
    Returns: {status, tx_hash, oracle_addr, check_url} or {status: tx_template_issued, tx_template}
    """
    p = params if isinstance(params, dict) else (params[0] if isinstance(params, list) and params and isinstance(params[0], dict) else {})
    wallet_addr = str(p.get('wallet_address', p.get('from_address', '')))
    oracle_addr = str(p.get('oracle_addr', wallet_addr))
    oracle_pub  = str(p.get('oracle_pub',  p.get('public_key', '')))
    mode        = str(p.get('mode',        'full'))
    ip_hint     = str(p.get('ip_hint',     ''))
    action      = str(p.get('action',      'register'))
    signature   = p.get('signature', {})
    nonce_val   = int(p.get('nonce', int(time.time_ns() // 1_000_000) % 2**31))
    ts_ns       = int(p.get('timestamp_ns', time.time_ns()))

    if not wallet_addr or not oracle_addr:
        return _rpc_error(-32602, "wallet_address and oracle_addr required", rpc_id)

    import hashlib as _hh
    cert_preimage = f"{oracle_addr}|{wallet_addr}|{oracle_pub}"
    cert_sig_hex  = str(p.get('cert_sig',      _hh.sha256(cert_preimage.encode()).hexdigest()))
    cert_auth_tag = str(p.get('cert_auth_tag', _hh.sha3_256(cert_preimage.encode()).hexdigest()[:32]))

    _ora_registry_addr = "qtcl1oracle_registry_000000000000000000000000"
    tx_payload = {
        'tx_type'     : 'oracle_reg',
        'from_address': wallet_addr,
        'to_address'  : _ora_registry_addr,
        'amount'      : 1,
        'fee'         : 0.01,
        'nonce'       : nonce_val,
        'timestamp_ns': ts_ns,
        'signature'   : signature,
        'input_data'  : {
            'oracle_addr'   : oracle_addr,
            'oracle_pub'    : oracle_pub,
            'cert_sig'      : cert_sig_hex,
            'cert_auth_tag' : cert_auth_tag,
            'mode'          : mode,
            'ip_hint'       : ip_hint,
            'action'        : action,
        },
        'metadata': {
            'oracle_addr': oracle_addr,
            'wallet_addr': wallet_addr,
            'cert_valid' : True,
            'action'     : action,
        },
    }

    # If no signature provided — return template for client to sign
    if not signature:
        return _rpc_ok({
            'status'     : 'tx_template_issued',
            'tx_template': tx_payload,
            'submit_to'  : 'qtcl_submitOracleReg (with signature) or POST /api/oracle/registry/submit',
            'note'       : 'Sign tx_template with your HLWE wallet, then resubmit with signature field.',
        }, rpc_id)

    try:
        if MEMPOOL:
            result, reason, accepted_tx = MEMPOOL.accept(tx_payload)
            if result.value not in ('accepted', 'duplicate'):
                return _rpc_error(-32001, f"Mempool rejected: {reason} [{result.value}]", rpc_id,
                                  {"result_code": result.value, "tx_template": tx_payload})
            tx_hash = accepted_tx.tx_hash if accepted_tx else ''
        else:
            tx_hash = _hh.sha3_256(
                f"oracle_reg:{wallet_addr}:{oracle_addr}:{ts_ns}".encode()
            ).hexdigest()

        return _rpc_ok({
            'status'    : 'submitted',
            'tx_hash'   : tx_hash,
            'oracle_addr': oracle_addr,
            'wallet_addr': wallet_addr,
            'action'    : action,
            'check_url' : f'/api/oracle/registry/{oracle_addr}',
            'note'      : 'TX in mempool — confirmed on next block seal.',
        }, rpc_id)
    except Exception as e:
        return _rpc_error(-32603, f"Oracle reg submission failed: {e}", rpc_id)


def _rpc_getEvents(params: Any, rpc_id: Any) -> dict:
    """qtcl_getEvents — poll recent RPC events (tx, block, oracle_snapshot, oracle_dm, oracle_measurements)."""
    try:
        # Normalise params → always a dict regardless of what the client sent
        if isinstance(params, dict):
            p = params
        elif isinstance(params, list) and params and isinstance(params[0], dict):
            p = params[0]
        else:
            p = {}
        since       = float(p.get('since', time.time() - 3600))
        event_types = str(p.get('types', 'all'))
        limit       = int(p.get('limit', 100))
        want_types  = set(event_types.split(',')) if event_types != 'all' else {'all'}
        events = []
        with _rpc_event_lock:
            for e in list(_rpc_event_log):
                if e['ts'] >= since and ('all' in want_types or e['type'] in want_types):
                    events.append(e)
                    if len(events) >= limit:
                        break
        return _rpc_ok({'events': events, 'count': len(events)}, rpc_id)
    except Exception as e:
        logger.exception(f"[RPC-METHOD] qtcl_getEvents exception: {e}")
        return _rpc_error(-32603, f"Events fetch failed: {str(e)}", rpc_id)


# ─── Method registry (O(1) dispatch) ─────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# RPC Methods: Oracle Measurement Broadcast (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def _rpc_registerMeasurementSubscriber(params: Any, rpc_id: Any) -> dict:
    """
    Subscribe to oracle measurement broadcasts via RPC push (WebSocket-ready).
    
    Request:
        {
            "jsonrpc": "2.0",
            "method": "qtcl_registerMeasurementSubscriber",
            "params": {
                "client_id": "miner_abc123",
                "callback_url": "http://localhost:9999/quantum/measurement",
                "burst_mode": true
            },
            "id": 1
        }
    
    Response (success):
        {
            "jsonrpc": "2.0",
            "result": {
                "registered": true,
                "subscriber_id": "miner_abc123",
                "measurement_frequency": "burst" | "throttled",
                "broadcast_url": "https://qtcl-blockchain.koyeb.app/rpc/_internal/measurement"
            },
            "id": 1
        }
    """
    try:
        if not isinstance(params, dict):
            return _rpc_error(-32602, "params must be object", rpc_id)
        
        client_id = params.get('client_id')
        callback_url = params.get('callback_url')
        burst_mode = params.get('burst_mode', False)
        
        if not client_id or not callback_url:
            return _rpc_error(-32602, "client_id and callback_url required", rpc_id)
        
        try:
            from oracle import get_oracle_measurement_broadcaster
            broadcaster = get_oracle_measurement_broadcaster()
            success = broadcaster.register_subscriber(client_id, callback_url, burst_mode)
            
            if success:
                return _rpc_ok({
                    'registered': True,
                    'subscriber_id': client_id,
                    'measurement_frequency': 'burst' if burst_mode else 'throttled',
                    'broadcast_url': "https://qtcl-blockchain.koyeb.app/rpc/_internal/measurement",
                }, rpc_id)
            else:
                return _rpc_error(-32000, "client already subscribed", rpc_id)
        except ImportError:
            return _rpc_error(-32603, "broadcast system not initialized", rpc_id)
    
    except Exception as e:
        return _rpc_error(-32603, f"Subscription failed: {str(e)}", rpc_id)


def _rpc_unregisterMeasurementSubscriber(params: Any, rpc_id: Any) -> dict:
    """Unsubscribe from oracle measurement broadcasts."""
    try:
        if not isinstance(params, dict):
            return _rpc_error(-32602, "params must be object", rpc_id)
        
        client_id = params.get('client_id')
        if not client_id:
            return _rpc_error(-32602, "client_id required", rpc_id)
        
        try:
            from oracle import get_oracle_measurement_broadcaster
            broadcaster = get_oracle_measurement_broadcaster()
            success = broadcaster.unregister_subscriber(client_id)
            
            return _rpc_ok({'unregistered': success}, rpc_id)
        except ImportError:
            return _rpc_error(-32603, "broadcast system not initialized", rpc_id)
    
    except Exception as e:
        return _rpc_error(-32603, f"Unsubscribe failed: {str(e)}", rpc_id)


def _rpc_listMeasurementSubscribers(params: Any, rpc_id: Any) -> dict:
    """
    List all active measurement subscribers (operator introspection).
    Returns active subscriber count, per-subscriber metrics, and broadcast controller status.
    """
    try:
        from oracle import get_oracle_measurement_broadcaster
        broadcaster = get_oracle_measurement_broadcaster()
        status = broadcaster.get_status()
        
        return _rpc_ok({
            'active_count': status.get('active_subscribers', 0),
            'is_running': status.get('is_running', False),
            'metrics': status.get('metrics', {}),
            'subscribers': status.get('subscribers', []),
        }, rpc_id)
    
    except ImportError:
        return _rpc_error(-32603, "broadcast system not initialized", rpc_id)
    except Exception as e:
        return _rpc_error(-32603, f"List failed: {str(e)}", rpc_id)


# ═══════════════════════════════════════════════════════════════════════════════════════
# QTCL-PoW VERIFIER  — canonical SHAKE-256 scratchpad + SHA3-256 64-round chain
#
# Must stay byte-for-byte identical to the client's _pow_worker inner loop in
# qtcl_client.py (_mine_inline, STAGE 4).  Any divergence here = invalid rejects.
#
# Algorithm:
#   scratchpad  = SHAKE-256("QTCL_SCRATCHPAD_v1:" + w_entropy_seed)[0:512 KiB]
#   header      = struct.pack('>Q I 32s 32s I I 40s 32s',
#                             height, timestamp_s,
#                             parent_hash_bytes[:32], merkle_root_bytes[:32],
#                             difficulty_bits, nonce,
#                             miner_address_bytes[:40], w_entropy_seed[:32])
#   state       = SHA3-256("QTCL_POW_v1:" + header)
#   for rnd in range(64):
#       wi      = uint32_be(state[0:4]) % N_WINDOWS      # N_WINDOWS = 8192
#       window  = scratchpad[wi*64 : wi*64+64]
#       state   = SHA3-256(state + window + struct.pack('>I', rnd))
#   return state.hex()
# ═══════════════════════════════════════════════════════════════════════════════════════

_POW_SCRATCHPAD_BYTES = 512 * 1024
_POW_WINDOW_BYTES     = 64
_POW_MIX_ROUNDS       = 64
_POW_N_WINDOWS        = _POW_SCRATCHPAD_BYTES // _POW_WINDOW_BYTES   # 8192
_POW_HDR_FMT          = '>Q I 32s 32s I I 40s 32s'
_POW_PREFIX           = b"QTCL_POW_v1:"
_POW_SCRATCHPAD_PFX   = b"QTCL_SCRATCHPAD_v1:"
_POW_RND_PACKED       = [struct.pack('>I', r) for r in range(_POW_MIX_ROUNDS)]


def qtcl_pow_hash(
    height: int,
    timestamp_s: int,
    parent_hash: str,
    merkle_root: str,
    difficulty_bits: int,
    nonce: int,
    miner_address: str,
    w_entropy_seed: bytes,
) -> str:
    """
    Compute the QTCL-PoW hash for a single nonce.  Pure Python mirror of the
    client's hot-path inner loop.  Returns the final state as a 64-char hex string.
    """
    import struct as _st
    _ph_parent = bytes.fromhex(parent_hash.zfill(64))[:32]
    _ph_merkle = bytes.fromhex(merkle_root.zfill(64))[:32]
    _ph_miner  = miner_address.encode()[:40].ljust(40, b'\x00')
    _ph_seed   = w_entropy_seed[:32]

    scratchpad = hashlib.shake_256(
        _POW_SCRATCHPAD_PFX + w_entropy_seed
    ).digest(_POW_SCRATCHPAD_BYTES)
    sp_mv = memoryview(scratchpad)

    WIN_OFFSETS = [i * _POW_WINDOW_BYTES for i in range(_POW_N_WINDOWS)]

    hdr = _st.pack(_POW_HDR_FMT,
                   height, timestamp_s,
                   _ph_parent, _ph_merkle,
                   difficulty_bits, nonce,
                   _ph_miner, _ph_seed)
    h0 = hashlib.sha3_256()
    h0.update(_POW_PREFIX)
    h0.update(hdr)
    state = h0.digest()

    for rnd in range(_POW_MIX_ROUNDS):
        wi = struct.unpack_from('>I', state, 0)[0] % _POW_N_WINDOWS
        o  = WIN_OFFSETS[wi]
        h  = hashlib.sha3_256()
        h.update(state)
        h.update(sp_mv[o : o + _POW_WINDOW_BYTES])
        h.update(_POW_RND_PACKED[rnd])
        state = h.digest()

    return state.hex()


def qtcl_pow_verify(
    height: int,
    parent_hash: str,
    merkle_root: str,
    timestamp_s: int,
    difficulty_bits: int,
    nonce: int,
    miner_address: str,
    w_entropy_seed: bytes,
    claimed_hash: str,
    block_timestamp_s: int = 0,   # alias accepted for compatibility
) -> tuple:
    """
    Verify a submitted block's PoW.

    Returns (True, "") on success or (False, reason_string) on failure.
    Raises nothing — all exceptions are caught and returned as failures.
    """
    try:
        if not claimed_hash or len(claimed_hash) != 64:
            return False, f"claimed_hash malformed (len={len(claimed_hash) if claimed_hash else 0})"

        _ts = timestamp_s or block_timestamp_s
        computed = qtcl_pow_hash(
            height=height,
            timestamp_s=_ts,
            parent_hash=parent_hash,
            merkle_root=merkle_root,
            difficulty_bits=difficulty_bits,
            nonce=nonce,
            miner_address=miner_address,
            w_entropy_seed=w_entropy_seed,
        )

        if computed != claimed_hash.lower():
            return False, (
                f"hash mismatch: computed={computed[:16]}… "
                f"claimed={claimed_hash[:16]}…"
            )

        prefix = '0' * difficulty_bits
        if not computed.startswith(prefix):
            return False, (
                f"difficulty not met: need {difficulty_bits} leading zeros, "
                f"got hash={computed[:difficulty_bits+4]}…"
            )

        return True, ""

    except Exception as e:
        return False, f"verifier exception: {type(e).__name__}: {e}"


def _rpc_submitBlock(params: Any, rpc_id: Any) -> dict:
    """qtcl_submitBlock — validate and persist a mined block directly (no Flask route needed)."""
    try:
        if not params or not isinstance(params, (list, tuple)) or len(params) < 1:
            return _rpc_error(-32602, "params[0] must be {header, transactions}", rpc_id)
        data = params[0]
        if not isinstance(data, dict):
            return _rpc_error(-32602, "params[0] must be a JSON object", rpc_id)

        hdr  = data.get("header", data)   # support flat or {header, transactions}
        txs  = data.get("transactions", [])

        height          = int(hdr.get("height", 0))
        block_hash      = str(hdr.get("block_hash", ""))
        parent_hash     = str(hdr.get("parent_hash", "0" * 64))
        merkle_root     = str(hdr.get("merkle_root", "0" * 64))
        timestamp_s     = int(hdr.get("timestamp_s", hdr.get("timestamp", 0)))
        nonce           = int(hdr.get("nonce", 0))
        miner_address   = str(hdr.get("miner_address", ""))
        difficulty_bits = int(hdr.get("difficulty_bits", hdr.get("difficulty", 4)))
        w_entropy_hex   = str(hdr.get("w_entropy_hash", hdr.get("w_entropy_seed", "")))
        w_state_fidelity = float(hdr.get("w_state_fidelity", 0.0) or 0.0)

        # ── Duplicate check ──────────────────────────────────────────────────
        existing = query_block_by_hash(block_hash)
        if existing:
            return _rpc_ok({"status": "duplicate", "height": height, "block_hash": block_hash}, rpc_id)

        # ── Height check ─────────────────────────────────────────────────────
        latest = query_latest_block()
        expected_height = (int(latest["height"]) + 1) if latest else 1
        if height != expected_height:
            tip = int(latest["height"]) if latest else 0
            return _rpc_error(-32001,
                f"Invalid height: expected {expected_height}, got {height}",
                rpc_id, {"tip": tip})

        # ── Parent hash check ────────────────────────────────────────────────
        if latest:
            expected_parent = latest.get("block_hash") or latest.get("hash", "")
            logger.critical(f"[RPC-submitBlock] 🔍 PARENT HASH CHECK h={height}: expected={expected_parent[:32] if expected_parent else 'NONE'}… got={parent_hash[:32] if parent_hash else 'NONE'}…")
            if parent_hash.lower() != expected_parent.lower():
                logger.error(f"[RPC-submitBlock] ❌ PARENT MISMATCH h={height} (expected {expected_parent[:16]}… got {parent_hash[:16]}…)")
                return _rpc_error(-32001,
                    f"Invalid parent_hash: expected {expected_parent[:16]}… got {parent_hash[:16]}…",
                    rpc_id)

        # ── PoW verification ─────────────────────────────────────────────────
        try:
            w_seed = bytes.fromhex(w_entropy_hex) if w_entropy_hex else b'\x00' * 32
            valid, reason = qtcl_pow_verify(
                height=height,
                parent_hash=parent_hash,
                merkle_root=merkle_root,
                timestamp_s=timestamp_s,
                difficulty_bits=difficulty_bits,
                nonce=nonce,
                miner_address=miner_address,
                w_entropy_seed=w_seed,
                claimed_hash=block_hash,
                block_timestamp_s=timestamp_s,
            )
            if not valid:
                return _rpc_error(-32003, f"PoW invalid: {reason}", rpc_id)
        except Exception as pe:
            logger.warning(f"[RPC-submitBlock] PoW verify error (non-fatal): {pe}")
            # Fall through — verify hash prefix at minimum
            if not block_hash.startswith('0' * difficulty_bits):
                return _rpc_error(-32003, f"Difficulty not met: {block_hash[:8]}", rpc_id)

        # ── Persist block + transactions (transaction 1 — no wallet writes) ──
        _block_rowcount = 0
        try:
            with get_db_cursor() as cur:
                cur.execute("""
                    INSERT INTO blocks
                    (height, block_number, block_hash, previous_hash, timestamp,
                     oracle_w_state_hash, validator_public_key, nonce,
                     difficulty, entropy_score, transactions_root)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (height) DO NOTHING
                """, (
                    height, height, block_hash, parent_hash, timestamp_s,
                    w_entropy_hex[:64] if w_entropy_hex else "0" * 64,
                    miner_address, nonce,
                    difficulty_bits, w_state_fidelity, merkle_root,
                ))
                # Capture rowcount IMMEDIATELY after block INSERT
                _block_rowcount = cur.rowcount

                # Persist user-supplied transactions
                for tx in (txs or []):
                    tx_id = tx.get("tx_id") or tx.get("tx_hash", "")
                    if not tx_id:
                        continue
                    cur.execute("""
                        INSERT INTO transactions
                        (tx_hash, from_address, to_address, amount,
                         tx_type, status, height, updated_at)
                        VALUES (%s, %s, %s, %s, %s, 'confirmed', %s, NOW())
                        ON CONFLICT (tx_hash) DO UPDATE
                          SET height     = EXCLUDED.height,
                              status     = 'confirmed',
                              updated_at = NOW()
                    """, (
                        tx_id,
                        tx.get("from_addr", "0" * 64),
                        tx.get("to_addr", ""),
                        float(tx.get("amount", 0)),
                        tx.get("tx_type", "transfer"),
                        height,
                    ))
        except Exception as dbe:
            logger.exception(f"[RPC-submitBlock] DB error: {dbe}")
            # Even if PG fails, we attempt to continue with SQLite if we can.
            # But normally we want the authoritative DB to succeed.
            # return _rpc_error(-32603, f"DB persist failed: {str(dbe)}", rpc_id)
        
        # ── Credit coinbase rewards (transaction 2 — isolated from block persist) ──
        # Separate get_db_cursor so a wallet schema error cannot roll back the block.
        # Deterministic from header: never scan txs (breaks when miner == treasury).
        if _block_rowcount > 0 and TessellationRewardSchedule:
            try:
                rewards          = TessellationRewardSchedule.get_rewards_for_height(height)
                miner_reward     = rewards.get('miner', 720)
                treasury_reward  = rewards.get('treasury', 80)
                treasury_address = TessellationRewardSchedule.TREASURY_ADDRESS

                # Canonical deterministic coinbase tx hashes for ledger
                _cb_miner_hash = hashlib.sha3_256(
                    json.dumps({"height": height, "to": miner_address,
                                "amount": miner_reward, "block_hash": block_hash,
                                "role": "miner"}, sort_keys=True, separators=(',', ':')).encode()
                ).hexdigest()
                _cb_treasury_hash = hashlib.sha3_256(
                    json.dumps({"height": height, "to": treasury_address,
                                "amount": treasury_reward, "block_hash": block_hash,
                                "role": "treasury"}, sort_keys=True, separators=(',', ':')).encode()
                ).hexdigest()

                with get_db_cursor() as cur:
                    # ── Miner wallet balance ─────────────────────────────────────
                    _miner_fp = hashlib.sha256(miner_address.encode()).hexdigest()[:64]
                    cur.execute("""
                        INSERT INTO wallet_addresses
                            (address, wallet_fingerprint, public_key,
                             balance, transaction_count, address_type)
                        VALUES (%s, %s, %s, %s, 1, 'miner')
                        ON CONFLICT (address) DO UPDATE SET
                            balance           = wallet_addresses.balance + EXCLUDED.balance,
                            transaction_count = wallet_addresses.transaction_count + 1
                    """, (miner_address, _miner_fp, _miner_fp, miner_reward))
                    logger.info(
                        f"[RPC-submitBlock] ⛏  Miner credited: {miner_reward} base units "
                        f"({miner_reward/100:.2f} QTCL) → {miner_address[:22]}…"
                    )

                    # ── Treasury wallet balance ──────────────────────────────────
                    if treasury_address and treasury_reward > 0:
                        _treas_fp = hashlib.sha256(treasury_address.encode()).hexdigest()[:64]
                        cur.execute("""
                            INSERT INTO wallet_addresses
                                (address, wallet_fingerprint, public_key,
                                 balance, transaction_count, address_type)
                            VALUES (%s, %s, %s, %s, 1, 'treasury')
                            ON CONFLICT (address) DO UPDATE SET
                                balance           = wallet_addresses.balance + EXCLUDED.balance,
                                transaction_count = wallet_addresses.transaction_count + 1
                        """, (treasury_address, _treas_fp, _treas_fp, treasury_reward))
                        logger.info(
                            f"[RPC-submitBlock] 💰 Treasury credited: {treasury_reward} base units "
                            f"({treasury_reward/100:.2f} QTCL) → {treasury_address[:22]}…"
                        )

                    # ── Canonical coinbase transaction rows ──────────────────────
                    cur.execute("""
                        INSERT INTO transactions
                            (tx_hash, from_address, to_address,
                             amount, tx_type, status, height, updated_at)
                        VALUES (%s, %s, %s, %s, 'coinbase', 'confirmed', %s, NOW())
                        ON CONFLICT (tx_hash) DO NOTHING
                    """, (_cb_miner_hash, "0" * 64, miner_address,
                          float(miner_reward), height))

                    if treasury_address and treasury_reward > 0:
                        cur.execute("""
                            INSERT INTO transactions
                                (tx_hash, from_address, to_address,
                                 amount, tx_type, status, height, updated_at)
                            VALUES (%s, %s, %s, %s, 'coinbase', 'confirmed', %s, NOW())
                            ON CONFLICT (tx_hash) DO NOTHING
                        """, (_cb_treasury_hash, "0" * 64, treasury_address,
                              float(treasury_reward), height))

            except Exception as credit_err:
                logger.error(
                    f"[RPC-submitBlock] ❌ Coinbase credit FAILED h={height}: {credit_err}",
                    exc_info=True
                )

        # ── Difficulty retarget ──────────────────────────────────────────────
        try:
            dm = get_difficulty_manager()
            dm.on_block_accepted(timestamp_s)
        except Exception:
            pass

        # ── Update chain state ──────────────────────────────────────────────
        try:
            with get_db_cursor() as cur:
                cur.execute("""
                    INSERT INTO chain_state (state_id, chain_height, head_block_hash, latest_coherence, updated_at)
                    VALUES (1, %s, %s, %s, NOW())
                    ON CONFLICT (state_id) DO UPDATE SET
                        chain_height = EXCLUDED.chain_height,
                        head_block_hash = EXCLUDED.head_block_hash,
                        latest_coherence = EXCLUDED.latest_coherence,
                        updated_at = NOW()
                """, (height, block_hash, w_state_fidelity))
        except Exception as cs_err:
            logger.warning(f"[RPC-submitBlock] Chain state update failed (non-fatal): {cs_err}")

        # ── In-memory blockchain index ────────────────────────────────────────
        try:
            from globals import get_blockchain
            get_blockchain().index_block(height, txs or [])
        except Exception:
            pass

        try: _resp_reward = TessellationRewardSchedule.get_miner_reward_qtcl(height) if TessellationRewardSchedule else 7.20
        except Exception: _resp_reward = 7.20
        if _block_rowcount == 0:
            logger.info(f"[RPC-submitBlock] 🔁 DUPLICATE h={height} hash={block_hash[:16]}…")
            return _rpc_ok({"status":"duplicate","height":height,"block_hash":block_hash}, rpc_id)
        logger.info(f"[RPC-submitBlock] ✅ ACCEPTED h={height} hash={block_hash[:16]}… miner={miner_address[:16]}… reward={_resp_reward} QTCL")
        return _rpc_ok({"status":"accepted","height":height,"block_hash":block_hash,"difficulty_bits":difficulty_bits,"miner_reward_qtcl":_resp_reward}, rpc_id)

    except Exception as e:
        logger.exception(f"[RPC] _rpc_submitBlock unhandled: {e}")
        return _rpc_error(-32603, f"Internal error: {str(e)}", rpc_id)


def _rpc_pushOracleDM(params: Any, rpc_id: Any) -> dict:
    """
    qtcl_pushOracleDM — accept a fused tripartite DM frame from a client oracle node.

    Params (dict):
        density_matrix_hex  str   — 1024-hex big-endian '>dd' 8×8 complex128 DM
        fidelity            float — W-state fidelity of the pushed DM  (0..1)
        oracle_type         str   — e.g. 'tripartite_client'
        node_ip             str   — caller's self-reported WAN IP (advisory)
        oracle_addr         str   — oracle signing address (qtcl1…)

    Server action:
        1. Validate DM hex (length, parse).
        2. Upsert into _CLIENT_DM_POOL keyed by oracle_addr.
        3. Evict oldest entries if pool > _CLIENT_POOL_MAX.
        4. Re-average pool → _client_consensus_dm_re/_im/_fid.
        5. Push composite snapshot (server 5-oracle fused with client pool) into
           _DM_SNAPSHOT_RING and _latest_snapshot so polling clients see it.
        6. Return {accepted, pool_size, client_consensus_fidelity}.
    ❤️  I love you — every push strengthens the distributed lattice
    """
    global _client_consensus_dm_re, _client_consensus_dm_im
    global _client_consensus_fid, _client_pool_count

    try:
        if not isinstance(params, dict):
            return _rpc_error(-32602, "params must be a dict", rpc_id)

        dm_hex     = params.get('density_matrix_hex', '')
        fidelity   = float(params.get('fidelity', 0.0))
        oracle_addr = str(params.get('oracle_addr', '') or f"anon_{int(time.time())}")
        node_ip    = str(params.get('node_ip', ''))
        oracle_type = str(params.get('oracle_type', 'tripartite_client'))

        # ── 1. Validate DM hex ─────────────────────────────────────────────
        if not dm_hex or len(dm_hex) != 2048:   # 64 × 2 × 8 bytes = 1024 bytes = 2048 hex chars
            return _rpc_error(-32602,
                f"density_matrix_hex must be 2048 hex chars (got {len(dm_hex)})", rpc_id)
        try:
            bdata = bytes.fromhex(dm_hex)
        except ValueError as _ve:
            return _rpc_error(-32602, f"density_matrix_hex not valid hex: {_ve}", rpc_id)

        dm_re = [0.0] * 64
        dm_im = [0.0] * 64
        for i in range(64):
            dm_re[i], dm_im[i] = struct.unpack_from('>dd', bdata, i * 16)

        # Sanity: trace should be ≈1
        tr = sum(dm_re[i * 8 + i] for i in range(8))
        if not (0.5 < tr < 1.5):
            return _rpc_error(-32602, f"DM trace out of range: {tr:.4f}", rpc_id)

        # ── 2 & 3. Upsert into pool, evict oldest if needed ───────────────
        with _CLIENT_DM_POOL_LOCK:
            _CLIENT_DM_POOL[oracle_addr] = {
                'dm_re':      dm_re,
                'dm_im':      dm_im,
                'fidelity':   max(0.0, min(1.0, fidelity)),
                'ts':         time.time(),
                'node_ip':    node_ip,
                'oracle_type': oracle_type,
            }
            # Evict oldest if over cap
            if len(_CLIENT_DM_POOL) > _CLIENT_POOL_MAX:
                _oldest = min(_CLIENT_DM_POOL, key=lambda k: _CLIENT_DM_POOL[k]['ts'])
                del _CLIENT_DM_POOL[_oldest]

            # ── 4. Re-average pool ─────────────────────────────────────────
            _recompute_client_consensus()
            _pool_size = _client_pool_count
            _cons_fid  = _client_consensus_fid

        # ── 5. Fuse client consensus with server 5-oracle snapshot ────────
        # Pull the current server snapshot (may be None if oracle not yet ready)
        try:
            with _snapshot_lock:
                _srv_snap = dict(_latest_snapshot) if _latest_snapshot else {}
        except Exception:
            _srv_snap = {}

        # Build a composite snapshot and push it into the ring + cache
        _srv_fid  = float(_srv_snap.get('w_state_fidelity') or
                          (_srv_snap.get('w_state') or {}).get('fidelity') or 0.0)
        _srv_dm_hex = _srv_snap.get('density_matrix_hex', '')

        # Weighted fuse: client contribution scales with _cons_fid, max 35%
        if _client_consensus_dm_re and any(v != 0.0 for v in _client_consensus_dm_re):
            try:
                _w_client = min(_cons_fid * 0.35, 0.35)
                _w_server = 1.0 - _w_client

                if _srv_dm_hex and len(_srv_dm_hex) == 2048:
                    _sd = bytes.fromhex(_srv_dm_hex)
                    _sre = [0.0]*64; _sim = [0.0]*64
                    for _i in range(64):
                        _sre[_i], _sim[_i] = struct.unpack_from('>dd', _sd, _i*16)
                    fused_re = [_w_server*_sre[i] + _w_client*_client_consensus_dm_re[i] for i in range(64)]
                    fused_im = [_w_server*_sim[i] + _w_client*_client_consensus_dm_im[i] for i in range(64)]
                    ftr = sum(fused_re[i*8+i] for i in range(8))
                    if ftr > 1e-12:
                        fused_re = [x/ftr for x in fused_re]
                        fused_im = [x/ftr for x in fused_im]
                    fused_hex = b''.join(struct.pack('>dd', fused_re[i], fused_im[i])
                                         for i in range(64)).hex()
                    fused_fid = _w_server*_srv_fid + _w_client*_cons_fid
                else:
                    fused_hex = dm_hex   # fallback: just use what client sent
                    fused_fid = fidelity

                composite = {
                    **_srv_snap,
                    'density_matrix_hex':    fused_hex,
                    'w_state_fidelity':      fused_fid,
                    'fidelity':              fused_fid,
                    'client_fused_fidelity': _cons_fid,
                    'client_oracle_count':   _pool_size,
                    'pq0_oracle_fidelity':   params.get('pq0_oracle_fidelity', fidelity),
                    'pq0_IV_fidelity':       params.get('pq0_IV_fidelity', fidelity),
                    'pq0_V_fidelity':        params.get('pq0_V_fidelity', fidelity),
                    'source':                'server+client_tripartite',
                    'ready':                 True,
                    'timestamp_ns':          int(time.time() * 1e9),
                }
                _broadcast_snapshot_to_database(composite)
            except Exception as _fe:
                logger.debug(f"[PUSH-DM] fuse error: {_fe}")

        logger.debug(
            f"[PUSH-DM] ✅ oracle_addr={oracle_addr[:16]} fid={fidelity:.4f} "
            f"pool={_pool_size} cons_fid={_cons_fid:.4f}"
        )
        return _rpc_ok({
            'accepted':                 True,
            'pool_size':                _pool_size,
            'client_consensus_fidelity': _cons_fid,
        }, rpc_id)

    except Exception as e:
        logger.exception(f"[RPC] qtcl_pushOracleDM: {e}")
        return _rpc_error(-32603, f"pushOracleDM failed: {e}", rpc_id)


def _rpc_getLatestDMSnapshot(params: Any, rpc_id: Any) -> dict:
    """qtcl_getLatestDMSnapshot — fetch latest density matrix snapshot from oracle ring buffer.
    
    Returns: {
        timestamp_ns, oracle_id, density_matrix_hex, purity, w_state_fidelity,
        von_neumann_entropy, coherence_l1, hlwe_signature, signature_valid, oracle_address
    }
    """
    try:
        with _DM_SNAPSHOT_LOCK:
            if not _DM_SNAPSHOT_RING:
                return _rpc_error(-32000, "No DM snapshots available yet", rpc_id)
            latest = list(_DM_SNAPSHOT_RING)[-1]
        logger.debug(f"[RPC-METHOD] qtcl_getLatestDMSnapshot: returned snapshot ts={latest.get('timestamp_ns')}")
        return _rpc_ok(latest, rpc_id)
    except Exception as e:
        logger.exception(f"[RPC-METHOD] qtcl_getLatestDMSnapshot: {e}")
        return _rpc_error(-32603, f"Internal error: {str(e)}", rpc_id)


def _rpc_getLatestDMSnapshots(params: Any, rpc_id: Any) -> dict:
    """qtcl_getLatestDMSnapshots — fetch last N DM snapshots (default 10, max 100)."""
    try:
        limit = 10
        if isinstance(params, list) and params:
            try:
                limit = min(int(params[0]), 100)
            except (ValueError, TypeError):
                pass
        elif isinstance(params, dict):
            try:
                limit = min(int(params.get("limit", 10)), 100)
            except (ValueError, TypeError):
                pass
        
        with _DM_SNAPSHOT_LOCK:
            snaps = list(_DM_SNAPSHOT_RING)[-limit:] if _DM_SNAPSHOT_RING else []
        
        logger.debug(f"[RPC-METHOD] qtcl_getLatestDMSnapshots: returned {len(snaps)} snapshots")
        return _rpc_ok({"snapshots": snaps, "count": len(snaps)}, rpc_id)
    except Exception as e:
        logger.exception(f"[RPC-METHOD] qtcl_getLatestDMSnapshots: {e}")
        return _rpc_error(-32603, f"Internal error: {str(e)}", rpc_id)



def _rpc_getMempool(params: Any, rpc_id: Any) -> dict:
    """qtcl_getMempool — pending transaction list for block building."""
    try:
        from mempool import get_pending_transactions as _get_pending
        max_count = 500
        if isinstance(params, list) and params:
            try: max_count = min(int(params[0]), 2000)
            except (ValueError, TypeError): pass
        txs = _get_pending(max_count=max_count)
        serialized = []
        for tx in txs:
            if hasattr(tx, '__dict__'):
                serialized.append({k: v for k, v in tx.__dict__.items() if not k.startswith('_')})
            elif isinstance(tx, dict):
                serialized.append(tx)
        logger.debug(f"[RPC-METHOD] qtcl_getMempool: returning {len(serialized)} txs")
        return _rpc_ok(serialized, rpc_id)
    except Exception as e:
        logger.exception(f"[RPC-METHOD] qtcl_getMempool: {e}")
        return _rpc_ok([], rpc_id)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTERPRISE P2P NETWORK — Inline Implementation (no external files)
# ═══════════════════════════════════════════════════════════════════════════════
P2P_BROADCAST_INTERVAL = 30
P2P_PEER_TIMEOUT = 300
P2P_MAX_PEERS = 100

class P2PPeer:
    """A peer in the P2P network. Peer = WALLET, not oracle."""
    def __init__(self, peer_id: str = "", wallet_address: str = "", external_addr: str = "", 
                 port: int = 9091, public_key: str = "", chain_height: int = 0,
                 last_seen: float = 0.0, first_seen: float = 0.0, is_alive: bool = True):
        self.peer_id = peer_id
        self.wallet_address = wallet_address
        self.external_addr = external_addr
        self.port = port
        self.public_key = public_key
        self.chain_height = chain_height
        self.last_seen = last_seen
        self.first_seen = first_seen
        self.is_alive = is_alive
    
    def to_dict(self) -> dict:
        return {"peer_id": self.peer_id, "wallet_address": self.wallet_address,
                "external_addr": self.external_addr, "port": self.port,
                "public_key": self.public_key, "chain_height": self.chain_height,
                "last_seen": self.last_seen, "first_seen": self.first_seen,
                "is_alive": self.is_alive}

class _P2PSQLiteStore:
    """SQLite store for peer persistence on client side."""
    def __init__(self, db_path: str = "peers.sqlite"):
        import sqlite3
        self.db_path = db_path
        self._lock = threading.RLock()
        conn = sqlite3.connect(db_path)
        conn.execute("""CREATE TABLE IF NOT EXISTS peer_registry (
            peer_id TEXT PRIMARY KEY, wallet_address TEXT, external_addr TEXT,
            port INTEGER, public_key TEXT, chain_height INTEGER, last_seen REAL,
            first_seen REAL, is_alive INTEGER)""")
        conn.commit()
        conn.close()
    
    def upsert_peer(self, peer: P2PPeer) -> bool:
        import sqlite3
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            now = time.time()
            if peer.first_seen == 0:
                peer.first_seen = now
            conn.execute("""INSERT OR REPLACE INTO peer_registry 
                (peer_id, wallet_address, external_addr, port, public_key, chain_height,
                 last_seen, first_seen, is_alive) VALUES (?,?,?,?,?,?,?,?,?)""",
                (peer.peer_id, peer.wallet_address, peer.external_addr, peer.port,
                 peer.public_key, peer.chain_height, peer.last_seen, peer.first_seen,
                 1 if peer.is_alive else 0))
            conn.commit()
            conn.close()
            return True
    
    def get_alive_peers(self) -> list:
        import sqlite3
        cutoff = time.time() - P2P_PEER_TIMEOUT
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""SELECT * FROM peer_registry 
                WHERE last_seen > ? AND is_alive = 1 ORDER BY last_seen DESC LIMIT ?""",
                (cutoff, P2P_MAX_PEERS)).fetchall()
            conn.close()
            return [P2PPeer(r["peer_id"], r["wallet_address"], r["external_addr"],
                     r["port"], r["public_key"], r["chain_height"], r["last_seen"],
                     r["first_seen"], bool(r["is_alive"])) for r in rows]

_p2p_dht_table: Dict[str, P2PPeer] = {}
_p2p_dht_lock = threading.RLock()
_p2p_seen_hashes: set = set()
_p2p_client_store: Optional[_P2PSQLiteStore] = None

def _p2p_rpc_get_dht_table(params, rpc_id):
    """qtcl_getDHTTable — Return the full DHT peer table."""
    try:
        limit = 100
        if isinstance(params, dict):
            limit = min(int(params.get("limit", 100)), P2P_MAX_PEERS)
        with _p2p_dht_lock:
            peers = list(_p2p_dht_table.values())[:limit]
        return {"peers": [p.to_dict() for p in peers], "count": len(peers), "timestamp": time.time()}
    except Exception as e:
        logger.error(f"[P2P-RPC] getDHTTable error: {e}")
        return {"peers": [], "count": 0, "timestamp": time.time()}

def _p2p_rpc_receive_dht_table(params, rpc_id):
    """qtcl_receiveDHTTable — Receive a DHT table from another peer."""
    try:
        dht_json = params.get("dht_table", "") if isinstance(params, dict) else ""
        from_peer = params.get("propagating_from", "") if isinstance(params, dict) else ""
        dht_hash = params.get("dht_hash", "") if isinstance(params, dict) else ""
        if not dht_json:
            return {"status": "error", "message": "dht_table required"}
        if dht_hash and dht_hash in _p2p_seen_hashes:
            return {"status": "already_seen", "dht_hash": dht_hash[:16]}
        import json
        doc = json.loads(dht_json)
        peers_data = doc.get("peers", [])
        new_count = 0
        with _p2p_dht_lock:
            for pd in peers_data:
                p = P2PPeer(pd.get("peer_id", ""), pd.get("wallet_address", ""),
                            pd.get("external_addr", ""), pd.get("port", 9091),
                            pd.get("public_key", ""), pd.get("chain_height", 0),
                            pd.get("last_seen", time.time()), pd.get("first_seen", 0),
                            pd.get("is_alive", True))
                if p.peer_id not in _p2p_dht_table:
                    new_count += 1
                p.last_seen = time.time()
                _p2p_dht_table[p.peer_id] = p
        if dht_hash:
            _p2p_seen_hashes.add(dht_hash)
            if len(_p2p_seen_hashes) > 10000:
                _p2p_seen_hashes = set(list(_p2p_seen_hashes)[-5000:])
        logger.info(f"[P2P] ← Received DHT from {from_peer[:16]}…: {len(peers_data)} peers ({new_count} new)")
        return {"status": "accepted", "peer_count": len(peers_data), "new_peers": new_count}
    except Exception as e:
        logger.error(f"[P2P-RPC] receiveDHTTable error: {e}")
        return {"status": "error", "message": str(e)}

def _p2p_rpc_peer_heartbeat(params, rpc_id):
    """qtcl_peerHeartbeat — Register a peer's heartbeat."""
    try:
        peer_id = params.get("peer_id", "") if isinstance(params, dict) else ""
        wallet_address = params.get("wallet_address", "") if isinstance(params, dict) else ""
        external_addr = params.get("external_addr", "") if isinstance(params, dict) else ""
        port = int(params.get("port", 9091)) if isinstance(params, dict) else 9091
        chain_height = int(params.get("chain_height", 0)) if isinstance(params, dict) else 0
        if not peer_id:
            return {"status": "error", "message": "peer_id required"}
        with _p2p_dht_lock:
            if peer_id in _p2p_dht_table:
                p = _p2p_dht_table[peer_id]
                p.last_seen = time.time()
                p.chain_height = max(p.chain_height, chain_height)
                p.is_alive = True
            else:
                p = P2PPeer(peer_id=peer_id, wallet_address=wallet_address,
                           external_addr=external_addr, port=port,
                           chain_height=chain_height, last_seen=time.time(),
                           first_seen=time.time(), is_alive=True)
                _p2p_dht_table[peer_id] = p
        return {"status": "ok", "peer_id": peer_id, "timestamp": time.time()}
    except Exception as e:
        logger.error(f"[P2P-RPC] peerHeartbeat error: {e}")
        return {"status": "error", "message": str(e)}


_RPC_METHODS: Dict[str, Any] = {
    "qtcl_submitBlock":       _rpc_submitBlock,
    "qtcl_getBlockHeight":    _rpc_getBlockHeight,
    "qtcl_getBalance":        _rpc_getBalance,
    "qtcl_getTransaction":    _rpc_getTransaction,
    "qtcl_getBlock":          _rpc_getBlock,
    "qtcl_getBlockRange":     _rpc_getBlockRange,
    "qtcl_getQuantumMetrics": _rpc_getQuantumMetrics,
    "qtcl_getPythPrice":      _rpc_getPythPrice,
    "qtcl_getMempoolStats":   _rpc_getMempoolStats,
    "qtcl_getMempool":        _rpc_getMempool,
    "qtcl_getPeers":          _rpc_getPeers,
    "qtcl_getPeersByNatGroup": _rpc_getPeersByNatGroup,
    "qtcl_registerPeer":      _rpc_registerPeer,      # ← NEW: miner bootstrap registration
    "qtcl_getMyAddr":         _rpc_getMyAddr,          # ← NEW: STUN — return caller's observed IP
    "qtcl_getHealth":         _rpc_getHealth,
    "qtcl_getEvents":         _rpc_getEvents,
    "qtcl_getOracleRegistry": _rpc_getOracleRegistry,
    "qtcl_getOracleRecord":   _rpc_getOracleRecord,
    "qtcl_submitOracleReg":   _rpc_submitOracleReg,
    "qtcl_registerMeasurementSubscriber": _rpc_registerMeasurementSubscriber,
    "qtcl_unregisterMeasurementSubscriber": _rpc_unregisterMeasurementSubscriber,
    "qtcl_listMeasurementSubscribers": _rpc_listMeasurementSubscribers,
    # ── NEW: Density Matrix Snapshot Streaming ──────────────────────────────────
    "qtcl_getLatestDMSnapshot": _rpc_getLatestDMSnapshot,
    "qtcl_getLatestDMSnapshots": _rpc_getLatestDMSnapshots,
    # ── NEW: Client Tripartite Oracle Push ──────────────────────────────────────
    "qtcl_pushOracleDM": _rpc_pushOracleDM,
    # P2P DHT methods
    "qtcl_getDHTTable":       _p2p_rpc_get_dht_table,
    "qtcl_receiveDHTTable":   _p2p_rpc_receive_dht_table,
    "qtcl_peerHeartbeat":     _p2p_rpc_peer_heartbeat,
}

# ═══════════════════════════════════════════════════════════════════════════════
# ENTERPRISE P2P NETWORK — Inline Implementation (no external files)
# ═══════════════════════════════════════════════════════════════════════════════
P2P_BROADCAST_INTERVAL = 30
P2P_PEER_TIMEOUT = 300
P2P_MAX_PEERS = 100

class P2PPeer:
    """A peer in the P2P network. Peer = WALLET, not oracle."""
    def __init__(self, peer_id: str = "", wallet_address: str = "", external_addr: str = "", 
                 port: int = 9091, public_key: str = "", chain_height: int = 0,
                 last_seen: float = 0.0, first_seen: float = 0.0, is_alive: bool = True):
        self.peer_id = peer_id
        self.wallet_address = wallet_address
        self.external_addr = external_addr
        self.port = port
        self.public_key = public_key
        self.chain_height = chain_height
        self.last_seen = last_seen
        self.first_seen = first_seen
        self.is_alive = is_alive
    
    def to_dict(self) -> dict:
        return {"peer_id": self.peer_id, "wallet_address": self.wallet_address,
                "external_addr": self.external_addr, "port": self.port,
                "public_key": self.public_key, "chain_height": self.chain_height,
                "last_seen": self.last_seen, "first_seen": self.first_seen,
                "is_alive": self.is_alive}

class _P2PSQLiteStore:
    """SQLite store for peer persistence on client side."""
    def __init__(self, db_path: str = "peers.sqlite"):
        import sqlite3
        self.db_path = db_path
        self._lock = threading.RLock()
        conn = sqlite3.connect(db_path)
        conn.execute("""CREATE TABLE IF NOT EXISTS peer_registry (
            peer_id TEXT PRIMARY KEY, wallet_address TEXT, external_addr TEXT,
            port INTEGER, public_key TEXT, chain_height INTEGER, last_seen REAL,
            first_seen REAL, is_alive INTEGER)""")
        conn.commit()
        conn.close()
    
    def upsert_peer(self, peer: P2PPeer) -> bool:
        import sqlite3
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            now = time.time()
            if peer.first_seen == 0:
                peer.first_seen = now
            conn.execute("""INSERT OR REPLACE INTO peer_registry 
                (peer_id, wallet_address, external_addr, port, public_key, chain_height,
                 last_seen, first_seen, is_alive) VALUES (?,?,?,?,?,?,?,?,?)""",
                (peer.peer_id, peer.wallet_address, peer.external_addr, peer.port,
                 peer.public_key, peer.chain_height, peer.last_seen, peer.first_seen,
                 1 if peer.is_alive else 0))
            conn.commit()
            conn.close()
            return True
    
    def get_alive_peers(self) -> list:
        import sqlite3
        cutoff = time.time() - P2P_PEER_TIMEOUT
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""SELECT * FROM peer_registry 
                WHERE last_seen > ? AND is_alive = 1 ORDER BY last_seen DESC LIMIT ?""",
                (cutoff, P2P_MAX_PEERS)).fetchall()
            conn.close()
            return [P2PPeer(r["peer_id"], r["wallet_address"], r["external_addr"],
                     r["port"], r["public_key"], r["chain_height"], r["last_seen"],
                     r["first_seen"], bool(r["is_alive"])) for r in rows]

_p2p_dht_table: Dict[str, P2PPeer] = {}
_p2p_dht_lock = threading.RLock()
_p2p_seen_hashes: set = set()
_p2p_client_store: Optional[_P2PSQLiteStore] = None

def _p2p_rpc_get_dht_table(params, rpc_id):
    """qtcl_getDHTTable — Return the full DHT peer table."""
    try:
        limit = 100
        if isinstance(params, dict):
            limit = min(int(params.get("limit", 100)), P2P_MAX_PEERS)
        with _p2p_dht_lock:
            peers = list(_p2p_dht_table.values())[:limit]
        return {"peers": [p.to_dict() for p in peers], "count": len(peers), "timestamp": time.time()}
    except Exception as e:
        logger.error(f"[P2P-RPC] getDHTTable error: {e}")
        return {"peers": [], "count": 0, "timestamp": time.time()}

def _p2p_rpc_receive_dht_table(params, rpc_id):
    """qtcl_receiveDHTTable — Receive a DHT table from another peer."""
    try:
        dht_json = params.get("dht_table", "") if isinstance(params, dict) else ""
        from_peer = params.get("propagating_from", "") if isinstance(params, dict) else ""
        dht_hash = params.get("dht_hash", "") if isinstance(params, dict) else ""
        if not dht_json:
            return {"status": "error", "message": "dht_table required"}
        if dht_hash and dht_hash in _p2p_seen_hashes:
            return {"status": "already_seen", "dht_hash": dht_hash[:16]}
        import json
        doc = json.loads(dht_json)
        peers_data = doc.get("peers", [])
        new_count = 0
        with _p2p_dht_lock:
            for pd in peers_data:
                p = P2PPeer(pd.get("peer_id", ""), pd.get("wallet_address", ""),
                            pd.get("external_addr", ""), pd.get("port", 9091),
                            pd.get("public_key", ""), pd.get("chain_height", 0),
                            pd.get("last_seen", time.time()), pd.get("first_seen", 0),
                            pd.get("is_alive", True))
                if p.peer_id not in _p2p_dht_table:
                    new_count += 1
                p.last_seen = time.time()
                _p2p_dht_table[p.peer_id] = p
        if dht_hash:
            _p2p_seen_hashes.add(dht_hash)
            if len(_p2p_seen_hashes) > 10000:
                _p2p_seen_hashes = set(list(_p2p_seen_hashes)[-5000:])
        logger.info(f"[P2P] ← Received DHT from {from_peer[:16]}…: {len(peers_data)} peers ({new_count} new)")
        return {"status": "accepted", "peer_count": len(peers_data), "new_peers": new_count}
    except Exception as e:
        logger.error(f"[P2P-RPC] receiveDHTTable error: {e}")
        return {"status": "error", "message": str(e)}

def _p2p_rpc_peer_heartbeat(params, rpc_id):
    """qtcl_peerHeartbeat — Receive heartbeat from a peer."""
    try:
        peer_id = params.get("peer_id", "") if isinstance(params, dict) else ""
        if peer_id:
            with _p2p_dht_lock:
                if peer_id in _p2p_dht_table:
                    _p2p_dht_table[peer_id].last_seen = time.time()
        return {"status": "ok", "timestamp": time.time()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def _p2p_fanout_broadcast():
    """Fan-out broadcast DHT table to all known peers."""
    import json
    from urllib.request import Request, urlopen
    from urllib.error import URLError, HTTPError
    with _p2p_dht_lock:
        peers = list(_p2p_dht_table.values())
    if len(peers) < 2:
        return
    dht_json = json.dumps({"version": 1, "timestamp": time.time(), "peer_count": len(peers),
                          "peers": [p.to_dict() for p in peers]}, separators=(',', ':'))
    dht_hash = hashlib.sha256(dht_json.encode()).hexdigest()
    if dht_hash in _p2p_seen_hashes:
        return
    _p2p_seen_hashes.add(dht_hash)
    sent = failed = 0
    for peer in peers:
        if peer.peer_id == ORACLE_ID:
            continue
        try:
            # Construct URL with port - external_addr may or may not include port
            if ':' in peer.external_addr:
                # Already has port in external_addr (e.g., "192.168.1.100:9091")
                url = f"http://{peer.external_addr}/rpc"
            else:
                # Use port from peer object
                url = f"http://{peer.external_addr}:{peer.port}/rpc"
            payload = json.dumps({"jsonrpc": "2.0", "method": "qtcl_receiveDHTTable",
                                  "params": {"dht_table": dht_json, "propagating_from": ORACLE_ID, "dht_hash": dht_hash},
                                  "id": 1}).encode()
            req = Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    sent += 1
        except Exception:
            failed += 1
        time.sleep(0.05)
    if sent > 0 or failed > 0:
        logger.info(f"[P2P] Fan-out: sent to {sent}, failed {failed}")

_p2p_broadcast_count = 0
_p2p_running = False
_p2p_broadcast_thread: Optional[threading.Thread] = None

def _p2p_broadcast_loop():
    """30-second DHT broadcast loop."""
    global _p2p_broadcast_count, _p2p_running
    logger.info("[P2P] Broadcast loop started")
    while _p2p_running:
        try:
            _p2p_broadcast_count += 1
            # Fetch peers from DB
            try:
                with get_db_cursor() as cur:
                    cur.execute("""SELECT node_id, external_addr, pubkey_hash, chain_height, last_seen
                        FROM peer_registry WHERE last_seen > NOW() - INTERVAL '10 minutes' 
                        AND ban_score < 100 LIMIT %s""", (P2P_MAX_PEERS,))
                    rows = cur.fetchall()
                    new_count = 0
                    with _p2p_dht_lock:
                        for row in rows:
                            nid, addr, pubk, height, last_seen = row
                            if nid not in _p2p_dht_table:
                                new_count += 1
                            ts = last_seen.timestamp() if hasattr(last_seen, "timestamp") else last_seen
                            _p2p_dht_table[nid] = P2PPeer(nid, "", addr or "", 9091, pubk or "", int(height or 0), ts)
                    logger.debug(f"[P2P] Cycle {_p2p_broadcast_count}: {len(rows)} peers from DB ({new_count} new)")
            except Exception as e:
                # Log but don't crash - use in-memory cache
                if "does not exist" in str(e):
                    logger.warning(f"[P2P] DB table missing - waiting for peer_registry to be created")
                else:
                    logger.warning(f"[P2P] DB fetch: {e}")
            # Fan-out broadcast
            _p2p_fanout_broadcast()
        except Exception as e:
            logger.error(f"[P2P] Broadcast cycle error: {e}")
        for _ in range(P2P_BROADCAST_INTERVAL * 2):
            if not _p2p_running:
                break
            time.sleep(0.5)
    logger.info("[P2P] Broadcast loop exited")

def _start_p2p_broadcast():
    """Start the P2P broadcast daemon."""
    global _p2p_running, _p2p_broadcast_thread
    if _p2p_running:
        return
    _p2p_running = True
    _p2p_broadcast_thread = threading.Thread(target=_p2p_broadcast_loop, daemon=True, name="P2PBroadcast")
    _p2p_broadcast_thread.start()
    logger.info(f"[P2P] ✅ DHT broadcaster started (30s interval)")

@app.route("/rpc", methods=["POST"])
def rpc_endpoint():
    """POST /rpc — JSON-RPC 2.0 endpoint for all P2P and blockchain operations."""
    try:
        body = request.get_data()
        if not body:
            return jsonify(_rpc_error(-32600, "Empty request", None)), 400
        
        result, status_code = _dispatch(body)
        
        if status_code >= 400:
            return jsonify(result), status_code
        if result is None:
            return "", 204
        return jsonify(result), 200
    except Exception as e:
        logger.exception(f"[RPC] Endpoint error: {e}")
        return jsonify(_rpc_error(-32603, str(e), None)), 500



# ═══════════════════════════════════════════════════════════════════════════════════
# DISTRIBUTED HASH TABLE (DHT) INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════════
@app.route("/rpc/methods", methods=["GET"])
def rpc_methods():
    """GET /rpc/methods — introspection: list all available RPC methods."""
    return jsonify({
        "jsonrpc":  _JSONRPC_VERSION,
        "endpoint": "/rpc",
        "methods":  _RPC_METHOD_META,
        "count":    len(_RPC_METHOD_META),
        "batch":    True,
        "ts":       time.time(),
    }), 200



def _get_pyth():
    """Return the Pyth oracle engine if initialized, else None. ❤️ I love you."""
    try:
        from oracle import ORACLE as _o
        return _o if _o is not None else None
    except Exception:
        return None


@app.route("/rpc/health", methods=["GET"])
def rpc_health():
    """GET /rpc/health — JSON-RPC engine and Pyth oracle health."""
    po = _get_pyth()
    return jsonify({
        "rpc_engine":       "ok",
        "jsonrpc_version":  _JSONRPC_VERSION,
        "method_count":     len(_RPC_METHODS),
        "pyth_ready":       po is not None,
        "pyth_stats":       po.stats() if po else {},
        "oracle_ready":     ORACLE_AVAILABLE,
        "lattice_ready":    LATTICE is not None,
        "uptime_s":         round(time.time() - _SERVER_START_TIME, 1),
        "ts":               time.time(),
    }), 200


@app.route("/health", methods=["GET"])
def health_bare():
    """GET /health — bare 200 OK for Koyeb health check (no JSON, fast)."""
    return "", 200


# ═══ STATIC FILE & ROOT SERVING ═══
@app.route("/", methods=["GET"])
def serve_root():
    """GET / — Serve index.html as the dashboard."""
    try:
        # First, try to serve from a dedicated static directory
        import os
        from flask import send_file
        index_path = os.path.join(os.path.dirname(__file__), 'index.html')
        if os.path.exists(index_path):
            return send_file(index_path, mimetype='text/html')
        
        # Fallback: inline the dashboard HTML (for production on Koyeb)
        return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>QTCL - Quantum Blockchain</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <style>
        *{margin:0;padding:0;box-sizing:border-box}
        body{font-family:'JetBrains Mono',monospace;background:#030610;color:#bbc8e0;min-height:100vh;padding:20px}
        #app{max-width:1200px;margin:0 auto;padding:20px;background:#080e20;border:1px solid rgba(0,220,255,.1);border-radius:8px}
        h1{color:#00dcff;margin-bottom:20px;font-size:28px;letter-spacing:2px}
        .status{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:15px;margin-bottom:30px}
        .card{background:#0a1228;border:1px solid rgba(0,220,255,.2);border-radius:6px;padding:15px}
        .card-title{color:#00dcff;font-size:12px;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}
        .card-value{font-size:18px;color:#fff;font-weight:bold}
        .card-unit{font-size:11px;color:#666;margin-top:4px}
        .loading{text-align:center;padding:40px;color:#666}
        .error{color:#ff3355;background:rgba(255,51,85,.1);padding:15px;border-radius:6px;margin:15px 0}
        .success{color:#00ff9d;background:rgba(0,255,157,.1);padding:15px;border-radius:6px;margin:15px 0}
    </style>
</head>
<body>
<div id="app">
    <h1>⚛️ QTCL — Quantum Temporal Coherence Ledger</h1>
    <div class="status">
        <div class="card">
            <div class="card-title">Oracle Status</div>
            <div class="card-value" id="oracle-status">Loading...</div>
        </div>
        <div class="card">
            <div class="card-title">Lattice State</div>
            <div class="card-value" id="lattice-status">Loading...</div>
        </div>
        <div class="card">
            <div class="card-title">Block Height</div>
            <div class="card-value" id="block-height">—</div>
        </div>
        <div class="card">
            <div class="card-title">Consensus</div>
            <div class="card-value" id="consensus">—</div>
        </div>
    </div>
    <div id="status-message" class="loading">Connecting to QTCL network...</div>
</div>
<script>
    async function updateStatus() {
        try {
            const health = await fetch('/health').then(() => '✅ Healthy');
            document.getElementById('oracle-status').textContent = health;
            
            try {
                const rpc = await fetch('/rpc', {
                    method:'POST',
                    headers:{'Content-Type':'application/json'},
                    body:JSON.stringify({
                        jsonrpc:'2.0',
                        method:'qtcl_getHealth',
                        params:[],
                        id:1
                    })
                }).then(r => r.json());
                
                if(rpc.result?.status) {
                    document.getElementById('lattice-status').textContent = '✅ Active';
                    document.getElementById('status-message').innerHTML = 
                        '<div class="success">✅ QTCL system is operational</div>';
                } else {
                    document.getElementById('status-message').innerHTML = 
                        '<div class="error">⚠️ System initializing...</div>';
                }
            } catch(e) {
                document.getElementById('status-message').innerHTML = 
                    '<div class="loading">System initializing... (quantum subsystems booting)</div>';
            }
        } catch(e) {
            document.getElementById('status-message').innerHTML = 
                '<div class="error">❌ Connection failed</div>';
        }
    }
    updateStatus();
    setInterval(updateStatus, 5000);
</script>
</body>
</html>
        """), 200
    except Exception as e:
        logger.error(f"[ROOT] Failed to serve index: {e}")
        return jsonify({"error": "Root endpoint failed", "detail": str(e)}), 500


@app.route("/rpc/hlwe/system-info", methods=["GET"])
def rpc_hlwe_system_info():
    """GET /rpc/hlwe/system-info — HLWE cryptographic system information.
    
    This endpoint is called by wsgi_config.py to verify HLWE is available
    without requiring direct imports at module load time.
    """
    try:
        from hlwe_engine import hlwe_system_info
        info = hlwe_system_info()
        return jsonify({
            "status": "ok",
            "hlwe_info": info,
            "timestamp": time.time(),
        }), 200
    except Exception as e:
        logger.error(f"[RPC-HLWE] Failed to get system info: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": time.time(),
        }), 500


# ──────────────────────────────────────────────────────────────────────────────
# RPC-only Architecture (no legacy REST endpoints)
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/rpc/_internal/measurement", methods=["POST"])
def rpc_measurement_broadcast_endpoint():
    """
    POST /rpc/_internal/measurement — Receive oracle measurement broadcast from controller.
    
    This endpoint is called by the RPC broadcast controller to distribute oracle 
    snapshots to subscribed clients. In normal operation, external callers should 
    use qtcl_registerMeasurementSubscriber RPC method to subscribe.
    
    Request (from broadcast controller):
        {
            "timestamp_ns": 1234567890000000,
            "cycle": 42,
            "w_state": {
                "fidelity": 0.7542,
                "coherence": 0.7605,
                ...
            },
            ...
        }
    
    Response:
        { "status": "processed" }
    """
    try:
        snap = request.get_json()
        if not snap:
            return jsonify({'status': 'invalid', 'error': 'no JSON payload'}), 400
        
        # Log broadcast receipt (optional, for debugging)
        cycle = snap.get('cycle', '?')
        fidelity = snap.get('w_state', {}).get('fidelity', 0)
        logger.debug(
            f"[BROADCAST-ENDPOINT] Received measurement | cycle={cycle} | "
            f"fidelity={fidelity:.4f}"
        )
        
        return jsonify({'status': 'processed'}), 200
    
    except Exception as e:
        logger.error(f"[BROADCAST-ENDPOINT] Error: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

# ══════════════════════════════════════════════════════════════════════════════
# PYTH PRICE ORACLE REST ROUTES
# (Complement to JSON-RPC — for REST-native integrations)
# ══════════════════════════════════════════════════════════════════════════════

def pyth_prices_rest():
    """
    GET /api/pyth/prices?symbols=BTC,ETH,SOL

    Returns an atomic Pyth snapshot for the requested symbols.
    Query params:
      symbols  — comma-separated list (default: all)
      refresh  — "true" to force bypass cache
    """
    po = _get_pyth()
    if po is None:
        return jsonify({"error": "Pyth oracle not initialized"}), 503

    symbols_raw = request.args.get("symbols", "")
    symbols: Optional[list] = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()] or None
    force   = request.args.get("refresh", "false").lower() == "true"

    try:
        snap = po.get_snapshot(symbols, force_refresh=force)
        resp = jsonify(snap.to_dict())
        resp.headers["Cache-Control"] = "public, max-age=5"
        resp.headers["X-Pyth-Snapshot"] = snap.snapshot_id[:16]
        resp.headers["X-Hermes-OK"]     = str(snap.hermes_ok).lower()
        return resp, 200
    except Exception as e:
        logger.error(f"[PYTH-REST] /api/pyth/prices error: {e}")
        return jsonify({"error": str(e)}), 500


def pyth_single_price_rest(symbol: str):
    """
    GET /api/pyth/price/BTC

    Single-symbol price convenience endpoint.
    Returns price_usd, confidence, age_seconds, publish_time.
    """
    po = _get_pyth()
    if po is None:
        return jsonify({"error": "Pyth oracle not initialized"}), 503

    sym = symbol.upper()
    try:
        snap = po.get_snapshot([sym])
        feed = snap.feeds.get(sym)
        if feed is None:
            return jsonify({
                "error":      f"Symbol {sym} not found",
                "available":  list(snap.feeds.keys()),
                "hermes_ok":  snap.hermes_ok,
            }), 404
        return jsonify({
            **feed.to_dict(),
            "snapshot_id": snap.snapshot_id,
            "hermes_ok":   snap.hermes_ok,
        }), 200
    except Exception as e:
        logger.error(f"[PYTH-REST] /api/pyth/price/{sym} error: {e}")
        return jsonify({"error": str(e)}), 500


def pyth_feed_catalog():
    """GET /api/pyth/feeds — full Pyth feed ID catalog (symbol → feed_id)."""
    return jsonify({
        "feeds":   {k: v for k, v in __import__("oracle").PYTH_FEED_IDS.items()},
        "count":   len(__import__("oracle").PYTH_FEED_IDS),
        "hermes":  "https://hermes.pyth.network",
        "ts":      time.time(),
    }), 200


def pyth_full_snapshot():
    """
    GET /api/pyth/snapshot — full HLWE-signed atomic price snapshot (all feeds).

    Returns the complete PythAtomicSnapshot including HLWE oracle signature,
    Byzantine outlier report, and raw attestation metadata.
    """
    po = _get_pyth()
    if po is None:
        return jsonify({"error": "Pyth oracle not initialized"}), 503
    try:
        snap = po.get_snapshot()   # all feeds
        return jsonify({
            **snap.to_dict(),
            "feed_count": len(snap.feeds),
            "outlier_count": len(snap.outliers),
        }), 200
    except Exception as e:
        logger.error(f"[PYTH-REST] /api/pyth/snapshot error: {e}")
        return jsonify({"error": str(e)}), 500


def pyth_oracle_stats():
    """GET /api/pyth/stats — Pyth oracle runtime statistics."""
    po = _get_pyth()
    if po is None:
        return jsonify({"error": "Pyth oracle not initialized"}), 503
    return jsonify(po.stats()), 200

@app.route("/rpc/oracle/snapshot", methods=["GET", "POST", "OPTIONS"])
def rpc_oracle_snapshot():
    """GET/POST /rpc/oracle/snapshot — Latest W-state snapshot fused with client tripartite pool."""
    if request.method == "OPTIONS":
        return "", 204
    try:
        with _snapshot_lock:
            if _latest_snapshot is None:
                # No server snapshot yet — but we may still have client pool data
                _base = {}
            else:
                _base = dict(_latest_snapshot)

        # ── Enrich with client tripartite pool consensus ───────────────────
        with _CLIENT_DM_POOL_LOCK:
            _c_re   = list(_client_consensus_dm_re)
            _c_im   = list(_client_consensus_dm_im)
            _c_fid  = _client_consensus_fid
            _c_cnt  = _client_pool_count

        if _c_cnt > 0 and any(v != 0.0 for v in _c_re):
            _base['client_fused_fidelity'] = round(_c_fid, 6)
            _base['client_oracle_count']   = _c_cnt
            # If server has no DM yet, surface the client consensus DM
            if not _base.get('density_matrix_hex'):
                import struct as _ss
                _base['density_matrix_hex'] = b''.join(
                    _ss.pack('>dd', _c_re[i], _c_im[i]) for i in range(64)
                ).hex()
                _base['w_state_fidelity'] = _c_fid
                _base['fidelity']         = _c_fid
                _base['source']           = 'client_tripartite_only'
                _base['ready']            = True
        else:
            _base['client_fused_fidelity'] = 0.0
            _base['client_oracle_count']   = 0

        if not _base:
            return jsonify({"jsonrpc":"2.0","error":{"code":-32000,"message":"No snapshot yet"},"id":None}), 202

        return jsonify({"jsonrpc":"2.0","result":_base,"id":request.args.get('id',1)}), 200
    except Exception as e:
        logger.error(f"[RPC-ORACLE] /rpc/oracle/snapshot error: {e}", exc_info=False)
        return jsonify({"jsonrpc":"2.0","error":{"code":-32603,"message":str(e)},"id":None}), 500

@app.route("/rpc/oracle/snapshots", methods=["GET", "POST", "OPTIONS"])
def rpc_oracle_snapshots():
    """GET/POST /rpc/oracle/snapshots — Ring buffer of last N snapshots."""
    if request.method == "OPTIONS":
        return "", 204
    try:
        limit=int(request.args.get('limit',10))
        with _rpc_event_lock:
            snaps=[e for e in list(_rpc_event_log) if e.get('type')=='snapshot'][-limit:]
        return jsonify({"jsonrpc":"2.0","result":{"snapshots":snaps,"count":len(snaps)},"id":request.args.get('id',1)}), 200
    except Exception as e:
        logger.error(f"[RPC-ORACLE] /rpc/oracle/snapshots error: {e}", exc_info=False)
        return jsonify({"jsonrpc":"2.0","error":{"code":-32603,"message":str(e)},"id":None}), 500

logger.info("[JSONRPC] ✅ JSON-RPC 2.0 engine mounted — /rpc, /rpc/methods, /rpc/health")
logger.info("[RPC-ORACLE] ✅ Oracle RPC routes mounted — /rpc/oracle/{snapshot,snapshots}")
logger.info("[PYTH]    ✅ Pyth REST routes mounted — /api/pyth/{prices,price/<sym>,feeds,snapshot,stats}")

# ⚛️ RPC SNAPSHOT BROADCAST SYSTEM (No SSE, Pure Database + HTTP Polling)
# ═════════════════════════════════════════════════════════════════════════════════

def _broadcast_snapshot_to_database(snapshot: dict) -> None:
    """
    RPC-based snapshot broadcast: queue to ring buffer + persist to SQLite for P2P sync.
    
    Architecture:
    - In-memory ring buffer (latest 1000) for fast /rpc polling
    - Async SQLite write to dm_pool table for P2P replication & miners
    - Dual persistence: Supabase (cloud) + SQLite (local mesh)
    """
    try:
        # 1. Update in-memory RPC cache
        _cache_snapshot(snapshot)
        
        # 2. Queue into DM snapshot ring buffer for streaming
        if 'density_matrix_hex' in snapshot:
            dm_snap = {
                'timestamp_ns': snapshot.get('timestamp_ns', int(time.time() * 1e9)),
                'oracle_id': snapshot.get('oracle_id'),
                'density_matrix_hex': snapshot.get('density_matrix_hex', ''),
                'purity': snapshot.get('purity'),
                'w_state_fidelity': snapshot.get('w_state_fidelity'),
                'von_neumann_entropy': snapshot.get('von_neumann_entropy'),
                'coherence_l1': snapshot.get('coherence_l1'),
                'coherence_renyi': snapshot.get('coherence_renyi'),
                'coherence_geometric': snapshot.get('coherence_geometric'),
                'quantum_discord': snapshot.get('quantum_discord'),
                'w_state_strength': snapshot.get('w_state_strength'),
                'phase_coherence': snapshot.get('phase_coherence'),
                'entanglement_witness': snapshot.get('entanglement_witness'),
                'trace_purity': snapshot.get('trace_purity'),
                'hlwe_signature': snapshot.get('hlwe_signature'),
                'signature_valid': snapshot.get('signature_valid', False),
                'oracle_address': snapshot.get('oracle_address'),
                'aer_noise_state': snapshot.get('aer_noise_state', {}),
                'measurement_counts': snapshot.get('measurement_counts', {}),
                'mermin_test': snapshot.get('mermin_test') or snapshot.get('bell_test'),  # Client expects mermin_test
                'bell_test': snapshot.get('bell_test'),  # Backward compatibility
                'lattice_refresh_counter': snapshot.get('lattice_refresh_counter'),
            }
            with _DM_SNAPSHOT_LOCK:
                _DM_SNAPSHOT_RING.append(dm_snap)
            logger.debug(f"[RPC-BROADCAST] ✅ DM snapshot queued (ring={len(_DM_SNAPSHOT_RING)}/1000)")
        
        # 3. Persist to SQLite asynchronously (P2P replication)
        def async_sqlite():
            try:
                import sqlite3
                from pathlib import Path
                db = Path.home() / "qtcl-miner" / "data" / "qtcl_blockchain.db"
                db.parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(str(db), timeout=5.0)
                cur = conn.cursor()
                cur.execute("""CREATE TABLE IF NOT EXISTS dm_pool (
                    id INTEGER PRIMARY KEY, timestamp_ns INTEGER, oracle_id INTEGER,
                    density_matrix_hex TEXT, purity REAL, w_state_fidelity REAL,
                    von_neumann_entropy REAL, coherence_l1 REAL, hlwe_signature TEXT,
                    signature_valid INTEGER, oracle_address TEXT, aer_noise_state TEXT,
                    measurement_counts TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
                cur.execute("""INSERT INTO dm_pool (
                    timestamp_ns, oracle_id, density_matrix_hex, purity, w_state_fidelity,
                    von_neumann_entropy, coherence_l1, hlwe_signature, signature_valid,
                    oracle_address, aer_noise_state, measurement_counts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                    snapshot.get('timestamp_ns'), snapshot.get('oracle_id'),
                    snapshot.get('density_matrix_hex', ''),
                    snapshot.get('purity'), snapshot.get('w_state_fidelity'),
                    snapshot.get('von_neumann_entropy'), snapshot.get('coherence_l1'),
                    json.dumps(snapshot.get('hlwe_signature')),
                    1 if snapshot.get('signature_valid') else 0,
                    snapshot.get('oracle_address'),
                    json.dumps(snapshot.get('aer_noise_state', {})),
                    json.dumps(snapshot.get('measurement_counts', {}))
                ))
                conn.commit()
                conn.close()
                logger.debug("[RPC-BROADCAST] ✅ DM → SQLite dm_pool")
            except Exception as e:
                logger.warning(f"[RPC-BROADCAST] SQLite write: {e}")
        
        threading.Thread(target=async_sqlite, daemon=True).start()
        
        # 4. Persist to Supabase (cloud backup, non-blocking)
        try:
            _persist_chirp_snapshot(snapshot)
        except Exception as e:
            logger.debug(f"[RPC-BROADCAST] Supabase skipped: {e}")
    except Exception as e:
        logger.error(f"[RPC-BROADCAST] Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════════════
# WSGI EXPORT FOR GUNICORN
# ═══════════════════════════════════════════════════════════════════════════════════════

# Gunicorn and wsgi_config.py require both 'app' and 'application' exports
application = app

