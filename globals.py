#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                                ║
║  QTCL GLOBAL STATE MANAGEMENT v3.3 — HypΓ INTEGRATED                                   ║
║                                                                                                ║
║  Thread-safe global state with explicit quantum entropy initialization.                       ║
║  Fail-fast if entropy sources unavailable — QUANTUM REAL ONLY.                                ║
║  No silent fallbacks to os.urandom() in critical path.                                        ║
║                                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import os
import sys
import logging
import threading
import json
import hashlib
import time
import hmac
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

# Database
try:
    import psycopg2
    from psycopg2.pool import ThreadedConnectionPool
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

# ═════════════════════════════════════════════════════════════════════════════════════════
# LOGGING (MUST BE FIRST)
# ═════════════════════════════════════════════════════════════════════════════════════════

if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════════════════
# POOL API (QUANTUM ENTROPY) — QUANTUM-ONLY
# ═════════════════════════════════════════════════════════════════════════════════════════

try:
    from qrng_ensemble import get_entropy_pool_manager, get_entropy, get_entropy_stats
    POOL_API_AVAILABLE = True
    logger.info("[GLOBALS] ✅ Pool API (Quantum Entropy) available")
except ImportError as e:
    POOL_API_AVAILABLE = False
    logger.warning(f"[GLOBALS] ⚠️  Pool API import failed: {e}")
    
    def get_entropy(size=32):
        raise RuntimeError(
            "[GLOBALS] Quantum entropy pool unavailable. "
            "Cannot initialize system without quantum entropy sources."
        )
    
    def get_entropy_pool_manager():
        raise RuntimeError(
            "[GLOBALS] Quantum entropy pool unavailable. "
            "Cannot initialize system without quantum entropy sources."
        )
    
    def get_entropy_stats():
        return {'error': 'entropy pool unavailable'}


# ═════════════════════════════════════════════════════════════════════════════════════════
# HypΓ ENGINE — post-quantum lattice cryptography
# ═════════════════════════════════════════════════════════════════════════════════════════

try:
    # ✅ FIXED: Import from hlwe package
    from hlwe.hyp_engine import HypGammaEngine, get_hyp_engine
    HYP_ENGINE_AVAILABLE = True
    logger.info("[GLOBALS] ✅ HypΓ engine available")
except ImportError as e:
    HYP_ENGINE_AVAILABLE = False
    logger.warning(f"[GLOBALS] ⚠️  HypΓ engine import failed: {e}")

    def get_hyp_engine():
        raise RuntimeError(
            "[GLOBALS] HypΓ engine unavailable. "
            "Cannot perform post-quantum crypto without hyp_engine.py."
        )

# ═════════════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════════════════

ENTROPY_SIZE_BYTES = 32  # 256 bits
BLOCK_FIELD_ENTROPY_KEY = 'current_block_field_entropy'

DB_POOL_SIZE = int(os.getenv('DB_POOL_SIZE', '10'))
DB_MAX_CONNECTIONS = int(os.getenv('DB_MAX_CONN', '20'))

WSTATE_MODE = os.getenv('WSTATE_MODE', 'normal').lower()
WSTATE_STRICT_THRESHOLD = 0.80
WSTATE_NORMAL_THRESHOLD = 0.75
WSTATE_RELAXED_THRESHOLD = 0.70
WSTATE_FIDELITY_THRESHOLD = {
    'strict': WSTATE_STRICT_THRESHOLD,
    'normal': WSTATE_NORMAL_THRESHOLD,
    'relaxed': WSTATE_RELAXED_THRESHOLD,
}.get(WSTATE_MODE, WSTATE_NORMAL_THRESHOLD)

ORACLE_MIN_PEERS = int(os.getenv('ORACLE_MIN_PEERS', '3'))
ORACLE_CONSENSUS_THRESHOLD = 2/3

MAX_PEERS = int(os.getenv('MAX_PEERS', '32'))

# ═══════════════════════════════════════════════════════════════════════════════════
# 🔨 MINING DIFFICULTY
# ═══════════════════════════════════════════════════════════════════════════════════

BLOCK_DIFFICULTY = int(os.getenv('BLOCK_DIFFICULTY', '4'))  # Leading zeroes for PoW

# ═══════════════════════════════════════════════════════════════════════════════════
# 💎 TESSELLATION REWARD SCHEDULE — HARDCODED / IMMUTABLE
# ═══════════════════════════════════════════════════════════════════════════════════

class TessellationRewardSchedule:
    """QTCL CANONICAL REWARD SCHEDULE — DO NOT MODIFY"""

    DEPTH_BOUNDARIES: Dict[int, int] = {
        5: 106_496,
        6: 958_464,
        7: 7_774_208,
        8: 62_300_160,
    }

    REWARDS: Dict[int, Dict[str, int]] = {
        5: {'miner': 720, 'treasury': 80},   # 7.20 + 0.80 = 8.00 QTCL
        6: {'miner': 740, 'treasury': 60},   # 7.40 + 0.60 = 8.00 QTCL
        7: {'miner': 760, 'treasury': 40},   # 7.60 + 0.40 = 8.00 QTCL
        8: {'miner': 780, 'treasury': 20},   # 7.80 + 0.20 = 8.00 QTCL
    }

    TOTAL_PER_BLOCK_BASE: int = 800
    TOTAL_BLOCKS: int = 62_300_160
    TOTAL_SUPPLY_BASE: int = 62_300_160 * 800
    TOTAL_SUPPLY_QTCL: int = 498_401_280

    TREASURY_ADDRESS: str = os.environ.get('QTCL_TREASURY_ADDRESS', 'qtcl1f5080131c276070d09bd2cd8c4bea99d046663b1')
    TREASURY_PUBKEY: str = '814becc6ae20614ada0611f283048ec90a40ba791606608509365caf35e6948c5b4b3ecade0c249f28c2dfd1bb496fa3e859cd61397f4dfefc6b380ac7f7f388bf1fc351b63859d14a41f32c0a116a62ef09baab9830561506e4f94ac6b0bb819e67d152d58f27d56cee5619a0de16fd13ba5536397ffa9375562e8a78c6bc58e82ea1307af58c62266821ddfbec1aa4afb851386522a6245aff660395b8fbbac399efccbe7c7b55386bb75f7ddb68ea2140048b8e7f88b6c522456a19be4d3be111aaaaacc91f366998edede709e3966d19c81885ec8e3ceb9d34edffacfbc533e118a8b46ea40529562c27dc4b8abcca46fbae038c149a976837b4d647a3714c07b40683de88562c30620e04784ed2f7b3f8f4a849d3d882b75529fc0e048a098a05b026d0141f5e46b112d60303b2b5b212c660f80e1a819c8157a82710af5ecf95574cf06b7c77d4739579bef8391c7f6754fa7f0608dc942526c4cb67ddf8ee3f65b258cd3ca3c65e4be929be2ebee0bb7954812a1ab59e2872514d3085a91ffe33f312a536a4c44a9ece867c5a8a27ada301ef6b24386ac58a51f8eba0d9109ffe438929d83a7d37dc4d8c01fc25c964f99c8e1d686757f8365c7dbe5f2a2a23544eca497f8ad1e1be5e02ea01a24781064a298e412806a0b0da90e62f72f0823593f79e054539008d3e2c14a145e6f5ee64ad9568971d11534aaae1145601c82df65745a65d048975452d233ff0043bc6ab745fb58697c86d126982d88ac485589fdbc67e75e4f064a79dc1e18c0b566d156d2cd8077f39c0e66e36fdf94e8e0e890fef8700c88781fed930be2fe7cfce94f985076704d405942821001d141243dc2f3c453ae55c7844787eb747d3f690b8955b2cbc7751d526b4217db76ffb2742091913e8e2beb6bcdefcc9a3eae5f27fb434eba3bdef91f89960e91dd5c6d2a96117c8894b95162173167a389c78957b388b787a0372cf9ef91065d3473c86529c9e434459926ba68a0040429dc8659594aee8944e8c0a9352fb4d0ff683cfed622f2983ae9c0026d7fd2c327dc4314e1a7bd6bf782d818a96640a2ce94192ba2d286e9af626aefe06e27c763928b03ae2a43434e587f24ff6841009b1641db9cfbe925de802fd93c062ca305bd4ac289c0a8b47bb91edd8788092259db797b2df41069782d5de389754ac667d09f8dfcc9e1744b4174cfbde41867e6831e15dab18ef66f93b2d7ea11d3bc1190e4cdcfbae5b9d929fb613976a21848dee1e970b09c8b05481d225197241587aa370f4f43d35ccf92ae3e910232a5776a67320126a782125e21fd8de873060c3d6d90494efd78e5fee2d6285f0c9af3f11934bcd85b2ce133e85578e3e157ace2fb12623aa6c091581887a792cfb6d5dcaa1f7b4b5caac1a4bb90236476df791edd9382788b5270981c1befa8875'

    @classmethod
    def get_depth_for_height(cls, height: int) -> int:
        """Return tessellation depth for a given block height."""
        for depth, boundary in cls.DEPTH_BOUNDARIES.items():
            if height < boundary:
                return depth
        return 8

    @classmethod
    def get_rewards_for_height(cls, height: int) -> Dict[str, int]:
        """Return {'miner': <base_units>, 'treasury': <base_units>}."""
        return dict(cls.REWARDS[cls.get_depth_for_height(height)])

    @classmethod
    def get_miner_reward_base(cls, height: int) -> int:
        """Miner reward in base units for block height."""
        return cls.REWARDS[cls.get_depth_for_height(height)]['miner']

    @classmethod
    def get_treasury_reward_base(cls, height: int) -> int:
        """Treasury reward in base units for block height."""
        return cls.REWARDS[cls.get_depth_for_height(height)]['treasury']

    @classmethod
    def get_miner_reward_qtcl(cls, height: int) -> float:
        """Miner reward in QTCL — display only."""
        return cls.get_miner_reward_base(height) / 100.0

    @classmethod
    def get_treasury_reward_qtcl(cls, height: int) -> float:
        """Treasury reward in QTCL — display only."""
        return cls.get_treasury_reward_base(height) / 100.0

    @classmethod
    def get_supply_breakdown(cls) -> Dict[str, Any]:
        """Full supply breakdown."""
        breakdown: Dict[str, Any] = {}
        total_miner_base = 0
        total_treasury_base = 0
        prev = 0
        for depth, boundary in cls.DEPTH_BOUNDARIES.items():
            blocks = boundary - prev
            r = cls.REWARDS[depth]
            m = blocks * r['miner']
            t = blocks * r['treasury']
            total_miner_base += m
            total_treasury_base += t
            breakdown[depth] = {
                'blocks': blocks,
                'miner_reward_qtcl': r['miner'] / 100.0,
                'treasury_reward_qtcl': r['treasury'] / 100.0,
                'total_per_block_qtcl': (r['miner'] + r['treasury']) / 100.0,
                'total_depth_miner_qtcl': m / 100.0,
                'total_depth_treasury_qtcl': t / 100.0,
            }
            prev = boundary
        breakdown['TOTAL'] = {
            'total_miner_qtcl': total_miner_base / 100.0,
            'total_treasury_qtcl': total_treasury_base / 100.0,
            'total_supply_qtcl': (total_miner_base + total_treasury_base) / 100.0,
        }
        return breakdown


MINING_COINBASE_REWARD = 720  # depth-5 miner base units

# Canonical null-sink burn address for oracle registration TXs.
# No private key exists for this address — it is a pure chain commitment.
# Mirrors the value previously defined in blockchain_entropy_mining.py.
ORACLE_REGISTRY_ADDRESS = "qtcl1oracle_registry_000000000000000000000000"

# ═════════════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ═════════════════════════════════════════════════════════════════════════════════════

_GLOBAL_STATE = {
    'initialized': False,
    '_initializing': False,
    'lock': threading.RLock(),

    # Database
    'db_pool': None,
    'db_url': None,
    'db_initialized': False,

    # Current Block Field Entropy
    'current_block_field': None,
    'block_field_entropy': None,
    'block_field_initialized': False,
    'entropy_stats': {},
    'entropy_efficiency': 0.0,

    # HypΓ Cryptography
    'hyp_engine': None,
    'hyp_initialized': False,

    # Quantum Lattice
    'lattice_controller': None,
    'heartbeat': None,
    'lattice_initialized': False,

    # Consensus
    'validators': {},
    'finality_depth': 32,
    'finality_threshold': 2/3,
    'finality_checkpoints': {},
    'quantum_witnesses': {},
    'pending_attestations': [],
    'current_epoch': 0,
    'slots_per_epoch': 256,

    # Server metadata
    'startup_time': datetime.now(timezone.utc).isoformat(),
    'server_version': '3.0',
    'integration': 'pool_api_quantum_hardened',
}

_STATE_LOCK = threading.RLock()

# ─────────────────────────────────────────────────────────────────────────────
# STATE ACCESSORS
# ─────────────────────────────────────────────────────────────────────────────

def get_state(key: str, default: Any = None) -> Any:
    """Thread-safe getter for global state"""
    with _STATE_LOCK:
        return _GLOBAL_STATE.get(key, default)


def set_state(key: str, value: Any) -> None:
    """Thread-safe setter for global state"""
    with _STATE_LOCK:
        _GLOBAL_STATE[key] = value
        logger.debug(f"[STATE] Set {key} = {type(value).__name__}")


def update_state(updates: Dict[str, Any]) -> None:
    """Batch update global state atomically"""
    with _STATE_LOCK:
        _GLOBAL_STATE.update(updates)
        logger.debug(f"[STATE] Batch update: {len(updates)} keys")

# ─────────────────────────────────────────────────────────────────────────────
# ENTROPY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_entropy_from_block_field() -> bytes:
    """Get entropy from quantum block field"""
    return get_entropy(ENTROPY_SIZE_BYTES)


def get_block_field_entropy() -> bytes:
    """Get current block field entropy from quantum pool"""
    entropy = get_state('block_field_entropy')
    if entropy:
        return entropy
    return get_entropy_from_block_field()

def get_current_block_field() -> Dict[str, Any]:
    """
    Return the current block field state dict.

    Block field = the live quantum entropy snapshot used as mining seed.
    Stored under 'current_block_field' in _GLOBAL_STATE.
    Returns {} if not yet initialised — callers must guard on empty dict.
    """
    with _STATE_LOCK:
        field = _GLOBAL_STATE.get('current_block_field')
    return field if isinstance(field, dict) else {}


def set_current_block_field(block_data: Dict[str, Any]) -> None:
    """
    Persist a new block field snapshot into global state and refresh the
    cached block_field_entropy from it.

    Called by the lattice controller and oracle after each measurement cycle
    so that miners always have a fresh quantum seed without a round-trip to
    pool_api on every nonce.

    Args:
        block_data: dict containing at minimum {'entropy': bytes | str, ...}
                    If 'entropy' is a hex string it is decoded automatically.
    """
    with _STATE_LOCK:
        _GLOBAL_STATE['current_block_field'] = block_data
        _GLOBAL_STATE['block_field_initialized'] = True

    # Refresh the cached entropy bytes from this block field
    raw = block_data.get('entropy') if isinstance(block_data, dict) else None
    if isinstance(raw, str):
        try:
            raw = bytes.fromhex(raw)
        except ValueError:
            raw = None
    if isinstance(raw, bytes) and len(raw) >= 16:
        set_state('block_field_entropy', raw[:ENTROPY_SIZE_BYTES])
        logger.debug(f"[GLOBALS] Block field entropy refreshed ({len(raw)} bytes)")
    else:
        # Fall back to pool_api for a fresh pull
        try:
            fresh = get_entropy(ENTROPY_SIZE_BYTES)
            set_state('block_field_entropy', fresh)
            logger.debug("[GLOBALS] Block field entropy refreshed via pool_api fallback")
        except Exception as e:
            logger.warning(f"[GLOBALS] Block field entropy refresh failed: {e}")


def initialize_block_field_entropy() -> bool:
    """
    Bootstrap the block field entropy subsystem on server startup.

    Pulls 32 bytes from the QRNG pool and seeds _GLOBAL_STATE so that
    get_block_field_entropy() returns real quantum entropy immediately
    instead of waiting for the first lattice measurement cycle.

    Returns True on success, False if pool_api is unavailable.
    Called by server.py during WSGI startup (inside try/except ImportError).
    """
    if not POOL_API_AVAILABLE:
        logger.warning("[GLOBALS] initialize_block_field_entropy: pool_api unavailable")
        return False
    try:
        seed = get_entropy(ENTROPY_SIZE_BYTES)
        with _STATE_LOCK:
            _GLOBAL_STATE['block_field_entropy'] = seed
            _GLOBAL_STATE['block_field_initialized'] = True
            if _GLOBAL_STATE.get('current_block_field') is None:
                _GLOBAL_STATE['current_block_field'] = {
                    'entropy': seed.hex(),
                    'source': 'pool_api_bootstrap',
                    'timestamp': time.time(),
                }
        logger.info(
            f"[GLOBALS] \u2705 Block field entropy initialised: {seed.hex()[:16]}\u2026"
        )
        return True
    except Exception as e:
        logger.error(f"[GLOBALS] initialize_block_field_entropy failed: {e}")
        return False


def initialize_system() -> bool:
    """
    Top-level system initialisation gate called by server.py on startup
    (imported as `initialize_system as init_entropy_system`).

    Runs in order:
      1. initialize_block_field_entropy()  — seed quantum entropy cache
      2. entropy_stats snapshot            — warm the stats cache

    Returns True when the entropy subsystem is ready for mining.
    Safe to call multiple times — idempotent via _initializing guard.
    """
    with _STATE_LOCK:
        if _GLOBAL_STATE.get('initialized'):
            return True
        if _GLOBAL_STATE.get('_initializing'):
            return False          # concurrent call — let the first one win
        _GLOBAL_STATE['_initializing'] = True

    try:
        ok = initialize_block_field_entropy()
        try:
            stats = get_entropy_stats()
            set_state('entropy_stats', stats)
        except Exception:
            pass
        with _STATE_LOCK:
            _GLOBAL_STATE['initialized'] = ok
            _GLOBAL_STATE['_initializing'] = False
        if ok:
            logger.info("[GLOBALS] \u2705 System initialised \u2014 quantum entropy subsystem ready")
        else:
            logger.warning("[GLOBALS] \u26a0\ufe0f  System initialised in degraded mode (no quantum entropy)")
        return ok
    except Exception as e:
        logger.error(f"[GLOBALS] initialize_system failed: {e}")
        with _STATE_LOCK:
            _GLOBAL_STATE['_initializing'] = False
        return False


# ─── In-memory blockchain index (lightweight, DB-authoritative) ──────────────
# server.py calls get_blockchain() from inside _rpc_getTransaction to do a
# fast in-process lookup before falling back to the DB cursor.  The object
# returned must have a get_transaction(tx_hash: str) method.

class _InMemoryBlockchainIndex:
    """
    Lightweight in-process index of recently confirmed transactions.

    NOT a full chain implementation — that lives in PostgreSQL.
    This index is populated by _rpc_submitBlock as blocks are accepted
    and used for O(1) tx lookups without a DB round-trip.

    Thread-safe via RLock.  Bounded to MAX_CACHED_TXS entries (LRU eviction).
    """

    MAX_CACHED_TXS = 50_000

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tx_index: Dict[str, Dict[str, Any]] = {}   # tx_hash -> tx dict

    def index_block(self, block_height: int, transactions: list) -> None:
        """Index all transactions from a newly accepted block."""
        with self._lock:
            for tx in transactions:
                tx_hash = tx.get('tx_hash') or tx.get('tx_id', '')
                if tx_hash:
                    self._tx_index[tx_hash] = dict(tx, block_height=block_height)
            # Evict oldest entries when over limit
            if len(self._tx_index) > self.MAX_CACHED_TXS:
                evict_n = len(self._tx_index) - self.MAX_CACHED_TXS
                for old_key in list(self._tx_index.keys())[:evict_n]:
                    del self._tx_index[old_key]

    def get_transaction(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        """Return cached tx dict or None if not in index (caller falls back to DB)."""
        with self._lock:
            return self._tx_index.get(tx_hash)

    def __len__(self) -> int:
        with self._lock:
            return len(self._tx_index)


_BLOCKCHAIN_INDEX: Optional['_InMemoryBlockchainIndex'] = None
_BLOCKCHAIN_LOCK = threading.Lock()


def get_blockchain() -> '_InMemoryBlockchainIndex':
    """
    Return the singleton in-memory blockchain index.

    Called by server.py _rpc_getTransaction for fast tx lookup.
    Creates the singleton on first call (lazy, thread-safe).
    """
    global _BLOCKCHAIN_INDEX
    if _BLOCKCHAIN_INDEX is None:
        with _BLOCKCHAIN_LOCK:
            if _BLOCKCHAIN_INDEX is None:
                _BLOCKCHAIN_INDEX = _InMemoryBlockchainIndex()
                logger.info("[GLOBALS] In-memory blockchain index created")
    return _BLOCKCHAIN_INDEX

# ─────────────────────────────────────────────────────────────────────────────
# LATTICE SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

LATTICE = None  # type: Optional[Any]
_LATTICE_LOCK = threading.Lock()


def get_lattice() -> Optional[Any]:
    """Return the global LatticeController (None until server sets it)."""
    return LATTICE


def set_lattice(lattice_instance) -> None:
    """Called by server.py once LatticeController is fully initialised."""
    global LATTICE
    with _LATTICE_LOCK:
        LATTICE = lattice_instance
    set_state('lattice_controller', lattice_instance)
    logger.info(f"[GLOBALS] LATTICE singleton set: {type(lattice_instance).__name__}")



# ─────────────────────────────────────────────────────────────────────────────
# HypΓ ENGINE SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

_HYP_ENGINE_INSTANCE = None
_HYP_ENGINE_LOCK = threading.Lock()


def get_global_hyp_engine():
    """Return the global HypGammaEngine singleton, initializing on first call."""
    global _HYP_ENGINE_INSTANCE
    if _HYP_ENGINE_INSTANCE is None:
        with _HYP_ENGINE_LOCK:
            if _HYP_ENGINE_INSTANCE is None:
                if HYP_ENGINE_AVAILABLE:
                    try:
                        _HYP_ENGINE_INSTANCE = get_hyp_engine()
                        set_state('hyp_engine', _HYP_ENGINE_INSTANCE)
                        set_state('hyp_initialized', True)
                        logger.info("[GLOBALS] ✅ HypΓ engine singleton created")
                    except Exception as exc:
                        logger.error(f"[GLOBALS] HypΓ engine init failed: {exc}")
                else:
                    logger.warning("[GLOBALS] HypΓ engine not available")
    return _HYP_ENGINE_INSTANCE


def set_hyp_engine(engine_instance) -> None:
    """Called by server.py to inject a pre-initialized HypGammaEngine."""
    global _HYP_ENGINE_INSTANCE
    with _HYP_ENGINE_LOCK:
        _HYP_ENGINE_INSTANCE = engine_instance
    set_state('hyp_engine', engine_instance)
    set_state('hyp_initialized', engine_instance is not None)
    logger.info(f"[GLOBALS] HypΓ engine set: {type(engine_instance).__name__}")


# ─────────────────────────────────────────────────────────────────────────────
# CONSENSUS STATE
# ─────────────────────────────────────────────────────────────────────────────

def record_quantum_witness(block_height: int, block_hash: str,
                            w_state_fidelity: float, timestamp_ns: int) -> bool:
    """Record a W-state quantum witness for a block."""
    with _STATE_LOCK:
        _GLOBAL_STATE['quantum_witnesses'][block_height] = {
            'hash': block_hash,
            'fidelity': float(w_state_fidelity),
            'timestamp_ns': int(timestamp_ns),
        }
    logger.debug(f"[GLOBALS] Quantum witness recorded: h={block_height} F={w_state_fidelity:.4f}")
    return True


def register_validator(pubkey: str, balance: int,
                        address: str = '') -> tuple:
    """Register a validator. Returns (success, validator_index)."""
    with _STATE_LOCK:
        validators = _GLOBAL_STATE['validators']
        idx = len(validators)
        validators[idx] = {
            'pubkey': pubkey,
            'address': address,
            'balance': int(balance),
            'status': 'pending',
            'activation_epoch': _GLOBAL_STATE.get('current_epoch', 0) + 4,
        }
    logger.info(f"[GLOBALS] Validator registered: idx={idx} pubkey={pubkey[:16]}…")
    return True, idx


def accept_attestation(validator_index: int, slot: int,
                        beacon_block_root: str, source_epoch: int,
                        target_epoch: int, signature: str) -> tuple:
    """Accept a validator attestation. Returns (success, reason)."""
    attestation = {
        'validator_index': int(validator_index),
        'slot': int(slot),
        'beacon_block_root': beacon_block_root,
        'source_epoch': int(source_epoch),
        'target_epoch': int(target_epoch),
        'signature': signature,
        'received_at': time.time(),
    }
    with _STATE_LOCK:
        pending = _GLOBAL_STATE.setdefault('pending_attestations', [])
        pending.append(attestation)
        if len(pending) > 1024:
            _GLOBAL_STATE['pending_attestations'] = pending[-1024:]
    logger.debug(f"[GLOBALS] Attestation accepted: val={validator_index} slot={slot}")
    return True, "attestation_accepted"


def compute_finality(check_height: int) -> Optional[int]:
    """Check if a block at check_height has reached finality (stub)."""
    with _STATE_LOCK:
        witnesses = _GLOBAL_STATE.get('quantum_witnesses', {})
        if check_height in witnesses:
            return check_height
    return None


# Export public symbols
__all__ = [
    'get_state', 'set_state', 'update_state',
    'get_entropy', 'get_entropy_stats', 'get_block_field_entropy',
    'get_entropy_from_block_field',
    'get_current_block_field', 'set_current_block_field',
    'initialize_block_field_entropy', 'initialize_system',
    'get_blockchain',
    'POOL_API_AVAILABLE',
    'WSTATE_FIDELITY_THRESHOLD', 'ORACLE_MIN_PEERS', 'MAX_PEERS', 'MINING_COINBASE_REWARD', 'ORACLE_REGISTRY_ADDRESS',
    'LATTICE', 'get_lattice', 'set_lattice',
    'record_quantum_witness', 'register_validator', 'accept_attestation',
    'compute_finality',
    'TessellationRewardSchedule',
    'HYP_ENGINE_AVAILABLE', 'get_global_hyp_engine', 'set_hyp_engine',
]
