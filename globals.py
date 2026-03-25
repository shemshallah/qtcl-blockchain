#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                                ║
║  QTCL GLOBAL STATE MANAGEMENT v3.2 — QUANTUM-ONLY HARDENED                                   ║
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
    from pool_api import get_entropy_pool_manager, get_entropy, get_entropy_stats
    POOL_API_AVAILABLE = True
    logger.info("[GLOBALS] ✅ Pool API (Quantum Entropy) available")
except ImportError as e:
    POOL_API_AVAILABLE = False
    logger.warning(f"[GLOBALS] ⚠️  Pool API import failed: {e}")
    
    # NO FALLBACK TO os.urandom() — FAIL-FAST if called
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
        5: {'miner': 720, 'treasury': 80},
        6: {'miner': 760, 'treasury': 40},
        7: {'miner': 780, 'treasury': 20},
        8: {'miner': 790, 'treasury': 10},
    }

    TOTAL_PER_BLOCK_BASE: int = 800
    TOTAL_BLOCKS: int = 62_300_160
    TOTAL_SUPPLY_BASE: int = 62_300_160 * 800
    TOTAL_SUPPLY_QTCL: int = 498_401_280

    TREASURY_ADDRESS: str = 'qtcl110fc58e3c441106cc1e54ae41da5d15868525a87'
    TREASURY_PUBKEY: str = '627944e93fd7f406175393da145524cf73eb7b2ed12505cd81810a69b0c4d7ac'

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

    # HLWE Cryptography
    'hlwe_system': None,
    'hlwe_initialized': False,

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
    'POOL_API_AVAILABLE',
    'WSTATE_FIDELITY_THRESHOLD', 'ORACLE_MIN_PEERS', 'MAX_PEERS', 'MINING_COINBASE_REWARD',
    'LATTICE', 'get_lattice', 'set_lattice',
    'record_quantum_witness', 'register_validator', 'accept_attestation',
    'compute_finality',
    'TessellationRewardSchedule',
]
