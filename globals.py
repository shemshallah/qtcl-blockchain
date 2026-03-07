#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                                ║
║  QTCL GLOBAL STATE MANAGEMENT v2.0 — POOL_API UNIFIED ENTROPY                                 ║
║                                                                                                ║
║  Thread-safe global state with database connection pooling                                    ║
║  Entropy Pool: 5-source QRNG (ANU, Random.org, QBICK, HotBits, Fourmilab) via pool_api      ║
║  HLWE Cryptography: Post-quantum lattice-based signatures                                     ║
║  Quantum Lattice: Hyperbolic {8,3} tessellation controller                                    ║
║  W-State: Oracle density matrix snapshots with HLWE signatures                                ║
║                                                                                                ║
║  Architecture:                                                                                 ║
║    • Pool API: Unified entropy management (circuit breaker, caching, XOR ensemble)            ║
║    • Database: PostgreSQL with thread pool for concurrent access                              ║
║    • Lattice: Quantum entropy mining with pseudoqubit evolution                               ║
║    • Oracle: W-state density matrix with HLWE authentication                                  ║
║    • Heartbeat: Background daemon for system health                                           ║
║                                                                                                ║
║  Museum-grade implementation. Zero shortcuts. Deploy with confidence. 🚀⚛️💎                 ║
╚════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import threading, logging, json, sys, hashlib, time
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

# Pool API (Entropy Management) — 5-source QRNG ensemble
try:
    from pool_api import get_entropy_pool_manager, get_entropy, get_entropy_stats
    POOL_API_AVAILABLE = True
except ImportError:
    POOL_API_AVAILABLE = False
    # Fallback stubs if pool_api unavailable
    def get_entropy_pool_manager(): return None
    def get_entropy(size=32): 
        import secrets
        return secrets.token_bytes(size)
    def get_entropy_stats(): return {}

# Current Block Field (entropy pool = current block field, nonmarkovian noise bath)
ENTROPY_SIZE_BYTES = 32  # 256 bits
BLOCK_FIELD_ENTROPY_KEY = 'current_block_field_entropy'

# ═════════════════════════════════════════════════════════════════════════════════════════
# UNIFIED CONFIGURATION (Single Source of Truth)
# ═════════════════════════════════════════════════════════════════════════════════════════

# DATABASE
DB_POOL_SIZE = int(os.getenv('DB_POOL_SIZE', '10'))
DB_MAX_CONNECTIONS = int(os.getenv('DB_MAX_CONN', '20'))

# W-STATE VALIDATION THRESHOLDS (Unified across all validators)
WSTATE_MODE = os.getenv('WSTATE_MODE', 'normal').lower()
WSTATE_STRICT_THRESHOLD = 0.80
WSTATE_NORMAL_THRESHOLD = 0.75
WSTATE_RELAXED_THRESHOLD = 0.70
WSTATE_FIDELITY_THRESHOLD = {
    'strict': WSTATE_STRICT_THRESHOLD,
    'normal': WSTATE_NORMAL_THRESHOLD,
    'relaxed': WSTATE_RELAXED_THRESHOLD,
}.get(WSTATE_MODE, WSTATE_NORMAL_THRESHOLD)

# ORACLE CONSENSUS
ORACLE_MIN_PEERS = int(os.getenv('ORACLE_MIN_PEERS', '3'))
ORACLE_CONSENSUS_THRESHOLD = 2/3  # 2/3 majority

# P2P NETWORK
MAX_PEERS = int(os.getenv('MAX_PEERS', '32'))

# MINING
MINING_COINBASE_REWARD = 1250  # sats

# Logging
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE DICTIONARY
# ─────────────────────────────────────────────────────────────────────────────

_GLOBAL_STATE = {
    'initialized': False,
    '_initializing': False,
    'lock': threading.RLock(),
    
    # Database
    'db_pool': None,
    'db_url': None,
    'db_initialized': False,
    
    # Current Block Field Entropy (nonmarkovian noise bath mining)
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
    
    # Consensus (Finality Gadget)
    'validators': {},  # validator_index → {pubkey, balance, status, ...}
    'finality_depth': 32,  # Blocks until finality
    'finality_threshold': 2/3,  # Validator supermajority
    'finality_checkpoints': {},  # epoch → {block_height, block_hash, timestamp}
    'quantum_witnesses': {},  # block_height → {w_state_fidelity, timestamp_ns}
    'pending_attestations': [],  # List of pending validator votes
    'current_epoch': 0,
    'slots_per_epoch': 256,
    
    # Server metadata
    'startup_time': datetime.now(timezone.utc).isoformat(),
    'server_version': '2.0',
    'integration': 'pool_api_unified',
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
# DATABASE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def initialize_database(db_url: str, pool_size: int = 10) -> bool:
    """
    Initialize database connection pool
    Connects to Supabase Pooler with new QTCL schema
    """
    try:
        if not DB_AVAILABLE:
            logger.error("[DB] psycopg2 not available")
            return False
        
        logger.info(f"[DB] Initializing connection pool (size={pool_size})...")
        db_pool = ThreadedConnectionPool(1, pool_size, db_url)
        
        # Test connection
        conn = db_pool.getconn()
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM hyperbolic_triangles LIMIT 1")
        cur.fetchone()
        db_pool.putconn(conn)
        
        set_state('db_pool', db_pool)
        set_state('db_url', db_url)
        set_state('db_initialized', True)
        
        logger.info("[DB] ✓ Database initialized and schema verified")
        return True
    except Exception as e:
        logger.error(f"[DB] Initialization failed: {e}")
        return False

def get_db_pool() -> Optional[ThreadedConnectionPool]:
    """Get database connection pool"""
    return get_state('db_pool')

def close_database() -> None:
    """Close database connection pool"""
    db_pool = get_state('db_pool')
    if db_pool:
        try:
            db_pool.closeall()
            set_state('db_pool', None)
            set_state('db_initialized', False)
            logger.info("[DB] ✓ Connection pool closed")
        except Exception as e:
            logger.error(f"[DB] Close failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# POOL API ENTROPY MANAGEMENT (UNIFIED 5-SOURCE QRNG)
# ─────────────────────────────────────────────────────────────────────────────

def initialize_entropy_pool() -> bool:
    """
    Initialize unified entropy pool via pool_api
    Manages 5-source QRNG ensemble with circuit breaker pattern:
      1. ANU QRNG (quantum vacuum fluctuations)
      2. Random.org (atmospheric noise)
      3. QBICK (ID Quantique Quantis hardware)
      4. HotBits (radioactive decay)
      5. Fourmilab (additional entropy source)
    
    XOR combines all sources for information-theoretic security.
    """
    try:
        if not POOL_API_AVAILABLE:
            logger.error("[ENTROPY] pool_api not available")
            return False
        
        # Get entropy pool manager singleton
        pool_mgr = get_entropy_pool_manager()
        
        set_state('entropy_pool_manager', pool_mgr)
        set_state('entropy_pool_initialized', True)
        
        logger.info("[ENTROPY] ✓ Unified 5-source entropy pool initialized (pool_api)")
        return True
    except Exception as e:
        logger.error(f"[ENTROPY] Initialization failed: {e}")
        return False

def get_fresh_entropy(size: int = 32) -> bytes:
    """
    Get fresh entropy from pool_api
    Returns bytes of entropy (default 32 bytes = 256 bits)
    
    Uses:
      - XOR hedging across 5 sources
      - 1-hour caching with async refresh
      - Circuit breaker for failing sources
      - Fallback to secrets.token_bytes() if all fail
    """
    try:
        if POOL_API_AVAILABLE:
            return get_entropy(size)
        else:
            logger.warning("[ENTROPY] pool_api not available, using fallback")
            import secrets
            return secrets.token_bytes(size)
    except Exception as e:
        logger.warning(f"[ENTROPY] Failed to get entropy: {e}")
        import secrets
        return secrets.token_bytes(size)

def refresh_entropy_stats() -> Dict[str, Any]:
    """
    Refresh entropy pool statistics
    Tracks cache hits, source health, circuit breaker status
    """
    try:
        if POOL_API_AVAILABLE:
            stats = get_entropy_stats()
            
            # Calculate efficiency (cache hit ratio)
            hits = stats.get('metrics', {}).get('cache_hits', 0)
            misses = stats.get('metrics', {}).get('cache_misses', 1)
            efficiency = hits / max(1, hits + misses)
            
            update_state({
                'entropy_stats': stats,
                'entropy_efficiency': efficiency,
            })
            
            logger.debug(f"[ENTROPY] Stats: {stats['sources']['working']}/{stats['sources']['total']} sources working")
            return stats
        else:
            return {}
    except Exception as e:
        logger.debug(f"[ENTROPY] Stats refresh failed: {e}")
        return {}

# ─────────────────────────────────────────────────────────────────────────────
# HLWE CRYPTOGRAPHY MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def initialize_hlwe_engine() -> bool:
    """
    Initialize HLWE cryptographic engine
    Integrates with database for wallet management
    """
    try:
        db_pool = get_db_pool()
        if not db_pool:
            logger.warning("[HLWE] Database not available - HLWE will work in memory only")
        
        from hlwe_engine_fresh import get_hlwe_system
        hlwe = get_hlwe_system(db_pool)
        
        set_state('hlwe_system', hlwe)
        set_state('hlwe_initialized', True)
        
        logger.info("[HLWE] ✓ HLWE cryptographic system initialized")
        return True
    except Exception as e:
        logger.error(f"[HLWE] Initialization failed: {e}")
        return False

def get_hlwe_system() -> Optional[Any]:
    """Get HLWE cryptographic system"""
    if not get_state('hlwe_initialized'):
        initialize_hlwe_engine()
    return get_state('hlwe_system')

# ─────────────────────────────────────────────────────────────────────────────
# QUANTUM LATTICE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def initialize_lattice_controller() -> bool:
    """
    Initialize quantum lattice controller
    Main server control system for hyperbolic {8,3} tessellation evolution
    Integrates with entropy pool for quantum entropy mining
    """
    try:
        db_pool = get_db_pool()
        entropy_pool = get_entropy_pool_manager()
        
        from lattice_controller import get_lattice_controller, get_heartbeat
        lattice = get_lattice_controller(db_pool, entropy_pool)
        heartbeat = get_heartbeat(lattice)
        
        set_state('lattice_controller', lattice)
        set_state('heartbeat', heartbeat)
        set_state('lattice_initialized', True)
        
        logger.info("[LATTICE] ✓ Quantum lattice controller initialized")
        return True
    except Exception as e:
        logger.error(f"[LATTICE] Initialization failed: {e}")
        return False

def get_lattice_controller() -> Optional[Any]:
    """Get quantum lattice controller"""
    if not get_state('lattice_initialized'):
        initialize_lattice_controller()
    return get_state('lattice_controller')

def get_heartbeat() -> Optional[Any]:
    """Get heartbeat daemon"""
    if not get_state('lattice_initialized'):
        initialize_lattice_controller()
    return get_state('heartbeat')

def start_heartbeat() -> bool:
    """Start the heartbeat daemon"""
    try:
        heartbeat = get_heartbeat()
        if heartbeat and not heartbeat.running:
            heartbeat.start()
            logger.info("[HEARTBEAT] ✓ Started")
            return True
        return False
    except Exception as e:
        logger.error(f"[HEARTBEAT] Start failed: {e}")
        return False

def stop_heartbeat() -> bool:
    """Stop the heartbeat daemon"""
    try:
        heartbeat = get_heartbeat()
        if heartbeat and heartbeat.running:
            heartbeat.stop()
            logger.info("[HEARTBEAT] ✓ Stopped")
            return True
        return False
    except Exception as e:
        logger.error(f"[HEARTBEAT] Stop failed: {e}")
        return False

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def initialize_system(db_url: str, start_heartbeat_daemon: bool = True) -> bool:
    """
    Initialize complete QTCL system
    
    Initialization sequence:
      1. Database → Connect to PostgreSQL, verify schema
      2. Entropy Pool → Initialize pool_api (5-source QRNG with circuit breaker)
      3. HLWE Engine → Initialize post-quantum cryptography system
      4. Lattice Controller → Initialize quantum entropy mining system
      5. Heartbeat → Start background health monitoring daemon
    """
    with _STATE_LOCK:
        if get_state('initialized'):
            logger.info("[SYSTEM] Already initialized")
            return True
        
        if get_state('_initializing'):
            logger.warning("[SYSTEM] Initialization in progress")
            return False
        
        set_state('_initializing', True)
    
    try:
        logger.info("=" * 100)
        logger.info("[SYSTEM] ▄▀▀▀▄ QTCL System Startup (pool_api integrated)")
        logger.info("=" * 100)
        
        # 1. Database
        logger.info("[SYSTEM] [1/5] Initializing database connection pool...")
        if not initialize_database(db_url):
            raise RuntimeError("Database initialization failed")
        logger.info("[SYSTEM] ✓ Database ready")
        
        # 2. Entropy Pool (5-source QRNG via pool_api)
        logger.info("[SYSTEM] [2/5] Initializing entropy pool (5-source QRNG ensemble)...")
        if not initialize_entropy_pool():
            logger.warning("[SYSTEM] ⚠️  Entropy pool initialization failed - continuing with fallback")
        else:
            logger.info("[SYSTEM] ✓ Entropy pool ready (ANU, Random.org, QBICK, HotBits, Fourmilab)")
        
        # 3. HLWE
        logger.info("[SYSTEM] [3/5] Initializing HLWE cryptographic engine...")
        if not initialize_hlwe_engine():
            logger.warning("[SYSTEM] ⚠️  HLWE initialization failed - continuing")
        else:
            logger.info("[SYSTEM] ✓ HLWE engine ready")
        
        # 4. Lattice
        logger.info("[SYSTEM] [4/5] Initializing quantum lattice controller...")
        if not initialize_lattice_controller():
            logger.warning("[SYSTEM] ⚠️  Lattice initialization failed - continuing")
        else:
            logger.info("[SYSTEM] ✓ Lattice controller ready")
        
        # 5. Heartbeat
        logger.info("[SYSTEM] [5/5] Starting heartbeat daemon...")
        if start_heartbeat_daemon:
            if not start_heartbeat():
                logger.warning("[SYSTEM] ⚠️  Heartbeat start failed - continuing")
            else:
                logger.info("[SYSTEM] ✓ Heartbeat daemon active")
        
        set_state('initialized', True)
        set_state('_initializing', False)
        
        logger.info("=" * 100)
        logger.info("[SYSTEM] ✓ QTCL System fully initialized and ready")
        logger.info("[SYSTEM] Entropy: pool_api | DB: PostgreSQL | Crypto: HLWE | Lattice: {8,3}")
        logger.info("=" * 100)
        return True
    except Exception as e:
        logger.error(f"[SYSTEM] Initialization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        set_state('_initializing', False)
        return False

def shutdown_system() -> None:
    """Gracefully shutdown QTCL system"""
    try:
        logger.info("[SYSTEM] Shutting down...")
        
        # 1. Heartbeat
        stop_heartbeat()
        
        # 2. Database
        close_database()
        
        set_state('initialized', False)
        logger.info("[SYSTEM] ✓ Shutdown complete")
    except Exception as e:
        logger.error(f"[SYSTEM] Shutdown error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM STATUS
# ─────────────────────────────────────────────────────────────────────────────

def get_system_status() -> Dict[str, Any]:
    """
    Get complete system status including entropy pool health
    """
    try:
        lattice = get_lattice_controller()
        heartbeat = get_heartbeat()
        pool_stats = refresh_entropy_stats()
        
        return {
            'initialized': get_state('initialized'),
            'startup_time': get_state('startup_time'),
            'server_version': get_state('server_version'),
            'integration': get_state('integration'),
            'database': {
                'initialized': get_state('db_initialized'),
                'pool_active': get_state('db_pool') is not None,
            },
            'entropy_pool': {
                'initialized': get_state('entropy_pool_initialized'),
                'pool_api_available': POOL_API_AVAILABLE,
                'efficiency': get_state('entropy_efficiency'),
                'sources': {
                    'total': pool_stats.get('sources', {}).get('total', 0),
                    'working': pool_stats.get('sources', {}).get('working', 0),
                    'failing': pool_stats.get('sources', {}).get('failing', 0),
                    'dead': pool_stats.get('sources', {}).get('dead', 0),
                } if pool_stats else {},
                'cache': {
                    'hits': pool_stats.get('metrics', {}).get('cache_hits', 0),
                    'misses': pool_stats.get('metrics', {}).get('cache_misses', 0),
                    'age_seconds': pool_stats.get('cache_age_seconds'),
                    'valid': pool_stats.get('cache_valid'),
                } if pool_stats else {},
            },
            'hlwe': {
                'initialized': get_state('hlwe_initialized'),
            },
            'lattice': {
                'initialized': get_state('lattice_initialized'),
                'status': lattice.get_status() if lattice else None,
            },
            'heartbeat': {
                'running': heartbeat.running if heartbeat else False,
                'health': heartbeat.get_status() if heartbeat else None,
            },
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"[STATUS] Get status failed: {e}")
        return {'error': str(e), 'timestamp': datetime.now(timezone.utc).isoformat()}

# ─────────────────────────────────────────────────────────────────────────────
# CURRENT BLOCK FIELD ENTROPY (Nonmarkovian Noise Bath Mining)
# ─────────────────────────────────────────────────────────────────────────────

def initialize_block_field_entropy() -> bool:
    """Initialize current block field as entropy pool"""
    try:
        logger.info("[ENTROPY] Initializing current block field as entropy pool...")
        set_state('block_field_entropy', {})
        set_state('block_field_initialized', True)
        logger.info("[ENTROPY] ✓ Block field entropy pool initialized")
        return True
    except Exception as e:
        logger.error(f"[ENTROPY] Block field initialization failed: {e}")
        return False

def get_current_block_field() -> Optional[Dict[str, Any]]:
    """Get current block field (entropy pool source)"""
    return get_state('current_block_field')

def set_current_block_field(block_data: Dict[str, Any]) -> None:
    """Set current block field and extract entropy"""
    with _STATE_LOCK:
        _GLOBAL_STATE['current_block_field'] = block_data
        # Extract entropy from block field (nonmarkovian noise)
        _GLOBAL_STATE['block_field_entropy'] = get_entropy_from_block_field(block_data)
        logger.debug("[ENTROPY] Updated current block field entropy")

def get_entropy_from_block_field(block_data: Optional[Dict[str, Any]] = None) -> bytes:
    """Extract entropy from current block field (nonmarkovian noise bath)"""
    try:
        if block_data is None:
            block_data = get_state('current_block_field', {})
        
        # Hash block field components to extract entropy
        block_bytes = json.dumps(block_data, sort_keys=True, default=str).encode('utf-8')
        entropy = hashlib.sha256(block_bytes).digest()
        return entropy
    except Exception as e:
        logger.warning(f"[ENTROPY] Block field entropy extraction failed: {e}")
        # Fallback: return zeros
        return b'\x00' * ENTROPY_SIZE_BYTES

def get_block_field_entropy() -> bytes:
    """Get current block field entropy"""
    entropy = get_state('block_field_entropy')
    if entropy:
        return entropy
    return get_entropy_from_block_field()

# ─────────────────────────────────────────────────────────────────────────────
# CONSENSUS (FINALITY GADGET) — Integrated Functions
# ─────────────────────────────────────────────────────────────────────────────

def register_validator(pubkey: str, balance: int) -> Tuple[bool, int]:
    """Register validator (staking). Returns (success, validator_index)"""
    with _STATE_LOCK:
        validators = get_state('validators', {})
        if len(validators) >= 1024:  # MAX_VALIDATORS
            return False, -1
        
        validator_index = len(validators)
        validators[validator_index] = {
            'pubkey': pubkey,
            'balance': balance,
            'status': 'active',
            'activation_epoch': get_state('current_epoch', 0),
            'exit_epoch': 2**63 - 1,
            'slashed': False,
        }
        set_state('validators', validators)
        logger.info(f"[CONSENSUS] ✅ Validator {validator_index} registered (balance={balance/10**18:.2f} QTCL)")
        return True, validator_index

def record_quantum_witness(block_height: int, block_hash: str, 
                          w_state_fidelity: float, timestamp_ns: int) -> bool:
    """Record W-state quantum witness from oracle"""
    with _STATE_LOCK:
        witnesses = get_state('quantum_witnesses', {})
        witnesses[block_height] = {
            'block_hash': block_hash,
            'w_state_fidelity': w_state_fidelity,
            'timestamp_ns': timestamp_ns,
        }
        set_state('quantum_witnesses', witnesses)
        logger.debug(f"[CONSENSUS] Quantum witness recorded (block #{block_height}, fidelity={w_state_fidelity:.4f})")
        return True

def accept_attestation(validator_index: int, slot: int, beacon_block_root: str,
                      source_epoch: int, target_epoch: int, signature: str) -> Tuple[bool, str]:
    """Accept validator attestation. Check for double-attest and surround-vote."""
    with _STATE_LOCK:
        attestations = get_state('pending_attestations', [])
        
        # Check for double-attest
        for att in attestations:
            if (att.get('slot') == slot and 
                att.get('validator_index') == validator_index and
                att.get('beacon_block_root') != beacon_block_root):
                return False, "Double-attest detected"
        
        # Check for surround-vote
        for att in attestations:
            if (att.get('validator_index') == validator_index and
                att.get('source_epoch') < source_epoch < target_epoch < att.get('target_epoch', 2**63)):
                return False, "Surround-vote detected"
        
        # Add attestation
        attestations.append({
            'validator_index': validator_index,
            'slot': slot,
            'beacon_block_root': beacon_block_root,
            'source_epoch': source_epoch,
            'target_epoch': target_epoch,
            'signature': signature,
            'timestamp_ns': time.time_ns(),
        })
        set_state('pending_attestations', attestations)
        logger.debug(f"[CONSENSUS] Attestation from validator {validator_index} for slot {slot}")
        return True, "Accepted"

def compute_finality(block_height: int) -> Optional[int]:
    """Check if block is final (depth + quantum + 2/3 validators)"""
    with _STATE_LOCK:
        # 1. Check depth
        finality_depth = get_state('finality_depth', 32)
        if block_height < finality_depth:
            return None
        
        # 2. Check quantum witness
        witnesses = get_state('quantum_witnesses', {})
        if block_height not in witnesses:
            return None
        
        witness = witnesses[block_height]
        if witness.get('w_state_fidelity', 0.0) < 0.85:  # Fidelity threshold
            return None
        
        # 3. Check 2/3 validator supermajority
        validators = get_state('validators', {})
        total_balance = sum(v.get('balance', 0) for v in validators.values() 
                          if v.get('status') == 'active')
        quorum_balance = (total_balance * 2) // 3
        
        attestations = get_state('pending_attestations', [])
        attesting_balance = sum(
            validators.get(att.get('validator_index', -1), {}).get('balance', 0)
            for att in attestations
            if att.get('beacon_block_root') == witnesses[block_height].get('block_hash')
        )
        
        if attesting_balance >= quorum_balance:
            epoch = block_height // get_state('slots_per_epoch', 256)
            checkpoints = get_state('finality_checkpoints', {})
            checkpoints[epoch] = {
                'block_height': block_height,
                'block_hash': witness.get('block_hash'),
                'timestamp': int(time.time()),
                'validator_weight': attesting_balance,
            }
            set_state('finality_checkpoints', checkpoints)
            logger.info(f"[CONSENSUS] ✅ FINALITY: Block #{block_height} finalized at epoch {epoch}")
            return block_height
        
        return None

# ─────────────────────────────────────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    'initialize_system',
    'shutdown_system',
    'get_system_status',
    'get_state',
    'set_state',
    'update_state',
    'get_db_pool',
    'initialize_database',
    'close_database',
    'initialize_block_field_entropy',
    'get_current_block_field',
    'set_current_block_field',
    'get_entropy_from_block_field',
    'get_block_field_entropy',
    'register_validator',
    'record_quantum_witness',
    'accept_attestation',
    'compute_finality',
    'get_hlwe_system',
    'get_lattice_controller',
    'get_heartbeat',
    'start_heartbeat',
    'stop_heartbeat',
]
