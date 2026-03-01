#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                        ║
║  QTCL GLOBAL STATE MANAGEMENT v1.0 — NEW SCHEMA INTEGRATION                          ║
║                                                                                        ║
║  Thread-safe global state with database connection pooling                            ║
║  QRNG Ensemble: 5-source quantum entropy                                              ║
║  HLWE Cryptography: Post-quantum wallet management                                    ║
║  Quantum Lattice: Main controller integration                                         ║
║  Direct schema awareness: All tables, indexes, constraints                            ║
║                                                                                        ║
║  REMOVED: Byzantine consensus, LYRA validator pool, all legacy commands               ║
║  FRESH: Schema-native state management and initialization                            ║
╚════════════════════════════════════════════════════════════════════════════════════════╝
"""

import threading, logging, json, sys
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

# Database
try:
    import psycopg2
    from psycopg2.pool import ThreadedConnectionPool
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

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
    
    # QRNG Ensemble
    'qrng_ensemble': None,
    'qrng_stats': {},
    'qrng_initialized': False,
    'qrng_entropy_estimate': 0.0,
    
    # HLWE Cryptography
    'hlwe_system': None,
    'hlwe_initialized': False,
    
    # Quantum Lattice
    'lattice_controller': None,
    'heartbeat': None,
    'lattice_initialized': False,
    
    # Server metadata
    'startup_time': datetime.now(timezone.utc).isoformat(),
    'server_version': '1.0',
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
# QRNG ENSEMBLE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def initialize_qrng_ensemble() -> bool:
    """
    Initialize QRNG Ensemble (5-source quantum entropy)
    Lazy loading - will be created on demand
    """
    try:
        # Mark as attempted - actual QRNG will be loaded when needed
        set_state('qrng_initialized', True)
        logger.info("[QRNG] ✓ QRNG ensemble ready (lazy load)")
        return True
    except Exception as e:
        logger.error(f"[QRNG] Initialization failed: {e}")
        return False

def get_qrng_ensemble() -> Optional[Any]:
    """Get QRNG ensemble (lazy load if needed)"""
    qrng = get_state('qrng_ensemble')
    if qrng is None and get_state('qrng_initialized'):
        try:
            # Lazy load QRNG on first access
            from qrng_ensemble import get_qrng_ensemble as create_qrng
            qrng = create_qrng()
            set_state('qrng_ensemble', qrng)
            logger.debug("[QRNG] ✓ Lazy-loaded QRNG ensemble")
        except Exception as e:
            logger.warning(f"[QRNG] Lazy load failed: {e}")
    return qrng

def refresh_qrng_stats() -> Dict[str, Any]:
    """Refresh QRNG statistics"""
    qrng = get_qrng_ensemble()
    if not qrng:
        return {}
    
    try:
        stats = qrng.get_entropy_stats() if hasattr(qrng, 'get_entropy_stats') else {}
        entropy = qrng.get_entropy_estimate() if hasattr(qrng, 'get_entropy_estimate') else 0.0
        
        update_state({
            'qrng_stats': stats,
            'qrng_entropy_estimate': entropy,
        })
        return stats
    except Exception as e:
        logger.debug(f"[QRNG] Stats refresh failed: {e}")
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
    Main server control system for lattice evolution
    """
    try:
        db_pool = get_db_pool()
        
        from quantum_lattice_control_server import get_lattice_controller, get_heartbeat
        lattice = get_lattice_controller(db_pool)
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
    Order: Database → QRNG → HLWE → Lattice → Heartbeat
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
        logger.info("=" * 80)
        logger.info("[SYSTEM] QTCL Startup Sequence")
        logger.info("=" * 80)
        
        # 1. Database
        if not initialize_database(db_url):
            raise RuntimeError("Database initialization failed")
        
        # 2. QRNG
        if not initialize_qrng_ensemble():
            logger.warning("[SYSTEM] QRNG initialization failed - continuing")
        
        # 3. HLWE
        if not initialize_hlwe_engine():
            logger.warning("[SYSTEM] HLWE initialization failed - continuing")
        
        # 4. Lattice
        if not initialize_lattice_controller():
            logger.warning("[SYSTEM] Lattice initialization failed - continuing")
        
        # 5. Heartbeat
        if start_heartbeat_daemon:
            if not start_heartbeat():
                logger.warning("[SYSTEM] Heartbeat start failed - continuing")
        
        set_state('initialized', True)
        set_state('_initializing', False)
        
        logger.info("[SYSTEM] ✓ QTCL System fully initialized")
        logger.info("=" * 80)
        return True
    except Exception as e:
        logger.error(f"[SYSTEM] Initialization failed: {e}")
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
    """Get complete system status"""
    try:
        lattice = get_lattice_controller()
        heartbeat = get_heartbeat()
        
        return {
            'initialized': get_state('initialized'),
            'startup_time': get_state('startup_time'),
            'server_version': get_state('server_version'),
            'database': {
                'initialized': get_state('db_initialized'),
                'pool': get_state('db_pool') is not None,
            },
            'qrng': {
                'initialized': get_state('qrng_initialized'),
                'entropy_estimate': get_state('qrng_entropy_estimate'),
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
        return {'error': str(e)}

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
    'get_qrng_ensemble',
    'refresh_qrng_stats',
    'get_hlwe_system',
    'get_lattice_controller',
    'get_heartbeat',
    'start_heartbeat',
    'stop_heartbeat',
]
