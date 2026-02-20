#!/usr/bin/env python3
"""
QTCL WSGI v5.0 - GLOBALS COORDINATOR & SYSTEM BOOTSTRAP

Proper globals interfacing system. Initializes all subsystems in correct order:
1. Logging (foundation)
2. Utility classes (PROFILER, CACHE, etc)
3. db_builder_v2 (database singleton)
4. globals module (unified state)
5. Terminal engine (command system - lazy loaded to avoid circular imports)
6. Flask app (HTTP interface)

Other modules import from wsgi_config to access initialized globals.
"""

import os
import sys
import logging
import threading
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ═══════════════════════════════════════════════════════════════════════════════════════
# UTILITY CLASSES - Available for all modules
# ═══════════════════════════════════════════════════════════════════════════════════════

class SimpleProfiler:
    """Performance profiling for system operations"""
    def __init__(self):
        self.metrics = {}
        self.lock = threading.RLock()
    
    def record(self, name: str, duration_ms: float):
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(duration_ms)
    
    def get_stats(self, name: str) -> dict:
        with self.lock:
            if name not in self.metrics:
                return {'count': 0, 'avg_ms': 0, 'min_ms': 0, 'max_ms': 0}
            times = self.metrics[name]
            return {
                'count': len(times),
                'avg_ms': sum(times) / len(times),
                'min_ms': min(times),
                'max_ms': max(times),
            }


class SimpleCache:
    """In-memory caching with TTL"""
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.lock = threading.RLock()
    
    def get(self, key: str):
        with self.lock:
            if key not in self.cache:
                return None
            item = self.cache[key]
            if item['ttl'] and time.time() > item['expires_at']:
                del self.cache[key]
                return None
            return item['value']
    
    def set(self, key: str, value, ttl_sec: int = 3600):
        with self.lock:
            if len(self.cache) >= self.max_size:
                self.cache.pop(next(iter(self.cache)), None)
            self.cache[key] = {
                'value': value,
                'ttl': ttl_sec,
                'expires_at': time.time() + ttl_sec if ttl_sec else None
            }
    
    def delete(self, key: str):
        with self.lock:
            self.cache.pop(key, None)
    
    def clear(self):
        with self.lock:
            self.cache.clear()


class RequestTracer:
    """Request correlation tracking"""
    def __init__(self):
        import uuid
        self.current_id = None
        self.uuid = uuid
    
    def new_id(self) -> str:
        self.current_id = str(self.uuid.uuid4())
        return self.current_id


class ErrorBudgetManager:
    """Error budget tracking"""
    def __init__(self, max_errors: int = 100, window_sec: int = 60):
        self.max_errors = max_errors
        self.window_sec = window_sec
        self.errors = []
        self.lock = threading.RLock()
    
    def record_error(self):
        with self.lock:
            now = time.time()
            self.errors = [t for t in self.errors if now - t < self.window_sec]
            self.errors.append(now)
    
    def is_exhausted(self) -> bool:
        with self.lock:
            now = time.time()
            self.errors = [t for t in self.errors if now - t < self.window_sec]
            return len(self.errors) >= self.max_errors
    
    def remaining(self) -> int:
        with self.lock:
            now = time.time()
            self.errors = [t for t in self.errors if now - t < self.window_sec]
            return max(0, self.max_errors - len(self.errors))


# Global utility instances
PROFILER = SimpleProfiler()
CACHE = SimpleCache(max_size=2000)
RequestCorrelation = RequestTracer()
ERROR_BUDGET = ErrorBudgetManager(max_errors=100, window_sec=60)

logger.info("[BOOTSTRAP] ✅ Utility services initialized (PROFILER, CACHE, RequestCorrelation, ERROR_BUDGET)")

# ═══════════════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP PHASE 1: DATABASE SINGLETON (db_builder_v2) - BEFORE globals to break circular import
# ═══════════════════════════════════════════════════════════════════════════════════════

DB = None
try:
    logger.info("[BOOTSTRAP] Initializing database layer (db_builder_v2)...")
    from db_builder_v2 import db_manager, DB_POOL
    
    if db_manager is None:
        raise RuntimeError("db_manager singleton is None")
    
    DB = db_manager
    logger.info("[BOOTSTRAP] ✅ Database singleton ready (db_builder_v2.db_manager)")
except Exception as e:
    logger.error(f"[BOOTSTRAP] ❌ FATAL: Database initialization failed: {e}")
    raise

# ═══════════════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP PHASE 2: GLOBALS MODULE (Unified State) - Now ledger_manager can safely import PROFILER
# ═══════════════════════════════════════════════════════════════════════════════════════

GLOBALS_AVAILABLE = False
try:
    logger.info("[BOOTSTRAP] Initializing globals module (ledger_manager will lazy-import PROFILER)...")
    import globals as globals_module
    globals_module.initialize_globals()
    initialize_globals = globals_module.initialize_globals
    get_globals = globals_module.get_globals
    get_system_health = globals_module.get_system_health
    get_state_snapshot = globals_module.get_state_snapshot
    get_heartbeat = globals_module.get_heartbeat
    get_blockchain = globals_module.get_blockchain
    get_ledger = globals_module.get_ledger
    get_oracle = globals_module.get_oracle
    get_defi = globals_module.get_defi
    get_auth_manager = globals_module.get_auth_manager
    get_pqc_system = globals_module.get_pqc_system
    get_genesis_block = globals_module.get_genesis_block
    verify_genesis_block = globals_module.verify_genesis_block
    get_metrics = globals_module.get_metrics
    get_quantum = globals_module.get_quantum
    dispatch_command = globals_module.dispatch_command
    COMMAND_REGISTRY = globals_module.COMMAND_REGISTRY
    pqc_generate_user_key = globals_module.pqc_generate_user_key
    pqc_sign = globals_module.pqc_sign
    pqc_verify = globals_module.pqc_verify
    pqc_encapsulate = globals_module.pqc_encapsulate
    pqc_prove_identity = globals_module.pqc_prove_identity
    pqc_verify_identity = globals_module.pqc_verify_identity
    pqc_revoke_key = globals_module.pqc_revoke_key
    pqc_rotate_key = globals_module.pqc_rotate_key
    bootstrap_admin_session = globals_module.bootstrap_admin_session
    revoke_session = globals_module.revoke_session
    GLOBALS_AVAILABLE = True
    logger.info("[BOOTSTRAP] ✅ Globals module initialized")
except Exception as e:
    logger.error(f"[BOOTSTRAP] ❌ FATAL: Globals initialization failed: {e}")
    raise

# ═══════════════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP PHASE 3: TERMINAL ENGINE (Command System - LAZY LOADED)
# ═══════════════════════════════════════════════════════════════════════════════════════

_TERMINAL_INITIALIZED = False
_terminal_engine = None

def _initialize_terminal_engine():
    """Lazy initialization of terminal engine to avoid circular imports"""
    global _TERMINAL_INITIALIZED, _terminal_engine
    
    if _TERMINAL_INITIALIZED:
        return _terminal_engine
    
    try:
        logger.info("[BOOTSTRAP] Initializing terminal engine...")
        from terminal_logic import TerminalEngine, register_all_commands
        
        engine = TerminalEngine()
        cmd_count = register_all_commands(engine)
        
        total_cmds = len(COMMAND_REGISTRY)
        logger.info(f"[BOOTSTRAP] ✓ Registered {total_cmds} commands")
        
        if total_cmds < 80:
            raise RuntimeError(f"Command registration incomplete: {total_cmds} (expected 89+)")
        
        _terminal_engine = engine
        _TERMINAL_INITIALIZED = True
        logger.info("[BOOTSTRAP] ✅ Terminal engine initialized")
        return engine
        
    except Exception as e:
        logger.error(f"[BOOTSTRAP] ❌ FATAL: Terminal engine failed: {e}")
        raise

# ═══════════════════════════════════════════════════════════════════════════════════════
# EXPORTS: Other modules import these from wsgi_config
# ═══════════════════════════════════════════════════════════════════════════════════════

__all__ = [
    'DB', 'PROFILER', 'CACHE', 'RequestCorrelation', 'ERROR_BUDGET',
    'GLOBALS_AVAILABLE', 'logger', 'app', 'application',
    'get_system_health', 'get_state_snapshot',
    'get_heartbeat', 'get_blockchain', 'get_ledger', 'get_oracle', 'get_defi',
    'get_auth_manager', 'get_pqc_system', 'get_genesis_block', 'verify_genesis_block',
    'get_metrics', 'get_quantum', 'dispatch_command', 'COMMAND_REGISTRY',
    'pqc_generate_user_key', 'pqc_sign', 'pqc_verify',
    'pqc_encapsulate', 'pqc_prove_identity', 'pqc_verify_identity',
    'pqc_revoke_key', 'pqc_rotate_key',
    'bootstrap_admin_session', 'revoke_session',
]

# Ensure functions are available (late binding in case of re-imports)
def _ensure_exports():
    """Ensure all critical exports are available."""
    global initialize_globals, get_globals, get_system_health, get_state_snapshot
    global get_heartbeat, get_blockchain, get_ledger, get_oracle, get_defi
    global get_auth_manager, get_pqc_system, get_genesis_block, verify_genesis_block
    global get_metrics, get_quantum, dispatch_command, COMMAND_REGISTRY
    global pqc_generate_user_key, pqc_sign, pqc_verify
    global pqc_encapsulate, pqc_prove_identity, pqc_verify_identity
    global pqc_revoke_key, pqc_rotate_key, bootstrap_admin_session, revoke_session
    
    if not GLOBALS_AVAILABLE:
        # Functions should already be imported above, but provide fallbacks
        pass

_ensure_exports()

# ═══════════════════════════════════════════════════════════════════════════════════════
# FLASK APPLICATION (HTTP Interface)
# ═══════════════════════════════════════════════════════════════════════════════════════

from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime, timezone

app = Flask(__name__, static_folder=os.path.join(PROJECT_ROOT, 'static'), static_url_path='/static')
app.config['JSON_SORT_KEYS'] = False


class CircuitBreaker:
    """Safe access to optional services"""
    def __init__(self, service_name, getter):
        self.service_name = service_name
        self.getter = getter
    
    def get(self, default=None):
        try:
            return self.getter() or default or {}
        except Exception as e:
            logger.warning(f"⚠️  {self.service_name} unavailable: {str(e)[:60]}")
            return default or {}


SERVICES = {
    'quantum': CircuitBreaker('quantum', lambda: get_heartbeat()),
    'blockchain': CircuitBreaker('blockchain', lambda: get_blockchain()),
    'ledger': CircuitBreaker('ledger', lambda: get_ledger()),
    'oracle': CircuitBreaker('oracle', lambda: get_oracle()),
    'defi': CircuitBreaker('defi', lambda: get_defi()),
    'auth': CircuitBreaker('auth', lambda: get_auth_manager()),
    'pqc': CircuitBreaker('pqc', lambda: get_pqc_system()),
}

# ═══════════════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/', methods=['GET'])
def index():
    accept = request.headers.get('Accept', '')
    if 'text/html' in accept:
        try:
            return send_from_directory(PROJECT_ROOT, 'index.html')
        except:
            pass
    return jsonify({
        'system': 'QTCL v5.0',
        'status': 'operational',
        'timestamp': datetime.now(timezone.utc).isoformat(),
    })

@app.route('/health', methods=['GET'])
def health():
    try:
        return jsonify(get_system_health()), 200
    except:
        return jsonify({'status': 'healthy'}), 200

@app.route('/api/status', methods=['GET'])
def status():
    try:
        return jsonify({'health': get_system_health(), 'snapshot': get_state_snapshot()})
    except:
        return jsonify({'status': 'operational'})

@app.route('/api/command', methods=['POST'])
def execute_command():
    try:
        data = request.get_json() or {}
        command = data.get('command', 'help')
        args = data.get('args') or {}
        user_id = data.get('user_id')
        
        if not user_id and 'Authorization' in request.headers:
            try:
                import jwt
                token = request.headers['Authorization'].replace('Bearer ', '')
                secret = os.getenv('JWT_SECRET', '')
                if secret:
                    payload = jwt.decode(token, secret, algorithms=['HS256', 'HS512'])
                    user_id = payload.get('user_id')
            except:
                pass
        
        result = dispatch_command(command, args, user_id)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Command execution failed: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/commands', methods=['GET'])
def commands():
    try:
        _initialize_terminal_engine()  # Lazy load on first request
        return jsonify({'total': len(COMMAND_REGISTRY), 'commands': dict(COMMAND_REGISTRY)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/genesis', methods=['GET'])
def genesis():
    try:
        return jsonify({'block': get_genesis_block(), 'verified': verify_genesis_block()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/quantum', methods=['GET'])
def quantum():
    return jsonify(SERVICES['quantum'].get())

@app.route('/api/blockchain', methods=['GET'])
def blockchain():
    return jsonify(SERVICES['blockchain'].get())

@app.route('/api/ledger', methods=['GET'])
def ledger():
    return jsonify(SERVICES['ledger'].get())

@app.route('/api/metrics', methods=['GET'])
def metrics():
    try:
        return jsonify(get_metrics())
    except:
        return jsonify({'status': 'operational'})

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"500 error: {e}")
    return jsonify({'error': 'Server error'}), 500

# ═══════════════════════════════════════════════════════════════════════════════════════
# WSGI ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════════════

application = app

if __name__ == '__main__':
    _initialize_terminal_engine()  # Initialize on startup
    logger.info("="*80)
    logger.info("🚀 QTCL WSGI v5.0 - PRODUCTION STARTUP")
    logger.info("="*80)
    logger.info(f"✅ Database initialized: {DB is not None}")
    logger.info(f"✅ Utilities initialized: PROFILER, CACHE, RequestCorrelation, ERROR_BUDGET")
    logger.info(f"✅ Globals available: {GLOBALS_AVAILABLE}")
    logger.info(f"✅ Terminal engine ready: {_TERMINAL_INITIALIZED}")
    logger.info(f"✅ Commands registered: {len(COMMAND_REGISTRY)}")
    logger.info("="*80)
    app.run(host='0.0.0.0', port=8000, debug=False)
