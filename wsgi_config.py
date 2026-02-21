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

# Only configure logging once - subsequent imports should not reconfigure
if not logging.getLogger().hasHandlers():
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
# BOOTSTRAP PHASE 1: DATABASE SINGLETON (db_builder_v2)
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
# BOOTSTRAP PHASE 2: GLOBALS MODULE (Unified State) - DEFERRED
# ═══════════════════════════════════════════════════════════════════════════════════════

GLOBALS_AVAILABLE = False
_GLOBALS_INIT_THREAD = None
_GLOBALS_INIT_TIMEOUT = 30  # seconds

def _initialize_globals_deferred():
    """Initialize globals in background thread with timeout."""
    global GLOBALS_AVAILABLE
    try:
        logger.info("[BOOTSTRAP] Starting deferred globals initialization (background thread)...")
        from globals import initialize_globals
        
        # Try with timeout
        initialize_globals()
        GLOBALS_AVAILABLE = True
        logger.info("[BOOTSTRAP] ✅ Globals module initialized in background")
    except Exception as e:
        logger.error(f"[BOOTSTRAP] ⚠️  Globals initialization failed: {e}")
        # Don't re-raise — let app start anyway with degraded functionality

try:
    logger.info("[BOOTSTRAP] Pre-loading globals module (imports only, no initialization)...")
    from globals import (
        initialize_globals, get_globals,
        get_system_health, get_state_snapshot,
        get_heartbeat, get_blockchain, get_ledger, get_oracle, get_defi,
        get_auth_manager, get_pqc_system, get_genesis_block, verify_genesis_block,
        get_metrics, get_quantum, dispatch_command, COMMAND_REGISTRY,
        pqc_generate_user_key, pqc_sign, pqc_verify,
        pqc_encapsulate, pqc_prove_identity, pqc_verify_identity,
        pqc_revoke_key, pqc_rotate_key, bootstrap_admin_session, revoke_session,
    )
    logger.info("[BOOTSTRAP] ✅ Globals module imports successful")
    
    # Start initialization in background thread to avoid blocking Flask startup
    _GLOBALS_INIT_THREAD = threading.Thread(target=_initialize_globals_deferred, daemon=True)
    _GLOBALS_INIT_THREAD.start()
    logger.info("[BOOTSTRAP] ✅ Globals initialization started in background (daemon thread)")
    
except Exception as e:
    logger.error(f"[BOOTSTRAP] ❌ FATAL: Could not import globals: {e}")
    raise

# ═══════════════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP PHASE 3: TERMINAL ENGINE (Command System - LAZY LOADED)
# ═══════════════════════════════════════════════════════════════════════════════════════

_TERMINAL_INITIALIZED = False
_TERMINAL_INIT_FAILED = False   # sticky failure flag — stop retrying after first failure
_terminal_engine = None
_terminal_lock = threading.Lock()  # guards the initialization critical section


def _initialize_terminal_engine():
    """
    Lazy, thread-safe initialization of the terminal engine.

    Thread-safety:
        Uses a double-checked locking pattern with a module-level threading.Lock.
        Without the lock, concurrent WSGI requests each see _TERMINAL_INITIALIZED=False
        and all spawn a full TerminalEngine (8 ParallelExecutor worker threads each).
        Five concurrent requests → 40 orphan threads + log flood.

    Failure semantics:
        If the first init attempt fails the flag _TERMINAL_INIT_FAILED is set so
        subsequent calls return immediately instead of retrying on every request.
        The COMMAND_REGISTRY stubs populated by globals.py still work — callers
        degrade gracefully without handlers but don't hammer the system.
    """
    global _TERMINAL_INITIALIZED, _TERMINAL_INIT_FAILED, _terminal_engine

    # Fast-path: already done (or permanently failed)
    if _TERMINAL_INITIALIZED or _TERMINAL_INIT_FAILED:
        return _terminal_engine

    # Slow-path: take the lock and re-check inside
    with _terminal_lock:
        if _TERMINAL_INITIALIZED or _TERMINAL_INIT_FAILED:
            return _terminal_engine

        try:
            logger.info("[BOOTSTRAP] Initializing terminal engine (thread-safe)...")
            from terminal_logic import TerminalEngine, register_all_commands

            engine = TerminalEngine()
            cmd_count = register_all_commands(engine)

            total_cmds = len(COMMAND_REGISTRY)
            logger.info(f"[BOOTSTRAP] ✓ Registered {total_cmds} commands "
                        f"({cmd_count} from terminal_logic handlers)")

            if total_cmds < 10:
                # Only hard-fail on truly empty registry — stubs count too
                raise RuntimeError(
                    f"Command registration catastrophically incomplete: {total_cmds} commands. "
                    "Check terminal_logic import errors above."
                )

            if total_cmds < 80:
                logger.warning(
                    f"[BOOTSTRAP] ⚠ Only {total_cmds} commands registered (expected 89+). "
                    "Some commands may be missing. Check terminal_logic dependency errors."
                )

            _terminal_engine = engine
            _TERMINAL_INITIALIZED = True
            logger.info("[BOOTSTRAP] ✅ Terminal engine initialized successfully")
            return engine

        except Exception as e:
            _TERMINAL_INIT_FAILED = True   # stop retrying on every request
            logger.error(
                f"[BOOTSTRAP] ❌ Terminal engine initialization failed: {e}\n"
                "  → Falling back to globals.py stub registry (basic commands still work).\n"
                "  → Fix the error above and restart to enable full command set.",
                exc_info=True,
            )
            return None

# ═══════════════════════════════════════════════════════════════════════════════════════
# EXPORTS: Other modules import these from wsgi_config
# ═══════════════════════════════════════════════════════════════════════════════════════

__all__ = [
    'DB', 'PROFILER', 'CACHE', 'RequestCorrelation', 'ERROR_BUDGET',
    'GLOBALS_AVAILABLE', 'logger',
    'get_system_health', 'get_state_snapshot',
    'get_heartbeat', 'get_blockchain', 'get_ledger', 'get_oracle', 'get_defi',
    'get_auth_manager', 'get_pqc_system', 'get_genesis_block', 'verify_genesis_block',
    'get_metrics', 'get_quantum', 'dispatch_command', 'COMMAND_REGISTRY',
    'pqc_generate_user_key', 'pqc_sign', 'pqc_verify',
    'pqc_encapsulate', 'pqc_prove_identity', 'pqc_verify_identity',
    'pqc_revoke_key', 'pqc_rotate_key',
    'bootstrap_admin_session', 'revoke_session',
]

# ═══════════════════════════════════════════════════════════════════════════════════════
# FLASK APPLICATION (HTTP Interface)
# ═══════════════════════════════════════════════════════════════════════════════════════

from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime, timezone

app = Flask(__name__, static_folder=os.path.join(PROJECT_ROOT, 'static'), static_url_path='/static')
app.config['JSON_SORT_KEYS'] = False

logger.info("[FLASK] ✅ Flask app created and ready to receive requests")
logger.info("[FLASK] 📝 Globals initialization continues in background thread (non-blocking)")
logger.info("[FLASK] 🚀 App will be fully functional once globals initialization completes")

@app.before_request
def log_request():
    """Log incoming requests"""
    logger.debug(f"[REQUEST] {request.method} {request.path}")

@app.after_request
def log_response(response):
    """Log outgoing responses"""
    logger.debug(f"[RESPONSE] {response.status_code}")
    return response


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
    """Minimal health check - returns immediately"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'globals_initialized': GLOBALS_AVAILABLE,
        'note': 'If globals_initialized=false, system is still starting up (normal on first boot)'
    }), 200

@app.route('/api/status', methods=['GET'])
def status():
    try:
        return jsonify({'health': get_system_health(), 'snapshot': get_state_snapshot()})
    except:
        return jsonify({'status': 'operational'})

@app.route('/api/command', methods=['POST'])
def execute_command():
    try:
        # Ensure terminal engine is initialized (thread-safe, idempotent)
        _initialize_terminal_engine()

        data = request.get_json() or {}
        command = data.get('command', 'help')
        args = data.get('args') or {}
        user_id = data.get('user_id')

        # ── JWT extraction — decode once, carry full payload downstream ────────
        raw_token = None
        role = None
        is_admin = False
        jwt_decode_error = None

        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            raw_token = auth_header[7:].strip()
        # Also accept token in JSON body (some clients send it there)
        if not raw_token:
            raw_token = data.get('token') or data.get('access_token')

        if raw_token:
            try:
                import jwt as _jwt
                # Collect all possible secrets to try (order: env → auth_handlers → globals)
                secrets_to_try = []
                
                # 1. Try environment variable (production override)
                env_secret = os.getenv('JWT_SECRET', '')
                if env_secret:
                    secrets_to_try.append(('ENV', env_secret))
                
                # 2. Try auth_handlers module (canonical location)
                try:
                    from auth_handlers import JWT_SECRET as _ahs_secret
                    if _ahs_secret:
                        secrets_to_try.append(('AUTH_HANDLERS', _ahs_secret))
                except Exception as e:
                    logger.debug(f"[auth] Could not import JWT_SECRET from auth_handlers: {e}")
                
                # 3. Try globals module (_get_jwt_secret)
                try:
                    from globals import _get_jwt_secret as _get_secret
                    _gs = _get_secret()
                    if _gs:
                        secrets_to_try.append(('GLOBALS', _gs))
                except Exception as e:
                    logger.debug(f"[auth] Could not get JWT_SECRET from globals: {e}")
                
                jwt_payload = {}
                decode_success = False
                
                # Try each secret until one works
                for secret_source, secret in secrets_to_try:
                    try:
                        jwt_payload = _jwt.decode(raw_token, secret, algorithms=['HS512', 'HS256'])
                        decode_success = True
                        logger.debug(f"[auth] JWT decoded successfully using {secret_source} secret")
                        break
                    except _jwt.InvalidSignatureError:
                        continue  # Try next secret
                    except Exception as e:
                        jwt_decode_error = str(e)
                        continue
                
                if decode_success and jwt_payload:
                    if not user_id:
                        user_id = jwt_payload.get('user_id')
                    role = jwt_payload.get('role', 'user')
                    is_admin = bool(jwt_payload.get('is_admin', False)) or role in ('admin', 'superadmin', 'super_admin')
                    logger.info(f"[auth] ✓ JWT validated: user={user_id} role={role} admin={is_admin}")
                elif raw_token:
                    # Token present but decode failed - log more details for debugging
                    logger.warning(f"[auth] ⚠️  JWT decode failed for token starting with {raw_token[:20]}... | Error: {jwt_decode_error}")
            except Exception as _je:
                logger.error(f"[auth] Unexpected error during JWT processing: {_je}", exc_info=True)

        result = dispatch_command(command, args, user_id, token=raw_token, role=role)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Command execution failed: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/commands', methods=['GET'])
def commands():
    try:
        # Initialize terminal engine once (thread-safe, idempotent after first call)
        # If it already initialized or permanently failed, this returns immediately
        _initialize_terminal_engine()

        # Build serializable command list (filter out handler functions)
        serializable_commands = {}
        for cmd_name, cmd_info in COMMAND_REGISTRY.items():
            # Copy all fields EXCEPT 'handler' (which is a function, not JSON serializable)
            serializable_commands[cmd_name] = {
                k: v for k, v in cmd_info.items()
                if k != 'handler' and not callable(v)
            }

        # Always return command registry (populated at module load time)
        return jsonify({
            'total': len(serializable_commands),
            'commands': serializable_commands,
            'status': 'success'
        })
    except Exception as e:
        logger.error(f"[/api/commands] Unexpected error: {e}", exc_info=True)
        return jsonify({
            'total': 0,
            'commands': {},
            'status': 'error',
            'error': str(e)
        }), 200  # Return 200 even with error

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

# ─── EAGER TERMINAL ENGINE INIT ───────────────────────────────────────────────
# Initialize the terminal engine NOW at process startup rather than lazily on
# the first incoming request.  This eliminates the race window where concurrent
# requests would each try to initialize simultaneously.
#
# If terminal_logic has a dependency problem the error is logged clearly here
# (during startup, where it's easy to spot) instead of surfacing on the first
# user request.  _initialize_terminal_engine() is idempotent — subsequent
# calls from the routes return immediately via the _TERMINAL_INITIALIZED flag.
#
# DO NOT call inside __name__ == '__main__' block — it must run in every WSGI
# worker process (gunicorn pre-forks after this point).
try:
    _initialize_terminal_engine()
except Exception as _boot_exc:
    logger.error(f"[STARTUP] Terminal engine init at startup failed (non-fatal): {_boot_exc}")

if __name__ == '__main__':
    # DO NOT initialize terminal engine here - it will lazy-load on first request
    # This avoids circular recursion during bootstrap
    logger.info("="*80)
    logger.info("🚀 QTCL WSGI v5.0 - PRODUCTION STARTUP")
    logger.info("="*80)
    logger.info(f"✅ Database initialized: {DB is not None}")
    logger.info(f"✅ Utilities initialized: PROFILER, CACHE, RequestCorrelation, ERROR_BUDGET")
    logger.info(f"✅ Globals available: {GLOBALS_AVAILABLE}")
    logger.info(f"✅ Terminal engine: LAZY-LOAD (initializes on first request)")
    logger.info(f"✅ Commands will be registered: len(COMMAND_REGISTRY)={len(COMMAND_REGISTRY)}")
    logger.info("="*80)
    app.run(host='0.0.0.0', port=8000, debug=False)
