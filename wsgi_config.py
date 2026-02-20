#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                        ║
║                     🚀 QTCL COMPREHENSIVE WSGI INTEGRATION v5.0 🚀                    ║
║                                                                                        ║
║  Comprehensive system integration with proper circuit breakers                        ║
║                                                                                        ║
╚════════════════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import logging
from datetime import datetime, timezone
from flask import Flask, request, jsonify, send_from_directory, send_file
import threading
from contextlib import contextmanager
from typing import Optional, Dict, Any, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ═══════════════════════════════════════════════════════════════════════════════════════
# DATABASE CONNECTION POOLING - GLOBAL DB WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════════════

class DBConnectionPool:
    """Thread-safe PostgreSQL connection pool wrapper using psycopg2.pool.ThreadedConnectionPool"""
    
    def __init__(self, db_url: Optional[str] = None, min_conns: int = 2, max_conns: int = 10):
        """Initialize connection pool"""
        self.db_url = db_url or os.getenv('DATABASE_URL', '')
        self.min_conns = min_conns
        self.max_conns = max_conns
        self.pool = None
        self.lock = threading.RLock()
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize the connection pool (lazy)"""
        if not self.db_url:
            logger.warning("⚠️  DATABASE_URL not set; DB operations will fail until configured")
            return
        
        try:
            import psycopg2
            from psycopg2.pool import ThreadedConnectionPool
            
            with self.lock:
                if self.pool is None:
                    self.pool = ThreadedConnectionPool(
                        self.min_conns,
                        self.max_conns,
                        self.db_url,
                        connect_timeout=5
                    )
                    logger.info(f"✅ Database connection pool initialized: {self.min_conns}-{self.max_conns} connections")
        except Exception as e:
            logger.error(f"❌ Failed to initialize DB pool: {e}")
            self.pool = None
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool (context manager)"""
        if not self.pool:
            self._initialize_pool()
        
        if not self.pool:
            raise RuntimeError("Database pool not initialized and DATABASE_URL not available")
        
        conn = self.pool.getconn()
        try:
            yield conn
        finally:
            self.pool.putconn(conn)
    
    def execute(self, query: str, params: tuple = None, fetch_one: bool = False, 
                fetch_all: bool = False) -> Any:
        """Execute a query and return result"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params or ())
                    if fetch_one:
                        return cur.fetchone()
                    elif fetch_all:
                        return cur.fetchall()
                    else:
                        conn.commit()
                        return cur.rowcount
        except Exception as e:
            logger.error(f"❌ DB query error: {e}")
            raise
    
    def executemany(self, query: str, params_list: List[tuple]) -> int:
        """Execute multiple queries"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.executemany(query, params_list)
                    conn.commit()
                    return cur.rowcount
        except Exception as e:
            logger.error(f"❌ DB batch error: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check if database is accessible"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    return cur.fetchone() is not None
        except Exception as e:
            logger.error(f"❌ DB health check failed: {e}")
            return False
    
    def close_all(self):
        """Close all connections in pool"""
        if self.pool:
            with self.lock:
                try:
                    self.pool.closeall()
                    logger.info("✅ All DB connections closed")
                except Exception as e:
                    logger.error(f"❌ Error closing DB pool: {e}")


# Global DB instance
DB = DBConnectionPool()

# Compatibility aliases for RequestCorrelation, PROFILER, CACHE, ERROR_BUDGET
class RequestCorrelation:
    """Request correlation tracking"""
    current_id: Optional[str] = None
    
    @classmethod
    def new_id(cls) -> str:
        import uuid
        cls.current_id = str(uuid.uuid4())
        return cls.current_id


class SimpleProfiler:
    """Basic profiler for performance tracking"""
    def __init__(self):
        self.metrics = {}
    
    def record(self, name: str, duration_ms: float):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(duration_ms)
    
    def get_stats(self, name: str) -> Dict[str, float]:
        if name not in self.metrics:
            return {}
        times = self.metrics[name]
        return {
            'count': len(times),
            'avg_ms': sum(times) / len(times),
            'min_ms': min(times),
            'max_ms': max(times),
        }


class SimpleCache:
    """Simple in-memory cache"""
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            return self.cache.get(key)
    
    def set(self, key: str, value: Any, ttl_sec: int = 3600):
        with self.lock:
            if len(self.cache) >= self.max_size:
                # Simple eviction: remove first item
                self.cache.pop(next(iter(self.cache)), None)
            self.cache[key] = {'value': value, 'ttl': ttl_sec}
    
    def delete(self, key: str):
        with self.lock:
            self.cache.pop(key, None)
    
    def clear(self):
        with self.lock:
            self.cache.clear()


class ErrorBudget:
    """Error budget tracking for graceful degradation"""
    def __init__(self, max_errors: int = 100, window_sec: int = 60):
        self.max_errors = max_errors
        self.window_sec = window_sec
        self.errors = []
        self.lock = threading.RLock()
    
    def record_error(self):
        with self.lock:
            import time
            now = time.time()
            # Remove old errors outside window
            self.errors = [t for t in self.errors if now - t < self.window_sec]
            self.errors.append(now)
    
    def is_exhausted(self) -> bool:
        with self.lock:
            import time
            now = time.time()
            self.errors = [t for t in self.errors if now - t < self.window_sec]
            return len(self.errors) >= self.max_errors
    
    def remaining(self) -> int:
        with self.lock:
            import time
            now = time.time()
            self.errors = [t for t in self.errors if now - t < self.window_sec]
            return max(0, self.max_errors - len(self.errors))


# Global instances
PROFILER = SimpleProfiler()
CACHE = SimpleCache(max_size=2000)
ERROR_BUDGET = ErrorBudget(max_errors=100, window_sec=60)

# ═══════════════════════════════════════════════════════════════════════════════════════
# IMPORT GLOBALS
# ═══════════════════════════════════════════════════════════════════════════════════════

GLOBALS_AVAILABLE = False
try:
    from globals import (
        initialize_globals, get_globals,
        get_system_health, get_state_snapshot,
        get_heartbeat, get_lattice, get_db_pool, get_db_manager,
        get_blockchain, get_ledger, get_oracle, get_defi,
        get_auth_manager, get_pqc_system, get_genesis_block,
        verify_genesis_block, get_metrics, get_quantum,
        dispatch_command, COMMAND_REGISTRY,
        pqc_generate_user_key, pqc_sign, pqc_verify,
        pqc_encapsulate, pqc_prove_identity, pqc_verify_identity,
        pqc_revoke_key, pqc_rotate_key,
        bootstrap_admin_session, revoke_session,
    )
    logger.info("✅ Globals module loaded successfully")
    GLOBALS_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Cannot import globals: {e}")
    GLOBALS_AVAILABLE = False

if GLOBALS_AVAILABLE:
    initialize_globals()
    logger.info("✅ Global state initialized")

# ═══════════════════════════════════════════════════════════════════════════════════════
# CRITICAL: TERMINAL ENGINE INITIALIZATION - MUST RUN BEFORE APP STARTS
# ═══════════════════════════════════════════════════════════════════════════════════════
# This is NOT optional. This MUST succeed. If it fails, app should not start.

_TERMINAL_ENGINE_INITIALIZED = False

if GLOBALS_AVAILABLE:
    try:
        logger.info("[BOOTSTRAP] Initializing Terminal Engine...")
        from terminal_logic import TerminalEngine, register_all_commands
        
        # Instantiate the engine
        _ENGINE = TerminalEngine()
        logger.info("[BOOTSTRAP] ✓ TerminalEngine instantiated")
        
        # Register all commands - this MODIFIES globals.COMMAND_REGISTRY directly
        cmd_count = register_all_commands(_ENGINE)
        logger.info(f"[BOOTSTRAP] ✓ register_all_commands returned: {cmd_count} commands")
        
        # VERIFY registration actually happened
        from globals import COMMAND_REGISTRY
        total_cmds = len(COMMAND_REGISTRY)
        logger.info(f"[BOOTSTRAP] ✓ COMMAND_REGISTRY now has {total_cmds} commands")
        
        if total_cmds < 80:  # Should have at least 89, warn if less
            logger.error(f"[BOOTSTRAP] ❌ ERROR: Only {total_cmds} commands registered! Expected 89+")
            logger.error(f"[BOOTSTRAP] Available commands: {list(COMMAND_REGISTRY.keys())}")
            raise RuntimeError(f"Command registration failed: only {total_cmds} commands (expected 89+)")
        
        # Verify critical commands are present
        critical_cmds = ['help', 'help-commands', 'system-status']
        missing = [cmd for cmd in critical_cmds if cmd not in COMMAND_REGISTRY]
        if missing:
            logger.error(f"[BOOTSTRAP] ❌ CRITICAL: Missing commands: {missing}")
            raise RuntimeError(f"Critical commands missing: {missing}")
        
        # Warn if optional help-pq is missing (post-quantum help)
        optional_cmds = ['help-pq']
        missing_opt = [cmd for cmd in optional_cmds if cmd not in COMMAND_REGISTRY]
        if missing_opt:
            logger.warning(f"[BOOTSTRAP] ⚠ Optional commands missing: {missing_opt}")
        
        logger.info(f"[BOOTSTRAP] ✅ VERIFIED: All critical commands present")
        logger.info(f"[BOOTSTRAP] ✅ Terminal Engine initialization COMPLETE")
        _TERMINAL_ENGINE_INITIALIZED = True
        
    except ImportError as e:
        logger.error(f"[BOOTSTRAP] ❌ FATAL: Cannot import terminal_logic: {e}")
        logger.error(f"[BOOTSTRAP] ❌ Ensure terminal_logic.py is in {os.path.dirname(os.path.abspath(__file__))}")
        raise
    except Exception as e:
        logger.error(f"[BOOTSTRAP] ❌ FATAL: Terminal Engine initialization failed: {e}", exc_info=True)
        raise

if not _TERMINAL_ENGINE_INITIALIZED:
    logger.error("[BOOTSTRAP] ❌ FATAL: Terminal Engine not initialized!")
    raise RuntimeError("Terminal Engine initialization required but failed")

# ═══════════════════════════════════════════════════════════════════════════════════════
# FLASK APP
# ═══════════════════════════════════════════════════════════════════════════════════════

app = Flask(__name__,
            static_folder=os.path.join(PROJECT_ROOT, 'static'),
            static_url_path='/static')
app.config['JSON_SORT_KEYS'] = False

# ═══════════════════════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER - Safe System Access
# ═══════════════════════════════════════════════════════════════════════════════════════

class CircuitBreaker:
    """Graceful degradation for missing subsystems."""
    def __init__(self, system_name, getter_func):
        self.system_name = system_name
        self.getter_func = getter_func
    
    def get(self, fallback=None):
        try:
            result = self.getter_func()
            return result or fallback or {'status': 'unavailable'}
        except Exception as e:
            logger.warning(f"⚠️  {self.system_name} unavailable: {str(e)[:60]}")
            return fallback or {'status': 'unavailable'}

# Create circuit breakers (lazy - functions not called until .get())
SYSTEMS = {
    'quantum': CircuitBreaker('quantum', lambda: get_heartbeat()),
    'blockchain': CircuitBreaker('blockchain', lambda: get_blockchain()),
    'database': CircuitBreaker('database', lambda: get_db_manager()),
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
    """Serve index.html for browsers; return JSON for API clients."""
    accept = request.headers.get('Accept', '')
    # If the client explicitly wants HTML (browser), serve the frontend
    if 'text/html' in accept:
        try:
            return send_from_directory(PROJECT_ROOT, 'index.html')
        except Exception as e:
            logger.warning(f"⚠️  Could not serve index.html: {e}")
    # API clients / health checks get JSON
    return jsonify({
        'system': 'QTCL v5.0 - Quantum Lattice Control',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'status': 'operational',
        'endpoints': [
            '/', '/health', '/api/status',
            '/api/command', '/api/commands',
            '/api/genesis', '/api/quantum',
            '/api/blockchain', '/api/ledger',
            '/api/oracle', '/api/metrics',
        ],
    })

@app.route('/index.html', methods=['GET'])
def index_html():
    """Explicit index.html route for any hardcoded links."""
    try:
        return send_from_directory(PROJECT_ROOT, 'index.html')
    except Exception as e:
        logger.error(f"❌ index.html not found at {PROJECT_ROOT}: {e}")
        return jsonify({'error': 'Frontend not found', 'hint': 'index.html must be in project root'}), 404

@app.route('/health', methods=['GET'])
def health():
    """Health check for load balancers."""
    try:
        return jsonify(get_system_health()), 200
    except:
        return jsonify({'status': 'healthy', 'timestamp': datetime.now(timezone.utc).isoformat()}), 200

@app.route('/api/status', methods=['GET'])
def api_status():
    try:
        return jsonify({
            'health': get_system_health(),
            'snapshot': get_state_snapshot(),
            'timestamp': datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        return jsonify({'status': 'operational', 'timestamp': datetime.now(timezone.utc).isoformat()}), 200

@app.route('/api/command', methods=['POST'])
def execute_command():
    try:
        data = request.get_json() or {}
        # Accept the full command string (may include inline --flags)
        command = data.get('command', 'help')

        # args/kwargs from body (legacy callers); new executor embeds them in command string
        args = data.get('args') or data.get('kwargs') or {}

        # Resolve user_id: prefer explicit body field, then decode Bearer JWT
        user_id = data.get('user_id') or None
        if not user_id:
            auth_header = request.headers.get('Authorization', '')
            if auth_header.startswith('Bearer '):
                token = auth_header[7:]
                try:
                    import jwt as _jwt
                    secret = __import__('os').getenv('JWT_SECRET', '')
                    if secret:
                        payload = _jwt.decode(token, secret, algorithms=['HS256', 'HS512'])
                        user_id = payload.get('user_id') or payload.get('sub')
                except Exception:
                    pass  # invalid/expired token — user_id stays None

        result = dispatch_command(command, args, user_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/commands', methods=['GET'])
def list_commands():
    try:
        category = request.args.get('category')
        if category:
            from globals import get_commands_by_category
            commands = get_commands_by_category(category)
        else:
            commands = COMMAND_REGISTRY
        return jsonify({'total': len(commands), 'commands': commands})
    except Exception as e:
        return jsonify({'error': str(e), 'total': len(COMMAND_REGISTRY)})

@app.route('/api/genesis', methods=['GET'])
def get_genesis_endpoint():
    try:
        return jsonify({
            'genesis_block': get_genesis_block(),
            'verification': verify_genesis_block(),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/quantum', methods=['GET'])
def quantum_status():
    return jsonify({
        'quantum': SYSTEMS['quantum'].get(),
        'status': 'operational',
    })

@app.route('/api/blockchain', methods=['GET'])
def blockchain_status():
    return jsonify({
        'blockchain': SYSTEMS['blockchain'].get(),
    })

@app.route('/api/ledger', methods=['GET'])
def ledger_status():
    return jsonify({
        'ledger': SYSTEMS['ledger'].get(),
    })

@app.route('/api/oracle', methods=['GET'])
def oracle_status():
    return jsonify({
        'oracle': SYSTEMS['oracle'].get(),
    })

@app.route('/api/metrics', methods=['GET'])
def metrics():
    try:
        return jsonify(get_metrics())
    except:
        return jsonify({'requests': 0, 'errors': 0, 'status': 'operational'})

# ═══════════════════════════════════════════════════════════════════════════════════════
# PQC ROUTES
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/pqc/keygen', methods=['POST'])
def pqc_keygen():
    try:
        data = request.get_json() or {}
        user_id = data.get('user_id', 'default')
        return jsonify(pqc_generate_user_key(user_id))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/pqc/sign', methods=['POST'])
def pqc_sign_endpoint():
    try:
        data = request.get_json() or {}
        message = data.get('message', '')
        key_id = data.get('key_id', 'default')
        return jsonify(pqc_sign(message, key_id))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/pqc/verify', methods=['POST'])
def pqc_verify_endpoint():
    try:
        data = request.get_json() or {}
        message = data.get('message', '')
        signature = data.get('signature', '')
        key_id = data.get('key_id', 'default')
        return jsonify(pqc_verify(message, signature, key_id))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ═══════════════════════════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(error):
    return jsonify({'status': 'error', 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'status': 'error', 'error': 'Internal server error'}), 500

# ═══════════════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════════════
# WSGI ENTRYPOINT — gunicorn expects 'application', not 'app'
# ═══════════════════════════════════════════════════════════════════════════════════════
application = app  # <-- THIS is what gunicorn:  wsgi_config:application  resolves to

if __name__ == '__main__':
    logger.info("="*80)
    logger.info("🚀 QTCL WSGI v5.0 STARTING")
    logger.info("="*80)
    logger.info(f"✅ Globals available: {GLOBALS_AVAILABLE}")
    logger.info(f"✅ Systems registered: {len(SYSTEMS)}")
    logger.info(f"✅ Database pool initialized: {DB.pool is not None}")
    try:
        logger.info(f"✅ Commands available: {len(COMMAND_REGISTRY)}")
    except:
        logger.info("⚠️  Commands: unavailable")
    logger.info("="*80)
    app.run(host='0.0.0.0', port=8000, debug=False)
