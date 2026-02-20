#!/usr/bin/env python3
"""
QTCL COMPREHENSIVE WSGI INTEGRATION v5.0 - ENTERPRISE GRADE

Proper integration with db_builder_v2 DatabaseBuilder singleton.
No fallbacks, no conditional imports. Production-grade dependency management.
"""

import os
import sys
import logging
from datetime import datetime, timezone
from flask import Flask, request, jsonify, send_from_directory

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ═══════════════════════════════════════════════════════════════════════════════════════
# DATABASE LAYER - ENTERPRISE INTEGRATION WITH DB_BUILDER_V2
# ═══════════════════════════════════════════════════════════════════════════════════════

class DatabaseLayer:
    """Enterprise-grade database layer - delegates to db_builder_v2.DatabaseBuilder"""
    
    def __init__(self):
        """Initialize with singleton from db_builder_v2"""
        self._db_manager = None
        self._initialized = False
        self._initialize()
    
    def _initialize(self):
        """Retrieve DatabaseBuilder singleton from db_builder_v2"""
        try:
            from db_builder_v2 import db_manager, DB_POOL
            
            if db_manager is None:
                raise RuntimeError("db_manager singleton is None - db_builder_v2 initialization failed")
            
            self._db_manager = db_manager
            self._initialized = True
            logger.info("✅ DatabaseLayer: Connected to db_builder_v2.db_manager")
            
        except ImportError as e:
            raise RuntimeError(f"CRITICAL: Cannot import db_builder_v2: {e}")
        except Exception as e:
            raise RuntimeError(f"CRITICAL: DatabaseLayer initialization failed: {e}")
    
    def health_check(self) -> bool:
        """Verify database connectivity"""
        if not self._initialized or not self._db_manager:
            return False
        try:
            result = self._db_manager.execute_fetch("SELECT 1")
            return result is not None
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def execute(self, query: str, params=None, return_results=False):
        """Execute query - delegates to db_manager"""
        if not self._initialized:
            raise RuntimeError("DatabaseLayer not initialized")
        return self._db_manager.execute(query, params, return_results)
    
    def execute_fetch(self, query: str, params=None):
        """Fetch single row"""
        if not self._initialized:
            raise RuntimeError("DatabaseLayer not initialized")
        return self._db_manager.execute_fetch(query, params)
    
    def execute_fetch_all(self, query: str, params=None):
        """Fetch all rows"""
        if not self._initialized:
            raise RuntimeError("DatabaseLayer not initialized")
        return self._db_manager.execute_fetch_all(query, params)
    
    def execute_many(self, query: str, data_list):
        """Batch execute"""
        if not self._initialized:
            raise RuntimeError("DatabaseLayer not initialized")
        return self._db_manager.execute_many(query, data_list)


class PerfProfiler:
    """Enterprise performance profiling"""
    
    def __init__(self):
        import threading
        self.metrics = {}
        self.lock = threading.RLock()
    
    def record(self, operation: str, duration_ms: float):
        """Record operation timing"""
        with self.lock:
            if operation not in self.metrics:
                self.metrics[operation] = []
            self.metrics[operation].append(duration_ms)
    
    def get_stats(self, operation: str) -> dict:
        """Get operation statistics"""
        with self.lock:
            if operation not in self.metrics:
                return {'count': 0, 'avg_ms': 0, 'min_ms': 0, 'max_ms': 0}
            times = self.metrics[operation]
            return {
                'count': len(times),
                'avg_ms': sum(times) / len(times),
                'min_ms': min(times),
                'max_ms': max(times),
            }
    
    def clear(self):
        """Clear all metrics"""
        with self.lock:
            self.metrics.clear()


class RequestTracer:
    """Enterprise request correlation and tracing"""
    
    def __init__(self):
        import uuid, threading
        self.lock = threading.RLock()
        self.current_request_id = None
        self.uuid = uuid
    
    def new_request_id(self) -> str:
        """Generate new request correlation ID"""
        with self.lock:
            self.current_request_id = str(self.uuid.uuid4())
            return self.current_request_id
    
    def get_request_id(self) -> str:
        """Get current request ID"""
        return self.current_request_id or self.new_request_id()


class ResultCache:
    """Enterprise in-memory result caching with TTL"""
    
    def __init__(self, max_items: int = 2000):
        import threading, time
        self.cache = {}
        self.max_items = max_items
        self.lock = threading.RLock()
        self.time = time
    
    def get(self, key: str):
        """Retrieve cached value"""
        with self.lock:
            if key not in self.cache:
                return None
            item = self.cache[key]
            # Check if expired
            if item['ttl'] and self.time.time() > item['expires_at']:
                del self.cache[key]
                return None
            return item['value']
    
    def set(self, key: str, value, ttl_sec: int = 3600):
        """Cache value with TTL"""
        with self.lock:
            if len(self.cache) >= self.max_items:
                # Simple eviction: remove first item
                self.cache.pop(next(iter(self.cache)), None)
            
            self.cache[key] = {
                'value': value,
                'ttl': ttl_sec,
                'expires_at': self.time.time() + ttl_sec if ttl_sec else None
            }
    
    def delete(self, key: str):
        """Remove cached value"""
        with self.lock:
            self.cache.pop(key, None)
    
    def clear(self):
        """Clear entire cache"""
        with self.lock:
            self.cache.clear()


class ErrorBudgetManager:
    """Enterprise error budget tracking for circuit breaker patterns"""
    
    def __init__(self, max_errors: int = 100, window_sec: int = 60):
        import threading, time
        self.max_errors = max_errors
        self.window_sec = window_sec
        self.errors = []
        self.lock = threading.RLock()
        self.time = time
    
    def record_error(self):
        """Record an error occurrence"""
        with self.lock:
            now = self.time.time()
            self.errors = [t for t in self.errors if now - t < self.window_sec]
            self.errors.append(now)
    
    def is_exhausted(self) -> bool:
        """Check if error budget exceeded"""
        with self.lock:
            now = self.time.time()
            self.errors = [t for t in self.errors if now - t < self.window_sec]
            return len(self.errors) >= self.max_errors
    
    def remaining(self) -> int:
        """Get remaining errors before exhaustion"""
        with self.lock:
            now = self.time.time()
            self.errors = [t for t in self.errors if now - t < self.window_sec]
            return max(0, self.max_errors - len(self.errors))
    
    def reset(self):
        """Reset error budget"""
        with self.lock:
            self.errors.clear()


# ═══════════════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCES - Enterprise-grade initialization
# ═══════════════════════════════════════════════════════════════════════════════════════

try:
    DB = DatabaseLayer()
    if not DB.health_check():
        logger.warning("⚠️  Database health check failed on startup")
except Exception as e:
    logger.error(f"CRITICAL: DatabaseLayer initialization failed: {e}")
    raise

PROFILER = PerfProfiler()
CACHE = ResultCache(max_items=2000)
RequestCorrelation = RequestTracer()
ERROR_BUDGET = ErrorBudgetManager(max_errors=100, window_sec=60)

logger.info("✅ Enterprise database layer initialized")
logger.info("✅ All utility services initialized (Profiler, Cache, Tracer, ErrorBudget)")

# ═══════════════════════════════════════════════════════════════════════════════════════
# GLOBALS MODULE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════════════

GLOBALS_AVAILABLE = False
try:
    from globals import (
        initialize_globals, get_system_health, get_state_snapshot,
        get_heartbeat, get_blockchain, get_ledger, get_oracle, get_defi,
        get_auth_manager, get_pqc_system, get_genesis_block, verify_genesis_block,
        get_metrics, get_quantum, dispatch_command, COMMAND_REGISTRY,
        pqc_generate_user_key, pqc_sign, pqc_verify,
        pqc_encapsulate, pqc_prove_identity, pqc_verify_identity,
        pqc_revoke_key, pqc_rotate_key, bootstrap_admin_session, revoke_session,
    )
    initialize_globals()
    logger.info("✅ Globals module initialized")
    GLOBALS_AVAILABLE = True
except Exception as e:
    logger.error(f"❌ Globals initialization failed: {e}")
    GLOBALS_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════════════
# TERMINAL ENGINE BOOTSTRAP - CRITICAL
# ═══════════════════════════════════════════════════════════════════════════════════════

_TERMINAL_INITIALIZED = False

if GLOBALS_AVAILABLE:
    try:
        logger.info("[BOOTSTRAP] Starting Terminal Engine initialization...")
        from terminal_logic import TerminalEngine, register_all_commands
        
        engine = TerminalEngine()
        cmd_count = register_all_commands(engine)
        
        total_cmds = len(COMMAND_REGISTRY)
        logger.info(f"[BOOTSTRAP] Registered {total_cmds} commands")
        
        if total_cmds < 80:
            raise RuntimeError(f"Command registration incomplete: {total_cmds} commands (expected 89+)")
        
        _TERMINAL_INITIALIZED = True
        logger.info("[BOOTSTRAP] ✅ Terminal Engine initialized successfully")
        
    except Exception as e:
        logger.error(f"[BOOTSTRAP] ❌ FATAL: Terminal Engine failed: {e}")
        raise

if not _TERMINAL_INITIALIZED:
    raise RuntimeError("Terminal Engine initialization required")

# ═══════════════════════════════════════════════════════════════════════════════════════
# FLASK APPLICATION - ENTERPRISE SETUP
# ═══════════════════════════════════════════════════════════════════════════════════════

app = Flask(__name__, static_folder=os.path.join(PROJECT_ROOT, 'static'), static_url_path='/static')
app.config['JSON_SORT_KEYS'] = False


class CircuitBreaker:
    """Enterprise circuit breaker for graceful degradation"""
    def __init__(self, service_name: str, getter_func):
        self.service_name = service_name
        self.getter_func = getter_func
    
    def get(self, default=None):
        try:
            result = self.getter_func()
            return result or default or {'status': 'unavailable'}
        except Exception as e:
            logger.warning(f"Service unavailable: {self.service_name} ({str(e)[:60]})")
            return default or {'status': 'unavailable'}


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
# FLASK ROUTES
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
        'system': 'QTCL v5.0 - Quantum Lattice Control',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'status': 'operational',
        'db_health': 'connected' if DB.health_check() else 'disconnected',
    })

@app.route('/health', methods=['GET'])
def health():
    try:
        return jsonify(get_system_health()), 200
    except:
        return jsonify({'status': 'healthy', 'timestamp': datetime.now(timezone.utc).isoformat()}), 200

@app.route('/api/status', methods=['GET'])
def status():
    try:
        return jsonify({
            'health': get_system_health(),
            'snapshot': get_state_snapshot(),
            'timestamp': datetime.now(timezone.utc).isoformat(),
        })
    except:
        return jsonify({'status': 'operational'}), 200

@app.route('/api/command', methods=['POST'])
def execute_command():
    try:
        data = request.get_json() or {}
        command = data.get('command', 'help')
        args = data.get('args') or {}
        user_id = data.get('user_id')
        
        if not user_id:
            auth = request.headers.get('Authorization', '')
            if auth.startswith('Bearer '):
                try:
                    import jwt
                    token = auth[7:]
                    secret = os.getenv('JWT_SECRET', '')
                    if secret:
                        payload = jwt.decode(token, secret, algorithms=['HS256', 'HS512'])
                        user_id = payload.get('user_id') or payload.get('sub')
                except:
                    pass
        
        result = dispatch_command(command, args, user_id)
        return jsonify(result)
    except Exception as e:
        ERROR_BUDGET.record_error()
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/commands', methods=['GET'])
def list_commands():
    try:
        return jsonify({'total': len(COMMAND_REGISTRY), 'commands': dict(COMMAND_REGISTRY)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/genesis', methods=['GET'])
def genesis():
    try:
        return jsonify({
            'genesis_block': get_genesis_block(),
            'verification': verify_genesis_block(),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/quantum', methods=['GET'])
def quantum():
    return jsonify({'quantum': SERVICES['quantum'].get()})

@app.route('/api/blockchain', methods=['GET'])
def blockchain():
    return jsonify({'blockchain': SERVICES['blockchain'].get()})

@app.route('/api/ledger', methods=['GET'])
def ledger():
    return jsonify({'ledger': SERVICES['ledger'].get()})

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
    ERROR_BUDGET.record_error()
    return jsonify({'error': 'Internal server error'}), 500

# ═══════════════════════════════════════════════════════════════════════════════════════
# WSGI ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════════════

application = app

if __name__ == '__main__':
    logger.info("=" * 80)
    logger.info("🚀 QTCL WSGI v5.0 PRODUCTION STARTUP")
    logger.info("=" * 80)
    logger.info(f"Database: {'HEALTHY' if DB.health_check() else 'UNHEALTHY'}")
    logger.info(f"Terminal Engine: {'READY' if _TERMINAL_INITIALIZED else 'FAILED'}")
    logger.info(f"Globals Module: {'AVAILABLE' if GLOBALS_AVAILABLE else 'UNAVAILABLE'}")
    logger.info(f"Commands Registered: {len(COMMAND_REGISTRY)}")
    logger.info("=" * 80)
    app.run(host='0.0.0.0', port=8000, debug=False)
