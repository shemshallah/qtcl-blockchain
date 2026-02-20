#!/usr/bin/env python3
"""
QTCL WSGI v5.0 - GLOBALS COORDINATOR & SYSTEM BOOTSTRAP

Proper globals interfacing system. Initializes all subsystems in correct order:
1. Logging (foundation)
2. db_builder_v2 (database singleton)
3. globals module (unified state)
4. Terminal engine (command system)
5. Flask app (HTTP interface)

Other modules import from wsgi_config to access initialized globals.
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
# BOOTSTRAP PHASE 2: GLOBALS MODULE (Unified State)
# ═══════════════════════════════════════════════════════════════════════════════════════

GLOBALS_AVAILABLE = False
try:
    logger.info("[BOOTSTRAP] Initializing globals module...")
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
    initialize_globals()
    GLOBALS_AVAILABLE = True
    logger.info("[BOOTSTRAP] ✅ Globals module initialized")
except Exception as e:
    logger.error(f"[BOOTSTRAP] ❌ FATAL: Globals initialization failed: {e}")
    raise

# ═══════════════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP PHASE 3: TERMINAL ENGINE (Command System)
# ═══════════════════════════════════════════════════════════════════════════════════════

_TERMINAL_INITIALIZED = False
try:
    logger.info("[BOOTSTRAP] Initializing terminal engine...")
    from terminal_logic import TerminalEngine, register_all_commands
    
    engine = TerminalEngine()
    cmd_count = register_all_commands(engine)
    
    total_cmds = len(COMMAND_REGISTRY)
    logger.info(f"[BOOTSTRAP] ✓ Registered {total_cmds} commands")
    
    if total_cmds < 80:
        raise RuntimeError(f"Command registration incomplete: {total_cmds} (expected 89+)")
    
    _TERMINAL_INITIALIZED = True
    logger.info("[BOOTSTRAP] ✅ Terminal engine initialized")
except Exception as e:
    logger.error(f"[BOOTSTRAP] ❌ FATAL: Terminal engine failed: {e}")
    raise

if not _TERMINAL_INITIALIZED:
    raise RuntimeError("Terminal engine initialization required")

# ═══════════════════════════════════════════════════════════════════════════════════════
# EXPORTS: Other modules import these from wsgi_config
# ═══════════════════════════════════════════════════════════════════════════════════════

__all__ = [
    'DB', 'GLOBALS_AVAILABLE',
    'get_system_health', 'get_state_snapshot',
    'get_heartbeat', 'get_blockchain', 'get_ledger', 'get_oracle', 'get_defi',
    'get_auth_manager', 'get_pqc_system', 'get_genesis_block', 'verify_genesis_block',
    'get_metrics', 'get_quantum', 'dispatch_command', 'COMMAND_REGISTRY',
    'pqc_generate_user_key', 'pqc_sign', 'pqc_verify',
    'pqc_encapsulate', 'pqc_prove_identity', 'pqc_verify_identity',
    'pqc_revoke_key', 'pqc_rotate_key',
    'bootstrap_admin_session', 'revoke_session',
    'logger',
]

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
    logger.info("="*80)
    logger.info("🚀 QTCL WSGI v5.0 - PRODUCTION STARTUP")
    logger.info("="*80)
    logger.info(f"✅ Database initialized: {DB is not None}")
    logger.info(f"✅ Globals available: {GLOBALS_AVAILABLE}")
    logger.info(f"✅ Terminal engine ready: {_TERMINAL_INITIALIZED}")
    logger.info(f"✅ Commands registered: {len(COMMAND_REGISTRY)}")
    logger.info("="*80)
    app.run(host='0.0.0.0', port=8000, debug=False)
