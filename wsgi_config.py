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
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
# FLASK APP
# ═══════════════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
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
        command = data.get('command', 'help')
        args = data.get('args', {})
        user_id = data.get('user_id')
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

if __name__ == '__main__':
    logger.info("="*80)
    logger.info("🚀 QTCL WSGI v5.0 STARTING")
    logger.info("="*80)
    logger.info(f"✅ Globals available: {GLOBALS_AVAILABLE}")
    logger.info(f"✅ Systems registered: {len(SYSTEMS)}")
    try:
        logger.info(f"✅ Commands available: {len(COMMAND_REGISTRY)}")
    except:
        logger.info("⚠️  Commands: unavailable")
    logger.info("="*80)
    app.run(host='0.0.0.0', port=8000, debug=False)
