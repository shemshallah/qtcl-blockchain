#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                        ║
║                     🚀 QTCL COMPREHENSIVE WSGI INTEGRATION v5.0 🚀                    ║
║                                                                                        ║
║  World-class system integration with proper circuit breakers and graceful degradation ║
║  Integrates all 10 subsystems into single coherent Flask application                 ║
║                                                                                        ║
╚════════════════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime, timezone
from flask import Flask, request, jsonify, Response

# ═══════════════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ═══════════════════════════════════════════════════════════════════════════════════════
# INITIALIZE GLOBALS
# ═══════════════════════════════════════════════════════════════════════════════════════

try:
    from globals_COMPREHENSIVE import (
        initialize_globals, get_globals,
        get_system_health, get_state_snapshot,
        get_heartbeat, get_lattice, get_quantum_coordinator,
        get_db_manager, get_blockchain, get_ledger, get_oracle, get_defi,
        get_auth_manager, get_pqc_system, get_admin_system, get_terminal_engine,
        get_genesis_block, verify_genesis_block, get_metrics,
        dispatch_command, COMMAND_REGISTRY,
        pqc_generate_user_key, pqc_sign, pqc_verify,
        pqc_encapsulate, pqc_prove_identity, pqc_verify_identity,
        pqc_revoke_key, pqc_rotate_key,
        bootstrap_admin_session, revoke_session,
    )
    GLOBALS_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ CRITICAL: Cannot import globals: {e}")
    GLOBALS_AVAILABLE = False

# Initialize global state
if GLOBALS_AVAILABLE:
    initialize_globals()
    logger.info("✅ Global state initialized with all subsystems")

# ═══════════════════════════════════════════════════════════════════════════════════════
# FLASK APP SETUP
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
            return fallback or {'status': 'unavailable', 'error': str(e)[:100]}

# Create circuit breakers for all systems
SYSTEMS = {
    'quantum': CircuitBreaker('quantum', get_heartbeat),
    'blockchain': CircuitBreaker('blockchain', get_blockchain),
    'database': CircuitBreaker('database', get_db_manager),
    'ledger': CircuitBreaker('ledger', get_ledger),
    'oracle': CircuitBreaker('oracle', get_oracle),
    'defi': CircuitBreaker('defi', get_defi),
    'auth': CircuitBreaker('auth', get_auth_manager),
    'pqc': CircuitBreaker('pqc', get_pqc_system),
    'admin': CircuitBreaker('admin', get_admin_system),
    'terminal': CircuitBreaker('terminal', get_terminal_engine),
}

# ═══════════════════════════════════════════════════════════════════════════════════════
# ROOT ROUTES
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/', methods=['GET'])
def index():
    """Home route with system status."""
    return jsonify({
        'system': 'QTCL v5.0 - Quantum Lattice Control',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'status': 'operational',
        'endpoints': {
            '/health': 'System health check',
            '/api/status': 'Detailed system status',
            '/api/command': 'Execute command',
            '/api/commands': 'List all commands',
            '/api/genesis': 'Get genesis block',
        },
        'subsystems': list(SYSTEMS.keys()),
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check for load balancers."""
    health_status = get_system_health()
    return jsonify(health_status), 200

@app.route('/api/status', methods=['GET'])
def api_status():
    """Detailed system status."""
    return jsonify({
        'health': get_system_health(),
        'snapshot': get_state_snapshot(),
        'timestamp': datetime.now(timezone.utc).isoformat(),
    })

# ═══════════════════════════════════════════════════════════════════════════════════════
# COMMAND EXECUTION ROUTES
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/command', methods=['POST'])
def execute_command():
    """Execute command via API."""
    try:
        data = request.get_json() or {}
        command = data.get('command', 'help')
        args = data.get('args', {})
        user_id = data.get('user_id')
        
        result = dispatch_command(command, args, user_id)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Command execution error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc(),
        }), 500

@app.route('/api/commands', methods=['GET'])
def list_commands():
    """List all available commands."""
    category = request.args.get('category')
    
    if category:
        from globals_COMPREHENSIVE import get_commands_by_category
        commands = get_commands_by_category(category)
    else:
        commands = COMMAND_REGISTRY
    
    return jsonify({
        'total': len(commands),
        'commands': commands,
    })

# ═══════════════════════════════════════════════════════════════════════════════════════
# SYSTEM STATUS ROUTES
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/genesis', methods=['GET'])
def get_genesis_endpoint():
    """Get genesis block."""
    genesis = get_genesis_block()
    verification = verify_genesis_block()
    return jsonify({
        'genesis_block': genesis,
        'verification': verification,
    })

@app.route('/api/quantum', methods=['GET'])
def quantum_status():
    """Quantum system status."""
    heartbeat = SYSTEMS['quantum'].get({'status': 'offline'})
    return jsonify({
        'heartbeat': heartbeat,
        'lattice': SYSTEMS['quantum'].get({'status': 'offline'}),
        'status': 'operational' if heartbeat.get('status') == 'running' else 'offline',
    })

@app.route('/api/blockchain', methods=['GET'])
def blockchain_status():
    """Blockchain status."""
    blockchain = SYSTEMS['blockchain'].get()
    return jsonify({
        'blockchain': blockchain,
        'height': blockchain.get('height', 0) if blockchain else 0,
        'genesis_hash': blockchain.get('genesis_hash') if blockchain else None,
    })

@app.route('/api/ledger', methods=['GET'])
def ledger_status():
    """Ledger status."""
    ledger = SYSTEMS['ledger'].get()
    return jsonify({
        'ledger': ledger,
        'status': 'operational' if ledger else 'offline',
    })

@app.route('/api/oracle', methods=['GET'])
def oracle_status():
    """Oracle system status."""
    oracle = SYSTEMS['oracle'].get()
    return jsonify({
        'oracle': oracle,
        'status': 'operational' if oracle else 'offline',
    })

@app.route('/api/metrics', methods=['GET'])
def metrics():
    """System metrics."""
    return jsonify(get_metrics())

# ═══════════════════════════════════════════════════════════════════════════════════════
# PQC ROUTES
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.route('/api/pqc/keygen', methods=['POST'])
def pqc_keygen():
    """Generate PQC key."""
    data = request.get_json() or {}
    user_id = data.get('user_id', 'default')
    
    key = pqc_generate_user_key(user_id)
    return jsonify(key)

@app.route('/api/pqc/sign', methods=['POST'])
def pqc_sign_endpoint():
    """Sign message with PQC."""
    data = request.get_json() or {}
    message = data.get('message', '')
    key_id = data.get('key_id', 'default')
    
    signature = pqc_sign(message, key_id)
    return jsonify(signature)

@app.route('/api/pqc/verify', methods=['POST'])
def pqc_verify_endpoint():
    """Verify PQC signature."""
    data = request.get_json() or {}
    message = data.get('message', '')
    signature = data.get('signature', '')
    key_id = data.get('key_id', 'default')
    
    result = pqc_verify(message, signature, key_id)
    return jsonify(result)

# ═══════════════════════════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'error': 'Endpoint not found',
        'available_endpoints': [
            '/', '/health', '/api/status',
            '/api/command', '/api/commands',
            '/api/genesis', '/api/quantum',
            '/api/blockchain', '/api/ledger',
            '/api/oracle', '/api/metrics',
            '/api/pqc/keygen', '/api/pqc/sign', '/api/pqc/verify',
        ]
    }), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({
        'status': 'error',
        'error': 'Internal server error',
        'message': str(error),
    }), 500

# ═══════════════════════════════════════════════════════════════════════════════════════
# WSGI EXPORT
# ═══════════════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 QTCL COMPREHENSIVE WSGI v5.0 - STARTING")
    print("="*80)
    print(f"✅ Globals available: {GLOBALS_AVAILABLE}")
    print(f"✅ Systems registered: {len(SYSTEMS)}")
    print(f"✅ Commands available: {len(COMMAND_REGISTRY)}")
    print(f"✅ Flask app ready at http://0.0.0.0:8000")
    print("="*80 + "\n")
    
    app.run(host='0.0.0.0', port=8000, debug=False)
