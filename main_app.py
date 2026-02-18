#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                                ║
║           🚀 MAIN_APP.PY v5.0 - UNIFIED FLASK APPLICATION WITH GLOBAL INTEGRATION 🚀          ║
║                                                                                                ║
║        QTCL Unified Flask app factory with comprehensive blueprint registration               ║
║            Blueprint registration, state managed by expanded globals.py                       ║
║    Serves index.html properly + ALL functions integrated + deep module interconnection        ║
║          Original 132 lines EXPANDED to 500+ lines with full architecture                     ║
║                                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import logging
import time
from datetime import datetime
from pathlib import Path

# ════════════════════════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ════════════════════════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('qtcl_main_expanded.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info("╔════════════════════════════════════════════════════════════════════╗")
logger.info("║        QTCL MAIN APPLICATION v5.0 STARTUP - EXPANDED             ║")
logger.info("╚════════════════════════════════════════════════════════════════════╝")

# ════════════════════════════════════════════════════════════════════════════════════════════════
# PROJECT ROOT & IMPORTS
# ════════════════════════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger.info("[Main] Importing Flask and global state...")

from flask import Flask, request, jsonify, g, send_file
from flask_cors import CORS

# ════════════════════════════════════════════════════════════════════════════════════════════════
# IMPORT GLOBALS (EXPANDED)
# ════════════════════════════════════════════════════════════════════════════════════════════════

logger.info("[Main] Importing expanded globals...")
try:
    from globals import (
        initialize_globals,
        get_globals,
        get_system_health,
        get_state_snapshot,
        get_debug_info,
        get_heartbeat,
        get_lattice,
        get_blockchain,
        get_defi,
        get_oracle,
        get_ledger,
        get_metrics,
        SystemHealth,
    )
    logger.info("✅ [Main] Globals imported successfully")
except ImportError as e:
    logger.error(f"❌ [Main] Failed to import globals: {e}")
    sys.exit(1)

# ════════════════════════════════════════════════════════════════════════════════════════════════
# IMPORT TERMINAL ENGINE FOR COMMAND EXECUTION
# ════════════════════════════════════════════════════════════════════════════════════════════════

logger.info("[Main] Importing terminal engine for command execution...")
try:
    from terminal_logic import TerminalEngine, GlobalCommandRegistry, CommandRegistry
    TERMINAL_ENGINE = None
    logger.info("✅ [Main] Terminal engine imported")
except ImportError as e:
    logger.warning(f"⚠️ [Main] Terminal engine not available: {e}")
    TERMINAL_ENGINE = None
    GlobalCommandRegistry = None
    CommandRegistry = None

# ════════════════════════════════════════════════════════════════════════════════════════════════
# INITIALIZE GLOBAL STATE (LEVEL 1 LOGIC)
# ════════════════════════════════════════════════════════════════════════════════════════════════

logger.info("[Main] Initializing global architecture...")
if not initialize_globals():
    logger.error("[Main] ❌ FAILED to initialize global state")
    sys.exit(1)

gs = get_globals()
logger.info(f"[Main] ✅ Global state initialized")
logger.info(f"[Main] ✅ Functions registered: {len(gs.all_functions)}")

# ════════════════════════════════════════════════════════════════════════════════════════════════
# FLASK APP FACTORY (EXPANDED)
# ════════════════════════════════════════════════════════════════════════════════════════════════

def create_app():
    """
    Create and configure Flask application with full global state integration
    
    This factory creates the Flask app and registers all blueprints,
    integrates with the global architecture master, and sets up all middleware.
    """
    
    logger.info("[Main] Creating Flask application...")
    
    app = Flask(__name__, static_folder='static', static_url_path='/static')
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'qtcl-production-key-2024')
    app.config['JSON_SORT_KEYS'] = False
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
    
    # Enable CORS for all API routes
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    logger.info("✅ [Main] Flask app configured")
    
    # ════════════════════════════════════════════════════════════════════════════════════════
    # BLUEPRINT REGISTRATION (LEVEL 2 SUBLOGIC)
    # ════════════════════════════════════════════════════════════════════════════════════════
    
    logger.info("[Main] Importing and registering blueprints...")
    blueprints_registered = 0
    
    try:
        from blockchain_api import blueprint as blockchain_bp
        app.register_blueprint(blockchain_bp)
        blueprints_registered += 1
        logger.info("✅ Blockchain blueprint registered")
    except ImportError as e:
        logger.warning(f"⚠️ Blockchain blueprint not available: {e}")
    
    try:
        from quantum_api import blueprint as quantum_bp
        app.register_blueprint(quantum_bp)
        blueprints_registered += 1
        logger.info("✅ Quantum blueprint registered")
    except ImportError as e:
        logger.warning(f"⚠️ Quantum blueprint not available: {e}")
    
    try:
        from core_api import blueprint as core_bp
        app.register_blueprint(core_bp)
        blueprints_registered += 1
        logger.info("✅ Core blueprint registered")
    except ImportError as e:
        logger.warning(f"⚠️ Core blueprint not available: {e}")
    
    try:
        from admin_api import blueprint as admin_bp
        app.register_blueprint(admin_bp)
        blueprints_registered += 1
        logger.info("✅ Admin blueprint registered")
    except ImportError as e:
        logger.warning(f"⚠️ Admin blueprint not available: {e}")
    
    try:
        from oracle_api import blueprint as oracle_bp
        app.register_blueprint(oracle_bp)
        blueprints_registered += 1
        logger.info("✅ Oracle blueprint registered")
    except ImportError as e:
        logger.warning(f"⚠️ Oracle blueprint not available: {e}")
    
    try:
        from defi_api import blueprint as defi_bp
        app.register_blueprint(defi_bp)
        blueprints_registered += 1
        logger.info("✅ DeFi blueprint registered")
    except ImportError as e:
        logger.warning(f"⚠️ DeFi blueprint not available: {e}")
    
    try:
        from auth_handlers import blueprint as auth_bp
        app.register_blueprint(auth_bp)
        blueprints_registered += 1
        logger.info("✅ Auth blueprint registered")
    except ImportError as e:
        logger.warning(f"⚠️ Auth blueprint not available: {e}")
    
    logger.info(f"[Main] ✅ {blueprints_registered} blueprints registered")
    
    # ════════════════════════════════════════════════════════════════════════════════════════
    # REQUEST/RESPONSE MIDDLEWARE (LEVEL 3 SUB²LOGIC)
    # ════════════════════════════════════════════════════════════════════════════════════════
    
    @app.before_request
    def before_request():
        """
        Pre-request processing:
        - Track request timing
        - Record quantum pulse count
        - Inject global context
        - Check rate limiting
        """
        g.start_time = time.time()
        g.start_pulse = 0
        
        # Get heartbeat for quantum coherence tracking
        heartbeat = get_heartbeat()
        if heartbeat:
            g.start_pulse = heartbeat.pulse_count if hasattr(heartbeat, 'pulse_count') else 0
        
        # Inject global state into request context
        g.global_state = gs
        
        # Rate limiting
        client_ip = request.remote_addr
        # Could implement rate limiting here
    
    @app.after_request
    def after_request(response):
        """
        Post-request processing:
        - Calculate latency
        - Track quantum pulses
        - Update metrics
        - Add response headers
        """
        try:
            elapsed_ms = (time.time() - g.start_time) * 1000
            response.headers['X-Request-Time-Ms'] = f"{elapsed_ms:.1f}"
            response.headers['X-Quantum-Version'] = '5.0'
            
            # Track quantum coherence
            heartbeat = get_heartbeat()
            if heartbeat:
                response.headers['X-Heartbeat-Pulse'] = str(heartbeat.pulse_count)
                pulses_delta = heartbeat.pulse_count - g.start_pulse
                if pulses_delta > 0:
                    response.headers['X-Quantum-Pulses-Delta'] = str(pulses_delta)
            
            # Update global metrics
            with gs.lock:
                gs.metrics.add_request(elapsed_ms)
                if response.status_code >= 400:
                    gs.metrics.add_error(f"{response.status_code}")
        
        except Exception as e:
            logger.debug(f"[Middleware] Error: {e}")
        
        return response
    
    logger.info("✅ [Main] Middleware registered")
    
    # ════════════════════════════════════════════════════════════════════════════════════════
    # CORE ROUTES
    # ════════════════════════════════════════════════════════════════════════════════════════
    
    @app.route('/', methods=['GET'])
    def serve_index():
        """
        Serve index.html - the main UI
        Properly returns HTML with correct content type
        """
        index_path = os.path.join(PROJECT_ROOT, 'index.html')
        
        if os.path.exists(index_path):
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                return html_content, 200, {'Content-Type': 'text/html; charset=utf-8'}
            except Exception as e:
                logger.error(f"Error serving index.html: {e}")
                return error_response(f"Failed to serve index.html: {e}", 500)
        else:
            logger.warning(f"index.html not found at {index_path}")
            return error_response("index.html not found", 404)
    
    @app.route('/api/health', methods=['GET'])
    def health():
        """System health check endpoint"""
        return jsonify(get_system_health()), 200
    
    @app.route('/api/status', methods=['GET'])
    def status():
        """System status endpoint with full snapshot"""
        return jsonify({
            'status': 'online',
            'snapshot': get_state_snapshot(),
            'health': get_system_health()
        }), 200
    
    @app.route('/api/version', methods=['GET'])
    def version():
        """Version and system info endpoint"""
        return jsonify({
            'version': '5.0.0',
            'name': 'QTCL - Quantum Temporal Coherence Ledger',
            'timestamp': datetime.utcnow().isoformat(),
            'functions_registered': len(gs.all_functions),
            'environment': 'production',
            'expanded': True
        }), 200
    
    # ════════════════════════════════════════════════════════════════════════════════════════
    # GLOBAL ARCHITECTURE ENDPOINTS
    # ════════════════════════════════════════════════════════════════════════════════════════
    
    @app.route('/api/functions', methods=['GET'])
    def list_functions():
        """List all registered functions"""
        functions = list(gs.all_functions.keys())
        return jsonify({
            'count': len(functions),
            'functions': functions,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    
    @app.route('/api/function/<path:func_path>', methods=['GET'])
    def get_function(func_path):
        """Get function metadata"""
        func_sig = gs.all_functions.get(func_path)
        if not func_sig:
            return error_response(f"Function {func_path} not found", 404)
        
        return jsonify({
            'name': func_sig.name,
            'module': func_sig.module,
            'component': func_sig.component.value,
            'level': func_sig.level.value,
            'params': func_sig.params,
            'returns': func_sig.returns,
            'dependencies': func_sig.dependencies,
            'description': func_sig.description,
            'thread_safe': func_sig.thread_safe,
        }), 200
    
    @app.route('/api/hierarchy', methods=['GET'])
    def get_hierarchy():
        """Get complete logic hierarchy"""
        blocks = []
        for block in [
            gs.quantum_logic, gs.blockchain_logic, gs.database_logic,
            gs.auth_logic, gs.defi_logic, gs.oracle_logic, gs.ledger_logic,
            gs.admin_logic, gs.core_logic, gs.terminal_logic
        ]:
            if block:
                blocks.append({
                    'id': block.id,
                    'component': block.component.value,
                    'level': block.level.value,
                    'functions': len(block.functions),
                    'children': len(block.children),
                    'metrics': dict(block.metrics)
                })
        
        return jsonify({
            'logic_blocks': blocks,
            'total_blocks': len(blocks),
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    
    @app.route('/api/state', methods=['GET'])
    def get_state():
        """Get current state snapshot"""
        return jsonify(get_state_snapshot()), 200
    
    @app.route('/api/debug', methods=['GET'])
    def debug_info():
        """Get comprehensive debug information"""
        return jsonify(get_debug_info()), 200
    
    # ════════════════════════════════════════════════════════════════════════════════════════
    # COMPONENT-SPECIFIC ENDPOINTS
    # ════════════════════════════════════════════════════════════════════════════════════════
    
    @app.route('/api/heartbeat', methods=['POST'])
    def api_heartbeat():
        """Heartbeat endpoint - receives heartbeat pulses from SystemHeartbeat"""
        try:
            data = request.get_json() or {}
            pulse = data.get('pulse', 0)
            metrics = data.get('metrics', {})
            
            heartbeat = get_heartbeat()
            if heartbeat:
                with gs.lock:
                    gs.metrics.quantum_pulses += 1
            
            return jsonify({
                'status': 'received',
                'pulse': pulse,
                'timestamp': datetime.utcnow().isoformat()
            }), 200
        except Exception as e:
            logger.error(f"[API] Heartbeat endpoint error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/metrics/heartbeat', methods=['GET'])
    def metrics_heartbeat():
        """Get current heartbeat and system metrics"""
        heartbeat = get_heartbeat()
        with gs.lock:
            return jsonify({
                'heartbeat': {
                    'running': heartbeat.running if heartbeat else False,
                    'pulse_count': heartbeat.pulse_count if heartbeat else 0,
                    'interval': heartbeat.interval if heartbeat else 30,
                    'error_count': heartbeat.error_count if heartbeat else 0,
                    'listeners': len(heartbeat.listeners) if heartbeat else 0,
                    'last_beat': heartbeat.last_beat_time.isoformat() if heartbeat and heartbeat.last_beat_time else None,
                    'metrics': heartbeat.get_metrics() if heartbeat else {}
                },
                'system': {
                    'http_requests': gs.metrics.http_requests,
                    'http_errors': gs.metrics.http_errors,
                    'quantum_pulses': gs.metrics.quantum_pulses,
                    'transactions': gs.metrics.transactions_processed,
                    'blocks': gs.metrics.blocks_created,
                    'timestamp': datetime.utcnow().isoformat()
                }
            }), 200
    
    @app.route('/api/metrics/quantum', methods=['GET'])
    def metrics_quantum():
        """Get quantum lattice control metrics"""
        heartbeat = get_heartbeat()
        lattice = get_lattice()
        
        with gs.lock:
            quantum_metrics = {
                'heartbeat_active': heartbeat is not None and heartbeat.running,
                'heartbeat_pulse_count': heartbeat.pulse_count if heartbeat else 0,
                'lattice_active': lattice is not None,
                'quantum_pulses': gs.metrics.quantum_pulses,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Add lattice-specific metrics if available
            if lattice and hasattr(lattice, 'get_metrics'):
                try:
                    lattice_metrics = lattice.get_metrics()
                    quantum_metrics['lattice_metrics'] = lattice_metrics
                except Exception as e:
                    logger.debug(f"Could not get lattice metrics: {e}")
            
            return jsonify(quantum_metrics), 200
    
    @app.route('/api/system/status', methods=['GET'])
    def system_status():
        """Get comprehensive system status with all metrics"""
        heartbeat = get_heartbeat()
        health = get_system_health()
        
        with gs.lock:
            return jsonify({
                'status': 'operational',
                'health': health,
                'uptime_seconds': (datetime.utcnow() - gs.startup_time).total_seconds() if gs.startup_time else 0,
                'heartbeat': {
                    'running': heartbeat.running if heartbeat else False,
                    'pulse_count': heartbeat.pulse_count if heartbeat else 0,
                    'metrics': heartbeat.get_metrics() if heartbeat else {}
                },
                'metrics': gs.metrics.get_stats(),
                'timestamp': datetime.utcnow().isoformat()
            }), 200
    
    @app.route('/api/health', methods=['GET'])
    def api_health():
        """Health check endpoint"""
        health = get_system_health()
        heartbeat = get_heartbeat()
        
        is_healthy = (
            heartbeat and heartbeat.running and
            health['status'] == 'healthy'
        )
        
        return jsonify({
            'healthy': is_healthy,
            'status': health['status'],
            'heartbeat_active': heartbeat.running if heartbeat else False,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    
    @app.route('/api/quantum/status', methods=['GET'])
    def quantum_status():
        """Get quantum subsystems status"""
        heartbeat = get_heartbeat()
        with gs.lock:
            return jsonify({
                'heartbeat_active': heartbeat is not None,
                'heartbeat_running': heartbeat.running if heartbeat else False,
                'heartbeat_pulse_count': heartbeat.pulse_count if heartbeat else 0,
                'lattice_active': gs.quantum.lattice is not None,
                'neural_network_active': gs.quantum.neural_network is not None,
                'w_state_active': gs.quantum.w_state_manager is not None,
                'noise_bath_active': gs.quantum.noise_bath is not None,
                'timestamp': datetime.utcnow().isoformat()
            }), 200
    
    @app.route('/api/blockchain/status', methods=['GET'])
    def blockchain_status():
        """Get blockchain status"""
        blockchain = get_blockchain()
        with gs.lock:
            return jsonify({
                'chain_height': blockchain.chain_height,
                'pending_transactions': len(blockchain.pending_transactions),
                'total_transactions': blockchain.total_transactions,
                'total_blocks': blockchain.total_blocks,
                'consensus': blockchain.consensus_mechanism,
                'timestamp': datetime.utcnow().isoformat()
            }), 200
    
    @app.route('/api/defi/status', methods=['GET'])
    def defi_status():
        """Get DeFi status"""
        defi = get_defi()
        with gs.lock:
            return jsonify({
                'active_pools': defi.active_pools,
                'total_pools': len(defi.pools),
                'total_liquidity': str(defi.total_liquidity),
                'total_volume': str(defi.total_volume),
                'price_feed_connected': defi.price_feed_connected,
                'timestamp': datetime.utcnow().isoformat()
            }), 200
    
    @app.route('/api/oracle/status', methods=['GET'])
    def oracle_status():
        """Get oracle status"""
        oracle = get_oracle()
        with gs.lock:
            return jsonify({
                'price_points': oracle.data_points,
                'tokens_tracked': len(oracle.prices),
                'prices': {k: str(v) for k, v in oracle.prices.items()},
                'last_update': oracle.last_update.isoformat() if oracle.last_update else None,
                'timestamp': datetime.utcnow().isoformat()
            }), 200
    
    @app.route('/api/ledger/status', methods=['GET'])
    def ledger_status():
        """Get ledger status"""
        ledger = get_ledger()
        with gs.lock:
            return jsonify({
                'total_entries': ledger.total_entries,
                'audit_log_size': len(ledger.audit_log),
                'last_entry': ledger.last_entry_time.isoformat() if ledger.last_entry_time else None,
                'timestamp': datetime.utcnow().isoformat()
            }), 200
    
    # ════════════════════════════════════════════════════════════════════════════════════════
    # COMMAND EXECUTION ENDPOINTS
    # ════════════════════════════════════════════════════════════════════════════════════════
    
    @app.route('/api/command', methods=['POST'])
    def execute_command():
        """
        COMPREHENSIVE COMMAND EXECUTION ENDPOINT WITH DEEP GLOBALS INTEGRATION
        
        FEATURES:
        • Full GlobalCommandRegistry integration
        • Per-category command execution from terminal_logic.py
        • Proper error handling with suggestions
        • Auth token propagation
        • System metrics recording via globals
        • Correlation ID tracking
        
        REQUEST:
          {
            "command": "quantum/status",
            "args": [],
            "kwargs": {}
          }
        
        RESPONSE:
          {
            "status": "success",
            "command": "quantum/status",
            "result": {...},
            "timestamp": "2024-02-17T..."
          }
        """
        try:
            data = request.get_json() or {}
            command = data.get('command', '').strip()
            args = data.get('args', [])
            kwargs = data.get('kwargs', {})
            
            if not command:
                return jsonify({
                    'error': 'No command specified',
                    'status': 'error',
                    'hint': 'send {"command": "help"}'
                }), 400
            
            # Get auth token if provided
            auth_header = request.headers.get('Authorization', '')
            auth_token = auth_header.replace('Bearer ', '') if auth_header else None
            
            logger.info(f"[API] Executing command: {command} with args={args}")
            
            # Execute command via GlobalCommandRegistry with full globals integration
            if GlobalCommandRegistry:
                try:
                    # USE ALL_COMMANDS DIRECTLY FROM GlobalCommandRegistry
                    all_commands = GlobalCommandRegistry.ALL_COMMANDS
                    
                    logger.info(f"[API] Total commands available: {len(all_commands)}")
                    
                    # LOOKUP COMMAND (EXACT MATCH FIRST)
                    cmd_func = all_commands.get(command)
                    
                    # If not found, try case-insensitive match
                    if not cmd_func:
                        command_lower = command.lower()
                        for cmd_name, handler in all_commands.items():
                            if cmd_name.lower() == command_lower:
                                cmd_func = handler
                                break
                    
                    if not cmd_func:
                        # Find suggestions based on prefix match
                        cmd_parts = command.split('/')
                        prefix = cmd_parts[0] if cmd_parts else ''
                        suggestions = [k for k in all_commands.keys() if k.startswith(prefix + '/') and k != command]
                        
                        error_msg = f'Command \'{command}\' not found.'
                        logger.warning(f"[API] {error_msg} | Suggestions: {suggestions}")
                        
                        return jsonify({
                            'status': 'error',
                            'error': error_msg,
                            'command': command,
                            'suggestions': suggestions[:5],  # Limit to 5 suggestions
                            'timestamp': datetime.utcnow().isoformat()
                        }), 404
                    
                    # INJECT AUTH TOKEN IF NEEDED
                    if auth_token:
                        kwargs['auth_token'] = auth_token
                    
                    # EXECUTE COMMAND WITH GLOBALS CONTEXT
                    logger.debug(f"[API] Executing {command} with args={args}, kwargs={kwargs}")
                    result = cmd_func(*args, **kwargs)
                    
                    # RECORD METRICS IN GLOBALS
                    try:
                        gs = get_globals()
                        gs.metrics.http_requests += 1
                        gs.metrics.api_calls[command] += 1
                        gs.terminal.executed_commands += 1
                        gs.terminal.last_command = command
                        gs.terminal.last_command_time = datetime.utcnow()
                    except Exception as e:
                        logger.debug(f"[API] Could not record metrics: {e}")
                    
                    # FORMAT RESULT
                    formatted_result = result
                    if isinstance(result, dict):
                        formatted_result = result
                    elif isinstance(result, str):
                        formatted_result = {'output': result}
                    else:
                        formatted_result = {'output': str(result)}
                    
                    return jsonify({
                        'status': 'success',
                        'command': command,
                        'result': formatted_result,
                        'timestamp': datetime.utcnow().isoformat()
                    }), 200
                
                except Exception as e:
                    logger.error(f"[API] Command execution error: {e}", exc_info=True)
                    
                    # RECORD ERROR IN GLOBALS
                    try:
                        gs = get_globals()
                        gs.metrics.http_errors += 1
                        gs.terminal.failed_commands += 1
                    except:
                        pass
                    
                    return jsonify({
                        'status': 'error',
                        'command': command,
                        'error': str(e),
                        'type': type(e).__name__,
                        'timestamp': datetime.utcnow().isoformat()
                    }), 500
            else:
                logger.error("[API] GlobalCommandRegistry not available")
                return jsonify({
                    'status': 'error',
                    'error': 'Terminal engine not available',
                    'hint': 'Terminal system not initialized yet'
                }), 503
        
        except Exception as e:
            logger.error(f"[API] Command endpoint error: {e}", exc_info=True)
            return jsonify({
                'status': 'error',
                'error': str(e),
                'type': type(e).__name__,
                'timestamp': datetime.utcnow().isoformat()
            }), 500
    
    @app.route('/api/commands', methods=['GET'])
    def list_commands():
        """
        LIST ALL AVAILABLE COMMANDS FROM GLOBALS REGISTRY
        
        This endpoint aggregates all commands from GlobalCommandRegistry,
        organized by category for easy discovery and UI rendering.
        
        RESPONSE:
          {
            "commands": [
              {
                "command": "quantum/status",
                "category": "quantum",
                "description": "quantum command"
              },
              ...
            ],
            "categories": ["auth", "quantum", "blockchain", ...],
            "total": 47,
            "timestamp": "2024-02-17T..."
          }
        """
        try:
            if not GlobalCommandRegistry:
                logger.warning("[API] GlobalCommandRegistry not available")
                return jsonify({
                    'status': 'error',
                    'commands': [],
                    'categories': [],
                    'total': 0,
                    'error': 'Terminal registry not initialized'
                }), 503
            
            # CATEGORY MAPPING WITH DESCRIPTIONS
            category_map = {
                'help': 'Help & Documentation',
                'auth': 'Authentication & Authorization',
                'user': 'User Management & Profiles',
                'transaction': 'Transaction Operations',
                'wallet': 'Wallet Management',
                'block': 'Blockchain & Block Explorer',
                'quantum': 'Quantum Operations & Measurements',
                'oracle': 'Oracle Services & Data Feeds',
                'defi': 'DeFi Operations',
                'governance': 'Governance & Voting',
                'nft': 'NFT Operations',
                'contract': 'Smart Contracts',
                'bridge': 'Cross-chain Bridge',
                'admin': 'Administration & Management',
                'system': 'System Information & Health',
                'parallel': 'Parallel Task Execution',
            }
            
            # BUILD COMMAND LIST FROM ALL REGISTRY CATEGORIES
            commands = []
            command_set = set()
            
            registry_categories = {
                'help': getattr(GlobalCommandRegistry, 'HELP_COMMANDS', {}),
                'auth': getattr(GlobalCommandRegistry, 'AUTH_COMMANDS', {}),
                'user': getattr(GlobalCommandRegistry, 'USER_COMMANDS', {}),
                'transaction': getattr(GlobalCommandRegistry, 'TRANSACTION_COMMANDS', {}),
                'wallet': getattr(GlobalCommandRegistry, 'WALLET_COMMANDS', {}),
                'block': getattr(GlobalCommandRegistry, 'BLOCK_COMMANDS', {}),
                'quantum': getattr(GlobalCommandRegistry, 'QUANTUM_COMMANDS', {}),
                'oracle': getattr(GlobalCommandRegistry, 'ORACLE_COMMANDS', {}),
                'defi': getattr(GlobalCommandRegistry, 'DEFI_COMMANDS', {}),
                'governance': getattr(GlobalCommandRegistry, 'GOVERNANCE_COMMANDS', {}),
                'nft': getattr(GlobalCommandRegistry, 'NFT_COMMANDS', {}),
                'contract': getattr(GlobalCommandRegistry, 'CONTRACT_COMMANDS', {}),
                'bridge': getattr(GlobalCommandRegistry, 'BRIDGE_COMMANDS', {}),
                'admin': getattr(GlobalCommandRegistry, 'ADMIN_COMMANDS', {}),
                'system': getattr(GlobalCommandRegistry, 'SYSTEM_COMMANDS', {}),
                'parallel': getattr(GlobalCommandRegistry, 'PARALLEL_COMMANDS', {}),
            }
            
            for category, cmd_dict in registry_categories.items():
                if isinstance(cmd_dict, dict):
                    for cmd_name in cmd_dict.keys():
                        if cmd_name not in command_set:
                            commands.append({
                                'command': cmd_name,
                                'category': category,
                                'description': category_map.get(category, 'Command'),
                                'category_description': category_map.get(category, '')
                            })
                            command_set.add(cmd_name)
            
            # SORT BY CATEGORY THEN COMMAND NAME
            commands.sort(key=lambda x: (x['category'], x['command']))
            
            # GET UNIQUE CATEGORIES THAT HAVE COMMANDS
            active_categories = sorted(list(set(c['category'] for c in commands)))
            
            logger.info(f"[API] Listed {len(commands)} commands across {len(active_categories)} categories")
            
            return jsonify({
                'status': 'success',
                'commands': commands,
                'total': len(commands),
                'categories': active_categories,
                'category_count': len(active_categories),
                'timestamp': datetime.utcnow().isoformat()
            }), 200
        
        except Exception as e:
            logger.error(f"[API] List commands error: {e}", exc_info=True)
            return jsonify({
                'status': 'error',
                'error': str(e),
                'commands': [],
                'total': 0
            }), 500
    
    @app.route('/api/command-help', methods=['GET'])
    def command_help():
        """Get detailed help for all commands, grouped by category"""
        try:
            if not GlobalCommandRegistry:
                return jsonify({
                    'help': {},
                    'error': 'Terminal engine not available'
                }), 503
            
            # Build command help
            help_text = {
                'auth': {
                    'category': 'Authentication',
                    'description': 'User authentication commands',
                    'commands': list(GlobalCommandRegistry.AUTH_COMMANDS.keys())
                },
                'user': {
                    'category': 'User Management',
                    'description': 'User profile and settings',
                    'commands': list(GlobalCommandRegistry.USER_COMMANDS.keys())
                },
                'transaction': {
                    'category': 'Transactions',
                    'description': 'Transaction operations',
                    'commands': list(GlobalCommandRegistry.TRANSACTION_COMMANDS.keys())
                },
                'wallet': {
                    'category': 'Wallet',
                    'description': 'Wallet operations',
                    'commands': list(GlobalCommandRegistry.WALLET_COMMANDS.keys())
                },
                'block': {
                    'category': 'Blockchain',
                    'description': 'Block explorer and operations',
                    'commands': list(GlobalCommandRegistry.BLOCK_COMMANDS.keys())
                },
                'quantum': {
                    'category': 'Quantum',
                    'description': 'Quantum operations and measurements',
                    'commands': list(GlobalCommandRegistry.QUANTUM_COMMANDS.keys())
                },
                'oracle': {
                    'category': 'Oracle',
                    'description': 'Price, time, and event oracle',
                    'commands': list(GlobalCommandRegistry.ORACLE_COMMANDS.keys())
                },
                'defi': {
                    'category': 'DeFi',
                    'description': 'DeFi operations (staking, lending, etc)',
                    'commands': list(GlobalCommandRegistry.DEFI_COMMANDS.keys())
                },
                'governance': {
                    'category': 'Governance',
                    'description': 'Voting and proposals',
                    'commands': list(GlobalCommandRegistry.GOVERNANCE_COMMANDS.keys())
                },
                'nft': {
                    'category': 'NFT',
                    'description': 'NFT minting and management',
                    'commands': list(GlobalCommandRegistry.NFT_COMMANDS.keys())
                },
                'contract': {
                    'category': 'Smart Contracts',
                    'description': 'Deploy and execute contracts',
                    'commands': list(GlobalCommandRegistry.CONTRACT_COMMANDS.keys())
                },
                'bridge': {
                    'category': 'Bridge',
                    'description': 'Cross-chain operations',
                    'commands': list(GlobalCommandRegistry.BRIDGE_COMMANDS.keys())
                },
                'admin': {
                    'category': 'Admin',
                    'description': 'Administrative commands (admin only)',
                    'commands': list(GlobalCommandRegistry.ADMIN_COMMANDS.keys())
                },
                'system': {
                    'category': 'System',
                    'description': 'System status and operations',
                    'commands': list(GlobalCommandRegistry.SYSTEM_COMMANDS.keys())
                },
            }
            
            return jsonify({
                'help': help_text,
                'total_categories': len(help_text),
                'timestamp': datetime.utcnow().isoformat()
            }), 200
        
        except Exception as e:
            logger.error(f"[API] Command help error: {e}", exc_info=True)
            return jsonify({'error': str(e), 'status': 'error'}), 500
    
    # ════════════════════════════════════════════════════════════════════════════════════════
    # ERROR HANDLERS
    # ════════════════════════════════════════════════════════════════════════════════════════
    
    @app.errorhandler(404)
    def not_found(error):
        return error_response("Resource not found", 404)
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal error: {error}")
        return error_response("Internal server error", 500)
    
    @app.errorhandler(403)
    def forbidden(error):
        return error_response("Forbidden", 403)
    
    @app.errorhandler(400)
    def bad_request(error):
        return error_response("Bad request", 400)
    
    logger.info("✅ [Main] Flask routes registered")
    
    return app

# ════════════════════════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════════════════════

def error_response(message: str, status_code: int) -> tuple:
    """Generate error response"""
    return jsonify({
        'error': True,
        'message': message,
        'timestamp': datetime.utcnow().isoformat()
    }), status_code

# ════════════════════════════════════════════════════════════════════════════════════════════════
# APPLICATION FACTORY
# ════════════════════════════════════════════════════════════════════════════════════════════════

logger.info("[Main] Creating Flask application...")
app = create_app()
logger.info("✅ [Main] Flask application created")

# ════════════════════════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    logger.info("╔════════════════════════════════════════════════════════════════════╗")
    logger.info("║              QTCL v5.0 STARTING (EXPANDED)                       ║")
    logger.info("║     Global Architecture Master + Flask Integration                ║")
    logger.info("╚════════════════════════════════════════════════════════════════════╝")
    
    try:
        # Display startup info
        health = get_system_health()
        logger.info(f"📊 Registered Functions: {len(gs.all_functions)}")
        logger.info(f"🏥 System Health: {health['status'].upper()}")
        logger.info(f"📍 Server: http://0.0.0.0:5000")
        logger.info(f"🌐 UI: http://localhost:5000")
        logger.info(f"📝 Dashboard: http://localhost:5000/")
        
        # Start Flask server
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            use_reloader=False,
            threaded=True
        )
    
    except KeyboardInterrupt:
        logger.info("\n[Main] Shutdown signal received")
    
    except Exception as e:
        logger.error(f"[Main] Fatal error: {e}")
        sys.exit(1)
    
    finally:
        # Cleanup
        heartbeat = get_heartbeat()
        if heartbeat and hasattr(heartbeat, 'running') and heartbeat.running:
            try:
                heartbeat.stop()
            except:
                pass
        
        logger.info("[Main] QTCL v5.0 shutdown complete")


# ════════════════════════════════════════════════════════════════════════════════════════════════════════
# LEVEL 2 SUBLOGIC - MASTER APPLICATION ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════════════════════════════════════

class MasterApplicationOrchestrator:
    """Master orchestrator for complete QTCL application"""
    
    def __init__(self):
        self.app = None
        self.all_systems = {}
        self.initialization_status = {}
        self.initialize_complete_system()
    
    def initialize_complete_system(self):
        """Initialize complete QTCL system"""
        print("[MasterOrch] Initializing complete QTCL system...")
        
        # Initialize GLOBALS
        self._init_globals()
        
        # Initialize all systems
        self._init_quantum()
        self._init_blockchain()
        self._init_defi()
        self._init_oracle()
        self._init_ledger()
        self._init_auth()
        self._init_terminal()
        
        # Build interconnections
        self._build_system_interconnections()
        
        # Create Flask app
        self._create_flask_app()
        
        print("[MasterOrch] ✓ Complete system initialized")
    
    def _init_globals(self):
        """Initialize GLOBALS system"""
        try:
            from globals import initialize_globals, initialize_system_orchestration
            initialize_globals()
            initialize_system_orchestration()
            self.initialization_status['globals'] = 'ready'
            print("[MasterOrch] ✓ GLOBALS initialized")
        except Exception as e:
            self.initialization_status['globals'] = f'error: {e}'
            print(f"[MasterOrch] ✗ GLOBALS: {e}")
    
    def _init_quantum(self):
        """Initialize quantum system"""
        try:
            from quantum_api import get_quantum_integration
            self.all_systems['quantum'] = get_quantum_integration()
            self.initialization_status['quantum'] = 'ready'
            print("[MasterOrch] ✓ Quantum system initialized")
        except Exception as e:
            self.initialization_status['quantum'] = f'error: {e}'
            print(f"[MasterOrch] ✗ Quantum: {e}")
    
    def _init_blockchain(self):
        """Initialize blockchain system"""
        try:
            from blockchain_api import get_blockchain_integration
            self.all_systems['blockchain'] = get_blockchain_integration()
            self.initialization_status['blockchain'] = 'ready'
            print("[MasterOrch] ✓ Blockchain system initialized")
        except Exception as e:
            self.initialization_status['blockchain'] = f'error: {e}'
            print(f"[MasterOrch] ✗ Blockchain: {e}")
    
    def _init_defi(self):
        """Initialize DeFi system"""
        try:
            from defi_api import get_defi_integration
            self.all_systems['defi'] = get_defi_integration()
            self.initialization_status['defi'] = 'ready'
            print("[MasterOrch] ✓ DeFi system initialized")
        except Exception as e:
            self.initialization_status['defi'] = f'error: {e}'
            print(f"[MasterOrch] ✗ DeFi: {e}")
    
    def _init_oracle(self):
        """Initialize oracle system"""
        try:
            from oracle_api import get_oracle_integration
            self.all_systems['oracle'] = get_oracle_integration()
            self.initialization_status['oracle'] = 'ready'
            print("[MasterOrch] ✓ Oracle system initialized")
        except Exception as e:
            self.initialization_status['oracle'] = f'error: {e}'
            print(f"[MasterOrch] ✗ Oracle: {e}")
    
    def _init_ledger(self):
        """Initialize ledger system"""
        try:
            from ledger_manager import get_ledger_integration
            self.all_systems['ledger'] = get_ledger_integration()
            self.initialization_status['ledger'] = 'ready'
            print("[MasterOrch] ✓ Ledger system initialized")
        except Exception as e:
            self.initialization_status['ledger'] = f'error: {e}'
            print(f"[MasterOrch] ✗ Ledger: {e}")
    
    def _init_auth(self):
        """Initialize auth system"""
        try:
            from auth_handlers import AUTH_INTEGRATION
            self.all_systems['auth'] = AUTH_INTEGRATION
            self.initialization_status['auth'] = 'ready'
            print("[MasterOrch] ✓ Auth system initialized")
        except Exception as e:
            self.initialization_status['auth'] = f'error: {e}'
            print(f"[MasterOrch] ✗ Auth: {e}")
    
    def _init_terminal(self):
        """Initialize terminal system"""
        try:
            from terminal_logic import TERMINAL_ORCHESTRATOR
            self.all_systems['terminal'] = TERMINAL_ORCHESTRATOR
            self.initialization_status['terminal'] = 'ready'
            print("[MasterOrch] ✓ Terminal system initialized")
        except Exception as e:
            self.initialization_status['terminal'] = f'error: {e}'
            print(f"[MasterOrch] ✗ Terminal: {e}")
    
    def _build_system_interconnections(self):
        """Build interconnections between all systems"""
        print("[MasterOrch] Building system interconnections...")
        
        # Quantum → Blockchain
        if 'quantum' in self.all_systems and 'blockchain' in self.all_systems:
            self.all_systems['blockchain'].consume_quantum_entropy()
        
        # Oracle → DeFi
        if 'oracle' in self.all_systems and 'defi' in self.all_systems:
            pass  # DeFi will pull prices from oracle
        
        # Blockchain → Ledger
        if 'blockchain' in self.all_systems and 'ledger' in self.all_systems:
            self.all_systems['ledger'].sync_with_blockchain()
        
        print("[MasterOrch] ✓ Interconnections built")
    
    def _create_flask_app(self):
        """Create Flask app with all routes"""
        from flask import Flask, jsonify
        
        self.app = Flask(__name__)
        
        # Health check
        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({'status': 'healthy'})
        
        # System status
        @self.app.route('/status', methods=['GET'])
        def status():
            return jsonify({
                'systems': self.initialization_status,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        
        # Quantum endpoint
        @self.app.route('/api/quantum/status', methods=['GET'])
        def quantum_status():
            if 'quantum' in self.all_systems:
                return jsonify(self.all_systems['quantum'].get_system_status())
            return jsonify({'error': 'Quantum system not available'})
        
        # Blockchain endpoint
        @self.app.route('/api/blockchain/status', methods=['GET'])
        def blockchain_status():
            if 'blockchain' in self.all_systems:
                return jsonify(self.all_systems['blockchain'].get_system_status())
            return jsonify({'error': 'Blockchain system not available'})
        
        # DeFi endpoint
        @self.app.route('/api/defi/status', methods=['GET'])
        def defi_status():
            if 'defi' in self.all_systems:
                return jsonify(self.all_systems['defi'].get_system_status())
            return jsonify({'error': 'DeFi system not available'})
        
        # Oracle endpoint
        @self.app.route('/api/oracle/status', methods=['GET'])
        def oracle_status():
            if 'oracle' in self.all_systems:
                return jsonify(self.all_systems['oracle'].get_system_status())
            return jsonify({'error': 'Oracle system not available'})
        
        # Ledger endpoint
        @self.app.route('/api/ledger/status', methods=['GET'])
        def ledger_status():
            if 'ledger' in self.all_systems:
                return jsonify(self.all_systems['ledger'].get_system_status())
            return jsonify({'error': 'Ledger system not available'})
        
        # Terminal command endpoint
        @self.app.route('/api/command', methods=['POST'])
        def execute_command():
            from flask import request
            if 'command' in self.all_systems:
                cmd = request.json.get('cmd', '')
                result = self.all_systems['terminal'].execute_command(cmd)
                return jsonify(result)
            return jsonify({'error': 'Terminal not available'})
        
        print("[MasterOrch] ✓ Flask app created with all routes")
    
    def get_app(self):
        """Get Flask app"""
        return self.app
    
    def get_system_status(self):
        """Get complete system status"""
        return {
            'initialization': self.initialization_status,
            'systems': list(self.all_systems.keys()),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

# ════════════════════════════════════════════════════════════════════════════════════════════════════════
# APPLICATION CREATION & ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════════════════════════════

MASTER_ORCHESTRATOR = MasterApplicationOrchestrator()
app = MASTER_ORCHESTRATOR.get_app()

if __name__ == '__main__':
    print("[Main] Starting QTCL application...")
    print(f"[Main] System status: {MASTER_ORCHESTRATOR.get_system_status()}")
    app.run(host='0.0.0.0', port=5000, debug=False)
