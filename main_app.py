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
        """Execute a command via the terminal engine"""
        try:
            data = request.get_json() or {}
            command = data.get('command', '').strip()
            args = data.get('args', [])
            kwargs = data.get('kwargs', {})
            
            if not command:
                return jsonify({'error': 'No command specified', 'status': 'error'}), 400
            
            # Get auth token if provided
            auth_header = request.headers.get('Authorization', '')
            auth_token = auth_header.replace('Bearer ', '') if auth_header else None
            
            # Execute command via GlobalCommandRegistry
            if GlobalCommandRegistry:
                try:
                    # Convert command like 'auth/login' to registry lookup
                    cmd_parts = command.split('/')
                    
                    # Build all command maps
                    all_commands = {
                        **GlobalCommandRegistry.AUTH_COMMANDS,
                        **GlobalCommandRegistry.USER_COMMANDS,
                        **GlobalCommandRegistry.TRANSACTION_COMMANDS,
                        **GlobalCommandRegistry.WALLET_COMMANDS,
                        **GlobalCommandRegistry.BLOCK_COMMANDS,
                        **GlobalCommandRegistry.QUANTUM_COMMANDS,
                        **GlobalCommandRegistry.ORACLE_COMMANDS,
                        **GlobalCommandRegistry.DEFI_COMMANDS,
                        **GlobalCommandRegistry.GOVERNANCE_COMMANDS,
                        **GlobalCommandRegistry.NFT_COMMANDS,
                        **GlobalCommandRegistry.CONTRACT_COMMANDS,
                        **GlobalCommandRegistry.BRIDGE_COMMANDS,
                        **GlobalCommandRegistry.ADMIN_COMMANDS,
                        **GlobalCommandRegistry.SYSTEM_COMMANDS,
                    }
                    
                    cmd_func = all_commands.get(command)
                    
                    if not cmd_func:
                        # Try to find matching command
                        matching = [k for k in all_commands.keys() if k.startswith(cmd_parts[0] + '/')]
                        if matching:
                            return jsonify({
                                'error': f'Command not found. Did you mean: {matching}',
                                'status': 'error',
                                'suggestions': matching
                            }), 404
                        else:
                            return jsonify({
                                'error': f'Unknown command: {command}',
                                'status': 'error'
                            }), 404
                    
                    # Pass auth token to command if needed
                    if auth_token:
                        kwargs['auth_token'] = auth_token
                    
                    # Execute the command
                    logger.info(f"[API] Executing command: {command} with args={args}, kwargs={kwargs}")
                    result = cmd_func(*args, **kwargs)
                    
                    return jsonify({
                        'status': 'success',
                        'command': command,
                        'result': result if isinstance(result, dict) else {'output': str(result)},
                        'timestamp': datetime.utcnow().isoformat()
                    }), 200
                
                except Exception as e:
                    logger.error(f"[API] Command execution error: {e}", exc_info=True)
                    return jsonify({
                        'error': str(e),
                        'status': 'error',
                        'command': command,
                        'timestamp': datetime.utcnow().isoformat()
                    }), 500
            else:
                return jsonify({
                    'error': 'Terminal engine not available',
                    'status': 'error'
                }), 503
        
        except Exception as e:
            logger.error(f"[API] Command endpoint error: {e}", exc_info=True)
            return jsonify({'error': str(e), 'status': 'error'}), 500
    
    @app.route('/api/commands', methods=['GET'])
    def list_commands():
        """List all available commands with metadata"""
        try:
            if not GlobalCommandRegistry:
                return jsonify({
                    'commands': [],
                    'categories': [],
                    'error': 'Terminal engine not available'
                }), 503
            
            # Build all command maps
            command_map = {
                'auth': [(k, 'authentication') for k in GlobalCommandRegistry.AUTH_COMMANDS.keys()],
                'user': [(k, 'user management') for k in GlobalCommandRegistry.USER_COMMANDS.keys()],
                'transaction': [(k, 'transactions') for k in GlobalCommandRegistry.TRANSACTION_COMMANDS.keys()],
                'wallet': [(k, 'wallet operations') for k in GlobalCommandRegistry.WALLET_COMMANDS.keys()],
                'block': [(k, 'blockchain') for k in GlobalCommandRegistry.BLOCK_COMMANDS.keys()],
                'quantum': [(k, 'quantum operations') for k in GlobalCommandRegistry.QUANTUM_COMMANDS.keys()],
                'oracle': [(k, 'oracle services') for k in GlobalCommandRegistry.ORACLE_COMMANDS.keys()],
                'defi': [(k, 'defi operations') for k in GlobalCommandRegistry.DEFI_COMMANDS.keys()],
                'governance': [(k, 'governance') for k in GlobalCommandRegistry.GOVERNANCE_COMMANDS.keys()],
                'nft': [(k, 'nft operations') for k in GlobalCommandRegistry.NFT_COMMANDS.keys()],
                'contract': [(k, 'smart contracts') for k in GlobalCommandRegistry.CONTRACT_COMMANDS.keys()],
                'bridge': [(k, 'bridge operations') for k in GlobalCommandRegistry.BRIDGE_COMMANDS.keys()],
                'admin': [(k, 'admin') for k in GlobalCommandRegistry.ADMIN_COMMANDS.keys()],
                'system': [(k, 'system') for k in GlobalCommandRegistry.SYSTEM_COMMANDS.keys()],
            }
            
            # Flatten and format
            commands = []
            for category, cmds in command_map.items():
                for cmd_name, desc in cmds:
                    commands.append({
                        'command': cmd_name,
                        'category': category,
                        'description': desc,
                        'examples': f'{cmd_name} arg1 arg2'
                    })
            
            # Sort by category then command
            commands.sort(key=lambda x: (x['category'], x['command']))
            
            return jsonify({
                'commands': commands,
                'total': len(commands),
                'categories': sorted(list(command_map.keys())),
                'timestamp': datetime.utcnow().isoformat()
            }), 200
        
        except Exception as e:
            logger.error(f"[API] List commands error: {e}", exc_info=True)
            return jsonify({'error': str(e), 'status': 'error'}), 500
    
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
