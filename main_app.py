#!/usr/bin/env python3
"""
QTCL MAIN APPLICATION - CLEAN ARCHITECTURE
Unified Flask app factory, blueprint registration, state managed by globals.py
"""

import os
import sys
import logging
import time
from datetime import datetime
from flask import Flask, request, jsonify, g
from flask_cors import CORS

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('qtcl_main.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info("[Main] Importing globals...")
from globals import initialize_globals, get_globals, get_heartbeat

logger.info("[Main] Initializing global state...")
if not initialize_globals():
    logger.error("[Main] FAILED to initialize global state")
    sys.exit(1)

logger.info("[Main] Global state initialized")

def create_app():
    """Create and configure Flask application"""
    logger.info("[Main] Creating Flask app...")
    
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
    app.config['JSON_SORT_KEYS'] = False
    
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    logger.info("[Main] Importing blueprints...")
    try:
        from blockchain_api import blueprint as blockchain_bp
        from quantum_api import blueprint as quantum_bp
        from core_api import blueprint as core_bp
        from admin_api import blueprint as admin_bp
        from oracle_api import blueprint as oracle_bp
        from defi_api import blueprint as defi_bp
        
        app.register_blueprint(blockchain_bp)
        app.register_blueprint(quantum_bp)
        app.register_blueprint(core_bp)
        app.register_blueprint(admin_bp)
        app.register_blueprint(oracle_bp)
        app.register_blueprint(defi_bp)
        
        logger.info("[Main] All 6 blueprints registered")
    
    except ImportError as e:
        logger.error(f"[Main] Failed to import blueprints: {e}")
        raise
    
    @app.before_request
    def before_request():
        """Pre-request processing"""
        g.start_time = time.time()
        g.start_pulse = 0
        heartbeat = get_heartbeat()
        if heartbeat:
            g.start_pulse = heartbeat.pulse_count
    
    @app.after_request
    def after_request(response):
        """Post-request processing"""
        try:
            elapsed_ms = (time.time() - g.start_time) * 1000
            response.headers['X-Request-Time-Ms'] = f"{elapsed_ms:.1f}"
            
            heartbeat = get_heartbeat()
            if heartbeat:
                response.headers['X-Heartbeat-Pulse'] = str(heartbeat.pulse_count)
                pulses_delta = heartbeat.pulse_count - g.start_pulse
                if pulses_delta > 0:
                    response.headers['X-Quantum-Pulses-Delta'] = str(pulses_delta)
            
            globals_inst = get_globals()
            with globals_inst.lock:
                globals_inst.metrics.add_request(elapsed_ms)
                if response.status_code >= 400:
                    globals_inst.metrics.add_error(f"{response.status_code}")
        except Exception as e:
            logger.debug(f"[Middleware] Error: {e}")
        
        return response
    
    @app.route('/health', methods=['GET'])
    def health():
        from globals import get_system_health
        return jsonify(get_system_health()), 200
    
    @app.route('/status', methods=['GET'])
    def status():
        from globals import get_state_snapshot, get_debug_info
        return jsonify({'status': 'online', 'snapshot': get_state_snapshot(), 'debug': get_debug_info()}), 200
    
    @app.route('/api/version', methods=['GET'])
    def version():
        return jsonify({'version': '4.0.0', 'name': 'QTCL', 'timestamp': datetime.utcnow().isoformat()}), 200
    
    logger.info("[Main] Flask app ready")
    return app

app = create_app()

if __name__ == '__main__':
    logger.info("="*80)
    logger.info("QTCL API v4.0.0 STARTING")
    logger.info("="*80)
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("\n[Main] Shutdown")
    finally:
        heartbeat = get_heartbeat()
        if heartbeat and heartbeat.running:
            heartbeat.stop()
        logger.info("[Main] Goodbye")
