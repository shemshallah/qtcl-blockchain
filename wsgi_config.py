#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════
QTCL WSGI CONFIGURATION - PRODUCTION DEPLOYMENT
Clean entry point for Gunicorn, uWSGI, and other WSGI servers
═══════════════════════════════════════════════════════════════════════════════════════
"""

import sys
import os
import logging
import traceback
import threading
import time
import fcntl
from datetime import datetime

# Setup project path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configure logging with both file and console output
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('qtcl_wsgi.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info("=" * 100)
logger.info("QTCL WSGI - PRODUCTION DEPLOYMENT INITIALIZATION")
logger.info(f"Project Root: {PROJECT_ROOT}")
logger.info(f"Python Version: {sys.version}")
logger.info(f"Timestamp: {datetime.now().isoformat()}")
logger.info("=" * 100)

# ═══════════════════════════════════════════════════════════════════════════════════════
# GLOBAL QUANTUM SYSTEM SINGLETON WITH LOCK FILE (NO EXTERNAL MODULE)
# ═══════════════════════════════════════════════════════════════════════════════════════

_QUANTUM_SYSTEM_INSTANCE = None
_QUANTUM_SYSTEM_LOCK = threading.RLock()
_QUANTUM_SYSTEM_INITIALIZED = False
_LOCK_FILE_PATH = '/tmp/quantum_system.lock'
_LOCK_FILE = None

def _acquire_lock_file(timeout: int = 30) -> bool:
    """Acquire filesystem lock (process-level synchronization)"""
    global _LOCK_FILE
    start_time = time.time()
    
    while True:
        try:
            _LOCK_FILE = open(_LOCK_FILE_PATH, 'w')
            fcntl.flock(_LOCK_FILE.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            logger.debug("[QuantumSystem] Lock file acquired")
            return True
        except (IOError, OSError):
            if time.time() - start_time > timeout:
                logger.warning(f"[QuantumSystem] Lock timeout after {timeout}s")
                return False
            time.sleep(0.1)

def _release_lock_file() -> None:
    """Release filesystem lock"""
    global _LOCK_FILE
    if _LOCK_FILE:
        try:
            fcntl.flock(_LOCK_FILE.fileno(), fcntl.LOCK_UN)
            _LOCK_FILE.close()
            _LOCK_FILE = None
            logger.debug("[QuantumSystem] Lock file released")
        except Exception as e:
            logger.warning(f"[QuantumSystem] Error releasing lock: {e}")

def initialize_quantum_system() -> None:
    """Initialize SINGLE global quantum system (process-safe with lock file) - NO DAEMON THREAD YET"""
    global _QUANTUM_SYSTEM_INSTANCE, _QUANTUM_SYSTEM_INITIALIZED
    
    with _QUANTUM_SYSTEM_LOCK:
        if _QUANTUM_SYSTEM_INITIALIZED:
            return
        
        _QUANTUM_SYSTEM_INITIALIZED = True
        
        try:
            from quantum_lattice_control_live_complete import QuantumLatticeControlLiveV5
            
            # Acquire lock file (process-level sync)
            if not _acquire_lock_file(timeout=30):
                logger.error("[QuantumSystem] Failed to acquire lock")
                return
            
            try:
                db_config = {
                    'host': os.getenv('SUPABASE_HOST', 'localhost'),
                    'port': int(os.getenv('SUPABASE_PORT', '5432')),
                    'database': os.getenv('SUPABASE_DB', 'postgres'),
                    'user': os.getenv('SUPABASE_USER', 'postgres'),
                    'password': os.getenv('SUPABASE_PASSWORD', 'postgres'),
                }
                app_url = os.getenv('APP_URL', 'http://localhost:5000')
                
                logger.info("[QuantumSystem] Creating SINGLE global quantum system...")
                _QUANTUM_SYSTEM_INSTANCE = QuantumLatticeControlLiveV5(
                    db_config=db_config,
                    app_url=app_url
                )
                logger.info("[QuantumSystem] ✓ Global quantum system initialized (SINGLETON)")
                logger.info("[QuantumSystem] ⚠ Daemon thread will start after Flask app initialization")
            
            finally:
                _release_lock_file()
        
        except Exception as e:
            logger.error(f"[QuantumSystem] Failed: {e}")
            logger.error(traceback.format_exc())

def get_quantum_system():
    """Get the initialized quantum system instance"""
    return _QUANTUM_SYSTEM_INSTANCE

def start_quantum_daemon():
    """Start the quantum system daemon thread (call after Flask is initialized)"""
    global _QUANTUM_SYSTEM_INSTANCE
    
    if not _QUANTUM_SYSTEM_INSTANCE:
        logger.error("[QuantumSystem] Cannot start daemon - quantum system not initialized")
        return
    
    try:
        import threading as _threading
        _cycle_thread = _threading.Thread(
            target=_QUANTUM_SYSTEM_INSTANCE.run_continuous,
            kwargs={'duration_hours': 87600},
            daemon=True,
            name='QuantumCycleThread'
        )
        _cycle_thread.start()
        logger.info("[QuantumSystem] ✓ Cycle daemon thread started after Flask initialization")
    except Exception as e:
        logger.error(f"[QuantumSystem] Failed to start daemon: {e}")
        logger.error(traceback.format_exc())

# Pre-initialize quantum system ONCE at module load
logger.info("Pre-initializing GLOBAL quantum system (SINGLETON with lock file)...")
initialize_quantum_system()
if _QUANTUM_SYSTEM_INSTANCE:
    logger.info("✓ GLOBAL quantum system pre-initialized at module load")
else:
    logger.warning("⚠ Quantum system not initialized")

logger.info("=" * 100)

# ═══════════════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════════════

REQUIRED_ENV_VARS = [
    'SUPABASE_HOST',
    'SUPABASE_USER',
    'SUPABASE_PASSWORD',
    'SUPABASE_PORT',
    'SUPABASE_DB'
]

missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing_vars:
    logger.warning(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
    logger.warning("⚠️  Using defaults for development - NOT FOR PRODUCTION")
else:
    logger.info("✓ All required environment variables configured")

# ═══════════════════════════════════════════════════════════════════════════════════════
# IMPORT MAIN APPLICATION WITH COMPREHENSIVE ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════════════════════

app = None
initialization_error = None

try:
    logger.info("Importing main application...")
    
    # Import factory functions from main_app.py
    from main_app import create_app, initialize_app
    
    logger.info("✓ Main application factory imported successfully")
    
    # Create app instance (routes are registered in the factory)
    logger.info("Creating Flask application instance...")
    app = create_app()
    logger.info(f"✓ Flask app created with {len(list(app.url_map.iter_rules()))} routes")
    
    # Initialize the application
    logger.info("Initializing application...")
    if initialize_app(app):
        logger.info("✓ Application initialized successfully")
    else:
        logger.warning("⚠ Application initialization returned False, but continuing...")
    
    logger.info("=" * 100)
    logger.info("✓ WSGI APPLICATION READY FOR DEPLOYMENT")
    logger.info("=" * 100)
    
except ImportError as e:
    logger.critical(f"✗ Failed to import main_app: {e}")
    logger.critical(traceback.format_exc())
    initialization_error = str(e)
    
except Exception as e:
    logger.critical(f"✗ Initialization error: {e}")
    logger.critical(traceback.format_exc())
    initialization_error = str(e)

# ═══════════════════════════════════════════════════════════════════════════════════════
# WSGI APPLICATION EXPORT WITH FALLBACK
# ═══════════════════════════════════════════════════════════════════════════════════════

if app is None:
    logger.error("✗ Creating minimal Flask app as fallback")
    from flask import Flask, jsonify
    
    app = Flask(__name__)
    
    @app.errorhandler(500)
    @app.errorhandler(400)
    @app.errorhandler(404)
    def error_handler(error):
        return jsonify({
            'error': 'Application initialization error',
            'details': initialization_error or str(error),
            'status': 'degraded'
        }), 500
    
    @app.route('/health', methods=['GET'])
    def health():
        if initialization_error:
            return jsonify({
                'status': 'unhealthy',
                'error': initialization_error
            }), 503
        return jsonify({'status': 'healthy'}), 200

# ═══════════════════════════════════════════════════════════════════════════════════════
# ADD HTML TERMINAL SERVING AT ROOT
# ═══════════════════════════════════════════════════════════════════════════════════════

def _load_html_terminal():
    """Load index.html terminal UI"""
    try:
        html_path = os.path.join(PROJECT_ROOT, 'index.html')
        if os.path.exists(html_path):
            with open(html_path, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        logger.error(f"[HTML] Error loading index.html: {e}")
    return None

_HTML_CONTENT = _load_html_terminal()

if app is not None:
    @app.route('/', methods=['GET'])
    def serve_html_terminal():
        """Serve HTML terminal UI at root"""
        if _HTML_CONTENT:
            return _HTML_CONTENT, 200, {'Content-Type': 'text/html; charset=utf-8'}
        else:
            return """<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>QTCL Terminal</title>
<style>body{background:#0a0a0f;color:#00ff88;font-family:monospace;display:flex;align-items:center;justify-content:center;height:100vh;margin:0}.box{text-align:center;padding:30px;border:2px solid #00ff88;border-radius:8px;background:rgba(0,255,136,0.1);box-shadow:0 0 20px rgba(0,255,136,0.3)}h1{color:#00ff88;text-shadow:0 0 10px #00ff88;margin:0}p{color:#00ffff;margin:8px 0}</style>
</head>
<body><div class="box"><h1>⚛️ QTCL Terminal</h1><p>Quantum Blockchain Interface</p><hr style="border-color:#00ff88;margin:15px 0"><p>⚠️ Terminal UI not available</p><p style="font-size:11px;opacity:0.7">Ensure index.html is deployed</p></div></body>
</html>""", 200, {'Content-Type': 'text/html; charset=utf-8'}
    
    logger.info("[HTML] ✓ HTML terminal route registered at /")

# Export Flask app as WSGI application (this is what Gunicorn looks for)
application = app

logger.info("")
logger.info("To run with Gunicorn:")
logger.info("  gunicorn -w 4 -b 0.0.0.0:5000 wsgi_config:application")
logger.info("")
logger.info("To run with uWSGI:")
logger.info("  uwsgi --http :5000 --wsgi-file wsgi_config.py --callable application --processes 4 --threads 2")
logger.info("")

# For direct execution (development only)
if __name__ == '__main__':
    logger.warning("WARNING: Running WSGI app directly (use Gunicorn for production)")
    app.run(
        host=os.getenv('API_HOST', '0.0.0.0'),
        port=int(os.getenv('API_PORT', '5000')),
        debug=os.getenv('FLASK_ENV') == 'development'
    )
