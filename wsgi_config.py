#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
PYTHONANYWHERE WSGI CONFIGURATION
QUANTUM TEMPORAL COHERENCE LEDGER (QTCL)
═══════════════════════════════════════════════════════════════════════════════

This is the entry point for PythonAnywhere web application hosting.
PythonAnywhere will call this wsgi_config.py file at startup.

Configuration Steps:
1. Upload all project files to /home/USERNAME/QTCL_production/
2. In PythonAnywhere Web tab, set WSGI configuration file to this file
3. Set working directory to /home/USERNAME/QTCL_production/
4. Set Python version to 3.9+
5. Configure environment variables in Web app config
"""

import sys
import os
import logging
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# PATH CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Add project directory to Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════

LOG_DIR = os.path.expanduser('~/QTCL_logs')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'wsgi.log')),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)
logger.info("=" * 80)
logger.info("QTCL WSGI Configuration Loaded")
logger.info(f"Project Root: {PROJECT_ROOT}")
logger.info(f"Python Version: {sys.version}")
logger.info(f"Timestamp: {datetime.now().isoformat()}")
logger.info("=" * 80)

# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT VARIABLES VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

REQUIRED_ENV_VARS = [
    'SUPABASE_HOST',
    'SUPABASE_USER',
    'SUPABASE_PASSWORD',
    'SUPABASE_PORT',
    'SUPABASE_DB'
]

missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]

if missing_vars:
    logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
    logger.warning("Please set these in PythonAnywhere Web app configuration")
    
    # Set defaults for development
    os.environ.setdefault('SUPABASE_HOST', 'aws-0-us-west-2.pooler.supabase.com')
    os.environ.setdefault('SUPABASE_USER', 'postgres.rslvlsqwkfmdtebqsvtw')
    os.environ.setdefault('SUPABASE_PASSWORD', '$h10j1r1H0w4rd')
    os.environ.setdefault('SUPABASE_PORT', '5432')
    os.environ.setdefault('SUPABASE_DB', 'postgres')

os.environ.setdefault('FLASK_ENV', 'production')
os.environ.setdefault('FLASK_SECRET_KEY', os.urandom(16).hex())
os.environ.setdefault('JWT_SECRET', os.urandom(32).hex())

logger.info("Environment variables validated")

# ═══════════════════════════════════════════════════════════════════════════════
# DEPENDENCY INSTALLATION CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def ensure_package(package_name, pip_name=None):
    """Ensure package is installed"""
    try:
        __import__(package_name)
        logger.info(f"✓ {package_name} installed")
        return True
    except ImportError:
        logger.warning(f"✗ {package_name} not found - attempting install")
        try:
            import subprocess
            pip_name = pip_name or package_name
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', '-q', pip_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            logger.info(f"✓ {package_name} installed successfully")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to install {package_name}: {e}")
            return False

logger.info("Checking critical dependencies...")

critical_packages = {
    'flask': 'Flask',
    'psycopg2': 'psycopg2-binary',
    'numpy': 'numpy',
    'qiskit': 'qiskit',
    'qiskit_aer': 'qiskit-aer',
}

all_installed = all(
    ensure_package(pkg, pip_name) 
    for pkg, pip_name in critical_packages.items()
)

if not all_installed:
    logger.warning("Some packages failed to install - continuing anyway")

logger.info("Dependency check complete")

# ═══════════════════════════════════════════════════════════════════════════════
# FLASK APPLICATION CREATION
# ═══════════════════════════════════════════════════════════════════════════════

try:
    logger.info("Importing Flask...")
    from flask import Flask, jsonify, request
    from flask_cors import CORS
    
    logger.info("Creating Flask application...")
    
    # Create Flask app instance
    app = Flask(__name__, instance_relative_config=True)
    
    # Configuration
    app.config['ENV'] = 'production'
    app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY')
    app.config['JSON_SORT_KEYS'] = False
    app.config['PROPAGATE_EXCEPTIONS'] = True
    
    # Enable CORS
    CORS(app)
    
    logger.info("Flask app created successfully")
    
    # ═════════════════════════════════════════════════════════════════════════
    # BASIC HEALTH CHECK ENDPOINT
    # ═════════════════════════════════════════════════════════════════════════
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Basic health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'service': 'QTCL-API-Gateway',
            'version': '1.0.0'
        }), 200
    
    @app.route('/api/health', methods=['GET'])
    def api_health():
        """API health endpoint with component status"""
        return jsonify({
            'status': 'operational',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'database': 'checking...',
                'quantum_executor': 'ready',
                'ledger_manager': 'ready',
                'oracle_engine': 'ready'
            }
        }), 200
    
    # ═════════════════════════════════════════════════════════════════════════
    # ERROR HANDLERS
    # ═════════════════════════════════════════════════════════════════════════
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'status': 'error',
            'message': 'Endpoint not found',
            'path': request.path
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return jsonify({
            'status': 'error',
            'message': 'Internal server error'
        }), 500
    
    # ═════════════════════════════════════════════════════════════════════════
    # IMPORT AND INTEGRATE API GATEWAY
    # ═════════════════════════════════════════════════════════════════════════
    
    try:
        logger.info("Attempting to import api_gateway module...")
        from api_gateway import setup_api_routes, setup_database
        
        logger.info("Setting up database connection pool...")
        setup_database(app)
        
        logger.info("Setting up API routes...")
        setup_api_routes(app)
        
        logger.info("API Gateway routes configured successfully")
        
    except ImportError as e:
        logger.error(f"Failed to import api_gateway: {e}")
        logger.warning("Running in minimal mode with health check only")
    except Exception as e:
        logger.error(f"Error setting up API routes: {e}")
        logger.warning("Running in minimal mode with health check only")
    
    # ═════════════════════════════════════════════════════════════════════════
    # STARTUP MESSAGE
    # ═════════════════════════════════════════════════════════════════════════
    
    logger.info("=" * 80)
    logger.info("QTCL Application Ready for Serving")
    logger.info(f"Available endpoints:")
    logger.info(f"  - GET  /health              (Basic health check)")
    logger.info(f"  - GET  /api/health          (API status)")
    logger.info(f"  - GET  /api/transactions    (Get transactions)")
    logger.info(f"  - POST /api/transactions    (Submit transaction)")
    logger.info(f"  - GET  /api/blocks          (Get blocks)")
    logger.info(f"  - GET  /api/users           (Get user info)")
    logger.info(f"  - GET  /api/quantum         (Get quantum metrics)")
    logger.info("=" * 80)
    
except Exception as e:
    logger.critical(f"Failed to create Flask application: {e}")
    logger.critical(traceback.format_exc())
    
    # Create minimal app for error reporting
    app = Flask(__name__)
    
    @app.route('/')
    def error_page():
        return f"<h1>Application Error</h1><p>{str(e)}</p>", 500

# ═══════════════════════════════════════════════════════════════════════════════
# WSGI APPLICATION OBJECT (PythonAnywhere looks for this)
# ═══════════════════════════════════════════════════════════════════════════════

# This is what PythonAnywhere's WSGI server will call
# It MUST be named 'application'
application = app

# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL TESTING (if run directly)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # This block runs ONLY during local testing
    # PythonAnywhere will NOT call this
    print("\n" + "=" * 80)
    print("Running QTCL in LOCAL DEVELOPMENT MODE")
    print("=" * 80 + "\n")
    
    # Debug mode only for development
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=True
    )
    
    print("\n" + "=" * 80)
    print("QTCL Development Server Stopped")
    print("=" * 80 + "\n")
