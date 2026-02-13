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
