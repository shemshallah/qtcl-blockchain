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

# Configure logging
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
    logger.error(f"✗ Missing environment variables: {', '.join(missing_vars)}")
    logger.warning("⚠️  Using defaults for development - NOT FOR PRODUCTION")
else:
    logger.info("✓ All required environment variables configured")

# ═══════════════════════════════════════════════════════════════════════════════════════
# IMPORT MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════════════

try:
    logger.info("Importing main application...")
    
    # Import the Flask application from main_app.py
    from main_app import app, initialize_app
    
    logger.info("✓ Main application imported successfully")
    
    # Initialize the application
    logger.info("Initializing application...")
    if initialize_app():
        logger.info("✓ Application initialized successfully")
    else:
        logger.error("✗ Application initialization failed")
        raise RuntimeError("Failed to initialize application")
    
except ImportError as e:
    logger.critical(f"✗ Failed to import main_app: {e}")
    logger.critical(traceback.format_exc())
    sys.exit(1)
except Exception as e:
    logger.critical(f"✗ Initialization error: {e}")
    logger.critical(traceback.format_exc())
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════════════
# WSGI APPLICATION EXPORT
# ═══════════════════════════════════════════════════════════════════════════════════════

logger.info("=" * 100)
logger.info("✓ WSGI APPLICATION READY FOR DEPLOYMENT")
logger.info("=" * 100)
logger.info("")
logger.info("To run with Gunicorn:")
logger.info("  gunicorn -w 4 -b 0.0.0.0:5000 wsgi_config:application")
logger.info("")
logger.info("To run with uWSGI:")
logger.info("  uwsgi --http :5000 --wsgi-file wsgi_config.py --callable application --processes 4 --threads 2")
logger.info("")

# Export Flask app as WSGI application
application = app

# For direct execution (development only)
if __name__ == '__main__':
    logger.warning("WARNING: Running WSGI app directly (use Gunicorn for production)")
    app.run(
        host=os.getenv('API_HOST', '0.0.0.0'),
        port=int(os.getenv('API_PORT', '5000')),
        debug=os.getenv('FLASK_ENV') == 'development'
    )
