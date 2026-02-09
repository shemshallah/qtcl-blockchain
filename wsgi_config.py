#!/usr/bin/env python3
"""
WSGI Configuration for Koyeb - FINAL WORKING VERSION
Imports Flask app from api_gateway.py which has all routes defined
"""

import sys
import os
import logging
import traceback
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
)

logger = logging.getLogger(__name__)
logger.info("=" * 80)
logger.info("QTCL WSGI Configuration - FINAL WORKING VERSION")
logger.info(f"Project Root: {PROJECT_ROOT}")
logger.info(f"Python Version: {sys.version}")
logger.info(f"Timestamp: {datetime.now().isoformat()}")
logger.info("=" * 80)

REQUIRED_ENV_VARS = [
    'SUPABASE_HOST',
    'SUPABASE_USER',
    'SUPABASE_PASSWORD',
    'SUPABASE_PORT',
    'SUPABASE_DB'
]

missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]

if missing_vars:
    logger.error(f"Missing REQUIRED environment variables: {', '.join(missing_vars)}")
    logger.error("Set these in Koyeb Environment Variables before deployment")
else:
    logger.info(f"✓ All required environment variables configured")

os.environ.setdefault('FLASK_ENV', 'production')
os.environ.setdefault('FLASK_SECRET_KEY', os.urandom(16).hex())
os.environ.setdefault('JWT_SECRET', os.urandom(32).hex())

logger.info("Environment variables validated")

try:
    logger.info("=" * 80)
    logger.info("IMPORTING FLASK APP FROM api_gateway.py")
    logger.info("=" * 80)
    
    # IMPORT THE APP THAT api_gateway.py ALREADY CREATED
    # This app has all @app.route decorators already registered
    from api_gateway import app
    
    logger.info("✓ Successfully imported Flask app from api_gateway.py")
    logger.info("✓ All api_gateway routes are already registered!")
    
    # Now add any additional setup needed
    try:
        from flask_cors import CORS
        CORS(app)
        logger.info("✓ CORS enabled")
    except Exception as e:
        logger.warning(f"⚠ Could not enable CORS: {e}")
    
    try:
        logger.info("Setting up database...")
        from db_config import setup_database
        setup_database(app)
        logger.info("✓ Database setup complete")
    except Exception as e:
        logger.warning(f"⚠ Database setup had issues: {e}")
    
    try:
        logger.info("Setting up transaction processor...")
        from transaction_processor import processor, register_transaction_routes
        processor.start()
        register_transaction_routes(app)
        logger.info("✓ Transaction processor ready")
    except ImportError:
        logger.warning("⚠ Transaction processor not available")
    except Exception as e:
        logger.warning(f"⚠ Transaction processor setup issue: {e}")
    
    logger.info("=" * 80)
    logger.info("QTCL Application Ready for Serving")
    logger.info("=" * 80)
    logger.info("Routes available:")
    logger.info("  [HEALTH]")
    logger.info("    GET  /health")
    logger.info("    GET  /api/health")
    logger.info("")
    logger.info("  [AUTHENTICATION]")
    logger.info("    POST /api/auth/login")
    logger.info("    POST /api/auth/verify")
    logger.info("    POST /api/v1/auth/register")
    logger.info("    POST /api/v1/auth/change-password")
    logger.info("")
    logger.info("  [TRANSACTIONS]")
    logger.info("    POST /api/transactions")
    logger.info("    GET  /api/transactions")
    logger.info("    GET  /api/transactions/<tx_id>")
    logger.info("    GET  /api/v1/transactions/menu")
    logger.info("    POST /api/v1/transactions/send-step-1")
    logger.info("    POST /api/v1/transactions/send-step-2")
    logger.info("    POST /api/v1/transactions/send-step-3")
    logger.info("")
    logger.info("  [DATA]")
    logger.info("    GET  /api/users")
    logger.info("    GET  /api/blocks")
    logger.info("    GET  /api/quantum")
    logger.info("")
    logger.info("  [QUANTUM VALIDATION]")
    logger.info("    POST /api/v1/blocks/<hash>/validate")
    logger.info("    GET  /api/v1/blocks/<hash>/measurements")
    logger.info("=" * 80)

except ImportError as e:
    logger.critical(f"✗ CRITICAL: Could not import app from api_gateway.py: {e}")
    logger.critical(f"Make sure api_gateway.py is in the same directory as wsgi_config.py")
    logger.critical(traceback.format_exc())
    
    # Fallback: Create minimal app
    logger.info("Creating fallback Flask app...")
    from flask import Flask, jsonify
    
    app = Flask(__name__)
    
    @app.route('/health')
    def health():
        return jsonify({'status': 'error', 'message': 'api_gateway import failed'})

except Exception as e:
    logger.critical(f"✗ FATAL ERROR: {e}")
    logger.critical(traceback.format_exc())
    
    from flask import Flask, jsonify
    app = Flask(__name__)
    
    @app.route('/')
    def error():
        return f"<h1>Application Error</h1><p>{str(e)}</p>", 500

# Export application for WSGI servers (Koyeb, etc)
application = app

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("Running QTCL in LOCAL DEVELOPMENT MODE")
    print("=" * 80 + "\n")
    
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=True
    )
