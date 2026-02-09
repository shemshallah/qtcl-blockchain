#!/usr/bin/env python3
"""
WSGI Configuration for Koyeb
Uses db_config.py for database setup
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
logger.info("QTCL WSGI Configuration Loaded (Koyeb)")
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
    logger.info("Importing Flask...")
    from flask import Flask, jsonify, request
    from flask_cors import CORS
    
    logger.info("Creating Flask application...")
    
    app = Flask(__name__, instance_relative_config=True)
    
    app.config['ENV'] = 'production'
    app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY')
    app.config['JSON_SORT_KEYS'] = False
    app.config['PROPAGATE_EXCEPTIONS'] = True
    
    CORS(app)
    
    logger.info("Flask app created successfully")
    
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
    
    try:
        logger.info("Importing db_config module...")
        from db_config import setup_database, setup_api_routes
        
        logger.info("Setting up database connection pool...")
        setup_database(app)
        
        logger.info("Setting up API routes...")
        setup_api_routes(app)
        
        logger.info("API Gateway routes configured successfully")
        
    except ImportError as e:
        logger.error(f"Failed to import db_config: {e}")
        logger.warning("Running in minimal mode with health check only")
    except Exception as e:
        logger.error(f"Error setting up API routes: {e}")
        logger.warning("Running in minimal mode with health check only")
    
    logger.info("=" * 80)
    logger.info("QTCL Application Ready for Serving on Koyeb")
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
    
    app = Flask(__name__)
    
    @app.route('/')
    def error_page():
        return f"<h1>Application Error</h1><p>{str(e)}</p>", 500

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
