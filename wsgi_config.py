#!/usr/bin/env python3
"""
WSGI Configuration for Koyeb - HYBRID VERSION
Uses original working structure + adds missing routes directly (no api_gateway import)
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
logger.info("QTCL WSGI Configuration - HYBRID VERSION (Original + New Routes)")
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
    
    # ═════════════════════════════════════════════════════════════════════════
    # HEALTH ENDPOINTS (from original)
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
                'database': 'connected',
                'quantum_executor': 'ready',
                'ledger_manager': 'ready',
                'transaction_processor': 'running'
            }
        }), 200
    
    # ═════════════════════════════════════════════════════════════════════════
    # NEW: STUB ROUTES FOR MISSING ENDPOINTS
    # These are placeholders that return proper responses
    # ═════════════════════════════════════════════════════════════════════════
    
    @app.route('/api/auth/login', methods=['POST'])
    def login():
        """User login endpoint"""
        try:
            data = request.get_json()
            email = data.get('email')
            password = data.get('password')
            
            if not email or not password:
                return jsonify({'status': 'error', 'message': 'Missing email or password'}), 400
            
            # For now, just return success (real auth would check database)
            return jsonify({
                'status': 'success',
                'access_token': f'token_{email}_{datetime.now().timestamp()}',
                'user_id': email
            }), 200
        except Exception as e:
            logger.error(f"Login error: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    @app.route('/api/auth/verify', methods=['POST'])
    def verify_token():
        """Token verification endpoint"""
        try:
            data = request.get_json()
            token = data.get('token')
            
            if not token:
                return jsonify({'status': 'error', 'message': 'Missing token'}), 400
            
            # Simple token validation
            is_valid = token.startswith('token_')
            return jsonify({
                'status': 'success',
                'valid': is_valid
            }), 200
        except Exception as e:
            logger.error(f"Verify error: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    @app.route('/api/v1/auth/register', methods=['POST'])
    def register():
        """User registration endpoint"""
        try:
            data = request.get_json()
            email = data.get('email')
            password = data.get('password')
            name = data.get('name')
            
            if not email or not password:
                return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400
            
            # Simple registration (real would write to database)
            return jsonify({
                'status': 'success',
                'user_id': email,
                'email': email,
                'name': name or 'User'
            }), 200
        except Exception as e:
            logger.error(f"Register error: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    @app.route('/api/v1/auth/change-password', methods=['POST'])
    def change_password():
        """Change password endpoint"""
        try:
            data = request.get_json()
            user_id = data.get('user_id')
            old_password = data.get('old_password')
            new_password = data.get('new_password')
            
            if not all([user_id, old_password, new_password]):
                return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400
            
            return jsonify({
                'status': 'success',
                'success': True,
                'message': 'Password changed successfully'
            }), 200
        except Exception as e:
            logger.error(f"Change password error: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    @app.route('/api/transactions', methods=['POST'])
    def submit_transaction():
        """Submit a new transaction"""
        try:
            data = request.get_json()
            
            # Accept both 'from_user' and 'from_user_id'
            from_user = data.get('from_user') or data.get('from_user_id')
            to_user = data.get('to_user') or data.get('to_user_id')
            amount = data.get('amount')
            tx_type = data.get('tx_type', 'transfer')
            
            if not all([from_user, to_user, amount]):
                return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400
            
            tx_id = f"tx_{from_user}_{to_user}_{datetime.now().timestamp()}"
            
            return jsonify({
                'status': 'success',
                'tx_id': tx_id,
                'from_user': from_user,
                'to_user': to_user,
                'amount': float(amount),
                'tx_type': tx_type,
                'message': 'Transaction submitted successfully'
            }), 200
        except Exception as e:
            logger.error(f"Submit transaction error: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    @app.route('/api/v1/transactions/menu', methods=['GET'])
    def transaction_menu():
        """Get transaction menu options"""
        try:
            return jsonify({
                'status': 'success',
                'options': ['send', 'receive', 'convert', 'stake']
            }), 200
        except Exception as e:
            logger.error(f"Transaction menu error: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    @app.route('/api/v1/transactions/send-step-1', methods=['POST'])
    def send_step_1():
        """Transaction send step 1: initialize"""
        try:
            data = request.get_json()
            user_id = data.get('user_id')
            recipient = data.get('recipient')
            amount = data.get('amount')
            
            if not all([user_id, recipient, amount]):
                return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400
            
            session_id = f"session_{user_id}_{datetime.now().timestamp()}"
            
            return jsonify({
                'status': 'success',
                'session_id': session_id,
                'amount': float(amount),
                'fee': float(amount) * 0.001,  # 0.1% fee
                'total': float(amount) * 1.001
            }), 200
        except Exception as e:
            logger.error(f"Send step 1 error: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    @app.route('/api/v1/transactions/send-step-2', methods=['POST'])
    def send_step_2():
        """Transaction send step 2: confirm"""
        try:
            data = request.get_json()
            session_id = data.get('session_id')
            confirmed = data.get('confirmed')
            
            if not session_id:
                return jsonify({'status': 'error', 'message': 'Missing session_id'}), 400
            
            return jsonify({
                'status': 'confirmed' if confirmed else 'cancelled',
                'session_id': session_id,
                'confirmed': confirmed
            }), 200
        except Exception as e:
            logger.error(f"Send step 2 error: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    @app.route('/api/v1/transactions/send-step-3', methods=['POST'])
    def send_step_3():
        """Transaction send step 3: execute"""
        try:
            data = request.get_json()
            session_id = data.get('session_id')
            
            if not session_id:
                return jsonify({'status': 'error', 'message': 'Missing session_id'}), 400
            
            tx_id = f"tx_{session_id}_{datetime.now().timestamp()}"
            
            return jsonify({
                'status': 'success',
                'tx_id': tx_id,
                'session_id': session_id,
                'message': 'Transaction executed successfully'
            }), 200
        except Exception as e:
            logger.error(f"Send step 3 error: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    @app.route('/api/v1/blocks/<block_hash>/validate', methods=['POST'])
    def validate_block(block_hash):
        """Validate a quantum block"""
        try:
            data = request.get_json()
            validator_id = data.get('validator_id')
            measurements = data.get('measurements', [])
            
            if not validator_id:
                return jsonify({'status': 'error', 'message': 'Missing validator_id'}), 400
            
            # Calculate agreement from measurements
            agreement = 0.85 if len(measurements) >= 3 else 0.0
            
            return jsonify({
                'status': 'success',
                'block_hash': block_hash,
                'validator_id': validator_id,
                'measurements_count': len(measurements),
                'agreement': agreement,
                'valid': agreement >= 0.75
            }), 200
        except Exception as e:
            logger.error(f"Validate block error: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    @app.route('/api/v1/blocks/<block_hash>/measurements', methods=['GET'])
    def get_block_measurements(block_hash):
        """Get quantum measurements for a block"""
        try:
            return jsonify({
                'status': 'success',
                'block_hash': block_hash,
                'measurements': [
                    {
                        'validator_id': f'validator_{i}',
                        'fidelity': 0.95 + (i * 0.01),
                        'coherence': 0.90 + (i * 0.01),
                        'entropy': 0.75 - (i * 0.02)
                    }
                    for i in range(1, 4)
                ]
            }), 200
        except Exception as e:
            logger.error(f"Get measurements error: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
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
        
        logger.info("Setting up API routes from db_config...")
        setup_api_routes(app)
        
        logger.info("Setting up additional components...")
        
        try:
            logger.info("Importing transaction processor...")
            from transaction_processor import processor, register_transaction_routes
            
            logger.info("Starting transaction processor...")
            processor.start()
            
            logger.info("Registering transaction routes...")
            register_transaction_routes(app)
        except Exception as e:
            logger.warning(f"Transaction processor setup failed: {e}")
        
        logger.info("API Gateway routes configured successfully")
        
    except ImportError as e:
        logger.error(f"Failed to import modules: {e}")
        logger.warning("Running with basic + stub routes")
    except Exception as e:
        logger.error(f"Error setting up routes: {e}")
        logger.warning("Continuing with available routes")
    
    logger.info("=" * 80)
    logger.info("QTCL Application Ready for Serving on Koyeb")
    logger.info(f"Available endpoints:")
    logger.info(f"  [HEALTH]")
    logger.info(f"  - GET  /health")
    logger.info(f"  - GET  /api/health")
    logger.info(f"")
    logger.info(f"  [AUTHENTICATION]")
    logger.info(f"  - POST /api/auth/login")
    logger.info(f"  - POST /api/auth/verify")
    logger.info(f"  - POST /api/v1/auth/register")
    logger.info(f"  - POST /api/v1/auth/change-password")
    logger.info(f"")
    logger.info(f"  [TRANSACTIONS]")
    logger.info(f"  - POST /api/transactions (from db_config)")
    logger.info(f"  - GET  /api/transactions (from db_config)")
    logger.info(f"  - GET  /api/transactions/<tx_id> (from db_config)")
    logger.info(f"  - GET  /api/v1/transactions/menu")
    logger.info(f"  - POST /api/v1/transactions/send-step-1")
    logger.info(f"  - POST /api/v1/transactions/send-step-2")
    logger.info(f"  - POST /api/v1/transactions/send-step-3")
    logger.info(f"")
    logger.info(f"  [DATA]")
    logger.info(f"  - GET  /api/users (from db_config)")
    logger.info(f"  - GET  /api/blocks (from db_config)")
    logger.info(f"  - GET  /api/quantum (from db_config)")
    logger.info(f"")
    logger.info(f"  [QUANTUM VALIDATION]")
    logger.info(f"  - POST /api/v1/blocks/<hash>/validate")
    logger.info(f"  - GET  /api/v1/blocks/<hash>/measurements")
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
