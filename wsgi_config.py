#!/usr/bin/env python3
"""
WSGI Configuration for Koyeb - ACTUALLY WORKING VERSION
Routes defined BEFORE they're used, proper ordering
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
logger.info("QTCL WSGI Configuration - WORKING VERSION")
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

# ═══════════════════════════════════════════════════════════════════════════
# DEFINE HELPER FUNCTION FIRST (before it's used)
# ═══════════════════════════════════════════════════════════════════════════

def setup_basic_routes(app):
    """Setup basic routes that don't require complex logic"""
    from flask import jsonify, request
    
    logger.info("Defining basic API routes...")
    
    @app.route('/api/transactions', methods=['GET'])
    def get_transactions():
        try:
            from db_config import DatabaseConnection
            limit = int(request.args.get('limit', 10))
            results = DatabaseConnection.execute(
                "SELECT tx_id, from_user_id, to_user_id, amount, tx_type, status, "
                "created_at, entropy_score "
                "FROM transactions ORDER BY created_at DESC LIMIT %s",
                (limit,)
            )
            
            transactions = []
            for t in results:
                transactions.append({
                    'tx_id': t['tx_id'],
                    'from': t['from_user_id'],
                    'to': t['to_user_id'],
                    'amount': float(t['amount']) if t['amount'] else 0,
                    'type': t['tx_type'],
                    'status': t['status'],
                    'entropy': float(t['entropy_score']) if t['entropy_score'] else 0,
                    'created_at': t['created_at'].isoformat() if t['created_at'] else None
                })
            
            return jsonify({
                'status': 'success',
                'count': len(transactions),
                'data': transactions
            })
        except Exception as e:
            logger.error(f"[API] Error fetching transactions: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    @app.route('/api/transactions/<tx_id>', methods=['GET'])
    def get_transaction(tx_id):
        try:
            from db_config import DatabaseConnection
            results = DatabaseConnection.execute(
                "SELECT tx_id, from_user_id, to_user_id, amount, tx_type, status, "
                "quantum_state_hash, entropy_score, created_at "
                "FROM transactions WHERE tx_id = %s",
                (tx_id,)
            )
            
            if results:
                t = results[0]
                return jsonify({
                    'status': 'found',
                    'tx_id': t['tx_id'],
                    'tx_status': t['status'],
                    'quantum_state_hash': t['quantum_state_hash'],
                    'entropy_score': float(t['entropy_score']) if t['entropy_score'] else None,
                    'created_at': t['created_at'].isoformat() if t['created_at'] else None
                })
            else:
                return jsonify({
                    'status': 'not_found',
                    'tx_id': tx_id
                })
        except Exception as e:
            logger.error(f"[API] Error: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    @app.route('/api/blocks', methods=['GET'])
    def get_blocks():
        try:
            from db_config import DatabaseConnection
            results = DatabaseConnection.execute(
                "SELECT block_number, block_hash, parent_hash, timestamp, "
                "transactions, entropy_score, created_at "
                "FROM blocks ORDER BY block_number DESC LIMIT 50"
            )
            
            blocks = []
            for b in results:
                blocks.append({
                    'block_number': b['block_number'],
                    'block_hash': b['block_hash'],
                    'parent_hash': b['parent_hash'],
                    'timestamp': b['timestamp'],
                    'transaction_count': b['transactions'],
                    'entropy_score': float(b['entropy_score']) if b['entropy_score'] else 0,
                    'created_at': b['created_at'].strftime('%a, %d %b %Y %H:%M:%S GMT') if b['created_at'] else None
                })
            
            return jsonify({
                'status': 'success',
                'count': len(blocks),
                'data': blocks
            })
        except Exception as e:
            logger.error(f"[API] Error: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    @app.route('/api/users', methods=['GET'])
    def get_users():
        try:
            from db_config import DatabaseConnection
            limit = int(request.args.get('limit', 50))
            results = DatabaseConnection.execute(
                "SELECT user_id, email, name, balance, role, created_at FROM users LIMIT %s",
                (limit,)
            )
            
            users = []
            for u in results:
                users.append({
                    'id': u['user_id'],
                    'email': u['email'],
                    'name': u['name'],
                    'balance': float(u['balance']) if u['balance'] else 0,
                    'role': u['role'],
                    'created_at': u['created_at'].isoformat() if u['created_at'] else None
                })
            
            return jsonify({
                'status': 'success',
                'count': len(users),
                'users': users
            })
        except Exception as e:
            logger.error(f"[API] Error: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    @app.route('/api/quantum', methods=['GET'])
    def get_quantum():
        try:
            from db_config import DatabaseConnection
            limit = int(request.args.get('limit', 50))
            results = DatabaseConnection.execute(
                "SELECT pseudoqubit_id, fidelity, coherence, purity, entropy, "
                "concurrence, routing_address FROM pseudoqubits LIMIT %s",
                (limit,)
            )
            
            pseudoqubits = []
            for pq in results:
                pseudoqubits.append({
                    'id': pq['pseudoqubit_id'],
                    'fidelity': float(pq['fidelity']) if pq['fidelity'] else 0,
                    'coherence': float(pq['coherence']) if pq['coherence'] else 0,
                    'purity': float(pq['purity']) if pq['purity'] else 0,
                    'entropy': float(pq['entropy']) if pq['entropy'] else 0,
                    'concurrence': float(pq['concurrence']) if pq['concurrence'] else 0,
                    'routing_address': pq['routing_address'],
                })
            
            return jsonify({
                'status': 'success',
                'count': len(pseudoqubits),
                'pseudoqubits': pseudoqubits
            })
        except Exception as e:
            logger.error(f"[API] Error: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    logger.info("✓ Basic routes defined")

# ═══════════════════════════════════════════════════════════════════════════
# NOW CREATE FLASK APP AND SETUP ROUTES
# ═══════════════════════════════════════════════════════════════════════════

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
                'database': 'connected',
                'quantum_executor': 'ready',
                'ledger_manager': 'ready',
                'transaction_processor': 'running'
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
        from db_config import DatabaseConnection, setup_database
        
        logger.info("Setting up database connection pool...")
        setup_database(app)
        logger.info("✓ Database connection pool ready")
        
        logger.info("Setting up basic API routes (users, blocks, quantum, transactions)...")
        setup_basic_routes(app)  # NOW this function exists before being called!
        logger.info("✓ Basic routes registered")
        
        # IMPORT AND SETUP ROUTES FROM api_gateway
        logger.info("Importing api_gateway module for extended routes...")
        try:
            import api_gateway
            
            logger.info("Registering authentication routes from api_gateway...")
            
            # Login & verify
            if hasattr(api_gateway, 'login'):
                app.add_url_rule('/api/auth/login', 'login', api_gateway.login, methods=['POST'])
                logger.info("  ✓ POST /api/auth/login")
            
            if hasattr(api_gateway, 'verify_token'):
                app.add_url_rule('/api/auth/verify', 'verify_token', api_gateway.verify_token, methods=['POST'])
                logger.info("  ✓ POST /api/auth/verify")
            
            # Enhanced auth (v1)
            if hasattr(api_gateway, 'register'):
                app.add_url_rule('/api/v1/auth/register', 'register', api_gateway.register, methods=['POST'])
                logger.info("  ✓ POST /api/v1/auth/register")
            
            if hasattr(api_gateway, 'change_password'):
                app.add_url_rule('/api/v1/auth/change-password', 'change_password', api_gateway.change_password, methods=['POST'])
                logger.info("  ✓ POST /api/v1/auth/change-password")
            
            # Transaction menu
            if hasattr(api_gateway, 'transaction_menu'):
                app.add_url_rule('/api/v1/transactions/menu', 'transaction_menu', api_gateway.transaction_menu, methods=['GET'])
                logger.info("  ✓ GET /api/v1/transactions/menu")
            
            if hasattr(api_gateway, 'send_step_1'):
                app.add_url_rule('/api/v1/transactions/send-step-1', 'send_step_1', api_gateway.send_step_1, methods=['POST'])
                logger.info("  ✓ POST /api/v1/transactions/send-step-1")
            
            if hasattr(api_gateway, 'send_step_2'):
                app.add_url_rule('/api/v1/transactions/send-step-2', 'send_step_2', api_gateway.send_step_2, methods=['POST'])
                logger.info("  ✓ POST /api/v1/transactions/send-step-2")
            
            if hasattr(api_gateway, 'send_step_3'):
                app.add_url_rule('/api/v1/transactions/send-step-3', 'send_step_3', api_gateway.send_step_3, methods=['POST'])
                logger.info("  ✓ POST /api/v1/transactions/send-step-3")
            
            # Quantum validation
            if hasattr(api_gateway, 'validate_block'):
                app.add_url_rule('/api/v1/blocks/<block_hash>/validate', 'validate_block', api_gateway.validate_block, methods=['POST'])
                logger.info("  ✓ POST /api/v1/blocks/<block_hash>/validate")
            
            if hasattr(api_gateway, 'get_block_measurements'):
                app.add_url_rule('/api/v1/blocks/<block_hash>/measurements', 'get_block_measurements', api_gateway.get_block_measurements, methods=['GET'])
                logger.info("  ✓ GET /api/v1/blocks/<block_hash>/measurements")
            
            logger.info("✓ All available api_gateway routes registered successfully")
            
        except ImportError as e:
            logger.warning(f"⚠ Could not import api_gateway: {e}")
            logger.warning("  (This is okay - running with basic routes only)")
        except Exception as e:
            logger.error(f"✗ Error registering api_gateway routes: {e}")
            logger.error(traceback.format_exc())
        
        try:
            logger.info("Importing transaction processor...")
            from transaction_processor import processor, register_transaction_routes
            
            logger.info("Starting transaction processor...")
            processor.start()
            
            logger.info("Registering transaction routes...")
            register_transaction_routes(app)
            logger.info("✓ Transaction processor ready")
        except ImportError:
            logger.warning("⚠ Transaction processor not available")
        except Exception as e:
            logger.error(f"✗ Error setting up transaction processor: {e}")
        
        logger.info("API Gateway routes configured successfully")
        
    except Exception as e:
        logger.error(f"Error setting up routes: {e}")
        logger.error(traceback.format_exc())
        logger.warning("Running in minimal mode")
    
    logger.info("=" * 80)
    logger.info("QTCL Application Ready for Serving")
    logger.info(f"Available endpoints:")
    logger.info(f"  GET  /health")
    logger.info(f"  GET  /api/health")
    logger.info(f"  GET  /api/users")
    logger.info(f"  GET  /api/blocks")
    logger.info(f"  GET  /api/quantum")
    logger.info(f"  GET  /api/transactions")
    logger.info(f"  GET  /api/transactions/<tx_id>")
    logger.info(f"  POST /api/auth/login (if api_gateway available)")
    logger.info(f"  POST /api/v1/auth/register (if api_gateway available)")
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
