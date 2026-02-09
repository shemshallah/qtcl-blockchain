#!/usr/bin/env python3
"""
db_config.py - CORRECT for qtcl_db_builder.py schema
Matches actual columns from production build
"""

import os
import sys
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from collections import deque
import threading
import time

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG FROM ENVIRONMENT VARIABLES
# ═══════════════════════════════════════════════════════════════════════════

class Config:
    """Production configuration - reads from environment"""
    
    SUPABASE_HOST = os.getenv('SUPABASE_HOST', '')
    SUPABASE_USER = os.getenv('SUPABASE_USER', '')
    SUPABASE_PASSWORD = os.getenv('SUPABASE_PASSWORD', '')
    SUPABASE_PORT = int(os.getenv('SUPABASE_PORT', '5432'))
    SUPABASE_DB = os.getenv('SUPABASE_DB', 'postgres')
    
    DB_POOL_SIZE = 5
    DB_POOL_TIMEOUT = 30
    DB_CONNECT_TIMEOUT = 15
    DB_RETRY_ATTEMPTS = 3
    DB_RETRY_DELAY_SECONDS = 2
    
    @staticmethod
    def validate():
        required = {
            'SUPABASE_HOST': Config.SUPABASE_HOST,
            'SUPABASE_USER': Config.SUPABASE_USER,
            'SUPABASE_PASSWORD': Config.SUPABASE_PASSWORD,
        }
        
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise RuntimeError(f"Missing required env vars: {missing}")
        
        logger.info(f"✓ Config valid: {Config.SUPABASE_HOST}:{Config.SUPABASE_PORT}/{Config.SUPABASE_DB}")

# ═══════════════════════════════════════════════════════════════════════════
# DATABASE CONNECTION WITH RETRY LOGIC
# ═══════════════════════════════════════════════════════════════════════════

class DatabaseConnection:
    """Connection pool with retry logic"""
    
    _instance = None
    _connections = deque(maxlen=Config.DB_POOL_SIZE)
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @staticmethod
    def _get_new_connection():
        """Create connection with retry logic"""
        last_error = None
        
        for attempt in range(1, Config.DB_RETRY_ATTEMPTS + 1):
            try:
                logger.info(f"[DB] Connection attempt {attempt}/{Config.DB_RETRY_ATTEMPTS}...")
                
                conn = psycopg2.connect(
                    host=Config.SUPABASE_HOST,
                    user=Config.SUPABASE_USER,
                    password=Config.SUPABASE_PASSWORD,
                    port=Config.SUPABASE_PORT,
                    database=Config.SUPABASE_DB,
                    connect_timeout=Config.DB_CONNECT_TIMEOUT,
                    application_name='qtcl_api'
                )
                conn.set_session(autocommit=True)
                logger.info(f"[DB] ✓ Connection established")
                return conn
                
            except psycopg2.OperationalError as e:
                last_error = e
                logger.warning(f"[DB] ✗ Connection failed: {e}")
                
                if attempt < Config.DB_RETRY_ATTEMPTS:
                    wait = Config.DB_RETRY_DELAY_SECONDS * attempt
                    logger.info(f"[DB] Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"[DB] All retry attempts failed")
                    raise last_error
            
            except psycopg2.Error as e:
                logger.error(f"[DB] Non-retriable error: {e}")
                raise
    
    @staticmethod
    def get_connection():
        """Get connection from pool or create new one"""
        with DatabaseConnection._lock:
            try:
                conn = DatabaseConnection._connections.popleft()
                if conn and not conn.closed:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                    logger.debug("[DB] Reused pooled connection")
                    return conn
            except (IndexError, psycopg2.Error):
                pass
        
        return DatabaseConnection._get_new_connection()
    
    @staticmethod
    def return_connection(conn):
        """Return connection to pool for reuse"""
        if conn and not conn.closed:
            with DatabaseConnection._lock:
                try:
                    DatabaseConnection._connections.append(conn)
                except Exception as e:
                    logger.warning(f"[DB] Failed to pool connection: {e}")
                    conn.close()
    
    @staticmethod
    def execute(query: str, params: tuple = None) -> list:
        """Execute SELECT query"""
        conn = DatabaseConnection.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params or ())
                results = cur.fetchall()
                logger.debug(f"[DB] Query returned {len(results)} rows")
                return results
        finally:
            DatabaseConnection.return_connection(conn)
    
    @staticmethod
    def execute_update(query: str, params: tuple = None) -> int:
        """Execute INSERT/UPDATE/DELETE query"""
        conn = DatabaseConnection.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(query, params or ())
                affected = cur.rowcount
                logger.debug(f"[DB] Query affected {affected} rows")
                return affected
        finally:
            DatabaseConnection.return_connection(conn)

# ═══════════════════════════════════════════════════════════════════════════
# SETUP FUNCTIONS FOR WSGI
# ═══════════════════════════════════════════════════════════════════════════

def setup_database(app):
    """Initialize database connection in Flask app"""
    try:
        logger.info("[INIT] Validating Supabase configuration...")
        Config.validate()
        
        logger.info("[INIT] Testing database connection...")
        test_conn = DatabaseConnection.get_connection()
        
        with test_conn.cursor() as cur:
            cur.execute("SELECT now()")
            ts = cur.fetchone()
        
        DatabaseConnection.return_connection(test_conn)
        
        logger.info(f"[INIT] ✓ Database ready")
        
        app.config['DB'] = DatabaseConnection
        
        return True
        
    except Exception as e:
        logger.critical(f"[INIT] ✗ Database initialization failed: {e}")
        raise RuntimeError(f"Database error: {e}")

def setup_api_routes(app):
    """Setup Flask routes - CORRECT COLUMNS FOR qtcl_db_builder.py"""
    from flask import jsonify
    
    @app.route('/api/transactions', methods=['GET'])
    def get_transactions():
        try:
            # Correct columns from transactions table
            results = DatabaseConnection.execute(
                "SELECT tx_id, from_user_id, to_user_id, amount, tx_type, status, "
                "created_at, entropy_score "
                "FROM transactions ORDER BY created_at DESC LIMIT 100"
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
            # Correct columns from users table (user_id, not id)
            limit = 50
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
            # Correct columns from pseudoqubits table (pseudoqubit_id, not id)
            limit = 50
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

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    Config.validate()
    conn = DatabaseConnection.get_connection()
    print("✓ Database connection OK")
    DatabaseConnection.return_connection(conn)
