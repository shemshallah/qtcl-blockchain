#!/usr/bin/env python3
"""
db_config.py - Drop-in database configuration for api_gateway
Fixes syntax errors and uses environment variables
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
# CONFIG - ALL FROM ENVIRONMENT VARIABLES
# ═══════════════════════════════════════════════════════════════════════════

class Config:
    """Production configuration - all from environment"""
    
    SUPABASE_HOST = os.getenv('SUPABASE_HOST', 'aws-0-us-west-2.pooler.supabase.com')
    SUPABASE_USER = os.getenv('SUPABASE_USER', 'postgres.rslvlsqwkfmdtebqsvtw')
    SUPABASE_PASSWORD = os.getenv('SUPABASE_PASSWORD', '')
    SUPABASE_PORT = int(os.getenv('SUPABASE_PORT', '5432'))
    SUPABASE_DB = os.getenv('SUPABASE_DB', 'postgres')
    
    API_PORT = 5000
    API_HOST = "0.0.0.0"
    DEBUG = False
    TESTING = False
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    JSONIFY_PRETTYPRINT_REGULAR = True
    
    TOKEN_TOTAL_SUPPLY = 1_000_000_000
    TOKEN_DECIMALS = 18
    TOKEN_WEI_PER_UNIT = 10 ** TOKEN_DECIMALS
    
    QISKIT_SHOTS = 1024
    QISKIT_QUBITS = 8
    MAX_CIRCUIT_EXECUTION_TIME_MS = 200
    SUPERPOSITION_TIMEOUT_SECONDS = 300
    COHERENCE_REFRESH_INTERVAL_SECONDS = 5
    ENTROPY_THRESHOLD_FOR_SUPERPOSITION = 0.8
    
    MAX_TXS_PER_SECOND = 1000
    MAX_TXS_PER_USER_PER_MINUTE = 60
    MAX_TXS_PER_USER_PER_HOUR = 500
    RATE_LIMIT_WINDOW_SECONDS = 60
    
    INITIAL_GAS_PRICE = 1
    GAS_PER_TRANSFER = 21000
    GAS_PER_CONTRACT_CALL = 100000
    GAS_PER_STAKE = 50000
    BLOCK_GAS_LIMIT = 8_000_000
    
    BLOCK_REWARD_EPOCH_1_BLOCKS = 52_560
    BLOCK_REWARD_EPOCH_1 = 100
    BLOCK_REWARD_EPOCH_2 = 50
    BLOCK_REWARD_EPOCH_3 = 25
    
    TESSELLATION_TYPE = (8, 3)
    MAX_TESSELLATION_DEPTH = 10
    TOTAL_PSEUDOQUBITS_ESTIMATED = 106_496
    
    MAX_EUCLIDEAN_DISTANCE_FOR_ROUTE = 0.3
    BANDWIDTH_SCORE_CALCULATION = "1.0 - (distance / max_distance)"
    
    USER_CACHE_TTL_SECONDS = 300
    QUBIT_CACHE_TTL_SECONDS = 300
    BLOCK_CACHE_TTL_SECONDS = 600
    ENABLE_QUERY_CACHING = True
    
    JWT_ALGORITHM = 'HS256'
    JWT_EXPIRATION_HOURS = 24
    ALLOWED_ORIGINS = ['*']
    ENABLE_REQUEST_LOGGING = True
    
    DB_POOL_SIZE = 5
    DB_POOL_TIMEOUT = 30
    DB_CONNECT_TIMEOUT = 15
    DB_RETRY_ATTEMPTS = 3
    DB_RETRY_DELAY_SECONDS = 2
    
    @staticmethod
    def validate():
        """Validate required config"""
        if not Config.SUPABASE_PASSWORD:
            raise RuntimeError("SUPABASE_PASSWORD not set")
        logger.info(f"✓ Config: {Config.SUPABASE_HOST}:{Config.SUPABASE_PORT}/{Config.SUPABASE_DB}")

# ═══════════════════════════════════════════════════════════════════════════
# DATABASE CONNECTION POOL WITH RETRY
# ═══════════════════════════════════════════════════════════════════════════

class DatabaseConnection:
    """Thread-safe connection pool with retry logic"""
    
    _instance = None
    _connections = deque(maxlen=Config.DB_POOL_SIZE)
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @staticmethod
    def _get_new_connection():
        """Create connection with retries"""
        last_error = None
        
        for attempt in range(1, Config.DB_RETRY_ATTEMPTS + 1):
            try:
                logger.info(f"[DB] Connection attempt {attempt}/{Config.DB_RETRY_ATTEMPTS}")
                
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
                logger.info(f"[DB] ✓ Connected")
                return conn
                
            except psycopg2.OperationalError as e:
                last_error = e
                logger.warning(f"[DB] Failed: {e}")
                
                if attempt < Config.DB_RETRY_ATTEMPTS:
                    wait = Config.DB_RETRY_DELAY_SECONDS * attempt
                    logger.info(f"[DB] Retry in {wait}s...")
                    time.sleep(wait)
                else:
                    raise last_error
            
            except psycopg2.Error as e:
                logger.error(f"[DB] Error: {e}")
                raise
    
    @staticmethod
    def get_connection():
        """Get connection from pool or create new"""
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
        """Return connection to pool"""
        if conn and not conn.closed:
            with DatabaseConnection._lock:
                try:
                    DatabaseConnection._connections.append(conn)
                except Exception as e:
                    logger.warning(f"[DB] Pool error: {e}")
                    conn.close()
        else:
            logger.warning("[DB] Cannot pool closed connection")
    
    @staticmethod
    def execute(query: str, params: tuple = None) -> list:
        """Execute SELECT"""
        conn = DatabaseConnection.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params or ())
                return cur.fetchall()
        finally:
            DatabaseConnection.return_connection(conn)
    
    @staticmethod
    def execute_update(query: str, params: tuple = None) -> int:
        """Execute INSERT/UPDATE/DELETE"""
        conn = DatabaseConnection.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(query, params or ())
                return cur.rowcount
        finally:
            DatabaseConnection.return_connection(conn)

# ═══════════════════════════════════════════════════════════════════════════
# SETUP FUNCTIONS FOR WSGI
# ═══════════════════════════════════════════════════════════════════════════

def setup_database(app):
    """Initialize database"""
    try:
        logger.info("[INIT] Validating Supabase...")
        Config.validate()
        
        logger.info("[INIT] Testing connection...")
        conn = DatabaseConnection.get_connection()
        
        with conn.cursor() as cur:
            cur.execute("SELECT now()")
            ts = cur.fetchone()
        
        DatabaseConnection.return_connection(conn)
        logger.info(f"[INIT] ✓ Database ready")
        
        app.config['DB'] = DatabaseConnection
        return True
        
    except Exception as e:
        logger.critical(f"[INIT] ✗ Database failed: {e}")
        raise RuntimeError(f"Database error: {e}")

def setup_api_routes(app):
    """Setup API endpoints"""
    from flask import jsonify
    
    @app.route('/api/transactions', methods=['GET'])
    def get_transactions():
        try:
            results = DatabaseConnection.execute(
                "SELECT * FROM transactions ORDER BY created_at DESC LIMIT 100"
            )
            return jsonify({'status': 'success', 'count': len(results), 'data': results})
        except Exception as e:
            logger.error(f"[API] Error: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    @app.route('/api/blocks', methods=['GET'])
    def get_blocks():
        try:
            results = DatabaseConnection.execute(
                "SELECT * FROM blocks ORDER BY block_number DESC LIMIT 50"
            )
            return jsonify({'status': 'success', 'count': len(results), 'data': results})
        except Exception as e:
            logger.error(f"[API] Error: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    @app.route('/api/users', methods=['GET'])
    def get_users():
        try:
            results = DatabaseConnection.execute(
                "SELECT id, username, created_at FROM users LIMIT 50"
            )
            return jsonify({'status': 'success', 'count': len(results), 'data': results})
        except Exception as e:
            logger.error(f"[API] Error: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    @app.route('/api/quantum', methods=['GET'])
    def get_quantum():
        try:
            results = DatabaseConnection.execute(
                "SELECT id, qubit_state, coherence_level FROM pseudoqubits LIMIT 50"
            )
            return jsonify({'status': 'success', 'count': len(results), 'data': results})
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
