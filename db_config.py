#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════
db_config.py - DATABASE UTILITIES FOR QTCL API
Clean database connection management - NO ROUTE DEFINITIONS
All routes defined in main_app.py only
═══════════════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import logging
import psycopg2
import time
import threading
from psycopg2.extras import RealDictCursor
from collections import deque
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════════════
# CONFIG FROM ENVIRONMENT VARIABLES
# ═══════════════════════════════════════════════════════════════════════════════════════

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
        """Validate required environment variables"""
        required = {
            'SUPABASE_HOST': Config.SUPABASE_HOST,
            'SUPABASE_USER': Config.SUPABASE_USER,
            'SUPABASE_PASSWORD': Config.SUPABASE_PASSWORD,
        }
        
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise RuntimeError(f"Missing required env vars: {missing}")
        
        logger.info(f"✓ Config valid: {Config.SUPABASE_HOST}:{Config.SUPABASE_PORT}/{Config.SUPABASE_DB}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# DATABASE CONNECTION WITH RETRY LOGIC
# ═══════════════════════════════════════════════════════════════════════════════════════

class DatabaseConnection:
    """Singleton connection pool with retry logic"""
    
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
                logger.debug(f"[DB] Query returned {len(results) if results else 0} rows")
                return results or []
        except Exception as e:
            logger.error(f"[DB] Query error: {e}")
            return []
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
        except Exception as e:
            logger.error(f"[DB] Update error: {e}")
            return 0
        finally:
            DatabaseConnection.return_connection(conn)

    @staticmethod
    def execute_one(query: str, params: tuple = None):
        """Execute SELECT query and return first row only"""
        conn = DatabaseConnection.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params or ())
                result = cur.fetchone()
                return result
        except Exception as e:
            logger.error(f"[DB] Query error: {e}")
            return None
        finally:
            DatabaseConnection.return_connection(conn)
    
    @staticmethod
    def execute_many(query: str, params_list: List[tuple] = None) -> int:
        """Execute multiple INSERT/UPDATE/DELETE queries"""
        if not params_list:
            return 0
        conn = DatabaseConnection.get_connection()
        try:
            with conn.cursor() as cur:
                total = 0
                for params in params_list:
                    cur.execute(query, params or ())
                    total += cur.rowcount
                return total
        except Exception as e:
            logger.error(f"[DB] Batch error: {e}")
            return 0
        finally:
            DatabaseConnection.return_connection(conn)

# ═══════════════════════════════════════════════════════════════════════════════════════
# SETUP FUNCTIONS FOR WSGI
# ═══════════════════════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Testing database configuration...")
    
    try:
        Config.validate()
        logger.info("✓ Configuration valid")
        
        conn = DatabaseConnection.get_connection()
        logger.info("✓ Database connection OK")
        DatabaseConnection.return_connection(conn)
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        sys.exit(1)
