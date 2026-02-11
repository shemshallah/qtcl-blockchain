#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════
db_config.py - DATABASE UTILITIES FOR QTCL API WITH DB_BUILDER V2 INTEGRATION
Integrated connection management + db_builder_v2 functionality in one module
═══════════════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import logging
import psycopg2
import time
import threading
import subprocess
from psycopg2.extras import RealDictCursor
from collections import deque
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════════════
# ENSURE DB_BUILDER_V2 DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════════════════════

def ensure_package(package, pip_name=None):
    """Install missing packages"""
    try:
        __import__(package)
    except ImportError:
        print(f"📦 Installing {pip_name or package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--break-system-packages", pip_name or package])

ensure_package("psycopg2", "psycopg2-binary")

# Import db_builder_v2 components
_db_builder_v2_available = False
try:
    from db_builder_v2 import (
        DatabaseBuilder,
        DatabaseOrchestrator,
        DatabaseValidator,
        QueryBuilder,
        BatchOperations,
        MigrationManager,
        PerformanceOptimizer,
        BackupManager,
        RandomOrgQRNG,
        ANUQuantumRNG,
        HybridQuantumEntropyEngine
    )
    _db_builder_v2_available = True
    logger.info("[DB_CONFIG] ✓ db_builder_v2 components imported successfully")
except ImportError as e:
    logger.warning(f"[DB_CONFIG] ⚠ db_builder_v2 not available: {e}")
    DatabaseBuilder = None
    DatabaseOrchestrator = None

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
    SUPABASE_PROJECT_ID = os.getenv('SUPABASE_PROJECT_ID', '6c312f3f-20ea-47cb-8c85-1dc8b5377eb3')
    
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
# DATABASE BUILDER V2 INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════════════

class DatabaseBuilderManager:
    """Manages db_builder_v2 integration - singleton wrapper"""
    
    _instance = None
    _builder = None
    _orchestrator = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.initialized = False
        logger.info("[DB_BUILDER_MGR] Initialized")
    
    @classmethod
    def get_builder(cls) -> Optional[DatabaseBuilder]:
        """Get singleton DatabaseBuilder instance"""
        if not _db_builder_v2_available or DatabaseBuilder is None:
            logger.warning("[DB_BUILDER_MGR] DatabaseBuilder not available")
            return None
        
        if cls._builder is None:
            try:
                with cls._lock:
                    if cls._builder is None:
                        cls._builder = DatabaseBuilder(
                            host=Config.SUPABASE_HOST,
                            user=Config.SUPABASE_USER,
                            password=Config.SUPABASE_PASSWORD,
                            port=Config.SUPABASE_PORT,
                            database=Config.SUPABASE_DB
                        )
                        logger.info("[DB_BUILDER_MGR] ✓ DatabaseBuilder created")
            except Exception as e:
                logger.error(f"[DB_BUILDER_MGR] Failed to create DatabaseBuilder: {e}")
                return None
        return cls._builder
    
    @classmethod
    def get_orchestrator(cls) -> Optional[DatabaseOrchestrator]:
        """Get singleton DatabaseOrchestrator instance"""
        if not _db_builder_v2_available or DatabaseOrchestrator is None:
            logger.warning("[DB_BUILDER_MGR] DatabaseOrchestrator not available")
            return None
        
        if cls._orchestrator is None:
            try:
                with cls._lock:
                    if cls._orchestrator is None:
                        cls._orchestrator = DatabaseOrchestrator()
                        logger.info("[DB_BUILDER_MGR] ✓ DatabaseOrchestrator created")
            except Exception as e:
                logger.error(f"[DB_BUILDER_MGR] Failed to create DatabaseOrchestrator: {e}")
                return None
        return cls._orchestrator
    
    @classmethod
    def full_initialization(cls, populate_pq: bool = False) -> bool:
        """Full database initialization via db_builder"""
        builder = cls.get_builder()
        if not builder:
            logger.error("[DB_BUILDER_MGR] Cannot initialize - builder not available")
            return False
        try:
            result = builder.full_initialization(populate_pq=populate_pq)
            logger.info(f"[DB_BUILDER_MGR] Full initialization {'✓ succeeded' if result else '✗ failed'}")
            return result
        except Exception as e:
            logger.error(f"[DB_BUILDER_MGR] Full initialization error: {e}")
            return False
    
    @classmethod
    def verify_schema(cls) -> bool:
        """Verify database schema exists"""
        builder = cls.get_builder()
        if not builder:
            logger.warning("[DB_BUILDER_MGR] Cannot verify - builder not available")
            return False
        try:
            result = builder.verify_schema()
            logger.info(f"[DB_BUILDER_MGR] Schema verification {'✓ passed' if result else '✗ failed'}")
            return result
        except Exception as e:
            logger.error(f"[DB_BUILDER_MGR] Schema verification error: {e}")
            return False
    
    @classmethod
    def create_schema(cls, drop_existing: bool = False) -> bool:
        """Create database schema"""
        builder = cls.get_builder()
        if not builder:
            return False
        try:
            builder.create_schema(drop_existing=drop_existing)
            logger.info("[DB_BUILDER_MGR] ✓ Schema created")
            return True
        except Exception as e:
            logger.error(f"[DB_BUILDER_MGR] Schema creation error: {e}")
            return False
    
    @classmethod
    def create_indexes(cls) -> bool:
        """Create database indexes"""
        builder = cls.get_builder()
        if not builder:
            return False
        try:
            builder.create_indexes()
            logger.info("[DB_BUILDER_MGR] ✓ Indexes created")
            return True
        except Exception as e:
            logger.error(f"[DB_BUILDER_MGR] Index creation error: {e}")
            return False
    
    @classmethod
    def apply_constraints(cls) -> bool:
        """Apply database constraints"""
        builder = cls.get_builder()
        if not builder:
            return False
        try:
            builder.apply_constraints()
            logger.info("[DB_BUILDER_MGR] ✓ Constraints applied")
            return True
        except Exception as e:
            logger.error(f"[DB_BUILDER_MGR] Constraint application error: {e}")
            return False
    
    @classmethod
    def initialize_genesis_data(cls) -> bool:
        """Initialize genesis block and initial users"""
        builder = cls.get_builder()
        if not builder:
            return False
        try:
            builder.initialize_genesis_data()
            logger.info("[DB_BUILDER_MGR] ✓ Genesis data initialized")
            return True
        except Exception as e:
            logger.error(f"[DB_BUILDER_MGR] Genesis initialization error: {e}")
            return False
    
    @classmethod
    def get_health_check(cls) -> Dict[str, Any]:
        """Get database health status"""
        builder = cls.get_builder()
        if not builder:
            return {'status': 'unavailable', 'error': 'Builder not available'}
        try:
            return builder.health_check()
        except Exception as e:
            logger.error(f"[DB_BUILDER_MGR] Health check error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    @classmethod
    def get_statistics(cls) -> Dict[str, Any]:
        """Get database statistics"""
        builder = cls.get_builder()
        if not builder:
            return {'error': 'Builder not available'}
        try:
            return builder.get_statistics()
        except Exception as e:
            logger.error(f"[DB_BUILDER_MGR] Statistics error: {e}")
            return {'error': str(e)}
    
    @classmethod
    def execute_query(cls, query: str, params: tuple = None) -> List[Dict]:
        """Execute query via builder"""
        builder = cls.get_builder()
        if not builder:
            return []
        try:
            return builder.execute(query, params, return_results=True)
        except Exception as e:
            logger.error(f"[DB_BUILDER_MGR] Query error: {e}")
            return []
    
    @classmethod
    def execute_update(cls, query: str, params: tuple = None) -> int:
        """Execute update via builder"""
        builder = cls.get_builder()
        if not builder:
            return 0
        try:
            return builder.execute(query, params, return_results=False)
        except Exception as e:
            logger.error(f"[DB_BUILDER_MGR] Update error: {e}")
            return 0

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
        app.config['DB_BUILDER'] = DatabaseBuilderManager
        
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
        
        if _db_builder_v2_available:
            logger.info("✓ db_builder_v2 available")
            builder_mgr = DatabaseBuilderManager()
            stats = builder_mgr.get_statistics()
            logger.info(f"✓ Builder manager initialized: {stats}")
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        sys.exit(1)