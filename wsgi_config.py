#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════
QTCL WSGI + DATABASE CONFIGURATION - UNIFIED PRODUCTION DEPLOYMENT
Absorbed db_config.py (709 lines) completely into wsgi_config.py
SINGLE FILE - All database + WSGI functionality globally accessible
═══════════════════════════════════════════════════════════════════════════════════════
"""

import sys
import os
import logging
import traceback
import threading
import time
import fcntl
import psycopg2
import subprocess
from datetime import datetime, timezone
from psycopg2.extras import RealDictCursor
from collections import deque
from typing import Optional, Dict, List, Any, Tuple

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
logger.info("QTCL UNIFIED WSGI + DATABASE - PRODUCTION INITIALIZATION")
logger.info(f"Project Root: {PROJECT_ROOT}")
logger.info(f"Python Version: {sys.version}")
logger.info(f"Timestamp: {datetime.now().isoformat()}")
logger.info("=" * 100)

# ═══════════════════════════════════════════════════════════════════════════════════════
# SECTION 1: DEPENDENCIES (FROM db_config.py)
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
    logger.info("[DB] ✓ db_builder_v2 components imported successfully")
except ImportError as e:
    logger.warning(f"[DB] ⚠ db_builder_v2 not available: {e}")
    DatabaseBuilder = None
    DatabaseOrchestrator = None

# ═══════════════════════════════════════════════════════════════════════════════════════
# SECTION 2: DATABASE CONFIG (VERBATIM FROM db_config.py)
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

class DatabaseConnection:
    """Singleton connection pool with retry logic (VERBATIM FROM db_config.py)"""
    
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

class DatabaseBuilderManager:
    """Manages db_builder_v2 integration - singleton wrapper (VERBATIM FROM db_config.py)"""
    
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

class TransactionManager:
    """Transaction management and analytics (VERBATIM FROM db_config.py)"""
    
    @staticmethod
    def create_transaction(from_user_id: str, to_user_id: str, amount: float, tx_type: str = 'transfer', metadata: Dict = None) -> Optional[str]:
        """Create new transaction"""
        try:
            import uuid
            tx_id = str(uuid.uuid4())
            timestamp = datetime.now(timezone.utc).isoformat()
            
            query = """
                INSERT INTO transactions (tx_id, from_user_id, to_user_id, amount, tx_type, status, created_at, updated_at, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            conn = DatabaseConnection.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(query, (tx_id, from_user_id, to_user_id, amount, tx_type, 'pending', timestamp, timestamp, psycopg2.extras.Json(metadata or {})))
                conn.commit()
                logger.info(f"[TX_MGR] ✓ Created transaction {tx_id}")
                return tx_id
            finally:
                DatabaseConnection.return_connection(conn)
        except Exception as e:
            logger.error(f"[TX_MGR] Create transaction error: {e}")
            return None
    
    @staticmethod
    def get_transaction_details(tx_id: str) -> Optional[Dict]:
        """Get transaction details"""
        try:
            query = """
                SELECT tx_id, from_user_id as sender_id, to_user_id as receiver_id, amount, 
                       tx_type as type, status, created_at, updated_at, metadata
                FROM transactions 
                WHERE tx_id = %s
            """
            result = DatabaseConnection.execute_one(query, (tx_id,))
            logger.debug(f"[TX_MGR] Retrieved transaction {tx_id}")
            return result
        except Exception as e:
            logger.error(f"[TX_MGR] Get transaction details error: {e}")
            return None
    
    @staticmethod
    def update_transaction_status(tx_id: str, status: str, admin_id: str = None, notes: str = None) -> bool:
        """Update transaction status"""
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            
            query = """
                UPDATE transactions 
                SET status = %s, updated_at = %s, error_message = %s
                WHERE tx_id = %s
            """
            
            conn = DatabaseConnection.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(query, (status, timestamp, notes, tx_id))
                conn.commit()
                logger.info(f"[TX_MGR] ✓ Transaction {tx_id} status updated to {status}")
                return True
            finally:
                DatabaseConnection.return_connection(conn)
        except Exception as e:
            logger.error(f"[TX_MGR] Update status error: {e}")
            return False
    
    @staticmethod
    def cancel_transaction(tx_id: str, reason: str = None) -> bool:
        """Cancel a pending transaction"""
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            
            tx = TransactionManager.get_transaction_details(tx_id)
            if not tx or tx.get('status') != 'pending':
                logger.warning(f"[TX_MGR] Cannot cancel non-pending transaction {tx_id}")
                return False
            
            query = """
                UPDATE transactions 
                SET status = %s, updated_at = %s, revert_reason = %s
                WHERE tx_id = %s AND status = %s
            """
            
            conn = DatabaseConnection.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(query, ('cancelled', timestamp, reason, tx_id, 'pending'))
                conn.commit()
                logger.info(f"[TX_MGR] ✓ Transaction {tx_id} cancelled")
                return True
            finally:
                DatabaseConnection.return_connection(conn)
        except Exception as e:
            logger.error(f"[TX_MGR] Cancel transaction error: {e}")
            return False
    
    @staticmethod
    def get_all_transactions(status: str = None, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get all transactions"""
        try:
            if status:
                query = """
                    SELECT tx_id, from_user_id as sender_id, to_user_id as receiver_id, amount, tx_type as type, 
                           status, created_at, updated_at, metadata
                    FROM transactions 
                    WHERE status = %s
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                """
                results = DatabaseConnection.execute(query, (status, limit, offset))
            else:
                query = """
                    SELECT tx_id, from_user_id as sender_id, to_user_id as receiver_id, amount, tx_type as type, 
                           status, created_at, updated_at, metadata
                    FROM transactions 
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                """
                results = DatabaseConnection.execute(query, (limit, offset))
            
            logger.debug(f"[TX_MGR] Retrieved {len(results)} transactions")
            return results or []
        except Exception as e:
            logger.error(f"[TX_MGR] Get all transactions error: {e}")
            return []
    
    @staticmethod
    def get_transaction_statistics() -> Dict:
        """Get transaction statistics"""
        try:
            stats_query = """
                SELECT 
                    COUNT(*) as total_transactions,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                    SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) as cancelled,
                    SUM(CASE WHEN status = 'completed' THEN amount ELSE 0 END) as total_amount
                FROM transactions
            """
            
            result = DatabaseConnection.execute_one(stats_query, ())
            logger.debug("[TX_MGR] Retrieved transaction statistics")
            return dict(result) if result else {}
        except Exception as e:
            logger.error(f"[TX_MGR] Get statistics error: {e}")
            return {}
    
    @staticmethod
    def bulk_approve_transactions(tx_ids: List[str], admin_id: str, notes: str = None) -> int:
        """Bulk approve multiple transactions"""
        try:
            count = 0
            for tx_id in tx_ids:
                if TransactionManager.update_transaction_status(tx_id, 'completed', admin_id, notes):
                    count += 1
            logger.info(f"[TX_MGR] Bulk approved {count}/{len(tx_ids)} transactions")
            return count
        except Exception as e:
            logger.error(f"[TX_MGR] Bulk approve error: {e}")
            return 0
    
    @staticmethod
    def bulk_reject_transactions(tx_ids: List[str], admin_id: str, reason: str = None) -> int:
        """Bulk reject multiple transactions"""
        try:
            count = 0
            for tx_id in tx_ids:
                if TransactionManager.update_transaction_status(tx_id, 'failed', admin_id, reason):
                    count += 1
            logger.info(f"[TX_MGR] Bulk rejected {count}/{len(tx_ids)} transactions")
            return count
        except Exception as e:
            logger.error(f"[TX_MGR] Bulk reject error: {e}")
            return 0

# ═══════════════════════════════════════════════════════════════════════════════════════
# SECTION 3: GLOBAL DATABASE ACCESSORS (MAKE EVERYTHING GLOBALLY CALLABLE)
# ═══════════════════════════════════════════════════════════════════════════════════════

_global_db_conn = None
_global_db_builder = None
_global_tx_mgr = None

def get_db_config() -> Config:
    """Global accessor: Get Config"""
    return Config

def get_db() -> DatabaseConnection:
    """Global accessor: Get database connection singleton"""
    global _global_db_conn
    if _global_db_conn is None:
        _global_db_conn = DatabaseConnection()
    return _global_db_conn

def get_db_builder() -> DatabaseBuilderManager:
    """Global accessor: Get database builder"""
    global _global_db_builder
    if _global_db_builder is None:
        _global_db_builder = DatabaseBuilderManager()
    return _global_db_builder

def get_tx_manager() -> TransactionManager:
    """Global accessor: Get transaction manager"""
    global _global_tx_mgr
    if _global_tx_mgr is None:
        _global_tx_mgr = TransactionManager()
    return _global_tx_mgr

def setup_database(app):
    """Initialize database connection in Flask app (FROM db_config.py)"""
    try:
        logger.info("[INIT] Validating Supabase configuration...")
        Config.validate()
        
        logger.info("[INIT] Testing database connection...")
        test_conn = get_db().get_connection()
        
        with test_conn.cursor() as cur:
            cur.execute("SELECT now()")
            ts = cur.fetchone()
        
        get_db().return_connection(test_conn)
        
        logger.info(f"[INIT] ✓ Database ready")
        
        app.config['DB'] = get_db()
        app.config['DB_BUILDER'] = get_db_builder()
        app.config['TX_MANAGER'] = get_tx_manager()
        
        return True
        
    except Exception as e:
        logger.critical(f"[INIT] ✗ Database initialization failed: {e}")
        raise RuntimeError(f"Database error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# SECTION 4: GLOBAL QUANTUM SYSTEM SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════════════

_QUANTUM_SYSTEM_INSTANCE = None
_QUANTUM_SYSTEM_LOCK = threading.RLock()
_QUANTUM_SYSTEM_INITIALIZED = False
_LOCK_FILE_PATH = '/tmp/quantum_system.lock'
_LOCK_FILE = None

def _acquire_lock_file(timeout: int = 30) -> bool:
    """Acquire filesystem lock"""
    global _LOCK_FILE
    start_time = time.time()
    
    while True:
        try:
            _LOCK_FILE = open(_LOCK_FILE_PATH, 'w')
            fcntl.flock(_LOCK_FILE.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            logger.debug("[QuantumSystem] Lock file acquired")
            return True
        except (IOError, OSError):
            if time.time() - start_time > timeout:
                logger.warning(f"[QuantumSystem] Lock timeout after {timeout}s")
                return False
            time.sleep(0.1)

def _release_lock_file() -> None:
    """Release filesystem lock"""
    global _LOCK_FILE
    if _LOCK_FILE:
        try:
            fcntl.flock(_LOCK_FILE.fileno(), fcntl.LOCK_UN)
            _LOCK_FILE.close()
            _LOCK_FILE = None
            logger.debug("[QuantumSystem] Lock file released")
        except Exception as e:
            logger.warning(f"[QuantumSystem] Error releasing lock: {e}")

def initialize_quantum_system() -> None:
    """Initialize SINGLE global quantum system"""
    global _QUANTUM_SYSTEM_INSTANCE, _QUANTUM_SYSTEM_INITIALIZED
    
    with _QUANTUM_SYSTEM_LOCK:
        if _QUANTUM_SYSTEM_INITIALIZED:
            return
        
        _QUANTUM_SYSTEM_INITIALIZED = True
        
        try:
            from quantum_lattice_control_live_complete import QuantumLatticeControlLiveV5
            
            if not _acquire_lock_file(timeout=30):
                logger.error("[QuantumSystem] Failed to acquire lock")
                return
            
            try:
                db_config = {
                    'host': os.getenv('SUPABASE_HOST', 'localhost'),
                    'port': int(os.getenv('SUPABASE_PORT', '5432')),
                    'database': os.getenv('SUPABASE_DB', 'postgres'),
                    'user': os.getenv('SUPABASE_USER', 'postgres'),
                    'password': os.getenv('SUPABASE_PASSWORD', 'postgres'),
                }
                app_url = os.getenv('APP_URL', 'http://localhost:5000')
                
                logger.info("[QuantumSystem] Creating SINGLE global quantum system...")
                _QUANTUM_SYSTEM_INSTANCE = QuantumLatticeControlLiveV5(
                    db_config=db_config,
                    app_url=app_url
                )
                logger.info("[QuantumSystem] ✓ Global quantum system initialized (SINGLETON)")
            
            finally:
                _release_lock_file()
        
        except Exception as e:
            logger.error(f"[QuantumSystem] Failed: {e}")
            logger.error(traceback.format_exc())

def get_quantum_system():
    """Global accessor: Get the initialized quantum system instance"""
    return _QUANTUM_SYSTEM_INSTANCE

def start_quantum_daemon():
    """Start the quantum system daemon thread"""
    global _QUANTUM_SYSTEM_INSTANCE
    
    if not _QUANTUM_SYSTEM_INSTANCE:
        logger.error("[QuantumSystem] Cannot start daemon - quantum system not initialized")
        return
    
    try:
        import threading as _threading
        _cycle_thread = _threading.Thread(
            target=_QUANTUM_SYSTEM_INSTANCE.run_continuous,
            kwargs={'duration_hours': 87600},
            daemon=True,
            name='QuantumCycleThread'
        )
        _cycle_thread.start()
        logger.info("[QuantumSystem] ✓ Cycle daemon thread started")
    except Exception as e:
        logger.error(f"[QuantumSystem] Failed to start daemon: {e}")
        logger.error(traceback.format_exc())

# Pre-initialize quantum system
logger.info("Pre-initializing GLOBAL quantum system (SINGLETON)...")
initialize_quantum_system()
if _QUANTUM_SYSTEM_INSTANCE:
    logger.info("✓ GLOBAL quantum system pre-initialized")
else:
    logger.warning("⚠ Quantum system not initialized")

logger.info("=" * 100)

# ═══════════════════════════════════════════════════════════════════════════════════════
# SECTION 5: ENVIRONMENT VALIDATION
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
# SECTION 6: FLASK APP INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════════════

app = None
initialization_error = None

try:
    logger.info("Importing main application...")
    
    from main_app import create_app, initialize_app
    
    logger.info("✓ Main application factory imported successfully")
    
    logger.info("Creating Flask application instance...")
    app = create_app()
    logger.info(f"✓ Flask app created with {len(list(app.url_map.iter_rules()))} routes")
    
    logger.info("Initializing application...")
    if initialize_app(app):
        logger.info("✓ Application initialized successfully")
    else:
        logger.warning("⚠ Application initialization returned False, but continuing...")
    
    logger.info("=" * 100)
    logger.info("✓ UNIFIED WSGI + DATABASE APPLICATION READY FOR DEPLOYMENT")
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
# SECTION 7: WSGI EXPORT
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

def _load_html_terminal():
    """Load index.html terminal UI"""
    try:
        html_path = os.path.join(PROJECT_ROOT, 'index.html')
        if os.path.exists(html_path):
            with open(html_path, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        logger.error(f"[HTML] Error loading index.html: {e}")
    return None

_HTML_CONTENT = _load_html_terminal()

if app is not None:
    @app.route('/', methods=['GET'])
    def serve_html_terminal():
        """Serve HTML terminal UI at root"""
        if _HTML_CONTENT:
            return _HTML_CONTENT, 200, {'Content-Type': 'text/html; charset=utf-8'}
        else:
            return """<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>QTCL Terminal</title>
<style>body{background:#0a0a0f;color:#00ff88;font-family:monospace;display:flex;align-items:center;justify-content:center;height:100vh;margin:0}.box{text-align:center;padding:30px;border:2px solid #00ff88;border-radius:8px;background:rgba(0,255,136,0.1);box-shadow:0 0 20px rgba(0,255,136,0.3)}h1{color:#00ff88;text-shadow:0 0 10px #00ff88;margin:0}p{color:#00ffff;margin:8px 0}</style>
</head>
<body><div class="box"><h1>⚛️ QTCL Terminal</h1><p>Quantum Blockchain Interface</p><hr style="border-color:#00ff88;margin:15px 0"><p>⚠️ Terminal UI not available</p><p style="font-size:11px;opacity:0.7">Ensure index.html is deployed</p></div></body>
</html>""", 200, {'Content-Type': 'text/html; charset=utf-8'}
    
    logger.info("[HTML] ✓ HTML terminal route registered at /")

application = app

logger.info("")
logger.info("To run with Gunicorn:")
logger.info("  gunicorn -w 4 -b 0.0.0.0:5000 wsgi_config:application")
logger.info("")
logger.info("To run with uWSGI:")
logger.info("  uwsgi --http :5000 --wsgi-file wsgi_config.py --callable application --processes 4 --threads 2")
logger.info("")

if __name__ == '__main__':
    logger.warning("WARNING: Running WSGI app directly (use Gunicorn for production)")
    app.run(
        host=os.getenv('API_HOST', '0.0.0.0'),
        port=int(os.getenv('API_PORT', '5000')),
        debug=os.getenv('FLASK_ENV') == 'development'
    )
