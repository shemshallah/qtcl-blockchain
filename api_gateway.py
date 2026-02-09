
#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
QUANTUM TEMPORAL COHERENCE LEDGER (QTCL)
COMPREHENSIVE TRANSACTION EXECUTION & QUANTUM MEASUREMENT INTEGRATION
Production-Grade, Deployment-Ready for PythonAnywhere
═══════════════════════════════════════════════════════════════════════════════

TOTAL CODE VOLUME: ~30,000+ lines across all scripts
DEPLOYMENT: PythonAnywhere
DATABASE: Supabase PostgreSQL (10 tables, 106K+ pseudoqubits)
QUANTUM: Qiskit + AER Simulator (1024 shots per tx)

COMPLETE SCRIPT 1: api_gateway.py
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import time
import hashlib
import hmac
import uuid
import logging
import asyncio
import jwt
import threading
import queue
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
import subprocess
from collections import defaultdict, deque
from functools import wraps
import traceback

# ═══════════════════════════════════════════════════════════════════════════════
# DEPENDENCY INSTALLATION
# ═══════════════════════════════════════════════════════════════════════════════

def ensure_packages():
    """Install required packages if missing"""
    packages = {
        'flask': 'Flask',
        'flask_cors': 'Flask-CORS',
        'psycopg2': 'psycopg2-binary',
        #'numpy': 'numpy',
        # 'pyjwt': 'PyJWT',
        'supabase': 'supabase',
        'qiskit': 'qiskit',
        'qiskit_aer': 'qiskit-aer',
        'redis': 'redis',
        'requests': 'requests'
    }
    
    for module, pip_name in packages.items():
        try:
            __import__(module)
        except ImportError:
            print(f"[INSTALL] Installing {pip_name}...")
            try:
                subprocess.check_call(
                    [sys.executable, '-m', 'pip', 'install', '-q', pip_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print(f"[INSTALL] ✓ {pip_name} installed")
            except Exception as e:
                print(f"[INSTALL] ✗ Failed to install {pip_name}: {e}")

ensure_packages()

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
"""Production configuration - reads from environment variables"""
# ─────────────────────────────────────────────────────────────────────────
# SUPABASE CONNECTION - FROM ENVIRONMENT VARIABLES
# ─────────────────────────────────────────────────────────────────────────
SUPABASE_HOST = os.getenv('SUPABASE_HOST', 'aws-0-us-west-2.pooler.supabase.com')
SUPABASE_USER = os.getenv('SUPABASE_USER', 'postgres.rslvlsqwkfmdtebqsvtw')
SUPABASE_PASSWORD = os.getenv('SUPABASE_PASSWORD', '')  # NO DEFAULT - FAIL IF NOT SET
SUPABASE_PORT = int(os.getenv('SUPABASE_PORT', '5432'))
SUPABASE_DB = os.getenv('SUPABASE_DB', 'postgres')
SUPABASE_JWT_SECRET = os.getenv('SUPABASE_JWT_SECRET', 'dev-secret-key-change-in-production')

@staticmethod
def validate():
    """Check required environment variables are set"""
    if not Config.SUPABASE_PASSWORD:
        raise RuntimeError("SUPABASE_PASSWORD environment variable not set")
    if not Config.SUPABASE_USER:
        raise RuntimeError("SUPABASE_USER environment variable not set")
    if not Config.SUPABASE_HOST:
        raise RuntimeError("SUPABASE_HOST environment variable not set")
    logger.info(f"✓ Supabase config validated: {Config.SUPABASE_HOST}")

    # ─────────────────────────────────────────────────────────────────────────
    # API CONFIGURATION
    # ─────────────────────────────────────────────────────────────────────────
    API_PORT = 5000
    API_HOST = "0.0.0.0"
    DEBUG = False
    TESTING = False
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max request
    JSONIFY_PRETTYPRINT_REGULAR = True
    
    # ─────────────────────────────────────────────────────────────────────────
    # TOKEN ECONOMICS (1 billion QTCL total)
    # ─────────────────────────────────────────────────────────────────────────
    TOKEN_TOTAL_SUPPLY = 1_000_000_000  # 1B QTCL
    TOKEN_DECIMALS = 18
    TOKEN_WEI_PER_UNIT = 10 ** TOKEN_DECIMALS
    
    # ─────────────────────────────────────────────────────────────────────────
    # QUANTUM CONFIGURATION
    # ─────────────────────────────────────────────────────────────────────────
    QISKIT_SHOTS = 1024
    QISKIT_QUBITS = 8
    MAX_CIRCUIT_EXECUTION_TIME_MS = 200
    SUPERPOSITION_TIMEOUT_SECONDS = 300  # 5 minutes
    COHERENCE_REFRESH_INTERVAL_SECONDS = 5
    ENTROPY_THRESHOLD_FOR_SUPERPOSITION = 0.8  # Must have 80%+ entropy
    
    # ─────────────────────────────────────────────────────────────────────────
    # RATE LIMITING
    # ─────────────────────────────────────────────────────────────────────────
    MAX_TXS_PER_SECOND = 1000
    MAX_TXS_PER_USER_PER_MINUTE = 60
    MAX_TXS_PER_USER_PER_HOUR = 500
    RATE_LIMIT_WINDOW_SECONDS = 60
    
    # ─────────────────────────────────────────────────────────────────────────
    # BLOCKCHAIN CONFIGURATION
    # ─────────────────────────────────────────────────────────────────────────
    INITIAL_GAS_PRICE = 1  # Wei per gas unit
    GAS_PER_TRANSFER = 21000
    GAS_PER_CONTRACT_CALL = 100000
    GAS_PER_STAKE = 50000
    BLOCK_GAS_LIMIT = 8_000_000
    
    # Block reward halving schedule
    BLOCK_REWARD_EPOCH_1_BLOCKS = 52_560  # ~1 year at 10s blocks
    BLOCK_REWARD_EPOCH_1 = 100  # QTCL per block
    BLOCK_REWARD_EPOCH_2 = 50   # QTCL per block (after 52,560)
    BLOCK_REWARD_EPOCH_3 = 25   # QTCL per block (after 105,120)
    
    # ─────────────────────────────────────────────────────────────────────────
    # HYPERBOLIC GEOMETRY
    # ─────────────────────────────────────────────────────────────────────────
    TESSELLATION_TYPE = (8, 3)  # {8,3} tessellation
    MAX_TESSELLATION_DEPTH = 10
    TOTAL_PSEUDOQUBITS_ESTIMATED = 106_496  # ~52 per triangle × ~2048 triangles
    
    # ─────────────────────────────────────────────────────────────────────────
    # ROUTING PARAMETERS
    # ─────────────────────────────────────────────────────────────────────────
    MAX_EUCLIDEAN_DISTANCE_FOR_ROUTE = 0.3
    BANDWIDTH_SCORE_CALCULATION = "1.0 - (distance / max_distance)"
    
    # ─────────────────────────────────────────────────────────────────────────
    # CACHE & PERFORMANCE
    # ─────────────────────────────────────────────────────────────────────────
    USER_CACHE_TTL_SECONDS = 300
    QUBIT_CACHE_TTL_SECONDS = 300
    BLOCK_CACHE_TTL_SECONDS = 600
    ENABLE_QUERY_CACHING = True
    
    # ─────────────────────────────────────────────────────────────────────────
    # SECURITY
    # ─────────────────────────────────────────────────────────────────────────
    JWT_ALGORITHM = 'HS256'
    JWT_EXPIRATION_HOURS = 24
    ALLOWED_ORIGINS = ['*']  # Change in production
    ENABLE_REQUEST_LOGGING = True
    
    # ─────────────────────────────────────────────────────────────────────────
    # CONNECTION POOL
    # ─────────────────────────────────────────────────────────────────────────
    DB_POOL_SIZE = 5
    DB_POOL_TIMEOUT = 30
    DB_POOL_RECYCLE_SECONDS = 3600

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class LoggerSetup:
    """Configure comprehensive logging"""
    
    @staticmethod
    def setup():
        """Initialize logging to file and console"""
        log_format = '[%(asctime)s] %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        
        # File handler
        file_handler = logging.FileHandler('qtcl_api_gateway.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        
        # Root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        return logging.getLogger(__name__)

logger = LoggerSetup.setup()

# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE CONNECTION MANAGEMENT WITH POOLING
# ═══════════════════════════════════════════════════════════════════════════════

class DatabaseConnection:
    """
    Singleton connection management with connection pooling
    Handles all database operations for api_gateway
    """
    
    _instance = None
    _connections = deque(maxlen=Config.DB_POOL_SIZE)
    _lock = threading.Lock()
    _last_cleanup = time.time()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        logger.info("Initializing DatabaseConnection singleton")
    
    @staticmethod
def _get_new_connection():
    \"\"\"Create a new database connection\"\"\"
    # Validate config
    for attempt in range(1, 4):  # Try 3 times
        try:
            logger.info(f"Database connection attempt {attempt}/3...")
            conn = psycopg2.connect(
                host=Config.SUPABASE_HOST,
                user=Config.SUPABASE_USER,
                password=Config.SUPABASE_PASSWORD,
                port=Config.SUPABASE_PORT,
                database=Config.SUPABASE_DB,
                connect_timeout=15,  # Increase from 10 to 15
                application_name='qtcl_api_gateway'
            )
            conn.set_session(autocommit=True)
            logger.info("✓ Database connection established")
            return conn
        except psycopg2.Error as e:
            logger.error(f"Connection failed: {e}")
            if attempt < 3:
                wait = 2 * attempt
                logger.info(f"Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    
    try:
        conn = psycopg2.connect(
                host=Config.SUPABASE_HOST,
                user=Config.SUPABASE_USER,
                password=Config.SUPABASE_PASSWORD,
                port=Config.SUPABASE_PORT,
                database=Config.SUPABASE_DB,
                connect_timeout=10,
                application_name='qtcl_api_gateway'
            )
            conn.set_session(autocommit=True)
            logger.debug("New database connection created")
            return conn
        except psycopg2.Error as e:
            logger.error(f"Failed to create database connection: {e}")
            raise
    
    @staticmethod
    def get_connection():
        """Get connection from pool or create new one"""
        with DatabaseConnection._lock:
            # Try to get from pool
            try:
                conn = DatabaseConnection._connections.popleft()
                if conn and not conn.closed:
                    # Test connection
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                    logger.debug("Reused pooled connection")
                    return conn
            except IndexError:
                pass
            
            # Create new connection
            return DatabaseConnection._get_new_connection()
    
    @staticmethod
    def return_connection(conn):
        """Return connection to pool"""
        if conn and not conn.closed:
            with DatabaseConnection._lock:
                try:
                    DatabaseConnection._connections.append(conn)
                    logger.debug("Connection returned to pool")
                except Exception as e:
                    logger.warning(f"Failed to return connection to pool: {e}")
                    conn.close()
        else:
            logger.warning("Attempted to return closed connection")
    
    @staticmethod
    def execute(query: str, params: tuple = None) -> List[Dict]:
        """Execute SELECT query and return results"""
        conn = DatabaseConnection.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params or ())
                results = cur.fetchall()
                logger.debug(f"SELECT query returned {len(results)} rows")
                return results
        except psycopg2.Error as e:
            logger.error(f"Query execution error: {e}\nQuery: {query}\nParams: {params}")
            raise
        finally:
            DatabaseConnection.return_connection(conn)
    
    @staticmethod
    def execute_one(query: str, params: tuple = None) -> Optional[Dict]:
        """Execute SELECT query and return single result"""
        results = DatabaseConnection.execute(query, params)
        return results[0] if results else None
    
    @staticmethod
    def execute_insert(query: str, params: tuple = None) -> int:
        """Execute INSERT/UPDATE/DELETE and return affected rows"""
        conn = DatabaseConnection.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(query, params or ())
                affected = cur.rowcount
                logger.debug(f"Execute insert affected {affected} rows")
                return affected
        except psycopg2.Error as e:
            logger.error(f"Insert execution error: {e}\nQuery: {query}\nParams: {params}")
            raise
        finally:
            DatabaseConnection.return_connection(conn)
    
    @staticmethod
    def execute_batch(query: str, params_list: List[tuple]) -> int:
        """Execute batch INSERT/UPDATE/DELETE"""
        conn = DatabaseConnection.get_connection()
        try:
            with conn.cursor() as cur:
                execute_batch(cur, query, params_list, page_size=1000)
                affected = cur.rowcount
                logger.debug(f"Batch execute affected {affected} rows")
                return affected
        except psycopg2.Error as e:
            logger.error(f"Batch execution error: {e}")
            raise
        finally:
            DatabaseConnection.return_connection(conn)
    
    @staticmethod
    def transaction(func):
        """Decorator for transaction handling"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            conn = DatabaseConnection.get_connection()
            try:
                conn.set_session(autocommit=False)
                result = func(conn, *args, **kwargs)
                conn.commit()
                logger.debug(f"Transaction {func.__name__} committed")
                return result
            except Exception as e:
                conn.rollback()
                logger.error(f"Transaction {func.__name__} rolled back: {e}")
                raise
            finally:
                conn.set_session(autocommit=True)
                DatabaseConnection.return_connection(conn)
        return wrapper
    
    @staticmethod
    def close_all():
        """Close all pooled connections"""
        with DatabaseConnection._lock:
            while len(DatabaseConnection._connections) > 0:
                try:
                    conn = DatabaseConnection._connections.popleft()
                    if conn and not conn.closed:
                        conn.close()
                except Exception as e:
                    logger.warning(f"Error closing connection: {e}")
        logger.info("All database connections closed")

# ═══════════════════════════════════════════════════════════════════════════════
# CACHING LAYER
# ═══════════════════════════════════════════════════════════════════════════════

class CacheLayer:
    """In-memory cache with TTL for performance optimization"""
    
    def __init__(self):
        self.cache = {}
        self.timestamps = {}
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        with self.lock:
            if key in self.cache:
                created_time = self.timestamps[key]
                if time.time() - created_time < Config.USER_CACHE_TTL_SECONDS:
                    logger.debug(f"Cache HIT: {key}")
                    return self.cache[key]
                else:
                    # Expired
                    del self.cache[key]
                    del self.timestamps[key]
                    logger.debug(f"Cache EXPIRED: {key}")
            logger.debug(f"Cache MISS: {key}")
        return None
    
    def set(self, key: str, value: Any):
        """Set value in cache with TTL"""
        with self.lock:
            self.cache[key] = value
            self.timestamps[key] = time.time()
            logger.debug(f"Cache SET: {key}")
    
    def delete(self, key: str):
        """Delete value from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                del self.timestamps[key]
                logger.debug(f"Cache DELETE: {key}")
    
    def clear(self):
        """Clear all cache"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            logger.info("Cache cleared")

cache = CacheLayer()

# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS / DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class User:
    """User model from 'users' table"""
    user_id: str
    email: str
    name: str
    role: str  # 'admin', 'founder', 'user'
    balance: int  # Stored as integer (Wei)
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    kyc_verified: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict"""
        return {
            'user_id': self.user_id,
            'email': self.email,
            'name': self.name,
            'role': self.role,
            'balance': str(self.balance),
            'balance_display': f"{self.balance / Config.TOKEN_WEI_PER_UNIT:.18f}",
            'is_active': self.is_active,
            'kyc_verified': self.kyc_verified,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }
    
    def has_sufficient_balance(self, amount: int) -> bool:
        """Check if user has sufficient balance"""
        return self.balance >= amount

@dataclass
class Pseudoqubit:
    """Pseudoqubit model from 'pseudoqubits' table"""
    pseudoqubit_id: int
    triangle_id: int
    placement_type: str  # vertex, edge, incenter, circumcenter, orthocenter, geodesic_grid
    depth: int
    position_poincare_real: float
    position_poincare_imag: float
    position_klein_real: float
    position_klein_imag: float
    hyperboloid_x: float
    hyperboloid_y: float
    hyperboloid_t: float
    boundary_distance: float
    fidelity: float
    coherence: float
    purity: float
    entropy: float
    concurrence: float
    curvature_local: float
    geodesic_density: float
    routing_address: str
    is_available: bool
    auth_user_id: Optional[str] = None
    assigned_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict"""
        return {
            'pseudoqubit_id': self.pseudoqubit_id,
            'triangle_id': self.triangle_id,
            'placement_type': self.placement_type,
            'depth': self.depth,
            'routing_address': self.routing_address,
            'position': {
                'poincare': {
                    'real': self.position_poincare_real,
                    'imag': self.position_poincare_imag
                },
                'klein': {
                    'real': self.position_klein_real,
                    'imag': self.position_klein_imag
                },
                'hyperboloid': {
                    'x': self.hyperboloid_x,
                    'y': self.hyperboloid_y,
                    't': self.hyperboloid_t
                }
            },
            'quantum_metrics': {
                'fidelity': self.fidelity,
                'coherence': self.coherence,
                'purity': self.purity,
                'entropy': self.entropy,
                'concurrence': self.concurrence
            },
            'geometric_properties': {
                'boundary_distance': self.boundary_distance,
                'curvature_local': self.curvature_local,
                'geodesic_density': self.geodesic_density
            },
            'is_available': self.is_available,
            'assigned_user': self.auth_user_id,
            'assigned_at': self.assigned_at.isoformat() if self.assigned_at else None
        }

@dataclass
class Transaction:
    """Transaction model from 'transactions' table"""
    tx_id: str
    from_user_id: str
    to_user_id: str
    amount: int
    tx_type: str  # 'transfer', 'mint', 'burn', 'stake'
    status: str  # 'pending', 'confirmed', 'failed'
    block_number: Optional[int] = None
    gas_used: int = 0
    gas_price: int = 1
    created_at: Optional[datetime] = None
    confirmed_at: Optional[datetime] = None
    quantum_state_hash: Optional[str] = None
    entropy_score: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict"""
        return {
            'tx_id': self.tx_id,
            'from_user_id': self.from_user_id,
            'to_user_id': self.to_user_id,
            'amount': str(self.amount),
            'amount_display': f"{self.amount / Config.TOKEN_WEI_PER_UNIT:.18f}",
            'tx_type': self.tx_type,
            'status': self.status,
            'block_number': self.block_number,
            'gas_used': self.gas_used,
            'gas_price': self.gas_price,
            'total_fee': str(self.gas_used * self.gas_price),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'confirmed_at': self.confirmed_at.isoformat() if self.confirmed_at else None,
            'quantum_state_hash': self.quantum_state_hash,
            'entropy_score': self.entropy_score
        }

@dataclass
class Block:
    """Block model from 'blocks' table"""
    block_number: int
    block_hash: str
    parent_hash: str
    timestamp: int
    transactions: int = 0
    validator_address: Optional[str] = None
    quantum_state_hash: Optional[str] = None
    entropy_score: Optional[float] = None
    floquet_cycle: int = 0
    merkle_root: Optional[str] = None
    difficulty: int = 0
    gas_used: int = 0
    gas_limit: int = Config.BLOCK_GAS_LIMIT
    miner_reward: int = 0
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict"""
        return {
            'block_number': self.block_number,
            'block_hash': self.block_hash,
            'parent_hash': self.parent_hash,
            'timestamp': self.timestamp,
            'transactions': self.transactions,
            'validator_address': self.validator_address,
            'quantum_state_hash': self.quantum_state_hash,
            'entropy_score': self.entropy_score,
            'floquet_cycle': self.floquet_cycle,
            'merkle_root': self.merkle_root,
            'difficulty': str(self.difficulty),
            'gas_used': self.gas_used,
            'gas_limit': self.gas_limit,
            'miner_reward': str(self.miner_reward),
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

# ═══════════════════════════════════════════════════════════════════════════════
# JWT AUTHENTICATION & SECURITY
# ═══════════════════════════════════════════════════════════════════════════════

class JWTHandler:
    """JWT token creation, verification, and handling"""
    
    SECRET = Config.SUPABASE_JWT_SECRET
    ALGORITHM = Config.JWT_ALGORITHM
    
    @staticmethod
    def create_token(user_id: str, email: str, role: str, expires_in_hours: int = 24) -> str:
        """Create JWT token for authenticated user"""
        now = datetime.utcnow()
        exp = now + timedelta(hours=expires_in_hours)
        
        payload = {
            'sub': user_id,
            'email': email,
            'role': role,
            'iat': int(now.timestamp()),
            'exp': int(exp.timestamp())
        }
        
        token = jwt.encode(payload, JWTHandler.SECRET, algorithm=JWTHandler.ALGORITHM)
        logger.info(f"Created JWT token for user {user_id}")
        return token
    
    @staticmethod
    def verify_token(token: str) -> Dict:
        """Verify JWT token and return payload"""
        try:
            payload = jwt.decode(token, JWTHandler.SECRET, algorithms=[JWTHandler.ALGORITHM])
            logger.debug(f"JWT verified for user {payload.get('sub')}")
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            raise ValueError("Token expired")
        except jwt.InvalidTokenError as e:
            logger.warning(f"JWT token invalid: {e}")
            raise ValueError("Invalid token")
    
    @staticmethod
    def extract_user_id(token: str) -> str:
        """Extract user_id from JWT token"""
        payload = JWTHandler.verify_token(token)
        return payload.get('sub')
    
    @staticmethod
    def require_auth(f):
        """Decorator for JWT-protected endpoints"""
        @wraps(f)
        def decorated(*args, **kwargs):
            auth_header = request.headers.get('Authorization', '')
            
            if not auth_header.startswith('Bearer '):
                logger.warning("Missing or invalid Authorization header")
                return jsonify({'error': 'Missing or invalid Authorization header'}), 401
            
            token = auth_header.replace('Bearer ', '')
            try:
                user_id = JWTHandler.extract_user_id(token)
                request.user_id = user_id
                return f(*args, **kwargs)
            except ValueError as e:
                logger.warning(f"JWT verification failed: {e}")
                return jsonify({'error': str(e)}), 401
        
        decorated.__name__ = f.__name__
        return decorated

# ═══════════════════════════════════════════════════════════════════════════════
# USER MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

class UserManager:
    """User account management, lookup, and balance operations"""
    
    @staticmethod
    def _row_to_user(row: Dict) -> User:
        """Convert database row to User object"""
        return User(
            user_id=row['user_id'],
            email=row['email'],
            name=row['name'],
            role=row['role'],
            balance=int(row['balance']),
            created_at=row['created_at'],
            last_login=row['last_login'],
            is_active=row['is_active'],
            kyc_verified=row['kyc_verified']
        )
    
    @staticmethod
    def get_user_by_id(user_id: str) -> Optional[User]:
        """Fetch user by user_id (SHA256 hash)"""
        # Check cache first
        cache_key = f"user:{user_id}"
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        try:
            row = DatabaseConnection.execute_one(
                "SELECT * FROM users WHERE user_id = %s",
                (user_id,)
            )
            
            if row:
                user = UserManager._row_to_user(row)
                cache.set(cache_key, user)
                logger.info(f"Retrieved user {user_id}")
                return user
            else:
                logger.warning(f"User not found: {user_id}")
                return None
        except Exception as e:
            logger.error(f"Error fetching user {user_id}: {e}")
            return None
    
    @staticmethod
    def get_user_by_email(email: str) -> Optional[User]:
        """Fetch user by email address"""
        try:
            row = DatabaseConnection.execute_one(
                "SELECT * FROM users WHERE email = %s",
                (email,)
            )
            
            if row:
                user = UserManager._row_to_user(row)
                cache_key = f"user:{user.user_id}"
                cache.set(cache_key, user)
                logger.info(f"Retrieved user by email: {email}")
                return user
            else:
                logger.warning(f"User not found by email: {email}")
                return None
        except Exception as e:
            logger.error(f"Error fetching user by email {email}: {e}")
            return None
    
    @staticmethod
    def get_all_users() -> List[User]:
        """Get all users (admin only)"""
        try:
            rows = DatabaseConnection.execute(
                "SELECT * FROM users ORDER BY balance DESC"
            )
            users = [UserManager._row_to_user(row) for row in rows]
            logger.info(f"Retrieved {len(users)} users")
            return users
        except Exception as e:
            logger.error(f"Error fetching all users: {e}")
            return []
    
    @staticmethod
    def get_balance(user_id: str) -> int:
        """Get user's current balance"""
        try:
            row = DatabaseConnection.execute_one(
                "SELECT balance FROM users WHERE user_id = %s",
                (user_id,)
            )
            balance = int(row['balance']) if row else 0
            logger.debug(f"Balance for {user_id}: {balance}")
            return balance
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return 0
    
    @staticmethod
    def update_balance(user_id: str, new_balance: int) -> bool:
        """Update user's balance atomically"""
        try:
            affected = DatabaseConnection.execute_insert(
                "UPDATE users SET balance = %s WHERE user_id = %s",
                (new_balance, user_id)
            )
            
            if affected > 0:
                # Invalidate cache
                cache.delete(f"user:{user_id}")
                logger.info(f"Updated balance for {user_id}: {new_balance}")
                return True
            else:
                logger.warning(f"No user found to update: {user_id}")
                return False
        except Exception as e:
            logger.error(f"Error updating balance: {e}")
            return False
    
    @staticmethod
    def add_to_balance(user_id: str, amount: int) -> bool:
        """Add amount to user's balance"""
        try:
            affected = DatabaseConnection.execute_insert(
                "UPDATE users SET balance = balance + %s WHERE user_id = %s",
                (amount, user_id)
            )
            
            if affected > 0:
                cache.delete(f"user:{user_id}")
                logger.info(f"Added {amount} to balance of {user_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error adding to balance: {e}")
            return False
    
    @staticmethod
    def subtract_from_balance(user_id: str, amount: int) -> bool:
        """Subtract amount from user's balance"""
        try:
            # Use transaction to ensure atomic operation
            @DatabaseConnection.transaction
            def _subtract(conn):
                with conn.cursor() as cur:
                    # Check balance first
                    cur.execute("SELECT balance FROM users WHERE user_id = %s FOR UPDATE", (user_id,))
                    row = cur.fetchone()
                    
                    if not row:
                        raise ValueError("User not found")
                    
                    current_balance = int(row[0])
                    if current_balance < amount:
                        raise ValueError("Insufficient balance")
                    
                    # Update balance
                    cur.execute(
                        "UPDATE users SET balance = balance - %s WHERE user_id = %s",
                        (amount, user_id)
                    )
                    
                    return cur.rowcount > 0
            
            if _subtract():
                cache.delete(f"user:{user_id}")
                logger.info(f"Subtracted {amount} from balance of {user_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error subtracting from balance: {e}")
            return False
    
    @staticmethod
    def update_last_login(user_id: str):
        """Update last_login timestamp"""
        try:
            DatabaseConnection.execute_insert(
                "UPDATE users SET last_login = NOW() WHERE user_id = %s",
                (user_id,)
            )
            cache.delete(f"user:{user_id}")
            logger.debug(f"Updated last_login for {user_id}")
        except Exception as e:
            logger.error(f"Error updating last_login: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# PSEUDOQUBIT MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

class PseudoqubitManager:
    """Pseudoqubit allocation, assignment, and retrieval"""
    
    @staticmethod
    def _row_to_pseudoqubit(row: Dict) -> Pseudoqubit:
        """Convert database row to Pseudoqubit object"""
        return Pseudoqubit(
            pseudoqubit_id=row['pseudoqubit_id'],
            triangle_id=row['triangle_id'],
            placement_type=row['placement_type'],
            depth=row['depth'],
            position_poincare_real=float(row['position_poincare_real']),
            position_poincare_imag=float(row['position_poincare_imag']),
            position_klein_real=float(row['position_klein_real']),
            position_klein_imag=float(row['position_klein_imag']),
            hyperboloid_x=float(row['hyperboloid_x']),
            hyperboloid_y=float(row['hyperboloid_y']),
            hyperboloid_t=float(row['hyperboloid_t']),
            boundary_distance=float(row['boundary_distance']),
            fidelity=float(row['fidelity']),
            coherence=float(row['coherence']),
            purity=float(row['purity']),
            entropy=float(row['entropy']),
            concurrence=float(row['concurrence']),
            curvature_local=float(row['curvature_local']),
            geodesic_density=float(row['geodesic_density']),
            routing_address=row['routing_address'],
            is_available=row['is_available'],
            auth_user_id=row['auth_user_id'],
            assigned_at=row['assigned_at']
        )
    
    @staticmethod
    def get_user_pseudoqubit(user_id: str) -> Optional[Pseudoqubit]:
        """Get user's assigned personal pseudoqubit"""
        cache_key = f"user_qubit:{user_id}"
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        try:
            row = DatabaseConnection.execute_one(
                "SELECT * FROM pseudoqubits WHERE auth_user_id = %s LIMIT 1",
                (user_id,)
            )
            
            if row:
                qubit = PseudoqubitManager._row_to_pseudoqubit(row)
                cache.set(cache_key, qubit)
                logger.debug(f"Retrieved user pseudoqubit for {user_id}")
                return qubit
            else:
                logger.warning(f"No pseudoqubit assigned to {user_id}")
                return None
        except Exception as e:
            logger.error(f"Error fetching user pseudoqubit: {e}")
            return None
    
    @staticmethod
    def get_shared_pool_qubits(count: int = 7) -> List[Pseudoqubit]:
        """Get N shared pool pseudoqubits (unassigned, high-fidelity)"""
        try:
            rows = DatabaseConnection.execute(
                """SELECT * FROM pseudoqubits 
                   WHERE auth_user_id IS NULL 
                   AND is_available = TRUE 
                   AND placement_type IN ('vertex', 'incenter', 'circumcenter')
                   ORDER BY fidelity DESC, coherence DESC 
                   LIMIT %s""",
                (count,)
            )
            
            qubits = [PseudoqubitManager._row_to_pseudoqubit(row) for row in rows]
            logger.info(f"Retrieved {len(qubits)} shared pool qubits")
            return qubits
        except Exception as e:
            logger.error(f"Error fetching pool qubits: {e}")
            return []
    
    @staticmethod
    def get_pseudoqubit_by_id(pseudoqubit_id: int) -> Optional[Pseudoqubit]:
        """Get pseudoqubit by ID"""
        cache_key = f"qubit:{pseudoqubit_id}"
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        try:
            row = DatabaseConnection.execute_one(
                "SELECT * FROM pseudoqubits WHERE pseudoqubit_id = %s",
                (pseudoqubit_id,)
            )
            
            if row:
                qubit = PseudoqubitManager._row_to_pseudoqubit(row)
                cache.set(cache_key, qubit)
                return qubit
            return None
        except Exception as e:
            logger.error(f"Error fetching pseudoqubit {pseudoqubit_id}: {e}")
            return None
    
    @staticmethod
    def get_pseudoqubit_by_routing_address(routing_address: str) -> Optional[Pseudoqubit]:
        """Lookup pseudoqubit by quantum routing address"""
        try:
            row = DatabaseConnection.execute_one(
                "SELECT * FROM pseudoqubits WHERE routing_address = %s",
                (routing_address,)
            )
            
            if row:
                return PseudoqubitManager._row_to_pseudoqubit(row)
            return None
        except Exception as e:
            logger.error(f"Error fetching pseudoqubit by routing address: {e}")
            return None
    
    @staticmethod
    def assign_pseudoqubit_to_user(user_id: str, pseudoqubit_id: int) -> bool:
        """Assign an unassigned pseudoqubit to a user"""
        try:
            affected = DatabaseConnection.execute_insert(
                """UPDATE pseudoqubits 
                   SET auth_user_id = %s, assigned_at = NOW(), is_available = FALSE 
                   WHERE pseudoqubit_id = %s AND auth_user_id IS NULL""",
                (user_id, pseudoqubit_id)
            )
            
            if affected > 0:
                logger.info(f"Assigned pseudoqubit {pseudoqubit_id} to user {user_id}")
                return True
            else:
                logger.warning(f"Failed to assign pseudoqubit {pseudoqubit_id}")
                return False
        except Exception as e:
            logger.error(f"Error assigning pseudoqubit: {e}")
            return False
    
    @staticmethod
    def get_available_pseudoqubits_count() -> int:
        """Get count of available pseudoqubits"""
        try:
            row = DatabaseConnection.execute_one(
                "SELECT COUNT(*) as count FROM pseudoqubits WHERE is_available = TRUE AND auth_user_id IS NULL"
            )
            count = row['count'] if row else 0
            logger.debug(f"Available pseudoqubits: {count}")
            return count
        except Exception as e:
            logger.error(f"Error counting available qubits: {e}")
            return 0

# ═══════════════════════════════════════════════════════════════════════════════
# TRANSACTION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

class TransactionManager:
    """Transaction creation, validation, and status management"""
    
    @staticmethod
    def _row_to_transaction(row: Dict) -> Transaction:
        """Convert database row to Transaction object"""
        return Transaction(
            tx_id=row['tx_id'],
            from_user_id=row['from_user_id'],
            to_user_id=row['to_user_id'],
            amount=int(row['amount']),
            tx_type=row['tx_type'],
            status=row['status'],
            block_number=row['block_number'],
            gas_used=int(row['gas_used']) if row['gas_used'] else 0,
            gas_price=int(row['gas_price']) if row['gas_price'] else 1,
            created_at=row['created_at'],
            confirmed_at=row['confirmed_at'],
            quantum_state_hash=row['quantum_state_hash'],
            entropy_score=float(row['entropy_score']) if row['entropy_score'] else None
        )
    
    @staticmethod
    def create_transaction(
        from_user_id: str,
        to_user_id: str,
        amount: int,
        tx_type: str = 'transfer',
        gas_price: int = 1
    ) -> Optional[str]:
        """Create new transaction and return tx_id"""
        try:
            # Generate transaction ID
            tx_id = hashlib.sha256(
                f"{from_user_id}{to_user_id}{amount}{time.time()}".encode()
            ).hexdigest()
            
            # Insert transaction
            DatabaseConnection.execute_insert(
                """INSERT INTO transactions 
                   (tx_id, from_user_id, to_user_id, amount, tx_type, status, gas_price, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())""",
                (tx_id, from_user_id, to_user_id, amount, tx_type, 'pending', gas_price)
            )
            
            logger.info(f"Created transaction {tx_id}")
            return tx_id
        except Exception as e:
            logger.error(f"Error creating transaction: {e}")
            return None
    
    @staticmethod
    def get_transaction(tx_id: str) -> Optional[Transaction]:
        """Get transaction by ID"""
        try:
            row = DatabaseConnection.execute_one(
                "SELECT * FROM transactions WHERE tx_id = %s",
                (tx_id,)
            )
            
            if row:
                return TransactionManager._row_to_transaction(row)
            return None
        except Exception as e:
            logger.error(f"Error fetching transaction: {e}")
            return None
    
    @staticmethod
    def update_transaction_status(tx_id: str, status: str) -> bool:
        """Update transaction status"""
        try:
            affected = DatabaseConnection.execute_insert(
                "UPDATE transactions SET status = %s WHERE tx_id = %s",
                (status, tx_id)
            )
            
            if affected > 0:
                logger.info(f"Updated transaction {tx_id} status to {status}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating transaction status: {e}")
            return False
    
    @staticmethod
    def update_transaction_quantum(
        tx_id: str,
        quantum_state_hash: str,
        entropy_score: float
    ) -> bool:
        """Update transaction with quantum measurement results"""
        try:
            affected = DatabaseConnection.execute_insert(
                """UPDATE transactions 
                   SET quantum_state_hash = %s, entropy_score = %s 
                   WHERE tx_id = %s""",
                (quantum_state_hash, entropy_score, tx_id)
            )
            
            if affected > 0:
                logger.info(f"Updated transaction {tx_id} with quantum data")
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating transaction quantum data: {e}")
            return False
    
    @staticmethod
    def confirm_transaction(tx_id: str, block_number: int) -> bool:
        """Confirm transaction and assign to block"""
        try:
            affected = DatabaseConnection.execute_insert(
                """UPDATE transactions 
                   SET status = %s, block_number = %s, confirmed_at = NOW() 
                   WHERE tx_id = %s""",
                ('confirmed', block_number, tx_id)
            )
            
            if affected > 0:
                logger.info(f"Confirmed transaction {tx_id} in block {block_number}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error confirming transaction: {e}")
            return False
    
    @staticmethod
    def get_user_transactions(user_id: str, limit: int = 100) -> List[Transaction]:
        """Get transactions for a user"""
        try:
            rows = DatabaseConnection.execute(
                """SELECT * FROM transactions 
                   WHERE from_user_id = %s OR to_user_id = %s 
                   ORDER BY created_at DESC 
                   LIMIT %s""",
                (user_id, user_id, limit)
            )
            
            transactions = [TransactionManager._row_to_transaction(row) for row in rows]
            logger.debug(f"Retrieved {len(transactions)} transactions for user {user_id}")
            return transactions
        except Exception as e:
            logger.error(f"Error fetching user transactions: {e}")
            return []

# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

class BlockManager:
    """Block creation, retrieval, and chain management"""
    
    @staticmethod
    def _row_to_block(row: Dict) -> Block:
        """Convert database row to Block object"""
        return Block(
            block_number=row['block_number'],
            block_hash=row['block_hash'],
            parent_hash=row['parent_hash'],
            timestamp=int(row['timestamp']),
            transactions=int(row['transactions']),
            validator_address=row['validator_address'],
            quantum_state_hash=row['quantum_state_hash'],
            entropy_score=float(row['entropy_score']) if row['entropy_score'] else None,
            floquet_cycle=int(row['floquet_cycle']) if row['floquet_cycle'] else 0,
            merkle_root=row['merkle_root'],
            difficulty=int(row['difficulty']) if row['difficulty'] else 0,
            gas_used=int(row['gas_used']) if row['gas_used'] else 0,
            gas_limit=int(row['gas_limit']) if row['gas_limit'] else Config.BLOCK_GAS_LIMIT,
            miner_reward=int(row['miner_reward']) if row['miner_reward'] else 0,
            created_at=row['created_at']
        )
    
    @staticmethod
    def get_block(block_number: int) -> Optional[Block]:
        """Get block by number"""
        cache_key = f"block:{block_number}"
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        try:
            row = DatabaseConnection.execute_one(
                "SELECT * FROM blocks WHERE block_number = %s",
                (block_number,)
            )
            
            if row:
                block = BlockManager._row_to_block(row)
                cache.set(cache_key, block)
                return block
            return None
        except Exception as e:
            logger.error(f"Error fetching block {block_number}: {e}")
            return None
    
    @staticmethod
    def get_latest_block() -> Optional[Block]:
        """Get latest block in chain"""
        try:
            row = DatabaseConnection.execute_one(
                "SELECT * FROM blocks ORDER BY block_number DESC LIMIT 1"
            )
            
            if row:
                return BlockManager._row_to_block(row)
            return None
        except Exception as e:
            logger.error(f"Error fetching latest block: {e}")
            return None
    
    @staticmethod
    def get_genesis_block() -> Optional[Block]:
        """Get genesis block (block 0)"""
        return BlockManager.get_block(0)
    
    @staticmethod
    def create_block(
        parent_hash: str,
        validator_address: str,
        merkle_root: str,
        quantum_state_hash: str,
        entropy_score: float,
        transactions_count: int = 0
    ) -> Optional[Block]:
        """Create new block"""
        try:
            # Get next block number
            latest = BlockManager.get_latest_block()
            block_number = (latest.block_number + 1) if latest else 0
            
            # Calculate block hash
            block_hash_input = f"{block_number}{parent_hash}{merkle_root}{quantum_state_hash}{time.time()}"
            block_hash = f"0x{hashlib.sha256(block_hash_input.encode()).hexdigest()}"
            
            # Calculate miner reward based on epoch
            if block_number <= Config.BLOCK_REWARD_EPOCH_1_BLOCKS:
                miner_reward = Config.BLOCK_REWARD_EPOCH_1 * Config.TOKEN_WEI_PER_UNIT
            elif block_number <= Config.BLOCK_REWARD_EPOCH_1_BLOCKS * 2:
                miner_reward = Config.BLOCK_REWARD_EPOCH_2 * Config.TOKEN_WEI_PER_UNIT
            else:
                miner_reward = Config.BLOCK_REWARD_EPOCH_3 * Config.TOKEN_WEI_PER_UNIT
            
            # Insert block
            DatabaseConnection.execute_insert(
                """INSERT INTO blocks 
                   (block_number, block_hash, parent_hash, timestamp, transactions, 
                    validator_address, quantum_state_hash, entropy_score, merkle_root, 
                    miner_reward, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())""",
                (block_number, block_hash, parent_hash, int(time.time()), 
                 transactions_count, validator_address, quantum_state_hash, 
                 entropy_score, merkle_root, miner_reward)
            )
            
            logger.info(f"Created block {block_number} with hash {block_hash}")
            return BlockManager.get_block(block_number)
        except Exception as e:
            logger.error(f"Error creating block: {e}")
            return None

# ═══════════════════════════════════════════════════════════════════════════════
# TRANSACTION VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

class TransactionValidator:
    """Comprehensive transaction validation"""
    
    @staticmethod
    def validate_transaction_input(tx_data: Dict) -> Tuple[bool, Optional[str]]:
        """
        Validate transaction input
        
        Checks:
        - Required fields present
        - Correct types
        - Non-negative amounts
        - Users exist
        - Sufficient balance
        - Valid operation type
        """
        
        # Check required fields
        required = ['from_user_id', 'to_user_id', 'amount', 'tx_type']
        for field in required:
            if field not in tx_data:
                return False, f"Missing required field: {field}"
        
        # Validate tx_type
        valid_types = ['transfer', 'mint', 'burn', 'stake']
        if tx_data.get('tx_type') not in valid_types:
            return False, f"Invalid tx_type. Must be one of: {', '.join(valid_types)}"
        
        # Validate amount
        try:
            amount = int(tx_data.get('amount', 0))
            if amount <= 0:
                return False, "Amount must be positive (> 0)"
            if amount > Config.TOKEN_TOTAL_SUPPLY * Config.TOKEN_WEI_PER_UNIT:
                return False, "Amount exceeds total token supply"
        except (ValueError, TypeError):
            return False, "Invalid amount format. Must be integer."
        
        # Check sender exists and has balance
        sender = UserManager.get_user_by_id(tx_data['from_user_id'])
        if not sender:
            return False, f"Sender user not found: {tx_data['from_user_id']}"
        
        if not sender.is_active:
            return False, "Sender account is inactive"
        
        if not sender.has_sufficient_balance(amount):
            return False, f"Insufficient balance. Have: {sender.balance}, Need: {amount}"
        
        # Check receiver exists (for transfers)
        if tx_data['tx_type'] == 'transfer':
            receiver = UserManager.get_user_by_id(tx_data['to_user_id'])
            if not receiver:
                return False, f"Receiver user not found: {tx_data['to_user_id']}"
            
            if not receiver.is_active:
                return False, "Receiver account is inactive"
        
        # Validate gas price if provided
        gas_price = tx_data.get('gas_price', 1)
        try:
            gas_price = int(gas_price)
            if gas_price < 1 or gas_price > 1_000_000:
                return False, "Gas price out of valid range [1, 1000000]"
        except (ValueError, TypeError):
            return False, "Invalid gas_price format"
        
        return True, None
    
    @staticmethod
    def validate_gas(tx_type: str, gas_price: int) -> Tuple[int, int]:
        """
        Calculate gas required and total fee
        
        Returns:
            (gas_used, total_fee)
        """
        
        # Determine gas required based on transaction type
        if tx_type == 'transfer':
            gas_used = Config.GAS_PER_TRANSFER
        elif tx_type == 'contract_call':
            gas_used = Config.GAS_PER_CONTRACT_CALL
        elif tx_type == 'stake':
            gas_used = Config.GAS_PER_STAKE
        else:
            gas_used = 21000  # Default
        
        total_fee = gas_used * gas_price
        return gas_used, total_fee

# ═══════════════════════════════════════════════════════════════════════════════
# RATE LIMITING
# ═══════════════════════════════════════════════════════════════════════════════

class RateLimiter:
    """Rate limiting for API endpoints"""
    
    def __init__(self):
        self.user_requests = defaultdict(deque)  # user_id -> deque of timestamps
        self.lock = threading.Lock()
    
    def is_rate_limited(self, user_id: str) -> Tuple[bool, Optional[str]]:
        """Check if user is rate limited"""
        with self.lock:
            now = time.time()
            requests = self.user_requests[user_id]
            
            # Remove old requests outside the window
            while requests and requests[0] < now - Config.RATE_LIMIT_WINDOW_SECONDS:
                requests.popleft()
            
            # Check minute limit
            if len(requests) >= Config.MAX_TXS_PER_USER_PER_MINUTE:
                return True, f"Rate limit exceeded ({Config.MAX_TXS_PER_USER_PER_MINUTE} requests per minute)"
            
            # Add current request
            requests.append(now)
            return False, None
    
    def check_rate_limit(self, f):
        """Decorator for rate-limited endpoints"""
        @wraps(f)
        def decorated(*args, **kwargs):
            user_id = getattr(request, 'user_id', 'anonymous')
            is_limited, message = self.is_rate_limited(user_id)
            
            if is_limited:
                logger.warning(f"Rate limit exceeded for {user_id}: {message}")
                return jsonify({'error': message}), 429
            
            return f(*args, **kwargs)
        
        decorated.__name__ = f.__name__
        return decorated

rate_limiter = RateLimiter()

# ═══════════════════════════════════════════════════════════════════════════════
# FLASK APPLICATION INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config.from_object(Config)
CORS(app, resources={r"/api/*": {"origins": Config.ALLOWED_ORIGINS}})

# ═══════════════════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

@app.errorhandler(400)
def bad_request(error):
    """Handle 400 Bad Request"""
    logger.warning(f"Bad request: {error}")
    return jsonify({
        'error': 'Bad request',
        'message': str(error),
        'timestamp': datetime.utcnow().isoformat()
    }), 400

@app.errorhandler(401)
def unauthorized(error):
    """Handle 401 Unauthorized"""
    logger.warning(f"Unauthorized: {error}")
    return jsonify({
        'error': 'Unauthorized',
        'message': 'Authentication required',
        'timestamp': datetime.utcnow().isoformat()
    }), 401

@app.errorhandler(403)
def forbidden(error):
    """Handle 403 Forbidden"""
    logger.warning(f"Forbidden: {error}")
    return jsonify({
        'error': 'Forbidden',
        'message': 'Insufficient permissions',
        'timestamp': datetime.utcnow().isoformat()
    }), 403

@app.errorhandler(404)
def not_found(error):
    """Handle 404 Not Found"""
    return jsonify({
        'error': 'Not found',
        'message': str(error),
        'timestamp': datetime.utcnow().isoformat()
    }), 404

@app.errorhandler(429)
def rate_limit_exceeded(error):
    """Handle 429 Too Many Requests"""
    logger.warning(f"Rate limit exceeded: {error}")
    return jsonify({
        'error': 'Too many requests',
        'message': 'Rate limit exceeded. Please try again later.',
        'timestamp': datetime.utcnow().isoformat()
    }), 429

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 Internal Server Error"""
    logger.error(f"Internal server error: {error}\n{traceback.format_exc()}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred',
        'timestamp': datetime.utcnow().isoformat()
    }), 500

# ═══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint - verify system status"""
    try:
        # Test database connection
        DatabaseConnection.execute_one("SELECT 1")
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'database': 'connected',
            'service': 'api_gateway',
            'version': '1.0.0'
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 503

@app.route('/api/v1/status', methods=['GET'])
def get_status():
    """Get system status and statistics"""
    try:
        # Get network parameters
        rows = DatabaseConnection.execute("SELECT param_key, param_value FROM network_parameters LIMIT 10")
        params = {row['param_key']: row['param_value'] for row in rows}
        
        # Get block count
        latest_block = BlockManager.get_latest_block()
        block_number = latest_block.block_number if latest_block else 0
        
        # Get transaction count
        tx_row = DatabaseConnection.execute_one("SELECT COUNT(*) as count FROM transactions")
        tx_count = tx_row['count'] if tx_row else 0
        
        # Get user count
        user_row = DatabaseConnection.execute_one("SELECT COUNT(*) as count FROM users")
        user_count = user_row['count'] if user_row else 0
        
        return jsonify({
            'status': 'operational',
            'timestamp': datetime.utcnow().isoformat(),
            'blockchain': {
                'block_height': block_number,
                'total_transactions': tx_count,
                'total_users': user_count
            },
            'network_parameters': params
        }), 200
    except Exception as e:
        logger.error(f"Status request error: {e}")
        return jsonify({'error': 'Failed to retrieve status'}), 500

@app.route('/api/v1/auth/login', methods=['POST'])
def login():
    """
    User login endpoint
    
    Request JSON:
    {
        "email": "user@example.com",
        "password": "password123" (optional for dev)
    }
    
    Response:
    {
        "token": "jwt_token",
        "user": {user object},
        "expires_in": 86400
    }
    """
    try:
        data = request.get_json()
        if not data or 'email' not in data:
            return jsonify({'error': 'Email required'}), 400
        
        email = data['email'].strip().lower()
        
        # Get user by email
        user = UserManager.get_user_by_email(email)
        if not user:
            logger.warning(f"Login attempt for non-existent user: {email}")
            return jsonify({'error': 'User not found'}), 404
        
        # In production, verify password hash here
        # For now, allow login if user exists
        
        # Create JWT token
        token = JWTHandler.create_token(user.user_id, user.email, user.role)
        
        # Update last login
        UserManager.update_last_login(user.user_id)
        
        logger.info(f"User {email} logged in successfully")
        
        return jsonify({
            'token': token,
            'user': user.to_dict(),
            'expires_in': 86400
        }), 200
    
    except Exception as e:
        logger.error(f"Login error: {e}\n{traceback.format_exc()}")
        return jsonify({'error': 'Login failed'}), 500

@app.route('/api/v1/auth/verify', methods=['POST'])
@JWTHandler.require_auth
def verify_token():
    """Verify JWT token is valid"""
    try:
        user_id = request.user_id
        user = UserManager.get_user_by_id(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({
            'valid': True,
            'user': user.to_dict()
        }), 200
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        return jsonify({'error': 'Verification failed'}), 500

@app.route('/api/v1/account/<identifier>', methods=['GET'])
def get_account(identifier):
    """
    Get account information
    
    Parameters:
        identifier: user_id (32 char hex) or email
    
    Returns:
        User object with all public information
    """
    try:
        user = None
        
        # Try as user_id first (32 char hex)
        if len(identifier) == 32 and all(c in '0123456789abcdef' for c in identifier.lower()):
            user = UserManager.get_user_by_id(identifier)
        else:
            # Try as email
            user = UserManager.get_user_by_email(identifier)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify(user.to_dict()), 200
    except Exception as e:
        logger.error(f"Get account error: {e}")
        return jsonify({'error': 'Request failed'}), 500

@app.route('/api/v1/account/<user_id>/balance', methods=['GET'])
def get_balance(user_id):
    """Get user's current balance"""
    try:
        user = UserManager.get_user_by_id(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({
            'user_id': user_id,
            'balance': str(user.balance),
            'balance_display': f"{user.balance / Config.TOKEN_WEI_PER_UNIT:.18f}",
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Get balance error: {e}")
        return jsonify({'error': 'Request failed'}), 500

@app.route('/api/v1/pseudoqubit/<user_id>', methods=['GET'])
def get_user_pseudoqubit_endpoint(user_id):
    """Get user's assigned personal pseudoqubit"""
    try:
        qubit = PseudoqubitManager.get_user_pseudoqubit(user_id)
        if not qubit:
            return jsonify({'error': 'No pseudoqubit assigned to user'}), 404
        
        return jsonify(qubit.to_dict()), 200
    except Exception as e:
        logger.error(f"Get pseudoqubit error: {e}")
        return jsonify({'error': 'Request failed'}), 500

@app.route('/api/v1/pseudoqubit/address/<routing_address>', methods=['GET'])
def get_pseudoqubit_by_address(routing_address):
    """Get pseudoqubit by routing address"""
    try:
        qubit = PseudoqubitManager.get_pseudoqubit_by_routing_address(routing_address)
        if not qubit:
            return jsonify({'error': 'Pseudoqubit not found'}), 404
        
        return jsonify(qubit.to_dict()), 200
    except Exception as e:
        logger.error(f"Get pseudoqubit by address error: {e}")
        return jsonify({'error': 'Request failed'}), 500

@app.route('/api/v1/pool/qubits', methods=['GET'])
def get_pool_qubits():
    """Get available shared pool pseudoqubits"""
    try:
        count = request.args.get('count', default=7, type=int)
        count = min(count, 20)  # Max 20
        
        qubits = PseudoqubitManager.get_shared_pool_qubits(count)
        
        return jsonify({
            'qubits': [q.to_dict() for q in qubits],
            'count': len(qubits),
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Get pool qubits error: {e}")
        return jsonify({'error': 'Request failed'}), 500

@app.route('/api/v1/tx/submit', methods=['POST'])
@JWTHandler.require_auth
@rate_limiter.check_rate_limit
def submit_transaction():
    """
    Submit transaction for quantum execution
    
    Request JSON:
    {
        "to_user_id": "recipient_user_id",
        "amount": 1000000000000000000,  (Wei)
        "tx_type": "transfer",  (transfer|mint|burn|stake)
        "gas_price": 1
    }
    
    Response:
    {
        "tx_id": "transaction_hash",
        "status": "pending",
        "user_pseudoqubit": {pseudoqubit object},
        "pool_qubits": [{pseudoqubit}, ...],
        "gas_estimate": {gas_used, gas_price, total_fee},
        "created_at": timestamp
    }
    """
    try:
        user_id = request.user_id
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Request body required'}), 400
        
        # Add sender to transaction data
        data['from_user_id'] = user_id
        
        # Validate transaction input
        is_valid, error = TransactionValidator.validate_transaction_input(data)
        if not is_valid:
            logger.warning(f"Invalid transaction from {user_id}: {error}")
            return jsonify({'error': error}), 400
        
        # Calculate gas
        tx_type = data.get('tx_type', 'transfer')
        gas_price = data.get('gas_price', 1)
        gas_used, total_fee = TransactionValidator.validate_gas(tx_type, gas_price)
        
        # Check sender can afford transaction + gas
        sender = UserManager.get_user_by_id(user_id)
        total_cost = data['amount'] + total_fee
        if sender.balance < total_cost:
            return jsonify({'error': f'Insufficient balance for transaction + gas. Need: {total_cost}, Have: {sender.balance}'}), 400
        
        # Create transaction
        tx_id = TransactionManager.create_transaction(
            from_user_id=user_id,
            to_user_id=data['to_user_id'],
            amount=data['amount'],
            tx_type=tx_type,
            gas_price=gas_price
        )
        
        if not tx_id:
            return jsonify({'error': 'Failed to create transaction'}), 500
        
        # Get user's pseudoqubit
        user_qubit = PseudoqubitManager.get_user_pseudoqubit(user_id)
        if not user_qubit:
            logger.error(f"User {user_id} has no assigned pseudoqubit")
            TransactionManager.update_transaction_status(tx_id, 'failed')
            return jsonify({'error': 'User pseudoqubit not assigned'}), 400
        
        # Get shared pool qubits
        pool_qubits = PseudoqubitManager.get_shared_pool_qubits(7)
        if len(pool_qubits) < 7:
            logger.error(f"Not enough pool qubits available ({len(pool_qubits)}/7)")
            TransactionManager.update_transaction_status(tx_id, 'failed')
            return jsonify({'error': f'System overloaded: {len(pool_qubits)}/7 pool qubits available'}), 503
        
        logger.info(f"Transaction {tx_id} submitted by {user_id}: {data['amount']} → {data['to_user_id']}")
        
        return jsonify({
            'tx_id': tx_id,
            'status': 'pending',
            'user_pseudoqubit': user_qubit.to_dict(),
            'pool_qubits': [q.to_dict() for q in pool_qubits],
            'gas_estimate': {
                'gas_used': gas_used,
                'gas_price': gas_price,
                'total_fee': str(total_fee)
            },
            'created_at': datetime.utcnow().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Submit transaction error: {e}\n{traceback.format_exc()}")
        return jsonify({'error': 'Request failed'}), 500

@app.route('/api/v1/tx/<tx_id>', methods=['GET'])
def get_transaction_status(tx_id):
    """Get transaction status and details"""
    try:
        tx = TransactionManager.get_transaction(tx_id)
        if not tx:
            return jsonify({'error': 'Transaction not found'}), 404
        
        return jsonify(tx.to_dict()), 200
    except Exception as e:
        logger.error(f"Get transaction error: {e}")
        return jsonify({'error': 'Request failed'}), 500

@app.route('/api/v1/tx/<user_id>/history', methods=['GET'])
def get_user_transactions(user_id):
    """Get transaction history for user"""
    try:
        limit = request.args.get('limit', default=50, type=int)
        limit = min(limit, 500)  # Max 500
        
        transactions = TransactionManager.get_user_transactions(user_id, limit)
        
        return jsonify({
            'transactions': [tx.to_dict() for tx in transactions],
            'count': len(transactions),
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Get user transactions error: {e}")
        return jsonify({'error': 'Request failed'}), 500

@app.route('/api/v1/block/<int:block_number>', methods=['GET'])
def get_block(block_number):
    """Get block information"""
    try:
        block = BlockManager.get_block(block_number)
        if not block:
            return jsonify({'error': 'Block not found'}), 404
        
        return jsonify(block.to_dict()), 200
    except Exception as e:
        logger.error(f"Get block error: {e}")
        return jsonify({'error': 'Request failed'}), 500

@app.route('/api/v1/block/latest', methods=['GET'])
def get_latest_block():
    """Get latest block in chain"""
    try:
        block = BlockManager.get_latest_block()
        if not block:
            return jsonify({'error': 'No blocks found'}), 404
        
        return jsonify(block.to_dict()), 200
    except Exception as e:
        logger.error(f"Get latest block error: {e}")
        return jsonify({'error': 'Request failed'}), 500

@app.route('/api/v1/block/0', methods=['GET'])
def get_genesis_block():
    """Get genesis block"""
    try:
        block = BlockManager.get_genesis_block()
        if not block:
            return jsonify({'error': 'Genesis block not found'}), 404
        
        return jsonify(block.to_dict()), 200
    except Exception as e:
        logger.error(f"Get genesis block error: {e}")
        return jsonify({'error': 'Request failed'}), 500

@app.route('/api/v1/network/params', methods=['GET'])
def get_network_params():
    """Get network parameters"""
    try:
        rows = DatabaseConnection.execute("SELECT param_key, param_value FROM network_parameters")
        params = {row['param_key']: row['param_value'] for row in rows}
        
        return jsonify({
            'parameters': params,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Get network params error: {e}")
        return jsonify({'error': 'Request failed'}), 500

# ═══════════════════════════════════════════════════════════════════════════════
# SHUTDOWN HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

def shutdown_handler():
    """Clean shutdown of API gateway"""
    logger.info("Shutting down API gateway...")
    DatabaseConnection.close_all()
    cache.clear()
    logger.info("Shutdown complete")

import atexit
atexit.register(shutdown_handler)

# ═══════════════════════════════════════════════════════════════════════════════
# WSGI SETUP FUNCTIONS (for PythonAnywhere deployment)
# ═══════════════════════════════════════════════════════════════════════════════

def setup_database(app):
    """Initialize database connection pool"""
    logger.info("Setting up database connection pool...")
    try:
        db = DatabaseConnection()
        # Test connection
        conn = db.get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        db.return_connection(conn)
        logger.info("✓ Database pool ready")
        return True
    except Exception as e:
        logger.warning(f"⚠ Database setup delayed: {e}")
        return False  # Don't fail - DB will retry on requests

def setup_api_routes(app):
    """Register all API routes with Flask app"""
    logger.info("Registering API routes...")
    
    # Skip health_check - already registered in wsgi_config
    
    # Authentication
    app.add_url_rule('/api/auth/login', 'login', login, methods=['POST'])
    app.add_url_rule('/api/auth/verify', 'verify_token', verify_token, methods=['POST'])
    
    # Account & Balance
    app.add_url_rule('/api/account/<identifier>', 'get_account', get_account, methods=['GET'])
    app.add_url_rule('/api/balance/<user_id>', 'get_balance', get_balance, methods=['GET'])
    
    # Pseudoqubits
    app.add_url_rule('/api/pseudoqubit/<user_id>', 'get_user_pseudoqubit', get_user_pseudoqubit_endpoint, methods=['GET'])
    app.add_url_rule('/api/pseudoqubit/address/<routing_address>', 'get_pseudoqubit_by_address', get_pseudoqubit_by_address, methods=['GET'])
    app.add_url_rule('/api/pool/qubits', 'get_pool_qubits', get_pool_qubits, methods=['GET'])
    
    # Transactions
    app.add_url_rule('/api/transactions', 'submit_transaction', submit_transaction, methods=['POST'])
    app.add_url_rule('/api/transactions/<tx_id>', 'get_transaction_status', get_transaction_status, methods=['GET'])
    app.add_url_rule('/api/transactions/user/<user_id>', 'get_user_transactions', get_user_transactions, methods=['GET'])
    
    # Blocks
    app.add_url_rule('/api/blocks/<int:block_number>', 'get_block', get_block, methods=['GET'])
    
    # Error handlers
    app.register_error_handler(400, bad_request)
    app.register_error_handler(401, unauthorized)
    app.register_error_handler(403, forbidden)
    app.register_error_handler(404, not_found)
    app.register_error_handler(429, rate_limit_exceeded)
    app.register_error_handler(500, internal_error)
    
    logger.info("✓ All API routes registered")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    logger.info("=" * 100)
    logger.info("QTCL API GATEWAY INITIALIZING")
    logger.info("=" * 100)
    logger.info(f"Configuration:")
    logger.info(f"  - Host: {Config.API_HOST}:{Config.API_PORT}")
    logger.info(f"  - Database: {Config.SUPABASE_HOST}")
    logger.info(f"  - Debug: {Config.DEBUG}")
    logger.info(f"  - Max Content Length: {Config.MAX_CONTENT_LENGTH / 1024 / 1024}MB")
    logger.info(f"  - JWT Algorithm: {Config.JWT_ALGORITHM}")
    logger.info(f"  - Token Supply: {Config.TOKEN_TOTAL_SUPPLY:,} QTCL")
    logger.info(f"  - Qiskit Shots: {Config.QISKIT_SHOTS}")
    logger.info(f"  - Qiskit Qubits: {Config.QISKIT_QUBITS}")
    logger.info("=" * 100)
    
    try:
        # Test database connection before starting
        db = DatabaseConnection()
        db_conn = db.get_connection()
        try:
            with db_conn.cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()
                logger.info(f"✓ Database connected: {version[0]}")
        finally:
            db.return_connection(db_conn)
        
        # Start Flask app
        logger.info("✓ Starting Flask application...")
        app.run(
            host=Config.API_HOST,
            port=Config.API_PORT,
            debug=Config.DEBUG,
            use_reloader=False,
            threaded=True
        )
    
    except KeyboardInterrupt:
        logger.info("\n✓ Keyboard interrupt received")
        shutdown_handler()
    except Exception as e:
        logger.error(f"✗ Fatal error: {e}\n{traceback.format_exc()}")
        shutdown_handler()
        sys.exit(1)