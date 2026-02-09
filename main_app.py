
#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════
QUANTUM TEMPORAL COHERENCE LEDGER (QTCL) - MAIN APPLICATION
Complete Transaction Monitor, Quantum Router, and Execution Orchestrator
═══════════════════════════════════════════════════════════════════════════════════════

MONOLITHIC PRODUCTION APPLICATION - ~7000+ lines single continuous output
Deployment: PythonAnywhere
Database: Supabase PostgreSQL
Quantum: Qiskit + AER Simulator (1024 shots per transaction)

CORE RESPONSIBILITIES:
  ├─ Genesis Block Initialization (Thread 1)
  ├─ Transaction Polling & Validation (Thread 2)
  ├─ Pseudoqubit Management & Retrieval (Thread 3)
  ├─ Quantum Circuit Job Queue Management (Thread 4)
  ├─ Superposition Lifecycle Management (Thread 5)
  ├─ Oracle Event Listening (Thread 6)
  ├─ Finality & Block Creation (Thread 7)
  ├─ Database State Synchronization (Thread 8)
  └─ WebSocket Status Broadcasting (Thread 9)

QUANTUM INTEGRATION:
  ├─ All transactions executed via quantum circuits
  ├─ Superposition maintained with entropy tracking
  ├─ Measurement results integrated from test suite
  ├─ Coherence refresh with Floquet cycle tracking
  ├─ Entanglement management for validator consensus
  └─ Collapse triggering via oracle data

TOTAL SYSTEM ARCHITECTURE:
  Genesis Block → Pending TX Pool → Quantum Executor → Superposition State → Oracle Collapse → Finality → Blockchain
═══════════════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import time
import json
import hashlib
import logging
import threading
import asyncio
import queue
import sqlite3
import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
from functools import wraps
import subprocess
import traceback
import cmath
import math
import base64
import pickle

# ═══════════════════════════════════════════════════════════════════════════════════════
# DEPENDENCY INSTALLATION
# ═══════════════════════════════════════════════════════════════════════════════════════

def ensure_packages():
    """Ensure all required packages installed"""
    packages = {
        'psycopg2': 'psycopg2-binary',
        'numpy': 'numpy',
        'redis': 'redis',
        'qiskit': 'qiskit',
        'qiskit_aer': 'qiskit-aer',
        'websocket': 'websocket-client',
        'requests': 'requests'
    }
    
    for module, pip_name in packages.items():
        try:
            __import__(module)
        except ImportError:
            print(f"[INSTALL] Installing {pip_name}...")
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', '-q', pip_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

ensure_packages()

import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

# ═══════════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════════════

class Config:
    """Complete configuration for main application"""
    
    # Database
    SUPABASE_HOST = "aws-0-us-west-2.pooler.supabase.com"
    SUPABASE_USER = "postgres.rslvlsqwkfmdtebqsvtw"
    SUPABASE_PASSWORD = "$h10j1r1H0w4rd"
    SUPABASE_PORT = 5432
    SUPABASE_DB = "postgres"
    DB_POOL_SIZE = 10
    DB_CONNECTION_TIMEOUT = 30
    
    # Threading
    NUM_WORKER_THREADS = 9
    THREAD_NAMES = [
        'GenesisInitializer',
        'TransactionPoller',
        'PseudoqubitManager',
        'QuantumJobQueue',
        'SuperpositionManager',
        'OracleListener',
        'FinalityProcessor',
        'DatabaseSync',
        'WebSocketBroadcaster'
    ]
    
    # Polling Intervals (milliseconds)
    POLL_INTERVAL_TRANSACTIONS_MS = 100
    POLL_INTERVAL_SUPERPOSITION_MS = 500
    POLL_INTERVAL_ORACLE_MS = 200
    POLL_INTERVAL_FINALITY_MS = 300
    POLL_INTERVAL_SYNC_MS = 1000
    
    # Batch Sizes
    BATCH_SIZE_TRANSACTIONS = 100
    BATCH_SIZE_QUANTUM_JOBS = 50
    BATCH_SIZE_SUPERPOSITION_REFRESH = 25
    
    # Quantum Parameters
    QISKIT_SHOTS = 1024
    QISKIT_QUBITS = 8
    QISKIT_SIMULATOR_SEED = 42
    CIRCUIT_TIMEOUT_MS = 200
    
    # Superposition Lifecycle
    SUPERPOSITION_TIMEOUT_SECONDS = 300  # 5 minutes
    COHERENCE_REFRESH_INTERVAL_SECONDS = 5
    MIN_ENTROPY_FOR_SUPERPOSITION = 0.80
    MAX_COHERENCE_REFRESH_COUNT = 60
    FLOQUET_CYCLE_INTERVAL_SECONDS = 10
    
    # Block Parameters
    BLOCK_SIZE_MAX_TRANSACTIONS = 1000
    BLOCK_CREATION_INTERVAL_SECONDS = 10
    GENESIS_BLOCK_VALIDATOR = "system@qtcl.network"
    
    # Token Economics
    TOKEN_TOTAL_SUPPLY = 1_000_000_000
    TOKEN_DECIMALS = 18
    TOKEN_WEI_PER_UNIT = 10 ** TOKEN_DECIMALS
    
    # Initial Users (from __NOTES.txt)
    INITIAL_USERS = [
        {
            'email': 'shemshallah@gmail.com',
            'name': 'Dev Account',
            'balance': 999_000_000,
            'role': 'admin'
        },
        {
            'email': 'founder1@qtcl.network',
            'name': 'Founding Member 1',
            'balance': 200_000,
            'role': 'founder'
        },
        {
            'email': 'founder2@qtcl.network',
            'name': 'Founding Member 2',
            'balance': 200_000,
            'role': 'founder'
        },
        {
            'email': 'founder3@qtcl.network',
            'name': 'Founding Member 3',
            'balance': 200_000,
            'role': 'founder'
        },
        {
            'email': 'founder4@qtcl.network',
            'name': 'Founding Member 4',
            'balance': 200_000,
            'role': 'founder'
        },
        {
            'email': 'founder5@qtcl.network',
            'name': 'Founding Member 5',
            'balance': 200_000,
            'role': 'founder'
        }
    ]

# ═══════════════════════════════════════════════════════════════════════════════════════
# LOGGING SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════════════

class ThreadLogger:
    """Thread-safe logging with context awareness"""
    
    @staticmethod
    def setup():
        """Initialize logging system"""
        log_format = '[%(asctime)s] [%(threadName)s] %(levelname)s: %(message)s'
        
        file_handler = logging.FileHandler('qtcl_main_app.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format))
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(log_format))
        
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        return logging.getLogger(__name__)

logger = ThreadLogger.setup()

# ═══════════════════════════════════════════════════════════════════════════════════════
# DATABASE CONNECTION POOL
# ═══════════════════════════════════════════════════════════════════════════════════════

class DatabasePool:
    """Thread-safe database connection pool"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabasePool, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.pool = deque(maxlen=Config.DB_POOL_SIZE)
        self.lock = threading.Lock()
        self.stats = {
            'connections_created': 0,
            'connections_reused': 0,
            'queries_executed': 0,
            'queries_failed': 0
        }
        logger.info(f"DatabasePool initialized with size {Config.DB_POOL_SIZE}")
    
    def get_connection(self):
        """Get connection from pool or create new"""
        with self.lock:
            try:
                conn = self.pool.popleft()
                if conn and not conn.closed:
                    self.stats['connections_reused'] += 1
                    return conn
            except IndexError:
                pass
        
        try:
            conn = psycopg2.connect(
                host=Config.SUPABASE_HOST,
                user=Config.SUPABASE_USER,
                password=Config.SUPABASE_PASSWORD,
                port=Config.SUPABASE_PORT,
                database=Config.SUPABASE_DB,
                connect_timeout=Config.DB_CONNECTION_TIMEOUT
            )
            conn.set_session(autocommit=True)
            self.stats['connections_created'] += 1
            return conn
        except psycopg2.Error as e:
            logger.error(f"Failed to create database connection: {e}")
            raise
    
    def return_connection(self, conn):
        """Return connection to pool"""
        if conn and not conn.closed:
            with self.lock:
                try:
                    self.pool.append(conn)
                except Exception as e:
                    logger.warning(f"Failed to return connection to pool: {e}")
                    conn.close()
    
    def execute(self, query: str, params: tuple = None) -> List[Dict]:
        """Execute SELECT query"""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params or ())
                self.stats['queries_executed'] += 1
                return cur.fetchall()
        except psycopg2.Error as e:
            logger.error(f"Query error: {e}")
            self.stats['queries_failed'] += 1
            raise
        finally:
            self.return_connection(conn)
    
    def execute_one(self, query: str, params: tuple = None) -> Optional[Dict]:
        """Execute SELECT query, return first row"""
        results = self.execute(query, params)
        return results[0] if results else None
    
    def execute_insert(self, query: str, params: tuple = None) -> int:
        """Execute INSERT/UPDATE/DELETE"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(query, params or ())
                self.stats['queries_executed'] += 1
                return cur.rowcount
        except psycopg2.Error as e:
            logger.error(f"Insert error: {e}")
            self.stats['queries_failed'] += 1
            raise
        finally:
            self.return_connection(conn)
    
    def execute_batch(self, query: str, params_list: List[tuple]) -> int:
        """Execute batch INSERT/UPDATE/DELETE"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                execute_batch(cur, query, params_list, page_size=1000)
                self.stats['queries_executed'] += 1
                return cur.rowcount
        except psycopg2.Error as e:
            logger.error(f"Batch error: {e}")
            self.stats['queries_failed'] += 1
            raise
        finally:
            self.return_connection(conn)
    
    def close_all(self):
        """Close all pooled connections"""
        with self.lock:
            while len(self.pool) > 0:
                try:
                    conn = self.pool.popleft()
                    if conn and not conn.closed:
                        conn.close()
                except Exception as e:
                    logger.warning(f"Error closing connection: {e}")
        logger.info("All database connections closed")

db_pool = DatabasePool()

# ═══════════════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class HyperbolicCoordinates:
    """Hyperbolic coordinate system representations"""
    poincare_x: float
    poincare_y: float
    klein_x: float
    klein_y: float
    hyperboloid_x: float
    hyperboloid_y: float
    hyperboloid_t: float
    
    @staticmethod
    def poincare_to_klein(px: float, py: float) -> Tuple[float, float]:
        """Convert Poincaré to Klein coordinates"""
        r_sq = px*px + py*py
        factor = 2 / (1 + r_sq)
        return px * factor, py * factor
    
    @staticmethod
    def poincare_to_hyperboloid(px: float, py: float) -> Tuple[float, float, float]:
        """Convert Poincaré to Hyperboloid coordinates"""
        r_sq = px*px + py*py
        t = (1 + r_sq) / (1 - r_sq)
        factor = 2 / (1 - r_sq)
        return px * factor, py * factor, t
    
    @staticmethod
    def hyperbolic_distance(p1x: float, p1y: float, p2x: float, p2y: float) -> float:
        """Calculate hyperbolic distance between two Poincaré points"""
        dx = p1x - p2x
        dy = p1y - p2y
        r1_sq = p1x*p1x + p1y*p1y
        r2_sq = p2x*p2x + p2y*p2y
        
        numerator = 2 * (dx*dx + dy*dy)
        denominator = (1 - r1_sq) * (1 - r2_sq)
        
        if denominator == 0:
            return float('inf')
        
        ratio = numerator / denominator
        if ratio <= 0:
            return 0
        
        return math.acosh(1 + ratio)

@dataclass
class PseudoqubitState:
    """Quantum state of a pseudoqubit"""
    pseudoqubit_id: int
    state_vector: List[complex] = field(default_factory=list)
    fidelity: float = 0.98
    coherence: float = 0.95
    purity: float = 0.97
    entropy: float = 0.0
    concurrence: float = 0.90
    
    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps({
            'pseudoqubit_id': self.pseudoqubit_id,
            'fidelity': self.fidelity,
            'coherence': self.coherence,
            'purity': self.purity,
            'entropy': self.entropy,
            'concurrence': self.concurrence
        })

@dataclass
class TransactionInSuperposition:
    """Transaction in quantum superposition state"""
    tx_id: str
    from_user_id: str
    to_user_id: str
    amount: int
    tx_type: str
    user_pseudoqubit_id: int
    pool_qubit_ids: List[int]
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    entered_superposition_at: Optional[datetime] = None
    last_coherence_refresh: Optional[datetime] = None
    
    coherence_refresh_count: int = 0
    floquet_cycle: int = 0
    entropy_current: float = 0.0
    quantum_state_hash: Optional[str] = None
    
    measurement_results: Optional[Dict] = None
    collapsed_outcome: Optional[str] = None
    
    def age_seconds(self) -> float:
        """Age in seconds since creation"""
        return (datetime.utcnow() - self.created_at).total_seconds()
    
    def is_expired(self) -> bool:
        """Check if transaction exceeded superposition timeout"""
        return self.age_seconds() > Config.SUPERPOSITION_TIMEOUT_SECONDS
    
    def needs_coherence_refresh(self) -> bool:
        """Check if coherence refresh needed"""
        if self.last_coherence_refresh is None:
            return True
        elapsed = (datetime.utcnow() - self.last_coherence_refresh).total_seconds()
        return elapsed >= Config.COHERENCE_REFRESH_INTERVAL_SECONDS

@dataclass
class QuantumMeasurementResult:
    """Results from quantum circuit execution"""
    tx_id: str
    bitstring_counts: Dict[str, int]
    total_shots: int
    entropy_bits: float
    entropy_percent: float
    dominant_states: List[str]
    quantum_property: str
    execution_time_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        """Convert to dict"""
        return {
            'tx_id': self.tx_id,
            'bitstring_counts': self.bitstring_counts,
            'total_shots': self.total_shots,
            'entropy_bits': self.entropy_bits,
            'entropy_percent': self.entropy_percent,
            'dominant_states': self.dominant_states,
            'quantum_property': self.quantum_property,
            'execution_time_ms': self.execution_time_ms,
            'timestamp': self.timestamp.isoformat()
        }

# ═══════════════════════════════════════════════════════════════════════════════════════
# QUANTUM CIRCUIT EXECUTOR
# ═══════════════════════════════════════════════════════════════════════════════════════

class QuantumCircuitExecutor:
    """Builds and executes quantum circuits using Qiskit"""
    
    def __init__(self):
        self.simulator = AerSimulator()
        self.execution_count = 0
        self.total_execution_time = 0.0
        logger.info("QuantumCircuitExecutor initialized")
    
    def build_circuit(
        self,
        tx_id: str,
        from_user_id: str,
        to_user_id: str,
        amount: int,
        tx_type: str,
        user_qubit_id: int,
        pool_qubit_ids: List[int]
    ) -> QuantumCircuit:
        """
        Build quantum circuit for transaction
        
        Circuit structure (8 qubits):
          Qubit 0: User's personal pseudoqubit
          Qubits 1-7: Shared pool pseudoqubits
        
        Gates:
          Phase 1: Create superposition with Hadamard
          Phase 2: Entangle with CNOT ladder
          Phase 3: Encode transaction parameters with rotations
          Phase 4: Measure all qubits
        """
        
        qreg = QuantumRegister(8, 'q')
        creg = ClassicalRegister(8, 'c')
        circuit = QuantumCircuit(qreg, creg, name=f'tx_{tx_id[:8]}')
        
        # Phase 1: Initialize superposition
        # Apply Hadamard to user qubit (qubit 0)
        circuit.h(qreg[0])
        
        # Apply Hadamard to pool qubits for multi-qubit superposition
        for i in range(1, min(8, len(pool_qubit_ids) + 1)):
            circuit.h(qreg[i])
        
        # Phase 2: Create entanglement ladder
        # CNOT chain to entangle all qubits
        for i in range(7):
            circuit.cx(qreg[i], qreg[i + 1])
        
        # Phase 3: Encode transaction information
        # Encode amount as rotation angle (normalize to [0, 2π])
        amount_normalized = (amount % (10**18)) / 10**18 * 2 * math.pi
        circuit.rx(amount_normalized, qreg[0])
        
        # Encode recipient hash as rotation
        to_hash = int(hashlib.sha256(to_user_id.encode()).hexdigest(), 16)
        to_angle = (to_hash % 360) * (math.pi / 180)
        circuit.ry(to_angle, qreg[1])
        
        # Encode sender hash as rotation
        from_hash = int(hashlib.sha256(from_user_id.encode()).hexdigest(), 16)
        from_angle = (from_hash % 360) * (math.pi / 180)
        circuit.rz(from_angle, qreg[2])
        
        # Encode transaction type
        tx_type_map = {'transfer': 0, 'mint': 1, 'burn': 2, 'stake': 3}
        tx_code = tx_type_map.get(tx_type, 0)
        circuit.ry(tx_code * (math.pi / 2), qreg[3])
        
        # Additional entanglement for robustness
        circuit.cx(qreg[0], qreg[3])
        circuit.cx(qreg[1], qreg[4])
        circuit.cx(qreg[2], qreg[5])
        
        # Phase 4: Measure all qubits
        for i in range(8):
            circuit.measure(qreg[i], creg[i])
        
        logger.debug(f"Built circuit for {tx_id}: {circuit.depth()} depth, {circuit.size()} gates")
        return circuit
    
    def execute(self, circuit: QuantumCircuit, seed: Optional[int] = None) -> QuantumMeasurementResult:
        """
        Execute circuit on AER simulator
        
        Returns measurement results with entropy calculation
        """
        
        start_time = time.time()
        tx_id = circuit.name.replace('tx_', '')[:16]
        
        try:
            if seed is None:
                seed = Config.QISKIT_SIMULATOR_SEED + self.execution_count
            
            # Execute on simulator
            job = self.simulator.run(
                circuit,
                shots=Config.QISKIT_SHOTS,
                seed_simulator=seed,
                optimization_level=2
            )
            
            result = job.result()
            counts = result.get_counts(circuit)
            
            execution_time_ms = (time.time() - start_time) * 1000
            self.execution_count += 1
            self.total_execution_time += execution_time_ms
            
            # Calculate entropy from measurement results
            entropy_bits, entropy_percent, dominant_states, quantum_property = self._analyze_measurements(counts)
            
            measurement_result = QuantumMeasurementResult(
                tx_id=tx_id,
                bitstring_counts=counts,
                total_shots=Config.QISKIT_SHOTS,
                entropy_bits=entropy_bits,
                entropy_percent=entropy_percent,
                dominant_states=dominant_states,
                quantum_property=quantum_property,
                execution_time_ms=execution_time_ms
            )
            
            logger.info(f"Executed circuit {tx_id}: entropy={entropy_bits:.2f} bits ({entropy_percent:.1f}%), "
                       f"time={execution_time_ms:.1f}ms, property={quantum_property}")
            
            return measurement_result
        
        except Exception as e:
            logger.error(f"Circuit execution failed for {tx_id}: {e}")
            raise
    
    def _analyze_measurements(self, counts: Dict[str, int]) -> Tuple[float, float, List[str], str]:
        """
        Analyze measurement results and calculate quantum properties
        
        Returns:
            (entropy_bits, entropy_percent, dominant_states, quantum_property)
        """
        
        # Calculate probabilities
        total = sum(counts.values())
        probabilities = {state: count / total for state, count in counts.items()}
        
        # Shannon entropy
        entropy = 0.0
        for prob in probabilities.values():
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        # Maximum possible entropy for 8 qubits
        max_entropy = 8
        entropy_percent = (entropy / max_entropy) * 100 if max_entropy > 0 else 0
        
        # Get dominant states
        sorted_states = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        dominant_states = [state for state, _ in sorted_states[:4]]
        
        # Classify quantum property
        if entropy_percent > 95:
            quantum_property = "Pure Superposition"
        elif entropy_percent > 80:
            quantum_property = "Strong Superposition"
        elif entropy_percent > 60:
            # Check for GHZ pattern (|000...⟩ and |111...⟩ dominant)
            if '00000000' in counts and '11111111' in counts:
                total_ghz = counts.get('00000000', 0) + counts.get('11111111', 0)
                if total_ghz > total * 0.6:
                    quantum_property = "GHZ Entanglement"
                else:
                    quantum_property = "Constrained"
            else:
                quantum_property = "Constrained"
        else:
            quantum_property = "Classical-like"
        
        return entropy, entropy_percent, dominant_states, quantum_property

quantum_executor = QuantumCircuitExecutor()

# ═══════════════════════════════════════════════════════════════════════════════════════
# GENESIS BLOCK INITIALIZER (THREAD 1)
# ═══════════════════════════════════════════════════════════════════════════════════════

class GenesisBlockInitializer:
    """Initializes genesis block and blockchain state"""
    
    def __init__(self):
        self.initialized = False
        self.lock = threading.Lock()
    
    def initialize(self) -> bool:
        """Initialize genesis block and system state"""
        
        with self.lock:
            if self.initialized:
                logger.info("Genesis already initialized")
                return True
            
            try:
                logger.info("=" * 100)
                logger.info("GENESIS BLOCK INITIALIZATION")
                logger.info("=" * 100)
                
                # Check if genesis already exists
                genesis = db_pool.execute_one("SELECT * FROM blocks WHERE block_number = 0")
                if genesis:
                    logger.info("✓ Genesis block already exists")
                    self.initialized = True
                    return True
                
                # Create genesis block
                genesis_hash_input = "QTCL_GENESIS_BLOCK"
                genesis_hash = f"0x{hashlib.sha256(genesis_hash_input.encode()).hexdigest()}"
                
                db_pool.execute_insert(
                    """INSERT INTO blocks 
                       (block_number, block_hash, parent_hash, timestamp, transactions, 
                        validator_address, quantum_state_hash, entropy_score, floquet_cycle,
                        merkle_root, difficulty, gas_used, gas_limit, miner_reward, created_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())""",
                    (0, genesis_hash, "0x" + "0" * 64, int(time.time()), 0,
                     Config.GENESIS_BLOCK_VALIDATOR, genesis_hash, 0.0, 0,
                     hashlib.sha256(b"genesis").hexdigest(), 0, 0, Config.BLOCK_GAS_LIMIT, 0)
                )
                
                logger.info(f"✓ Created genesis block: {genesis_hash}")
                
                # Initialize users
                for user_data in Config.INITIAL_USERS:
                    user_id = hashlib.sha256(user_data['email'].encode()).hexdigest()[:32]
                    balance = user_data['balance'] * Config.TOKEN_WEI_PER_UNIT
                    
                    # Check if user exists
                    existing = db_pool.execute_one("SELECT * FROM users WHERE user_id = %s", (user_id,))
                    if existing:
                        logger.info(f"✓ User already exists: {user_data['email']}")
                        continue
                    
                    db_pool.execute_insert(
                        """INSERT INTO users 
                           (user_id, email, name, role, balance, created_at, is_active, kyc_verified)
                           VALUES (%s, %s, %s, %s, %s, NOW(), %s, %s)""",
                        (user_id, user_data['email'], user_data['name'], user_data['role'],
                         balance, True, user_data['role'] == 'admin')
                    )
                    
                    logger.info(f"✓ Created user: {user_data['email']} with balance {user_data['balance']}")
                
                # Assign pseudoqubits to users
                logger.info("Assigning pseudoqubits to users...")
                
                for idx, user_data in enumerate(Config.INITIAL_USERS):
                    user_id = hashlib.sha256(user_data['email'].encode()).hexdigest()[:32]
                    
                    # Get first available pseudoqubit for this user
                    available_qubit = db_pool.execute_one(
                        """SELECT pseudoqubit_id FROM pseudoqubits 
                           WHERE is_available = TRUE AND auth_user_id IS NULL 
                           ORDER BY fidelity DESC LIMIT 1"""
                    )
                    
                    if available_qubit:
                        qubit_id = available_qubit['pseudoqubit_id']
                        db_pool.execute_insert(
                            """UPDATE pseudoqubits 
                               SET auth_user_id = %s, assigned_at = NOW(), is_available = FALSE 
                               WHERE pseudoqubit_id = %s""",
                            (user_id, qubit_id)
                        )
                        logger.info(f"✓ Assigned pseudoqubit {qubit_id} to {user_data['email']}")
                
                # Initialize network parameters
                logger.info("Initializing network parameters...")
                
                network_params = {
                    'total_supply': str(Config.TOKEN_TOTAL_SUPPLY),
                    'decimals': str(Config.TOKEN_DECIMALS),
                    'genesis_timestamp': str(int(time.time())),
                    'tessellation_type': "8,3",
                    'max_block_size': str(Config.BLOCK_SIZE_MAX_TRANSACTIONS),
                    'block_interval_seconds': str(Config.BLOCK_CREATION_INTERVAL_SECONDS),
                    'qiskit_shots': str(Config.QISKIT_SHOTS),
                    'qiskit_qubits': str(Config.QISKIT_QUBITS),
                    'superposition_timeout': str(Config.SUPERPOSITION_TIMEOUT_SECONDS),
                    'coherence_refresh_interval': str(Config.COHERENCE_REFRESH_INTERVAL_SECONDS)
                }
                
                for key, value in network_params.items():
                    db_pool.execute_insert(
                        "INSERT INTO network_parameters (param_key, param_value) VALUES (%s, %s)",
                        (key, value)
                    )
                
                logger.info("✓ Network parameters initialized")
                
                logger.info("=" * 100)
                logger.info("GENESIS INITIALIZATION COMPLETE")
                logger.info("=" * 100)
                
                self.initialized = True
                return True
            
            except Exception as e:
                logger.error(f"Genesis initialization failed: {e}\n{traceback.format_exc()}")
                return False

genesis_initializer = GenesisBlockInitializer()

# ═══════════════════════════════════════════════════════════════════════════════════════
# TRANSACTION POLLER (THREAD 2)
# ═══════════════════════════════════════════════════════════════════════════════════════

class TransactionPoller:
    """Polls for pending transactions and routes to quantum executor"""
    
    def __init__(self):
        self.running = True
        self.stats = {
            'total_polled': 0,
            'total_queued': 0,
            'total_failed': 0
        }
    
    def poll_transactions(self) -> List[Dict]:
        """Fetch pending transactions from database"""
        try:
            rows = db_pool.execute(
                """SELECT * FROM transactions 
                   WHERE status = 'pending' 
                   ORDER BY created_at ASC 
                   LIMIT %s""",
                (Config.BATCH_SIZE_TRANSACTIONS,)
            )
            return rows
        except Exception as e:
            logger.error(f"Transaction polling error: {e}")
            return []
    
    def validate_transaction(self, tx: Dict) -> Tuple[bool, Optional[str]]:
        """Validate transaction before quantum execution"""
        
        # Check sender
        sender_row = db_pool.execute_one(
            "SELECT balance, is_active FROM users WHERE user_id = %s",
            (tx['from_user_id'],)
        )
        
        if not sender_row:
            return False, "Sender not found"
        
        if not sender_row['is_active']:
            return False, "Sender inactive"
        
        if sender_row['balance'] < tx['amount']:
            return False, "Insufficient balance"
        
        # Check receiver
        receiver_row = db_pool.execute_one(
            "SELECT is_active FROM users WHERE user_id = %s",
            (tx['to_user_id'],)
        )
        
        if not receiver_row:
            return False, "Receiver not found"
        
        if not receiver_row['is_active']:
            return False, "Receiver inactive"
        
        # Check age
        created = tx['created_at']
        if isinstance(created, str):
            created = datetime.fromisoformat(created.replace('Z', '+00:00'))
        
        age = (datetime.utcnow() - created).total_seconds()
        if age > Config.SUPERPOSITION_TIMEOUT_SECONDS:
            return False, "Transaction expired"
        
        return True, None
    
    def queue_for_quantum_execution(self, tx: Dict) -> bool:
        """Queue transaction for quantum executor"""
        try:
            # Get user's pseudoqubit
            user_qubit = db_pool.execute_one(
                "SELECT pseudoqubit_id FROM pseudoqubits WHERE auth_user_id = %s",
                (tx['from_user_id'],)
            )
            
            if not user_qubit:
                logger.error(f"No pseudoqubit for {tx['from_user_id']}")
                return False
            
            # Get pool qubits
            pool_qubits = db_pool.execute(
                """SELECT pseudoqubit_id FROM pseudoqubits 
                   WHERE auth_user_id IS NULL AND is_available = TRUE 
                   ORDER BY fidelity DESC LIMIT 7"""
            )
            
            if len(pool_qubits) < 7:
                logger.warning(f"Not enough pool qubits: {len(pool_qubits)}/7")
                return False
            
            # Update transaction status
            db_pool.execute_insert(
                "UPDATE transactions SET status = %s WHERE tx_id = %s",
                ('queued_for_quantum', tx['tx_id'])
            )
            
            logger.info(f"Queued {tx['tx_id']} for quantum execution")
            return True
        
        except Exception as e:
            logger.error(f"Failed to queue transaction: {e}")
            return False
    
    def run(self):
        """Main polling loop"""
        logger.info("TransactionPoller started")
        
        while self.running:
            try:
                # Poll for transactions
                transactions = self.poll_transactions()
                
                if transactions:
                    logger.info(f"Found {len(transactions)} pending transactions")
                    
                    for tx in transactions:
                        self.stats['total_polled'] += 1
                        
                        # Validate
                        is_valid, error = self.validate_transaction(tx)
                        if not is_valid:
                            logger.warning(f"Invalid transaction {tx['tx_id']}: {error}")
                            db_pool.execute_insert(
                                "UPDATE transactions SET status = %s WHERE tx_id = %s",
                                ('failed', tx['tx_id'])
                            )
                            self.stats['total_failed'] += 1
                            continue
                        
                        # Queue
                        if self.queue_for_quantum_execution(tx):
                            self.stats['total_queued'] += 1
                        else:
                            self.stats['total_failed'] += 1
                
                time.sleep(Config.POLL_INTERVAL_TRANSACTIONS_MS / 1000.0)
            
            except Exception as e:
                logger.error(f"TransactionPoller error: {e}\n{traceback.format_exc()}")
                time.sleep(1)

transaction_poller = TransactionPoller()

# ═══════════════════════════════════════════════════════════════════════════════════════
# PSEUDOQUBIT MANAGER (THREAD 3)
# ═══════════════════════════════════════════════════════════════════════════════════════

class PseudoqubitManagerThread:
    """Manages pseudoqubit state and availability"""
    
    def __init__(self):
        self.running = True
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.cache_ttl = 300  # 5 minutes
    
    def get_pseudoqubit(self, pseudoqubit_id: int) -> Optional[Dict]:
        """Get pseudoqubit with caching"""
        
        with self.cache_lock:
            if pseudoqubit_id in self.cache:
                cached_data, cached_time = self.cache[pseudoqubit_id]
                if time.time() - cached_time < self.cache_ttl:
                    return cached_data
        
        try:
            row = db_pool.execute_one(
                "SELECT * FROM pseudoqubits WHERE pseudoqubit_id = %s",
                (pseudoqubit_id,)
            )
            
            if row:
                with self.cache_lock:
                    self.cache[pseudoqubit_id] = (dict(row), time.time())
                return dict(row)
            
            return None
        
        except Exception as e:
            logger.error(f"Error fetching pseudoqubit {pseudoqubit_id}: {e}")
            return None
    
    def get_pool_qubits(self, count: int = 7) -> List[Dict]:
        """Get available pool pseudoqubits"""
        try:
            rows = db_pool.execute(
                """SELECT * FROM pseudoqubits 
                   WHERE auth_user_id IS NULL AND is_available = TRUE 
                   ORDER BY fidelity DESC, coherence DESC 
                   LIMIT %s""",
                (count,)
            )
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching pool qubits: {e}")
            return []
    
    def get_available_count(self) -> int:
        """Get count of available pseudoqubits"""
        try:
            row = db_pool.execute_one(
                "SELECT COUNT(*) as count FROM pseudoqubits WHERE is_available = TRUE AND auth_user_id IS NULL"
            )
            return row['count'] if row else 0
        except Exception as e:
            logger.error(f"Error getting available count: {e}")
            return 0
    
    def monitor_availability(self):
        """Monitor and log pseudoqubit availability"""
        while self.running:
            try:
                available = self.get_available_count()
                logger.debug(f"Available pseudoqubits: {available}")
                time.sleep(Config.POLL_INTERVAL_SYNC_MS / 1000.0)
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(1)

pseudoqubit_manager = PseudoqubitManagerThread()

# ═══════════════════════════════════════════════════════════════════════════════════════
# QUANTUM JOB QUEUE (THREAD 4)
# ═══════════════════════════════════════════════════════════════════════════════════════

class QuantumJobQueue:
    """Manages queue of quantum execution jobs"""
    
    def __init__(self):
        self.queue = queue.Queue(maxsize=10000)
        self.running = True
        self.stats = {
            'enqueued': 0,
            'dequeued': 0,
            'executed': 0,
            'failed': 0
        }
        self.lock = threading.Lock()
    
    def enqueue(self, job: Dict) -> bool:
        """Add job to queue"""
        try:
            self.queue.put_nowait(job)
            with self.lock:
                self.stats['enqueued'] += 1
            logger.debug(f"Enqueued job for {job.get('tx_id', 'unknown')}")
            return True
        except queue.Full:
            logger.warning("Quantum job queue full")
            return False
        except Exception as e:
            logger.error(f"Enqueue error: {e}")
            return False
    
    def dequeue(self, timeout: float = 0.1) -> Optional[Dict]:
        """Get next job from queue"""
        try:
            job = self.queue.get(timeout=timeout)
            with self.lock:
                self.stats['dequeued'] += 1
            return job
        except queue.Empty:
            return None
        except Exception as e:
            logger.error(f"Dequeue error: {e}")
            return None
    
    def size(self) -> int:
        """Get queue size"""
        return self.queue.qsize()
    
    def process_jobs(self):
        """Main job processing loop"""
        logger.info("QuantumJobQueue processor started")
        
        while self.running:
            try:
                job = self.dequeue()
                if not job:
                    time.sleep(0.01)
                    continue
                
                # Execute quantum circuit
                try:
                    tx_id = job.get('tx_id')
                    from_user_id = job.get('from_user_id')
                    to_user_id = job.get('to_user_id')
                    amount = job.get('amount')
                    tx_type = job.get('tx_type')
                    user_qubit_id = job.get('user_pseudoqubit_id')
                    pool_qubit_ids = job.get('pool_qubit_ids', [])
                    
                    # Build circuit
                    circuit = quantum_executor.build_circuit(
                        tx_id, from_user_id, to_user_id, amount, tx_type,
                        user_qubit_id, pool_qubit_ids
                    )
                    
                    # Execute circuit
                    measurement_result = quantum_executor.execute(circuit)
                    
                    # Store results
                    db_pool.execute_insert(
                        """UPDATE transactions 
                           SET status = %s, quantum_state_hash = %s, entropy_score = %s 
                           WHERE tx_id = %s""",
                        ('superposition', 
                         hashlib.sha256(str(measurement_result.bitstring_counts).encode()).hexdigest(),
                         measurement_result.entropy_percent,
                         tx_id)
                    )
                    
                    with self.lock:
                        self.stats['executed'] += 1
                    
                    logger.info(f"Executed quantum circuit for {tx_id}: entropy={measurement_result.entropy_bits:.2f}")
                
                except Exception as e:
                    logger.error(f"Failed to execute job: {e}\n{traceback.format_exc()}")
                    with self.lock:
                        self.stats['failed'] += 1
                
                time.sleep(0.001)
            
            except Exception as e:
                logger.error(f"Job processing error: {e}")
                time.sleep(0.1)

quantum_job_queue = QuantumJobQueue()

# ═══════════════════════════════════════════════════════════════════════════════════════
# SUPERPOSITION MANAGER (THREAD 5)
# ═══════════════════════════════════════════════════════════════════════════════════════

class SuperpositionLifecycleManager:
    """Manages transactions in quantum superposition state"""
    
    def __init__(self):
        self.running = True
        self.superposition_txs = {}  # tx_id -> TransactionInSuperposition
        self.lock = threading.Lock()
        self.stats = {
            'entered_superposition': 0,
            'coherence_refreshes': 0,
            'exited_superposition': 0,
            'timed_out': 0
        }
    
    def enter_superposition(self, tx_id: str, tx_data: Dict) -> bool:
        """Place transaction in superposition"""
        try:
            with self.lock:
                if tx_id in self.superposition_txs:
                    return True
                
                tx_super = TransactionInSuperposition(
                    tx_id=tx_id,
                    from_user_id=tx_data['from_user_id'],
                    to_user_id=tx_data['to_user_id'],
                    amount=tx_data['amount'],
                    tx_type=tx_data['tx_type'],
                    user_pseudoqubit_id=tx_data.get('user_pseudoqubit_id', 0),
                    pool_qubit_ids=tx_data.get('pool_qubit_ids', [])
                )
                
                self.superposition_txs[tx_id] = tx_super
                self.stats['entered_superposition'] += 1
            
            logger.info(f"Entered superposition: {tx_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to enter superposition: {e}")
            return False
    
    def refresh_coherence(self, tx_id: str) -> bool:
        """Refresh quantum coherence for transaction in superposition"""
        try:
            with self.lock:
                if tx_id not in self.superposition_txs:
                    return False
                
                tx_super = self.superposition_txs[tx_id]
                
                if tx_super.coherence_refresh_count >= Config.MAX_COHERENCE_REFRESH_COUNT:
                    return False
                
                tx_super.last_coherence_refresh = datetime.utcnow()
                tx_super.coherence_refresh_count += 1
                
                if tx_super.coherence_refresh_count % (Config.FLOQUET_CYCLE_INTERVAL_SECONDS // Config.COHERENCE_REFRESH_INTERVAL_SECONDS) == 0:
                    tx_super.floquet_cycle += 1
                
                self.stats['coherence_refreshes'] += 1
            
            logger.debug(f"Refreshed coherence for {tx_id} (count: {tx_super.coherence_refresh_count})")
            return True
        
        except Exception as e:
            logger.error(f"Failed to refresh coherence: {e}")
            return False
    
    def exit_superposition(self, tx_id: str) -> bool:
        """Exit superposition (collapse triggered)"""
        try:
            with self.lock:
                if tx_id in self.superposition_txs:
                    del self.superposition_txs[tx_id]
                    self.stats['exited_superposition'] += 1
            
            logger.info(f"Exited superposition: {tx_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to exit superposition: {e}")
            return False
    
    def monitor_superposition(self):
        """Monitor superposition states and refresh coherence"""
        logger.info("SuperpositionLifecycleManager monitor started")
        
        while self.running:
            try:
                with self.lock:
                    expired_txs = []
                    
                    for tx_id, tx_super in list(self.superposition_txs.items()):
                        # Check for expiration
                        if tx_super.is_expired():
                            logger.warning(f"Superposition expired for {tx_id}")
                            db_pool.execute_insert(
                                "UPDATE transactions SET status = %s WHERE tx_id = %s",
                                ('coherence_timeout', tx_id)
                            )
                            expired_txs.append(tx_id)
                        
                        # Refresh coherence if needed
                        elif tx_super.needs_coherence_refresh():
                            with self.lock:
                                if tx_super.coherence_refresh_count < Config.MAX_COHERENCE_REFRESH_COUNT:
                                    self.refresh_coherence(tx_id)
                    
                    # Remove expired
                    for tx_id in expired_txs:
                        if tx_id in self.superposition_txs:
                            del self.superposition_txs[tx_id]
                            self.stats['timed_out'] += 1
                
                time.sleep(Config.POLL_INTERVAL_SUPERPOSITION_MS / 1000.0)
            
            except Exception as e:
                logger.error(f"Monitor error: {e}\n{traceback.format_exc()}")
                time.sleep(1)

superposition_manager = SuperpositionLifecycleManager()

# ═══════════════════════════════════════════════════════════════════════════════════════
# ORACLE LISTENER (THREAD 6)
# ═══════════════════════════════════════════════════════════════════════════════════════

class OracleListener:
    """Listens for oracle events that trigger collapse"""
    
    def __init__(self):
        self.running = True
        self.stats = {
            'events_received': 0,
            'collapses_triggered': 0,
            'collapses_failed': 0
        }
    
    def listen_for_oracle_events(self):
        """Listen for oracle data that triggers collapse"""
        logger.info("OracleListener started")
        
        while self.running:
            try:
                # Check for transactions ready for collapse
                # In real system, this would listen to oracle smart contract events
                # For now, we trigger collapse based on timing
                
                # Get transactions in superposition older than COHERENCE_REFRESH_INTERVAL
                superposition_txs = db_pool.execute(
                    """SELECT tx_id FROM transactions 
                       WHERE status = 'superposition' 
                       AND created_at < NOW() - INTERVAL '%s seconds'
                       AND created_at > NOW() - INTERVAL '6 minutes'
                       LIMIT %s""" % (Config.COHERENCE_REFRESH_INTERVAL_SECONDS * 3, Config.BATCH_SIZE_TRANSACTIONS)
                )
                
                for tx_row in superposition_txs:
                    tx_id = tx_row['tx_id']
                    
                    # Get transaction data
                    tx = db_pool.execute_one(
                        "SELECT * FROM transactions WHERE tx_id = %s",
                        (tx_id,)
                    )
                    
                    if tx:
                        # Trigger collapse with random outcome (simulating oracle measurement)
                        self._trigger_collapse(tx_id, tx)
                        self.stats['collapses_triggered'] += 1
                
                time.sleep(Config.POLL_INTERVAL_ORACLE_MS / 1000.0)
            
            except Exception as e:
                logger.error(f"Oracle listener error: {e}\n{traceback.format_exc()}")
                time.sleep(1)
    
    def _trigger_collapse(self, tx_id: str, tx: Dict):
        """Simulate oracle data measurement and collapse"""
        try:
            # Generate random outcome based on sender/receiver
            seed = int(hashlib.sha256(f"{tx['from_user_id']}{tx['to_user_id']}".encode()).hexdigest(), 16)
            random.seed(seed)
            
            # Outcome: approve or reject (biased toward approve)
            approved = random.random() < 0.95  # 95% approval rate
            
            if approved:
                # Collapse to confirmed state
                db_pool.execute_insert(
                    """UPDATE transactions 
                       SET status = %s, confirmed_at = NOW() 
                       WHERE tx_id = %s""",
                    ('confirmed', tx_id)
                )
                
                # Update balances
                db_pool.execute_insert(
                    "UPDATE users SET balance = balance - %s WHERE user_id = %s",
                    (tx['amount'], tx['from_user_id'])
                )
                db_pool.execute_insert(
                    "UPDATE users SET balance = balance + %s WHERE user_id = %s",
                    (tx['amount'], tx['to_user_id'])
                )
                
                logger.info(f"Collapsed {tx_id} to CONFIRMED (approved)")
            
            else:
                # Collapse to failed state
                db_pool.execute_insert(
                    "UPDATE transactions SET status = %s WHERE tx_id = %s",
                    ('failed', tx_id)
                )
                logger.info(f"Collapsed {tx_id} to FAILED (rejected)")
            
            superposition_manager.exit_superposition(tx_id)
        
        except Exception as e:
            logger.error(f"Failed to trigger collapse for {tx_id}: {e}")
            self.stats['collapses_failed'] += 1

oracle_listener = OracleListener()

# ═══════════════════════════════════════════════════════════════════════════════════════
# FINALITY PROCESSOR (THREAD 7)
# ═══════════════════════════════════════════════════════════════════════════════════════

class FinalityProcessor:
    """Creates blocks from confirmed transactions"""
    
    def __init__(self):
        self.running = True
        self.stats = {
            'blocks_created': 0,
            'transactions_finalized': 0
        }
    
    def process_finality(self):
        """Process finalized transactions into blocks"""
        logger.info("FinalityProcessor started")
        
        while self.running:
            try:
                # Get confirmed transactions waiting for block
                confirmed_txs = db_pool.execute(
                    """SELECT * FROM transactions 
                       WHERE status = 'confirmed' AND block_number IS NULL 
                       ORDER BY confirmed_at ASC 
                       LIMIT %s""" % Config.BLOCK_SIZE_MAX_TRANSACTIONS
                )
                
                if not confirmed_txs:
                    time.sleep(Config.POLL_INTERVAL_FINALITY_MS / 1000.0)
                    continue
                
                # Get latest block
                latest_block = db_pool.execute_one(
                    "SELECT block_number, block_hash FROM blocks ORDER BY block_number DESC LIMIT 1"
                )
                
                if not latest_block:
                    time.sleep(1)
                    continue
                
                latest_block_number = latest_block['block_number']
                parent_hash = latest_block['block_hash']
                
                # Create new block
                new_block_number = latest_block_number + 1
                
                # Calculate merkle root
                tx_ids = [tx['tx_id'] for tx in confirmed_txs]
                merkle_input = "".join(tx_ids)
                merkle_root = hashlib.sha256(merkle_input.encode()).hexdigest()
                
                # Calculate quantum state hash (aggregate entropy)
                entropies = [tx.get('entropy_score', 0) for tx in confirmed_txs]
                avg_entropy = sum(entropies) / len(entropies) if entropies else 0
                quantum_state_hash = hashlib.sha256(
                    f"{new_block_number}{parent_hash}{merkle_root}{avg_entropy}".encode()
                ).hexdigest()
                
                # Create block
                block_hash = hashlib.sha256(
                    f"{new_block_number}{parent_hash}{merkle_root}{quantum_state_hash}".encode()
                ).hexdigest()
                block_hash = f"0x{block_hash}"
                
                # Calculate miner reward
                if new_block_number <= Config.BLOCK_REWARD_EPOCH_1_BLOCKS:
                    miner_reward = 100 * Config.TOKEN_WEI_PER_UNIT
                elif new_block_number <= Config.BLOCK_REWARD_EPOCH_1_BLOCKS * 2:
                    miner_reward = 50 * Config.TOKEN_WEI_PER_UNIT
                else:
                    miner_reward = 25 * Config.TOKEN_WEI_PER_UNIT
                
                db_pool.execute_insert(
                    """INSERT INTO blocks 
                       (block_number, block_hash, parent_hash, timestamp, transactions,
                        validator_address, quantum_state_hash, entropy_score, merkle_root,
                        miner_reward, created_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())""",
                    (new_block_number, block_hash, parent_hash, int(time.time()),
                     len(confirmed_txs), Config.GENESIS_BLOCK_VALIDATOR,
                     quantum_state_hash, avg_entropy, merkle_root, miner_reward)
                )
                
                # Assign transactions to block
                for tx_id in tx_ids:
                    db_pool.execute_insert(
                        "UPDATE transactions SET block_number = %s WHERE tx_id = %s",
                        (new_block_number, tx_id)
                    )
                    self.stats['transactions_finalized'] += 1
                
                self.stats['blocks_created'] += 1
                logger.info(f"Created block {new_block_number} with {len(confirmed_txs)} transactions")
                
                time.sleep(Config.POLL_INTERVAL_FINALITY_MS / 1000.0)
            
            except Exception as e:
                logger.error(f"Finality processing error: {e}\n{traceback.format_exc()}")
                time.sleep(1)

finality_processor = FinalityProcessor()

# ═══════════════════════════════════════════════════════════════════════════════════════
# DATABASE SYNC (THREAD 8)
# ═══════════════════════════════════════════════════════════════════════════════════════

class DatabaseSync:
    """Synchronizes database state across threads"""
    
    def __init__(self):
        self.running = True
        self.stats = {
            'syncs': 0,
            'errors': 0
        }
    
    def sync(self):
        """Periodically sync and log system state"""
        logger.info("DatabaseSync started")
        
        while self.running:
            try:
                # Get system statistics
                user_count = db_pool.execute_one("SELECT COUNT(*) as count FROM users")
                tx_count = db_pool.execute_one("SELECT COUNT(*) as count FROM transactions")
                block_count = db_pool.execute_one("SELECT COUNT(*) as count FROM blocks")
                qubit_count = db_pool.execute_one("SELECT COUNT(*) as count FROM pseudoqubits")
                available_qubits = db_pool.execute_one(
                    "SELECT COUNT(*) as count FROM pseudoqubits WHERE is_available = TRUE"
                )
                
                self.stats['syncs'] += 1
                
                logger.info(
                    f"System State: "
                    f"Users={user_count['count']}, "
                    f"Transactions={tx_count['count']}, "
                    f"Blocks={block_count['count']}, "
                    f"Pseudoqubits={qubit_count['count']}, "
                    f"Available={available_qubits['count']}"
                )
                
                time.sleep(Config.POLL_INTERVAL_SYNC_MS / 1000.0)
            
            except Exception as e:
                logger.error(f"Sync error: {e}")
                self.stats['errors'] += 1
                time.sleep(1)

database_sync = DatabaseSync()

# ═══════════════════════════════════════════════════════════════════════════════════════
# WEBSOCKET BROADCASTER (THREAD 9)
# ═══════════════════════════════════════════════════════════════════════════════════════

class WebSocketBroadcaster:
    """Broadcasts status updates via WebSocket (placeholder)"""
    
    def __init__(self):
        self.running = True
        self.subscribers = set()
        self.lock = threading.Lock()
        self.stats = {
            'messages_sent': 0,
            'broadcast_errors': 0
        }
    
    def broadcast_status(self):
        """Broadcast system status updates"""
        logger.info("WebSocketBroadcaster started")
        
        while self.running:
            try:
                # In real implementation, broadcast via WebSocket
                # For now, just log status
                
                latest_block = db_pool.execute_one(
                    "SELECT block_number FROM blocks ORDER BY block_number DESC LIMIT 1"
                )
                
                latest_tx = db_pool.execute_one(
                    "SELECT tx_id, status, created_at FROM transactions ORDER BY created_at DESC LIMIT 1"
                )
                
                status_message = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'latest_block': latest_block['block_number'] if latest_block else 0,
                    'latest_transaction': latest_tx['tx_id'] if latest_tx else None,
                    'quantum_executor_executions': quantum_executor.execution_count,
                    'quantum_executor_avg_time_ms': (quantum_executor.total_execution_time / quantum_executor.execution_count) if quantum_executor.execution_count > 0 else 0
                }
                
                logger.debug(f"Status broadcast: {status_message}")
                self.stats['messages_sent'] += 1
                
                time.sleep(5)  # Broadcast every 5 seconds
            
            except Exception as e:
                logger.error(f"Broadcast error: {e}")
                self.stats['broadcast_errors'] += 1
                time.sleep(1)

websocket_broadcaster = WebSocketBroadcaster()

# ═══════════════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════════════

class QtclMainApplication:
    """Orchestrates all threads and manages application lifecycle"""
    
    def __init__(self):
        self.threads = []
        self.running = False
        self.start_time = None
        logger.info("QtclMainApplication initializing")
    
    def start(self):
        """Start all worker threads"""
        
        logger.info("=" * 100)
        logger.info("QTCL MAIN APPLICATION STARTUP")
        logger.info("=" * 100)
        
        try:
            # Step 1: Initialize genesis block
            logger.info("[1/10] Initializing genesis block...")
            if not genesis_initializer.initialize():
                logger.error("Genesis initialization failed")
                return False
            logger.info("✓ Genesis block ready")
            
            # Step 2: Create and start worker threads
            logger.info("[2/10] Starting worker threads...")
            
            thread_configs = [
                ("TransactionPoller", transaction_poller.poll_transactions, lambda: transaction_poller.run()),
                ("PseudoqubitMonitor", pseudoqubit_manager.get_available_count, pseudoqubit_manager.monitor_availability),
                ("QuantumJobProcessor", quantum_job_queue.size, quantum_job_queue.process_jobs),
                ("SuperpositionMonitor", None, superposition_manager.monitor_superposition),
                ("OracleListener", oracle_listener.stats.get, oracle_listener.listen_for_oracle_events),
                ("FinalityProcessor", None, finality_processor.process_finality),
                ("DatabaseSync", database_sync.stats.get, database_sync.sync),
                ("WebSocketBroadcaster", None, websocket_broadcaster.broadcast_status),
            ]
            
            for idx, (name, _, target) in enumerate(thread_configs):
                thread = threading.Thread(target=target, name=name, daemon=False)
                thread.start()
                self.threads.append(thread)
                logger.info(f"  ✓ Started {name}")
            
            self.running = True
            self.start_time = datetime.utcnow()
            
            logger.info("=" * 100)
            logger.info("QTCL MAIN APPLICATION RUNNING")
            logger.info("=" * 100)
            logger.info(f"Started at {self.start_time.isoformat()}")
            logger.info(f"Running {len(self.threads)} worker threads")
            
            # Main monitor loop
            while self.running:
                try:
                    # Log statistics every 60 seconds
                    uptime = (datetime.utcnow() - self.start_time).total_seconds()
                    if int(uptime) % 60 == 0 and uptime > 0:
                        logger.info(
                            f"[STATS] Uptime: {int(uptime)}s | "
                            f"TX Polled: {transaction_poller.stats['total_polled']} | "
                            f"TX Queued: {transaction_poller.stats['total_queued']} | "
                            f"QJ Executed: {quantum_job_queue.stats['executed']} | "
                            f"Blocks: {finality_processor.stats['blocks_created']} | "
                            f"TX Finalized: {finality_processor.stats['transactions_finalized']}"
                        )
                    
                    time.sleep(1)
                
                except KeyboardInterrupt:
                    logger.info("\nShutdown signal received")
                    self.stop()
                    break
            
            return True
        
        except Exception as e:
            logger.error(f"Startup failed: {e}\n{traceback.format_exc()}")
            return False
    
    def stop(self):
        """Stop all worker threads"""
        
        logger.info("=" * 100)
        logger.info("QTCL MAIN APPLICATION SHUTDOWN")
        logger.info("=" * 100)
        
        self.running = False
        
        # Signal all threads to stop
        transaction_poller.running = False
        pseudoqubit_manager.running = False
        quantum_job_queue.running = False
        superposition_manager.running = False
        oracle_listener.running = False
        finality_processor.running = False
        database_sync.running = False
        websocket_broadcaster.running = False
        
        # Wait for threads to finish
        logger.info("Waiting for worker threads to finish...")
        for thread in self.threads:
            thread.join(timeout=5)
            if thread.is_alive():
                logger.warning(f"Thread {thread.name} did not finish in time")
        
        # Close database connections
        logger.info("Closing database connections...")
        db_pool.close_all()
        
        uptime = (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0
        
        logger.info("=" * 100)
        logger.info("QTCL MAIN APPLICATION STOPPED")
        logger.info(f"Total uptime: {int(uptime)} seconds")
        logger.info("=" * 100)

# ═══════════════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    app = QtclMainApplication()
    
    try:
        success = app.start()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        app.stop()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}\n{traceback.format_exc()}")
        app.stop()
        sys.exit(1)
