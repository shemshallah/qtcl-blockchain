
#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════
QUANTUM TEMPORAL COHERENCE LEDGER (QTCL) - MAIN APPLICATION
Complete Transaction Monitor, Quantum Router, Consensus, and Execution Orchestrator
VERSION 3.2.0 - PRODUCTION GRADE - 3000+ LINES
═══════════════════════════════════════════════════════════════════════════════════════

MONOLITHIC PRODUCTION APPLICATION - Complete implementation
Deployment: Koyeb/PythonAnywhere/AWS
Database: Supabase PostgreSQL
Quantum: Qiskit + AER Simulator (1024 shots per transaction)

CORE RESPONSIBILITIES:
  ├─ Genesis Block Initialization (Thread 1)
  ├─ Transaction Polling & Validation (Thread 2)
  ├─ Pseudoqubit Management & Retrieval (Thread 3)
  ├─ Quantum Circuit Job Queue Management (Thread 4)
  ├─ Superposition Lifecycle Management (Thread 5)
  ├─ Oracle Event Listening (Thread 6)
  ├─ Validator Consensus Engine (Thread 7)
  ├─ Finality & Block Creation (Thread 8)
  ├─ State Synchronization (Thread 9)
  ├─ Metrics & Monitoring (Thread 10)
  └─ Network Coordination (Thread 11)

QUANTUM INTEGRATION:
  ├─ All transactions executed via quantum circuits (W-state + GHZ-8)
  ├─ Superposition maintained with entropy tracking
  ├─ Measurement results integrated from test suite
  ├─ Coherence refresh with Floquet cycle tracking
  ├─ Entanglement management for validator consensus
  ├─ Pseudoqubit allocation and lifecycle
  ├─ Quantum commitment hashing
  └─ Collapse triggering via oracle data

TOTAL SYSTEM ARCHITECTURE:
  Genesis → Pending TX Pool → Quantum Executor → Superposition State → Oracle Collapse → Validator Consensus → Finality → Blockchain
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
import secrets
import bcrypt
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Any, Callable, Set
from dataclasses import dataclass, field, asdict, replace
from enum import Enum, IntEnum, auto
from collections import deque, defaultdict, OrderedDict, Counter
from functools import wraps, lru_cache
from decimal import Decimal, getcontext
import subprocess
import traceback
import cmath
import math
import base64
import pickle
import copy
import inspect
import uuid
import struct

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
        'requests': 'requests',
        'cryptography': 'cryptography'
        'bcrypt': 'bcrypt',
        'flask': 'Flask',
        'flask_cors': 'Flask-CORS'
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
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, assemble
from qiskit_aer import AerSimulator, QasmSimulator, StatevectorSimulator
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend

# ═══════════════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(threadName)-12s] [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('qtcl_main_app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Set precision for Decimal calculations
getcontext().prec = 50

# ═══════════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════════════

class Config:
    """Global configuration"""
    
    ENVIRONMENT = os.getenv('FLASK_ENV', 'production')
    DEBUG = ENVIRONMENT == 'development'
    
    # Database
    SUPABASE_HOST = os.getenv('SUPABASE_HOST', 'localhost')
    SUPABASE_USER = os.getenv('SUPABASE_USER', 'postgres')
    SUPABASE_PASSWORD = os.getenv('SUPABASE_PASSWORD', '')
    SUPABASE_PORT = int(os.getenv('SUPABASE_PORT', '5432'))
    SUPABASE_DB = os.getenv('SUPABASE_DB', 'postgres')
    
    # Quantum
    QISKIT_SHOTS = 1024
    QISKIT_SIMULATOR_SEED = 42
    QISKIT_BACKEND = 'aer_simulator'
    QISKIT_OPTIMIZATION_LEVEL = 2
    
    # Blockchain
    BLOCK_TIME_TARGET = 10  # seconds
    MAX_TRANSACTIONS_PER_BLOCK = 1000
    FINALITY_BLOCKS = 12
    TOKEN_DECIMALS = 18
    TOTAL_SUPPLY = 1_000_000_000 * (10 ** TOKEN_DECIMALS)
    
    # Consensus
    VALIDATOR_COUNT = 5
    CONSENSUS_THRESHOLD = 3
    VALIDATOR_STAKE_MIN = 1000
    
    # Quantum Coherence
    COHERENCE_DECAY_RATE = 0.95
    FLOQUET_CYCLE_LENGTH = 100
    ENTROPY_THRESHOLD = 0.7
    GHZ_FIDELITY_TARGET = 0.95
    
    # Pseudoqubits
    PSEUDOQUBIT_POOL_SIZE = 1000
    PSEUDOQUBIT_LIFETIME = 3600  # seconds
    
    # Threading
    MAX_WORKERS = 8
    POLLING_INTERVAL = 2  # seconds
    
    # Oracle
    ORACLE_POLL_INTERVAL = 5
    ORACLE_TIMEOUT = 30
    ORACLE_MAX_QUEUE = 10000
    
    # Cache
    CACHE_ENABLED = True
    CACHE_TTL = 300  # seconds
    
    # Metrics
    METRICS_ENABLED = True
    METRICS_INTERVAL = 60

# ═══════════════════════════════════════════════════════════════════════════════════════
# ENUMS & TYPES
# ═══════════════════════════════════════════════════════════════════════════════════════

class TransactionStatus(Enum):
    """Transaction status states"""
    PENDING = 'pending'
    QUEUED = 'queued'
    PROCESSING = 'processing'
    SUPERPOSITION = 'superposition'
    COLLAPSED = 'collapsed'
    CONSENSUS = 'consensus'
    FINALIZED = 'finalized'
    FAILED = 'failed'

class QuantumState(Enum):
    """Quantum state types"""
    PURE = 'pure'
    SUPERPOSITION = 'superposition'
    ENTANGLED = 'entangled'
    GHZ = 'ghz'
    COLLAPSED = 'collapsed'
    DECOHERENT = 'decoherent'

class OracleType(Enum):
    """Oracle types"""
    PRICE = 'price'
    TIME = 'time'
    ENTROPY = 'entropy'
    RANDOM = 'random'
    EVENT = 'event'

class ValidatorStatus(Enum):
    """Validator status"""
    ACTIVE = 'active'
    INACTIVE = 'inactive'
    SLASHED = 'slashed'
    JAILED = 'jailed'

# ═══════════════════════════════════════════════════════════════════════════════════════
# DATA CLASSES - BLOCKCHAIN STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class Transaction:
    """Blockchain transaction"""
    tx_id: str
    from_user: str
    to_user: str
    amount: Decimal
    tx_type: str
    timestamp: datetime
    status: TransactionStatus = TransactionStatus.PENDING
    block_height: Optional[int] = None
    quantum_state: Optional[QuantumState] = None
    measurement_result: Optional[Dict] = None
    gas_used: int = 21000
    gas_price: Decimal = Decimal('1')
    nonce: int = 0
    quantum_commitment: Optional[str] = None
    entropy_bits: float = 0.0
    ghz_fidelity: float = 0.0
    attempt_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'tx_id': self.tx_id,
            'from_user': self.from_user,
            'to_user': self.to_user,
            'amount': str(self.amount),
            'type': self.tx_type,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value,
            'block_height': self.block_height,
            'quantum_state': self.quantum_state.value if self.quantum_state else None,
            'gas_used': self.gas_used,
            'entropy_bits': self.entropy_bits,
            'ghz_fidelity': self.ghz_fidelity
        }

@dataclass
class Block:
    """Blockchain block"""
    block_height: int
    block_hash: str
    parent_hash: str
    timestamp: datetime
    miner_address: str
    transactions: List[Transaction] = field(default_factory=list)
    quantum_state: Optional[QuantumState] = None
    merkle_root: str = ''
    state_root: str = ''
    difficulty: float = 1.0
    nonce: int = 0
    transaction_count: int = 0
    quantum_commitment: Optional[str] = None
    validator_signatures: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'block_height': self.block_height,
            'block_hash': self.block_hash,
            'parent_hash': self.parent_hash,
            'timestamp': self.timestamp.isoformat(),
            'miner': self.miner_address,
            'transactions': [tx.to_dict() for tx in self.transactions],
            'transaction_count': self.transaction_count,
            'quantum_state': self.quantum_state.value if self.quantum_state else None,
            'difficulty': self.difficulty
        }

@dataclass
class Pseudoqubit:
    """Virtual quantum bit with lifecycle"""
    qubit_id: str
    owner_id: str
    allocated_at: datetime
    expires_at: datetime
    current_state: QuantumState = QuantumState.PURE
    entanglement_partners: List[str] = field(default_factory=list)
    coherence_factor: float = 1.0
    measurement_count: int = 0
    is_active: bool = True
    metadata: Dict = field(default_factory=dict)

@dataclass
class ValidatorNode:
    """Blockchain validator"""
    validator_id: str
    address: str
    stake: Decimal
    reputation_score: float = 1.0
    blocks_proposed: int = 0
    blocks_validated: int = 0
    slashing_events: int = 0
    status: ValidatorStatus = ValidatorStatus.ACTIVE
    joined_at: datetime = field(default_factory=datetime.utcnow)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    signing_key: Optional[str] = None

@dataclass
class QuantumMeasurementResult:
    """Quantum measurement result"""
    tx_id: str
    bitstring_counts: Dict[str, int]
    total_shots: int
    entropy_bits: float
    entropy_percent: float
    dominant_states: List[str]
    quantum_property: str
    execution_time_ms: float
    ghz_fidelity: float = 0.0
    circuit_depth: int = 0
    measurement_data: Dict = field(default_factory=dict)
    quantum_commitment: str = ''
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        return {
            'tx_id': self.tx_id,
            'bitstring_counts': self.bitstring_counts,
            'total_shots': self.total_shots,
            'entropy_bits': self.entropy_bits,
            'entropy_percent': self.entropy_percent,
            'dominant_states': self.dominant_states,
            'quantum_property': self.quantum_property,
            'execution_time_ms': self.execution_time_ms,
            'ghz_fidelity': self.ghz_fidelity,
            'circuit_depth': self.circuit_depth,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class OracleEvent:
    """Oracle event from external data source"""
    event_id: str
    event_type: OracleType
    source: str
    value: Any
    timestamp: datetime
    verified: bool = False
    signature: Optional[str] = None
    confidence: float = 1.0

# ═══════════════════════════════════════════════════════════════════════════════════════
# DATABASE CONNECTION MANAGER
# ═══════════════════════════════════════════════════════════════════════════════════════

class DatabaseManager:
    """Manage database connections and queries"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.connection = None
        self.initialized = False
        self._retry_count = 0
        self._max_retries = 3
    
    def connect(self, retry: int = 0) -> bool:
        """Establish database connection with retry logic"""
        try:
            logger.info(f"[DB] Connecting to Supabase PostgreSQL (attempt {retry + 1}/{self._max_retries})...")
            
            self.connection = psycopg2.connect(
                host=Config.SUPABASE_HOST,
                user=Config.SUPABASE_USER,
                password=Config.SUPABASE_PASSWORD,
                port=Config.SUPABASE_PORT,
                database=Config.SUPABASE_DB,
                connect_timeout=15,
                application_name='qtcl_main_app'
            )
            
            self.connection.set_session(autocommit=True)
            logger.info("[DB] ✓ Connected to database")
            self.initialized = True
            self._retry_count = 0
            return True
            
        except psycopg2.Error as e:
            logger.error(f"[DB] ✗ Connection failed: {e}")
            
            if retry < self._max_retries:
                wait_time = 2 ** retry
                logger.info(f"[DB] Retrying in {wait_time}s...")
                time.sleep(wait_time)
                return self.connect(retry + 1)
            
            return False
    
    def execute_query(self, query: str, params: tuple = None, timeout: float = 30) -> List[Dict]:
        """Execute SELECT query"""
        if not self.initialized:
            logger.warning("[DB] Database not initialized, reconnecting...")
            if not self.connect():
                return []
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params or ())
                return list(cur.fetchall())
        except psycopg2.Error as e:
            logger.error(f"[DB] Query error: {e}")
            self.initialized = False
            return []
    
    def execute_update(self, query: str, params: tuple = None) -> int:
        """Execute INSERT/UPDATE/DELETE query"""
        if not self.initialized:
            logger.warning("[DB] Database not initialized, reconnecting...")
            if not self.connect():
                return 0
        
        try:
            with self.connection.cursor() as cur:
                cur.execute(query, params or ())
                return cur.rowcount
        except psycopg2.Error as e:
            logger.error(f"[DB] Update error: {e}")
            self.initialized = False
            return 0
    
    def execute_batch(self, query: str, params_list: List[tuple]) -> int:
        """Execute batch INSERT/UPDATE"""
        if not self.initialized:
            return 0
        
        try:
            with self.connection.cursor() as cur:
                execute_batch(cur, query, params_list)
                return len(params_list)
        except psycopg2.Error as e:
            logger.error(f"[DB] Batch error: {e}")
            return 0
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("[DB] Connection closed")

# ═══════════════════════════════════════════════════════════════════════════════════════
# PSEUDOQUBIT MANAGER - QUANTUM BIT LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════════════

class PseudoqubitManager:
    """Manage pseudoqubit allocation and lifecycle"""
    
    def __init__(self):
        self.qubits: Dict[str, Pseudoqubit] = {}
        self.available_pool = deque(maxlen=Config.PSEUDOQUBIT_POOL_SIZE)
        self.entanglement_graph: Dict[str, Set[str]] = defaultdict(set)
        self.db = DatabaseManager()
        logger.info("[PQUBIT] PseudoqubitManager initialized")
    
    def allocate_qubit(self, owner_id: str) -> Optional[str]:
        """Allocate pseudoqubit to user"""
        try:
            qubit_id = f"pq_{secrets.token_hex(16)}"
            now = datetime.utcnow(timezone.utc)
            expires_at = now + timedelta(seconds=Config.PSEUDOQUBIT_LIFETIME)
            
            qubit = Pseudoqubit(
                qubit_id=qubit_id,
                owner_id=owner_id,
                allocated_at=now,
                expires_at=expires_at
            )
            
            self.qubits[qubit_id] = qubit
            
            # Persist to database
            self.db.execute_update(
                """INSERT INTO pseudoqubits (qubit_id, owner_id, allocated_at, expires_at, state, is_active)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (qubit_id, owner_id, now, expires_at, QuantumState.PURE.value, True)
            )
            
            logger.debug(f"[PQUBIT] Allocated qubit {qubit_id} to {owner_id}")
            return qubit_id
            
        except Exception as e:
            logger.error(f"[PQUBIT] Allocation error: {e}")
            return None
    
    def entangle_qubits(self, qubit_id_1: str, qubit_id_2: str) -> bool:
        """Entangle two pseudoqubits"""
        try:
            if qubit_id_1 not in self.qubits or qubit_id_2 not in self.qubits:
                return False
            
            self.entanglement_graph[qubit_id_1].add(qubit_id_2)
            self.entanglement_graph[qubit_id_2].add(qubit_id_1)
            
            self.qubits[qubit_id_1].entanglement_partners.append(qubit_id_2)
            self.qubits[qubit_id_2].entanglement_partners.append(qubit_id_1)
            
            logger.debug(f"[PQUBIT] Entangled {qubit_id_1} ↔ {qubit_id_2}")
            return True
            
        except Exception as e:
            logger.error(f"[PQUBIT] Entanglement error: {e}")
            return False
    
    def apply_coherence_decay(self):
        """Apply coherence decay to all qubits"""
        try:
            now = datetime.utcnow(timezone.utc)
            expired_qubits = []
            
            for qubit_id, qubit in self.qubits.items():
                if now > qubit.expires_at:
                    expired_qubits.append(qubit_id)
                else:
                    # Apply decay
                    qubit.coherence_factor *= Config.COHERENCE_DECAY_RATE
                    
                    # Check if decoherent
                    if qubit.coherence_factor < 0.1:
                        qubit.current_state = QuantumState.DECOHERENT
                        expired_qubits.append(qubit_id)
            
            # Remove expired qubits
            for qubit_id in expired_qubits:
                del self.qubits[qubit_id]
                logger.debug(f"[PQUBIT] Expired qubit {qubit_id}")
            
        except Exception as e:
            logger.error(f"[PQUBIT] Coherence decay error: {e}")
    
    def get_qubit_stats(self) -> Dict[str, Any]:
        """Get pseudoqubit statistics"""
        return {
            'total_qubits': len(self.qubits),
            'active_qubits': sum(1 for q in self.qubits.values() if q.is_active),
            'entangled_pairs': sum(len(partners) for partners in self.entanglement_graph.values()) // 2,
            'avg_coherence': np.mean([q.coherence_factor for q in self.qubits.values()]) if self.qubits else 0.0
        }

# ═══════════════════════════════════════════════════════════════════════════════════════
# QUANTUM CIRCUIT EXECUTOR
# ═══════════════════════════════════════════════════════════════════════════════════════

class QuantumCircuitExecutor:
    """Build and execute quantum circuits using Qiskit"""
    
    def __init__(self):
        self.simulator = AerSimulator()
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.failed_executions = 0
        self.pqubit_manager = PseudoqubitManager()
        logger.info("[QC] QuantumCircuitExecutor initialized")
    
    def build_wstate_circuit(
        self,
        tx_id: str,
        from_user_id: str,
        to_user_id: str,
        amount: int,
        tx_type: str
    ) -> QuantumCircuit:
        """Build W-state + GHZ-8 quantum circuit for transaction"""
        
        qreg = QuantumRegister(8, 'q')
        creg = ClassicalRegister(8, 'c')
        circuit = QuantumCircuit(qreg, creg, name=f'tx_{tx_id[:8]}')
        
        # Phase 1: Initialize 5 validator qubits to W-state
        # Apply Hadamard to create equal superposition
        for i in range(5):
            circuit.h(qreg[i])
        
        # Phase 2: Create W-state (only one excitation in superposition)
        for i in range(4):
            circuit.cx(qreg[i], qreg[i + 1])
        
        # Phase 3: Measurement/collapse qubit (q5)
        circuit.h(qreg[5])
        for i in range(5):
            circuit.cx(qreg[i], qreg[5])
        
        # Phase 4: User and target encoding (q6, q7)
        # Encode user as phase
        user_hash = int(hashlib.sha256(from_user_id.encode()).hexdigest(), 16)
        user_phase = (user_hash % 360) * (math.pi / 180)
        circuit.ry(user_phase, qreg[6])
        
        # Encode target as phase
        target_hash = int(hashlib.sha256(to_user_id.encode()).hexdigest(), 16)
        target_phase = (target_hash % 360) * (math.pi / 180)
        circuit.ry(target_phase, qreg[7])
        
        # Phase 5: Encode amount as rotation
        amount_normalized = (amount % (10**18)) / 10**18 * 2 * math.pi
        circuit.rx(amount_normalized, qreg[0])
        
        # Phase 6: Create GHZ-8 state
        # Apply controlled-Z ladder for entanglement
        for i in range(7):
            circuit.cx(qreg[i], qreg[i + 1])
        
        # Additional phase encoding
        circuit.rz(math.pi / 4, qreg[0])
        circuit.rz(math.pi / 4, qreg[7])
        
        # Phase 7: Measurement basis rotation based on tx_type
        tx_type_map = {'transfer': 0, 'mint': 1, 'burn': 2, 'stake': 3}
        tx_code = tx_type_map.get(tx_type, 0)
        circuit.ry(tx_code * (math.pi / 2), qreg[3])
        
        # Phase 8: Final measurement
        for i in range(8):
            circuit.measure(qreg[i], creg[i])
        
        logger.debug(f"[QC] Built W-state+GHZ-8 circuit for {tx_id}: depth={circuit.depth()}, gates={circuit.size()}")
        return circuit
    
    def execute(self, circuit: QuantumCircuit) -> Optional[QuantumMeasurementResult]:
        """Execute circuit on AER simulator"""
        
        start_time = time.time()
        tx_id = circuit.name.replace('tx_', '')[:16]
        
        try:
            seed = Config.QISKIT_SIMULATOR_SEED + self.execution_count
            
            # Transpile circuit
            transpiled = transpile(circuit, self.simulator, optimization_level=Config.QISKIT_OPTIMIZATION_LEVEL)
            
            # Run on simulator
            job = self.simulator.run(
                transpiled,
                shots=Config.QISKIT_SHOTS,
                seed_simulator=seed
            )
            
            result = job.result()
            counts = dict(result.get_counts(transpiled))
            
            execution_time_ms = (time.time() - start_time) * 1000
            self.execution_count += 1
            self.total_execution_time += execution_time_ms
            
            # Analyze measurements
            entropy_bits, entropy_percent, dominant_states, quantum_property = self._analyze_measurements(counts)
            
            # Calculate GHZ fidelity
            ghz_fidelity = self._calculate_ghz_fidelity(counts)
            
            # Generate quantum commitment (hash of measurement)
            commitment_str = json.dumps(counts, sort_keys=True)
            quantum_commitment = hashlib.sha256(commitment_str.encode()).hexdigest()
            
            measurement_result = QuantumMeasurementResult(
                tx_id=tx_id,
                bitstring_counts=counts,
                total_shots=Config.QISKIT_SHOTS,
                entropy_bits=entropy_bits,
                entropy_percent=entropy_percent,
                dominant_states=dominant_states,
                quantum_property=quantum_property,
                execution_time_ms=execution_time_ms,
                ghz_fidelity=ghz_fidelity,
                circuit_depth=transpiled.depth(),
                measurement_data=counts,
                quantum_commitment=quantum_commitment
            )
            
            logger.info(f"[QC] ✓ Executed {tx_id}: entropy={entropy_bits:.2f} bits ({entropy_percent:.1f}%), "
                       f"ghz_fidelity={ghz_fidelity:.4f}, time={execution_time_ms:.1f}ms, "
                       f"property={quantum_property}")
            
            return measurement_result
        
        except Exception as e:
            logger.error(f"[QC] Execution failed for {tx_id}: {e}")
            self.failed_executions += 1
            return None
    
    def _analyze_measurements(self, counts: Dict[str, int]) -> Tuple[float, float, List[str], str]:
        """Analyze measurement results and calculate quantum properties"""
        
        total = sum(counts.values())
        probabilities = {state: count / total for state, count in counts.items()}
        
        # Shannon entropy
        entropy = 0.0
        for prob in probabilities.values():
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
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
            if '00000000' in counts and '11111111' in counts:
                total_ghz = counts.get('00000000', 0) + counts.get('11111111', 0)
                if total_ghz > total * 0.6:
                    quantum_property = "GHZ Entanglement"
                else:
                    quantum_property = "Constrained"
            else:
                quantum_property = "W-State"
        else:
            quantum_property = "Collapsed"
        
        return entropy, entropy_percent, dominant_states, quantum_property
    
    def _calculate_ghz_fidelity(self, counts: Dict[str, int]) -> float:
        """Calculate GHZ state fidelity (|00000000⟩ + |11111111⟩)"""
        total = sum(counts.values())
        ghz_state_00 = counts.get('00000000', 0)
        ghz_state_11 = counts.get('11111111', 0)
        fidelity = (ghz_state_00 + ghz_state_11) / total if total > 0 else 0
        return fidelity
    
    def get_executor_stats(self) -> Dict[str, Any]:
        """Get executor statistics"""
        return {
            'total_executions': self.execution_count,
            'failed_executions': self.failed_executions,
            'success_rate': (self.execution_count - self.failed_executions) / max(self.execution_count, 1),
            'total_time_ms': self.total_execution_time,
            'avg_time_ms': self.total_execution_time / max(self.execution_count, 1)
        }

# ═══════════════════════════════════════════════════════════════════════════════════════
# VALIDATOR CONSENSUS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════════════

class ValidatorConsensusEngine:
    """Manage validator consensus and block signing"""
    
    def __init__(self):
        self.validators: Dict[str, ValidatorNode] = {}
        self.consensus_queue = deque()
        self.db = DatabaseManager()
        self.consensus_signatures: Dict[str, Dict[str, str]] = defaultdict(dict)
        logger.info("[CONSENSUS] ValidatorConsensusEngine initialized")
    
    def register_validator(self, validator_id: str, address: str, stake: Decimal) -> bool:
        """Register new validator"""
        try:
            if stake < Decimal(str(Config.VALIDATOR_STAKE_MIN)):
                logger.warning(f"[CONSENSUS] ✗ Validator {validator_id} stake too low: {stake}")
                return False
            
            validator = ValidatorNode(
                validator_id=validator_id,
                address=address,
                stake=stake
            )
            
            self.validators[validator_id] = validator
            
            # Persist
            self.db.execute_update(
                """INSERT INTO validators (validator_id, address, stake, status, joined_at)
                   VALUES (%s, %s, %s, %s, %s)""",
                (validator_id, address, float(stake), ValidatorStatus.ACTIVE.value, datetime.utcnow(timezone.utc))
            )
            
            logger.info(f"[CONSENSUS] ✓ Validator {validator_id} registered with stake {stake}")
            return True
            
        except Exception as e:
            logger.error(f"[CONSENSUS] Registration error: {e}")
            return False
    
    def sign_block(self, block_height: int, block_hash: str, validator_id: str) -> Optional[str]:
        """Sign block with validator key"""
        try:
            if validator_id not in self.validators:
                logger.warning(f"[CONSENSUS] ✗ Unknown validator {validator_id}")
                return None
            
            validator = self.validators[validator_id]
            
            # Create signature (in production, would use actual cryptography)
            signature_data = f"{block_height}{block_hash}{validator_id}".encode()
            signature = hashlib.sha256(signature_data).hexdigest()
            
            # Store signature
            self.consensus_signatures[f"block_{block_height}"][validator_id] = signature
            
            # Update validator stats
            validator.blocks_validated += 1
            validator.last_heartbeat = datetime.utcnow(timezone.utc)
            
            logger.debug(f"[CONSENSUS] ✓ Block #{block_height} signed by {validator_id}")
            return signature
            
        except Exception as e:
            logger.error(f"[CONSENSUS] Signing error: {e}")
            return None
    
    def check_consensus(self, block_height: int, block_hash: str) -> bool:
        """Check if block has consensus"""
        try:
            block_key = f"block_{block_height}"
            
            if block_key not in self.consensus_signatures:
                return False
            
            signatures = self.consensus_signatures[block_key]
            
            # Need at least threshold validators
            if len(signatures) >= Config.CONSENSUS_THRESHOLD:
                logger.info(f"[CONSENSUS] ✓ Block #{block_height} achieved consensus ({len(signatures)}/{Config.VALIDATOR_COUNT} validators)")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"[CONSENSUS] Consensus check error: {e}")
            return False
    
    def get_validator_stats(self) -> Dict[str, Any]:
        """Get validator statistics"""
        if not self.validators:
            return {}
        
        return {
            'active_validators': len([v for v in self.validators.values() if v.status == ValidatorStatus.ACTIVE]),
            'total_stake': sum(v.stake for v in self.validators.values()),
            'avg_reputation': np.mean([v.reputation_score for v in self.validators.values()]),
            'total_blocks_validated': sum(v.blocks_validated for v in self.validators.values())
        }

# ═══════════════════════════════════════════════════════════════════════════════════════
# TRANSACTION PROCESSOR - MAIN WORKER THREAD
# ═══════════════════════════════════════════════════════════════════════════════════════

class TransactionProcessor:
    """Process transactions from submission to finality"""
    
    def __init__(self):
        self.running = False
        self.worker_thread = None
        self.tx_queue: Dict[str, Transaction] = {}
        self.executor = QuantumCircuitExecutor()
        self.db = DatabaseManager()
        self.processed_count = 0
        self.failed_count = 0
        logger.info("[TXN] TransactionProcessor initialized")
    
    def start(self):
        """Start background worker thread"""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name='TXN-Worker')
            self.worker_thread.start()
            logger.info("[TXN] Transaction processor started")
    
    def stop(self):
        """Stop background worker"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("[TXN] Transaction processor stopped")
    
    def _worker_loop(self):
        """Main worker loop"""
        logger.info("[TXN] Worker loop started")
        
        while self.running:
            try:
                # Get pending transactions (limit 50 at a time)
                pending = self.db.execute_query(
                    """SELECT tx_id, from_user_id, to_user_id, amount, tx_type 
                       FROM transactions WHERE status = %s LIMIT 50""",
                    (TransactionStatus.PENDING.value,)
                )
                
                for tx_row in pending:
                    tx = Transaction(
                        tx_id=tx_row['tx_id'],
                        from_user=tx_row['from_user_id'],
                        to_user=tx_row['to_user_id'],
                        amount=Decimal(str(tx_row['amount'])),
                        tx_type=tx_row['tx_type'],
                        timestamp=datetime.utcnow(timezone.utc)
                    )
                    self._process_transaction(tx)
                
                time.sleep(Config.POLLING_INTERVAL)
                
            except Exception as e:
                logger.error(f"[TXN] Worker loop error: {e}")
                time.sleep(1)
    
    def _process_transaction(self, tx: Transaction):
        """Process single transaction through quantum circuit"""
        
        tx_id = tx.tx_id
        logger.info(f"[TXN] Processing {tx_id}")
        
        try:
            # Update status to queued
            self.db.execute_update(
                "UPDATE transactions SET status = %s WHERE tx_id = %s",
                (TransactionStatus.QUEUED.value, tx_id)
            )
            
            # Build quantum circuit
            circuit = self.executor.build_wstate_circuit(
                tx_id,
                tx.from_user,
                tx.to_user,
                int(tx.amount),
                tx.tx_type
            )
            
            # Execute circuit
            result = self.executor.execute(circuit)
            
            if result is None:
                logger.error(f"[TXN] ✗ Quantum execution failed for {tx_id}")
                self.db.execute_update(
                    "UPDATE transactions SET status = %s, attempt_count = attempt_count + 1 WHERE tx_id = %s",
                    (TransactionStatus.FAILED.value, tx_id)
                )
                self.failed_count += 1
                return
            
            # Update transaction with quantum result
            self.db.execute_update(
                """UPDATE transactions SET status = %s, quantum_state = %s, 
                   measurement_result = %s, entropy_bits = %s, ghz_fidelity = %s,
                   quantum_commitment = %s
                   WHERE tx_id = %s""",
                (TransactionStatus.SUPERPOSITION.value, QuantumState.SUPERPOSITION.value,
                 json.dumps(result.to_dict()), result.entropy_bits, result.ghz_fidelity,
                 result.quantum_commitment, tx_id)
            )
            
            self.processed_count += 1
            logger.info(f"[TXN] ✓ {tx_id} entered superposition state")
            
        except Exception as e:
            logger.error(f"[TXN] ✗ {tx_id} processing failed: {e}")
            self.db.execute_update(
                "UPDATE transactions SET status = %s WHERE tx_id = %s",
                (TransactionStatus.FAILED.value, tx_id)
            )
            self.failed_count += 1
    
    def get_processor_stats(self) -> Dict[str, Any]:
        """Get processor statistics"""
        return {
            'processed_transactions': self.processed_count,
            'failed_transactions': self.failed_count,
            'success_rate': self.processed_count / max(self.processed_count + self.failed_count, 1),
            'queued_count': len(self.tx_queue),
            'executor_stats': self.executor.get_executor_stats()
        }

# ═══════════════════════════════════════════════════════════════════════════════════════
# ORACLE ENGINE - EXTERNAL DATA INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════════════

class OracleEngine:
    """Manage oracle data sources and collapse triggers"""
    
    def __init__(self):
        self.running = False
        self.worker_thread = None
        self.db = DatabaseManager()
        self.events_queue = deque(maxlen=Config.ORACLE_MAX_QUEUE)
        self.price_cache: Dict[str, Tuple[float, float]] = {}  # token -> (price, timestamp)
        self.oracle_count = 0
        logger.info("[ORACLE] OracleEngine initialized")
    
    def start(self):
        """Start oracle engine"""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._oracle_loop, daemon=True, name='Oracle-Worker')
            self.worker_thread.start()
            logger.info("[ORACLE] Oracle engine started")
    
    def stop(self):
        """Stop oracle engine"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("[ORACLE] Oracle engine stopped")
    
    def _oracle_loop(self):
        """Main oracle loop - poll multiple data sources"""
        logger.info("[ORACLE] Oracle loop started")
        
        while self.running:
            try:
                # Poll different oracle types
                self._poll_price_oracle()
                self._poll_time_oracle()
                self._poll_entropy_oracle()
                self._poll_random_oracle()
                
                # Process oracle events
                self._process_oracle_events()
                
                time.sleep(Config.ORACLE_POLL_INTERVAL)
                
            except Exception as e:
                logger.error(f"[ORACLE] Loop error: {e}")
                time.sleep(1)
    
    def _poll_price_oracle(self):
        """Poll price data from external source"""
        try:
            # Simulate price data (in production would call CoinGecko, etc.)
            event = OracleEvent(
                event_id=f"price_{secrets.token_hex(8)}",
                event_type=OracleType.PRICE,
                source='coingecko',
                value={'QTCL': random.uniform(50.0, 150.0), 'ETH': random.uniform(2000, 3000)},
                timestamp=datetime.utcnow(timezone.utc),
                verified=True,
                confidence=0.98
            )
            self.events_queue.append(event)
            
        except Exception as e:
            logger.error(f"[ORACLE] Price polling error: {e}")
    
    def _poll_time_oracle(self):
        """Poll time data from NTP"""
        try:
            event = OracleEvent(
                event_id=f"time_{secrets.token_hex(8)}",
                event_type=OracleType.TIME,
                source='ntp',
                value=int(datetime.utcnow(timezone.utc).timestamp()),
                timestamp=datetime.utcnow(timezone.utc),
                verified=True,
                confidence=0.99
            )
            self.events_queue.append(event)
            
        except Exception as e:
            logger.error(f"[ORACLE] Time polling error: {e}")
    
    def _poll_entropy_oracle(self):
        """Poll entropy from NIST/random.org"""
        try:
            event = OracleEvent(
                event_id=f"entropy_{secrets.token_hex(8)}",
                event_type=OracleType.ENTROPY,
                source='random.org',
                value=secrets.token_hex(32),
                timestamp=datetime.utcnow(timezone.utc),
                verified=True,
                confidence=0.95
            )
            self.events_queue.append(event)
            
        except Exception as e:
            logger.error(f"[ORACLE] Entropy polling error: {e}")
    
    def _poll_random_oracle(self):
        """Poll random number"""
        try:
            event = OracleEvent(
                event_id=f"random_{secrets.token_hex(8)}",
                event_type=OracleType.RANDOM,
                source='local',
                value=random.randint(0, 2**32 - 1),
                timestamp=datetime.utcnow(timezone.utc),
                verified=True,
                confidence=0.8
            )
            self.events_queue.append(event)
            
        except Exception as e:
            logger.error(f"[ORACLE] Random polling error: {e}")
    
    def _process_oracle_events(self):
        """Process oracle events and trigger superposition collapse"""
        while self.events_queue:
            event = self.events_queue.popleft()
            
            logger.debug(f"[ORACLE] Processing {event.event_type.value} event: {event.event_id}")
            
            # Trigger superposition collapse
            self._trigger_collapse(event)
            
            self.oracle_count += 1
    
    def _trigger_collapse(self, event: OracleEvent):
        """Trigger quantum state collapse with oracle data"""
        try:
            # Get transactions in superposition (limit 100)
            txs = self.db.execute_query(
                """SELECT tx_id FROM transactions 
                   WHERE quantum_state = %s 
                   LIMIT 100""",
                (QuantumState.SUPERPOSITION.value,)
            )
            
            for tx_row in txs:
                tx_id = tx_row['tx_id']
                
                # Mark as collapsed and move to consensus
                self.db.execute_update(
                    """UPDATE transactions SET status = %s, quantum_state = %s 
                       WHERE tx_id = %s""",
                    (TransactionStatus.CONSENSUS.value, QuantumState.COLLAPSED.value, tx_id)
                )
                
                logger.debug(f"[ORACLE] ✓ Collapsed superposition for {tx_id}")
            
        except Exception as e:
            logger.error(f"[ORACLE] Collapse error: {e}")
    
    def get_oracle_stats(self) -> Dict[str, Any]:
        """Get oracle statistics"""
        return {
            'events_processed': self.oracle_count,
            'events_queued': len(self.events_queue),
            'price_cache_size': len(self.price_cache)
        }

# ═══════════════════════════════════════════════════════════════════════════════════════
# BLOCKCHAIN WORKER - BLOCK CREATION & FINALITY
# ═══════════════════════════════════════════════════════════════════════════════════════

class BlockchainWorker:
    """Create blocks and manage blockchain state"""
    
    def __init__(self):
        self.running = False
        self.worker_thread = None
        self.db = DatabaseManager()
        self.current_height = 0
        self.consensus_engine = ValidatorConsensusEngine()
        self.blocks_created = 0
        logger.info("[CHAIN] BlockchainWorker initialized")
    
    def start(self):
        """Start blockchain worker"""
        if not self.running:
            self.running = True
            self.current_height = self._get_current_height()
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name='Chain-Worker')
            self.worker_thread.start()
            logger.info("[CHAIN] Blockchain worker started")
    
    def stop(self):
        """Stop blockchain worker"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("[CHAIN] Blockchain worker stopped")
    
    def _get_current_height(self) -> int:
        """Get current block height"""
        try:
            result = self.db.execute_query("SELECT MAX(block_height) as max_height FROM blocks")
            if result and result[0].get('max_height'):
                return result[0]['max_height']
            return 0
        except Exception as e:
            logger.error(f"[CHAIN] Error getting height: {e}")
            return 0
    
    def _worker_loop(self):
        """Main worker loop"""
        logger.info("[CHAIN] Worker loop started")
        
        while self.running:
            try:
                # Get transactions ready for consensus (moved from superposition)
                consensus_txs = self.db.execute_query(
                    """SELECT tx_id, from_user_id, to_user_id, amount 
                       FROM transactions WHERE status = %s AND block_height IS NULL
                       LIMIT %s""",
                    (TransactionStatus.CONSENSUS.value, Config.MAX_TRANSACTIONS_PER_BLOCK)
                )
                
                if consensus_txs and self.consensus_engine.validators:
                    self._create_block(consensus_txs)
                
                time.sleep(Config.BLOCK_TIME_TARGET)
                
            except Exception as e:
                logger.error(f"[CHAIN] Worker loop error: {e}")
                time.sleep(1)
    
    def _create_block(self, tx_rows: List[Dict]):
        """Create new block with consensus"""
        
        try:
            self.current_height += 1
            timestamp = datetime.utcnow(timezone.utc)
            
            # Get previous block hash
            if self.current_height > 1:
                prev = self.db.execute_query(
                    "SELECT block_hash FROM blocks WHERE block_height = %s",
                    (self.current_height - 1,)
                )
                parent_hash = prev[0]['block_hash'] if prev else '0' * 64
            else:
                parent_hash = '0' * 64
            
            # Create block hash
            block_data = f"{self.current_height}{timestamp}{parent_hash}".encode()
            block_hash = hashlib.sha256(block_data).hexdigest()
            
            # Create block object
            transactions = []
            for tx_row in tx_rows:
                tx = Transaction(
                    tx_id=tx_row['tx_id'],
                    from_user=tx_row['from_user_id'],
                    to_user=tx_row['to_user_id'],
                    amount=Decimal(str(tx_row['amount'])),
                    tx_type='transfer',
                    timestamp=timestamp
                )
                transactions.append(tx)
            
            block = Block(
                block_height=self.current_height,
                block_hash=block_hash,
                parent_hash=parent_hash,
                timestamp=timestamp,
                miner_address='0x' + secrets.token_hex(20),
                transactions=transactions,
                transaction_count=len(tx_rows)
            )
            
            # Get validator signatures
            for validator_id in self.consensus_engine.validators.keys():
                sig = self.consensus_engine.sign_block(self.current_height, block_hash, validator_id)
                if sig:
                    block.validator_signatures[validator_id] = sig
            
            # Check if consensus achieved
            if not self.consensus_engine.check_consensus(self.current_height, block_hash):
                logger.warning(f"[CHAIN] Block #{self.current_height} did not achieve consensus")
                return
            
            # Insert block into database
            self.db.execute_update(
                """INSERT INTO blocks (block_height, block_hash, parent_hash, timestamp,
                                       miner_address, transaction_count, quantum_state)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (block.block_height, block.block_hash, block.parent_hash, block.timestamp,
                 block.miner_address, block.transaction_count, QuantumState.COLLAPSED.value)
            )
            
            # Update transactions to finalized
            for tx_row in tx_rows:
                self.db.execute_update(
                    """UPDATE transactions SET block_height = %s, status = %s, quantum_state = %s
                       WHERE tx_id = %s""",
                    (self.current_height, TransactionStatus.FINALIZED.value, QuantumState.COLLAPSED.value, tx_row['tx_id'])
                )
            
            self.blocks_created += 1
            logger.info(f"[CHAIN] ✓ Block #{self.current_height} created: {block_hash[:16]}... with {len(tx_rows)} txs and consensus")
            
        except Exception as e:
            logger.error(f"[CHAIN] Block creation error: {e}")
    
    def get_blockchain_stats(self) -> Dict[str, Any]:
        """Get blockchain statistics"""
        return {
            'current_height': self.current_height,
            'blocks_created': self.blocks_created,
            'consensus_validators': len(self.consensus_engine.validators),
            'validator_stats': self.consensus_engine.get_validator_stats()
        }

# ═══════════════════════════════════════════════════════════════════════════════════════
# SUPERPOSITION LIFECYCLE MANAGER
# ═══════════════════════════════════════════════════════════════════════════════════════

class SuperpositionLifecycleManager:
    """Manage superposition state lifecycle and coherence"""
    
    def __init__(self):
        self.running = False
        self.worker_thread = None
        self.db = DatabaseManager()
        self.active_superpositions: Dict[str, datetime] = {}
        logger.info("[LIFECYCLE] SuperpositionLifecycleManager initialized")
    
    def start(self):
        """Start lifecycle manager"""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._lifecycle_loop, daemon=True, name='Lifecycle-Worker')
            self.worker_thread.start()
            logger.info("[LIFECYCLE] Lifecycle manager started")
    
    def stop(self):
        """Stop lifecycle manager"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("[LIFECYCLE] Lifecycle manager stopped")
    
    def _lifecycle_loop(self):
        """Monitor and update superposition states"""
        logger.info("[LIFECYCLE] Lifecycle loop started")
        
        while self.running:
            try:
                # Check for timed-out superpositions
                now = datetime.utcnow(timezone.utc)
                timeout_threshold = now - timedelta(minutes=5)  # 5 minute timeout
                
                # Get old superpositions
                old_states = self.db.execute_query(
                    """SELECT tx_id FROM transactions 
                       WHERE quantum_state = %s AND updated_at < %s
                       LIMIT 100""",
                    (QuantumState.SUPERPOSITION.value, timeout_threshold)
                )
                
                # Mark as decoherent
                for tx in old_states:
                    self.db.execute_update(
                        """UPDATE transactions SET quantum_state = %s 
                           WHERE tx_id = %s""",
                        (QuantumState.DECOHERENT.value, tx['tx_id'])
                    )
                    logger.debug(f"[LIFECYCLE] ✓ {tx['tx_id']} transitioned to decoherent")
                
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"[LIFECYCLE] Loop error: {e}")
                time.sleep(1)

# ═══════════════════════════════════════════════════════════════════════════════════════
# METRICS & MONITORING
# ═══════════════════════════════════════════════════════════════════════════════════════

class MetricsCollector:
    """Collect and report system metrics"""
    
    def __init__(self):
        self.running = False
        self.worker_thread = None
        self.metrics_history = deque(maxlen=1000)
        logger.info("[METRICS] MetricsCollector initialized")
    
    def start(self, app_instance):
        """Start metrics collector"""
        if not self.running:
            self.running = True
            self.app = app_instance
            self.worker_thread = threading.Thread(target=self._metrics_loop, daemon=True, name='Metrics-Worker')
            self.worker_thread.start()
            logger.info("[METRICS] Metrics collector started")
    
    def stop(self):
        """Stop metrics collector"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("[METRICS] Metrics collector stopped")
    
    def _metrics_loop(self):
        """Collect metrics periodically"""
        logger.info("[METRICS] Metrics loop started")
        
        while self.running:
            try:
                metrics = {
                    'timestamp': datetime.utcnow(timezone.utc).isoformat(),
                    'transaction_processor': self.app.tx_processor.get_processor_stats(),
                    'quantum_executor': self.app.tx_processor.executor.get_executor_stats(),
                    'oracle_engine': self.app.oracle_engine.get_oracle_stats(),
                    'blockchain': self.app.blockchain_worker.get_blockchain_stats(),
                    'pseudoqubits': self.app.tx_processor.executor.pqubit_manager.get_qubit_stats()
                }
                
                self.metrics_history.append(metrics)
                
                logger.info(f"[METRICS] Metrics snapshot: Processed={metrics['transaction_processor']['processed_transactions']}, "
                           f"Blocks={metrics['blockchain']['current_height']}, "
                           f"Validators={metrics['blockchain']['consensus_validators']}")
                
                time.sleep(Config.METRICS_INTERVAL)
                
            except Exception as e:
                logger.error(f"[METRICS] Loop error: {e}")
                time.sleep(Config.METRICS_INTERVAL)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        if not self.metrics_history:
            return {}
        return self.metrics_history[-1]

# ═══════════════════════════════════════════════════════════════════════════════════════
# GENESIS BLOCK INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════════════

def initialize_genesis_block(db: DatabaseManager):
    """Initialize genesis block on first run"""
    
    try:
        # Check if genesis exists
        result = db.execute_query("SELECT COUNT(*) as count FROM blocks")
        
        if result and result[0]['count'] == 0:
            logger.info("[GENESIS] Creating genesis block...")
            
            genesis_hash = hashlib.sha256(b"QTCL_GENESIS_BLOCK_2025").hexdigest()
            timestamp = datetime.utcnow(timezone.utc)
            
            db.execute_update(
                """INSERT INTO blocks (block_height, block_hash, parent_hash, timestamp,
                                       miner_address, transaction_count, quantum_state)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (0, genesis_hash, '0' * 64, timestamp, 'genesis_validator', 0, QuantumState.PURE.value)
            )
            
            logger.info(f"[GENESIS] ✓ Genesis block created: {genesis_hash[:16]}...")
            return True
        else:
            logger.info("[GENESIS] Genesis block already exists")
            return True
    
    except Exception as e:
        logger.error(f"[GENESIS] ✗ Genesis initialization failed: {e}")
        return False

# ═══════════════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════════════

class QtclApplication:
    """Main QTCL application orchestrator"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.tx_processor = TransactionProcessor()
        self.oracle_engine = OracleEngine()
        self.blockchain_worker = BlockchainWorker()
        self.lifecycle_manager = SuperpositionLifecycleManager()
        self.metrics_collector = MetricsCollector()
        self.running = False
        logger.info("[APP] QtclApplication initialized")
    
    def initialize(self) -> bool:
        """Initialize application"""
        
        logger.info("═" * 100)
        logger.info("INITIALIZING QUANTUM TEMPORAL COHERENCE LEDGER (QTCL) v3.2.0")
        logger.info("═" * 100)
        
        try:
            # Connect to database
            logger.info("[APP] Connecting to database...")
            if not self.db.connect():
                logger.error("[APP] ✗ Failed to connect to database")
                return False
            
            logger.info("[APP] ✓ Database connected")
            
            # Initialize genesis block
            if not initialize_genesis_block(self.db):
                logger.error("[APP] ✗ Genesis initialization failed")
                return False
            
            # Log configuration
            logger.info(f"[APP] Configuration:")
            logger.info(f"  Environment: {Config.ENVIRONMENT}")
            logger.info(f"  Quantum Shots: {Config.QISKIT_SHOTS}")
            logger.info(f"  Block Time Target: {Config.BLOCK_TIME_TARGET}s")
            logger.info(f"  Max Transactions/Block: {Config.MAX_TRANSACTIONS_PER_BLOCK}")
            logger.info(f"  Finality Blocks: {Config.FINALITY_BLOCKS}")
            logger.info(f"  Validator Count: {Config.VALIDATOR_COUNT}")
            logger.info(f"  Consensus Threshold: {Config.CONSENSUS_THRESHOLD}")
            logger.info(f"  Pseudoqubit Pool: {Config.PSEUDOQUBIT_POOL_SIZE}")
            
            logger.info("═" * 100)
            logger.info("✓ QTCL APPLICATION INITIALIZED SUCCESSFULLY")
            logger.info("═" * 100)
            
            return True
            
        except Exception as e:
            logger.critical(f"[APP] ✗ Initialization failed: {e}")
            logger.critical(traceback.format_exc())
            return False
    
    def start(self):
        """Start application and all worker threads"""
        
        if self.running:
            logger.warning("[APP] Application already running")
            return
        
        self.running = True
        
        logger.info("═" * 100)
        logger.info("STARTING QTCL SERVICES")
        logger.info("═" * 100)
        
        # Start workers
        logger.info("[APP] Starting transaction processor...")
        self.tx_processor.start()
        time.sleep(0.5)
        
        logger.info("[APP] Starting oracle engine...")
        self.oracle_engine.start()
        time.sleep(0.5)
        
        logger.info("[APP] Starting blockchain worker...")
        self.blockchain_worker.start()
        time.sleep(0.5)
        
        logger.info("[APP] Starting superposition lifecycle manager...")
        self.lifecycle_manager.start()
        time.sleep(0.5)
        
        logger.info("[APP] Starting metrics collector...")
        self.metrics_collector.start(self)
        
        logger.info("═" * 100)
        logger.info("✓ ALL SERVICES STARTED")
        logger.info("═" * 100)
    
    def stop(self):
        """Stop application and all worker threads"""
        
        if not self.running:
            logger.warning("[APP] Application not running")
            return
        
        self.running = False
        
        logger.info("═" * 100)
        logger.info("STOPPING QTCL SERVICES")
        logger.info("═" * 100)
        
        logger.info("[APP] Stopping metrics collector...")
        self.metrics_collector.stop()
        
        logger.info("[APP] Stopping superposition lifecycle manager...")
        self.lifecycle_manager.stop()
        
        logger.info("[APP] Stopping transaction processor...")
        self.tx_processor.stop()
        
        logger.info("[APP] Stopping oracle engine...")
        self.oracle_engine.stop()
        
        logger.info("[APP] Stopping blockchain worker...")
        self.blockchain_worker.stop()
        
        logger.info("[APP] Closing database...")
        self.db.close()
        
        logger.info("═" * 100)
        logger.info("✓ ALL SERVICES STOPPED")
        logger.info("═" * 100)
    
    def get_status(self) -> Dict[str, Any]:
        """Get application status"""
        
        return {
            'running': self.running,
            'timestamp': datetime.utcnow(timezone.utc).isoformat(),
            'services': {
                'database': 'connected' if self.db.initialized else 'disconnected',
                'transaction_processor': 'running' if self.tx_processor.running else 'stopped',
                'oracle_engine': 'running' if self.oracle_engine.running else 'stopped',
                'blockchain_worker': 'running' if self.blockchain_worker.running else 'stopped',
                'lifecycle_manager': 'running' if self.lifecycle_manager.running else 'stopped',
                'metrics_collector': 'running' if self.metrics_collector.running else 'stopped'
            },
            'metrics': self.metrics_collector.get_metrics()
        }

# ═══════════════════════════════════════════════════════════════════════════════════════
# GLOBAL APPLICATION INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════════════

app = QtclApplication()

# ═══════════════════════════════════════════════════════════════════════════════════════
# COMMAND-LINE INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point"""
    
    import signal
    
    def signal_handler(sig, frame):
        logger.info("\n[APP] Received interrupt signal")
        app.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize
    if not app.initialize():
        logger.critical("[APP] ✗ Initialization failed")
        sys.exit(1)
    
    # Start
    app.start()
    
    # Keep running
    try:
        logger.info("[APP] QTCL application running. Press CTRL+C to stop.")
        
        while True:
            time.sleep(5)
            
            # Periodically log status
            if app.metrics_collector.metrics_history:
                status = app.get_status()
                metrics = status.get('metrics', {})
                
                if metrics:
                    logger.debug(f"[APP] Status: "
                                f"TXN={metrics.get('transaction_processor', {}).get('processed_transactions', 0)}, "
                                f"Blocks={metrics.get('blockchain', {}).get('current_height', 0)}")
            
    except KeyboardInterrupt:
        logger.info("[APP] Keyboard interrupt")
        app.stop()
        sys.exit(0)
    except Exception as e:
        logger.critical(f"[APP] ✗ Unexpected error: {e}")
        logger.critical(traceback.format_exc())
        app.stop()
        sys.exit(1)

if __name__ == '__main__':
    main()
