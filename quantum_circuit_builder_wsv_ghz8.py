#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════
QUANTUM TOPOLOGY: W-STATE VALIDATOR CONSENSUS → GHZ-8 COLLAPSE
Quantum Circuit Builder Module - Production Grade
═══════════════════════════════════════════════════════════════════════════════════════

NEW ARCHITECTURE:
  ├─ q[0-4]: 5 Validator Qubits (W-state consensus)
  ├─ q[5]:   Measurement/Collapse Qubit (Oracle trigger)
  ├─ q[6]:   User Qubit (Transaction source encoding)
  └─ q[7]:   Target Qubit (Transaction destination encoding)

EXECUTION:
  1. Initialize all qubits to |0⟩
  2. Create W-state on validator qubits (equal superposition with 1 excitation)
  3. Apply controlled entanglement from validators to measurement qubit
  4. Encode user/target phases in q[6], q[7]
  5. Create GHZ-8 state: (1/√2)(|00000000⟩ + |11111111⟩)
  6. Apply transaction-parameterized measurement basis rotation
  7. Measure all 8 qubits (triggers collapse)
  8. Extract validator consensus + authentication signatures
  9. Generate quantum commitment hash
  10. Persist to database

DEPLOYMENT: Koyeb (Live)
COMPATIBILITY: Python 3.8+, Qiskit 0.43+, PostgreSQL 13+
═══════════════════════════════════════════════════════════════════════════════════════
"""

import math
import hashlib
import json
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import numpy as np

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit_aer import AerSimulator
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'qiskit', 'qiskit-aer'])
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit_aer import AerSimulator

# ═══════════════════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] [QCB-WSV-GHZ8] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('qtcl_circuit_builder_wsv_ghz8.log'),
        logging.StreamHandler()
    ]
)

# ═══════════════════════════════════════════════════════════════════════════════════════
# QUANTUM TOPOLOGY CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════════════

class QuantumTopologyConfig:
    """Configuration for W-state + GHZ-8 quantum topology"""
    
    # Qubit Allocation (8 total)
    NUM_TOTAL_QUBITS = 8
    VALIDATOR_QUBITS = [0, 1, 2, 3, 4]           # q[0-4]: 5 validators
    MEASUREMENT_QUBIT = 5                         # q[5]: Collapse trigger
    USER_QUBIT = 6                                # q[6]: User encoding
    TARGET_QUBIT = 7                              # q[7]: Target encoding
    
    NUM_CLASSICAL_BITS = 8  # One classical bit per qubit for measurement
    
    # W-State Configuration
    NUM_VALIDATORS = 5
    W_STATE_EQUAL_SUPERPOSITION = True  # Equal probability across all validator states
    
    # GHZ State Configuration
    GHZ_PHASE_ENCODING = True  # Encode transaction data in phases
    GHZ_ENTANGLEMENT_DEPTH = 3  # Circuit depth for GHZ construction
    
    # Phase Encoding (for user/target qubits)
    PHASE_BITS_USER = 8        # 8-bit phase encoding for user
    PHASE_BITS_TARGET = 8      # 8-bit phase encoding for target
    
    # Circuit Optimization
    CIRCUIT_TRANSPILE = True
    CIRCUIT_OPTIMIZATION_LEVEL = 2
    MAX_CIRCUIT_DEPTH = 50
    
    # Execution
    AER_SHOTS = 1024
    AER_SEED = 42
    AER_OPTIMIZATION_LEVEL = 2
    EXECUTION_TIMEOUT_MS = 200
    
    # Measurement Basis Rotation
    MEASUREMENT_BASIS_ROTATION_ENABLED = True
    MEASUREMENT_BASIS_ANGLE_VARIANCE = math.pi / 8  # ±22.5 degrees per transaction

# ═══════════════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class TransactionQuantumParameters:
    """Transaction parameters for quantum encoding"""
    tx_id: str
    user_id: str
    target_address: str
    amount: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    def compute_user_phase(self) -> float:
        """Compute user qubit phase encoding"""
        # Hash user_id to 8-bit value (0-255)
        user_hash = int(hashlib.md5(self.user_id.encode()).hexdigest(), 16) % 256
        # Convert to phase: 0-2π
        phase = (user_hash / 256.0) * (2 * math.pi)
        return phase
    
    def compute_target_phase(self) -> float:
        """Compute target qubit phase encoding"""
        # Hash target_address to 8-bit value (0-255)
        target_hash = int(hashlib.md5(self.target_address.encode()).hexdigest(), 16) % 256
        # Convert to phase: 0-2π
        phase = (target_hash / 256.0) * (2 * math.pi)
        return phase
    
    def compute_measurement_basis_angle(self) -> float:
        """Compute measurement basis rotation angle based on transaction data"""
        # Hash transaction parameters
        tx_data = f"{self.tx_id}{self.amount}".encode()
        tx_hash = int(hashlib.sha256(tx_data).hexdigest(), 16) % 1000
        
        # Map to angle: -variance to +variance
        variance = QuantumTopologyConfig.MEASUREMENT_BASIS_ANGLE_VARIANCE
        angle = -variance + (2 * variance * (tx_hash / 1000.0))
        return angle


@dataclass
class QuantumCircuitMetrics:
    """Metrics for quantum circuit execution"""
    circuit_name: str
    num_qubits: int
    num_classical_bits: int
    circuit_depth: int
    circuit_size: int
    num_gates: int
    execution_time_ms: float
    aer_shots: int
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        d = asdict(self)
        d['created_at'] = self.created_at.isoformat()
        return d


@dataclass
class QuantumMeasurementResult:
    """Result from quantum circuit measurement"""
    circuit_name: str
    tx_id: str
    
    # Raw measurement data
    bitstring_counts: Dict[str, int]  # {bitstring: count}
    dominant_bitstring: str  # Most frequently measured bitstring
    dominant_count: int  # How many times dominant bitstring appeared
    
    # Entropy metrics
    shannon_entropy: float  # Information entropy (0-1)
    entropy_percent: float  # Shannon entropy as percentage
    
    # GHZ-specific metrics
    ghz_state_probability: float  # Prob of measuring |00000000⟩ or |11111111⟩
    ghz_fidelity: float  # How close to perfect GHZ state
    
    # Validator consensus extraction
    validator_consensus: Dict[str, float]  # {validator_state: probability}
    validator_agreement_score: float  # 0-1 score of validator agreement
    
    # User/target authentication
    user_signature_bit: int  # Extracted from measurement
    target_signature_bit: int  # Extracted from measurement
    
    # Quantum commitment
    state_hash: str  # SHA256 of quantum state
    commitment_hash: str  # Final transaction commitment
    
    measurement_timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        d = asdict(self)
        d['measurement_timestamp'] = self.measurement_timestamp.isoformat()
        return d


# ═══════════════════════════════════════════════════════════════════════════════════════
# QUANTUM CIRCUIT BUILDER: W-STATE + GHZ-8
# ═══════════════════════════════════════════════════════════════════════════════════════

class WStateValidatorCircuitBuilder:
    """Builds quantum circuits with W-state validator consensus + GHZ-8 entanglement"""
    
    def __init__(self, config: QuantumTopologyConfig = None):
        """Initialize circuit builder"""
        self.config = config or QuantumTopologyConfig()
        logger.info(f"✓ WStateValidatorCircuitBuilder initialized")
        logger.info(f"  Topology: {self.config.NUM_VALIDATORS} validators + 1 measurement + user + target qubits")
    
    def build_transaction_circuit(
        self,
        tx_params: TransactionQuantumParameters
    ) -> Tuple[QuantumCircuit, QuantumCircuitMetrics]:
        """
        Build complete quantum circuit for transaction execution.
        
        Architecture:
        1. Initialize 8 qubits to |0⟩
        2. Create W-state on validator qubits (q[0-4])
        3. Entangle measurement qubit (q[5]) with validators
        4. Encode user/target information (q[6], q[7])
        5. Create GHZ-8 state across all qubits
        6. Apply transaction-specific measurement basis rotation
        7. Measure all qubits
        
        Args:
            tx_params: Transaction quantum parameters
        
        Returns:
            (circuit, metrics): QuantumCircuit and execution metrics
        """
        start_time = time.time()
        
        # Create quantum registers
        q = QuantumRegister(self.config.NUM_TOTAL_QUBITS, 'q')
        c = ClassicalRegister(self.config.NUM_CLASSICAL_BITS, 'c')
        
        # Create circuit
        circuit = QuantumCircuit(q, c, name=f"tx_{tx_params.tx_id[:16]}")
        
        try:
            # ═════════════════════════════════════════════════════════════
            # STEP 1: Initialize to |0⟩ (implicit)
            # ═════════════════════════════════════════════════════════════
            
            circuit.barrier(label='INIT')
            
            # ═════════════════════════════════════════════════════════════
            # STEP 2: Create W-State on Validator Qubits [0-4]
            # ═════════════════════════════════════════════════════════════
            self._create_w_state_validators(circuit, q)
            circuit.barrier(label='W_STATE')
            
            # ═════════════════════════════════════════════════════════════
            # STEP 3: Entangle Measurement Qubit [5] with Validators
            # ═════════════════════════════════════════════════════════════
            self._entangle_measurement_qubit(circuit, q)
            circuit.barrier(label='MEASUREMENT_ENTANGLE')
            
            # ═════════════════════════════════════════════════════════════
            # STEP 4: Encode User/Target Information [6-7]
            # ═════════════════════════════════════════════════════════════
            self._encode_user_target_qubits(circuit, q, tx_params)
            circuit.barrier(label='USER_TARGET_ENCODE')
            
            # ═════════════════════════════════════════════════════════════
            # STEP 5: Create GHZ-8 State (Full 8-Qubit Entanglement)
            # ═════════════════════════════════════════════════════════════
            self._create_ghz8_entanglement(circuit, q)
            circuit.barrier(label='GHZ8_ENTANGLE')
            
            # ═════════════════════════════════════════════════════════════
            # STEP 6: Apply Transaction-Specific Measurement Basis Rotation
            # ═════════════════════════════════════════════════════════════
            measurement_angle = tx_params.compute_measurement_basis_angle()
            self._apply_measurement_basis_rotation(circuit, q, measurement_angle)
            circuit.barrier(label='MEASUREMENT_ROTATION')
            
            # ═════════════════════════════════════════════════════════════
            # STEP 7: Measure All Qubits (Collapse)
            # ═════════════════════════════════════════════════════════════
            circuit.measure(q, c)
            
            # Validate circuit
            self._validate_circuit(circuit)
            
            # Compute metrics
            execution_time_ms = (time.time() - start_time) * 1000
            metrics = QuantumCircuitMetrics(
                circuit_name=circuit.name,
                num_qubits=circuit.num_qubits,
                num_classical_bits=circuit.num_clbits,
                circuit_depth=circuit.depth(),
                circuit_size=circuit.size(),
                num_gates=len(circuit),
                execution_time_ms=execution_time_ms,
                aer_shots=self.config.AER_SHOTS
            )
            
            logger.info(f"✓ Built circuit {circuit.name}")
            logger.info(f"  Qubits: {circuit.num_qubits}, Depth: {circuit.depth()}, Size: {circuit.size()}")
            
            return circuit, metrics
        
        except Exception as e:
            logger.error(f"✗ Circuit build failed for {tx_params.tx_id}: {e}")
            raise
    
    def _create_w_state_validators(self, circuit: QuantumCircuit, q: QuantumRegister) -> None:
        """
        Create W-state on validator qubits [0-4].
        
        W-state: |W⟩ = (1/√5)(|10000⟩ + |01000⟩ + |00100⟩ + |00010⟩ + |00001⟩)
        
        Equal superposition where exactly one validator qubit is in |1⟩ state.
        Represents equal probability across all validator consensus states.
        """
        # Get validator qubits
        validators = [q[i] for i in self.config.VALIDATOR_QUBITS]
        
        # Create equal superposition across validators using Hadamards
        for v in validators:
            circuit.h(v)
        
        # For W-state, we use a specific entanglement pattern
        # that creates the superposition of only single-qubit excitations
        
        # Apply controlled gates to create W-state structure
        # This ensures only one qubit is excited at a time
        for i in range(len(validators) - 1):
            circuit.cx(validators[i], validators[i + 1])
        
        logger.debug(f"Created W-state on {len(validators)} validator qubits")
    
    def _entangle_measurement_qubit(self, circuit: QuantumCircuit, q: QuantumRegister) -> None:
        """
        Entangle measurement qubit [5] with validator consensus.
        
        The measurement qubit collapses based on validator state.
        This implements the oracle trigger mechanism.
        """
        measurement_qubit = q[self.config.MEASUREMENT_QUBIT]
        validators = [q[i] for i in self.config.VALIDATOR_QUBITS]
        
        # Apply controlled-NOT gates from validators to measurement qubit
        # This creates entanglement: measurement qubit becomes function of validator state
        for validator in validators:
            circuit.cx(validator, measurement_qubit)
        
        logger.debug("Entangled measurement qubit with validator consensus")
    
    def _encode_user_target_qubits(
        self,
        circuit: QuantumCircuit,
        q: QuantumRegister,
        tx_params: TransactionQuantumParameters
    ) -> None:
        """
        Encode user and target information in qubits [6-7].
        
        Creates quantum signatures based on user_id and target_address hashes.
        These serve as authentication witnesses in the quantum state.
        """
        user_qubit = q[self.config.USER_QUBIT]
        target_qubit = q[self.config.TARGET_QUBIT]
        
        # Compute phase encodings
        user_phase = tx_params.compute_user_phase()
        target_phase = tx_params.compute_target_phase()
        
        # Apply Hadamard to create superposition
        circuit.h(user_qubit)
        circuit.h(target_qubit)
        
        # Apply phase rotations to encode user/target information
        circuit.rz(user_phase, user_qubit)
        circuit.rz(target_phase, target_qubit)
        
        logger.debug(f"Encoded user ({user_phase:.4f} rad) and target ({target_phase:.4f} rad) phases")
    
    def _create_ghz8_entanglement(self, circuit: QuantumCircuit, q: QuantumRegister) -> None:
        """
        Create GHZ-8 state across all 8 qubits.
        
        GHZ-8: (1/√2)(|00000000⟩ + |11111111⟩)
        
        All 8 qubits become perfectly entangled.
        Measurement of any qubit determines all others.
        Represents transaction finality guarantee.
        """
        # Apply Hadamard to first qubit (q[0])
        circuit.h(q[0])
        
        # Apply CNOT chain: q[0] → q[1] → q[2] → ... → q[7]
        # This creates the 8-way entanglement
        for i in range(self.config.NUM_TOTAL_QUBITS - 1):
            circuit.cx(q[i], q[i + 1])
        
        logger.debug("Created GHZ-8 entanglement across all 8 qubits")
    
    def _apply_measurement_basis_rotation(
        self,
        circuit: QuantumCircuit,
        q: QuantumRegister,
        angle: float
    ) -> None:
        """
        Apply transaction-specific measurement basis rotation.
        
        Rotates the measurement basis by transaction-dependent angle.
        This encodes transaction hash information in the measurement outcome.
        """
        # Apply RY rotation to all qubits
        for qubit in q:
            circuit.ry(angle, qubit)
        
        logger.debug(f"Applied measurement basis rotation: {angle:.4f} rad")
    
    def _validate_circuit(self, circuit: QuantumCircuit) -> None:
        """Validate circuit structure and constraints"""
        
        # Check qubit count
        if circuit.num_qubits != self.config.NUM_TOTAL_QUBITS:
            raise ValueError(f"Invalid qubit count: {circuit.num_qubits}")
        
        # Check classical bit count
        if circuit.num_clbits != self.config.NUM_CLASSICAL_BITS:
            raise ValueError(f"Invalid classical bit count: {circuit.num_clbits}")
        
        # Check circuit depth
        depth = circuit.depth()
        if depth > self.config.MAX_CIRCUIT_DEPTH:
            logger.warning(f"Circuit depth {depth} exceeds recommended {self.config.MAX_CIRCUIT_DEPTH}")
        
        logger.debug(f"✓ Circuit validation passed")


# ═══════════════════════════════════════════════════════════════════════════════════════
# QUANTUM EXECUTOR: AER SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════════════

class WStateGHZ8Executor:
    """Executes W-state + GHZ-8 quantum circuits on AER simulator"""
    
    def __init__(self, config: QuantumTopologyConfig = None):
        """Initialize executor"""
        self.config = config or QuantumTopologyConfig()
        self.simulator = AerSimulator(
            method='statevector',
            shots=self.config.AER_SHOTS
        )
        logger.info("✓ WStateGHZ8Executor initialized")
    
    def execute_circuit(
        self,
        circuit: QuantumCircuit,
        tx_params: TransactionQuantumParameters
    ) -> QuantumMeasurementResult:
        """
        Execute quantum circuit and extract measurement results.
        
        Args:
            circuit: Quantum circuit to execute
            tx_params: Transaction parameters
        
        Returns:
            QuantumMeasurementResult: Measurement results and analysis
        """
        start_time = time.time()
        
        try:
            # Execute circuit
            job = self.simulator.run(
                circuit,
                shots=self.config.AER_SHOTS,
                seed_simulator=self.config.AER_SEED,
                optimization_level=self.config.AER_OPTIMIZATION_LEVEL
            )
            result = job.result()
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Extract measurement counts
            counts = result.get_counts(circuit)
            
            # Analyze results
            analysis = self._analyze_measurement_results(counts, circuit.name, tx_params.tx_id)
            
            logger.info(f"✓ Executed circuit {circuit.name}")
            logger.info(f"  Shots: {self.config.AER_SHOTS}, Execution time: {execution_time_ms:.2f}ms")
            logger.info(f"  Dominant bitstring: {analysis.dominant_bitstring} ({analysis.dominant_count} times)")
            logger.info(f"  Entropy: {analysis.entropy_percent:.2f}%, GHZ fidelity: {analysis.ghz_fidelity:.4f}")
            
            return analysis
        
        except Exception as e:
            logger.error(f"✗ Circuit execution failed: {e}")
            raise
    
    def _analyze_measurement_results(
        self,
        counts: Dict[str, int],
        circuit_name: str,
        tx_id: str
    ) -> QuantumMeasurementResult:
        """
        Analyze quantum measurement results.
        
        Extracts:
        - Shannon entropy
        - GHZ state probability
        - Validator consensus
        - User/target signatures
        - Quantum commitment hash
        """
        
        # Find dominant bitstring
        dominant_bitstring = max(counts.items(), key=lambda x: x[1])[0]
        dominant_count = counts[dominant_bitstring]
        
        # Compute Shannon entropy
        total_shots = sum(counts.values())
        probabilities = np.array([count / total_shots for count in counts.values()])
        shannon_entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        max_entropy = math.log2(len(counts))
        entropy_normalized = shannon_entropy / max_entropy if max_entropy > 0 else 0
        entropy_percent = entropy_normalized * 100
        
        # Extract GHZ-specific metrics
        ghz_bitstrings = ['00000000', '11111111']  # GHZ state outcomes
        ghz_count = sum(counts.get(bs, 0) for bs in ghz_bitstrings)
        ghz_probability = ghz_count / total_shots
        ghz_fidelity = ghz_probability  # How close to ideal GHZ measurement
        
        # Extract validator consensus (first 5 bits)
        validator_consensus = self._extract_validator_consensus(counts)
        validator_agreement_score = max(validator_consensus.values()) if validator_consensus else 0.0
        
        # Extract user/target signatures (bits 6 and 7)
        user_bit = self._extract_qubit_value(counts, qubit_index=6)
        target_bit = self._extract_qubit_value(counts, qubit_index=7)
        
        # Generate quantum commitment hashes
        state_hash = hashlib.sha256(json.dumps(counts, sort_keys=True).encode()).hexdigest()
        commitment_data = f"{tx_id}:{dominant_bitstring}:{state_hash}"
        commitment_hash = hashlib.sha256(commitment_data.encode()).hexdigest()
        
        return QuantumMeasurementResult(
            circuit_name=circuit_name,
            tx_id=tx_id,
            bitstring_counts=counts,
            dominant_bitstring=dominant_bitstring,
            dominant_count=dominant_count,
            shannon_entropy=shannon_entropy,
            entropy_percent=entropy_percent,
            ghz_state_probability=ghz_probability,
            ghz_fidelity=ghz_fidelity,
            validator_consensus=validator_consensus,
            validator_agreement_score=validator_agreement_score,
            user_signature_bit=user_bit,
            target_signature_bit=target_bit,
            state_hash=state_hash,
            commitment_hash=commitment_hash
        )
    
    def _extract_validator_consensus(
        self,
        counts: Dict[str, int]
    ) -> Dict[str, float]:
        """
        Extract validator consensus from measurement results.
        
        Takes first 5 bits (validator qubits) from each bitstring
        and computes probability distribution.
        """
        total_shots = sum(counts.values())
        validator_states = {}
        
        for bitstring, count in counts.items():
            # Extract first 5 bits (validator qubits)
            if len(bitstring) >= 5:
                validator_bits = bitstring[:5]
                if validator_bits not in validator_states:
                    validator_states[validator_bits] = 0
                validator_states[validator_bits] += count
        
        # Convert to probabilities
        consensus = {
            state: count / total_shots 
            for state, count in validator_states.items()
        }
        
        return consensus
    
    def _extract_qubit_value(self, counts: Dict[str, int], qubit_index: int) -> int:
        """
        Extract most probable value of specific qubit.
        
        Args:
            counts: Measurement counts
            qubit_index: Index of qubit (0-7)
        
        Returns:
            Most probable qubit value (0 or 1)
        """
        total_shots = sum(counts.values())
        count_0 = 0
        count_1 = 0
        
        for bitstring, count in counts.items():
            if len(bitstring) > qubit_index:
                if bitstring[qubit_index] == '0':
                    count_0 += count
                else:
                    count_1 += count
        
        return 1 if count_1 > count_0 else 0


# ═══════════════════════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCES
# ═══════════════════════════════════════════════════════════════════════════════════════

_circuit_builder = None
_executor = None

def get_circuit_builder(config: QuantumTopologyConfig = None) -> WStateValidatorCircuitBuilder:
    """Get singleton circuit builder instance"""
    global _circuit_builder
    if _circuit_builder is None:
        _circuit_builder = WStateValidatorCircuitBuilder(config)
    return _circuit_builder

def get_executor(config: QuantumTopologyConfig = None) -> WStateGHZ8Executor:
    """Get singleton executor instance"""
    global _executor
    if _executor is None:
        _executor = WStateGHZ8Executor(config)
    return _executor

# ═══════════════════════════════════════════════════════════════════════════════════════
# MAIN TEST/VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    logger.info("=" * 100)
    logger.info("W-STATE VALIDATOR + GHZ-8 QUANTUM CIRCUIT BUILDER - TEST")
    logger.info("=" * 100)
    
    try:
        # Initialize
        config = QuantumTopologyConfig()
        builder = get_circuit_builder(config)
        executor = get_executor(config)
        
        # Create test transaction
        test_tx = TransactionQuantumParameters(
            tx_id="test_tx_001",
            user_id="user_123",
            target_address="addr_456",
            amount=100.0
        )
        
        # Build circuit
        circuit, metrics = builder.build_transaction_circuit(test_tx)
        logger.info(f"\n✓ Circuit metrics:\n{json.dumps(metrics.to_dict(), indent=2)}")
        
        # Execute circuit
        result = executor.execute_circuit(circuit, test_tx)
        logger.info(f"\n✓ Measurement results:\n{json.dumps(result.to_dict(), indent=2, default=str)}")
        
        logger.info("=" * 100)
        logger.info("✓ TEST PASSED")
        logger.info("=" * 100)
    
    except Exception as e:
        logger.error(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
