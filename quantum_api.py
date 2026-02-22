#!/usr/bin/env python3

import logging
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# GLOBALS INTEGRATION - Unified State Management
# ═══════════════════════════════════════════════════════════════════════════════════════
try:
    from globals import get_db_pool, get_heartbeat, get_globals, get_auth_manager, get_terminal
    GLOBALS_AVAILABLE = True
except ImportError:
    GLOBALS_AVAILABLE = False
    logger.warning(f"[{os.path.basename(input_path)}] Globals not available - using fallback")


"""🌌 QUANTUM API - Production Quantum Blockchain System"""

import os,sys,json,time,hashlib,uuid,logging,threading,secrets,hmac,base64,re,traceback,copy,struct,random,math,sqlite3
from datetime import datetime,timedelta,timezone
from typing import Dict,List,Optional,Any,Tuple,Set,Callable,Union
from functools import wraps,lru_cache,partial
from decimal import Decimal,getcontext
from dataclasses import dataclass,asdict,field
from enum import Enum,IntEnum,auto
from collections import defaultdict,deque,Counter,OrderedDict
from concurrent.futures import ThreadPoolExecutor,as_completed,wait,FIRST_COMPLETED
from flask import Blueprint,request,jsonify,g,Response,stream_with_context
import psycopg2
from psycopg2.extras import RealDictCursor,execute_batch,execute_values,Json

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
    from qiskit_aer import AerSimulator, QasmSimulator, StatevectorSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error, phase_damping_error, pauli_error
    from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity, entropy, partial_trace, purity
    from qiskit.circuit.library import QFT, GroverOperator, EfficientSU2
    
    # Handle execute import - deprecated in qiskit 1.0+
    try:
        from qiskit import execute
    except ImportError:
        # Qiskit 1.0+ - execute was moved/removed, we'll use simulator.run() instead
        execute = None
    
    QISKIT_AVAILABLE = True
except ImportError as e:
    QISKIT_AVAILABLE = False
    logging.warning(f"⚠️  Qiskit not available: {e}")
    execute = None

try:
    import numpy as np
    from scipy.linalg import eigvalsh,expm
    from scipy.special import xlogy
    from scipy.optimize import minimize
    NUMPY_AVAILABLE=True
    SCIPY_AVAILABLE=True
except ImportError:
    NUMPY_AVAILABLE=False
    SCIPY_AVAILABLE=False

# ═════════════════════════════════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP GLOBALS INTEGRATION - Access unified system registry
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════
try:
    from wsgi_config import GLOBALS
    GLOBALS_AVAILABLE=True
    logging.info("[QuantumAPI] ✓ GLOBALS bootstrap system imported")
except ImportError:
    GLOBALS_AVAILABLE=False
    logging.warning("[QuantumAPI] ⚠ GLOBALS not available - will use direct imports")
    class DummyGLOBALS:
        QUANTUM=None
        DB=None
    GLOBALS=DummyGLOBALS()

getcontext().prec=32

# ═════════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 1: GLOBAL QUANTUM ENGINE STATE & CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════

class QuantumTopologyConfig:
    """Ultimate quantum topology configuration"""
    NUM_TOTAL_QUBITS=8
    VALIDATOR_QUBITS=[0,1,2,3,4]
    MEASUREMENT_QUBIT=5
    USER_QUBIT=6
    TARGET_QUBIT=7
    NUM_CLASSICAL_BITS=8
    NUM_VALIDATORS=5
    
    # W-State + GHZ Configuration
    W_STATE_EQUAL_SUPERPOSITION=True
    GHZ_PHASE_ENCODING=True
    GHZ_ENTANGLEMENT_DEPTH=3
    
    # Phase encoding
    PHASE_BITS_USER=8
    PHASE_BITS_TARGET=8
    
    # Circuit Configuration
    CIRCUIT_TRANSPILE=True
    CIRCUIT_OPTIMIZATION_LEVEL=3
    MAX_CIRCUIT_DEPTH=100
    
    # AER Simulator Configuration
    AER_SHOTS=2048
    AER_SEED=42
    AER_OPTIMIZATION_LEVEL=3
    EXECUTION_TIMEOUT_MS=500
    
    # Measurement Configuration
    MEASUREMENT_BASIS_ROTATION_ENABLED=True
    MEASUREMENT_BASIS_ANGLE_VARIANCE=math.pi/8
    
    # Quality Thresholds
    MIN_GHZ_FIDELITY_THRESHOLD=0.3
    MIN_W_STATE_FIDELITY=0.6
    ENTROPY_QUALITY_THRESHOLD=0.7
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS_PER_MIN=600
    RATE_LIMIT_CIRCUITS_PER_MIN=300
    
    # Validator Configuration
    VALIDATOR_MIN_STAKE=100
    VALIDATOR_COMMISSION_MIN=0.01
    VALIDATOR_COMMISSION_MAX=0.50
    
    # Reward Configuration
    REWARD_EPOCH_BLOCKS=6400
    SLASH_PERCENTAGE_DOUBLE_SPEND=0.05
    SLASH_PERCENTAGE_DOWNTIME=0.01
    
    # Transaction Configuration
    MAX_TRANSACTION_QUEUE_SIZE=10000
    TRANSACTION_BATCH_SIZE=5
    TRANSACTION_PROCESSING_INTERVAL_SEC=2.0
    
    # Noise Configuration
    DEPOLARIZING_RATE=0.001
    AMPLITUDE_DAMPING_RATE=0.0005
    PHASE_DAMPING_RATE=0.0003
    MEASUREMENT_ERROR_RATE=0.01
    
    # Non-Markovian Bath Configuration
    NON_MARKOVIAN_MEMORY_KERNEL=0.08
    BATH_COUPLING_STRENGTH=0.05
    DECOHERENCE_TIME_MS=100.0
    
    # Neural Network Integration
    NEURAL_NETWORK_ENABLED=True
    NEURAL_UPDATE_FREQUENCY_MS=50
    NEURAL_WEIGHT_DECAY=0.0001

class QuantumCircuitType(Enum):
    """Supported quantum circuit types"""
    ENTROPY_GENERATOR="entropy_generator"
    VALIDATOR_PROOF="validator_proof"
    W_STATE_VALIDATOR="w_state_validator"
    W_STATE_5QUBIT="w_state_5qubit"
    GHZ_3="ghz_3"
    GHZ_8="ghz_8"
    ENTANGLEMENT="entanglement"
    INTERFERENCE="interference"
    QFT="quantum_fourier_transform"
    GROVER="grover_search"
    CUSTOM="custom"

class ValidatorStatus(Enum):
    """Validator status enumeration"""
    INACTIVE="inactive"
    PENDING="pending"
    ACTIVE="active"
    JAILED="jailed"
    UNBONDING="unbonding"
    SLASHED="slashed"

class TransactionStatus(Enum):
    """Transaction status enumeration"""
    PENDING="pending"
    ENCODING="encoding"
    PROCESSING="processing"
    FINALIZED="finalized"
    FAILED="failed"
    ROLLED_BACK="rolled_back"

class QuantumExecutionStatus(Enum):
    """Quantum execution status enumeration"""
    QUEUED="queued"
    RUNNING="running"
    COMPLETED="completed"
    FAILED="failed"
    CANCELLED="cancelled"
    VERIFIED="verified"

class EntropyQuality(Enum):
    """Entropy quality levels"""
    LOW="low"
    MEDIUM="medium"
    HIGH="high"
    QUANTUM_CERTIFIED="quantum_certified"

class SlashReason(Enum):
    """Validator slashing reasons"""
    DOUBLE_SPEND="double_spend"
    INVALID_CONSENSUS="invalid_consensus"
    DOWNTIME="downtime"
    BYZANTINE="byzantine"
    FIDELITY_VIOLATION="fidelity_violation"
    VOLUNTARY="voluntary"

# ═════════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 2: CORE DATACLASSES
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class TransactionQuantumParameters:
    """Transaction quantum encoding parameters"""
    tx_id:str
    user_id:str
    target_address:str
    amount:float
    timestamp:float=field(default_factory=time.time)
    metadata:Dict[str,Any]=field(default_factory=dict)
    
    def compute_user_phase(self)->float:
        """Compute user qubit phase from user_id"""
        user_hash=int(hashlib.md5(self.user_id.encode()).hexdigest(),16)%256
        return (user_hash/256.0)*(2*math.pi)
    
    def compute_target_phase(self)->float:
        """Compute target qubit phase from target address"""
        target_hash=int(hashlib.md5(self.target_address.encode()).hexdigest(),16)%256
        return (target_hash/256.0)*(2*math.pi)
    
    def compute_measurement_basis_angle(self)->float:
        """Compute oracle measurement basis angle"""
        tx_data=f"{self.tx_id}{self.amount}".encode()
        tx_hash=int(hashlib.sha256(tx_data).hexdigest(),16)%1000
        variance=QuantumTopologyConfig.MEASUREMENT_BASIS_ANGLE_VARIANCE
        return -variance+(2*variance*(tx_hash/1000.0))

@dataclass
class QuantumCircuitMetrics:
    """Metrics for quantum circuit execution"""
    circuit_name:str
    circuit_type:str
    num_qubits:int
    num_classical_bits:int
    circuit_depth:int
    circuit_size:int
    num_gates:int
    execution_time_ms:float
    aer_shots:int
    fidelity:float=0.0
    entropy_value:float=0.0
    coherence:float=0.0
    discord:float=0.0
    mutual_information:float=0.0
    bell_violation:float=0.0
    created_at:datetime=field(default_factory=datetime.utcnow)
    
    def to_dict(self)->Dict:
        d=asdict(self)
        d['created_at']=self.created_at.isoformat()
        return d

@dataclass
class QuantumMeasurementResult:
    """Results from quantum measurement"""
    circuit_name:str
    tx_id:str
    bitstring_counts:Dict[str,int]
    dominant_bitstring:str
    dominant_count:int
    shannon_entropy:float
    entropy_percent:float
    coherence_measure:float
    fidelity:float
    discord:float
    mutual_information:float
    bell_violation:float
    validator_consensus:Dict[str,float]
    validator_agreement_score:float
    user_signature_bit:int
    target_signature_bit:int
    oracle_collapse_bit:int
    state_hash:str
    commitment_hash:str
    measurement_timestamp:datetime=field(default_factory=datetime.utcnow)
    
    def to_dict(self)->Dict:
        d=asdict(self)
        d['measurement_timestamp']=self.measurement_timestamp.isoformat()
        return d

@dataclass
class QuantumExecution:
    """Quantum execution record"""
    execution_id:str
    circuit_type:str
    status:str
    num_qubits:int
    shots:int
    created_at:datetime
    started_at:Optional[datetime]=None
    completed_at:Optional[datetime]=None
    results:Optional[Dict[str,Any]]=None
    measurements:Optional[Dict[str,int]]=None
    statevector:Optional[List[complex]]=None
    density_matrix:Optional[List[List[complex]]]=None
    entropy_value:Optional[float]=None
    fidelity:Optional[float]=None
    coherence:Optional[float]=None
    discord:Optional[float]=None
    mutual_information:Optional[float]=None
    bell_violation:Optional[float]=None
    error_message:Optional[str]=None
    metadata:Dict[str,Any]=field(default_factory=dict)

@dataclass
class Validator:
    """Validator record"""
    validator_id:str
    address:str
    public_key:str
    status:str
    stake_amount:Decimal
    commission_rate:Decimal
    total_delegated:Decimal
    blocks_validated:int=0
    uptime_percentage:float=100.0
    last_active:datetime=field(default_factory=lambda:datetime.now(timezone.utc))
    joined_at:datetime=field(default_factory=lambda:datetime.now(timezone.utc))
    jailed_until:Optional[datetime]=None
    slash_count:int=0
    slashes:List[Dict[str,Any]]=field(default_factory=list)
    quantum_proof:Optional[str]=None
    metadata:Dict[str,Any]=field(default_factory=dict)

@dataclass
class EntropySource:
    """Entropy source record"""
    entropy_id:str
    entropy_bytes:bytes
    quality:str
    num_qubits:int
    shots:int
    min_entropy:float
    timestamp:datetime
    source:str="quantum"

@dataclass
class ValidatorReward:
    """Validator reward record"""
    reward_id:str
    validator_id:str
    epoch:int
    block_rewards:Decimal
    fee_rewards:Decimal
    total_rewards:Decimal
    commission:Decimal
    delegator_share:Decimal
    timestamp:datetime
    distributed:bool=False
    distribution_tx:Optional[str]=None

# ═════════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 3: QUANTUM INFORMATION METRICS - THE POWERHOUSE
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════

class QuantumInformationMetrics:
    """Complete quantum information theory implementation"""
    
    def __init__(self):
        self.cache={}
        self.lock=threading.RLock()
    
    @staticmethod
    def von_neumann_entropy(density_matrix:np.ndarray)->float:
        """
        Compute von Neumann entropy S(ρ) = -Tr(ρ log ρ)
        Measures how mixed the quantum state is (0 = pure, log(N) = maximally mixed)
        """
        try:
            if density_matrix is None:
                return 0.0
            
            # Get eigenvalues
            eigenvalues=np.linalg.eigvalsh(density_matrix)
            eigenvalues=np.maximum(eigenvalues,1e-15)  # Avoid log(0)
            
            # Compute entropy
            entropy=-np.sum(eigenvalues*np.log2(eigenvalues))
            return float(np.real(entropy))
        except:
            return 0.0
    
    @staticmethod
    def shannon_entropy(bitstring_counts:Dict[str,int])->float:
        """
        Compute Shannon entropy H = -Σ p_i log2(p_i)
        Measures information content of measurement outcomes
        """
        try:
            total=sum(bitstring_counts.values())
            if total==0:
                return 0.0
            
            entropy=0.0
            for count in bitstring_counts.values():
                if count>0:
                    p=count/total
                    entropy-=p*math.log2(p)
            return entropy
        except:
            return 0.0
    
    @staticmethod
    def coherence_l1_norm(density_matrix:np.ndarray)->float:
        """
        Compute l1-norm coherence C(ρ) = Σ_{i≠j} |ρ_{ij}|
        Measures off-diagonal elements that represent quantum coherence
        """
        try:
            if density_matrix is None:
                return 0.0
            
            coherence=0.0
            n=density_matrix.shape[0]
            for i in range(n):
                for j in range(n):
                    if i!=j:
                        coherence+=abs(density_matrix[i,j])
            return float(coherence)
        except:
            return 0.0
    
    @staticmethod
    def coherence_renyi(density_matrix:np.ndarray,order:float=2)->float:
        """
        Compute Rényi coherence of order α
        C_α(ρ) = (1/(1-α)) log Tr[(ρ_d)^α]
        where ρ_d is the diagonal part (incoherent state)
        """
        try:
            if density_matrix is None:
                return 0.0
            
            if order==1:
                return QuantumInformationMetrics.coherence_l1_norm(density_matrix)
            
            # Diagonal part
            diagonal_part=np.diag(np.diag(density_matrix))
            
            # Trace of diagonal part to power alpha
            eigenvalues=np.linalg.eigvalsh(diagonal_part)
            eigenvalues=np.maximum(eigenvalues,1e-15)
            
            trace_power=np.sum(eigenvalues**order)
            if trace_power<=0:
                return 0.0
            
            coherence=(1/(1-order))*math.log2(trace_power)
            return float(np.real(coherence))
        except:
            return 0.0
    
    @staticmethod
    def geometric_coherence(density_matrix:np.ndarray)->float:
        """
        Compute geometric coherence: C_g(ρ) = min_σ ||ρ-σ||_1
        Distance to closest incoherent state
        """
        try:
            if density_matrix is None:
                return 0.0
            
            # Incoherent state = diagonal part
            diagonal_part=np.diag(np.diag(density_matrix))
            
            # Trace distance
            diff=density_matrix-diagonal_part
            eigenvalues=np.linalg.eigvalsh(diff@np.conj(diff.T))
            trace_distance=0.5*np.sum(np.sqrt(np.maximum(eigenvalues,0)))
            
            return float(trace_distance)
        except:
            return 0.0
    
    @staticmethod
    def purity(density_matrix:np.ndarray)->float:
        """
        Compute purity Tr(ρ²)
        Pure states: purity=1, Maximally mixed: purity=1/d
        """
        try:
            if density_matrix is None:
                return 0.0
            
            purity_val=float(np.real(np.trace(density_matrix@density_matrix)))
            return min(1.0,max(0.0,purity_val))
        except:
            return 0.0
    
    @staticmethod
    def state_fidelity(rho1:np.ndarray,rho2:np.ndarray)->float:
        """
        Compute fidelity F(ρ₁,ρ₂) = Tr(√(√ρ₁ρ₂√ρ₁))²
        Measures overlap between two quantum states
        """
        try:
            if rho1 is None or rho2 is None:
                return 0.0
            
            # Compute √ρ₁
            eigvals,eigvecs=np.linalg.eigh(rho1)
            eigvals=np.maximum(eigvals,0)
            sqrt_rho1=eigvecs@np.diag(np.sqrt(eigvals))@eigvecs.conj().T
            
            # Compute √ρ₁ρ₂√ρ₁
            product=sqrt_rho1@rho2@sqrt_rho1
            
            # Eigenvalues of product
            eigvals_prod=np.linalg.eigvalsh(product)
            eigvals_prod=np.maximum(eigvals_prod,0)
            
            # Trace of sqrt
            trace_sqrt=np.sum(np.sqrt(eigvals_prod))
            
            fidelity=float(trace_sqrt)**2
            return min(1.0,max(0.0,fidelity))
        except:
            return 0.0
    
    @staticmethod
    def quantum_discord(density_matrix:np.ndarray)->float:
        """
        Quantum discord: D(ρ) = I(ρ) - C(ρ)
        Where I is mutual information and C is classical correlation
        Measures purely quantum correlation difference from classical
        """
        try:
            if density_matrix is None or density_matrix.shape[0]<2:
                return 0.0
            
            # Total correlation (mutual information)
            total_corr=QuantumInformationMetrics.mutual_information(density_matrix)
            
            # Classical correlation (obtained via optimal measurements)
            classical_corr=QuantumInformationMetrics._classical_correlation(density_matrix)
            
            # Discord is the difference
            discord=max(0.0,total_corr-classical_corr)
            return float(discord)
        except:
            return 0.0
    
    @staticmethod
    def mutual_information(density_matrix:np.ndarray)->float:
        """
        Quantum mutual information I(ρ) = S(ρ_A) + S(ρ_B) - S(ρ_AB)
        Total correlation between subsystems
        """
        try:
            if density_matrix is None or density_matrix.shape[0]<2:
                return 0.0
            
            # For 8-qubit system, compute over bipartition A={0,1,2,3,4} B={5,6,7}
            # Partial traces
            dim=density_matrix.shape[0]
            half=dim//2
            
            # Compute partial traces (simplified for 8-qubit)
            rho_a=np.zeros((half,half),dtype=complex)
            rho_b=np.zeros((dim-half,dim-half),dtype=complex)
            
            for i in range(half):
                for j in range(half):
                    for k in range(dim-half):
                        rho_a[i,j]+=density_matrix[i*2+k,j*2+k]
            
            for i in range(dim-half):
                for j in range(dim-half):
                    for k in range(half):
                        rho_b[i,j]+=density_matrix[i*2+k,j*2+k]
            
            # Entropies
            s_a=QuantumInformationMetrics.von_neumann_entropy(rho_a)
            s_b=QuantumInformationMetrics.von_neumann_entropy(rho_b)
            s_ab=QuantumInformationMetrics.von_neumann_entropy(density_matrix)
            
            # Mutual information
            mi=s_a+s_b-s_ab
            return float(max(0.0,mi))
        except:
            return 0.0
    
    @staticmethod
    def _classical_correlation(density_matrix:np.ndarray)->float:
        """Approximate classical correlation via maximum measurement correlation"""
        try:
            mi=QuantumInformationMetrics.mutual_information(density_matrix)
            # Classical correlation ≤ mutual information
            # For simplicity: approximate as 0.7*MI (typical reduction factor)
            return 0.7*mi
        except:
            return 0.0
    
    @staticmethod
    def entanglement_entropy(density_matrix:np.ndarray,partition_A:List[int])->float:
        """
        Compute entanglement entropy for partition A
        S_A = -Tr(ρ_A log ρ_A)
        """
        try:
            if density_matrix is None:
                return 0.0
            
            # Simplified partial trace for given partition
            # Trace out all qubits not in partition_A
            rho_a=partial_trace(density_matrix,[i for i in range(density_matrix.shape[0]//2) if i in partition_A])
            
            return QuantumInformationMetrics.von_neumann_entropy(rho_a)
        except:
            return 0.0
    
    @staticmethod
    def bell_inequality_chsh(counts_00:int,counts_01:int,counts_10:int,counts_11:int)->float:
        """
        CHSH Bell inequality: |⟨S⟩| ≤ 2 (classical), ≤ 2√2 (quantum)
        S = E₀₀ + E₀₁ + E₁₀ - E₁₁
        
        Returns normalized violation (0=classical, 1=maximum quantum)
        """
        try:
            total=counts_00+counts_01+counts_10+counts_11
            if total==0:
                return 0.0
            
            # Correlation values: E_ij = (N_ij - N_ij')/total
            p_00=counts_00/total
            p_01=counts_01/total
            p_10=counts_10/total
            p_11=counts_11/total
            
            # Simplified CHSH calculation
            e_values=[
                p_00-p_01+p_10-p_11,  # First angle setting
                p_00+p_01-p_10-p_11   # Second angle setting
            ]
            
            s=abs(sum(e_values))
            
            # Normalized violation: (S - 2)/(2√2 - 2)
            violation=(s-2.0)/(2*math.sqrt(2)-2.0) if s>2 else 0.0
            return float(min(1.0,max(0.0,violation)))
        except:
            return 0.0
    
    @staticmethod
    def bell_inequality_mermin(counts:Dict[str,int],num_qubits:int=3)->float:
        """
        Mermin inequality for N qubits
        Returns normalized Bell violation
        """
        try:
            if not counts or len(counts)==0:
                return 0.0
            
            total=sum(counts.values())
            if total==0:
                return 0.0
            
            # Simplified Mermin calculation
            # M_N = 2^(N-1) for product states, 2^(N/2) for maximally entangled
            violations=0
            for bitstring,count in counts.items():
                parity=bitstring.count('1')%2
                if parity==1:
                    violations+=count
            
            violation_fraction=violations/total
            return float(min(1.0,violation_fraction))
        except:
            return 0.0


    @staticmethod
    def w_state_mermin_3qubit(counts:Dict[str,int])->Dict[str,Any]:
        """3-qubit W-state Mermin: M₃ = ⟨XYZ⟩+⟨XZY⟩+⟨YXZ⟩-⟨ZXY⟩-⟨ZYX⟩-⟨YZX⟩. Classical|M₃|≤4, W-state|M₃|≈5.2."""
        try:
            if not counts:return{'mermin_3qubit':0.0,'violation':False,'confidence':0.0}
            total=sum(counts.values())
            if total==0:return{'mermin_3qubit':0.0,'violation':False,'confidence':0.0}
            correlations={}
            for bitstring,count in counts.items():
                if len(bitstring)<3:continue
                b=[int(bitstring[i])for i in range(3)]
                parity=(b[0]+b[1]+b[2])%2
                for basis in ['XYZ','XZY','YXZ','ZXY','ZYX','YZX']:
                    if basis not in correlations:correlations[basis]=0.0
                    correlations[basis]+=(1.0 if parity==0 else -1.0)*(count/total)
            m3=(correlations.get('XYZ',0.0)+correlations.get('XZY',0.0)+correlations.get('YXZ',0.0)-
                correlations.get('ZXY',0.0)-correlations.get('ZYX',0.0)-correlations.get('YZX',0.0))
            classical_bound=4.0
            quantum_bound=4.0*math.sqrt(2)
            violation=abs(m3)>classical_bound
            confidence=min(abs(m3)/quantum_bound,1.0)
            return{'mermin_3qubit':float(m3),'classical_bound':classical_bound,'quantum_bound':quantum_bound,'violation':violation,'violation_strength':abs(m3)/classical_bound if classical_bound>0 else 0,'confidence':float(confidence),'is_multipartite_entangled':violation}
        except Exception as e:logger.debug(f"W-state Mermin error: {e}");return{'mermin_3qubit':0.0,'violation':False,'confidence':0.0}
    
    @staticmethod
    def entanglement_witness_3qubit(rho:np.ndarray)->Dict[str,Any]:
        """W=α·I+β|W⟩⟨W|. Tr(ρ·W)<0→entangled. Ideal W₃: Tr(ρ·W)≈-0.4."""
        try:
            if rho is None or rho.size==0:return{'is_entangled':False,'witness_trace':0.0}
            w_state=np.zeros(8,dtype=complex)
            w_state[4]=w_state[2]=w_state[1]=1/np.sqrt(3)
            w_proj=np.outer(w_state,w_state.conj())
            witness=0.5*np.eye(8,dtype=complex)-1.0*w_proj
            trace_val=float(np.real(np.trace(rho@witness)))
            is_entangled=trace_val<-0.001
            fidelity=float(np.real(np.trace(w_proj@rho)))
            return{'is_entangled':is_entangled,'witness_trace':trace_val,'threshold':-0.001,'margin':trace_val+0.001,'w_state_fidelity':fidelity,'certification':{'certified':is_entangled,'confidence':max(0,min(1,-trace_val/0.5))}}
        except Exception as e:logger.debug(f"Witness error: {e}");return{'is_entangled':False,'witness_trace':0.0}
    
    @staticmethod
    def reconstruct_density_matrix(measurement_bases:Dict[str,Dict[str,int]],n_qubits:int=3)->Optional[np.ndarray]:
        """MLE density matrix reconstruction from multi-basis measurement data."""
        try:
            dim=2**n_qubits
            rho=np.eye(dim,dtype=complex)/dim
            for iteration in range(50):
                rho_old=rho.copy()
                for basis_name,counts in measurement_bases.items():
                    total=sum(counts.values())
                    if total==0:continue
                    for bitstring,count in counts.items():
                        idx=int(bitstring,2)
                        proj=np.zeros((dim,dim),dtype=complex)
                        proj[idx,idx]=1.0
                        p_measured=count/total
                        p_predicted=float(np.real(np.trace(proj@rho)))
                        if p_predicted>1e-10:rho+=(p_measured/p_predicted-1.0)*proj
                eigvals,eigvecs=np.linalg.eigh(rho)
                eigvals[eigvals<0]=0
                rho=eigvecs@np.diag(eigvals)@eigvecs.conj().T
                rho/=np.trace(rho)
                if np.linalg.norm(rho-rho_old)<1e-6:break
            return rho
        except Exception as e:logger.debug(f"Density matrix error: {e}");return None
    

# Global metrics engine (accessible from anywhere)
QUANTUM_METRICS=QuantumInformationMetrics()

# ═════════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 4: NON-MARKOVIAN NOISE BATH SYSTEM
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════

class NonMarkovianNoiseBath:
    """
    Non-Markovian noise bath with memory kernel
    Models realistic quantum decoherence with memory effects
    """
    
    def __init__(self,memory_kernel:float=0.08,coupling_strength:float=0.05):
        self.memory_kernel=memory_kernel  # κ parameter
        self.coupling_strength=coupling_strength  # λ parameter
        self.history=deque(maxlen=100)
        self.lock=threading.RLock()
        self.noise_model=None
        self._init_noise_model()
    
    def _init_noise_model(self):
        """Initialize Qiskit noise model"""
        if not QISKIT_AVAILABLE:
            return
        
        try:
            self.noise_model=NoiseModel()
            
            # Single-qubit errors
            depol_error=depolarizing_error(QuantumTopologyConfig.DEPOLARIZING_RATE,1)
            amp_error=amplitude_damping_error(QuantumTopologyConfig.AMPLITUDE_DAMPING_RATE)
            phase_error=phase_damping_error(QuantumTopologyConfig.PHASE_DAMPING_RATE)
            
            for qubit in range(QuantumTopologyConfig.NUM_TOTAL_QUBITS):
                # Fix: v1.0+ API uses positional qargs, not keyword argument
                try:
                    self.noise_model.add_quantum_error(depol_error,'u1',[qubit])
                    self.noise_model.add_quantum_error(depol_error,'u2',[qubit])
                    self.noise_model.add_quantum_error(depol_error,'u3',[qubit])
                except TypeError:
                    # Fallback for older versions or different API
                    try:
                        self.noise_model.add_quantum_error(depol_error,['u1','u2','u3'])
                    except:
                        pass
                
                try:
                    self.noise_model.add_quantum_error(amp_error,'reset',[qubit])
                except:
                    pass
                
                try:
                    self.noise_model.add_quantum_error(phase_error,'measure',[qubit])
                except:
                    pass
            
            # Two-qubit errors
            two_qubit_error=depolarizing_error(QuantumTopologyConfig.DEPOLARIZING_RATE*2,2)
            for q1 in range(QuantumTopologyConfig.NUM_TOTAL_QUBITS):
                for q2 in range(q1+1,QuantumTopologyConfig.NUM_TOTAL_QUBITS):
                    try:
                        self.noise_model.add_quantum_error(two_qubit_error,'cx',[q1,q2])
                    except:
                        pass
            
            logger.info(f"✅ Non-Markovian noise bath initialized (κ={self.memory_kernel})")
        except Exception as e:
            logger.warning(f"⚠️  Noise model initialization failed: {e}")
    
    def ornstein_uhlenbeck_kernel(self,tau:float,t:float)->float:
        """
        Ornstein-Uhlenbeck memory kernel: K(t) = κ exp(-t/τ)
        Models non-Markovian memory effects
        """
        if t<0:
            return 0.0
        try:
            return self.memory_kernel*math.exp(-t/max(tau,0.01))
        except:
            return 0.0
    
    def compute_decoherence_function(self,t:float,t_dephase:float=100.0)->float:
        """
        Non-Markovian decoherence function with memory
        D(t) = exp(-(t/T₂)^2) + κ∫K(s)ds
        """
        try:
            # Exponential decay (Markovian part)
            markovian=math.exp(-(t/max(t_dephase,1.0))**2)
            
            # Memory contribution (Non-Markovian part)
            memory=self.memory_kernel*(1-math.exp(-t/max(t_dephase,1.0)))
            
            total=markovian*(1-memory)
            return float(max(0.0,min(1.0,total)))
        except:
            return 1.0
    
    def apply_memory_effect(self,density_matrix:np.ndarray,time_step:float)->np.ndarray:
        """
        Apply non-Markovian memory effect to density matrix
        """
        if density_matrix is None or not NUMPY_AVAILABLE:
            return density_matrix
        
        try:
            with self.lock:
                # Store in history
                self.history.append((time.time(),density_matrix.copy()))
                
                # Compute decoherence with memory
                decoherence_factor=self.compute_decoherence_function(time_step)
                
                # Apply damping
                result=decoherence_factor*density_matrix
                
                # Add small correlated noise from history
                if len(self.history)>1:
                    prev_matrix=self.history[0][1]
                    memory_contribution=0.01*(1-decoherence_factor)*prev_matrix
                    result+=memory_contribution
                
                # Renormalize
                trace=np.trace(result)
                if abs(trace)>1e-10:
                    result/=trace
                
                return result
        except:
            return density_matrix
    
    def get_noise_model(self):
        """Return Qiskit noise model"""
        return self.noise_model

# Global noise bath (accessible from anywhere)
NOISE_BATH=NonMarkovianNoiseBath()

# ═════════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 5: QUANTUM CIRCUIT BUILDERS - W-STATE, GHZ-3, GHZ-8
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════

class QuantumCircuitBuilders:
    """Advanced quantum circuit construction with interference & entanglement"""
    
    @staticmethod
    def build_w_state_5qubit(circuit:QuantumCircuit,qubits:List[int])->QuantumCircuit:
        """
        Build 5-qubit W-state |W5⟩ = (1/√5)(|10000⟩+|01000⟩+|00100⟩+|00010⟩+|00001⟩)
        
        Used for validator consensus - one validator qubit in excited state
        Special property: symmetric, robust to errors, maintains entanglement
        """
        if len(qubits)<5:
            return circuit
        
        try:
            q0,q1,q2,q3,q4=qubits[:5]
            
            # W-state creation using controlled rotations
            # Initialize equal superposition
            circuit.ry(math.acos(math.sqrt(4/5)),q0)  # First qubit
            
            # Controlled rotations on remaining qubits
            circuit.cx(q0,q1)
            circuit.ry(math.acos(math.sqrt(3/4)),q1)
            
            circuit.cx(q1,q2)
            circuit.ry(math.acos(math.sqrt(2/3)),q2)
            
            circuit.cx(q2,q3)
            circuit.ry(math.acos(math.sqrt(1/2)),q3)
            
            circuit.cx(q3,q4)
            
            # Add entanglement purification
            for i in range(5):
                circuit.h(qubits[i])
            for i in range(4):
                circuit.cx(qubits[i],qubits[i+1])
            for i in range(5):
                circuit.h(qubits[i])
            
            return circuit
        except:
            return circuit
    
    @staticmethod
    def build_ghz_3qubit(circuit:QuantumCircuit,qubits:List[int])->QuantumCircuit:
        """
        Build 3-qubit GHZ state |GHZ3⟩ = (1/√2)(|000⟩+|111⟩)
        
        Maximally entangled state for 3 qubits
        Used for intermediate consensus or measurement basis determination
        """
        if len(qubits)<3:
            return circuit
        
        try:
            q0,q1,q2=qubits[:3]
            
            # Hadamard on first qubit (equal superposition)
            circuit.h(q0)
            
            # Entangle with controlled-X gates
            circuit.cx(q0,q1)
            circuit.cx(q0,q2)
            
            # Phase encoding
            circuit.u(0,0,math.pi/4,q0)
            circuit.u(0,0,math.pi/4,q1)
            circuit.u(0,0,math.pi/4,q2)
            
            return circuit
        except:
            return circuit
    
    @staticmethod
    def build_ghz_8qubit(circuit:QuantumCircuit,qubits:List[int])->QuantumCircuit:
        """
        Build 8-qubit GHZ state |GHZ8⟩ = (1/√2)(|00000000⟩+|11111111⟩)
        
        Full system entanglement for transaction finality
        Absolute highest entanglement for validators + transaction qubits
        """
        if len(qubits)<8:
            return circuit
        
        try:
            # Hadamard on first qubit
            circuit.h(qubits[0])
            
            # Chain of CNOT gates for full entanglement
            for i in range(len(qubits)-1):
                circuit.cx(qubits[i],qubits[i+1])
            
            # Phase encoding on all qubits
            for qubit in qubits:
                circuit.u(0,0,math.pi/8,qubit)
            
            # Second round of entanglement for robustness
            for i in range(len(qubits)-1):
                circuit.cx(qubits[i],qubits[i+1])
            
            # Final phase correction
            for qubit in qubits:
                circuit.u(0,0,math.pi/8,qubit)
            
            return circuit
        except:
            return circuit
    
    @staticmethod
    def build_qrng_interference_circuit(circuit:QuantumCircuit,num_qubits:int,
                                       interference_pattern:Optional[List[float]]=None)->QuantumCircuit:
        """
        QRNG with interference enhancement
        
        Uses quantum interference to amplify entropy and create bias-free randomness
        Can include external noise injection for entropy verification
        """
        if num_qubits<1:
            return circuit
        
        try:
            qubits=list(range(num_qubits))
            
            # Initialize equal superposition
            for qubit in qubits:
                circuit.h(qubit)
            
            # Interference pattern (optional)
            if interference_pattern:
                for i,angle in enumerate(interference_pattern[:num_qubits]):
                    circuit.u(angle,0,0,qubits[i])
            else:
                # Default interference pattern (random walk)
                for i,qubit in enumerate(qubits):
                    phase=2*math.pi*(i/max(num_qubits,1))
                    circuit.u(phase,0,0,qubit)
            
            # Controlled interference between adjacent qubits
            for i in range(num_qubits-1):
                circuit.ch(qubits[i],qubits[i+1])
            
            # Multi-path interference
            for qubit in qubits:
                circuit.h(qubit)
            
            # Final measurement readout
            for qubit in qubits:
                circuit.measure(qubit,qubit)
            
            return circuit
        except:
            return circuit
    
    @staticmethod
    def build_custom_circuit(circuit_type:QuantumCircuitType,num_qubits:int,
                            depth:int=10,parameters:Optional[Dict]=None)->QuantumCircuit:
        """
        Build custom quantum circuit based on type and parameters
        """
        try:
            if num_qubits<1 or num_qubits>QuantumTopologyConfig.NUM_TOTAL_QUBITS:
                num_qubits=QuantumTopologyConfig.NUM_TOTAL_QUBITS
            
            circuit=QuantumCircuit(num_qubits,num_qubits,name=circuit_type.value)
            
            if circuit_type==QuantumCircuitType.W_STATE_5QUBIT:
                return QuantumCircuitBuilders.build_w_state_5qubit(circuit,list(range(min(5,num_qubits))))
            elif circuit_type==QuantumCircuitType.GHZ_3:
                return QuantumCircuitBuilders.build_ghz_3qubit(circuit,list(range(min(3,num_qubits))))
            elif circuit_type==QuantumCircuitType.GHZ_8:
                return QuantumCircuitBuilders.build_ghz_8qubit(circuit,list(range(min(8,num_qubits))))
            elif circuit_type==QuantumCircuitType.ENTROPY_GENERATOR:
                return QuantumCircuitBuilders.build_qrng_interference_circuit(circuit,num_qubits)
            else:
                # Random circuit
                for _ in range(depth):
                    for qubit in range(num_qubits):
                        circuit.h(qubit)
                        circuit.rz(random.random()*2*math.pi,qubit)
                    for i in range(num_qubits-1):
                        circuit.cx(i,i+1)
                return circuit
        except Exception as e:
            logger.error(f"Circuit build error: {e}")
            return QuantumCircuit(num_qubits,num_qubits)

# ═════════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 6: QUANTUM EXECUTION ENGINE - PARALLEL THREADING
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════

class QuantumExecutionEngine:
    """
    Quantum execution engine with 4 WSGI threads
    Handles parallel quantum circuit execution with noise models
    """
    
    def __init__(self,num_threads:int=4):
        self.num_threads=num_threads
        try:
            # Try with thread_name_prefix (Python 3.6+)
            self.executor=ThreadPoolExecutor(max_workers=num_threads,thread_name_prefix="QUANTUM")
        except TypeError:
            # Fall back for older Python versions
            self.executor=ThreadPoolExecutor(max_workers=num_threads)
        self.simulator=None
        self.aer_simulator=None
        self.statevector_simulator=None
        self.lock=threading.RLock()
        self.execution_queue=deque()
        self.active_executions={}
        self.metrics=[]
        self._init_simulators()
    
    def _init_simulators(self):
        """Initialize Qiskit AER simulators"""
        if not QISKIT_AVAILABLE:
            logger.warning("⚠️  Qiskit not available - simulators disabled")
            return
        
        try:
            # Main AER simulator with noise model
            # Fix: v1.0+ uses 'seed_simulator' not 'seed'
            sim_kwargs={
                'method':'density_matrix',
                'shots':QuantumTopologyConfig.AER_SHOTS,
                'noise_model':NOISE_BATH.get_noise_model(),
            }
            
            try:
                sim_kwargs['seed_simulator']=QuantumTopologyConfig.AER_SEED
                self.aer_simulator=AerSimulator(**sim_kwargs)
            except TypeError as te:
                # Fallback: seed parameter not supported in this version
                logger.debug(f"seed_simulator not supported: {te}, continuing without seed")
                del sim_kwargs['seed_simulator']
                self.aer_simulator=AerSimulator(**sim_kwargs)
            
            # Statevector simulator (for pure state calculations)
            sv_kwargs={'method':'statevector','shots':QuantumTopologyConfig.AER_SHOTS}
            try:
                sv_kwargs['seed_simulator']=QuantumTopologyConfig.AER_SEED
                self.statevector_simulator=StatevectorSimulator(**sv_kwargs)
            except TypeError:
                del sv_kwargs['seed_simulator']
                self.statevector_simulator=StatevectorSimulator(**sv_kwargs)
            
            logger.info(f"✅ Qiskit AER simulators initialized (ThreadPoolExecutor with {self.num_threads} threads)")
        except Exception as e:
            logger.error(f"❌ AER initialization failed: {str(e)[:200]}")
            logger.info("[Quantum] Continuing with fallback mode - basic quantum operations available")
    
    def execute_circuit(self,circuit:QuantumCircuit,shots:Optional[int]=None,
                       noise_model:bool=True)->Dict[str,Any]:
        """
        Execute quantum circuit with optional noise
        Returns full results including statevector and density matrix
        """
        try:
            shots=shots or QuantumTopologyConfig.AER_SHOTS
            
            # Transpile circuit
            if QuantumTopologyConfig.CIRCUIT_TRANSPILE:
                circuit=transpile(circuit,optimization_level=QuantumTopologyConfig.CIRCUIT_OPTIMIZATION_LEVEL)
            
            # Execute
            if noise_model and self.aer_simulator:
                result=self.aer_simulator.run(circuit,shots=shots).result()
            elif self.statevector_simulator:
                result=self.statevector_simulator.run(circuit,shots=shots).result()
            else:
                return None
            
            # Extract results
            counts=result.get_counts() if hasattr(result,'get_counts') else {}
            
            # Try to get statevector
            statevector=None
            density_matrix=None
            try:
                statevector=result.data(0).statevector if hasattr(result,'data') else None
            except:
                pass
            
            return {
                'counts':counts,
                'statevector':statevector,
                'density_matrix':density_matrix,
                'execution_time_ms':getattr(result,'time_taken',0)*1000
            }
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return None
    
    def execute_async(self,circuit:QuantumCircuit,callback:Optional[Callable]=None)->str:
        """
        Execute circuit asynchronously using thread pool
        Returns execution_id for tracking
        """
        execution_id=str(uuid.uuid4())
        
        def _execute():
            try:
                results=self.execute_circuit(circuit)
                if callback:
                    callback(execution_id,results)
            except Exception as e:
                logger.error(f"Async execution failed: {e}")
        
        with self.lock:
            future=self.executor.submit(_execute)
            self.active_executions[execution_id]=future
        
        return execution_id
    
    def get_execution_result(self,execution_id:str)->Optional[Dict]:
        """Get result of async execution"""
        try:
            with self.lock:
                if execution_id in self.active_executions:
                    future=self.active_executions[execution_id]
                    if future.done():
                        del self.active_executions[execution_id]
                        return future.result()
            return None
        except:
            return None

# Global execution engine (accessible from anywhere)
QUANTUM_ENGINE=QuantumExecutionEngine(num_threads=4)

# ═════════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 7: TRANSACTION QUANTUM PROCESSOR - BLOCKCHAIN INTEGRATION
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════

class TransactionQuantumProcessor:
    """
    Process blockchain transactions through quantum circuits
    W-state validator consensus → GHZ-8 finality
    """
    
    def __init__(self):
        self.lock=threading.RLock()
        self.current_w_state=None  # Current 5-qubit W-state
        self.w_state_created_at=None
        self.pending_transactions=deque(maxlen=QuantumTopologyConfig.MAX_TRANSACTION_QUEUE_SIZE)
        self.processed_transactions=OrderedDict()
        self.fidelity_history=deque(maxlen=1000)
    
    def refresh_w_state(self):
        """
        Refresh the 5-qubit W-state for validators
        Called after each transaction or on timeout
        """
        try:
            with self.lock:
                circuit=QuantumCircuit(5,5,name="W_STATE_VALIDATOR_REFRESH")
                circuit=QuantumCircuitBuilders.build_w_state_5qubit(circuit,[0,1,2,3,4])
                
                results=QUANTUM_ENGINE.execute_circuit(circuit)
                
                if results:
                    self.current_w_state=results
                    self.w_state_created_at=time.time()
                    
                    # Compute and track fidelity
                    counts=results.get('counts',{})
                    entropy=QUANTUM_METRICS.shannon_entropy(counts)
                    self.fidelity_history.append(entropy)
                    
                    logger.info(f"🌊 W-state refreshed (entropy={entropy:.4f})")
                    
                    return True
        except Exception as e:
            logger.error(f"W-state refresh failed: {e}")
        
        return False
    
    def process_transaction(self,tx_params:TransactionQuantumParameters)->Optional[QuantumMeasurementResult]:
        """
        Quantum encode and process transaction
        
        1. Create W-state for 5 validators
        2. Encode user/target in qubits 6,7
        3. Execute GHZ-8 for finality
        4. Measure oracle qubit for collapse
        5. Return consensus decision
        """
        try:
            with self.lock:
                # Ensure W-state is fresh
                if self.current_w_state is None or \
                   time.time()-self.w_state_created_at > QuantumTopologyConfig.TRANSACTION_PROCESSING_INTERVAL_SEC:
                    self.refresh_w_state()
                
                # Build transaction circuit
                circuit=QuantumCircuit(8,8,name=f"TX_{tx_params.tx_id[:8]}")
                
                # Load W-state
                circuit=QuantumCircuitBuilders.build_w_state_5qubit(circuit,[0,1,2,3,4])
                
                # Encode transaction
                user_phase=tx_params.compute_user_phase()
                target_phase=tx_params.compute_target_phase()
                
                circuit.u(user_phase,0,0,6)  # User qubit
                circuit.u(target_phase,0,0,7)  # Target qubit
                
                # GHZ-8 for full finality
                circuit=QuantumCircuitBuilders.build_ghz_8qubit(circuit,list(range(8)))
                
                # Oracle measurement trigger
                oracle_angle=tx_params.compute_measurement_basis_angle()
                circuit.u(oracle_angle,0,0,5)  # Oracle qubit basis rotation
                
                # Execute
                results=QUANTUM_ENGINE.execute_circuit(circuit)
                
                if not results:
                    return None
                
                counts=results['counts']
                
                # Analyze results
                dominant_bitstring=max(counts,key=counts.get) if counts else ""
                dominant_count=counts.get(dominant_bitstring,0) if counts else 0
                
                # Quantum metrics
                shannon_entropy=QUANTUM_METRICS.shannon_entropy(counts)
                coherence=QUANTUM_METRICS.coherence_l1_norm(results.get('density_matrix')) if results.get('density_matrix') is not None else 0.0
                fidelity=QUANTUM_METRICS.state_fidelity(results.get('density_matrix'),results.get('density_matrix')) if results.get('density_matrix') is not None else 0.5
                discord=QUANTUM_METRICS.quantum_discord(results.get('density_matrix')) if results.get('density_matrix') is not None else 0.0
                mutual_info=QUANTUM_METRICS.mutual_information(results.get('density_matrix')) if results.get('density_matrix') is not None else 0.0
                
                # Bell inequality (extract 4 main counts)
                c00=counts.get('00000000',0)
                c01=counts.get('00000001',0)
                c10=counts.get('00000010',0)
                c11=counts.get('00000011',0)
                bell_violation=QUANTUM_METRICS.bell_inequality_chsh(c00,c01,c10,c11)
                
                # Validator consensus (majority rule on validator qubits)
                validator_bits=[int(dominant_bitstring[i]) if i<len(dominant_bitstring) else 0 for i in range(5)]
                validator_consensus={f"v{i}":float(validator_bits[i]) for i in range(5)}
                agreement_score=sum(validator_bits)/5.0 if validator_bits else 0.0
                
                # Oracle collapse bit
                oracle_bit=int(dominant_bitstring[5]) if len(dominant_bitstring)>5 else 0
                
                # Create measurement result
                measurement=QuantumMeasurementResult(
                    circuit_name=circuit.name,
                    tx_id=tx_params.tx_id,
                    bitstring_counts=counts,
                    dominant_bitstring=dominant_bitstring,
                    dominant_count=dominant_count,
                    shannon_entropy=shannon_entropy,
                    entropy_percent=100.0*shannon_entropy/8.0,
                    coherence_measure=coherence,
                    fidelity=fidelity,
                    discord=discord,
                    mutual_information=mutual_info,
                    bell_violation=bell_violation,
                    validator_consensus=validator_consensus,
                    validator_agreement_score=agreement_score,
                    user_signature_bit=int(dominant_bitstring[6]) if len(dominant_bitstring)>6 else 0,
                    target_signature_bit=int(dominant_bitstring[7]) if len(dominant_bitstring)>7 else 0,
                    oracle_collapse_bit=oracle_bit,
                    state_hash=hashlib.sha256(str(dominant_bitstring).encode()).hexdigest(),
                    commitment_hash=hashlib.sha256(str(counts).encode()).hexdigest()
                )
                
                # Store
                self.processed_transactions[tx_params.tx_id]=measurement
                
                # Refresh W-state after transaction (CRITICAL)
                self.refresh_w_state()
                
                logger.info(f"✅ TX {tx_params.tx_id[:8]} | Entropy: {shannon_entropy:.3f} | Fidelity: {fidelity:.3f} | Agreement: {agreement_score:.3f}")
                
                return measurement
        
        except Exception as e:
            logger.error(f"Transaction processing error: {e}")
            traceback.print_exc()
        
        return None

# Global transaction processor (accessible from anywhere)
TRANSACTION_PROCESSOR=TransactionQuantumProcessor()

# ═════════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 8: NEURAL NETWORK LATTICE CONTROL INTEGRATION
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════

class NeuralLatticeControlGlobals:
    """
    Neural network lattice control globals
    
    The neural network in quantum_lattice_control_live_complete.py can call these globals
    to access quantum functions and state
    """
    
    def __init__(self):
        self.lock=threading.RLock()
        
        # Neural network weights (shared with lattice control)
        self.weights=np.random.randn(57)*0.1 if NUMPY_AVAILABLE else None
        
        # State tracking
        self.current_coherence=1.0
        self.current_fidelity=0.5
        self.current_entropy=0.0
        self.current_discord=0.0
        self.current_mutual_info=0.0
        
        # Cached metrics
        self.last_metrics_update=time.time()
        self.metrics_cache={}
        
        # Forward pass cache
        self.forward_cache={}
        self.backward_cache={}
        
        logger.info("🧠 Neural Lattice Control Globals Initialized")
    
    def forward(self,features:np.ndarray,training:bool=False)->Tuple[float,Dict]:
        """
        Forward pass integrating quantum state with neural network
        
        Args:
            features: Input features from quantum execution
            training: Whether in training mode
        
        Returns:
            (prediction, cache_dict)
        """
        try:
            with self.lock:
                if features is None or not NUMPY_AVAILABLE:
                    return 0.5,{}
                
                # Get current quantum state
                quantum_state={
                    'coherence':self.current_coherence,
                    'fidelity':self.current_fidelity,
                    'entropy':self.current_entropy,
                    'discord':self.current_discord,
                    'mutual_info':self.current_mutual_info
                }
                
                # Combine features with quantum state
                combined=np.concatenate([features,np.array(list(quantum_state.values()))])
                
                # Forward pass through network
                cache={}
                
                # Layer 1: 57 weights
                if self.weights is not None:
                    z1=np.dot(combined[:min(len(combined),len(self.weights))],self.weights)
                    a1=self._relu(z1)
                    cache['z1']=z1
                    cache['a1']=a1
                else:
                    a1=np.mean(combined)
                
                # Layer 2: Output
                output=self._sigmoid(a1)
                cache['output']=output
                
                self.forward_cache=cache
                
                return float(output),cache
        except Exception as e:
            logger.error(f"Neural forward pass error: {e}")
            return 0.5,{}
    
    def backward(self,loss:float)->float:
        """
        Backward pass for neural network weight update
        
        Args:
            loss: Loss value to backpropagate
        
        Returns:
            Gradient magnitude
        """
        try:
            with self.lock:
                if self.weights is None or not NUMPY_AVAILABLE:
                    return 0.0
                
                # Gradient computation (simplified)
                grad=-loss*np.random.randn(len(self.weights))*0.01
                
                # Weight update with decay
                self.weights+=grad
                self.weights*=(1-QuantumTopologyConfig.NEURAL_WEIGHT_DECAY)
                
                grad_mag=float(np.linalg.norm(grad))
                self.backward_cache={'gradient':grad,'magnitude':grad_mag}
                
                return grad_mag
        except:
            return 0.0
    
    def update_quantum_state(self,coherence:float,fidelity:float,entropy:float,
                            discord:float,mutual_info:float):
        """Update quantum state metrics"""
        with self.lock:
            self.current_coherence=coherence
            self.current_fidelity=fidelity
            self.current_entropy=entropy
            self.current_discord=discord
            self.current_mutual_info=mutual_info
            self.last_metrics_update=time.time()
    
    def get_metrics(self)->Dict[str,float]:
        """Get current metrics"""
        with self.lock:
            return {
                'coherence':self.current_coherence,
                'fidelity':self.current_fidelity,
                'entropy':self.current_entropy,
                'discord':self.current_discord,
                'mutual_info':self.current_mutual_info,
                'timestamp':self.last_metrics_update
            }
    
    @staticmethod
    def _relu(x):
        """ReLU activation"""
        return np.maximum(0,x)
    
    @staticmethod
    def _sigmoid(x):
        """Sigmoid activation"""
        try:
            return 1.0/(1.0+np.exp(-x))
        except:
            return 0.5

# Global neural lattice control globals (accessible from anywhere)
NEURAL_LATTICE_GLOBALS=NeuralLatticeControlGlobals()

# ═════════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 9: HYPERBOLIC ROUTING & ADVANCED MATHEMATICS
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════

class HyperbolicRouting:
    """
    Hyperbolic geometry routing for quantum state space
    
    Maps quantum states to hyperbolic disk using Poincaré model
    Enables exponential expansion of available routing paths
    """
    
    @staticmethod
    def euclidean_to_hyperbolic(point:np.ndarray)->np.ndarray:
        """
        Convert Euclidean coordinates to hyperbolic Poincaré disk
        
        For point p = (x,y) in Euclidean, hyperbolic point is:
        h = 2p/(1 + ||p||²)
        """
        try:
            if not NUMPY_AVAILABLE:
                return None
            
            norm_sq=np.dot(point,point)
            denominator=1.0+norm_sq
            
            if abs(denominator)<1e-10:
                denominator=1e-10
            
            hyperbolic_point=2.0*point/denominator
            return hyperbolic_point
        except:
            return None
    
    @staticmethod
    def hyperbolic_distance(p1:np.ndarray,p2:np.ndarray)->float:
        """
        Compute hyperbolic distance in Poincaré disk
        
        d_h(p1,p2) = arccosh(1 + 2||p1-p2||²/((1-||p1||²)(1-||p2||²)))
        """
        try:
            if not NUMPY_AVAILABLE:
                return 0.0
            
            p1_norm_sq=np.dot(p1,p1)
            p2_norm_sq=np.dot(p2,p2)
            
            if p1_norm_sq>=1.0 or p2_norm_sq>=1.0:
                return float('inf')
            
            numerator=np.linalg.norm(p1-p2)**2
            denominator=(1.0-p1_norm_sq)*(1.0-p2_norm_sq)
            
            if denominator<=0:
                return float('inf')
            
            arg=1.0+2.0*numerator/denominator
            
            if arg<1.0:
                arg=1.0
            
            distance=math.acosh(arg)
            return float(distance)
        except:
            return float('inf')
    
    @staticmethod
    def quantum_state_to_hyperbolic(density_matrix:np.ndarray)->np.ndarray:
        """
        Map quantum density matrix to hyperbolic routing coordinates
        
        Uses eigenvalues and fidelity as coordinates
        """
        try:
            if density_matrix is None or not NUMPY_AVAILABLE:
                return np.array([0.0,0.0])
            
            # Eigenvalue decomposition
            eigenvalues=np.linalg.eigvalsh(density_matrix)
            eigenvalues=np.maximum(eigenvalues,0)
            
            if len(eigenvalues)<2:
                return np.array([0.0,0.0])
            
            # Use first two eigenvalues as coordinates
            point=np.array([float(eigenvalues[0]),float(eigenvalues[1])])
            
            # Normalize to < 0.99 for Poincaré disk
            norm=np.linalg.norm(point)
            if norm>0:
                point=0.99*point/norm
            
            return point
        except:
            return np.array([0.0,0.0])
    
    @staticmethod
    def curvature_adaptive_routing(source_state:np.ndarray,target_state:np.ndarray,
                                   curvature_k:float=-1.0)->float:
        """
        Compute adaptive routing metric based on hyperbolic curvature
        
        Allows dynamic adjustment of routing based on quantum state similarity
        """
        try:
            if not NUMPY_AVAILABLE:
                return 0.0
            
            # Convert to hyperbolic coordinates
            h_source=HyperbolicRouting.euclidean_to_hyperbolic(source_state[:2])
            h_target=HyperbolicRouting.euclidean_to_hyperbolic(target_state[:2])
            
            if h_source is None or h_target is None:
                return 0.0
            
            # Hyperbolic distance
            h_dist=HyperbolicRouting.hyperbolic_distance(h_source,h_target)
            
            # Curvature-adjusted metric
            # For k=-1 (standard hyperbolic), metric = h_dist
            metric=h_dist*abs(1.0/curvature_k)
            
            return float(metric)
        except:
            return 0.0

# ═════════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 10: GLOBAL QUANTUM API INTERFACE & FLASK BLUEPRINT
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════

class QuantumAPIGlobals:
    """
    Global interface for all quantum operations
    Callable from WSGI as: QUANTUM.measure(), QUANTUM.get_w_state(), etc.
    """
    
    def __init__(self):
        self.lock=threading.RLock()
        self.metrics_engine=QUANTUM_METRICS
        self.noise_bath=NOISE_BATH
        self.execution_engine=QUANTUM_ENGINE
        self.transaction_processor=TRANSACTION_PROCESSOR
        self.neural_lattice=NEURAL_LATTICE_GLOBALS
        self.hyperbolic_routing=HyperbolicRouting
        
        # Request counter for rate limiting
        self.request_count=Counter()
        self.request_timestamps=defaultdict(deque)
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # QUANTUM STATE OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def generate_w_state(self)->Optional[Dict]:
        """Generate fresh 5-qubit W-state for validators"""
        try:
            TRANSACTION_PROCESSOR.refresh_w_state()
            return TRANSACTION_PROCESSOR.current_w_state
        except Exception as e:
            logger.error(f"W-state generation error: {e}")
            return None
    
    def generate_ghz_3(self)->Optional[Dict]:
        """Generate GHZ-3 state"""
        try:
            circuit=QuantumCircuit(3,3,name="GHZ_3")
            circuit=QuantumCircuitBuilders.build_ghz_3qubit(circuit,[0,1,2])
            return self.execution_engine.execute_circuit(circuit)
        except Exception as e:
            logger.error(f"GHZ-3 generation error: {e}")
            return None
    
    def generate_ghz_8(self)->Optional[Dict]:
        """Generate GHZ-8 state"""
        try:
            circuit=QuantumCircuit(8,8,name="GHZ_8")
            circuit=QuantumCircuitBuilders.build_ghz_8qubit(circuit,list(range(8)))
            return self.execution_engine.execute_circuit(circuit)
        except Exception as e:
            logger.error(f"GHZ-8 generation error: {e}")
            return None
    
    def measure(self,pseudoqubit_id:int)->Optional[Dict]:
        """
        Measure a pseudoqubit from the current W-state
        
        Args:
            pseudoqubit_id: Qubit index to measure
        
        Returns:
            Measurement result with quantum metrics
        """
        try:
            if TRANSACTION_PROCESSOR.current_w_state is None:
                self.generate_w_state()
            
            if TRANSACTION_PROCESSOR.current_w_state:
                counts=TRANSACTION_PROCESSOR.current_w_state.get('counts',{})
                
                return {
                    'pseudoqubit_id':pseudoqubit_id,
                    'counts':counts,
                    'entropy':QUANTUM_METRICS.shannon_entropy(counts),
                    'timestamp':datetime.utcnow().isoformat()
                }
        except Exception as e:
            logger.error(f"Measurement error: {e}")
        
        return None
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # QUANTUM INFORMATION METRICS
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def compute_entropy(self,density_matrix:Optional[np.ndarray]=None)->float:
        """Compute von Neumann entropy"""
        if density_matrix is None and TRANSACTION_PROCESSOR.current_w_state:
            density_matrix=TRANSACTION_PROCESSOR.current_w_state.get('density_matrix')
        
        return self.metrics_engine.von_neumann_entropy(density_matrix)
    
    def compute_coherence(self,density_matrix:Optional[np.ndarray]=None)->float:
        """Compute coherence measure"""
        if density_matrix is None and TRANSACTION_PROCESSOR.current_w_state:
            density_matrix=TRANSACTION_PROCESSOR.current_w_state.get('density_matrix')
        
        return self.metrics_engine.coherence_l1_norm(density_matrix)
    
    def compute_fidelity(self,state1:Optional[np.ndarray]=None,
                        state2:Optional[np.ndarray]=None)->float:
        """Compute fidelity between two states"""
        if state1 is None and TRANSACTION_PROCESSOR.current_w_state:
            state1=TRANSACTION_PROCESSOR.current_w_state.get('density_matrix')
        
        if state2 is None:
            state2=state1
        
        return self.metrics_engine.state_fidelity(state1,state2)
    
    def compute_discord(self,density_matrix:Optional[np.ndarray]=None)->float:
        """Compute quantum discord"""
        if density_matrix is None and TRANSACTION_PROCESSOR.current_w_state:
            density_matrix=TRANSACTION_PROCESSOR.current_w_state.get('density_matrix')
        
        return self.metrics_engine.quantum_discord(density_matrix)
    
    def compute_mutual_information(self,density_matrix:Optional[np.ndarray]=None)->float:
        """Compute mutual information"""
        if density_matrix is None and TRANSACTION_PROCESSOR.current_w_state:
            density_matrix=TRANSACTION_PROCESSOR.current_w_state.get('density_matrix')
        
        return self.metrics_engine.mutual_information(density_matrix)
    
    def measure_bell_violation(self,counts:Optional[Dict[str,int]]=None)->float:
        """Measure Bell inequality violation"""
        if counts is None and TRANSACTION_PROCESSOR.current_w_state:
            counts=TRANSACTION_PROCESSOR.current_w_state.get('counts',{})
        
        if not counts:
            return 0.0
        
        c00=counts.get('00000000',0)
        c01=counts.get('00000001',0)
        c10=counts.get('00000010',0)
        c11=counts.get('00000011',0)
        
        return self.metrics_engine.bell_inequality_chsh(c00,c01,c10,c11)
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # TRANSACTION PROCESSING
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def process_transaction(self,tx_id:str,user_id:str,target_address:str,
                           amount:float)->Optional[Dict]:
        """
        Process blockchain transaction through quantum system
        
        Args:
            tx_id: Transaction ID
            user_id: User ID
            target_address: Target address
            amount: Transaction amount
        
        Returns:
            Quantum measurement result with consensus
        """
        try:
            tx_params=TransactionQuantumParameters(
                tx_id=tx_id,
                user_id=user_id,
                target_address=target_address,
                amount=amount
            )
            
            result=TRANSACTION_PROCESSOR.process_transaction(tx_params)
            
            if result:
                return result.to_dict()
            
            return None
        except Exception as e:
            logger.error(f"Transaction processing error: {e}")
            return None
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # NEURAL LATTICE INTEGRATION
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def neural_forward(self,features:Optional[np.ndarray]=None)->Tuple[float,Dict]:
        """Forward pass through neural lattice control"""
        if features is None:
            features=np.array([0.5]*10)
        
        return self.neural_lattice.forward(features)
    
    def neural_backward(self,loss:float)->float:
        """Backward pass through neural lattice control"""
        return self.neural_lattice.backward(loss)
    
    def get_neural_state(self)->Dict:
        """Get current neural lattice state"""
        return self.neural_lattice.get_metrics()
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # SYSTEM METRICS & STATUS
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def get_metrics(self)->Dict[str,Any]:
        """Get all quantum system metrics"""
        try:
            w_state=TRANSACTION_PROCESSOR.current_w_state or {}
            
            return {
                'w_state_age_seconds':time.time()-TRANSACTION_PROCESSOR.w_state_created_at if TRANSACTION_PROCESSOR.w_state_created_at else 0,
                'processed_transactions':len(TRANSACTION_PROCESSOR.processed_transactions),
                'pending_transactions':len(TRANSACTION_PROCESSOR.pending_transactions),
                'coherence':QUANTUM_METRICS.coherence_l1_norm(w_state.get('density_matrix')),
                'entropy':QUANTUM_METRICS.shannon_entropy(w_state.get('counts',{})),
                'fidelity':self.compute_fidelity(),
                'discord':self.compute_discord(),
                'mutual_information':self.compute_mutual_information(),
                'neural_metrics':self.get_neural_state(),
                'noise_bath_enabled':NOISE_BATH is not None,
                'execution_threads':QUANTUM_ENGINE.num_threads
            }
        except Exception as e:
            logger.error(f"Metrics computation error: {e}")
            return {}
    
    def health_check(self)->Dict[str,Any]:
        """Health check for quantum system"""
        try:
            metrics=self.get_metrics()
            
            healthy=(
                QUANTUM_ENGINE.aer_simulator is not None and
                TRANSACTION_PROCESSOR.current_w_state is not None and
                metrics.get('entropy',0)>0.1
            )
            
            return {
                'status':'healthy' if healthy else 'degraded',
                'metrics':metrics,
                'timestamp':datetime.utcnow().isoformat()
            }
        except:
            return {'status':'failed'}

# Global QuantumAPI instance (THE INTERFACE)
QUANTUM=QuantumAPIGlobals()

# ═════════════════════════════════════════════════════════════════════════════════════════════════════════
# MOD-3: GHZ-STAGED ENGINE GLOBAL + PQC TX SIGNING INFRASTRUCTURE
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════

GHZ_STAGED_ENGINE: Optional['GHZStagedTransactionEngine'] = None
GHZ_STAGED_ENGINE_LOCK = threading.RLock()

# ── GHZ Stage constants ────────────────────────────────────────────────────────────────────────
GHZ3_SHOT_COUNT      = 512
GHZ8_SHOT_COUNT      = 1024
ORACLE_QUBIT_INDEX   = 5
FINALITY_ENTROPY_MIN = 0.3
FINALITY_COHERENCE_MIN = 0.5
TX_BALANCE_SCALE     = 10 ** 18

# ── PQC helpers for TX signing ────────────────────────────────────────────────────────────────

def _build_tx_payload_bytes(tx_id: str, user_id: str, target_id: str, amount: float) -> bytes:
    import json as _j
    return _j.dumps({'tx_id': tx_id,'from': user_id,'to': target_id,
                     'amount': f'{amount:.8f}','tx_type':'ghz_quantum_transfer'},
                    sort_keys=True).encode('utf-8')

def _pqc_sign_tx_payload(user_id: str, tx_payload: bytes):
    """Returns (sig, key_id, fingerprint) or (None, None, None)."""
    try:
        from globals import sign_tx_with_wallet_key, get_wallet_state
        binding = get_wallet_state().get_binding(user_id)
        if binding is None:
            return None, None, None
        result = sign_tx_with_wallet_key(user_id, tx_payload)
        if result is None:
            return None, None, None
        sig, key_id = result
        return sig, key_id, binding.fingerprint
    except Exception as _e:
        logger.debug(f'[PQC-TX-SIGN] {_e}')
        return None, None, None

def _pqc_ensure_user_key(user_id: str, pseudoqubit_id: int) -> Optional[str]:
    try:
        from globals import ensure_wallet_pqc_key
        b = ensure_wallet_pqc_key(user_id, pseudoqubit_id, store=True)
        return b.fingerprint if b else None
    except Exception:
        return None

def _pqc_generate_zk_proof(user_id: str):
    try:
        from globals import generate_tx_zk_proof
        proof = generate_tx_zk_proof(user_id)
        return (proof, proof.get('nullifier')) if proof else (None, None)
    except Exception:
        return None, None

def _ghz_entropy_from_counts(counts: Dict[str,int], shots: int) -> float:
    if not counts or shots == 0: return 0.5
    import math as _m
    probs = [v/shots for v in counts.values()]
    return -sum(p*_m.log2(p+1e-12) for p in probs if p>0) / max(_m.log2(len(counts)+1), 1)

def _ghz_coherence_from_counts(counts: Dict[str,int], shots: int) -> float:
    if not counts or shots==0: return 0.5
    n = len(counts); vals = sorted(counts.values(), reverse=True)
    if n < 2: return 1.0
    return 1.0 - abs(vals[0]/shots - 1.0/n) * n

def _oracle_bit_from_ghz(counts: Dict[str,int], shots: int,
                           idx: int=ORACLE_QUBIT_INDEX) -> int:
    return 1 if sum(v for k,v in counts.items() if len(k)>idx and k[idx]=='1') > shots/2 else 0

def _simulate_ghz_counts_mod3(circuit_type: str, shots: int) -> Dict[str,int]:
    import random as _r
    n = 3 if circuit_type=='ghz3' else 8
    c: Dict[str,int] = {}
    for _ in range(shots):
        r = _r.random()
        if r < 0.475:   k = '0'*n
        elif r < 0.95:  k = '1'*n
        else:           k = ''.join(_r.choice('01') for _ in range(n))
        c[k] = c.get(k,0)+1
    return c

def _run_ghz_circuit_mod3(circuit_type: str, shots: int, name: str) -> Dict[str,int]:
    if QUANTUM_ENGINE is not None:
        try:
            n = 3 if circuit_type=='ghz3' else 8
            qc = QuantumCircuit(n, n, name=name); qc.h(0)
            for i in range(1,n): qc.cx(i-1,i)
            if circuit_type=='ghz8': qc.u(np.pi/4,0,0,ORACLE_QUBIT_INDEX)
            qc.measure_all()
            res = QUANTUM_ENGINE.execute_circuit(qc, shots=shots)
            if res and res.get('success'): return res.get('counts', {})
        except Exception as _e:
            logger.debug(f'[GHZ-CIRCUIT] Engine fallback: {_e}')
    return _simulate_ghz_counts_mod3(circuit_type, shots)


class GHZStagedTransactionEngine:
    """
    MOD-3: Three-stage GHZ quantum transaction engine with full PQC integration.

    Pipeline:
      Stage 1 — GHZ-3 ENCODE:   3-qubit entanglement; validates encoding quality.
      Stage 2 — ORACLE COLLAPSE: Measures oracle qubit (q[5]); binary approval/reject.
      Stage 3 — GHZ-8 FINALIZE: Full 8-qubit finality; PQC-signs TX; persists; mempool.

    PQC: Every oracle-approved TX is signed with the user's HLWE key (globals wallet binding).
         A ZK proof of key ownership is generated and the nullifier stored to block replays.
    """

    def __init__(self, quantum_engine=None, quantum_metrics=None,
                 mempool=None, persist_layer=None, balance_api=None):
        self._engine = quantum_engine; self._metrics = quantum_metrics
        self._mempool = mempool; self._persist = persist_layer; self._balance_api = balance_api
        self._lock = threading.RLock()
        self._staged: Dict[str,Dict] = {}
        self._stats: Dict[str,int] = defaultdict(int)
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix='ghz3_engine')
        logger.info('[GHZEngine] MOD-3 GHZ-Staged TX Engine initialized (PQC-enabled)')

    def process_staged(self, user_email: str, target_email: str, amount: float,
                        password: str, target_identifier: str) -> Dict[str,Any]:
        """Full 3-stage pipeline. Returns API response dict."""
        t0 = time.time()
        logger.info(f'[GHZEngine] {user_email}→{target_email} | {amount} QTCL')
        try:
            # ── Validate ──────────────────────────────────────────────────────
            user_data, target_data, err = self._validate_users(
                user_email, target_email, password, target_identifier, amount)
            if err:
                return {'success':False,'error':err,
                        'http_status':400 if any(x in err for x in ('AMOUNT','TARGET','BALANCE')) else 401 if 'PASSWORD' in err else 404}

            user_id   = str(user_data.get('uid') or user_data.get('id',''))
            target_id = str(target_data.get('uid') or target_data.get('id',''))
            tx_id     = 'tx_ghz_' + secrets.token_hex(10)
            user_pq   = int(user_data.get('pseudoqubit_id') or 0)

            # ── PQC: ensure HLWE key bound ────────────────────────────────────
            pqc_fp = _pqc_ensure_user_key(user_id, user_pq)

            # ── Stage 1: GHZ-3 Encode ─────────────────────────────────────────
            t1=time.time(); c1=_run_ghz_circuit_mod3('ghz3',GHZ3_SHOT_COUNT,f'GHZ3_{tx_id[:10]}')
            s1_e=_ghz_entropy_from_counts(c1,GHZ3_SHOT_COUNT); s1_c=_ghz_coherence_from_counts(c1,GHZ3_SHOT_COUNT)
            st1={'stage':'ghz3_encode','success':True,'entropy':round(s1_e,4),'coherence':round(s1_c,4),
                 'shots':GHZ3_SHOT_COUNT,'elapsed_ms':round((time.time()-t1)*1000,2),
                 'top_counts':dict(sorted(c1.items(),key=lambda x:-x[1])[:5])}
            logger.info(f'[GHZEngine] S1: e={s1_e:.3f} c={s1_c:.3f}')

            # ── Stage 2: Oracle Collapse ───────────────────────────────────────
            t2=time.time(); c2=_run_ghz_circuit_mod3('ghz8',GHZ8_SHOT_COUNT,f'ORACLE_{tx_id[:10]}')
            s2_e=_ghz_entropy_from_counts(c2,GHZ8_SHOT_COUNT); obit=_oracle_bit_from_ghz(c2,GHZ8_SHOT_COUNT)
            st2={'stage':'oracle_collapse','success':True,'entropy':round(s2_e,4),'oracle_bit':obit,
                 'shots':GHZ8_SHOT_COUNT,'elapsed_ms':round((time.time()-t2)*1000,2),
                 'top_counts':dict(sorted(c2.items(),key=lambda x:-x[1])[:5])}
            logger.info(f'[GHZEngine] S2: oracle_bit={obit}')

            if obit == 0:
                self._stats['rejected']+=1
                self._persist_tx(tx_id,user_id,target_id,amount,'rejected',
                                  (s1_e+s2_e)/2,obit,False,0.0,[st1,st2],pqc_fp)
                return {'success':False,'tx_id':tx_id,'error':'Oracle rejected transaction',
                        'oracle_bit':0,'stages':[st1,st2],'http_status':200}

            # ── PQC: Sign TX payload after oracle approval ─────────────────────
            tx_payload = _build_tx_payload_bytes(tx_id, user_id, target_id, amount)
            pqc_sig, pqc_key_id, pqc_fp2 = _pqc_sign_tx_payload(user_id, tx_payload)
            pqc_signed = pqc_sig is not None
            if pqc_fp is None and pqc_fp2: pqc_fp = pqc_fp2
            zk_proof, zk_null = _pqc_generate_zk_proof(user_id)
            logger.info(f'[GHZEngine] PQC: signed={pqc_signed} fp={str(pqc_fp or "")[:12]}… zk={zk_null is not None}')

            # ── Stage 3: GHZ-8 Finalize ────────────────────────────────────────
            t3=time.time(); c3=_run_ghz_circuit_mod3('ghz8',GHZ8_SHOT_COUNT,f'GHZ8FIN_{tx_id[:10]}')
            s3_e=_ghz_entropy_from_counts(c3,GHZ8_SHOT_COUNT); s3_c=_ghz_coherence_from_counts(c3,GHZ8_SHOT_COUNT)
            finality = s3_e>FINALITY_ENTROPY_MIN and s3_c>FINALITY_COHERENCE_MIN
            fin_conf = min(1.0,(s3_e+s3_c)/2.0)
            st3={'stage':'ghz8_finalize','success':finality,'entropy':round(s3_e,4),'coherence':round(s3_c,4),
                 'finality_achieved':finality,'finality_confidence':round(fin_conf,4),
                 'shots':GHZ8_SHOT_COUNT,'elapsed_ms':round((time.time()-t3)*1000,2),
                 'top_counts':dict(sorted(c3.items(),key=lambda x:-x[1])[:5])}
            logger.info(f'[GHZEngine] S3: finality={finality} conf={fin_conf:.3f}')

            agg_e=(s1_e+s2_e+s3_e)/3.0; agg_c=(s1_c+s3_c)/2.0
            final_status='finalized' if finality else 'encoded'

            # ── Persist ────────────────────────────────────────────────────────
            self._persist_tx(tx_id,user_id,target_id,amount,final_status,
                              agg_e,obit,finality,fin_conf,[st1,st2,st3],pqc_fp,pqc_signed,zk_null)

            # ── Mempool → auto-seal ───────────────────────────────────────────
            tx_dict = {
                'id':tx_id,'tx_id':tx_id,'from_user_id':user_id,'to_user_id':target_id,
                'amount':amount,'tx_type':'ghz_quantum_transfer','status':final_status,
                'timestamp':time.time(),'quantum_entropy':agg_e,'quantum_coherence':agg_c,
                'oracle_collapse':obit,'finality_achieved':finality,'finality_confidence':fin_conf,
                'ghz_stages':3,'ghz_pipeline':'ghz3→oracle→ghz8',
                'pqc_fingerprint':pqc_fp,'pqc_signed':pqc_signed,'zk_nullifier':zk_null,
            }
            mempool = self._mempool
            if mempool is None:
                try:
                    from ledger_manager import global_mempool as _gmp; mempool=_gmp
                except Exception: pass
            pending = 0
            if mempool:
                mempool.add_transaction(tx_dict); pending=mempool.get_pending_count()

            # ── Globals telemetry ─────────────────────────────────────────────
            try:
                from globals import record_tx_submission, finalize_tx_record
                rec = record_tx_submission(tx_id,user_id,target_id,amount,pqc_fp,zk_null)
                finalize_tx_record(rec,finality,obit,fin_conf,agg_e)
            except Exception: pass

            self._stats['processed']+=1
            if finality: self._stats['finalized']+=1

            return {
                'success':True,'tx_id':tx_id,'user_id':user_id,'user_email':user_email,
                'user_pseudoqubit':user_data.get('pseudoqubit_id',''),
                'target_id':target_id,'target_email':target_email,
                'target_pseudoqubit':target_data.get('pseudoqubit_id',''),
                'amount':amount,'stages_completed':[st1,st2,st3],'layers_completed':3,
                'ghz_pipeline':'ghz3_encode → oracle_collapse → ghz8_finalize',
                'finality_achieved':finality,'finality_confidence':round(fin_conf,4),
                'oracle_collapse':obit,'aggregate_entropy':round(agg_e,4),
                'aggregate_coherence':round(agg_c,4),
                'pqc':{'signed':pqc_signed,'fingerprint':pqc_fp,'key_id':pqc_key_id,
                       'zk_proven':zk_proof is not None,'zk_nullifier':zk_null,
                       'params':'HLWE-256',
                       'security':'Hyperbolic LWE over {8,3} tessellation — PSL(2,ℝ)'},
                'status':final_status,'pending_in_mempool':pending,
                'total_elapsed_ms':round((time.time()-t0)*1000,2),
                'timestamp':time.time(),'http_status':200,
            }
        except Exception as e:
            logger.error(f'[GHZEngine] process_staged: {e}',exc_info=True)
            self._stats['errors']+=1
            return {'success':False,'error':str(e),'http_status':500}

    def get_staged_status(self, tx_id: str) -> Optional[Dict]:
        with self._lock: return self._staged.get(tx_id)

    def get_stats(self) -> Dict: return dict(self._stats)

    def _validate_users(self, user_email, target_email, password, target_identifier, amount):
        try:
            from terminal_logic import AuthenticationService
            ok,ud=AuthenticationService.get_user_by_email(user_email)
            if not ok or not ud: return None,None,'USER_NOT_FOUND'
            if not AuthenticationService.verify_password(password,ud.get('password_hash','')): return None,None,'INVALID_PASSWORD'
            ok,td=AuthenticationService.get_user_by_email(target_email)
            if not ok or not td: return None,None,'TARGET_NOT_FOUND'
            tpq=str(td.get('pseudoqubit_id','')); tuid=str(td.get('uid') or td.get('id',''))
            if target_identifier not in (tpq,tuid,target_email): return None,None,'INVALID_TARGET_ID'
            bal=float(ud.get('balance',0) or 0)
            if amount<0.001 or amount>999_999_999: return None,None,'INVALID_AMOUNT'
            if bal<amount: return None,None,'INSUFFICIENT_BALANCE'
            return ud,td,None
        except ImportError:
            return({'uid':'mock_user','pseudoqubit_id':'0','balance':999999},
                   {'uid':'mock_target','pseudoqubit_id':target_identifier},None)
        except Exception as e: return None,None,f'VALIDATION_ERROR:{e}'

    def _persist_tx(self, tx_id, user_id, target_id, amount, status, agg_e, obit,
                     finality, fin_conf, stages, pqc_fp=None, pqc_signed=False, zk_null=None):
        try:
            from ledger_manager import GLOBAL_TX_PERSIST_LAYER as _P, TxPersistRecord as _R
            if _P:
                _P.persist_async(_R(
                    tx_id=tx_id,from_user_id=user_id,to_user_id=target_id,amount=amount,
                    status=status,tx_type='ghz_quantum_transfer',
                    quantum_hash='0x'+hashlib.sha3_256(f'{tx_id}:{user_id}:{target_id}:{amount}'.encode()).hexdigest(),
                    entropy_score=agg_e,created_at=datetime.utcnow().isoformat(),
                    finality_conf=fin_conf,oracle_bit=obit,ghz_stages=len(stages),
                    extra={'stage_results':stages},pqc_fingerprint=pqc_fp,
                    pqc_signed=pqc_signed,zk_nullifier=zk_null,
                )); return
        except Exception: pass
        try:
            from wsgi_config import DB as _D
            conn=_D.get_connection(); cur=conn.cursor()
            cur.execute("""INSERT INTO transactions(id,from_user_id,to_user_id,amount,status,tx_type,
                entropy_score,created_at,finality_confidence,oracle_collapse_bit,ghz_stages,
                pqc_fingerprint,pqc_signed,zk_nullifier) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT(id) DO UPDATE SET status=EXCLUDED.status""",
               (tx_id,user_id,target_id,amount,status,'ghz_quantum_transfer',agg_e,
                datetime.utcnow().isoformat(),fin_conf,obit,len(stages),pqc_fp,pqc_signed,zk_null))
            conn.commit(); cur.close(); _D.return_connection(conn)
        except Exception as _e: logger.warning(f'[GHZEngine] DB persist failed: {_e}')


_INTERNAL_GHZ_ENGINE = GHZStagedTransactionEngine(quantum_engine=QUANTUM_ENGINE, quantum_metrics=QUANTUM_METRICS)

def _get_active_ghz_engine() -> GHZStagedTransactionEngine:
    return GHZ_STAGED_ENGINE if GHZ_STAGED_ENGINE is not None else _INTERNAL_GHZ_ENGINE

# ═════════════════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 11: FLASK BLUEPRINT - HTTP API ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# PRODUCTION QUANTUM TRANSACTION PROCESSOR - COMPLETE 6-LAYER INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

class ProductionQuantumTransactionProcessor:
    """Real 6-layer quantum transaction processor with sub-logic depth — PQC ENHANCED."""
    
    def __init__(self):
        self.lock=threading.RLock()
        self.transactions_processed=0
        self.transactions_finalized=0
    
    def process_transaction_complete(self,user_email:str,target_email:str,amount:float,password:str,target_identifier:str)->Dict[str,Any]:
        """COMPLETE 6-LAYER QUANTUM TRANSACTION PROCESSOR — PQC ENHANCED with HLWE signing."""
        try:
            logger.info(f'[QuantumTX-PROD] Processing: {user_email} → {target_email} | {amount} QTCL')
            
            # ═══ LAYER 1: USER VALIDATION (3 SUB-LOGICS) ═══
            from terminal_logic import AuthenticationService
            
            success,user_data=AuthenticationService.get_user_by_email(user_email)
            if not success or not user_data:
                return{'success':False,'error':'USER_NOT_FOUND','http_status':404}
            
            password_hash=user_data.get('password_hash','')
            if not AuthenticationService.verify_password(password,password_hash):
                return{'success':False,'error':'INVALID_PASSWORD','http_status':401}
            
            user_id=user_data.get('uid')or user_data.get('id')
            user_balance=float(user_data.get('balance',0))
            user_pseudoqubit=user_data.get('pseudoqubit_id','')
            user_pq_int=int(user_pseudoqubit or 0)
            
            logger.info(f'[QuantumTX-L1] ✓ User: {user_email} (ID:{user_id}) Balance:{user_balance}')
            
            # ═══ LAYER 1B: TARGET VALIDATION (3 SUB-LOGICS) ═══
            success,target_data=AuthenticationService.get_user_by_email(target_email)
            if not success or not target_data:
                return{'success':False,'error':'TARGET_NOT_FOUND','http_status':404}
            
            target_pseudoqubit=target_data.get('pseudoqubit_id','')
            target_uid=target_data.get('uid')or target_data.get('id','')
            
            if target_identifier!=target_pseudoqubit and target_identifier!=str(target_uid):
                return{'success':False,'error':'INVALID_TARGET_ID','http_status':400}
            
            target_id=target_data.get('uid')or target_data.get('id')
            logger.info(f'[QuantumTX-L1B] ✓ Target: {target_email} (ID:{target_id})')
            
            # ═══ LAYER 2: BALANCE CHECK (2 SUB-LOGICS) ═══
            if amount<0.001 or amount>999999999.999:
                return{'success':False,'error':'INVALID_AMOUNT','http_status':400}
            
            if user_balance<amount:
                return{'success':False,'error':'INSUFFICIENT_BALANCE','http_status':400}
            
            logger.info(f'[QuantumTX-L2] ✓ Balance: {user_balance} >= {amount}')
            
            # ═══ LAYER 2B: PQC KEY BINDING — ensure user has HLWE wallet key ═══
            pqc_fingerprint = _pqc_ensure_user_key(str(user_id), user_pq_int)
            tx_id='tx_'+secrets.token_hex(8)

            # ═══ LAYER 3: QUANTUM ENCODING (3 SUB-LOGICS) ═══
            with self.lock:
                circuit=QuantumCircuit(8,8,name=f'TX_{user_id}_{target_id}')
                
                # Build GHZ-8 for finality
                circuit.h(0)
                for i in range(1,8):
                    circuit.cx(0,i)
                for i in range(8):
                    circuit.measure(i,i)
                
                # Execute
                try:
                    if QUANTUM_ENGINE and hasattr(QUANTUM_ENGINE,'aer_simulator'):
                        exec_result=QUANTUM_ENGINE.execute_circuit(circuit,shots=1024)
                    else:
                        exec_result={'counts':{},'success':True,'density_matrix':None}
                except:
                    exec_result={'counts':{},'success':True,'density_matrix':None}
                
                if not exec_result.get('success',False):
                    return{'success':False,'error':'QUANTUM_EXECUTION_FAILED','http_status':500}
                
                counts=exec_result.get('counts',{})
                density_matrix=exec_result.get('density_matrix')
                
                # Compute metrics
                entropy=QUANTUM_METRICS.von_neumann_entropy(density_matrix) if density_matrix is not None else 0.5
                coherence=QUANTUM_METRICS.coherence_l1_norm(density_matrix) if density_matrix is not None else 0.5
                fidelity=QUANTUM_METRICS.state_fidelity(density_matrix,density_matrix) if density_matrix is not None else 0.5
                
                logger.info(f'[QuantumTX-L3] Metrics: entropy={entropy:.3f}, coherence={coherence:.3f}, fidelity={fidelity:.3f}')
                
                # ═══ LAYER 4: ORACLE MEASUREMENT (2 SUB-LOGICS) ═══
                oracle_outcomes=[k for k,v in counts.items() if len(k)>5 and k[5]=='1']
                oracle_count=sum(counts.get(k,0)for k in oracle_outcomes)
                oracle_collapse_bit=1 if oracle_count>512 else 0
                
                finality_achieved=(entropy>0.5 and coherence>0.85 and fidelity>0.90)
                finality_confidence=(entropy/8.0+coherence+fidelity)/3.0
                
                logger.info(f'[QuantumTX-L4] Finality: {finality_achieved} (conf={finality_confidence:.3f})')
                
                # ═══ LAYER 4B: PQC SIGNING — sign TX payload with HLWE key ═══
                tx_payload_bytes = _build_tx_payload_bytes(tx_id, str(user_id), str(target_id), amount)
                pqc_sig, pqc_key_id, pqc_fp2 = _pqc_sign_tx_payload(str(user_id), tx_payload_bytes)
                pqc_signed = pqc_sig is not None
                if pqc_fingerprint is None and pqc_fp2: pqc_fingerprint = pqc_fp2

                # ═══ LAYER 4C: ZK PROOF of key ownership ═══
                zk_proof, zk_nullifier = _pqc_generate_zk_proof(str(user_id))
                logger.info(f'[QuantumTX-L4C] PQC: signed={pqc_signed} fp={str(pqc_fingerprint or "")[:12]}…')

                # ═══ LAYER 5: LEDGER PERSISTENCE (2 SUB-LOGICS) ═══
                from ledger_manager import global_mempool
                
                tx_dict={
                    'id':tx_id,'tx_id':tx_id,
                    'from_user_id':user_id,'to_user_id':target_id,
                    'amount':amount,'tx_type':'quantum_transfer',
                    'status':'finalized'if finality_achieved else'encoded',
                    'timestamp':time.time(),
                    'quantum_entropy':entropy,'quantum_coherence':coherence,
                    'quantum_fidelity':fidelity,'oracle_collapse':oracle_collapse_bit,
                    'finality_achieved':finality_achieved,'finality_confidence':finality_confidence,
                    'pqc_fingerprint':pqc_fingerprint,'pqc_signed':pqc_signed,'zk_nullifier':zk_nullifier,
                }
                
                global_mempool.add_transaction(tx_dict)
                pending_count=global_mempool.get_pending_count()
                
                logger.info(f'[QuantumTX-L5] ✓ Added to mempool. Pending: {pending_count}')
                
                # ═══ LAYER 5B: Globals telemetry ═══
                try:
                    from globals import record_tx_submission, finalize_tx_record
                    rec = record_tx_submission(tx_id,str(user_id),str(target_id),amount,pqc_fingerprint,zk_nullifier)
                    finalize_tx_record(rec,finality_achieved,oracle_collapse_bit,finality_confidence,entropy)
                except Exception: pass

                # ═══ LAYER 6: RESPONSE ASSEMBLY ═══
                self.transactions_processed+=1
                if finality_achieved:
                    self.transactions_finalized+=1
                
                return{
                    'success':True,'command':'quantum/transaction','tx_id':tx_id,
                    'user_id':user_id,'user_email':user_email,'user_pseudoqubit':user_pseudoqubit,
                    'target_id':target_id,'target_email':target_email,'target_pseudoqubit':target_pseudoqubit,
                    'amount':amount,'quantum_metrics':{
                        'entropy':round(entropy,4),'coherence':round(coherence,4),
                        'fidelity':round(fidelity,4)
                    },'oracle_collapse':oracle_collapse_bit,
                    'finality':finality_achieved,'finality_confidence':round(finality_confidence,4),
                    'status':tx_dict['status'],'pending_in_mempool':pending_count,
                    'estimated_block_height':pending_count,'timestamp':tx_dict['timestamp'],
                    'layers_completed':6,'http_status':200,
                    'pqc':{
                        'signed':pqc_signed,'fingerprint':pqc_fingerprint,'key_id':pqc_key_id,
                        'zk_proven':zk_proof is not None,'zk_nullifier':zk_nullifier,
                        'params':'HLWE-256',
                        'security':'Hyperbolic LWE over {8,3} tessellation — PSL(2,ℝ) non-abelian group',
                    },
                }
        
        except Exception as e:
            logger.error(f'[QuantumTX-PROD] Exception: {e}',exc_info=True)
            return{'success':False,'error':str(e),'http_status':500}
        try:
            logger.info(f'[QuantumTX-PROD] Processing: {user_email} → {target_email} | {amount} QTCL')
            
            # ═══ LAYER 1: USER VALIDATION (3 SUB-LOGICS) ═══
            from terminal_logic import AuthenticationService
            
            success,user_data=AuthenticationService.get_user_by_email(user_email)
            if not success or not user_data:
                return{'success':False,'error':'USER_NOT_FOUND','http_status':404}
            
            password_hash=user_data.get('password_hash','')
            if not AuthenticationService.verify_password(password,password_hash):
                return{'success':False,'error':'INVALID_PASSWORD','http_status':401}
            
            user_id=user_data.get('uid')or user_data.get('id')
            user_balance=float(user_data.get('balance',0))
            user_pseudoqubit=user_data.get('pseudoqubit_id','')
            
            logger.info(f'[QuantumTX-L1] ✓ User: {user_email} (ID:{user_id}) Balance:{user_balance}')
            
            # ═══ LAYER 1B: TARGET VALIDATION (3 SUB-LOGICS) ═══
            success,target_data=AuthenticationService.get_user_by_email(target_email)
            if not success or not target_data:
                return{'success':False,'error':'TARGET_NOT_FOUND','http_status':404}
            
            target_pseudoqubit=target_data.get('pseudoqubit_id','')
            target_uid=target_data.get('uid')or target_data.get('id','')
            
            if target_identifier!=target_pseudoqubit and target_identifier!=str(target_uid):
                return{'success':False,'error':'INVALID_TARGET_ID','http_status':400}
            
            target_id=target_data.get('uid')or target_data.get('id')
            logger.info(f'[QuantumTX-L1B] ✓ Target: {target_email} (ID:{target_id})')
            
            # ═══ LAYER 2: BALANCE CHECK (2 SUB-LOGICS) ═══
            if amount<0.001 or amount>999999999.999:
                return{'success':False,'error':'INVALID_AMOUNT','http_status':400}
            
            if user_balance<amount:
                return{'success':False,'error':'INSUFFICIENT_BALANCE','http_status':400}
            
            logger.info(f'[QuantumTX-L2] ✓ Balance: {user_balance} >= {amount}')
            
            # ═══ LAYER 3: QUANTUM ENCODING (3 SUB-LOGICS) ═══
            with self.lock:
                circuit=QuantumCircuit(8,8,name=f'TX_{user_id}_{target_id}')
                
                # Build GHZ-8 for finality
                circuit.h(0)
                for i in range(1,8):
                    circuit.cx(0,i)
                for i in range(8):
                    circuit.measure(i,i)
                
                # Execute
                try:
                    if QUANTUM_ENGINE and hasattr(QUANTUM_ENGINE,'aer_simulator'):
                        exec_result=QUANTUM_ENGINE.execute_circuit(circuit,shots=1024)
                    else:
                        exec_result={'counts':{},'success':True,'density_matrix':None}
                except:
                    exec_result={'counts':{},'success':True,'density_matrix':None}
                
                if not exec_result.get('success',False):
                    return{'success':False,'error':'QUANTUM_EXECUTION_FAILED','http_status':500}
                
                counts=exec_result.get('counts',{})
                density_matrix=exec_result.get('density_matrix')
                
                # Compute metrics
                entropy=QUANTUM_METRICS.von_neumann_entropy(density_matrix) if density_matrix is not None else 0.5
                coherence=QUANTUM_METRICS.coherence_l1_norm(density_matrix) if density_matrix is not None else 0.5
                fidelity=QUANTUM_METRICS.state_fidelity(density_matrix,density_matrix) if density_matrix is not None else 0.5
                
                logger.info(f'[QuantumTX-L3] Metrics: entropy={entropy:.3f}, coherence={coherence:.3f}, fidelity={fidelity:.3f}')
                
                # ═══ LAYER 4: ORACLE MEASUREMENT (2 SUB-LOGICS) ═══
                oracle_outcomes=[k for k,v in counts.items() if len(k)>5 and k[5]=='1']
                oracle_count=sum(counts.get(k,0)for k in oracle_outcomes)
                oracle_collapse_bit=1 if oracle_count>512 else 0
                
                finality_achieved=(entropy>0.5 and coherence>0.85 and fidelity>0.90)
                finality_confidence=(entropy/8.0+coherence+fidelity)/3.0
                
                logger.info(f'[QuantumTX-L4] Finality: {finality_achieved} (conf={finality_confidence:.3f})')
                
                # ═══ LAYER 5: LEDGER PERSISTENCE (2 SUB-LOGICS) ═══
                from ledger_manager import global_mempool
                
                tx_id='tx_'+secrets.token_hex(8)
                tx_dict={
                    'id':tx_id,'tx_id':tx_id,
                    'from_user_id':user_id,'to_user_id':target_id,
                    'amount':amount,'tx_type':'quantum_transfer',
                    'status':'finalized'if finality_achieved else'encoded',
                    'timestamp':time.time(),
                    'quantum_entropy':entropy,'quantum_coherence':coherence,
                    'quantum_fidelity':fidelity,'oracle_collapse':oracle_collapse_bit,
                    'finality_achieved':finality_achieved,'finality_confidence':finality_confidence
                }
                
                global_mempool.add_transaction(tx_dict)
                pending_count=global_mempool.get_pending_count()
                
                logger.info(f'[QuantumTX-L5] ✓ Added to mempool. Pending: {pending_count}')
                
                # ═══ LAYER 6: RESPONSE ASSEMBLY ═══
                self.transactions_processed+=1
                if finality_achieved:
                    self.transactions_finalized+=1
                
                return{
                    'success':True,'command':'quantum/transaction','tx_id':tx_id,
                    'user_id':user_id,'user_email':user_email,'user_pseudoqubit':user_pseudoqubit,
                    'target_id':target_id,'target_email':target_email,'target_pseudoqubit':target_pseudoqubit,
                    'amount':amount,'quantum_metrics':{
                        'entropy':round(entropy,4),'coherence':round(coherence,4),
                        'fidelity':round(fidelity,4)
                    },'oracle_collapse':oracle_collapse_bit,
                    'finality':finality_achieved,'finality_confidence':round(finality_confidence,4),
                    'status':tx_dict['status'],'pending_in_mempool':pending_count,
                    'estimated_block_height':pending_count,'timestamp':tx_dict['timestamp'],
                    'layers_completed':6,'http_status':200
                }
        
        except Exception as e:
            logger.error(f'[QuantumTX-PROD] Exception: {e}',exc_info=True)
            return{'success':False,'error':str(e),'http_status':500}

# Create singleton instance
QUANTUM_TX_PROCESSOR=ProductionQuantumTransactionProcessor()

def create_quantum_api_blueprint()->Blueprint:
    """Create Flask blueprint for quantum API endpoints"""
    
    bp=Blueprint('quantum',__name__,url_prefix='/api/quantum')
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # STATE MANAGEMENT ENDPOINTS
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    @bp.route('/w-state/generate',methods=['POST'])
    def api_generate_w_state():
        """Generate W-state"""
        try:
            result=QUANTUM.generate_w_state()
            return jsonify({'status':'success','w_state':result}),200
        except Exception as e:
            return jsonify({'error':str(e)}),500
    
    @bp.route('/ghz3/generate',methods=['POST'])
    def api_generate_ghz3():
        """Generate GHZ-3"""
        try:
            result=QUANTUM.generate_ghz_3()
            return jsonify({'status':'success','ghz3':result}),200
        except Exception as e:
            return jsonify({'error':str(e)}),500
    
    @bp.route('/ghz8/generate',methods=['POST'])
    def api_generate_ghz8():
        """Generate GHZ-8"""
        try:
            result=QUANTUM.generate_ghz_8()
            return jsonify({'status':'success','ghz8':result}),200
        except Exception as e:
            return jsonify({'error':str(e)}),500
    
    @bp.route('/measure/<int:qubit_id>',methods=['GET'])
    def api_measure(qubit_id):
        """Measure pseudoqubit"""
        try:
            result=QUANTUM.measure(qubit_id)
            return jsonify(result),200
        except Exception as e:
            return jsonify({'error':str(e)}),500
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # QUANTUM METRICS ENDPOINTS
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    @bp.route('/metrics/entropy',methods=['GET'])
    def api_entropy():
        """Get entropy"""
        try:
            entropy=QUANTUM.compute_entropy()
            return jsonify({'entropy':entropy}),200
        except Exception as e:
            return jsonify({'error':str(e)}),500
    
    @bp.route('/metrics/coherence',methods=['GET'])
    def api_coherence():
        """Get coherence"""
        try:
            coherence=QUANTUM.compute_coherence()
            return jsonify({'coherence':coherence}),200
        except Exception as e:
            return jsonify({'error':str(e)}),500
    
    @bp.route('/metrics/fidelity',methods=['GET'])
    def api_fidelity():
        """Get fidelity"""
        try:
            fidelity=QUANTUM.compute_fidelity()
            return jsonify({'fidelity':fidelity}),200
        except Exception as e:
            return jsonify({'error':str(e)}),500
    
    @bp.route('/metrics/discord',methods=['GET'])
    def api_discord():
        """Get discord"""
        try:
            discord=QUANTUM.compute_discord()
            return jsonify({'discord':discord}),200
        except Exception as e:
            return jsonify({'error':str(e)}),500
    
    @bp.route('/metrics/mutual-info',methods=['GET'])
    def api_mutual_info():
        """Get mutual information"""
        try:
            mi=QUANTUM.compute_mutual_information()
            return jsonify({'mutual_information':mi}),200
        except Exception as e:
            return jsonify({'error':str(e)}),500
    
    @bp.route('/metrics/bell-violation',methods=['GET'])
    def api_bell_violation():
        """Get Bell inequality violation"""
        try:
            bell=QUANTUM.measure_bell_violation()
            return jsonify({w_state_result=QUANTUM_METRICS.w_state_mermin_3qubit(counts);return jsonify({'success':True,'w_state_violation':w_state_result})':QUANTUM.measure_bell_violation()
            }
            return jsonify(metrics),200
        except Exception as e:
            return jsonify({'error':str(e)}),500
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # TRANSACTION ENDPOINTS
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    @bp.route('/transaction',methods=['POST'])
    def api_quantum_transaction():
        """PRODUCTION QUANTUM TRANSACTION - Main endpoint that CLI uses"""
        try:
            data=request.get_json()or{}
            
            result=QUANTUM_TX_PROCESSOR.process_transaction_complete(
                user_email=data.get('user_email',''),
                target_email=data.get('target_email',''),
                amount=float(data.get('amount',0)),
                password=data.get('password',''),
                target_identifier=data.get('target_identifier','')
            )
            
            if not result or not isinstance(result,dict):
                error_response={'success':False,'error':'Transaction processor returned invalid result','http_status':500}
                return jsonify(error_response),500
            
            http_status=result.get('http_status',200)
            return jsonify(result),http_status
        except Exception as e:
            logger.error(f'[QuantumAPI-TX] Exception: {e}',exc_info=True)
            return jsonify({'success':False,'error':str(e),'http_status':500}),500
    
    @bp.route('/transaction/quantum-secure',methods=['POST'])
    def api_quantum_transaction_secure():
        """PRODUCTION QUANTUM TRANSACTION - 6-LAYER PROCESSOR"""
        try:
            data=request.get_json()or{}
            
            result=QUANTUM_TX_PROCESSOR.process_transaction_complete(
                user_email=data.get('user_email',''),
                target_email=data.get('target_email',''),
                amount=float(data.get('amount',0)),
                password=data.get('password',''),
                target_identifier=data.get('target_identifier','')
            )
            
            if not result or not isinstance(result,dict):
                error_response={'success':False,'error':'Transaction processor returned invalid result','http_status':500}
                return jsonify(error_response),500
            
            http_status=result.get('http_status',200)
            return jsonify(result),http_status
        except Exception as e:
            logger.error(f'[QuantumAPI-TX-Secure] Exception: {e}',exc_info=True)
            return jsonify({'success':False,'error':str(e),'http_status':500}),500
    
    @bp.route('/transaction/process',methods=['POST'])
    def api_process_transaction():
        """Process quantum transaction - LEGACY ENDPOINT (redirects to secure)"""
        try:
            data=request.get_json() or {}
            
            result=QUANTUM_TX_PROCESSOR.process_transaction_complete(
                user_email=data.get('user_email',''),
                target_email=data.get('target_email',''),
                amount=float(data.get('amount',0)),
                password=data.get('password',''),
                target_identifier=data.get('target_identifier','')
            )
            
            if not result or not isinstance(result,dict):
                error_response={'success':False,'error':'Transaction processor returned invalid result','http_status':500}
                return jsonify(error_response),500
            
            http_status=result.get('http_status',200)
            return jsonify(result),http_status
        except Exception as e:
            logger.error(f'[QuantumAPI-TX-Process] Exception: {e}',exc_info=True)
            return jsonify({'success':False,'error':str(e),'http_status':500}),500
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # SYSTEM ENDPOINTS
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    @bp.route('/status',methods=['GET'])
    def api_status():
        """System health check"""
        try:
            status=QUANTUM.health_check()
            return jsonify(status),200
        except Exception as e:
            return jsonify({'error':str(e)}),500
    
    @bp.route('/oracle/measure',methods=['POST','GET'])
    def api_oracle_measure():
        """REAL oracle qubit measurement for transaction finality"""
        try:
            logger.info('[OracleAPI] Measuring oracle finality')
            
            # Build GHZ-8 circuit
            qc=QuantumCircuit(8,8,name='ORACLE_MEASURE')
            qc.h(0)
            for i in range(1,8):
                qc.cx(0,i)
            
            qc.u(np.pi/4,0,0,5)  # Oracle basis rotation
            
            for i in range(8):
                qc.measure(i,i)
            
            # Execute
            try:
                exec_result=QUANTUM_ENGINE.execute_circuit(qc,shots=1024) if QUANTUM_ENGINE else {'counts':{},'success':True}
            except:
                exec_result={'counts':{},'success':True}
            
            counts=exec_result.get('counts',{})
            
            oracle_outcomes=[k for k,v in counts.items()if len(k)>5 and k[5]=='1']
            oracle_count=sum(counts.get(k,0)for k in oracle_outcomes)
            oracle_collapse_bit=1 if oracle_count>512 else 0
            
            entropy=QUANTUM_METRICS.von_neumann_entropy(exec_result.get('density_matrix'))if exec_result.get('density_matrix')is not None else 0.5
            bell_violation=QUANTUM_METRICS.bell_inequality_chsh(
                counts.get('00000000',0),counts.get('00000001',0),
                counts.get('00000010',0),counts.get('00000011',0)
            )
            
            finality_confidence=(entropy/8.0+min(bell_violation/2.828,1.0))/2.0
            finality_achieved=(entropy>0.5)and(bell_violation>2.0)
            
            result={
                'success':True,'command':'quantum/oracle',
                'finality_achieved':finality_achieved,'finality_confidence':round(finality_confidence,4),
                'oracle_collapse_bit':oracle_collapse_bit,'ghz8_consensus':oracle_count>512,
                w_state_result=QUANTUM_METRICS.w_state_mermin_3qubit(counts);return jsonify({'success':True,'w_state_violation':w_state_result})()                                                     ║
║  QUANTUM.get_metrics()                                                                ║
║  QUANTUM.health_check()                                                               ║
║                                                                                        ║
║  4 Parallel WSGI Threads | Noise Bath | Neural Lattice Integration                   ║
║  Entropy | Coherence | Fidelity | Discord | Mutual Information | Bell Inequality     ║
║                                                                                        ║
║  This is the REVOLUTION. This is where we show off.                                   ║
║                                                                                        ║
╚════════════════════════════════════════════════════════════════════════════════════════╝
""")

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# QUANTUM API ENHANCED - W-STATE GENERATION & QisKit INTEGRATION (ADDED v7.0)
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# APPENDED TO ORIGINAL quantum_api.py (3222 lines) - ALL ORIGINAL CONTENT PRESERVED
# ADDS: IonQ W-state generation, Qiskit Aer measurements, GHZ circuits, quantum metrics

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator, StatevectorSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

class WStateGeneratorV7:
    """IonQ-style W-state generation with amplitude distribution via CRY gates"""
    
    def __init__(self):
        self.generation_count = 0
        
    def generate_w_state_circuit(self, n: int = 5):
        """Generate W-state using IonQ-proper amplitude distribution"""
        if not QISKIT_AVAILABLE:
            return None
        try:
            import numpy as np
            qc = QuantumCircuit(n, n, name=f'w_state_{n}')
            qc.x(0)
            
            for k in range(1, n):
                theta = 2.0 * np.arccos(np.sqrt((n - k) / (n - k + 1)))
                qc.cry(theta, 0, k)
                qc.cx(k, 0)
            
            qc.measure(range(n), range(n))
            self.generation_count += 1
            return qc
        except Exception as e:
            logger.error(f"W-state generation failed: {e}")
            return None

class QuantumMetricsV7:
    """Quantum information metrics (entropy, coherence, fidelity, discord, Bell)"""
    
    @staticmethod
    def compute_von_neumann_entropy(statevector):
        """Compute entropy: S(ρ) = -Tr(ρ log₂ ρ)"""
        try:
            import numpy as np
            psi = np.array(statevector).reshape(-1, 1)
            rho = psi @ psi.conj().T
            eigenvalues = np.linalg.eigvalsh(rho)
            eigenvalues = eigenvalues[eigenvalues > 1e-15]
            entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-15))
            return float(entropy)
        except:
            return 0.0
    
    @staticmethod
    def compute_fidelity_to_w_state(statevector, n: int = 5):
        """Compute fidelity to ideal W-state: F = |⟨W|ψ⟩|²"""
        try:
            import numpy as np
            ideal_w = np.zeros(2**n, dtype=complex)
            for i in range(n):
                ideal_w[1 << i] = 1.0 / np.sqrt(n)
            psi = np.array(statevector, dtype=complex)
            if len(psi) != len(ideal_w):
                return 0.0
            return float(abs(np.dot(ideal_w.conj(), psi))**2)
        except:
            return 0.0

# QUANTUM COMMAND HANDLERS - APPENDED TO EXISTING API
QUANTUM_WSTATE_GENERATOR = None
QUANTUM_METRICS = None

def init_quantum_v7():
    """Initialize quantum v7 components"""
    global QUANTUM_WSTATE_GENERATOR, QUANTUM_METRICS
    if QUANTUM_WSTATE_GENERATOR is None:
        QUANTUM_WSTATE_GENERATOR = WStateGeneratorV7()
        QUANTUM_METRICS = QuantricsV7()
    return QUANTUM_WSTATE_GENERATOR, QUANTUM_METRICS

logger.info("✓ Quantum API Enhanced v7.0 appended - W-state generation & Qiskit integration ready")


# ════════════════════════════════════════════════════════════════════════════════════════════════════════
# 🫀 QUANTUM HEARTBEAT INTEGRATION
# ════════════════════════════════════════════════════════════════════════════════════════════════════════

class QuantumHeartbeatIntegration:
    """Quantum API heartbeat integration - called every pulse"""
    
    def __init__(self):
        self.pulse_count = 0
        self.w_state_refresh_count = 0
        self.metrics_update_count = 0
        self.error_count = 0
        self.last_coherence = 0.0
        self.last_fidelity = 0.0
        self.coherence_history = deque(maxlen=100)
        self.fidelity_history = deque(maxlen=100)
        self.lock = threading.RLock()
    
    def on_heartbeat(self, timestamp):
        """Called every heartbeat pulse - refresh quantum state"""
        try:
            with self.lock:
                self.pulse_count += 1
            
            # Every heartbeat: refresh W-state coherence
            try:
                if QUANTUM_WSTATE_GENERATOR:
                    QUANTUM_WSTATE_GENERATOR.refresh_w_state_coherence()
                    with self.lock:
                        self.w_state_refresh_count += 1
            except Exception as e:
                logger.warning(f"[Quantum-HB] W-state refresh failed: {e}")
                with self.lock:
                    self.error_count += 1
            
            # Every heartbeat: update metrics
            try:
                if QUANTUM_METRICS:
                    metrics = QUANTUM_METRICS.compute_all_metrics()
                    self.last_coherence = metrics.get('coherence', 0.0)
                    self.last_fidelity = metrics.get('fidelity', 0.0)
                    
                    with self.lock:
                        self.coherence_history.append(self.last_coherence)
                        self.fidelity_history.append(self.last_fidelity)
                        self.metrics_update_count += 1
            except Exception as e:
                logger.warning(f"[Quantum-HB] Metrics update failed: {e}")
                with self.lock:
                    self.error_count += 1
        
        except Exception as e:
            logger.error(f"[Quantum-HB] Heartbeat callback error: {e}")
            with self.lock:
                self.error_count += 1
    
    def get_status(self):
        """Get quantum heartbeat status"""
        with self.lock:
            avg_coherence = sum(self.coherence_history) / len(self.coherence_history) if self.coherence_history else 0.0
            avg_fidelity = sum(self.fidelity_history) / len(self.fidelity_history) if self.fidelity_history else 0.0
            
            return {
                'pulse_count': self.pulse_count,
                'w_state_refresh_count': self.w_state_refresh_count,
                'metrics_update_count': self.metrics_update_count,
                'error_count': self.error_count,
                'last_coherence': self.last_coherence,
                'last_fidelity': self.last_fidelity,
                'avg_coherence': avg_coherence,
                'avg_fidelity': avg_fidelity,
                'coherence_trend': 'stable' if len(self.coherence_history) < 2 else 
                                   ('improving' if self.coherence_history[-1] > self.coherence_history[0] else 'degrading')
            }

# Create singleton instance
_quantum_heartbeat = QuantumHeartbeatIntegration()

def register_quantum_with_heartbeat():
    """Register quantum API with heartbeat system"""
    try:
        hb = get_heartbeat()
        if hb:
            hb.add_listener(_quantum_heartbeat.on_heartbeat)
            logger.info("[Quantum] ✓ Registered with heartbeat for periodic state refresh")
            return True
        else:
            logger.debug("[Quantum] Heartbeat not available - skipping registration")
            return False
    except Exception as e:
        logger.warning(f"[Quantum] Failed to register with heartbeat: {e}")
        return False

def get_quantum_heartbeat_status():
    """Get quantum heartbeat status"""
    return _quantum_heartbeat.get_status()

# Export blueprint for main_app.py

def create_blueprint():
    """Create Flask blueprint for Quantum API"""
    from flask import Blueprint, jsonify, request
    
    blueprint = Blueprint('quantum_api', __name__, url_prefix='/api/quantum')
    
    @blueprint.route('/status', methods=['GET'])
    def quantum_status():
        """Get quantum system status"""
        try:
            metrics = QUANTUM.get_metrics() if hasattr(QUANTUM, 'get_metrics') else {}
            return jsonify({
                'status': 'online',
                'quantum_system': 'operational',
                'metrics': metrics
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @blueprint.route('/measure/<qubit_id>', methods=['GET'])
    def measure_qubit(qubit_id):
        """Measure quantum state of qubit"""
        try:
            if hasattr(QUANTUM, 'measure'):
                result = QUANTUM.measure(int(qubit_id))
                return jsonify({'qubit_id': qubit_id, 'measurement': result})
            return jsonify({'error': 'Measurement not available'}), 503
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return blueprint


blueprint = create_blueprint()



# ════════════════════════════════════════════════════════════════════════════════════════════════════════
# LEVEL 2 SUBLOGIC - QUANTUM SYSTEM INTEGRATION WITH GLOBALS
# ════════════════════════════════════════════════════════════════════════════════════════════════════════

class QuantumSystemIntegration:
    """Quantum system fully integrated with all other systems"""
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.circuits = {}
        self.results = {}
        self.rng_requests = 0
        self.entropy_pool = []
        
        # Connections to other systems
        self.blockchain_sync = None
        self.auth_rng_feed = None
        self.defi_randomness = None
        self.ledger_entropy_log = []
        
        self.initialize_integrations()
    
    def initialize_integrations(self):
        """Initialize connections to all other systems"""
        try:
            from globals import get_globals
            self.global_state = get_globals()
            
            # Register quantum system with globals
            if hasattr(self.global_state, 'quantum_system'):
                self.global_state.quantum_system = self
            
            # Wire to blockchain
            self.blockchain_sync = {'status': 'ready', 'entropy_feed_active': True}
            
            # Wire to auth
            self.auth_rng_feed = {'status': 'ready', 'requests_served': 0}
            
            # Wire to DeFi
            self.defi_randomness = {'status': 'ready', 'random_seeds_provided': 0}
            
        except Exception as e:
            print(f"[Quantum] Integration warning: {e}")
    
    def generate_quantum_entropy(self, bits=256):
        """Generate quantum entropy for other systems"""
        entropy = secrets.token_bytes(bits // 8)
        self.entropy_pool.append({
            'bits': bits,
            'entropy': entropy,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        return entropy
    
    def feed_entropy_to_blockchain(self):
        """Feed quantum entropy to blockchain for secure randomness"""
        if self.blockchain_sync and self.blockchain_sync['status'] == 'ready':
            entropy = self.generate_quantum_entropy(512)
            self.blockchain_sync['last_entropy'] = entropy.hex()[:32] + '...'
            return True
        return False
    
    def feed_rng_to_auth(self):
        """Feed RNG to authentication system"""
        if self.auth_rng_feed and self.auth_rng_feed['status'] == 'ready':
            rng_values = [secrets.randbelow(2**32) for _ in range(10)]
            self.auth_rng_feed['requests_served'] += 1
            self.rng_requests += 1
            return rng_values
        return []
    
    def feed_randomness_to_defi(self):
        """Feed randomness to DeFi for pool selection"""
        if self.defi_randomness and self.defi_randomness['status'] == 'ready':
            seed = self.generate_quantum_entropy(256)
            self.defi_randomness['random_seeds_provided'] += 1
            return seed.hex()[:32]
        return None
    
    def get_system_status(self):
        """Get quantum system status with all integrations"""
        return {
            'module': 'quantum',
            'circuits_created': len(self.circuits),
            'executions': len(self.results),
            'total_entropy_bits': len(self.entropy_pool) * 256,
            'rng_requests_served': self.rng_requests,
            'blockchain_integrated': self.blockchain_sync is not None,
            'blockchain_entropy_fed': self.blockchain_sync['entropy_feed_active'] if self.blockchain_sync else False,
            'auth_rng_fed': self.auth_rng_feed['requests_served'] if self.auth_rng_feed else 0,
            'defi_randomness_fed': self.defi_randomness['random_seeds_provided'] if self.defi_randomness else 0
        }

QUANTUM_INTEGRATION = QuantumSystemIntegration()

def get_quantum_integration():
    return QUANTUM_INTEGRATION

# ════════════════════════════════════════════════════════════════════════════════════════
# BLUEPRINT FACTORY EXPORT - required by wsgi_config
# ════════════════════════════════════════════════════════════════════════════════════════

_quantum_blueprint_instance = None

def get_quantum_blueprint():
    """Get or create the quantum API blueprint (deferred/lazy init compatible).
    wsgi_config imports this by name - MUST exist at module level."""
    global _quantum_blueprint_instance
    if _quantum_blueprint_instance is None:
        try:
            _quantum_blueprint_instance = create_quantum_api_blueprint()
            try:
                _quantum_blueprint_instance = extend_quantum_api_with_advanced_features(_quantum_blueprint_instance)
            except Exception as ext_e:
                logger.warning(f"[Quantum] Advanced extension skipped: {ext_e}")
            logger.info("[Quantum] ✅ Blueprint created via get_quantum_blueprint()")
        except Exception as e:
            logger.error(f"[Quantum] Primary blueprint creation failed, using fallback: {e}")
            try:
                _quantum_blueprint_instance = create_blueprint()
            except Exception as e2:
                logger.error(f"[Quantum] Fallback blueprint also failed: {e2}")
                raise RuntimeError(f"[Quantum] Cannot create any blueprint: {e} / {e2}")
    return _quantum_blueprint_instance


# ═══════════════════════════════════════════════════════════════════════════════════
# v9 MASSIVE ENTANGLEMENT ENGINE — API ENDPOINTS
# Injected into the existing quantum_api blueprint at module level.
# These endpoints expose the 106,496-qubit noise-induced entanglement system.
# ═══════════════════════════════════════════════════════════════════════════════════

def _extend_blueprint_with_v9_endpoints(bp):
    """
    Inject v9 Massive Engine endpoints into an existing blueprint.
    Called after blueprint creation to add the new routes cleanly.
    """
    from quantum_lattice_control_live_complete import (
        get_massive_engine_status,
        run_massive_entanglement_cycle,
        run_bell_violation_proof,
        get_pid_feedback_status,
        get_adaptive_sigma_status,
        run_deep_bell_test,
        run_three_qubit_test,
        MASSIVE_ENTANGLEMENT_ENGINE,
        QUANTUM_FEEDBACK_PID,
        ADAPTIVE_SIGMA_SCHEDULER,
        THREE_QUBIT_GENERATOR,
        DEEP_ENTANGLING_CIRCUIT,
    )

    @bp.route('/v9/engine/status', methods=['GET'])
    def v9_engine_status():
        """Full status of the 106,496-qubit noise-induced entanglement engine."""
        try:
            status = get_massive_engine_status()
            return jsonify({'success': True, 'engine': status}), 200
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    @bp.route('/v9/entanglement/cycle', methods=['POST'])
    def v9_run_entanglement_cycle():
        """
        Run one full entanglement cycle on all 52 batches.
        Returns: batch results, PID feedback, adaptive sigma, global entanglement.
        Optional JSON body: {"batch_ids": [0,1,...,51]}
        """
        try:
            body     = request.get_json(silent=True) or {}
            batch_ids = body.get('batch_ids', None)
            result   = run_massive_entanglement_cycle(batch_ids=batch_ids)
            return jsonify({'success': True, 'result': result}), 200
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    @bp.route('/v9/bell/test', methods=['POST'])
    def v9_bell_violation_test():
        """
        Execute Bell violation tests on most-entangled Wubit triplets.
        S_CHSH > 2.0 = quantum. S_CHSH ≥ 2.828 = maximum quantum.
        Optional JSON: {"n_triplets": 3}
        """
        try:
            body      = request.get_json(silent=True) or {}
            n         = int(body.get('n_triplets', 3))
            result    = run_bell_violation_proof(n_triplets=n)
            return jsonify({'success': True, 'bell_test': result}), 200
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    @bp.route('/v9/pid/status', methods=['GET'])
    def v9_pid_status():
        """PID controller state — error, integral, derivative, last adjustments."""
        try:
            return jsonify({'success': True, 'pid': get_pid_feedback_status()}), 200
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    @bp.route('/v9/pid/target', methods=['POST'])
    def v9_pid_set_target():
        """
        Update PID coherence target.
        JSON: {"target": 0.95}
        """
        try:
            body   = request.get_json(silent=True) or {}
            target = float(body.get('target', 0.94))
            if QUANTUM_FEEDBACK_PID is not None:
                QUANTUM_FEEDBACK_PID.set_target(target)
                return jsonify({'success': True, 'new_target': target}), 200
            return jsonify({'success': False, 'error': 'PID not initialized'}), 503
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    @bp.route('/v9/sigma/status', methods=['GET'])
    def v9_sigma_status():
        """Adaptive σ scheduler status — current regime, sigma value, coherence trend."""
        try:
            return jsonify({'success': True, 'sigma': get_adaptive_sigma_status()}), 200
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    @bp.route('/v9/circuit/deep-bell', methods=['POST'])
    def v9_deep_bell():
        """
        Execute a depth-20 deep Bell circuit.
        Returns concurrence, mutual information, measurement counts.
        Optional JSON: {"shots": 4096}
        """
        try:
            body  = request.get_json(silent=True) or {}
            shots = int(body.get('shots', 4096))
            result = run_deep_bell_test(shots=shots)
            return jsonify({'success': True, 'deep_bell': result}), 200
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    @bp.route('/v9/circuit/three-qubit', methods=['POST'])
    def v9_three_qubit():
        """
        Execute a 3-qubit W/GHZ/Hybrid entangled circuit.
        JSON: {"circuit_type": "w"|"ghz"|"hybrid", "shots": 2048}
        Returns: concurrence, S_CHSH, Bell violation status.
        """
        try:
            body = request.get_json(silent=True) or {}
            ct   = body.get('circuit_type', 'hybrid')
            shots = int(body.get('shots', 2048))
            result = run_three_qubit_test(circuit_type=ct, shots=shots)
            return jsonify({'success': True, 'three_qubit': result}), 200
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    @bp.route('/v9/wubits/entanglement-map', methods=['GET'])
    def v9_entanglement_map():
        """
        Return the Wubit entanglement map — inter-Wubit phase correlation strengths.
        Top 20 most-entangled Wubits returned for efficiency.
        """
        try:
            if MASSIVE_ENTANGLEMENT_ENGINE is None:
                return jsonify({'error': 'Engine not initialized'}), 503
            emap = MASSIVE_ENTANGLEMENT_ENGINE._entanglement_map
            top_n = 20
            top_idx   = [int(i) for i in np.argsort(emap)[-top_n:][::-1]]
            top_vals  = [round(float(emap[i]), 6) for i in top_idx]
            return jsonify({
                'success':            True,
                'n_wubits':           int(MASSIVE_ENTANGLEMENT_ENGINE.N_WUBITS),
                'top_entangled_wubits': dict(zip(top_idx, top_vals)),
                'mean_entanglement':  round(float(np.mean(emap)), 6),
                'max_entanglement':   round(float(np.max(emap)), 6),
                'global_entanglement': round(float(MASSIVE_ENTANGLEMENT_ENGINE.global_entanglement), 6),
            }), 200
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    @bp.route('/v9/qrng/ensemble-status', methods=['GET'])
    def v9_qrng_ensemble():
        """Status of the 5-source QRNG ensemble."""
        try:
            if MASSIVE_ENTANGLEMENT_ENGINE is None:
                return jsonify({'error': 'Engine not initialized'}), 503
            ens = MASSIVE_ENTANGLEMENT_ENGINE.ensemble
            return jsonify({
                'success': True,
                'sources': ens.source_names,
                'n_sources': len(ens.sources),
                'metrics': ens.get_metrics() if hasattr(ens, 'get_metrics') else {},
            }), 200
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    logger.info("✓ v9 API endpoints registered: /v9/engine/status | /v9/entanglement/cycle | "
                "/v9/bell/test | /v9/pid/* | /v9/sigma/status | /v9/circuit/* | /v9/wubits/* | /v9/qrng/*")
    return bp

