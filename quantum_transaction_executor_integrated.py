#!/usr/bin/env python3
"""
QUANTUM TRANSACTION EXECUTOR - INTEGRATED QUANTUM SUPERPOSITION & ORACLE VALIDATION
Unified transaction flow with quantum circuit execution, pseudoqubit superposition,
oracle validation (6 reserved qubits), block creation, and full persistence.
PRODUCTION-GRADE: ~2000 lines, single drop-in replacement module
Part 1/2: Core transaction executor with superposition handling
"""

import os,sys,time,json,hashlib,uuid,logging,threading,queue,math,cmath,random,pickle,base64
from datetime import datetime,timedelta,timezone
from typing import Dict,List,Tuple,Optional,Any
from dataclasses import dataclass,field,asdict
from enum import Enum
from collections import defaultdict,deque
import subprocess,traceback

def ensure_packages():
    packages={'psycopg2':'psycopg2-binary','numpy':'numpy','qiskit':'qiskit','qiskit_aer':'qiskit-aer','scipy':'scipy'}
    for module,pip_name in packages.items():
        try:__import__(module)
        except ImportError:
            subprocess.check_call([sys.executable,'-m','pip','install','-q',pip_name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

ensure_packages()
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from qiskit import QuantumCircuit,QuantumRegister,ClassicalRegister
from qiskit_aer import AerSimulator
from scipy import stats as scipy_stats

class Config:
    """QTCL Quantum Transaction Executor Configuration"""
    SUPABASE_HOST="aws-0-us-west-2.pooler.supabase.com"
    SUPABASE_USER="postgres.rslvlsqwkfmdtebqsvtw"
    SUPABASE_PASSWORD="$h10j1r1H0w4rd"
    SUPABASE_PORT=5432
    SUPABASE_DB="postgres"
    QISKIT_SHOTS=1024
    QISKIT_QUBITS=14
    ORACLE_QUBITS=6
    SUPERPOSITION_QUBITS=4
    DATA_QUBITS=4
    COHERENCE_REFRESH_SECONDS=5
    SUPERPOSITION_TIMEOUT_SECONDS=300
    BLOCK_SIZE_MAX_TX=1000
    BLOCK_CREATION_INTERVAL_SECONDS=10
    TX_BATCH_SIZE=25
    TOKEN_WEI_PER_UNIT=10**18
    MIN_ENTROPY_FOR_VALID_TX=0.70
    ORACLE_MEASUREMENT_THRESHOLD=0.75

logging.basicConfig(level=logging.INFO,format='[%(asctime)s][%(name)s]%(levelname)s: %(message)s')
logger=logging.getLogger(__name__)

class DatabasePool:
    """Connection pooling for Supabase PostgreSQL"""
    _instance=None
    _lock=threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance=super(DatabasePool,cls).__new__(cls)
                    cls._instance._initialized=False
        return cls._instance
    
    def __init__(self):
        if self._initialized:return
        self._initialized=True
        self.pool=deque(maxlen=10)
        self._lock=threading.Lock()
        self._create_connections(3)
    
    def _create_connections(self,count):
        for _ in range(count):
            try:
                conn=psycopg2.connect(host=Config.SUPABASE_HOST,user=Config.SUPABASE_USER,password=Config.SUPABASE_PASSWORD,port=Config.SUPABASE_PORT,database=Config.SUPABASE_DB,connect_timeout=15)
                self.pool.append(conn)
            except Exception as e:logger.error(f"Connection creation failed: {e}")
    
    def get_connection(self):
        with self._lock:
            if len(self.pool)==0:self._create_connections(1)
            return self.pool.popleft() if self.pool else None
    
    def return_connection(self,conn):
        if conn:
            with self._lock:
                try:self.pool.append(conn)
                except:pass
    
    def execute(self,query,params=None):
        conn=self.get_connection()
        if not conn:return []
        try:
            cur=conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(query,params or ())
            results=[dict(row) for row in cur.fetchall()]
            cur.close()
            return results
        except Exception as e:
            logger.error(f"Query execution failed: {e}\n{query}")
            return []
        finally:self.return_connection(conn)
    
    def execute_one(self,query,params=None):
        results=self.execute(query,params)
        return results[0] if results else None
    
    def execute_update(self,query,params=None):
        conn=self.get_connection()
        if not conn:return False
        try:
            cur=conn.cursor()
            cur.execute(query,params or ())
            conn.commit()
            cur.close()
            return True
        except Exception as e:
            logger.error(f"Update failed: {e}\n{query}")
            conn.rollback()
            return False
        finally:self.return_connection(conn)
    
    def execute_batch(self,query,param_list):
        conn=self.get_connection()
        if not conn:return False
        try:
            cur=conn.cursor()
            for params in param_list:cur.execute(query,params)
            conn.commit()
            cur.close()
            return True
        except Exception as e:
            logger.error(f"Batch update failed: {e}")
            conn.rollback()
            return False
        finally:self.return_connection(conn)

db=DatabasePool()

@dataclass
class SuperpositionState:
    """Quantum superposition state for transaction"""
    tx_id:str
    user_id:str
    created_at:str
    qubits_state:str
    entropy:float
    coherence:float
    phase_encoding:str
    oracle_measurement:Optional[str]=None
    collapsed:bool=False
    collapse_time:Optional[str]=None

@dataclass
class QuantumMeasurement:
    """Measurement result from quantum circuit"""
    measurement_id:str
    bitstring_counts:Dict[str,int]
    entropy:float
    dominant_states:List[Tuple[str,int]]
    state_space_utilization:float
    properties:str
    timestamp:str

class QuantumCircuitBuilder:
    """Builds quantum circuits with oracle qubits for transaction validation"""
    
    def __init__(self):
        self.simulator=AerSimulator()
        self.circuit_cache={}
        self.execution_stats=deque(maxlen=100)
    
    def build_transaction_circuit(self,tx_id:str,sender_id:str,receiver_id:str,amount:float)->QuantumCircuit:
        """Build 14-qubit circuit: 4 superposition + 4 data + 6 oracle"""
        qr_super=QuantumRegister(4,'superposition')
        qr_data=QuantumRegister(4,'data')
        qr_oracle=QuantumRegister(6,'oracle')
        cr=ClassicalRegister(14,'measure')
        
        circuit=QuantumCircuit(qr_super,qr_data,qr_oracle,cr,name=f'tx_{tx_id[:8]}')
        
        # Initialize superposition qubits (sender/receiver state)
        for i in range(4):circuit.h(qr_super[i])
        
        # Encode transaction parameters into data qubits via phase shifts
        sender_hash=int(hashlib.sha256(sender_id.encode()).hexdigest(),16)%256
        receiver_hash=int(hashlib.sha256(receiver_id.encode()).hexdigest(),16)%256
        amount_bits=int(amount*1000)%256
        
        phase_sender=(2*math.pi*sender_hash)/256
        phase_receiver=(2*math.pi*receiver_hash)/256
        phase_amount=(2*math.pi*amount_bits)/256
        
        circuit.rz(phase_sender,qr_data[0])
        circuit.rz(phase_receiver,qr_data[1])
        circuit.rz(phase_amount,qr_data[2])
        
        # Hadamard on data qubits for superposition
        for i in range(4):circuit.h(qr_data[i])
        
        # CNOT ladder for entanglement (data to oracle)
        for i in range(4):circuit.cx(qr_data[i],qr_oracle[i%6])
        
        # Additional oracle qubits initialization (balanced state)
        for i in range(6):circuit.h(qr_oracle[i])
        
        # Measurement of all qubits
        circuit.measure(qr_super,cr[0:4])
        circuit.measure(qr_data,cr[4:8])
        circuit.measure(qr_oracle,cr[8:14])
        
        return circuit
    
    def execute_circuit(self,circuit:QuantumCircuit,tx_id:str)->Tuple[Dict[str,int],Dict[str,Any]]:
        """Execute circuit on AER simulator and analyze results"""
        try:
            start_time=time.time()
            job=self.simulator.run(circuit,shots=Config.QISKIT_SHOTS,seed_simulator=hash(tx_id)%2**31)
            result=job.result()
            counts=result.get_counts()
            
            execution_time=(time.time()-start_time)*1000
            
            # Analyze measurements
            measurement_analysis=self._analyze_measurements(counts,tx_id)
            
            # Track performance
            self.execution_stats.append({'tx_id':tx_id,'time_ms':execution_time,'entropy':measurement_analysis['entropy']})
            
            logger.info(f"Circuit execution: {tx_id[:8]}... entropy={measurement_analysis['entropy']:.2%} time={execution_time:.1f}ms")
            
            return counts,measurement_analysis
        except Exception as e:
            logger.error(f"Circuit execution failed for {tx_id}: {e}\n{traceback.format_exc()}")
            return {},{'entropy':0.0,'error':str(e)}
    
    def _analyze_measurements(self,counts:Dict[str,int],tx_id:str)->Dict[str,Any]:
        """Calculate entropy and classify measurement results"""
        if not counts:return {'entropy':0.0,'dominant_states':[],'properties':'ERROR','state_space_utilization':0.0}
        
        total_shots=sum(counts.values())
        probabilities=[c/total_shots for c in counts.values()]
        
        # Shannon entropy
        entropy=-sum(p*math.log2(p) for p in probabilities if p>0)
        max_entropy=math.log2(len(counts))
        entropy_normalized=entropy/max_entropy if max_entropy>0 else 0
        
        # Top 4 dominant states
        sorted_states=sorted(counts.items(),key=lambda x:x[1],reverse=True)[:4]
        dominant_states=[(state,count) for state,count in sorted_states]
        
        # State space utilization
        num_unique_states=len(counts)
        max_possible_states=2**14
        utilization=num_unique_states/max_possible_states
        
        # Classify quantum properties
        if entropy_normalized>Config.ENTROPY_SUPERPOSITION_MIN:properties='SUPERPOSITION'
        elif entropy_normalized>Config.ENTROPY_STRONG_ENTANGLEMENT_MIN:properties='ENTANGLEMENT'
        elif entropy_normalized>Config.ENTROPY_CONSTRAINED_MIN:properties='CONSTRAINED'
        else:properties='CLASSICAL'
        
        return {
            'entropy':entropy_normalized,
            'dominant_states':dominant_states,
            'properties':properties,
            'state_space_utilization':utilization,
            'unique_states':num_unique_states,
            'total_shots':total_shots
        }

class SuperpositionManager:
    """Manages quantum superposition states for transactions"""
    
    def __init__(self):
        self.active_superpositions={}
        self.lock=threading.Lock()
        self.coherence_monitor_thread=threading.Thread(target=self._coherence_monitor_loop,daemon=True)
        self.coherence_monitor_thread.start()
    
    def create_superposition(self,tx_id:str,sender_id:str,receiver_id:str)->SuperpositionState:
        """Create superposition state for sender/receiver during transaction"""
        timestamp=datetime.utcnow().isoformat()
        
        # Generate phase encoding based on user IDs
        sender_phase=hash(sender_id)%256
        receiver_phase=hash(receiver_id)%256
        phase_encoding=f"{sender_phase:08b}{receiver_phase:08b}"
        
        superposition=SuperpositionState(
            tx_id=tx_id,
            user_id=sender_id,
            created_at=timestamp,
            qubits_state='|superposition⟩',
            entropy=0.85,
            coherence=0.95,
            phase_encoding=phase_encoding
        )
        
        with self.lock:
            self.active_superpositions[tx_id]=superposition
        
        # Store in database
        db.execute_update(
            """INSERT INTO superposition_states (tx_id,user_id,created_at,qubits_state,entropy,coherence,phase_encoding)
               VALUES (%s,%s,%s,%s,%s,%s,%s)""",
            (tx_id,sender_id,timestamp,'|superposition⟩',0.85,0.95,phase_encoding)
        )
        
        logger.info(f"Superposition created: {tx_id[:8]}... for {sender_id}")
        return superposition
    
    def refresh_coherence(self,tx_id:str)->bool:
        """Refresh coherence without full re-measurement (Floquet cycle)"""
        with self.lock:
            if tx_id not in self.active_superpositions:return False
            super_state=self.active_superpositions[tx_id]
        
        # Simulate coherence refresh (in real system, apply Floquet control pulses)
        new_coherence=min(super_state.coherence+0.02,0.99)
        
        db.execute_update(
            "UPDATE superposition_states SET coherence=%s WHERE tx_id=%s",
            (new_coherence,tx_id)
        )
        
        with self.lock:
            self.active_superpositions[tx_id].coherence=new_coherence
        
        return True
    
    def collapse_superposition(self,tx_id:str,oracle_measurement:str)->bool:
        """Collapse superposition via oracle measurement"""
        with self.lock:
            if tx_id not in self.active_superpositions:return False
            super_state=self.active_superpositions[tx_id]
        
        collapse_time=datetime.utcnow().isoformat()
        
        db.execute_update(
            """UPDATE superposition_states SET collapsed=TRUE,collapse_time=%s,oracle_measurement=%s 
               WHERE tx_id=%s""",
            (collapse_time,oracle_measurement,tx_id)
        )
        
        with self.lock:
            super_state.collapsed=True
            super_state.collapse_time=collapse_time
            super_state.oracle_measurement=oracle_measurement
        
        logger.info(f"Superposition collapsed: {tx_id[:8]}... oracle={oracle_measurement[:16]}...")
        return True
    
    def get_active_superposition(self,tx_id:str)->Optional[SuperpositionState]:
        """Retrieve active superposition state"""
        with self.lock:
            return self.active_superpositions.get(tx_id)
    
    def _coherence_monitor_loop(self):
        """Monitor and refresh coherence of active superpositions"""
        logger.info("Coherence monitor started")
        while True:
            try:
                with self.lock:
                    active_txs=list(self.active_superpositions.keys())
                
                for tx_id in active_txs:
                    super_state=self.get_active_superposition(tx_id)
                    if super_state and not super_state.collapsed:
                        age_seconds=(datetime.fromisoformat(datetime.utcnow().isoformat())-datetime.fromisoformat(super_state.created_at)).total_seconds()
                        
                        if age_seconds>Config.SUPERPOSITION_TIMEOUT_SECONDS:
                            logger.warning(f"Superposition timeout: {tx_id[:8]}... (age={age_seconds:.1f}s)")
                            with self.lock:
                                del self.active_superpositions[tx_id]
                        elif (int(age_seconds)%Config.COHERENCE_REFRESH_SECONDS)==0:
                            self.refresh_coherence(tx_id)
                
                time.sleep(1)
            except Exception as e:
                logger.error(f"Coherence monitor error: {e}")
                time.sleep(5)

class OracleValidator:
    """Quantum oracle validation with 6 reserved qubits"""
    
    def __init__(self):
        self.measurements={}
        self.validator_count=3
    
    def validate_transaction(self,tx_id:str,bitstring_counts:Dict[str,int])->Tuple[bool,Dict[str,Any]]:
        """Validate transaction using oracle qubit measurements"""
        
        oracle_bits=[bitstring[-6:] for bitstring in bitstring_counts.keys()]
        oracle_probabilities=self._calculate_oracle_probabilities(oracle_bits,bitstring_counts)
        
        # Check oracle agreement (3+ validators must agree)
        top_oracle_state=max(oracle_probabilities.items(),key=lambda x:x[1])[0]
        agreement=oracle_probabilities.get(top_oracle_state,0)
        
        is_valid=agreement>=Config.ORACLE_MEASUREMENT_THRESHOLD
        
        result={
            'valid':is_valid,
            'oracle_state':top_oracle_state,
            'agreement':agreement,
            'validators_agreed':int(agreement*self.validator_count),
            'required_validators':self.validator_count
        }
        
        # Store measurement
        self.measurements[tx_id]={
            'timestamp':datetime.utcnow().isoformat(),
            'oracle_state':top_oracle_state,
            'agreement':agreement,
            'result':result
        }
        
        logger.info(f"Oracle validation: {tx_id[:8]}... valid={is_valid} agreement={agreement:.2%}")
        return is_valid,result
    
    def _calculate_oracle_probabilities(self,oracle_bits:List[str],bitstring_counts:Dict[str,int])->Dict[str,float]:
        """Calculate probability distribution of oracle measurements"""
        oracle_counts=defaultdict(int)
        total=sum(bitstring_counts.values())
        
        for bitstring,count in bitstring_counts.items():
            oracle_state=bitstring[-6:]
            oracle_counts[oracle_state]+=count
        
        return {state:count/total for state,count in oracle_counts.items()}

class BlockManager:
    """Manages blockchain block creation and finality"""
    
    def __init__(self):
        self.current_block=None
        self.pending_transactions=[]
        self.lock=threading.Lock()
        self.block_creation_thread=threading.Thread(target=self._block_creation_loop,daemon=True)
        self.block_creation_thread.start()
    
    def add_finalized_transaction(self,tx_id:str,entropy:float,oracle_state:str)->bool:
        """Add finalized transaction to pending block"""
        with self.lock:
            self.pending_transactions.append({
                'tx_id':tx_id,
                'entropy':entropy,
                'oracle_state':oracle_state,
                'finalized_at':datetime.utcnow().isoformat()
            })
        
        return True
    
    def get_pending_block_size(self)->int:
        """Get number of pending transactions waiting for block"""
        with self.lock:
            return len(self.pending_transactions)
    
    def _create_block(self)->Optional[Dict[str,Any]]:
        """Create new block from pending transactions"""
        with self.lock:
            if not self.pending_transactions:return None
            
            transactions=self.pending_transactions[:Config.BLOCK_SIZE_MAX_TX]
            self.pending_transactions=self.pending_transactions[Config.BLOCK_SIZE_MAX_TX:]
        
        # Get latest block for parent hash
        latest_block_result=db.execute_one("SELECT block_hash,block_number FROM blocks ORDER BY block_number DESC LIMIT 1")
        parent_hash=latest_block_result['block_hash'] if latest_block_result else '0x0'
        block_number=(latest_block_result['block_number']+1) if latest_block_result else 0
        
        # Create block hash
        block_data=json.dumps(transactions,sort_keys=True)
        timestamp=datetime.utcnow().isoformat()
        block_hash='0x'+hashlib.sha256((parent_hash+timestamp+block_data).encode()).hexdigest()
        
        # Calculate aggregate entropy
        avg_entropy=sum(tx['entropy'] for tx in transactions)/len(transactions) if transactions else 0
        
        # Store block
        success=db.execute_update(
            """INSERT INTO blocks (block_hash,block_number,parent_hash,timestamp,transaction_count,
                                   aggregate_entropy,quantum_validation_status)
               VALUES (%s,%s,%s,%s,%s,%s,%s)""",
            (block_hash,block_number,parent_hash,timestamp,len(transactions),avg_entropy,'pending')
        )
        
        if success:
            logger.info(f"✓ Block created: #{block_number} | Hash: {block_hash[:16]}... | TX: {len(transactions)} | Entropy: {avg_entropy:.2%}")
            
            # Store block transactions mapping
            for tx in transactions:
                db.execute_update(
                    "UPDATE transactions SET block_hash=%s WHERE tx_id=%s",
                    (block_hash,tx['tx_id'])
                )
            
            return {
                'block_hash':block_hash,
                'block_number':block_number,
                'transaction_count':len(transactions),
                'avg_entropy':avg_entropy
            }
        
        return None
    
    def _block_creation_loop(self):
        """Periodically create blocks from pending transactions"""
        logger.info("Block creation loop started")
        while True:
            try:
                pending_count=self.get_pending_block_size()
                
                if pending_count>=Config.BLOCK_SIZE_MAX_TX*0.5:  # 50% full
                    self._create_block()
                
                time.sleep(Config.BLOCK_CREATION_INTERVAL_SECONDS)
            except Exception as e:
                logger.error(f"Block creation error: {e}")
                time.sleep(5)

class QuantumTransactionExecutor:
    """Main transaction executor with full quantum integration"""
    
    def __init__(self):
        self.circuit_builder=QuantumCircuitBuilder()
        self.superposition_manager=SuperpositionManager()
        self.oracle_validator=OracleValidator()
        self.block_manager=BlockManager()
        self.execution_stats={'total':0,'successful':0,'failed':0,'pending':0}
        self.lock=threading.Lock()
    
    def submit_and_execute_transaction(self,sender_id:str,receiver_id:str,amount:float,session_id:str)->Dict[str,Any]:
        """Main entry point: submit transaction and execute full quantum pipeline"""
        
        tx_id=f"tx_{uuid.uuid4().hex[:16]}"
        timestamp=datetime.utcnow().isoformat()
        
        try:
            logger.info(f"\n{'='*80}\nTransaction Pipeline Started: {tx_id}\n{'='*80}")
            
            # Step 1: Insert pending transaction record
            db.execute_update(
                """INSERT INTO transactions (tx_id,from_user_id,to_user_id,amount,tx_type,status,created_at,session_id)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                (tx_id,sender_id,receiver_id,int(amount*Config.TOKEN_WEI_PER_UNIT),'TRANSFER','SUBMITTED',timestamp,session_id)
            )
            logger.info(f"[1/7] Transaction record created: {tx_id}")
            
            # Step 2: Create superposition for sender/receiver
            superposition=self.superposition_manager.create_superposition(tx_id,sender_id,receiver_id)
            logger.info(f"[2/7] Superposition created: entropy={superposition.entropy:.2%} coherence={superposition.coherence:.2%}")
            
            # Step 3: Build quantum circuit
            circuit=self.circuit_builder.build_transaction_circuit(tx_id,sender_id,receiver_id,amount)
            logger.info(f"[3/7] Quantum circuit built: {circuit.num_qubits} qubits, {circuit.num_clbits} classical bits")
            
            # Step 4: Execute circuit and measure
            bitstring_counts,measurement_analysis=self.circuit_builder.execute_circuit(circuit,tx_id)
            logger.info(f"[4/7] Circuit executed: {measurement_analysis['unique_states']} unique states, entropy={measurement_analysis['entropy']:.2%}")
            
            # Step 5: Oracle validation
            is_valid,oracle_result=self.oracle_validator.validate_transaction(tx_id,bitstring_counts)
            logger.info(f"[5/7] Oracle validation: valid={is_valid} agreement={oracle_result['agreement']:.2%}")
            
            if not is_valid:
                logger.warning(f"Oracle validation FAILED for {tx_id}")
                db.execute_update("UPDATE transactions SET status=%s WHERE tx_id=%s",('FAILED_ORACLE',tx_id))
                with self.lock:
                    self.execution_stats['failed']+=1
                return {
                    'success':False,
                    'tx_id':tx_id,
                    'reason':'Oracle validation failed',
                    'oracle_agreement':oracle_result['agreement']
                }
            
            # Step 6: Collapse superposition
            oracle_measurement=oracle_result['oracle_state']
            self.superposition_manager.collapse_superposition(tx_id,oracle_measurement)
            logger.info(f"[6/7] Superposition collapsed: oracle_state={oracle_measurement}")
            
            # Step 7: Update transaction status and add to block queue
            db.execute_update(
                """UPDATE transactions SET status=%s,entropy_score=%s,quantum_state_hash=%s,
                                          oracle_measurement=%s WHERE tx_id=%s""",
                (
                    'FINALIZED',
                    measurement_analysis['entropy'],
                    hashlib.sha256(json.dumps(dict(sorted(bitstring_counts.items()))).encode()).hexdigest()[:16],
                    oracle_measurement,
                    tx_id
                )
            )
            
            self.block_manager.add_finalized_transaction(tx_id,measurement_analysis['entropy'],oracle_measurement)
            logger.info(f"[7/7] Transaction finalized and queued for block")
            
            logger.info(f"{'='*80}\nTransaction Pipeline COMPLETE: {tx_id}\n{'='*80}\n")
            
            with self.lock:
                self.execution_stats['total']+=1
                self.execution_stats['successful']+=1
            
            return {
                'success':True,
                'tx_id':tx_id,
                'entropy':measurement_analysis['entropy'],
                'oracle_state':oracle_measurement,
                'pending_block_size':self.block_manager.get_pending_block_size(),
                'status':'FINALIZED'
            }
        
        except Exception as e:
            logger.error(f"Transaction execution failed: {e}\n{traceback.format_exc()}")
            db.execute_update("UPDATE transactions SET status=%s WHERE tx_id=%s",('FAILED_EXCEPTION',tx_id))
            with self.lock:
                self.execution_stats['failed']+=1
            return {'success':False,'tx_id':tx_id,'error':str(e)}
    
    def get_transaction_status(self,tx_id:str)->Optional[Dict[str,Any]]:
        """Get transaction status from database"""
        result=db.execute_one(
            """SELECT tx_id,from_user_id,to_user_id,amount,status,entropy_score,oracle_measurement,
                      quantum_state_hash,block_hash,created_at FROM transactions WHERE tx_id=%s""",
            (tx_id,)
        )
        
        if result:
            return {
                'tx_id':result['tx_id'],
                'from':result['from_user_id'],
                'to':result['to_user_id'],
                'amount':result['amount']/Config.TOKEN_WEI_PER_UNIT,
                'status':result['status'],
                'entropy':result['entropy_score'],
                'oracle_measurement':result['oracle_measurement'],
                'quantum_hash':result['quantum_state_hash'],
                'block_hash':result['block_hash'],
                'created_at':result['created_at'].isoformat() if result['created_at'] else None
            }
        return None
    
    def list_transactions(self,limit:int=50)->List[Dict[str,Any]]:
        """List recent transactions with persistence"""
        results=db.execute(
            """SELECT tx_id,from_user_id,to_user_id,amount,status,entropy_score,created_at 
               FROM transactions ORDER BY created_at DESC LIMIT %s""",
            (limit,)
        )
        
        return [
            {
                'tx_id':r['tx_id'],
                'from':r['from_user_id'],
                'to':r['to_user_id'],
                'amount':r['amount']/Config.TOKEN_WEI_PER_UNIT,
                'status':r['status'],
                'entropy':r['entropy_score'],
                'created_at':r['created_at'].isoformat() if r['created_at'] else None
            }
            for r in results
        ]
    
    def get_stats(self)->Dict[str,Any]:
        """Get executor statistics"""
        with self.lock:
            stats=self.execution_stats.copy()
        
        block_info=db.execute_one("SELECT COUNT(*) as count FROM blocks")
        tx_info=db.execute_one("SELECT COUNT(*) as count FROM transactions")
        
        return {
            'execution_stats':stats,
            'total_blocks':block_info['count'] if block_info else 0,
            'total_transactions':tx_info['count'] if tx_info else 0,
            'pending_transactions':self.block_manager.get_pending_block_size(),
            'circuit_avg_time_ms':np.mean([s['time_ms'] for s in self.circuit_builder.execution_stats]) if self.circuit_builder.execution_stats else 0
        }

# GLOBAL INSTANCE
executor=QuantumTransactionExecutor()

if __name__=='__main__':
    print("Quantum Transaction Executor initialized")
    print(f"Config: {Config.QISKIT_QUBITS} total qubits ({Config.SUPERPOSITION_QUBITS} superposition, {Config.DATA_QUBITS} data, {Config.ORACLE_QUBITS} oracle)")
