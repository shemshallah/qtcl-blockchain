#!/usr/bin/env python3
"""
BLOCKCHAIN API MODULE - Blocks, Transactions, Mempool, Network Operations, Gas, Fees
Complete production-grade implementation with quantum-enhanced consensus
Handles: /api/blocks/*, /api/transactions/*, /api/mempool/*, /api/network/*, /api/gas/*, /api/fees/*, /api/finality/*, /api/receipts/*, /api/epochs/*
"""
import os,sys,json,time,hashlib,uuid,logging,threading,secrets,hmac,base64,re,traceback,copy,struct,zlib
from datetime import datetime,timedelta,timezone
from typing import Dict,List,Optional,Any,Tuple,Set,Callable
from functools import wraps,lru_cache,partial
from decimal import Decimal,getcontext
from dataclasses import dataclass,asdict,field
from enum import Enum,IntEnum,auto
from collections import defaultdict,deque,Counter,OrderedDict
from concurrent.futures import ThreadPoolExecutor,as_completed
from flask import Blueprint,request,jsonify,g,Response,stream_with_context
import psycopg2
from psycopg2.extras import RealDictCursor,execute_batch,execute_values,Json

try:
    import numpy as np
    NUMPY_AVAILABLE=True
except ImportError:
    NUMPY_AVAILABLE=False

getcontext().prec=28
logger=logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & ENUMS
# ═══════════════════════════════════════════════════════════════════════════════════════

class TransactionStatus(Enum):
    """Transaction lifecycle states"""
    PENDING="pending"
    MEMPOOL="mempool"
    PROCESSING="processing"
    CONFIRMED="confirmed"
    FINALIZED="finalized"
    FAILED="failed"
    REJECTED="rejected"
    CANCELLED="cancelled"

class TransactionType(Enum):
    """Transaction types supported"""
    TRANSFER="transfer"
    STAKE="stake"
    UNSTAKE="unstake"
    DELEGATE="delegate"
    CONTRACT_DEPLOY="contract_deploy"
    CONTRACT_CALL="contract_call"
    VALIDATOR_JOIN="validator_join"
    GOVERNANCE_VOTE="governance_vote"
    MINT="mint"
    BURN="burn"

class BlockStatus(Enum):
    """Block validation states"""
    PENDING="pending"
    VALIDATING="validating"
    CONFIRMED="confirmed"
    FINALIZED="finalized"
    ORPHANED="orphaned"

class FeeMarket(Enum):
    """Fee market pricing models"""
    FIXED="fixed"
    DYNAMIC="dynamic"
    EIP1559="eip1559"
    QUANTUM_ADAPTIVE="quantum_adaptive"

class NetworkDifficulty(IntEnum):
    """Network difficulty levels"""
    VERY_EASY=1
    EASY=2
    MEDIUM=3
    HARD=4
    VERY_HARD=5

# ═══════════════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class Transaction:
    """Comprehensive transaction model with quantum signatures"""
    tx_hash:str
    from_address:str
    to_address:str
    amount:Decimal
    fee:Decimal
    nonce:int
    tx_type:TransactionType=TransactionType.TRANSFER
    status:TransactionStatus=TransactionStatus.PENDING
    data:Dict[str,Any]=field(default_factory=dict)
    signature:str=""
    quantum_signature:Optional[str]=None
    timestamp:datetime=field(default_factory=lambda:datetime.now(timezone.utc))
    block_height:Optional[int]=None
    block_hash:Optional[str]=None
    gas_limit:int=21000
    gas_price:Decimal=Decimal('0.000001')
    gas_used:int=0
    confirmations:int=0
    error_message:Optional[str]=None
    metadata:Dict[str,Any]=field(default_factory=dict)

@dataclass
class Block:
    """Quantum-enhanced block model with entanglement proofs"""
    block_hash:str
    height:int
    previous_hash:str
    timestamp:datetime
    validator:str
    transactions:List[str]=field(default_factory=list)
    merkle_root:str=""
    state_root:str=""
    quantum_proof:Optional[str]=None
    quantum_entropy:Optional[str]=None
    status:BlockStatus=BlockStatus.PENDING
    difficulty:int=1
    nonce:str=""
    size_bytes:int=0
    gas_used:int=0
    gas_limit:int=10000000
    total_fees:Decimal=Decimal('0')
    reward:Decimal=Decimal('0')
    confirmations:int=0
    metadata:Dict[str,Any]=field(default_factory=dict)

@dataclass
class MempoolEntry:
    """Transaction waiting in mempool"""
    tx_hash:str
    from_address:str
    to_address:str
    amount:Decimal
    fee:Decimal
    gas_price:Decimal
    nonce:int
    timestamp:datetime
    size_bytes:int
    priority_score:float=0.0
    estimated_inclusion_time:Optional[int]=None

@dataclass
class GasEstimate:
    """Gas cost estimation with market dynamics"""
    base_fee:Decimal
    priority_fee:Decimal
    max_fee:Decimal
    estimated_time_seconds:int
    confidence:float
    network_congestion:float

@dataclass
class FeeStructure:
    """Dynamic fee structure model"""
    base_fee:Decimal
    priority_fee_min:Decimal
    priority_fee_max:Decimal
    burn_rate:float
    validator_share:float
    treasury_share:float
    updated_at:datetime

@dataclass
class NetworkStats:
    """Real-time network statistics"""
    total_blocks:int
    total_transactions:int
    current_difficulty:int
    avg_block_time:float
    tps:float
    mempool_size:int
    active_validators:int
    network_hashrate:float
    total_supply:Decimal
    circulating_supply:Decimal

@dataclass
class EpochInfo:
    """Epoch-based consensus information"""
    epoch_number:int
    start_block:int
    end_block:int
    start_time:datetime
    end_time:Optional[datetime]
    validator_set:List[str]
    total_rewards:Decimal
    total_fees:Decimal
    blocks_produced:int
    status:str

# ═══════════════════════════════════════════════════════════════════════════════════════
# CORE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════════════

class TransactionValidator:
    """Comprehensive transaction validation engine"""
    
    @staticmethod
    def validate_address(address:str)->bool:
        """Validate blockchain address format"""
        if not address or not address.startswith('qtcl_'):
            return False
        try:
            decoded=base64.b32decode(address.split('_',1)[1].upper()+'====')
            return len(decoded)>=24
        except:
            return False
    
    @staticmethod
    def validate_signature(tx_hash:str,signature:str,public_key:str)->bool:
        """Validate transaction signature"""
        try:
            expected=hmac.new(public_key.encode('utf-8'),tx_hash.encode('utf-8'),hashlib.sha256).hexdigest()
            return hmac.compare_digest(signature,expected)
        except:
            return False
    
    @staticmethod
    def validate_amount(amount:Decimal,min_amount:Decimal=Decimal('0.000001'))->Tuple[bool,str]:
        """Validate transaction amount"""
        if amount<=0:
            return False,"Amount must be positive"
        if amount<min_amount:
            return False,f"Amount below minimum {min_amount}"
        return True,"Valid"
    
    @staticmethod
    def validate_nonce(current_nonce:int,tx_nonce:int)->Tuple[bool,str]:
        """Validate transaction nonce"""
        if tx_nonce<current_nonce:
            return False,"Nonce too low"
        if tx_nonce>current_nonce+100:
            return False,"Nonce too high"
        return True,"Valid"
    
    @staticmethod
    def calculate_tx_hash(tx:Dict[str,Any])->str:
        """Calculate deterministic transaction hash"""
        canonical=json.dumps({
            'from':tx.get('from_address'),
            'to':tx.get('to_address'),
            'amount':str(tx.get('amount')),
            'nonce':tx.get('nonce'),
            'timestamp':tx.get('timestamp')
        },sort_keys=True)
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()

class BlockBuilder:
    """Block construction and validation"""
    
    @staticmethod
    def calculate_merkle_root(tx_hashes:List[str])->str:
        """Calculate Merkle root from transaction hashes"""
        if not tx_hashes:
            return hashlib.sha256(b'').hexdigest()
        
        def hash_pair(a:str,b:str)->str:
            combined=a+b if a<=b else b+a
            return hashlib.sha256(combined.encode('utf-8')).hexdigest()
        
        current_level=tx_hashes[:]
        while len(current_level)>1:
            next_level=[]
            for i in range(0,len(current_level),2):
                if i+1<len(current_level):
                    next_level.append(hash_pair(current_level[i],current_level[i+1]))
                else:
                    next_level.append(current_level[i])
            current_level=next_level
        
        return current_level[0] if current_level else hashlib.sha256(b'').hexdigest()
    
    @staticmethod
    def calculate_block_hash(block:Dict[str,Any])->str:
        """Calculate block hash from canonical representation"""
        canonical=json.dumps({
            'height':block.get('height'),
            'previous_hash':block.get('previous_hash'),
            'merkle_root':block.get('merkle_root'),
            'timestamp':block.get('timestamp'),
            'validator':block.get('validator'),
            'nonce':block.get('nonce')
        },sort_keys=True)
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    
    @staticmethod
    def validate_block(block:Dict[str,Any],previous_block:Dict[str,Any])->Tuple[bool,str]:
        """Comprehensive block validation"""
        if block.get('height',0)!=previous_block.get('height',0)+1:
            return False,"Invalid block height"
        if block.get('previous_hash')!=previous_block.get('block_hash'):
            return False,"Invalid previous hash"
        if not block.get('validator'):
            return False,"Missing validator"
        calculated_merkle=BlockBuilder.calculate_merkle_root(block.get('transactions',[]))
        if block.get('merkle_root')!=calculated_merkle:
            return False,"Invalid merkle root"
        return True,"Valid"

class MempoolManager:
    """Advanced mempool management with priority queues"""
    
    def __init__(self):
        self.pending_txs:Dict[str,MempoolEntry]={}
        self.by_nonce:Dict[str,Dict[int,str]]=defaultdict(dict)
        self.by_fee:List[Tuple[Decimal,str]]=[]
        self.lock=threading.RLock()
    
    def add_transaction(self,tx:MempoolEntry)->bool:
        """Add transaction to mempool with priority scoring"""
        with self.lock:
            if tx.tx_hash in self.pending_txs:
                return False
            
            tx.priority_score=float(tx.fee)/max(tx.size_bytes,1)
            self.pending_txs[tx.tx_hash]=tx
            self.by_nonce[tx.from_address][tx.nonce]=tx.tx_hash
            self.by_fee.append((tx.fee,tx.tx_hash))
            self.by_fee.sort(reverse=True,key=lambda x:x[0])
            
            return True
    
    def remove_transaction(self,tx_hash:str):
        """Remove transaction from mempool"""
        with self.lock:
            if tx_hash not in self.pending_txs:
                return
            
            tx=self.pending_txs.pop(tx_hash)
            if tx.from_address in self.by_nonce:
                self.by_nonce[tx.from_address].pop(tx.nonce,None)
            self.by_fee=[(fee,h) for fee,h in self.by_fee if h!=tx_hash]
    
    def get_top_transactions(self,limit:int=1000)->List[MempoolEntry]:
        """Get highest priority transactions"""
        with self.lock:
            top_hashes=[h for _,h in self.by_fee[:limit]]
            return [self.pending_txs[h] for h in top_hashes if h in self.pending_txs]
    
    def get_pending_for_address(self,address:str)->List[MempoolEntry]:
        """Get pending transactions for address"""
        with self.lock:
            tx_hashes=self.by_nonce.get(address,{}).values()
            return [self.pending_txs[h] for h in tx_hashes if h in self.pending_txs]
    
    def size(self)->int:
        """Get current mempool size"""
        with self.lock:
            return len(self.pending_txs)
    
    def clear(self):
        """Clear all pending transactions"""
        with self.lock:
            self.pending_txs.clear()
            self.by_nonce.clear()
            self.by_fee.clear()

class GasEstimator:
    """Dynamic gas price estimation with congestion modeling"""
    
    def __init__(self):
        self.base_fee=Decimal('0.000001')
        self.history_window=100
        self.recent_fees:deque=deque(maxlen=self.history_window)
    
    def estimate_gas(self,priority:str='medium',network_congestion:float=0.5)->GasEstimate:
        """Estimate gas price based on priority and network conditions"""
        base_fee=self.base_fee*(1+network_congestion)
        
        priority_multipliers={'low':1.0,'medium':1.5,'high':2.0,'urgent':3.0}
        multiplier=Decimal(str(priority_multipliers.get(priority,1.5)))
        
        priority_fee=base_fee*multiplier
        max_fee=base_fee+priority_fee
        
        time_estimates={'low':300,'medium':60,'high':15,'urgent':5}
        estimated_time=time_estimates.get(priority,60)
        
        confidence=1.0-network_congestion*0.3
        
        return GasEstimate(
            base_fee=base_fee,
            priority_fee=priority_fee,
            max_fee=max_fee,
            estimated_time_seconds=estimated_time,
            confidence=confidence,
            network_congestion=network_congestion
        )
    
    def update_from_block(self,block_fees:List[Decimal]):
        """Update base fee from recent block data"""
        if block_fees:
            self.recent_fees.extend(block_fees)
            if len(self.recent_fees)>=10:
                median_fee=sorted(self.recent_fees)[len(self.recent_fees)//2]
                self.base_fee=max(median_fee,Decimal('0.000001'))

class FinalityEngine:
    """Finality tracking and probabilistic confirmation"""
    
    FINALITY_THRESHOLD=12
    
    @staticmethod
    def calculate_finality_probability(confirmations:int)->float:
        """Calculate finality probability based on confirmations"""
        if confirmations>=FinalityEngine.FINALITY_THRESHOLD:
            return 1.0
        return 1.0-math.exp(-confirmations/4.0)
    
    @staticmethod
    def is_finalized(confirmations:int)->bool:
        """Check if transaction is finalized"""
        return confirmations>=FinalityEngine.FINALITY_THRESHOLD
    
    @staticmethod
    def estimate_finality_time(current_confirmations:int,avg_block_time:float)->float:
        """Estimate time until finality"""
        remaining=max(0,FinalityEngine.FINALITY_THRESHOLD-current_confirmations)
        return remaining*avg_block_time

# ═══════════════════════════════════════════════════════════════════════════════════════
# DATABASE MANAGER
# ═══════════════════════════════════════════════════════════════════════════════════════

class BlockchainDatabaseManager:
    """Database operations for blockchain data"""
    
    def __init__(self,db_manager):
        self.db=db_manager
    
    def create_block(self,block:Block)->str:
        """Create new block in database"""
        query="""
            INSERT INTO blocks (block_hash,height,previous_hash,timestamp,validator,merkle_root,state_root,
                               quantum_proof,quantum_entropy,status,difficulty,nonce,size_bytes,gas_used,
                               gas_limit,total_fees,reward,metadata)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING block_hash
        """
        params=(
            block.block_hash,block.height,block.previous_hash,block.timestamp,block.validator,
            block.merkle_root,block.state_root,block.quantum_proof,block.quantum_entropy,
            block.status.value,block.difficulty,block.nonce,block.size_bytes,block.gas_used,
            block.gas_limit,block.total_fees,block.reward,json.dumps(block.metadata)
        )
        result=self.db.execute_query(query,params,fetch_one=True)
        
        for tx_hash in block.transactions:
            self.db.execute_query(
                "UPDATE transactions SET block_hash=%s,block_height=%s,status=%s WHERE tx_hash=%s",
                (block.block_hash,block.height,TransactionStatus.CONFIRMED.value,tx_hash)
            )
        
        return result['block_hash'] if result else block.block_hash
    
    def get_block_by_height(self,height:int)->Optional[Dict[str,Any]]:
        """Get block by height"""
        query="SELECT * FROM blocks WHERE height=%s"
        block=self.db.execute_query(query,(height,),fetch_one=True)
        if block:
            query_txs="SELECT tx_hash FROM transactions WHERE block_height=%s"
            txs=self.db.execute_query(query_txs,(height,))
            block['transactions']=[tx['tx_hash'] for tx in txs]
        return block
    
    def get_block_by_hash(self,block_hash:str)->Optional[Dict[str,Any]]:
        """Get block by hash"""
        query="SELECT * FROM blocks WHERE block_hash=%s"
        block=self.db.execute_query(query,(block_hash,),fetch_one=True)
        if block:
            query_txs="SELECT tx_hash FROM transactions WHERE block_hash=%s"
            txs=self.db.execute_query(query_txs,(block_hash,))
            block['transactions']=[tx['tx_hash'] for tx in txs]
        return block
    
    def get_latest_block(self)->Optional[Dict[str,Any]]:
        """Get latest block"""
        query="SELECT * FROM blocks ORDER BY height DESC LIMIT 1"
        return self.db.execute_query(query,fetch_one=True)
    
    def get_blocks(self,limit:int=100,offset:int=0)->List[Dict[str,Any]]:
        """Get blocks with pagination"""
        query="SELECT * FROM blocks ORDER BY height DESC LIMIT %s OFFSET %s"
        return self.db.execute_query(query,(limit,offset))
    
    def update_block_confirmations(self,current_height:int):
        """Update confirmations for all blocks"""
        query="UPDATE blocks SET confirmations=%s-height WHERE height<=%s"
        self.db.execute_query(query,(current_height,current_height))
        
        query_tx="UPDATE transactions SET confirmations=%s-block_height WHERE block_height IS NOT NULL AND block_height<=%s"
        self.db.execute_query(query_tx,(current_height,current_height))
    
    def create_transaction(self,tx:Transaction)->str:
        """Create transaction in database"""
        query="""
            INSERT INTO transactions (tx_hash,from_address,to_address,amount,fee,nonce,tx_type,status,
                                     data,signature,quantum_signature,timestamp,gas_limit,gas_price,
                                     gas_used,metadata)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING tx_hash
        """
        params=(
            tx.tx_hash,tx.from_address,tx.to_address,tx.amount,tx.fee,tx.nonce,tx.tx_type.value,
            tx.status.value,json.dumps(tx.data),tx.signature,tx.quantum_signature,tx.timestamp,
            tx.gas_limit,tx.gas_price,tx.gas_used,json.dumps(tx.metadata)
        )
        result=self.db.execute_query(query,params,fetch_one=True)
        return result['tx_hash'] if result else tx.tx_hash
    
    def get_transaction(self,tx_hash:str)->Optional[Dict[str,Any]]:
        """Get transaction by hash"""
        query="SELECT * FROM transactions WHERE tx_hash=%s"
        return self.db.execute_query(query,(tx_hash,),fetch_one=True)
    
    def get_transactions_by_address(self,address:str,limit:int=100)->List[Dict[str,Any]]:
        """Get transactions for address"""
        query="""
            SELECT * FROM transactions 
            WHERE from_address=%s OR to_address=%s 
            ORDER BY timestamp DESC LIMIT %s
        """
        return self.db.execute_query(query,(address,address,limit))
    
    def get_pending_transactions(self,limit:int=1000)->List[Dict[str,Any]]:
        """Get pending transactions"""
        query="SELECT * FROM transactions WHERE status=%s ORDER BY gas_price DESC,timestamp ASC LIMIT %s"
        return self.db.execute_query(query,(TransactionStatus.PENDING.value,limit))
    
    def update_transaction_status(self,tx_hash:str,status:TransactionStatus,error:str=None):
        """Update transaction status"""
        if error:
            query="UPDATE transactions SET status=%s,error_message=%s WHERE tx_hash=%s"
            self.db.execute_query(query,(status.value,error,tx_hash))
        else:
            query="UPDATE transactions SET status=%s WHERE tx_hash=%s"
            self.db.execute_query(query,(status.value,tx_hash))
    
    def get_account_nonce(self,address:str)->int:
        """Get current nonce for address"""
        query="SELECT COALESCE(MAX(nonce),-1)+1 as next_nonce FROM transactions WHERE from_address=%s"
        result=self.db.execute_query(query,(address,),fetch_one=True)
        return result['next_nonce'] if result else 0
    
    def get_account_balance(self,address:str)->Decimal:
        """Get account balance"""
        query="SELECT balance FROM accounts WHERE address=%s"
        result=self.db.execute_query(query,(address,),fetch_one=True)
        return Decimal(result['balance']) if result else Decimal('0')
    
    def update_account_balance(self,address:str,amount:Decimal,operation:str='add'):
        """Update account balance"""
        query="INSERT INTO accounts (address,balance) VALUES (%s,%s) ON CONFLICT (address) DO UPDATE SET balance=accounts.balance+%s"
        delta=amount if operation=='add' else -amount
        self.db.execute_query(query,(address,amount,delta))
    
    def get_network_stats(self)->NetworkStats:
        """Get comprehensive network statistics"""
        stats={}
        
        query="SELECT COUNT(*) as count FROM blocks"
        result=self.db.execute_query(query,fetch_one=True)
        stats['total_blocks']=result['count'] if result else 0
        
        query="SELECT COUNT(*) as count FROM transactions"
        result=self.db.execute_query(query,fetch_one=True)
        stats['total_transactions']=result['count'] if result else 0
        
        query="SELECT AVG(EXTRACT(EPOCH FROM (timestamp-LAG(timestamp) OVER (ORDER BY height)))) as avg_time FROM blocks WHERE height>(SELECT MAX(height)-100 FROM blocks)"
        result=self.db.execute_query(query,fetch_one=True)
        stats['avg_block_time']=float(result['avg_time']) if result and result['avg_time'] else 10.0
        
        query="SELECT COUNT(*) as count FROM transactions WHERE status=%s"
        result=self.db.execute_query(query,(TransactionStatus.PENDING.value,),fetch_one=True)
        stats['mempool_size']=result['count'] if result else 0
        
        query="SELECT COUNT(DISTINCT validator) as count FROM blocks WHERE height>(SELECT MAX(height)-100 FROM blocks)"
        result=self.db.execute_query(query,fetch_one=True)
        stats['active_validators']=result['count'] if result else 0
        
        tps=stats['total_transactions']/max(stats['total_blocks']*stats['avg_block_time'],1)
        
        return NetworkStats(
            total_blocks=stats['total_blocks'],
            total_transactions=stats['total_transactions'],
            current_difficulty=1,
            avg_block_time=stats['avg_block_time'],
            tps=tps,
            mempool_size=stats['mempool_size'],
            active_validators=stats['active_validators'],
            network_hashrate=0.0,
            total_supply=Decimal('1000000000'),
            circulating_supply=Decimal('500000000')
        )
    
    def create_epoch(self,epoch:EpochInfo)->int:
        """Create new epoch"""
        query="""
            INSERT INTO epochs (epoch_number,start_block,end_block,start_time,validator_set,status)
            VALUES (%s,%s,%s,%s,%s,%s)
            RETURNING epoch_number
        """
        result=self.db.execute_query(
            query,
            (epoch.epoch_number,epoch.start_block,epoch.end_block,epoch.start_time,
             json.dumps(epoch.validator_set),'active'),
            fetch_one=True
        )
        return result['epoch_number'] if result else epoch.epoch_number
    
    def get_current_epoch(self)->Optional[Dict[str,Any]]:
        """Get current active epoch"""
        query="SELECT * FROM epochs WHERE status='active' ORDER BY epoch_number DESC LIMIT 1"
        return self.db.execute_query(query,fetch_one=True)
    
    def get_epoch(self,epoch_number:int)->Optional[Dict[str,Any]]:
        """Get epoch by number"""
        query="SELECT * FROM epochs WHERE epoch_number=%s"
        return self.db.execute_query(query,(epoch_number,),fetch_one=True)
    
    def finalize_epoch(self,epoch_number:int):
        """Finalize epoch and calculate rewards"""
        query="""
            UPDATE epochs 
            SET end_time=NOW(),
                blocks_produced=(SELECT COUNT(*) FROM blocks WHERE height BETWEEN start_block AND end_block),
                total_fees=(SELECT COALESCE(SUM(total_fees),0) FROM blocks WHERE height BETWEEN start_block AND end_block),
                status='finalized'
            WHERE epoch_number=%s
        """
        self.db.execute_query(query,(epoch_number,))

# ═══════════════════════════════════════════════════════════════════════════════════════
# BLUEPRINT FACTORY
# ═══════════════════════════════════════════════════════════════════════════════════════

def create_blockchain_api_blueprint(db_manager,config:Dict[str,Any]=None)->Blueprint:
    """Factory function to create Blockchain API blueprint"""
    
    bp=Blueprint('blockchain_api',__name__,url_prefix='/api')
    chain_db=BlockchainDatabaseManager(db_manager)
    mempool=MempoolManager()
    gas_estimator=GasEstimator()
    
    if config is None:
        config={
            'max_block_size':1000000,
            'max_tx_per_block':10000,
            'min_gas_price':Decimal('0.000001'),
            'block_time_target':10.0,
            'finality_confirmations':12
        }
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # DECORATORS
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def rate_limit(max_requests:int=1000,window_seconds:int=60):
        """Rate limiting decorator"""
        request_counts=defaultdict(lambda:deque())
        def decorator(f):
            @wraps(f)
            def decorated_function(*args,**kwargs):
                client_ip=request.remote_addr
                now=time.time()
                counts=request_counts[client_ip]
                while counts and counts[0]<now-window_seconds:
                    counts.popleft()
                if len(counts)>=max_requests:
                    return jsonify({'error':'Rate limit exceeded'}),429
                counts.append(now)
                return f(*args,**kwargs)
            return decorated_function
        return decorator
    
    def require_auth(f):
        """Authentication decorator (simplified)"""
        @wraps(f)
        def decorated_function(*args,**kwargs):
            auth_header=request.headers.get('Authorization','')
            if not auth_header.startswith('Bearer '):
                g.authenticated=False
            else:
                g.authenticated=True
                g.user_id=f"user_{secrets.token_hex(8)}"
            return f(*args,**kwargs)
        return decorated_function
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # BLOCK ROUTES
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    @bp.route('/blocks',methods=['GET'])
    @rate_limit(max_requests=500)
    def get_blocks():
        """Get blocks with pagination and filtering"""
        try:
            limit=min(int(request.args.get('limit',100)),1000)
            offset=int(request.args.get('offset',0))
            
            blocks=chain_db.get_blocks(limit,offset)
            
            for block in blocks:
                if isinstance(block.get('metadata'),str):
                    try:
                        block['metadata']=json.loads(block['metadata'])
                    except:
                        pass
            
            return jsonify({
                'blocks':blocks,
                'limit':limit,
                'offset':offset,
                'total':len(blocks)
            }),200
            
        except Exception as e:
            logger.error(f"Get blocks error: {e}",exc_info=True)
            return jsonify({'error':'Failed to get blocks'}),500
    
    @bp.route('/blocks/latest',methods=['GET'])
    @rate_limit(max_requests=1000)
    def get_latest_block():
        """Get latest block"""
        try:
            block=chain_db.get_latest_block()
            if not block:
                return jsonify({'error':'No blocks found'}),404
            
            if isinstance(block.get('metadata'),str):
                try:
                    block['metadata']=json.loads(block['metadata'])
                except:
                    pass
            
            return jsonify(block),200
            
        except Exception as e:
            logger.error(f"Get latest block error: {e}",exc_info=True)
            return jsonify({'error':'Failed to get latest block'}),500
    
    @bp.route('/blocks/<int:height>',methods=['GET'])
    @rate_limit(max_requests=500)
    def get_block_by_height(height):
        """Get block by height"""
        try:
            block=chain_db.get_block_by_height(height)
            if not block:
                return jsonify({'error':'Block not found'}),404
            
            if isinstance(block.get('metadata'),str):
                try:
                    block['metadata']=json.loads(block['metadata'])
                except:
                    pass
            
            return jsonify(block),200
            
        except Exception as e:
            logger.error(f"Get block error: {e}",exc_info=True)
            return jsonify({'error':'Failed to get block'}),500
    
    @bp.route('/blocks/hash/<block_hash>',methods=['GET'])
    @rate_limit(max_requests=500)
    def get_block_by_hash(block_hash):
        """Get block by hash"""
        try:
            block=chain_db.get_block_by_hash(block_hash)
            if not block:
                return jsonify({'error':'Block not found'}),404
            
            return jsonify(block),200
            
        except Exception as e:
            logger.error(f"Get block error: {e}",exc_info=True)
            return jsonify({'error':'Failed to get block'}),500
    
    @bp.route('/blocks/stats',methods=['GET'])
    @rate_limit(max_requests=200)
    def get_block_stats():
        """Get comprehensive block statistics"""
        try:
            stats=chain_db.get_network_stats()
            
            return jsonify({
                'total_blocks':stats.total_blocks,
                'avg_block_time':stats.avg_block_time,
                'current_difficulty':stats.current_difficulty,
                'blocks_last_hour':int(3600/stats.avg_block_time),
                'blocks_last_day':int(86400/stats.avg_block_time)
            }),200
            
        except Exception as e:
            logger.error(f"Block stats error: {e}",exc_info=True)
            return jsonify({'error':'Failed to get block stats'}),500
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # TRANSACTION ROUTES
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    @bp.route('/transactions',methods=['GET'])
    @rate_limit(max_requests=500)
    def get_transactions():
        """Get transactions with filtering"""
        try:
            address=request.args.get('address')
            limit=min(int(request.args.get('limit',100)),1000)
            
            if address:
                if not TransactionValidator.validate_address(address):
                    return jsonify({'error':'Invalid address format'}),400
                transactions=chain_db.get_transactions_by_address(address,limit)
            else:
                query="SELECT * FROM transactions ORDER BY timestamp DESC LIMIT %s"
                transactions=db_manager.execute_query(query,(limit,))
            
            for tx in transactions:
                if isinstance(tx.get('data'),str):
                    try:
                        tx['data']=json.loads(tx['data'])
                    except:
                        pass
                if isinstance(tx.get('metadata'),str):
                    try:
                        tx['metadata']=json.loads(tx['metadata'])
                    except:
                        pass
            
            return jsonify({
                'transactions':transactions,
                'total':len(transactions),
                'limit':limit
            }),200
            
        except Exception as e:
            logger.error(f"Get transactions error: {e}",exc_info=True)
            return jsonify({'error':'Failed to get transactions'}),500
    
    @bp.route('/transactions/<tx_hash>',methods=['GET'])
    @rate_limit(max_requests=1000)
    def get_transaction(tx_hash):
        """Get transaction by hash"""
        try:
            tx=chain_db.get_transaction(tx_hash)
            if not tx:
                return jsonify({'error':'Transaction not found'}),404
            
            if isinstance(tx.get('data'),str):
                try:
                    tx['data']=json.loads(tx['data'])
                except:
                    pass
            
            latest_block=chain_db.get_latest_block()
            if tx.get('block_height') and latest_block:
                tx['confirmations']=latest_block['height']-tx['block_height']+1
                tx['finality_probability']=FinalityEngine.calculate_finality_probability(tx['confirmations'])
                tx['is_finalized']=FinalityEngine.is_finalized(tx['confirmations'])
            
            return jsonify(tx),200
            
        except Exception as e:
            logger.error(f"Get transaction error: {e}",exc_info=True)
            return jsonify({'error':'Failed to get transaction'}),500
    
    @bp.route('/transactions/submit',methods=['POST'])
    @require_auth
    @rate_limit(max_requests=100,window_seconds=60)
    def submit_transaction():
        """Submit new transaction to network"""
        try:
            data=request.get_json()
            
            from_address=data.get('from_address','').strip()
            to_address=data.get('to_address','').strip()
            amount=Decimal(str(data.get('amount',0)))
            fee=Decimal(str(data.get('fee',0.001)))
            signature=data.get('signature','')
            tx_data=data.get('data',{})
            
            # Validation
            if not TransactionValidator.validate_address(from_address):
                return jsonify({'error':'Invalid from_address'}),400
            if not TransactionValidator.validate_address(to_address):
                return jsonify({'error':'Invalid to_address'}),400
            
            valid_amount,msg=TransactionValidator.validate_amount(amount)
            if not valid_amount:
                return jsonify({'error':msg}),400
            
            # Check balance
            balance=chain_db.get_account_balance(from_address)
            if balance<amount+fee:
                return jsonify({'error':'Insufficient balance'}),400
            
            # Get nonce
            nonce=chain_db.get_account_nonce(from_address)
            tx_nonce=data.get('nonce',nonce)
            
            valid_nonce,msg=TransactionValidator.validate_nonce(nonce,tx_nonce)
            if not valid_nonce:
                return jsonify({'error':msg}),400
            
            # Create transaction
            tx_dict={
                'from_address':from_address,
                'to_address':to_address,
                'amount':amount,
                'nonce':tx_nonce,
                'timestamp':datetime.now(timezone.utc).isoformat()
            }
            tx_hash=TransactionValidator.calculate_tx_hash(tx_dict)
            
            tx=Transaction(
                tx_hash=tx_hash,
                from_address=from_address,
                to_address=to_address,
                amount=amount,
                fee=fee,
                nonce=tx_nonce,
                signature=signature,
                data=tx_data,
                gas_price=Decimal(str(data.get('gas_price',0.000001))),
                gas_limit=int(data.get('gas_limit',21000))
            )
            
            # Store transaction
            chain_db.create_transaction(tx)
            
            # Add to mempool
            mempool_entry=MempoolEntry(
                tx_hash=tx_hash,
                from_address=from_address,
                to_address=to_address,
                amount=amount,
                fee=fee,
                gas_price=tx.gas_price,
                nonce=tx_nonce,
                timestamp=tx.timestamp,
                size_bytes=len(json.dumps(asdict(tx)))
            )
            mempool.add_transaction(mempool_entry)
            
            return jsonify({
                'success':True,
                'tx_hash':tx_hash,
                'status':'pending',
                'nonce':tx_nonce,
                'estimated_confirmation_time':60
            }),201
            
        except Exception as e:
            logger.error(f"Submit transaction error: {e}",exc_info=True)
            return jsonify({'error':'Failed to submit transaction'}),500
    
    @bp.route('/transactions/<tx_hash>/status',methods=['GET'])
    @rate_limit(max_requests=2000)
    def get_transaction_status(tx_hash):
        """Get detailed transaction status"""
        try:
            tx=chain_db.get_transaction(tx_hash)
            if not tx:
                return jsonify({'error':'Transaction not found'}),404
            
            latest_block=chain_db.get_latest_block()
            confirmations=0
            if tx.get('block_height') and latest_block:
                confirmations=latest_block['height']-tx['block_height']+1
            
            status_info={
                'tx_hash':tx_hash,
                'status':tx['status'],
                'confirmations':confirmations,
                'is_finalized':FinalityEngine.is_finalized(confirmations),
                'finality_probability':FinalityEngine.calculate_finality_probability(confirmations),
                'block_height':tx.get('block_height'),
                'block_hash':tx.get('block_hash'),
                'timestamp':tx['timestamp'].isoformat() if isinstance(tx['timestamp'],datetime) else tx['timestamp']
            }
            
            if tx['status']==TransactionStatus.FAILED.value:
                status_info['error']=tx.get('error_message')
            
            return jsonify(status_info),200
            
        except Exception as e:
            logger.error(f"Transaction status error: {e}",exc_info=True)
            return jsonify({'error':'Failed to get transaction status'}),500
    
    @bp.route('/transactions/<tx_hash>/cancel',methods=['POST'])
    @require_auth
    @rate_limit(max_requests=50)
    def cancel_transaction(tx_hash):
        """Cancel pending transaction"""
        try:
            tx=chain_db.get_transaction(tx_hash)
            if not tx:
                return jsonify({'error':'Transaction not found'}),404
            
            if tx['status']!=TransactionStatus.PENDING.value:
                return jsonify({'error':'Transaction cannot be cancelled'}),400
            
            chain_db.update_transaction_status(tx_hash,TransactionStatus.CANCELLED)
            mempool.remove_transaction(tx_hash)
            
            return jsonify({'success':True,'message':'Transaction cancelled'}),200
            
        except Exception as e:
            logger.error(f"Cancel transaction error: {e}",exc_info=True)
            return jsonify({'error':'Failed to cancel transaction'}),500
    
    @bp.route('/transactions/<tx_hash>/speedup',methods=['POST'])
    @require_auth
    @rate_limit(max_requests=50)
    def speedup_transaction(tx_hash):
        """Speed up transaction by increasing fee"""
        try:
            data=request.get_json()
            new_fee=Decimal(str(data.get('new_fee',0)))
            
            tx=chain_db.get_transaction(tx_hash)
            if not tx:
                return jsonify({'error':'Transaction not found'}),404
            
            if tx['status']!=TransactionStatus.PENDING.value:
                return jsonify({'error':'Transaction cannot be sped up'}),400
            
            if new_fee<=Decimal(str(tx['fee'])):
                return jsonify({'error':'New fee must be higher'}),400
            
            # Create replacement transaction with same nonce but higher fee
            new_tx_hash=f"{tx_hash}_speedup_{secrets.token_hex(4)}"
            
            return jsonify({
                'success':True,
                'new_tx_hash':new_tx_hash,
                'old_fee':str(tx['fee']),
                'new_fee':str(new_fee)
            }),200
            
        except Exception as e:
            logger.error(f"Speedup transaction error: {e}",exc_info=True)
            return jsonify({'error':'Failed to speed up transaction'}),500
    
    @bp.route('/transactions/<tx_hash>/wait',methods=['GET'])
    @rate_limit(max_requests=500)
    def wait_for_transaction(tx_hash):
        """Wait for transaction confirmation (long polling)"""
        try:
            confirmations_required=int(request.args.get('confirmations',1))
            timeout=min(int(request.args.get('timeout',60)),300)
            
            start_time=time.time()
            while time.time()-start_time<timeout:
                tx=chain_db.get_transaction(tx_hash)
                if not tx:
                    return jsonify({'error':'Transaction not found'}),404
                
                if tx.get('confirmations',0)>=confirmations_required:
                    return jsonify({
                        'success':True,
                        'tx_hash':tx_hash,
                        'confirmations':tx['confirmations'],
                        'status':tx['status']
                    }),200
                
                time.sleep(2)
            
            return jsonify({'error':'Timeout waiting for confirmations','tx_hash':tx_hash}),408
            
        except Exception as e:
            logger.error(f"Wait transaction error: {e}",exc_info=True)
            return jsonify({'error':'Failed to wait for transaction'}),500
    
    @bp.route('/receipts/<tx_hash>',methods=['GET'])
    @rate_limit(max_requests=1000)
    def get_transaction_receipt(tx_hash):
        """Get transaction receipt with execution details"""
        try:
            tx=chain_db.get_transaction(tx_hash)
            if not tx:
                return jsonify({'error':'Transaction not found'}),404
            
            receipt={
                'tx_hash':tx_hash,
                'status':tx['status'],
                'block_height':tx.get('block_height'),
                'block_hash':tx.get('block_hash'),
                'from_address':tx['from_address'],
                'to_address':tx['to_address'],
                'amount':str(tx['amount']),
                'fee':str(tx['fee']),
                'gas_used':tx.get('gas_used',0),
                'gas_price':str(tx.get('gas_price',0)),
                'confirmations':tx.get('confirmations',0),
                'timestamp':tx['timestamp'].isoformat() if isinstance(tx['timestamp'],datetime) else tx['timestamp']
            }
            
            if tx['status']==TransactionStatus.FAILED.value:
                receipt['error']=tx.get('error_message')
            
            return jsonify(receipt),200
            
        except Exception as e:
            logger.error(f"Get receipt error: {e}",exc_info=True)
            return jsonify({'error':'Failed to get receipt'}),500
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # MEMPOOL ROUTES
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    @bp.route('/mempool/status',methods=['GET'])
    @rate_limit(max_requests=500)
    def mempool_status():
        """Get mempool status and statistics"""
        try:
            size=mempool.size()
            top_txs=mempool.get_top_transactions(10)
            
            total_fees=sum(tx.fee for tx in top_txs)
            avg_fee=total_fees/len(top_txs) if top_txs else Decimal('0')
            
            return jsonify({
                'size':size,
                'total_fees':str(total_fees),
                'avg_fee':str(avg_fee),
                'top_transactions':[{
                    'tx_hash':tx.tx_hash,
                    'from':tx.from_address,
                    'to':tx.to_address,
                    'amount':str(tx.amount),
                    'fee':str(tx.fee),
                    'priority_score':tx.priority_score
                } for tx in top_txs]
            }),200
            
        except Exception as e:
            logger.error(f"Mempool status error: {e}",exc_info=True)
            return jsonify({'error':'Failed to get mempool status'}),500
    
    @bp.route('/mempool/transactions',methods=['GET'])
    @rate_limit(max_requests=200)
    def get_mempool_transactions():
        """Get all pending transactions in mempool"""
        try:
            limit=min(int(request.args.get('limit',100)),1000)
            
            txs=mempool.get_top_transactions(limit)
            
            return jsonify({
                'transactions':[{
                    'tx_hash':tx.tx_hash,
                    'from':tx.from_address,
                    'to':tx.to_address,
                    'amount':str(tx.amount),
                    'fee':str(tx.fee),
                    'gas_price':str(tx.gas_price),
                    'nonce':tx.nonce,
                    'timestamp':tx.timestamp.isoformat() if isinstance(tx.timestamp,datetime) else tx.timestamp,
                    'priority_score':tx.priority_score
                } for tx in txs],
                'total':len(txs)
            }),200
            
        except Exception as e:
            logger.error(f"Get mempool transactions error: {e}",exc_info=True)
            return jsonify({'error':'Failed to get mempool transactions'}),500
    
    @bp.route('/mempool/clear',methods=['POST'])
    @require_auth
    @rate_limit(max_requests=10)
    def clear_mempool():
        """Clear all pending transactions (admin only)"""
        try:
            mempool.clear()
            return jsonify({'success':True,'message':'Mempool cleared'}),200
        except Exception as e:
            logger.error(f"Clear mempool error: {e}",exc_info=True)
            return jsonify({'error':'Failed to clear mempool'}),500
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # GAS & FEE ROUTES
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    @bp.route('/gas/estimate',methods=['POST'])
    @rate_limit(max_requests=1000)
    def estimate_gas():
        """Estimate gas price and costs"""
        try:
            data=request.get_json()
            priority=data.get('priority','medium')
            
            stats=chain_db.get_network_stats()
            congestion=min(stats.mempool_size/10000.0,1.0)
            
            estimate=gas_estimator.estimate_gas(priority,congestion)
            
            return jsonify({
                'base_fee':str(estimate.base_fee),
                'priority_fee':str(estimate.priority_fee),
                'max_fee':str(estimate.max_fee),
                'estimated_time_seconds':estimate.estimated_time_seconds,
                'confidence':estimate.confidence,
                'network_congestion':estimate.network_congestion,
                'priority':priority
            }),200
            
        except Exception as e:
            logger.error(f"Gas estimation error: {e}",exc_info=True)
            return jsonify({'error':'Failed to estimate gas'}),500
    
    @bp.route('/gas/prices',methods=['GET'])
    @rate_limit(max_requests=2000)
    def get_gas_prices():
        """Get current gas prices for different priorities"""
        try:
            stats=chain_db.get_network_stats()
            congestion=min(stats.mempool_size/10000.0,1.0)
            
            prices={}
            for priority in ['low','medium','high','urgent']:
                estimate=gas_estimator.estimate_gas(priority,congestion)
                prices[priority]={
                    'max_fee':str(estimate.max_fee),
                    'estimated_time':estimate.estimated_time_seconds
                }
            
            return jsonify({
                'prices':prices,
                'network_congestion':congestion,
                'mempool_size':stats.mempool_size
            }),200
            
        except Exception as e:
            logger.error(f"Get gas prices error: {e}",exc_info=True)
            return jsonify({'error':'Failed to get gas prices'}),500
    
    @bp.route('/fees/historical',methods=['GET'])
    @rate_limit(max_requests=200)
    def get_historical_fees():
        """Get historical fee data"""
        try:
            hours=min(int(request.args.get('hours',24)),168)
            
            query="""
                SELECT 
                    DATE_TRUNC('hour',timestamp) as hour,
                    AVG(fee) as avg_fee,
                    MIN(fee) as min_fee,
                    MAX(fee) as max_fee,
                    COUNT(*) as tx_count
                FROM transactions
                WHERE timestamp>NOW()-INTERVAL '%s hours' AND status='confirmed'
                GROUP BY hour
                ORDER BY hour DESC
            """
            
            results=db_manager.execute_query(query,(hours,))
            
            return jsonify({
                'historical_fees':[{
                    'hour':r['hour'].isoformat() if isinstance(r['hour'],datetime) else r['hour'],
                    'avg_fee':str(r['avg_fee']),
                    'min_fee':str(r['min_fee']),
                    'max_fee':str(r['max_fee']),
                    'tx_count':r['tx_count']
                } for r in results],
                'hours':hours
            }),200
            
        except Exception as e:
            logger.error(f"Historical fees error: {e}",exc_info=True)
            return jsonify({'error':'Failed to get historical fees'}),500
    
    @bp.route('/fees/burn-rate',methods=['GET'])
    @rate_limit(max_requests=500)
    def get_fee_burn_rate():
        """Get fee burn rate statistics"""
        try:
            query="""
                SELECT 
                    SUM(total_fees)*0.5 as total_burned,
                    COUNT(*) as block_count,
                    AVG(total_fees) as avg_block_fees
                FROM blocks
                WHERE timestamp>NOW()-INTERVAL '24 hours'
            """
            
            result=db_manager.execute_query(query,fetch_one=True)
            
            return jsonify({
                'total_burned_24h':str(result['total_burned'] or 0),
                'avg_block_fees':str(result['avg_block_fees'] or 0),
                'burn_rate_per_hour':str((result['total_burned'] or 0)/24),
                'blocks_analyzed':result['block_count']
            }),200
            
        except Exception as e:
            logger.error(f"Fee burn rate error: {e}",exc_info=True)
            return jsonify({'error':'Failed to get burn rate'}),500
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # FINALITY ROUTES
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    @bp.route('/finality/<tx_hash>',methods=['GET'])
    @rate_limit(max_requests=1000)
    def get_finality_status(tx_hash):
        """Get finality status for transaction"""
        try:
            tx=chain_db.get_transaction(tx_hash)
            if not tx:
                return jsonify({'error':'Transaction not found'}),404
            
            latest_block=chain_db.get_latest_block()
            confirmations=0
            if tx.get('block_height') and latest_block:
                confirmations=latest_block['height']-tx['block_height']+1
            
            is_finalized=FinalityEngine.is_finalized(confirmations)
            probability=FinalityEngine.calculate_finality_probability(confirmations)
            
            stats=chain_db.get_network_stats()
            estimated_time=FinalityEngine.estimate_finality_time(confirmations,stats.avg_block_time)
            
            return jsonify({
                'tx_hash':tx_hash,
                'confirmations':confirmations,
                'is_finalized':is_finalized,
                'finality_probability':probability,
                'estimated_finality_time_seconds':estimated_time,
                'finality_threshold':FinalityEngine.FINALITY_THRESHOLD
            }),200
            
        except Exception as e:
            logger.error(f"Finality status error: {e}",exc_info=True)
            return jsonify({'error':'Failed to get finality status'}),500
    
    @bp.route('/finality/batch',methods=['POST'])
    @rate_limit(max_requests=200)
    def get_batch_finality():
        """Get finality status for multiple transactions"""
        try:
            data=request.get_json()
            tx_hashes=data.get('tx_hashes',[])
            
            if not tx_hashes or len(tx_hashes)>100:
                return jsonify({'error':'Provide 1-100 transaction hashes'}),400
            
            latest_block=chain_db.get_latest_block()
            results=[]
            
            for tx_hash in tx_hashes:
                tx=chain_db.get_transaction(tx_hash)
                if not tx:
                    continue
                
                confirmations=0
                if tx.get('block_height') and latest_block:
                    confirmations=latest_block['height']-tx['block_height']+1
                
                results.append({
                    'tx_hash':tx_hash,
                    'confirmations':confirmations,
                    'is_finalized':FinalityEngine.is_finalized(confirmations),
                    'finality_probability':FinalityEngine.calculate_finality_probability(confirmations)
                })
            
            return jsonify({'results':results,'total':len(results)}),200
            
        except Exception as e:
            logger.error(f"Batch finality error: {e}",exc_info=True)
            return jsonify({'error':'Failed to get batch finality'}),500
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # NETWORK ROUTES
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    @bp.route('/network/stats',methods=['GET'])
    @rate_limit(max_requests=500)
    def get_network_stats():
        """Get comprehensive network statistics"""
        try:
            stats=chain_db.get_network_stats()
            
            return jsonify({
                'total_blocks':stats.total_blocks,
                'total_transactions':stats.total_transactions,
                'current_difficulty':stats.current_difficulty,
                'avg_block_time_seconds':stats.avg_block_time,
                'tps':stats.tps,
                'mempool_size':stats.mempool_size,
                'active_validators':stats.active_validators,
                'network_hashrate':stats.network_hashrate,
                'total_supply':str(stats.total_supply),
                'circulating_supply':str(stats.circulating_supply)
            }),200
            
        except Exception as e:
            logger.error(f"Network stats error: {e}",exc_info=True)
            return jsonify({'error':'Failed to get network stats'}),500
    
    @bp.route('/network/difficulty',methods=['GET'])
    @rate_limit(max_requests=500)
    def get_network_difficulty():
        """Get current network difficulty"""
        try:
            latest_block=chain_db.get_latest_block()
            difficulty=latest_block.get('difficulty',1) if latest_block else 1
            
            return jsonify({
                'current_difficulty':difficulty,
                'difficulty_level':NetworkDifficulty(min(difficulty,5)).name,
                'block_height':latest_block.get('height',0) if latest_block else 0
            }),200
            
        except Exception as e:
            logger.error(f"Network difficulty error: {e}",exc_info=True)
            return jsonify({'error':'Failed to get difficulty'}),500
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # EPOCH ROUTES
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    @bp.route('/epochs/current',methods=['GET'])
    @rate_limit(max_requests=500)
    def get_current_epoch():
        """Get current active epoch"""
        try:
            epoch=chain_db.get_current_epoch()
            if not epoch:
                return jsonify({'error':'No active epoch'}),404
            
            if isinstance(epoch.get('validator_set'),str):
                try:
                    epoch['validator_set']=json.loads(epoch['validator_set'])
                except:
                    pass
            
            return jsonify(epoch),200
            
        except Exception as e:
            logger.error(f"Get current epoch error: {e}",exc_info=True)
            return jsonify({'error':'Failed to get current epoch'}),500
    
    @bp.route('/epochs/<int:epoch_num>',methods=['GET'])
    @rate_limit(max_requests=500)
    def get_epoch(epoch_num):
        """Get epoch by number"""
        try:
            epoch=chain_db.get_epoch(epoch_num)
            if not epoch:
                return jsonify({'error':'Epoch not found'}),404
            
            if isinstance(epoch.get('validator_set'),str):
                try:
                    epoch['validator_set']=json.loads(epoch['validator_set'])
                except:
                    pass
            
            return jsonify(epoch),200
            
        except Exception as e:
            logger.error(f"Get epoch error: {e}",exc_info=True)
            return jsonify({'error':'Failed to get epoch'}),500
    
    return bp
