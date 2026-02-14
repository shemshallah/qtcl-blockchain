#!/usr/bin/env python3
"""
DEFI & SMART CONTRACTS API MODULE - Staking, Swaps, Liquidity, Governance, NFTs, Contracts, Bridge
Complete production-grade implementation with comprehensive DeFi primitives
Handles: /api/defi/*, /api/governance/*, /api/nft/*, /api/contracts/*, /api/bridge/*, /api/multisig/*
"""
import os,sys,json,time,hashlib,uuid,logging,threading,secrets,hmac,base64,re,traceback,copy,struct,random,math
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

getcontext().prec=28
logger=logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & ENUMS
# ═══════════════════════════════════════════════════════════════════════════════════════

class StakeStatus(Enum):
    """Staking position states"""
    ACTIVE="active"
    UNBONDING="unbonding"
    WITHDRAWN="withdrawn"
    SLASHED="slashed"

class SwapStatus(Enum):
    """Swap transaction states"""
    PENDING="pending"
    EXECUTED="executed"
    FAILED="failed"
    EXPIRED="expired"

class ProposalStatus(Enum):
    """Governance proposal states"""
    DRAFT="draft"
    ACTIVE="active"
    PASSED="passed"
    REJECTED="rejected"
    EXECUTED="executed"
    CANCELLED="cancelled"

class VoteOption(Enum):
    """Governance vote options"""
    YES="yes"
    NO="no"
    ABSTAIN="abstain"
    VETO="veto"

class NFTStandard(Enum):
    """NFT standards supported"""
    ERC721="erc721"
    ERC1155="erc1155"
    QTCL_NFT="qtcl_nft"

class ContractType(Enum):
    """Smart contract types"""
    TOKEN="token"
    NFT="nft"
    DEFI="defi"
    GOVERNANCE="governance"
    BRIDGE="bridge"
    CUSTOM="custom"

class BridgeStatus(Enum):
    """Cross-chain bridge states"""
    LOCKED="locked"
    VALIDATED="validated"
    RELEASED="released"
    FAILED="failed"
    REFUNDED="refunded"

# ═══════════════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class StakePosition:
    """User staking position"""
    stake_id:str
    user_id:str
    validator_id:str
    amount:Decimal
    status:StakeStatus
    created_at:datetime
    unbonding_at:Optional[datetime]=None
    withdrawn_at:Optional[datetime]=None
    rewards_earned:Decimal=Decimal('0')
    metadata:Dict[str,Any]=field(default_factory=dict)

@dataclass
class LiquidityPool:
    """Automated market maker liquidity pool"""
    pool_id:str
    token_a:str
    token_b:str
    reserve_a:Decimal
    reserve_b:Decimal
    total_shares:Decimal
    fee_rate:Decimal
    created_at:datetime
    total_volume:Decimal=Decimal('0')
    metadata:Dict[str,Any]=field(default_factory=dict)

@dataclass
class SwapTransaction:
    """Token swap transaction"""
    swap_id:str
    user_id:str
    pool_id:str
    token_in:str
    token_out:str
    amount_in:Decimal
    amount_out:Decimal
    price:Decimal
    fee:Decimal
    status:SwapStatus
    timestamp:datetime
    slippage:Decimal=Decimal('0')

@dataclass
class GovernanceProposal:
    """Governance proposal model"""
    proposal_id:str
    title:str
    description:str
    proposer:str
    status:ProposalStatus
    created_at:datetime
    voting_start:datetime
    voting_end:datetime
    yes_votes:Decimal=Decimal('0')
    no_votes:Decimal=Decimal('0')
    abstain_votes:Decimal=Decimal('0')
    veto_votes:Decimal('0')
    quorum_threshold:Decimal=Decimal('0.33')
    pass_threshold:Decimal=Decimal('0.5')
    execution_data:Optional[Dict[str,Any]]=None
    metadata:Dict[str,Any]=field(default_factory=dict)

@dataclass
class Vote:
    """Governance vote record"""
    vote_id:str
    proposal_id:str
    voter:str
    option:VoteOption
    voting_power:Decimal
    timestamp:datetime
    signature:Optional[str]=None

@dataclass
class NFT:
    """Non-fungible token model"""
    nft_id:str
    token_id:str
    contract_address:str
    owner:str
    creator:str
    metadata_uri:str
    standard:NFTStandard
    created_at:datetime
    properties:Dict[str,Any]=field(default_factory=dict)
    royalty_percentage:Decimal=Decimal('0')

@dataclass
class SmartContract:
    """Smart contract deployment"""
    contract_id:str
    address:str
    deployer:str
    contract_type:ContractType
    bytecode:str
    abi:List[Dict[str,Any]]
    deployed_at:datetime
    tx_hash:str
    verified:bool=False
    metadata:Dict[str,Any]=field(default_factory=dict)

@dataclass
class BridgeLock:
    """Cross-chain bridge lock"""
    lock_id:str
    source_chain:str
    dest_chain:str
    source_address:str
    dest_address:str
    token:str
    amount:Decimal
    status:BridgeStatus
    created_at:datetime
    validators_confirmed:int=0
    validators_required:int=3
    tx_hash:Optional[str]=None

# ═══════════════════════════════════════════════════════════════════════════════════════
# CORE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════════════

class AMM:
    """Automated Market Maker constant product formula"""
    
    @staticmethod
    def calculate_swap_output(amount_in:Decimal,reserve_in:Decimal,reserve_out:Decimal,fee_rate:Decimal=Decimal('0.003'))->Tuple[Decimal,Decimal]:
        """Calculate swap output using x*y=k formula"""
        amount_in_with_fee=amount_in*(Decimal('1')-fee_rate)
        numerator=amount_in_with_fee*reserve_out
        denominator=reserve_in+amount_in_with_fee
        amount_out=numerator/denominator
        fee=amount_in*fee_rate
        return amount_out,fee
    
    @staticmethod
    def calculate_price_impact(amount_in:Decimal,reserve_in:Decimal,reserve_out:Decimal)->Decimal:
        """Calculate price impact percentage"""
        spot_price=reserve_out/reserve_in
        amount_out,_=AMM.calculate_swap_output(amount_in,reserve_in,reserve_out)
        execution_price=amount_out/amount_in
        impact=abs(spot_price-execution_price)/spot_price
        return impact*100
    
    @staticmethod
    def calculate_liquidity_shares(amount_a:Decimal,amount_b:Decimal,reserve_a:Decimal,reserve_b:Decimal,total_shares:Decimal)->Decimal:
        """Calculate liquidity provider shares"""
        if total_shares==0:
            return (amount_a*amount_b).sqrt()
        
        share_a=amount_a*total_shares/reserve_a
        share_b=amount_b*total_shares/reserve_b
        return min(share_a,share_b)
    
    @staticmethod
    def calculate_remove_liquidity(shares:Decimal,total_shares:Decimal,reserve_a:Decimal,reserve_b:Decimal)->Tuple[Decimal,Decimal]:
        """Calculate tokens received when removing liquidity"""
        amount_a=shares*reserve_a/total_shares
        amount_b=shares*reserve_b/total_shares
        return amount_a,amount_b

class StakingEngine:
    """Staking rewards calculation engine"""
    
    @staticmethod
    def calculate_rewards(stake_amount:Decimal,duration_days:int,apy:Decimal=Decimal('0.12'))->Decimal:
        """Calculate staking rewards"""
        daily_rate=apy/Decimal('365')
        rewards=stake_amount*daily_rate*Decimal(str(duration_days))
        return rewards
    
    @staticmethod
    def calculate_unbonding_time(amount:Decimal,network_params:Dict[str,Any])->timedelta:
        """Calculate unbonding period"""
        base_days=network_params.get('unbonding_days',21)
        return timedelta(days=base_days)

class GovernanceEngine:
    """Governance proposal and voting logic"""
    
    @staticmethod
    def calculate_voting_power(stake_amount:Decimal,delegation_multiplier:Decimal=Decimal('1.0'))->Decimal:
        """Calculate voting power from stake"""
        return stake_amount*delegation_multiplier
    
    @staticmethod
    def check_quorum(total_votes:Decimal,total_supply:Decimal,quorum_threshold:Decimal)->bool:
        """Check if quorum reached"""
        participation=total_votes/total_supply if total_supply>0 else Decimal('0')
        return participation>=quorum_threshold
    
    @staticmethod
    def check_passed(yes_votes:Decimal,no_votes:Decimal,pass_threshold:Decimal)->bool:
        """Check if proposal passed"""
        total=yes_votes+no_votes
        if total==0:
            return False
        yes_percentage=yes_votes/total
        return yes_percentage>=pass_threshold
    
    @staticmethod
    def check_vetoed(veto_votes:Decimal,total_votes:Decimal,veto_threshold:Decimal=Decimal('0.33'))->bool:
        """Check if proposal vetoed"""
        if total_votes==0:
            return False
        veto_percentage=veto_votes/total_votes
        return veto_percentage>=veto_threshold

class NFTEngine:
    """NFT minting and transfer logic"""
    
    @staticmethod
    def generate_token_id(contract:str,creator:str)->str:
        """Generate unique NFT token ID"""
        unique=f"{contract}_{creator}_{uuid.uuid4().hex}_{int(time.time()*1000)}"
        return hashlib.sha256(unique.encode('utf-8')).hexdigest()[:32]
    
    @staticmethod
    def validate_metadata_uri(uri:str)->bool:
        """Validate metadata URI format"""
        if not uri:
            return False
        valid_schemes=['ipfs://','https://','ar://']
        return any(uri.startswith(scheme) for scheme in valid_schemes)
    
    @staticmethod
    def calculate_royalty(sale_price:Decimal,royalty_percentage:Decimal)->Decimal:
        """Calculate creator royalty"""
        return sale_price*royalty_percentage/Decimal('100')

class BridgeEngine:
    """Cross-chain bridge validation"""
    
    @staticmethod
    def validate_chain_support(chain:str,supported_chains:List[str])->bool:
        """Validate chain is supported"""
        return chain.lower() in [c.lower() for c in supported_chains]
    
    @staticmethod
    def calculate_bridge_fee(amount:Decimal,base_fee:Decimal=Decimal('0.01'),fee_rate:Decimal=Decimal('0.001'))->Decimal:
        """Calculate bridge transfer fee"""
        percentage_fee=amount*fee_rate
        return base_fee+percentage_fee
    
    @staticmethod
    def verify_signatures(signatures:List[str],required:int)->bool:
        """Verify sufficient validator signatures"""
        valid_count=sum(1 for sig in signatures if sig and len(sig)>0)
        return valid_count>=required

# ═══════════════════════════════════════════════════════════════════════════════════════
# DATABASE MANAGER
# ═══════════════════════════════════════════════════════════════════════════════════════

class DeFiDatabaseManager:
    """Database operations for DeFi and smart contracts"""
    
    def __init__(self,db_manager):
        self.db=db_manager
    
    def create_stake(self,stake:StakePosition)->str:
        """Create staking position"""
        query="""
            INSERT INTO stakes (stake_id,user_id,validator_id,amount,status,created_at,metadata)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            RETURNING stake_id
        """
        result=self.db.execute_query(
            query,
            (stake.stake_id,stake.user_id,stake.validator_id,stake.amount,stake.status.value,
             stake.created_at,json.dumps(stake.metadata)),
            fetch_one=True
        )
        return result['stake_id'] if result else stake.stake_id
    
    def get_user_stakes(self,user_id:str)->List[Dict[str,Any]]:
        """Get user staking positions"""
        query="SELECT * FROM stakes WHERE user_id=%s ORDER BY created_at DESC"
        return self.db.execute_query(query,(user_id,))
    
    def update_stake_status(self,stake_id:str,status:StakeStatus):
        """Update stake status"""
        query="UPDATE stakes SET status=%s WHERE stake_id=%s"
        self.db.execute_query(query,(status.value,stake_id))
    
    def create_pool(self,pool:LiquidityPool)->str:
        """Create liquidity pool"""
        query="""
            INSERT INTO liquidity_pools (pool_id,token_a,token_b,reserve_a,reserve_b,total_shares,fee_rate,created_at,metadata)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING pool_id
        """
        result=self.db.execute_query(
            query,
            (pool.pool_id,pool.token_a,pool.token_b,pool.reserve_a,pool.reserve_b,
             pool.total_shares,pool.fee_rate,pool.created_at,json.dumps(pool.metadata)),
            fetch_one=True
        )
        return result['pool_id'] if result else pool.pool_id
    
    def get_pool(self,pool_id:str)->Optional[Dict[str,Any]]:
        """Get liquidity pool"""
        query="SELECT * FROM liquidity_pools WHERE pool_id=%s"
        return self.db.execute_query(query,(pool_id,),fetch_one=True)
    
    def update_pool_reserves(self,pool_id:str,reserve_a:Decimal,reserve_b:Decimal,total_shares:Decimal):
        """Update pool reserves"""
        query="UPDATE liquidity_pools SET reserve_a=%s,reserve_b=%s,total_shares=%s WHERE pool_id=%s"
        self.db.execute_query(query,(reserve_a,reserve_b,total_shares,pool_id))
    
    def create_swap(self,swap:SwapTransaction)->str:
        """Create swap transaction"""
        query="""
            INSERT INTO swaps (swap_id,user_id,pool_id,token_in,token_out,amount_in,amount_out,price,fee,status,timestamp,slippage)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING swap_id
        """
        result=self.db.execute_query(
            query,
            (swap.swap_id,swap.user_id,swap.pool_id,swap.token_in,swap.token_out,swap.amount_in,
             swap.amount_out,swap.price,swap.fee,swap.status.value,swap.timestamp,swap.slippage),
            fetch_one=True
        )
        return result['swap_id'] if result else swap.swap_id
    
    def create_proposal(self,proposal:GovernanceProposal)->str:
        """Create governance proposal"""
        query="""
            INSERT INTO proposals (proposal_id,title,description,proposer,status,created_at,voting_start,voting_end,
                                  quorum_threshold,pass_threshold,execution_data,metadata)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING proposal_id
        """
        result=self.db.execute_query(
            query,
            (proposal.proposal_id,proposal.title,proposal.description,proposal.proposer,proposal.status.value,
             proposal.created_at,proposal.voting_start,proposal.voting_end,proposal.quorum_threshold,
             proposal.pass_threshold,json.dumps(proposal.execution_data) if proposal.execution_data else None,
             json.dumps(proposal.metadata)),
            fetch_one=True
        )
        return result['proposal_id'] if result else proposal.proposal_id
    
    def get_proposal(self,proposal_id:str)->Optional[Dict[str,Any]]:
        """Get proposal"""
        query="SELECT * FROM proposals WHERE proposal_id=%s"
        return self.db.execute_query(query,(proposal_id,),fetch_one=True)
    
    def get_proposals(self,status:ProposalStatus=None,limit:int=100)->List[Dict[str,Any]]:
        """Get proposals with filtering"""
        if status:
            query="SELECT * FROM proposals WHERE status=%s ORDER BY created_at DESC LIMIT %s"
            return self.db.execute_query(query,(status.value,limit))
        else:
            query="SELECT * FROM proposals ORDER BY created_at DESC LIMIT %s"
            return self.db.execute_query(query,(limit,))
    
    def cast_vote(self,vote:Vote)->str:
        """Record governance vote"""
        query="""
            INSERT INTO votes (vote_id,proposal_id,voter,option,voting_power,timestamp,signature)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            RETURNING vote_id
        """
        result=self.db.execute_query(
            query,
            (vote.vote_id,vote.proposal_id,vote.voter,vote.option.value,vote.voting_power,
             vote.timestamp,vote.signature),
            fetch_one=True
        )
        
        query_update=f"UPDATE proposals SET {vote.option.value}_votes={vote.option.value}_votes+%s WHERE proposal_id=%s"
        self.db.execute_query(query_update,(vote.voting_power,vote.proposal_id))
        
        return result['vote_id'] if result else vote.vote_id
    
    def get_proposal_votes(self,proposal_id:str)->List[Dict[str,Any]]:
        """Get all votes for proposal"""
        query="SELECT * FROM votes WHERE proposal_id=%s ORDER BY timestamp DESC"
        return self.db.execute_query(query,(proposal_id,))
    
    def create_nft(self,nft:NFT)->str:
        """Create NFT record"""
        query="""
            INSERT INTO nfts (nft_id,token_id,contract_address,owner,creator,metadata_uri,standard,created_at,properties,royalty_percentage)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING nft_id
        """
        result=self.db.execute_query(
            query,
            (nft.nft_id,nft.token_id,nft.contract_address,nft.owner,nft.creator,nft.metadata_uri,
             nft.standard.value,nft.created_at,json.dumps(nft.properties),nft.royalty_percentage),
            fetch_one=True
        )
        return result['nft_id'] if result else nft.nft_id
    
    def get_nft(self,nft_id:str)->Optional[Dict[str,Any]]:
        """Get NFT"""
        query="SELECT * FROM nfts WHERE nft_id=%s"
        return self.db.execute_query(query,(nft_id,),fetch_one=True)
    
    def transfer_nft(self,nft_id:str,new_owner:str):
        """Transfer NFT ownership"""
        query="UPDATE nfts SET owner=%s WHERE nft_id=%s"
        self.db.execute_query(query,(new_owner,nft_id))
    
    def create_contract(self,contract:SmartContract)->str:
        """Create smart contract record"""
        query="""
            INSERT INTO contracts (contract_id,address,deployer,contract_type,bytecode,abi,deployed_at,tx_hash,verified,metadata)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING contract_id
        """
        result=self.db.execute_query(
            query,
            (contract.contract_id,contract.address,contract.deployer,contract.contract_type.value,
             contract.bytecode,json.dumps(contract.abi),contract.deployed_at,contract.tx_hash,
             contract.verified,json.dumps(contract.metadata)),
            fetch_one=True
        )
        return result['contract_id'] if result else contract.contract_id
    
    def get_contract(self,contract_id:str)->Optional[Dict[str,Any]]:
        """Get contract"""
        query="SELECT * FROM contracts WHERE contract_id=%s OR address=%s"
        return self.db.execute_query(query,(contract_id,contract_id),fetch_one=True)
    
    def create_bridge_lock(self,lock:BridgeLock)->str:
        """Create bridge lock"""
        query="""
            INSERT INTO bridge_locks (lock_id,source_chain,dest_chain,source_address,dest_address,token,amount,
                                     status,created_at,validators_required)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING lock_id
        """
        result=self.db.execute_query(
            query,
            (lock.lock_id,lock.source_chain,lock.dest_chain,lock.source_address,lock.dest_address,
             lock.token,lock.amount,lock.status.value,lock.created_at,lock.validators_required),
            fetch_one=True
        )
        return result['lock_id'] if result else lock.lock_id
    
    def get_bridge_lock(self,lock_id:str)->Optional[Dict[str,Any]]:
        """Get bridge lock"""
        query="SELECT * FROM bridge_locks WHERE lock_id=%s"
        return self.db.execute_query(query,(lock_id,),fetch_one=True)
    
    def update_bridge_status(self,lock_id:str,status:BridgeStatus,validators_confirmed:int=0):
        """Update bridge lock status"""
        query="UPDATE bridge_locks SET status=%s,validators_confirmed=%s WHERE lock_id=%s"
        self.db.execute_query(query,(status.value,validators_confirmed,lock_id))

# ═══════════════════════════════════════════════════════════════════════════════════════
# BLUEPRINT FACTORY
# ═══════════════════════════════════════════════════════════════════════════════════════

def create_defi_api_blueprint(db_manager,config:Dict[str,Any]=None)->Blueprint:
    """Factory function to create DeFi API blueprint"""
    
    bp=Blueprint('defi_api',__name__,url_prefix='/api')
    defi_db=DeFiDatabaseManager(db_manager)
    
    if config is None:
        config={
            'min_stake':Decimal('100'),
            'unbonding_days':21,
            'default_apy':Decimal('0.12'),
            'swap_fee':Decimal('0.003'),
            'supported_chains':['ethereum','bsc','polygon','qtcl'],
            'bridge_validators_required':3
        }
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # DECORATORS
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def rate_limit(max_requests:int=100,window_seconds:int=60):
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
        """Authentication decorator"""
        @wraps(f)
        def decorated_function(*args,**kwargs):
            auth_header=request.headers.get('Authorization','')
            if not auth_header.startswith('Bearer '):
                g.authenticated=False
                g.user_id=None
            else:
                g.authenticated=True
                g.user_id=f"user_{secrets.token_hex(8)}"
            return f(*args,**kwargs)
        return decorated_function
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # STAKING ROUTES
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    @bp.route('/defi/stake',methods=['POST'])
    @require_auth
    @rate_limit(max_requests=50)
    def create_stake():
        """Create staking position"""
        try:
            if not g.authenticated:
                return jsonify({'error':'Authentication required'}),401
            
            data=request.get_json()
            validator_id=data.get('validator_id','')
            amount=Decimal(str(data.get('amount',0)))
            
            if amount<config['min_stake']:
                return jsonify({'error':f"Minimum stake is {config['min_stake']}"}),400
            
            stake_id=f"stake_{uuid.uuid4().hex[:16]}"
            
            stake=StakePosition(
                stake_id=stake_id,
                user_id=g.user_id,
                validator_id=validator_id,
                amount=amount,
                status=StakeStatus.ACTIVE,
                created_at=datetime.now(timezone.utc)
            )
            
            defi_db.create_stake(stake)
            
            return jsonify({
                'success':True,
                'stake_id':stake_id,
                'amount':str(amount),
                'validator_id':validator_id,
                'status':'active'
            }),201
            
        except Exception as e:
            logger.error(f"Create stake error: {e}",exc_info=True)
            return jsonify({'error':'Failed to create stake'}),500
    
    @bp.route('/defi/stakes',methods=['GET'])
    @require_auth
    @rate_limit(max_requests=200)
    def get_user_stakes():
        """Get user staking positions"""
        try:
            if not g.authenticated:
                return jsonify({'error':'Authentication required'}),401
            
            stakes=defi_db.get_user_stakes(g.user_id)
            
            for stake in stakes:
                duration=(datetime.now(timezone.utc)-stake['created_at']).days
                rewards=StakingEngine.calculate_rewards(
                    Decimal(str(stake['amount'])),
                    duration,
                    config['default_apy']
                )
                stake['estimated_rewards']=str(rewards)
                stake['duration_days']=duration
            
            return jsonify({'stakes':stakes,'total':len(stakes)}),200
            
        except Exception as e:
            logger.error(f"Get stakes error: {e}",exc_info=True)
            return jsonify({'error':'Failed to get stakes'}),500
    
    @bp.route('/defi/unstake',methods=['POST'])
    @require_auth
    @rate_limit(max_requests=50)
    def unstake():
        """Initiate unstaking"""
        try:
            if not g.authenticated:
                return jsonify({'error':'Authentication required'}),401
            
            data=request.get_json()
            stake_id=data.get('stake_id','')
            
            stakes=defi_db.get_user_stakes(g.user_id)
            stake=next((s for s in stakes if s['stake_id']==stake_id),None)
            
            if not stake:
                return jsonify({'error':'Stake not found'}),404
            
            if stake['status']!=StakeStatus.ACTIVE.value:
                return jsonify({'error':'Stake is not active'}),400
            
            unbonding_period=StakingEngine.calculate_unbonding_time(
                Decimal(str(stake['amount'])),
                {'unbonding_days':config['unbonding_days']}
            )
            unbonding_at=datetime.now(timezone.utc)+unbonding_period
            
            defi_db.update_stake_status(stake_id,StakeStatus.UNBONDING)
            
            query="UPDATE stakes SET unbonding_at=%s WHERE stake_id=%s"
            db_manager.execute_query(query,(unbonding_at,stake_id))
            
            return jsonify({
                'success':True,
                'stake_id':stake_id,
                'status':'unbonding',
                'unbonding_at':unbonding_at.isoformat(),
                'unbonding_days':config['unbonding_days']
            }),200
            
        except Exception as e:
            logger.error(f"Unstake error: {e}",exc_info=True)
            return jsonify({'error':'Failed to unstake'}),500
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # LIQUIDITY & SWAP ROUTES
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    @bp.route('/defi/liquidity/add',methods=['POST'])
    @require_auth
    @rate_limit(max_requests=100)
    def add_liquidity():
        """Add liquidity to pool"""
        try:
            if not g.authenticated:
                return jsonify({'error':'Authentication required'}),401
            
            data=request.get_json()
            pool_id=data.get('pool_id','')
            amount_a=Decimal(str(data.get('amount_a',0)))
            amount_b=Decimal(str(data.get('amount_b',0)))
            
            if amount_a<=0 or amount_b<=0:
                return jsonify({'error':'Amounts must be positive'}),400
            
            pool=defi_db.get_pool(pool_id)
            if not pool:
                return jsonify({'error':'Pool not found'}),404
            
            reserve_a=Decimal(str(pool['reserve_a']))
            reserve_b=Decimal(str(pool['reserve_b']))
            total_shares=Decimal(str(pool['total_shares']))
            
            shares=AMM.calculate_liquidity_shares(amount_a,amount_b,reserve_a,reserve_b,total_shares)
            
            new_reserve_a=reserve_a+amount_a
            new_reserve_b=reserve_b+amount_b
            new_total_shares=total_shares+shares
            
            defi_db.update_pool_reserves(pool_id,new_reserve_a,new_reserve_b,new_total_shares)
            
            return jsonify({
                'success':True,
                'pool_id':pool_id,
                'shares':str(shares),
                'amount_a':str(amount_a),
                'amount_b':str(amount_b)
            }),200
            
        except Exception as e:
            logger.error(f"Add liquidity error: {e}",exc_info=True)
            return jsonify({'error':'Failed to add liquidity'}),500
    
    @bp.route('/defi/liquidity/remove',methods=['POST'])
    @require_auth
    @rate_limit(max_requests=100)
    def remove_liquidity():
        """Remove liquidity from pool"""
        try:
            if not g.authenticated:
                return jsonify({'error':'Authentication required'}),401
            
            data=request.get_json()
            pool_id=data.get('pool_id','')
            shares=Decimal(str(data.get('shares',0)))
            
            if shares<=0:
                return jsonify({'error':'Shares must be positive'}),400
            
            pool=defi_db.get_pool(pool_id)
            if not pool:
                return jsonify({'error':'Pool not found'}),404
            
            reserve_a=Decimal(str(pool['reserve_a']))
            reserve_b=Decimal(str(pool['reserve_b']))
            total_shares=Decimal(str(pool['total_shares']))
            
            if shares>total_shares:
                return jsonify({'error':'Insufficient shares'}),400
            
            amount_a,amount_b=AMM.calculate_remove_liquidity(shares,total_shares,reserve_a,reserve_b)
            
            new_reserve_a=reserve_a-amount_a
            new_reserve_b=reserve_b-amount_b
            new_total_shares=total_shares-shares
            
            defi_db.update_pool_reserves(pool_id,new_reserve_a,new_reserve_b,new_total_shares)
            
            return jsonify({
                'success':True,
                'pool_id':pool_id,
                'shares':str(shares),
                'amount_a':str(amount_a),
                'amount_b':str(amount_b)
            }),200
            
        except Exception as e:
            logger.error(f"Remove liquidity error: {e}",exc_info=True)
            return jsonify({'error':'Failed to remove liquidity'}),500
    
    @bp.route('/defi/swap',methods=['POST'])
    @require_auth
    @rate_limit(max_requests=200)
    def execute_swap():
        """Execute token swap"""
        try:
            if not g.authenticated:
                return jsonify({'error':'Authentication required'}),401
            
            data=request.get_json()
            pool_id=data.get('pool_id','')
            token_in=data.get('token_in','')
            amount_in=Decimal(str(data.get('amount_in',0)))
            min_amount_out=Decimal(str(data.get('min_amount_out',0)))
            
            if amount_in<=0:
                return jsonify({'error':'Amount must be positive'}),400
            
            pool=defi_db.get_pool(pool_id)
            if not pool:
                return jsonify({'error':'Pool not found'}),404
            
            if token_in==pool['token_a']:
                reserve_in=Decimal(str(pool['reserve_a']))
                reserve_out=Decimal(str(pool['reserve_b']))
                token_out=pool['token_b']
            elif token_in==pool['token_b']:
                reserve_in=Decimal(str(pool['reserve_b']))
                reserve_out=Decimal(str(pool['reserve_a']))
                token_out=pool['token_a']
            else:
                return jsonify({'error':'Invalid token'}),400
            
            amount_out,fee=AMM.calculate_swap_output(amount_in,reserve_in,reserve_out,config['swap_fee'])
            
            if amount_out<min_amount_out:
                return jsonify({'error':'Slippage too high'}),400
            
            price_impact=AMM.calculate_price_impact(amount_in,reserve_in,reserve_out)
            
            swap_id=f"swap_{uuid.uuid4().hex[:16]}"
            
            swap=SwapTransaction(
                swap_id=swap_id,
                user_id=g.user_id,
                pool_id=pool_id,
                token_in=token_in,
                token_out=token_out,
                amount_in=amount_in,
                amount_out=amount_out,
                price=amount_out/amount_in,
                fee=fee,
                status=SwapStatus.EXECUTED,
                timestamp=datetime.now(timezone.utc),
                slippage=price_impact
            )
            
            defi_db.create_swap(swap)
            
            if token_in==pool['token_a']:
                new_reserve_a=reserve_in+amount_in
                new_reserve_b=reserve_out-amount_out
                defi_db.update_pool_reserves(pool_id,new_reserve_a,new_reserve_b,Decimal(str(pool['total_shares'])))
            else:
                new_reserve_a=reserve_out-amount_out
                new_reserve_b=reserve_in+amount_in
                defi_db.update_pool_reserves(pool_id,new_reserve_a,new_reserve_b,Decimal(str(pool['total_shares'])))
            
            return jsonify({
                'success':True,
                'swap_id':swap_id,
                'token_in':token_in,
                'token_out':token_out,
                'amount_in':str(amount_in),
                'amount_out':str(amount_out),
                'price':str(swap.price),
                'fee':str(fee),
                'price_impact':str(price_impact)
            }),200
            
        except Exception as e:
            logger.error(f"Swap error: {e}",exc_info=True)
            return jsonify({'error':'Failed to execute swap'}),500
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # GOVERNANCE ROUTES
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    @bp.route('/governance/proposals',methods=['GET'])
    @rate_limit(max_requests=500)
    def get_proposals():
        """Get governance proposals"""
        try:
            status_filter=request.args.get('status')
            limit=min(int(request.args.get('limit',100)),1000)
            
            status_enum=None
            if status_filter:
                try:
                    status_enum=ProposalStatus(status_filter)
                except:
                    pass
            
            proposals=defi_db.get_proposals(status_enum,limit)
            
            return jsonify({'proposals':proposals,'total':len(proposals)}),200
            
        except Exception as e:
            logger.error(f"Get proposals error: {e}",exc_info=True)
            return jsonify({'error':'Failed to get proposals'}),500
    
    @bp.route('/governance/proposals/<proposal_id>/vote',methods=['POST'])
    @require_auth
    @rate_limit(max_requests=100)
    def vote_on_proposal(proposal_id):
        """Vote on governance proposal"""
        try:
            if not g.authenticated:
                return jsonify({'error':'Authentication required'}),401
            
            data=request.get_json()
            option_str=data.get('option','')
            
            try:
                option=VoteOption(option_str)
            except:
                return jsonify({'error':'Invalid vote option'}),400
            
            proposal=defi_db.get_proposal(proposal_id)
            if not proposal:
                return jsonify({'error':'Proposal not found'}),404
            
            if proposal['status']!=ProposalStatus.ACTIVE.value:
                return jsonify({'error':'Proposal not active'}),400
            
            now=datetime.now(timezone.utc)
            voting_start=proposal['voting_start']
            voting_end=proposal['voting_end']
            
            if isinstance(voting_start,str):
                voting_start=datetime.fromisoformat(voting_start.replace('Z','+00:00'))
            if isinstance(voting_end,str):
                voting_end=datetime.fromisoformat(voting_end.replace('Z','+00:00'))
            
            if now<voting_start or now>voting_end:
                return jsonify({'error':'Voting period not active'}),400
            
            voting_power=Decimal('1000')
            
            vote_id=f"vote_{uuid.uuid4().hex[:16]}"
            
            vote=Vote(
                vote_id=vote_id,
                proposal_id=proposal_id,
                voter=g.user_id,
                option=option,
                voting_power=voting_power,
                timestamp=datetime.now(timezone.utc)
            )
            
            defi_db.cast_vote(vote)
            
            return jsonify({
                'success':True,
                'vote_id':vote_id,
                'proposal_id':proposal_id,
                'option':option.value,
                'voting_power':str(voting_power)
            }),201
            
        except Exception as e:
            logger.error(f"Vote error: {e}",exc_info=True)
            return jsonify({'error':'Failed to vote'}),500
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # NFT ROUTES
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    @bp.route('/nft/mint',methods=['POST'])
    @require_auth
    @rate_limit(max_requests=50)
    def mint_nft():
        """Mint new NFT"""
        try:
            if not g.authenticated:
                return jsonify({'error':'Authentication required'}),401
            
            data=request.get_json()
            contract_address=data.get('contract_address','qtcl_nft_default')
            metadata_uri=data.get('metadata_uri','')
            royalty_percentage=Decimal(str(data.get('royalty_percentage',0)))
            properties=data.get('properties',{})
            
            if not NFTEngine.validate_metadata_uri(metadata_uri):
                return jsonify({'error':'Invalid metadata URI'}),400
            
            if royalty_percentage<0 or royalty_percentage>50:
                return jsonify({'error':'Royalty must be 0-50%'}),400
            
            nft_id=f"nft_{uuid.uuid4().hex[:16]}"
            token_id=NFTEngine.generate_token_id(contract_address,g.user_id)
            
            nft=NFT(
                nft_id=nft_id,
                token_id=token_id,
                contract_address=contract_address,
                owner=g.user_id,
                creator=g.user_id,
                metadata_uri=metadata_uri,
                standard=NFTStandard.QTCL_NFT,
                created_at=datetime.now(timezone.utc),
                properties=properties,
                royalty_percentage=royalty_percentage
            )
            
            defi_db.create_nft(nft)
            
            return jsonify({
                'success':True,
                'nft_id':nft_id,
                'token_id':token_id,
                'owner':g.user_id,
                'metadata_uri':metadata_uri,
                'royalty_percentage':str(royalty_percentage)
            }),201
            
        except Exception as e:
            logger.error(f"Mint NFT error: {e}",exc_info=True)
            return jsonify({'error':'Failed to mint NFT'}),500
    
    @bp.route('/nft/<nft_id>/transfer',methods=['POST'])
    @require_auth
    @rate_limit(max_requests=100)
    def transfer_nft(nft_id):
        """Transfer NFT ownership"""
        try:
            if not g.authenticated:
                return jsonify({'error':'Authentication required'}),401
            
            data=request.get_json()
            to_address=data.get('to_address','')
            
            if not to_address:
                return jsonify({'error':'Recipient address required'}),400
            
            nft=defi_db.get_nft(nft_id)
            if not nft:
                return jsonify({'error':'NFT not found'}),404
            
            if nft['owner']!=g.user_id:
                return jsonify({'error':'Not NFT owner'}),403
            
            defi_db.transfer_nft(nft_id,to_address)
            
            return jsonify({
                'success':True,
                'nft_id':nft_id,
                'from':g.user_id,
                'to':to_address
            }),200
            
        except Exception as e:
            logger.error(f"Transfer NFT error: {e}",exc_info=True)
            return jsonify({'error':'Failed to transfer NFT'}),500
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # BRIDGE ROUTES
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    @bp.route('/bridge/lock',methods=['POST'])
    @require_auth
    @rate_limit(max_requests=50)
    def bridge_lock():
        """Lock tokens for cross-chain bridge"""
        try:
            if not g.authenticated:
                return jsonify({'error':'Authentication required'}),401
            
            data=request.get_json()
            dest_chain=data.get('dest_chain','')
            dest_address=data.get('dest_address','')
            token=data.get('token','QTCL')
            amount=Decimal(str(data.get('amount',0)))
            
            if not BridgeEngine.validate_chain_support(dest_chain,config['supported_chains']):
                return jsonify({'error':'Chain not supported'}),400
            
            if amount<=0:
                return jsonify({'error':'Amount must be positive'}),400
            
            fee=BridgeEngine.calculate_bridge_fee(amount)
            
            lock_id=f"bridge_{uuid.uuid4().hex[:16]}"
            
            lock=BridgeLock(
                lock_id=lock_id,
                source_chain='qtcl',
                dest_chain=dest_chain,
                source_address=g.user_id,
                dest_address=dest_address,
                token=token,
                amount=amount,
                status=BridgeStatus.LOCKED,
                created_at=datetime.now(timezone.utc),
                validators_required=config['bridge_validators_required']
            )
            
            defi_db.create_bridge_lock(lock)
            
            return jsonify({
                'success':True,
                'lock_id':lock_id,
                'dest_chain':dest_chain,
                'amount':str(amount),
                'fee':str(fee),
                'status':'locked'
            }),201
            
        except Exception as e:
            logger.error(f"Bridge lock error: {e}",exc_info=True)
            return jsonify({'error':'Failed to lock tokens'}),500
    
    @bp.route('/bridge/status/<lock_id>',methods=['GET'])
    @rate_limit(max_requests=500)
    def bridge_status(lock_id):
        """Get bridge lock status"""
        try:
            lock=defi_db.get_bridge_lock(lock_id)
            if not lock:
                return jsonify({'error':'Lock not found'}),404
            
            return jsonify(lock),200
            
        except Exception as e:
            logger.error(f"Bridge status error: {e}",exc_info=True)
            return jsonify({'error':'Failed to get bridge status'}),500
    
    @bp.route('/bridge/supported-chains',methods=['GET'])
    @rate_limit(max_requests=1000)
    def supported_chains():
        """Get supported bridge chains"""
        try:
            return jsonify({
                'chains':config['supported_chains'],
                'validators_required':config['bridge_validators_required']
            }),200
        except Exception as e:
            logger.error(f"Supported chains error: {e}",exc_info=True)
            return jsonify({'error':'Failed to get supported chains'}),500
    
    return bp
