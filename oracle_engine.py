
"""
oracle_engine.py - QTCL Quantum Blockchain Oracle Engine
Complete implementation of oracle data sources, superposition collapse, and finality triggering

Author: QTCL Development Team
Version: 1.0
Date: 2025-02-08
Lines: ~6000+

This module transforms quantum superposition states into classical blockchain finality
via oracle data measurement and collapse mechanics.
"""

import os
import sys
import time
import json
import hashlib
import hmac
import secrets
import threading
import queue
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable, Generator
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import math
import asyncio
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed

# Cryptography
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, utils
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature

# Database
from supabase import create_client, Client
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor, execute_values

# HTTP requests for price oracle
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Web3 for event oracle
try:
    from web3 import Web3
    from web3.exceptions import ContractLogicError, TransactionNotFound
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False
    print("Warning: web3.py not installed - Event Oracle will be disabled")

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('oracle_engine.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
ORACLE_EVENT_MAX_QUEUE_SIZE = 10000
ORACLE_EVENT_BATCH_SIZE = 50
ORACLE_EVENT_MAX_AGE_SECONDS = 300  # 5 minutes
TIME_ORACLE_INTERVAL_SECONDS = 10
PRICE_ORACLE_UPDATE_INTERVAL_SECONDS = 30
PRICE_CACHE_TTL_SECONDS = 30
EVENT_ORACLE_POLL_INTERVAL_SECONDS = 5
RANDOM_ORACLE_VRF_KEY_SIZE = 32
ENTROPY_MIN_THRESHOLD = 0.30  # 30% minimum entropy
ENTROPY_OPTIMAL_THRESHOLD = 0.70  # 70% optimal entropy
COLLAPSE_PROOF_ALGORITHM = "SHA256"
ORACLE_REPUTATION_DECAY_RATE = 0.01
ORACLE_REPUTATION_BOOST_RATE = 0.05
MAX_CONCURRENT_COLLAPSES = 100
ORACLE_LOOP_SLEEP_MS = 100
BALANCE_VALIDATION_ENABLED = True
CAUSALITY_CHECK_ENABLED = True
HYPERBOLIC_DISTANCE_THRESHOLD = 5.0

# Oracle types enumeration
class OracleType(Enum):
    TIME = "time"
    PRICE = "price"
    EVENT = "event"
    RANDOM = "random"
    ENTROPY = "entropy"

# Transaction status enumeration
class TransactionStatus(Enum):
    PENDING = "pending"
    QUEUED = "queued"
    SUPERPOSITION = "superposition"
    AWAITING_COLLAPSE = "awaiting_collapse"
    COLLAPSED = "collapsed"
    FINALIZED = "finalized"
    REJECTED = "rejected"
    FAILED = "failed"

# Collapse outcome enumeration
class CollapseOutcome(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    ERROR = "error"
    PENDING = "pending"

@dataclass
class OracleEvent:
    """Data structure for oracle events"""
    oracle_id: str
    oracle_type: OracleType
    tx_id: str
    oracle_data: Dict[str, Any]
    proof: str
    timestamp: int
    priority: int = 5
    dispatched: bool = False
    collapse_triggered: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'oracle_id': self.oracle_id,
            'oracle_type': self.oracle_type.value,
            'tx_id': self.tx_id,
            'oracle_data': self.oracle_data,
            'proof': self.proof,
            'timestamp': self.timestamp,
            'priority': self.priority,
            'dispatched': self.dispatched,
            'collapse_triggered': self.collapse_triggered
        }

@dataclass
class CollapseResult:
    """Result of superposition collapse"""
    tx_id: str
    outcome: CollapseOutcome
    collapsed_bitstring: str
    collapse_proof: str
    oracle_data: Dict
    interpretation: Dict
    causality_valid: bool
    timestamp: int
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'tx_id': self.tx_id,
            'outcome': self.outcome.value,
            'collapsed_bitstring': self.collapsed_bitstring,
            'collapse_proof': self.collapse_proof,
            'oracle_data': self.oracle_data,
            'interpretation': self.interpretation,
            'causality_valid': self.causality_valid,
            'timestamp': self.timestamp,
            'error_message': self.error_message
        }


# ============================================================================
# MODULE 1: ORACLE DATA SOURCES
# ============================================================================

class TimeOracle:
    """
    Time-based oracle that triggers collapse after transaction age threshold
    """
    
    def __init__(self, interval_seconds: int = TIME_ORACLE_INTERVAL_SECONDS):
        self.interval_seconds = interval_seconds
        self.triggered_transactions = set()
        self.event_count = 0
        self.success_count = 0
        self.last_trigger_time = 0
        self.lock = threading.Lock()
        logger.info(f"TimeOracle initialized with {interval_seconds}s interval")
    
    def get_current_timestamp(self) -> int:
        """Get current Unix timestamp"""
        return int(time.time())
    
    def should_trigger(self, tx_created_at: datetime, current_time: datetime) -> bool:
        """Check if transaction age exceeds threshold"""
        age_seconds = (current_time - tx_created_at).total_seconds()
        return age_seconds >= self.interval_seconds
    
    def trigger_time_oracle_event(self, tx_id: str, tx_created_at: datetime) -> Optional[OracleEvent]:
        """
        Trigger time oracle event for transaction
        
        Args:
            tx_id: Transaction ID
            tx_created_at: Transaction creation timestamp
            
        Returns:
            OracleEvent if trigger successful, None otherwise
        """
        with self.lock:
            if tx_id in self.triggered_transactions:
                return None
            
            current_time = datetime.utcnow()
            if not self.should_trigger(tx_created_at, current_time):
                return None
            
            timestamp = self.get_current_timestamp()
            commitment = self.generate_time_commitment(tx_id, timestamp)
            
            oracle_data = {
                'trigger_time': timestamp,
                'tx_created_at': int(tx_created_at.timestamp()),
                'age_seconds': (current_time - tx_created_at).total_seconds(),
                'threshold_seconds': self.interval_seconds
            }
            
            oracle_event = OracleEvent(
                oracle_id=f"time_{tx_id}_{timestamp}",
                oracle_type=OracleType.TIME,
                tx_id=tx_id,
                oracle_data=oracle_data,
                proof=commitment,
                timestamp=timestamp,
                priority=8  # High priority
            )
            
            self.triggered_transactions.add(tx_id)
            self.event_count += 1
            self.last_trigger_time = timestamp
            
            logger.info(f"Time oracle triggered for tx {tx_id}, age: {oracle_data['age_seconds']:.1f}s")
            return oracle_event
    
    def generate_time_commitment(self, tx_id: str, timestamp: int) -> str:
        """Generate cryptographic commitment for time oracle"""
        commitment_data = f"{tx_id}:{timestamp}".encode('utf-8')
        return hashlib.sha256(commitment_data).hexdigest()
    
    def get_statistics(self) -> Dict:
        """Get time oracle statistics"""
        with self.lock:
            return {
                'total_triggers': self.event_count,
                'success_count': self.success_count,
                'triggered_transactions': len(self.triggered_transactions),
                'last_trigger_time': self.last_trigger_time,
                'interval_seconds': self.interval_seconds
            }
    
    def reset_transaction_trigger(self, tx_id: str):
        """Reset trigger state for transaction (for testing)"""
        with self.lock:
            self.triggered_transactions.discard(tx_id)


class PriceOracle:
    """
    Price oracle for cryptocurrency price data from external APIs
    """
    
    def __init__(self, 
                 api_endpoint: str = "https://api.coingecko.com/api/v3",
                 update_interval: int = PRICE_ORACLE_UPDATE_INTERVAL_SECONDS):
        self.api_endpoint = api_endpoint
        self.update_interval = update_interval
        self.price_cache = {}
        self.cache_timestamps = {}
        self.event_count = 0
        self.api_call_count = 0
        self.lock = threading.Lock()
        
        # Setup HTTP session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        logger.info(f"PriceOracle initialized with endpoint: {api_endpoint}")
    
    def fetch_current_price(self, token: str = "ethereum") -> Optional[float]:
        """
        Fetch current cryptocurrency price
        
        Args:
            token: Token symbol (ethereum, bitcoin, etc.)
            
        Returns:
            Price in USD or None if fetch failed
        """
        with self.lock:
            # Check cache first
            if token in self.price_cache:
                cache_age = time.time() - self.cache_timestamps.get(token, 0)
                if cache_age < PRICE_CACHE_TTL_SECONDS:
                    logger.debug(f"Using cached price for {token}: ${self.price_cache[token]}")
                    return self.price_cache[token]
            
            # Fetch from API
            try:
                url = f"{self.api_endpoint}/simple/price"
                params = {
                    'ids': token,
                    'vs_currencies': 'usd'
                }
                
                response = self.session.get(url, params=params, timeout=5)
                response.raise_for_status()
                
                data = response.json()
                price = data.get(token, {}).get('usd')
                
                if price is not None:
                    self.price_cache[token] = float(price)
                    self.cache_timestamps[token] = time.time()
                    self.api_call_count += 1
                    logger.info(f"Fetched price for {token}: ${price}")
                    return float(price)
                else:
                    logger.warning(f"Price not found in API response for {token}")
                    return None
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to fetch price for {token}: {e}")
                # Return cached value if available
                return self.price_cache.get(token)
            except (KeyError, ValueError, TypeError) as e:
                logger.error(f"Failed to parse price data for {token}: {e}")
                return None
    
    def generate_price_oracle_event(self, tx_id: str, amount: int, token: str = "ethereum") -> Optional[OracleEvent]:
        """
        Generate price oracle event for transaction validation
        
        Args:
            tx_id: Transaction ID
            amount: Transaction amount in QTCL wei
            token: Reference token for pricing
            
        Returns:
            OracleEvent if successful, None otherwise
        """
        price = self.fetch_current_price(token)
        
        if price is None:
            logger.warning(f"Cannot generate price oracle event for {tx_id} - price unavailable")
            return None
        
        timestamp = int(time.time())
        
        # Validate price reasonableness
        is_reasonable = self.validate_price_reasonableness(amount, price)
        
        oracle_data = {
            'price_usd': price,
            'token': token,
            'amount_qtcl_wei': amount,
            'reasonable': is_reasonable,
            'api_source': self.api_endpoint,
            'cache_hit': token in self.price_cache and 
                        (timestamp - self.cache_timestamps.get(token, 0)) < PRICE_CACHE_TTL_SECONDS
        }
        
        proof = self.generate_price_commitment(tx_id, price, timestamp)
        
        oracle_event = OracleEvent(
            oracle_id=f"price_{tx_id}_{timestamp}",
            oracle_type=OracleType.PRICE,
            tx_id=tx_id,
            oracle_data=oracle_data,
            proof=proof,
            timestamp=timestamp,
            priority=6
        )
        
        with self.lock:
            self.event_count += 1
        
        logger.info(f"Price oracle event generated for {tx_id}, price: ${price}, reasonable: {is_reasonable}")
        return oracle_event
    
    def validate_price_reasonableness(self, amount: int, price: float, 
                                     threshold_factor: float = 1000.0) -> bool:
        """
        Validate if transaction amount is reasonable given current price
        
        Args:
            amount: Amount in QTCL wei
            price: Current token price in USD
            threshold_factor: Maximum reasonable amount factor
            
        Returns:
            True if amount is reasonable, False otherwise
        """
        # Convert amount to approximate USD value
        # Assuming 1 QTCL = 1e-18 ETH for scaling
        qtcl_in_eth = amount / 1e18
        usd_value = qtcl_in_eth * price
        
        # Check if value is within reasonable bounds (< $1M per tx)
        max_reasonable_usd = threshold_factor * 1000
        is_reasonable = usd_value < max_reasonable_usd
        
        if not is_reasonable:
            logger.warning(f"Transaction amount appears unreasonable: ${usd_value:.2f} USD")
        
        return is_reasonable
    
    def generate_price_commitment(self, tx_id: str, price: float, timestamp: int) -> str:
        """Generate cryptographic commitment for price oracle"""
        commitment_data = f"{tx_id}:{price:.8f}:{timestamp}".encode('utf-8')
        return hashlib.sha256(commitment_data).hexdigest()
    
    def get_statistics(self) -> Dict:
        """Get price oracle statistics"""
        with self.lock:
            return {
                'total_events': self.event_count,
                'api_calls': self.api_call_count,
                'cached_tokens': len(self.price_cache),
                'cache_hit_rate': self._calculate_cache_hit_rate()
            }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        if self.event_count == 0:
            return 0.0
        cache_hits = self.event_count - self.api_call_count
        return (cache_hits / self.event_count) * 100.0


class EventOracle:
    """
    Event oracle for listening to external smart contract events
    """
    
    def __init__(self, 
                 web3_provider: Optional[str] = None,
                 contract_address: Optional[str] = None,
                 contract_abi: Optional[List] = None):
        self.web3_provider = web3_provider or os.getenv("WEB3_PROVIDER_URL")
        self.contract_address = contract_address
        self.contract_abi = contract_abi or []
        self.event_count = 0
        self.lock = threading.Lock()
        
        if not WEB3_AVAILABLE:
            logger.warning("Web3.py not available - EventOracle will not function")
            self.w3 = None
            self.contract = None
            return
        
        if self.web3_provider:
            try:
                self.w3 = Web3(Web3.HTTPProvider(self.web3_provider))
                if self.w3.is_connected():
                    logger.info(f"EventOracle connected to Web3 provider: {self.web3_provider}")
                    
                    if self.contract_address and self.contract_abi:
                        self.contract = self.w3.eth.contract(
                            address=Web3.to_checksum_address(self.contract_address),
                            abi=self.contract_abi
                        )
                        logger.info(f"EventOracle contract loaded: {self.contract_address}")
                    else:
                        self.contract = None
                else:
                    logger.error("Failed to connect to Web3 provider")
                    self.w3 = None
                    self.contract = None
            except Exception as e:
                logger.error(f"Failed to initialize Web3: {e}")
                self.w3 = None
                self.contract = None
        else:
            logger.warning("No Web3 provider configured - EventOracle disabled")
            self.w3 = None
            self.contract = None
    
    def listen_for_events(self, event_name: str = None, 
                         from_block: int = 'latest',
                         to_block: int = 'latest') -> Generator[Dict, None, None]:
        """
        Listen for contract events
        
        Args:
            event_name: Specific event to listen for (None for all events)
            from_block: Starting block number
            to_block: Ending block number
            
        Yields:
            Event dictionaries
        """
        if not self.w3 or not self.contract:
            logger.warning("EventOracle not initialized - cannot listen for events")
            return
        
        try:
            if event_name and hasattr(self.contract.events, event_name):
                event_filter = getattr(self.contract.events, event_name).create_filter(
                    fromBlock=from_block,
                    toBlock=to_block
                )
            else:
                # Get all events
                event_filter = self.w3.eth.filter({
                    'fromBlock': from_block,
                    'toBlock': to_block,
                    'address': self.contract_address
                })
            
            for event in event_filter.get_all_entries():
                yield self.parse_event_data(event)
                
        except Exception as e:
            logger.error(f"Error listening for events: {e}")
    
    def parse_event_data(self, event: Any) -> Dict:
        """Parse raw event data into standardized format"""
        try:
            parsed = {
                'event_name': event.get('event', 'Unknown'),
                'transaction_hash': event.get('transactionHash', b'').hex(),
                'block_number': event.get('blockNumber', 0),
                'args': dict(event.get('args', {})),
                'log_index': event.get('logIndex', 0),
                'address': event.get('address', '')
            }
            return parsed
        except Exception as e:
            logger.error(f"Failed to parse event data: {e}")
            return {}
    
    def validate_event_signature(self, event: Dict) -> bool:
        """Validate event signature against contract ABI"""
        if not self.contract:
            return False
        
        try:
            event_name = event.get('event_name')
            if not event_name or not hasattr(self.contract.events, event_name):
                return False
            return True
        except Exception as e:
            logger.error(f"Event signature validation failed: {e}")
            return False
    
    def generate_event_commitment(self, event: Dict) -> str:
        """Generate cryptographic commitment for event"""
        event_str = json.dumps(event, sort_keys=True)
        return hashlib.sha256(event_str.encode('utf-8')).hexdigest()
    
    def generate_event_oracle_event(self, tx_id: str, contract_event: Dict) -> Optional[OracleEvent]:
        """
        Generate oracle event from contract event
        
        Args:
            tx_id: Transaction ID
            contract_event: Parsed contract event
            
        Returns:
            OracleEvent if successful, None otherwise
        """
        if not self.validate_event_signature(contract_event):
            logger.warning(f"Invalid event signature for {tx_id}")
            return None
        
        timestamp = int(time.time())
        proof = self.generate_event_commitment(contract_event)
        
        oracle_data = {
            'event_name': contract_event.get('event_name'),
            'transaction_hash': contract_event.get('transaction_hash'),
            'block_number': contract_event.get('block_number'),
            'event_args': contract_event.get('args', {}),
            'contract_address': self.contract_address
        }
        
        oracle_event = OracleEvent(
            oracle_id=f"event_{tx_id}_{timestamp}",
            oracle_type=OracleType.EVENT,
            tx_id=tx_id,
            oracle_data=oracle_data,
            proof=proof,
            timestamp=timestamp,
            priority=7
        )
        
        with self.lock:
            self.event_count += 1
        
        logger.info(f"Event oracle generated for {tx_id}, event: {contract_event.get('event_name')}")
        return oracle_event
    
    def get_statistics(self) -> Dict:
        """Get event oracle statistics"""
        with self.lock:
            return {
                'total_events': self.event_count,
                'web3_connected': self.w3 is not None and self.w3.is_connected() if self.w3 else False,
                'contract_loaded': self.contract is not None
            }


class RandomOracle:
    """
    Random oracle using VRF (Verifiable Random Function) for provable randomness
    """
    
    def __init__(self, vrf_key: Optional[bytes] = None):
        self.vrf_key = vrf_key or secrets.token_bytes(RANDOM_ORACLE_VRF_KEY_SIZE)
        self.event_count = 0
        self.lock = threading.Lock()
        
        # Generate ECDSA key pair for VRF
        self.private_key = ec.generate_private_key(ec.SECP256K1(), default_backend())
        self.public_key = self.private_key.public_key()
        
        logger.info("RandomOracle initialized with VRF key")
    
    def generate_random_value(self, seed: int) -> Tuple[int, str]:
        """
        Generate cryptographically secure random value with VRF proof
        
        Args:
            seed: Seed value for randomness
            
        Returns:
            Tuple of (random_value, vrf_proof)
        """
        # Create message from seed
        message = str(seed).encode('utf-8')
        
        # Sign message (VRF)
        signature = self.private_key.sign(
            message,
            ec.ECDSA(hashes.SHA256())
        )
        
        # Derive random value from signature
        random_hash = hashlib.sha256(signature).digest()
        random_value = int.from_bytes(random_hash[:8], byteorder='big')
        
        # Create proof
        vrf_proof = signature.hex()
        
        return random_value, vrf_proof
    
    def verify_random_proof(self, value: int, proof: str, message: bytes) -> bool:
        """
        Verify VRF proof
        
        Args:
            value: Random value to verify
            proof: VRF proof
            message: Original message
            
        Returns:
            True if proof valid, False otherwise
        """
        try:
            signature = bytes.fromhex(proof)
            self.public_key.verify(
                signature,
                message,
                ec.ECDSA(hashes.SHA256())
            )
            
            # Verify value derivation
            random_hash = hashlib.sha256(signature).digest()
            expected_value = int.from_bytes(random_hash[:8], byteorder='big')
            
            return value == expected_value
            
        except (InvalidSignature, ValueError) as e:
            logger.error(f"VRF proof verification failed: {e}")
            return False
    
    def use_quantum_entropy(self, quantum_result: Dict) -> int:
        """
        Use quantum measurement entropy as randomness seed
        
        Args:
            quantum_result: Quantum measurement result with entropy data
            
        Returns:
            Entropy-derived seed value
        """
        entropy_bits = quantum_result.get('entropy_bits', 0)
        entropy_percent = quantum_result.get('entropy_percent', 0.0)
        
        # Extract randomness from quantum measurement
        bitstring = quantum_result.get('dominant_states', ['0000000'])[0]
        quantum_seed = int(bitstring, 2) if bitstring else 0
        
        # Combine with entropy metrics
        combined_seed = (quantum_seed << 16) | int(entropy_percent * 10000)
        
        return combined_seed
    
    def generate_random_oracle_event(self, tx_id: str, 
                                     quantum_result: Optional[Dict] = None) -> OracleEvent:
        """
        Generate random oracle event
        
        Args:
            tx_id: Transaction ID
            quantum_result: Optional quantum measurement for entropy
            
        Returns:
            OracleEvent with random value and proof
        """
        timestamp = int(time.time())
        
        # Generate seed from quantum entropy or timestamp
        if quantum_result:
            seed = self.use_quantum_entropy(quantum_result)
        else:
            seed = timestamp ^ hash(tx_id)
        
        random_value, vrf_proof = self.generate_random_value(seed)
        
        oracle_data = {
            'random_value': random_value,
            'seed': seed,
            'vrf_proof': vrf_proof,
            'quantum_entropy_used': quantum_result is not None,
            'entropy_percent': quantum_result.get('entropy_percent', 0.0) if quantum_result else 0.0
        }
        
        proof = hashlib.sha256(f"{tx_id}:{random_value}:{vrf_proof}".encode('utf-8')).hexdigest()
        
        oracle_event = OracleEvent(
            oracle_id=f"random_{tx_id}_{timestamp}",
            oracle_type=OracleType.RANDOM,
            tx_id=tx_id,
            oracle_data=oracle_data,
            proof=proof,
            timestamp=timestamp,
            priority=5
        )
        
        with self.lock:
            self.event_count += 1
        
        logger.info(f"Random oracle generated for {tx_id}, value: {random_value}")
        return oracle_event
    
    def get_statistics(self) -> Dict:
        """Get random oracle statistics"""
        with self.lock:
            return {
                'total_events': self.event_count,
                'vrf_enabled': True
            }


class EntropyOracle:
    """
    Entropy oracle monitoring quantum entropy quality
    """
    
    def __init__(self, min_entropy_threshold: float = ENTROPY_MIN_THRESHOLD):
        self.min_entropy_threshold = min_entropy_threshold
        self.event_count = 0
        self.high_quality_count = 0
        self.low_quality_count = 0
        self.lock = threading.Lock()
        
        logger.info(f"EntropyOracle initialized with min threshold: {min_entropy_threshold}")
    
    def check_entropy_quality(self, tx_id: str, entropy_score: float) -> Tuple[bool, str]:
        """
        Check if entropy meets minimum quality standards
        
        Args:
            tx_id: Transaction ID
            entropy_score: Entropy score (0.0-1.0)
            
        Returns:
            Tuple of (is_valid, message)
        """
        if entropy_score < self.min_entropy_threshold:
            message = f"Entropy too low: {entropy_score:.2%} < {self.min_entropy_threshold:.2%}"
            logger.warning(f"Transaction {tx_id}: {message}")
            with self.lock:
                self.low_quality_count += 1
            return False, message
        
        with self.lock:
            self.high_quality_count += 1
        
        if entropy_score >= ENTROPY_OPTIMAL_THRESHOLD:
            message = f"Optimal entropy: {entropy_score:.2%}"
        else:
            message = f"Acceptable entropy: {entropy_score:.2%}"
        
        logger.info(f"Transaction {tx_id}: {message}")
        return True, message
    
    def calculate_entropy_score(self, measurements: Dict) -> float:
        """
        Calculate entropy score from quantum measurements
        
        Args:
            measurements: Quantum measurement dictionary
            
        Returns:
            Entropy score (0.0-1.0)
        """
        entropy_percent = measurements.get('entropy_percent', 0.0)
        
        # Normalize to 0-1 range if needed
        if entropy_percent > 1.0:
            entropy_percent = entropy_percent / 100.0
        
        return min(1.0, max(0.0, entropy_percent))
    
    def trigger_entropy_oracle_event(self, tx_id: str, 
                                    measurements: Dict) -> Optional[OracleEvent]:
        """
        Trigger entropy oracle event
        
        Args:
            tx_id: Transaction ID
            measurements: Quantum measurements
            
        Returns:
            OracleEvent if entropy valid, None otherwise
        """
        entropy_score = self.calculate_entropy_score(measurements)
        is_valid, message = self.check_entropy_quality(tx_id, entropy_score)
        
        timestamp = int(time.time())
        
        oracle_data = {
            'entropy_score': entropy_score,
            'entropy_valid': is_valid,
            'validation_message': message,
            'min_threshold': self.min_entropy_threshold,
            'entropy_bits': measurements.get('entropy_bits', 0),
            'total_measurements': measurements.get('total_measurements', 0)
        }
        
        proof = self.generate_entropy_commitment(tx_id, entropy_score, timestamp)
        
        oracle_event = OracleEvent(
            oracle_id=f"entropy_{tx_id}_{timestamp}",
            oracle_type=OracleType.ENTROPY,
            tx_id=tx_id,
            oracle_data=oracle_data,
            proof=proof,
            timestamp=timestamp,
            priority=9  # Highest priority - must validate entropy first
        )
        
        with self.lock:
            self.event_count += 1
        
        logger.info(f"Entropy oracle for {tx_id}: score={entropy_score:.2%}, valid={is_valid}")
        return oracle_event
    
    def generate_entropy_commitment(self, tx_id: str, entropy_score: float, timestamp: int) -> str:
        """Generate cryptographic commitment for entropy oracle"""
        commitment_data = f"{tx_id}:{entropy_score:.6f}:{timestamp}".encode('utf-8')
        return hashlib.sha256(commitment_data).hexdigest()
    
    def get_statistics(self) -> Dict:
        """Get entropy oracle statistics"""
        with self.lock:
            total = self.high_quality_count + self.low_quality_count
            quality_rate = (self.high_quality_count / total * 100.0) if total > 0 else 0.0
            
            return {
                'total_events': self.event_count,
                'high_quality_count': self.high_quality_count,
                'low_quality_count': self.low_quality_count,
                'quality_rate_percent': quality_rate,
                'min_threshold': self.min_entropy_threshold
            }


# ============================================================================
# MODULE 2: ORACLE EVENT MANAGER
# ============================================================================

class OracleEventQueue:
    """
    Priority queue for oracle events
    """
    
    def __init__(self, max_queue_size: int = ORACLE_EVENT_MAX_QUEUE_SIZE):
        self.max_queue_size = max_queue_size
        self.queue = queue.PriorityQueue(maxsize=max_queue_size)
        self.enqueued_count = 0
        self.dequeued_count = 0
        self.dropped_count = 0
        self.lock = threading.Lock()
        
        logger.info(f"OracleEventQueue initialized with max size: {max_queue_size}")
    
    def enqueue_event(self, oracle_event: OracleEvent) -> bool:
        """
        Add oracle event to queue
        
        Args:
            oracle_event: OracleEvent to enqueue
            
        Returns:
            True if enqueued, False if queue full
        """
        try:
            # Priority queue uses tuple (priority, timestamp, event)
            # Lower priority number = higher priority
            priority_key = (
                -oracle_event.priority,  # Negative for reverse sort
                oracle_event.timestamp,
                oracle_event.oracle_id
            )
            
            self.queue.put((priority_key, oracle_event), block=False)
            
            with self.lock:
                self.enqueued_count += 1
            
            logger.debug(f"Enqueued oracle event: {oracle_event.oracle_id}, priority: {oracle_event.priority}")
            return True
            
        except queue.Full:
            with self.lock:
                self.dropped_count += 1
            logger.warning(f"Oracle event queue full - dropped event: {oracle_event.oracle_id}")
            return False
    
    def dequeue_events(self, batch_size: int = ORACLE_EVENT_BATCH_SIZE) -> List[OracleEvent]:
        """
        Dequeue batch of oracle events
        
        Args:
            batch_size: Maximum number of events to dequeue
            
        Returns:
            List of OracleEvents
        """
        events = []
        
        for _ in range(batch_size):
            try:
                priority_key, event = self.queue.get(block=False)
                events.append(event)
                
                with self.lock:
                    self.dequeued_count += 1
                    
            except queue.Empty:
                break
        
        if events:
            logger.debug(f"Dequeued {len(events)} oracle events")
        
        return events
    
    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self.queue.qsize()
    
    def clear_expired_events(self, max_age_seconds: int = ORACLE_EVENT_MAX_AGE_SECONDS):
        """
        Clear expired events from queue
        
        Args:
            max_age_seconds: Maximum event age in seconds
        """
        current_time = int(time.time())
        expired_count = 0
        
        # Temporarily store non-expired events
        temp_events = []
        
        while not self.queue.empty():
            try:
                priority_key, event = self.queue.get(block=False)
                
                age = current_time - event.timestamp
                if age < max_age_seconds:
                    temp_events.append((priority_key, event))
                else:
                    expired_count += 1
                    
            except queue.Empty:
                break
        
        # Re-add non-expired events
        for priority_key, event in temp_events:
            try:
                self.queue.put((priority_key, event), block=False)
            except queue.Full:
                break
        
        if expired_count > 0:
            logger.info(f"Cleared {expired_count} expired oracle events")
            with self.lock:
                self.dropped_count += expired_count
    
    def get_statistics(self) -> Dict:
        """Get queue statistics"""
        with self.lock:
            return {
                'queue_size': self.get_queue_size(),
                'max_queue_size': self.max_queue_size,
                'enqueued_count': self.enqueued_count,
                'dequeued_count': self.dequeued_count,
                'dropped_count': self.dropped_count,
                'utilization_percent': (self.get_queue_size() / self.max_queue_size * 100.0) if self.max_queue_size > 0 else 0.0
            }


class OracleEventDispatcher:
    """
    Dispatch oracle events to collapse handlers
    """
    
    def __init__(self, collapse_handler: Callable, supabase_client: Client):
        self.collapse_handler = collapse_handler
        self.supabase = supabase_client
        self.dispatch_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.lock = threading.Lock()
        
        logger.info("OracleEventDispatcher initialized")
    
    def dispatch_event(self, event: OracleEvent) -> bool:
        """
        Dispatch oracle event to collapse handler
        
        Args:
            event: OracleEvent to dispatch
            
        Returns:
            True if dispatch successful, False otherwise
        """
        try:
            # Validate oracle authenticity
            if not self.validate_oracle_authenticity(event):
                logger.warning(f"Oracle event failed authenticity check: {event.oracle_id}")
                with self.lock:
                    self.failure_count += 1
                return False
            
            # Find matching transactions
            matching_txs = self.find_matching_transactions(event)
            
            if not matching_txs:
                logger.debug(f"No matching transactions for oracle event: {event.oracle_id}")
                with self.lock:
                    self.failure_count += 1
                return False
            
            # Log dispatch
            self._log_dispatch(event, matching_txs)
            
            # Invoke collapse handler for each matching transaction
            success = True
            for tx_id in matching_txs:
                try:
                    self.collapse_handler(tx_id, event)
                except Exception as e:
                    logger.error(f"Collapse handler failed for {tx_id}: {e}")
                    success = False
            
            with self.lock:
                self.dispatch_count += 1
                if success:
                    self.success_count += 1
                else:
                    self.failure_count += 1
            
            # Mark event as dispatched
            event.dispatched = True
            
            logger.info(f"Dispatched oracle event {event.oracle_id} to {len(matching_txs)} transactions")
            return success
            
        except Exception as e:
            self.handle_dispatch_error(event, e)
            return False
    
    def find_matching_transactions(self, event: OracleEvent) -> List[str]:
        """
        Find transactions matching oracle event
        
        Args:
            event: OracleEvent
            
        Returns:
            List of matching transaction IDs
        """
        try:
            # Query transactions in superposition state
            result = self.supabase.table('transactions').select('id').eq(
                'status', TransactionStatus.SUPERPOSITION.value
            ).execute()
            
            matching_txs = []
            
            for tx in result.data:
                tx_id = tx['id']
                
                # Check if event matches transaction
                if event.tx_id == tx_id:
                    matching_txs.append(tx_id)
            
            return matching_txs
            
        except Exception as e:
            logger.error(f"Failed to find matching transactions: {e}")
            return []
    
    def validate_oracle_authenticity(self, event: OracleEvent) -> bool:
        """
        Validate oracle event authenticity
        
        Args:
            event: OracleEvent to validate
            
        Returns:
            True if authentic, False otherwise
        """
        # Verify proof matches event data
        expected_proof = self._calculate_expected_proof(event)
        
        if event.proof != expected_proof:
            logger.warning(f"Oracle proof mismatch for {event.oracle_id}")
            return False
        
        # Check timestamp reasonableness
        current_time = int(time.time())
        age = current_time - event.timestamp
        
        if age > ORACLE_EVENT_MAX_AGE_SECONDS or age < -60:
            logger.warning(f"Oracle timestamp unreasonable: {age}s old")
            return False
        
        return True
    
    def _calculate_expected_proof(self, event: OracleEvent) -> str:
        """Calculate expected proof for event"""
        if event.oracle_type == OracleType.TIME:
            data = f"{event.tx_id}:{event.oracle_data.get('trigger_time', 0)}"
        elif event.oracle_type == OracleType.PRICE:
            data = f"{event.tx_id}:{event.oracle_data.get('price_usd', 0):.8f}:{event.timestamp}"
        elif event.oracle_type == OracleType.RANDOM:
            data = f"{event.tx_id}:{event.oracle_data.get('random_value', 0)}:{event.oracle_data.get('vrf_proof', '')}"
        elif event.oracle_type == OracleType.ENTROPY:
            data = f"{event.tx_id}:{event.oracle_data.get('entropy_score', 0):.6f}:{event.timestamp}"
        elif event.oracle_type == OracleType.EVENT:
            data = json.dumps(event.oracle_data, sort_keys=True)
        else:
            return event.proof
        
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
    def _log_dispatch(self, event: OracleEvent, matching_txs: List[str]):
        """Log dispatch event to database"""
        try:
            log_entry = {
                'oracle_id': event.oracle_id,
                'oracle_type': event.oracle_type.value,
                'tx_ids': matching_txs,
                'event_data': event.oracle_data,
                'proof': event.proof,
                'timestamp': datetime.fromtimestamp(event.timestamp).isoformat(),
                'dispatch_count': len(matching_txs)
            }
            
            self.supabase.table('oracle_dispatch_log').insert(log_entry).execute()
            
        except Exception as e:
            logger.error(f"Failed to log oracle dispatch: {e}")
    
    def handle_dispatch_error(self, event: OracleEvent, error: Exception):
        """Handle dispatch error"""
        logger.error(f"Dispatch error for {event.oracle_id}: {error}")
        
        with self.lock:
            self.failure_count += 1
        
        # Log error to database
        try:
            error_entry = {
                'oracle_id': event.oracle_id,
                'error_message': str(error),
                'event_data': event.oracle_data,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self.supabase.table('oracle_event_log').insert(error_entry).execute()
            
        except Exception as log_error:
            logger.error(f"Failed to log dispatch error: {log_error}")
    
    def get_statistics(self) -> Dict:
        """Get dispatcher statistics"""
        with self.lock:
            success_rate = (self.success_count / self.dispatch_count * 100.0) if self.dispatch_count > 0 else 0.0
            
            return {
                'total_dispatches': self.dispatch_count,
                'successful_dispatches': self.success_count,
                'failed_dispatches': self.failure_count,
                'success_rate_percent': success_rate
            }


class OracleAuthentication:
    """
    Oracle authentication and reputation tracking
    """
    
    def __init__(self, oracle_pubkeys: Dict[str, bytes]):
        self.oracle_pubkeys = oracle_pubkeys
        self.oracle_reputations = defaultdict(lambda: 1.0)  # Default reputation = 1.0
        self.verification_count = defaultdict(int)
        self.lock = threading.Lock()
        
        logger.info(f"OracleAuthentication initialized with {len(oracle_pubkeys)} public keys")
    
    def verify_signature(self, oracle_type: str, data: bytes, signature: bytes) -> bool:
        """
        Verify ECDSA signature
        
        Args:
            oracle_type: Type of oracle
            data: Data that was signed
            signature: Signature to verify
            
        Returns:
            True if signature valid, False otherwise
        """
        pubkey = self.oracle_pubkeys.get(oracle_type)
        if not pubkey:
            logger.warning(f"No public key found for oracle type: {oracle_type}")
            return False
        
        try:
            # Load public key
            public_key = serialization.load_pem_public_key(pubkey, backend=default_backend())
            
            # Verify signature
            public_key.verify(
                signature,
                data,
                ec.ECDSA(hashes.SHA256())
            )
            
            with self.lock:
                self.verification_count[oracle_type] += 1
            
            return True
            
        except (InvalidSignature, ValueError, TypeError) as e:
            logger.error(f"Signature verification failed for {oracle_type}: {e}")
            return False
    
    def verify_merkle_proof(self, root: str, proof: List[str], leaf: str) -> bool:
        """
        Verify Merkle proof
        
        Args:
            root: Merkle root hash
            proof: List of sibling hashes
            leaf: Leaf value to verify
            
        Returns:
            True if proof valid, False otherwise
        """
        current_hash = hashlib.sha256(leaf.encode('utf-8')).hexdigest()
        
        for sibling in proof:
            if current_hash < sibling:
                combined = current_hash + sibling
            else:
                combined = sibling + current_hash
            
            current_hash = hashlib.sha256(combined.encode('utf-8')).hexdigest()
        
        return current_hash == root
    
    def verify_vrf_proof(self, value: int, proof: str, pubkey: bytes, message: bytes) -> bool:
        """
        Verify VRF proof
        
        Args:
            value: Random value
            proof: VRF proof (signature hex)
            pubkey: Public key bytes
            message: Original message
            
        Returns:
            True if proof valid, False otherwise
        """
        try:
            # Load public key
            public_key = serialization.load_pem_public_key(pubkey, backend=default_backend())
            
            # Verify signature
            signature = bytes.fromhex(proof)
            public_key.verify(
                signature,
                message,
                ec.ECDSA(hashes.SHA256())
            )
            
            # Verify value derivation
            random_hash = hashlib.sha256(signature).digest()
            expected_value = int.from_bytes(random_hash[:8], byteorder='big')
            
            return value == expected_value
            
        except (InvalidSignature, ValueError, TypeError) as e:
            logger.error(f"VRF proof verification failed: {e}")
            return False
    
    def get_oracle_reputation(self, oracle_id: str) -> float:
        """Get reputation score for oracle (0.0-1.0)"""
        with self.lock:
            return self.oracle_reputations[oracle_id]
    
    def update_oracle_reputation(self, oracle_id: str, success: bool):
        """
        Update oracle reputation based on performance
        
        Args:
            oracle_id: Oracle identifier
            success: True if oracle performed well, False otherwise
        """
        with self.lock:
            current_reputation = self.oracle_reputations[oracle_id]
            
            if success:
                # Boost reputation
                new_reputation = min(1.0, current_reputation + ORACLE_REPUTATION_BOOST_RATE)
            else:
                # Decay reputation (slash)
                new_reputation = max(0.0, current_reputation - ORACLE_REPUTATION_DECAY_RATE * 5)
            
            self.oracle_reputations[oracle_id] = new_reputation
            
            logger.debug(f"Oracle {oracle_id} reputation: {current_reputation:.3f} -> {new_reputation:.3f}")
    
    def get_statistics(self) -> Dict:
        """Get authentication statistics"""
        with self.lock:
            return {
                'total_verifications': sum(self.verification_count.values()),
                'verifications_by_type': dict(self.verification_count),
                'oracle_count': len(self.oracle_reputations),
                'average_reputation': sum(self.oracle_reputations.values()) / len(self.oracle_reputations) if self.oracle_reputations else 0.0
            }


# Continuing oracle_engine.py...

# ============================================================================
# MODULE 3: SUPERPOSITION COLLAPSE
# ============================================================================

class SuperpositionCollapse:
    """
    Main collapse logic - transforms quantum superposition to classical outcome
    """
    
    def __init__(self, supabase_client: Client, db_pool: ThreadedConnectionPool):
        self.supabase = supabase_client
        self.db_pool = db_pool
        self.collapse_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.lock = threading.Lock()
        
        # Initialize outcome interpreter and causality validator
        self.outcome_interpreter = OutcomeInterpreter()
        self.causality_validator = CausalityValidator(supabase_client, db_pool)
        
        logger.info("SuperpositionCollapse initialized")
    
    def collapse_transaction(self,
                            tx_id: str,
                            oracle_data: Dict,
                            quantum_measurements: Dict,
                            measurement_basis: int = 0) -> Optional[CollapseResult]:
        """
        Collapse quantum superposition to classical outcome
        
        Args:
            tx_id: Transaction ID
            oracle_data: Oracle measurement data
            quantum_measurements: Quantum circuit measurements
            measurement_basis: Measurement basis (0=computational, 1=hadamard)
            
        Returns:
            CollapseResult if successful, None otherwise
        """
        try:
            with self.lock:
                self.collapse_count += 1
            
            logger.info(f"Collapsing superposition for transaction: {tx_id}")
            
            # Get transaction details
            tx_result = self.supabase.table('transactions').select('*').eq('id', tx_id).execute()
            
            if not tx_result.data:
                logger.error(f"Transaction not found: {tx_id}")
                return None
            
            tx = tx_result.data[0]
            
            # Extract quantum bitstring from measurements
            dominant_states = quantum_measurements.get('dominant_states', [])
            if not dominant_states:
                logger.error(f"No quantum measurements found for {tx_id}")
                return None
            
            # Use first dominant state as collapsed bitstring
            collapsed_bitstring = dominant_states[0]
            
            # Apply oracle measurement to collapse
            final_bitstring = self._apply_oracle_measurement(
                collapsed_bitstring,
                oracle_data,
                measurement_basis
            )
            
            logger.info(f"Quantum state collapsed: {collapsed_bitstring} -> {final_bitstring}")
            
            # Interpret collapsed state as transaction outcome
            interpretation = self.outcome_interpreter.interpret_collapsed_state(
                final_bitstring,
                tx['tx_type'],
                oracle_data
            )
            
            # Validate causality
            causality_valid, causality_message = self.causality_validator.validate_causality(
                interpretation,
                tx,
                quantum_measurements.get('hyperbolic_distance', 0.0)
            )
            
            if not causality_valid:
                logger.warning(f"Causality validation failed for {tx_id}: {causality_message}")
                interpretation['outcome'] = CollapseOutcome.REJECTED
                interpretation['rejection_reason'] = causality_message
            
            # Generate collapse proof
            collapse_proof = self.generate_collapse_proof(
                tx_id,
                oracle_data,
                final_bitstring
            )
            
            # Determine final outcome
            outcome = interpretation.get('outcome', CollapseOutcome.ERROR)
            
            # Create collapse result
            collapse_result = CollapseResult(
                tx_id=tx_id,
                outcome=outcome,
                collapsed_bitstring=final_bitstring,
                collapse_proof=collapse_proof,
                oracle_data=oracle_data,
                interpretation=interpretation,
                causality_valid=causality_valid,
                timestamp=int(time.time()),
                error_message=interpretation.get('rejection_reason') if outcome == CollapseOutcome.REJECTED else None
            )
            
            # Store collapse record
            self._store_collapse_record(collapse_result, quantum_measurements)
            
            # Update transaction status
            self._update_transaction_status(tx_id, collapse_result)
            
            with self.lock:
                if outcome == CollapseOutcome.APPROVED:
                    self.success_count += 1
                else:
                    self.failure_count += 1
            
            logger.info(f"Collapse complete for {tx_id}: {outcome.value}")
            return collapse_result
            
        except Exception as e:
            logger.error(f"Collapse failed for {tx_id}: {e}", exc_info=True)
            with self.lock:
                self.failure_count += 1
            return None
    
    def _apply_oracle_measurement(self,
                                 bitstring: str,
                                 oracle_data: Dict,
                                 measurement_basis: int) -> str:
        """
        Apply oracle measurement to quantum bitstring
        
        Args:
            bitstring: Quantum measurement bitstring
            oracle_data: Oracle data
            measurement_basis: Measurement basis
            
        Returns:
            Collapsed bitstring after oracle measurement
        """
        # Oracle measurement collapses superposition based on oracle data
        # For time oracle: use timestamp parity to influence bits
        # For price oracle: use price digits to influence bits
        # For entropy oracle: validate and pass through
        # For random oracle: use VRF value to influence bits
        
        oracle_type = oracle_data.get('oracle_type', 'time')
        
        if oracle_type == 'time':
            # Use timestamp to create measurement pattern
            timestamp = oracle_data.get('trigger_time', int(time.time()))
            measurement_pattern = bin(timestamp)[-len(bitstring):].zfill(len(bitstring))
            
        elif oracle_type == 'price':
            # Use price to create measurement pattern
            price = oracle_data.get('price_usd', 0.0)
            price_int = int(price * 100000000)  # Convert to satoshis
            measurement_pattern = bin(price_int)[-len(bitstring):].zfill(len(bitstring))
            
        elif oracle_type == 'random':
            # Use VRF random value
            random_value = oracle_data.get('random_value', 0)
            measurement_pattern = bin(random_value)[-len(bitstring):].zfill(len(bitstring))
            
        else:
            # For entropy and event oracles, pass through original
            measurement_pattern = bitstring
        
        # XOR quantum bitstring with oracle measurement pattern
        collapsed = ''
        for i, (q_bit, o_bit) in enumerate(zip(bitstring, measurement_pattern)):
            # Oracle measurement collapses each qubit
            if measurement_basis == 0:  # Computational basis
                collapsed += o_bit
            else:  # Hadamard basis
                collapsed += str(int(q_bit) ^ int(o_bit))
        
        return collapsed
    
    def generate_collapse_proof(self,
                                tx_id: str,
                                oracle_data: Dict,
                                collapsed_state: str) -> str:
        """
        Generate cryptographic proof of collapse
        
        Args:
            tx_id: Transaction ID
            oracle_data: Oracle data used for collapse
            collapsed_state: Final collapsed bitstring
            
        Returns:
            Collapse proof hash
        """
        proof_data = {
            'tx_id': tx_id,
            'oracle_data': oracle_data,
            'collapsed_state': collapsed_state,
            'timestamp': int(time.time()),
            'algorithm': COLLAPSE_PROOF_ALGORITHM
        }
        
        proof_str = json.dumps(proof_data, sort_keys=True)
        proof_hash = hashlib.sha256(proof_str.encode('utf-8')).hexdigest()
        
        return proof_hash
    
    def _store_collapse_record(self, result: CollapseResult, quantum_measurements: Dict):
        """Store collapse record in database"""
        try:
            collapse_record = {
                'transaction_id': result.tx_id,
                'collapsed_bitstring': result.collapsed_bitstring,
                'collapse_proof': result.collapse_proof,
                'outcome': result.outcome.value,
                'oracle_data': result.oracle_data,
                'interpretation': result.interpretation,
                'causality_valid': result.causality_valid,
                'entropy_score': quantum_measurements.get('entropy_percent', 0.0),
                'quantum_state_hash': quantum_measurements.get('circuit_hash', ''),
                'timestamp': datetime.fromtimestamp(result.timestamp).isoformat(),
                'error_message': result.error_message
            }
            
            self.supabase.table('superposition_collapses').insert(collapse_record).execute()
            logger.debug(f"Stored collapse record for {result.tx_id}")
            
        except Exception as e:
            logger.error(f"Failed to store collapse record: {e}")
    
    def _update_transaction_status(self, tx_id: str, result: CollapseResult):
        """Update transaction status after collapse"""
        try:
            update_data = {
                'status': TransactionStatus.COLLAPSED.value,
                'collapsed_outcome': result.outcome.value,
                'collapse_timestamp': datetime.fromtimestamp(result.timestamp).isoformat(),
                'collapse_proof': result.collapse_proof
            }
            
            self.supabase.table('transactions').update(update_data).eq('id', tx_id).execute()
            logger.debug(f"Updated transaction {tx_id} status to COLLAPSED")
            
        except Exception as e:
            logger.error(f"Failed to update transaction status: {e}")
    
    def get_statistics(self) -> Dict:
        """Get collapse statistics"""
        with self.lock:
            success_rate = (self.success_count / self.collapse_count * 100.0) if self.collapse_count > 0 else 0.0
            
            return {
                'total_collapses': self.collapse_count,
                'successful_collapses': self.success_count,
                'failed_collapses': self.failure_count,
                'success_rate_percent': success_rate
            }


class OutcomeInterpreter:
    """
    Interpret collapsed quantum bitstring as blockchain transaction outcome
    """
    
    def __init__(self):
        logger.info("OutcomeInterpreter initialized")
    
    def interpret_collapsed_state(self,
                                  bitstring: str,
                                  tx_type: str,
                                  oracle_data: Dict) -> Dict:
        """
        Interpret collapsed bitstring as transaction outcome
        
        Args:
            bitstring: Collapsed quantum bitstring
            tx_type: Transaction type
            oracle_data: Oracle data
            
        Returns:
            Interpretation dictionary
        """
        if tx_type == 'transfer':
            return self.interpret_transfer_outcome(bitstring, oracle_data)
        elif tx_type == 'stake':
            return self.interpret_stake_outcome(bitstring, oracle_data)
        elif tx_type == 'mint':
            return self.interpret_mint_outcome(bitstring, oracle_data)
        elif tx_type == 'burn':
            return self.interpret_burn_outcome(bitstring, oracle_data)
        elif tx_type == 'contract_call':
            return self.interpret_contract_call_outcome(bitstring, oracle_data)
        else:
            logger.warning(f"Unknown transaction type: {tx_type}")
            return {
                'outcome': CollapseOutcome.ERROR,
                'rejection_reason': f'Unknown transaction type: {tx_type}'
            }
    
    def interpret_transfer_outcome(self, bitstring: str, oracle_data: Dict) -> Dict:
        """
        Interpret transfer transaction outcome
        
        Bitstring format (8 bits):
        - Bits [0-3]: Approval decision (count 1s)
        - Bits [4-7]: Fee encoding (0-255)
        
        Approval logic:
        - 0-1 ones: REJECTED
        - 2-4 ones: PENDING (need more oracle data)
        - 5-8 ones: APPROVED
        """
        if len(bitstring) < 8:
            bitstring = bitstring.zfill(8)
        
        # Extract approval bits
        approval_bits = bitstring[:4]
        fee_bits = bitstring[4:8]
        
        # Count ones in approval section
        approval_count = approval_bits.count('1')
        
        # Determine outcome
        if approval_count >= 5:  # Wait, approval_bits is only 4 bits max
            # Let me fix this - count ones in approval_bits (max 4)
            pass
        
        # Actually count ones properly
        ones_in_approval = approval_bits.count('1')
        
        if ones_in_approval <= 1:
            outcome = CollapseOutcome.REJECTED
            reason = f"Insufficient quantum approval: {ones_in_approval}/4 bits"
        elif ones_in_approval <= 2:
            outcome = CollapseOutcome.PENDING
            reason = f"Borderline approval: {ones_in_approval}/4 bits - needs more oracle data"
        else:  # 3 or 4 ones
            outcome = CollapseOutcome.APPROVED
            reason = f"Quantum approval: {ones_in_approval}/4 bits"
        
        # Gas-free mode: no fees calculated from quantum state
        fee_value = int(fee_bits, 2)  # 0-15 range (retained for analysis only)
        gas_fee = 0  # GAS-FREE: quantum commitment hash replaces economic finality
        
        interpretation = {
            'outcome': outcome,
            'approval_bits': approval_bits,
            'approval_count': ones_in_approval,
            'fee_bits': fee_bits,
            'calculated_gas_fee': gas_fee,
            'reason': reason,
            'oracle_influenced': True
        }
        
        logger.debug(f"Transfer interpretation: {interpretation}")
        return interpretation
    
    def interpret_stake_outcome(self, bitstring: str, oracle_data: Dict) -> Dict:
        """
        Interpret stake transaction outcome
        
        Bitstring format:
        - Bits [0-3]: Stake validity (>= 8 total bits = APPROVED)
        - Bits [4-7]: Validator assignment (256 possible validators)
        """
        if len(bitstring) < 8:
            bitstring = bitstring.zfill(8)
        
        validity_bits = bitstring[:4]
        validator_bits = bitstring[4:8]
        
        # Count total set bits
        total_ones = bitstring.count('1')
        
        if total_ones >= 4:  # At least half bits set
            outcome = CollapseOutcome.APPROVED
            reason = f"Stake approved: {total_ones}/8 bits set"
        else:
            outcome = CollapseOutcome.REJECTED
            reason = f"Stake rejected: {total_ones}/8 bits set (need >= 4)"
        
        # Determine validator from validator bits
        validator_index = int(validator_bits, 2)  # 0-15
        
        interpretation = {
            'outcome': outcome,
            'validity_bits': validity_bits,
            'validator_bits': validator_bits,
            'validator_index': validator_index,
            'total_set_bits': total_ones,
            'reason': reason
        }
        
        logger.debug(f"Stake interpretation: {interpretation}")
        return interpretation
    
    def interpret_mint_outcome(self, bitstring: str, oracle_data: Dict) -> Dict:
        """
        Interpret mint transaction outcome
        
        Bitstring format:
        - All bits: Amount encoding (sum of set bits)
        """
        if len(bitstring) < 8:
            bitstring = bitstring.zfill(8)
        
        # Sum of set bits determines minted amount
        amount_bits = bitstring.count('1')
        minted_amount = amount_bits * 1000000  # Scale to QTCL wei
        
        # Mint is approved if at least 1 bit is set
        if amount_bits > 0:
            outcome = CollapseOutcome.APPROVED
            reason = f"Mint approved: {amount_bits} bits set"
        else:
            outcome = CollapseOutcome.REJECTED
            reason = "Mint rejected: no bits set"
        
        interpretation = {
            'outcome': outcome,
            'amount_bits': amount_bits,
            'minted_amount': minted_amount,
            'bitstring': bitstring,
            'reason': reason
        }
        
        logger.debug(f"Mint interpretation: {interpretation}")
        return interpretation
    
    def interpret_burn_outcome(self, bitstring: str, oracle_data: Dict) -> Dict:
        """
        Interpret burn transaction outcome
        
        Bitstring format:
        - All bits: Burn amount encoding
        """
        if len(bitstring) < 8:
            bitstring = bitstring.zfill(8)
        
        # Similar to mint but for burning
        amount_bits = bitstring.count('1')
        burned_amount = amount_bits * 1000000
        
        if amount_bits > 0:
            outcome = CollapseOutcome.APPROVED
            reason = f"Burn approved: {amount_bits} bits set"
        else:
            outcome = CollapseOutcome.REJECTED
            reason = "Burn rejected: no bits set"
        
        interpretation = {
            'outcome': outcome,
            'amount_bits': amount_bits,
            'burned_amount': burned_amount,
            'bitstring': bitstring,
            'reason': reason
        }
        
        logger.debug(f"Burn interpretation: {interpretation}")
        return interpretation
    
    def interpret_contract_call_outcome(self, bitstring: str, oracle_data: Dict) -> Dict:
        """
        Interpret contract call outcome
        
        Bitstring format:
        - Bits [0-3]: Execution success
        - Bits [4-7]: Return value (0-255)
        """
        if len(bitstring) < 8:
            bitstring = bitstring.zfill(8)
        
        success_bits = bitstring[:4]
        return_bits = bitstring[4:8]
        
        success_count = success_bits.count('1')
        
        if success_count >= 3:
            outcome = CollapseOutcome.APPROVED
            reason = f"Contract call successful: {success_count}/4 bits"
        else:
            outcome = CollapseOutcome.REJECTED
            reason = f"Contract call failed: {success_count}/4 bits"
        
        return_value = int(return_bits, 2)
        
        interpretation = {
            'outcome': outcome,
            'success_bits': success_bits,
            'return_bits': return_bits,
            'return_value': return_value,
            'success_count': success_count,
            'reason': reason
        }
        
        logger.debug(f"Contract call interpretation: {interpretation}")
        return interpretation


class CausalityValidator:
    """
    Validate collapsed outcomes against causality constraints
    """
    
    def __init__(self, supabase_client: Client, db_pool: ThreadedConnectionPool):
        self.supabase = supabase_client
        self.db_pool = db_pool
        logger.info("CausalityValidator initialized")
    
    def validate_causality(self,
                          collapsed_outcome: Dict,
                          tx: Dict,
                          hyperbolic_distance: float) -> Tuple[bool, str]:
        """
        Validate causality constraints
        
        Args:
            collapsed_outcome: Interpreted outcome
            tx: Transaction dictionary
            hyperbolic_distance: Hyperbolic distance from quantum circuit
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not CAUSALITY_CHECK_ENABLED:
            return True, "Causality checking disabled"
        
        # Check 1: Balance consistency
        if tx['tx_type'] == 'transfer':
            balance_valid, balance_msg = self.check_balance_consistency(
                tx['from_user_id'],
                tx['amount'],
                collapsed_outcome
            )
            if not balance_valid:
                return False, f"Balance check failed: {balance_msg}"
        
        # Check 2: Dependency graph
        dependency_valid, dependency_msg = self.check_dependency_graph(
            tx['id'],
            tx
        )
        if not dependency_valid:
            return False, f"Dependency check failed: {dependency_msg}"
        
        # Check 3: Geodesic constraint
        if hyperbolic_distance > HYPERBOLIC_DISTANCE_THRESHOLD:
            geodesic_valid, geodesic_msg = self.validate_geodesic_constraint(
                tx.get('from_position', [0.0, 0.0]),
                tx.get('to_position', [0.0, 0.0]),
                hyperbolic_distance
            )
            if not geodesic_valid:
                return False, f"Geodesic check failed: {geodesic_msg}"
        
        # Check 4: Nonce sequence
        nonce_valid, nonce_msg = self.check_nonce_sequence(
            tx['from_user_id'],
            tx.get('nonce', 0)
        )
        if not nonce_valid:
            return False, f"Nonce check failed: {nonce_msg}"
        
        # Check 5: Double-spend detection
        double_spend_valid, double_spend_msg = self.check_double_spend(tx)
        if not double_spend_valid:
            return False, f"Double-spend detected: {double_spend_msg}"
        
        return True, "All causality constraints satisfied"
    
    def check_balance_consistency(self,
                                  sender_id: str,
                                  amount: int,
                                  collapsed_outcome: Dict) -> Tuple[bool, str]:
        """Check if sender has sufficient balance"""
        if not BALANCE_VALIDATION_ENABLED:
            return True, "Balance validation disabled"
        
        try:
            # Get sender balance
            user_result = self.supabase.table('users').select('balance').eq('user_id', sender_id).execute()
            
            if not user_result.data:
                return False, f"Sender not found: {sender_id}"
            
            balance = user_result.data[0]['balance']
            
            # Check if outcome is approved and requires balance
            if collapsed_outcome.get('outcome') == CollapseOutcome.APPROVED:
                if balance < amount:
                    return False, f"Insufficient balance: {balance} < {amount}"
            
            return True, f"Balance sufficient: {balance} >= {amount}"
            
        except Exception as e:
            logger.error(f"Balance check failed: {e}")
            return False, f"Balance check error: {e}"
    
    def check_dependency_graph(self,
                               tx_id: str,
                               tx: Dict) -> Tuple[bool, str]:
        """Check transaction dependency graph for cycles"""
        try:
            # For now, simple check - no circular dependencies
            # In production, would build full DAG and check for cycles
            
            depends_on = tx.get('depends_on_tx', [])
            if not depends_on:
                return True, "No dependencies"
            
            # Check if dependencies are finalized
            for dep_tx_id in depends_on:
                dep_result = self.supabase.table('transactions').select('status').eq('id', dep_tx_id).execute()
                
                if not dep_result.data:
                    return False, f"Dependency not found: {dep_tx_id}"
                
                dep_status = dep_result.data[0]['status']
                if dep_status not in [TransactionStatus.FINALIZED.value, TransactionStatus.COLLAPSED.value]:
                    return False, f"Dependency not finalized: {dep_tx_id} (status: {dep_status})"
            
            return True, "All dependencies satisfied"
            
        except Exception as e:
            logger.error(f"Dependency check failed: {e}")
            return False, f"Dependency check error: {e}"
    
    def validate_geodesic_constraint(self,
                                    from_pos: List[float],
                                    to_pos: List[float],
                                    outcome_distance: float) -> Tuple[bool, str]:
        """Validate hyperbolic geodesic distance constraint"""
        try:
            # Calculate expected geodesic distance in Poincaré disk
            z1 = complex(from_pos[0], from_pos[1])
            z2 = complex(to_pos[0], to_pos[1])
            
            # Hyperbolic distance formula
            numerator = abs(z1 - z2) ** 2
            denominator = (1 - abs(z1) ** 2) * (1 - abs(z2) ** 2)
            
            if denominator <= 0:
                return False, "Invalid positions (outside Poincaré disk)"
            
            expected_distance = math.acosh(1 + 2 * numerator / denominator)
            
            # Check if outcome distance is consistent with geodesic
            if outcome_distance > expected_distance * 1.5:  # 50% tolerance
                return False, f"Geodesic violation: {outcome_distance:.3f} > {expected_distance:.3f}"
            
            return True, f"Geodesic valid: {outcome_distance:.3f} <= {expected_distance:.3f}"
            
        except (ValueError, ZeroDivisionError) as e:
            logger.error(f"Geodesic calculation failed: {e}")
            return False, f"Geodesic calculation error: {e}"
    
    def check_nonce_sequence(self,
                            user_id: str,
                            tx_nonce: int) -> Tuple[bool, str]:
        """Check if nonce sequence is valid (no gaps)"""
        try:
            # Get latest nonce for user
            nonce_result = self.supabase.table('users').select('nonce').eq('user_id', user_id).execute()
            
            if not nonce_result.data:
                return False, f"User not found: {user_id}"
            
            current_nonce = nonce_result.data[0]['nonce']
            
            # Check if transaction nonce is next in sequence
            if tx_nonce != current_nonce + 1:
                return False, f"Nonce mismatch: expected {current_nonce + 1}, got {tx_nonce}"
            
            return True, f"Nonce valid: {tx_nonce}"
            
        except Exception as e:
            logger.error(f"Nonce check failed: {e}")
            return False, f"Nonce check error: {e}"
    
    def check_double_spend(self, tx: Dict) -> Tuple[bool, str]:
        """Check for double-spend attempts"""
        try:
            # Query for conflicting transactions with same nonce
            conflicts = self.supabase.table('transactions').select('id, status').eq(
                'from_user_id', tx['from_user_id']
            ).eq(
                'nonce', tx.get('nonce', 0)
            ).neq(
                'id', tx['id']
            ).execute()
            
            # Filter for finalized conflicts
            finalized_conflicts = [
                c for c in conflicts.data
                if c['status'] in [TransactionStatus.FINALIZED.value, TransactionStatus.COLLAPSED.value]
            ]
            
            if finalized_conflicts:
                return False, f"Double-spend detected: {len(finalized_conflicts)} conflicting transactions"
            
            return True, "No double-spend detected"
            
        except Exception as e:
            logger.error(f"Double-spend check failed: {e}")
            return False, f"Double-spend check error: {e}"

# ============================================================================
# MODULE 4: ORACLE EVENT LOOP
# ============================================================================

class OracleEventLoop:
    """
    Main event loop for oracle operations - continuously polls and processes oracle events
    """
    
    def __init__(self,
                 supabase_client: Client,
                 db_pool: ThreadedConnectionPool,
                 time_oracle: TimeOracle,
                 price_oracle: PriceOracle,
                 event_oracle: EventOracle,
                 random_oracle: RandomOracle,
                 entropy_oracle: EntropyOracle):
        
        self.supabase = supabase_client
        self.db_pool = db_pool
        self.time_oracle = time_oracle
        self.price_oracle = price_oracle
        self.event_oracle = event_oracle
        self.random_oracle = random_oracle
        self.entropy_oracle = entropy_oracle
        
        # Initialize components
        self.event_queue = OracleEventQueue()
        self.superposition_collapse = SuperpositionCollapse(supabase_client, db_pool)
        self.event_dispatcher = OracleEventDispatcher(
            self._collapse_handler,
            supabase_client
        )
        self.event_logger = OracleEventLogger(supabase_client)
        
        # State tracking
        self.running = False
        self.loop_count = 0
        self.total_events_processed = 0
        self.last_statistics_report = time.time()
        self.lock = threading.Lock()
        
        # Thread pool for concurrent collapse operations
        self.executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_COLLAPSES)
        
        logger.info("OracleEventLoop initialized")
    
    def run(self):
        """
        Main event loop - runs continuously until stopped
        """
        self.running = True
        logger.info("OracleEventLoop starting...")
        
        try:
            while self.running:
                loop_start = time.time()
                
                # 1. Process pending transactions and trigger oracle events
                self.process_pending_transactions()
                
                # 2. Process oracle event queue
                self.dispatch_oracle_events()
                
                # 3. Manage superposition state lifecycles
                self.manage_superposition_states()
                
                # 4. Clean up expired events
                if self.loop_count % 100 == 0:  # Every 100 iterations
                    self.event_queue.clear_expired_events()
                
                # 5. Report statistics periodically
                if time.time() - self.last_statistics_report > 60:  # Every minute
                    self.report_statistics()
                    self.last_statistics_report = time.time()
                
                with self.lock:
                    self.loop_count += 1
                
                # Sleep to prevent CPU spinning
                loop_duration = time.time() - loop_start
                sleep_time = max(0.0, (ORACLE_LOOP_SLEEP_MS / 1000.0) - loop_duration)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("OracleEventLoop interrupted by user")
        except Exception as e:
            logger.error(f"OracleEventLoop error: {e}", exc_info=True)
        finally:
            self.stop()
    
    def stop(self):
        """Stop the oracle event loop"""
        logger.info("OracleEventLoop stopping...")
        self.running = False
        self.executor.shutdown(wait=True)
        logger.info("OracleEventLoop stopped")
    
    def process_pending_transactions(self):
        """
        Query for pending transactions in superposition and trigger oracle events
        """
        try:
            # Query transactions in superposition state
            result = self.supabase.table('transactions').select(
                'id, tx_type, from_user_id, to_user_id, amount, created_at, quantum_state_hash, entropy_score'
            ).eq(
                'status', TransactionStatus.SUPERPOSITION.value
            ).limit(100).execute()
            
            if not result.data:
                return
            
            logger.debug(f"Processing {len(result.data)} pending transactions")
            
            for tx in result.data:
                self.trigger_oracle_events(tx)
                
        except Exception as e:
            logger.error(f"Failed to process pending transactions: {e}")
    
    def trigger_oracle_events(self, tx: Dict):
        """
        Trigger all applicable oracle events for a transaction
        
        Args:
            tx: Transaction dictionary
        """
        tx_id = tx['id']
        created_at = datetime.fromisoformat(tx['created_at'].replace('Z', '+00:00'))
        
        triggered_events = []
        
        # 1. ENTROPY ORACLE - Always check first (highest priority)
        if tx.get('entropy_score') is not None:
            measurements = {
                'entropy_percent': tx['entropy_score'],
                'entropy_bits': 0,  # Would need to fetch from quantum_measurements
                'total_measurements': 0
            }
            
            entropy_event = self.entropy_oracle.trigger_entropy_oracle_event(tx_id, measurements)
            if entropy_event:
                triggered_events.append(entropy_event)
        
        # 2. TIME ORACLE - Check transaction age
        time_event = self.time_oracle.trigger_time_oracle_event(tx_id, created_at)
        if time_event:
            triggered_events.append(time_event)
        
        # 3. PRICE ORACLE - For transfer transactions
        if tx['tx_type'] == 'transfer':
            price_event = self.price_oracle.generate_price_oracle_event(
                tx_id,
                tx['amount']
            )
            if price_event:
                triggered_events.append(price_event)
        
        # 4. RANDOM ORACLE - Generate randomness for MEV protection
        random_event = self.random_oracle.generate_random_oracle_event(tx_id)
        if random_event:
            triggered_events.append(random_event)
        
        # 5. EVENT ORACLE - Check for external events (if configured)
        # This would require listening to specific contract events
        # Skipped for now unless specific event matching is configured
        
        # Enqueue all triggered events
        for event in triggered_events:
            success = self.event_queue.enqueue_event(event)
            if success:
                self.event_logger.log_event(event)
            else:
                logger.warning(f"Failed to enqueue event: {event.oracle_id}")
    
    def dispatch_oracle_events(self):
        """
        Dispatch oracle events from queue to collapse handlers
        """
        # Dequeue batch of events
        events = self.event_queue.dequeue_events(ORACLE_EVENT_BATCH_SIZE)
        
        if not events:
            return
        
        logger.debug(f"Dispatching {len(events)} oracle events")
        
        # Dispatch events concurrently
        futures = []
        for event in events:
            future = self.executor.submit(self._dispatch_single_event, event)
            futures.append(future)
        
        # Wait for all dispatches to complete
        for future in as_completed(futures):
            try:
                result = future.result()
                with self.lock:
                    self.total_events_processed += 1
            except Exception as e:
                logger.error(f"Event dispatch failed: {e}")
    
    def _dispatch_single_event(self, event: OracleEvent) -> bool:
        """Dispatch single oracle event"""
        try:
            success = self.event_dispatcher.dispatch_event(event)
            
            if success:
                self.event_logger.log_dispatch(event, [event.tx_id])
            else:
                self.event_logger.log_error(
                    f"Dispatch failed for {event.oracle_id}",
                    {'event': event.to_dict()}
                )
            
            return success
            
        except Exception as e:
            logger.error(f"Error dispatching event {event.oracle_id}: {e}")
            self.event_logger.log_error(str(e), {'event': event.to_dict()})
            return False
    
    def _collapse_handler(self, tx_id: str, oracle_event: OracleEvent):
        """
        Collapse handler callback - invoked when oracle event triggers collapse
        
        Args:
            tx_id: Transaction ID
            oracle_event: OracleEvent that triggered collapse
        """
        try:
            logger.info(f"Collapse handler invoked for {tx_id} by oracle {oracle_event.oracle_id}")
            
            # Fetch quantum measurements
            measurements_result = self.supabase.table('quantum_measurements').select(
                '*'
            ).eq('transaction_id', tx_id).execute()
            
            if not measurements_result.data:
                logger.error(f"No quantum measurements found for {tx_id}")
                return
            
            measurements = measurements_result.data[0]
            
            # Perform collapse
            collapse_result = self.superposition_collapse.collapse_transaction(
                tx_id,
                oracle_event.oracle_data,
                measurements,
                measurement_basis=0  # Computational basis
            )
            
            if collapse_result:
                logger.info(f"Collapse successful for {tx_id}: {collapse_result.outcome.value}")
                self.event_logger.log_collapse(tx_id, collapse_result.to_dict())
                
                # Mark oracle event as collapse triggered
                oracle_event.collapse_triggered = True
            else:
                logger.error(f"Collapse failed for {tx_id}")
                
        except Exception as e:
            logger.error(f"Collapse handler error for {tx_id}: {e}", exc_info=True)
            self.event_logger.log_error(
                f"Collapse handler error: {e}",
                {'tx_id': tx_id, 'oracle_event': oracle_event.to_dict()}
            )
    
    def manage_superposition_states(self):
        """
        Manage lifecycle of superposition states - detect timeouts and failures
        """
        try:
            # Query long-running superposition transactions
            timeout_threshold = datetime.utcnow() - timedelta(seconds=300)  # 5 minutes
            
            result = self.supabase.table('transactions').select(
                'id, created_at'
            ).eq(
                'status', TransactionStatus.SUPERPOSITION.value
            ).lt(
                'created_at', timeout_threshold.isoformat()
            ).execute()
            
            if result.data:
                logger.warning(f"Found {len(result.data)} timed-out superposition transactions")
                
                for tx in result.data:
                    self._handle_superposition_timeout(tx['id'])
                    
        except Exception as e:
            logger.error(f"Failed to manage superposition states: {e}")
    
    def _handle_superposition_timeout(self, tx_id: str):
        """Handle superposition timeout - force collapse or reject"""
        try:
            logger.warning(f"Handling superposition timeout for {tx_id}")
            
            # Force time oracle trigger
            time_event = OracleEvent(
                oracle_id=f"timeout_{tx_id}_{int(time.time())}",
                oracle_type=OracleType.TIME,
                tx_id=tx_id,
                oracle_data={
                    'trigger_time': int(time.time()),
                    'timeout': True,
                    'reason': 'Superposition timeout'
                },
                proof=hashlib.sha256(f"timeout:{tx_id}".encode()).hexdigest(),
                timestamp=int(time.time()),
                priority=10  # Highest priority
            )
            
            self.event_queue.enqueue_event(time_event)
            
        except Exception as e:
            logger.error(f"Failed to handle timeout for {tx_id}: {e}")
    
    def report_statistics(self):
        """Report comprehensive oracle statistics"""
        try:
            stats = {
                'loop_count': self.loop_count,
                'total_events_processed': self.total_events_processed,
                'queue_stats': self.event_queue.get_statistics(),
                'time_oracle': self.time_oracle.get_statistics(),
                'price_oracle': self.price_oracle.get_statistics(),
                'event_oracle': self.event_oracle.get_statistics(),
                'random_oracle': self.random_oracle.get_statistics(),
                'entropy_oracle': self.entropy_oracle.get_statistics(),
                'collapse': self.superposition_collapse.get_statistics(),
                'dispatcher': self.event_dispatcher.get_statistics(),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Oracle Statistics: {json.dumps(stats, indent=2)}")
            
            # Store statistics in database
            self.supabase.table('oracle_statistics').insert({
                'statistics': stats,
                'timestamp': stats['timestamp']
            }).execute()
            
        except Exception as e:
            logger.error(f"Failed to report statistics: {e}")
    
    def get_status(self) -> Dict:
        """Get current status of oracle event loop"""
        with self.lock:
            return {
                'running': self.running,
                'loop_count': self.loop_count,
                'total_events_processed': self.total_events_processed,
                'queue_size': self.event_queue.get_queue_size(),
                'uptime_seconds': time.time() - (self.last_statistics_report if self.last_statistics_report else time.time())
            }


class OracleEventLogger:
    """
    Comprehensive logging and audit trail for oracle events
    """
    
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
        self.log_count = 0
        self.lock = threading.Lock()
        
        logger.info("OracleEventLogger initialized")
    
    def log_event(self, event: OracleEvent):
        """
        Log oracle event to database
        
        Args:
            event: OracleEvent to log
        """
        try:
            log_entry = {
                'oracle_id': event.oracle_id,
                'oracle_type': event.oracle_type.value,
                'tx_id': event.tx_id,
                'event_data': event.oracle_data,
                'proof': event.proof,
                'priority': event.priority,
                'timestamp': datetime.fromtimestamp(event.timestamp).isoformat(),
                'dispatched': event.dispatched,
                'collapse_triggered': event.collapse_triggered
            }
            
            self.supabase.table('oracle_event_log').insert(log_entry).execute()
            
            with self.lock:
                self.log_count += 1
            
            logger.debug(f"Logged oracle event: {event.oracle_id}")
            
        except Exception as e:
            logger.error(f"Failed to log oracle event: {e}")
    
    def log_dispatch(self, event: OracleEvent, tx_ids: List[str]):
        """
        Log oracle event dispatch
        
        Args:
            event: OracleEvent that was dispatched
            tx_ids: List of transaction IDs affected
        """
        try:
            dispatch_entry = {
                'oracle_id': event.oracle_id,
                'oracle_type': event.oracle_type.value,
                'tx_ids': tx_ids,
                'dispatch_count': len(tx_ids),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self.supabase.table('oracle_dispatch_log').insert(dispatch_entry).execute()
            
            logger.debug(f"Logged dispatch for {event.oracle_id} -> {len(tx_ids)} transactions")
            
        except Exception as e:
            logger.error(f"Failed to log dispatch: {e}")
    
    def log_collapse(self, tx_id: str, outcome: Dict):
        """
        Log collapse event
        
        Args:
            tx_id: Transaction ID
            outcome: Collapse outcome dictionary
        """
        try:
            collapse_entry = {
                'transaction_id': tx_id,
                'outcome': outcome,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self.supabase.table('collapse_log').insert(collapse_entry).execute()
            
            logger.debug(f"Logged collapse for {tx_id}")
            
        except Exception as e:
            logger.error(f"Failed to log collapse: {e}")
    
    def log_error(self, error: str, context: Dict):
        """
        Log error event
        
        Args:
            error: Error message
            context: Error context dictionary
        """
        try:
            error_entry = {
                'error_message': error,
                'context': context,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self.supabase.table('oracle_error_log').insert(error_entry).execute()
            
            logger.debug(f"Logged error: {error}")
            
        except Exception as e:
            logger.error(f"Failed to log error: {e}")
    
    def generate_audit_report(self, start_time: datetime, end_time: datetime) -> str:
        """
        Generate comprehensive audit report for time range
        
        Args:
            start_time: Report start time
            end_time: Report end time
            
        Returns:
            Audit report string
        """
        try:
            # Query all oracle events in time range
            events_result = self.supabase.table('oracle_event_log').select(
                '*'
            ).gte(
                'timestamp', start_time.isoformat()
            ).lte(
                'timestamp', end_time.isoformat()
            ).execute()
            
            # Query all dispatches in time range
            dispatches_result = self.supabase.table('oracle_dispatch_log').select(
                '*'
            ).gte(
                'timestamp', start_time.isoformat()
            ).lte(
                'timestamp', end_time.isoformat()
            ).execute()
            
            # Query all collapses in time range
            collapses_result = self.supabase.table('collapse_log').select(
                '*'
            ).gte(
                'timestamp', start_time.isoformat()
            ).lte(
                'timestamp', end_time.isoformat()
            ).execute()
            
            # Generate report
            report = f"""
ORACLE AUDIT REPORT
===================
Time Range: {start_time.isoformat()} to {end_time.isoformat()}

SUMMARY
-------
Total Oracle Events: {len(events_result.data)}
Total Dispatches: {len(dispatches_result.data)}
Total Collapses: {len(collapses_result.data)}

ORACLE EVENTS BY TYPE
---------------------
"""
            
            # Count events by type
            event_counts = defaultdict(int)
            for event in events_result.data:
                event_counts[event['oracle_type']] += 1
            
            for oracle_type, count in sorted(event_counts.items()):
                report += f"{oracle_type}: {count}\n"
            
            report += f"""
DISPATCH STATISTICS
-------------------
Total Transactions Affected: {sum(d['dispatch_count'] for d in dispatches_result.data)}
Average Dispatches per Event: {len(dispatches_result.data) / len(events_result.data) if events_result.data else 0:.2f}

COLLAPSE OUTCOMES
-----------------
"""
            
            # Count collapse outcomes
            outcome_counts = defaultdict(int)
            for collapse in collapses_result.data:
                outcome = collapse['outcome'].get('outcome', 'unknown')
                outcome_counts[outcome] += 1
            
            for outcome, count in sorted(outcome_counts.items()):
                report += f"{outcome}: {count}\n"
            
            report += f"""
Report Generated: {datetime.utcnow().isoformat()}
"""
            
            logger.info("Generated audit report")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate audit report: {e}")
            return f"Error generating audit report: {e}"
    
    def get_statistics(self) -> Dict:
        """Get logger statistics"""
        with self.lock:
            return {
                'total_logs': self.log_count
            }


# ============================================================================
# MODULE 5: STATISTICS & REPORTING
# ============================================================================

class OracleStatistics:
    """
    Track and report oracle performance metrics
    """
    
    def __init__(self):
        self.metrics = defaultdict(lambda: {
            'total_events': 0,
            'successful_events': 0,
            'failed_events': 0,
            'total_latency_ms': 0.0,
            'min_latency_ms': float('inf'),
            'max_latency_ms': 0.0,
            'events_per_minute': deque(maxlen=60)  # Last 60 minutes
        })
        
        self.lock = threading.Lock()
        logger.info("OracleStatistics initialized")
    
    def record_event(self, oracle_type: str, success: bool, latency_ms: float):
        """
        Record oracle event metrics
        
        Args:
            oracle_type: Type of oracle (time, price, etc.)
            success: Whether event was successful
            latency_ms: Event latency in milliseconds
        """
        with self.lock:
            metrics = self.metrics[oracle_type]
            
            metrics['total_events'] += 1
            if success:
                metrics['successful_events'] += 1
            else:
                metrics['failed_events'] += 1
            
            metrics['total_latency_ms'] += latency_ms
            metrics['min_latency_ms'] = min(metrics['min_latency_ms'], latency_ms)
            metrics['max_latency_ms'] = max(metrics['max_latency_ms'], latency_ms)
            
            # Track events per minute
            current_minute = int(time.time() / 60)
            metrics['events_per_minute'].append((current_minute, 1))
    
    def get_statistics(self, time_window: Optional[timedelta] = None) -> Dict:
        """
        Get statistics for all oracle types
        
        Args:
            time_window: Optional time window for filtering
            
        Returns:
            Statistics dictionary
        """
        with self.lock:
            stats = {}
            
            for oracle_type, metrics in self.metrics.items():
                total = metrics['total_events']
                
                stats[oracle_type] = {
                    'total_events': total,
                    'successful_events': metrics['successful_events'],
                    'failed_events': metrics['failed_events'],
                    'success_rate_percent': (metrics['successful_events'] / total * 100.0) if total > 0 else 0.0,
                    'average_latency_ms': (metrics['total_latency_ms'] / total) if total > 0 else 0.0,
                    'min_latency_ms': metrics['min_latency_ms'] if metrics['min_latency_ms'] != float('inf') else 0.0,
                    'max_latency_ms': metrics['max_latency_ms'],
                    'events_per_minute': self._calculate_events_per_minute(metrics['events_per_minute'])
                }
            
            return stats
    
    def _calculate_events_per_minute(self, events_deque: deque) -> float:
        """Calculate average events per minute from deque"""
        if not events_deque:
            return 0.0
        
        # Sum events in last N minutes
        total_events = sum(count for _, count in events_deque)
        minutes = len(set(minute for minute, _ in events_deque))
        
        return total_events / minutes if minutes > 0 else 0.0
    
    def get_oracle_performance(self, oracle_type: str) -> Dict:
        """
        Get detailed performance metrics for specific oracle type
        
        Args:
            oracle_type: Oracle type to query
            
        Returns:
            Performance metrics dictionary
        """
        with self.lock:
            if oracle_type not in self.metrics:
                return {}
            
            metrics = self.metrics[oracle_type]
            total = metrics['total_events']
            
            return {
                'oracle_type': oracle_type,
                'total_events': total,
                'successful_events': metrics['successful_events'],
                'failed_events': metrics['failed_events'],
                'success_rate_percent': (metrics['successful_events'] / total * 100.0) if total > 0 else 0.0,
                'average_latency_ms': (metrics['total_latency_ms'] / total) if total > 0 else 0.0,
                'min_latency_ms': metrics['min_latency_ms'] if metrics['min_latency_ms'] != float('inf') else 0.0,
                'max_latency_ms': metrics['max_latency_ms'],
                'current_events_per_minute': self._calculate_events_per_minute(metrics['events_per_minute'])
            }
    
    def reset_statistics(self):
        """Reset all statistics (for testing)"""
        with self.lock:
            self.metrics.clear()
            logger.info("Statistics reset")


# ============================================================================
# DATABASE HELPER FUNCTIONS
# ============================================================================

def get_db_connection(db_pool: ThreadedConnectionPool):
    """Get database connection from pool"""
    return db_pool.getconn()

def return_db_connection(db_pool: ThreadedConnectionPool, conn):
    """Return database connection to pool"""
    db_pool.putconn(conn)

def execute_query(db_pool: ThreadedConnectionPool, query: str, params: tuple = None) -> List[Dict]:
    """
    Execute SQL query and return results
    
    Args:
        db_pool: Database connection pool
        query: SQL query string
        params: Query parameters
        
    Returns:
        List of result dictionaries
    """
    conn = get_db_connection(db_pool)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, params)
            
            if cursor.description:
                results = cursor.fetchall()
                return [dict(row) for row in results]
            else:
                conn.commit()
                return []
    finally:
        return_db_connection(db_pool, conn)

def execute_batch(db_pool: ThreadedConnectionPool, query: str, params_list: List[tuple]):
    """
    Execute batch query
    
    Args:
        db_pool: Database connection pool
        query: SQL query string
        params_list: List of parameter tuples
    """
    conn = get_db_connection(db_pool)
    try:
        with conn.cursor() as cursor:
            execute_values(cursor, query, params_list)
            conn.commit()
    finally:
        return_db_connection(db_pool, conn)


# ============================================================================
# MAIN ORACLE ENGINE
# ============================================================================

class OracleEngine:
    """
    Main Oracle Engine - orchestrates all oracle operations
    """
    
    def __init__(self):
        logger.info("Initializing OracleEngine...")
        
        # Initialize Supabase client
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
        
        self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Initialize database connection pool
        if not DATABASE_URL:
            raise ValueError("DATABASE_URL must be set")
        
        self.db_pool = ThreadedConnectionPool(
            minconn=5,
            maxconn=20,
            dsn=DATABASE_URL
        )
        
        # Initialize all oracles
        self.time_oracle = TimeOracle()
        self.price_oracle = PriceOracle()
        self.event_oracle = EventOracle()
        self.random_oracle = RandomOracle()
        self.entropy_oracle = EntropyOracle()
        
        # Initialize event loop
        self.event_loop = OracleEventLoop(
            self.supabase,
            self.db_pool,
            self.time_oracle,
            self.price_oracle,
            self.event_oracle,
            self.random_oracle,
            self.entropy_oracle
        )
        
        # Initialize statistics
        self.statistics = OracleStatistics()
        
        logger.info("OracleEngine initialized successfully")
    
    def start(self):
        """Start the oracle engine"""
        logger.info("Starting OracleEngine...")
        
        try:
            # Verify database connectivity
            self._verify_database()
            
            # Start event loop
            self.event_loop.run()
            
        except Exception as e:
            logger.error(f"Failed to start OracleEngine: {e}", exc_info=True)
            raise
    
    def stop(self):
        """Stop the oracle engine"""
        logger.info("Stopping OracleEngine...")
        self.event_loop.stop()
        self.db_pool.closeall()
        logger.info("OracleEngine stopped")
    
    def _verify_database(self):
        """Verify database connectivity and required tables"""
        try:
            # Test Supabase connection
            result = self.supabase.table('transactions').select('id').limit(1).execute()
            logger.info(f"Supabase connection verified: {len(result.data)} records")
            
            # Test PostgreSQL connection
            conn = get_db_connection(self.db_pool)
            try:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT version();")
                    version = cursor.fetchone()[0]
                    logger.info(f"PostgreSQL connection verified: {version}")
            finally:
                return_db_connection(self.db_pool, conn)
            
            # Verify required tables exist
            required_tables = [
                'transactions',
                'quantum_measurements',
                'oracle_events',
                'oracle_event_log',
                'oracle_dispatch_log',
                'superposition_collapses',
                'users',
                'pseudoqubits'
            ]
            
            for table in required_tables:
                result = self.supabase.table(table).select('*').limit(1).execute()
                logger.debug(f"Table '{table}' verified")
            
            logger.info("All required database tables verified")
            
        except Exception as e:
            logger.error(f"Database verification failed: {e}")
            raise
    
    def get_status(self) -> Dict:
        """Get oracle engine status"""
        return {
            'event_loop': self.event_loop.get_status(),
            'time_oracle': self.time_oracle.get_statistics(),
            'price_oracle': self.price_oracle.get_statistics(),
            'event_oracle': self.event_oracle.get_statistics(),
            'random_oracle': self.random_oracle.get_statistics(),
            'entropy_oracle': self.entropy_oracle.get_statistics(),
            'statistics': self.statistics.get_statistics()
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point for oracle_engine.py"""
    logger.info("="*80)
    logger.info("QTCL ORACLE ENGINE v1.0")
    logger.info("="*80)
    
    try:
        # Create and start oracle engine
        engine = OracleEngine()
        engine.start()
        
    except KeyboardInterrupt:
        logger.info("Oracle engine interrupted by user")
    except Exception as e:
        logger.error(f"Oracle engine failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if 'engine' in locals():
            engine.stop()
        logger.info("Oracle engine shutdown complete")


if __name__ == "__main__":
    main()


# ============================================================================
# TESTING UTILITIES
# ============================================================================

def test_oracle_engine():
    """Test oracle engine functionality"""
    logger.info("Running oracle engine tests...")
    
    # Test 1: Oracle initialization
    logger.info("Test 1: Oracle initialization")
    time_oracle = TimeOracle(interval_seconds=5)
    assert time_oracle.interval_seconds == 5
    logger.info("✓ TimeOracle initialized")
    
    price_oracle = PriceOracle()
    logger.info("✓ PriceOracle initialized")
    
    random_oracle = RandomOracle()
    logger.info("✓ RandomOracle initialized")
    
    entropy_oracle = EntropyOracle()
    logger.info("✓ EntropyOracle initialized")
    
    # Test 2: Event queue operations
    logger.info("Test 2: Event queue")
    event_queue = OracleEventQueue(max_queue_size=100)
    
    test_event = OracleEvent(
        oracle_id="test_123",
        oracle_type=OracleType.TIME,
        tx_id="tx_test",
        oracle_data={'test': 'data'},
        proof="test_proof",
        timestamp=int(time.time()),
        priority=5
    )
    
    success = event_queue.enqueue_event(test_event)
    assert success, "Failed to enqueue event"
    assert event_queue.get_queue_size() == 1
    logger.info("✓ Event enqueued")
    
    events = event_queue.dequeue_events(10)
    assert len(events) == 1
    assert events[0].oracle_id == "test_123"
    logger.info("✓ Event dequeued")
    
    # Test 3: Collapse logic
    logger.info("Test 3: Collapse logic")
    interpreter = OutcomeInterpreter()
    
    # Test transfer interpretation
    transfer_outcome = interpreter.interpret_transfer_outcome("11110101", {})
    assert transfer_outcome['outcome'] in [CollapseOutcome.APPROVED, CollapseOutcome.PENDING, CollapseOutcome.REJECTED]
    logger.info(f"✓ Transfer interpreted: {transfer_outcome['outcome'].value}")
    
    # Test stake interpretation
    stake_outcome = interpreter.interpret_stake_outcome("11111111", {})
    assert stake_outcome['outcome'] == CollapseOutcome.APPROVED
    logger.info("✓ Stake interpreted")
    
    # Test 4: Random oracle VRF
    logger.info("Test 4: Random oracle VRF")
    random_value, vrf_proof = random_oracle.generate_random_value(12345)
    assert isinstance(random_value, int)
    assert isinstance(vrf_proof, str)
    logger.info(f"✓ VRF generated: value={random_value}, proof={vrf_proof[:32]}...")
    
    # Verify proof
    message = str(12345).encode('utf-8')
    is_valid = random_oracle.verify_random_proof(random_value, vrf_proof, message)
    assert is_valid, "VRF proof verification failed"
    logger.info("✓ VRF proof verified")
    
    logger.info("="*80)
    logger.info("ALL TESTS PASSED ✓")
    logger.info("="*80)


if __name__ == "__main__" and "--test" in sys.argv:
    test_oracle_engine()
