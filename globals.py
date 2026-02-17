#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                       ║
║               🌟 GLOBALS.PY - UNIFIED SYSTEM STATE MANAGEMENT 🌟                     ║
║                                                                                       ║
║              Single source of truth for entire QTCL application                       ║
║          All quantum, database, auth, terminal state centralized here                ║
║         Thread-safe, lazy-initialized, fully instrumented & monitorable              ║
║                                                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
"""

import threading
import logging
import time
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════════════
# ENUMS & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════════════

class SystemHealth(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"

class InitStatus(Enum):
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    FAILED = "failed"

# ═══════════════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class QuantumSubsystems:
    """Quantum systems state"""
    heartbeat: Optional[Any] = None
    lattice: Optional[Any] = None
    neural_network: Optional[Any] = None
    w_state_manager: Optional[Any] = None
    noise_bath: Optional[Any] = None
    entropy_ensemble: Optional[Any] = None
    quantum_coordinator: Optional[Any] = None
    
    def all_initialized(self) -> bool:
        """Check if all quantum systems are initialized"""
        return all([
            self.heartbeat is not None,
            self.lattice is not None,
            self.neural_network is not None,
            self.w_state_manager is not None,
            self.noise_bath is not None
        ])
    
    def get_health(self) -> Dict[str, Any]:
        """Get health of all quantum systems"""
        return {
            'heartbeat_running': self.heartbeat.running if self.heartbeat else False,
            'heartbeat_pulse_count': self.heartbeat.pulse_count if self.heartbeat else 0,
            'heartbeat_listeners': len(self.heartbeat.listeners) if self.heartbeat else 0,
            'heartbeat_errors': self.heartbeat.error_count if self.heartbeat else 0,
            'lattice_ready': self.lattice is not None,
            'neural_network_ready': self.neural_network is not None,
            'w_state_ready': self.w_state_manager is not None,
            'noise_bath_ready': self.noise_bath is not None,
        }

@dataclass
class DatabaseState:
    """Database connection & state"""
    pool: Optional[Any] = None
    connection_count: int = 0
    failed_connections: int = 0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None
    healthy: bool = False
    
    def mark_connection(self, success: bool = True):
        if success:
            self.connection_count += 1
            self.healthy = True
        else:
            self.failed_connections += 1
            self.last_error_time = datetime.utcnow()

@dataclass
class AuthenticationState:
    """Authentication systems state"""
    jwt_manager: Optional[Any] = None
    session_store: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    active_sessions: int = 0
    failed_attempts: Dict[str, List[datetime]] = field(default_factory=lambda: defaultdict(list))
    
    def get_failed_attempts(self, user_id: str, window_minutes: int = 15) -> int:
        """Get failed login attempts in time window"""
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        self.failed_attempts[user_id] = [
            t for t in self.failed_attempts[user_id] if t > cutoff
        ]
        return len(self.failed_attempts[user_id])

@dataclass
class TerminalState:
    """Command execution engine state"""
    engine: Optional[Any] = None
    command_registry: Dict[str, Callable] = field(default_factory=dict)
    executed_commands: int = 0
    failed_commands: int = 0
    last_command: Optional[str] = None
    last_command_time: Optional[datetime] = None

@dataclass
class ApplicationMetrics:
    """Application-wide metrics"""
    http_requests: int = 0
    http_errors: int = 0
    quantum_pulses: int = 0
    transactions_processed: int = 0
    blocks_created: int = 0
    api_calls: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    request_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    error_log: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add_request(self, duration_ms: float):
        """Track request timing"""
        self.http_requests += 1
        self.request_times.append({'time': datetime.utcnow(), 'duration_ms': duration_ms})
    
    def add_error(self, error: str):
        """Log error"""
        self.http_errors += 1
        self.error_log.append({'time': datetime.utcnow(), 'error': error})
    
    def get_stats(self) -> Dict[str, Any]:
        """Get metrics snapshot"""
        avg_request_time = 0
        if self.request_times:
            avg_request_time = sum(r['duration_ms'] for r in self.request_times) / len(self.request_times)
        
        return {
            'http_requests': self.http_requests,
            'http_errors': self.http_errors,
            'error_rate': self.http_errors / max(self.http_requests, 1),
            'quantum_pulses': self.quantum_pulses,
            'transactions': self.transactions_processed,
            'blocks': self.blocks_created,
            'avg_request_time_ms': avg_request_time,
            'unique_endpoints': len(self.api_calls),
        }

@dataclass
class RateLimiting:
    """Rate limiting state"""
    store: Dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=100)))
    limits: Dict[str, int] = field(default_factory=lambda: {'default': 100})  # requests per minute
    
    def check_limit(self, key: str, limit: Optional[int] = None) -> bool:
        """Check if request is within rate limit"""
        if limit is None:
            limit = self.limits.get(key, self.limits['default'])
        
        now = time.time()
        cutoff = now - 60  # 1 minute window
        
        # Clean old entries
        self.store[key] = deque(
            [t for t in self.store[key] if t > cutoff],
            maxlen=100
        )
        
        if len(self.store[key]) >= limit:
            return False
        
        self.store[key].append(now)
        return True

@dataclass
class GlobalState:
    """UNIFIED APPLICATION STATE - SINGLE SOURCE OF TRUTH"""
    
    # Initialization & status
    lock: threading.RLock = field(default_factory=threading.RLock)
    init_status: InitStatus = InitStatus.NOT_STARTED
    init_start_time: Optional[datetime] = None
    init_errors: List[str] = field(default_factory=list)
    
    # System components
    quantum: QuantumSubsystems = field(default_factory=QuantumSubsystems)
    database: DatabaseState = field(default_factory=DatabaseState)
    auth: AuthenticationState = field(default_factory=AuthenticationState)
    terminal: TerminalState = field(default_factory=TerminalState)
    
    # Application state
    metrics: ApplicationMetrics = field(default_factory=ApplicationMetrics)
    rate_limiting: RateLimiting = field(default_factory=RateLimiting)
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Session management
    sessions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    request_context: Dict[str, Any] = field(default_factory=dict)
    
    # System health
    startup_time: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    
    def get_system_health(self) -> Dict[str, Any]:
        """Comprehensive system health check"""
        with self.lock:
            quantum_health = self.quantum.get_health()
            
            is_healthy = (
                quantum_health.get('heartbeat_running', False) and
                quantum_health.get('heartbeat_listeners', 0) == 3 and
                quantum_health.get('heartbeat_errors', 0) < 10 and
                self.database.healthy
            )
            
            health_status = SystemHealth.HEALTHY if is_healthy else SystemHealth.DEGRADED
            if not quantum_health.get('heartbeat_running', False):
                health_status = SystemHealth.OFFLINE
            
            return {
                'status': health_status.value,
                'timestamp': datetime.utcnow().isoformat(),
                'uptime_seconds': (datetime.utcnow() - self.startup_time).total_seconds() if self.startup_time else 0,
                'quantum': quantum_health,
                'database': {
                    'healthy': self.database.healthy,
                    'connection_count': self.database.connection_count,
                    'failed_attempts': self.database.failed_connections,
                },
                'metrics': self.metrics.get_stats(),
                'init_status': self.init_status.value,
            }
    
    def snapshot(self) -> Dict[str, Any]:
        """Get complete state snapshot"""
        with self.lock:
            return {
                'init_status': self.init_status.value,
                'quantum': self.quantum.get_health(),
                'metrics': self.metrics.get_stats(),
                'sessions_active': len(self.sessions),
                'timestamp': datetime.utcnow().isoformat(),
            }

# ═══════════════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE & SINGLETON ACCESSOR
# ═══════════════════════════════════════════════════════════════════════════════════════

_GLOBAL_STATE: Optional[GlobalState] = None

def get_globals() -> GlobalState:
    """Get global state instance (lazy initialization)"""
    global _GLOBAL_STATE
    if _GLOBAL_STATE is None:
        _GLOBAL_STATE = GlobalState()
        _GLOBAL_STATE.startup_time = datetime.utcnow()
        logger.info("🌟 [Globals] Global state created")
    return _GLOBAL_STATE

# ═══════════════════════════════════════════════════════════════════════════════════════
# INITIALIZATION ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════════════

def initialize_globals() -> bool:
    """Initialize all global systems - CENTRAL INITIALIZATION POINT"""
    globals_inst = get_globals()
    
    with globals_inst.lock:
        if globals_inst.init_status == InitStatus.INITIALIZED:
            logger.info("[Globals] Already initialized")
            return True
        
        if globals_inst.init_status == InitStatus.INITIALIZING:
            logger.warning("[Globals] Already initializing...")
            return False
        
        globals_inst.init_status = InitStatus.INITIALIZING
        globals_inst.init_start_time = datetime.utcnow()
        logger.info("🚀 [Globals] Starting initialization sequence...")
    
    try:
        # Initialize quantum systems
        _init_quantum_systems(globals_inst)
        
        # Initialize database
        _init_database(globals_inst)
        
        # Initialize authentication
        _init_authentication(globals_inst)
        
        # Initialize terminal
        _init_terminal(globals_inst)
        
        # Mark as initialized
        with globals_inst.lock:
            globals_inst.init_status = InitStatus.INITIALIZED
            elapsed = (datetime.utcnow() - globals_inst.init_start_time).total_seconds()
            logger.info(f"✅ [Globals] INITIALIZATION COMPLETE ({elapsed:.2f}s)")
        
        return True
    
    except Exception as e:
        with globals_inst.lock:
            globals_inst.init_status = InitStatus.FAILED
            globals_inst.init_errors.append(str(e))
            logger.error(f"❌ [Globals] INITIALIZATION FAILED: {e}")
        return False

def _init_quantum_systems(globals_inst: GlobalState):
    """Initialize all quantum subsystems"""
    logger.info("[Globals] Initializing quantum systems...")
    try:
        from quantum_lattice_control_live_complete import (
            HEARTBEAT, LATTICE, LATTICE_NEURAL_REFRESH, 
            W_STATE_ENHANCED, NOISE_BATH_ENHANCED, QUANTUM_COORDINATOR
        )
        
        with globals_inst.lock:
            globals_inst.quantum.heartbeat = HEARTBEAT
            globals_inst.quantum.lattice = LATTICE
            globals_inst.quantum.neural_network = LATTICE_NEURAL_REFRESH
            globals_inst.quantum.w_state_manager = W_STATE_ENHANCED
            globals_inst.quantum.noise_bath = NOISE_BATH_ENHANCED
            globals_inst.quantum.quantum_coordinator = QUANTUM_COORDINATOR
        
        # Ensure heartbeat is running
        if HEARTBEAT and not HEARTBEAT.running:
            HEARTBEAT.start()
            logger.info("[Globals] ✅ HEARTBEAT started")
        
        if globals_inst.quantum.all_initialized():
            logger.info("[Globals] ✅ All quantum systems online")
        else:
            logger.warning("[Globals] ⚠️ Some quantum systems not initialized")
    
    except ImportError as e:
        logger.error(f"[Globals] ✗ Quantum import error: {e}")
        raise

def _init_database(globals_inst: GlobalState):
    """Initialize database connection pool"""
    logger.info("[Globals] Initializing database...")
    try:
        from db_builder_v2 import DB_POOL, init_db
        
        with globals_inst.lock:
            globals_inst.database.pool = DB_POOL
            globals_inst.database.healthy = True
        
        # Initialize schema
        init_db()
        
        logger.info("[Globals] ✅ Database initialized")
    
    except ImportError as e:
        logger.error(f"[Globals] ✗ Database import error: {e}")
        globals_inst.database.healthy = False

def _init_authentication(globals_inst: GlobalState):
    """Initialize authentication systems"""
    logger.info("[Globals] Initializing authentication...")
    try:
        from auth_handlers import JWTTokenManager
        
        jwt_manager = JWTTokenManager()
        with globals_inst.lock:
            globals_inst.auth.jwt_manager = jwt_manager
        
        logger.info("[Globals] ✅ Authentication initialized")
    
    except ImportError as e:
        logger.error(f"[Globals] ✗ Auth import error: {e}")

def _init_terminal(globals_inst: GlobalState):
    """Initialize terminal/command execution"""
    logger.info("[Globals] Initializing terminal...")
    try:
        from terminal_logic import TerminalEngine, GlobalCommandRegistry
        
        engine = TerminalEngine()
        with globals_inst.lock:
            globals_inst.terminal.engine = engine
            globals_inst.terminal.command_registry = GlobalCommandRegistry.ALL_COMMANDS if hasattr(GlobalCommandRegistry, 'ALL_COMMANDS') else {}
        
        logger.info("[Globals] ✅ Terminal initialized")
    
    except ImportError as e:
        logger.error(f"[Globals] ✗ Terminal import error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# CONVENIENCE ACCESSOR FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════════════

def get_heartbeat():
    """Get heartbeat instance"""
    return get_globals().quantum.heartbeat

def get_lattice():
    """Get quantum lattice"""
    return get_globals().quantum.lattice

def get_quantum_neural():
    """Get neural network"""
    return get_globals().quantum.neural_network

def get_quantum_wstate():
    """Get W-state manager"""
    return get_globals().quantum.w_state_manager

def get_quantum_noise():
    """Get noise bath"""
    return get_globals().quantum.noise_bath

def get_db_pool():
    """Get database pool"""
    return get_globals().database.pool

def get_auth_manager():
    """Get JWT auth manager"""
    return get_globals().auth.jwt_manager

def get_terminal():
    """Get terminal engine"""
    return get_globals().terminal.engine

def get_metrics():
    """Get application metrics"""
    return get_globals().metrics

def get_rate_limiter():
    """Get rate limiter"""
    return get_globals().rate_limiting

def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value"""
    return get_globals().config.get(key, default)

def set_config(key: str, value: Any):
    """Set configuration value"""
    with get_globals().lock:
        get_globals().config[key] = value

def get_system_health() -> Dict[str, Any]:
    """Get current system health"""
    return get_globals().get_system_health()

def get_state_snapshot() -> Dict[str, Any]:
    """Get complete state snapshot"""
    return get_globals().snapshot()

# ═══════════════════════════════════════════════════════════════════════════════════════
# REQUEST-SCOPED CONTEXT
# ═══════════════════════════════════════════════════════════════════════════════════════

def set_request_context(key: str, value: Any):
    """Set request-scoped context"""
    with get_globals().lock:
        if 'request_id' not in get_globals().request_context:
            get_globals().request_context['request_id'] = {}
        get_globals().request_context[key] = value

def get_request_context(key: str, default: Any = None) -> Any:
    """Get request-scoped context"""
    return get_globals().request_context.get(key, default)

def clear_request_context():
    """Clear request context"""
    with get_globals().lock:
        get_globals().request_context.clear()

# ═══════════════════════════════════════════════════════════════════════════════════════
# SESSION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════════════

def create_session(session_id: str, user_data: Dict[str, Any]) -> bool:
    """Create new session"""
    with get_globals().lock:
        get_globals().sessions[session_id] = {
            'user_data': user_data,
            'created_at': datetime.utcnow(),
            'last_access': datetime.utcnow(),
        }
        get_globals().auth.active_sessions += 1
    return True

def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Get session data"""
    session = get_globals().sessions.get(session_id)
    if session:
        with get_globals().lock:
            session['last_access'] = datetime.utcnow()
    return session

def delete_session(session_id: str) -> bool:
    """Delete session"""
    with get_globals().lock:
        if session_id in get_globals().sessions:
            del get_globals().sessions[session_id]
            get_globals().auth.active_sessions -= 1
            return True
    return False

# ═══════════════════════════════════════════════════════════════════════════════════════
# RATE LIMITING INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════════════

def check_rate_limit(key: str, limit: Optional[int] = None) -> bool:
    """Check if request is within rate limit"""
    return get_globals().rate_limiting.check_limit(key, limit)

# ═══════════════════════════════════════════════════════════════════════════════════════
# MONITORING & DEBUGGING
# ═══════════════════════════════════════════════════════════════════════════════════════

def get_debug_info() -> Dict[str, Any]:
    """Get comprehensive debug information"""
    g = get_globals()
    return {
        'init_status': g.init_status.value,
        'init_errors': g.init_errors,
        'quantum_health': g.quantum.get_health(),
        'database_healthy': g.database.healthy,
        'auth_active_sessions': g.auth.active_sessions,
        'metrics': g.metrics.get_stats(),
        'uptime_seconds': (datetime.utcnow() - g.startup_time).total_seconds() if g.startup_time else 0,
    }

logger.info("✅ [Globals] Module loaded - ready for initialization")
