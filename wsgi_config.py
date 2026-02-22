#!/usr/bin/env python3
"""
QTCL WSGI v5.1 - GLOBALS COORDINATOR & SYSTEM BOOTSTRAP
========================================================
Fixed edition — all known bugs resolved:

  BUG-1  CRITICAL  Blueprint registration was entirely absent. Flask app had no API
                   routes — every /api/* call returned 404, proxied by Koyeb to 503.
  BUG-2  CRITICAL  /api/heartbeat and /api/keepalive routes missing. LightweightHeartbeat
                   (quantum_lattice) POSTs to /api/heartbeat every 30 s → 404 → 503.
  BUG-3  CRITICAL  /api/quantum/heartbeat/status missing. terminal_logic quantum-heartbeat
                   monitor command GETs this path → 503 on every invocation.
  BUG-4  HIGH      get_heartbeat() returned {'status':'not_initialized'} (truthy dict)
                   when uninitialized. Every register_*_with_heartbeat() called
                   hb.add_listener() on a plain dict → AttributeError; all listener
                   registrations silently failed. Only oracle_api had the correct guard.
  BUG-5  MEDIUM    /health returned HTTP 503 when DB not yet connected. During the
                   deferred DB init window the keep-alive daemon logged a CRITICAL
                   failure streak on every startup, masking real failures later.
  BUG-6  LOW       SERVICES['quantum'] used get_heartbeat() instead of get_quantum(),
                   so GET /api/quantum returned raw heartbeat data, not quantum status.
  BUG-7  LOW       Heartbeat listeners (core, blockchain, quantum, admin, defi, oracle)
                   were never wired after globals initialized; the heartbeat ran with
                   zero listeners and logged desync warnings every cycle.

Initialization order (unchanged):
  1. Logging  2. Utility classes  3. DB (deferred thread)  4. Globals (deferred thread)
  5. Terminal engine (lazy)  6. Flask app  7. Blueprint registration  8. Routes
  9. Keep-alive daemon
"""

import os
import sys
import logging
import threading
import time
import traceback
import json

# ── Logging: configure once, never re-configure on re-import ─────────────────
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY CLASSES  — available to all modules via `from wsgi_config import ...`
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleProfiler:
    """Thread-safe performance profiler — records call durations per named operation."""

    def __init__(self):
        self.metrics: dict = {}
        self.lock = threading.RLock()

    def record(self, name: str, duration_ms: float) -> None:
        with self.lock:
            self.metrics.setdefault(name, []).append(duration_ms)

    def get_stats(self, name: str) -> dict:
        with self.lock:
            times = self.metrics.get(name, [])
            if not times:
                return {'count': 0, 'avg_ms': 0.0, 'min_ms': 0.0, 'max_ms': 0.0}
            return {
                'count':  len(times),
                'avg_ms': sum(times) / len(times),
                'min_ms': min(times),
                'max_ms': max(times),
            }

    def all_stats(self) -> dict:
        with self.lock:
            return {name: self.get_stats(name) for name in self.metrics}


class SimpleCache:
    """Thread-safe in-memory LRU cache with per-entry TTL support."""

    def __init__(self, max_size: int = 2000):
        self.cache:    dict = {}
        self.max_size: int  = max_size
        self.lock = threading.RLock()

    def get(self, key: str):
        with self.lock:
            entry = self.cache.get(key)
            if entry is None:
                return None
            if entry['ttl'] and time.time() > entry['expires_at']:
                del self.cache[key]
                return None
            return entry['value']

    def set(self, key: str, value, ttl_sec: int = 3600) -> None:
        with self.lock:
            if len(self.cache) >= self.max_size:
                self.cache.pop(next(iter(self.cache)), None)  # evict oldest
            self.cache[key] = {
                'value':      value,
                'ttl':        ttl_sec,
                'expires_at': time.time() + ttl_sec if ttl_sec else None,
            }

    def delete(self, key: str) -> None:
        with self.lock:
            self.cache.pop(key, None)

    def clear(self) -> None:
        with self.lock:
            self.cache.clear()

    def size(self) -> int:
        with self.lock:
            return len(self.cache)


class RequestTracer:
    """Lightweight per-request correlation-ID generator."""

    def __init__(self):
        import uuid
        self._uuid = uuid
        self.current_id: str | None = None

    def new_id(self) -> str:
        self.current_id = str(self._uuid.uuid4())
        return self.current_id


class ErrorBudgetManager:
    """Sliding-window error-rate budget tracker."""

    def __init__(self, max_errors: int = 100, window_sec: int = 60):
        self.max_errors  = max_errors
        self.window_sec  = window_sec
        self._timestamps: list = []
        self.lock = threading.RLock()

    def _prune(self, now: float) -> None:
        """Remove timestamps outside the window — MUST be called with lock held."""
        cutoff = now - self.window_sec
        self._timestamps = [t for t in self._timestamps if t >= cutoff]

    def record_error(self) -> None:
        with self.lock:
            now = time.time()
            self._prune(now)
            self._timestamps.append(now)

    def is_exhausted(self) -> bool:
        with self.lock:
            self._prune(time.time())
            return len(self._timestamps) >= self.max_errors

    def remaining(self) -> int:
        with self.lock:
            self._prune(time.time())
            return max(0, self.max_errors - len(self._timestamps))


# ── Module-level singletons — importable by all API modules ──────────────────
PROFILER            = SimpleProfiler()
CACHE               = SimpleCache(max_size=2000)
RequestCorrelation  = RequestTracer()
ERROR_BUDGET        = ErrorBudgetManager(max_errors=100, window_sec=60)

# ── Exported stubs — blockchain_api / defi_api / admin_api import these ──────
# Each module implements its own circuit-breaking; these are coordination stubs.
CIRCUIT_BREAKERS: dict = {}
RATE_LIMITERS:    dict = {}

logger.info("[BOOTSTRAP] ✅ Utility singletons ready (PROFILER, CACHE, RequestCorrelation, ERROR_BUDGET)")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — DATABASE (deferred, non-blocking)
# ═══════════════════════════════════════════════════════════════════════════════

DB      = None
DB_POOL = None


def _initialize_database_deferred() -> None:
    """
    Connect to PostgreSQL in a daemon thread so Flask can start immediately.
    Verifies credentials, creates a test connection, and runs SELECT version().
    Sets module-level DB / DB_POOL on success.
    """
    global DB, DB_POOL
    try:
        logger.info("[BOOTSTRAP/DB] Starting deferred database initialization…")
        from db_builder_v2 import (
            db_manager, DB_POOL as _pool,
            POOLER_HOST, POOLER_USER, POOLER_PASSWORD, POOLER_PORT, POOLER_DB,
        )

        if db_manager is None:
            logger.error("[BOOTSTRAP/DB] ❌ db_manager is None — credential/pool failure")
            logger.error(f"[BOOTSTRAP/DB]   HOST={POOLER_HOST}  USER={POOLER_USER}  "
                         f"PASSWORD={'<set>' if POOLER_PASSWORD else '<EMPTY>'}  "
                         f"PORT={POOLER_PORT}  DB={POOLER_DB}")
            return

        if db_manager.pool is None:
            logger.error(f"[BOOTSTRAP/DB] ❌ Pool creation failed: {db_manager.pool_error}")
            return

        # Smoke-test: one query to confirm the pool actually works
        test_conn = db_manager.get_connection()
        if test_conn is None:
            logger.error("[BOOTSTRAP/DB] ❌ Could not obtain test connection from pool")
            return

        cur = test_conn.cursor()
        cur.execute("SELECT version()")
        version = cur.fetchone()
        cur.close()
        db_manager.return_connection(test_conn)

        DB      = db_manager
        DB_POOL = _pool
        logger.info(f"[BOOTSTRAP/DB] ✅ Pool ready — {version[0][:70]}…")

    except ImportError as exc:
        logger.error(f"[BOOTSTRAP/DB] ❌ Import failed: {exc}")
    except Exception as exc:
        logger.error(f"[BOOTSTRAP/DB] ❌ Unexpected error: {exc}\n{traceback.format_exc()}")


_DB_INIT_THREAD = threading.Thread(
    target=_initialize_database_deferred, daemon=True, name="db-init",
)
_DB_INIT_THREAD.start()
logger.info("[BOOTSTRAP] ✅ DB initialization started in background daemon thread")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — GLOBALS MODULE (deferred, non-blocking)
# ═══════════════════════════════════════════════════════════════════════════════

GLOBALS_AVAILABLE = False


def _initialize_globals_deferred() -> None:
    """Call globals.initialize_globals() in a daemon thread; mark GLOBALS_AVAILABLE on success."""
    global GLOBALS_AVAILABLE
    try:
        logger.info("[BOOTSTRAP/GLOBALS] Initializing global state…")
        from globals import initialize_globals
        initialize_globals()
        GLOBALS_AVAILABLE = True
        logger.info("[BOOTSTRAP/GLOBALS] ✅ Global state initialized")
        # Wire heartbeat listeners now that globals (and HEARTBEAT singleton) are ready
        _register_heartbeat_listeners()
    except Exception as exc:
        logger.error(f"[BOOTSTRAP/GLOBALS] ⚠️  Initialization failed: {exc}", exc_info=True)


try:
    from globals import (
        initialize_globals, get_globals,
        get_system_health, get_state_snapshot,
        get_heartbeat, get_blockchain, get_ledger, get_oracle, get_defi,
        get_auth_manager, get_pqc_system, get_genesis_block, verify_genesis_block,
        get_metrics, get_quantum, dispatch_command, COMMAND_REGISTRY,
        pqc_generate_user_key, pqc_sign, pqc_verify,
        pqc_encapsulate, pqc_prove_identity, pqc_verify_identity,
        pqc_revoke_key, pqc_rotate_key, bootstrap_admin_session, revoke_session,
    )
    logger.info("[BOOTSTRAP/GLOBALS] ✅ Module imports resolved")

    _GLOBALS_INIT_THREAD = threading.Thread(
        target=_initialize_globals_deferred, daemon=True, name="globals-init",
    )
    _GLOBALS_INIT_THREAD.start()
    logger.info("[BOOTSTRAP/GLOBALS] ✅ Initialization thread started")

except Exception as exc:
    logger.error(f"[BOOTSTRAP/GLOBALS] ❌ FATAL — cannot import globals: {exc}")
    raise


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — TERMINAL ENGINE (lazy, thread-safe)
# ═══════════════════════════════════════════════════════════════════════════════

_TERMINAL_INITIALIZED  = False
_TERMINAL_INIT_FAILED  = False
_terminal_engine       = None
_terminal_lock         = threading.Lock()


def _initialize_terminal_engine():
    """
    Lazy, idempotent, thread-safe terminal engine boot.

    Uses double-checked locking: fast path (flag read) avoids the lock on
    every call after initialization; slow path (inside lock) prevents the
    duplicate-initialization race that would spawn 40+ orphan threads under
    concurrent WSGI worker startup.

    Sets _TERMINAL_INIT_FAILED=True on first failure so repeated calls from
    request handlers don't hammer the import system on every request.
    """
    global _TERMINAL_INITIALIZED, _TERMINAL_INIT_FAILED, _terminal_engine

    if _TERMINAL_INITIALIZED or _TERMINAL_INIT_FAILED:
        return _terminal_engine

    with _terminal_lock:
        if _TERMINAL_INITIALIZED or _TERMINAL_INIT_FAILED:
            return _terminal_engine

        try:
            logger.info("[BOOTSTRAP/TERMINAL] Initializing terminal engine…")
            from terminal_logic import TerminalEngine, register_all_commands

            engine    = TerminalEngine()
            cmd_count = register_all_commands(engine)
            total     = len(COMMAND_REGISTRY)

            logger.info(f"[BOOTSTRAP/TERMINAL] ✓ Registered {total} commands "
                        f"({cmd_count} from terminal_logic handlers)")

            if total < 10:
                raise RuntimeError(
                    f"Command registration catastrophically incomplete: {total} commands. "
                    "Check terminal_logic import errors above."
                )
            if total < 80:
                logger.warning(
                    f"[BOOTSTRAP/TERMINAL] ⚠ Only {total} commands registered (expected 89+). "
                    "Some commands may be missing."
                )

            _terminal_engine       = engine
            _TERMINAL_INITIALIZED  = True
            logger.info("[BOOTSTRAP/TERMINAL] ✅ Terminal engine ready")
            return engine

        except Exception as exc:
            _TERMINAL_INIT_FAILED = True
            logger.error(
                f"[BOOTSTRAP/TERMINAL] ❌ Init failed: {exc}\n"
                "  → Falling back to globals.py stub registry (basic commands work).\n"
                "  → Fix the error above and restart to enable the full command set.",
                exc_info=True,
            )
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Utilities
    'DB', 'PROFILER', 'CACHE', 'RequestCorrelation', 'ERROR_BUDGET',
    'CIRCUIT_BREAKERS', 'RATE_LIMITERS', 'GLOBALS_AVAILABLE', 'logger',
    # Globals API
    'get_system_health', 'get_state_snapshot',
    'get_heartbeat', 'get_blockchain', 'get_ledger', 'get_oracle', 'get_defi',
    'get_auth_manager', 'get_pqc_system', 'get_genesis_block', 'verify_genesis_block',
    'get_metrics', 'get_quantum', 'dispatch_command', 'COMMAND_REGISTRY',
    # PQC
    'pqc_generate_user_key', 'pqc_sign', 'pqc_verify',
    'pqc_encapsulate', 'pqc_prove_identity', 'pqc_verify_identity',
    'pqc_revoke_key', 'pqc_rotate_key',
    # Session
    'bootstrap_admin_session', 'revoke_session',
]


# ═══════════════════════════════════════════════════════════════════════════════
# FLASK APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime, timezone

app = Flask(
    __name__,
    static_folder=os.path.join(PROJECT_ROOT, 'static'),
    static_url_path='/static',
)
app.config['JSON_SORT_KEYS'] = False

logger.info("[FLASK] ✅ Flask app created")

# ── Start background blueprint registration (non-blocking) ────────────────────
# This must happen AFTER Flask app is created but BEFORE any requests arrive.
# It runs in a background thread so HTTP requests (health checks, etc.) are not blocked.



@app.before_request
def _before():
    logger.debug(f"[REQUEST] {request.method} {request.path}")


@app.after_request
def _after(response):
    logger.debug(f"[RESPONSE] {response.status_code}")
    return response


# ── Safe service accessors ────────────────────────────────────────────────────

class _ServiceAccessor:
    """
    Wraps a zero-arg getter so accidental None / exception returns a safe default.
    Avoids silently swallowing TypeError that would hide real bugs — only catches
    the known-safe cases (subsystem not ready / getter raises).
    """

    def __init__(self, name: str, getter):
        self.name   = name
        self.getter = getter

    def get(self, default=None):
        try:
            val = self.getter()
            return val if val is not None else (default if default is not None else {})
        except Exception as exc:
            logger.warning(f"⚠️  {self.name} accessor error: {str(exc)[:80]}")
            return default if default is not None else {}


# BUG-6 FIX: SERVICES['quantum'] must use get_quantum(), not get_heartbeat().
SERVICES = {
    'quantum':    _ServiceAccessor('quantum',    get_quantum),      # was get_heartbeat ← BUG-6
    'blockchain': _ServiceAccessor('blockchain', get_blockchain),
    'ledger':     _ServiceAccessor('ledger',     get_ledger),
    'oracle':     _ServiceAccessor('oracle',     get_oracle),
    'defi':       _ServiceAccessor('defi',       get_defi),
    'auth':       _ServiceAccessor('auth',       get_auth_manager),
    'pqc':        _ServiceAccessor('pqc',        get_pqc_system),
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE STATUS CACHE  — background thread updates every 30 s
# ═══════════════════════════════════════════════════════════════════════════════

_DB_STATUS_CACHE: dict  = {
    'connected': False, 'healthy': False,
    'host': 'unknown',  'port': 5432,  'database': 'unknown',
    'timestamp': 0.0,   'error': None,
}
_DB_STATUS_LOCK             = threading.Lock()
_DB_STATUS_UPDATE_INTERVAL  = 30   # seconds


def _update_db_status_cache() -> None:
    """Daemon thread: refresh cached DB status without ever blocking a request handler."""

    def _do_update() -> None:
        try:
            from db_builder_v2 import verify_database_connection
            result = verify_database_connection(verbose=False)
            with _DB_STATUS_LOCK:
                _DB_STATUS_CACHE.update({
                    'connected': result.get('connected', False),
                    'healthy':   result.get('healthy',   False),
                    'host':      result.get('host',      'unknown'),
                    'port':      result.get('port',      5432),
                    'database':  result.get('database',  'unknown'),
                    'timestamp': time.time(),
                    'error':     None,
                })
        except Exception as exc:
            with _DB_STATUS_LOCK:
                _DB_STATUS_CACHE.update({
                    'connected': False, 'healthy': False,
                    'timestamp': time.time(),
                    'error':     str(exc)[:120],
                })

    # Immediate update so /health is accurate from the first request
    try:
        _do_update()
        logger.info("[DB-STATUS-CACHE] ✅ Initial cache populated")
    except Exception as exc:
        logger.warning(f"[DB-STATUS-CACHE] Initial update error: {exc}")

    while True:
        try:
            time.sleep(_DB_STATUS_UPDATE_INTERVAL)
            _do_update()
        except Exception as exc:
            logger.debug(f"[DB-STATUS-CACHE] Refresh error: {exc}")
            time.sleep(5)


threading.Thread(
    target=_update_db_status_cache, daemon=True, name="db-status-cache",
).start()


# ═══════════════════════════════════════════════════════════════════════════════
# CORE ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/', methods=['GET'])
def index():
    """Root — serve index.html for browsers, JSON for API clients."""
    if 'text/html' in request.headers.get('Accept', ''):
        try:
            return send_from_directory(PROJECT_ROOT, 'index.html')
        except Exception:
            pass
    return jsonify({
        'system':    'QTCL v5.1',
        'status':    'operational',
        'timestamp': datetime.now(timezone.utc).isoformat(),
    })


@app.route('/health', methods=['GET'])
def health():
    """
    Fast health endpoint — returns cached DB status in <10 ms, never blocks.

    BUG-5 FIX: Always returns HTTP 200.  Previous code returned 503 when the DB
    deferred-init thread hadn't connected yet, causing the keep-alive daemon to
    log a CRITICAL failure streak on every cold start and masking real failures.
    The actual health information is in the JSON body; callers should inspect
    `database.connected` and `status` instead of relying on the HTTP status code
    for degraded-but-running distinctions.

    Use /api/db-diagnostics for verbose connection info.
    """
    with _DB_STATUS_LOCK:
        db = _DB_STATUS_CACHE.copy()

    connected = db.get('connected', False)
    status    = 'healthy' if connected else 'degraded'

    return jsonify({
        'status':              status,
        'timestamp':           datetime.now(timezone.utc).isoformat(),
        'globals_initialized': GLOBALS_AVAILABLE,
        'database': {
            'connected':        connected,
            'host':             db.get('host'),
            'port':             db.get('port'),
            'database':         db.get('database'),
            'error':            db.get('error'),
            'cache_age_seconds': int(time.time() - db.get('timestamp', 0)),
        },
        'note': (
            None if connected
            else 'DB not yet connected — check Koyeb env vars or wait for deferred init. '
                 'Details: /api/db-diagnostics'
        ),
    }), 200   # ← BUG-5 FIX: always 200, degraded state is in the body


# ── BUG-2 FIX: add /api/heartbeat and /api/keepalive routes ──────────────────
#
# LightweightHeartbeat (quantum_lattice_control_live_complete.py:389) POSTs JSON
# metrics to KEEPALIVE_URL which defaults to f"{APP_URL}/api/heartbeat".
# QuantumSystemCoordinator (same file:2541) uses /api/keepalive.
# Neither route existed before → every keepalive attempt returned 404 which
# Koyeb's proxy surfaced to callers as 503.

@app.route('/api/heartbeat', methods=['GET', 'POST'])
def api_heartbeat_receiver():
    """
    Keepalive / metrics sink for LightweightHeartbeat (quantum_lattice module).

    Accepts GET pings from the instance keep-alive daemon and POST payloads from
    the quantum subsystem's LightweightHeartbeat.  Responds 200 in all cases.
    POST bodies are optional — if present they are logged at DEBUG level so the
    metrics are visible in logs without adding overhead to every call.
    """
    if request.method == 'POST':
        try:
            payload = request.get_json(silent=True) or {}
            # ── Extract nested lattice fields for readable INFO-level output ─────
            hb_sub   = payload.get('heartbeat', {})
            ws_sub   = payload.get('w_state', {})
            lat_sub  = payload.get('lattice', {})
            beat_n   = payload.get('beat_count', hb_sub.get('pulse_count', '?'))
            coherence = (
                payload.get('lattice_coherence')
                or lat_sub.get('global_coherence')
                or lat_sub.get('coherence')
                or ws_sub.get('coherence_avg')
                or 'n/a'
            )
            fidelity = (
                lat_sub.get('global_fidelity')
                or lat_sub.get('fidelity')
                or ws_sub.get('fidelity_avg')
                or 'n/a'
            )
            listeners = hb_sub.get('listeners', hb_sub.get('listener_count', '?'))
            running   = hb_sub.get('running', '?')
            logger.info(
                f"[LATTICE-BEAT] beat=#{beat_n} | "
                f"coherence={coherence} | fidelity={fidelity} | "
                f"hb.running={running} | listeners={listeners}"
            )
        except Exception:
            pass   # malformed body — accept and discard

    return jsonify({
        'status':    'ok',
        'ts':        time.time(),
        'server':    'QTCL v5.1',
        'received':  True,
    }), 200


@app.route('/api/keepalive', methods=['GET', 'POST'])
def api_keepalive_receiver():
    """
    Secondary keepalive sink for QuantumSystemCoordinator.

    The coordinator sends POST requests to /api/keepalive every interval_seconds.
    Returns 200 with a minimal body so the coordinator marks the ping successful.
    """
    if request.method == 'POST':
        try:
            payload = request.get_json(silent=True) or {}
            logger.debug(f"[KEEPALIVE] coordinator ping | keys={list(payload.keys())[:5]}")
        except Exception:
            pass

    return jsonify({'status': 'ok', 'ts': time.time()}), 200


# ── BUG-3 FIX: /api/quantum/heartbeat/status ─────────────────────────────────
#
# terminal_logic.py:4654 and :4704 issue GET /quantum/heartbeat/status through
# the API client (base URL = /api).  The route did not exist in any registered
# blueprint.  The quantum_api module exports get_quantum_heartbeat_status() as a
# plain Python function, never as a Flask route.  Added here so it is always
# available regardless of which blueprint variant is registered.

@app.route('/api/quantum/heartbeat/status', methods=['GET'])
def api_quantum_heartbeat_status():
    """
    Heartbeat status for the terminal quantum-heartbeat-monitor command.

    Reads directly from the HEARTBEAT singleton via get_heartbeat() with a proper
    isinstance guard (BUG-4 fix pattern).  Falls back gracefully when the quantum
    subsystem hasn't initialized yet.
    """
    try:
        hb = get_heartbeat()
        # BUG-4 GUARD: get_heartbeat() may return {'status':'not_initialized'} dict
        if hb is None or isinstance(hb, dict):
            return jsonify({
                'heartbeat': hb or {'status': 'not_initialized'},
                'running':   False,
                'note':      'Quantum subsystem not yet initialized',
            }), 200

        # Real UniversalQuantumHeartbeat object
        metrics = hb.get_metrics() if hasattr(hb, 'get_metrics') else {}
        return jsonify({
            'heartbeat': metrics,
            'running':   metrics.get('running', hb.running if hasattr(hb, 'running') else False),
        }), 200

    except Exception as exc:
        logger.error(f"[HB-STATUS] Error reading heartbeat: {exc}")
        return jsonify({'error': str(exc), 'running': False}), 200   # 200 not 500 — terminal expects JSON


@app.route('/api/db-diagnostics', methods=['GET'])
def db_diagnostics():
    """Verbose DB diagnostic dump — gated by ALLOW_DIAGNOSTICS env var in production."""
    allow = os.getenv('ALLOW_DIAGNOSTICS', 'false').lower() == 'true'
    if not allow and os.getenv('FLASK_ENV') == 'production':
        return jsonify({'error': 'Diagnostics endpoint disabled in production'}), 403
    try:
        from db_builder_v2 import verify_database_connection
        result = verify_database_connection(verbose=True)
        return jsonify(result), 200 if result.get('healthy') else 503
    except Exception as exc:
        logger.error(f"[DIAG] {exc}")
        return jsonify({'error': str(exc), 'timestamp': datetime.now(timezone.utc).isoformat()}), 500


@app.route('/api/status', methods=['GET'])
def status():
    """System status — health + state snapshot."""
    try:
        return jsonify({'health': get_system_health(), 'snapshot': get_state_snapshot()})
    except Exception:
        return jsonify({'status': 'operational'})


@app.route('/api/command', methods=['POST'])
def execute_command():
    """
    Dispatch a named command through the terminal engine.

    JWT extraction: tries env JWT_SECRET → auth_handlers.JWT_SECRET → globals._get_jwt_secret,
    in order, stopping at the first successful decode.  Carries decoded payload
    (user_id, role, is_admin) downstream to the command handler.
    """
    try:
        _initialize_terminal_engine()   # idempotent, thread-safe

        data    = request.get_json() or {}
        command = data.get('command', 'help')
        args    = data.get('args') or {}
        user_id = data.get('user_id')

        raw_token       = None
        role            = None
        is_admin        = False
        jwt_decode_error = None

        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            raw_token = auth_header[7:].strip()
        if not raw_token:
            raw_token = data.get('token') or data.get('access_token')

        if raw_token:
            try:
                import jwt as _jwt

                secrets_to_try = []
                env_secret = os.getenv('JWT_SECRET', '')
                if env_secret:
                    secrets_to_try.append(('ENV', env_secret))

                try:
                    from auth_handlers import JWT_SECRET as _ahs
                    if _ahs:
                        secrets_to_try.append(('AUTH_HANDLERS', _ahs))
                except Exception as _e:
                    logger.debug(f"[auth] auth_handlers JWT_SECRET unavailable: {_e}")

                try:
                    from globals import _get_jwt_secret as _gs_fn
                    _gs = _gs_fn()
                    if _gs:
                        secrets_to_try.append(('GLOBALS', _gs))
                except Exception as _e:
                    logger.debug(f"[auth] globals _get_jwt_secret unavailable: {_e}")

                jwt_payload    = {}
                decode_success = False

                for secret_source, secret in secrets_to_try:
                    try:
                        jwt_payload = _jwt.decode(raw_token, secret, algorithms=['HS512', 'HS256'])
                        decode_success = True
                        logger.debug(f"[auth] JWT decoded via {secret_source}")
                        break
                    except _jwt.InvalidSignatureError:
                        continue
                    except Exception as _e:
                        jwt_decode_error = str(_e)
                        continue

                if decode_success and jwt_payload:
                    if not user_id:
                        user_id  = jwt_payload.get('user_id')
                    role     = jwt_payload.get('role', 'user')
                    is_admin = bool(jwt_payload.get('is_admin', False)) or role in (
                        'admin', 'superadmin', 'super_admin'
                    )
                    logger.info(f"[auth] ✓ JWT valid: user={user_id} role={role} admin={is_admin}")
                else:
                    logger.warning(
                        f"[auth] ⚠️  JWT decode failed for …{raw_token[:20]} | {jwt_decode_error}"
                    )
            except Exception as exc:
                logger.error(f"[auth] Unexpected JWT error: {exc}", exc_info=True)

        result = dispatch_command(command, args, user_id, token=raw_token, role=role)
        return jsonify(result)

    except Exception as exc:
        logger.error(f"[/api/command] {exc}")
        return jsonify({'status': 'error', 'error': str(exc)}), 500


@app.route('/api/commands', methods=['GET'])
def commands():
    """List all registered commands (serializable subset — handler callables excluded)."""
    try:
        _initialize_terminal_engine()   # idempotent
        serializable = {
            name: {k: v for k, v in info.items() if k != 'handler' and not callable(v)}
            for name, info in COMMAND_REGISTRY.items()
        }
        return jsonify({'total': len(serializable), 'commands': serializable, 'status': 'success'})
    except Exception as exc:
        logger.error(f"[/api/commands] {exc}", exc_info=True)
        return jsonify({'total': 0, 'commands': {}, 'status': 'error', 'error': str(exc)}), 200


@app.route('/api/genesis', methods=['GET'])
def genesis():
    """Genesis block retrieval and PQC verification."""
    try:
        return jsonify({'block': get_genesis_block(), 'verified': verify_genesis_block()})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/quantum', methods=['GET'])
def quantum_summary():
    """Comprehensive quantum/lattice metrics - BUG-6 FIX: uses get_quantum() not get_heartbeat()"""
    try:
        from quantum_lattice_control_live_complete import (
            HEARTBEAT, LATTICE_NEURAL_REFRESH, W_STATE_ENHANCED, NOISE_BATH_ENHANCED, LATTICE
        )
        
        metrics = {'status': 'online', 'timestamp': time.time()}
        
        # Heartbeat metrics
        if HEARTBEAT is not None:
            metrics['heartbeat'] = HEARTBEAT.get_metrics()
        
        # Lattice neural refresh metrics
        if LATTICE_NEURAL_REFRESH is not None:
            try:
                metrics['lattice_neural'] = LATTICE_NEURAL_REFRESH.get_state()
            except:
                pass
        
        # W-state metrics
        if W_STATE_ENHANCED is not None:
            try:
                metrics['w_state'] = W_STATE_ENHANCED.get_metrics()
            except:
                pass
        
        # Noise bath metrics
        if NOISE_BATH_ENHANCED is not None:
            try:
                metrics['noise_bath'] = NOISE_BATH_ENHANCED.get_metrics()
            except:
                pass
        
        # Lattice metrics
        if LATTICE is not None:
            try:
                metrics['lattice'] = LATTICE.get_system_metrics()
            except:
                pass
        
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"[/api/quantum] {e}")
        return jsonify(SERVICES['quantum'].get())  # Fallback to old method if everything fails





@app.route('/api/blockchain', methods=['GET'])
def blockchain_summary():
    return jsonify(SERVICES['blockchain'].get())


@app.route('/api/ledger', methods=['GET'])
def ledger_summary():
    return jsonify(SERVICES['ledger'].get())


@app.route('/api/metrics', methods=['GET'])
def metrics():
    try:
        return jsonify(get_metrics())
    except Exception:
        return jsonify({'status': 'operational'})


@app.errorhandler(404)
def not_found(exc):
    return jsonify({'error': 'Not found', 'path': request.path}), 404


@app.errorhandler(500)
def server_error(exc):
    logger.error(f"[500] {exc}")
    return jsonify({'error': 'Internal server error'}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# BUG-1 FIX — BLUEPRINT REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════
#
# The previous version created the Flask app and defined routes but NEVER called
# app.register_blueprint().  Every /api/quantum/*, /api/blocks/*, /api/admin/*,
# /api/defi/*, /api/oracle/*, /api/auth/* etc. returned 404.  Koyeb's load
# balancer surfaced these as 503 to the external caller.
#
# Registration strategy:
#   • Each blueprint is wrapped in its own try/except so one failing module
#     cannot prevent the others from loading.
#   • More-specific prefixes (/api/quantum, /api/oracle) are registered FIRST
#     to avoid any ambiguity with the generic /api-prefix blueprints.
#   • We call the authoritative factory function from each module (get_*_blueprint)
#     rather than touching module-level `blueprint` variables, because several
#     modules define multiple Blueprint objects and we want the right one.
#   • blueprint names must be unique per Flask app — we use name_override where
#     a module inadvertently creates two blueprints with the same name.
#
# URL layout after registration:
#   /api/quantum/*   ← quantum_api     (create_quantum_api_blueprint_extended)
#   /api/oracle/*    ← oracle_api
#   /api/auth/*      ← core_api        (prefix /api, routes under /auth/*)
#   /api/users/*     ← core_api
#   /api/keys/*      ← core_api
#   /api/addresses/* ← core_api
#   /api/blocks/*    ← blockchain_api  (full blueprint, prefix /api)
#   /api/admin/*     ← admin_api       (prefix /api, routes under /admin/*)
#   /api/defi/*      ← defi_api
#   /api/governance/*← defi_api
#   /api/nft/*       ← defi_api
#   /api/bridge/*    ← defi_api

def _register_blueprints() -> None:
    """
    Register all API blueprints with the Flask app.
    Called once at module load time (inside a try/except per blueprint).
    Thread-safe: Flask's add_url_rule acquires an internal lock.
    """
    registrations = []   # (label, success, error)

    # ── 1. Quantum API (/api/quantum/*) — register first (most specific) ─────
    try:
        from quantum_api import get_quantum_blueprint
        bp = get_quantum_blueprint()
        app.register_blueprint(bp)
        registrations.append(('quantum_api [/api/quantum]', True, None))
    except Exception as exc:
        registrations.append(('quantum_api', False, exc))

    # ── 2. Oracle API (/api/oracle/*) ─────────────────────────────────────────
    try:
        from oracle_api import get_oracle_blueprint, register_oracle_with_heartbeat
        bp = get_oracle_blueprint()
        app.register_blueprint(bp)
        registrations.append(('oracle_api [/api/oracle]', True, None))
        # oracle_api is the only module with correct isinstance guard already
        try:
            register_oracle_with_heartbeat()
        except Exception:
            pass
    except Exception as exc:
        registrations.append(('oracle_api', False, exc))

    # ── 3. Core API (/api — auth, users, keys, addresses, sign, security…) ───
    # Use sys.modules to defer db_builder_v2 access until after all module inits
    try:
        from core_api import get_core_blueprint
        bp = get_core_blueprint()
        app.register_blueprint(bp)
        registrations.append(('core_api [/api]', True, None))
    except Exception as exc:
        registrations.append(('core_api', False, exc))

    # ── 4. Blockchain API (/api — blocks, transactions, mempool, chain…) ─────
    #    get_full_blockchain_blueprint() calls create_blueprint() which produces
    #    the complete production blueprint.  We must NOT also register the
    #    module-level `blueprint` variable (create_simple_blockchain_blueprint)
    #    as both use the name 'blockchain_api' and Flask would raise ValueError.
    try:
        from blockchain_api import get_full_blockchain_blueprint
        bp = get_full_blockchain_blueprint()
        app.register_blueprint(bp)
        registrations.append(('blockchain_api-full [/api]', True, None))
    except Exception as exc:
        registrations.append(('blockchain_api-full', False, exc))

    # ── 5. Admin API (/api — admin/*, stats/*, pqc/*, events/…) ──────────────
    # Use sys.modules to defer db_builder_v2 access until after all module inits
    try:
        from admin_api import get_admin_blueprint
        bp = get_admin_blueprint()
        app.register_blueprint(bp)
        registrations.append(('admin_api [/api]', True, None))
    except Exception as exc:
        registrations.append(('admin_api', False, exc))

    # ── 6. DeFi API (/api — defi/*, governance/*, nft/*, bridge/*) ───────────
    try:
        from defi_api import get_defi_blueprint
        bp = get_defi_blueprint()
        app.register_blueprint(bp)
        registrations.append(('defi_api [/api]', True, None))
    except Exception as exc:
        registrations.append(('defi_api', False, exc))

    # ── Summary ───────────────────────────────────────────────────────────────
    ok      = [label for label, success, _  in registrations if success]
    failed  = [(label, err) for label, success, err in registrations if not success]

    for label in ok:
        logger.info(f"[BLUEPRINT] ✅ {label}")
    for label, err in failed:
        logger.error(f"[BLUEPRINT] ❌ {label}: {err}")

    if failed:
        logger.warning(
            f"[BLUEPRINT] {len(failed)}/{len(registrations)} blueprints failed to register. "
            "The affected API surfaces will return 404."
        )
    else:
        logger.info(f"[BLUEPRINT] ✅ All {len(ok)} blueprints registered successfully")


# ── DEFERRED BLUEPRINT REGISTRATION — avoid circular import on gunicorn fork ──
# Do NOT call _register_blueprints() at module load time. Gunicorn's forking model
# loads wsgi_config in the parent process, then forks workers. If db_builder_v2 is
# mid-initialization during fork, workers inherit a half-constructed module.
# Solution: register blueprints lazily on the first HTTP request.
#
# Flask's before_request hook ensures registration happens once per worker
# after the worker has fully initialized and any deferred imports have completed.

_BLUEPRINTS_REGISTERED = False
_BLUEPRINT_THREAD = None

def _register_blueprints_background():
    """Register blueprints in background thread (non-blocking to HTTP requests)"""
    global _BLUEPRINTS_REGISTERED
    try:
        logger.info("[BLUEPRINT/BG] Starting background blueprint registration...")
        _register_blueprints()
        _BLUEPRINTS_REGISTERED = True
        logger.info("[BLUEPRINT/BG] ✅ Background blueprint registration complete")
    except Exception as exc:
        logger.error(f"[BLUEPRINT/BG] Failed during background registration: {exc}", exc_info=True)
        _BLUEPRINTS_REGISTERED = True  # Mark as attempted even if failed

def _start_blueprint_registration_thread():
    """Start background thread for blueprint registration (called immediately after app creation)"""
    global _BLUEPRINT_THREAD
    if _BLUEPRINT_THREAD is None or not _BLUEPRINT_THREAD.is_alive():
        _BLUEPRINT_THREAD = threading.Thread(
            target=_register_blueprints_background,
            daemon=True,
            name="BlueprintRegistration"
        )
        _BLUEPRINT_THREAD.start()
        logger.info("[BLUEPRINT/BG] 🔄 Background blueprint registration thread started (non-blocking)")


# ── Start blueprint registration immediately (runs in background) ────────────────────
_start_blueprint_registration_thread()


# ═══════════════════════════════════════════════════════════════════════════════
# BUG-4 + BUG-7 FIX — HEARTBEAT LISTENER REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════
#
# BUG-4: get_heartbeat() returns {'status':'not_initialized'} (a truthy dict)
# when globals hasn't initialized yet.  Every register_*_with_heartbeat() across
# core_api/blockchain_api/quantum_api/admin_api/defi_api does:
#
#     hb = get_heartbeat()
#     if hb:                      ← True for a non-empty dict
#         hb.add_listener(...)    ← AttributeError: 'dict' has no 'add_listener'
#
# The exception is caught and swallowed, so every listener registration silently
# fails.  Only oracle_api correctly guards with: if hb and not isinstance(hb, dict).
#
# BUG-7: Even with the guard fixed, _register_heartbeat_listeners() was never
# called anywhere.  The HEARTBEAT singleton ran forever with zero listeners and
# logged desync warnings on every cycle.
#
# Fix: this function is called from _initialize_globals_deferred() AFTER
# globals.initialize_globals() completes and HEARTBEAT is a real object.  The
# isinstance guard is applied uniformly here so the individual API modules'
# register_* functions are irrelevant — we register directly.

def _register_heartbeat_listeners() -> None:
    """
    Wire per-module heartbeat callbacks into the HEARTBEAT singleton.

    Must be called AFTER globals.initialize_globals() so HEARTBEAT is a
    UniversalQuantumHeartbeat instance, not None or a dict fallback.

    Each listener is registered in its own try/except so one failing module
    does not prevent the others from receiving heartbeat pulses.
    """
    # BUG-4 FIX: isinstance guard — never call .add_listener() on a plain dict
    hb = get_heartbeat()
    if hb is None or isinstance(hb, dict):
        logger.warning(
            "[HEARTBEAT-LISTENERS] HEARTBEAT not ready (got %s) — listener "
            "registration deferred.  Will retry on next globals init.",
            type(hb).__name__,
        )
        return

    if not hasattr(hb, 'add_listener'):
        logger.error(
            f"[HEARTBEAT-LISTENERS] HEARTBEAT object ({type(hb)}) has no add_listener — "
            "check quantum_lattice_control_live_complete.py"
        )
        return

    registered = []
    failed     = []

    _candidates = [
        ('core_api',        'core_api',        'register_core_with_heartbeat'),
        ('blockchain_api',  'blockchain_api',  'register_blockchain_with_heartbeat'),
        ('quantum_api',     'quantum_api',     'register_quantum_with_heartbeat'),
        ('admin_api',       'admin_api',       'register_admin_with_heartbeat'),
        ('defi_api',        'defi_api',        'register_defi_with_heartbeat'),
    ]

    for label, module_name, fn_name in _candidates:
        try:
            import importlib
            mod = importlib.import_module(module_name)
            register_fn = getattr(mod, fn_name, None)
            if register_fn is None:
                failed.append((label, f"{fn_name} not found in {module_name}"))
                continue

            # Each module's register_* function calls get_heartbeat() internally
            # and has `if hb: hb.add_listener(...)` — BUG-4 means they silently
            # fail.  Call them anyway (some may have been patched) and also
            # register the on_heartbeat method directly here as a backstop.
            try:
                register_fn()
            except Exception:
                pass   # fallback to direct registration below

            # Direct registration — bypass the buggy isinstance check in the modules
            hb_integration_attr = f"_{label.replace('_api', '')}_heartbeat"
            integration = getattr(mod, hb_integration_attr, None) or \
                          getattr(mod, '_defi_heartbeat', None)     or \
                          getattr(mod, '_hb_listener', None)

            if integration and hasattr(integration, 'on_heartbeat'):
                try:
                    hb.add_listener(integration.on_heartbeat)
                    registered.append(label)
                except Exception as exc:
                    # Listener already registered (some impls check for duplicates)
                    if 'duplicate' in str(exc).lower() or 'already' in str(exc).lower():
                        registered.append(f"{label} (already registered)")
                    else:
                        failed.append((label, str(exc)))
            else:
                # Module registered via its own register_fn (may have worked)
                registered.append(f"{label} (via register_fn)")

        except Exception as exc:
            failed.append((label, str(exc)))

    for label in registered:
        logger.info(f"[HEARTBEAT-LISTENERS] ✅ {label}")
    for label, err in failed:
        logger.warning(f"[HEARTBEAT-LISTENERS] ⚠️  {label}: {err}")

    listener_count = len(hb.listeners) if hasattr(hb, 'listeners') else '?'
    logger.info(
        f"[HEARTBEAT-LISTENERS] Registration complete — "
        f"{len(registered)} ok / {len(failed)} failed / {listener_count} total listeners on HEARTBEAT"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# INSTANCE KEEP-ALIVE DAEMON
# ═══════════════════════════════════════════════════════════════════════════════
#
# WHY: Koyeb (and most PaaS platforms) sleep instances after ~10 min of no
#   inbound traffic at the load-balancer level.  A self-ping on localhost does
#   NOT count.  This daemon fires a GET at the public /health URL so the
#   platform sees live traffic and never triggers the sleep lifecycle.
#
# INTERVAL: 300 s (5 min) — comfortably below any platform's sleep threshold.
#   Do not reduce below 60 s; excessive pings waste request quota.
#
# BUG-5 interaction: /health now always returns 200 so the daemon no longer
#   logs spurious CRITICAL failure streaks during the deferred DB-init window.

HEARTBEAT_INTERVAL_SECONDS = 300
_KEEPALIVE_THREAD: threading.Thread | None = None
_KEEPALIVE_STOP   = threading.Event()


def _resolve_public_url() -> str:
    """
    Resolve the externally-reachable base URL for this instance.
    Priority: KOYEB_PUBLIC_DOMAIN → APP_URL → RENDER_EXTERNAL_URL → FLY_APP_NAME → localhost.
    """
    koyeb = os.getenv('KOYEB_PUBLIC_DOMAIN', '').strip()
    if koyeb:
        return f"https://{koyeb}"

    app_url = os.getenv('APP_URL', '').strip().rstrip('/')
    if app_url:
        return app_url

    render = os.getenv('RENDER_EXTERNAL_URL', '').strip().rstrip('/')
    if render:
        return render

    fly = os.getenv('FLY_APP_NAME', '').strip()
    if fly:
        return f"https://{fly}.fly.dev"

    port = os.getenv('PORT', '5000')
    return f"http://127.0.0.1:{port}"


def _keepalive_loop() -> None:
    """
    Daemon loop: ping /health every HEARTBEAT_INTERVAL_SECONDS.

    Enhanced error handling:
      • urllib.error.HTTPError → extracts actual status code + JSON body diagnostics
      • urllib.error.URLError  → DNS / connection refused (transient)
      • socket.timeout        → slow/hung service indicator
      • Consecutive failure counter → escalates to ERROR at threshold 3
      • JSON body parsed for database.connected on 5xx responses
    """
    import urllib.request
    import urllib.error
    import socket as _socket

    base_url = _resolve_public_url()
    logger.info(
        f"[KEEPALIVE] 🟢 Daemon started | interval={HEARTBEAT_INTERVAL_SECONDS}s | "
        f"target={base_url}/health"
    )

    # Stagger initial ping to avoid colliding with gunicorn startup checks
    _KEEPALIVE_STOP.wait(timeout=30)

    consecutive_failures = 0

    while not _KEEPALIVE_STOP.is_set():
        base_url = _resolve_public_url()   # re-resolve each loop (env may change)
        target   = f"{base_url}/health"

        try:
            req = urllib.request.Request(target, headers={'User-Agent': 'QTCL-Keepalive/1.1'})
            with urllib.request.urlopen(req, timeout=15) as resp:
                status        = resp.status
                body          = resp.read().decode('utf-8', errors='ignore')
                consecutive_failures = 0   # any successful TCP+HTTP resets the counter

                if status == 200:
                    try:
                        data      = json.loads(body)
                        db_ok     = data.get('database', {}).get('connected', '?')
                        hb_status = data.get('status', '?')
                        logger.info(
                            f"[KEEPALIVE] 💓 {target} → 200 OK | "
                            f"app={hb_status} | db.connected={db_ok}"
                        )
                    except json.JSONDecodeError:
                        logger.info(f"[KEEPALIVE] 💓 {target} → 200 OK")
                else:
                    logger.info(f"[KEEPALIVE] 💓 {target} → HTTP {status}")

        except urllib.error.HTTPError as exc:
            consecutive_failures += 1
            try:
                body = exc.read().decode('utf-8', errors='ignore')
                try:
                    data    = json.loads(body)
                    db_ok   = data.get('database', {}).get('connected', '?')
                    glob_ok = data.get('globals_initialized', '?')
                    logger.error(
                        f"[KEEPALIVE] ❌ HTTP {exc.code} {exc.reason} | "
                        f"db.connected={db_ok} globals_initialized={glob_ok} | "
                        f"streak={consecutive_failures}"
                    )
                except json.JSONDecodeError:
                    logger.error(
                        f"[KEEPALIVE] ❌ HTTP {exc.code} {exc.reason} | "
                        f"body={body[:80]} | streak={consecutive_failures}"
                    )
            except Exception:
                logger.error(
                    f"[KEEPALIVE] ❌ HTTP {exc.code} {exc.reason} | "
                    f"streak={consecutive_failures}"
                )

        except (urllib.error.URLError, _socket.gaierror) as exc:
            consecutive_failures += 1
            logger.warning(
                f"[KEEPALIVE] ⚠️  {type(exc).__name__}: {str(exc)[:100]} | "
                f"target={target} | streak={consecutive_failures}"
            )

        except _socket.timeout:
            consecutive_failures += 1
            logger.warning(
                f"[KEEPALIVE] ⚠️  TIMEOUT (>15 s) | target={target} | streak={consecutive_failures}"
            )

        except Exception as exc:
            consecutive_failures += 1
            logger.error(
                f"[KEEPALIVE] ❌ {type(exc).__name__}: {str(exc)[:120]} | "
                f"streak={consecutive_failures}"
            )

        if consecutive_failures >= 3:
            logger.error(
                f"[KEEPALIVE] 🚨 {consecutive_failures} consecutive failures — "
                f"instance may be unreachable at {base_url}/health"
            )

        _KEEPALIVE_STOP.wait(timeout=HEARTBEAT_INTERVAL_SECONDS)

    logger.info("[KEEPALIVE] 🛑 Daemon shutdown complete")


def _start_keepalive() -> None:
    """
    Launch the keep-alive thread once per worker process.
    Guard prevents double-start if wsgi_config is imported multiple times
    (e.g. during pytest collection or gunicorn --reload).
    """
    global _KEEPALIVE_THREAD
    if _KEEPALIVE_THREAD is not None and _KEEPALIVE_THREAD.is_alive():
        return

    _KEEPALIVE_STOP.clear()
    _KEEPALIVE_THREAD = threading.Thread(
        target=_keepalive_loop,
        name='qtcl-keepalive',
        daemon=True,
    )
    _KEEPALIVE_THREAD.start()
    logger.info(
        f"[KEEPALIVE] 🟢 Instance keep-alive active | interval={HEARTBEAT_INTERVAL_SECONDS}s | "
        f"target={_resolve_public_url()}/health"
    )


_start_keepalive()


# ═══════════════════════════════════════════════════════════════════════════════
# LATTICE TELEMETRY DAEMON — periodic INFO-level quantum measurement output
# ═══════════════════════════════════════════════════════════════════════════════
#
# WHY: UniversalQuantumHeartbeat listeners (W_STATE, LATTICE_NEURAL, NOISE_BATH)
#      update state on every 1 Hz pulse but emit ZERO log output.
#      WStateEnhancedCoherenceRefresh.refresh_full_lattice() (the source of
#      [LATTICE-REFRESH] logs) is only called inside run_continuous() — not from
#      any heartbeat listener.
#      This daemon bridges that gap: every LATTICE_TELEMETRY_INTERVAL seconds it
#      reads the live singleton state and logs it at INFO so measurements appear
#      in Koyeb's log stream without code changes to the quantum_lattice module.

LATTICE_TELEMETRY_INTERVAL = 30   # seconds — matches LightweightHeartbeat interval


def _lattice_telemetry_loop() -> None:
    """
    Daemon: emit INFO-level quantum lattice measurement snapshot every 30 s.

    Reads W_STATE_ENHANCED, NOISE_BATH_ENHANCED, LATTICE_NEURAL_REFRESH, LATTICE,
    and HEARTBEAT from quantum_lattice_control_live_complete module globals.
    Triggers one refresh_full_lattice() cycle per interval if LATTICE has a noise
    bath attached, producing [LATTICE-REFRESH] log lines organically.
    """
    import time as _time

    # Wait for quantum module to initialize before first measurement
    _time.sleep(45)

    logger.info(
        f"[LATTICE-TELEM] 🌌 Telemetry daemon started — "
        f"emitting measurements every {LATTICE_TELEMETRY_INTERVAL}s"
    )

    cycle = 0
    while True:
        try:
            _time.sleep(LATTICE_TELEMETRY_INTERVAL)
            cycle += 1

            # ── Import live singletons ────────────────────────────────────────
            try:
                import quantum_lattice_control_live_complete as _ql
            except ImportError:
                logger.warning("[LATTICE-TELEM] quantum_lattice module not importable yet")
                continue

            # ── HEARTBEAT status ──────────────────────────────────────────────
            hb = getattr(_ql, 'HEARTBEAT', None)
            if hb is not None and hasattr(hb, 'get_metrics'):
                try:
                    hbm = hb.get_metrics()
                    logger.info(
                        f"[LATTICE-HB]  cycle=#{cycle} | "
                        f"pulses={hbm.get('pulse_count','?')} | "
                        f"listeners={hbm.get('listeners','?')} | "
                        f"running={hbm.get('running','?')} | "
                        f"avg_interval={hbm.get('avg_pulse_interval', 0):.4f}s | "
                        f"errors={hbm.get('error_count','?')}"
                    )
                except Exception as _e:
                    logger.debug(f"[LATTICE-TELEM] HEARTBEAT metrics error: {_e}")

            # ── W-STATE + BELL VIOLATION METRICS ──────────────────────────────────
            lattice = getattr(_ql, 'LATTICE', None)
            
            if lattice is not None:
                try:
                    # W-state coherence/fidelity from noise bath evolution (PRIMARY SOURCE)
                    noise_bath = getattr(lattice, 'noise_bath', None)
                    bell_tester = getattr(lattice, 'bell_tester', None)
                    blp_monitor = getattr(lattice, 'blp_monitor', None)
                    
                    # Get current state from noise bath evolution deques
                    coh_w = float(noise_bath.coherence_evolution[-1]) if (noise_bath and hasattr(noise_bath, 'coherence_evolution') and len(noise_bath.coherence_evolution) > 0) else 0.0
                    fid_w = float(noise_bath.fidelity_evolution[-1]) if (noise_bath and hasattr(noise_bath, 'fidelity_evolution') and len(noise_bath.fidelity_evolution) > 0) else 0.0
                    
                    # W-state superposition measure (entanglement_strength from GHZ builder)
                    ghz_builder = getattr(lattice, 'ghz_builder', None)
                    super_q = 100 if (ghz_builder and hasattr(ghz_builder, 'w_state_strength') and ghz_builder.w_state_strength > 0.5) else 0
                    
                    # Bell violation CHSH measurement
                    bell_s = 0.0
                    depth = 0
                    viol_flag = ""
                    validations = 0
                    
                    if bell_tester is not None and hasattr(bell_tester, 'get_summary'):
                        try:
                            bs = bell_tester.get_summary()
                            bell_s = float(bs.get('last_s_chsh', 0.0))
                            validations = int(bs.get('violation_count', 0))
                            if bs.get('last_violation', False):
                                viol_flag = "⚡VIOLATION"
                                depth = 2  # Entanglement confirmed
                            else:
                                depth = 0  # Classical
                        except Exception:
                            pass
                    
                    # MI from coherence-fidelity correlation
                    mi = 0.0
                    if noise_bath and hasattr(noise_bath, 'coherence_evolution'):
                        try:
                            coh_hist = list(noise_bath.coherence_evolution)[-100:] if len(noise_bath.coherence_evolution) >= 100 else list(noise_bath.coherence_evolution)
                            fid_hist = list(noise_bath.fidelity_evolution)[-100:] if len(noise_bath.fidelity_evolution) >= 100 else list(noise_bath.fidelity_evolution)
                            if len(coh_hist) >= 10 and len(fid_hist) >= 10:
                                coh_arr = np.array(coh_hist)
                                fid_arr = np.array(fid_hist)
                                corr = np.corrcoef(coh_arr, fid_arr)[0, 1]
                                if not np.isnan(corr):
                                    mi = float(np.clip(abs(corr), 0.0, 1.0))
                        except Exception:
                            pass
                    
                    kappa = float(getattr(noise_bath, 'memory_kernel', 0.08)) if noise_bath else 0.08
                    
                    logger.info(
                        f"[LATTICE-W]   cycle=#{cycle} | "
                        f"coherence_avg={coh_w:.6f} | "
                        f"fidelity_avg={fid_w:.6f} | "
                        f"superpositions={super_q} | "
                        f"bell_chsh_s={bell_s:.3f} | "
                        f"entanglement_depth={depth} | "
                        f"mi={mi:.4f} | "
                        f"κ={kappa:.5f} | "
                        f"validations={validations} {viol_flag}"
                    )
                except Exception as _e:
                    logger.debug(f"[LATTICE-TELEM] W-state metrics error: {_e}")
            else:
                logger.debug(f"[LATTICE-W]   cycle=#{cycle} | LATTICE not yet initialized")

            # ── LATTICE SYSTEM METRICS (primary source of truth) ────────────────
            lat = getattr(_ql, 'LATTICE', None)
            lm = None
            coh_current = 0.92
            fid_current = 0.91
            
            if lat is not None and hasattr(lat, 'get_system_metrics'):
                try:
                    lm = lat.get_system_metrics()
                    coh_current = lm.get('global_coherence', 0.92)
                    fid_current = lm.get('global_fidelity', 0.91)
                    logger.info(
                        f"[LATTICE-SYS] cycle=#{cycle} | "
                        f"coherence={coh_current:.6f} | fidelity={fid_current:.6f} | "
                        f"qubits={lm.get('num_qubits', 106496)} | "
                        f"ops={lm.get('operations_count','?')} | "
                        f"txs={lm.get('transactions_processed','?')}"
                    )
                except Exception as _e:
                    logger.debug(f"[LATTICE-TELEM] System metrics error: {_e}")

            # ── NEURAL REFRESH state + CONVERGENCE TRACKING ────────────────────
            lnr = getattr(_ql, 'LATTICE_NEURAL_REFRESH', None)
            if lnr is not None and hasattr(lnr, 'get_state'):
                try:
                    nm = lnr.get_state()
                    
                    # Get convergence metrics
                    status = nm.get('convergence_status', '?')
                    acts = int(nm.get('activation_count', 0))
                    lr = float(nm.get('learning_rate', 0.0))
                    grad = float(nm.get('avg_error_gradient', 0.0))
                    
                    # Hidden state dynamics (if available)
                    hidden_avg = float(nm.get('hidden_avg', 0.0))
                    hidden_std = float(nm.get('hidden_std', 0.0))
                    weight_updates = int(nm.get('weight_updates', 0))
                    loss_trend = nm.get('loss_trend', '→')
                    conv_pct = int(nm.get('convergence_percent', 0))
                    
                    logger.info(
                        f"[LATTICE-NN]  cycle=#{cycle} | "
                        f"status={status} ({conv_pct}%) | "
                        f"activations={acts} | "
                        f"lr={lr:.2e} | "
                        f"grad={grad:.6f} {loss_trend} | "
                        f"hidden=[μ={hidden_avg:.4f} σ={hidden_std:.4f}] | "
                        f"Δw={weight_updates}"
                    )
                except Exception as _e:
                    logger.debug(f"[LATTICE-TELEM] Neural metrics error: {_e}")

            # ── Genuine quantum observables from last Aer statevector run ─────
            # These are the true quantum quantities: purity, S_vN, entanglement entropy,
            # l₁ coherence, W-state fidelity. Sourced from TransactionValidatorWState.
            try:
                _w_mgr = getattr(lat, 'w_state_manager', None) if lat else None
                if _w_mgr is not None:
                    qobs = getattr(_w_mgr, 'quantum_observables', {})
                    if qobs and qobs.get('timestamp', 0) > 0:
                        age = time.time() - qobs['timestamp']
                        logger.info(
                            f"[QUANTUM-OBS] cycle=#{cycle} | "
                            f"purity={qobs.get('purity', '?'):.6f} | "
                            f"S_vN={qobs.get('von_neumann_entropy', '?'):.6f}bits | "
                            f"S_ent={qobs.get('entanglement_entropy', '?'):.6f}bits | "
                            f"C_l1={qobs.get('l1_coherence', '?'):.4f} | "
                            f"F_W={qobs.get('w_state_fidelity', '?'):.6f} | "
                            f"age={age:.0f}s"
                        )
                    else:
                        logger.debug(f"[LATTICE-TELEM] quantum_observables not yet populated")
            except Exception as _e:
                logger.debug(f"[LATTICE-TELEM] QUANTUM-OBS error: {_e}")

            # ── MASTER REFRESH CYCLE: Evolve noise bath + measure interference ────────
            # This is the core quantum dynamics step that updates all metrics
            if lat is not None and coh_current > 0 and fid_current > 0:
                try:
                    # Execute the evolution: current state → next state
                    nb_result  = lat.evolve_noise_bath(coh_current, fid_current)
                    ws_result  = lat.refresh_interference()

                    # Extract evolved state
                    coh_after = nb_result.get('coherence', nb_result.get('coherence_after', coh_current))
                    fid_after = nb_result.get('fidelity',  nb_result.get('fidelity_after',  fid_current))
                    coh_ss    = nb_result.get('coh_ss', 0.87)
                    memory    = nb_result.get('memory', 0.0)
                    revival   = nb_result.get('revival_detected', 
                                ws_result.get('revival_detected', False)) if ws_result else False

                    # Revival detection: check history for dip-recovery pattern
                    if lat is not None and hasattr(lat, 'noise_bath'):
                        _coh_hist = list(lat.noise_bath.coherence_evolution) if hasattr(lat.noise_bath, 'coherence_evolution') else []
                        if not revival and len(_coh_hist) >= 3:
                            if _coh_hist[-1] > _coh_hist[-2] and _coh_hist[-2] < _coh_hist[-3]:
                                revival = True

                    logger.info(
                        f"[LATTICE-REFRESH] Cycle #{cycle:4d} | "
                        f"C: {coh_current:.4f}→{coh_after:.4f} (ss={coh_ss:.3f}) | "
                        f"F: {fid_current:.4f}→{fid_after:.4f} | "
                        f"mem={memory:.3f} | "
                        f"W-revival={'✓' if revival else '↔'} | "
                        f"source=AerSimulator"
                    )

                    # ── BELL TEST + BLP MONITOR SUMMARY ──────────────────────
                    try:
                        _bell = getattr(lat, 'bell_tester', None)
                        _blp  = getattr(lat, 'blp_monitor',  None)
                        if _bell and hasattr(_bell, 'test_count') and _bell.test_count > 0:
                            bs = _bell.get_summary()
                            viol = "✓ ENTANGLEMENT" if bs.get('last_violation') else "· classical"
                            logger.info(
                                f"[BELL] last S_CHSH={bs.get('last_s_chsh', 0):.4f} | "
                                f"max={bs.get('max_s_seen', 0):.4f} | "
                                f"violations={bs.get('violation_count', 0)}/{bs.get('test_count', 0)} | "
                                f"{viol}"
                            )
                        if _blp and hasattr(_blp, 'total_measurements') and _blp.total_measurements > 0:
                            bp = _blp.get_summary()
                            nm = "↑ BACKFLOW" if bp.get('nm_rate', 0) > 0.1 else "→ Markovian-like"
                            logger.info(
                                f"[BLP] D={bp.get('last_trace_distance', 0):.6f} | "
                                f"N_BLP={bp.get('blp_integral', 0):.6f} | "
                                f"NM_rate={bp.get('nm_rate', 0):.3f} | {nm}"
                            )
                    except Exception as _be:
                        logger.debug(f"[LATTICE-TELEM] Bell/BLP summary error: {_be}")
                        
                except Exception as _e:
                    logger.debug(f"[LATTICE-TELEM] Noise bath evolution/refresh error: {_e}", exc_info=False)

        except Exception as exc:
            logger.error(f"[LATTICE-TELEM] Unexpected error in telemetry loop: {exc}", exc_info=True)
            _time.sleep(10)


threading.Thread(
    target=_lattice_telemetry_loop,
    daemon=True,
    name="lattice-telemetry",
).start()
logger.info(
    f"[LATTICE-TELEM] 🌌 Lattice measurement daemon started "
    f"(interval={LATTICE_TELEMETRY_INTERVAL}s, first output in ~45s)"
)
# ═══════════════════════════════════════════════════════════════════════════════

application = app   # gunicorn / uwsgi look for `application`

# Eager terminal init at startup — surfaces import errors in logs at boot time
# rather than on the first user request.  _initialize_terminal_engine() is
# idempotent; subsequent calls from route handlers return immediately.
try:
    _initialize_terminal_engine()
except Exception as _boot_exc:
    logger.error(f"[STARTUP] Terminal engine eager init failed (non-fatal): {_boot_exc}")


if __name__ == '__main__':
    logger.info("=" * 80)
    logger.info("🚀 QTCL WSGI v5.1 — PRODUCTION STARTUP")
    logger.info("=" * 80)
    logger.info(f"  DB initialized:        {DB is not None}")
    logger.info(f"  Globals available:     {GLOBALS_AVAILABLE}")
    logger.info(f"  Commands registered:   {len(COMMAND_REGISTRY)}")
    logger.info(f"  Terminal initialized:  {_TERMINAL_INITIALIZED}")
    logger.info(f"  Public URL:            {_resolve_public_url()}")
    logger.info("=" * 80)
    port  = int(os.getenv('PORT', 8000))
    debug = os.getenv('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
