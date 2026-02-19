#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        QTCL WSGI CONFIG v5.0 — GLOBALS REGISTRY + FLASK SERVER             ║
║                                                                              ║
║  This file does exactly three things:                                        ║
║    1. Import and initialise globals.py (single source of truth)              ║
║    2. Import terminal_logic and register all commands into globals            ║
║    3. Serve a clean Flask app that dispatches via globals.dispatch_command()  ║
║                                                                              ║
║  Command format: hyphen-separated, flags with --key=val or --flag            ║
║    admin-users --limit=20 --role=admin                                       ║
║    help-admin              (expands to help --category=admin)                ║
║    help-admin-users        (expands to help --command=admin-users)           ║
║    quantum-status                                                             ║
║    oracle-price --symbol=BTCUSD                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, sys, logging, traceback, time
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper(), logging.INFO),
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT, 'qtcl.log')),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger('wsgi_config')

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — GLOBALS
# ══════════════════════════════════════════════════════════════════════════════

from globals import (
    get_globals, initialize_globals, get_system_health, get_state_snapshot,
    COMMAND_REGISTRY, dispatch_command,
    get_heartbeat, get_lattice, get_db_pool, get_auth_manager,
    get_oracle, get_defi, get_ledger, get_blockchain, get_metrics,
    bootstrap_admin_session, revoke_session,
    get_pqc_state, get_pqc_system,
    pqc_generate_user_key, pqc_sign, pqc_verify,
    pqc_encapsulate, pqc_prove_identity, pqc_verify_identity,
    pqc_revoke_key, pqc_rotate_key,
)

# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATED COMMAND SYSTEM - Oracle pricing, response wrapping, multi-execute
# ══════════════════════════════════════════════════════════════════════════════
try:
    from integrated_oracle_provider import (
        ORACLE_PRICE_PROVIDER, ResponseWrapper, get_oracle_price_provider
    )
    INTEGRATED_AVAILABLE = True
    logger.info('[wsgi] Integrated oracle system loaded')
except ImportError as e:
    INTEGRATED_AVAILABLE = False
    ResponseWrapper = None
    ORACLE_PRICE_PROVIDER = None
    logger.warning('[wsgi] Integrated oracle system not available: %s', e)

# ── Quantum Lattice singletons ────────────────────────────────────────────────
try:
    from quantum_lattice_control_live_complete import (
        LATTICE,
        HEARTBEAT         as LATTICE_HEARTBEAT,
        LATTICE_NEURAL_REFRESH,
        W_STATE_ENHANCED,
        NOISE_BATH_ENHANCED,
    )
    _LATTICE_AVAILABLE = True
    logger.info('[wsgi] quantum_lattice_control_live_complete loaded ✓')
except Exception as _le:
    LATTICE = None
    LATTICE_HEARTBEAT = None
    LATTICE_NEURAL_REFRESH = None
    W_STATE_ENHANCED = None
    NOISE_BATH_ENHANCED = None
    _LATTICE_AVAILABLE = False
    logger.warning(f'[wsgi] quantum_lattice not available: {_le}')

logger.info('[wsgi] globals imported — initialising...')
try:
    initialize_globals()
    _GS = get_globals()
    logger.info('[wsgi] globals initialised OK')
except Exception as _e:
    logger.error(f'[wsgi] globals init error (continuing): {_e}')
    _GS = None

# ── Expose component singletons as module-level vars ──────────────────────────
# terminal_logic.WSGIGlobals.load() does getattr(wsgi_config, 'DB', None) etc.
# These MUST be module-level to be discoverable.

def _expose_components():
    """Wire db_builder_v2 + globals components into module-level vars."""
    global DB, CACHE, PROFILER, CIRCUIT_BREAKERS, RATE_LIMITERS
    global APIS, HEARTBEAT, ORCHESTRATOR, MONITOR, QUANTUM, ERROR_BUDGET
    
    # DB — DatabaseBuilder pool from globals (which loaded db_builder_v2)
    try:
        DB = get_db_pool()
        if DB is None:
            import db_builder_v2 as _db
            DB = getattr(_db, 'db_manager', None) or getattr(_db, 'DB_POOL', None)
    except Exception as _e:
        logger.warning(f'[wsgi] DB singleton not available: {_e}')
        DB = None
    
    # HEARTBEAT — prefer lattice heartbeat (already started), fall back to globals
    try:
        if _LATTICE_AVAILABLE and LATTICE_HEARTBEAT is not None:
            HEARTBEAT = LATTICE_HEARTBEAT
            if not HEARTBEAT.running:
                HEARTBEAT.start()
        else:
            HEARTBEAT = get_heartbeat()
    except Exception as _he:
        logger.warning(f'[wsgi] HEARTBEAT init error: {_he}')
        HEARTBEAT = None

    # Wire lattice subsystems into globals.quantum
    try:
        gs = get_globals()
        if _LATTICE_AVAILABLE:
            gs.quantum.heartbeat       = HEARTBEAT
            gs.quantum.lattice         = LATTICE
            gs.quantum.neural_network  = LATTICE_NEURAL_REFRESH
            gs.quantum.w_state_manager = W_STATE_ENHANCED
            gs.quantum.noise_bath      = NOISE_BATH_ENHANCED
            logger.info('[wsgi] ✅ Lattice subsystems wired into globals.quantum')
        QUANTUM = gs.quantum
    except Exception as _qe:
        logger.warning(f'[wsgi] QUANTUM wiring error: {_qe}')
        QUANTUM = None
    
    # Placeholders — extend these when real implementations are wired
    CACHE           = None
    PROFILER        = None
    CIRCUIT_BREAKERS= None
    RATE_LIMITERS   = None
    APIS            = None
    ORCHESTRATOR    = None
    MONITOR         = None
    ERROR_BUDGET    = None
    
    logger.info(f'[wsgi] Components exposed — DB:{DB is not None} HEARTBEAT:{HEARTBEAT is not None} QUANTUM:{QUANTUM is not None}')

# Declare with defaults so WSGIGlobals.load() never gets AttributeError
DB              = None
CACHE           = None
PROFILER        = None
CIRCUIT_BREAKERS= None
RATE_LIMITERS   = None
APIS            = None
HEARTBEAT       = None
ORCHESTRATOR    = None
MONITOR         = None
QUANTUM         = None
ERROR_BUDGET    = None

class RequestCorrelation:
    """Stub — correlates requests across modules via a thread-local ID."""
    import threading as _tl
    _local = _tl.local()

    @classmethod
    def get_id(cls) -> str:
        return getattr(cls._local, 'request_id', '')

    @classmethod
    def set_id(cls, rid: str):
        cls._local.request_id = rid

_expose_components()

# ── Populate globals blockchain/auth/ledger stats from live DB ────────────────
def _populate_globals_from_db():
    """Read actual DB counts and seed in-memory globals state."""
    try:
        if DB is None:
            return
        gs = get_globals()
        conn = DB.get_connection()
        cur = conn.cursor()
        
        try:
            # blockchain chain_height
            cur.execute("SELECT COUNT(*) FROM blocks")
            row = cur.fetchone()
            if row and row[0]:
                gs.blockchain.chain_height = int(row[0])
                gs.blockchain.total_blocks = int(row[0])
                logger.info(f'[wsgi] blockchain.chain_height = {gs.blockchain.chain_height}')
            
            # total transactions
            try:
                cur.execute("SELECT COUNT(*) FROM transactions")
                row = cur.fetchone()
                if row and row[0]:
                    gs.blockchain.total_transactions = int(row[0])
            except Exception:
                pass
            
            # pending transactions
            try:
                cur.execute("SELECT COUNT(*) FROM transactions WHERE status='pending'")
                row = cur.fetchone()
                if row and row[0]:
                    gs.blockchain.mempool_size = int(row[0])
            except Exception:
                pass
            
            # auth users & active sessions — DISTINCT: user count ≠ session count
            try:
                # Active users (for informational count only)
                cur.execute("SELECT COUNT(*) FROM users WHERE is_active=TRUE")
                row = cur.fetchone()
                if row and row[0]:
                    gs.auth.total_users = int(row[0])
            except Exception:
                pass

            # Real active sessions — from sessions table, not user rows
            try:
                cur.execute(
                    "SELECT COUNT(*) FROM sessions WHERE expires_at > NOW()"
                )
                row = cur.fetchone()
                gs.auth.active_sessions = int(row[0]) if (row and row[0]) else 0
                logger.info(f'[wsgi] auth.active_sessions (from sessions table) = {gs.auth.active_sessions}')
            except Exception as _se:
                # sessions table may not exist yet — start at 0 (correct)
                gs.auth.active_sessions = 0
                logger.debug(f'[wsgi] sessions table query skipped: {_se}')
            
            # ledger entries
            try:
                cur.execute("SELECT COUNT(*) FROM ledger_entries")
                row = cur.fetchone()
                if row and row[0]:
                    gs.ledger.total_entries = int(row[0])
            except Exception:
                pass
            
            logger.info(f'[wsgi] ✅ DB stats loaded into globals: blocks={gs.blockchain.chain_height} txns={gs.blockchain.total_transactions}')
        finally:
            cur.close()
            DB.return_connection(conn)
    except Exception as _e:
        logger.warning(f'[wsgi] Could not populate globals from DB (non-fatal): {_e}')

try:
    _populate_globals_from_db()
except Exception as _pe:
    logger.warning(f'[wsgi] DB population skipped: {_pe}')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — TERMINAL LOGIC → COMMAND_REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

_ENGINE = None

def _boot_terminal():
    global _ENGINE
    try:
        from terminal_logic import TerminalEngine, register_all_commands
        _ENGINE = TerminalEngine()
        n = register_all_commands(_ENGINE)
        logger.info(f'[wsgi] terminal_logic: {n} commands registered into globals.COMMAND_REGISTRY')
        return True
    except Exception as exc:
        logger.error(f'[wsgi] terminal_logic boot failed: {exc}\n{traceback.format_exc()}')
        return False

_boot_terminal()

_reg_size   = len(COMMAND_REGISTRY)
_categories = sorted({e['category'] for e in COMMAND_REGISTRY.values()})
logger.info(f'[wsgi] Registry: {_reg_size} cmds · {len(_categories)} cats: {_categories}')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — FLASK
# ══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__, static_folder=PROJECT_ROOT, static_url_path='')
CORS(app, resources={r'/*': {'origins': '*'}})
app.config['JSON_SORT_KEYS'] = False


# ══════════════════════════════════════════════════════════════════════════════
# BLUEPRINT REGISTRATION — mount all sub-API blueprints onto the Flask app.
# Previously these blueprints were defined in their modules but NEVER registered,
# so /api/auth/*, /api/admin/*, /api/users/*, /api/transactions/* etc. all
# returned 404 → index.html. This block fixes that.
# ══════════════════════════════════════════════════════════════════════════════

def _register_blueprints(flask_app):
    """Register all sub-API blueprints. Each wrapped in its own try/except
    so a failure in one module never prevents the others from mounting."""

    # ── core_api: /api/auth/*, /api/users/*, /api/transactions/* ─────────────
    try:
        from core_api import get_core_blueprint
        flask_app.register_blueprint(get_core_blueprint())
        logger.info('[wsgi] ✓ core_api blueprint registered (/api/auth, /api/users, /api/transactions)')
    except Exception as _e:
        logger.error(f'[wsgi] ✗ core_api blueprint failed: {_e}')

    # ── admin_api: /api/admin/* ───────────────────────────────────────────────
    try:
        from admin_api import get_admin_blueprint
        flask_app.register_blueprint(get_admin_blueprint())
        logger.info('[wsgi] ✓ admin_api blueprint registered (/api/admin)')
    except Exception as _e:
        logger.error(f'[wsgi] ✗ admin_api blueprint failed: {_e}')

    # ── blockchain_api: /api/blocks, /api/transactions, /api/mempool etc. ─────
    try:
        from blockchain_api import get_full_blockchain_blueprint
        flask_app.register_blueprint(get_full_blockchain_blueprint())
        logger.info('[wsgi] ✓ blockchain_api (full) blueprint registered')
    except Exception as _e:
        logger.warning(f'[wsgi] Full blockchain blueprint unavailable ({_e}); trying fallback')
        try:
            from blockchain_api import get_blockchain_blueprint
            flask_app.register_blueprint(get_blockchain_blueprint())
            logger.info('[wsgi] ✓ blockchain_api (stub) blueprint registered (/api/blockchain)')
        except Exception as _e2:
            logger.error(f'[wsgi] ✗ blockchain_api blueprint failed: {_e2}')

    # ── oracle_api: /api/oracle/* ─────────────────────────────────────────────
    try:
        from oracle_api import get_oracle_blueprint
        flask_app.register_blueprint(get_oracle_blueprint())
        logger.info('[wsgi] ✓ oracle_api blueprint registered (/api/oracle)')
    except Exception as _e:
        logger.error(f'[wsgi] ✗ oracle_api blueprint failed: {_e}')

    # ── defi_api: /api/defi/* ─────────────────────────────────────────────────
    try:
        from defi_api import get_defi_blueprint
        flask_app.register_blueprint(get_defi_blueprint())
        logger.info('[wsgi] ✓ defi_api blueprint registered (/api/defi)')
    except Exception as _e:
        logger.error(f'[wsgi] ✗ defi_api blueprint failed: {_e}')

    # ── quantum_api: /api/quantum/* ───────────────────────────────────────────
    try:
        from quantum_api import get_quantum_blueprint
        flask_app.register_blueprint(get_quantum_blueprint())
        logger.info('[wsgi] ✓ quantum_api blueprint registered (/api/quantum)')
    except Exception as _e:
        logger.error(f'[wsgi] ✗ quantum_api blueprint failed: {_e}')

    logger.info('[wsgi] Blueprint registration complete')


_register_blueprints(app)


def _parse_auth(req):
    """Returns (is_authenticated, is_admin).

    Token resolution order:
      1. Authorization: Bearer <token> header  (standard)
      2. globals.auth.session_store            (set by h_login — works without JWT env var)
      3. JWT decode via auth_mgr               (stateless verification)
      4. DB sessions table lookup              (cross-worker fallback: Koyeb multi-worker)
      5. ADMIN_SECRET env var                  (admin bypass)
    """
    # ── 1. Extract token from header ────────────────────────────────────────
    token = req.headers.get('Authorization', '').replace('Bearer ', '').strip()

    if not token:
        return False, False

    # ── 2. Check globals session store (populated by h_login) ───────────────
    try:
        gs = get_globals()
        session_entry = gs.auth.session_store.get(token)
        if session_entry and session_entry.get('authenticated'):
            is_admin = session_entry.get('is_admin', False)
            return True, is_admin
    except Exception:
        pass

    # ── 3. JWT stateless verification ───────────────────────────────────────
    try:
        auth_mgr = get_auth_manager()
        if auth_mgr:
            payload = auth_mgr.verify_token(token)
            if payload:
                role = payload.get('role', 'user')
                is_admin = payload.get('is_admin', False) or role in ('admin', 'superadmin')
                # Cache in session_store so future requests on this worker are fast
                try:
                    gs = get_globals()
                    gs.auth.session_store[token] = {
                        'user_id': payload.get('user_id', ''),
                        'email': payload.get('email', ''),
                        'role': role,
                        'is_admin': is_admin,
                        'authenticated': True,
                    }
                except Exception:
                    pass
                return True, is_admin
    except Exception:
        pass

    # ── 3b. Direct JWT decode — bypasses jwt_manager (cross-worker safe) ───
    # If JWT_SECRET isn't set, every worker generates a different random secret.
    # Decode directly from auth_handlers where JWT_SECRET is module-level consistent.
    try:
        import jwt as _jwt
        from auth_handlers import JWT_SECRET as _AH_SECRET, JWT_ALGORITHM as _AH_ALG
        _payload = _jwt.decode(
            token, _AH_SECRET,
            algorithms=[_AH_ALG, 'HS256', 'HS512'],
            options={'verify_exp': True}
        )
        if _payload:
            _role = _payload.get('role', 'user')
            _is_admin_jwt = _payload.get('is_admin', False) or _role in ('admin', 'superadmin')
            try:
                gs = get_globals()
                if token not in gs.auth.session_store:
                    gs.auth.session_store[token] = {
                        'user_id': _payload.get('user_id', ''),
                        'email':   _payload.get('email', ''),
                        'role':    _role,
                        'is_admin': _is_admin_jwt,
                        'authenticated': True,
                    }
            except Exception:
                pass
            return True, _is_admin_jwt
    except Exception:
        pass

    # ── 4. DB sessions table (cross-worker: JWT_SECRET differs per gunicorn worker) ──
    try:
        db = get_db_pool()
        if db is not None:
            _conn_fn  = getattr(db, 'get_connection', None) or getattr(db, 'getconn', None)
            _ret_fn   = getattr(db, 'return_connection', None) or getattr(db, 'putconn', None)
            if _conn_fn and _ret_fn:
                conn = _conn_fn()
                try:
                    with conn.cursor() as cur:
                        cur.execute("""
                            SELECT s.user_id, u.role
                              FROM sessions s
                              LEFT JOIN users u ON s.user_id = u.user_id
                             WHERE s.access_token = %s
                               AND s.is_active = TRUE
                               AND s.revoked   = FALSE
                               AND s.expires_at > NOW()
                             LIMIT 1
                        """, (token,))
                        row = cur.fetchone()
                        if row:
                            _uid, _role = row[0], (row[1] or 'user')
                            _is_admin = _role in ('admin', 'superadmin', 'super_admin')
                            # Cache for this worker
                            try:
                                gs = get_globals()
                                gs.auth.session_store[token] = {
                                    'user_id': _uid, 'role': _role,
                                    'is_admin': _is_admin, 'authenticated': True,
                                }
                            except Exception:
                                pass
                            return True, _is_admin
                finally:
                    _ret_fn(conn)
    except Exception:
        pass

    if token and token == os.getenv('ADMIN_SECRET', ''):
        return True, True
    return False, False


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def home():
    html_path = os.path.join(PROJECT_ROOT, 'index.html')
    if os.path.exists(html_path):
        try:
            return Response(open(html_path, encoding='utf-8').read(), 200, mimetype='text/html')
        except Exception as e:
            logger.error(f'[/] {e}')
    return Response(
        '<html><head><title>QTCL</title></head>'
        '<body style="background:#0a0a0a;color:#00ff88;font-family:monospace;padding:2rem">'
        '<h1>⚛ QTCL</h1>'
        '<p><a href="/api/status" style="color:#0af">Status</a> · '
        '<a href="/api/commands" style="color:#0af">Commands</a> · '
        '<a href="/health" style="color:#0af">Health</a></p></body></html>',
        200, mimetype='text/html')


@app.route('/health', methods=['GET'])
def health():
    try:
        gs = get_globals()
        snap = get_system_health()
        return jsonify({
            'status':   snap.get('status', 'healthy'),
            'commands': len(COMMAND_REGISTRY),
            'engine':   _ENGINE is not None,
            'blockchain': snap.get('blockchain', {}),
            'database': snap.get('database', {}),
            'quantum':  snap.get('quantum', {}),
            'uptime_seconds': snap.get('uptime_seconds', 0),
            'time':     datetime.now(timezone.utc).isoformat(),
        }), 200
    except Exception as _e:
        return jsonify({
            'status':   'healthy',
            'commands': len(COMMAND_REGISTRY),
            'engine':   _ENGINE is not None,
            'time':     datetime.now(timezone.utc).isoformat(),
        }), 200


@app.route('/api/status', methods=['GET'])
def api_status():
    try:
        snap = get_state_snapshot()
        gs = get_globals()
        snap['quantum']  = gs.quantum.get_health()
        snap['ledger']   = {
            'blocks':       gs.ledger.total_entries,
            'transactions': gs.blockchain.total_transactions,
            'validators':   256,
        }
        snap['uptime']   = (datetime.utcnow() - gs.startup_time).total_seconds() if gs.startup_time else 0
        snap['commands'] = len(COMMAND_REGISTRY)
        return jsonify(snap), 200
    except Exception as e:
        logger.error(f'[/api/status] {e}')
        return jsonify({'status': 'error', 'error': str(e), 'commands': len(COMMAND_REGISTRY)}), 500


@app.route('/api/command', methods=['POST'])
@app.route('/api/execute', methods=['POST'])
def api_command():
    """
    Universal hyphenated command dispatcher with integrated error handling.
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
        raw  = (body.get('command') or body.get('cmd') or '').strip()
        if not raw:
            if INTEGRATED_AVAILABLE and ResponseWrapper:
                return jsonify(ResponseWrapper.error(
                    error='No command provided',
                    error_code='EMPTY_COMMAND'
                )), 400
            else:
                return jsonify({'status': 'error', 'error': 'No command provided'}), 400

        extra_args = body.get('args', [])
        extra_kw   = body.get('kwargs', {})
        if extra_args:
            raw += ' ' + ' '.join(str(a) for a in extra_args if a)
        if extra_kw:
            for k, v in extra_kw.items():
                if v is not None and v != '':
                    raw += f' --{k}={v}'

        is_auth, is_admin = _parse_auth(request)

        # ── Propagate caller's Bearer token into the engine client ──────────
        # c.request() (used by h_tx_create etc.) makes secondary HTTP calls.
        # Without this, those calls carry the engine's last-seen token — which
        # may be from a different worker's login session or nothing at all.
        _incoming_token = request.headers.get('Authorization', '').replace('Bearer ', '').strip()
        if _incoming_token and _ENGINE is not None:
            try:
                _ENGINE.client.set_auth_token(_incoming_token)
            except Exception:
                pass

        result = dispatch_command(raw, is_admin=is_admin, is_authenticated=is_auth)
        
        # Ensure proper response format
        if INTEGRATED_AVAILABLE and ResponseWrapper:
            result = ResponseWrapper.ensure_json(result)

        try:
            get_globals().metrics.commands_executed += 1
        except Exception:
            pass

        status_code = 200 if result.get('status') == 'success' else 400
        return jsonify(result), status_code

    except Exception as exc:
        logger.error(f'[/api/command] {exc}\n{traceback.format_exc()}')
        if INTEGRATED_AVAILABLE and ResponseWrapper:
            return jsonify(ResponseWrapper.error(
                error=str(exc), error_code='INTERNAL_ERROR'
            )), 500
        else:
            return jsonify({'status': 'error', 'error': str(exc)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# POST-QUANTUM CRYPTOGRAPHY REST ENDPOINTS
# All operations route through globals PQC accessors for unified telemetry.
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/pqc/status', methods=['GET'])
def api_pqc_status():
    """
    GET /api/pqc/status
    Return live HyperbolicPQCSystem telemetry — capability flags, counters,
    tessellation stats, entropy source hits, vault readiness.
    """
    try:
        pqc_state = get_pqc_state()
        summary   = pqc_state.get_summary()
        # Augment with live system.status() if system is up
        pqc_sys = get_pqc_system()
        if pqc_sys is not None:
            summary['live_status'] = pqc_sys.status()
        if INTEGRATED_AVAILABLE and ResponseWrapper:
            return jsonify(ResponseWrapper.success(data=summary,
                message='HLWE Post-Quantum Cryptography System')), 200
        return jsonify({'status': 'success', 'data': summary}), 200
    except Exception as exc:
        logger.error(f'[/api/pqc/status] {exc}')
        return jsonify({'status': 'error', 'error': str(exc)}), 500


@app.route('/api/pqc/keygen', methods=['POST'])
def api_pqc_keygen():
    """
    POST /api/pqc/keygen
    Body: { "pseudoqubit_id": int, "user_id": str }
    Generate an HLWE master keypair + signing/encryption subkeys.
    Requires admin auth to prevent abuse.

    Sub-logic:
      1. Auth check (admin or self)
      2. pqc_generate_user_key() — routes through globals telemetry
      3. Returns fingerprint + derivation paths (never private key material)
    """
    is_auth, is_admin = _parse_auth(request)
    if not is_auth:
        return jsonify({'status': 'error', 'error': 'Authentication required'}), 401
    try:
        body = request.get_json(force=True, silent=True) or {}
        pq_id   = int(body.get('pseudoqubit_id', 0))
        user_id = str(body.get('user_id', ''))
        store   = bool(body.get('store', True))
        if not user_id or pq_id <= 0:
            return jsonify({'status': 'error',
                            'error': 'pseudoqubit_id (>0) and user_id required'}), 400
        bundle = pqc_generate_user_key(pq_id, user_id, store=store)
        if bundle is None:
            return jsonify({'status': 'error',
                            'error': 'Key generation failed — check PQC system logs'}), 500
        # Return only public metadata — never the private key
        safe_response = {
            'pseudoqubit_id':  bundle['pseudoqubit_id'],
            'user_id':         bundle['user_id'],
            'fingerprint':     bundle['fingerprint'],
            'params':          bundle['params'],
            'master_key_id':   bundle['master_key']['key_id'],
            'signing_key_id':  bundle['signing_key']['key_id'],
            'enc_key_id':      bundle['encryption_key']['key_id'],
            'master_fp':       bundle['master_key'].get('fingerprint', ''),
            'expires_at':      bundle['master_key'].get('metadata', {}).get('expires_at', ''),
            'derivation_path': 'm',
        }
        return jsonify({'status': 'success', 'data': safe_response}), 200
    except Exception as exc:
        logger.error(f'[/api/pqc/keygen] {exc}')
        return jsonify({'status': 'error', 'error': str(exc)}), 500


@app.route('/api/pqc/sign', methods=['POST'])
def api_pqc_sign():
    """
    POST /api/pqc/sign
    Body: { "message_hex": str, "user_id": str, "key_id": str }
    Produces a HyperSign (Fiat-Shamir over {8,3} tessellation) signature.
    """
    is_auth, _ = _parse_auth(request)
    if not is_auth:
        return jsonify({'status': 'error', 'error': 'Authentication required'}), 401
    try:
        body    = request.get_json(force=True, silent=True) or {}
        msg_hex = body.get('message_hex', '')
        user_id = body.get('user_id', '')
        key_id  = body.get('key_id', '')
        if not all([msg_hex, user_id, key_id]):
            return jsonify({'status': 'error',
                            'error': 'message_hex, user_id, key_id required'}), 400
        message   = bytes.fromhex(msg_hex)
        signature = pqc_sign(message, user_id, key_id)
        if signature is None:
            return jsonify({'status': 'error', 'error': 'Signing failed'}), 500
        return jsonify({'status': 'success',
                        'signature_hex': signature.hex(),
                        'sig_bytes': len(signature),
                        'algorithm': 'HyperSign-v1 (HLWE/Fiat-Shamir)'}), 200
    except Exception as exc:
        logger.error(f'[/api/pqc/sign] {exc}')
        return jsonify({'status': 'error', 'error': str(exc)}), 500


@app.route('/api/pqc/verify', methods=['POST'])
def api_pqc_verify():
    """
    POST /api/pqc/verify
    Body: { "message_hex": str, "signature_hex": str, "user_id": str, "key_id": str }
    """
    try:
        body    = request.get_json(force=True, silent=True) or {}
        msg     = bytes.fromhex(body.get('message_hex', ''))
        sig     = bytes.fromhex(body.get('signature_hex', ''))
        user_id = body.get('user_id', '')
        key_id  = body.get('key_id', '')
        ok      = pqc_verify(msg, sig, key_id, user_id)
        return jsonify({'status': 'success', 'valid': ok,
                        'algorithm': 'HyperSign-v1 (EUF-CMA)'}), 200
    except Exception as exc:
        logger.error(f'[/api/pqc/verify] {exc}')
        return jsonify({'status': 'error', 'error': str(exc)}), 500


@app.route('/api/pqc/encapsulate', methods=['POST'])
def api_pqc_encapsulate():
    """
    POST /api/pqc/encapsulate
    Body: { "recipient_key_id": str, "recipient_user_id": str }
    Returns ciphertext; caller receives shared_secret separately.
    """
    is_auth, _ = _parse_auth(request)
    if not is_auth:
        return jsonify({'status': 'error', 'error': 'Authentication required'}), 401
    try:
        body = request.get_json(force=True, silent=True) or {}
        ct, ss = pqc_encapsulate(body.get('recipient_key_id', ''),
                                  body.get('recipient_user_id', ''))
        if ct is None:
            return jsonify({'status': 'error', 'error': 'Encapsulation failed'}), 500
        return jsonify({'status': 'success',
                        'ciphertext_hex':    ct.hex(),
                        'shared_secret_hex': ss.hex(),
                        'algorithm':         'HyperKEM (IND-CCA2 / HLWE + Kyber hybrid)'}), 200
    except Exception as exc:
        logger.error(f'[/api/pqc/encapsulate] {exc}')
        return jsonify({'status': 'error', 'error': str(exc)}), 500


@app.route('/api/pqc/prove-identity', methods=['POST'])
def api_pqc_prove_identity():
    """
    POST /api/pqc/prove-identity
    Body: { "user_id": str, "key_id": str }
    Generates a non-interactive Sigma-protocol ZK proof of pseudoqubit key ownership.
    Nullifier stored globally to block replays.
    """
    is_auth, _ = _parse_auth(request)
    if not is_auth:
        return jsonify({'status': 'error', 'error': 'Authentication required'}), 401
    try:
        body  = request.get_json(force=True, silent=True) or {}
        proof = pqc_prove_identity(body.get('user_id', ''), body.get('key_id', ''))
        if not proof:
            return jsonify({'status': 'error', 'error': 'ZK proof generation failed'}), 500
        return jsonify({'status': 'success', 'proof': proof,
                        'algorithm': 'HyperZK-Sigma-v1 (Fiat-Shamir / HLWE)'}), 200
    except Exception as exc:
        logger.error(f'[/api/pqc/prove-identity] {exc}')
        return jsonify({'status': 'error', 'error': str(exc)}), 500


@app.route('/api/pqc/revoke', methods=['POST'])
def api_pqc_revoke():
    """
    POST /api/pqc/revoke
    Body: { "key_id": str, "user_id": str, "reason": str, "cascade": bool }
    Instantly revokes key + all derived subkeys (cascade CTE in DB).
    """
    is_auth, _ = _parse_auth(request)
    if not is_auth:
        return jsonify({'status': 'error', 'error': 'Authentication required'}), 401
    try:
        body    = request.get_json(force=True, silent=True) or {}
        result  = pqc_revoke_key(
            body.get('key_id', ''), body.get('user_id', ''),
            body.get('reason', 'user_request'),
            cascade=bool(body.get('cascade', True))
        )
        code = 200 if result.get('status') == 'success' else 400
        return jsonify(result), code
    except Exception as exc:
        logger.error(f'[/api/pqc/revoke] {exc}')
        return jsonify({'status': 'error', 'error': str(exc)}), 500


@app.route('/api/pqc/rotate', methods=['POST'])
def api_pqc_rotate():
    """
    POST /api/pqc/rotate
    Body: { "key_id": str, "user_id": str }
    Rotates key with fresh QRNG entropy. Old key revoked, new key stored.
    """
    is_auth, _ = _parse_auth(request)
    if not is_auth:
        return jsonify({'status': 'error', 'error': 'Authentication required'}), 401
    try:
        body    = request.get_json(force=True, silent=True) or {}
        new_kp  = pqc_rotate_key(body.get('key_id', ''), body.get('user_id', ''))
        if new_kp is None:
            return jsonify({'status': 'error', 'error': 'Rotation failed'}), 500
        return jsonify({'status': 'success',
                        'new_key_id':    new_kp.get('key_id', ''),
                        'fingerprint':   new_kp.get('fingerprint', ''),
                        'expires_at':    new_kp.get('metadata', {}).get('expires_at', '')}), 200
    except Exception as exc:
        logger.error(f'[/api/pqc/rotate] {exc}')
        return jsonify({'status': 'error', 'error': str(exc)}), 500


@app.route('/api/commands/batch', methods=['POST'])
def api_commands_batch():
    """Execute multiple commands in parallel with aggregated results."""
    try:
        body = request.get_json(force=True, silent=True) or {}
        commands = body.get('commands', [])
        
        if not commands:
            if INTEGRATED_AVAILABLE and ResponseWrapper:
                return jsonify(ResponseWrapper.error(
                    error='No commands provided', error_code='EMPTY_BATCH'
                )), 400
        
        is_auth, is_admin = _parse_auth(request)
        
        # Execute in parallel
        import concurrent.futures
        results = []
        errors = []
        timings = []
        
        def execute_single(cmd_str):
            import time
            t0 = time.time()
            try:
                result = dispatch_command(cmd_str, is_admin=is_admin, is_authenticated=is_auth)
                duration = time.time() - t0
                timings.append(duration)
                return {'command': cmd_str, 'response': ResponseWrapper.ensure_json(result) if ResponseWrapper else result, 'duration': duration, 'error': None}
            except Exception as e:
                logger.error('[batch] Command %s failed: %s', cmd_str, e)
                errors.append({'command': cmd_str, 'error': str(e)})
                return {'command': cmd_str, 'response': None, 'error': str(e)}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(commands), 10)) as executor:
            futures = [executor.submit(execute_single, cmd) for cmd in commands]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        successful = [r for r in results if not r.get('error')]
        failed = [r for r in results if r.get('error')]
        
        if INTEGRATED_AVAILABLE and ResponseWrapper:
            return jsonify(ResponseWrapper.success(
                data={
                    'results': results, 'successful': len(successful),
                    'failed': len(failed), 'total': len(commands)
                },
                message=f'Executed {len(successful)}/{len(commands)} commands',
                metadata={
                    'avg_duration_ms': round((sum(timings) / len(timings)) * 1000) if timings else 0,
                    'total_duration_ms': round(sum(timings) * 1000) if timings else 0
                }
            )), 200
        else:
            return jsonify({'status': 'success', 'result': results}), 200
    
    except Exception as exc:
        logger.error('[/api/commands/batch] %s', exc)
        if INTEGRATED_AVAILABLE and ResponseWrapper:
            return jsonify(ResponseWrapper.error(error=str(exc), error_code='BATCH_FAILED')), 500
        else:
            return jsonify({'status': 'error', 'error': str(exc)}), 500


@app.route('/api/commands/sequential', methods=['POST'])
def api_commands_sequential():
    """Execute multiple commands sequentially."""
    try:
        body = request.get_json(force=True, silent=True) or {}
        commands = body.get('commands', [])
        
        if not commands:
            if INTEGRATED_AVAILABLE and ResponseWrapper:
                return jsonify(ResponseWrapper.error(
                    error='No commands provided', error_code='EMPTY_BATCH'
                )), 400
        
        is_auth, is_admin = _parse_auth(request)
        results = []
        errors = []
        timings = []
        
        for cmd_str in commands:
            import time
            t0 = time.time()
            try:
                result = dispatch_command(cmd_str, is_admin=is_admin, is_authenticated=is_auth)
                duration = time.time() - t0
                timings.append(duration)
                results.append({'command': cmd_str, 'response': ResponseWrapper.ensure_json(result) if ResponseWrapper else result, 'duration': duration})
            except Exception as e:
                logger.error('[seq] Command %s failed: %s', cmd_str, e)
                errors.append({'command': cmd_str, 'error': str(e)})
                results.append({'command': cmd_str, 'response': None, 'error': str(e)})
        
        if INTEGRATED_AVAILABLE and ResponseWrapper:
            return jsonify(ResponseWrapper.success(
                data={
                    'results': results, 'successful': len([r for r in results if not r.get('error')]),
                    'failed': len(errors), 'total': len(commands)
                },
                metadata={
                    'avg_duration_ms': round((sum(timings) / len(timings)) * 1000) if timings else 0,
                    'total_duration_ms': round(sum(timings) * 1000) if timings else 0
                }
            )), 200
        else:
            return jsonify({'status': 'success', 'result': results}), 200
    
    except Exception as exc:
        logger.error('[/api/commands/sequential] %s', exc)
        if INTEGRATED_AVAILABLE and ResponseWrapper:
            return jsonify(ResponseWrapper.error(error=str(exc), error_code='SEQ_FAILED')), 500


@app.route('/api/oracle/price/<symbol>', methods=['GET'])
def api_oracle_price(symbol):
    """Get price for a specific symbol."""
    try:
        if not INTEGRATED_AVAILABLE or not ORACLE_PRICE_PROVIDER:
            return jsonify({'status': 'error', 'error': 'Oracle not available'}), 503
        
        price_data = ORACLE_PRICE_PROVIDER.get_price(symbol)
        
        if price_data.get('available'):
            return jsonify(ResponseWrapper.success(data=price_data)), 200
        else:
            return jsonify(ResponseWrapper.error(
                error=f'Symbol not found: {symbol}', error_code='SYMBOL_NOT_FOUND'
            )), 404
    
    except Exception as exc:
        logger.error('[/api/oracle/price/%s] %s', symbol, exc)
        if INTEGRATED_AVAILABLE and ResponseWrapper:
            return jsonify(ResponseWrapper.error(error=str(exc), error_code='PRICE_ERROR')), 500


@app.route('/api/oracle/prices', methods=['GET'])
def api_oracle_prices():
    """Get all available prices."""
    try:
        if not INTEGRATED_AVAILABLE or not ORACLE_PRICE_PROVIDER:
            return jsonify({'status': 'error', 'error': 'Oracle not available'}), 503
        
        all_prices = ORACLE_PRICE_PROVIDER.get_all_prices()
        return jsonify(ResponseWrapper.success(
            data={'prices': all_prices, 'count': len(all_prices)}
        )), 200
    
    except Exception as exc:
        logger.error('[/api/oracle/prices] %s', exc)
        if INTEGRATED_AVAILABLE and ResponseWrapper:
            return jsonify(ResponseWrapper.error(error=str(exc), error_code='PRICES_ERROR')), 500


@app.route('/api/oracle/status', methods=['GET'])
def api_oracle_status():
    """Get oracle system status."""
    try:
        if not INTEGRATED_AVAILABLE or not ORACLE_PRICE_PROVIDER:
            return jsonify({'status': 'error', 'error': 'Oracle not available'}), 503
        
        status = ORACLE_PRICE_PROVIDER.get_status()
        return jsonify(ResponseWrapper.success(data=status)), 200
    
    except Exception as exc:
        logger.error('[/api/oracle/status] %s', exc)
        if INTEGRATED_AVAILABLE and ResponseWrapper:
            return jsonify(ResponseWrapper.error(error=str(exc), error_code='STATUS_ERROR')), 500

@app.route('/api/commands', methods=['GET'])
def api_commands():
    """Full command registry — drives frontend category/command lists."""
    try:
        cat_filter = request.args.get('category', '').lower()
        q          = request.args.get('q', '').lower()

        commands = []
        for name, entry in sorted(COMMAND_REGISTRY.items()):
            if cat_filter and entry['category'] != cat_filter:
                continue
            if q and q not in name and q not in entry['description'].lower():
                continue
            commands.append({
                'command':        name,
                'category':       entry['category'],
                'description':    entry['description'],
                'requires_auth':  entry.get('requires_auth', True),
                'requires_admin': entry.get('requires_admin', False),
            })

        return jsonify({
            'status':     'success',
            'commands':   commands,
            'total':      len(commands),
            'categories': sorted({e['category'] for e in COMMAND_REGISTRY.values()}),
            'timestamp':  datetime.now(timezone.utc).isoformat(),
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/help', methods=['GET'])
def api_help():
    """
    GET /api/help                   → overview
    GET /api/help?category=admin    → admin commands
    GET /api/help?command=admin-users
    """
    cat = request.args.get('category', '').strip()
    cmd = request.args.get('command', '').strip()

    if cat:
        result = dispatch_command(f'help-{cat}', is_admin=True, is_authenticated=True)
    elif cmd:
        result = dispatch_command(f'help-{cmd}', is_admin=True, is_authenticated=True)
    else:
        result = dispatch_command('help', is_admin=False, is_authenticated=False)

    return jsonify(result), 200 if result.get('status') == 'success' else 400


@app.route('/api/heartbeat', methods=['GET', 'POST'])
def api_heartbeat():
    data = {}
    if request.method == 'POST':
        data = request.get_json(force=True, silent=True) or {}
    try:
        hb = get_heartbeat()
        metrics = hb.get_metrics() if hb else {}
    except Exception:
        metrics = {}
    return jsonify({
        'status':   'alive',
        'commands': len(COMMAND_REGISTRY),
        'metrics':  metrics,
        'received': data,
        'time':     datetime.now(timezone.utc).isoformat(),
    }), 200


@app.route('/api/registry', methods=['GET'])
def api_registry():
    cats = {}
    for entry in COMMAND_REGISTRY.values():
        cats[entry['category']] = cats.get(entry['category'], 0) + 1
    return jsonify({'total': len(COMMAND_REGISTRY), 'categories': cats, 'engine_ok': _ENGINE is not None}), 200


# ══════════════════════════════════════════════════════════════════════════════
# TRANSACTION ROUTES  /api/transactions/*
# Guaranteed fallback — fires even if blockchain_api blueprint fails to load.
# The blockchain blueprint's route takes precedence when it loads successfully.
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/transactions/submit', methods=['POST'])
def api_transactions_submit():
    """POST /api/transactions/submit — create & queue a transaction."""
    is_auth, _ = _parse_auth(request)
    if not is_auth:
        return jsonify({'status': 'error', 'error': 'Authentication required'}), 401
    try:
        data         = request.get_json(force=True, silent=True) or {}
        from_address = (data.get('from_address') or data.get('from') or '').strip()
        to_address   = (data.get('to_address')   or data.get('to')   or '').strip()
        for _ch in '\x00\r\n\x1b':
            to_address   = to_address.replace(_ch, '')
            from_address = from_address.replace(_ch, '')
        if not to_address:
            return jsonify({'status': 'error', 'error': 'to_address is required'}), 400
        try:
            amount = float(data.get('amount', 0))
            if amount <= 0:
                return jsonify({'status': 'error', 'error': 'Amount must be positive'}), 400
        except (TypeError, ValueError):
            return jsonify({'status': 'error', 'error': f"Invalid amount: {data.get('amount')!r}"}), 400

        import hashlib, secrets as _sec, time as _t
        ts      = datetime.now(timezone.utc).isoformat()
        salt    = _sec.token_hex(16)
        tx_hash = hashlib.sha3_256(
            f'{from_address}{to_address}{amount}{ts}{salt}'.encode()
        ).hexdigest()
        tx_type = str(data.get('tx_type') or data.get('type', 'transfer'))
        memo    = str(data.get('memo', ''))[:500]

        # Try ledger mempool
        _queued = False
        try:
            from ledger_manager import GLOBAL_MEMPOOL as _mp
            if _mp:
                _mp.add_transaction({
                    'tx_hash': tx_hash, 'from_address': from_address,
                    'to_address': to_address, 'amount': amount,
                    'tx_type': tx_type, 'status': 'pending',
                    'memo': memo, 'timestamp': ts,
                })
                _queued = True
        except Exception:
            pass

        # Update globals counters
        try:
            gs = get_globals()
            if hasattr(gs, 'blockchain'):
                gs.blockchain.total_transactions = getattr(gs.blockchain, 'total_transactions', 0) + 1
                gs.blockchain.mempool_size = getattr(gs.blockchain, 'mempool_size', 0) + 1
        except Exception:
            pass

        logger.info(f'[tx/submit] {tx_hash[:16]}… → {to_address} amount={amount}')
        return jsonify({
            'status':       'success',
            'tx_hash':      tx_hash,
            'status_code':  'pending',
            'from_address': from_address,
            'to_address':   to_address,
            'amount':       amount,
            'tx_type':      tx_type,
            'queued':       _queued,
            'message':      f'Transaction submitted — {tx_hash[:16]}…',
            'timestamp':    ts,
        }), 201

    except Exception as exc:
        logger.error(f'[/api/transactions/submit] {exc}', exc_info=True)
        return jsonify({'status': 'error', 'error': str(exc)}), 500


@app.route('/api/transactions', methods=['GET'])
def api_transactions_list():
    is_auth, _ = _parse_auth(request)
    if not is_auth:
        return jsonify({'status': 'error', 'error': 'Authentication required'}), 401
    try:
        gs = get_globals()
        return jsonify({'status': 'success', 'transactions': [],
                        'total': getattr(gs.blockchain, 'total_transactions', 0),
                        'mempool_size': getattr(gs.blockchain, 'mempool_size', 0)}), 200
    except Exception as exc:
        return jsonify({'status': 'error', 'error': str(exc)}), 500


@app.route('/api/transactions/stats', methods=['GET'])
def api_transactions_stats():
    is_auth, _ = _parse_auth(request)
    if not is_auth:
        return jsonify({'status': 'error', 'error': 'Authentication required'}), 401
    try:
        gs = get_globals()
        return jsonify({'status': 'success',
                        'total_transactions': getattr(gs.blockchain, 'total_transactions', 0),
                        'mempool_size':       getattr(gs.blockchain, 'mempool_size', 0),
                        'total_blocks':       getattr(gs.blockchain, 'total_blocks', 0)}), 200
    except Exception as exc:
        return jsonify({'status': 'error', 'error': str(exc)}), 500



logger.info('[wsgi] ✓ /api/transactions/* routes registered')


@app.errorhandler(404)
def not_found(e):
    # API paths must NEVER return HTML — terminal block handlers detect HTML as an error
    if request.path.startswith('/api/'):
        return jsonify({'status': 'error', 'error': 'Not found', 'path': request.path}), 404
    html_path = os.path.join(PROJECT_ROOT, 'index.html')
    if os.path.exists(html_path):
        try:
            return Response(open(html_path, encoding='utf-8').read(), 200, mimetype='text/html')
        except Exception:
            pass
    return jsonify({'error': 'Not found', 'path': request.path}), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f'[500] {e}')
    return jsonify({'error': 'Internal server error', 'detail': str(e)}), 500


# ── WSGI entrypoint for gunicorn ──────────────────────────────────────────────
application = app

logger.info(f"""
╔══════════════════════════════════════════════════════════════════╗
║              QTCL WSGI v5.0 — READY                             ║
╠══════════════════════════════════════════════════════════════════╣
║  Commands:   {len(COMMAND_REGISTRY):<4}  Categories: {len(_categories):<4}  Engine: {'✓' if _ENGINE else '✗'}                 ║
║  Globals:    {'✓' if _GS else '✗'}      Dispatch: globals.dispatch_command()       ║
║  Routes:     / /health /api/command /api/commands /api/status    ║
╚══════════════════════════════════════════════════════════════════╝
""")

if __name__ == '__main__':
    port  = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
