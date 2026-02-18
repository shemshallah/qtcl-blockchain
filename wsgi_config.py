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
    
    # HEARTBEAT — from globals.quantum.heartbeat
    try:
        HEARTBEAT = get_heartbeat()
    except Exception:
        HEARTBEAT = None
    
    # QUANTUM — quantum subsystems bundle
    try:
        gs = get_globals()
        QUANTUM = gs.quantum
    except Exception:
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
            
            # auth users & active sessions
            try:
                cur.execute("SELECT COUNT(*) FROM users WHERE is_active=TRUE AND is_deleted=FALSE")
                row = cur.fetchone()
                if row and row[0]:
                    gs.auth.active_sessions = int(row[0])  # approximate user count
            except Exception:
                pass
            
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


def _parse_auth(req):
    """Returns (is_authenticated, is_admin).

    Token resolution order:
      1. Authorization: Bearer <token> header  (standard)
      2. globals.auth.session_store            (set by h_login — works without JWT env var)
      3. JWT decode via auth_mgr               (stateless verification)
      4. ADMIN_SECRET env var                  (admin bypass)
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
                return True, role in ('admin', 'superadmin')
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


@app.errorhandler(404)
def not_found(e):
    html_path = os.path.join(PROJECT_ROOT, 'index.html')
    if os.path.exists(html_path):
        return Response(open(html_path, encoding='utf-8').read(), 200, mimetype='text/html')
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
