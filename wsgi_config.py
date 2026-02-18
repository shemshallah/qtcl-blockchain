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

logger.info('[wsgi] globals imported — initialising...')
try:
    initialize_globals()
    _GS = get_globals()
    logger.info('[wsgi] globals initialised OK')
except Exception as _e:
    logger.error(f'[wsgi] globals init error (continuing): {_e}')
    _GS = None

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
    """Returns (is_authenticated, is_admin)"""
    token = req.headers.get('Authorization', '').replace('Bearer ', '').strip()
    if not token:
        return False, False
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
    Universal hyphenated command dispatcher.

    Body: { "command": "admin-users --limit=10" }
          { "command": "help-admin" }
          { "command": "oracle-price", "kwargs": {"symbol": "BTCUSD"} }

    Response: { "status": "success"|"error", "result": {...} }
              { "status": "error", "error": "...", "suggestions": [...] }
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
        raw  = (body.get('command') or body.get('cmd') or '').strip()
        if not raw:
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

        try:
            get_globals().metrics.commands_executed += 1
        except Exception:
            pass

        return jsonify(result), 200 if result.get('status') == 'success' else 400

    except Exception as exc:
        logger.error(f'[/api/command] {exc}\n{traceback.format_exc()}')
        return jsonify({'status': 'error', 'error': str(exc)}), 500


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
