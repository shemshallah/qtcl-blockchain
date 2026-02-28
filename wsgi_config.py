#!/usr/bin/env python3
"""
wsgi_config.py — QTCL Production WSGI entry point.

Single source of truth for:
  • Boot sequence (globals → DB → quantum → Flask)
  • All HTTP routes (delegating to mega_command_system — NO separate blueprints)
  • WSGI/Gunicorn compatibility

All commands are handled by mega_command_system.dispatch_command_sync().
The blueprint files (core_api, admin_api, blockchain_api, etc.) are NOT registered
here — every operation goes through POST /api/command.
"""

import os
import sys
import logging
import threading
from datetime import datetime, timezone

# ── Logging ──────────────────────────────────────────────────────────────────
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )
logger = logging.getLogger(__name__)

# ── Qiskit noise suppression (must run before any qiskit import) ──────────────
_QISKIT_NOISE_PREFIXES = (
    'qiskit.passmanager',
    'qiskit.compiler.transpiler',
    'qiskit.transpiler',
)

class _QiskitPassFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        return not any(record.name.startswith(p) for p in _QISKIT_NOISE_PREFIXES)

_qf = _QiskitPassFilter()
for _h in logging.root.handlers:
    _h.addFilter(_qf)
for _ql in _QISKIT_NOISE_PREFIXES:
    logging.getLogger(_ql).setLevel(logging.WARNING)
if not logging.root.handlers:
    _fb = logging.StreamHandler()
    _fb.addFilter(_qf)
    logging.root.addHandler(_fb)

# ════════════════════════════════════════════════════════════════════════════
# PHASE 0 — GLOBAL STATE
# ════════════════════════════════════════════════════════════════════════════
logger.info('[BOOTSTRAP] PHASE 0: Global state...')
try:
    from globals import initialize_globals, set_global_state, get_global_state
    if not initialize_globals():
        logger.warning('[BOOTSTRAP] Global state had issues, continuing')
    logger.info('[BOOTSTRAP] ✓ PHASE 0 complete')
except Exception as e:
    logger.error(f'[BOOTSTRAP] FATAL global state: {e}', exc_info=True)
    raise

# ════════════════════════════════════════════════════════════════════════════
# ENV VALIDATION
# ════════════════════════════════════════════════════════════════════════════
_REQUIRED_ENV = {
    'POOLER_HOST':     'PostgreSQL hostname',
    'POOLER_USER':     'PostgreSQL username',
    'POOLER_PASSWORD': 'PostgreSQL password',
    'POOLER_PORT':     'PostgreSQL port (default: 5432)',
    'POOLER_DB':       'Database name',
}
_missing = [f'  • {k}: {v}' for k, v in _REQUIRED_ENV.items() if not os.environ.get(k)]
if _missing:
    for m in _missing:
        logger.error(f'[BOOTSTRAP] Missing env var: {m}')
    raise RuntimeError(
        'Missing required environment variables:\n' + '\n'.join(_missing) +
        '\n\nSet POOLER_HOST, POOLER_USER, POOLER_PASSWORD, POOLER_PORT, POOLER_DB'
    )
logger.info('[BOOTSTRAP] ✓ All env vars present')

# ════════════════════════════════════════════════════════════════════════════
# PHASE 1 — DATABASE
# ════════════════════════════════════════════════════════════════════════════
logger.info('[BOOTSTRAP] PHASE 1: Database...')
DB_MANAGER = None
try:
    from db_builder_v2 import DatabaseBuilder
    DB_MANAGER = DatabaseBuilder()
    set_global_state('db_manager', DB_MANAGER)

    if DB_MANAGER.pool is None:
        raise RuntimeError(f'DB pool init failed: {DB_MANAGER.pool_error}')

    # Smoke test
    _conn = DB_MANAGER.pool.getconn()
    with _conn.cursor() as _cur:
        _cur.execute('SELECT 1')
    DB_MANAGER.pool.putconn(_conn)
    logger.info(f'[BOOTSTRAP] ✓ PHASE 1 complete — {DB_MANAGER.host}:{DB_MANAGER.port}/{DB_MANAGER.database}')

    try:
        DB_MANAGER.start_reconnect_daemon()
        logger.info('[BOOTSTRAP] ✓ DB reconnect daemon started')
    except Exception as _de:
        logger.warning(f'[BOOTSTRAP] Reconnect daemon failed (non-fatal): {_de}')

except ImportError as e:
    logger.error(f'[BOOTSTRAP] Cannot import DatabaseBuilder: {e}', exc_info=True)
    raise
except Exception as e:
    logger.error(f'[BOOTSTRAP] DB init failed: {e}', exc_info=True)
    raise

# ════════════════════════════════════════════════════════════════════════════
# PHASE 2 — QUANTUM LATTICE CONTROL
# ════════════════════════════════════════════════════════════════════════════
logger.info('[BOOTSTRAP] PHASE 2: Quantum lattice control...')
HEARTBEAT = None
LATTICE = None
QUANTUM_COORDINATOR = None
STABILIZER = None
try:
    from quantum_lattice_control import (
        initialize_quantum_system,
        HEARTBEAT,
        LATTICE,
        QUANTUM_COORDINATOR,
    )
    initialize_quantum_system()
    logger.info('[BOOTSTRAP] ✓ PHASE 2 complete — quantum lattice active')
except Exception as e:
    logger.error(f'[BOOTSTRAP] quantum_lattice_control init failed (non-fatal): {e}', exc_info=True)

# ════════════════════════════════════════════════════════════════════════════
# PHASE 2.5 — HLWE GENESIS ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════════
logger.info('[BOOTSTRAP] PHASE 2.5: HLWE Genesis Orchestrator...')
HLWEGenesisOrchestrator = None
try:
    from hlwe_engine import HLWEGenesisOrchestrator
    set_global_state('genesis_orchestrator', HLWEGenesisOrchestrator)
    logger.info('[BOOTSTRAP] ✓ HLWE Genesis Orchestrator ready')

    if os.environ.get('GENESIS_AUTO_INIT', 'false').lower() in ('true', '1', 'yes'):
        try:
            from db_builder_v2 import check_genesis_exists
            genesis_ok, _ = check_genesis_exists()
            if not genesis_ok:
                logger.info('[BOOTSTRAP] Auto-initializing genesis block...')
                ok, gb = HLWEGenesisOrchestrator.initialize_genesis(
                    validator_id=os.environ.get('GENESIS_VALIDATOR', 'GENESIS_VALIDATOR'),
                    chain_id=os.environ.get('CHAIN_ID', 'QTCL-MAINNET'),
                    entropy_sources=int(os.environ.get('GENESIS_ENTROPY_SOURCES', '5')),
                    force_overwrite=os.environ.get('GENESIS_FORCE_OVERWRITE', 'false').lower() in ('true', '1'),
                )
                if ok:
                    logger.info(f"[BOOTSTRAP] ✓ Genesis: {gb.get('block_hash','?')[:16]}...")
                else:
                    logger.warning('[BOOTSTRAP] Genesis auto-init failed (non-fatal)')
            else:
                logger.info('[BOOTSTRAP] ✓ Genesis block already exists')
        except Exception as _ge:
            logger.warning(f'[BOOTSTRAP] Auto-genesis failed (non-fatal): {_ge}')

except ImportError as e:
    logger.warning(f'[BOOTSTRAP] HLWE Genesis not available (non-fatal): {e}')
except Exception as e:
    logger.warning(f'[BOOTSTRAP] HLWE Genesis error (non-fatal): {e}')

# ════════════════════════════════════════════════════════════════════════════
# PHASE 3 — MEGA COMMAND SYSTEM (pre-warm registry)
# ════════════════════════════════════════════════════════════════════════════
logger.info('[BOOTSTRAP] PHASE 3: Mega command system...')
try:
    from mega_command_system import (
        dispatch_command_sync,
        dispatch_cli_command,
        list_commands_sync,
        get_command_info_sync,
        parse_cli_command,
        get_registry,
    )
    # Export aliases expected by main_app
    COMMAND_REGISTRY = get_registry()
    dispatch_command = dispatch_command_sync  # alias for main_app import
    logger.info(f'[BOOTSTRAP] ✓ PHASE 3 complete — {len(COMMAND_REGISTRY.commands)} commands registered')
except ImportError as e:
    logger.error(f'[BOOTSTRAP] FATAL: Cannot import mega_command_system: {e}', exc_info=True)
    raise
except Exception as e:
    logger.error(f'[BOOTSTRAP] mega_command_system init failed: {e}', exc_info=True)
    raise

# ════════════════════════════════════════════════════════════════════════════
# FLASK APP
# ════════════════════════════════════════════════════════════════════════════
try:
    from flask import Flask, request, g, jsonify
    logger.info('[BOOTSTRAP] Flask imported')
except ImportError:
    logger.error('[BOOTSTRAP] Flask not available — install it: pip install flask')
    raise

app = None

def create_app() -> Flask:
    """Create and configure the Flask application."""
    global app, HEARTBEAT, LATTICE, QUANTUM_COORDINATOR, STABILIZER

    _app = Flask(__name__)
    _app.config['JSON_SORT_KEYS'] = False

    # ── STATIC / UI ──────────────────────────────────────────────────────────
    @_app.route('/', methods=['GET'])
    @_app.route('/index.html', methods=['GET'])
    def serve_ui():
        """Serve the QTCL terminal UI."""
        try:
            index_path = os.path.join(os.path.dirname(__file__), 'index.html')
            if os.path.exists(index_path):
                with open(index_path, 'r') as f:
                    content = f.read()
                return content, 200, {
                    'Content-Type': 'text/html; charset=utf-8',
                    'Cache-Control': 'no-cache, no-store, must-revalidate, max-age=0',
                    'Pragma': 'no-cache',
                    'Expires': '0',
                }
        except Exception as e:
            logger.error(f'[UI] Error serving index.html: {e}')
        return jsonify({'error': 'UI not found', 'hint': 'Use POST /api/command'}), 404

    # ── HEALTH / VERSION ─────────────────────────────────────────────────────
    @_app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'heartbeat': HEARTBEAT.running if HEARTBEAT and hasattr(HEARTBEAT, 'running') else False,
            'lattice': LATTICE is not None,
            'db': DB_MANAGER is not None and DB_MANAGER.pool is not None,
            'commands': len(COMMAND_REGISTRY.commands) if COMMAND_REGISTRY else 0,
        }), 200

    @_app.route('/version', methods=['GET'])
    def version():
        return jsonify({
            'version': '6.0.0',
            'codename': 'QTCL',
            'quantum_lattice': 'v9 unified',
            'command_system': 'mega_command_system',
            'nonmarkovian_bath': 'κ=0.070',
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }), 200

    @_app.route('/metrics', methods=['GET'])
    def metrics():
        try:
            registry = get_registry()
            stats = {
                name: cmd.get_stats()
                for name, cmd in registry.commands.items()
                if hasattr(cmd, 'get_stats')
            }
            return jsonify(stats), 200
        except Exception as e:
            logger.error(f'[API] Metrics error: {e}')
            return jsonify({'error': str(e)}), 500

    # ── COMMAND REGISTRY INFO ────────────────────────────────────────────────
    @_app.route('/api/commands', methods=['GET'])
    def list_commands():
        try:
            category = request.args.get('category')
            result = list_commands_sync(category)
            return jsonify(result), 200
        except Exception as e:
            logger.error(f'[API] /api/commands error: {e}', exc_info=True)
            return jsonify({'error': str(e)}), 500

    @_app.route('/api/commands/<command_name>', methods=['GET'])
    def get_command_info(command_name):
        try:
            info = get_command_info_sync(command_name)
            if not info:
                return jsonify({'error': f'Unknown command: {command_name}'}), 404
            return jsonify(info), 200
        except Exception as e:
            logger.error(f'[API] /api/commands/{command_name} error: {e}', exc_info=True)
            return jsonify({'error': str(e)}), 500

    # ── COMMAND EXECUTION — THE ONE TRUE ENTRY POINT ─────────────────────────
    @_app.route('/api/command', methods=['POST'])
    def execute_command():
        """
        Execute any QTCL command via mega_command_system.

        Accepts two body formats:
          JSON:  {"command": "auth-login", "args": {"username": "x", "password": "y"}}
          CLI:   {"command": "auth-login username=x password=y"}
          Raw:   Content-Type: text/plain  →  auth-login username=x password=y
        """
        try:
            user_id = getattr(g, 'user_id', None)
            token   = request.headers.get('Authorization', '').replace('Bearer ', '').strip() or None
            role    = getattr(g, 'user_role', 'user')

            command = None
            args    = {}

            data = request.get_json(silent=True)
            if data and isinstance(data, dict):
                command = (data.get('command') or '').strip()
                args    = data.get('args') or data.get('kwargs') or {}
                if not isinstance(args, dict):
                    args = {}
            else:
                # Fallback: raw text body (CLI piped input)
                raw = request.get_data(as_text=True).strip()
                if raw:
                    command, args = parse_cli_command(raw)

            if not command:
                return jsonify({'status': 'error', 'error': 'No command specified'}), 400

            # Special case: help shortcut
            if command.lower() == 'help':
                return jsonify({
                    'status': 'success',
                    'result': list_commands_sync(),
                    'execution_time_ms': 0.0,
                }), 200

            result = dispatch_command_sync(
                command=command,
                args=args,
                user_id=user_id,
                token=token,
                role=role,
            )
            return jsonify(result), 200

        except Exception as e:
            logger.error(f'[API] /api/command unhandled error: {e}', exc_info=True)
            return jsonify({'status': 'error', 'error': str(e)}), 500

    # ── INFRASTRUCTURE ENDPOINTS ─────────────────────────────────────────────
    @_app.route('/api/heartbeat', methods=['POST'])
    def heartbeat_receiver():
        try:
            data = request.get_json(silent=True) or {}
            logger.info(f"[HEARTBEAT] beat #{data.get('beat_count', '?')}")
            return jsonify({'status': 'received'}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @_app.route('/api/quantum/status', methods=['GET'])
    def quantum_status():
        try:
            if QUANTUM_COORDINATOR:
                return jsonify(QUANTUM_COORDINATOR.get_status()), 200
            return jsonify({'status': 'unavailable', 'reason': 'quantum coordinator not initialized'}), 503
        except Exception as e:
            logger.error(f'[API] quantum status error: {e}')
            return jsonify({'error': str(e)}), 500

    @_app.route('/api/health/stabilizer', methods=['GET'])
    def stabilizer_health():
        try:
            from quantum_lattice_control import STABILIZER as _stab
            if not _stab:
                return jsonify({'status': 'unavailable'}), 503
            return jsonify(_stab.get_health_report()), 200
        except Exception as e:
            logger.error(f'[API] stabilizer health error: {e}', exc_info=True)
            return jsonify({'error': str(e)}), 500

    # ── GLOBAL NO-CACHE + JSON CONTENT-TYPE ENFORCEMENT ──────────────────────
    @_app.after_request
    def set_headers(response):
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
        response.headers['Pragma']        = 'no-cache'
        response.headers['Expires']       = '0'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    logger.info('[BOOTSTRAP] ✓ All routes registered')
    return _app


def get_wsgi_app() -> Flask:
    global app
    if app is None:
        app = create_app()
    return app


# ── Module-level boot (Gunicorn imports this module, not __main__) ────────────
if __name__ != '__main__':
    app = get_wsgi_app()
    logger.info('[BOOTSTRAP] ✓ WSGI app ready')

# ── Gunicorn entry point ─────────────────────────────────────────────────────
application = app if app is not None else get_wsgi_app()

__all__ = [
    'app',
    'application',
    'get_wsgi_app',
    'create_app',
    # Exported so main_app.py import succeeds
    'COMMAND_REGISTRY',
    'dispatch_command',
]

logger.info('[BOOTSTRAP] ✓ wsgi_config loaded')
