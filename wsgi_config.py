#!/usr/bin/env python3
"""
Flask WSGI configuration with unified mega_command_system and quantum_lattice_control.
"""

import os
import sys

# ── OQS / liboqs guard — MUST be set before any module that does `import oqs` ──
# oqs.oqs._load_liboqs() runs at import time and tries to git-clone liboqs from
# source if the shared library (.so) is absent.  On Koyeb there is no cmake/gcc
# toolchain and the target git branch (0.14.1) no longer exists, so the clone
# fails with "Remote branch 0.14.1 not found" → RuntimeError → process exits.
#
# OQS_SKIP_SETUP=1 tells oqs to skip the auto-install countdown entirely.
# OQS_BUILD=0      is the alternate env var checked by some oqs builds.
# Both must be set before the first `import oqs` anywhere in the process.
os.environ.setdefault('OQS_SKIP_SETUP', '1')
os.environ.setdefault('OQS_BUILD', '0')
# ─────────────────────────────────────────────────────────────────────────────

import logging
import threading
from datetime import datetime, timezone

# Logging
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )
logger = logging.getLogger(__name__)

# ── Qiskit noise suppression — ROOT-HANDLER FILTER (bulletproof) ──────────────
# quantum_lattice_control installs this filter too, but wsgi_config must also
# install it here because gunicorn worker processes may reach this module via a
# different import path, or wsgi_config's basicConfig may fire after QLC's.
# Attaching a Filter to root handlers is idempotent and safe to call twice.
_QISKIT_NOISE_PREFIXES = (
    "qiskit.passmanager",
    "qiskit.compiler.transpiler",
    "qiskit.transpiler",
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
# ─────────────────────────────────────────────────────────────────────────────

# ════════════════════════════════════════════════════════════════════════════════════════
# CRITICAL: PHASE 0 — INITIALIZE GLOBAL STATE (ABSOLUTELY FIRST)
# ════════════════════════════════════════════════════════════════════════════════════════

logger.info("[BOOTSTRAP] PHASE 0: Initializing global state infrastructure...")
try:
    from globals import initialize_globals, set_global_state, get_global_state
    
    if not initialize_globals():
        logger.warning("[BOOTSTRAP] Global state had initialization issues, continuing anyway")
    logger.info("[BOOTSTRAP] ✓ PHASE 0 complete: Global state ready")
except Exception as e:
    logger.error(f"[BOOTSTRAP] CRITICAL: Global state init failed: {e}", exc_info=True)
    raise

# ════════════════════════════════════════════════════════════════════════════════════════
# CRITICAL: Validate Database Credentials Before Attempting Connection
# ════════════════════════════════════════════════════════════════════════════════════════

logger.info("[BOOTSTRAP] Validating database credentials...")

_required_env_vars = {
    'POOLER_HOST': 'PostgreSQL hostname (e.g., localhost or cloud.example.com)',
    'POOLER_USER': 'PostgreSQL username',
    'POOLER_PASSWORD': 'PostgreSQL password',
    'POOLER_PORT': 'PostgreSQL port (default: 5432)',
    'POOLER_DB': 'Database name',
}

_missing_vars = []
for var, description in _required_env_vars.items():
    value = os.environ.get(var)
    if not value:
        _missing_vars.append(f"  • {var}: {description}")
        logger.error(f"[BOOTSTRAP] Missing env var: {var}")
    else:
        if 'PASSWORD' in var:
            display = '*' * 5
        else:
            display = value
        logger.info(f"[BOOTSTRAP] Found {var} = {display}")

if _missing_vars:
    logger.error("[BOOTSTRAP] ❌ CRITICAL: Missing environment variables:")
    for var_msg in _missing_vars:
        logger.error(var_msg)
    logger.error("[BOOTSTRAP] Set these before starting the app:")
    logger.error("[BOOTSTRAP]   export POOLER_HOST=your-host")
    logger.error("[BOOTSTRAP]   export POOLER_USER=postgres")
    logger.error("[BOOTSTRAP]   export POOLER_PASSWORD=your-password")
    logger.error("[BOOTSTRAP]   export POOLER_PORT=5432")
    logger.error("[BOOTSTRAP]   export POOLER_DB=qtcl_db")
    raise RuntimeError("Missing required database credentials")

logger.info("[BOOTSTRAP] ✓ All database credentials found")

# ════════════════════════════════════════════════════════════════════════════════════════
# CRITICAL: PHASE 1 — INITIALIZE DATABASE MANAGER (BEFORE EVERYTHING ELSE)
# ════════════════════════════════════════════════════════════════════════════════════════

logger.info("[BOOTSTRAP] PHASE 1: Initializing database manager...")
DB_MANAGER = None  # Module-level, so it persists

try:
    from db_builder_v2 import DatabaseBuilder
    
    # Create database manager with connection pool
    # Reads from environment: POOLER_HOST, POOLER_USER, POOLER_PASSWORD, POOLER_PORT, POOLER_DB
    logger.info("[BOOTSTRAP]   Creating DatabaseBuilder...")
    DB_MANAGER = DatabaseBuilder()
    
    # CRITICAL: Register in global state so get_db_manager() works
    logger.info("[BOOTSTRAP]   Storing db_manager in global state...")
    set_global_state('db_manager', DB_MANAGER)
    
    if DB_MANAGER.pool is not None:
        logger.info(f"[BOOTSTRAP] ✓ PHASE 1 complete: Database CONNECTED")
        logger.info(f"[BOOTSTRAP]   Pool Status: ACTIVE")
        logger.info(f"[BOOTSTRAP]   Host: {DB_MANAGER.host}:{DB_MANAGER.port}")
        logger.info(f"[BOOTSTRAP]   Database: {DB_MANAGER.database}")
        logger.info(f"[BOOTSTRAP]   Pool Size: {DB_MANAGER.pool_size}")
        logger.info(f"[BOOTSTRAP]   Min Connections: 1")
        
        # Verify pool is actually working with a test query
        try:
            conn = DB_MANAGER.pool.getconn()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            DB_MANAGER.pool.putconn(conn)
            logger.info("[BOOTSTRAP]   Pool Test: ✓ SUCCESS (SELECT 1 executed)")
        except Exception as pool_test_err:
            logger.warning(f"[BOOTSTRAP]   Pool Test: ⚠️ FAILED — {pool_test_err}")
        
        # Start reconnection daemon for resilience
        try:
            DB_MANAGER.start_reconnect_daemon()
            logger.info("[BOOTSTRAP] ✓ Reconnection daemon started (auto-recovery enabled)")
        except Exception as daemon_err:
            logger.warning(f"[BOOTSTRAP]   Reconnection daemon failed: {daemon_err}")
    else:
        # Pool failed to create
        logger.error(f"[BOOTSTRAP] ✗ PHASE 1 FAILED: Database pool NOT initialized")
        logger.error(f"[BOOTSTRAP]   Error: {DB_MANAGER.pool_error}")
        logger.error(f"[BOOTSTRAP]   Credentials to verify:")
        logger.error(f"[BOOTSTRAP]     POOLER_HOST = {os.environ.get('POOLER_HOST')}")
        logger.error(f"[BOOTSTRAP]     POOLER_USER = {os.environ.get('POOLER_USER')}")
        logger.error(f"[BOOTSTRAP]     POOLER_PASSWORD = {'*' * len(os.environ.get('POOLER_PASSWORD', ''))}")
        logger.error(f"[BOOTSTRAP]     POOLER_PORT = {os.environ.get('POOLER_PORT')}")
        logger.error(f"[BOOTSTRAP]     POOLER_DB = {os.environ.get('POOLER_DB')}")
        logger.error(f"[BOOTSTRAP]   FIX: Verify PostgreSQL is running and credentials are correct")
        raise RuntimeError(f"Database pool initialization failed: {DB_MANAGER.pool_error}")
        
except ImportError as e:
    logger.error(f"[BOOTSTRAP] CRITICAL: Cannot import DatabaseBuilder: {e}", exc_info=True)
    raise
except Exception as e:
    logger.error(f"[BOOTSTRAP] CRITICAL: Database initialization failed: {e}", exc_info=True)
    raise

# Verify DB_MANAGER is in global state
_check_db = get_global_state('db_manager')
if _check_db is None:
    logger.error("[BOOTSTRAP] CRITICAL: db_manager is None in global state!")
    logger.error("[BOOTSTRAP] Something went wrong storing db_manager")
    raise RuntimeError("Failed to store db_manager in global state")
elif _check_db.pool is None:
    logger.error("[BOOTSTRAP] CRITICAL: db_manager.pool is None!")
    logger.error("[BOOTSTRAP] Database connection failed")
    raise RuntimeError("Database pool is None")
else:
    logger.info("[BOOTSTRAP] ✓ Verified: db_manager is in global state and pool is active")

# ════════════════════════════════════════════════════════════════════════════════════════
# PHASE 2: INITIALIZE QUANTUM LATTICE CONTROL
# ════════════════════════════════════════════════════════════════════════════════════════

logger.info("[BOOTSTRAP] Initializing quantum_lattice_control...")
try:
    from quantum_lattice_control import (
        initialize_quantum_system, 
        HEARTBEAT, 
        LATTICE,
        QUANTUM_COORDINATOR
    )
    initialize_quantum_system()
    logger.info("[BOOTSTRAP] ✓ quantum_lattice_control initialized")
    logger.info("[BOOTSTRAP] ✓ HEARTBEAT running (check logs for 15s/30s cycles)")
except Exception as e:
    logger.error(f"[BOOTSTRAP] Failed to init quantum_lattice_control: {e}", exc_info=True)

# ════════════════════════════════════════════════════════════════════════════════════════
# FLASK SETUP
# ════════════════════════════════════════════════════════════════════════════════════════

try:
    from flask import Flask, request, g, jsonify, send_file
    logger.info("[BOOTSTRAP] Flask imported successfully")
except ImportError:
    logger.error("[BOOTSTRAP] Flask not available")
    Flask = None

app = None

def create_app():
    """Create Flask application."""
    global app
    
    app = Flask(__name__)
    app.config['JSON_SORT_KEYS'] = False
    
    logger.info("[BOOTSTRAP] Flask app created")
    
    # Import mega_command_system (required)
    try:
        from mega_command_system import (
            dispatch_command_sync,
            list_commands_sync,
            get_command_info_sync,
        )
        logger.info("[BOOTSTRAP] ✓ Mega command system imported")
    except ImportError as e:
        logger.error(f"[BOOTSTRAP] ✗ FATAL: Cannot import mega_command_system: {e}")
        raise
    
    # ── ENDPOINTS ────────────────────────────────────────────────────────────────────
    
    @app.route('/index.html', methods=['GET'])
    def serve_index():
        """Serve index.html with no-cache headers."""
        try:
            index_path = os.path.join(os.path.dirname(__file__), 'index.html')
            if os.path.exists(index_path):
                with open(index_path, 'r') as f:
                    return f.read(), 200, {
                        'Content-Type': 'text/html; charset=utf-8',
                        'Cache-Control': 'no-cache, no-store, must-revalidate, max-age=0',
                        'Pragma': 'no-cache',
                        'Expires': '0',
                    }
            return "index.html not found", 404
        except Exception as e:
            logger.error(f"[API] Error serving index.html: {e}")
            return str(e), 500
    
    @app.route('/', methods=['GET'])
    def index():
        """Redirect to index.html or serve dashboard."""
        try:
            index_path = os.path.join(os.path.dirname(__file__), 'index.html')
            if os.path.exists(index_path):
                with open(index_path, 'r') as f:
                    return f.read(), 200, {
                        'Content-Type': 'text/html; charset=utf-8',
                        'Cache-Control': 'no-cache, no-store, must-revalidate, max-age=0',
                        'Pragma': 'no-cache',
                        'Expires': '0',
                    }
        except Exception as e:
            logger.warning(f"[API] Could not load index.html: {e}")
        
        # Fallback API status page
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>QTCL - Quantum Blockchain</title>
    <style>
        body {
            font-family: 'Courier New', monospace;
            background: #0a0e27;
            color: #00ff00;
            margin: 20px;
            line-height: 1.6;
        }
        h1 { color: #00ffff; }
        h2 { color: #ffff00; margin-top: 30px; }
        .section { 
            background: #1a1f3a; 
            padding: 15px;
            margin: 15px 0;
            border-left: 3px solid #00ff00;
            border-radius: 3px;
        }
        a { color: #00ffff; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .endpoint { margin: 8px 0; padding: 5px; }
        .status { color: #00ff00; }
        code { background: #0a0e27; padding: 2px 5px; }
        table { width: 100%; border-collapse: collapse; }
        td { padding: 8px; border-bottom: 1px solid #333; }
    </style>
</head>
<body>
    <h1>⚡ QTCL - Quantum Blockchain System</h1>
    <p>Unified mega_command_system (72 commands) + quantum_lattice_control</p>
    
    <div class="section">
        <h2>🌐 API Endpoints</h2>
        <table>
            <tr>
                <td><strong>GET /</strong></td>
                <td>This page</td>
            </tr>
            <tr>
                <td><strong>GET /health</strong></td>
                <td>System health</td>
            </tr>
            <tr>
                <td><strong>GET /version</strong></td>
                <td>Version info</td>
            </tr>
            <tr>
                <td><strong>GET /api/commands</strong></td>
                <td>List all 72 commands</td>
            </tr>
            <tr>
                <td><strong>GET /api/commands/&lt;name&gt;</strong></td>
                <td>Get command info</td>
            </tr>
            <tr>
                <td><strong>POST /api/command</strong></td>
                <td>Execute a command</td>
            </tr>
            <tr>
                <td><strong>GET /api/quantum/status</strong></td>
                <td>Quantum system status</td>
            </tr>
            <tr>
                <td><strong>POST /api/heartbeat</strong></td>
                <td>Heartbeat metrics receiver</td>
            </tr>
            <tr>
                <td><strong>GET /metrics</strong></td>
                <td>Command execution metrics</td>
            </tr>
        </table>
    </div>
    
    <div class="section">
        <h2>📊 System Status</h2>
        <p><span class="status">✓ Heartbeat:</span> Running (1.0 Hz, 15s check + 30s report)</p>
        <p><span class="status">✓ Quantum Lattice:</span> Active (NonMarkovian noise bath, W-state recovery)</p>
        <p><span class="status">✓ Commands:</span> 72 available</p>
    </div>
    
    <div class="section">
        <h2>🔧 Quick Test</h2>
        <pre>
# Health check
curl https://your-domain.koyeb.app/health

# Quantum status
curl https://your-domain.koyeb.app/api/quantum/status

# Execute command
curl -X POST https://your-domain.koyeb.app/api/command \\
  -H "Content-Type: application/json" \\
  -d '{"command": "system-stats"}'
        </pre>
    </div>
</body>
</html>
        """
        return html, 200, {
            'Content-Type': 'text/html; charset=utf-8',
            'Cache-Control': 'no-cache, no-store, must-revalidate, max-age=0',
            'Pragma': 'no-cache',
            'Expires': '0',
        }
    
    @app.route('/api/command', methods=['POST'])
    def execute_command():
        """Execute a command."""
        try:
            data = request.get_json() or {}
            command = data.get('command', '').strip().lower()
            args = data.get('args', {})
            user_id = getattr(g, 'user_id', None)
            token = request.headers.get('Authorization', '').replace('Bearer ', '')
            role = getattr(g, 'user_role', 'user')
            
            # Special handling for help command
            if command == 'help':
                result = list_commands_sync()
                return jsonify({
                    'status': 'success',
                    'result': result,
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
            logger.error(f"[API] Error in /api/command: {e}", exc_info=True)
            return jsonify({'status': 'error', 'error': str(e)}), 500
    
    @app.route('/api/commands', methods=['GET'])
    def list_commands():
        """List all available commands."""
        try:
            category = request.args.get('category')
            result = list_commands_sync(category)
            return jsonify(result), 200
        except Exception as e:
            logger.error(f"[API] Error in /api/commands: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/commands/<command_name>', methods=['GET'])
    def get_command_info(command_name):
        """Get info about a specific command."""
        try:
            info = get_command_info_sync(command_name)
            if not info:
                return jsonify({'error': 'Command not found'}), 404
            return jsonify(info), 200
        except Exception as e:
            logger.error(f"[API] Error in /api/commands/<n>: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/heartbeat', methods=['POST'])
    def heartbeat_receiver():
        """Receive heartbeat metrics."""
        try:
            data = request.get_json() or {}
            logger.info(f"[HEARTBEAT-RECEIVER] Received beat #{data.get('beat_count', 0)}")
            return jsonify({'status': 'received'}), 200
        except Exception as e:
            logger.error(f"[HEARTBEAT-RECEIVER] Error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/quantum/status', methods=['GET'])
    def quantum_status():
        """Get quantum system status."""
        try:
            if QUANTUM_COORDINATOR:
                status = QUANTUM_COORDINATOR.get_status()
                return jsonify(status), 200
            return jsonify({'status': 'unavailable'}), 503
        except Exception as e:
            logger.error(f"[API] Quantum status error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/health/stabilizer', methods=['GET'])
    def get_stabilizer_health():
        """Return comprehensive enterprise stabilizer health report."""
        try:
            from quantum_lattice_control import STABILIZER
            if not STABILIZER:
                return jsonify({'status': 'unavailable', 'message': 'Stabilizer not initialized'}), 503
            
            report = STABILIZER.get_health_report()
            return jsonify(report), 200
        except Exception as e:
            logger.error(f"[API] Stabilizer health error: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/metrics', methods=['GET'])
    def metrics():
        """Command execution metrics."""
        try:
            from mega_command_system import get_registry
            registry = get_registry()
            stats = {}
            for cmd_name, cmd in registry.commands.items():
                if hasattr(cmd, 'get_stats'):
                    stats[cmd_name] = cmd.get_stats()
            return jsonify(stats), 200
        except Exception as e:
            logger.error(f"[API] Metrics error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """System health check."""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'heartbeat': HEARTBEAT.running if HEARTBEAT else False,
            'lattice': LATTICE is not None,
        }), 200
    
    @app.route('/version', methods=['GET'])
    def version():
        """Return system version information."""
        return jsonify({
            'version': '6.0.0',
            'codename': 'QTCL',
            'quantum_lattice': 'v9 unified (v6/v7/v8/v9 consolidated)',
            'command_system': 'mega_command_system',
            'heartbeat': '15s check + 30s report',
            'nonmarkovian_bath': 'κ=0.070',
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }), 200
    
    logger.info("[BOOTSTRAP] ✓ All endpoints registered")
    
    # ── NO-CACHE HEADERS ─────────────────────────────────────────────────────────
    @app.after_request
    def set_no_cache(response):
        """Force no-cache on ALL responses."""
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        logger.debug(f"[NO-CACHE] Applied to {request.path}")
        return response
    
    return app

def get_wsgi_app():
    """Get WSGI-compatible app instance."""
    global app
    if app is None:
        app = create_app()
    return app

# Create app on module load
if __name__ != '__main__':
    app = get_wsgi_app()
    logger.info("[BOOTSTRAP] ✓ WSGI app ready")

# WSGI Entry Point (required by Gunicorn)
application = app if app is not None else get_wsgi_app()

__all__ = ['app', 'application', 'get_wsgi_app', 'create_app']

logger.info("[BOOTSTRAP] ✓ wsgi_config loaded successfully")
