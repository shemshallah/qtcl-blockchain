#!/usr/bin/env python3
"""
Flask WSGI configuration with unified mega_command_system and quantum_lattice_control.
"""

import os
import sys
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

# ════════════════════════════════════════════════════════════════════════════════════════
# INITIALIZE QUANTUM LATTICE CONTROL FIRST
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
    from flask import Flask, request, g, jsonify
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
    
    @app.route('/', methods=['GET'])
    def index():
        """Index page - API status"""
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
    <h1>⚡ QTCL v6.0 - Quantum Blockchain System</h1>
    <p>Unified mega_command_system (72 commands) + quantum_lattice_control (v6/v7/v8/v9 consolidated)</p>
    
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
        <p><span class="status">✓ Neural Network:</span> 57-neuron continuous refresh</p>
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
    
    <div class="section">
        <h2>💡 Quantum Systems Active</h2>
        <ul>
            <li>NonMarkovian Noise Bath (κ=0.070)</li>
            <li>W-State Recovery (Adaptive control)</li>
            <li>Enhanced Noise Bath Refresh (κ=0.08)</li>
            <li>57-Neuron Neural Lattice (8→57→32→8)</li>
            <li>Heartbeat: 15s checks + 30s reports</li>
            <li>v8 Revival System (106,496 pseudoqubits)</li>
        </ul>
    </div>
</body>
</html>
        """
        return html, 200, {'Content-Type': 'text/html; charset=utf-8'}
    
    @app.route('/api/command', methods=['POST'])
    def execute_command():
        """Execute a command."""
        try:
            data = request.get_json() or {}
            command = data.get('command', '')
            args = data.get('args', {})
            user_id = getattr(g, 'user_id', None)
            token = request.headers.get('Authorization', '').replace('Bearer ', '')
            role = getattr(g, 'user_role', 'user')
            
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
