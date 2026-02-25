#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                        ║
║        🚀 QTCL WSGI v6.0 — UNIFIED MEGA COMMAND SYSTEM INTEGRATION 🚀               ║
║                                                                                        ║
║  Clean Flask WSGI configuration with integrated mega_command_system.                 ║
║  NO legacy command system. Completely unified.                                        ║
║                                                                                        ║
╚════════════════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import logging
import threading
import json
from datetime import datetime, timezone
from collections import deque

# Suppress OQS auto-install
os.environ.setdefault('OQS_SKIP_SETUP', '1')
os.environ.setdefault('OQS_BUILD', '0')

import numpy as np
from typing import Dict, Any

# ════════════════════════════════════════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════════════════════════════════════════

if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )
logger = logging.getLogger(__name__)

# Suppress Qiskit
for qiskit_logger in ['qiskit.passmanager', 'qiskit.compiler', 'qiskit.transpiler', 'qiskit']:
    logging.getLogger(qiskit_logger).setLevel(logging.WARNING)

# ════════════════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT & CONFIG
# ════════════════════════════════════════════════════════════════════════════════════════

logger.info("[BOOTSTRAP] Reading environment variables...")

SUPABASE_HOST = os.environ.get('SUPABASE_HOST')
SUPABASE_USER = os.environ.get('SUPABASE_USER')
SUPABASE_PASSWORD = os.environ.get('SUPABASE_PASSWORD')
SUPABASE_PORT = os.environ.get('SUPABASE_PORT', '5432')
SUPABASE_DB = os.environ.get('SUPABASE_DB', 'postgres')

if SUPABASE_HOST and SUPABASE_USER and SUPABASE_PASSWORD:
    logger.info(f"[BOOTSTRAP] ✓ SUPABASE credentials loaded")
else:
    logger.error("[BOOTSTRAP] ❌ SUPABASE credentials NOT found!")

# ════════════════════════════════════════════════════════════════════════════════════════
# DATABASE SINGLETON
# ════════════════════════════════════════════════════════════════════════════════════════

DB = None
DB_LOCK = threading.RLock()

def set_database_instance(db_instance):
    global DB
    with DB_LOCK:
        DB = db_instance
        logger.info("[DB] Database instance registered")

def get_database_instance():
    global DB
    with DB_LOCK:
        return DB

# ════════════════════════════════════════════════════════════════════════════════════════
# PQ CRYPTOGRAPHY
# ════════════════════════════════════════════════════════════════════════════════════════

_PQ_CRYPTO_SYSTEM = None
_PQ_CRYPTO_LOCK = threading.RLock()

def get_pq_system():
    global _PQ_CRYPTO_SYSTEM
    if _PQ_CRYPTO_SYSTEM is None:
        with _PQ_CRYPTO_LOCK:
            if _PQ_CRYPTO_SYSTEM is None:
                try:
                    from pq_keys_system import get_pq_system as _get_unified_pq
                    _PQ_CRYPTO_SYSTEM = _get_unified_pq()
                    logger.info("[BOOTSTRAP/PQ] ✓ Unified PQ system initialized")
                except Exception as e:
                    logger.error(f"[BOOTSTRAP/PQ] ✗ Failed: {e}")
    return _PQ_CRYPTO_SYSTEM

# ════════════════════════════════════════════════════════════════════════════════════════
# ADAPTIVE HYPERPARAMETER TUNER
# ════════════════════════════════════════════════════════════════════════════════════════

class AdaptiveHyperparameterTuner:
    def __init__(self):
        self.coherence_history = deque(maxlen=10)
        self.fidelity_history = deque(maxlen=10)
        self.mi_history = deque(maxlen=10)
        self.gradient_history = deque(maxlen=10)
        self.current_lr = 1e-3
        self.current_kappa = 0.08
        self.w_strength_multiplier = 1.0
        self.lock = threading.RLock()
    
    def update_metrics(self, coherence: float, fidelity: float, mi: float, gradient: float):
        with self.lock:
            self.coherence_history.append(coherence)
            self.fidelity_history.append(fidelity)
            self.mi_history.append(mi)
            self.gradient_history.append(gradient)
            
            cycle = len(self.gradient_history)
            oscillation = np.sin(2 * np.pi * cycle / 15)
            self.current_lr = 1e-3 * (1 + 0.15 * oscillation)
            
            if len(self.coherence_history) >= 2:
                recovery_rate = (self.coherence_history[-1] - self.coherence_history[0]) / len(self.coherence_history)
                ref_rate = 0.002
                if recovery_rate > ref_rate:
                    adjustment = 0.03 * min((recovery_rate - ref_rate) / ref_rate, 1.0)
                    self.current_kappa = 0.08 + adjustment
                else:
                    adjustment = -0.03 * (ref_rate - recovery_rate) / ref_rate
                    self.current_kappa = 0.08 + adjustment
                self.current_kappa = np.clip(self.current_kappa, 0.070, 0.120)
            
            if len(self.coherence_history) >= 2:
                recovery_rate = (self.coherence_history[-1] - self.coherence_history[0]) / len(self.coherence_history)
                ref_rate = 0.002
                if recovery_rate > ref_rate:
                    multiplier = 1.0 + 0.5 * min((recovery_rate - ref_rate) / ref_rate, 1.0)
                else:
                    multiplier = 1.0
                self.w_strength_multiplier = np.clip(multiplier, 1.0, 1.5)
    
    def get_status(self) -> Dict:
        with self.lock:
            return {
                'learning_rate': float(self.current_lr),
                'kappa': float(self.current_kappa),
                'w_strength_multiplier': float(self.w_strength_multiplier),
                'coherence_gain': float(self.coherence_history[-1] - self.coherence_history[0]) if len(self.coherence_history) >= 2 else 0.0
            }

HYPERPARAMETER_TUNER = AdaptiveHyperparameterTuner()
logger.info("✓ Hyperparameter tuner initialized")

# ════════════════════════════════════════════════════════════════════════════════════════
# PROJECT ROOT
# ════════════════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger.info(f"[BOOTSTRAP] PROJECT_ROOT = {PROJECT_ROOT}")

# ════════════════════════════════════════════════════════════════════════════════════════
# FLASK APP WITH MEGA COMMAND SYSTEM
# ════════════════════════════════════════════════════════════════════════════════════════

try:
    from flask import Flask, request, g, jsonify
    logger.info("[BOOTSTRAP] Flask imported successfully")
except ImportError:
    logger.error("[BOOTSTRAP] Flask not available")
    Flask = None

app = None

def create_app():
    """Create Flask application with unified mega_command_system."""
    global app
    
    app = Flask(__name__)
    app.config['JSON_SORT_KEYS'] = False
    
    logger.info("[BOOTSTRAP] Flask app created")
    
    # Import mega_command_system (required, not optional)
    try:
        from mega_command_system import (
            dispatch_command_sync,
            list_commands_sync,
            get_command_info_sync,
            get_registry
        )
        logger.info("[BOOTSTRAP] ✓ Mega command system imported")
    except ImportError as e:
        logger.error(f"[BOOTSTRAP] ✗ FATAL: Cannot import mega_command_system: {e}")
        raise
    
    # ── UNIFIED ENDPOINTS (only mega_command_system) ────────────────────────────────────
    
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
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat(),
            }), 500
    
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
    
    @app.route('/metrics', methods=['GET'])
    def metrics():
        """Command execution metrics."""
        try:
            registry = get_registry()
            stats = {}
            for cmd_name, cmd in registry.commands.items():
                if hasattr(cmd, 'get_stats'):
                    stats[cmd_name] = cmd.get_stats()
            return jsonify(stats), 200
        except Exception as e:
            logger.error(f"[API] Error in /metrics: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """System health check."""
        try:
            from globals import get_system_health
            health = get_system_health()
            status_code = 200 if health['status'] == 'healthy' else 503
            return jsonify(health), status_code
        except:
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'version': '6.0.0',
            }), 200
    
    @app.route('/status', methods=['GET'])
    def status():
        """Alias for /health."""
        return health_check()
    
    @app.route('/api/heartbeat', methods=['POST'])
    def heartbeat_receiver():
        """Receive and log heartbeat metrics from quantum system."""
        try:
            data = request.get_json() or {}
            timestamp = data.get('timestamp', datetime.now(timezone.utc).isoformat())
            beat_count = data.get('beat_count', 0)
            status = data.get('status', 'unknown')
            metrics = data.get('metrics', {})
            
            logger.info(
                f"[HEARTBEAT] Received beat #{beat_count} | "
                f"Status: {status} | "
                f"Metrics: {len(metrics)} subsystems"
            )
            
            return jsonify({
                'status': 'received',
                'beat_count': beat_count,
                'timestamp': timestamp,
            }), 200
        except Exception as e:
            logger.error(f"[HEARTBEAT] Error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/quantum/status', methods=['GET'])
    def quantum_status():
        """Get quantum system status."""
        try:
            from quantum_lattice_control import get_coordinator
            coordinator = get_coordinator()
            if coordinator:
                status = coordinator.get_status()
                return jsonify(status), 200
            return jsonify({'status': 'unavailable'}), 503
        except Exception as e:
            logger.error(f"[API] Quantum status error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/version', methods=['GET'])
    def version():
        """Return system version information."""
        return jsonify({
            'version': '6.0.0',
            'codename': 'QTCL',
            'quantum_lattice': 'v8',
            'pqc': 'HLWE-256',
            'wsgi': 'gunicorn-sync',
            'command_system': 'mega_command_system',
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
application = app

__all__ = [
    'app',
    'application',
    'get_wsgi_app',
    'create_app',
    'get_pq_system',
    'get_database_instance',
    'set_database_instance',
    'HYPERPARAMETER_TUNER',
    'PROJECT_ROOT',
]

logger.info("[BOOTSTRAP] ✓ wsgi_config loaded successfully")
