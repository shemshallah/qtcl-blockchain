#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                            ║
║  ⚛️  WSGI ENTRY POINT v2.2 — INSTANT HEALTH + DEFERRED INIT ⚛️                        ║
║                                                                                            ║
║  Unified WSGI server entry point for Gunicorn, Heroku, Koyeb, Railway, Fly.io, etc.      ║
║  Museum-grade orchestration with proper initialization sequencing.                        ║
║                                                                                            ║
║  KEY FEATURES:                                                                            ║
║    • /health returns 200 IMMEDIATELY on startup (no blocking imports)                     ║
║    • Heavy initialization deferred to background threads                                   ║
║    • Database and lattice init in background                                              ║
║                                                                                            ║
║  Made by Claude. Museum-grade production code. Zero shortcuts. 🚀⚛️💎                    ║
║                                                                                            ║
╚════════════════════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import threading
import logging

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_HYP_DIR = os.path.join(_REPO_ROOT, 'hlwe')
if _HYP_DIR not in sys.path:
    sys.path.insert(0, _HYP_DIR)

_STARTUP_TIME = time.time()

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s]: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

logger.info("╔" + "═" * 90 + "╗")
logger.info("║" + "  🌌 QUANTUM LATTICE BLOCKCHAIN — WSGI ENTRY POINT v2.2 LOADING 🌌".center(90) + "║")
logger.info("╚" + "═" * 90 + "╝")

# ═══ PHASE 0: Create minimal health app BEFORE heavy imports ═══
# This allows /health to respond immediately while heavy init happens in background
logger.info("[WSGI-0] Creating instant health Flask app...")

from flask import Flask, Response
_health_app = Flask('__health__')

@_health_app.route("/health")
def instant_health():
    """INSTANT health check - no dependencies, returns immediately."""
    return "", 200

@_health_app.route("/ready")
def instant_ready():
    """Readiness probe - checks if full app is loaded."""
    try:
        from server import _LATTICE_READY, _DB_READY
        if _LATTICE_READY and (_DB_READY or os.getenv('DATABASE_URL', '') == ''):
            return "", 200
        return "", 503
    except:
        return "", 503

logger.info(f"[WSGI-0] ✅ Instant health app ready at {time.time() - _STARTUP_TIME:.2f}s")

# ═══ PHASE 1: Start heavy imports in background ═══
def _load_server_in_background():
    """Load server.py (with all heavy imports) in background thread."""
    global _server_loaded
    try:
        from server import app, application
        _server_loaded = True
        logger.info(f"[WSGI-BG] ✅ Server module loaded at {time.time() - _STARTUP_TIME:.2f}s")
    except Exception as e:
        logger.error(f"[WSGI-BG] ❌ Server load failed: {e}")

_server_loaded = False
_server_thread = threading.Thread(target=_load_server_in_background, daemon=True, name='ServerLoad')
_server_thread.start()

# ═══ PHASE 2: Use full app once loaded, fall back to health app ═══
class _DualApp:
    """WSGI app that serves health instantly, then switches to full app."""
    def __init__(self, health_app):
        self._health = health_app
        self._full = None
        self._lock = threading.Lock()
    
    def _get_full(self):
        if self._full is None:
            with self._lock:
                if self._full is None:
                    try:
                        from server import app as _fa
                        self._full = _fa
                        logger.info(f"[WSGI] ✅ Full app active at {time.time() - _STARTUP_TIME:.2f}s")
                    except:
                        pass
        return self._full
    
    def __call__(self, environ, start_response):
        path = environ.get('PATH_INFO', '')
        # Serve /health and /ready from instant health app
        if path in ('/health', '/ready', '/health/', '/ready/'):
            return self._health(environ, start_response)
        # For all other paths, use full app if loaded
        full = self._get_full()
        if full is not None:
            return full(environ, start_response)
        # Full app not ready yet - return 503
        status = '503 Service Initializing'
        headers = [('Content-Type', 'text/plain')]
        start_response(status, headers)
        return [b'Service initializing, please retry...']

# Create the dual app
app = _DualApp(_health_app)
application = app  # alias for gunicorn

logger.info(f"[WSGI] ✅ Dual app ready at {time.time() - _STARTUP_TIME:.2f}s — /health responds immediately")
logger.info("[WSGI] Background server loading in progress...")

__all__ = ['app', 'application']
