#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                                          ║
║     ⚛️  WSGI ENTRY POINT v1.0 — Quantum Lattice Blockchain Production Deployment ⚛️                    ║
║                                                                                                          ║
║  WSGI server entry point for Gunicorn, Heroku, Koyeb, Railway, Fly.io, etc                            ║
║  Minimal, clean, no side effects                                                                       ║
║                                                                                                          ║
║  Usage:                                                                                                 ║
║    gunicorn -w1 -b0.0.0.0:5000 wsgi:app                                                                ║
║    gunicorn -w1 -b0.0.0.0:$PORT wsgi:app  (with PORT env var - Koyeb/Heroku)                          ║
║                                                                                                          ║
║  This file simply imports and exports the Flask app from server.py                                    ║
║  All initialization happens in server.py and globals.py                                               ║
║                                                                                                          ║
║  Made by Claude. Museum-grade production code. 🚀⚛️💎                                                   ║
║                                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s]: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

logger.info("╔" + "═" * 86 + "╗")
logger.info("║" + " " * 86 + "║")
logger.info("║" + "  🌌 QUANTUM LATTICE BLOCKCHAIN — WSGI ENTRY POINT LOADING 🌌".center(86) + "║")
logger.info("║" + " " * 86 + "║")
logger.info("╚" + "═" * 86 + "╝")

# Import Flask app from server.py
logger.info("[WSGI] Importing Flask application from server.py...")

try:
    from server import app, application
    logger.info("[WSGI] ✅ Flask app imported successfully")
    logger.info("[WSGI] ✅ WSGI entry point ready")
except ImportError as e:
    logger.error(f"[WSGI] ❌ Failed to import app from server.py: {e}")
    logger.error("[WSGI] Make sure server.py exists and exports 'app' and 'application'")
    raise
except Exception as e:
    logger.error(f"[WSGI] ❌ Unexpected error during app import: {e}")
    import traceback
    traceback.print_exc()
    raise

logger.info("")
logger.info("╔" + "═" * 86 + "╗")
logger.info("║" + " " * 86 + "║")
logger.info("║" + "  ✅ WSGI APPLICATION READY FOR DEPLOYMENT".center(86) + "║")
logger.info("║" + " " * 86 + "║")
logger.info("║  Entry: wsgi:app".ljust(86) + "║")
logger.info("║  Command: gunicorn -w1 -b0.0.0.0:5000 wsgi:app".ljust(86) + "║")
logger.info("║" + " " * 86 + "║")
logger.info("╚" + "═" * 86 + "╝")


# This is the WSGI application object
# gunicorn expects to find 'application' or 'app' here
__all__ = ['app', 'application']
