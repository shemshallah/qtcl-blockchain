#!/usr/bin/env python3
"""
main_app.py — Unified application shim.

Production entry point is wsgi_config:application (see Procfile).
This file simply re-exports the canonical Flask app so that any import
of main_app.app or main_app.application also resolves correctly.

All logic, blueprints, globals, and command registry live in wsgi_config.py.
"""

import os
import sys
import logging

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configure logging ONLY if not already configured
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

logger = logging.getLogger(__name__)

try:
    # Import the canonical app — this initialises globals, registers blueprints,
    # boots terminal_logic, and wires all commands. Done exactly once.
    from wsgi_config import app, application, COMMAND_REGISTRY, dispatch_command
    logger.info('[main_app] ✓ Delegating to wsgi_config — single unified app')
except Exception as _e:
    logger.error(f'[main_app] FATAL: Could not import wsgi_config: {_e}', exc_info=True)
    raise

if __name__ == '__main__':
    port  = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    logger.info(f'[main_app] Starting on port {port} debug={debug}')
    app.run(host='0.0.0.0', port=port, debug=debug)
