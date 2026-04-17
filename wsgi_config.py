#!/usr/bin/env python3
"""
WSGI entry point for Gunicorn/Koyeb.

Loads server module directly - /health is already defined in server.py.
Lattice and oracle initialize in background threads (non-blocking).
"""

import os
import sys
import time

# Add hlwe to path
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_HYP_DIR = os.path.join(_REPO_ROOT, 'hlwe')
if _HYP_DIR not in sys.path:
    sys.path.insert(0, _HYP_DIR)

_STARTUP = time.time()

print(f"[WSGI] Loading QTCL blockchain server...")

# Import server - /health endpoint is defined in server.py
from server import app, application

print(f"[WSGI] ✅ Server ready at {time.time() - _STARTUP:.1f}s")
print(f"[WSGI] /health endpoint available immediately")
print(f"[WSGI] Lattice/oracle initializing in background...")

__all__ = ['app', 'application']
