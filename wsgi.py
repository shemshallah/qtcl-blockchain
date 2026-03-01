#!/usr/bin/env python3
"""
wsgi.py — WSGI Entry Point for Production Deployment

This file is the entry point for gunicorn/WSGI servers.
Used by Koyeb, Heroku, Railway, Fly.io, etc.

Entry: gunicorn wsgi:app
"""

import os
import logging

# Configure logging before importing app
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s]: %(message)s'
)

logger = logging.getLogger(__name__)

# Import the Flask app from server.py
try:
    from server import app, application
    logger.info("✅ [WSGI] App loaded successfully")
except Exception as e:
    logger.error(f"❌ [WSGI] Failed to load app: {e}")
    raise

# WSGI entry point for gunicorn
if __name__ == '__main__':
    logger.info("[WSGI] Server started by gunicorn")
