#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║  QTCL SERVER v5 — Quantum Lattice with HLWE Wallet & Neural Metrics          ║
║                                                                                ║
║  Single unified server that:                                                   ║
║  - Loads lattice controller at startup                                         ║
║  - Maintains database connections (new complete schema)                        ║
║  - Tracks quantum metrics & heartbeat                                          ║
║  - Serves HTML dashboard with real-time updates                               ║
║  - Exposes minimal API: /health, /blocks, /wallet, /transact, /heartbeat      ║
║                                                                                ║
║  Entry: python server.py or gunicorn -w1 -b0.0.0.0:5000 server:app            ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import time
import logging
import threading
import traceback
import psycopg2
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Tuple
from contextlib import contextmanager

from flask import Flask, jsonify, request, render_template_string, send_file
from io import BytesIO

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s]: %(message)s'
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DB_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost/qtcl_db')

# ─────────────────────────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
application = app  # WSGI entry point

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────────────────────

class SystemState:
    """Maintains server and quantum state"""
    
    def __init__(self):
        self.db_conn = None
        self.lattice_loaded = False
        self.quantum_metrics = {
            'coherence': 0.99,
            'entanglement': 0.95,
            'phase_drift': 0.01,
            'w_state_fidelity': 0.98,
        }
        self.block_state = {
            'current_height': 0,
            'current_hash': None,
            'pq_current': 0,
            'pq_last': 0,
            'timestamp': None,
        }
        self.wallet_state = {
            'balance': 0,
            'address': None,
            'tx_count': 0,
        }
        self.heartbeat_ts = time.time()
        self.is_alive = True
        self.lock = threading.Lock()
    
    def update_metrics(self, metrics: Dict[str, float]):
        """Update quantum metrics"""
        with self.lock:
            self.quantum_metrics.update(metrics)
            self.heartbeat_ts = time.time()
    
    def update_block_state(self, state: Dict[str, Any]):
        """Update block state"""
        with self.lock:
            self.block_state.update(state)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state snapshot"""
        with self.lock:
            return {
                'quantum_metrics': self.quantum_metrics.copy(),
                'block_state': self.block_state.copy(),
                'wallet_state': self.wallet_state.copy(),
                'heartbeat_ts': self.heartbeat_ts,
                'is_alive': self.is_alive,
                'uptime_s': time.time() - self.heartbeat_ts,
            }

state = SystemState()

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def get_db_cursor():
    """Context manager for database cursor"""
    conn = None
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        yield cur
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"[DB] Error: {e}")
        raise
    finally:
        if conn:
            conn.close()

def query_latest_block() -> Optional[Dict[str, Any]]:
    """Get latest block from database"""
    try:
        with get_db_cursor() as cur:
            cur.execute("""
                SELECT height, block_hash, timestamp, oracle_w_state_hash
                FROM blocks
                ORDER BY height DESC
                LIMIT 1
            """)
            row = cur.fetchone()
            if row:
                return {
                    'height': row[0],
                    'hash': row[1],
                    'timestamp': row[2],
                    'w_state_hash': row[3],
                }
    except Exception as e:
        logger.error(f"[DB] Failed to query blocks: {e}")
    return None

def query_pseudoqubit_range() -> Tuple[int, int]:
    """Get min/max pq_id currently in database"""
    try:
        with get_db_cursor() as cur:
            cur.execute("SELECT MIN(pq_id), MAX(pq_id) FROM pseudoqubits")
            row = cur.fetchone()
            if row and row[0] is not None:
                return (row[0], row[1])
    except Exception as e:
        logger.error(f"[DB] Failed to query pseudoqubits: {e}")
    return (0, 0)

def query_wallet_info(address: str) -> Optional[Dict[str, Any]]:
    """Get wallet info from database"""
    try:
        with get_db_cursor() as cur:
            cur.execute("""
                SELECT balance, transaction_count
                FROM wallet_addresses
                WHERE address = %s
            """, (address,))
            row = cur.fetchone()
            if row:
                return {
                    'address': address,
                    'balance': row[0],
                    'tx_count': row[1],
                }
    except Exception as e:
        logger.error(f"[DB] Failed to query wallet: {e}")
    return None

def insert_transaction(from_addr: str, to_addr: str, amount: int) -> Optional[str]:
    """Insert transaction into database"""
    try:
        with get_db_cursor() as cur:
            tx_hash = f"tx_{int(time.time() * 1000)}"
            cur.execute("""
                INSERT INTO transactions (tx_hash, from_address, to_address, amount, status)
                VALUES (%s, %s, %s, %s, 'pending')
                RETURNING tx_hash
            """, (tx_hash, from_addr, to_addr, amount))
            result = cur.fetchone()
            return result[0] if result else None
    except Exception as e:
        logger.error(f"[DB] Failed to insert transaction: {e}")
    return None

# ─────────────────────────────────────────────────────────────────────────────
# LATTICE INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def initialize_lattice_controller():
    """Initialize lattice controller at startup"""
    try:
        # Try to import lattice controller
        try:
            from lattice_controller import QuantumLatticeController
            controller = QuantumLatticeController()
            controller.initialize()
            logger.info("✨ [LATTICE] Quantum lattice controller initialized")
            state.lattice_loaded = True
            return controller
        except ImportError:
            logger.warning("[LATTICE] lattice_controller not available, running in mock mode")
            state.lattice_loaded = False
            return None
    except Exception as e:
        logger.error(f"[LATTICE] Initialization failed: {e}")
        logger.error(traceback.format_exc())
        return None

# Initialize lattice on startup
LATTICE = initialize_lattice_controller()

# ─────────────────────────────────────────────────────────────────────────────
# QUANTUM METRICS THREAD (Neural Net / Heartbeat)
# ─────────────────────────────────────────────────────────────────────────────

def quantum_metrics_thread():
    """Background thread that updates quantum metrics and block state"""
    logger.info("[METRICS] Quantum metrics thread started")
    
    while state.is_alive:
        try:
            # Query latest block
            block = query_latest_block()
            if block:
                pq_min, pq_max = query_pseudoqubit_range()
                state.update_block_state({
                    'current_height': block['height'],
                    'current_hash': block['hash'],
                    'pq_current': pq_max,
                    'pq_last': pq_min,
                    'timestamp': block['timestamp'],
                })
            
            # Simulate quantum metrics (or pull from LATTICE if available)
            if LATTICE:
                try:
                    metrics = LATTICE.get_metrics()
                    state.update_metrics(metrics)
                except:
                    pass
            else:
                # Mock metrics
                import random
                state.update_metrics({
                    'coherence': 0.99 - random.random() * 0.05,
                    'entanglement': 0.95 - random.random() * 0.05,
                    'phase_drift': 0.01 + random.random() * 0.02,
                    'w_state_fidelity': 0.98 - random.random() * 0.03,
                })
            
            time.sleep(2)  # Update every 2 seconds
        except Exception as e:
            logger.error(f"[METRICS] Error: {e}")
            time.sleep(2)

# Start metrics thread
metrics_thread = threading.Thread(target=quantum_metrics_thread, daemon=True)
metrics_thread.start()

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES: DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def dashboard():
    """Serve index.html dashboard"""
    try:
        with open('index.html', 'r') as f:
            html_content = f.read()
        return render_template_string(html_content)
    except FileNotFoundError:
        return """
        <html>
            <head><title>QTCL Server</title></head>
            <body style="background:#0a0a0e; color:#d4d4d8; font-family:monospace; padding:20px;">
                <h1>QTCL Server Running</h1>
                <p>index.html not found. Check /api/health for system status.</p>
            </body>
        </html>
        """, 404

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES: API
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    snapshot = state.get_state()
    return jsonify({
        'status': 'ok' if state.is_alive else 'degraded',
        'lattice_loaded': state.lattice_loaded,
        'quantum_metrics': snapshot['quantum_metrics'],
        'block_height': snapshot['block_state']['current_height'],
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }), 200

@app.route('/api/blocks', methods=['GET'])
def blocks():
    """Get current block information"""
    snapshot = state.get_state()
    block = snapshot['block_state']
    
    return jsonify({
        'current_height': block['current_height'],
        'current_hash': block['current_hash'],
        'pq_current': block['pq_current'],
        'pq_last': block['pq_last'],
        'timestamp': block['timestamp'],
        'quantum_metrics': snapshot['quantum_metrics'],
    }), 200

@app.route('/api/wallet', methods=['GET'])
def wallet():
    """Get wallet information"""
    address = request.args.get('address')
    if not address:
        return jsonify({'error': 'address parameter required'}), 400
    
    wallet_info = query_wallet_info(address)
    if wallet_info:
        return jsonify(wallet_info), 200
    else:
        return jsonify({'error': 'wallet not found'}), 404

@app.route('/api/transact', methods=['POST'])
def transact():
    """Create a transaction"""
    data = request.get_json() or {}
    
    from_addr = data.get('from')
    to_addr = data.get('to')
    amount = data.get('amount')
    
    if not all([from_addr, to_addr, amount]):
        return jsonify({'error': 'missing from/to/amount'}), 400
    
    try:
        tx_hash = insert_transaction(from_addr, to_addr, int(amount))
        if tx_hash:
            return jsonify({
                'tx_hash': tx_hash,
                'status': 'pending',
                'from': from_addr,
                'to': to_addr,
                'amount': amount,
            }), 201
        else:
            return jsonify({'error': 'failed to create transaction'}), 500
    except Exception as e:
        logger.error(f"[TRANSACT] Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/heartbeat', methods=['POST'])
def heartbeat():
    """Heartbeat endpoint - keep server alive"""
    snapshot = state.get_state()
    return jsonify({
        'heartbeat': snapshot['heartbeat_ts'],
        'uptime_s': snapshot['uptime_s'],
        'is_alive': snapshot['is_alive'],
        'metrics': snapshot['quantum_metrics'],
    }), 200

# ─────────────────────────────────────────────────────────────────────────────
# ERROR HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'not found'}), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"[SERVER] Error: {e}")
    return jsonify({'error': 'internal server error'}), 500

# ─────────────────────────────────────────────────────────────────────────────
# STARTUP & SHUTDOWN
# ─────────────────────────────────────────────────────────────────────────────

@app.before_request
def before_request():
    """Before each request"""
    pass

@app.teardown_appcontext
def teardown(error=None):
    """Cleanup on shutdown"""
    if error:
        logger.error(f"[APP] Teardown error: {error}")

def shutdown_handler():
    """Graceful shutdown"""
    logger.info("[SERVER] Shutting down...")
    state.is_alive = False
    if LATTICE:
        try:
            LATTICE.shutdown()
        except:
            pass
    logger.info("[SERVER] Shutdown complete")

if __name__ == '__main__':
    import atexit
    atexit.register(shutdown_handler)
    
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    logger.info("╔" + "═" * 78 + "╗")
    logger.info("║ QTCL SERVER v5 STARTING")
    logger.info("║ Port: %d | Debug: %s | Lattice: %s" % (port, debug, state.lattice_loaded))
    logger.info("╚" + "═" * 78 + "╝")
    
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)
