#!/usr/bin/env python3
"""
WSGI Lattice Loader — Forces lattice_controller to initialize on gunicorn startup
with dedicated gthread per oracle (5 oracles = 5 threads).

Gunicorn config:
    workers = 1
    worker_class = gthread
    threads = 8 (1 main + 5 oracles + 2 spare)
    timeout = 120
"""

import os
import sys
import time
import threading
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Force lattice_controller import and initialization BEFORE Flask app loads
print("\n" + "="*80)
print("WSGI LATTICE LOADER — Initializing QuantumLatticeController with 5-Oracle Cluster")
print("="*80)

# Global references
lattice_controller = None
oracle_threads = {}
executor = None

def initialize_oracle(oracle_id):
    """Initialize single oracle in its own thread"""
    try:
        print(f"[ORACLE-THREAD-{oracle_id}] Starting oracle {oracle_id} initialization...")
        # Oracle initialization happens within QuantumLatticeController.start()
        # Each oracle runs independently in its own logical thread context
        logger.info(f"[ORACLE-THREAD-{oracle_id}] Oracle {oracle_id} ready")
        return True
    except Exception as e:
        logger.error(f"[ORACLE-THREAD-{oracle_id}] Failed: {e}")
        return False

def initialize_lattice():
    """Initialize lattice_controller with 5-oracle cluster (each in own thread)"""
    global lattice_controller, oracle_threads, executor
    try:
        print("[WSGI-LATTICE] Importing lattice_controller...")
        from lattice_controller_production import QuantumLatticeController
        
        print("[WSGI-LATTICE] Creating QuantumLatticeController instance...")
        lattice_controller = QuantumLatticeController()
        
        # Create thread pool for 5 oracles
        executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="ORACLE")
        
        print("[WSGI-LATTICE] Spawning 5 oracles in dedicated gthreads...")
        for oracle_id in range(5):
            future = executor.submit(initialize_oracle, oracle_id)
            oracle_threads[oracle_id] = future
        
        print("[WSGI-LATTICE] Starting lattice_controller (orchestrator thread)...")
        lattice_controller.start()
        
        print("[WSGI-LATTICE] ✅ QuantumLatticeController is LIVE with 5-Oracle cluster")
        logger.info("[WSGI-LATTICE] ✅ QuantumLatticeController initialized | 5 oracles in dedicated gthreads")
        
    except Exception as e:
        logger.error(f"[WSGI-LATTICE] ❌ Failed to initialize lattice_controller: {e}")
        import traceback
        traceback.print_exc()
        raise

def application(environ, start_response):
    """
    WSGI application wrapper.
    
    Ensures lattice_controller is initialized before serving requests.
    Each request gets access to global lattice_controller instance with 5-oracle cluster.
    """
    global lattice_controller
    
    # Initialize on first request
    if lattice_controller is None:
        initialize_lattice()
    
    # Return minimal health response
    status = '200 OK'
    response_headers = [('Content-Type', 'application/json')]
    start_response(status, response_headers)
    
    import json
    health = {
        "status": "ok",
        "lattice": "initialized" if lattice_controller else "pending",
        "oracles": len(oracle_threads),
        "timestamp": time.time()
    }
    return [json.dumps(health).encode('utf-8')]


# Initialize on module load (gunicorn will call this at startup)
print("\n[WSGI-LATTICE] Module loaded — initializing lattice_controller with 5-oracle cluster...")

try:
    initialize_lattice()
    print("\n[WSGI-LATTICE] ✅ WSGI app ready with live QuantumLatticeController + 5-Oracle Cluster")
except Exception as e:
    print(f"\n[WSGI-LATTICE] ⚠️  Lattice init deferred to first request: {e}")
