# QUANTUM TEMPORAL COHERENCE LEDGER - PROCFILE (KOYEB)
# Production-Grade Process Management - FIXED FOR CIRCULAR IMPORT STABILITY
# 
# CRITICAL CHANGE: Running ONLY web process to stabilize initialization.
# All worker processes are DISABLED until circular import is verified fixed.
# They were causing 7x re-initialization from logging.basicConfig and module re-imports.

web: gunicorn -w 4 -b 0.0.0.0:${PORT:-5000} --worker-class sync --timeout 300 --access-logfile - --error-logfile - wsgi_config:application

# DISABLED UNTIL CIRCULAR IMPORT STABILITY VERIFIED
# worker: python -c "import sys,logging,time; logging.basicConfig(level=logging.INFO); sys.path.insert(0, '.'); from quantum_lattice_control_live_complete import LATTICE; from db_builder_v2 import db_manager; print('[WORKER] Started'); [(print('[WORKER] Processing...'), time.sleep(10)) for _ in iter(int, 1)]"
# 
# oracle: python -c "import sys,logging,time; logging.basicConfig(level=logging.INFO); sys.path.insert(0, '.'); from oracle_api import blueprint; print('[ORACLE] Started'); [(print('[ORACLE] Feed...'), time.sleep(30)) for _ in iter(int, 1)]"
# 
# release: python db_builder_v2.py
# 
# scheduler: python -c "import sys,logging,time; logging.basicConfig(level=logging.INFO); sys.path.insert(0, '.'); from quantum_lattice_control_live_complete import HEARTBEAT; print('[SCHEDULER] Started'); [(print('[SCHEDULER] Health...'), time.sleep(60)) for _ in iter(int, 1)]"
# 
# finality: python -c "import sys,logging,time; logging.basicConfig(level=logging.INFO); sys.path.insert(0, '.'); from blockchain_api import blueprint; print('[FINALITY] Started'); [(print('[FINALITY] Check...'), time.sleep(5)) for _ in iter(int, 1)]"
# 
# cache-warmer: python -c "import sys,logging,time; logging.basicConfig(level=logging.INFO); sys.path.insert(0, '.'); print('[CACHE] Started'); [(print('[CACHE] Warm...'), time.sleep(300)) for _ in iter(int, 1)]"
# 
# cleanup: python -c "import sys,logging,time; logging.basicConfig(level=logging.INFO); sys.path.insert(0, '.'); print('[CLEANUP] Started'); [(print('[CLEANUP] Clean...'), time.sleep(300)) for _ in iter(int, 1)]"
# 
# defi-monitor: python -c "import sys,logging,time; logging.basicConfig(level=logging.INFO); sys.path.insert(0, '.'); from defi_api import blueprint; print('[DEFI] Started'); [(print('[DEFI] Monitor...'), time.sleep(60)) for _ in iter(int, 1)]"
# 
# governance: python -c "import sys,logging,time; logging.basicConfig(level=logging.INFO); sys.path.insert(0, '.'); print('[GOVERNANCE] Started'); [(print('[GOVERNANCE] Vote...'), time.sleep(120)) for _ in iter(int, 1)]"
# 
# admin: python -c "import sys,logging,time; logging.basicConfig(level=logging.INFO); sys.path.insert(0, '.'); from admin_api import AdminSessionManager, blueprint; print('[ADMIN] Started'); [(print('[ADMIN] Ready...'), time.sleep(30)) for _ in iter(int, 1)]"
# 
# heartbeat: python -c "import sys,logging,time; logging.basicConfig(level=logging.INFO); sys.path.insert(0, '.'); from quantum_lattice_control_live_complete import HEARTBEAT; print('[HEARTBEAT] Started'); [(print('[HEARTBEAT] Pulse...'), time.sleep(10)) for _ in iter(int, 1)]"
