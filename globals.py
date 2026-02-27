#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                        ║
║            🚀 QTCL v5.0 GLOBAL STATE & CONFIGURATION (INTEGRATED v3.0) 🚀             ║
║                                                                                        ║
║  Pure global state with QRNG Ensemble & HLWE Engine integration                       ║
║  Command routing moved to mega_command_system, crypto in hlwe_engine                  ║
║  Complete QRNG awareness, atomic entropy operations everywhere                        ║
║                                                                                        ║
╚════════════════════════════════════════════════════════════════════════════════════════╝
"""

import json,logging,threading,platform as _platform
from collections import defaultdict,deque as _deque
from datetime import datetime,timezone
from typing import Dict,Any,Optional,List,Tuple,Callable
import traceback

if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger=logging.getLogger(__name__)

_GLOBAL_STATE={
    'initialized':False,'_initializing':False,'lock':threading.Lock(),
    'qrng_ensemble':None,'qrng_stats':None,'qrng_last_refresh':0.0,'qrng_circuit_breakers':{},'qrng_pool_size':0,'qrng_active_sources':0,'qrng_entropy_estimate':0.0,
    'hlwe_system':None,'hlwe_params':None,'pqc_system':None,'pqc_state':None,
}

_STATE_LOCK=threading.RLock()

def get_global_state(key:str,default:Any=None)->Any:
    """Thread-safe getter for global state"""
    with _STATE_LOCK:return _GLOBAL_STATE.get(key,default)

def set_global_state(key:str,value:Any)->None:
    """Thread-safe setter for global state"""
    with _STATE_LOCK:
        _GLOBAL_STATE[key]=value
        logger.debug(f"[GLOBAL] Set {key} = {type(value).__name__}")

def update_global_state(updates:Dict[str,Any])->None:
    """Batch update global state atomically"""
    with _STATE_LOCK:
        _GLOBAL_STATE.update(updates)
        logger.debug(f"[GLOBAL] Batch update: {len(updates)} keys")

def get_qrng_ensemble():
    """Get QRNG ensemble singleton (lazy-load aware)"""
    return get_global_state('qrng_ensemble')

def initialize_qrng_ensemble()->bool:
    """Initialize QRNG ensemble with 5-source quantum entropy"""
    try:
        from qrng_ensemble import get_qrng_ensemble as create_qrng,QuantumEntropyEnsemble
        qrng=create_qrng()
        set_global_state('qrng_ensemble',qrng)
        logger.info("[GLOBAL] ✓ QRNG Ensemble initialized with 5 sources")
        return True
    except Exception as e:
        logger.warning(f"[GLOBAL] QRNG Ensemble initialization failed: {e}")
        return False

def refresh_qrng_stats()->Dict[str,Any]:
    """Refresh QRNG statistics in global state"""
    qrng=get_qrng_ensemble()
    if not qrng:return {}
    try:
        stats=qrng.get_entropy_stats()
        active=sum(1 for s in stats.get('sources',{}).values()if s.get('success_rate',0)>50)
        update_global_state({
            'qrng_stats':stats,
            'qrng_pool_size':stats.get('pool_size',0),
            'qrng_active_sources':active,
            'qrng_entropy_estimate':qrng.get_entropy_estimate(),
            'qrng_last_refresh':datetime.now(timezone.utc).timestamp(),
            'qrng_circuit_breakers':stats.get('circuit_breakers',{})
        })
        return stats
    except Exception as e:
        logger.warning(f"[GLOBAL] QRNG stats refresh failed: {e}")
        return {}

def get_hlwe_system():
    """Get HLWE cryptographic system singleton"""
    return get_global_state('hlwe_system')

def initialize_hlwe_engine(db_url:Optional[str]=None,params=None)->bool:
    """Initialize HLWE cryptographic engine with QRNG integration"""
    try:
        from hlwe_engine import get_pq_system,HLWE_256
        qrng=get_qrng_ensemble()
        hlwe=get_pq_system(params or HLWE_256,db_url,qrng)
        set_global_state('hlwe_system',hlwe)
        set_global_state('pqc_system',hlwe)
        logger.info("[GLOBAL] ✓ HLWE Engine initialized with QRNG ensemble integration")
        return True
    except Exception as e:
        logger.warning(f"[GLOBAL] HLWE Engine initialization failed: {e}")
        return False

def get_heartbeat()->Optional[Any]:
    return get_global_state('heartbeat')

def get_lattice()->Optional[Any]:
    return get_global_state('lattice')

def get_blockchain()->Optional[Any]:
    return get_global_state('blockchain')

def get_db_pool()->Optional[Any]:
    return get_global_state('db_pool')

def get_db_manager()->Optional[Any]:
    return get_global_state('db_manager')

def get_ledger()->Optional[Any]:
    return get_global_state('ledger')

def get_oracle()->Optional[Any]:
    return get_global_state('oracle')

def get_defi()->Optional[Any]:
    return get_global_state('defi')

def get_auth_manager()->Optional[Any]:
    return get_global_state('auth_manager')

def get_pqc_system()->Optional[Any]:
    """Get PQC system (uses HLWE engine internally)"""
    return get_global_state('pqc_system')

class QuantumMetricsHarvester:
    """Lightweight harvester for 15-second metric intervals with QRNG metrics"""
    def __init__(self,db_connection_getter:Optional[Callable]=None):
        self.get_db=db_connection_getter
        self.running=False
        self.harvest_interval=15
        self.verbose_interval=30
        self.harvest_count=0
        self.write_count=0
        self.error_count=0
        self._lock=threading.RLock()
    def harvest(self)->Dict[str,Any]:
        """Collect current metrics from global state including QRNG"""
        try:
            metrics={
                'timestamp':datetime.now(timezone.utc).isoformat(),
                'engine':'QTCL-QE v8.0 + HLWE v2.0',
                'source':'live_harvest',
                'python_version':_platform.python_version(),
                'platform':_platform.platform(),
            }
            lattice=get_lattice()
            if lattice is not None and hasattr(lattice,'get_metrics'):
                try:
                    quantum_metrics=lattice.get_metrics()
                    metrics['quantum']=quantum_metrics
                except Exception as e:
                    logger.warning(f"[Harvester] Failed to get quantum metrics: {e}")
            blockchain=get_blockchain()
            if blockchain is not None and hasattr(blockchain,'get_metrics'):
                try:
                    chain_metrics=blockchain.get_metrics()
                    metrics['blockchain']=chain_metrics
                except Exception as e:
                    logger.warning(f"[Harvester] Failed to get blockchain metrics: {e}")
            qrng=get_qrng_ensemble()
            if qrng:
                try:
                    stats=refresh_qrng_stats()
                    metrics['qrng']={
                        'pool_size':stats.get('pool_size',0),
                        'pool_max':stats.get('pool_max_size',0),
                        'active_sources':get_global_state('qrng_active_sources',0),
                        'total_sources':len(stats.get('sources',{})),
                        'entropy_estimate':get_global_state('qrng_entropy_estimate',0.0)
                    }
                except Exception as e:
                    logger.warning(f"[Harvester] Failed to get QRNG metrics: {e}")
            return metrics
        except Exception as e:
            self.error_count+=1
            logger.error(f"[Harvester] Error during harvest: {e}",exc_info=True)
            return {'timestamp':datetime.now(timezone.utc).isoformat(),'error':str(e)}
    def get_status(self)->Dict[str,Any]:
        """Return harvester status"""
        with self._lock:
            return {
                'running':self.running,
                'harvest_count':self.harvest_count,
                'write_count':self.write_count,
                'error_count':self.error_count,
                'last_harvest':get_global_state('metrics_last_harvest'),
            }

def get_metrics()->Dict[str,Any]:
    """Get system metrics snapshot including QRNG and HLWE status"""
    metrics={
        'timestamp':datetime.now(timezone.utc).isoformat(),
        'quantum':{},
        'blockchain':{},
        'database':{},
        'qrng':{},
        'hlwe':{},
        'system':{
            'python':_platform.python_version(),
            'platform':_platform.platform(),
        },
    }
    lattice=get_lattice()
    if lattice is not None and hasattr(lattice,'get_metrics'):
        try:
            metrics['quantum']=lattice.get_metrics()
        except Exception as e:
            logger.warning(f"[metrics] Failed to get quantum metrics: {e}")
    blockchain=get_blockchain()
    if blockchain is not None and hasattr(blockchain,'get_metrics'):
        try:
            metrics['blockchain']=blockchain.get_metrics()
        except Exception as e:
            logger.warning(f"[metrics] Failed to get blockchain metrics: {e}")
    qrng=get_qrng_ensemble()
    if qrng:
        try:
            stats=refresh_qrng_stats()
            metrics['qrng']={
                'pool_size':stats.get('pool_size',0),
                'pool_max':stats.get('pool_max_size',0),
                'pool_fill_percent':stats.get('pool_fill_percentage',0.0),
                'active_sources':get_global_state('qrng_active_sources',0),
                'entropy_estimate':get_global_state('qrng_entropy_estimate',0.0),
            }
        except Exception as e:
            logger.warning(f"[metrics] Failed to get QRNG metrics: {e}")
    hlwe=get_hlwe_system()
    if hlwe:
        try:
            metrics['hlwe']=hlwe.get_system_status()
        except Exception as e:
            logger.warning(f"[metrics] Failed to get HLWE metrics: {e}")
    return metrics

def get_module_status()->Dict[str,str]:
    """Get status of all system modules"""
    status={
        'heartbeat':'online' if get_heartbeat() is not None else 'offline',
        'lattice':'online' if get_lattice() is not None else 'offline',
        'blockchain':'online' if get_blockchain() is not None else 'offline',
        'database':'online' if get_db_pool() is not None else 'offline',
        'ledger':'online' if get_ledger() is not None else 'offline',
        'oracle':'online' if get_oracle() is not None else 'offline',
        'defi':'online' if get_defi() is not None else 'offline',
        'auth':'online' if get_auth_manager() is not None else 'offline',
        'pqc':'online' if get_pqc_system() is not None else 'offline',
        'qrng':'online' if get_qrng_ensemble() is not None else 'offline',
    }
    return status

def get_metric(metric_name:str,default:Any=None)->Any:
    """Get a metric by name (legacy API)"""
    metrics=get_metrics()
    parts=metric_name.split('.')
    current=metrics
    for part in parts:
        if isinstance(current,dict):
            current=current.get(part)
        else:
            return default
    return current if current is not None else default

def initialize_globals(db_url:Optional[str]=None)->bool:
    """Initialize global state with QRNG and HLWE integration. Called by wsgi_config at startup."""
    with _STATE_LOCK:
        if get_global_state('initialized'):
            logger.info("[GLOBAL] Already initialized, skipping")
            return True
        if get_global_state('_initializing'):
            logger.warning("[GLOBAL] Initialization in progress, preventing recursion")
            return False
        try:
            set_global_state('_initializing',True)
            set_global_state('metrics_harvester',QuantumMetricsHarvester())
            initialize_qrng_ensemble()
            initialize_hlwe_engine(db_url)
            set_global_state('initialized',True)
            logger.info("[GLOBAL] ✓ Global state initialized (QRNG + HLWE)")
            return True
        except Exception as e:
            logger.error(f"[GLOBAL] ✗ Failed to initialize: {e}",exc_info=True)
            return False
        finally:
            set_global_state('_initializing',False)

def get_system_health()->Dict[str,Any]:
    """Get comprehensive system health status"""
    return {
        'status':'healthy' if get_global_state('initialized') else 'degraded',
        'timestamp':datetime.now(timezone.utc).isoformat(),
        'modules':get_module_status(),
        'version':'5.0.0',
        'codename':'QTCL',
        'quantum_lattice':'v8',
        'pqc':'HLWE-256 + QRNG Ensemble',
        'wsgi':'gunicorn-sync',
    }

def shutdown_globals()->None:
    """Clean up global state during shutdown"""
    with _STATE_LOCK:
        logger.info("[GLOBAL] Shutting down global state")
        harvester=get_global_state('metrics_harvester')
        if harvester is not None and hasattr(harvester,'running'):
            harvester.running=False
        db_pool=get_global_state('db_pool')
        if db_pool is not None and hasattr(db_pool,'closeall'):
            try:
                db_pool.closeall()
                logger.info("[GLOBAL] Database pool closed")
            except Exception as e:
                logger.error(f"[GLOBAL] Error closing DB pool: {e}")
        qrng=get_qrng_ensemble()
        if qrng is not None and hasattr(qrng,'close'):
            try:
                qrng.close()
                logger.info("[GLOBAL] QRNG ensemble closed")
            except Exception as e:
                logger.error(f"[GLOBAL] Error closing QRNG: {e}")
        hlwe=get_hlwe_system()
        if hlwe is not None and hasattr(hlwe,'close'):
            try:
                hlwe.close()
                logger.info("[GLOBAL] HLWE system closed")
            except Exception as e:
                logger.error(f"[GLOBAL] Error closing HLWE: {e}")
        logger.info("[GLOBAL] ✓ Global state shutdown complete")

__all__=[
    'get_global_state','set_global_state','update_global_state',
    'get_heartbeat','get_lattice','get_blockchain','get_db_pool','get_db_manager','get_ledger','get_oracle','get_defi','get_auth_manager','get_pqc_system',
    'get_metrics','get_metric','get_module_status','get_system_health','QuantumMetricsHarvester',
    'initialize_globals','shutdown_globals',
    'get_qrng_ensemble','initialize_qrng_ensemble','refresh_qrng_stats',
    'get_hlwe_system','initialize_hlwe_engine',
    '_GLOBAL_STATE','_STATE_LOCK',
]

logger.info("[GLOBAL] ✓ Integrated globals module loaded (QRNG + HLWE enabled)")
