#!/usr/bin/env python3
"""
LIGHTWEIGHT INDEPENDENT HEARTBEAT SYSTEM
Separate from lattice refresh cycle - runs on its own timer.

This heartbeat:
- POSTs to /api/keepalive every 60 seconds
- Non-blocking daemon thread
- Independent of W-state refresh timing
- Minimal overhead (~10ms per ping)
- Automatic retry on failure
"""

import requests
import threading
import time
import logging
from datetime import datetime
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class LightweightHeartbeat:
    """
    Independent keepalive heartbeat for HTTP endpoints.
    Runs on separate thread, no interference with main system.
    """
    
    def __init__(self, 
                 endpoint: str = "http://qtcl-blockchain.koyeb.app/api/keepalive",
                 interval_seconds: int = 60,
                 timeout_seconds: int = 5):
        """
        Initialize lightweight heartbeat.
        
        Args:
            endpoint: URL to POST keepalive to
            interval_seconds: How often to ping (default 60s)
            timeout_seconds: HTTP request timeout
        """
        self.endpoint = endpoint
        self.interval = interval_seconds
        self.timeout = timeout_seconds
        
        self.running = False
        self.thread = None
        self.lock = threading.RLock()
        
        # Metrics
        self.ping_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.last_ping_time = None
        self.last_success_time = None
    
    def start(self) -> None:
        """Start the heartbeat daemon thread"""
        with self.lock:
            if self.running:
                logger.warning("Heartbeat already running")
                return
            
            self.running = True
        
        self.thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="LightweightHeartbeat"
        )
        self.thread.start()
        logger.info(f"✓ Lightweight heartbeat started ({self.interval}s interval)")
    
    def stop(self) -> None:
        """Stop the heartbeat thread"""
        with self.lock:
            self.running = False
        
        if self.thread:
            self.thread.join(timeout=5)
        
        logger.info("✓ Heartbeat stopped")
    
    def _run_loop(self) -> None:
        """Main heartbeat loop (runs in daemon thread)"""
        logger.debug(f"Heartbeat loop started (endpoint: {self.endpoint})")
        
        while self.running:
            try:
                # Wait for interval
                time.sleep(self.interval)
                
                if not self.running:
                    break
                
                # Send ping
                self._send_ping()
            
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
                time.sleep(5)  # Brief backoff on error
    
    def _send_ping(self) -> None:
        """Send single keepalive ping"""
        try:
            with self.lock:
                self.ping_count += 1
                self.last_ping_time = datetime.now()
            
            payload = {
                'source': 'quantum_lattice_keepalive',
                'ping': self.ping_count,
                'timestamp': datetime.now().isoformat()
            }
            
            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout,
                headers={
                    'User-Agent': 'QuantumLatticeHeartbeat/1.0',
                    'Content-Type': 'application/json'
                }
            )
            
            with self.lock:
                if response.status_code == 200:
                    self.success_count += 1
                    self.last_success_time = datetime.now()
                    logger.debug(f"💓 Keepalive ping #{self.ping_count} OK")
                else:
                    self.failure_count += 1
                    logger.debug(f"⚠ Keepalive ping #{self.ping_count} HTTP {response.status_code}")
        
        except requests.exceptions.Timeout:
            with self.lock:
                self.failure_count += 1
            logger.debug(f"⏱ Keepalive ping timeout")
        
        except requests.exceptions.ConnectionError:
            with self.lock:
                self.failure_count += 1
            logger.debug(f"🔌 Keepalive connection error")
        
        except Exception as e:
            with self.lock:
                self.failure_count += 1
            logger.warning(f"❌ Keepalive ping error: {type(e).__name__}")
    
    def get_status(self) -> Dict:
        """Get heartbeat status metrics"""
        with self.lock:
            uptime = None
            if self.last_ping_time:
                uptime = (datetime.now() - self.last_ping_time).total_seconds()
            
            return {
                'running': self.running,
                'endpoint': self.endpoint,
                'interval_seconds': self.interval,
                'ping_count': self.ping_count,
                'success_count': self.success_count,
                'failure_count': self.failure_count,
                'success_rate': (
                    self.success_count / max(self.ping_count, 1)
                ),
                'last_ping_time': self.last_ping_time.isoformat() if self.last_ping_time else None,
                'last_success_time': self.last_success_time.isoformat() if self.last_success_time else None,
                'seconds_since_last_ping': uptime
            }


# Global instance (optional singleton pattern)
_global_heartbeat: Optional[LightweightHeartbeat] = None

def get_global_heartbeat() -> LightweightHeartbeat:
    """Get or create global heartbeat instance"""
    global _global_heartbeat
    if _global_heartbeat is None:
        _global_heartbeat = LightweightHeartbeat()
    return _global_heartbeat


if __name__ == '__main__':
    # Test the heartbeat
    logging.basicConfig(level=logging.DEBUG)
    
    hb = LightweightHeartbeat(
        endpoint="http://localhost:5000/api/keepalive",
        interval_seconds=5
    )
    hb.start()
    
    try:
        for i in range(10):
            time.sleep(2)
            status = hb.get_status()
            print(f"Status: {status['ping_count']} pings, {status['success_count']} OK")
    finally:
        hb.stop()
