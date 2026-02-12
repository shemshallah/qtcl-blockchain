#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
QUANTUM LATTICE HEARTBEAT - NOISE REFRESH INTEGRATION
═══════════════════════════════════════════════════════════════════════════════

ARCHITECTURE:
The heartbeat IS the noise refresh cycle. When the NonMarkovianNoiseBath 
refreshes entropy (every 52 batches/cycle), it sends an HTTP heartbeat signal.

This is elegant because:
✓ Heartbeat happens alongside actual work (entropy refresh)
✓ No separate threading/timing logic needed
✓ Integrated into the quantum processing pipeline
✓ Heartbeat rate naturally aligns with cycle completion

INTEGRATION POINT:
NonMarkovianNoiseBath._get_quantum_noise() → entropy.fetch_quantum_bytes()
↓
Calls optional heartbeat callback on cycle boundary

═══════════════════════════════════════════════════════════════════════════════
"""

import requests
import logging
from typing import Optional, Callable, Dict, Any

logger = logging.getLogger(__name__)


class NoiseRefreshHeartbeat:
    """
    Sends heartbeat signal as part of noise bath entropy refresh.
    
    This callback is invoked when the noise bath refreshes entropy,
    naturally creating keep-alive signals without separate scheduling.
    """
    
    def __init__(self, app_url: Optional[str] = None):
        """
        Args:
            app_url: Flask app base URL (e.g., 'http://localhost:5000')
        """
        import os
        self.app_url = app_url or os.getenv('APP_URL', 'http://localhost:5000')
        self.heartbeat_endpoint = f"{self.app_url}/api/keep-alive"
        self.last_beat_cycle = -1
    
    def on_noise_cycle_complete(self, cycle_number: int, metrics: Dict[str, Any]) -> None:
        """
        Called by noise bath when a complete cycle completes.
        
        Args:
            cycle_number: Current cycle number
            metrics: Cycle metrics dict with coherence, fidelity, sigma, etc.
        """
        # Only send heartbeat once per cycle (on first batch of next cycle)
        # to avoid 52 requests per cycle
        if cycle_number > self.last_beat_cycle:
            self._send_beat(cycle_number, metrics)
            self.last_beat_cycle = cycle_number
    
    def _send_beat(self, cycle_number: int, metrics: Dict[str, Any]) -> None:
        """Send heartbeat asynchronously (non-blocking)"""
        import threading
        
        def _send():
            try:
                payload = {
                    'cycle': cycle_number,
                    'timestamp': __import__('datetime').datetime.now().isoformat(),
                    'metrics': {
                        'sigma': metrics.get('sigma'),
                        'coherence': metrics.get('coherence_after'),
                        'fidelity': metrics.get('fidelity_after', 1.0)
                    }
                }
                
                response = requests.post(
                    self.heartbeat_endpoint,
                    json=payload,
                    timeout=3,
                    headers={'User-Agent': 'QuantumLatticeHeartbeat/1.0'}
                )
                
                if response.status_code == 200:
                    logger.debug(f"♥ [Cycle {cycle_number}] Heartbeat sent")
                else:
                    logger.warning(f"⚠ [Cycle {cycle_number}] HTTP {response.status_code}")
            
            except requests.exceptions.Timeout:
                logger.debug(f"⏱ [Cycle {cycle_number}] Heartbeat timeout (non-critical)")
            except requests.exceptions.RequestException as e:
                logger.debug(f"⚠ [Cycle {cycle_number}] Heartbeat failed: {type(e).__name__}")
            except Exception as e:
                logger.debug(f"❌ [Cycle {cycle_number}] Heartbeat error: {e}")
        
        # Send in background thread (non-blocking)
        thread = threading.Thread(target=_send, daemon=True)
        thread.start()



# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION COMPLETE
# ═══════════════════════════════════════════════════════════════════════════════
# 
# This module is ready to use. Integration instructions have been followed in:
# - quantum_lattice_control_live_complete.py
# 
# The heartbeat system will activate automatically when quantum cycles complete.
#
