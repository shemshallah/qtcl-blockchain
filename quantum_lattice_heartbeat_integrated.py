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
# INTEGRATION INSTRUCTIONS FOR quantum_lattice_control_live_complete.py
# ═══════════════════════════════════════════════════════════════════════════════

"""
STEP 1: Add import at top of quantum_lattice_control_live_complete.py
─────────────────────────────────────────────────────────────────────

from quantum_lattice_heartbeat import NoiseRefreshHeartbeat

STEP 2: In QuantumLatticeControlSystem.__init__() (around line 1470)
─────────────────────────────────────────────────────────────────────────

    def __init__(self):
        # ... existing code ...
        
        self.noise_bath = NonMarkovianNoiseBath(self.entropy_ensemble)
        
        # ADD THIS: Initialize heartbeat (ADDED)
        self.heartbeat = NoiseRefreshHeartbeat()
        self.noise_bath.set_heartbeat_callback(self.on_cycle_complete)
        
        # ... rest of existing code ...
    
    def on_cycle_complete(self, cycle_num: int, metrics: Dict) -> None:
        """Callback when cycle completes - sends heartbeat"""
        self.heartbeat.on_noise_cycle_complete(cycle_num, metrics)

STEP 3: Modify NonMarkovianNoiseBath class (in quantum_lattice_control_live_complete.py)
──────────────────────────────────────────────────────────────────────────────────────────

In the __init__ method (around line 422), add:
    
    def __init__(self, entropy_ensemble: QuantumEntropyEnsemble):
        # ... existing code ...
        self.lock = threading.RLock()
        
        # ADD THIS: Heartbeat callback (ADDED)
        self.heartbeat_callback: Optional[Callable] = None
        
        logger.info(f"Non-Markovian Noise Bath initialized...")

Then add this method to the NonMarkovianNoiseBath class:

    def set_heartbeat_callback(self, callback: Optional[Callable]) -> None:
        \"\"\"Set callback to invoke when cycle completes\"\"\"
        self.heartbeat_callback = callback

Then modify execute_cycle() in QuantumLatticeControlSystem (line 1540):
────────────────────────────────────────────────────────────────────────

    def execute_cycle(self) -> Dict:
        \"\"\"Execute complete system cycle (all 52 batches)\"\"\"
        if not self.running:
            logger.error("System not running")
            return {}
        
        with self.lock:
            self.cycle_count += 1
            cycle_start = time.time()
        
        logger.info(f"\\n[Cycle {self.cycle_count}] Starting {self.noise_bath.NUM_BATCHES} batches...")
        
        batch_results = []
        
        for batch_id in range(self.noise_bath.NUM_BATCHES):
            result = self.batch_pipeline.execute(batch_id, self.entropy_ensemble)
            batch_results.append(result)
            
            self.analytics.record_batch(batch_id, result)
            
            with self.lock:
                self.total_batches_executed += 1
            
            if (batch_id + 1) % 13 == 0:
                logger.debug(f"  Progress: {batch_id + 1}/{self.noise_bath.NUM_BATCHES}")
        
        cycle_time = time.time() - cycle_start
        with self.lock:
            self.total_time_compute += cycle_time
        
        avg_sigma = np.mean([r['sigma'] for r in batch_results])
        avg_coh = np.mean([r['coherence_after'] for r in batch_results])
        avg_fid = np.mean([r['fidelity_after'] for r in batch_results])
        avg_loss = np.mean([r['neural_loss'] for r in batch_results])
        avg_change = np.mean([r['net_change'] for r in batch_results])
        
        self.analytics.record_cycle(
            avg_coh, avg_fid, avg_sigma, avg_loss, avg_change, cycle_time
        )
        
        anomalies = self.analytics.detect_anomalies()
        
        if self.cycle_count % 10 == 0:
            self.checkpoint_mgr.save(
                self.cycle_count,
                self.sigma_controller,
                self.get_status()
            )
        
        logger.info(
            f"[Cycle {self.cycle_count}] ✓ Complete ({cycle_time:.1f}s) | "
            f"σ={avg_sigma:.2f} | C={avg_coh:.6f} | F={avg_fid:.6f} | "
            f"ΔC={avg_change:+.6f} | L={avg_loss:.6f} | "
            f"A={len(anomalies)}"
        )
        
        # SEND HEARTBEAT (ADDED)
        cycle_metrics = {
            'sigma': float(avg_sigma),
            'coherence_after': float(avg_coh),
            'fidelity_after': float(avg_fid),
            'cycle_time': cycle_time
        }
        self.on_cycle_complete(self.cycle_count, cycle_metrics)
        
        return {
            'cycle': self.cycle_count,
            'duration': cycle_time,
            'batches_completed': len(batch_results),
            'avg_sigma': avg_sigma,
            'avg_coherence': avg_coh,
            'avg_fidelity': avg_fid,
            'avg_loss': avg_loss,
            'avg_net_change': avg_change,
            'anomalies': anomalies,
            'throughput_batches_per_sec': len(batch_results) / cycle_time
        }

STEP 4: Update main_app.py - Add keep-alive endpoint
──────────────────────────────────────────────────────

In initialize_app() function, add:

    @flask_app.route('/api/keep-alive', methods=['GET', 'POST'])
    def keep_alive_endpoint():
        \"\"\"Keep-alive endpoint for quantum lattice heartbeat\"\"\"
        from datetime import datetime
        return jsonify({
            'status': 'alive',
            'timestamp': datetime.now().isoformat(),
            'source': 'quantum_lattice_noise_refresh',
            'message': 'Instance is awake and responsive'
        }), 200

═══════════════════════════════════════════════════════════════════════════════
RESULT
═══════════════════════════════════════════════════════════════════════════════

Each quantum lattice cycle (~90-100 seconds):
  1. Noise bath refreshes entropy
  2. 52 batches execute
  3. Cycle completes
  4. Heartbeat sent to /api/keep-alive
  5. Koyeb inactivity timer resets
  6. Instance stays awake

This is elegant because the heartbeat IS the noise refresh cycle itself.
No separate scheduling, threading, or timing logic needed.
Pure integration of keep-alive into the quantum processing pipeline.

═══════════════════════════════════════════════════════════════════════════════
"""
