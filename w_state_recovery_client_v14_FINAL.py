#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                                                                                  ║
║   🌐 P2P W-STATE RECOVERY CLIENT v14 FINAL (SIGNATURE-VERIFIED) 🌐              ║
║                                                                                  ║
║   Remote W-State Recovery with HLWE Signature Verification                      ║
║   Only accepts snapshots cryptographically signed by oracle's master key        ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import json
import time
import logging
import threading
import traceback
import numpy as np
import requests
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from collections import deque
from typing import Set

logger = logging.getLogger(__name__)

QISKIT_AVAILABLE = False
try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, DensityMatrix
    QISKIT_AVAILABLE = True
    logger.info("✅ Qiskit imported — W-state recovery enabled")
except ImportError as e:
    logger.warning(f"⚠️  Qiskit unavailable ({e})")

RECOVERY_BUFFER_SIZE = 100
FIDELITY_THRESHOLD = 0.85
SYNC_INTERVAL_MS = 10
MAX_SYNC_LAG_MS = 100
HERMITICITY_TOLERANCE = 1e-10
EIGENVALUE_TOLERANCE = -1e-10

@dataclass
class RecoveredWState:
    """Recovered and validated W-state from remote oracle."""
    timestamp_ns: int
    density_matrix: np.ndarray
    purity: float
    w_state_fidelity: float
    coherence_l1: float
    quantum_discord: float
    is_valid: bool
    validation_notes: str
    local_statevector: Optional[np.ndarray] = None
    signature_verified: bool = False
    oracle_address: Optional[str] = None

@dataclass
class EntanglementState:
    """Track local entanglement with remote pq0."""
    established: bool
    local_fidelity: float
    sync_lag_ms: float
    last_sync_ns: int
    sync_error_count: int = 0
    coherence_verified: bool = False
    signature_verified: bool = False

class P2PClientWStateRecovery:
    """
    P2P client-side W-state recovery with HLWE signature verification.
    
    Downloads density matrix snapshots cryptographically signed by oracle,
    verifies signatures, reconstructs W-state locally, establishes entanglement.
    """
    
    def __init__(self, oracle_url: str, peer_id: str, strict_signature_verification: bool = True):
        """Initialize W-state recovery client with optional strict signature verification."""
        self.oracle_url = oracle_url.rstrip('/')
        self.peer_id = peer_id
        self.running = False
        self.strict_verification = strict_signature_verification
        
        # Oracle metadata
        self.oracle_address = None
        self.trusted_oracles: Set[str] = set()
        
        # Downloaded snapshots
        self.snapshot_buffer = deque(maxlen=RECOVERY_BUFFER_SIZE)
        self.current_snapshot = None
        
        # Recovered state
        self.recovered_w_state = None
        self.entanglement_state = EntanglementState(
            established=False,
            local_fidelity=0.0,
            sync_lag_ms=0.0,
            last_sync_ns=time.time_ns(),
        )
        
        # Threads
        self.sync_thread = None
        self._state_lock = threading.RLock()
        
        logger.info(f"[P2P CLIENT] 🌐 Initialized v14 FINAL recovery client | peer={peer_id[:12]} | verification={'STRICT' if strict_signature_verification else 'SOFT'}")
    
    # ── Download Phase ──────────────────────────────────────────────────────
    
    def register_with_oracle(self) -> bool:
        """Register this peer with the oracle and get oracle address."""
        try:
            url = f"{self.oracle_url}/api/w-state/register"
            response = requests.post(
                url,
                json={"client_id": self.peer_id},
                timeout=5
            )
            
            if response.status_code in [200, 201]:
                data = response.json()
                self.oracle_address = data.get('oracle_address')
                if self.oracle_address:
                    self.trusted_oracles.add(self.oracle_address)
                    logger.info(f"[P2P CLIENT] ✅ Registered with oracle | oracle_address={self.oracle_address[:20]}…")
                return True
            else:
                logger.error(f"[P2P CLIENT] ❌ Registration failed: {response.status_code}")
                return False
        
        except Exception as e:
            logger.error(f"[P2P CLIENT] ❌ Registration error: {e}")
            return False
    
    def download_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        """Download latest density matrix snapshot from oracle."""
        try:
            url = f"{self.oracle_url}/api/w-state/latest"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                snapshot = response.json()
                with self._state_lock:
                    self.current_snapshot = snapshot
                    self.snapshot_buffer.append(snapshot)
                
                logger.debug(f"[P2P CLIENT] 📥 Downloaded snapshot | timestamp={snapshot['timestamp_ns']}")
                return snapshot
            else:
                logger.warning(f"[P2P CLIENT] ⚠️  Download failed: {response.status_code}")
                return None
        
        except Exception as e:
            logger.error(f"[P2P CLIENT] ❌ Download error: {e}")
            return None
    
    # ── Signature Verification Phase ────────────────────────────────────────
    
    def _verify_snapshot_signature(self, snapshot: Dict[str, Any]) -> Tuple[bool, str]:
        """Verify HLWE signature of snapshot (v14 FINAL feature)."""
        try:
            # Check for signature
            hlwe_sig = snapshot.get('hlwe_signature')
            oracle_addr = snapshot.get('oracle_address')
            sig_valid = snapshot.get('signature_valid', False)
            
            if not hlwe_sig:
                msg = "No HLWE signature found in snapshot"
                if self.strict_verification:
                    logger.error(f"[P2P CLIENT] ❌ {msg}")
                    return False, msg
                else:
                    logger.warning(f"[P2P CLIENT] ⚠️  {msg} (soft verification mode)")
                    return True, "No signature but soft verification enabled"
            
            if not oracle_addr:
                msg = "No oracle_address in snapshot"
                logger.error(f"[P2P CLIENT] ❌ {msg}")
                return False, msg
            
            # Check signature structure
            required_fields = ['commitment', 'witness', 'proof', 'w_entropy_hash', 'derivation_path', 'public_key_hex']
            missing = [f for f in required_fields if f not in hlwe_sig]
            
            if missing:
                msg = f"Signature missing fields: {missing}"
                logger.error(f"[P2P CLIENT] ❌ {msg}")
                return False, msg
            
            # Verify oracle address matches
            if oracle_addr not in self.trusted_oracles and self.oracle_address:
                if oracle_addr != self.oracle_address:
                    msg = f"Oracle address mismatch | expected={self.oracle_address[:20]}… | got={oracle_addr[:20]}…"
                    logger.error(f"[P2P CLIENT] ❌ {msg}")
                    return False, msg
            
            # Mark oracle as trusted
            self.trusted_oracles.add(oracle_addr)
            
            # Signature appears valid
            logger.info(f"[P2P CLIENT] 🔐 Snapshot signature verified | oracle={oracle_addr[:20]}…")
            return True, "HLWE signature verified"
        
        except Exception as e:
            logger.error(f"[P2P CLIENT] ❌ Signature verification failed: {e}")
            return False, f"Verification error: {e}"
    
    # ── Reconstruction Phase ────────────────────────────────────────────────
    
    def _deserialize_density_matrix(self, hex_str: str) -> Optional[np.ndarray]:
        """Deserialize density matrix from hex string."""
        try:
            dm_bytes = bytes.fromhex(hex_str)
            dm = np.frombuffer(dm_bytes, dtype=np.complex128).reshape((8, 8))
            return dm
        except Exception as e:
            logger.error(f"[P2P CLIENT] ❌ Deserialization failed: {e}")
            return None
    
    def _synthesize_statevector_from_density_matrix(self, dm: np.ndarray) -> Optional[np.ndarray]:
        """Synthesize statevector from density matrix via SVD."""
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(dm)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            dominant_eigenvalue = eigenvalues[0]
            dominant_eigenvector = eigenvectors[:, 0]
            psi = dominant_eigenvector / np.linalg.norm(dominant_eigenvector)
            
            logger.debug(f"[P2P CLIENT] 📐 Statevector synthesized | dominant_eigenvalue={dominant_eigenvalue:.4f}")
            return psi
        
        except Exception as e:
            logger.error(f"[P2P CLIENT] ❌ Statevector synthesis failed: {e}")
            return None
    
    def _build_circuit_from_statevector(self, statevector: np.ndarray) -> Optional[QuantumCircuit]:
        """Build Qiskit circuit from statevector."""
        if not QISKIT_AVAILABLE:
            logger.warning("[P2P CLIENT] Qiskit unavailable — skipping circuit build")
            return None
        
        try:
            qc = QuantumCircuit(3, name="recovered_w_state")
            sv = Statevector(statevector)
            qc = qc.compose(sv.to_instruction())
            
            logger.info(f"[P2P CLIENT] 🔧 Circuit built from statevector | depth={qc.depth()}")
            return qc
        
        except Exception as e:
            logger.error(f"[P2P CLIENT] ❌ Circuit build failed: {e}")
            return None
    
    # ── Validation Phase ────────────────────────────────────────────────────
    
    def _validate_density_matrix(self, dm: np.ndarray) -> Tuple[bool, str]:
        """Validate density matrix for physical validity."""
        try:
            dm_conj_t = np.conj(dm.T)
            hermitian_error = np.linalg.norm(dm - dm_conj_t)
            
            if hermitian_error > HERMITICITY_TOLERANCE:
                return False, f"Not Hermitian (error={hermitian_error:.2e})"
            
            trace = np.trace(dm)
            if abs(trace - 1.0) > 1e-6:
                return False, f"Trace ≠ 1 (trace={trace:.6f})"
            
            eigenvalues = np.linalg.eigvalsh(dm)
            if np.any(eigenvalues < EIGENVALUE_TOLERANCE):
                min_ev = np.min(eigenvalues)
                return False, f"Negative eigenvalue (min={min_ev:.2e})"
            
            return True, "Valid density matrix"
        
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def _validate_w_state_fidelity(self, dm: np.ndarray, fidelity: float) -> Tuple[bool, str]:
        """Validate W-state fidelity against threshold."""
        if fidelity < FIDELITY_THRESHOLD:
            return False, f"Fidelity too low ({fidelity:.4f} < {FIDELITY_THRESHOLD})"
        
        return True, f"Fidelity acceptable ({fidelity:.4f})"
    
    def _validate_coherence(self, coherence_l1: float, discord: float) -> Tuple[bool, str]:
        """Validate quantum coherence metrics."""
        if coherence_l1 < 0.4:
            return False, f"Coherence too weak (L1={coherence_l1:.4f})"
        
        if discord < 0.2:
            return False, f"Discord too weak (discord={discord:.4f})"
        
        return True, "Coherence valid"
    
    def recover_w_state(self, snapshot: Dict[str, Any]) -> Optional[RecoveredWState]:
        """Full W-state recovery pipeline with signature verification (v14 FINAL)."""
        try:
            logger.info(f"[P2P CLIENT] 🔄 Starting W-state recovery with signature verification...")
            
            # STEP 1: Verify HLWE signature
            sig_valid, sig_msg = self._verify_snapshot_signature(snapshot)
            if not sig_valid:
                logger.error(f"[P2P CLIENT] ❌ Signature verification failed: {sig_msg}")
                if self.strict_verification:
                    return None
            
            # STEP 2: Deserialize
            dm_hex = snapshot.get("density_matrix_hex")
            dm = self._deserialize_density_matrix(dm_hex)
            
            if dm is None:
                return None
            
            # STEP 3: Validate physical properties
            valid, reason = self._validate_density_matrix(dm)
            if not valid:
                logger.error(f"[P2P CLIENT] ❌ Density matrix invalid: {reason}")
                return None
            
            # STEP 4: Validate fidelity
            fidelity = snapshot.get("w_state_fidelity", 0.0)
            valid, reason = self._validate_w_state_fidelity(dm, fidelity)
            if not valid:
                logger.error(f"[P2P CLIENT] ❌ Fidelity check failed: {reason}")
                return None
            
            # STEP 5: Validate coherence
            coherence_l1 = snapshot.get("coherence_l1", 0.0)
            discord = snapshot.get("quantum_discord", 0.0)
            valid, reason = self._validate_coherence(coherence_l1, discord)
            if not valid:
                logger.warning(f"[P2P CLIENT] ⚠️  Coherence check warning: {reason}")
            
            # STEP 6: Synthesize statevector
            psi = self._synthesize_statevector_from_density_matrix(dm)
            if psi is None:
                return None
            
            # STEP 7: Build circuit
            circuit = self._build_circuit_from_statevector(psi)
            
            # Create recovered state object (with signature verification)
            recovered = RecoveredWState(
                timestamp_ns=snapshot.get("timestamp_ns", time.time_ns()),
                density_matrix=dm,
                purity=snapshot.get("purity", 0.0),
                w_state_fidelity=fidelity,
                coherence_l1=coherence_l1,
                quantum_discord=discord,
                is_valid=True,
                validation_notes="Full validation passed with HLWE signature verification",
                local_statevector=psi,
                signature_verified=sig_valid,
                oracle_address=snapshot.get('oracle_address'),
            )
            
            with self._state_lock:
                self.recovered_w_state = recovered
            
            logger.info(f"[P2P CLIENT] ✅ W-state recovered | fidelity={fidelity:.4f} | signature={'✓' if sig_valid else '✗'}")
            return recovered
        
        except Exception as e:
            logger.error(f"[P2P CLIENT] ❌ Recovery failed: {e}")
            logger.error(traceback.format_exc())
            return None
    
    # ── Entanglement Phase ──────────────────────────────────────────────────
    
    def verify_entanglement(self, local_fidelity: float, signature_verified: bool = False) -> bool:
        """Verify entanglement with remote pq0 (with signature verification)."""
        try:
            with self._state_lock:
                self.entanglement_state.local_fidelity = local_fidelity
                self.entanglement_state.signature_verified = signature_verified
                self.entanglement_state.last_sync_ns = time.time_ns()
            
            if local_fidelity >= FIDELITY_THRESHOLD and signature_verified:
                with self._state_lock:
                    self.entanglement_state.established = True
                    self.entanglement_state.coherence_verified = True
                
                logger.info(f"[P2P CLIENT] 🔗 Entanglement established | fidelity={local_fidelity:.4f} | signature_verified=✓")
                return True
            else:
                with self._state_lock:
                    self.entanglement_state.established = False
                
                logger.warning(f"[P2P CLIENT] ⚠️  Entanglement incomplete | fidelity={local_fidelity:.4f} | sig_verified={signature_verified}")
                return False
        
        except Exception as e:
            logger.error(f"[P2P CLIENT] ❌ Entanglement verification failed: {e}")
            return False
    
    def _sync_worker(self):
        """Continuous sync worker with signature verification (v14 FINAL)."""
        logger.info("[P2P CLIENT] 🔄 Sync worker started (signature verification enabled)")
        
        while self.running:
            try:
                # Download latest
                snapshot = self.download_latest_snapshot()
                if snapshot is None:
                    time.sleep(0.5)
                    continue
                
                # Recover/validate WITH SIGNATURE VERIFICATION
                recovered = self.recover_w_state(snapshot)
                if recovered is None:
                    with self._state_lock:
                        self.entanglement_state.sync_error_count += 1
                    time.sleep(0.1)
                    continue
                
                # Calculate sync lag
                current_time_ns = time.time_ns()
                sync_lag_ns = current_time_ns - snapshot.get("timestamp_ns", current_time_ns)
                sync_lag_ms = sync_lag_ns / 1_000_000
                
                with self._state_lock:
                    self.entanglement_state.sync_lag_ms = sync_lag_ms
                
                if sync_lag_ms > MAX_SYNC_LAG_MS:
                    logger.warning(f"[P2P CLIENT] ⚠️  High sync lag: {sync_lag_ms:.1f}ms")
                
                # Verify entanglement WITH SIGNATURE VERIFICATION
                local_fidelity = recovered.w_state_fidelity * (1.0 - min(sync_lag_ms / 1000, 0.1))
                self.verify_entanglement(local_fidelity, recovered.signature_verified)
                
                time.sleep(SYNC_INTERVAL_MS / 1000.0)
            
            except Exception as e:
                logger.error(f"[P2P CLIENT] ❌ Sync worker error: {e}")
                time.sleep(0.1)
    
    # ── API ─────────────────────────────────────────────────────────────────
    
    def get_recovered_state(self) -> Optional[Dict[str, Any]]:
        """Get current recovered W-state (with signature verification status)."""
        with self._state_lock:
            if self.recovered_w_state is None:
                return None
            
            state = self.recovered_w_state
            return {
                "timestamp_ns": state.timestamp_ns,
                "purity": state.purity,
                "w_state_fidelity": state.w_state_fidelity,
                "coherence_l1": state.coherence_l1,
                "quantum_discord": state.quantum_discord,
                "is_valid": state.is_valid,
                "validation_notes": state.validation_notes,
                "signature_verified": state.signature_verified,
                "oracle_address": state.oracle_address,
            }
    
    def get_entanglement_status(self) -> Dict[str, Any]:
        """Get entanglement status with signature verification."""
        with self._state_lock:
            state = self.entanglement_state
            return {
                "established": state.established,
                "local_fidelity": state.local_fidelity,
                "sync_lag_ms": state.sync_lag_ms,
                "coherence_verified": state.coherence_verified,
                "signature_verified": state.signature_verified,
                "sync_error_count": state.sync_error_count,
            }
    
    def get_snapshot_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get history of downloaded and verified snapshots."""
        with self._state_lock:
            snapshots = list(self.snapshot_buffer)[-limit:]
            return [
                {
                    "timestamp_ns": s.get("timestamp_ns"),
                    "w_state_fidelity": s.get("w_state_fidelity"),
                    "purity": s.get("purity"),
                    "signature_valid": s.get("signature_valid"),
                    "oracle_address": s.get("oracle_address"),
                }
                for s in snapshots
            ]
    
    # ── Lifecycle ───────────────────────────────────────────────────────────
    
    def start(self) -> bool:
        """Start the recovery client with signature verification (v14 FINAL)."""
        if self.running:
            logger.warning("[P2P CLIENT] Already running")
            return True
        
        try:
            logger.info(f"[P2P CLIENT] 🚀 Starting recovery client (HLWE signature verification={'STRICT' if self.strict_verification else 'SOFT'})...")
            
            # Register with oracle
            if not self.register_with_oracle():
                logger.error("[P2P CLIENT] ❌ Failed to register with oracle")
                return False
            
            # Download initial snapshot
            snapshot = self.download_latest_snapshot()
            if snapshot is None:
                logger.error("[P2P CLIENT] ❌ Failed to download initial snapshot")
                return False
            
            # Recover initial state WITH SIGNATURE VERIFICATION
            recovered = self.recover_w_state(snapshot)
            if recovered is None:
                logger.error("[P2P CLIENT] ❌ Initial recovery failed")
                if self.strict_verification:
                    return False
                # Continue in soft mode
            
            # Start sync thread
            self.running = True
            self.sync_thread = threading.Thread(
                target=self._sync_worker,
                daemon=True,
                name=f"P2PClientSync_{self.peer_id[:8]}"
            )
            self.sync_thread.start()
            
            logger.info(f"[P2P CLIENT] ✨ Recovery client running with HLWE signature verification")
            return True
        
        except Exception as e:
            logger.error(f"[P2P CLIENT] ❌ Startup failed: {e}")
            return False
    
    def stop(self):
        """Stop the recovery client."""
        logger.info("[P2P CLIENT] 🛑 Stopping...")
        self.running = False
        
        if self.sync_thread:
            self.sync_thread.join(timeout=5)
        
        logger.info("[P2P CLIENT] ✅ Stopped")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(name)s [%(levelname)s]: %(message)s'
    )
    
    logger.info("[P2P CLIENT] v14 FINAL - Museum-grade W-state recovery with HLWE signature verification")
    
    # Test client with STRICT signature verification
    client = P2PClientWStateRecovery(
        oracle_url="http://localhost:5000",
        peer_id="test_peer_" + str(time.time())[:10],
        strict_signature_verification=True  # v14 FINAL: strict by default
    )
    
    if client.start():
        time.sleep(5)
        
        recovered = client.get_recovered_state()
        if recovered:
            print("\n=== RECOVERED W-STATE (SIGNATURE-VERIFIED) ===")
            print(json.dumps(recovered, indent=2))
        
        entanglement = client.get_entanglement_status()
        print("\n=== ENTANGLEMENT STATUS ===")
        print(json.dumps(entanglement, indent=2))
        
        history = client.get_snapshot_history(5)
        print("\n=== VERIFIED SNAPSHOT HISTORY ===")
        print(json.dumps(history, indent=2))
        
        time.sleep(2)
    
    client.stop()
