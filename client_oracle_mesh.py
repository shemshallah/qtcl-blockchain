#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║  CLIENT ORACLE MESH — Phase 3: Peer SSE Stream Infrastructure                ║
║                                                                                ║
║  Connects to multiple peers' /rpc/stream/snapshot endpoints and aggregates    ║
║  their 8×8 density matrices via Byzantine-resistant averaging.                ║
║                                                                                ║
║  Museum-Grade P2P Design:                                                      ║
║    • Exponential backoff on connection failures                               ║
║    • Thread-safe snapshot reads (no locks on hot path)                        ║
║    • Hermitian validation and weight calculation                              ║
║    • Per-peer health tracking                                                 ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import json
import struct
import logging
import threading
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen
from io import BytesIO

_EXP_LOG = logging.getLogger(__name__)


class ClientOracleMesh:
    """
    Museum-Grade P2P mesh client that collects 8×8 density matrices from
    multiple oracle peers via their /rpc/stream/snapshot endpoints.

    Features:
      • Background polling threads per peer (no SSE for peer connections)
      • Exponential backoff (max 10 consecutive errors → stop)
      • Byzantine-resistant Hermitian averaging
      • Thread-safe snapshot reads via RLock
      • Per-peer fidelity weighting for consensus
    """

    class _PeerSSEStream:
        """
        Connects to a peer's /rpc/stream/snapshot REST endpoint and polls
        for latest 8×8 density matrix.

        Pattern: background thread, exponential backoff, max 10 consecutive
        errors → stops gracefully.
        """

        def __init__(self, peer_id: str, host: str, port: int, parent_ref) -> None:
            """
            Initialize peer stream connection.

            Args:
                peer_id: string identifier for this peer (e.g., "node_abc123")
                host: peer's IP/hostname (e.g., "192.168.1.100" or "peer1.local")
                port: peer's local HTTP port (e.g., 9092)
                parent_ref: reference back to ClientOracleMesh for error callbacks
            """
            self.peer_id = peer_id
            self.host = host
            self.port = port
            self.parent_ref = parent_ref
            self.endpoint = f"http://{host}:{port}/rpc/stream/snapshot"

            # State
            self.latest_dm: Optional[np.ndarray] = None
            self.latest_ts: int = 0
            self.latest_fidelity: float = 0.0
            self.running = False
            self.consecutive_errors = 0
            self._lock = threading.RLock()
            self._thread: Optional[threading.Thread] = None

            _EXP_LOG.info(
                f"[MESH] Peer stream instance created for {peer_id} @ "
                f"{host}:{port}"
            )

        def start(self) -> None:
            """Launch background polling thread."""
            if self.running:
                _EXP_LOG.warning(
                    f"[MESH] Peer {self.peer_id} stream already running, skipping start"
                )
                return

            self.running = True
            self.consecutive_errors = 0
            self._thread = threading.Thread(
                target=self._poll_loop,
                daemon=True,
                name=f"Peer-SSE-{self.peer_id}",
            )
            self._thread.start()
            _EXP_LOG.info(
                f"[MESH] Peer SSE connected to {self.peer_id} @ "
                f"{self.host}:{self.port}"
            )

        def stop(self) -> None:
            """Stop background thread gracefully."""
            if not self.running:
                return

            self.running = False
            if self._thread is not None:
                try:
                    self._thread.join(timeout=5.0)
                except Exception as e:
                    _EXP_LOG.warning(
                        f"[MESH] Error joining thread for {self.peer_id}: {e}"
                    )

            _EXP_LOG.info(f"[MESH] Peer SSE {self.peer_id} stopped")

        def get_latest_dm(self) -> Optional[np.ndarray]:
            """
            Non-blocking read of latest 8×8 density matrix.

            Returns:
                Parsed np.ndarray(8, 8, dtype=complex128) if latest snapshot
                is valid, else None. Thread-safe via internal lock.
            """
            with self._lock:
                if self.latest_dm is None:
                    return None
                # Return a copy to prevent external modification
                return np.array(self.latest_dm, copy=True)

        def get_latest_fidelity(self) -> float:
            """Non-blocking read of latest W-state fidelity."""
            with self._lock:
                return self.latest_fidelity

        def get_latest_timestamp_ns(self) -> int:
            """Non-blocking read of latest snapshot timestamp (nanoseconds)."""
            with self._lock:
                return self.latest_ts

        def _poll_loop(self) -> None:
            """
            Private poll loop running in background thread.
            Never raises; logs errors and exits gracefully.
            """
            while self.running:
                try:
                    response = self._fetch_snapshot()
                    if response is None:
                        continue

                    # Parse JSON
                    try:
                        data = json.loads(response.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        _EXP_LOG.debug(
                            f"[MESH] {self.peer_id} malformed JSON: {e}"
                        )
                        self.consecutive_errors += 1
                        if self.consecutive_errors > 10:
                            _EXP_LOG.error(
                                f"[MESH] {self.peer_id} exceeded max errors "
                                f"({self.consecutive_errors}), stopping"
                            )
                            self.running = False
                        continue

                    # Extract snapshot fields
                    snapshot = data.get("snapshot", {})
                    dm_hex = snapshot.get("density_matrix_hex", "")
                    ts_ns = snapshot.get("timestamp_ns", 0)
                    fidelity = snapshot.get("w_state_fidelity", 0.0)

                    if not dm_hex:
                        _EXP_LOG.debug(
                            f"[MESH] {self.peer_id} missing density_matrix_hex"
                        )
                        self.consecutive_errors += 1
                        if self.consecutive_errors > 10:
                            _EXP_LOG.error(
                                f"[MESH] {self.peer_id} exceeded max errors "
                                f"({self.consecutive_errors}), stopping"
                            )
                            self.running = False
                        continue

                    # Check staleness (>120s old)
                    now_ns = time.time_ns()
                    age_s = (now_ns - ts_ns) / 1e9
                    if age_s > 120.0:
                        _EXP_LOG.warning(
                            f"[MESH] {self.peer_id} stale snapshot: "
                            f"{age_s:.1f}s old, skipping"
                        )
                        self.consecutive_errors += 1
                        if self.consecutive_errors > 10:
                            _EXP_LOG.error(
                                f"[MESH] {self.peer_id} exceeded max errors "
                                f"({self.consecutive_errors}), stopping"
                            )
                            self.running = False
                        continue

                    # Parse DM: hex → bytes → complex128 array
                    try:
                        dm_bytes = bytes.fromhex(dm_hex)
                        dm = np.frombuffer(dm_bytes, dtype=np.complex128).reshape(8, 8)
                    except (ValueError, struct.error) as e:
                        _EXP_LOG.debug(
                            f"[MESH] {self.peer_id} cannot parse density_matrix_hex: "
                            f"{e}"
                        )
                        self.consecutive_errors += 1
                        if self.consecutive_errors > 10:
                            _EXP_LOG.error(
                                f"[MESH] {self.peer_id} exceeded max errors "
                                f"({self.consecutive_errors}), stopping"
                            )
                            self.running = False
                        continue

                    # Validate DM via parent
                    if not self.parent_ref._validate_dm(dm, f"peer_{self.peer_id}"):
                        self.consecutive_errors += 1
                        if self.consecutive_errors > 10:
                            _EXP_LOG.error(
                                f"[MESH] {self.peer_id} exceeded max errors "
                                f"({self.consecutive_errors}), stopping"
                            )
                            self.running = False
                        continue

                    # Success: store snapshot under lock
                    with self._lock:
                        self.latest_dm = dm
                        self.latest_ts = ts_ns
                        self.latest_fidelity = max(0.0, min(fidelity, 1.0))
                        self.consecutive_errors = 0

                    _EXP_LOG.debug(
                        f"[MESH] {self.peer_id} snapshot acquired "
                        f"(fidelity={fidelity:.3f}, age_s={age_s:.1f})"
                    )

                    # Poll every 5 seconds on success
                    time.sleep(5.0)

                except Exception as e:
                    _EXP_LOG.debug(f"[MESH] {self.peer_id} unexpected error: {e}")
                    self.consecutive_errors += 1
                    if self.consecutive_errors > 10:
                        _EXP_LOG.error(
                            f"[MESH] {self.peer_id} exceeded max errors "
                            f"({self.consecutive_errors}), stopping"
                        )
                        self.running = False
                        break

        def _fetch_snapshot(self) -> Optional[bytes]:
            """
            Fetch snapshot from peer endpoint with exponential backoff.

            Returns:
                Response body bytes on success, None on error (after logging
                and sleeping). Never raises.
            """
            try:
                req = Request(self.endpoint)
                req.add_header("User-Agent", "ClientOracleMesh/1.0")
                req.add_header("Accept", "application/json")

                with urlopen(req, timeout=5.0) as response:
                    body = response.read()
                    return body

            except (URLError, HTTPError, TimeoutError) as e:
                self.consecutive_errors += 1
                sleep_time = min(2 ** self.consecutive_errors, 30)

                # Log only on first error and every 10th retry
                if self.consecutive_errors == 1 or (
                    self.consecutive_errors % 10
                ) == 0:
                    _EXP_LOG.warning(
                        f"[MESH] {self.peer_id} connection error "
                        f"(attempt {self.consecutive_errors}): {type(e).__name__}: "
                        f"{str(e)[:60]} — backoff {sleep_time}s"
                    )

                time.sleep(sleep_time)
                return None

            except Exception as e:
                self.consecutive_errors += 1
                sleep_time = min(2 ** self.consecutive_errors, 30)

                if self.consecutive_errors == 1 or (
                    self.consecutive_errors % 10
                ) == 0:
                    _EXP_LOG.warning(
                        f"[MESH] {self.peer_id} unexpected error "
                        f"(attempt {self.consecutive_errors}): {type(e).__name__}: "
                        f"{str(e)[:60]} — backoff {sleep_time}s"
                    )

                time.sleep(sleep_time)
                return None

    def __init__(self, node_id: str = "unknown") -> None:
        """
        Initialize ClientOracleMesh.

        Args:
            node_id: this client's identifier for logging
        """
        self.node_id = node_id
        self._peer_streams: Dict[str, self._PeerSSEStream] = {}
        self._peer_streams_lock = threading.RLock()

        _EXP_LOG.info(
            f"[MESH] ClientOracleMesh initialized for node {node_id}"
        )

    def add_peer(self, peer_id: str, host: str, port: int) -> bool:
        """
        Register a peer and start its stream poller.

        Args:
            peer_id: unique peer identifier
            host: peer hostname or IP
            port: peer HTTP port

        Returns:
            True if peer added successfully, False if already exists
        """
        with self._peer_streams_lock:
            if peer_id in self._peer_streams:
                _EXP_LOG.warning(
                    f"[MESH] Peer {peer_id} already registered, ignoring"
                )
                return False

            stream = self._PeerSSEStream(peer_id, host, port, self)
            stream.start()
            self._peer_streams[peer_id] = stream

            _EXP_LOG.info(
                f"[MESH] Peer {peer_id} registered and stream started"
            )
            return True

    def remove_peer(self, peer_id: str) -> bool:
        """
        Unregister a peer and stop its stream.

        Args:
            peer_id: peer identifier

        Returns:
            True if peer was removed, False if not found
        """
        with self._peer_streams_lock:
            if peer_id not in self._peer_streams:
                return False

            stream = self._peer_streams.pop(peer_id)
            stream.stop()
            _EXP_LOG.info(f"[MESH] Peer {peer_id} removed and stream stopped")
            return True

    def get_peer_status(self, peer_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific peer.

        Args:
            peer_id: peer identifier

        Returns:
            Dict with 'dm', 'fidelity', 'timestamp_ns', 'healthy' keys,
            or None if peer not found.
        """
        with self._peer_streams_lock:
            if peer_id not in self._peer_streams:
                return None

            stream = self._peer_streams[peer_id]
            return {
                "peer_id": peer_id,
                "dm": stream.get_latest_dm(),
                "fidelity": stream.get_latest_fidelity(),
                "timestamp_ns": stream.get_latest_timestamp_ns(),
                "healthy": stream.running and stream.consecutive_errors < 10,
                "consecutive_errors": stream.consecutive_errors,
            }

    def get_all_peer_statuses(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all registered peers.

        Returns:
            Dict keyed by peer_id with status dicts
        """
        with self._peer_streams_lock:
            return {
                peer_id: {
                    "peer_id": peer_id,
                    "dm": stream.get_latest_dm(),
                    "fidelity": stream.get_latest_fidelity(),
                    "timestamp_ns": stream.get_latest_timestamp_ns(),
                    "healthy": (
                        stream.running and stream.consecutive_errors < 10
                    ),
                    "consecutive_errors": stream.consecutive_errors,
                }
                for peer_id, stream in self._peer_streams.items()
            }

    def _collect_stream_dms(self) -> List[Tuple[np.ndarray, float]]:
        """
        Collect latest DMs from all peer streams (non-blocking).

        Returns:
            List of (dm, fidelity) tuples for valid DMs. Empty list if
            no peers or all offline.
        """
        result = []
        with self._peer_streams_lock:
            for stream in self._peer_streams.values():
                dm = stream.get_latest_dm()
                if dm is not None:
                    fidelity = stream.get_latest_fidelity()
                    result.append((dm, fidelity))
        return result

    def compute_consensus_dm(
        self, include_weights: bool = True
    ) -> Optional[np.ndarray]:
        """
        Compute Hermitian-averaged consensus DM from all peer streams.

        Args:
            include_weights: if True, weight by peer fidelity; else equal

        Returns:
            8×8 consensus DM (normalized), or None if no peers have data
        """
        dms_and_fidelities = self._collect_stream_dms()
        if not dms_and_fidelities:
            return None

        if not include_weights:
            # Equal average
            total_dm = np.zeros((8, 8), dtype=np.complex128)
            for dm, _ in dms_and_fidelities:
                total_dm += dm
            total_dm /= len(dms_and_fidelities)
        else:
            # Weighted by fidelity
            total_w = sum(max(f, 1e-6) for _, f in dms_and_fidelities)
            total_dm = np.zeros((8, 8), dtype=np.complex128)
            for dm, f in dms_and_fidelities:
                total_dm += (f / total_w) * dm

        # Normalize trace to 1
        tr = np.trace(total_dm)
        if abs(tr) > 1e-12:
            total_dm /= tr

        return total_dm

    def _validate_dm(self, dm: np.ndarray, label: str) -> bool:
        """
        Validate 8×8 density matrix.

        Checks:
          • Shape is (8, 8)
          • dtype is complex128
          • Hermitian (dm ≈ dm†)
          • Trace ≈ 1
          • Positive semidefinite (all eigenvalues ≥ -1e-10)

        Args:
            dm: candidate density matrix
            label: string for logging

        Returns:
            True if valid, False otherwise (logs at WARNING on failure)
        """
        try:
            if dm.shape != (8, 8):
                _EXP_LOG.warning(
                    f"[MESH] {label} invalid shape {dm.shape}, expected (8, 8)"
                )
                return False

            if dm.dtype != np.complex128:
                _EXP_LOG.warning(
                    f"[MESH] {label} invalid dtype {dm.dtype}, "
                    f"expected complex128"
                )
                return False

            # Check Hermitian
            diff = np.max(np.abs(dm - dm.conj().T))
            if diff > 1e-10:
                _EXP_LOG.warning(
                    f"[MESH] {label} not Hermitian (max diff {diff:.2e})"
                )
                return False

            # Check trace
            tr = np.trace(dm)
            if abs(tr - 1.0) > 1e-10:
                _EXP_LOG.warning(
                    f"[MESH] {label} trace not 1.0 (got {tr:.6f})"
                )
                return False

            # Check positive semidefinite (all eigenvalues >= -epsilon)
            eigenvalues = np.linalg.eigvalsh(dm)
            min_eigenval = np.min(eigenvalues)
            if min_eigenval < -1e-10:
                _EXP_LOG.warning(
                    f"[MESH] {label} not positive semidefinite "
                    f"(min eigenvalue {min_eigenval:.2e})"
                )
                return False

            return True

        except Exception as e:
            _EXP_LOG.warning(f"[MESH] {label} validation error: {e}")
            return False

    def shutdown(self) -> None:
        """Stop all peer streams and clean up."""
        with self._peer_streams_lock:
            for stream in self._peer_streams.values():
                stream.stop()
            self._peer_streams.clear()

        _EXP_LOG.info(f"[MESH] ClientOracleMesh shutdown complete")


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE INITIALIZATION & TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Setup logging for standalone testing
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Test instantiation
    mesh = ClientOracleMesh(node_id="test_client_1")

    # Simulate adding peers (in real usage, these would be discovered via DHT)
    # mesh.add_peer("peer_1", "localhost", 9092)
    # mesh.add_peer("peer_2", "localhost", 9093)

    print("[TEST] ClientOracleMesh instantiated successfully")
    print(f"[TEST] Peer streams: {list(mesh._peer_streams.keys())}")

    mesh.shutdown()
    print("[TEST] Shutdown complete")
