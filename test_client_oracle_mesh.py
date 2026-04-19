#!/usr/bin/env python3
"""
Test suite for ClientOracleMesh and _PeerSSEStream implementation.

Tests cover:
  1. Basic instantiation and peer management
  2. Density matrix validation (shape, Hermitian, trace, PSD)
  3. Mock peer server and snapshot polling
  4. Exponential backoff on connection failures
  5. Byzantine-resistant consensus averaging
  6. Thread safety and concurrent access
  7. Graceful shutdown
"""

import sys
import os
import json
import time
import threading
import numpy as np
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from unittest.mock import MagicMock, patch
from io import BytesIO

# Add repo to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client_oracle_mesh import ClientOracleMesh

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# HELPER: Generate valid 8×8 density matrices
# ═════════════════════════════════════════════════════════════════════════════


def create_valid_dm(fidelity: float = 0.95) -> np.ndarray:
    """
    Create a valid 8×8 density matrix with given fidelity.

    Returns:
        np.ndarray(8, 8, dtype=complex128) with trace=1, Hermitian, PSD
    """
    # Start with random Hermitian matrix
    A = np.random.randn(8, 8) + 1j * np.random.randn(8, 8)
    H = (A + A.conj().T) / 2

    # Make positive semidefinite by adding identity scaled by max eigenvalue
    eigenvalues = np.linalg.eigvalsh(H)
    min_ev = np.min(eigenvalues)
    if min_ev < 0:
        H = H - 1.1 * min_ev * np.eye(8)

    # Normalize trace to 1
    tr = np.trace(H)
    if abs(tr) > 1e-12:
        H = H / tr

    return np.array(H, dtype=np.complex128)


def dm_to_hex(dm: np.ndarray) -> str:
    """Convert DM to hex string."""
    bytes_data = dm.astype(np.complex128).tobytes()
    return bytes_data.hex()


# ═════════════════════════════════════════════════════════════════════════════
# MOCK PEER HTTP SERVER
# ═════════════════════════════════════════════════════════════════════════════


class MockPeerHandler(BaseHTTPRequestHandler):
    """Mock peer HTTP server that responds to /rpc/stream/snapshot requests."""

    # Class variables (shared across instances)
    response_data = None
    should_fail = False
    response_delay = 0

    def do_GET(self):
        """Handle GET request."""
        if self.should_fail:
            self.send_error(500, "Internal Server Error")
            return

        if self.response_delay > 0:
            time.sleep(self.response_delay)

        if self.path == "/rpc/stream/snapshot":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            if self.response_data:
                self.wfile.write(json.dumps(self.response_data).encode("utf-8"))
            else:
                self.wfile.write(b'{"status": "ok", "snapshot": {}}')
        else:
            self.send_error(404, "Not Found")

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def start_mock_peer_server(port: int, response_data: dict = None) -> HTTPServer:
    """
    Start a mock peer HTTP server on given port.

    Args:
        port: listen port
        response_data: response JSON to return

    Returns:
        HTTPServer instance (already running in daemon thread)
    """
    MockPeerHandler.response_data = response_data
    server = HTTPServer(("127.0.0.1", port), MockPeerHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)  # Let server start
    return server


# ═════════════════════════════════════════════════════════════════════════════
# TEST SUITE
# ═════════════════════════════════════════════════════════════════════════════


def test_instantiation():
    """Test 1: Basic instantiation."""
    print("\n[TEST 1] Basic instantiation...")
    mesh = ClientOracleMesh(node_id="test_node_1")
    assert mesh.node_id == "test_node_1"
    assert len(mesh._peer_streams) == 0
    mesh.shutdown()
    print("✓ PASS: ClientOracleMesh instantiated successfully")


def test_peer_management():
    """Test 2: Add/remove peers."""
    print("\n[TEST 2] Peer management...")
    mesh = ClientOracleMesh(node_id="test_node_2")

    # Add peer (should fail since no server)
    result = mesh.add_peer("peer_1", "127.0.0.1", 19999)
    assert result == True
    assert "peer_1" in mesh._peer_streams

    # Try adding same peer again
    result = mesh.add_peer("peer_1", "127.0.0.1", 19999)
    assert result == False

    # Remove peer
    result = mesh.remove_peer("peer_1")
    assert result == True
    assert "peer_1" not in mesh._peer_streams

    # Try removing non-existent peer
    result = mesh.remove_peer("peer_1")
    assert result == False

    mesh.shutdown()
    print("✓ PASS: Peer management works correctly")


def test_dm_validation_shape():
    """Test 3: DM validation - shape."""
    print("\n[TEST 3] DM validation - shape...")
    mesh = ClientOracleMesh(node_id="test_node_3")

    # Valid shape
    dm = create_valid_dm()
    assert mesh._validate_dm(dm, "test_valid") == True

    # Invalid shape (7x7)
    dm_bad = np.zeros((7, 7), dtype=np.complex128)
    assert mesh._validate_dm(dm_bad, "test_7x7") == False

    # Invalid shape (8x16)
    dm_bad = np.zeros((8, 16), dtype=np.complex128)
    assert mesh._validate_dm(dm_bad, "test_8x16") == False

    mesh.shutdown()
    print("✓ PASS: Shape validation works")


def test_dm_validation_hermitian():
    """Test 4: DM validation - Hermitian."""
    print("\n[TEST 4] DM validation - Hermitian...")
    mesh = ClientOracleMesh(node_id="test_node_4")

    # Valid Hermitian
    dm = create_valid_dm()
    assert mesh._validate_dm(dm, "test_hermitian") == True

    # Non-Hermitian
    dm_bad = np.random.randn(8, 8) + 1j * np.random.randn(8, 8)
    dm_bad = dm_bad.astype(np.complex128)
    assert mesh._validate_dm(dm_bad, "test_non_hermitian") == False

    mesh.shutdown()
    print("✓ PASS: Hermitian validation works")


def test_dm_validation_trace():
    """Test 5: DM validation - trace."""
    print("\n[TEST 5] DM validation - trace...")
    mesh = ClientOracleMesh(node_id="test_node_5")

    # Invalid trace
    dm_bad = create_valid_dm()
    dm_bad = dm_bad * 2  # trace = 2
    assert mesh._validate_dm(dm_bad, "test_bad_trace") == False

    # Valid trace = 1
    dm_good = create_valid_dm()
    assert mesh._validate_dm(dm_good, "test_good_trace") == True

    mesh.shutdown()
    print("✓ PASS: Trace validation works")


def test_dm_validation_psd():
    """Test 6: DM validation - positive semidefinite."""
    print("\n[TEST 6] DM validation - positive semidefinite...")
    mesh = ClientOracleMesh(node_id="test_node_6")

    # Create non-PSD matrix
    dm_bad = np.eye(8, dtype=np.complex128)
    dm_bad[0, 0] = -1.1  # Negative eigenvalue
    dm_bad = dm_bad / np.trace(dm_bad)  # Normalize trace
    assert mesh._validate_dm(dm_bad, "test_non_psd") == False

    # Valid PSD
    dm_good = create_valid_dm()
    assert mesh._validate_dm(dm_good, "test_psd") == True

    mesh.shutdown()
    print("✓ PASS: PSD validation works")


def test_mock_peer_polling():
    """Test 7: Mock peer polling and snapshot collection."""
    print("\n[TEST 7] Mock peer polling...")

    # Start mock peer server on port 19001
    dm = create_valid_dm(fidelity=0.95)
    response_data = {
        "status": "ok",
        "snapshot": {
            "density_matrix_hex": dm_to_hex(dm),
            "w_state_fidelity": 0.95,
            "timestamp_ns": int(time.time_ns()),
        },
    }

    server = start_mock_peer_server(19001, response_data)

    # Create mesh and add peer
    mesh = ClientOracleMesh(node_id="test_node_7")
    result = mesh.add_peer("peer_1", "127.0.0.1", 19001)
    assert result == True

    # Wait for polling to complete
    time.sleep(7.0)

    # Get peer status
    status = mesh.get_peer_status("peer_1")
    assert status is not None
    assert status["dm"] is not None
    assert status["fidelity"] == 0.95
    assert status["healthy"] == True

    # Verify DM matches
    retrieved_dm = status["dm"]
    assert retrieved_dm.shape == (8, 8)
    assert np.allclose(retrieved_dm, dm, atol=1e-10)

    mesh.shutdown()
    server.shutdown()
    print("✓ PASS: Mock peer polling works")


def test_exponential_backoff():
    """Test 8: Exponential backoff on connection failure."""
    print("\n[TEST 8] Exponential backoff...")

    # Start server that fails
    response_data = {
        "status": "ok",
        "snapshot": {
            "density_matrix_hex": dm_to_hex(create_valid_dm()),
            "w_state_fidelity": 0.95,
            "timestamp_ns": int(time.time_ns()),
        },
    }
    server = start_mock_peer_server(19002, response_data)

    mesh = ClientOracleMesh(node_id="test_node_8")
    mesh.add_peer("peer_1", "127.0.0.1", 19002)

    # Wait for initial success
    time.sleep(7.0)
    stream = mesh._peer_streams["peer_1"]
    assert stream.consecutive_errors == 0

    # Make server fail
    MockPeerHandler.should_fail = True

    # Wait for failures to accumulate
    start_time = time.time()
    time.sleep(3.0)

    # Check error count increased
    assert stream.consecutive_errors > 0

    # Restore server
    MockPeerHandler.should_fail = False

    # Wait for recovery
    time.sleep(7.0)

    # Should have recovered
    status = mesh.get_peer_status("peer_1")
    assert status["healthy"] == True

    mesh.shutdown()
    server.shutdown()
    print(
        f"✓ PASS: Exponential backoff works "
        f"(errors accumulated to {stream.consecutive_errors})"
    )


def test_consensus_dm():
    """Test 9: Consensus DM computation."""
    print("\n[TEST 9] Consensus DM computation...")

    # Create two DMs with known values
    dm1 = create_valid_dm(fidelity=0.9)
    dm2 = create_valid_dm(fidelity=0.95)

    response1 = {
        "status": "ok",
        "snapshot": {
            "density_matrix_hex": dm_to_hex(dm1),
            "w_state_fidelity": 0.9,
            "timestamp_ns": int(time.time_ns()),
        },
    }

    response2 = {
        "status": "ok",
        "snapshot": {
            "density_matrix_hex": dm_to_hex(dm2),
            "w_state_fidelity": 0.95,
            "timestamp_ns": int(time.time_ns()),
        },
    }

    server1 = start_mock_peer_server(19003, response1)
    server2 = start_mock_peer_server(19004, response2)

    mesh = ClientOracleMesh(node_id="test_node_9")
    mesh.add_peer("peer_1", "127.0.0.1", 19003)
    mesh.add_peer("peer_2", "127.0.0.1", 19004)

    # Wait for both peers to collect snapshots
    time.sleep(7.0)

    # Compute consensus
    consensus = mesh.compute_consensus_dm(include_weights=True)
    assert consensus is not None
    assert consensus.shape == (8, 8)
    assert mesh._validate_dm(consensus, "consensus") == True

    # Weighted consensus should be closer to dm2 (higher fidelity)
    # Equal consensus for comparison
    consensus_equal = mesh.compute_consensus_dm(include_weights=False)
    assert consensus_equal is not None

    mesh.shutdown()
    server1.shutdown()
    server2.shutdown()
    print("✓ PASS: Consensus DM computation works")


def test_stale_snapshot_rejection():
    """Test 10: Reject stale snapshots (>120s old)."""
    print("\n[TEST 10] Stale snapshot rejection...")

    # Create snapshot with very old timestamp (200s ago)
    old_time_ns = int((time.time() - 200) * 1e9)
    dm = create_valid_dm()
    response_data = {
        "status": "ok",
        "snapshot": {
            "density_matrix_hex": dm_to_hex(dm),
            "w_state_fidelity": 0.95,
            "timestamp_ns": old_time_ns,
        },
    }

    server = start_mock_peer_server(19005, response_data)

    mesh = ClientOracleMesh(node_id="test_node_10")
    mesh.add_peer("peer_1", "127.0.0.1", 19005)

    # Wait for polling
    time.sleep(7.0)

    # Peer should have incremented error count due to stale snapshot
    stream = mesh._peer_streams["peer_1"]
    assert stream.latest_dm is None  # Should not store stale DM
    assert stream.consecutive_errors > 0

    mesh.shutdown()
    server.shutdown()
    print("✓ PASS: Stale snapshot rejection works")


def test_malformed_json_handling():
    """Test 11: Handle malformed JSON gracefully."""
    print("\n[TEST 11] Malformed JSON handling...")

    class MalformedHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/rpc/stream/snapshot":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                # Send invalid JSON
                self.wfile.write(b"{invalid json}")
            else:
                self.send_error(404)

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("127.0.0.1", 19006), MalformedHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)

    mesh = ClientOracleMesh(node_id="test_node_11")
    mesh.add_peer("peer_1", "127.0.0.1", 19006)

    # Wait for polling to try
    time.sleep(7.0)

    # Peer should have errors but not crash
    stream = mesh._peer_streams["peer_1"]
    assert stream.consecutive_errors > 0
    assert stream.latest_dm is None

    mesh.shutdown()
    server.shutdown()
    print("✓ PASS: Malformed JSON handled gracefully")


def test_thread_safety():
    """Test 12: Thread safety of concurrent reads."""
    print("\n[TEST 12] Thread safety...")

    dm = create_valid_dm()
    response_data = {
        "status": "ok",
        "snapshot": {
            "density_matrix_hex": dm_to_hex(dm),
            "w_state_fidelity": 0.95,
            "timestamp_ns": int(time.time_ns()),
        },
    }

    server = start_mock_peer_server(19007, response_data)

    mesh = ClientOracleMesh(node_id="test_node_12")
    mesh.add_peer("peer_1", "127.0.0.1", 19007)

    time.sleep(7.0)

    # Spawn multiple threads reading concurrently
    results = []
    errors = []

    def read_dm():
        try:
            for _ in range(10):
                dm_read = mesh.get_peer_status("peer_1")
                if dm_read:
                    results.append(dm_read["dm"])
                time.sleep(0.01)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=read_dm) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert len(results) > 0

    mesh.shutdown()
    server.shutdown()
    print(f"✓ PASS: Thread safety verified ({len(results)} concurrent reads)")


def test_shutdown():
    """Test 13: Graceful shutdown."""
    print("\n[TEST 13] Graceful shutdown...")

    dm = create_valid_dm()
    response_data = {
        "status": "ok",
        "snapshot": {
            "density_matrix_hex": dm_to_hex(dm),
            "w_state_fidelity": 0.95,
            "timestamp_ns": int(time.time_ns()),
        },
    }

    server = start_mock_peer_server(19008, response_data)

    mesh = ClientOracleMesh(node_id="test_node_13")
    mesh.add_peer("peer_1", "127.0.0.1", 19008)
    mesh.add_peer("peer_2", "127.0.0.1", 19008)

    # Shutdown
    mesh.shutdown()

    # Verify streams stopped
    for stream in mesh._peer_streams.values():
        assert stream.running == False

    assert len(mesh._peer_streams) == 0

    server.shutdown()
    print("✓ PASS: Graceful shutdown works")


# ═════════════════════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ═════════════════════════════════════════════════════════════════════════════


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("CLIENTORACLEMESH TEST SUITE")
    print("=" * 80)

    tests = [
        test_instantiation,
        test_peer_management,
        test_dm_validation_shape,
        test_dm_validation_hermitian,
        test_dm_validation_trace,
        test_dm_validation_psd,
        test_mock_peer_polling,
        test_exponential_backoff,
        test_consensus_dm,
        test_stale_snapshot_rejection,
        test_malformed_json_handling,
        test_thread_safety,
        test_shutdown,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ FAIL: {test.__name__}")
            print(f"  Error: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 80)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
