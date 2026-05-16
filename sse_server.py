#!/usr/bin/env python3
"""
Standalone SSE Server — Quantum Information Streaming Service

Handles concurrent clients streaming 16³ density matrix snapshots,
block events, and metrics. Uses eventlet green threads (async).

Connection Management:
- Max 5 concurrent SSE streams per client IP
- Max 500 total concurrent streams globally
- Automatic cleanup of stale connections
- Connection deduplication by IP

Architecture:
- /rpc/oracle/snapshot GET — stream real-time 16³ quantum snapshots
- /rpc/events/blocks GET — stream block events
- /rpc/metrics/push GET — stream metrics (placeholder)
- /push/snapshot POST — internal: main server pushes snapshot, fan-out to all clients
- /push/block POST — internal: lattice_controller pushes block, fan-out to all clients
- /health GET — instant health check for Koyeb
"""

import os
import json
import queue
import threading
import logging
import time
from collections import defaultdict, deque
from flask import Flask, Response, request

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONNECTION LIMITS
# ═══════════════════════════════════════════════════════════════════════════════

MAX_CONNECTIONS_PER_IP = int(os.environ.get("SSE_MAX_PER_IP", 2))
MAX_TOTAL_CONNECTIONS = int(os.environ.get("SSE_MAX_TOTAL", 500))
CONNECTION_CLEANUP_INTERVAL = 10.0  # seconds


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRATION HELPER — attach SSE routes to any Flask app
# ═══════════════════════════════════════════════════════════════════════════════

_registered_apps: list = []

def register_sse_routes(flask_app) -> None:
    """Register all SSE routes onto *flask_app* so SSE endpoints are served
    directly from the main server — no proxy, no separate process needed."""
    if flask_app in _registered_apps:
        return
    _registered_apps.append(flask_app)

    flask_app.add_url_rule("/rpc/oracle/snapshot",   view_func=rpc_oracle_snapshot,   methods=["GET", "OPTIONS"])
    flask_app.add_url_rule("/rpc/events/blocks",     view_func=rpc_events_blocks,     methods=["GET", "OPTIONS"])
    flask_app.add_url_rule("/rpc/blocks/stream",     view_func=rpc_blocks_stream,     methods=["GET"])
    flask_app.add_url_rule("/rpc/metrics/push",      view_func=rpc_metrics_push,      methods=["GET"])
    flask_app.add_url_rule("/rpc/oracle/consensus",  view_func=rpc_oracle_consensus,  methods=["GET", "OPTIONS"])
    flask_app.add_url_rule("/rpc/events/mempool",   view_func=rpc_events_mempool,   methods=["GET", "OPTIONS"])
    flask_app.add_url_rule("/rpc/events/db_sync",   view_func=sse_db_sync,          methods=["GET"])
    # Internal push endpoints (for direct fan-out bypass; server.py now uses in-memory fan-out)
    flask_app.add_url_rule("/push/snapshot",         view_func=push_snapshot,         methods=["POST"])
    flask_app.add_url_rule("/push/block",            view_func=push_block,            methods=["POST"])
    flask_app.add_url_rule("/push/metric",           view_func=push_metric,           methods=["POST"])
    flask_app.add_url_rule("/push/oracle_consensus", view_func=push_oracle_consensus, methods=["POST"])
    flask_app.add_url_rule("/push/mempool",          view_func=push_mempool,          methods=["POST"])
    logger.info(f"[SSE] Registered {len(_registered_apps)} SSE route group(s)")

# ═══════════════════════════════════════════════════════════════════════════════
# IN-MEMORY STATE: Per-client queues for fan-out
# ═══════════════════════════════════════════════════════════════════════════════

class SSEClient:
    """Represents a single SSE client connection."""
    __slots__ = ('client_id', 'ip', 'queue', 'connected_at', 'last_activity')
    
    def __init__(self, client_id, ip):
        self.client_id = client_id
        self.ip = ip
        self.queue = queue.Queue(maxsize=50)
        self.connected_at = time.time()
        self.last_activity = time.time()
    
    def touch(self):
        self.last_activity = time.time()


class SSEChannel:
    """Manages clients for a single SSE endpoint."""
    
    def __init__(self, name):
        self.name = name
        self.clients = []  # list of SSEClient
        self.lock = threading.RLock()
        self._next_id = 0
        self._id_lock = threading.Lock()
    
    def _alloc_id(self):
        with self._id_lock:
            self._next_id += 1
            return self._next_id
    
    def get_client_counts(self):
        """Return (total_clients, per_ip_counts)."""
        with self.lock:
            per_ip = defaultdict(int)
            for c in self.clients:
                per_ip[c.ip] += 1
            return len(self.clients), dict(per_ip)
    
    def connect(self, client_ip):
        """Register a new client, enforcing connection limits."""
        with self.lock:
            # Count connections from this IP
            ip_count = sum(1 for c in self.clients if c.ip == client_ip)
            
            if ip_count >= MAX_CONNECTIONS_PER_IP:
                logger.debug(
                    f"[SSE-{self.name}] IP {client_ip} has {ip_count} conns (max {MAX_CONNECTIONS_PER_IP}), evicting oldest"
                )
                # Close oldest connection from this IP
                oldest = None
                oldest_time = float('inf')
                for c in self.clients:
                    if c.ip == client_ip and c.connected_at < oldest_time:
                        oldest = c
                        oldest_time = c.connected_at
                
                if oldest:
                    self._close_client(oldest, reason="per_ip_limit")
            
            # Check global limit
            if len(self.clients) >= MAX_TOTAL_CONNECTIONS:
                logger.warning(
                    f"[SSE-{self.name}] Global limit reached ({len(self.clients)}/{MAX_TOTAL_CONNECTIONS}). "
                    f"Closing oldest connection."
                )
                # Close oldest connection overall
                oldest = min(self.clients, key=lambda c: c.connected_at, default=None)
                if oldest:
                    self._close_client(oldest, reason="global_limit")
            
            # Create new client
            client = SSEClient(self._alloc_id(), client_ip)
            self.clients.append(client)
            
            total, per_ip = self.get_client_counts()
            ip_str = f" (IP {client_ip} now has {per_ip.get(client_ip, 0)} conns)"
            logger.debug(f"[SSE-{self.name}] client {client.client_id} connected{ip_str}. Total: {total}")
            
            return client
    
    def disconnect(self, client):
        """Remove a client from the channel."""
        with self.lock:
            self._close_client(client, reason="disconnect")
    
    def _close_client(self, client, reason="cleanup"):
        """Close a client connection and remove from list."""
        if client in self.clients:
            self.clients.remove(client)
            # Signal closure by putting None in queue
            try:
                client.queue.put_nowait(None)
            except queue.Full:
                pass
            total, per_ip = self.get_client_counts()
            logger.debug(
                f"[SSE-{self.name}] client {client.client_id} closed ({reason}). "
                f"Remaining: {total}"
            )
    
    def cleanup_stale(self, max_age=60.0):
        """Remove clients that haven't received data in max_age seconds."""
        now = time.time()
        with self.lock:
            stale = [c for c in self.clients if (now - c.last_activity) > max_age]
            for c in stale:
                self._close_client(c, reason="stale")
            return len(stale)
    
    def subscribe(self, client_ip="0.0.0.0"):
        """Subscribe a client — alias for connect(). Returns an SSEClient whose
        queue can be consumed via client.queue.get()."""
        return self.connect(client_ip)

    def unsubscribe(self, client):
        """Unsubscribe a client — alias for disconnect()."""
        self.disconnect(client)

    def fan_out(self, payload):
        """Send payload to all connected clients."""
        with self.lock:
            for client in list(self.clients):
                client.touch()
                try:
                    client.queue.put_nowait(payload)
                except queue.Full:
                    try:
                        client.queue.get_nowait()
                        client.queue.put_nowait(payload)
                    except Exception:
                        pass
            return len(self.clients)


# Initialize channels
_snapshot_channel = SSEChannel("snapshot")
_blocks_channel = SSEChannel("blocks")
_metrics_channel = SSEChannel("metrics")
_oracle_consensus_channel = SSEChannel("oracle_consensus")
_mempool_channel = SSEChannel("mempool")
_db_sync_channel = SSEChannel("db_sync")


# ═══════════════════════════════════════════════════════════════════════════════
# CONNECTION CLEANUP WORKER
# ═══════════════════════════════════════════════════════════════════════════════

def _cleanup_worker():
    """Background thread: periodically remove stale connections."""
    while True:
        time.sleep(CONNECTION_CLEANUP_INTERVAL)
        try:
            total_removed = 0
            total_removed += _snapshot_channel.cleanup_stale(max_age=15.0)
            total_removed += _blocks_channel.cleanup_stale(max_age=15.0)
            total_removed += _metrics_channel.cleanup_stale(max_age=15.0)
            total_removed += _oracle_consensus_channel.cleanup_stale(max_age=15.0)
            total_removed += _mempool_channel.cleanup_stale(max_age=15.0)
            
            if total_removed > 0:
                snap_total, _ = _snapshot_channel.get_client_counts()
                block_total, _ = _blocks_channel.get_client_counts()
                metric_total, _ = _metrics_channel.get_client_counts()
                consensus_total, _ = _oracle_consensus_channel.get_client_counts()
                logger.info(
                    f"[SSE-CLEANUP] Removed {total_removed} stale connections. "
                    f"Current: snapshot={snap_total}, blocks={block_total}, metrics={metric_total}, consensus={consensus_total}"
                )
        except Exception as e:
            logger.error(f"[SSE-CLEANUP] Error: {e}")


# Start cleanup thread
_cleanup_thread = threading.Thread(target=_cleanup_worker, daemon=True, name="sse-cleanup")
_cleanup_thread.start()


# ═══════════════════════════════════════════════════════════════════════════════
# SSE ENDPOINTS — Stream to connected clients
# ═══════════════════════════════════════════════════════════════════════════════

def _sse_generator(channel, client):
    """Generic SSE generator for any channel.

    Yields SSE events from the client's queue.  When the queue is empty for
    1 second, yields a heartbeat comment (`: heartbeat\\n\\n`).  If the
    downstream socket is broken (client disconnected), the yield will raise
    GeneratorExit (or occasionally BrokenPipeError/ConnectionResetError)
    which triggers cleanup in the finally block.
    """
    try:
        while True:
            try:
                data = client.queue.get(timeout=1.0)
                if data is None:
                    # Shutdown signal from _close_client
                    break
                if data:
                    yield f"data: {json.dumps(data)}\n\n"
            except queue.Empty:
                client.touch()
                # Heartbeat keeps the connection alive and — critically —
                # triggers a socket write that will fail if the client is gone.
                yield ": heartbeat\n\n"
    except (GeneratorExit, BrokenPipeError, ConnectionResetError, OSError):
        pass  # Client disconnected — normal for SSE
    finally:
        channel.disconnect(client)


def _get_client_ip():
    """Extract client IP from request, handling proxies."""
    # Check X-Forwarded-For header first (for reverse proxies)
    forwarded = request.headers.get('X-Forwarded-For')
    if forwarded:
        return forwarded.split(',')[0].strip()
    # Check X-Real-IP
    real_ip = request.headers.get('X-Real-IP')
    if real_ip:
        return real_ip.strip()
    # Fall back to remote_addr
    return request.remote_addr or "unknown"


@app.route("/rpc/oracle/snapshot", methods=["GET", "POST", "OPTIONS"])
def rpc_oracle_snapshot():
    """SSE stream: Real-time 16³ density matrix snapshots for quantum clients."""
    if request.method == "OPTIONS":
        return "", 204

    client_ip = _get_client_ip()
    client = _snapshot_channel.connect(client_ip)

    return Response(
        _sse_generator(_snapshot_channel, client),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.route("/rpc/events/blocks", methods=["GET", "POST", "OPTIONS"])
def rpc_events_blocks():
    """SSE stream: Real-time block events for blockchain clients."""
    if request.method == "OPTIONS":
        return "", 204

    client_ip = _get_client_ip()
    client = _blocks_channel.connect(client_ip)

    return Response(
        _sse_generator(_blocks_channel, client),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.route("/rpc/blocks/stream", methods=["GET"])
def rpc_blocks_stream():
    """SSE stream: Real-time block information (height, hash, timestamp)."""
    client_ip = _get_client_ip()
    client = _blocks_channel.connect(client_ip)

    def blocks_stream_generator():
        try:
            while True:
                try:
                    block = client.queue.get(timeout=1.0)
                    if block is None:
                        break
                    if block:
                        formatted = {
                            "height": block.get("height", 0),
                            "hash": block.get("hash") or block.get("block_hash", ""),
                            "block_hash": block.get("hash") or block.get("block_hash", ""),
                            "timestamp": block.get("timestamp") or block.get("timestamp_s", 0),
                            "timestamp_s": block.get("timestamp") or block.get("timestamp_s", 0),
                            "miner_address": block.get("miner_address", ""),
                        }
                        yield f"data: {json.dumps(formatted)}\n\n"
                except queue.Empty:
                    client.touch()
                    yield ": heartbeat\n\n"
        except GeneratorExit:
            logger.debug(f"[SSE-blocks/stream] client {client.client_id} GeneratorExit")
        finally:
            _blocks_channel.disconnect(client)

    return Response(
        blocks_stream_generator(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.route("/rpc/metrics/push", methods=["GET"])
def rpc_metrics_push():
    """SSE stream: Metrics push (placeholder — currently streams heartbeats only)."""
    client_ip = _get_client_ip()
    client = _metrics_channel.connect(client_ip)

    return Response(
        _sse_generator(_metrics_channel, client),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL PUSH ENDPOINTS — Main server pushes data, SSE service fan-outs
# ═══════════════════════════════════════════════════════════════════════════════


def _fan_out_snapshot(payload: dict) -> int:
    """Fan-out a snapshot payload to all connected /rpc/oracle/snapshot clients."""
    return _snapshot_channel.fan_out(payload)


def _fan_out_block(payload: dict) -> int:
    """Fan-out a block payload to all connected /rpc/events/blocks clients."""
    return _blocks_channel.fan_out(payload)


def _fan_out_metric(payload: dict) -> int:
    """Fan-out a metric payload to all connected /rpc/metrics/push clients."""
    return _metrics_channel.fan_out(payload)


def _fan_out_oracle_consensus(payload: dict) -> int:
    """Fan-out oracle consensus payload to all connected /rpc/oracle/consensus clients."""
    return _oracle_consensus_channel.fan_out(payload)


@app.route("/push/snapshot", methods=["POST"])
def push_snapshot():
    """Internal: main server pushes snapshot frame, fan-out to all /rpc/oracle/snapshot clients."""
    try:
        payload = request.get_json()
        if not payload:
            return {"error": "No payload"}, 400
        n = _fan_out_snapshot(payload)
        logger.debug(f"[SSE] /push/snapshot fan-out to {n} clients")
        return {"status": "ok", "clients": n}, 200
    except Exception as e:
        logger.error(f"[SSE] /push/snapshot error: {e}")
        return {"error": str(e)}, 500


@app.route("/push/block", methods=["POST"])
def push_block():
    """Internal: main server pushes block event, fan-out to all /rpc/events/blocks clients."""
    try:
        payload = request.get_json()
        if not payload:
            return {"error": "No payload"}, 400
        n = _fan_out_block(payload)
        logger.debug(f"[SSE] /push/block fan-out to {n} clients")
        return {"status": "ok", "clients": n}, 200
    except Exception as e:
        logger.error(f"[SSE] /push/block error: {e}")
        return {"error": str(e)}, 500


@app.route("/push/metric", methods=["POST"])
def push_metric():
    """Internal: main server pushes metric, fan-out to all /rpc/metrics/push clients."""
    try:
        payload = request.get_json()
        if not payload:
            return {"error": "No payload"}, 400
        n = _fan_out_metric(payload)
        return {"status": "ok", "clients": n}, 200
    except Exception as e:
        logger.error(f"[SSE] /push/metric error: {e}")
        return {"error": str(e)}, 500


@app.route("/push/oracle_consensus", methods=["POST"])
def push_oracle_consensus():
    """Internal: oracle pushes consensus event, fan-out to all /rpc/oracle/consensus clients."""
    try:
        payload = request.get_json()
        if not payload:
            return {"error": "No payload"}, 400
        n = _fan_out_oracle_consensus(payload)
        logger.debug(f"[SSE] /push/oracle_consensus fan-out to {n} clients")
        return {"status": "ok", "clients": n}, 200
    except Exception as e:
        logger.error(f"[SSE] /push/oracle_consensus error: {e}")
        return {"error": str(e)}, 500


@app.route("/push/db_sync", methods=["POST"])
def push_db_sync():
    """Internal push: fires after block settlement commits to Neon DB.
    Clients subscribed to /rpc/events/db_sync know to re-fetch authoritative state."""
    try:
        payload = request.get_json() or {}
        n = _db_sync_channel.fan_out(payload)
        logger.debug(f"[SSE] /push/db_sync fan-out to {n} clients")
        return {"status": "ok", "clients": n}, 200
    except Exception as e:
        logger.error(f"[SSE] /push/db_sync error: {e}")
        return {"error": str(e)}, 500


def _fan_out_mempool(payload: dict) -> int:
    """Fan-out a mempool event payload to all connected /rpc/events/mempool clients."""
    return _mempool_channel.fan_out(payload)


@app.route("/push/mempool", methods=["POST"])
def push_mempool():
    """Internal: mempool pushes new TX event, fan-out to all /rpc/events/mempool clients."""
    try:
        payload = request.get_json()
        if not payload:
            return {"error": "No payload"}, 400
        n = _fan_out_mempool(payload)
        logger.debug(f"[SSE] /push/mempool fan-out to {n} clients")
        return {"status": "ok", "clients": n}, 200
    except Exception as e:
        logger.error(f"[SSE] /push/mempool error: {e}")
        return {"error": str(e)}, 500


@app.route("/rpc/events/mempool", methods=["GET", "POST", "OPTIONS"])
def rpc_events_mempool():
    """SSE stream: Real-time mempool transaction events for miners and explorers."""
    if request.method == "OPTIONS":
        return "", 204

    client_ip = _get_client_ip()
    client = _mempool_channel.connect(client_ip)

    return Response(
        _sse_generator(_mempool_channel, client),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.route("/rpc/oracle/consensus", methods=["GET", "POST", "OPTIONS"])
def rpc_oracle_consensus():
    """SSE stream: Real-time oracle consensus events (attestations, finalizations)."""
    if request.method == "OPTIONS":
        return "", 204

    client_ip = _get_client_ip()
    client = _oracle_consensus_channel.connect(client_ip)

    return Response(
        _sse_generator(_oracle_consensus_channel, client),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════════════


@app.route("/health", methods=["GET"])
def health():
    """Instant health check for Koyeb."""
    snap_total, snap_per_ip = _snapshot_channel.get_client_counts()
    block_total, block_per_ip = _blocks_channel.get_client_counts()
    metric_total, metric_per_ip = _metrics_channel.get_client_counts()
    consensus_total, consensus_per_ip = _oracle_consensus_channel.get_client_counts()
    
    return {
        "status": "ok",
        "service": "sse",
        "snapshot_clients": snap_total,
        "blocks_clients": block_total,
        "metrics_clients": metric_total,
        "consensus_clients": consensus_total,
        "limits": {
            "max_per_ip": MAX_CONNECTIONS_PER_IP,
            "max_total": MAX_TOTAL_CONNECTIONS,
        },
        "top_ips": {
            "snapshot": dict(sorted(snap_per_ip.items(), key=lambda x: -x[1])[:5]),
            "blocks": dict(sorted(block_per_ip.items(), key=lambda x: -x[1])[:5]),
            "consensus": dict(sorted(consensus_per_ip.items(), key=lambda x: -x[1])[:5]),
        },
    }, 200




@app.route("/rpc/events/db_sync")
def sse_db_sync():
    """SSE stream: fires after every block settlement commits to Neon.
    Frontend subscribes here and re-fetches block + TX data on each pulse."""
    client_ip = _get_client_ip()
    client = _db_sync_channel.connect(client_ip)

    return Response(
        _sse_generator(_db_sync_channel, client),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


if __name__ == "__main__":
    # Development only — production uses gunicorn
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8001)))
