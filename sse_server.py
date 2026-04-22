#!/usr/bin/env python3
"""
Standalone SSE Server — Quantum Information Streaming Service

Handles unlimited concurrent clients streaming 16³ density matrix snapshots,
block events, and metrics. Uses eventlet green threads (async) instead of
synchronous OS threads, so thousands of concurrent SSE connections don't
exhaust gunicorn worker pool.

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
import requests
from flask import Flask, Response, request

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# IN-MEMORY STATE: Per-client queues for fan-out
# ═══════════════════════════════════════════════════════════════════════════════

_snapshot_clients = []  # list of (client_id, queue.Queue) tuples
_snapshot_lock = threading.RLock()

_blocks_clients = []
_blocks_lock = threading.RLock()

_metrics_clients = []
_metrics_lock = threading.RLock()

_transactions_clients = []
_transactions_lock = threading.RLock()

_next_client_id = 0
_client_id_lock = threading.Lock()


def _allocate_client_id():
    global _next_client_id
    with _client_id_lock:
        _next_client_id += 1
        return _next_client_id


# ═══════════════════════════════════════════════════════════════════════════════
# SSE ENDPOINTS — Stream to connected clients
# ═══════════════════════════════════════════════════════════════════════════════


@app.route("/rpc/oracle/snapshot", methods=["GET", "POST", "OPTIONS"])
def rpc_oracle_snapshot():
    """SSE stream: Real-time 16³ density matrix snapshots for quantum clients.

    Unlimited concurrent connections supported (green threads via eventlet).
    """
    if request.method == "OPTIONS":
        return "", 204

    client_id = _allocate_client_id()
    client_queue = queue.Queue(maxsize=50)

    with _snapshot_lock:
        _snapshot_clients.append((client_id, client_queue))

    logger.info(f"[SSE] /rpc/oracle/snapshot client {client_id} connected")

    def snapshot_generator():
        try:
            while True:
                try:
                    frame = client_queue.get(timeout=1.0)
                    if frame:
                        yield f"data: {json.dumps(frame)}\n\n"
                except queue.Empty:
                    yield ": heartbeat\n\n"
        except GeneratorExit:
            logger.info(f"[SSE] /rpc/oracle/snapshot client {client_id} disconnected")
        finally:
            with _snapshot_lock:
                _snapshot_clients[:] = [
                    (cid, q) for cid, q in _snapshot_clients if cid != client_id
                ]

    return Response(
        snapshot_generator(),
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

    client_id = _allocate_client_id()
    client_queue = queue.Queue(maxsize=50)

    with _blocks_lock:
        _blocks_clients.append((client_id, client_queue))

    logger.info(f"[SSE] /rpc/events/blocks client {client_id} connected")

    def blocks_generator():
        try:
            while True:
                try:
                    event = client_queue.get(timeout=1.0)
                    if event:
                        yield f"data: {json.dumps(event)}\n\n"
                except queue.Empty:
                    yield ": heartbeat\n\n"
        except GeneratorExit:
            logger.info(f"[SSE] /rpc/events/blocks client {client_id} disconnected")
        finally:
            with _blocks_lock:
                _blocks_clients[:] = [
                    (cid, q) for cid, q in _blocks_clients if cid != client_id
                ]

    return Response(
        blocks_generator(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.route("/rpc/blocks/recent", methods=["GET"])
def rpc_blocks_recent():
    """RPC GET: Fetch recent blocks with transactions (up to 20).
    
    Query params:
      - limit: max blocks to return (default 20, max 100)
      - offset: pagination offset (default 0)
    
    This endpoint queries the main server via RPC calls since SSE server
    has no direct DB access. Falls back to empty list if server unavailable.
    """
    try:
        import requests
        
        limit = min(int(request.args.get("limit", 20)), 100)
        offset = int(request.args.get("offset", 0))
        
        # Get server address from environment or default
        server_url = os.getenv('MAIN_SERVER_URL', 'http://localhost:8000')
        
        # Query main server for block height
        try:
            height_resp = requests.get(
                f"{server_url}/rpc?method=qtcl_getBlockHeight&params=%5B%5D&id=1",
                timeout=5
            )
            height_data = height_resp.json()
            max_height = height_data.get("result", {}).get("height", 0) if height_data.get("result") else 0
        except:
            max_height = 0
        
        blocks = []
        if max_height > 0:
            from_h = max(0, max_height - limit + 1)
            # Fetch individual blocks with transactions
            for h in range(max_height, max(from_h - 1, -1), -1):
                try:
                    block_resp = requests.get(
                        f"{server_url}/rpc?method=qtcl_getBlock&params=%5B{h}%5D&id=1",
                        timeout=3
                    )
                    block_data = block_resp.json()
                    if block_data.get("result"):
                        blocks.append(block_data["result"])
                except:
                    pass
        
        return {
            "jsonrpc": "2.0",
            "result": blocks,
            "id": None
        }, 200
    except Exception as e:
        logger.error(f"[SSE] /rpc/blocks/recent error: {e}")
        return {"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e)}, "id": None}, 500


@app.route("/rpc/blocks/stream", methods=["GET"])
def rpc_blocks_stream():
    """SSE stream: Real-time block information (height, hash, timestamp).

    Streams block data in format: {"height": N, "hash": "...", "timestamp": ...}
    """
    client_id = _allocate_client_id()
    client_queue = queue.Queue(maxsize=50)

    with _blocks_lock:
        _blocks_clients.append((client_id, client_queue))

    logger.info(f"[SSE] /rpc/blocks/stream client {client_id} connected")

    def blocks_stream_generator():
        try:
            while True:
                try:
                    block = client_queue.get(timeout=1.0)
                    if block:
                        # Format block data with height, hash, timestamp
                        formatted = {
                            "height": block.get("height", 0),
                            "hash": block.get("hash", ""),
                            "timestamp": block.get("timestamp", 0),
                        }
                        yield f"data: {json.dumps(formatted)}\n\n"
                except queue.Empty:
                    yield ": heartbeat\n\n"
        except GeneratorExit:
            logger.info(f"[SSE] /rpc/blocks/stream client {client_id} disconnected")
        finally:
            with _blocks_lock:
                _blocks_clients[:] = [
                    (cid, q) for cid, q in _blocks_clients if cid != client_id
                ]

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

    client_id = _allocate_client_id()
    client_queue = queue.Queue(maxsize=50)

    with _metrics_lock:
        _metrics_clients.append((client_id, client_queue))

    logger.info(f"[SSE] /rpc/metrics/push client {client_id} connected")

    def metrics_generator():
        try:
            while True:
                try:
                    metric = client_queue.get(timeout=2.0)
                    if metric:
                        yield f"data: {json.dumps(metric)}\n\n"
                except queue.Empty:
                    yield ": heartbeat\n\n"
        except GeneratorExit:
            logger.info(f"[SSE] /rpc/metrics/push client {client_id} disconnected")
        finally:
            with _metrics_lock:
                _metrics_clients[:] = [
                    (cid, q) for cid, q in _metrics_clients if cid != client_id
                ]

    return Response(
        metrics_generator(),
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


@app.route("/push/snapshot", methods=["POST"])
def push_snapshot():
    """Internal: main server pushes snapshot frame, fan-out to all /rpc/oracle/snapshot clients."""
    try:
        payload = request.get_json()
        if not payload:
            return {"error": "No payload"}, 400

        with _snapshot_lock:
            for client_id, client_queue in _snapshot_clients:
                try:
                    client_queue.put_nowait(payload)
                except queue.Full:
                    # Drop oldest item and retry (fire-and-forget: don't care if we fail)
                    try:
                        client_queue.get_nowait()
                        client_queue.put_nowait(payload)
                    except:
                        pass

        logger.debug(
            f"[SSE] /push/snapshot fan-out to {len(_snapshot_clients)} clients"
        )
        return {"status": "ok", "clients": len(_snapshot_clients)}, 200
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

        with _blocks_lock:
            for client_id, client_queue in _blocks_clients:
                try:
                    client_queue.put_nowait(payload)
                except queue.Full:
                    try:
                        client_queue.get_nowait()
                        client_queue.put_nowait(payload)
                    except:
                        pass

        logger.debug(f"[SSE] /push/block fan-out to {len(_blocks_clients)} clients")
        return {"status": "ok", "clients": len(_blocks_clients)}, 200
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

        with _metrics_lock:
            for client_id, client_queue in _metrics_clients:
                try:
                    client_queue.put_nowait(payload)
                except queue.Full:
                    try:
                        client_queue.get_nowait()
                        client_queue.put_nowait(payload)
                    except:
                        pass

        return {"status": "ok", "clients": len(_metrics_clients)}, 200
    except Exception as e:
        logger.error(f"[SSE] /push/metric error: {e}")
        return {"error": str(e)}, 500


@app.route("/push/transaction", methods=["POST"])
def push_transaction():
    """Internal: main server pushes transaction event, fan-out to /rpc/transactions/stream."""
    try:
        payload = request.get_json()
        if not payload:
            return {"error": "No payload"}, 400
        with _transactions_lock:
            for client_id, client_queue in _transactions_clients:
                try:
                    client_queue.put_nowait(payload)
                except queue.Full:
                    try:
                        client_queue.get_nowait()
                        client_queue.put_nowait(payload)
                    except:
                        pass
        return {"status": "ok", "clients": len(_transactions_clients)}, 200
    except Exception as e:
        logger.error(f"[SSE] /push/transaction error: {e}")
        return {"error": str(e)}, 500


@app.route("/rpc/transactions/stream", methods=["GET", "OPTIONS"])
def rpc_transactions_stream():
    """SSE stream: Real-time transaction events."""
    if request.method == "OPTIONS":
        return "", 204
    client_id = _allocate_client_id()
    client_queue = queue.Queue(maxsize=200)
    with _transactions_lock:
        _transactions_clients.append((client_id, client_queue))
    logger.info(f"[SSE] /rpc/transactions/stream client {client_id} connected")

    def tx_generator():
        try:
            while True:
                try:
                    tx = client_queue.get(timeout=1.0)
                    if tx:
                        yield f"data: {json.dumps(tx)}\n\n"
                except queue.Empty:
                    yield ": heartbeat\n\n"
        except GeneratorExit:
            logger.info(f"[SSE] /rpc/transactions/stream client {client_id} disconnected")
        finally:
            with _transactions_lock:
                _transactions_clients[:] = [
                    (cid, q) for cid, q in _transactions_clients if cid != client_id
                ]

    return Response(
        tx_generator(),
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
    return {
        "status": "ok",
        "service": "sse",
        "snapshot_clients": len(_snapshot_clients),
        "blocks_clients": len(_blocks_clients),
        "metrics_clients": len(_metrics_clients),
        "transactions_clients": len(_transactions_clients),
    }, 200


@app.route("/rpc/config/difficulty", methods=["GET"])
def rpc_config_difficulty():
    """RPC GET: Return current block mining difficulty (leading zeroes).
    
    Used by miners to know target PoW difficulty when building blocks.
    """
    import os
    difficulty = int(os.getenv('BLOCK_DIFFICULTY', '4'))
    return {
        "jsonrpc": "2.0",
        "result": {
            "difficulty": difficulty,
            "leading_zeroes": difficulty,
            "description": f"Blocks require {difficulty} leading zero bytes in hash"
        },
        "id": None
    }, 200


@app.route("/rpc/config", methods=["GET"])
def rpc_config():
    """RPC GET: Return all configurable parameters for block building.
    
    Includes: difficulty, rewards, network constants.
    Authoritative source for all network parameters.
    """
    import os
    return {
        "jsonrpc": "2.0",
        "result": {
            "block_difficulty": int(os.getenv('BLOCK_DIFFICULTY', '4')),
            "miner_reward": float(os.getenv('MINER_REWARD', '7.2')),
            "treasury_reward": float(os.getenv('TREASURY_REWARD', '0.8')),
            "max_peers": int(os.getenv('MAX_PEERS', '32')),
            "oracle_min_peers": int(os.getenv('ORACLE_MIN_PEERS', '3')),
            "wstate_mode": os.getenv('WSTATE_MODE', 'normal'),
        },
        "id": None
    }, 200


@app.route("/rpc/config/rewards", methods=["GET"])
def rpc_config_rewards():
    """RPC GET: Return miner and treasury reward amounts (authoritative, server-only).
    
    Used by miners to construct proper coinbase transactions.
    Cannot be overridden by client.
    """
    import os
    miner = float(os.getenv('MINER_REWARD', '7.2'))
    treasury = float(os.getenv('TREASURY_REWARD', '0.8'))
    return {
        "jsonrpc": "2.0",
        "result": {
            "miner_reward": miner,
            "treasury_reward": treasury,
            "total_per_block": miner + treasury,
        },
        "id": None
    }, 200


@app.route("/rpc/transactions/recent", methods=["GET"])
def rpc_transactions_recent():
    """RPC GET: Fetch recent transactions (paginated).
    
    Query params:
      - page: page number (default 0)
      - per_page: items per page (default 50, max 200)
      - type: filter by type (coinbase, transfer, miner_reward, etc.)
      - address: filter by from_addr or to_addr
    
    Queries main server RPC endpoint qtcl_getTransactions.
    """
    try:
        server_url = os.getenv('MAIN_SERVER_URL', 'http://localhost:8000')
        
        page = int(request.args.get("page", 0))
        per_page = min(int(request.args.get("per_page", 50)), 200)
        tx_type = request.args.get("type")
        address = request.args.get("address")
        
        # Build params for main server RPC call
        params = {
            "page": page,
            "per_page": per_page,
        }
        if tx_type:
            params["type"] = tx_type
        if address:
            params["address"] = address
        
        # Encode params as JSON for RPC call
        import json
        params_json = json.dumps(params)
        
        try:
            resp = requests.get(
                f"{server_url}/rpc?method=qtcl_getTransactions&params={params_json}&id=1",
                timeout=5
            )
            data = resp.json()
            return data, 200
        except:
            return {
                "jsonrpc": "2.0",
                "result": {"transactions": [], "total": 0, "pages": 0, "page": page},
                "id": None
            }, 200
    except Exception as e:
        logger.error(f"[SSE] /rpc/transactions/recent error: {e}")
        return {"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e)}, "id": None}, 500


@app.route("/rpc/block/<int:height>/transactions", methods=["GET"])
def rpc_block_transactions(height):
    """RPC GET: Fetch all transactions in a specific block.
    
    Returns transactions array with miner + treasury rewards.
    Queries main server qtcl_getBlockTransactions RPC method.
    """
    try:
        server_url = os.getenv('MAIN_SERVER_URL', 'http://localhost:8000')
        
        try:
            resp = requests.get(
                f"{server_url}/rpc?method=qtcl_getBlockTransactions&params=%5B{height}%5D&id=1",
                timeout=3
            )
            data = resp.json()
            return data, 200
        except:
            return {
                "jsonrpc": "2.0",
                "result": {"height": height, "tx_count": 0, "transactions": []},
                "id": None
            }, 200
    except Exception as e:
        logger.error(f"[SSE] /rpc/block/{height}/transactions error: {e}")
        return {"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e)}, "id": None}, 500


if __name__ == "__main__":
    # Development only — production uses gunicorn
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8001)))
