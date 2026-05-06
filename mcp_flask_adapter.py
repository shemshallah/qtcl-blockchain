#!/usr/bin/env python3
"""
================================================================================
MCP FLASK ADAPTER — Mount Modern MCP Server into Existing Flask App
================================================================================

Purpose:
  Bridges the modern MCP 2025-06-18 server (mcp_server.py / FastMCP) into the
  existing QTCL Flask application so both the old JSON-RPC routes and the new
  MCP endpoints coexist on the same port / same gunicorn worker pool.

Architecture:
  ┌─────────────────────────────────────────┐
  │  Gunicorn (port 8000)                   │
  │  ┌─────────────┐  ┌──────────────────┐  │
  │  │ Flask App   │  │ MCP ASGI App     │  │
  │  │ /rpc        │  │ /mcp             │  │
  │  │ /mcp/health │  │ /mcp/sse (legacy)│  │
  │  └─────────────┘  └──────────────────┘  │
  │         both served by same process      │
  └─────────────────────────────────────────┘

Usage (in your existing server.py or wsgi_config.py):

    from flask import Flask
    from mcp_flask_adapter import register_mcp_routes

    app = Flask(__name__)
    register_mcp_routes(app, rpc_url="http://localhost:8000/rpc")

    # Your existing routes...
    # app.register_blueprint(api_blueprint)

    if __name__ == "__main__":
        app.run()

Endpoints Added:
  GET  /mcp         — MCP streamable HTTP (or JSON handshake for GET)
  POST /mcp         — MCP streamable HTTP primary endpoint
  GET  /mcp/sse     — Legacy SSE (backward compatible)
  POST /mcp/message — Legacy message endpoint
  GET  /mcp/health  — Combined health status

Dependencies:
  pip install mcp>=1.23.0 flask>=2.0 starlette>=0.27.0 uvicorn>=0.23.0

Note:
  The FastMCP streamable-http transport uses Starlette (ASGI). When mounted
  inside Flask (WSGI), we proxy requests through a small WSGI-ASGI bridge.
  For production at scale, run the MCP server standalone on a separate port
  or switch the entire app to an ASGI framework (FastAPI, Quart, etc.).
================================================================================
"""

from __future__ import annotations

import os
import sys
import json
import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────
QTCL_RPC_URL = os.environ.get("QTCL_RPC_URL", "http://localhost:8000/rpc")
MCP_PROTOCOL_VERSION = "2025-06-18"

# ── Lazy imports (fail gracefully if SDK not installed) ───────────────────────
try:
    from mcp.server.fastmcp import FastMCP
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    logger.warning("[MCP Adapter] Official MCP Python SDK not installed. "
                   "Legacy mode only. Run: pip install 'mcp>=1.23.0'")


# ═══════════════════════════════════════════════════════════════════════════════
# §1  Create Modern MCP Server Instance (reuses mcp_server.py logic)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_mcp_app(rpc_url: str) -> Optional[Any]:
    """Build a FastMCP application with all QTCL tools, resources, and prompts."""
    if not SDK_AVAILABLE:
        return None

    mcp = FastMCP(
        "qtcl-blockchain",
        stateless_http=True,
        json_response=True,
    )

    # ── Re-import tool implementations from mcp_server.py ──
    # We avoid circular imports by lazy-loading
    try:
        import mcp_server as _ms
    except Exception:
        _ms = None

    # If mcp_server module is importable, reuse its helpers. Otherwise define inline.
    if _ms is not None and hasattr(_ms, '_wallet_create'):
        _wallet_create = _ms._wallet_create
        _sign_message = _ms._sign_message
        qtcl_rpc = _ms.qtcl_rpc
    else:
        # Inline fallback (duplicated minimally for standalone use)
        import urllib.request
        _rpc_counter = [0]
        def qtcl_rpc(method: str, params=None) -> Any:
            _rpc_counter[0] += 1
            payload = {
                "jsonrpc": "2.0",
                "method": method,
                "params": params if params is not None else [],
                "id": _rpc_counter[0],
            }
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                rpc_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = json.loads(resp.read().decode())
                if "error" in body:
                    raise RuntimeError(body["error"].get("message", "RPC error"))
                return body.get("result", body)
            except Exception as e:
                raise RuntimeError(f"QTCL RPC '{method}' failed: {e}")

        async def _wallet_create(label: str = "") -> dict:
            return qtcl_rpc("qtcl_hyp_generateKeypair", {})

        async def _sign_message(message_hex: str, private_key: str) -> dict:
            return qtcl_rpc("qtcl_hyp_signMessage", {
                "message": message_hex,
                "private_key": private_key,
            })

    # ── Tools ───────────────────────────────────────────────────────────────

    @mcp.tool()
    async def qtcl_create_wallet(label: str = "") -> str:
        """Create a new QTCL wallet backed by a real HypΓ post-quantum keypair."""
        result = await _wallet_create(label)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    async def qtcl_sign_message(message_hex: str, private_key: str) -> str:
        """Sign a 32-byte message hash with a HypΓ private key using Schnorr-Γ."""
        if len(message_hex) != 64:
            raise ValueError(f"message_hex must be 64 hex chars; got {len(message_hex)}")
        result = await _sign_message(message_hex, private_key)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_balance(address: str) -> str:
        """Check QTCL balance for any address."""
        return json.dumps(qtcl_rpc("qtcl_getBalance", [address]), indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_utxos(address: str, limit: int = 1000) -> str:
        """List UTXOs for an address."""
        p = {"address": address}
        if limit:
            p["limit"] = int(limit)
        return json.dumps(qtcl_rpc("qtcl_getUTXOs", [p]), indent=2, default=str)

    @mcp.tool()
    async def qtcl_send_transaction(
        from_address: str, to_address: str, amount: float,
        memo: str = "", signature: str = "", public_key: str = "", nonce: int = 0
    ) -> str:
        """Submit a signed UTXO transaction. Flat fee: 1 qsat."""
        p = {"from_address": from_address, "to_address": to_address, "amount": amount}
        for k, v in (("memo", memo), ("signature", signature), ("public_key", public_key), ("nonce", nonce)):
            if v:
                p[k] = v
        return json.dumps(qtcl_rpc("qtcl_submitTransaction", [p]), indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_transaction(tx_hash: str) -> str:
        """Look up a transaction by hash."""
        return json.dumps(qtcl_rpc("qtcl_getTransaction", [tx_hash]), indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_chain_info() -> str:
        """Current blockchain state."""
        return json.dumps({
            "chain": qtcl_rpc("qtcl_getBlockHeight"),
            "mempool": qtcl_rpc("qtcl_getMempoolStats"),
            "health": qtcl_rpc("qtcl_getHealth"),
        }, indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_block(height: int = -1, hash: str = "") -> str:
        """Block by height or hash. Use height=0 for genesis. Omit both for latest block."""
        if hash:
            key = hash
        elif height >= 0:
            key = height
        else:
            tip = qtcl_rpc("qtcl_getBlockHeight")
            key = tip.get("height", 0) if isinstance(tip, dict) else tip
        return json.dumps(qtcl_rpc("qtcl_getBlock", [key]), indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_recent_transactions(address: str = "", per_page: int = 20) -> str:
        """Recent transactions, optionally filtered by address."""
        p = {"page": 0, "per_page": min(int(per_page), 50)}
        if address:
            p["address"] = address
        return json.dumps(qtcl_rpc("qtcl_getTransactions", p), indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_quantum_metrics() -> str:
        """Live quantum coherence metrics."""
        return json.dumps(qtcl_rpc("qtcl_getQuantumMetrics"), indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_oracle_registry(limit: int = 10) -> str:
        """List registered quantum oracles."""
        return json.dumps(qtcl_rpc("qtcl_getOracleRegistry", {"limit": limit}), indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_peers(limit: int = 20) -> str:
        """List active P2P peers."""
        return json.dumps(qtcl_rpc("qtcl_getPeers", [limit]), indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_price() -> str:
        """QTCL quantum coherence metrics (no public USD exchange)."""
        return json.dumps(qtcl_rpc("qtcl_getQuantumMetrics"), indent=2, default=str)

    # ── Resources ─────────────────────────────────────────────────────────────

    @mcp.resource("chain://height")
    async def get_block_height() -> str:
        return str(qtcl_rpc("qtcl_getBlockHeight"))

    @mcp.resource("chain://health")
    async def get_health() -> str:
        return json.dumps(qtcl_rpc("qtcl_getHealth"), indent=2, default=str)

    @mcp.resource("price://qtcl-quantum")
    async def get_qtcl_price() -> str:
        return str(qtcl_rpc("qtcl_getQuantumMetrics"))

    @mcp.resource("docs://capability")
    async def get_capability_doc() -> str:
        return json.dumps({
            "name": "QTCL — Quantum Temporal Coherence Ledger",
            "version": "3.0.0",
            "protocol": f"JSON-RPC 2.0 + MCP {MCP_PROTOCOL_VERSION}",
            "tools": 12,
            "resources": 4,
            "transports": ["streamable-http", "stdio"],
        }, indent=2)

    # ── Prompts ───────────────────────────────────────────────────────────────

    @mcp.prompt()
    def wallet_helper(task: str = "create") -> str:
        if task == "create":
            return (
                "You are helping a user create a QTCL post-quantum wallet.\n"
                "Call qtcl_create_wallet to generate a keypair, then securely present:\n"
                "  - address (64-char hex)\n"
                "  - public_key (long hex)\n"
                "  - private_key (critical: user must save this)\n"
                "Warn the user that the server does not retain private keys."
            )
        elif task == "send":
            return (
                "You are helping a user send QTCL. The workflow is:\n"
                "1. qtcl_get_balance(from_address) — check funds\n"
                "2. qtcl_get_utxos(from_address) — select inputs\n"
                "3. qtcl_sign_message(tx_hash, private_key) — authorize\n"
                "4. qtcl_send_transaction(...) — submit\n"
                "Flat fee is 1 qsat. ~18s finality."
            )
        return "How can I help you with QTCL today?"

    return mcp


# ═══════════════════════════════════════════════════════════════════════════════
# §2  WSGI-ASGI Bridge for Mounting inside Flask
# ═══════════════════════════════════════════════════════════════════════════════

class _ASGIBridge:
    """Minimal WSGI-to-ASGI bridge for running Starlette (MCP) inside Flask."""

    def __init__(self, asgi_app: Any):
        self.asgi_app = asgi_app

    def __call__(self, environ, start_response):
        import asyncio
        from io import BytesIO

        method = environ.get("REQUEST_METHOD", "GET")
        path = environ.get("PATH_INFO", "/")
        query = environ.get("QUERY_STRING", "")
        if query:
            path += "?" + query

        headers = []
        for key, value in environ.items():
            if key.startswith("HTTP_"):
                name = key[5:].replace("_", "-").title()
                headers.append((name.encode(), value.encode()))
            elif key in ("CONTENT_TYPE", "CONTENT_LENGTH"):
                name = key.replace("_", "-").title()
                headers.append((name.encode(), value.encode()))

        body = b""
        if method in ("POST", "PUT", "PATCH"):
            content_length = int(environ.get("CONTENT_LENGTH", 0) or 0)
            if content_length:
                body = environ["wsgi.input"].read(content_length)

        scope = {
            "type": "http",
            "method": method,
            "path": path,
            "raw_path": path.encode(),
            "query_string": query.encode(),
            "headers": headers,
            "scheme": environ.get("wsgi.url_scheme", "http"),
            "server": (environ.get("SERVER_NAME", "localhost"), int(environ.get("SERVER_PORT", 80))),
            "client": (environ.get("REMOTE_ADDR", "127.0.0.1"), int(environ.get("REMOTE_PORT", 0) or 0)),
        }

        response_started = False
        status_code = 200
        response_headers = []
        response_body = BytesIO()

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message):
            nonlocal response_started, status_code, response_headers
            if message["type"] == "http.response.start":
                response_started = True
                status_code = message["status"]
                response_headers = [(k.decode() if isinstance(k, bytes) else k,
                                       v.decode() if isinstance(v, bytes) else v)
                                      for k, v in message.get("headers", [])]
            elif message["type"] == "http.response.body":
                response_body.write(message.get("body", b""))

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.asgi_app(scope, receive, send))
        finally:
            loop.close()

        start_response(f"{status_code} OK", response_headers)
        return [response_body.getvalue()]


# ═══════════════════════════════════════════════════════════════════════════════
# §3  Flask Registration
# ═══════════════════════════════════════════════════════════════════════════════

def register_mcp_routes(app: Any, rpc_url: Optional[str] = None) -> bool:
    """
    Register modern MCP routes on an existing Flask application.

    Args:
        app:      Flask application instance
        rpc_url:  QTCL JSON-RPC backend URL (default: env QTCL_RPC_URL)

    Returns:
        True if modern MCP routes were registered, False if fallback legacy only.
    """
    rpc_url = rpc_url or os.environ.get("QTCL_RPC_URL", "http://localhost:8000/rpc")

    if not SDK_AVAILABLE:
        logger.warning("[MCP Adapter] MCP SDK unavailable — registering legacy routes only")
        _register_legacy_only(app)
        return False

    mcp = _make_mcp_app(rpc_url)
    if mcp is None:
        _register_legacy_only(app)
        return False

    # Extract the underlying Starlette app from FastMCP and wrap it
    try:
        starlette_app = mcp._mcp_server.app
    except AttributeError:
        # FastMCP internal structure may vary by version — fallback
        starlette_app = mcp.app if hasattr(mcp, "app") else None

    if starlette_app is None:
        logger.warning("[MCP Adapter] Could not extract ASGI app — legacy routes only")
        _register_legacy_only(app)
        return False

    bridge = _ASGIBridge(starlette_app)

    @app.route("/mcp", methods=["GET", "POST", "OPTIONS"])
    @app.route("/mcp/<path:path>", methods=["GET", "POST", "OPTIONS"])
    def mcp_streamable_http(path=""):
        from flask import request
        return bridge(request.environ, lambda status, headers: None)

    # Legacy SSE (backward compatible)
    _register_legacy_only(app)

    logger.info("[MCP Adapter] Modern MCP 2025-06-18 routes registered on /mcp")
    logger.info("[MCP Adapter] Legacy SSE preserved on /mcp/sse and /mcp/message")
    return True


def _register_legacy_only(app: Any):
    """Register only the legacy v2.x SSE routes (no FastMCP)."""
    import uuid, queue
    from flask import Response, request, jsonify

    _sessions: dict = {}
    _sessions_lock = __import__("threading").Lock()

    @app.route("/mcp", methods=["OPTIONS"])
    @app.route("/mcp/sse", methods=["OPTIONS"])
    def mcp_options():
        return Response("", status=204, headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Accept, Mcp-Session-Id, Authorization",
        })

    @app.route("/mcp/sse", methods=["GET"])
    def mcp_sse():
        sid = str(uuid.uuid4())
        q: queue.Queue = queue.Queue(maxsize=100)
        with _sessions_lock:
            _sessions[sid] = q

        def generate():
            yield f"event: endpoint\ndata: /mcp\n\n"
            try:
                while True:
                    try:
                        msg = q.get(timeout=25.0)
                        if msg is None:
                            break
                        yield f"event: message\ndata: {json.dumps(msg)}\n\n"
                    except queue.Empty:
                        yield ": heartbeat\n\n"
            except GeneratorExit:
                pass
            finally:
                with _sessions_lock:
                    _sessions.pop(sid, None)

        return Response(generate(), mimetype="text/event-stream", headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        })

    @app.route("/mcp/message", methods=["POST"])
    def mcp_message():
        sid = request.args.get("session_id", "")
        with _sessions_lock:
            q = _sessions.get(sid)
        if not q:
            return jsonify({"error": "Invalid or expired session"}), 400
        try:
            msg = request.get_json(force=True, silent=True)
            if not msg:
                return jsonify({"error": "No JSON body"}), 400
            return jsonify({"status": "queued", "session_id": sid}), 202
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/mcp/health", methods=["GET"])
    def mcp_health():
        return jsonify({
            "status": "ok",
            "server": "qtcl-blockchain",
            "version": "3.0.0",
            "protocol": MCP_PROTOCOL_VERSION,
            "transport": "streamable-http",
            "legacy_sse": True,
            "sdk": "mcp-python-sdk" if SDK_AVAILABLE else "not-installed",
            "tools": 12,
            "resources": 4,
            "prompts": 1,
            "note": "Legacy SSE deprecated — migrate to /mcp streamable HTTP",
        })

    logger.info("[MCP Adapter] Legacy SSE routes registered: /mcp/sse, /mcp/message, /mcp/health")


# ═══════════════════════════════════════════════════════════════════════════════
# §4  Standalone Test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from flask import Flask
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    app = Flask(__name__)
    ok = register_mcp_routes(app)

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  QTCL MCP Flask Adapter — Test Mode                          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Modern MCP: {'YES (FastMCP)' if ok else 'NO (legacy only)'}")
    print("  Endpoints:")
    print("    POST/GET /mcp          — Streamable HTTP (MCP 2025-06-18)")
    print("    GET      /mcp/sse      — Legacy SSE (deprecated)")
    print("    POST     /mcp/message  — Legacy message (deprecated)")
    print("    GET      /mcp/health   — Health check")
    print()
    print("  Starting Flask on http://127.0.0.1:8000 ...")
    app.run(host="127.0.0.1", port=8000, debug=False)
