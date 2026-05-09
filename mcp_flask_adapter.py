#!/usr/bin/env python3
"""
================================================================================
MCP FLASK ADAPTER v4.0 — QTCL MCP 2025-06-18 Compatibility Shim
================================================================================

ARCHITECTURE (v4.0 — pure Flask, zero ASGI):
  MCP 2025-06-18 is now implemented NATIVELY in server.py as plain Flask routes.
  This file is retained for:
    1. Backward-compatible import surface (register_mcp_routes still works)
    2. stdio transport entry point for Claude Desktop / Cursor / CLI agents
    3. Standalone test/dev mode

  The old _ASGIBridge / Starlette approach is removed — it was fundamentally
  incompatible with Gunicorn gthread workers (asyncio event loop conflicts) and
  caused silent failures under concurrent load.

Endpoints (all registered natively in server.py):
  POST   /mcp          — Streamable HTTP primary channel (MCP 2025-06-18)
  GET    /mcp          — Server info JSON (stateless handshake)
  OPTIONS /mcp         — CORS preflight (204)
  GET    /mcp/sse      — Legacy SSE (2024-11-05 backward compat)
  POST   /mcp/message  — Legacy message channel
  GET    /mcp/health   — Health / capability summary
  GET    /mcp/capability — Full agent capability document

Tool surface (13 tools, 4 resources, 1 prompt — all in server.py):
  qtcl_create_wallet, qtcl_sign_message, qtcl_get_balance, qtcl_get_utxos,
  qtcl_send_transaction, qtcl_get_transaction, qtcl_get_chain_info,
  qtcl_get_block, qtcl_get_recent_transactions, qtcl_get_quantum_metrics,
  qtcl_get_oracle_registry, qtcl_get_peers, qtcl_get_price

Dependencies:
  pip install flask>=2.0 mcp>=1.23.0 (optional, for stdio transport only)
================================================================================
"""

from __future__ import annotations

import os
import sys
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
QTCL_RPC_URL      = os.environ.get("QTCL_RPC_URL", "http://localhost:8000/rpc")
MCP_PROTOCOL_VERSION = "2025-06-18"
MCP_SERVER_NAME   = "qtcl-blockchain"
MCP_SERVER_VERSION = "3.0.0"

# ── Optional MCP SDK (only needed for stdio transport) ───────────────────────
try:
    from mcp.server.fastmcp import FastMCP as _FastMCP
    SDK_AVAILABLE = True
except ImportError:
    _FastMCP = None  # type: ignore
    SDK_AVAILABLE = False
    logger.debug("[MCP Adapter] MCP Python SDK not installed — stdio transport unavailable. "
                 "HTTP transport works without it. Run: pip install 'mcp>=1.23.0'")


# ═══════════════════════════════════════════════════════════════════════════════
# §1  register_mcp_routes — compatibility shim
#     In v4+, MCP routes are registered natively in server.py.
#     This function is a no-op when called after server.py is loaded,
#     but provides backward compatibility for any external code calling it.
# ═══════════════════════════════════════════════════════════════════════════════

def register_mcp_routes(app: Any, rpc_url: Optional[str] = None) -> bool:
    """
    Register MCP routes on a Flask application.

    In production (server.py), MCP routes are already registered natively.
    This function detects that and skips re-registration to avoid conflicts.
    In standalone/test mode, it registers a minimal native implementation.

    Returns True if routes were registered (or already present), False on error.
    """
    # Check if native MCP routes are already mounted on this app
    existing_rules = {rule.rule for rule in app.url_map.iter_rules()}
    if "/mcp" in existing_rules:
        logger.info("[MCP Adapter] Native MCP routes already registered — skipping re-registration")
        return True

    logger.info("[MCP Adapter] Registering MCP 2025-06-18 routes (standalone mode)...")
    return _register_native_mcp(app, rpc_url or QTCL_RPC_URL)


def _register_native_mcp(app: Any, rpc_url: str) -> bool:
    """
    Register MCP 2025-06-18 Streamable HTTP routes as pure Flask.
    Zero ASGI, zero asyncio, zero Starlette — runs on any WSGI server.
    """
    import urllib.request
    import threading
    import queue
    import uuid

    _rpc_id_counter = [0]
    _lock = threading.Lock()

    def _rpc(method: str, params: Any = None) -> Any:
        with _lock:
            _rpc_id_counter[0] += 1
            rid = _rpc_id_counter[0]
        payload = {"jsonrpc": "2.0", "method": method, "params": params or [], "id": rid}
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            rpc_url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = json.loads(resp.read().decode())
            if "error" in body:
                raise RuntimeError(body["error"].get("message", "RPC error"))
            return body.get("result", body)
        except Exception as e:
            raise RuntimeError(f"RPC '{method}' failed: {e}")

    _CORS = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS, DELETE",
        "Access-Control-Allow-Headers": "Content-Type, Accept, Mcp-Session-Id, Authorization, Last-Event-ID",
        "Access-Control-Expose-Headers": "Mcp-Session-Id",
    }

    _MCP_TOOLS_LOCAL = [
        {"name": "qtcl_get_balance",          "description": "Check QTCL balance for an address.", "inputSchema": {"type": "object", "properties": {"address": {"type": "string"}}, "required": ["address"]}},
        {"name": "qtcl_get_chain_info",        "description": "Current blockchain state.", "inputSchema": {"type": "object", "properties": {}}},
        {"name": "qtcl_get_quantum_metrics",   "description": "Live quantum coherence metrics.", "inputSchema": {"type": "object", "properties": {}}},
        {"name": "qtcl_get_recent_transactions","description": "Recent transactions.", "inputSchema": {"type": "object", "properties": {"address": {"type": "string"}, "per_page": {"type": "integer", "default": 20}}}},
        {"name": "qtcl_get_peers",             "description": "Active P2P peers.", "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer", "default": 20}}}},
        {"name": "qtcl_get_oracle_registry",   "description": "Registered quantum oracles.", "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer", "default": 10}}}},
    ]

    from flask import request as _request, Response as _Response, jsonify as _jsonify

    def _tool_dispatch(name: str, args: dict) -> dict:
        try:
            if name == "qtcl_get_balance":
                r = _rpc("qtcl_getBalance", [args.get("address", "")])
            elif name == "qtcl_get_chain_info":
                r = {"chain": _rpc("qtcl_getBlockHeight"), "mempool": _rpc("qtcl_getMempoolStats"), "health": _rpc("qtcl_getHealth")}
            elif name == "qtcl_get_quantum_metrics":
                r = _rpc("qtcl_getQuantumMetrics")
            elif name == "qtcl_get_recent_transactions":
                p = {"page": 0, "per_page": min(int(args.get("per_page", 20)), 50)}
                if args.get("address"):
                    p["address"] = args["address"]
                r = _rpc("qtcl_getTransactions", [p])
            elif name == "qtcl_get_peers":
                r = _rpc("qtcl_getPeers", [int(args.get("limit", 20))])
            elif name == "qtcl_get_oracle_registry":
                r = _rpc("qtcl_getOracleRegistry", [{"limit": int(args.get("limit", 10))}])
            else:
                return {"jsonrpc": "2.0", "error": {"code": -32601, "message": f"Unknown tool: {name}"}, "id": None}
            return {"jsonrpc": "2.0", "result": {"content": [{"type": "text", "text": json.dumps(r, indent=2, default=str)}]}, "id": None}
        except Exception as e:
            return {"jsonrpc": "2.0", "result": {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True}, "id": None}

    def _handle_msg(body_bytes: bytes, sid: Optional[str]):
        try:
            msg = json.loads(body_bytes)
        except Exception as e:
            return {"jsonrpc": "2.0", "error": {"code": -32700, "message": f"Parse error: {e}"}, "id": None}, None

        method = msg.get("method", "")
        params = msg.get("params", {})
        req_id = msg.get("id")
        new_sid = None

        if method == "initialize":
            new_sid = sid or str(uuid.uuid4())
            return {"jsonrpc": "2.0", "result": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "serverInfo": {"name": MCP_SERVER_NAME, "version": MCP_SERVER_VERSION},
                "capabilities": {"tools": {"listChanged": False}, "resources": {"subscribe": False}, "prompts": {"listChanged": False}, "logging": {}},
            }, "id": req_id}, new_sid

        if method == "notifications/initialized" or req_id is None:
            return None, None

        if method == "tools/list":
            return {"jsonrpc": "2.0", "result": {"tools": _MCP_TOOLS_LOCAL}, "id": req_id}, None

        if method == "tools/call":
            name = params.get("name", "") if isinstance(params, dict) else ""
            args = params.get("arguments", {}) if isinstance(params, dict) else {}
            r = _tool_dispatch(name, args or {})
            r["id"] = req_id
            return r, None

        if method == "ping":
            return {"jsonrpc": "2.0", "result": {}, "id": req_id}, None

        return {"jsonrpc": "2.0", "error": {"code": -32601, "message": f"Method not found: {method}"}, "id": req_id}, None

    @app.route("/mcp", methods=["OPTIONS"])
    @app.route("/mcp/<path:subpath>", methods=["OPTIONS"])
    def _mcp_opts(subpath=""):
        r = _Response("", status=204)
        for k, v in _CORS.items():
            r.headers[k] = v
        return r

    @app.route("/mcp", methods=["POST"])
    def _mcp_post():
        sid = _request.headers.get("Mcp-Session-Id", "").strip() or None
        body = _request.get_data()
        if not body:
            out = json.dumps({"jsonrpc": "2.0", "error": {"code": -32700, "message": "Empty body"}, "id": None}).encode()
            r = _Response(out, status=200, mimetype="application/json")
            for k, v in _CORS.items():
                r.headers[k] = v
            return r
        try:
            parsed = json.loads(body)
        except Exception as e:
            out = json.dumps({"jsonrpc": "2.0", "error": {"code": -32700, "message": str(e)}, "id": None}).encode()
            r = _Response(out, status=200, mimetype="application/json")
            for k, v in _CORS.items():
                r.headers[k] = v
            return r
        new_sid = sid
        if isinstance(parsed, list):
            responses = []
            for item in parsed:
                resp_obj, maybe = _handle_msg(json.dumps(item).encode(), new_sid)
                if maybe:
                    new_sid = maybe
                if resp_obj is not None:
                    responses.append(resp_obj)
            out = json.dumps(responses).encode()
        else:
            resp_obj, maybe = _handle_msg(body, sid)
            if maybe:
                new_sid = maybe
            out = json.dumps(resp_obj).encode() if resp_obj is not None else b"{}"
        r = _Response(out, status=200, mimetype="application/json")
        for k, v in _CORS.items():
            r.headers[k] = v
        if new_sid:
            r.headers["Mcp-Session-Id"] = new_sid
        return r

    @app.route("/mcp", methods=["GET"])
    def _mcp_get():
        info = {"jsonrpc": "2.0", "method": "server/info", "params": {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "serverInfo": {"name": MCP_SERVER_NAME, "version": MCP_SERVER_VERSION},
            "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
            "transport": "streamable-http",
        }}
        r = _Response(json.dumps(info).encode(), status=200, mimetype="application/json")
        for k, v in _CORS.items():
            r.headers[k] = v
        return r

    _sessions_local: dict = {}
    _sessions_lock_local = threading.RLock()

    @app.route("/mcp/sse", methods=["GET"])
    def _mcp_sse():
        sid = str(uuid.uuid4())
        q: queue.Queue = queue.Queue(maxsize=100)
        with _sessions_lock_local:
            _sessions_local[sid] = q

        def gen():
            yield f"event: endpoint\ndata: /mcp?session_id={sid}\n\n"
            try:
                while True:
                    try:
                        msg = q.get(timeout=20.0)
                        if msg is None:
                            break
                        yield f"event: message\ndata: {json.dumps(msg, default=str)}\n\n"
                    except queue.Empty:
                        yield ": heartbeat\n\n"
            finally:
                with _sessions_lock_local:
                    _sessions_local.pop(sid, None)

        r = _Response(gen(), mimetype="text/event-stream")
        r.headers.update({"Cache-Control": "no-cache", "Connection": "keep-alive",
                          "X-Accel-Buffering": "no", "Mcp-Session-Id": sid})
        for k, v in _CORS.items():
            r.headers[k] = v
        return r

    @app.route("/mcp/message", methods=["POST"])
    def _mcp_msg():
        sid = _request.args.get("session_id", "")
        with _sessions_lock_local:
            q = _sessions_local.get(sid)
        if not q:
            return _jsonify({"error": "Invalid session"}), 404
        body = _request.get_data()
        if not body:
            return _jsonify({"error": "Empty body"}), 400
        resp_obj, _ = _handle_msg(body, sid)
        if resp_obj:
            try:
                q.put_nowait(resp_obj)
            except queue.Full:
                return _jsonify({"error": "Queue full"}), 429
        return _jsonify({"status": "accepted"}), 202

    @app.route("/mcp/health", methods=["GET"])
    def _mcp_health():
        out = json.dumps({
            "status": "ok", "server": MCP_SERVER_NAME, "version": MCP_SERVER_VERSION,
            "protocol": MCP_PROTOCOL_VERSION, "transport": "streamable-http",
            "tools": len(_MCP_TOOLS_LOCAL), "rpc_url": rpc_url, "ts": __import__("time").time(),
        }).encode()
        r = _Response(out, status=200, mimetype="application/json")
        for k, v in _CORS.items():
            r.headers[k] = v
        return r

    logger.info(f"[MCP Adapter] ✅ Native MCP 2025-06-18 routes registered (standalone mode, rpc_url={rpc_url})")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# §2  stdio transport — for Claude Desktop / Cursor / CLI agents
# ═══════════════════════════════════════════════════════════════════════════════

def run_stdio_transport(rpc_url: str = QTCL_RPC_URL) -> None:
    """
    Run MCP server over stdio (for Claude Desktop, Cursor, Windsurf, CLI).

    Requires: pip install 'mcp>=1.23.0'
    Usage: python mcp_flask_adapter.py --transport stdio
    """
    if not SDK_AVAILABLE or _FastMCP is None:
        print("[MCP] ERROR: MCP Python SDK not installed.", file=sys.stderr)
        print("[MCP]        Run: pip install 'mcp>=1.23.0'", file=sys.stderr)
        sys.exit(1)

    import urllib.request

    _counter = [0]

    def _rpc(method: str, params: Any = None) -> Any:
        _counter[0] += 1
        payload = {"jsonrpc": "2.0", "method": method, "params": params or [], "id": _counter[0]}
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            rpc_url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode())
        if "error" in body:
            raise RuntimeError(body["error"].get("message", "RPC error"))
        return body.get("result", body)

    mcp = _FastMCP("qtcl-blockchain")

    @mcp.tool()
    async def qtcl_get_balance(address: str) -> str:
        """Check QTCL balance for any address."""
        return json.dumps(_rpc("qtcl_getBalance", [address]), indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_chain_info() -> str:
        """Current blockchain state: height, mempool, health."""
        return json.dumps({
            "chain": _rpc("qtcl_getBlockHeight"),
            "mempool": _rpc("qtcl_getMempoolStats"),
            "health": _rpc("qtcl_getHealth"),
        }, indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_quantum_metrics() -> str:
        """Live quantum coherence metrics."""
        return json.dumps(_rpc("qtcl_getQuantumMetrics"), indent=2, default=str)

    @mcp.tool()
    async def qtcl_send_transaction(from_address: str, to_address: str, amount: float,
                                     memo: str = "", signature: str = "", public_key: str = "") -> str:
        """Submit a signed UTXO transaction. Flat fee: 1 qsat."""
        p = {"from_address": from_address, "to_address": to_address, "amount": amount}
        for k, v in (("memo", memo), ("signature", signature), ("public_key", public_key)):
            if v:
                p[k] = v
        return json.dumps(_rpc("qtcl_submitTransaction", [p]), indent=2, default=str)

    @mcp.tool()
    async def qtcl_create_wallet(label: str = "") -> str:
        """Create a new post-quantum QTCL wallet."""
        return json.dumps(_rpc("qtcl_hyp_generateKeypair", []), indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_block(height: int = -1, hash: str = "") -> str:
        """Get block by height or hash. Omit both for latest."""
        if hash:
            key: Any = hash
        elif height >= 0:
            key = height
        else:
            tip = _rpc("qtcl_getBlockHeight")
            key = tip.get("height", 0) if isinstance(tip, dict) else 0
        return json.dumps(_rpc("qtcl_getBlock", [key]), indent=2, default=str)

    @mcp.resource("chain://height")
    async def res_height() -> str:
        return str(_rpc("qtcl_getBlockHeight"))

    @mcp.resource("chain://health")
    async def res_health() -> str:
        return json.dumps(_rpc("qtcl_getHealth"), indent=2, default=str)

    @mcp.prompt()
    def wallet_helper(task: str = "create") -> str:
        if task == "send":
            return ("QTCL send: 1) qtcl_get_balance 2) qtcl_get_utxos 3) qtcl_sign_message "
                    "4) qtcl_send_transaction. Fee: 1 qsat. Finality: ~18s.")
        return ("Create QTCL wallet: call qtcl_create_wallet → save private_key locally "
                "(server never retains it). Cryptography: Schnorr-Γ post-quantum.")

    print(f"[MCP] Starting stdio transport → {rpc_url}", file=sys.stderr)
    mcp.run(transport="stdio")


# ═══════════════════════════════════════════════════════════════════════════════
# §3  Standalone entrypoint
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description="QTCL MCP Adapter")
    ap.add_argument("--transport", choices=["stdio", "http"], default="http")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--rpc-url", default=QTCL_RPC_URL)
    args = ap.parse_args()

    if args.transport == "stdio":
        run_stdio_transport(args.rpc_url)
    else:
        from flask import Flask
        _app = Flask(__name__)
        register_mcp_routes(_app, args.rpc_url)

        print()
        print("╔══════════════════════════════════════════════════════════════════╗")
        print("║  QTCL MCP Adapter v4.0 — HTTP Standalone Mode                    ║")
        print("║  MCP 2025-06-18 Streamable HTTP | Pure Flask | Zero ASGI          ║")
        print("╚══════════════════════════════════════════════════════════════════╝")
        print(f"  RPC backend: {args.rpc_url}")
        print(f"  Endpoints:")
        print(f"    POST/GET /mcp          — Streamable HTTP (MCP 2025-06-18)")
        print(f"    GET      /mcp/sse      — Legacy SSE (2024-11-05)")
        print(f"    POST     /mcp/message  — Legacy message channel")
        print(f"    GET      /mcp/health   — Health check")
        print()
        _app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
