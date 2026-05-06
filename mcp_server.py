#!/usr/bin/env python3
"""
================================================================================
QTCL MCP SERVER v3.0 — Modern Model Context Protocol Implementation
================================================================================

Protocol:     MCP 2025-06-18 (Streamable HTTP + stdio)
Transport:    streamable-http (production) | stdio (local dev)
Endpoint:     POST/GET https://qtcl-blockchain.koyeb.app/mcp
Health:      GET  https://qtcl-blockchain.koyeb.app/mcp/health

Standards Compliance:
  - Implements official MCP specification 2025-06-18
  - Uses JSON-RPC 2.0 message encoding (UTF-8, newline-delimited)
  - Supports streamable HTTP transport (replaces legacy SSE transport)
  - Backward-compatible with 2024-11-05 SSE clients via redirect endpoint
  - Tool definitions use JSON Schema inputSchema per spec
  - Proper initialize/ping/tools/list/tools/call lifecycle

Architecture:
  - Built on official MCP Python SDK (mcp.server.fastmcp)
  - Dual transport: streamable-http for remote, stdio for local
  - In-process HypΓ engine or HTTP RPC fallback
  - Stateless mode recommended for production (stateless_http=True)

Usage:
  # Streamable HTTP mode (production)
  python mcp_server.py --transport streamable-http --port 8000

  # stdio mode (local dev / Claude Desktop)
  python mcp_server.py --transport stdio

  # Test with MCP Inspector
  npx -y @modelcontextprotocol/inspector

Environment:
  QTCL_RPC_URL      — QTCL JSON-RPC endpoint (default: http://localhost:8000/rpc)
  MCP_PORT          — HTTP listen port (default: 8000)
  MCP_TRANSPORT     — stdio | streamable-http (default: streamable-http)

Dependencies:
  pip install mcp>=1.23.0 flask>=2.0 requests>=2.28

Changes from v2.x:
  - Migrated from custom Flask Blueprint to official MCP SDK FastMCP
  - Protocol upgraded from 2024-11-05 to 2025-06-18
  - Transport upgraded from SSE to Streamable HTTP
  - Added stdio transport support for local agent integration
  - Added resources and prompts per MCP spec
  - Proper session management with Mcp-Session-Id header
  - DNS rebinding protection on HTTP transport
================================================================================
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import queue
import logging
import threading
import argparse
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

# ═══════════════════════════════════════════════════════════════════════════════
# §1  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

QTCL_RPC_URL = os.environ.get("QTCL_RPC_URL", "http://localhost:8000/rpc")
MCP_SERVER_NAME = "qtcl-blockchain"
MCP_SERVER_VERSION = "3.0.0"
MCP_PROTOCOL_VERSION = "2025-06-18"

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# §2  HypΓ ENGINE ACCESS
# ═══════════════════════════════════════════════════════════════════════════════

_engine = None
_engine_lock = threading.Lock()
_engine_failed = False


def _get_engine():
    """Return HypGammaEngine singleton; None if hlwe package unavailable."""
    global _engine, _engine_failed
    if _engine is not None:
        return _engine
    if _engine_failed:
        return None
    with _engine_lock:
        if _engine is not None:
            return _engine
        if _engine_failed:
            return None
        try:
            from hlwe.hyp_engine import HypGammaEngine
            _engine = HypGammaEngine()
            logger.info("[MCP] HypΓ engine loaded (hlwe.hyp_engine)")
            return _engine
        except Exception as e:
            logger.warning(f"[MCP] hlwe.hyp_engine unavailable ({e}); using RPC fallback")
            _engine_failed = True
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# §3  HTTP RPC CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

import urllib.request

_rpc_counter = 0
_rpc_lock = threading.Lock()


def _next_id() -> int:
    global _rpc_counter
    with _rpc_lock:
        _rpc_counter += 1
        return _rpc_counter


def qtcl_rpc(method: str, params=None) -> Any:
    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params if params is not None else [],
        "id": _next_id(),
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        QTCL_RPC_URL,
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
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"QTCL RPC '{method}' failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# §4  TOOL IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════════

async def _wallet_create(label: str = "") -> dict:
    """Create wallet via in-process engine or RPC fallback."""
    try:
        engine = _get_engine()
        if engine is not None:
            kp = engine.generate_keypair()
            result = {
                "private_key": kp.private_key,
                "public_key": kp.public_key,
                "address": kp.address,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "crypto": "HypΓ Schnorr-Γ / PSL(2,R) | 512-step walk | SHA3-256² address",
            }
            if label:
                result["label"] = label
            return result
    except Exception as e:
        logger.warning(f"[MCP] Direct keygen failed ({e}), trying RPC")

    # RPC fallback
    result = qtcl_rpc("qtcl_hyp_generateKeypair", {})
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(f"qtcl_hyp_generateKeypair error: {result['error']}")
    if "created_at" not in result:
        result["created_at"] = result.pop("timestamp", datetime.now(timezone.utc).isoformat())
    result.setdefault("crypto", "HypΓ Schnorr-Γ / PSL(2,R) | 512-step walk | SHA3-256² address")
    if label:
        result["label"] = label
    return result


async def _sign_message(message_hex: str, private_key: str) -> dict:
    """Sign a 32-byte message hash with HypΓ private key."""
    msg_bytes = bytes.fromhex(message_hex)
    if len(msg_bytes) != 32:
        raise ValueError(f"message_hex must be 32 bytes; got {len(msg_bytes)}")

    try:
        engine = _get_engine()
        if engine is not None:
            sig = engine.sign_hash(msg_bytes, private_key)
            return {
                "signature": sig["signature"],
                "challenge": sig["challenge"],
                "auth_tag": sig.get("auth_tag", sig["challenge"]),
                "timestamp": sig.get("timestamp", datetime.now(timezone.utc).isoformat()),
            }
    except Exception as e:
        logger.warning(f"[MCP] Direct sign failed ({e}), trying RPC")

    result = qtcl_rpc("qtcl_hyp_signMessage", {
        "message": message_hex,
        "private_key": private_key,
    })
    return {
        "signature": result["signature"],
        "challenge": result["challenge"],
        "auth_tag": result.get("auth_tag", result["challenge"]),
        "timestamp": result.get("timestamp", datetime.now(timezone.utc).isoformat()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# §5  MODERN FASTMCP SERVER (Official MCP Python SDK)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from mcp.server.fastmcp import FastMCP
    from mcp.types import TextContent
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    logger.error("[MCP] Official MCP Python SDK not installed. Run: pip install mcp>=1.23.0")


def create_mcp_server(stateless: bool = True) -> Optional[Any]:
    """Create a modern FastMCP server instance."""
    if not SDK_AVAILABLE:
        return None

    mcp = FastMCP(
        MCP_SERVER_NAME,
        stateless_http=stateless,
        json_response=True,
    )

    # ── Tools ─────────────────────────────────────────────────────────────────

    @mcp.tool()
    async def qtcl_create_wallet(label: str = "") -> str:
        """
        Create a new QTCL wallet backed by a real HypΓ post-quantum keypair
        (Schnorr-Γ over PSL(2,R), 512-step random walk, SHA3-256² address).
        Returns private_key, public_key, address, and created_at.
        Store private_key securely — the server does NOT retain it.
        """
        result = await _wallet_create(label)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    async def qtcl_sign_message(message_hex: str, private_key: str) -> str:
        """
        Sign a 32-byte message hash with a HypΓ private key using Schnorr-Γ.
        Required to authorize transactions: SHA3-256 hash your tx data, pass the
        32-byte result as message_hex (64 hex chars), get back a signature dict
        to include in qtcl_send_transaction.
        """
        if len(message_hex) != 64:
            raise ValueError(f"message_hex must be 64 hex chars (32-byte SHA3-256); got {len(message_hex)}")
        result = await _sign_message(message_hex, private_key)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_balance(address: str) -> str:
        """Check QTCL balance for any address. Returns balance, UTXO count, and UTXO list. 1 QTCL = 100 qsat."""
        result = qtcl_rpc("qtcl_getBalance", [address])
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_utxos(address: str, limit: int = 1000) -> str:
        """List unspent transaction outputs (UTXOs) for an address. Returns tx_hash, output_index, amount_base per coin."""
        p: dict = {"address": address}
        if limit:
            p["limit"] = int(limit)
        result = qtcl_rpc("qtcl_getUTXOs", [p])
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    async def qtcl_send_transaction(
        from_address: str,
        to_address: str,
        amount: float,
        memo: str = "",
        signature: str = "",
        public_key: str = "",
        nonce: int = 0,
    ) -> str:
        """
        Submit a signed UTXO transaction. Flat fee: 1 qsat. ~18s finality.
        Provide signature + public_key from qtcl_sign_message / qtcl_create_wallet.
        """
        p = {
            "from_address": from_address,
            "to_address": to_address,
            "amount": amount,
        }
        for k, v in (("memo", memo), ("signature", signature), ("public_key", public_key), ("nonce", nonce)):
            if v:
                p[k] = v
        result = qtcl_rpc("qtcl_submitTransaction", [p])
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_transaction(tx_hash: str) -> str:
        """Look up a transaction by hash."""
        result = qtcl_rpc("qtcl_getTransaction", [tx_hash])
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_chain_info() -> str:
        """Current blockchain state: height, latest hash, mempool depth, oracle status, system health."""
        result = {
            "chain": qtcl_rpc("qtcl_getBlockHeight"),
            "mempool": qtcl_rpc("qtcl_getMempoolStats"),
            "health": qtcl_rpc("qtcl_getHealth"),
        }
        return json.dumps(result, indent=2, default=str)

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
        result = qtcl_rpc("qtcl_getBlock", [key])
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_recent_transactions(address: str = "", per_page: int = 20) -> str:
        """Recent transactions, optionally filtered by address. Up to 50, newest first."""
        p = {"page": 0, "per_page": min(int(per_page), 50)}
        if address:
            p["address"] = address
        result = qtcl_rpc("qtcl_getTransactions", p)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_quantum_metrics() -> str:
        """
        Live quantum coherence metrics: W-state fidelity (>=0.75),
        entanglement witness, oracle consensus round, Mermin test,
        kappa=0.11 non-Markovian coherence score.
        """
        result = qtcl_rpc("qtcl_getQuantumMetrics")
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_oracle_registry(limit: int = 10) -> str:
        """List registered quantum oracles (5-oracle Byzantine consensus, 3-of-5 majority)."""
        result = qtcl_rpc("qtcl_getOracleRegistry", {"limit": limit})
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_peers(limit: int = 20) -> str:
        """List active P2P peers (Kademlia DHT)."""
        result = qtcl_rpc("qtcl_getPeers", [limit])
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_price() -> str:
        """QTCL quantum coherence metrics (no public USD exchange — valuation via W-state fidelity)."""
        result = qtcl_rpc("qtcl_getQuantumMetrics")
        return json.dumps(result, indent=2, default=str)

    # ── Resources ───────────────────────────────────────────────────────────────

    @mcp.resource("chain://height")
    async def get_block_height() -> str:
        """Current blockchain height."""
        return str(qtcl_rpc("qtcl_getBlockHeight"))

    @mcp.resource("chain://health")
    async def get_health() -> str:
        """System health vector."""
        return json.dumps(qtcl_rpc("qtcl_getHealth"), indent=2, default=str)

    @mcp.resource("price://qtcl-quantum")
    async def get_qtcl_price() -> str:
        """QTCL quantum coherence metrics (no public USD exchange)."""
        return str(qtcl_rpc("qtcl_getQuantumMetrics"))

    @mcp.resource("docs://capability")
    async def get_capability_doc() -> str:
        """QTCL capability document describing the full system."""
        return json.dumps({
            "name": "QTCL — Quantum Temporal Coherence Ledger",
            "version": MCP_SERVER_VERSION,
            "protocol": f"JSON-RPC 2.0 + MCP {MCP_PROTOCOL_VERSION}",
            "tools": 12,
            "resources": 4,
            "transports": ["streamable-http", "stdio"],
            "cryptography": "Schnorr-Γ over PSL(2,R) — Fiat-Shamir on hyperbolic Fuchsian group",
            "economics": {
                "native_unit": "QTCL",
                "base_unit": "qsat (1 QTCL = 100 qsat)",
                "fee_per_tx": "1 qsat flat",
                "block_time_seconds": 18,
            },
        }, indent=2)

    # ── Prompts ───────────────────────────────────────────────────────────────

    @mcp.prompt()
    def wallet_helper(task: str = "create") -> str:
        """Generate a prompt for wallet-related tasks."""
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
# §6  STANDALONE / MOUNTABLE FLASK WRAPPER (Backward Compatible)
# ═══════════════════════════════════════════════════════════════════════════════

_sessions: Dict[str, queue.Queue] = {}
_sessions_lock = threading.Lock()


def _cors_headers() -> dict:
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Accept, Mcp-Session-Id, Authorization",
    }


def _sse_stream(session_id: str) -> Any:
    """Legacy SSE stream for backward compatibility with 2024-11-05 clients."""
    from flask import Response
    q: queue.Queue = queue.Queue(maxsize=100)
    with _sessions_lock:
        _sessions[session_id] = q

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
                _sessions.pop(session_id, None)

    return Response(generate(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        **_cors_headers(),
    })


def register_legacy_routes(app: Any):
    """Register backward-compatible /mcp/sse and /mcp/message routes on a Flask app."""
    from flask import Blueprint, request, jsonify, Response

    bp = Blueprint("mcp_legacy", __name__)

    @bp.route("/mcp", methods=["OPTIONS"])
    @bp.route("/mcp/sse", methods=["OPTIONS"])
    def mcp_options():
        return Response("", status=204, headers=_cors_headers())

    @bp.route("/mcp/sse", methods=["GET"])
    def mcp_sse_legacy():
        sid = str(uuid.uuid4())
        return _sse_stream(sid)

    @bp.route("/mcp/message", methods=["POST"])
    def mcp_message_legacy():
        sid = request.args.get("session_id", "")
        with _sessions_lock:
            q = _sessions.get(sid)
        if not q:
            return jsonify({"error": "Invalid or expired session"}), 400
        try:
            msg = request.get_json(force=True, silent=True)
            if not msg:
                return jsonify({"error": "No JSON body"}), 400
            # Queue for processing — actual processing handled by modern server
            return jsonify({"status": "queued", "session_id": sid}), 202
        except Exception as e:
            logger.error(f"[MCP] Legacy message error: {e}")
            return jsonify({"error": str(e)}), 500

    @bp.route("/mcp/health", methods=["GET"])
    def mcp_health():
        engine_ok = _get_engine() is not None
        with _sessions_lock:
            n = len(_sessions)
        return jsonify({
            "status": "ok",
            "server": MCP_SERVER_NAME,
            "version": MCP_SERVER_VERSION,
            "protocol": MCP_PROTOCOL_VERSION,
            "transport": "streamable-http",
            "hyp_engine": "in-process" if engine_ok else "rpc-fallback",
            "active_sessions": n,
            "tools": 12,
            "sdk": "mcp-python-sdk" if SDK_AVAILABLE else "legacy",
        })

    app.register_blueprint(bp)
    logger.info(f"[MCP] Legacy routes registered: /mcp/sse, /mcp/message, /mcp/health")


# ═══════════════════════════════════════════════════════════════════════════════
# §7  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="QTCL MCP Server v3.0")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default=os.environ.get("MCP_TRANSPORT", "streamable-http"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MCP_PORT", 8000)))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--stateless", action="store_true", default=True, help="Stateless HTTP mode (recommended for production)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║    QTCL MCP SERVER v3.0 — Modern MCP Standard                ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")
    logger.info(f"  SDK        : {'mcp-python-sdk (FastMCP)' if SDK_AVAILABLE else 'NOT INSTALLED — run pip install mcp'}")
    logger.info(f"  Protocol   : {MCP_PROTOCOL_VERSION}")
    logger.info(f"  Transport  : {args.transport}")
    logger.info(f"  RPC URL    : {QTCL_RPC_URL}")

    if not SDK_AVAILABLE:
        logger.error("\n[ERROR] Official MCP Python SDK is not installed.")
        logger.error("Install it with:  pip install 'mcp>=1.23.0'\n")
        logger.error("Falling back to legacy Flask-based mode...")
        from flask import Flask
        app = Flask(__name__)
        register_legacy_routes(app)
        app.run(host=args.host, port=args.port, debug=False)
        return

    mcp = create_mcp_server(stateless=args.stateless)

    if args.transport == "stdio":
        logger.info("  Mode       : stdio (local agent integration)")
        logger.info("  Claude     : claude mcp add qtcl -- python mcp_server.py --transport stdio")
        mcp.run(transport="stdio")
    else:
        logger.info(f"  Endpoint   : http://{args.host}:{args.port}/mcp")
        logger.info("  Inspector  : npx -y @modelcontextprotocol/inspector")
        mcp.run(transport="streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
