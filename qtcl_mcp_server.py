#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               QTCL MCP SERVER v2.1 — Agent-Native Blockchain Access        ║
║                                                                            ║
║  Transport:  Streamable HTTP (MCP 2024-11-05 / 2025-03-26 compatible)      ║
║  Endpoint:   POST  https://qtcl-blockchain.koyeb.app/mcp                   ║
║  Health:     GET   https://qtcl-blockchain.koyeb.app/mcp/health            ║
║                                                                            ║
║  Key generation: calls hlwe.hyp_engine.HypGammaEngine directly (in-proc)  ║
║  or falls back to qtcl_hyp_generateKeypair RPC if not in server context.   ║
║                                                                            ║
║  Changes from v2.0:                                                        ║
║    - qtcl_create_wallet: calls real HypGammaEngine from hlwe package,      ║
║      NOT random hashes. Falls back to HTTP RPC if engine unavailable.      ║
║    - qtcl_sign_message: NEW tool — agents can sign arbitrary messages       ║
║      with a HypΓ private key (needed to authorize transactions).           ║
║    - Fixed kp.timestamp bug: HypKeyPair NamedTuple has no timestamp field, ║
║      we inject created_at ourselves from time.time().                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import json
import time
import uuid
import queue
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from flask import Flask, Response, request, jsonify

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# §1  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

QTCL_RPC_URL         = os.environ.get("QTCL_RPC_URL", "http://localhost:8000/rpc")
MCP_SERVER_NAME      = "qtcl-blockchain"
MCP_SERVER_VERSION   = "2.1.0"
MCP_PROTOCOL_VERSION = "2024-11-05"

# ═══════════════════════════════════════════════════════════════════════════════
# §2  HypΓ ENGINE ACCESS
#
#  Priority:
#    1. Direct in-process: from hlwe.hyp_engine import HypGammaEngine
#       (always available when running as a Blueprint on server.py)
#    2. HTTP RPC fallback: qtcl_hyp_generateKeypair / qtcl_hyp_signMessage
#       (used if standalone or if hlwe package unavailable)
# ═══════════════════════════════════════════════════════════════════════════════

_engine        = None
_engine_lock   = threading.Lock()
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
            from hlwe.hyp_engine import HypGammaEngine  # package path on Koyeb
            _engine = HypGammaEngine()
            logger.info("[MCP] ✅ HypΓ engine loaded (hlwe.hyp_engine)")
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
_rpc_lock    = threading.Lock()


def _next_id() -> int:
    global _rpc_counter
    with _rpc_lock:
        _rpc_counter += 1
        return _rpc_counter


def qtcl_rpc(method: str, params=None) -> Any:
    payload = {
        "jsonrpc": "2.0",
        "method":  method,
        "params":  params if params is not None else [],
        "id":      _next_id(),
    }
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
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
# §4  TOOL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

MCP_TOOLS: List[Dict] = [

    {
        "name": "qtcl_create_wallet",
        "description": (
            "Create a new QTCL wallet backed by a real HypΓ post-quantum keypair "
            "(Schnorr-Γ over PSL(2,R), 512-step random walk, SHA3-256² address). "
            "Returns private_key, public_key, address, and created_at. "
            "Store private_key securely — the server does NOT retain it."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "description": "Optional label (e.g. 'agent-payments')"}
            },
            "required": []
        }
    },
    {
        "name": "qtcl_sign_message",
        "description": (
            "Sign a 32-byte message hash with a HypΓ private key using Schnorr-Γ. "
            "Required to authorize transactions: SHA3-256 hash your tx data, pass the "
            "32-byte result as message_hex (64 hex chars), get back a signature dict "
            "to include in qtcl_send_transaction."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "message_hex": {
                    "type": "string",
                    "description": "SHA3-256 hash of the message, hex-encoded (exactly 64 hex chars = 32 bytes)"
                },
                "private_key": {
                    "type": "string",
                    "description": "HypΓ private key from qtcl_create_wallet"
                }
            },
            "required": ["message_hex", "private_key"]
        }
    },
    {
        "name": "qtcl_get_balance",
        "description": "Check QTCL balance for any address. Returns balance, UTXO count, and UTXO list. 1 QTCL = 100 qsat.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "address": {"type": "string", "description": "64-char hex QTCL address"}
            },
            "required": ["address"]
        }
    },
    {
        "name": "qtcl_get_utxos",
        "description": "List unspent transaction outputs (UTXOs) for an address. Returns tx_hash, output_index, amount_base per coin.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "address": {"type": "string", "description": "64-char hex QTCL address"},
                "limit":   {"type": "integer", "description": "Max UTXOs (default 1000)"}
            },
            "required": ["address"]
        }
    },
    {
        "name": "qtcl_send_transaction",
        "description": (
            "Submit a signed UTXO transaction. Flat fee: 1 qsat. ~18s finality. "
            "Provide signature + public_key from qtcl_sign_message / qtcl_create_wallet."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "from_address": {"type": "string"},
                "to_address":   {"type": "string"},
                "amount":       {"type": "number",  "description": "Amount in QTCL"},
                "memo":         {"type": "string",  "description": "Optional memo, max 256 chars"},
                "signature":    {"type": "string",  "description": "Schnorr-Γ signature hex"},
                "public_key":   {"type": "string",  "description": "HypΓ public key hex"},
                "nonce":        {"type": "integer"}
            },
            "required": ["from_address", "to_address", "amount"]
        }
    },
    {
        "name": "qtcl_get_transaction",
        "description": "Look up a transaction by hash.",
        "inputSchema": {
            "type": "object",
            "properties": {"tx_hash": {"type": "string"}},
            "required": ["tx_hash"]
        }
    },
    {
        "name": "qtcl_get_chain_info",
        "description": "Current blockchain state: height, latest hash, mempool depth, oracle status, system health.",
        "inputSchema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "qtcl_get_block",
        "description": "Block by height or hash. Omit both params for latest block.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "height": {"type": "integer"},
                "hash":   {"type": "string"}
            },
            "required": []
        }
    },
    {
        "name": "qtcl_get_recent_transactions",
        "description": "Recent transactions, optionally filtered by address. Up to 50, newest first.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "address":  {"type": "string"},
                "per_page": {"type": "integer", "description": "Max results 1-50 (default 20)"}
            },
            "required": []
        }
    },
    {
        "name": "qtcl_get_quantum_metrics",
        "description": (
            "Live quantum coherence metrics: W-state fidelity (>=0.75), "
            "entanglement witness, oracle consensus round, Mermin test, "
            "kappa=0.11 non-Markovian coherence score."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "qtcl_get_oracle_registry",
        "description": "List registered quantum oracles (5-oracle Byzantine consensus, 3-of-5 majority).",
        "inputSchema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "description": "Max results (default 10)"}},
            "required": []
        }
    },
    {
        "name": "qtcl_get_peers",
        "description": "List active P2P peers (Kademlia DHT).",
        "inputSchema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "description": "Max peers (default 20)"}},
            "required": []
        }
    },
    {
        "name": "qtcl_get_price",
        "description": "QTCL/USD price from Pyth Network oracle feed.",
        "inputSchema": {"type": "object", "properties": {}, "required": []}
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# §5  TOOL HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

def _wallet_create_direct(label: str) -> dict:
    """
    Generate real HypΓ keypair using the in-process engine.

    HypKeyPair = NamedTuple(private_key, public_key, address)
    Note: no timestamp field — server.py has a bug accessing kp.timestamp.
    We inject created_at ourselves.
    """
    engine = _get_engine()
    if engine is None:
        raise RuntimeError("HypΓ engine not available in-process")

    kp = engine.generate_keypair()
    # kp.private_key : 512-char string of digits 0-3 (walk indices)
    # kp.public_key  : hex-encoded PSL(2,R) matrix (~1200+ chars)
    # kp.address     : 64-char hex SHA3-256^2 of public key bytes
    result: dict = {
        "private_key": kp.private_key,
        "public_key":  kp.public_key,
        "address":     kp.address,
        "created_at":  datetime.now(timezone.utc).isoformat(),
        "crypto":      "HypΓ Schnorr-Γ / PSL(2,R) | 512-step walk | SHA3-256² address",
    }
    if label:
        result["label"] = label
    return result


def _wallet_create_rpc(label: str) -> dict:
    """
    Generate keypair via HTTP RPC.

    server.py's qtcl_hyp_generateKeypair tries kp.timestamp which raises
    AttributeError (HypKeyPair has no timestamp field). If the bug is still
    present, the RPC will return an error — we surface it clearly.
    """
    result = qtcl_rpc("qtcl_hyp_generateKeypair", {})
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(f"qtcl_hyp_generateKeypair error: {result['error']}")
    # Normalise timestamp field: server may or may not have the bug fixed
    if "created_at" not in result:
        result["created_at"] = result.pop("timestamp", datetime.now(timezone.utc).isoformat())
    result.setdefault("crypto", "HypΓ Schnorr-Γ / PSL(2,R) | 512-step walk | SHA3-256² address")
    if label:
        result["label"] = label
    return result


def _sign_direct(message_hex: str, private_key: str) -> dict:
    """Sign via in-process engine."""
    engine = _get_engine()
    if engine is None:
        raise RuntimeError("HypΓ engine not available in-process")

    msg_bytes = bytes.fromhex(message_hex)
    if len(msg_bytes) != 32:
        raise ValueError(f"message_hex must be 32 bytes; got {len(msg_bytes)}")

    sig = engine.sign_hash(msg_bytes, private_key)
    # sig is a dict with keys: signature, challenge, auth_tag, timestamp, ...
    return {
        "signature": sig["signature"],
        "challenge": sig["challenge"],
        "auth_tag":  sig.get("auth_tag", sig["challenge"]),
        "timestamp": sig.get("timestamp", datetime.now(timezone.utc).isoformat()),
    }


def _sign_rpc(message_hex: str, private_key: str) -> dict:
    """Sign via HTTP RPC fallback."""
    result = qtcl_rpc("qtcl_hyp_signMessage", {
        "message":     message_hex,
        "private_key": private_key,
    })
    return {
        "signature": result["signature"],
        "challenge": result["challenge"],
        "auth_tag":  result.get("auth_tag", result["challenge"]),
        "timestamp": result.get("timestamp", datetime.now(timezone.utc).isoformat()),
    }


def _handle_tool_call(tool_name: str, arguments: dict) -> Any:

    # ── Wallet / crypto ─────────────────────────────────────────────────────

    if tool_name == "qtcl_create_wallet":
        label = arguments.get("label", "")
        try:
            return _wallet_create_direct(label)
        except Exception as e:
            logger.warning(f"[MCP] Direct keygen failed ({e}), trying RPC")
            return _wallet_create_rpc(label)

    elif tool_name == "qtcl_sign_message":
        msg_hex = arguments.get("message_hex", "").strip()
        priv    = arguments.get("private_key", "").strip()
        if not msg_hex or not priv:
            raise ValueError("message_hex and private_key are required")
        if len(msg_hex) != 64:
            raise ValueError(
                f"message_hex must be 64 hex chars (32-byte SHA3-256); got {len(msg_hex)}"
            )
        try:
            return _sign_direct(msg_hex, priv)
        except Exception as e:
            logger.warning(f"[MCP] Direct sign failed ({e}), trying RPC")
            return _sign_rpc(msg_hex, priv)

    # ── Chain queries ────────────────────────────────────────────────────────

    elif tool_name == "qtcl_get_balance":
        return qtcl_rpc("qtcl_getBalance", [arguments["address"]])

    elif tool_name == "qtcl_get_utxos":
        p: dict = {"address": arguments["address"]}
        if arguments.get("limit"):
            p["limit"] = int(arguments["limit"])
        return qtcl_rpc("qtcl_getUTXOs", [p])

    elif tool_name == "qtcl_send_transaction":
        p = {
            "from_address": arguments["from_address"],
            "to_address":   arguments["to_address"],
            "amount":       arguments["amount"],
        }
        for k in ("memo", "signature", "public_key", "nonce"):
            if arguments.get(k) is not None:
                p[k] = arguments[k]
        return qtcl_rpc("qtcl_submitTransaction", [p])

    elif tool_name == "qtcl_get_transaction":
        return qtcl_rpc("qtcl_getTransaction", [arguments["tx_hash"]])

    elif tool_name == "qtcl_get_chain_info":
        return {
            "chain":   qtcl_rpc("qtcl_getBlockHeight"),
            "mempool": qtcl_rpc("qtcl_getMempoolStats"),
            "health":  qtcl_rpc("qtcl_getHealth"),
        }

    elif tool_name == "qtcl_get_block":
        if "height" in arguments:
            key = arguments["height"]
        elif "hash" in arguments:
            key = arguments["hash"]
        else:
            tip = qtcl_rpc("qtcl_getBlockHeight")
            key = tip.get("height", 0) if isinstance(tip, dict) else tip
        return qtcl_rpc("qtcl_getBlock", [key])

    elif tool_name == "qtcl_get_recent_transactions":
        p = {
            "page":     0,
            "per_page": min(int(arguments.get("per_page", 20)), 50),
        }
        if arguments.get("address"):
            p["address"] = arguments["address"]
        return qtcl_rpc("qtcl_getTransactions", p)

    elif tool_name == "qtcl_get_quantum_metrics":
        return qtcl_rpc("qtcl_getQuantumMetrics")

    elif tool_name == "qtcl_get_oracle_registry":
        return qtcl_rpc("qtcl_getOracleRegistry", {"limit": arguments.get("limit", 10)})

    elif tool_name == "qtcl_get_peers":
        return qtcl_rpc("qtcl_getPeers", [arguments.get("limit", 20)])

    elif tool_name == "qtcl_get_price":
        return qtcl_rpc("qtcl_getPythPrice")

    else:
        raise ValueError(f"Unknown tool: {tool_name}")


# ═══════════════════════════════════════════════════════════════════════════════
# §6  MCP PROTOCOL DISPATCHER
# ═══════════════════════════════════════════════════════════════════════════════

def _ok(msg_id: Any, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}

def _err(msg_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def handle_mcp_message(msg: dict) -> Optional[dict]:
    method = msg.get("method", "")
    msg_id = msg.get("id")
    params = msg.get("params", {})

    if method == "initialize":
        return _ok(msg_id, {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities":    {"tools": {"listChanged": False}},
            "serverInfo":      {"name": MCP_SERVER_NAME, "version": MCP_SERVER_VERSION},
        })

    if method.startswith("notifications/"):
        return None

    if method == "ping":
        return _ok(msg_id, {})

    if method == "tools/list":
        return _ok(msg_id, {"tools": MCP_TOOLS})

    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        try:
            result = _handle_tool_call(tool_name, arguments)
            return _ok(msg_id, {
                "content": [{"type": "text", "text": json.dumps(result, indent=2, default=str)}]
            })
        except Exception as e:
            logger.error(f"[MCP] tool error ({tool_name}): {e}", exc_info=True)
            return _ok(msg_id, {
                "content": [{"type": "text", "text": f"Error: {e}"}],
                "isError": True,
            })

    return _err(msg_id, -32601, f"Method not found: {method}")


# ═══════════════════════════════════════════════════════════════════════════════
# §7  STREAMABLE HTTP TRANSPORT
# ═══════════════════════════════════════════════════════════════════════════════

_sessions: Dict[str, queue.Queue] = {}
_sessions_lock = threading.Lock()


def _cors() -> dict:
    return {
        "Access-Control-Allow-Origin":  "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Accept, Mcp-Session-Id, Authorization",
    }


def _sse_stream(session_id: str) -> Response:
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
        "Connection":    "keep-alive",
        "X-Accel-Buffering": "no",
        **_cors(),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# §8  FLASK BLUEPRINT
# ═══════════════════════════════════════════════════════════════════════════════

def _create_mcp_blueprint():
    from flask import Blueprint
    bp = Blueprint("mcp", __name__)

    @bp.route("/mcp", methods=["OPTIONS"])
    @bp.route("/mcp/sse", methods=["OPTIONS"])
    def mcp_options():
        return Response("", status=204, headers=_cors())

    @bp.route("/mcp", methods=["GET"])
    def mcp_get():
        sid = request.headers.get("Mcp-Session-Id") or str(uuid.uuid4())
        logger.info(f"[MCP] SSE connect sid={sid[:8]}")
        return _sse_stream(sid)

    @bp.route("/mcp", methods=["POST"])
    def mcp_post():
        try:
            msg = request.get_json(force=True, silent=True)
            if not msg:
                return jsonify(_err(None, -32700, "Parse error")), 400
        except Exception:
            return jsonify(_err(None, -32700, "Parse error")), 400

        response   = handle_mcp_message(msg)
        is_init    = msg.get("method") == "initialize"
        sid        = request.headers.get("Mcp-Session-Id")
        if not sid and is_init:
            sid = str(uuid.uuid4())

        resp_headers = {**_cors()}
        if sid:
            resp_headers["Mcp-Session-Id"] = sid

        if response is None:
            return Response("", status=202, headers=resp_headers)

        if "text/event-stream" in request.headers.get("Accept", ""):
            resp_headers.update({"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
            return Response(
                f"event: message\ndata: {json.dumps(response)}\n\n",
                status=200, headers=resp_headers
            )

        resp_headers["Content-Type"] = "application/json"
        return Response(json.dumps(response), status=200, headers=resp_headers)

    @bp.route("/mcp/health", methods=["GET"])
    def mcp_health():
        engine_ok = _get_engine() is not None
        with _sessions_lock:
            n = len(_sessions)
        return jsonify({
            "status":          "ok",
            "server":          MCP_SERVER_NAME,
            "version":         MCP_SERVER_VERSION,
            "protocol":        MCP_PROTOCOL_VERSION,
            "transport":       "streamable-http",
            "hyp_engine":      "in-process" if engine_ok else "rpc-fallback",
            "active_sessions": n,
            "tools":           len(MCP_TOOLS),
        })

    # ── Legacy SSE / message endpoints (backwards compat) ──────────────────
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
            response = handle_mcp_message(msg)
            if response is not None:
                try:
                    q.put_nowait(response)
                except queue.Full:
                    q.get_nowait()
                    q.put_nowait(response)
            return "", 202
        except Exception as e:
            logger.error(f"[MCP] Legacy message error: {e}")
            return jsonify({"error": str(e)}), 500

    return bp


# ═══════════════════════════════════════════════════════════════════════════════
# §9  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def register_mcp_routes(app):
    """Register MCP endpoints on the main QTCL Flask app. Call from server.py."""
    app.register_blueprint(_create_mcp_blueprint())
    engine_status = "in-process ✅" if _get_engine() is not None else "rpc-fallback ⚠️"
    logger.info(f"[MCP] ✅ QTCL MCP server v{MCP_SERVER_VERSION} registered — {len(MCP_TOOLS)} tools")
    logger.info(f"[MCP]    Transport  : Streamable HTTP")
    logger.info(f"[MCP]    Protocol   : {MCP_PROTOCOL_VERSION}")
    logger.info(f"[MCP]    HypΓ       : {engine_status}")
    logger.info(f"[MCP]    Primary    : /mcp")
    logger.info(f"[MCP]    Legacy SSE : /mcp/sse")


# ═══════════════════════════════════════════════════════════════════════════════
# §10  STANDALONE MODE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║    QTCL MCP SERVER v2.1 — Streamable HTTP + HypΓ Engine     ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")
    port = int(os.environ.get("MCP_PORT", 8002))
    logger.info(f"  Tools    : {len(MCP_TOOLS)}")
    logger.info(f"  Protocol : {MCP_PROTOCOL_VERSION}")
    logger.info(f"  RPC URL  : {QTCL_RPC_URL}")
    logger.info(f"  Endpoint : http://0.0.0.0:{port}/mcp")

    standalone = Flask(__name__)
    standalone.register_blueprint(_create_mcp_blueprint())
    standalone.run(host="0.0.0.0", port=port, debug=False)
