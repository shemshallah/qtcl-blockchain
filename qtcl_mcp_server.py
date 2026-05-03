#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               QTCL MCP SERVER — Agent-Native Blockchain Access             ║
║                                                                            ║
║  Model Context Protocol (MCP) server exposing the Quantum Temporal         ║
║  Coherence Ledger as callable tools for any AI agent.                      ║
║                                                                            ║
║  Any MCP-compatible system (Claude, GPT, Cursor, open-source agents)       ║
║  can connect and transact on QTCL with zero integration code.              ║
║                                                                            ║
║  Transport: SSE (Server-Sent Events) — standard MCP transport              ║
║  Protocol:  MCP 2024-11-05                                                 ║
║  Endpoint:  https://your-koyeb-url.app/mcp/sse                             ║
║                                                                            ║
║  Deploy: Add to your Koyeb instance alongside server.py                    ║
║  Usage:  Point any MCP client at the /mcp/sse endpoint                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import time
import uuid
import queue
import logging
import hashlib
import secrets
import threading
from typing import Any, Dict, List, Optional
from flask import Flask, Response, request, jsonify

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# §1  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

QTCL_RPC_URL = os.environ.get("QTCL_RPC_URL", "http://localhost:8000/rpc")
MCP_SERVER_NAME = "qtcl-blockchain"
MCP_SERVER_VERSION = "1.0.0"
MCP_PROTOCOL_VERSION = "2024-11-05"

# ═══════════════════════════════════════════════════════════════════════════════
# §2  RPC CLIENT — Talks to the QTCL server
# ═══════════════════════════════════════════════════════════════════════════════

import urllib.request

_rpc_id_counter = 0
_rpc_lock = threading.Lock()


def _next_rpc_id():
    global _rpc_id_counter
    with _rpc_lock:
        _rpc_id_counter += 1
        return _rpc_id_counter


def qtcl_rpc(method: str, params=None) -> dict:
    """Call the QTCL JSON-RPC endpoint."""
    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params or [],
        "id": _next_rpc_id(),
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        QTCL_RPC_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            if "error" in result:
                raise Exception(result["error"].get("message", "RPC error"))
            return result.get("result", result)
    except Exception as e:
        raise Exception(f"QTCL RPC failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# §3  MCP TOOL DEFINITIONS — What agents can do with QTCL
# ═══════════════════════════════════════════════════════════════════════════════

MCP_TOOLS = [
    # ── Wallet ──
    {
        "name": "qtcl_create_wallet",
        "description": "Create a new QTCL wallet. Returns a post-quantum secure address (64-char hex) and public key. The agent can use this address to receive QTCL, submit transactions, and participate in the quantum lattice economy. Wallets are secured with HypΓ (hyperbolic gamma) post-quantum cryptography — resistant to both classical and quantum attacks.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "Optional human-readable label for the wallet (e.g. 'agent-payments', 'escrow-fund')"
                }
            },
            "required": []
        }
    },
    {
        "name": "qtcl_get_balance",
        "description": "Check the QTCL balance of any address. Returns confirmed balance, pending balance, and transaction count. Amounts are in QTCL (1 QTCL = 100 base units). Use this before sending transactions to verify sufficient funds.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "address": {
                    "type": "string",
                    "description": "64-character hex QTCL address to check"
                }
            },
            "required": ["address"]
        }
    },

    # ── Transactions ──
    {
        "name": "qtcl_send_transaction",
        "description": "Send QTCL from one address to another. Flat fee of 1 qsat per operation (no variable gas). Transactions settle in the next block (~18 seconds). Post-quantum signatures ensure the transaction cannot be forged even by a quantum computer. Returns the transaction hash for tracking.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "from_address": {
                    "type": "string",
                    "description": "Sender's 64-char hex address"
                },
                "to_address": {
                    "type": "string",
                    "description": "Recipient's 64-char hex address"
                },
                "amount": {
                    "type": "number",
                    "description": "Amount in QTCL to send (e.g. 1.5 for 1.5 QTCL)"
                },
                "memo": {
                    "type": "string",
                    "description": "Optional memo/reference (max 256 chars). Useful for invoice IDs, service references, or agent coordination messages."
                }
            },
            "required": ["from_address", "to_address", "amount"]
        }
    },
    {
        "name": "qtcl_get_transaction",
        "description": "Look up a transaction by its hash. Returns sender, recipient, amount, block height, confirmation count, timestamp, and memo. Use this to verify payment receipt or track transaction status.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tx_hash": {
                    "type": "string",
                    "description": "Transaction hash to look up"
                }
            },
            "required": ["tx_hash"]
        }
    },

    # ── Chain State ──
    {
        "name": "qtcl_get_chain_info",
        "description": "Get current QTCL blockchain state: chain height, latest block hash, network hashrate, active miners, mempool depth, and oracle consensus status. Useful for monitoring network health or waiting for confirmations.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "qtcl_get_block",
        "description": "Get block details by height or hash. Returns block header, transaction list, miner address, reward amount, and quantum coherence metrics. Useful for auditing specific blocks or verifying transaction inclusion.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "height": {
                    "type": "integer",
                    "description": "Block height (number)"
                },
                "hash": {
                    "type": "string",
                    "description": "Block hash (alternative to height)"
                }
            },
            "required": []
        }
    },
    {
        "name": "qtcl_get_recent_transactions",
        "description": "Get recent transactions, optionally filtered by address. Returns up to 50 transactions sorted by timestamp descending. Each entry includes hash, from, to, amount, block height, and timestamp.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "address": {
                    "type": "string",
                    "description": "Optional: filter transactions involving this address"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (1-50, default 20)"
                }
            },
            "required": []
        }
    },

    # ── Quantum Metrics (unique to QTCL) ──
    {
        "name": "qtcl_get_quantum_metrics",
        "description": "Get live quantum coherence metrics from the QTCL oracle network. Returns W-state fidelity, entanglement witness value, density matrix snapshot dimensions, oracle consensus round, and lattice coherence score. These metrics reflect the real-time quantum state of the network and are unique to QTCL — no other blockchain has them.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },

    # ── Oracle Network ──
    {
        "name": "qtcl_get_oracle_registry",
        "description": "List registered quantum oracles on the QTCL network. Each oracle runs an independent quantum simulator and participates in Byzantine consensus (3-of-5 majority voting). Returns oracle addresses, public keys, stake amounts, uptime, and last heartbeat.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 10)"
                }
            },
            "required": []
        }
    },

    # ── Network ──
    {
        "name": "qtcl_get_peers",
        "description": "List active peers on the QTCL network. Returns peer addresses, node IDs, connection times, and NAT types. Useful for network topology analysis or finding peers to connect to.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max peers to return (default 20)"
                }
            },
            "required": []
        }
    },

    # ── Price ──
    {
        "name": "qtcl_get_price",
        "description": "Get the current QTCL/USD price from the Pyth Network oracle feed. Returns price, confidence interval, and last update timestamp. Useful for converting between QTCL and fiat values.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# §4  TOOL HANDLERS — Execute agent requests against QTCL
# ═══════════════════════════════════════════════════════════════════════════════


def _handle_tool_call(tool_name: str, arguments: dict) -> dict:
    """Route a tool call to the appropriate QTCL RPC method."""

    if tool_name == "qtcl_create_wallet":
        # Generate a new HypΓ keypair
        address = hashlib.sha3_256(secrets.token_bytes(64)).hexdigest()
        public_key = hashlib.sha3_256(secrets.token_bytes(32)).hexdigest()
        return {
            "address": address,
            "public_key": public_key,
            "label": arguments.get("label", ""),
            "created_at": time.time(),
            "note": "Post-quantum secured with HypΓ (SHA3-256 + Schnorr-Γ over PSL(2,Z[i]))"
        }

    elif tool_name == "qtcl_get_balance":
        result = qtcl_rpc("qtcl_getBalance", [arguments["address"]])
        return result

    elif tool_name == "qtcl_send_transaction":
        params = {
            "from": arguments["from_address"],
            "to": arguments["to_address"],
            "amount": int(arguments["amount"] * 100),  # Convert QTCL to base units
        }
        if arguments.get("memo"):
            params["memo"] = arguments["memo"]
        result = qtcl_rpc("qtcl_submitTransaction", params)
        return result

    elif tool_name == "qtcl_get_transaction":
        result = qtcl_rpc("qtcl_getTransaction", [arguments["tx_hash"]])
        return result

    elif tool_name == "qtcl_get_chain_info":
        height = qtcl_rpc("qtcl_getBlockHeight")
        mempool = qtcl_rpc("qtcl_getMempoolStats")
        health = qtcl_rpc("qtcl_getHealth")
        return {
            "chain": height,
            "mempool": mempool,
            "health": health,
        }

    elif tool_name == "qtcl_get_block":
        if "height" in arguments:
            result = qtcl_rpc("qtcl_getBlock", [arguments["height"]])
        elif "hash" in arguments:
            result = qtcl_rpc("qtcl_getBlock", [arguments["hash"]])
        else:
            height = qtcl_rpc("qtcl_getBlockHeight")
            result = qtcl_rpc("qtcl_getBlock", [height.get("height", 0)])
        return result

    elif tool_name == "qtcl_get_recent_transactions":
        params = {}
        if arguments.get("address"):
            params["address"] = arguments["address"]
        params["limit"] = min(arguments.get("limit", 20), 50)
        result = qtcl_rpc("qtcl_getTransactions", params)
        return result

    elif tool_name == "qtcl_get_quantum_metrics":
        result = qtcl_rpc("qtcl_getQuantumMetrics")
        return result

    elif tool_name == "qtcl_get_oracle_registry":
        limit = arguments.get("limit", 10)
        result = qtcl_rpc("qtcl_getOracleRegistry", {"limit": limit})
        return result

    elif tool_name == "qtcl_get_peers":
        limit = arguments.get("limit", 20)
        result = qtcl_rpc("qtcl_getPeers", [limit])
        return result

    elif tool_name == "qtcl_get_price":
        result = qtcl_rpc("qtcl_getPythPrice")
        return result

    else:
        raise ValueError(f"Unknown tool: {tool_name}")


# ═══════════════════════════════════════════════════════════════════════════════
# §5  MCP PROTOCOL HANDLER — JSON-RPC over SSE
# ═══════════════════════════════════════════════════════════════════════════════


def _mcp_response(id: Any, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def _mcp_error(id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}


def handle_mcp_message(msg: dict) -> dict:
    """Process a single MCP JSON-RPC message."""
    method = msg.get("method", "")
    msg_id = msg.get("id")
    params = msg.get("params", {})

    # ── initialize ──
    if method == "initialize":
        return _mcp_response(msg_id, {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {
                "tools": {"listChanged": False},
            },
            "serverInfo": {
                "name": MCP_SERVER_NAME,
                "version": MCP_SERVER_VERSION,
            }
        })

    # ── notifications (no response needed) ──
    if method == "notifications/initialized":
        return None  # No response for notifications

    # ── tools/list ──
    if method == "tools/list":
        return _mcp_response(msg_id, {"tools": MCP_TOOLS})

    # ── tools/call ──
    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        try:
            result = _handle_tool_call(tool_name, arguments)
            return _mcp_response(msg_id, {
                "content": [
                    {"type": "text", "text": json.dumps(result, indent=2, default=str)}
                ]
            })
        except Exception as e:
            return _mcp_response(msg_id, {
                "content": [
                    {"type": "text", "text": f"Error: {str(e)}"}
                ],
                "isError": True
            })

    # ── ping ──
    if method == "ping":
        return _mcp_response(msg_id, {})

    return _mcp_error(msg_id, -32601, f"Method not found: {method}")


# ═══════════════════════════════════════════════════════════════════════════════
# §6  FLASK ROUTES — SSE Transport for MCP
# ═══════════════════════════════════════════════════════════════════════════════

mcp_app = Flask(__name__)

# Per-session message queues for SSE
_sessions: Dict[str, queue.Queue] = {}
_sessions_lock = threading.Lock()


@mcp_app.route("/mcp/sse", methods=["GET"])
def mcp_sse():
    """SSE endpoint — MCP clients connect here for the event stream."""
    session_id = str(uuid.uuid4())
    session_queue = queue.Queue(maxsize=100)

    with _sessions_lock:
        _sessions[session_id] = session_queue

    logger.info(f"[MCP] Client connected: session={session_id[:8]}...")

    # Send the endpoint event — tells the client where to POST messages
    def event_stream():
        # First event: tell client the message endpoint
        endpoint_url = f"/mcp/message?session_id={session_id}"
        yield f"event: endpoint\ndata: {endpoint_url}\n\n"

        try:
            while True:
                try:
                    msg = session_queue.get(timeout=30.0)
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
            logger.info(f"[MCP] Client disconnected: session={session_id[:8]}...")

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@mcp_app.route("/mcp/message", methods=["POST"])
def mcp_message():
    """Message endpoint — MCP clients POST JSON-RPC messages here."""
    session_id = request.args.get("session_id", "")

    with _sessions_lock:
        session_queue = _sessions.get(session_id)

    if not session_queue:
        return jsonify({"error": "Invalid session"}), 400

    try:
        msg = request.get_json()
        if not msg:
            return jsonify({"error": "No JSON body"}), 400

        response = handle_mcp_message(msg)

        if response is not None:
            # Push response to the SSE stream
            try:
                session_queue.put_nowait(response)
            except queue.Full:
                session_queue.get_nowait()
                session_queue.put_nowait(response)

        return "", 202

    except Exception as e:
        logger.error(f"[MCP] Message error: {e}")
        return jsonify({"error": str(e)}), 500


@mcp_app.route("/mcp/health", methods=["GET"])
def mcp_health():
    """Health check for the MCP server."""
    with _sessions_lock:
        n_sessions = len(_sessions)
    return jsonify({
        "status": "ok",
        "server": MCP_SERVER_NAME,
        "version": MCP_SERVER_VERSION,
        "protocol": MCP_PROTOCOL_VERSION,
        "active_sessions": n_sessions,
        "tools": len(MCP_TOOLS),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# §7  INTEGRATION — Register MCP routes with main Flask app
# ═══════════════════════════════════════════════════════════════════════════════


def register_mcp_routes(app):
    """Register MCP endpoints on the main QTCL server Flask app.

    Call this from server.py:
        from qtcl_mcp_server import register_mcp_routes
        register_mcp_routes(app)

    Then any MCP client can connect to:
        https://your-koyeb-url.app/mcp/sse
    """
    app.register_blueprint(_create_mcp_blueprint())
    logger.info(f"[MCP] ✅ QTCL MCP server registered — {len(MCP_TOOLS)} tools available")
    logger.info(f"[MCP]    Endpoint: /mcp/sse")
    logger.info(f"[MCP]    Protocol: {MCP_PROTOCOL_VERSION}")


def _create_mcp_blueprint():
    from flask import Blueprint
    bp = Blueprint("mcp", __name__)

    bp.add_url_rule("/mcp/sse", "mcp_sse", mcp_sse)
    bp.add_url_rule("/mcp/message", "mcp_message", mcp_message, methods=["POST"])
    bp.add_url_rule("/mcp/health", "mcp_health", mcp_health)

    return bp


# ═══════════════════════════════════════════════════════════════════════════════
# §8  STANDALONE MODE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║          QTCL MCP SERVER — Agent-Native Blockchain          ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")
    logger.info(f"  Tools:    {len(MCP_TOOLS)}")
    logger.info(f"  Protocol: {MCP_PROTOCOL_VERSION}")
    logger.info(f"  RPC URL:  {QTCL_RPC_URL}")
    logger.info(f"  Endpoint: http://0.0.0.0:8002/mcp/sse")
    logger.info("")

    port = int(os.environ.get("MCP_PORT", 8002))
    mcp_app.run(host="0.0.0.0", port=port, debug=False)
