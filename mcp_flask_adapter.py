#!/usr/bin/env python3
"""
================================================================================
MCP FLASK ADAPTER v5.0 — QTCL Streamable HTTP + SSE Dual-Protocol MCP Layer
================================================================================

ARCHITECTURE (v5.0 — ground-up rewrite, zero drift from spec):

  This file IS the MCP implementation for the QTCL server. It registers pure
  Flask routes on the main app (server.py calls register_mcp_routes(app)).
  Zero ASGI, zero asyncio, zero Starlette, zero proxy — runs clean on any
  WSGI server (Gunicorn gthread, sync, eventlet).

PROTOCOL SUPPORT (both sides fully wired):
  ┌──────────────────┬──────────────────────────────────────────────────────┐
  │ Version          │ Endpoints                                            │
  ├──────────────────┼──────────────────────────────────────────────────────┤
  │ MCP 2025-06-18   │ POST /mcp  (Streamable HTTP — primary)              │
  │ (Claude.ai)      │ GET  /mcp  (server info JSON)                       │
  │                  │ DELETE /mcp (session teardown)                       │
  ├──────────────────┼──────────────────────────────────────────────────────┤
  │ MCP 2024-11-05   │ GET  /mcp/sse     (legacy SSE stream)               │
  │ (legacy clients) │ POST /mcp/message (legacy message channel)          │
  ├──────────────────┼──────────────────────────────────────────────────────┤
  │ Both             │ GET  /mcp/health  (health + capability summary)      │
  │                  │ GET  /mcp/capability (full agent capability doc)     │
  └──────────────────┴──────────────────────────────────────────────────────┘

LIFECYCLE (MCP 2025-06-18 Streamable HTTP):
  1. Client → POST /mcp  { initialize, protocolVersion, capabilities }
     Server → 200  { result: { protocolVersion, serverInfo, capabilities } }
             + Mcp-Session-Id header
  2. Client → POST /mcp  { notifications/initialized }  (no id — notification)
     Server → 200  (empty body, 202 also accepted by clients)
  3. Client → POST /mcp  { tools/list }
     Server → 200  { result: { tools: [...] } }
  4. Client → POST /mcp  { tools/call, name, arguments }
     Server → 200  { result: { content: [{ type: "text", text: "..." }] } }
  5. Client → DELETE /mcp  (session teardown)
     Server → 200

LIFECYCLE (MCP 2024-11-05 SSE):
  1. Client → GET /mcp/sse
     Server → text/event-stream  event:endpoint  data:/mcp/message?session_id=UUID
  2. Client → POST /mcp/message?session_id=UUID  { initialize }
     Server → queues response to SSE stream
  3. Client reads SSE stream for all responses

PARAM DISPATCH (matches server.py _rpc_* signatures exactly):
  Every tool call maps to a qtcl_* RPC method through _rpc() which hits
  the live /rpc endpoint. Param packing is verified against the actual
  server.py handler signatures to prevent silent -32602 errors.

FIXES vs v4.x:
  ✓ Protocol version negotiation: echoes client version if supported, else 2025-06-18
  ✓ notifications/initialized: never crashes, always 200 with empty body
  ✓ resources/list + prompts/list: fully implemented (Claude.ai requests these)
  ✓ Batch request handling: processes each item correctly with shared session
  ✓ getTransactions: sends dict (not [dict]) — matches server.py expectation
  ✓ getOracleRegistry: sends [{"limit":...}] — matches params[0] unpack path
  ✓ getPeers: sends [limit_int] — matches params[0] int unpack path
  ✓ All CORS headers match MCP spec (Last-Event-ID included)
  ✓ Session state is fork-safe (no threading.RLock across fork boundaries)
  ✓ SSE /mcp/sse sends correct endpoint URL with ?session_id= query param
  ✓ DELETE /mcp handler present and correct
  ✓ Proper JSON error for unknown tools (not silent empty result)
  ✓ isError flag on tool failure responses
  ✓ SDK_AVAILABLE properly scoped before use in register path

Dependencies:
  flask>=2.0   (required)
  mcp>=1.9.0   (optional — only for stdio transport mode)
================================================================================
"""

from __future__ import annotations

import json
import logging
import os
import queue
import sys
import threading
import time
import urllib.request
import uuid
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# §0  CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

QTCL_RPC_URL        = os.environ.get("QTCL_RPC_URL", "http://localhost:8000/rpc")
MCP_SERVER_NAME     = "qtcl-blockchain"
MCP_SERVER_VERSION  = "5.0.0"

# Both versions we speak — 2025-06-18 is preferred; 2024-11-05 for legacy clients.
_PROTO_PREFERRED    = "2025-06-18"
_PROTO_LEGACY       = "2024-11-05"
_SUPPORTED_PROTOS   = {_PROTO_PREFERRED, _PROTO_LEGACY}

# ──────────────────────────────────────────────────────────────────────────────
# §1  RPC CLIENT — reentrant, thread-safe, no shared state between workers
# ──────────────────────────────────────────────────────────────────────────────

_rpc_id_tl = threading.local()   # per-thread counter — fork safe, gunicorn safe


def _next_rpc_id() -> int:
    if not hasattr(_rpc_id_tl, "counter"):
        _rpc_id_tl.counter = 0
    _rpc_id_tl.counter += 1
    return _rpc_id_tl.counter


def _rpc(method: str, params: Any = None, rpc_url: str = QTCL_RPC_URL, timeout: float = 20.0) -> Any:
    """Issue a JSON-RPC 2.0 call to the QTCL server and return result or raise RuntimeError."""
    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params if params is not None else [],
        "id": _next_rpc_id(),
    }
    raw = json.dumps(payload, default=str).encode("utf-8")
    req = urllib.request.Request(
        rpc_url,
        data=raw,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        if "error" in body:
            err = body["error"]
            raise RuntimeError(f"[{err.get('code', -1)}] {err.get('message', 'RPC error')}")
        return body.get("result", body)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"QTCL RPC '{method}' transport failure: {exc}") from exc


# ──────────────────────────────────────────────────────────────────────────────
# §2  TOOL DEFINITIONS — JSON Schema inputSchema, matching MCP spec exactly
# ──────────────────────────────────────────────────────────────────────────────

_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "qtcl_create_wallet",
        "description": (
            "Create a new QTCL post-quantum wallet backed by a real HypΓ keypair "
            "(Schnorr-Γ over PSL(2,R), 512-step random walk, SHA3-256² address). "
            "Returns private_key, public_key, address, and created_at. "
            "Store private_key securely — the server never retains it."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "description": "Optional human-readable wallet label"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "qtcl_sign_message",
        "description": (
            "Sign a 32-byte message hash with a HypΓ private key using Schnorr-Γ. "
            "To sign a transaction, compute: SHA3-256(JSON.dumps({\"sender\": from_addr, "
            "\"recipient\": to_addr, \"amount\": amount_float, \"nonce\": nonce_int}, "
            "sort_keys=True)) → pass the 64-char hex as message_hex. "
            "The nonce MUST match the nonce you will use in qtcl_send_transaction. "
            "Returns the full signature dict (with canonical R/Z matrix fields) — "
            "pass the entire JSON output as the signature field to qtcl_send_transaction."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "message_hex": {"type": "string", "description": "64 hex chars (32-byte SHA3-256 hash of tx signing payload)"},
                "private_key": {"type": "string", "description": "HypΓ private key from qtcl_create_wallet (512-char base-4 walk)"},
            },
            "required": ["message_hex", "private_key"],
            "additionalProperties": False,
        },
    },
    {
        "name": "qtcl_get_balance",
        "description": "Check QTCL balance for any address. Returns balance in base units (qsat), UTXO count, and UTXO list. 1 QTCL = 100 qsat.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "address": {"type": "string", "description": "64-char hex QTCL address"},
            },
            "required": ["address"],
            "additionalProperties": False,
        },
    },
    {
        "name": "qtcl_get_utxos",
        "description": "List unspent transaction outputs (UTXOs) for an address. Returns tx_hash, output_index, amount_base per coin.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "address": {"type": "string", "description": "64-char hex QTCL address"},
                "limit":   {"type": "integer", "description": "Max UTXOs to return (default 1000)", "default": 1000},
            },
            "required": ["address"],
            "additionalProperties": False,
        },
    },
    {
        "name": "qtcl_send_transaction",
        "description": (
            "Submit a signed UTXO transaction to the QTCL network. "
            "Flat fee: 1 qsat. Finality: ~18 seconds. "
            "Provide the full signature JSON from qtcl_sign_message as the signature field, "
            "and the public_key from qtcl_create_wallet. The nonce must match the nonce "
            "used in the signing payload. If nonce is omitted, one is auto-generated."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "from_address": {"type": "string", "description": "Sender's 64-char hex QTCL address"},
                "to_address":   {"type": "string", "description": "Recipient's 64-char hex QTCL address"},
                "amount":       {"type": "number", "description": "Amount in QTCL (not qsat). 1 QTCL = 100 qsat."},
                "memo":         {"type": "string", "description": "Optional transaction memo (max 256 chars)"},
                "signature":    {"type": "string", "description": "Full signature JSON from qtcl_sign_message (includes R, Z, challenge, c_full)"},
                "public_key":   {"type": "string", "description": "HypΓ public key hex from qtcl_create_wallet"},
                "nonce":        {"type": "integer", "description": "Replay-prevention nonce (must match signing payload nonce). Auto-generated if omitted."},
            },
            "required": ["from_address", "to_address", "amount"],
            "additionalProperties": False,
        },
    },
    {
        "name": "qtcl_get_transaction",
        "description": "Look up a QTCL transaction by its SHA3-256 hash. Returns full tx details, status, and block height.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tx_hash": {"type": "string", "description": "64-char hex transaction hash"},
            },
            "required": ["tx_hash"],
            "additionalProperties": False,
        },
    },
    {
        "name": "qtcl_get_chain_info",
        "description": "Current blockchain state: height, latest block hash, mempool depth, oracle status, and system health vector.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "qtcl_get_block",
        "description": "Retrieve a block by height (integer) or hash (hex string). Omit both for the latest block.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "height": {"type": "integer", "description": "Block height (0 = genesis; omit for latest)"},
                "hash":   {"type": "string",  "description": "Block hash hex (takes priority over height)"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "qtcl_get_recent_transactions",
        "description": "List recent transactions, optionally filtered by address. Returns newest first. Max 50 per call.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "address":  {"type": "string",  "description": "Optional: filter by sender or receiver address"},
                "per_page": {"type": "integer", "description": "Results per page (default 20, max 50)", "default": 20},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "qtcl_get_quantum_metrics",
        "description": (
            "Live quantum coherence metrics: W-state fidelity (≥0.75 healthy), "
            "entanglement witness (NPT criterion), oracle consensus round, "
            "Mermin inequality test, and kappa=0.11 non-Markovian coherence score."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "qtcl_get_oracle_registry",
        "description": "List registered quantum oracle nodes participating in 5-oracle Byzantine consensus (3-of-5 majority).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max oracles to return (default 10, max 100)", "default": 10},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "qtcl_get_peers",
        "description": "List active P2P peers in the QTCL Kademlia DHT network.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max peers to return (default 20)", "default": 20},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "qtcl_get_price",
        "description": (
            "QTCL network quantum coherence metrics and valuation signals. "
            "Note: QTCL has no public USD exchange. Returns W-state fidelity, "
            "entanglement witness, and oracle coherence as network health proxy."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# §3  RESOURCES — MCP resource list (Claude.ai requests resources/list)
# ──────────────────────────────────────────────────────────────────────────────

_RESOURCES: List[Dict[str, Any]] = [
    {
        "uri":         "chain://height",
        "name":        "Current Block Height",
        "description": "Live chain tip height as a plain integer string.",
        "mimeType":    "text/plain",
    },
    {
        "uri":         "chain://health",
        "name":        "System Health Vector",
        "description": "Full JSON health snapshot: DB, mempool, oracle, quantum metrics.",
        "mimeType":    "application/json",
    },
    {
        "uri":         "price://qtcl-usd",
        "name":        "QTCL Quantum Metrics",
        "description": "Quantum coherence metrics proxy for network value (no USD exchange).",
        "mimeType":    "application/json",
    },
    {
        "uri":         "docs://capability",
        "name":        "QTCL Capability Document",
        "description": "Full agent capability document describing the QTCL blockchain system.",
        "mimeType":    "application/json",
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# §4  PROMPTS — MCP prompt list (Claude.ai requests prompts/list)
# ──────────────────────────────────────────────────────────────────────────────

_PROMPTS: List[Dict[str, Any]] = [
    {
        "name":        "wallet_helper",
        "description": "Step-by-step guidance for QTCL wallet operations (create or send).",
        "arguments": [
            {
                "name":        "task",
                "description": "Operation: 'create' or 'send'",
                "required":    False,
            }
        ],
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# §5  TOOL DISPATCH — maps tool names → RPC calls with correct param packing
#     Each mapping is verified against the actual server.py handler signature.
# ──────────────────────────────────────────────────────────────────────────────

def _dispatch_tool(name: str, args: Dict[str, Any], rpc_url: str) -> Dict[str, Any]:
    """
    Execute a named MCP tool call and return a proper MCP result block.

    Returns:
        {"content": [{"type": "text", "text": "..."}], "isError": False}
    or on error:
        {"content": [{"type": "text", "text": "Error: ..."}], "isError": True}

    Param packing rules (verified against server.py _rpc_* signatures):
      - getBalance:            params=[address_str]                         ← list[0] str
      - getUTXOs:              params=[{"address":..., "limit":...}]        ← list[0] dict
      - getTransaction:        params=[tx_hash_str]                         ← list[0] str
      - getBlock:              params=[height_or_hash]                      ← list[0] int|str
      - getTransactions:       params={"page":0, "per_page":N, ...}         ← flat dict (NOT list-wrapped)
      - getOracleRegistry:     params=[{"limit":N}]                         ← list[0] dict
      - getPeers:              params=[limit_int]                           ← list[0] int
      - submitTransaction:     params=[{...tx_fields...}]                   ← list[0] dict
      - hyp_generateKeypair:   params={}                                    ← dict (or [])
      - hyp_signMessage:       params={"message":hex, "private_key":key}    ← dict
    """
    def _ok(data: Any) -> Dict[str, Any]:
        text = json.dumps(data, indent=2, default=str) if not isinstance(data, str) else data
        return {"content": [{"type": "text", "text": text}], "isError": False}

    def _err(msg: str) -> Dict[str, Any]:
        return {"content": [{"type": "text", "text": f"Error: {msg}"}], "isError": True}

    try:
        # ── Wallet creation ─────────────────────────────────────────────────
        if name == "qtcl_create_wallet":
            r = _rpc("qtcl_hyp_generateKeypair", {}, rpc_url)
            if "created_at" not in r:
                r["created_at"] = r.pop("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            r.setdefault("crypto", "HypΓ Schnorr-Γ / PSL(2,R) | 512-step walk | SHA3-256² address")
            lbl = args.get("label", "")
            if lbl:
                r["label"] = lbl
            return _ok(r)

        # ── Sign message ────────────────────────────────────────────────────
        elif name == "qtcl_sign_message":
            mhex = args.get("message_hex", "")
            pk   = args.get("private_key", "")
            if not mhex or not pk:
                return _err("message_hex and private_key are required")
            if len(mhex) != 64:
                return _err(f"message_hex must be 64 hex chars (32-byte hash); got {len(mhex)}")
            # server.py qtcl_hyp_signMessage returns the full sig dict with canonical fields
            r = _rpc("qtcl_hyp_signMessage", {"message": mhex, "private_key": pk}, rpc_url)
            # FIX: The result from RPC is a flat dict:
            #   {signature: hex_str, challenge: hex_str, R: {...}, Z: {...}, c_full: ..., ...}
            # The MCP caller needs to pass the ENTIRE dict (as a JSON string) as the
            # "signature" field to qtcl_send_transaction. Package it so the caller can
            # just use result.signature directly:
            #   result.signature_for_tx = JSON string of the full sig dict
            #   (plus the raw fields for inspection)
            sig_for_tx = json.dumps(r, default=str)
            r["signature_for_tx"] = sig_for_tx
            r["_usage"] = (
                "Pass 'signature_for_tx' as the 'signature' parameter to qtcl_send_transaction. "
                "It contains the full sig dict (R, Z, c_full, challenge) needed for verification."
            )
            return _ok(r)

        # ── Balance ─────────────────────────────────────────────────────────
        elif name == "qtcl_get_balance":
            addr = args.get("address", "")
            if not addr:
                return _err("address is required")
            # server.py _rpc_getBalance: params[0] str or params["address"] str
            r = _rpc("qtcl_getBalance", [addr], rpc_url)
            return _ok(r)

        # ── UTXOs ───────────────────────────────────────────────────────────
        elif name == "qtcl_get_utxos":
            addr  = args.get("address", "")
            limit = int(args.get("limit", 1000))
            if not addr:
                return _err("address is required")
            # server.py _rpc_getUTXOs: params[0] dict or params[0] str
            r = _rpc("qtcl_getUTXOs", [{"address": addr, "limit": limit}], rpc_url)
            return _ok(r)

        # ── Send transaction ────────────────────────────────────────────────
        elif name == "qtcl_send_transaction":
            fa = args.get("from_address", "")
            ta = args.get("to_address", "")
            am = args.get("amount")
            if not fa or not ta or am is None:
                return _err("from_address, to_address, and amount are required")
            p: Dict[str, Any] = {"from_address": fa, "to_address": ta, "amount": float(am)}

            # Memo
            memo = args.get("memo", "")
            if memo:
                p["memo"] = memo

            # Nonce — MUST pass through even if 0 (mempool requires it)
            # FIX: old code filtered nonce==0 with `v != 0`, breaking required field
            nonce_val = args.get("nonce")
            if nonce_val is not None:
                p["nonce"] = int(nonce_val)
            else:
                # Auto-generate nonce if not provided (nanosecond timestamp)
                p["nonce"] = int(time.time() * 1e9)

            # Signature — ensure full sig dict with canonical fields is embedded
            # Accept signature_for_tx (from qtcl_sign_message output) as alias
            sig_raw = args.get("signature", "") or args.get("signature_for_tx", "")
            pub_key = args.get("public_key", "")
            if sig_raw:
                # Parse signature string to dict if needed
                sig_dict = None
                if isinstance(sig_raw, str):
                    try:
                        sig_dict = json.loads(sig_raw)
                    except (json.JSONDecodeError, TypeError):
                        sig_dict = {"signature": sig_raw}
                elif isinstance(sig_raw, dict):
                    sig_dict = dict(sig_raw)

                if sig_dict:
                    # Inject public_key into sig dict if not already present
                    if pub_key:
                        if not sig_dict.get("public_key"):
                            sig_dict["public_key"] = pub_key
                        if not sig_dict.get("public_key_hex"):
                            sig_dict["public_key_hex"] = pub_key
                    p["signature"] = json.dumps(sig_dict, default=str)
                else:
                    p["signature"] = sig_raw

            # Public key fields — mempool checks both top-level and inside sig
            if pub_key:
                p["public_key"] = pub_key
                p["sender_public_key_hex"] = pub_key

            # server.py _rpc_submitTransaction: params[0] dict
            r = _rpc("qtcl_submitTransaction", [p], rpc_url)
            return _ok(r)

        # ── Get transaction ─────────────────────────────────────────────────
        elif name == "qtcl_get_transaction":
            txh = args.get("tx_hash", "")
            if not txh:
                return _err("tx_hash is required")
            # server.py _rpc_getTransaction: params[0] str
            r = _rpc("qtcl_getTransaction", [txh], rpc_url)
            return _ok(r)

        # ── Chain info (composite) ──────────────────────────────────────────
        elif name == "qtcl_get_chain_info":
            result = {
                "chain":   _rpc("qtcl_getBlockHeight", [], rpc_url),
                "mempool": _rpc("qtcl_getMempoolStats", [], rpc_url),
                "health":  _rpc("qtcl_getHealth",      [], rpc_url),
            }
            return _ok(result)

        # ── Get block ───────────────────────────────────────────────────────
        elif name == "qtcl_get_block":
            blk_hash   = args.get("hash", "")
            blk_height = args.get("height")
            if blk_hash:
                key: Any = blk_hash
            elif blk_height is not None and int(blk_height) >= 0:
                key = int(blk_height)
            else:
                tip = _rpc("qtcl_getBlockHeight", [], rpc_url)
                key = tip.get("height", 0) if isinstance(tip, dict) else int(tip or 0)
            # server.py _rpc_getBlock: params[0] int|str
            r = _rpc("qtcl_getBlock", [key], rpc_url)
            return _ok(r)

        # ── Recent transactions ─────────────────────────────────────────────
        elif name == "qtcl_get_recent_transactions":
            pp = min(int(args.get("per_page", 20)), 50)
            # server.py _rpc_getTransactions: flat dict (NOT list-wrapped) OR list[0] dict
            # Using flat dict — the handler checks isinstance(params, dict) first.
            p2: Dict[str, Any] = {"page": 0, "per_page": pp}
            addr2 = args.get("address", "")
            if addr2:
                p2["address"] = addr2
            r = _rpc("qtcl_getTransactions", p2, rpc_url)
            return _ok(r)

        # ── Quantum metrics ─────────────────────────────────────────────────
        elif name == "qtcl_get_quantum_metrics":
            r = _rpc("qtcl_getQuantumMetrics", [], rpc_url)
            return _ok(r)

        # ── Oracle registry ─────────────────────────────────────────────────
        elif name == "qtcl_get_oracle_registry":
            lim = int(args.get("limit", 10))
            # server.py _rpc_getOracleRegistry: params[0] dict  (checked via list path)
            r = _rpc("qtcl_getOracleRegistry", [{"limit": lim}], rpc_url)
            return _ok(r)

        # ── Peers ───────────────────────────────────────────────────────────
        elif name == "qtcl_get_peers":
            lim2 = int(args.get("limit", 20))
            # server.py _rpc_getPeers: params[0] int (list path) or params["limit"] dict path
            r = _rpc("qtcl_getPeers", [lim2], rpc_url)
            return _ok(r)

        # ── Price / network vitals ──────────────────────────────────────────
        elif name == "qtcl_get_price":
            r = _rpc("qtcl_getQuantumMetrics", [], rpc_url)
            return _ok({
                "note":    "QTCL has no public USD exchange. Returning quantum coherence metrics as network value proxy.",
                "metrics": r,
            })

        else:
            return _err(f"Unknown tool: {name!r}. Call tools/list for valid tool names.")

    except RuntimeError as rpc_err:
        logger.warning(f"[MCP] Tool {name!r} RPC error: {rpc_err}")
        return _err(str(rpc_err))
    except Exception as exc:
        logger.exception(f"[MCP] Tool {name!r} unexpected error: {exc}")
        return _err(f"Unexpected error: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# §6  RESOURCE DISPATCH — reads live data for resource URIs
# ──────────────────────────────────────────────────────────────────────────────

def _read_resource(uri: str, rpc_url: str) -> str:
    """Return resource content as a string, or raise ValueError for unknown URI."""
    if uri == "chain://height":
        r = _rpc("qtcl_getBlockHeight", [], rpc_url)
        return str(r.get("height", r) if isinstance(r, dict) else r)
    elif uri == "chain://health":
        return json.dumps(_rpc("qtcl_getHealth", [], rpc_url), indent=2, default=str)
    elif uri == "price://qtcl-usd":
        return json.dumps(_rpc("qtcl_getQuantumMetrics", [], rpc_url), indent=2, default=str)
    elif uri == "docs://capability":
        return json.dumps({
            "name":         "QTCL — Quantum Temporal Coherence Ledger",
            "version":      MCP_SERVER_VERSION,
            "protocol":     f"JSON-RPC 2.0 + MCP {_PROTO_PREFERRED}",
            "tools":        len(_TOOLS),
            "resources":    len(_RESOURCES),
            "prompts":      len(_PROMPTS),
            "transports":   ["streamable-http (MCP 2025-06-18)", "SSE (MCP 2024-11-05)"],
            "cryptography": "Schnorr-Γ over PSL(2,R) — Fiat-Shamir on hyperbolic Fuchsian group",
            "economics": {
                "native_unit":      "QTCL",
                "base_unit":        "qsat (1 QTCL = 100 qsat)",
                "fee_per_tx":       "1 qsat flat",
                "block_time_secs":  18,
                "lattice":          "{8,3} hyperbolic Poincaré tessellation | 106,496 pseudoqubits",
                "consensus":        "W-state oracle BFT (5-oracle, 3-of-5 majority)",
            },
        }, indent=2)
    else:
        raise ValueError(f"Unknown resource URI: {uri!r}")


# ──────────────────────────────────────────────────────────────────────────────
# §7  PROMPT DISPATCH
# ──────────────────────────────────────────────────────────────────────────────

def _get_prompt(name: str, args: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return MCP prompt messages list."""
    if name == "wallet_helper":
        task = args.get("task", "create")
        if task == "send":
            text = (
                "You are helping a user send QTCL. The workflow is:\n"
                "1. qtcl_get_balance(from_address) — verify sufficient funds\n"
                "2. Choose a nonce (e.g. int(time.time() * 1e9) — nanosecond timestamp)\n"
                "3. Build signing payload: JSON.dumps({\"sender\": from_addr, "
                "\"recipient\": to_addr, \"amount\": amount_float, \"nonce\": nonce_int}, "
                "sort_keys=True)\n"
                "4. Compute SHA3-256 hash of that payload → 64-char hex\n"
                "5. qtcl_sign_message(message_hex=that_hash, private_key=key) — authorize\n"
                "6. qtcl_send_transaction(from_address, to_address, amount, "
                "signature=full_sig_json, public_key=key, nonce=same_nonce) — submit\n"
                "CRITICAL: The nonce in step 3 MUST match the nonce in step 6.\n"
                "The amount in step 3 MUST match the amount in step 6 (both in QTCL, not qsat).\n"
                "Flat fee: 1 qsat. Finality: ~18 seconds."
            )
        else:
            text = (
                "You are helping a user create a QTCL post-quantum wallet.\n"
                "Call qtcl_create_wallet to generate a HypΓ keypair, then present:\n"
                "  • address (64-char hex) — the public receiving address\n"
                "  • public_key — share freely\n"
                "  • private_key — CRITICAL: user must save this offline.\n"
                "                  The server NEVER retains private keys.\n"
                "Cryptography: Schnorr-Γ over PSL(2,R) with SHA3-256² address derivation."
            )
        return [{"role": "user", "content": {"type": "text", "text": text}}]
    raise ValueError(f"Unknown prompt: {name!r}")


# ──────────────────────────────────────────────────────────────────────────────
# §8  JSON-RPC MESSAGE HANDLER — the core brain
# ──────────────────────────────────────────────────────────────────────────────

def _handle_jsonrpc(
    msg: Dict[str, Any],
    session_id: Optional[str],
    rpc_url: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Handle a single JSON-RPC 2.0 message and return (response_or_None, new_session_id_or_None).

    Returns (None, None) for pure notifications (no id, no response expected).
    Returns (response_dict, maybe_sid) for requests.
    The new_session_id is only non-None on initialize — it's the session to assign.
    """
    method   = msg.get("method", "")
    params   = msg.get("params") or {}
    req_id   = msg.get("id")          # None means notification
    is_notif = req_id is None         # JSON-RPC 2.0: id absent = notification

    # ── Notification: no response, ever ─────────────────────────────────────
    # notifications/initialized, $/cancelRequest, etc.
    if is_notif:
        return None, None

    def _ok(result: Any) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "result": result, "id": req_id}

    def _err(code: int, message: str) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "error": {"code": code, "message": message}, "id": req_id}

    # ── initialize ───────────────────────────────────────────────────────────
    if method == "initialize":
        client_proto = params.get("protocolVersion", _PROTO_PREFERRED) if isinstance(params, dict) else _PROTO_PREFERRED
        # Negotiate: echo client version if supported; offer our preferred otherwise
        negotiated   = client_proto if client_proto in _SUPPORTED_PROTOS else _PROTO_PREFERRED
        new_sid      = session_id or str(uuid.uuid4())
        return _ok({
            "protocolVersion": negotiated,
            "serverInfo":      {"name": MCP_SERVER_NAME, "version": MCP_SERVER_VERSION},
            "capabilities":    {
                "tools":     {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
                "prompts":   {"listChanged": False},
                "logging":   {},
            },
        }), new_sid

    # ── ping ─────────────────────────────────────────────────────────────────
    if method == "ping":
        return _ok({}), None

    # ── tools/list ───────────────────────────────────────────────────────────
    if method == "tools/list":
        return _ok({"tools": _TOOLS}), None

    # ── tools/call ───────────────────────────────────────────────────────────
    if method == "tools/call":
        if not isinstance(params, dict):
            return _err(-32602, "params must be an object for tools/call"), None
        tool_name = params.get("name", "")
        tool_args = params.get("arguments") or {}
        if not tool_name:
            return _err(-32602, "tools/call requires params.name"), None
        result = _dispatch_tool(tool_name, tool_args if isinstance(tool_args, dict) else {}, rpc_url)
        return _ok(result), None

    # ── resources/list ───────────────────────────────────────────────────────
    if method == "resources/list":
        return _ok({"resources": _RESOURCES}), None

    # ── resources/read ───────────────────────────────────────────────────────
    if method == "resources/read":
        uri = (params.get("uri", "") if isinstance(params, dict) else "")
        if not uri:
            return _err(-32602, "resources/read requires params.uri"), None
        try:
            content = _read_resource(uri, rpc_url)
            mime    = "application/json" if content.lstrip().startswith("{") else "text/plain"
            return _ok({"contents": [{"uri": uri, "mimeType": mime, "text": content}]}), None
        except ValueError as ve:
            return _err(-32002, str(ve)), None
        except RuntimeError as re:
            return _err(-32603, str(re)), None

    # ── prompts/list ─────────────────────────────────────────────────────────
    if method == "prompts/list":
        return _ok({"prompts": _PROMPTS}), None

    # ── prompts/get ──────────────────────────────────────────────────────────
    if method == "prompts/get":
        pname = (params.get("name", "") if isinstance(params, dict) else "")
        pargs = (params.get("arguments") or {}) if isinstance(params, dict) else {}
        if not pname:
            return _err(-32602, "prompts/get requires params.name"), None
        try:
            messages = _get_prompt(pname, pargs if isinstance(pargs, dict) else {})
            return _ok({"description": f"Prompt: {pname}", "messages": messages}), None
        except ValueError as ve:
            return _err(-32002, str(ve)), None

    # ── completion/complete ──────────────────────────────────────────────────
    # (Claude.ai may send this; return empty completion to avoid method-not-found crash)
    if method == "completion/complete":
        return _ok({"completion": {"values": [], "hasMore": False}}), None

    # ── Unknown method ───────────────────────────────────────────────────────
    return _err(-32601, f"Method not found: {method!r}"), None


# ──────────────────────────────────────────────────────────────────────────────
# §9  CORS HEADERS
# ──────────────────────────────────────────────────────────────────────────────

_CORS: Dict[str, str] = {
    "Access-Control-Allow-Origin":      "*",
    "Access-Control-Allow-Methods":     "GET, POST, OPTIONS, DELETE",
    "Access-Control-Allow-Headers":     "Content-Type, Accept, Mcp-Session-Id, Authorization, Last-Event-ID",
    "Access-Control-Expose-Headers":    "Mcp-Session-Id",
    "Access-Control-Max-Age":           "86400",
}

def _add_cors(r: Any) -> Any:
    for k, v in _CORS.items():
        r.headers[k] = v
    return r


# ──────────────────────────────────────────────────────────────────────────────
# §10  REGISTER_MCP_ROUTES — the public API, called from server.py
# ──────────────────────────────────────────────────────────────────────────────

def register_mcp_routes(app: Any, rpc_url: Optional[str] = None) -> bool:
    """
    Register all MCP routes on a Flask app.

    In server.py, called as: register_mcp_routes(app)
    Detects if /mcp is already registered and skips to prevent double-registration.
    Returns True on success, False on error.
    """
    existing = {rule.rule for rule in app.url_map.iter_rules()}
    if "/mcp" in existing:
        logger.info("[MCP] ✅ /mcp already registered — skipping re-registration")
        return True
    try:
        _wire_routes(app, rpc_url or QTCL_RPC_URL)
        return True
    except Exception as exc:
        logger.exception(f"[MCP] ❌ Route registration failed: {exc}")
        return False


def _wire_routes(app: Any, rpc_url: str) -> None:
    """Wire all MCP HTTP routes onto the Flask app. Called exactly once."""
    from flask import request as _req, Response as _Resp, jsonify as _json

    # Per-process session store — fork safe (each gunicorn worker has own dict)
    # Keyed by session_id string → queue.Queue for SSE legacy path
    _sse_sessions: Dict[str, queue.Queue] = {}
    _sse_lock = threading.Lock()

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _json_resp(data: Any, status: int = 200, extra_headers: Optional[Dict] = None) -> Any:
        body = json.dumps(data, default=str).encode("utf-8")
        r    = _Resp(body, status=status, mimetype="application/json; charset=utf-8")
        _add_cors(r)
        if extra_headers:
            for k, v in extra_headers.items():
                r.headers[k] = v
        return r

    def _empty_resp(status: int = 200) -> Any:
        r = _Resp("", status=status, mimetype="text/plain")
        _add_cors(r)
        return r

    def _process_body(raw_body: bytes, session_id: Optional[str]) -> Tuple[Any, Optional[str]]:
        """
        Parse raw bytes as JSON-RPC (single object or batch array).
        Returns (response_payload, new_session_id).
        response_payload may be dict, list, or None.
        """
        if not raw_body:
            return {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Empty request body"}, "id": None}, None

        try:
            parsed = json.loads(raw_body.decode("utf-8"))
        except Exception as e:
            return {"jsonrpc": "2.0", "error": {"code": -32700, "message": f"JSON parse error: {e}"}, "id": None}, None

        new_sid = session_id

        # ── Batch ────────────────────────────────────────────────────────────
        if isinstance(parsed, list):
            responses = []
            for item in parsed:
                if not isinstance(item, dict):
                    responses.append({"jsonrpc": "2.0", "error": {"code": -32600, "message": "Invalid request in batch"}, "id": None})
                    continue
                resp, maybe_sid = _handle_jsonrpc(item, new_sid, rpc_url)
                if maybe_sid:
                    new_sid = maybe_sid
                if resp is not None:
                    responses.append(resp)
            return responses if responses else None, new_sid

        # ── Single ───────────────────────────────────────────────────────────
        if not isinstance(parsed, dict):
            return {"jsonrpc": "2.0", "error": {"code": -32600, "message": "Request must be an object"}, "id": None}, None

        resp, maybe_sid = _handle_jsonrpc(parsed, new_sid, rpc_url)
        if maybe_sid:
            new_sid = maybe_sid
        return resp, new_sid

    # ── CORS preflight ───────────────────────────────────────────────────────

    @app.route("/mcp", methods=["OPTIONS"])
    @app.route("/mcp/", methods=["OPTIONS"])
    @app.route("/mcp/<path:subpath>", methods=["OPTIONS"])
    def _mcp_preflight(subpath: str = "") -> Any:
        return _empty_resp(204)

    # ── POST /mcp — Streamable HTTP primary channel (MCP 2025-06-18) ─────────

    @app.route("/mcp", methods=["POST"])
    def _mcp_post() -> Any:
        sid      = (_req.headers.get("Mcp-Session-Id") or "").strip() or None
        raw_body = _req.get_data()
        payload, new_sid = _process_body(raw_body, sid)

        extra: Dict[str, str] = {}
        if new_sid:
            extra["Mcp-Session-Id"] = new_sid

        # Notification-only batch or pure notification → 202 with empty body
        if payload is None:
            r = _Resp("", status=202, mimetype="text/plain")
            _add_cors(r)
            for k, v in extra.items():
                r.headers[k] = v
            return r

        return _json_resp(payload, 200, extra)

    # ── GET /mcp — server info (stateless handshake / capability probe) ──────

    @app.route("/mcp", methods=["GET"])
    def _mcp_get() -> Any:
        """
        Claude.ai probes GET /mcp to discover server capabilities before
        starting the initialize handshake. Return a JSON capability document.
        This is NOT a JSON-RPC response — it's a plain JSON discovery doc.
        """
        info = {
            "server":          MCP_SERVER_NAME,
            "version":         MCP_SERVER_VERSION,
            "protocolVersions": [_PROTO_PREFERRED, _PROTO_LEGACY],
            "transport":       "streamable-http",
            "endpoint":        "/mcp",
            "capabilities": {
                "tools":     {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
                "prompts":   {"listChanged": False},
                "logging":   {},
            },
            "toolCount":    len(_TOOLS),
            "resourceCount": len(_RESOURCES),
            "promptCount":  len(_PROMPTS),
        }
        return _json_resp(info)

    # ── DELETE /mcp — session teardown (MCP 2025-06-18) ─────────────────────

    @app.route("/mcp", methods=["DELETE"])
    def _mcp_delete() -> Any:
        sid = (_req.headers.get("Mcp-Session-Id") or "").strip()
        if sid:
            with _sse_lock:
                q = _sse_sessions.pop(sid, None)
            if q:
                try:
                    q.put_nowait(None)   # Signal SSE stream to close
                except queue.Full:
                    pass
        return _empty_resp(200)

    # ── GET /mcp/sse — Legacy SSE stream (MCP 2024-11-05 backward compat) ────

    @app.route("/mcp/sse", methods=["GET"])
    def _mcp_sse() -> Any:
        sid = str(uuid.uuid4())
        q: queue.Queue = queue.Queue(maxsize=200)
        with _sse_lock:
            _sse_sessions[sid] = q

        def _generate():
            # MCP 2024-11-05: first event MUST be "endpoint" with the message URL
            # Include session_id so client can POST to the right session
            yield f"event: endpoint\ndata: /mcp/message?session_id={sid}\n\n"
            try:
                while True:
                    try:
                        msg = q.get(timeout=25.0)
                        if msg is None:   # teardown sentinel
                            break
                        yield f"event: message\ndata: {json.dumps(msg, default=str)}\n\n"
                    except queue.Empty:
                        yield ": heartbeat\n\n"   # keep-alive ping
            except GeneratorExit:
                pass
            finally:
                with _sse_lock:
                    _sse_sessions.pop(sid, None)

        r = _Resp(
            _generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control":     "no-cache",
                "Connection":        "keep-alive",
                "X-Accel-Buffering": "no",      # Nginx: disable proxy buffering
                "Mcp-Session-Id":    sid,
                **_CORS,
            },
        )
        return r

    # ── POST /mcp/message — Legacy message channel (MCP 2024-11-05) ──────────

    @app.route("/mcp/message", methods=["POST"])
    def _mcp_message() -> Any:
        sid      = (_req.args.get("session_id") or "").strip()
        raw_body = _req.get_data()

        with _sse_lock:
            q = _sse_sessions.get(sid)

        if not q:
            return _json_resp(
                {"jsonrpc": "2.0", "error": {"code": -32001, "message": f"Session not found: {sid!r}"}, "id": None},
                status=404,
            )

        payload, new_sid = _process_body(raw_body, sid)

        # Push response to SSE queue
        if payload is not None:
            msgs = payload if isinstance(payload, list) else [payload]
            for m in msgs:
                try:
                    q.put_nowait(m)
                except queue.Full:
                    return _json_resp({"error": "SSE queue full"}, 429)

        return _json_resp({"status": "accepted", "session_id": sid}, 202)

    # ── GET /mcp/health ───────────────────────────────────────────────────────

    @app.route("/mcp/health", methods=["GET"])
    def _mcp_health() -> Any:
        with _sse_lock:
            n_sse = len(_sse_sessions)
        return _json_resp({
            "status":           "ok",
            "server":           MCP_SERVER_NAME,
            "version":          MCP_SERVER_VERSION,
            "protocols":        [_PROTO_PREFERRED, _PROTO_LEGACY],
            "transport":        "streamable-http + SSE",
            "tools":            len(_TOOLS),
            "resources":        len(_RESOURCES),
            "prompts":          len(_PROMPTS),
            "active_sse":       n_sse,
            "rpc_url":          rpc_url,
            "ts":               time.time(),
        })

    # ── GET /mcp/capability — Full capability document ────────────────────────

    @app.route("/mcp/capability", methods=["GET"])
    def _mcp_capability() -> Any:
        return _json_resp({
            "name":         MCP_SERVER_NAME,
            "version":      MCP_SERVER_VERSION,
            "protocols":    [_PROTO_PREFERRED, _PROTO_LEGACY],
            "transports":   {
                _PROTO_PREFERRED: {"type": "streamable-http", "endpoint": "/mcp", "methods": ["POST", "GET", "DELETE"]},
                _PROTO_LEGACY:    {"type": "sse", "endpoint": "/mcp/sse", "message_endpoint": "/mcp/message"},
            },
            "tools":     _TOOLS,
            "resources": _RESOURCES,
            "prompts":   _PROMPTS,
            "economics": {
                "native_unit": "QTCL",
                "base_unit":   "qsat (1 QTCL = 100 qsat)",
                "fee_per_tx":  "1 qsat flat",
                "block_time":  "~18 seconds",
            },
            "cryptography": "Schnorr-Γ over PSL(2,R) | GeodesicLWE | SHA3-256² addresses",
            "lattice":       "{8,3} hyperbolic Poincaré tessellation | 106,496 pseudoqubits",
        })

    logger.info(
        f"[MCP] ✅ MCP routes registered — "
        f"Streamable HTTP (2025-06-18) + SSE legacy (2024-11-05) | "
        f"{len(_TOOLS)} tools | {len(_RESOURCES)} resources | {len(_PROMPTS)} prompts | "
        f"rpc_url={rpc_url}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# §11  stdio TRANSPORT — Claude Desktop / Cursor / Windsurf / CLI agents
# ──────────────────────────────────────────────────────────────────────────────

def run_stdio_transport(rpc_url: str = QTCL_RPC_URL) -> None:
    """
    Run MCP server over stdio for local agent integration.

    Usage:
        python mcp_flask_adapter.py --transport stdio [--rpc-url URL]

    Requires: pip install 'mcp>=1.9.0'
    """
    try:
        from mcp.server.fastmcp import FastMCP as _FastMCP
    except ImportError:
        print("[MCP] ERROR: MCP Python SDK not installed. Run: pip install 'mcp>=1.9.0'", file=sys.stderr)
        sys.exit(1)

    mcp = _FastMCP(MCP_SERVER_NAME)

    def _r(method: str, params: Any = None) -> Any:
        return _rpc(method, params, rpc_url)

    @mcp.tool()
    async def qtcl_create_wallet(label: str = "") -> str:
        """Create a new post-quantum QTCL wallet (HypΓ Schnorr-Γ keypair)."""
        r = _r("qtcl_hyp_generateKeypair", {})
        if "created_at" not in r:
            r["created_at"] = r.pop("timestamp", "")
        r.setdefault("crypto", "HypΓ Schnorr-Γ / PSL(2,R)")
        if label:
            r["label"] = label
        return json.dumps(r, indent=2, default=str)

    @mcp.tool()
    async def qtcl_sign_message(message_hex: str, private_key: str) -> str:
        """Sign a 32-byte hash with HypΓ private key (Schnorr-Γ)."""
        return json.dumps(_r("qtcl_hyp_signMessage", {"message": message_hex, "private_key": private_key}), indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_balance(address: str) -> str:
        """Check QTCL balance for any address."""
        return json.dumps(_r("qtcl_getBalance", [address]), indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_utxos(address: str, limit: int = 1000) -> str:
        """List UTXOs for an address."""
        return json.dumps(_r("qtcl_getUTXOs", [{"address": address, "limit": limit}]), indent=2, default=str)

    @mcp.tool()
    async def qtcl_send_transaction(from_address: str, to_address: str, amount: float,
                                     memo: str = "", signature: str = "", public_key: str = "") -> str:
        """Submit a signed UTXO transaction. Flat fee: 1 qsat."""
        p: Dict[str, Any] = {"from_address": from_address, "to_address": to_address, "amount": amount}
        for k, v in (("memo", memo), ("signature", signature), ("public_key", public_key)):
            if v:
                p[k] = v
        return json.dumps(_r("qtcl_submitTransaction", [p]), indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_transaction(tx_hash: str) -> str:
        """Look up a transaction by hash."""
        return json.dumps(_r("qtcl_getTransaction", [tx_hash]), indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_chain_info() -> str:
        """Current blockchain state: height, mempool, health."""
        return json.dumps({
            "chain":   _r("qtcl_getBlockHeight"),
            "mempool": _r("qtcl_getMempoolStats"),
            "health":  _r("qtcl_getHealth"),
        }, indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_block(height: int = -1, hash: str = "") -> str:
        """Get block by height or hash. Omit both for latest."""
        if hash:
            key: Any = hash
        elif height >= 0:
            key = height
        else:
            tip = _r("qtcl_getBlockHeight")
            key = tip.get("height", 0) if isinstance(tip, dict) else int(tip or 0)
        return json.dumps(_r("qtcl_getBlock", [key]), indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_recent_transactions(address: str = "", per_page: int = 20) -> str:
        """Recent transactions, optionally filtered by address."""
        p: Dict[str, Any] = {"page": 0, "per_page": min(per_page, 50)}
        if address:
            p["address"] = address
        return json.dumps(_r("qtcl_getTransactions", p), indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_quantum_metrics() -> str:
        """Live quantum coherence metrics."""
        return json.dumps(_r("qtcl_getQuantumMetrics"), indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_oracle_registry(limit: int = 10) -> str:
        """List registered quantum oracle nodes."""
        return json.dumps(_r("qtcl_getOracleRegistry", [{"limit": limit}]), indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_peers(limit: int = 20) -> str:
        """List active P2P peers."""
        return json.dumps(_r("qtcl_getPeers", [limit]), indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_price() -> str:
        """QTCL quantum coherence metrics as network value proxy."""
        return json.dumps({
            "note":    "QTCL has no public USD exchange.",
            "metrics": _r("qtcl_getQuantumMetrics"),
        }, indent=2, default=str)

    @mcp.resource("chain://height")
    async def _res_height() -> str:
        r = _r("qtcl_getBlockHeight")
        return str(r.get("height", r) if isinstance(r, dict) else r)

    @mcp.resource("chain://health")
    async def _res_health() -> str:
        return json.dumps(_r("qtcl_getHealth"), indent=2, default=str)

    @mcp.prompt()
    def wallet_helper(task: str = "create") -> str:
        msgs = _get_prompt("wallet_helper", {"task": task})
        return msgs[0]["content"]["text"]

    print(f"[MCP] ▶ stdio transport → rpc_url={rpc_url}", file=sys.stderr)
    mcp.run(transport="stdio")


# ──────────────────────────────────────────────────────────────────────────────
# §12  STANDALONE ENTRYPOINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    ap = argparse.ArgumentParser(description="QTCL MCP Adapter v5.0")
    ap.add_argument("--transport", choices=["stdio", "http"], default="http")
    ap.add_argument("--port",      type=int, default=int(os.environ.get("MCP_PORT", "8000")))
    ap.add_argument("--rpc-url",   default=QTCL_RPC_URL)
    args = ap.parse_args()

    if args.transport == "stdio":
        run_stdio_transport(args.rpc_url)
    else:
        from flask import Flask as _Flask
        _app = _Flask(__name__)
        _wire_routes(_app, args.rpc_url)

        print()
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║  QTCL MCP Adapter v5.0 — Standalone HTTP Mode                        ║")
        print("║  MCP 2025-06-18 Streamable HTTP + MCP 2024-11-05 SSE Legacy           ║")
        print("║  Pure Flask | Zero ASGI | Gunicorn-safe | Fork-safe                   ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")
        print(f"  RPC backend  : {args.rpc_url}")
        print(f"  Endpoints    :")
        print(f"    POST/GET/DELETE /mcp          — Streamable HTTP (MCP 2025-06-18)")
        print(f"    GET             /mcp/sse       — Legacy SSE stream (MCP 2024-11-05)")
        print(f"    POST            /mcp/message   — Legacy message channel")
        print(f"    GET             /mcp/health    — Health + capability summary")
        print(f"    GET             /mcp/capability — Full tool/resource/prompt manifest")
        print(f"  Tools: {len(_TOOLS)} | Resources: {len(_RESOURCES)} | Prompts: {len(_PROMPTS)}")
        print()
        _app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
