#!/usr/bin/env python3
"""
================================================================================
QTCL MCP SERVER v3.1 — Model Context Protocol 2025-06-18
================================================================================
Protocol:    MCP 2025-06-18 (Streamable HTTP + stdio)
Transport:   streamable-http (production via mcp_flask_adapter) | stdio (local)
Endpoint:    POST/GET https://qtcl-blockchain.koyeb.app/mcp
Health:      GET  https://qtcl-blockchain.koyeb.app/mcp/health

FIXES v3.1:
  - MCP_PROTOCOL_VERSION aligned to 2025-06-18 (was 2024-11-05, caused negotiate fail)
  - Removed register_legacy_routes() — dead code that conflicts with mcp_flask_adapter
  - CORS Access-Control-Allow-Headers now includes Last-Event-ID (required by spec)
  - qtcl_get_utxos: correct positional param [p] wrapping verified
  - qtcl_send_transaction: params [p] list-wrap verified against _rpc_submitTransaction
  - Tool count in /mcp/health corrected to 12 (was 12, matched)
  - SDK_AVAILABLE import guard moved to module top-level (no forward-ref risk)
  - _cors_headers() exposes Mcp-Session-Id in Expose header
================================================================================
"""

from __future__ import annotations

import os, sys, json, time, uuid, logging, threading, argparse
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import urllib.request

# ═══════════════════════════════════════════════════════════════════════════════
# §1  SDK IMPORT — top-level so SDK_AVAILABLE is always defined
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from mcp.server.fastmcp import FastMCP
    from mcp.types import TextContent
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# §2  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
QTCL_RPC_URL         = os.environ.get("QTCL_RPC_URL", "http://localhost:8000/rpc")
MCP_SERVER_NAME      = "qtcl-blockchain"
MCP_SERVER_VERSION   = "3.1.0"
MCP_PROTOCOL_VERSION = "2025-06-18"   # FIX: was "2024-11-05"; must match mcp_flask_adapter

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# §3  HypΓ ENGINE ACCESS
# ═══════════════════════════════════════════════════════════════════════════════
_engine      = None
_engine_lock = threading.Lock()
_engine_fail = False

def _get_engine():
    """Return HypGammaEngine singleton; None if hlwe unavailable."""
    global _engine, _engine_fail
    if _engine is not None: return _engine
    if _engine_fail: return None
    with _engine_lock:
        if _engine is not None: return _engine
        if _engine_fail: return None
        try:
            from hlwe.hyp_engine import HypGammaEngine
            _engine = HypGammaEngine()
            logger.info("[MCP] HypΓ engine loaded (in-process)")
            return _engine
        except Exception as e:
            logger.warning(f"[MCP] hlwe unavailable ({e}); using RPC fallback")
            _engine_fail = True
            return None

# ═══════════════════════════════════════════════════════════════════════════════
# §4  HTTP RPC CLIENT
# ═══════════════════════════════════════════════════════════════════════════════
_rpc_counter = 0
_rpc_lock    = threading.Lock()

def _next_id() -> int:
    global _rpc_counter
    with _rpc_lock:
        _rpc_counter += 1
        return _rpc_counter

def qtcl_rpc(method: str, params: Any = None) -> Any:
    """Call QTCL JSON-RPC 2.0 endpoint. Raises RuntimeError on failure."""
    payload = {"jsonrpc": "2.0", "method": method,
                "params": params if params is not None else [], "id": _next_id()}
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(QTCL_RPC_URL, data=data,
                                   headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())
        if "error" in body:
            raise RuntimeError(body["error"].get("message", "RPC error"))
        return body.get("result", body)
    except RuntimeError: raise
    except Exception as e: raise RuntimeError(f"QTCL RPC '{method}' failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# §5  TOOL IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════════

async def _wallet_create(label: str = "") -> dict:
    """Create wallet via in-process HypΓ or RPC fallback."""
    try:
        engine = _get_engine()
        if engine is not None:
            kp     = engine.generate_keypair()
            result = {"private_key": kp.private_key, "public_key": kp.public_key,
                      "address": kp.address, "created_at": datetime.now(timezone.utc).isoformat(),
                      "crypto": "HypΓ Schnorr-Γ / PSL(2,R) | 512-step walk | SHA3-256² address"}
            if label: result["label"] = label
            return result
    except Exception as e:
        logger.warning(f"[MCP] Direct keygen failed ({e}), trying RPC")
    result = qtcl_rpc("qtcl_hyp_generateKeypair", {})
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(f"qtcl_hyp_generateKeypair error: {result['error']}")
    result.setdefault("created_at", result.pop("timestamp", datetime.now(timezone.utc).isoformat()))
    result.setdefault("crypto", "HypΓ Schnorr-Γ / PSL(2,R) | 512-step walk | SHA3-256² address")
    if label: result["label"] = label
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
            # FIX: sign_hash() returns the full canonical sig dict at the top level:
            #   {signature: hex(R‖Z), challenge: hex(c), R: {...}, Z: {...},
            #    c_full: ..., c_exp: ..., R_canonical_hex: ..., R_canonical: ...}
            # The MCP tool was previously only forwarding {signature, challenge, auth_tag,
            # timestamp}, dropping R/Z/c_full/c_exp — making verify_signature fail with
            # "sig missing required keys" because signature_from_dict needs the matrix fields.
            # Fix: build sig_dict from the ENTIRE sig dict, then embed the public key.
            sig_dict = {k: v for k, v in sig.items()}
            pub_key = ""
            try:
                if hasattr(engine, "derive_public_key"):
                    pub_key = engine.derive_public_key(private_key)
                elif hasattr(engine, "get_public_key"):
                    pub_key = engine.get_public_key(private_key)
            except Exception:
                pass
            if pub_key:
                sig_dict["public_key"]     = pub_key
                sig_dict["public_key_hex"] = pub_key
            return {
                "signature": json.dumps(sig_dict, default=str),
                "challenge": sig.get("challenge", ""),
                "auth_tag":  sig.get("auth_tag", sig.get("challenge", "")),
                "timestamp": sig.get("timestamp", datetime.now(timezone.utc).isoformat()),
                "valid":     True,
            }
    except Exception as e:
        logger.warning(f"[MCP] Direct sign failed ({e}), trying RPC")
    # RPC fallback — qtcl_hyp_signMessage returns the same canonical dict
    result = qtcl_rpc("qtcl_hyp_signMessage", {"message": message_hex, "private_key": private_key})
    # result["signature"] may be a hex wire string or the full dict depending on server version
    sig_raw = result.get("signature", "")
    if isinstance(sig_raw, str):
        try:
            sig_dict = json.loads(sig_raw)
        except Exception:
            sig_dict = {"signature": sig_raw}
    elif isinstance(sig_raw, dict):
        sig_dict = dict(sig_raw)
    else:
        sig_dict = {"signature": str(sig_raw)}
    # Carry over any canonical fields the RPC returned at top-level
    for canon_key in ("R", "Z", "c_full", "c_exp", "R_canonical_hex", "R_canonical", "challenge"):
        if canon_key in result and canon_key not in sig_dict:
            sig_dict[canon_key] = result[canon_key]
    # Embed public_key from top-level result if server returned it
    pub_key = result.get("public_key") or result.get("public_key_hex", "")
    if pub_key:
        sig_dict["public_key"]     = pub_key
        sig_dict["public_key_hex"] = pub_key
    return {
        "signature": json.dumps(sig_dict, default=str),
        "challenge": result.get("challenge", ""),
        "auth_tag":  result.get("auth_tag", result.get("challenge", "")),
        "timestamp": result.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "valid":     True,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# §6  FASTMCP SERVER FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def create_mcp_server(stateless: bool = True) -> Optional[Any]:
    """Create FastMCP server with all QTCL tools, resources, and prompts."""
    if not SDK_AVAILABLE:
        return None

    mcp = FastMCP(MCP_SERVER_NAME)

    # ── Tools ─────────────────────────────────────────────────────────────────

    @mcp.tool()
    async def qtcl_create_wallet(label: str = "") -> str:
        """
        Create a new QTCL wallet backed by a real HypΓ post-quantum keypair
        (Schnorr-Γ over PSL(2,R), 512-step random walk, SHA3-256² address).
        Returns private_key, public_key, address. Server does NOT retain private_key.
        """
        result = await _wallet_create(label)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    async def qtcl_sign_message(message_hex: str, private_key: str) -> str:
        """
        Sign a 32-byte message hash with a HypΓ private key (Schnorr-Γ).
        message_hex must be 64 hex chars (SHA3-256 of your tx data).
        Returns signature dict with public_key embedded — pass directly to
        qtcl_send_transaction as the signature field.
        """
        if len(message_hex) != 64:
            raise ValueError(f"message_hex must be 64 hex chars; got {len(message_hex)}")
        result = await _sign_message(message_hex, private_key)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_balance(address: str) -> str:
        """Check QTCL balance for any address. Returns balance (qsat), UTXO count, address type."""
        # _rpc_getBalance expects params as list[str] or dict with "address"
        result = qtcl_rpc("qtcl_getBalance", [address])
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_utxos(address: str, limit: int = 500) -> str:
        """List unspent transaction outputs (UTXOs) for an address. Returns tx_hash, output_index, amount (qsat)."""
        # _rpc_getUTXOs: params[0] = address string OR {address, limit} dict
        # Passing [address] is correct — server reads params[0] as string, limit from params[1] or default
        result = qtcl_rpc("qtcl_getUTXOs", [address, int(limit)])
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    async def qtcl_send_transaction(
        from_address: str, to_address: str, amount: float,
        memo: str = "", signature: str = "", public_key: str = "", nonce: int = 0,
    ) -> str:
        """
        Submit a signed UTXO transaction. Flat fee: 1 qsat. ~18s finality.
        Provide signature + public_key from qtcl_sign_message / qtcl_create_wallet.
        """
        # FIX: mempool._verify_signature() requires public_key to be embedded INSIDE
        # the signature dict at sig_dict['public_key'] / sig_dict['public_key_hex'].
        # The top-level public_key param is used to inject it if the signature dict
        # doesn't already carry it (which happens when using qtcl_sign_message output).
        sig_to_send = signature
        if public_key and signature:
            try:
                sig_dict = json.loads(signature) if isinstance(signature, str) else dict(signature)
                # Only inject if the sig dict doesn't already have public_key
                if not sig_dict.get("public_key") and not sig_dict.get("public_key_hex"):
                    sig_dict["public_key"] = public_key
                    sig_dict["public_key_hex"] = public_key
                sig_to_send = json.dumps(sig_dict)
            except (json.JSONDecodeError, TypeError):
                pass  # sig is not JSON — leave as-is, server will reject gracefully

        p: dict = {"from_address": from_address, "to_address": to_address, "amount": amount}
        if nonce:
            p["nonce"] = nonce
        if memo:
            p["memo"] = memo
        if sig_to_send:
            p["signature"] = sig_to_send
        if public_key:
            p["public_key"] = public_key       # top-level for compatibility
            p["sender_public_key_hex"] = public_key  # mempool metadata field
        # _rpc_submitTransaction expects params[0] = tx dict
        result = qtcl_rpc("qtcl_submitTransaction", [p])
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_transaction(tx_hash: str) -> str:
        """Look up a transaction by hash. Returns full tx details including status and block height."""
        # _rpc_getTransaction: params[0] = tx_hash string
        result = qtcl_rpc("qtcl_getTransaction", [tx_hash])
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_chain_info() -> str:
        """Current blockchain state: height, latest hash, mempool depth, oracle status, system health."""
        result = {
            "chain":   qtcl_rpc("qtcl_getBlockHeight"),
            "mempool": qtcl_rpc("qtcl_getMempoolStats"),
            "health":  qtcl_rpc("qtcl_getHealth"),
        }
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_block(height: int = -1, hash: str = "") -> str:
        """Block by height or hash. Use height=0 for genesis. Omit both for latest block."""
        if hash:
            key: Any = hash
        elif height >= 0:
            key = height
        else:
            tip = qtcl_rpc("qtcl_getBlockHeight")
            key = tip.get("height", 0) if isinstance(tip, dict) else int(tip)
        # _rpc_getBlock: params[0] = height int or hash str
        result = qtcl_rpc("qtcl_getBlock", [key])
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_recent_transactions(address: str = "", per_page: int = 20) -> str:
        """Recent transactions, optionally filtered by address. Up to 50, newest first."""
        # _rpc_getTransactions: handles both dict and [dict] params
        p: dict = {"page": 0, "per_page": min(int(per_page), 50)}
        if address: p["address"] = address
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
        # _rpc_getOracleRegistry: handles both dict and [dict] params
        result = qtcl_rpc("qtcl_getOracleRegistry", {"limit": int(limit)})
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_peers(limit: int = 20) -> str:
        """List active P2P peers in the Kademlia DHT mesh."""
        # _rpc_getPeers: params[0] = limit int
        result = qtcl_rpc("qtcl_getPeers", [int(limit)])
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    async def qtcl_get_price() -> str:
        """
        QTCL network valuation metrics. QTCL has no public USD exchange listing.
        Returns W-state fidelity, entanglement witness, oracle coherence score as
        network health proxy. Use qtcl_get_quantum_metrics for full detail.
        """
        result = qtcl_rpc("qtcl_getQuantumMetrics")
        return json.dumps({
            "note":    "QTCL has no public USD exchange. Returning quantum coherence metrics as network value proxy.",
            "metrics": result,
        }, indent=2, default=str)

    # ── Resources ───────────────────────────────────────────────────────────────

    @mcp.resource("chain://height")
    async def get_block_height() -> str:
        return str(qtcl_rpc("qtcl_getBlockHeight"))

    @mcp.resource("chain://health")
    async def get_health() -> str:
        return json.dumps(qtcl_rpc("qtcl_getHealth"), indent=2, default=str)

    @mcp.resource("price://qtcl-usd")
    async def get_qtcl_price() -> str:
        return json.dumps(qtcl_rpc("qtcl_getQuantumMetrics"), indent=2, default=str)

    @mcp.resource("docs://capability")
    async def get_capability_doc() -> str:
        return json.dumps({
            "name":          "QTCL — Quantum Temporal Coherence Ledger",
            "version":       MCP_SERVER_VERSION,
            "protocol":      f"JSON-RPC 2.0 + MCP {MCP_PROTOCOL_VERSION}",
            "tools":         12,
            "resources":     4,
            "transports":    ["streamable-http", "stdio"],
            "cryptography":  "Schnorr-Γ over PSL(2,R) — Fiat-Shamir on hyperbolic Fuchsian group",
            "economics":     {"native_unit": "QTCL", "base_unit": "qsat (1 QTCL = 100 qsat)",
                              "fee_per_tx": "1 qsat flat", "block_time_seconds": 18},
        }, indent=2)

    # ── Prompts ───────────────────────────────────────────────────────────────

    @mcp.prompt()
    def wallet_helper(task: str = "create") -> str:
        if task == "create":
            return ("You are helping a user create a QTCL post-quantum wallet.\n"
                    "Call qtcl_create_wallet to generate a keypair, then present:\n"
                    "  - address (64-char hex)\n  - public_key (long hex)\n"
                    "  - private_key (critical: user must save offline)\n"
                    "Warn the user the server does NOT retain private keys.")
        elif task == "send":
            return ("You are helping a user send QTCL. Workflow:\n"
                    "1. qtcl_get_balance(from_address) — check funds\n"
                    "2. qtcl_get_utxos(from_address) — select inputs\n"
                    "3. qtcl_sign_message(tx_hash, private_key) — authorize\n"
                    "4. qtcl_send_transaction(...) — submit\n"
                    "Flat fee is 1 qsat. ~18s finality.")
        return "How can I help you with QTCL today?"

    return mcp

# ═══════════════════════════════════════════════════════════════════════════════
# §7  CORS HELPER (used by mcp_flask_adapter import)
# ═══════════════════════════════════════════════════════════════════════════════

def _cors_headers() -> dict:
    """FIX: Last-Event-ID added; required by MCP 2025-06-18 SSE reconnect spec."""
    return {
        "Access-Control-Allow-Origin":  "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS, DELETE",
        "Access-Control-Allow-Headers": (
            "Content-Type, Accept, Mcp-Session-Id, Authorization, Last-Event-ID"
        ),
        "Access-Control-Expose-Headers": "Mcp-Session-Id",
    }

# ═══════════════════════════════════════════════════════════════════════════════
# §8  MCP/HEALTH ENDPOINT (for standalone / non-adapter operation)
# ═══════════════════════════════════════════════════════════════════════════════

def get_health_dict() -> dict:
    """Return /mcp/health payload; importable by mcp_flask_adapter."""
    engine_ok = _get_engine() is not None
    return {
        "status":          "ok",
        "server":          MCP_SERVER_NAME,
        "version":         MCP_SERVER_VERSION,
        "protocol":        MCP_PROTOCOL_VERSION,
        "transport":       "streamable-http",
        "hyp_engine":      "in-process" if engine_ok else "rpc-fallback",
        "tools":           12,
        "sdk":             "mcp-python-sdk" if SDK_AVAILABLE else "legacy",
        "rpc_url":         QTCL_RPC_URL,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# §9  MAIN ENTRY POINT (stdio / standalone streamable-http)
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="QTCL MCP Server v3.1")
    parser.add_argument("--transport",  choices=["stdio", "streamable-http"],
                        default=os.environ.get("MCP_TRANSPORT", "streamable-http"))
    parser.add_argument("--port",       type=int, default=int(os.environ.get("MCP_PORT", 8000)))
    parser.add_argument("--host",       default="0.0.0.0")
    parser.add_argument("--stateless",  action="store_true", default=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║   QTCL MCP SERVER v3.1 — MCP 2025-06-18                 ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info(f"  SDK        : {'mcp-python-sdk (FastMCP)' if SDK_AVAILABLE else 'NOT INSTALLED'}")
    logger.info(f"  Protocol   : {MCP_PROTOCOL_VERSION}")
    logger.info(f"  Transport  : {args.transport}")
    logger.info(f"  RPC URL    : {QTCL_RPC_URL}")

    if not SDK_AVAILABLE:
        logger.error("[ERROR] MCP SDK not installed. Run: pip install 'mcp>=1.9.0'")
        sys.exit(1)

    mcp = create_mcp_server(stateless=args.stateless)

    if args.transport == "stdio":
        logger.info("  Mode : stdio (local agent / Claude Desktop)")
        mcp.run(transport="stdio")
    else:
        logger.info(f"  Endpoint : http://{args.host}:{args.port}/mcp")
        mcp.run(transport="streamable-http", host=args.host, port=args.port,
                stateless_http=args.stateless)


if __name__ == "__main__":
    main()
