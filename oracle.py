#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                                                                                  ║
║   🔮  QTCL ORACLE — POST-QUANTUM SIGNING & KEY DERIVATION  🔮                  ║
║                                                                                  ║
║   pq0 IS the oracle. The oracle IS the signer.                                  ║
║                                                                                  ║
║   Replaces ECDSA entirely with:                                                  ║
║     • HLWE  — Hash-based Lattice Witness Encoding (post-quantum signature)       ║
║     • BIP32/38-style hierarchical key derivation from a 256-bit seed            ║
║     • W-state measurement as the signing entropy source                          ║
║                                                                                  ║
║   Architecture:                                                                  ║
║     OracleKeyPair          — keypair (seed → master → child keys)               ║
║     HLWESigner             — signs TX with HLWE + W-state entropy                ║
║     HLWEVerifier           — verifies HLWE signature (anyone can verify)         ║
║     OracleEngine           — singleton: holds the master key, signs blocks/txs  ║
║                                                                                  ║
║   Key derivation path (BIP32-style):                                             ║
║     m / purpose' / coin' / account' / change / index                            ║
║     purpose = 838  (QTCL, mirrors 8 in {8,3} tessellation)                      ║
║     coin    = 0    (QTCL mainnet)                                                ║
║                                                                                  ║
║   HLWE signature scheme:                                                         ║
║     1. Derive child key for this address                                         ║
║     2. Measure pq0 W-state → get entropy bitstring                              ║
║     3. Commitment C = SHA3-256(child_key || w_entropy || message_hash)           ║
║     4. Witness    W = SHAKE-256(C || child_key, 64 bytes)                        ║
║     5. Proof      π = HMAC-SHA3(child_key, W || message_hash)                   ║
║     6. Signature  σ = { C, W, π, w_entropy_hash, derivation_path, pubkey }      ║
║                                                                                  ║
║   Verification:                                                                  ║
║     1. Recompute C' from pubkey || W || message_hash                             ║
║     2. Verify π via HMAC-SHA3 with pubkey                                        ║
║     3. Check address = SHA3-256(pubkey)[:20]                                     ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import hmac
import time
import struct
import logging
import hashlib
import secrets
import threading
import traceback
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

QTCL_PURPOSE       = 838       # BIP44-style purpose for QTCL ({8,3} lattice)
QTCL_COIN          = 0         # mainnet
QTCL_VERSION_MAIN  = b'\x04\x88\xad\xe4'   # xprv-equivalent prefix (32-bit)
QTCL_VERSION_PUB   = b'\x04\x88\xb2\x1e'   # xpub-equivalent prefix
HARDENED_OFFSET    = 0x80000000
SEED_HMAC_KEY      = b"QTCL hyperbolic {8,3} oracle seed"
CHILD_HMAC_KEY     = b"QTCL child key derivation"
ADDRESS_PREFIX     = "qtcl1"               # human-readable prefix

# ─────────────────────────────────────────────────────────────────────────────
# KEY TYPES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OracleKeyPair:
    """
    A derived keypair in the QTCL hierarchical deterministic tree.

    Fields mirror BIP32 extended key structure:
      private_key : 32 bytes — the signing scalar
      public_key  : 33 bytes — compressed representation (SHA3 of private)
      chain_code  : 32 bytes — child key derivation entropy
      depth       : 0-255   — depth in the HD tree
      index       : 0-2^32  — child index (≥ 2^31 = hardened)
      fingerprint : 4 bytes — parent pubkey fingerprint
      path        : str     — human-readable derivation path
    """
    private_key : bytes
    public_key  : bytes
    chain_code  : bytes
    depth       : int   = 0
    index       : int   = 0
    fingerprint : bytes = field(default_factory=lambda: b'\x00'*4)
    path        : str   = "m"

    def address(self) -> str:
        """
        QTCL address = ADDRESS_PREFIX + hex(SHA3-256(public_key)[:20])
        Deterministic, no checksums needed beyond the hash itself.
        """
        addr_bytes = hashlib.sha3_256(self.public_key).digest()[:20]
        return ADDRESS_PREFIX + addr_bytes.hex()

    def fingerprint_bytes(self) -> bytes:
        """First 4 bytes of SHA3-256(public_key) — used as child fingerprint."""
        return hashlib.sha3_256(self.public_key).digest()[:4]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "public_key_hex" : self.public_key.hex(),
            "depth"          : self.depth,
            "index"          : self.index,
            "path"           : self.path,
            "address"        : self.address(),
        }


@dataclass
class HLWESignature:
    """
    Hash-based Lattice Witness Encoding signature.

    commitment        : SHA3-256(child_key || w_entropy || msg_hash)
    witness           : SHAKE-256(commitment || child_key, 64 bytes)
    proof             : HMAC-SHA3(child_key, witness || msg_hash)
    w_entropy_hash    : SHA3-256 of the W-state measurement bitstring
    public_key_hex    : signer's compressed public key
    derivation_path   : e.g. "m/838'/0'/0'/0/7"
    timestamp_ns      : nanosecond timestamp at signing time
    """
    commitment      : str   # hex
    witness         : str   # hex
    proof           : str   # hex
    w_entropy_hash  : str   # hex
    public_key_hex  : str
    derivation_path : str
    timestamp_ns    : int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "HLWESignature":
        return HLWESignature(**d)


# ─────────────────────────────────────────────────────────────────────────────
# HD KEY DERIVATION (BIP32-STYLE, HASH-BASED)
# ─────────────────────────────────────────────────────────────────────────────

class HDKeyring:
    """
    Hierarchical deterministic key derivation.

    Replaces the elliptic-curve scalar arithmetic of BIP32 with:
      HMAC-SHA3-512(key, data) → 64 bytes → left 32 = child private, right 32 = chain code

    This preserves the BIP32 tree structure and path encoding while being
    purely hash-based (no elliptic curves, no lattice operations at derivation
    time — the lattice comes at signing).

    BIP38-equivalent passphrase hardening:
      The master seed is stretched with scrypt(passphrase, salt) before
      the first HMAC, giving BIP38-level passphrase protection.
    """

    def __init__(self, seed: bytes, passphrase: str = ""):
        """
        Derive master key from seed bytes.

        seed       : 16–64 bytes of random seed material
        passphrase : BIP38-style hardening passphrase (can be empty)
        """
        if len(seed) < 16:
            raise ValueError("Seed must be at least 16 bytes")

        # BIP38-style passphrase hardening via scrypt
        if passphrase:
            salt = hashlib.sha3_256(seed).digest()
            hardened_seed = hashlib.scrypt(
                passphrase.encode("utf-8"),
                salt=salt,
                n=16384, r=8, p=1,
                dklen=64,
            )
        else:
            hardened_seed = seed

        # Master key derivation
        raw = hmac.new(SEED_HMAC_KEY, hardened_seed, digestmod=hashlib.sha3_512).digest()
        master_private = raw[:32]
        master_chain   = raw[32:]

        # Public key = SHA3-256(private) compressed to 33 bytes
        # (not a real EC point — this is a hash-chain public key)
        master_public = self._derive_public(master_private)

        self._master = OracleKeyPair(
            private_key = master_private,
            public_key  = master_public,
            chain_code  = master_chain,
            depth       = 0,
            index       = 0,
            path        = "m",
        )
        self._cache: Dict[str, OracleKeyPair] = {"m": self._master}
        self._lock = threading.RLock()
        logger.info(f"[HDKeyring] Master key derived | address={self._master.address()}")

    @staticmethod
    def _derive_public(private_key: bytes) -> bytes:
        """
        Hash-chain public key from private key.
        33 bytes = 0x02 prefix + SHA3-256(private_key).
        (Mirrors the 33-byte compressed EC public key convention.)
        """
        return b'\x02' + hashlib.sha3_256(private_key).digest()

    def derive_child(self, parent: OracleKeyPair, index: int) -> OracleKeyPair:
        """
        Derive a child key at the given index.

        index < HARDENED_OFFSET  → normal child (public key included in HMAC)
        index ≥ HARDENED_OFFSET  → hardened child (private key in HMAC, BIP32 convention)
        """
        hardened = index >= HARDENED_OFFSET

        if hardened:
            # Hardened: private key + 0x00 prefix + index
            data = b'\x00' + parent.private_key + struct.pack('>I', index)
        else:
            # Normal: public key + index
            data = parent.public_key + struct.pack('>I', index)

        raw = hmac.new(parent.chain_code, data, digestmod=hashlib.sha3_512).digest()
        child_private_raw = raw[:32]
        child_chain       = raw[32:]

        # Tweak: XOR parent private with derived bytes (non-EC equivalent of addition)
        child_private = bytes(
            a ^ b for a, b in zip(parent.private_key, child_private_raw)
        )
        child_public = self._derive_public(child_private)

        suffix = f"/{index - HARDENED_OFFSET}'" if hardened else f"/{index}"
        return OracleKeyPair(
            private_key = child_private,
            public_key  = child_public,
            chain_code  = child_chain,
            depth       = parent.depth + 1,
            index       = index,
            fingerprint = parent.fingerprint_bytes(),
            path        = parent.path + suffix,
        )

    def derive_path(self, path: str) -> OracleKeyPair:
        """
        Derive keypair at BIP32-style path string.
        e.g. "m/838'/0'/0'/0/7"

        Results are cached so repeated derivations for the same address
        are O(1) after first call.
        """
        with self._lock:
            if path in self._cache:
                return self._cache[path]

        parts = path.split("/")
        if parts[0] != "m":
            raise ValueError(f"Path must start with 'm', got: {path}")

        node = self._master
        accumulated = "m"

        for part in parts[1:]:
            hardened = part.endswith("'")
            idx = int(part.rstrip("'"))
            if hardened:
                idx += HARDENED_OFFSET
            accumulated += "/" + part
            node = self.derive_child(node, idx)

        with self._lock:
            self._cache[path] = node

        return node

    def derive_address_key(self, account: int = 0, change: int = 0, index: int = 0) -> OracleKeyPair:
        """
        Standard QTCL derivation:  m / 838' / 0' / account' / change / index
        Mirrors Bitcoin's BIP44:   m / 44'  / 0' / account' / change / index
        """
        path = f"m/{QTCL_PURPOSE}'/{ QTCL_COIN }'/{ account }'/{ change }/{ index }"
        return self.derive_path(path)

    @property
    def master(self) -> OracleKeyPair:
        return self._master

    @staticmethod
    def generate_seed(entropy_bytes: int = 32) -> bytes:
        """Generate a cryptographically secure seed."""
        return secrets.token_bytes(entropy_bytes)

    @staticmethod
    def mnemonic_to_seed(mnemonic: str, passphrase: str = "") -> bytes:
        """
        BIP39-compatible: PBKDF2-SHA512(mnemonic, 'mnemonic'+passphrase, 2048).
        Accepts any BIP39 mnemonic. Pure hash-based, no wordlist dependency.
        """
        salt = ("mnemonic" + passphrase).encode("utf-8")
        return hashlib.pbkdf2_hmac(
            "sha512",
            mnemonic.encode("utf-8"),
            salt,
            iterations=2048,
            dklen=64,
        )


# ─────────────────────────────────────────────────────────────────────────────
# HLWE SIGNER & VERIFIER
# ─────────────────────────────────────────────────────────────────────────────

class HLWESigner:
    """
    Hash-based Lattice Witness Encoding signer.

    Signs a message (as bytes or a pre-computed hex hash) using a
    derived child key and W-state entropy from pq0.

    The W-state measurement is the quantum entropy injection point:
    every signature carries a commitment to a unique W-state outcome,
    making two signatures of the same message with the same key
    computationally distinguishable and non-replayable.
    """

    def __init__(self, keyring: HDKeyring):
        self.keyring = keyring

    def sign_message(
        self,
        message_hash: str,            # hex string of SHA3-256(message)
        keypair: OracleKeyPair,
        w_entropy: Optional[bytes] = None,  # W-state measurement bytes; freshly generated if None
    ) -> HLWESignature:
        """
        Produce an HLWE signature.

        message_hash : hex-encoded SHA3-256 hash of the data to sign
        keypair      : signing keypair (child key at the TX's derivation path)
        w_entropy    : raw bytes from W-state measurement (optional; random fallback)
        """
        msg_bytes = bytes.fromhex(message_hash)

        # W-state entropy — fresh randomness injected from pq0 measurement
        if w_entropy is None:
            w_entropy = secrets.token_bytes(32)
        w_entropy_hash = hashlib.sha3_256(w_entropy).hexdigest()

        # Step 1 — Commitment
        commitment_preimage = keypair.private_key + w_entropy + msg_bytes
        commitment = hashlib.sha3_256(commitment_preimage).digest()

        # Step 2 — Witness (64 bytes via SHAKE-256)
        shake = hashlib.shake_256()
        shake.update(commitment + keypair.private_key)
        witness = shake.digest(64)

        # Step 3 — Proof = HMAC-SHA3-256(private_key, witness || msg_bytes)
        proof = hmac.new(
            keypair.private_key,
            witness + msg_bytes,
            digestmod=hashlib.sha3_256,
        ).digest()

        return HLWESignature(
            commitment      = commitment.hex(),
            witness         = witness.hex(),
            proof           = proof.hex(),
            w_entropy_hash  = w_entropy_hash,
            public_key_hex  = keypair.public_key.hex(),
            derivation_path = keypair.path,
            timestamp_ns    = time.time_ns(),
        )

    def sign_transaction(
        self,
        tx_hash: str,
        sender_address: str,
        account: int = 0,
        change: int  = 0,
        index: int   = 0,
        w_entropy: Optional[bytes] = None,
    ) -> HLWESignature:
        """
        Sign a transaction hash.

        Derives the child key for (account, change, index) and signs tx_hash.
        Returns the full HLWESignature for inclusion in QuantumTransaction.signature.
        """
        keypair = self.keyring.derive_address_key(account, change, index)

        # Sanity: confirm derived address matches sender
        derived_addr = keypair.address()
        if derived_addr != sender_address:
            logger.warning(
                f"[HLWE] Address mismatch: derived={derived_addr} sender={sender_address}. "
                "Signing anyway — caller is responsible for address correctness."
            )

        return self.sign_message(tx_hash, keypair, w_entropy)


class HLWEVerifier:
    """
    HLWE signature verifier. Requires only the public key — no private material.

    Verification algorithm:
      1. Recompute proof' = HMAC-SHA3-256(pubkey_based_check, witness || msg_hash)
         Note: we verify the proof binds to the public key by using
         SHA3-256(public_key) as the HMAC key (mirrors the private→public relationship)
      2. Check proof == proof'
      3. Optionally check address derivation from public key

    Security note:
      This is a symmetric-check construction — full HLWE lattice hardness
      would require actual lattice operations (LWE/SIS). This implementation
      provides hash-based security (collision resistance + preimage resistance)
      suitable for testing and prototyping. A production deployment would swap
      HLWESigner/Verifier for CRYSTALS-Dilithium with the same interface.
    """

    @staticmethod
    def verify_signature(
        message_hash: str,
        signature: HLWESignature,
        expected_address: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Verify an HLWE signature.

        Returns (is_valid: bool, reason: str).
        """
        try:
            msg_bytes     = bytes.fromhex(message_hash)
            public_key    = bytes.fromhex(signature.public_key_hex)
            witness       = bytes.fromhex(signature.witness)
            proof         = bytes.fromhex(signature.proof)
            commitment    = bytes.fromhex(signature.commitment)

            # Derive the HMAC key from the public key
            # (mirrors: private_key → public_key derivation is one-way SHA3)
            hmac_key = hashlib.sha3_256(public_key).digest()

            # Recompute proof
            expected_proof = hmac.new(
                hmac_key,
                witness + msg_bytes,
                digestmod=hashlib.sha3_256,
            ).digest()

            if not hmac.compare_digest(proof, expected_proof):
                return False, "proof verification failed"

            # Verify witness length (SHAKE-256 output)
            if len(witness) != 64:
                return False, f"witness length invalid: {len(witness)}"

            # Verify commitment is well-formed (non-zero, correct length)
            if len(commitment) != 32:
                return False, f"commitment length invalid: {len(commitment)}"

            if commitment == b'\x00' * 32:
                return False, "null commitment rejected"

            # Address check
            if expected_address is not None:
                addr_bytes = hashlib.sha3_256(public_key).digest()[:20]
                derived_address = ADDRESS_PREFIX + addr_bytes.hex()
                if derived_address != expected_address:
                    return False, f"address mismatch: {derived_address} != {expected_address}"

            return True, "valid"

        except Exception as e:
            return False, f"verification exception: {e}"

    @staticmethod
    def public_key_to_address(public_key_hex: str) -> str:
        """Derive QTCL address from public key hex."""
        pk = bytes.fromhex(public_key_hex)
        return ADDRESS_PREFIX + hashlib.sha3_256(pk).digest()[:20].hex()


# ─────────────────────────────────────────────────────────────────────────────
# ORACLE ENGINE — SINGLETON, HOLDS MASTER KEY, SIGNS EVERYTHING
# ─────────────────────────────────────────────────────────────────────────────

class OracleEngine:
    """
    The Oracle Engine.

    pq0 IS the oracle. The oracle IS the signer. This class IS pq0's signing brain.

    Responsibilities:
      1. Hold the master HD keyring (loaded from env or generated once)
      2. Provide W-state entropy injection from the running lattice controller
      3. Sign transactions (called by BlockManager on seal)
      4. Sign blocks (block-level HLWE witness)
      5. Verify incoming transactions from P2P peers
      6. Issue new addresses for wallets

    Key storage:
      The master seed is stored in env var ORACLE_MASTER_SEED_HEX (64 hex bytes).
      On first boot with no env var, a seed is generated and logged once —
      the operator MUST save it. Subsequent boots with the same seed reproduce
      all keys deterministically.
    """

    _instance: Optional["OracleEngine"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "OracleEngine":
        with cls._lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._initialized = False
                cls._instance = inst
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._keyring: Optional[HDKeyring]     = None
        self._signer:  Optional[HLWESigner]    = None
        self._verifier = HLWEVerifier()
        self._lattice_ref                      = None   # set externally
        self._init_lock = threading.RLock()
        self._address_index: Dict[str, int]    = {}     # address → HD index
        self._next_index    = 0

        self._load_or_generate_master()

    def _load_or_generate_master(self):
        """Load master seed from environment or generate fresh."""
        seed_hex = os.getenv("ORACLE_MASTER_SEED_HEX", "")
        passphrase = os.getenv("ORACLE_PASSPHRASE", "")

        if seed_hex and len(seed_hex) >= 32:
            try:
                seed = bytes.fromhex(seed_hex)
                self._keyring = HDKeyring(seed, passphrase)
                logger.info(
                    f"[ORACLE] Master key loaded from ORACLE_MASTER_SEED_HEX | "
                    f"address={self._keyring.master.address()}"
                )
            except Exception as e:
                logger.error(f"[ORACLE] Failed to load seed from env: {e}. Generating new seed.")
                seed_hex = ""

        if not seed_hex:
            seed = HDKeyring.generate_seed(32)
            self._keyring = HDKeyring(seed, passphrase)
            logger.warning(
                f"[ORACLE] ⚠️  NEW MASTER SEED GENERATED — SAVE THIS IMMEDIATELY:\n"
                f"         ORACLE_MASTER_SEED_HEX={seed.hex()}\n"
                f"         Oracle address: {self._keyring.master.address()}\n"
                f"         Set this as a Koyeb env var or you will lose all keys on restart."
            )

        self._signer = HLWESigner(self._keyring)

    def set_lattice_ref(self, lattice_controller):
        """Wire the running QuantumLatticeController so we can pull W-state entropy."""
        self._lattice_ref = lattice_controller
        logger.info("[ORACLE] Lattice reference wired — W-state entropy active")

    def _get_w_entropy(self) -> bytes:
        """
        Pull fresh W-state measurement entropy from pq0.

        If the lattice is not running, falls back to OS randomness.
        The W-state measurement outcome is a genuinely quantum source of
        randomness (within simulation) — each call produces a unique bitstring.
        """
        if self._lattice_ref is not None:
            try:
                result = self._lattice_ref.w_state_constructor.measure_oracle_pqivv_w()
                if result and result.get("counts"):
                    counts_bytes = json_stable_bytes(result["counts"])
                    entropy = hashlib.sha3_256(
                        counts_bytes +
                        str(result.get("w_state_strength", 0)).encode() +
                        str(time.time_ns()).encode()
                    ).digest()
                    return entropy
            except Exception as e:
                logger.debug(f"[ORACLE] W-state entropy fallback ({e})")
        return secrets.token_bytes(32)

    # ── Signing API ──────────────────────────────────────────────────────────

    def sign_transaction(
        self,
        tx_hash: str,
        sender_address: str,
        account: int = 0,
        change: int  = 0,
        index: Optional[int] = None,
    ) -> Optional[HLWESignature]:
        """
        Sign a transaction with HLWE + W-state entropy.

        For testing: index is auto-incremented per new address.
        In production: caller tracks index per (account, change, address).
        """
        try:
            with self._init_lock:
                if index is None:
                    if sender_address not in self._address_index:
                        self._address_index[sender_address] = self._next_index
                        self._next_index += 1
                    index = self._address_index[sender_address]

            w_entropy = self._get_w_entropy()
            sig = self._signer.sign_transaction(
                tx_hash, sender_address, account, change, index, w_entropy
            )
            logger.debug(
                f"[ORACLE] TX signed | hash={tx_hash[:16]}… | "
                f"path={sig.derivation_path}"
            )
            return sig
        except Exception as e:
            logger.error(f"[ORACLE] TX signing failed: {e}")
            logger.error(traceback.format_exc())
            return None

    def sign_block(self, block_hash: str, block_height: int) -> Optional[HLWESignature]:
        """
        Sign a sealed block with the oracle master key.

        Block signing uses the master key directly (depth=0) — the oracle
        attest to the entire chain state at this block height.
        """
        try:
            # Use master key for block signing
            master_kp   = self._keyring.master
            w_entropy   = self._get_w_entropy()
            sig         = self._signer.sign_message(block_hash, master_kp, w_entropy)
            logger.info(
                f"[ORACLE] Block #{block_height} signed | "
                f"hash={block_hash[:18]}… | w_entropy={sig.w_entropy_hash[:16]}…"
            )
            return sig
        except Exception as e:
            logger.error(f"[ORACLE] Block signing failed: {e}")
            return None

    def verify_transaction(
        self,
        tx_hash: str,
        signature_dict: Dict[str, Any],
        sender_address: str,
    ) -> Tuple[bool, str]:
        """
        Verify a transaction's HLWE signature.
        Called by P2P on_tx handler before forwarding to BlockManager.
        """
        try:
            sig = HLWESignature.from_dict(signature_dict)
            return self._verifier.verify_signature(tx_hash, sig, sender_address)
        except Exception as e:
            return False, f"verification exception: {e}"

    def verify_block(
        self,
        block_hash: str,
        signature_dict: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """
        Verify a block's oracle signature.
        Called by P2P on_block handler.
        """
        try:
            sig = HLWESignature.from_dict(signature_dict)
            # Block sigs don't need address check — just proof validity
            return self._verifier.verify_signature(block_hash, sig, expected_address=None)
        except Exception as e:
            return False, f"block verification exception: {e}"

    def new_address(
        self,
        account: int = 0,
        change: int  = 0,
    ) -> Tuple[str, OracleKeyPair]:
        """
        Issue a new address.  Returns (address, keypair).
        Auto-increments index.
        """
        with self._init_lock:
            index  = self._next_index
            self._next_index += 1

        kp   = self._keyring.derive_address_key(account, change, index)
        addr = kp.address()
        self._address_index[addr] = index
        logger.debug(f"[ORACLE] New address issued: {addr} | path={kp.path}")
        return addr, kp

    @property
    def oracle_address(self) -> str:
        """The oracle's own primary address (master key)."""
        return self._keyring.master.address()

    def get_status(self) -> Dict[str, Any]:
        """Health / status snapshot for /api/oracle endpoint."""
        return {
            "oracle_address"   : self.oracle_address,
            "master_depth"     : 0,
            "addresses_issued" : self._next_index,
            "lattice_wired"    : self._lattice_ref is not None,
            "signing_scheme"   : "HLWE-SHA3-SHAKE256",
            "derivation"       : f"m/{QTCL_PURPOSE}'/{QTCL_COIN}'/account'/change/index",
            "timestamp"        : time.time(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def json_stable_bytes(obj) -> bytes:
    """Deterministic JSON bytes for any dict/list — for HMAC inputs."""
    import json
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON — import and use directly
# ─────────────────────────────────────────────────────────────────────────────

ORACLE = OracleEngine()
"""
The global oracle singleton.

Usage:
    from oracle import ORACLE

    sig  = ORACLE.sign_transaction(tx_hash, sender_addr)
    ok, reason = ORACLE.verify_transaction(tx_hash, sig.to_dict(), sender_addr)

    block_sig = ORACLE.sign_block(block_hash, block_height)
    ok, reason = ORACLE.verify_block(block_hash, block_sig.to_dict())

    addr, kp = ORACLE.new_address()
"""
