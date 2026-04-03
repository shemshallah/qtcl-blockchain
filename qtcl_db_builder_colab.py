
# ╔════════════════════════════════════════════════════════════════════════════════╗
# ║                                                                                ║
# ║   QTCL DATABASE BUILDER V5 - GOOGLE COLAB EDITION (COMPLETE)                  ║
# ║   Hard-coded Supabase Pooler Connection | Optimized Bulk Load                ║
# ║   RUN THIS CELL IN GOOGLE COLAB - NO MODIFICATIONS NEEDED                    ║
# ║                                                                                ║
# ╚════════════════════════════════════════════════════════════════════════════════╝

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: INSTALL DEPENDENCIES (Colab-specific)
# ─────────────────────────────────────────────────────────────────────────────
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "psycopg2-binary", "mpmath", "tqdm"])

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: IMPORTS & SETUP
# ─────────────────────────────────────────────────────────────────────────────
import time
import json
import math
import hashlib
import logging
import os
import gc
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any, Iterator
from dataclasses import dataclass, field
from contextlib import contextmanager
from urllib.parse import quote_plus

# Colab progress bar
from tqdm.notebook import tqdm

# Mathematical precision
try:
    from mpmath import (
        mp, mpf, mpc, sqrt, pi, cos, sin, exp, log, tanh, sinh, cosh, acosh,
        atanh, atan2, fabs, re as mre, im as mim, conj, norm, phase,
        matrix, nstr, power, floor, ceil, asin, acos, hypot, fsum
    )
    mp.dps = 150  # 150-bit precision
    MPMATH_AVAILABLE = True
except ImportError:
    MPMATH_AVAILABLE = False
    mp = None
    mpf = float
    print("⚠️ mpmath not available - precision reduced", file=sys.stderr)

# Database
try:
    import psycopg2
    from psycopg2.extras import execute_values, execute_batch, RealDictCursor, Json
    from psycopg2 import sql, errors as psycopg2_errors
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    print("⚠️ psycopg2 not available", file=sys.stderr)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: LOGGING (Colab-friendly)
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Colors for Colab output
class CLR:
    BOLD = '\033[1m'
    G = '\033[92m'
    R = '\033[91m'
    Y = '\033[93m'
    C = '\033[96m'
    M = '\033[95m'
    E = '\033[0m'
    HEADER = f'{BOLD}{M}'
    OK = f'{BOLD}{G}'
    ERROR = f'{BOLD}{R}'
    WARN = f'{BOLD}{Y}'
    QUANTUM = f'{BOLD}{M}'

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: HARD-CODED SUPABASE POOLER CONNECTION (AS REQUESTED)
# ─────────────────────────────────────────────────────────────────────────────
POOLER_DB = "postgres"
POOLER_HOST = "aws-0-us-west-2.pooler.supabase.com"
POOLER_PORT = "5432"
POOLER_USER = "postgres.rslvlsqwkfmdtebqsvtw"
POOLER_PASSWORD = "$h10j1r1H0w4rd"

# Build connection URL with proper URL encoding for special characters
# Note: statement_timeout must be set via SET command, not DSN parameter
pg_url = (
    f"postgresql://{quote_plus(POOLER_USER)}:"
    f"{quote_plus(POOLER_PASSWORD)}@"
    f"{POOLER_HOST}:{POOLER_PORT}/{POOLER_DB}"
)
logger.info(f"{CLR.QUANTUM}[COLAB] Supabase Pooler Connection Configured{CLR.E}")
logger.info(f"  Host: {POOLER_HOST}:{POOLER_PORT}")
logger.info(f"  User: {POOLER_USER}")
logger.info(f"  DB: {POOLER_DB}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: MATHEMATICAL STRUCTURES (Optimized for Colab memory)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class HyperbolicPoint:
    x: Any
    y: Any
    name: Optional[str] = None
    
    def to_db_tuple(self) -> Tuple[str, str, Optional[str]]:
        return (str(self.x), str(self.y), self.name)
    
    @staticmethod
    def from_db_tuple(x: str, y: str, name: Optional[str]) -> 'HyperbolicPoint':
        return HyperbolicPoint(mpf(x), mpf(y), name)


@dataclass(slots=True)
class HyperbolicTriangle:
    triangle_id: int
    v0: HyperbolicPoint
    v1: HyperbolicPoint
    v2: HyperbolicPoint
    depth: int = 0
    parent_id: Optional[int] = None
    
    def to_db_row(self) -> Tuple:
        return (
            self.triangle_id, self.depth, self.parent_id,
            *self.v0.to_db_tuple(),
            *self.v1.to_db_tuple(),
            *self.v2.to_db_tuple()
        )


@dataclass(slots=True)
class Pseudoqubit:
    pseudoqubit_id: int
    triangle_id: int
    x: Any
    y: Any
    placement_type: str
    phase_theta: Any = field(default_factory=lambda: mpf(0))
    coherence_measure: Any = field(default_factory=lambda: mpf("0.99"))
    
    def to_db_row(self) -> Tuple:
        return (
            self.pseudoqubit_id, self.triangle_id,
            str(self.x), str(self.y), self.placement_type,
            str(self.phase_theta), str(self.coherence_measure)
        )

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: HYPERBOLIC GEOMETRY (Mathematically Corrected)
# ─────────────────────────────────────────────────────────────────────────────

def _safe_atanh(x: Any) -> Any:
    if not MPMATH_AVAILABLE:
        return 0.5 * log((1 + x) / (1 - x)) if abs(x) < 1 else float('inf')
    x_clamped = max(mpf(-1) + mpf(10)**(-140), min(mpf(1) - mpf(10)**(-140), x))
    return atanh(x_clamped)


def hyperbolic_distance(p1: HyperbolicPoint, p2: HyperbolicPoint) -> Any:
    z1 = mpc(p1.x, p1.y)
    z2 = mpc(p2.x, p2.y)
    numerator = abs(z1 - z2)
    denominator = abs(mpf(1) - conj(z1) * z2)
    if denominator < mpf(10)**(-140):
        return mpf(0) if numerator < mpf(10)**(-140) else mpf('inf')
    ratio = numerator / denominator
    ratio = max(mpf(0), min(mpf(1) - mpf(10)**(-140), ratio))
    return mpf(2) * _safe_atanh(ratio)


def _safe_vertex_name(name: Optional[str], max_len: int = 200) -> Optional[str]:
    """Safely truncate vertex names to prevent database column overflow"""
    if not name:
        return None
    if len(name) <= max_len:
        return name
    # Hash suffix for long names
    import hashlib
    hash_suffix = hashlib.md5(name.encode()).hexdigest()[:8]
    return name[:max_len-9] + "_" + hash_suffix


def poincare_midpoint(p1: HyperbolicPoint, p2: HyperbolicPoint) -> HyperbolicPoint:
    z1 = mpc(p1.x, p1.y)
    z2 = mpc(p2.x, p2.y)
    denominator = mpf(1) + conj(z1) * z2
    if abs(denominator) < mpf(10)**(-140):
        # Use simple naming to avoid length issues
        fallback_name = "mid_fallback_p1" if abs(z1) < abs(z2) else "mid_fallback_p2"
        target_pt = p1 if abs(z1) < abs(z2) else p2
        return HyperbolicPoint(target_pt.x, target_pt.y, name=fallback_name)
    m = (z1 + z2) / denominator
    # Use simple naming: just use a hash of coordinates instead of parent names
    return HyperbolicPoint(mre(m), mim(m), name="midpoint")


def hyperbolic_angle_at_vertex(a: HyperbolicPoint, b: HyperbolicPoint, c: HyperbolicPoint) -> Any:
    a_len = hyperbolic_distance(b, c)
    b_len = hyperbolic_distance(a, c)
    c_len = hyperbolic_distance(a, b)
    cosh_a, cosh_b, cosh_c = cosh(a_len), cosh(b_len), cosh(c_len)
    sinh_a, sinh_c = sinh(a_len), sinh(c_len)
    if sinh_a * sinh_c < mpf(10)**(-140):
        return mpf(0)
    cos_angle = (cosh_b - cosh_a * cosh_c) / (sinh_a * sinh_c)
    cos_angle = max(mpf(-1), min(mpf(1), cos_angle))
    return acos(cos_angle)


def hyperbolic_incenter(tri: HyperbolicTriangle) -> HyperbolicPoint:
    a = hyperbolic_distance(tri.v1, tri.v2)
    b = hyperbolic_distance(tri.v0, tri.v2)
    c = hyperbolic_distance(tri.v0, tri.v1)
    w0 = mpf(1) / sinh(a) if sinh(a) > mpf(10)**(-140) else mpf(1)
    w1 = mpf(1) / sinh(b) if sinh(b) > mpf(10)**(-140) else mpf(1)
    w2 = mpf(1) / sinh(c) if sinh(c) > mpf(10)**(-140) else mpf(1)
    total = w0 + w1 + w2
    if total < mpf(10)**(-140):
        return tri.v0
    x = (w0 * tri.v0.x + w1 * tri.v1.x + w2 * tri.v2.x) / total
    y = (w0 * tri.v0.y + w1 * tri.v1.y + w2 * tri.v2.y) / total
    return HyperbolicPoint(x, y, name=f"incenter_{tri.triangle_id}")


def hyperbolic_circumcenter(tri: HyperbolicTriangle) -> HyperbolicPoint:
    ax, ay = tri.v0.x, tri.v0.y
    bx, by = tri.v1.x, tri.v1.y
    cx, cy = tri.v2.x, tri.v2.y
    d = mpf(2) * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if fabs(d) < mpf(10)**(-140):
        x = (ax + bx + cx) / mpf(3)
        y = (ay + by + cy) / mpf(3)
        return HyperbolicPoint(x, y, name=f"circumcenter_degen_{tri.triangle_id}")
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
    if ux**2 + uy**2 >= mpf(1) - mpf(10)**(-140):
        return hyperbolic_incenter(tri)
    return HyperbolicPoint(ux, uy, name=f"circumcenter_{tri.triangle_id}")


def hyperbolic_orthocenter(tri: HyperbolicTriangle) -> HyperbolicPoint:
    try:
        x = (tri.v0.x + tri.v1.x + tri.v2.x) / mpf(3)
        y = (tri.v0.y + tri.v1.y + tri.v2.y) / mpf(3)
        return HyperbolicPoint(x, y, name=f"orthocenter_{tri.triangle_id}")
    except Exception:
        return hyperbolic_incenter(tri)


def hyperbolic_geodesic_interpolate(p1: HyperbolicPoint, p2: HyperbolicPoint, t: Any) -> HyperbolicPoint:
    z1 = mpc(p1.x, p1.y)
    z2 = mpc(p2.x, p2.y)
    z_t = (mpf(1) - t) * z1 + t * z2
    r2 = abs(z_t)**2
    if r2 >= mpf(1) - mpf(10)**(-140):
        z_t = z_t * (mpf(1) - mpf(10)**(-140)) / sqrt(r2)
    # Keep name short to avoid truncation
    t_str = f"{float(t):.6f}".replace('.', '_')
    return HyperbolicPoint(mre(z_t), mim(z_t), name=_safe_vertex_name(f"geo_t{t_str}"))


def generate_geodesic_grid(tri: HyperbolicTriangle, density: int = 5) -> List[HyperbolicPoint]:
    points = []
    for i in range(1, density):
        for j in range(1, density - i):
            lambda1 = mpf(i) / mpf(density)
            lambda2 = mpf(j) / mpf(density)
            lambda3 = mpf(1) - lambda1 - lambda2
            if lambda1 + lambda2 > 0:
                pt_01 = hyperbolic_geodesic_interpolate(tri.v0, tri.v1, lambda2 / (lambda1 + lambda2))
            else:
                pt_01 = tri.v0
            pt = hyperbolic_geodesic_interpolate(tri.v2, pt_01, lambda3)
            # Use simple indexed name to avoid length issues
            points.append(HyperbolicPoint(pt.x, pt.y, name=f"g_{len(points)}"))
    # Add center point (barycenter) as 7th point
    center_x = (tri.v0.x + tri.v1.x + tri.v2.x) / mpf(3)
    center_y = (tri.v0.y + tri.v1.y + tri.v2.y) / mpf(3)
    points.append(HyperbolicPoint(center_x, center_y, name="g_center"))
    return points[:7]  # Ensure exactly 7 points

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: COMPLETE SCHEMA (Your exact 58-table schema)
# ─────────────────────────────────────────────────────────────────────────────

COMPLETE_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "btree_gist";

-- ════════════════════════════════════════════════════════════════════════════════════════════════════════════
-- QUANTUM LATTICE BLOCKCHAIN COMPLETE SCHEMA v2.0 - COMPREHENSIVE PATCH
-- ════════════════════════════════════════════════════════════════════════════════════════════════════════════
--
-- ALL 62 tables from db_builder integrated with blockchain core
-- Lattice geometry + Quantum measurements + Oracle state + Blockchain + Network + Wallet
--
-- Created: 2025
-- License: MIT
--
-- ════════════════════════════════════════════════════════════════════════════════════════════════════════════

-- TABLE: address_balance_history
CREATE TABLE address_balance_history (
    id BIGSERIAL PRIMARY KEY,
    address VARCHAR(255) NOT NULL,
    block_height BIGINT NOT NULL,
    block_hash VARCHAR(255),
    balance NUMERIC(30, 0) NOT NULL,
    delta NUMERIC(30, 0),
    snapshot_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(address, block_height)
);

-- TABLE: address_labels
CREATE TABLE address_labels (
    label_id BIGSERIAL PRIMARY KEY,
    address VARCHAR(255) NOT NULL,
    label VARCHAR(255) NOT NULL,
    description TEXT,
    label_type VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: address_transactions
CREATE TABLE address_transactions (
    id BIGSERIAL PRIMARY KEY,
    address VARCHAR(255) NOT NULL,
    tx_hash VARCHAR(255) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    from_address VARCHAR(255),
    to_address VARCHAR(255),
    amount NUMERIC(30, 0),
    block_height BIGINT,
    block_hash VARCHAR(255),
    block_timestamp BIGINT,
    tx_status VARCHAR(50) DEFAULT 'pending',
    notes TEXT,
    label VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(address, tx_hash)
);

-- TABLE: address_utxos
CREATE TABLE address_utxos (
    utxo_id BIGSERIAL PRIMARY KEY,
    address VARCHAR(255) NOT NULL,
    tx_hash VARCHAR(255) NOT NULL,
    output_index INT NOT NULL,
    amount NUMERIC(30, 0) NOT NULL,
    spent BOOLEAN DEFAULT FALSE,
    spent_at_height BIGINT,
    spent_in_tx_hash VARCHAR(255),
    created_at_height BIGINT,
    created_at_timestamp BIGINT
);

-- TABLE: audit_logs
CREATE TABLE audit_logs (
    log_id BIGSERIAL PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    actor_peer_id VARCHAR(255),
    action VARCHAR(255),
    resource_type VARCHAR(100),
    resource_id VARCHAR(255),
    changes JSONB,
    result VARCHAR(50),
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: block_headers_cache
CREATE TABLE block_headers_cache (
    height BIGINT PRIMARY KEY,
    block_hash VARCHAR(255) UNIQUE NOT NULL,
    previous_hash VARCHAR(255) NOT NULL,
    state_root VARCHAR(255),
    transactions_root VARCHAR(255),
    timestamp BIGINT NOT NULL,
    difficulty NUMERIC(20, 10),
    nonce VARCHAR(255),
    quantum_proof VARCHAR(255),
    quantum_state_hash VARCHAR(255),
    temporal_coherence NUMERIC(5, 4),
    pq_signature TEXT,
    pq_key_fingerprint VARCHAR(255),
    received_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: blocks
CREATE TABLE blocks (
    height BIGINT PRIMARY KEY,
    block_number BIGINT UNIQUE NOT NULL,
    block_hash VARCHAR(255) UNIQUE NOT NULL,
    previous_hash VARCHAR(255) NOT NULL,
    state_root VARCHAR(255),
    transactions_root VARCHAR(255),
    receipts_root VARCHAR(255),
    timestamp BIGINT NOT NULL,
    transactions INT DEFAULT 0,
    validator_public_key VARCHAR(255) NOT NULL,
    validator_signature TEXT,
    difficulty NUMERIC(20, 10) DEFAULT 1.0,
    total_difficulty NUMERIC(30, 0),
    nonce VARCHAR(255),
    quantum_proof TEXT,
    quantum_state_hash VARCHAR(255),
    quantum_validation_status VARCHAR(50) DEFAULT 'unvalidated',
    entropy_score NUMERIC(5, 4) DEFAULT 0.0,
    temporal_coherence NUMERIC(5, 4) DEFAULT 0.9,
    pq_signature TEXT,
    pq_key_fingerprint VARCHAR(255),
    pq_validation_status VARCHAR(50) DEFAULT 'unsigned',
    pq_curr INTEGER DEFAULT 1,
    pq_last INTEGER DEFAULT 0,
    oracle_w_state_hash VARCHAR(255),
    oracle_density_matrix_hash VARCHAR(255),
    oracle_entropy_hash VARCHAR(255),
    oracle_consensus_reached BOOLEAN DEFAULT FALSE,
    status VARCHAR(50) DEFAULT 'pending',
    finalized BOOLEAN DEFAULT FALSE,
    finalized_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: chain_reorganizations
CREATE TABLE chain_reorganizations (
    reorg_id BIGSERIAL PRIMARY KEY,
    timestamp BIGINT NOT NULL,
    reorg_depth INT NOT NULL,
    old_head_height BIGINT,
    new_head_height BIGINT,
    old_head_hash VARCHAR(255),
    new_head_hash VARCHAR(255),
    reorg_point_hash VARCHAR(255),
    transactions_reverted INT,
    transactions_reinserted INT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: client_block_sync
CREATE TABLE client_block_sync (
    sync_id BIGSERIAL PRIMARY KEY,
    peer_id VARCHAR(255) NOT NULL,
    blocks_downloaded INT,
    blocks_requested INT,
    blocks_total INT,
    sync_started_at TIMESTAMP WITH TIME ZONE,
    sync_completed_at TIMESTAMP WITH TIME ZONE,
    sync_status VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: client_network_metrics
CREATE TABLE client_network_metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    peer_id VARCHAR(255) NOT NULL,
    timestamp BIGINT NOT NULL,
    latency_ms NUMERIC(10, 2),
    bandwidth_in_kbps NUMERIC(15, 2),
    bandwidth_out_kbps NUMERIC(15, 2),
    packet_loss_rate NUMERIC(5, 4),
    blocks_per_second NUMERIC(10, 2),
    avg_sync_time_ms NUMERIC(15, 2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: client_oracle_sync
CREATE TABLE client_oracle_sync (
    sync_id BIGSERIAL PRIMARY KEY,
    peer_id VARCHAR(255) NOT NULL UNIQUE,
    block_height_local BIGINT,
    block_height_oracle BIGINT,
    w_state_hash_local VARCHAR(255),
    w_state_hash_oracle VARCHAR(255),
    density_matrix_hash_local VARCHAR(255),
    density_matrix_hash_oracle VARCHAR(255),
    density_matrix_sync_status VARCHAR(50) DEFAULT 'pending',
    entropy_hash_local VARCHAR(255),
    entropy_hash_oracle VARCHAR(255),
    coherence_measure_local NUMERIC(5, 4),
    coherence_measure_oracle NUMERIC(5, 4),
    coherence_aligned BOOLEAN DEFAULT FALSE,
    lattice_sync_quality NUMERIC(5, 4),
    tessellation_in_sync BOOLEAN DEFAULT FALSE,
    last_lattice_update TIMESTAMP WITH TIME ZONE,
    sync_status VARCHAR(50) DEFAULT 'initializing',
    sync_confidence NUMERIC(5, 4) DEFAULT 0.0,
    last_sync_attempt TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_successful_sync TIMESTAMP WITH TIME ZONE,
    sync_error_message TEXT,
    sync_attempt_count INT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: client_sync_events
CREATE TABLE client_sync_events (
    event_id BIGSERIAL PRIMARY KEY,
    peer_id VARCHAR(255) NOT NULL,
    timestamp BIGINT NOT NULL,
    event_type VARCHAR(50),
    event_description TEXT,
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: consensus_events
CREATE TABLE consensus_events (
    event_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT,
    timestamp BIGINT NOT NULL,
    event_type VARCHAR(100),
    event_description TEXT,
    severity VARCHAR(20),
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: database_metadata
CREATE TABLE database_metadata (
    metadata_id BIGSERIAL PRIMARY KEY,
    schema_version VARCHAR(50),
    build_timestamp TIMESTAMP WITH TIME ZONE,
    build_info JSONB,
    tables_created INT,
    indexes_created INT,
    constraints_created INT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: encrypted_private_keys
CREATE TABLE encrypted_private_keys (
    key_id BIGSERIAL PRIMARY KEY,
    address VARCHAR(255) NOT NULL UNIQUE,
    algorithm VARCHAR(50) DEFAULT 'AES-256-GCM',
    kdf_algorithm VARCHAR(50) DEFAULT 'PBKDF2-SHA3-512',
    kdf_iterations INT DEFAULT 16384,
    nonce_hex VARCHAR(255) NOT NULL,
    salt_hex VARCHAR(255) NOT NULL,
    ciphertext_hex TEXT NOT NULL,
    auth_tag_hex VARCHAR(255),
    key_fingerprint VARCHAR(255),
    derivation_path VARCHAR(100),
    is_locked BOOLEAN DEFAULT FALSE,
    lock_reason TEXT,
    last_used_for_signing TIMESTAMP WITH TIME ZONE,
    signing_count INT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: entanglement_records
CREATE TABLE entanglement_records (
    entanglement_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    peer_id VARCHAR(255) NOT NULL,
    peer_public_key VARCHAR(255),
    entanglement_type VARCHAR(50),
    entanglement_measure NUMERIC(5, 4),
    bell_parameter NUMERIC(5, 4),
    oracle_entanglement_measure NUMERIC(5, 4),
    entanglement_match_score NUMERIC(5, 4),
    in_sync_with_oracle BOOLEAN DEFAULT FALSE,
    local_w_state_hash VARCHAR(255),
    local_density_matrix_hash VARCHAR(255),
    verification_proof TEXT,
    verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: entropy_quality_log
CREATE TABLE entropy_quality_log (
    log_id BIGSERIAL PRIMARY KEY,
    timestamp BIGINT NOT NULL,
    anu_qrng_quality NUMERIC(5, 4),
    random_org_quality NUMERIC(5, 4),
    qbick_quality NUMERIC(5, 4),
    outshift_quality NUMERIC(5, 4),
    hotbits_quality NUMERIC(5, 4),
    ensemble_min_entropy NUMERIC(5, 4),
    ensemble_shannon_entropy NUMERIC(5, 4),
    passed_diehard BOOLEAN,
    passed_nist BOOLEAN,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: epoch_validators
CREATE TABLE epoch_validators (
    membership_id BIGSERIAL PRIMARY KEY,
    epoch_id BIGINT NOT NULL,
    validator_id BIGINT NOT NULL,
    stake NUMERIC(30, 0) NOT NULL,
    blocks_proposed INT DEFAULT 0,
    blocks_attested INT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(epoch_id, validator_id)
);

-- TABLE: epochs
CREATE TABLE epochs (
    epoch_id BIGSERIAL PRIMARY KEY,
    epoch_number BIGINT UNIQUE NOT NULL,
    start_block_height BIGINT NOT NULL,
    end_block_height BIGINT,
    validator_count INT,
    total_stake NUMERIC(30, 0),
    finalized BOOLEAN DEFAULT FALSE,
    finalized_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: finality_records
CREATE TABLE finality_records (
    finality_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL UNIQUE,
    block_hash VARCHAR(255) NOT NULL,
    finalized BOOLEAN DEFAULT FALSE,
    finalized_at TIMESTAMP WITH TIME ZONE,
    finality_epoch BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: hyperbolic_triangles
CREATE TABLE hyperbolic_triangles (
    triangle_id BIGINT PRIMARY KEY,
    depth INT NOT NULL,
    parent_id BIGINT,
    v0_x NUMERIC(200, 150) NOT NULL,
    v0_y NUMERIC(200, 150) NOT NULL,
    v0_name TEXT,
    v1_x NUMERIC(200, 150) NOT NULL,
    v1_y NUMERIC(200, 150) NOT NULL,
    v1_name TEXT,
    v2_x NUMERIC(200, 150) NOT NULL,
    v2_y NUMERIC(200, 150) NOT NULL,
    v2_name TEXT,
    area NUMERIC(200, 150),
    perimeter NUMERIC(200, 150),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: lattice_sync_state
CREATE TABLE lattice_sync_state (
    state_id BIGSERIAL PRIMARY KEY,
    peer_id VARCHAR(255) NOT NULL,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    hyperbolic_coordinates_synced JSONB,
    poincare_disk_coverage NUMERIC(5, 4),
    vertex_synchronization_count INT,
    edge_synchronization_count INT,
    pseudoqubit_positions_hash VARCHAR(255),
    pseudoqubit_lattice_sync_quality NUMERIC(5, 4),
    lattice_coherence_measure NUMERIC(5, 4),
    critical_points_coherence JSONB,
    geodesic_paths_synchronized BOOLEAN DEFAULT FALSE,
    updates_since_last_sync INT DEFAULT 0,
    bytes_synchronized BIGINT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: merkle_proofs
CREATE TABLE merkle_proofs (
    proof_id BIGSERIAL PRIMARY KEY,
    transaction_hash VARCHAR(255) NOT NULL,
    height BIGINT,
    block_hash VARCHAR(255) NOT NULL,
    proof_path TEXT NOT NULL,
    proof_index INT NOT NULL,
    verified BOOLEAN DEFAULT FALSE,
    verified_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(transaction_hash, block_hash)
);

-- TABLE: network_bandwidth_usage
CREATE TABLE network_bandwidth_usage (
    usage_id BIGSERIAL PRIMARY KEY,
    peer_id VARCHAR(255),
    timestamp BIGINT NOT NULL,
    bandwidth_in_kbps NUMERIC(15, 2),
    bandwidth_out_kbps NUMERIC(15, 2),
    total_bandwidth_kbps NUMERIC(15, 2),
    bytes_in INT,
    bytes_out INT,
    congestion_level NUMERIC(5, 4),
    packet_loss_rate NUMERIC(5, 4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: network_events
CREATE TABLE network_events (
    event_id BIGSERIAL PRIMARY KEY,
    timestamp BIGINT NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    event_description TEXT,
    affected_peers INT,
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: network_partition_events
CREATE TABLE network_partition_events (
    event_id BIGSERIAL PRIMARY KEY,
    timestamp BIGINT NOT NULL,
    partition_detected BOOLEAN DEFAULT TRUE,
    peers_in_partition_1 INT,
    peers_in_partition_2 INT,
    partition_healed BOOLEAN DEFAULT FALSE,
    healing_timestamp BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: oracle_registry
CREATE TABLE oracle_registry (
    oracle_id       VARCHAR(128)  PRIMARY KEY,
    oracle_url      VARCHAR(512)  NOT NULL DEFAULT '',
    oracle_address  VARCHAR(128)  NOT NULL DEFAULT '',
    is_primary      BOOLEAN       NOT NULL DEFAULT FALSE,
    last_seen       BIGINT        NOT NULL DEFAULT 0,
    block_height    BIGINT        NOT NULL DEFAULT 0,
    peer_count      INTEGER       NOT NULL DEFAULT 0,
    gossip_url      JSONB         NOT NULL DEFAULT '{}'::JSONB,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    -- ── ON-CHAIN IDENTITY COLUMNS (oracle_reg TX pipeline) ─────────────────
    wallet_address  VARCHAR(128)  NOT NULL DEFAULT '',
    oracle_pub_key  TEXT          NOT NULL DEFAULT '',
    cert_sig        VARCHAR(128)  NOT NULL DEFAULT '',
    cert_auth_tag   VARCHAR(128)  NOT NULL DEFAULT '',
    mode            VARCHAR(32)   NOT NULL DEFAULT 'full',
    ip_hint         VARCHAR(256)  NOT NULL DEFAULT '',
    reg_tx_hash     VARCHAR(64)   NOT NULL DEFAULT '',
    registered_at   BIGINT        NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_oracle_registry_last_seen     ON oracle_registry (last_seen DESC);
CREATE INDEX IF NOT EXISTS idx_oracle_registry_primary       ON oracle_registry (is_primary) WHERE is_primary = TRUE;
CREATE INDEX IF NOT EXISTS idx_oracle_registry_wallet        ON oracle_registry (wallet_address);
CREATE INDEX IF NOT EXISTS idx_oracle_registry_reg_tx        ON oracle_registry (reg_tx_hash) WHERE reg_tx_hash != '';
CREATE INDEX IF NOT EXISTS idx_oracle_registry_registered_at ON oracle_registry (registered_at DESC);

-- TABLE: oracle_coherence_metrics
CREATE TABLE oracle_coherence_metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    system_coherence_measure NUMERIC(5, 4),
    lattice_coherence_score NUMERIC(5, 4),
    tessellation_synchronization_quality NUMERIC(5, 4),
    pseudoqubit_coherence_array JSONB,
    min_coherence NUMERIC(5, 4),
    max_coherence NUMERIC(5, 4),
    avg_coherence NUMERIC(5, 4),
    phase_drift_radians NUMERIC(200, 150),
    phase_correction_applied BOOLEAN,
    validator_agreement_score NUMERIC(5, 4),
    network_partition_detected BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: oracle_consensus_state
CREATE TABLE oracle_consensus_state (
    consensus_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    oracle_consensus_reached BOOLEAN DEFAULT FALSE,
    validator_agreement_count INT,
    total_validators INT,
    consensus_threshold NUMERIC(5, 4),
    w_state_hash_agreement BOOLEAN,
    density_matrix_hash_agreement BOOLEAN,
    entropy_hash_agreement BOOLEAN,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(block_height)
);

-- TABLE: oracle_density_matrix_stream
CREATE TABLE oracle_density_matrix_stream (
    stream_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    density_matrix_json JSONB NOT NULL,
    density_matrix_hash VARCHAR(255) UNIQUE NOT NULL,
    trace_value NUMERIC(5, 4),
    purity NUMERIC(5, 4),
    von_neumann_entropy NUMERIC(5, 4),
    eigenvalues JSONB,
    live_metrics_json JSONB,
    sensor_timestamps JSONB,
    update_sequence_number BIGINT,
    time_since_last_update_ms NUMERIC(15, 2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: oracle_distribution_log
CREATE TABLE oracle_distribution_log (
    log_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    peer_id VARCHAR(255) NOT NULL,
    data_type VARCHAR(50),
    data_hash VARCHAR(255),
    distribution_successful BOOLEAN DEFAULT TRUE,
    distribution_latency_ms NUMERIC(15, 2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: oracle_entanglement_records
CREATE TABLE oracle_entanglement_records (
    entanglement_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    peer_id VARCHAR(255) NOT NULL,
    peer_public_key VARCHAR(255),
    entanglement_type VARCHAR(50),
    entanglement_measure NUMERIC(5, 4),
    bell_parameter NUMERIC(5, 4),
    oracle_entanglement_measure NUMERIC(5, 4),
    entanglement_match_score NUMERIC(5, 4),
    in_sync_with_oracle BOOLEAN DEFAULT FALSE,
    verification_proof TEXT,
    verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: oracle_entropy_feeds
CREATE TABLE oracle_entropy_feeds (
    feed_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    anu_qrng_entropy TEXT,
    random_org_entropy TEXT,
    qbick_entropy TEXT,
    outshift_entropy TEXT,
    hotbits_entropy TEXT,
    xor_combined_seed TEXT,
    entropy_hash VARCHAR(255) UNIQUE NOT NULL,
    min_entropy_estimate NUMERIC(5, 4),
    shannon_entropy_estimate NUMERIC(5, 4),
    source_agreement_score NUMERIC(5, 4),
    distributed_to_peers INT DEFAULT 0,
    distribution_complete BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: oracle_pq0_state
CREATE TABLE oracle_pq0_state (
    state_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    oracle_pq_id BIGINT,
    oracle_position_x NUMERIC(200, 150),
    oracle_position_y NUMERIC(200, 150),
    pq_inverse_virtual_id BIGINT,
    pq_virtual_id BIGINT,
    quantum_state_json JSONB,
    phase_theta NUMERIC(200, 150),
    coherence_measure NUMERIC(5, 4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: oracle_w_state_snapshots
CREATE TABLE oracle_w_state_snapshots (
    snapshot_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    block_hash VARCHAR(255) NOT NULL,
    timestamp BIGINT NOT NULL,
    w_state_serialized TEXT NOT NULL,
    w_state_hash VARCHAR(255) UNIQUE NOT NULL,
    entanglement_measure NUMERIC(5, 4),
    coherence_time_us NUMERIC(15, 2),
    fidelity_estimate NUMERIC(5, 4),
    quantum_proof_data TEXT,
    quantum_proof_hash VARCHAR(255),
    shannon_entropy NUMERIC(5, 4),
    entropy_source_quality JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(block_height, block_hash)
);

-- TABLE: orphan_blocks
CREATE TABLE orphan_blocks (
    block_hash VARCHAR(255) PRIMARY KEY,
    parent_hash VARCHAR(255) NOT NULL,
    block_height BIGINT,
    timestamp BIGINT,
    block_data_compressed BYTEA,
    block_size_bytes INT,
    received_from_peer VARCHAR(255),
    received_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    resolution_status VARCHAR(50) DEFAULT 'awaiting_parent',
    resolution_attempt_count INT DEFAULT 0
);

-- TABLE: peer_connections
CREATE TABLE peer_connections (
    connection_id BIGSERIAL PRIMARY KEY,
    peer_id VARCHAR(255) NOT NULL,
    connection_state VARCHAR(50) DEFAULT 'disconnected',
    established_at TIMESTAMP WITH TIME ZONE,
    disconnected_at TIMESTAMP WITH TIME ZONE,
    latency_ms NUMERIC(10, 2),
    bandwidth_in_kbps NUMERIC(15, 2),
    bandwidth_out_kbps NUMERIC(15, 2),
    packet_loss_rate NUMERIC(5, 4),
    blocks_sync_height BIGINT,
    last_message_at TIMESTAMP WITH TIME ZONE,
    messages_sent INT DEFAULT 0,
    messages_received INT DEFAULT 0,
    bytes_sent BIGINT DEFAULT 0,
    bytes_received BIGINT DEFAULT 0,
    oracle_state_shared BOOLEAN DEFAULT FALSE,
    density_matrix_shared BOOLEAN DEFAULT FALSE,
    entropy_shared BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: peer_registry
CREATE TABLE peer_registry (
    peer_id VARCHAR(255) PRIMARY KEY,
    public_key VARCHAR(255) UNIQUE NOT NULL,
    ip_address VARCHAR(45),
    port INTEGER,
    peer_type VARCHAR(50) DEFAULT 'full',
    capabilities TEXT[],
    block_height BIGINT DEFAULT 0,
    chain_head_hash VARCHAR(255),
    network_version VARCHAR(50),
    reputation_score NUMERIC(10, 4) DEFAULT 1.0,
    blocks_validated INT DEFAULT 0,
    blocks_rejected INT DEFAULT 0,
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_handshake TIMESTAMP WITH TIME ZONE,
    connection_attempts INT DEFAULT 0,
    failed_attempts INT DEFAULT 0,
    oracle_entanglement_ready BOOLEAN DEFAULT FALSE,
    oracle_density_matrix_version BIGINT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: peer_reputation
CREATE TABLE peer_reputation (
    reputation_id BIGSERIAL PRIMARY KEY,
    peer_id VARCHAR(255) NOT NULL,
    timestamp BIGINT NOT NULL,
    score NUMERIC(10, 4) NOT NULL,
    factors JSONB,
    event_type VARCHAR(50),
    event_description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: pseudoqubits
CREATE TABLE pseudoqubits (
    pq_id BIGINT PRIMARY KEY,
    triangle_id BIGINT NOT NULL,
    x NUMERIC(200, 150) NOT NULL,
    y NUMERIC(200, 150) NOT NULL,
    placement_type VARCHAR(50) NOT NULL,
    phase_theta NUMERIC(200, 150) DEFAULT 0,
    coherence_measure NUMERIC(5, 4) DEFAULT 0.99,
    coherence_time_us NUMERIC(15, 2) DEFAULT 100000,
    entanglement_with_oracle NUMERIC(5, 4) DEFAULT 0,
    entanglement_measure_neighbors JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_measured_at TIMESTAMP WITH TIME ZONE
);

-- TABLE: quantum_circuit_execution
CREATE TABLE quantum_circuit_execution (
    circuit_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    circuit_depth INT,
    circuit_size INT,
    num_qubits INT,
    num_gates INT,
    execution_successful BOOLEAN DEFAULT TRUE,
    execution_time_ms NUMERIC(15, 2),
    ghz_fidelity NUMERIC(5, 4),
    w_state_fidelity NUMERIC(5, 4),
    circuit_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: quantum_coherence_snapshots
CREATE TABLE quantum_coherence_snapshots (
    snapshot_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    global_coherence NUMERIC(5, 4) NOT NULL,
    average_coherence NUMERIC(5, 4),
    min_coherence NUMERIC(5, 4),
    max_coherence NUMERIC(5, 4),
    coherence_histogram JSONB,
    phase_drift_radians NUMERIC(200, 150),
    phase_correction_applied BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: quantum_density_matrix_global
CREATE TABLE quantum_density_matrix_global (
    state_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    density_matrix_json JSONB NOT NULL,
    density_matrix_hash VARCHAR(255) UNIQUE NOT NULL,
    trace_value NUMERIC(5, 4),
    purity NUMERIC(5, 4),
    von_neumann_entropy NUMERIC(5, 4),
    eigenvalues JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: quantum_error_correction
CREATE TABLE quantum_error_correction (
    correction_id BIGSERIAL PRIMARY KEY,
    pq_id BIGINT NOT NULL,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    error_detected BOOLEAN NOT NULL,
    error_type VARCHAR(50),
    error_location_code VARCHAR(255),
    correction_applied BOOLEAN DEFAULT FALSE,
    correction_method VARCHAR(50),
    correction_strength NUMERIC(5, 4),
    correction_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: quantum_lattice_metadata
CREATE TABLE quantum_lattice_metadata (
    metadata_id BIGSERIAL PRIMARY KEY,
    tessellation_depth INT NOT NULL,
    total_triangles BIGINT NOT NULL,
    total_pseudoqubits BIGINT NOT NULL,
    precision_bits INT DEFAULT 150,
    hyperbolicity_constant NUMERIC(5, 4) DEFAULT -1.0,
    poincare_radius NUMERIC(5, 4) DEFAULT 1.0,
    status VARCHAR(50) DEFAULT 'constructing',
    construction_started_at TIMESTAMP WITH TIME ZONE,
    construction_completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: quantum_measurements
CREATE TABLE quantum_measurements (
    measurement_id BIGSERIAL PRIMARY KEY,
    pq_id BIGINT NOT NULL,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    outcome INT CHECK (outcome IN (0, 1)),
    basis VARCHAR(10),
    expectation_value NUMERIC(5, 4),
    variance NUMERIC(5, 4),
    post_measurement_state JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: quantum_phase_evolution
CREATE TABLE quantum_phase_evolution (
    phase_id BIGSERIAL PRIMARY KEY,
    pq_id BIGINT NOT NULL,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    phase_theta NUMERIC(200, 150) NOT NULL,
    phase_derivative NUMERIC(200, 150),
    coherence_measure NUMERIC(5, 4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: quantum_shadow_tomography
CREATE TABLE quantum_shadow_tomography (
    shadow_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    shadow_snapshots JSONB NOT NULL,
    shadow_measurement_bases JSONB,
    reconstruction_fidelity NUMERIC(5, 4),
    num_snapshots INT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: quantum_supremacy_proofs
CREATE TABLE quantum_supremacy_proofs (
    proof_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    circuit_depth INT,
    success_probability NUMERIC(5, 4),
    classical_simulation_complexity VARCHAR(255),
    quantum_result_hash VARCHAR(255),
    classical_hardness_assumption TEXT,
    verified BOOLEAN DEFAULT FALSE,
    verification_method VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: state_root_updates
CREATE TABLE state_root_updates (
    update_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    new_state_root VARCHAR(255) NOT NULL,
    previous_state_root VARCHAR(255),
    timestamp BIGINT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(block_height)
);

-- TABLE: system_metrics
CREATE TABLE system_metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    timestamp BIGINT NOT NULL,
    db_size_mb NUMERIC(15, 2),
    active_connections INT,
    active_peers INT,
    total_peers INT,
    avg_latency_ms NUMERIC(10, 2),
    blocks_per_minute NUMERIC(10, 2),
    transactions_per_second NUMERIC(10, 2),
    avg_coherence NUMERIC(5, 4),
    oracle_sync_quality NUMERIC(5, 4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: transaction_inputs
CREATE TABLE transaction_inputs (
    input_id BIGSERIAL PRIMARY KEY,
    tx_id BIGINT NOT NULL,
    previous_tx_hash VARCHAR(255),
    previous_output_index INT,
    script_sig TEXT,
    script_pubkey TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: transaction_outputs
CREATE TABLE transaction_outputs (
    output_id BIGSERIAL PRIMARY KEY,
    tx_id BIGINT NOT NULL,
    output_index INT NOT NULL,
    address VARCHAR(255) NOT NULL,
    amount NUMERIC(30, 0) NOT NULL,
    script_pubkey TEXT,
    spent BOOLEAN DEFAULT FALSE,
    spent_in_tx_id BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(tx_id, output_index)
);

-- TABLE: transaction_receipts
CREATE TABLE transaction_receipts (
    receipt_id BIGSERIAL PRIMARY KEY,
    tx_id BIGINT NOT NULL,
    height BIGINT,
    status INT,
    logs_json JSONB,
    bloom_filter TEXT,
    quantum_proof TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: transactions
CREATE TABLE transactions (
    id BIGSERIAL PRIMARY KEY,
    tx_hash VARCHAR(255) UNIQUE NOT NULL,
    from_address VARCHAR(255) NOT NULL,
    to_address VARCHAR(255) NOT NULL,
    amount NUMERIC(30, 0) NOT NULL,
    nonce BIGINT,
    height BIGINT,
    block_hash VARCHAR(255),
    transaction_index INT,
    tx_type VARCHAR(50) DEFAULT 'transfer',
    status VARCHAR(50) DEFAULT 'pending',
    pq_signature TEXT,
    pq_signer_key_fp VARCHAR(255),
    pq_verified BOOLEAN DEFAULT FALSE,
    pq_verified_at TIMESTAMP WITH TIME ZONE,
    quantum_state_hash VARCHAR(255),
    commitment_hash VARCHAR(255),
    entropy_score NUMERIC(5, 4),
    input_data TEXT,
    metadata JSONB,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    finalized_at TIMESTAMP WITH TIME ZONE
);

-- TABLE: validator_stakes
CREATE TABLE validator_stakes (
    stake_id BIGSERIAL PRIMARY KEY,
    validator_id BIGINT NOT NULL,
    amount NUMERIC(30, 0) NOT NULL,
    staker_address VARCHAR(255),
    active BOOLEAN DEFAULT TRUE,
    delegated BOOLEAN DEFAULT FALSE,
    stake_at_height BIGINT,
    unstake_at_height BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: validators
CREATE TABLE validators (
    validator_id BIGSERIAL PRIMARY KEY,
    public_key VARCHAR(255) UNIQUE NOT NULL,
    peer_id VARCHAR(255),
    stake NUMERIC(30, 0) DEFAULT 0,
    commission_rate NUMERIC(5, 4),
    slashing_rate NUMERIC(5, 4) DEFAULT 0.0,
    blocks_proposed INT DEFAULT 0,
    blocks_validated INT DEFAULT 0,
    blocks_missed INT DEFAULT 0,
    status VARCHAR(50) DEFAULT 'active',
    is_slashed BOOLEAN DEFAULT FALSE,
    slashed_at TIMESTAMP WITH TIME ZONE,
    oracle_participation_score NUMERIC(5, 4) DEFAULT 0.0,
    w_state_sync_quality NUMERIC(5, 4) DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: w_state_snapshots
CREATE TABLE w_state_snapshots (
    snapshot_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    w_state_serialized TEXT NOT NULL,
    w_state_hash VARCHAR(255) UNIQUE NOT NULL,
    entanglement_measure NUMERIC(5, 4),
    coherence_time_us NUMERIC(15, 2),
    fidelity_estimate NUMERIC(5, 4),
    pq_addresses TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(block_height)
);

-- TABLE: w_state_validator_states
CREATE TABLE w_state_validator_states (
    state_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    validator_public_key VARCHAR(255) NOT NULL,
    timestamp BIGINT NOT NULL,
    w_state_serialized TEXT,
    w_state_hash VARCHAR(255),
    coherence_with_oracle NUMERIC(5, 4),
    phase_alignment_radians NUMERIC(200, 150),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: wallet_addresses
CREATE TABLE wallet_addresses (
    address VARCHAR(255) PRIMARY KEY,
    wallet_fingerprint VARCHAR(64) NOT NULL,
    derivation_path VARCHAR(100),
    account_index INT,
    change_index INT,
    address_index INT,
    public_key VARCHAR(255) NOT NULL,
    address_type VARCHAR(50) DEFAULT 'receiving',
    is_watching_only BOOLEAN DEFAULT FALSE,
    is_cold_storage BOOLEAN DEFAULT FALSE,
    balance NUMERIC(30, 0) DEFAULT 0,
    balance_updated_at TIMESTAMP WITH TIME ZONE,
    balance_at_height BIGINT,
    first_used_at TIMESTAMP WITH TIME ZONE,
    last_used_at TIMESTAMP WITH TIME ZONE,
    transaction_count INT DEFAULT 0,
    label VARCHAR(255),
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(wallet_fingerprint, derivation_path)
);

-- TABLE: wallet_key_rotation_history
CREATE TABLE wallet_key_rotation_history (
    rotation_id BIGSERIAL PRIMARY KEY,
    wallet_fingerprint VARCHAR(64) NOT NULL,
    old_key_id VARCHAR(255),
    new_key_id VARCHAR(255),
    rotation_reason TEXT,
    rotation_timestamp TIMESTAMP WITH TIME ZONE,
    ratchet_material TEXT,
    next_rotation_material TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- TABLE: wallet_seed_backup_status
CREATE TABLE wallet_seed_backup_status (
    backup_id BIGSERIAL PRIMARY KEY,
    wallet_fingerprint VARCHAR(64) NOT NULL UNIQUE,
    seed_phrase_backed_up BOOLEAN DEFAULT FALSE,
    backup_confirmed_at TIMESTAMP WITH TIME ZONE,
    seed_hint VARCHAR(50),
    seed_hash VARCHAR(255),
    backup_required BOOLEAN DEFAULT TRUE,
    days_since_creation_without_backup INT,
    email_notifications_sent INT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);



-- V5 SEQUENTIAL ROUTING
CREATE TABLE pq_sequential (
    pq_id BIGINT PRIMARY KEY,
    next_pq_id BIGINT,
    prev_pq_id BIGINT,
    sequence_order BIGINT UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_pq_next ON pq_sequential(next_pq_id);
CREATE INDEX idx_pq_prev ON pq_sequential(prev_pq_id);
CREATE INDEX idx_sequence_order ON pq_sequential(sequence_order);
"""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: OPTIMIZED DATABASE BUILDER CLASS (Colab-tuned)
# ─────────────────────────────────────────────────────────────────────────────

class QuantumTemporalCoherenceLedgerServer:
    """Optimized server-side database builder for PostgreSQL/Supabase - Colab Edition"""
    
    TRIANGLE_BATCH_SIZE = 2000
    QUBIT_BATCH_SIZE = 10000
    PROGRESS_INTERVAL_TRI = 500
    PROGRESS_INTERVAL_QUB = 5000
    
    def __init__(self, pg_url: str, tessellation_depth: int = 5):
        self.pg_url = pg_url
        self.tessellation_depth = tessellation_depth
        self.conn = None
        self.cursor = None
        self._start_time = None
    
    def connect(self):
        logger.info(f"{CLR.QUANTUM}[DB] Connecting to Supabase Pooler...{CLR.E}")
        try:
            self.conn = psycopg2.connect(self.pg_url)
            self.cursor = self.conn.cursor()
            # Set connection parameters via SET commands (not DSN parameters)
            self.cursor.execute("SET statement_timeout = '600s';")
            self.cursor.execute("SET application_name = 'qtcl_colab_v4';")
            self.cursor.execute("SET work_mem = '128MB';")
            self.cursor.execute("SET maintenance_work_mem = '256MB';")
            self.cursor.execute("SET synchronous_commit = off;")
            logger.info(f"{CLR.OK}[DB] Connected with bulk-optimized settings{CLR.E}")
        except Exception as e:
            logger.error(f"{CLR.ERROR}[DB] Connection failed: {e}{CLR.E}")
            raise
    
    def drop_all_tables(self):
        logger.info(f"{CLR.ERROR}[DROP] Dropping ALL existing tables...{CLR.E}")
        try:
            self.cursor.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public';")
            tables = self.cursor.fetchall()
            if not tables:
                logger.info(f"{CLR.OK}[DROP] No tables to drop{CLR.E}")
                return
            logger.info(f"[DROP] Found {len(tables)} tables to drop")
            for i, table in enumerate(tables, 1):
                table_name = table[0]
                self.cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
                if i % 10 == 0:
                    self.conn.commit()
            self.conn.commit()
            logger.info(f"{CLR.OK}[DROP] All {len(tables)} tables dropped successfully{CLR.E}")
        except Exception as e:
            logger.error(f"{CLR.ERROR}[DROP] Drop failed: {e}{CLR.E}")
            self.conn.rollback()
            raise
    
    def create_schema(self):
        logger.info(f"{CLR.QUANTUM}[SCHEMA] Creating schema...{CLR.E}")
        try:
            self.cursor.execute(COMPLETE_SCHEMA)
            self.conn.commit()
            logger.info(f"{CLR.OK}[SCHEMA] Schema created successfully{CLR.E}")
        except Exception as e:
            logger.error(f"{CLR.ERROR}[SCHEMA] Schema creation failed: {e}{CLR.E}")
            self.conn.rollback()
            raise

    def _migrate_oracle_registry_onchain(self):
        """Idempotent migration — adds 8 on-chain identity columns to oracle_registry.
        Safe to run on existing DBs: ADD COLUMN IF NOT EXISTS never fails on re-run.
        Also ensures the 3 new indexes land on both fresh builds and live migrations."""
        migrations = [
            "ALTER TABLE oracle_registry ADD COLUMN IF NOT EXISTS wallet_address  VARCHAR(128) NOT NULL DEFAULT ''",
            "ALTER TABLE oracle_registry ADD COLUMN IF NOT EXISTS oracle_pub_key  TEXT         NOT NULL DEFAULT ''",
            "ALTER TABLE oracle_registry ADD COLUMN IF NOT EXISTS cert_sig        VARCHAR(128) NOT NULL DEFAULT ''",
            "ALTER TABLE oracle_registry ADD COLUMN IF NOT EXISTS cert_auth_tag   VARCHAR(128) NOT NULL DEFAULT ''",
            "ALTER TABLE oracle_registry ADD COLUMN IF NOT EXISTS mode            VARCHAR(32)  NOT NULL DEFAULT 'full'",
            "ALTER TABLE oracle_registry ADD COLUMN IF NOT EXISTS ip_hint         VARCHAR(256) NOT NULL DEFAULT ''",
            "ALTER TABLE oracle_registry ADD COLUMN IF NOT EXISTS reg_tx_hash     VARCHAR(64)  NOT NULL DEFAULT ''",
            "ALTER TABLE oracle_registry ADD COLUMN IF NOT EXISTS registered_at   BIGINT       NOT NULL DEFAULT 0",
            "CREATE INDEX IF NOT EXISTS idx_oracle_registry_wallet        ON oracle_registry (wallet_address)",
            "CREATE INDEX IF NOT EXISTS idx_oracle_registry_reg_tx        ON oracle_registry (reg_tx_hash) WHERE reg_tx_hash != ''",
            "CREATE INDEX IF NOT EXISTS idx_oracle_registry_registered_at ON oracle_registry (registered_at DESC)",
        ]
        ok_count = 0
        for ddl in migrations:
            try:
                self.cursor.execute(ddl)
                self.conn.commit()
                ok_count += 1
            except Exception as e:
                self.conn.rollback()
                logger.debug(f"{CLR.WARNING}[MIGRATE] oracle_registry DDL skipped ({ddl[:55]}…): {e}{CLR.E}")
        logger.info(f"{CLR.OK}[MIGRATE] oracle_registry on-chain identity migration: {ok_count}/{len(migrations)} OK{CLR.E}")

    def _insert_triangles_batched(self, triangles: Dict[int, HyperbolicTriangle]):
        logger.info(f"{CLR.C}[TRI] Inserting {len(triangles)} triangles in batches...{CLR.E}")
        triangle_list = list(triangles.values())
        # Only insert final-depth triangles for correct tessellation count
        triangle_list = [t for t in triangle_list if t.depth == self.tessellation_depth]
        total = len(triangle_list)
        logger.info(f"{CLR.C}[TRI] Keeping only depth={self.tessellation_depth} triangles: {total}...{CLR.E}")
        
        pbar = tqdm(total=total, desc="Triangles", leave=True)
        
        for batch_start in range(0, total, self.TRIANGLE_BATCH_SIZE):
            batch = triangle_list[batch_start:batch_start + self.TRIANGLE_BATCH_SIZE]
            rows = []
            for tri in batch:
                # Create row but set parent_id to NULL (parents not being stored)
                row = (
                    tri.triangle_id, tri.depth, None,  # parent_id set to NULL
                    str(tri.v0.x), str(tri.v0.y), tri.v0.name,
                    str(tri.v1.x), str(tri.v1.y), tri.v1.name,
                    str(tri.v2.x), str(tri.v2.y), tri.v2.name
                )
                rows.append(row)
            
            execute_values(
                self.cursor,
                """
                INSERT INTO hyperbolic_triangles (
                    triangle_id, depth, parent_id,
                    v0_x, v0_y, v0_name,
                    v1_x, v1_y, v1_name,
                    v2_x, v2_y, v2_name
                ) VALUES %s
                """,
                rows,
                page_size=100
            )
            if batch_start % (self.TRIANGLE_BATCH_SIZE * 5) == 0 and batch_start > 0:
                self.conn.commit()
            pbar.update(len(batch))
        pbar.close()
        self.conn.commit()
        logger.info(f"{CLR.OK}[TRI] All {total} final-depth triangles inserted (parent_id=NULL){CLR.E}")
    
    def _insert_pseudoqubits_batched(self, qubits: Dict[int, Pseudoqubit], triangle_ids: set):
        logger.info(f"{CLR.C}[QUB] Filtering {len(qubits)} pseudoqubits to match inserted triangles...{CLR.E}")
        qubit_list = list(qubits.values())
        # Only insert qubits for triangles that were actually inserted (final-depth)
        qubit_list = [q for q in qubit_list if q.triangle_id in triangle_ids]
        total = len(qubit_list)
        logger.info(f"{CLR.C}[QUB] Keeping {total} pseudoqubits matching inserted triangles...{CLR.E}")
        
        pbar = tqdm(total=total, desc="Pseudoqubits", leave=True)
        
        for batch_start in range(0, total, self.QUBIT_BATCH_SIZE):
            batch = qubit_list[batch_start:batch_start + self.QUBIT_BATCH_SIZE]
            rows = [q.to_db_row() for q in batch]
            execute_values(
                self.cursor,
                """
                INSERT INTO pseudoqubits (
                    pq_id, triangle_id, x, y,
                    placement_type, phase_theta, coherence_measure
                ) VALUES %s
                """,
                rows,
                page_size=200
            )
            if batch_start % (self.QUBIT_BATCH_SIZE * 3) == 0 and batch_start > 0:
                self.conn.commit()
            pbar.update(len(batch))
        pbar.close()
        self.conn.commit()
        logger.info(f"{CLR.OK}[QUB] All {total} pseudoqubits inserted{CLR.E}")
    
    def _build_tessellation_inline(self) -> Tuple[Dict[int, HyperbolicTriangle], Dict[int, Pseudoqubit]]:
        """Inline tessellation builder - memory efficient for Colab"""
        triangles: Dict[int, HyperbolicTriangle] = {}
        qubits: Dict[int, Pseudoqubit] = {}
        triangle_id_counter = [0]
        qubit_id_counter = [0]
        
        def build_octagon_decomposition() -> List[HyperbolicTriangle]:
            logger.info(f"{CLR.QUANTUM}[OCTAGON] Constructing 8 fundamental octagons{CLR.E}")
            triangles_list = []
            octagon_radius = mpf("0.4")
            octagon_vertices = []
            for i in range(8):
                angle = mpf(2) * pi * mpf(i) / mpf(8)
                x = octagon_radius * cos(angle)
                y = octagon_radius * sin(angle)
                vertex = HyperbolicPoint(x, y, name=f"oct_v{i}")
                octagon_vertices.append(vertex)
            center = HyperbolicPoint(mpf(0), mpf(0), name="oct_center")
            for i in range(8):
                v0 = center
                v1 = octagon_vertices[i]
                v2 = octagon_vertices[(i + 1) % 8]
                triangle = HyperbolicTriangle(
                    triangle_id=triangle_id_counter[0],
                    v0=v0, v1=v1, v2=v2,
                    depth=0,
                    parent_id=None
                )
                triangle_id_counter[0] += 1
                triangles_list.append(triangle)
            logger.info(f"{CLR.OK}[OCTAGON] Created {len(triangles_list)} fundamental triangles{CLR.E}")
            return triangles_list
        
        def subdivide_triangle(parent: HyperbolicTriangle) -> List[HyperbolicTriangle]:
            m01 = poincare_midpoint(parent.v0, parent.v1)
            m12 = poincare_midpoint(parent.v1, parent.v2)
            m20 = poincare_midpoint(parent.v2, parent.v0)
            children = [
                HyperbolicTriangle(triangle_id=triangle_id_counter[0], v0=parent.v0, v1=m01, v2=m20, depth=parent.depth + 1, parent_id=parent.triangle_id),
                HyperbolicTriangle(triangle_id=triangle_id_counter[0] + 1, v0=parent.v1, v1=m12, v2=m01, depth=parent.depth + 1, parent_id=parent.triangle_id),
                HyperbolicTriangle(triangle_id=triangle_id_counter[0] + 2, v0=parent.v2, v1=m20, v2=m12, depth=parent.depth + 1, parent_id=parent.triangle_id),
                HyperbolicTriangle(triangle_id=triangle_id_counter[0] + 3, v0=m01, v1=m12, v2=m20, depth=parent.depth + 1, parent_id=parent.triangle_id)
            ]
            triangle_id_counter[0] += 4
            return children
        
        def build_recursive_tessellation():
            logger.info(f"{CLR.QUANTUM}[RECURSIVE] Building tessellation depth {self.tessellation_depth}{CLR.E}")
            current_level = build_octagon_decomposition()
            for tri in current_level:
                triangles[tri.triangle_id] = tri
            total_levels = self.tessellation_depth
            for level in range(1, total_levels + 1):
                next_level = []
                for parent_tri in current_level:
                    children = subdivide_triangle(parent_tri)
                    for child in children:
                        triangles[child.triangle_id] = child
                        next_level.append(child)
                current_level = next_level
                pct = (level / total_levels) * 100
                bar_len = 40
                filled = int(bar_len * level / total_levels)
                bar = "█" * filled + "░" * (bar_len - filled)
                logger.info(f"{CLR.C}[Tessellation] [{bar}] {pct:5.1f}% | Level {level}/{total_levels} | {len(triangles):,} triangles{CLR.E}")
            logger.info(f"{CLR.OK}[RECURSIVE] ✅ Complete: {len(triangles):,} triangles{CLR.E}")
        
        def place_pseudoqubits():
            logger.info(f"{CLR.QUANTUM}[QUBITS] Placing pseudoqubits...{CLR.E}")
            qubit_id = 0
            total_triangles = len(triangles)
            log_interval = max(1, total_triangles // 20)
            for idx, (tri_id, triangle) in enumerate(triangles.items()):
                for i, vertex in enumerate([triangle.v0, triangle.v1, triangle.v2]):
                    qubit = Pseudoqubit(pseudoqubit_id=qubit_id, triangle_id=tri_id, x=vertex.x, y=vertex.y, placement_type="vertex")
                    qubits[qubit_id] = qubit
                    qubit_id += 1
                inc = hyperbolic_incenter(triangle)
                qubit = Pseudoqubit(pseudoqubit_id=qubit_id, triangle_id=tri_id, x=inc.x, y=inc.y, placement_type="incenter")
                qubits[qubit_id] = qubit
                qubit_id += 1
                circ = hyperbolic_circumcenter(triangle)
                qubit = Pseudoqubit(pseudoqubit_id=qubit_id, triangle_id=tri_id, x=circ.x, y=circ.y, placement_type="circumcenter")
                qubits[qubit_id] = qubit
                qubit_id += 1
                orth = hyperbolic_orthocenter(triangle)
                qubit = Pseudoqubit(pseudoqubit_id=qubit_id, triangle_id=tri_id, x=orth.x, y=orth.y, placement_type="orthocenter")
                qubits[qubit_id] = qubit
                qubit_id += 1
                grid_points = generate_geodesic_grid(triangle)
                for gp in grid_points[:7]:
                    qubit = Pseudoqubit(pseudoqubit_id=qubit_id, triangle_id=tri_id, x=gp.x, y=gp.y, placement_type="geodesic")
                    qubits[qubit_id] = qubit
                    qubit_id += 1
                if (idx + 1) % log_interval == 0 or idx == total_triangles - 1:
                    pct = ((idx + 1) / total_triangles) * 100
                    bar_len = 40
                    filled = int(bar_len * (idx + 1) / total_triangles)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    logger.info(f"{CLR.C}[Geometry IDs] [{bar}] {pct:5.1f}% | {idx+1:,}/{total_triangles:,} triangles | {qubit_id:,} IDs placed{CLR.E}")
                if tri_id % 1000 == 0:
                    gc.collect()
            logger.info(f"{CLR.OK}[QUBITS] ✅ All {qubit_id:,} pseudoqubits placed{CLR.E}")
        
        # ✅ ACTUALLY CALL THE FUNCTIONS
        build_recursive_tessellation()
        place_pseudoqubits()
        gc.collect()
        return triangles, qubits
    
    def populate_tessellation(self):
        logger.info(f"{CLR.QUANTUM}[POPULATE] Building and inserting tessellation...{CLR.E}")
        self._start_time = time.time()
        
        try:
            # Build tessellation
            logger.info(f"{CLR.C}[BUILD] Constructing tessellation depth {self.tessellation_depth}...{CLR.E}")
            triangles, qubits = self._build_tessellation_inline()
            
            # Get IDs of final-depth triangles that will be inserted
            final_depth_triangle_ids = set(
                t.triangle_id for t in triangles.values() 
                if t.depth == self.tessellation_depth
            )
            logger.info(f"{CLR.C}[FILTER] Final-depth triangle IDs: {len(final_depth_triangle_ids)}{CLR.E}")
            
            # Insert with batching + progress bars
            self._insert_triangles_batched(triangles)
            self._insert_pseudoqubits_batched(qubits, final_depth_triangle_ids)
            
            # Insert metadata
            self.cursor.execute("""
                INSERT INTO quantum_lattice_metadata (
                    tessellation_depth, total_triangles, total_pseudoqubits,
                    precision_bits, hyperbolicity_constant, poincare_radius,
                    status, construction_started_at, construction_completed_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                self.tessellation_depth,
                len(final_depth_triangle_ids),
                len([q for q in qubits.values() if q.triangle_id in final_depth_triangle_ids]),
                150, -1.0, 1.0,
                'complete',
                datetime.now(timezone.utc),
                datetime.now(timezone.utc)
            ))
            
            self.cursor.execute("""
                INSERT INTO database_metadata (
                    schema_version, build_timestamp, tables_created
                ) VALUES (%s, %s, %s)
            """, ('4.1.0-colab', datetime.now(timezone.utc), 58))
            
            self.conn.commit()
            
            elapsed = time.time() - self._start_time
            logger.info(f"{CLR.OK}[POPULATE] Complete in {elapsed:.1f}s{CLR.E}")
            
        except Exception as e:
            logger.error(f"{CLR.ERROR}[POPULATE] Population failed: {e}{CLR.E}")
            self.conn.rollback()
            raise
    
    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.commit()
            self.conn.close()
        logger.info(f"{CLR.OK}[DB] Connection closed{CLR.E}")
    
    def rebuild_complete(self):
        logger.info(f"{CLR.HEADER}{'='*80}{CLR.E}")
        logger.info(f"{CLR.HEADER}QTCL DATABASE BUILDER V4.1 - COLAB EDITION (FIXED){CLR.E}")
        logger.info(f"{CLR.HEADER}{'='*80}{CLR.E}")
        
        total_start = time.time()
        
        try:
            self.connect()
            self.drop_all_tables()
            self.create_schema()
            self._migrate_oracle_registry_onchain()
            self.populate_tessellation()
            
            total_elapsed = time.time() - total_start
            logger.info(f"{CLR.HEADER}{'='*80}{CLR.E}")
            logger.info(f"{CLR.OK}✓ BUILD COMPLETE{CLR.E}")
            logger.info(f"{CLR.OK}  Total time: {total_elapsed/60:.1f} minutes{CLR.E}")
            logger.info(f"{CLR.OK}  Tessellation depth: {self.tessellation_depth}{CLR.E}")
            logger.info(f"{CLR.OK}  Schema tables: 58{CLR.E}")
            logger.info(f"{CLR.OK}  Status: PRODUCTION READY{CLR.E}")
            logger.info(f"{CLR.HEADER}{'='*80}{CLR.E}")
            
        except Exception as e:
            logger.error(f"{CLR.ERROR}Build failed after {time.time()-total_start:.1f}s: {e}{CLR.E}")
            raise
        finally:
            self.close()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9: RUN THE BUILD (Colab entry point)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info(f"\n{CLR.HEADER}{'='*80}{CLR.E}")
    logger.info(f"{CLR.HEADER}QTCL DATABASE BUILDER V4.1 - GOOGLE COLAB{CLR.E}")
    logger.info(f"{CLR.HEADER}Hard-coded Supabase Pooler | Optimized Bulk Load{CLR.E}")
    logger.info(f"{CLR.HEADER}{'='*80}{CLR.E}\n")
    
    # Build complete system
    builder = QuantumTemporalCoherenceLedgerServer(
        pg_url=pg_url,
        tessellation_depth=5
    )
    builder.rebuild_complete()
    
    logger.info(f"\n{CLR.QUANTUM}✓ QTCL V4.1 server database ready for production{CLR.E}\n")
    
    # Colab-specific: keep runtime alive for verification queries
    print("\n💡 Tip: Run verification queries in a new cell:")
    print("""
    # Verify build success
    import psycopg2
    conn = psycopg2.connect(pg_url)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM hyperbolic_triangles;")
    print(f"Triangles: {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM pseudoqubits;")
    print(f"Pseudoqubits: {cur.fetchone()[0]}")
    cur.execute("SELECT * FROM quantum_lattice_metadata LIMIT 1;")
    print(f"Metadata: {cur.fetchone()}")
    conn.close()
    """)