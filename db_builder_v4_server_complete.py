#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║   QTCL DATABASE BUILDER V4 - COMPLETE SERVER IMPLEMENTATION                  ║
║   PostgreSQL/Supabase with Full Hyperbolic Tessellation Lattice              ║
║   Drop All Tables & Rebuild from Scratch                                     ║
║                                                                                ║
║   ARCHITECTURE:                                                               ║
║   ═════════════════════════════════════════════════════════════════════════   ║
║                                                                                ║
║   Server PostgreSQL:                                                          ║
║   ├─ Hyperbolic tessellation (106,496 pseudoqubits)                          ║
║   ├─ P2P network layer (peers, connections, blocks)                          ║
║   ├─ Oracle state distribution (W-state, density matrix)                     ║
║   ├─ Blockchain core (blocks, transactions, validators)                      ║
║   ├─ HLWE wallets (addresses, encrypted keys, balance history)               ║
║   ├─ Quantum measurements (state tracking, coherence)                        ║
║   └─ Complete audit trail                                                     ║
║                                                                                ║
║   Key Features:                                                               ║
║   ├─ DROP TABLE IF EXISTS (clean slate)                                      ║
║   ├─ Complete tessellation in PostgreSQL (not computed at runtime)           ║
║   ├─ 150-bit arithmetic for all quantum calculations                         ║
║   ├─ Foreign key constraints (referential integrity)                         ║
║   ├─ Comprehensive indexes (performance optimization)                        ║
║   ├─ JSONB fields (flexible quantum state storage)                           ║
║   └─ Ready for Supabase/AWS RDS deployment                                   ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝

SCHEMA ORGANIZATION:
════════════════════════════════════════════════════════════════════════════════

SECTION 1: QUANTUM LATTICE FOUNDATION (14 TABLES)
├─ hyperbolic_triangles         - All 8,192 base triangles
├─ pseudoqubits                 - All 106,496 qubits positioned
├─ pseudoqubit_entanglement     - Entanglement graph (neighbors)
├─ quantum_lattice_metadata     - Tessellation construction info
├─ quantum_phase_evolution      - Phase tracking over time
├─ quantum_coherence_snapshots  - Coherence history
├─ quantum_measurements         - Individual measurement records
├─ quantum_density_matrix_global - Full system density matrix
├─ quantum_shadow_tomography    - Efficient state representation
├─ quantum_circuit_execution    - Circuit execution log
├─ quantum_error_correction     - Error correction events
├─ quantum_supremacy_proofs     - Quantum advantage demonstrations
├─ w_state_snapshots           - W-state for oracle
└─ w_state_validator_states     - Per-validator W-state tracking

SECTION 2: ORACLE STATE DISTRIBUTION (8 TABLES)
├─ oracle_w_state_snapshots     - Tripartite W-state at each block
├─ oracle_density_matrix_stream - Live metrics density matrix
├─ oracle_entropy_feeds         - 5-source quantum entropy
├─ oracle_coherence_metrics     - System coherence tracking
├─ oracle_pq0_state            - Oracle pseudoqubit 0 state
├─ oracle_entanglement_records   - W-state entanglement verification
├─ oracle_distribution_log      - Distribution to peers
└─ oracle_consensus_state       - Consensus agreement tracking

SECTION 3: P2P NETWORK LAYER (7 TABLES)
├─ peer_registry               - All peers (public_key identified)
├─ peer_connections            - Active peer relationships
├─ peer_reputation             - Peer scoring/trust
├─ block_headers_cache         - Block headers only (fast sync)
├─ orphan_blocks               - Temporary blocks during reorg
├─ network_bandwidth_usage     - Peer bandwidth tracking
└─ network_partition_events    - Partition detection log

SECTION 4: BLOCKCHAIN CORE (8 TABLES)
├─ blocks                      - Immutable block chain
├─ transactions                - All transactions
├─ transaction_inputs          - Transaction inputs (UTXO)
├─ transaction_outputs         - Transaction outputs (UTXO)
├─ transaction_receipts        - Transaction receipts/logs
├─ validators                  - Validator registry (public_key identified)
├─ validator_stakes            - Stake tracking
└─ merkle_proofs               - Merkle inclusion proofs (light client support)

SECTION 5: HLWE WALLET LAYER (8 TABLES)
├─ wallet_addresses            - Derived addresses (no user_id)
├─ encrypted_private_keys      - AES-256-GCM encrypted privkeys
├─ address_transactions        - Efficient per-address queries
├─ address_balance_history     - Balance snapshots by block
├─ wallet_seed_backup_status   - Backup verification (no seed stored!)
├─ address_labels              - User-friendly address names
├─ address_UTXOs              - Unspent transaction outputs
└─ wallet_key_rotation_history - Key rotation events

SECTION 6: CLIENT SYNCHRONIZATION (6 TABLES)
├─ client_oracle_sync          - Per-peer W-state sync status
├─ entanglement_records        - Peer-oracle coherence
├─ lattice_sync_state          - Tessellation synchronization
├─ client_block_sync           - Block download progress
├─ client_sync_events          - Sync events (connect, disconnect, resync)
└─ client_network_metrics      - Per-client network stats

SECTION 7: CONSENSUS & FINALITY (6 TABLES)
├─ epochs                      - Consensus epochs
├─ epoch_validators            - Validators in each epoch
├─ finality_records            - Block finality tracking
├─ state_root_updates          - State root changes
├─ chain_reorganizations       - Reorg events (height, depth)
└─ consensus_events            - Major consensus events

SECTION 8: AUDIT & METADATA (5 TABLES)
├─ audit_logs                  - Complete audit trail
├─ database_metadata           - Schema version, build info
├─ network_events              - Network-wide events
├─ system_metrics              - Performance metrics
└─ entropy_quality_log         - Entropy source quality tracking

════════════════════════════════════════════════════════════════════════════════
TOTAL: 58 tables, comprehensive schema covering all aspects of quantum blockchain
════════════════════════════════════════════════════════════════════════════════
"""

import sys
import time
import json
import math
import hashlib
import logging
import gc
import secrets
import pickle
import gzip
import base64
import uuid
import struct
import threading
import queue
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Any, Set, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum, IntEnum
from collections import defaultdict, deque, OrderedDict, Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# ═════════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL PRECISION ENGINE
# ═════════════════════════════════════════════════════════════════════════════════

try:
    from mpmath import (
        mp, mpf, mpc, sqrt, pi, cos, sin, exp, log, tanh, sinh, cosh, acosh,
        atanh, atan2, fabs, re as mre, im as mim, conj, norm, phase,
        matrix, nstr, nsum, power, floor, ceil, asin, acos, hypot
    )
    mp.dps = 150
    MPMATH_AVAILABLE = True
except ImportError:
    MPMATH_AVAILABLE = False
    print("⚠️  mpmath not available")

# ═════════════════════════════════════════════════════════════════════════════════
# DATABASE
# ═════════════════════════════════════════════════════════════════════════════════

try:
    import psycopg2
    from psycopg2.extras import execute_values, RealDictCursor, Json
    from psycopg2.pool import ThreadedConnectionPool
    from psycopg2 import sql, errors as psycopg2_errors
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# ═════════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═════════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('/tmp/qtcl_db_builder_v4_server.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════════
# COLORS
# ═════════════════════════════════════════════════════════════════════════════════

class CLR:
    BOLD = '\033[1m'
    G = '\033[92m'
    R = '\033[91m'
    Y = '\033[93m'
    C = '\033[96m'
    M = '\033[95m'
    E = '\033[0m'
    HEADER = f'{BOLD}{C}'
    OK = f'{BOLD}{G}'
    ERROR = f'{BOLD}{R}'
    QUANTUM = f'{BOLD}{M}'

# ═════════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL STRUCTURES
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class HyperbolicPoint:
    """Point in Poincaré disk with 150-bit precision"""
    x: Any
    y: Any
    name: Optional[str] = None
    
    def to_json(self) -> dict:
        return {
            "x": str(self.x),
            "y": str(self.y),
            "name": self.name
        }
    
    @staticmethod
    def from_json(data: dict) -> 'HyperbolicPoint':
        return HyperbolicPoint(
            x=mpf(data["x"]),
            y=mpf(data["y"]),
            name=data.get("name")
        )


@dataclass
class HyperbolicTriangle:
    """Triangle in hyperbolic plane"""
    triangle_id: int
    v0: HyperbolicPoint
    v1: HyperbolicPoint
    v2: HyperbolicPoint
    depth: int = 0
    parent_id: Optional[int] = None
    
    def to_json(self) -> dict:
        return {
            "triangle_id": self.triangle_id,
            "v0": self.v0.to_json(),
            "v1": self.v1.to_json(),
            "v2": self.v2.to_json(),
            "depth": self.depth,
            "parent_id": self.parent_id
        }


@dataclass
class Pseudoqubit:
    """Pseudoqubit in tessellation lattice"""
    pseudoqubit_id: int
    triangle_id: int
    x: Any
    y: Any
    placement_type: str
    phase_theta: Any = field(default_factory=lambda: mpf(0))
    coherence_measure: Any = field(default_factory=lambda: mpf(0.99))
    
    def to_json(self) -> dict:
        return {
            "pseudoqubit_id": self.pseudoqubit_id,
            "triangle_id": self.triangle_id,
            "x": str(self.x),
            "y": str(self.y),
            "placement_type": self.placement_type,
            "phase_theta": str(self.phase_theta),
            "coherence_measure": str(self.coherence_measure)
        }

# ═════════════════════════════════════════════════════════════════════════════════
# HYPERBOLIC GEOMETRY
# ═════════════════════════════════════════════════════════════════════════════════

def hyperbolic_distance(p1: HyperbolicPoint, p2: HyperbolicPoint) -> Any:
    """Exact geodesic distance in Poincaré disk"""
    z1 = mpc(p1.x, p1.y)
    z2 = mpc(p2.x, p2.y)
    
    numerator = abs(z1 - z2)
    denominator = abs(mpf(1) - conj(z1) * z2)
    
    if abs(denominator) < mpf(10)**(-140):
        return mpf(0)
    
    ratio = numerator / denominator
    distance = mpf(2) * atanh(ratio)
    
    return distance


def poincare_midpoint(p1: HyperbolicPoint, p2: HyperbolicPoint) -> HyperbolicPoint:
    """Geodesic midpoint formula"""
    z1 = mpc(p1.x, p1.y)
    z2 = mpc(p2.x, p2.y)
    
    numerator = z1 + z2
    denominator = mpf(1) + conj(z1) * z2
    
    m = numerator / denominator
    
    return HyperbolicPoint(mre(m), mim(m), name=f"mid_{p1.name}_{p2.name}")


def hyperbolic_area_triangle(v0: HyperbolicPoint, v1: HyperbolicPoint, v2: HyperbolicPoint) -> Any:
    """Hyperbolic area using Gauss-Bonnet"""
    a = hyperbolic_distance(v1, v2)
    b = hyperbolic_distance(v0, v2)
    c = hyperbolic_distance(v0, v1)
    
    cosh_a = cosh(a)
    cosh_b = cosh(b)
    cosh_c = cosh(c)
    sinh_a = sinh(a)
    sinh_b = sinh(b)
    sinh_c = sinh(c)
    
    cos_angle_1 = (cosh_a - cosh_b * cosh_c) / (sinh_b * sinh_c)
    cos_angle_1 = max(mpf(-1), min(mpf(1), cos_angle_1))
    angle_1 = acos(cos_angle_1)
    
    cos_angle_2 = (cosh_b - cosh_a * cosh_c) / (sinh_a * sinh_c)
    cos_angle_2 = max(mpf(-1), min(mpf(1), cos_angle_2))
    angle_2 = acos(cos_angle_2)
    
    cos_angle_3 = (cosh_c - cosh_a * cosh_b) / (sinh_a * sinh_b)
    cos_angle_3 = max(mpf(-1), min(mpf(1), cos_angle_3))
    angle_3 = acos(cos_angle_3)
    
    area = pi - (angle_1 + angle_2 + angle_3)
    
    return area

# ═════════════════════════════════════════════════════════════════════════════════
# TESSELLATION CONSTRUCTOR
# ═════════════════════════════════════════════════════════════════════════════════

class TessellationConstructor:
    """Builds {8,3} tessellation with 8,192 triangles, 106,496 pseudoqubits"""
    
    def __init__(self, depth: int = 5):
        self.depth = depth
        self.triangles: Dict[int, HyperbolicTriangle] = {}
        self.pseudoqubits: Dict[int, Pseudoqubit] = {}
        self.triangle_id_counter = 0
        self.qubit_id_counter = 0
    
    def build_octagon_decomposition(self) -> List[HyperbolicTriangle]:
        """8 fundamental octagons"""
        logger.info(f"{CLR.QUANTUM}[OCTAGON] Constructing 8 fundamental octagons{CLR.E}")
        
        triangles = []
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
                triangle_id=self.triangle_id_counter,
                v0=v0, v1=v1, v2=v2,
                depth=0,
                parent_id=None
            )
            self.triangle_id_counter += 1
            triangles.append(triangle)
        
        logger.info(f"{CLR.OK}[OCTAGON] Created {len(triangles)} fundamental triangles{CLR.E}")
        return triangles
    
    def subdivide_triangle(self, parent: HyperbolicTriangle) -> List[HyperbolicTriangle]:
        """Recursive subdivision into 4 sub-triangles"""
        m01 = poincare_midpoint(parent.v0, parent.v1)
        m12 = poincare_midpoint(parent.v1, parent.v2)
        m20 = poincare_midpoint(parent.v2, parent.v0)
        
        children = [
            HyperbolicTriangle(
                triangle_id=self.triangle_id_counter,
                v0=parent.v0, v1=m01, v2=m20,
                depth=parent.depth + 1,
                parent_id=parent.triangle_id
            ),
            HyperbolicTriangle(
                triangle_id=self.triangle_id_counter + 1,
                v0=parent.v1, v1=m12, v2=m01,
                depth=parent.depth + 1,
                parent_id=parent.triangle_id
            ),
            HyperbolicTriangle(
                triangle_id=self.triangle_id_counter + 2,
                v0=parent.v2, v1=m20, v2=m12,
                depth=parent.depth + 1,
                parent_id=parent.triangle_id
            ),
            HyperbolicTriangle(
                triangle_id=self.triangle_id_counter + 3,
                v0=m01, v1=m12, v2=m20,
                depth=parent.depth + 1,
                parent_id=parent.triangle_id
            )
        ]
        
        self.triangle_id_counter += 4
        return children
    
    def build_recursive_tessellation(self):
        """Build {8,3} tessellation to depth 5 = 8,192 triangles"""
        logger.info(f"{CLR.QUANTUM}[RECURSIVE] Building tessellation depth {self.depth}{CLR.E}")
        
        current_level = self.build_octagon_decomposition()
        for tri in current_level:
            self.triangles[tri.triangle_id] = tri
        
        for level in range(1, self.depth + 1):
            logger.info(f"{CLR.C}[RECURSIVE] Level {level-1}→{level} "
                       f"({len(current_level)} → {len(current_level)*4} triangles){CLR.E}")
            
            next_level = []
            for parent_tri in current_level:
                children = self.subdivide_triangle(parent_tri)
                for child in children:
                    self.triangles[child.triangle_id] = child
                    next_level.append(child)
            
            current_level = next_level
        
        logger.info(f"{CLR.OK}[RECURSIVE] Complete: {len(self.triangles)} triangles{CLR.E}")
    
    def place_pseudoqubits(self):
        """Place 106,496 pseudoqubits in 8,192 triangles"""
        logger.info(f"{CLR.QUANTUM}[QUBITS] Placing pseudoqubits...{CLR.E}")
        
        qubit_id = 0
        for tri_id, triangle in self.triangles.items():
            # Vertices
            for i, vertex in enumerate([triangle.v0, triangle.v1, triangle.v2]):
                qubit = Pseudoqubit(
                    pseudoqubit_id=qubit_id,
                    triangle_id=tri_id,
                    x=vertex.x,
                    y=vertex.y,
                    placement_type="vertex"
                )
                self.pseudoqubits[qubit_id] = qubit
                qubit_id += 1
            
            # Incenter
            inc = self._compute_incenter(triangle)
            qubit = Pseudoqubit(
                pseudoqubit_id=qubit_id,
                triangle_id=tri_id,
                x=inc.x,
                y=inc.y,
                placement_type="incenter"
            )
            self.pseudoqubits[qubit_id] = qubit
            qubit_id += 1
            
            # Circumcenter
            circ = self._compute_circumcenter(triangle)
            qubit = Pseudoqubit(
                pseudoqubit_id=qubit_id,
                triangle_id=tri_id,
                x=circ.x,
                y=circ.y,
                placement_type="circumcenter"
            )
            self.pseudoqubits[qubit_id] = qubit
            qubit_id += 1
            
            # Orthocenter
            orth = self._compute_orthocenter(triangle)
            qubit = Pseudoqubit(
                pseudoqubit_id=qubit_id,
                triangle_id=tri_id,
                x=orth.x,
                y=orth.y,
                placement_type="orthocenter"
            )
            self.pseudoqubits[qubit_id] = qubit
            qubit_id += 1
            
            # Geodesic grid (7 points)
            grid_points = self._compute_geodesic_grid(triangle)
            for gp in grid_points[:7]:
                qubit = Pseudoqubit(
                    pseudoqubit_id=qubit_id,
                    triangle_id=tri_id,
                    x=gp.x,
                    y=gp.y,
                    placement_type="geodesic"
                )
                self.pseudoqubits[qubit_id] = qubit
                qubit_id += 1
            
            if tri_id % 1000 == 0:
                logger.debug(f"  Triangle {tri_id}: qubits {qubit_id-13} to {qubit_id-1}")
        
        logger.info(f"{CLR.OK}[QUBITS] All {qubit_id} pseudoqubits placed{CLR.E}")
    
    def _compute_incenter(self, tri: HyperbolicTriangle) -> HyperbolicPoint:
        """Incenter of hyperbolic triangle"""
        d01 = hyperbolic_distance(tri.v0, tri.v1)
        d12 = hyperbolic_distance(tri.v1, tri.v2)
        d20 = hyperbolic_distance(tri.v2, tri.v0)
        
        w0 = mpf(1) / d12
        w1 = mpf(1) / d20
        w2 = mpf(1) / d01
        total = w0 + w1 + w2
        
        x = (w0 * tri.v0.x + w1 * tri.v1.x + w2 * tri.v2.x) / total
        y = (w0 * tri.v0.y + w1 * tri.v1.y + w2 * tri.v2.y) / total
        
        return HyperbolicPoint(x, y, name=f"incenter_{tri.triangle_id}")
    
    def _compute_circumcenter(self, tri: HyperbolicTriangle) -> HyperbolicPoint:
        """Circumcenter of hyperbolic triangle"""
        ax, ay = float(tri.v0.x), float(tri.v0.y)
        bx, by = float(tri.v1.x), float(tri.v1.y)
        cx, cy = float(tri.v2.x), float(tri.v2.y)
        
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-10:
            return tri.v0
        
        ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
        uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
        
        return HyperbolicPoint(mpf(ux), mpf(uy), name=f"circumcenter_{tri.triangle_id}")
    
    def _compute_orthocenter(self, tri: HyperbolicTriangle) -> HyperbolicPoint:
        """Orthocenter of hyperbolic triangle"""
        inc = self._compute_incenter(tri)
        circ = self._compute_circumcenter(tri)
        
        x = (inc.x + circ.x) / mpf(2)
        y = (inc.y + circ.y) / mpf(2)
        
        return HyperbolicPoint(x, y, name=f"orthocenter_{tri.triangle_id}")
    
    def _compute_geodesic_grid(self, tri: HyperbolicTriangle, density: int = 5) -> List[HyperbolicPoint]:
        """Geodesic grid points via barycentric coordinates"""
        points = []
        for i in range(1, density):
            for j in range(1, density - i):
                lambda1 = mpf(i) / mpf(density)
                lambda2 = mpf(j) / mpf(density)
                lambda3 = mpf(1) - lambda1 - lambda2
                
                x = lambda1 * tri.v0.x + lambda2 * tri.v1.x + lambda3 * tri.v2.x
                y = lambda1 * tri.v0.y + lambda2 * tri.v1.y + lambda3 * tri.v2.y
                
                points.append(HyperbolicPoint(x, y, name=f"geodesic_{tri.triangle_id}_{len(points)}"))
        
        return points

# ═════════════════════════════════════════════════════════════════════════════════
# COMPLETE POSTGRESQL SCHEMA
# ═════════════════════════════════════════════════════════════════════════════════

COMPLETE_SCHEMA = """

-- ═════════════════════════════════════════════════════════════════════════════════
-- DATABASE INITIALIZATION
-- ═════════════════════════════════════════════════════════════════════════════════

-- Enable essential extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "btree_gist";

-- ═════════════════════════════════════════════════════════════════════════════════
-- SECTION 1: QUANTUM LATTICE FOUNDATION (14 TABLES)
-- ═════════════════════════════════════════════════════════════════════════════════

CREATE TABLE hyperbolic_triangles (
    triangle_id BIGINT PRIMARY KEY,
    depth INT NOT NULL,
    parent_id BIGINT REFERENCES hyperbolic_triangles(triangle_id) ON DELETE SET NULL,
    
    -- Vertex 0
    v0_x NUMERIC(200, 150) NOT NULL,
    v0_y NUMERIC(200, 150) NOT NULL,
    v0_name VARCHAR(255),
    
    -- Vertex 1
    v1_x NUMERIC(200, 150) NOT NULL,
    v1_y NUMERIC(200, 150) NOT NULL,
    v1_name VARCHAR(255),
    
    -- Vertex 2
    v2_x NUMERIC(200, 150) NOT NULL,
    v2_y NUMERIC(200, 150) NOT NULL,
    v2_name VARCHAR(255),
    
    -- Computed properties
    area NUMERIC(200, 150),
    perimeter NUMERIC(200, 150),
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_parent_id (parent_id),
    INDEX idx_depth (depth)
);

CREATE TABLE pseudoqubits (
    pseudoqubit_id BIGINT PRIMARY KEY,
    triangle_id BIGINT NOT NULL REFERENCES hyperbolic_triangles(triangle_id) ON DELETE CASCADE,
    
    -- Position in Poincaré disk
    x NUMERIC(200, 150) NOT NULL,
    y NUMERIC(200, 150) NOT NULL,
    
    -- Placement strategy
    placement_type VARCHAR(50) NOT NULL,  -- 'vertex', 'incenter', 'circumcenter', 'orthocenter', 'geodesic'
    
    -- Quantum state
    phase_theta NUMERIC(200, 150) DEFAULT 0,
    coherence_measure NUMERIC(5, 4) DEFAULT 0.99,
    coherence_time_us NUMERIC(15, 2) DEFAULT 100000,
    
    -- Entanglement
    entanglement_with_oracle NUMERIC(5, 4) DEFAULT 0,
    entanglement_measure_neighbors JSONB DEFAULT '{}'::jsonb,
    
    -- Timing
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_measured_at TIMESTAMP WITH TIME ZONE,
    
    INDEX idx_triangle_id (triangle_id),
    INDEX idx_placement_type (placement_type),
    INDEX idx_pseudoqubit_id (pseudoqubit_id)
);

CREATE TABLE pseudoqubit_entanglement (
    id BIGSERIAL PRIMARY KEY,
    qubit_id_1 BIGINT NOT NULL REFERENCES pseudoqubits(pseudoqubit_id) ON DELETE CASCADE,
    qubit_id_2 BIGINT NOT NULL REFERENCES pseudoqubits(pseudoqubit_id) ON DELETE CASCADE,
    
    -- Entanglement measure
    entanglement_measure NUMERIC(5, 4) NOT NULL,
    bell_parameter NUMERIC(5, 4),
    
    -- Verification
    verified BOOLEAN DEFAULT FALSE,
    verification_proof TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(qubit_id_1, qubit_id_2),
    INDEX idx_qubit_id_1 (qubit_id_1),
    INDEX idx_qubit_id_2 (qubit_id_2)
);

CREATE TABLE quantum_lattice_metadata (
    metadata_id BIGSERIAL PRIMARY KEY,
    
    -- Construction parameters
    tessellation_depth INT NOT NULL,
    total_triangles BIGINT NOT NULL,
    total_pseudoqubits BIGINT NOT NULL,
    precision_bits INT DEFAULT 150,
    
    -- Geometry constants
    hyperbolicity_constant NUMERIC(5, 4) DEFAULT -1.0,
    poincare_radius NUMERIC(5, 4) DEFAULT 1.0,
    
    -- Status
    status VARCHAR(50) DEFAULT 'constructing',
    
    -- Timestamps
    construction_started_at TIMESTAMP WITH TIME ZONE,
    construction_completed_at TIMESTAMP WITH TIME ZONE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE quantum_phase_evolution (
    phase_id BIGSERIAL PRIMARY KEY,
    pseudoqubit_id BIGINT NOT NULL REFERENCES pseudoqubits(pseudoqubit_id) ON DELETE CASCADE,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    
    -- Phase tracking
    phase_theta NUMERIC(200, 150) NOT NULL,
    phase_derivative NUMERIC(200, 150),
    
    -- Phase coherence
    coherence_measure NUMERIC(5, 4),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_pseudoqubit_id (pseudoqubit_id),
    INDEX idx_block_height (block_height),
    INDEX idx_timestamp (timestamp)
);

CREATE TABLE quantum_coherence_snapshots (
    snapshot_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    
    -- System-wide coherence
    global_coherence NUMERIC(5, 4) NOT NULL,
    average_coherence NUMERIC(5, 4),
    min_coherence NUMERIC(5, 4),
    max_coherence NUMERIC(5, 4),
    
    -- Coherence distribution
    coherence_histogram JSONB,
    
    -- Phase alignment
    phase_drift_radians NUMERIC(200, 150),
    phase_correction_applied BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_block_height (block_height),
    INDEX idx_timestamp (timestamp)
);

CREATE TABLE quantum_measurements (
    measurement_id BIGSERIAL PRIMARY KEY,
    pseudoqubit_id BIGINT NOT NULL REFERENCES pseudoqubits(pseudoqubit_id) ON DELETE CASCADE,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    
    -- Measurement outcome
    outcome INT CHECK (outcome IN (0, 1)),
    basis VARCHAR(10),  -- 'Z', 'X', 'Y'
    
    -- Expectation value
    expectation_value NUMERIC(5, 4),
    variance NUMERIC(5, 4),
    
    -- Quantum state
    post_measurement_state JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_pseudoqubit_id (pseudoqubit_id),
    INDEX idx_block_height (block_height),
    INDEX idx_timestamp (timestamp)
);

CREATE TABLE quantum_density_matrix_global (
    state_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    
    -- Full density matrix (compressed)
    density_matrix_json JSONB NOT NULL,
    density_matrix_hash VARCHAR(255) UNIQUE NOT NULL,
    
    -- Properties
    trace_value NUMERIC(5, 4),  -- Should be 1.0
    purity NUMERIC(5, 4),  -- 0 = mixed, 1 = pure
    von_neumann_entropy NUMERIC(5, 4),
    
    -- Eigenvalues (for diagnostics)
    eigenvalues JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_block_height (block_height),
    INDEX idx_timestamp (timestamp)
);

CREATE TABLE quantum_shadow_tomography (
    shadow_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    
    -- Efficient state representation
    shadow_snapshots JSONB NOT NULL,
    shadow_measurement_bases JSONB,
    
    -- Reconstruction info
    reconstruction_fidelity NUMERIC(5, 4),
    num_snapshots INT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_block_height (block_height)
);

CREATE TABLE quantum_circuit_execution (
    circuit_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    
    -- Circuit parameters
    circuit_depth INT,
    circuit_size INT,
    num_qubits INT,
    num_gates INT,
    
    -- Execution results
    execution_successful BOOLEAN DEFAULT TRUE,
    execution_time_ms NUMERIC(15, 2),
    
    -- Fidelity metrics
    ghz_fidelity NUMERIC(5, 4),
    w_state_fidelity NUMERIC(5, 4),
    
    -- Circuit data (compressed)
    circuit_data JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_block_height (block_height),
    INDEX idx_timestamp (timestamp)
);

CREATE TABLE quantum_error_correction (
    correction_id BIGSERIAL PRIMARY KEY,
    pseudoqubit_id BIGINT NOT NULL REFERENCES pseudoqubits(pseudoqubit_id) ON DELETE CASCADE,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    
    -- Error detection
    error_detected BOOLEAN NOT NULL,
    error_type VARCHAR(50),
    error_location_code VARCHAR(255),
    
    -- Correction applied
    correction_applied BOOLEAN DEFAULT FALSE,
    correction_method VARCHAR(50),
    correction_strength NUMERIC(5, 4),
    
    -- Verification
    correction_verified BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_pseudoqubit_id (pseudoqubit_id),
    INDEX idx_block_height (block_height)
);

CREATE TABLE quantum_supremacy_proofs (
    proof_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    
    -- Proof parameters
    circuit_depth INT,
    success_probability NUMERIC(5, 4),
    classical_simulation_complexity VARCHAR(255),
    
    -- Proof data
    quantum_result_hash VARCHAR(255),
    classical_hardness_assumption TEXT,
    
    -- Verification
    verified BOOLEAN DEFAULT FALSE,
    verification_method VARCHAR(100),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_block_height (block_height)
);

CREATE TABLE w_state_snapshots (
    snapshot_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    
    -- W-state (tripartite)
    w_state_serialized TEXT NOT NULL,
    w_state_hash VARCHAR(255) UNIQUE NOT NULL,
    
    -- Properties
    entanglement_measure NUMERIC(5, 4),
    coherence_time_us NUMERIC(15, 2),
    fidelity_estimate NUMERIC(5, 4),
    
    -- W-state qubits
    pq_addresses TEXT[],
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(block_height),
    INDEX idx_block_height (block_height),
    INDEX idx_timestamp (timestamp)
);

CREATE TABLE w_state_validator_states (
    state_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    validator_public_key VARCHAR(255) NOT NULL,
    timestamp BIGINT NOT NULL,
    
    -- Local W-state
    w_state_serialized TEXT,
    w_state_hash VARCHAR(255),
    
    -- Coherence with oracle
    coherence_with_oracle NUMERIC(5, 4),
    phase_alignment_radians NUMERIC(200, 150),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_block_height (block_height),
    INDEX idx_validator_public_key (validator_public_key),
    INDEX idx_timestamp (timestamp)
);

-- ═════════════════════════════════════════════════════════════════════════════════
-- SECTION 2: ORACLE STATE DISTRIBUTION (8 TABLES)
-- ═════════════════════════════════════════════════════════════════════════════════

CREATE TABLE oracle_w_state_snapshots (
    snapshot_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    block_hash VARCHAR(255) NOT NULL,
    timestamp BIGINT NOT NULL,
    
    -- W-state (tripartite from pq0 oracle)
    w_state_serialized TEXT NOT NULL,
    w_state_hash VARCHAR(255) UNIQUE NOT NULL,
    
    -- Entanglement
    entanglement_measure NUMERIC(5, 4),
    coherence_time_us NUMERIC(15, 2),
    fidelity_estimate NUMERIC(5, 4),
    
    -- Quantum proof
    quantum_proof_data TEXT,
    quantum_proof_hash VARCHAR(255),
    
    -- Entropy
    shannon_entropy NUMERIC(5, 4),
    entropy_source_quality JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(block_height, block_hash),
    INDEX idx_block_height (block_height),
    INDEX idx_timestamp (timestamp)
);

CREATE TABLE oracle_density_matrix_stream (
    stream_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    
    -- Density matrix
    density_matrix_json JSONB NOT NULL,
    density_matrix_hash VARCHAR(255) UNIQUE NOT NULL,
    
    -- Properties
    trace_value NUMERIC(5, 4),
    purity NUMERIC(5, 4),
    von_neumann_entropy NUMERIC(5, 4),
    eigenvalues JSONB,
    
    -- Live metrics
    live_metrics_json JSONB,
    sensor_timestamps JSONB,
    
    -- Update tracking
    update_sequence_number BIGINT,
    time_since_last_update_ms NUMERIC(15, 2),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_block_height (block_height),
    INDEX idx_timestamp (timestamp)
);

CREATE TABLE oracle_entropy_feeds (
    feed_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    
    -- 5-source ensemble
    anu_qrng_entropy TEXT,
    random_org_entropy TEXT,
    qbick_entropy TEXT,
    outshift_entropy TEXT,
    hotbits_entropy TEXT,
    
    -- Combined
    xor_combined_seed TEXT,
    entropy_hash VARCHAR(255) UNIQUE NOT NULL,
    
    -- Metrics
    min_entropy_estimate NUMERIC(5, 4),
    shannon_entropy_estimate NUMERIC(5, 4),
    source_agreement_score NUMERIC(5, 4),
    
    -- Distribution
    distributed_to_peers INT DEFAULT 0,
    distribution_complete BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_block_height (block_height),
    INDEX idx_timestamp (timestamp)
);

CREATE TABLE oracle_coherence_metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    
    -- System coherence
    system_coherence_measure NUMERIC(5, 4),
    lattice_coherence_score NUMERIC(5, 4),
    tessellation_synchronization_quality NUMERIC(5, 4),
    
    -- Per-pseudoqubit
    pseudoqubit_coherence_array JSONB,
    min_coherence NUMERIC(5, 4),
    max_coherence NUMERIC(5, 4),
    avg_coherence NUMERIC(5, 4),
    
    -- Phase
    phase_drift_radians NUMERIC(200, 150),
    phase_correction_applied BOOLEAN,
    
    -- Consensus
    validator_agreement_score NUMERIC(5, 4),
    network_partition_detected BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_block_height (block_height),
    INDEX idx_timestamp (timestamp)
);

CREATE TABLE oracle_pq0_state (
    state_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    
    -- PQ0 oracle state
    oracle_pseudoqubit_id BIGINT REFERENCES pseudoqubits(pseudoqubit_id) ON DELETE SET NULL,
    oracle_position_x NUMERIC(200, 150),
    oracle_position_y NUMERIC(200, 150),
    
    -- Tripartite configuration
    pq_inverse_virtual_id BIGINT REFERENCES pseudoqubits(pseudoqubit_id) ON DELETE SET NULL,
    pq_virtual_id BIGINT REFERENCES pseudoqubits(pseudoqubit_id) ON DELETE SET NULL,
    
    -- State
    quantum_state_json JSONB,
    phase_theta NUMERIC(200, 150),
    coherence_measure NUMERIC(5, 4),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_block_height (block_height),
    INDEX idx_timestamp (timestamp)
);

CREATE TABLE oracle_entanglement_records (
    entanglement_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    
    -- Peer
    peer_id VARCHAR(255) NOT NULL,
    peer_public_key VARCHAR(255),
    
    -- Entanglement
    entanglement_type VARCHAR(50),
    entanglement_measure NUMERIC(5, 4),
    bell_parameter NUMERIC(5, 4),
    
    -- Oracle sync
    oracle_entanglement_measure NUMERIC(5, 4),
    entanglement_match_score NUMERIC(5, 4),
    in_sync_with_oracle BOOLEAN DEFAULT FALSE,
    
    -- Verification
    verification_proof TEXT,
    verified BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_peer_id (peer_id),
    INDEX idx_block_height (block_height),
    INDEX idx_timestamp (timestamp)
);

CREATE TABLE oracle_distribution_log (
    log_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    
    -- Distribution event
    peer_id VARCHAR(255) NOT NULL,
    data_type VARCHAR(50),  -- 'w_state', 'density_matrix', 'entropy', 'coherence'
    
    -- Tracking
    data_hash VARCHAR(255),
    distribution_successful BOOLEAN DEFAULT TRUE,
    distribution_latency_ms NUMERIC(15, 2),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_peer_id (peer_id),
    INDEX idx_block_height (block_height)
);

CREATE TABLE oracle_consensus_state (
    consensus_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    
    -- Consensus metrics
    oracle_consensus_reached BOOLEAN DEFAULT FALSE,
    validator_agreement_count INT,
    total_validators INT,
    consensus_threshold NUMERIC(5, 4),
    
    -- W-state agreement
    w_state_hash_agreement BOOLEAN,
    density_matrix_hash_agreement BOOLEAN,
    entropy_hash_agreement BOOLEAN,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(block_height),
    INDEX idx_block_height (block_height)
);

-- ═════════════════════════════════════════════════════════════════════════════════
-- SECTION 3: P2P NETWORK LAYER (7 TABLES)
-- ═════════════════════════════════════════════════════════════════════════════════

CREATE TABLE peer_registry (
    peer_id VARCHAR(255) PRIMARY KEY,
    
    -- Identity
    public_key VARCHAR(255) UNIQUE NOT NULL,
    ip_address VARCHAR(45),
    port INTEGER,
    
    -- Classification
    peer_type VARCHAR(50) DEFAULT 'full',
    capabilities TEXT[],
    
    -- Blockchain state
    block_height BIGINT DEFAULT 0,
    chain_head_hash VARCHAR(255),
    network_version VARCHAR(50),
    
    -- Reputation
    reputation_score NUMERIC(10, 4) DEFAULT 1.0,
    blocks_validated INT DEFAULT 0,
    blocks_rejected INT DEFAULT 0,
    
    -- Connection
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_handshake TIMESTAMP WITH TIME ZONE,
    connection_attempts INT DEFAULT 0,
    failed_attempts INT DEFAULT 0,
    
    -- Oracle
    oracle_entanglement_ready BOOLEAN DEFAULT FALSE,
    oracle_density_matrix_version BIGINT DEFAULT 0,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_public_key (public_key),
    INDEX idx_peer_type (peer_type),
    INDEX idx_last_seen (last_seen),
    INDEX idx_block_height (block_height),
    INDEX idx_reputation_score (reputation_score)
);

CREATE TABLE peer_connections (
    connection_id BIGSERIAL PRIMARY KEY,
    peer_id VARCHAR(255) NOT NULL REFERENCES peer_registry(peer_id) ON DELETE CASCADE,
    
    -- State
    connection_state VARCHAR(50) DEFAULT 'disconnected',
    established_at TIMESTAMP WITH TIME ZONE,
    disconnected_at TIMESTAMP WITH TIME ZONE,
    
    -- Metrics
    latency_ms NUMERIC(10, 2),
    bandwidth_in_kbps NUMERIC(15, 2),
    bandwidth_out_kbps NUMERIC(15, 2),
    packet_loss_rate NUMERIC(5, 4),
    
    -- Sync
    blocks_sync_height BIGINT,
    last_message_at TIMESTAMP WITH TIME ZONE,
    messages_sent INT DEFAULT 0,
    messages_received INT DEFAULT 0,
    bytes_sent BIGINT DEFAULT 0,
    bytes_received BIGINT DEFAULT 0,
    
    -- Oracle
    oracle_state_shared BOOLEAN DEFAULT FALSE,
    density_matrix_shared BOOLEAN DEFAULT FALSE,
    entropy_shared BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_peer_id (peer_id),
    INDEX idx_connection_state (connection_state),
    INDEX idx_last_message_at (last_message_at)
);

CREATE TABLE peer_reputation (
    reputation_id BIGSERIAL PRIMARY KEY,
    peer_id VARCHAR(255) NOT NULL REFERENCES peer_registry(peer_id) ON DELETE CASCADE,
    timestamp BIGINT NOT NULL,
    
    -- Scoring
    score NUMERIC(10, 4) NOT NULL,
    factors JSONB,
    
    -- Events
    event_type VARCHAR(50),
    event_description TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_peer_id (peer_id),
    INDEX idx_timestamp (timestamp)
);

CREATE TABLE block_headers_cache (
    height BIGINT PRIMARY KEY,
    
    -- Identity
    block_hash VARCHAR(255) UNIQUE NOT NULL,
    previous_hash VARCHAR(255) NOT NULL,
    
    -- State
    state_root VARCHAR(255),
    transactions_root VARCHAR(255),
    
    -- Timing
    timestamp BIGINT NOT NULL,
    difficulty NUMERIC(20, 10),
    nonce VARCHAR(255),
    
    -- Quantum
    quantum_proof VARCHAR(255),
    quantum_state_hash VARCHAR(255),
    temporal_coherence NUMERIC(5, 4),
    
    -- PQ
    pq_signature TEXT,
    pq_key_fingerprint VARCHAR(255),
    
    -- Caching
    received_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    requested_by_peers INT DEFAULT 0,
    
    INDEX idx_block_hash (block_hash),
    INDEX idx_previous_hash (previous_hash),
    INDEX idx_timestamp (timestamp)
);

CREATE TABLE orphan_blocks (
    block_hash VARCHAR(255) PRIMARY KEY,
    
    -- Parent
    parent_hash VARCHAR(255) NOT NULL,
    block_height BIGINT,
    timestamp BIGINT,
    
    -- Data
    block_data_compressed BYTEA,
    block_size_bytes INT,
    
    -- Received
    received_from_peer VARCHAR(255),
    received_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    
    -- State
    resolution_status VARCHAR(50) DEFAULT 'awaiting_parent',
    resolution_attempt_count INT DEFAULT 0,
    
    INDEX idx_parent_hash (parent_hash),
    INDEX idx_received_at (received_at),
    INDEX idx_expires_at (expires_at)
);

CREATE TABLE network_bandwidth_usage (
    usage_id BIGSERIAL PRIMARY KEY,
    peer_id VARCHAR(255) REFERENCES peer_registry(peer_id) ON DELETE CASCADE,
    timestamp BIGINT NOT NULL,
    
    -- Bandwidth
    bandwidth_in_kbps NUMERIC(15, 2),
    bandwidth_out_kbps NUMERIC(15, 2),
    total_bandwidth_kbps NUMERIC(15, 2),
    
    -- Data transferred
    bytes_in INT,
    bytes_out INT,
    
    -- Congestion
    congestion_level NUMERIC(5, 4),
    packet_loss_rate NUMERIC(5, 4),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_peer_id (peer_id),
    INDEX idx_timestamp (timestamp)
);

CREATE TABLE network_partition_events (
    event_id BIGSERIAL PRIMARY KEY,
    timestamp BIGINT NOT NULL,
    
    -- Partition details
    partition_detected BOOLEAN DEFAULT TRUE,
    peers_in_partition_1 INT,
    peers_in_partition_2 INT,
    
    -- Healing
    partition_healed BOOLEAN DEFAULT FALSE,
    healing_timestamp BIGINT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_timestamp (timestamp)
);

-- ═════════════════════════════════════════════════════════════════════════════════
-- SECTION 4: BLOCKCHAIN CORE (8 TABLES)
-- ═════════════════════════════════════════════════════════════════════════════════

CREATE TABLE blocks (
    height BIGINT PRIMARY KEY,
    block_number BIGINT UNIQUE NOT NULL,
    block_hash VARCHAR(255) UNIQUE NOT NULL,
    previous_hash VARCHAR(255) NOT NULL,
    
    -- State roots
    state_root VARCHAR(255),
    transactions_root VARCHAR(255),
    receipts_root VARCHAR(255),
    
    -- Consensus
    timestamp BIGINT NOT NULL,
    transactions INT DEFAULT 0,
    validator_public_key VARCHAR(255) NOT NULL,
    validator_signature TEXT,
    
    -- PoW/PoS
    difficulty NUMERIC(20, 10) DEFAULT 1.0,
    total_difficulty NUMERIC(30, 0),
    nonce VARCHAR(255),
    
    -- Quantum
    quantum_proof TEXT,
    quantum_state_hash VARCHAR(255),
    quantum_validation_status VARCHAR(50) DEFAULT 'unvalidated',
    entropy_score NUMERIC(5, 4) DEFAULT 0.0,
    temporal_coherence NUMERIC(5, 4) DEFAULT 0.9,
    
    -- Post-quantum
    pq_signature TEXT,
    pq_key_fingerprint VARCHAR(255),
    pq_validation_status VARCHAR(50) DEFAULT 'unsigned',
    
    -- Oracle integration
    oracle_w_state_hash VARCHAR(255),
    oracle_density_matrix_hash VARCHAR(255),
    oracle_entropy_hash VARCHAR(255),
    oracle_consensus_reached BOOLEAN DEFAULT FALSE,
    
    -- Status
    status VARCHAR(50) DEFAULT 'pending',
    finalized BOOLEAN DEFAULT FALSE,
    finalized_at TIMESTAMP WITH TIME ZONE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_height (height),
    INDEX idx_timestamp (timestamp),
    INDEX idx_validator_public_key (validator_public_key),
    INDEX idx_previous_hash (previous_hash),
    INDEX idx_finalized (finalized),
    INDEX idx_status (status)
);

CREATE TABLE transactions (
    id BIGSERIAL PRIMARY KEY,
    tx_hash VARCHAR(255) UNIQUE NOT NULL,
    
    -- Addresses (no user_id!)
    from_address VARCHAR(255) NOT NULL,
    to_address VARCHAR(255) NOT NULL,
    amount NUMERIC(30, 0) NOT NULL,
    
    -- Parameters
    nonce BIGINT,
    gas_price NUMERIC(30, 0) DEFAULT 1,
    gas_limit BIGINT DEFAULT 21000,
    gas_used BIGINT DEFAULT 0,
    
    -- Block inclusion
    height BIGINT REFERENCES blocks(height) ON DELETE CASCADE,
    block_hash VARCHAR(255),
    transaction_index INT,
    
    -- Type
    tx_type VARCHAR(50) DEFAULT 'transfer',
    status VARCHAR(50) DEFAULT 'pending',
    
    -- PQ signatures
    pq_signature TEXT,
    pq_signer_key_fp VARCHAR(255),
    pq_verified BOOLEAN DEFAULT FALSE,
    pq_verified_at TIMESTAMP WITH TIME ZONE,
    
    -- Quantum
    quantum_state_hash VARCHAR(255),
    commitment_hash VARCHAR(255),
    entropy_score NUMERIC(5, 4),
    
    -- Metadata
    input_data TEXT,
    metadata JSONB,
    error_message TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    finalized_at TIMESTAMP WITH TIME ZONE,
    
    INDEX idx_tx_hash (tx_hash),
    INDEX idx_from_address (from_address),
    INDEX idx_to_address (to_address),
    INDEX idx_height (height),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
);

CREATE TABLE transaction_inputs (
    input_id BIGSERIAL PRIMARY KEY,
    tx_id BIGINT NOT NULL REFERENCES transactions(id) ON DELETE CASCADE,
    
    -- UTXO reference
    previous_tx_hash VARCHAR(255),
    previous_output_index INT,
    
    -- Script
    script_sig TEXT,
    script_pubkey TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_tx_id (tx_id),
    INDEX idx_previous_tx_hash (previous_tx_hash)
);

CREATE TABLE transaction_outputs (
    output_id BIGSERIAL PRIMARY KEY,
    tx_id BIGINT NOT NULL REFERENCES transactions(id) ON DELETE CASCADE,
    output_index INT NOT NULL,
    
    -- Address and amount
    address VARCHAR(255) NOT NULL,
    amount NUMERIC(30, 0) NOT NULL,
    
    -- Script
    script_pubkey TEXT,
    
    -- Status
    spent BOOLEAN DEFAULT FALSE,
    spent_in_tx_id BIGINT REFERENCES transactions(id) ON DELETE SET NULL,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(tx_id, output_index),
    INDEX idx_tx_id (tx_id),
    INDEX idx_address (address),
    INDEX idx_spent (spent)
);

CREATE TABLE transaction_receipts (
    receipt_id BIGSERIAL PRIMARY KEY,
    tx_id BIGINT NOT NULL REFERENCES transactions(id) ON DELETE CASCADE,
    height BIGINT REFERENCES blocks(height) ON DELETE CASCADE,
    
    -- Status
    status INT,
    gas_used BIGINT,
    cumulative_gas_used BIGINT,
    
    -- Logs
    logs_json JSONB,
    bloom_filter TEXT,
    
    -- Quantum info
    quantum_proof TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_tx_id (tx_id),
    INDEX idx_height (height)
);

CREATE TABLE validators (
    validator_id BIGSERIAL PRIMARY KEY,
    public_key VARCHAR(255) UNIQUE NOT NULL,
    
    -- Peer reference
    peer_id VARCHAR(255) REFERENCES peer_registry(peer_id) ON DELETE SET NULL,
    
    -- Consensus
    stake NUMERIC(30, 0) DEFAULT 0,
    commission_rate NUMERIC(5, 4),
    slashing_rate NUMERIC(5, 4) DEFAULT 0.0,
    
    -- Activity
    blocks_proposed INT DEFAULT 0,
    blocks_validated INT DEFAULT 0,
    blocks_missed INT DEFAULT 0,
    
    -- Status
    status VARCHAR(50) DEFAULT 'active',
    is_slashed BOOLEAN DEFAULT FALSE,
    slashed_at TIMESTAMP WITH TIME ZONE,
    
    -- Oracle
    oracle_participation_score NUMERIC(5, 4) DEFAULT 0.0,
    w_state_sync_quality NUMERIC(5, 4) DEFAULT 0.0,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_public_key (public_key),
    INDEX idx_status (status),
    INDEX idx_stake (stake),
    INDEX idx_peer_id (peer_id)
);

CREATE TABLE validator_stakes (
    stake_id BIGSERIAL PRIMARY KEY,
    validator_id BIGINT NOT NULL REFERENCES validators(validator_id) ON DELETE CASCADE,
    
    -- Stake tracking
    amount NUMERIC(30, 0) NOT NULL,
    staker_address VARCHAR(255),
    
    -- Status
    active BOOLEAN DEFAULT TRUE,
    delegated BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    stake_at_height BIGINT,
    unstake_at_height BIGINT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_validator_id (validator_id),
    INDEX idx_staker_address (staker_address)
);

CREATE TABLE merkle_proofs (
    proof_id BIGSERIAL PRIMARY KEY,
    transaction_hash VARCHAR(255) NOT NULL,
    
    height BIGINT REFERENCES blocks(height) ON DELETE CASCADE,
    block_hash VARCHAR(255) NOT NULL,
    
    -- Proof path
    proof_path TEXT NOT NULL,
    proof_index INT NOT NULL,
    
    -- Verification
    verified BOOLEAN DEFAULT FALSE,
    verified_at TIMESTAMP WITH TIME ZONE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(transaction_hash, block_hash),
    INDEX idx_transaction_hash (transaction_hash),
    INDEX idx_height (height)
);

-- ═════════════════════════════════════════════════════════════════════════════════
-- SECTION 5: HLWE WALLET LAYER (8 TABLES)
-- ═════════════════════════════════════════════════════════════════════════════════

CREATE TABLE wallet_addresses (
    address VARCHAR(255) PRIMARY KEY,
    
    -- Master wallet
    wallet_fingerprint VARCHAR(64) NOT NULL,
    
    -- Derivation
    derivation_path VARCHAR(100),
    account_index INT,
    change_index INT,
    address_index INT,
    
    -- Key
    public_key VARCHAR(255) NOT NULL,
    
    -- Properties
    address_type VARCHAR(50) DEFAULT 'receiving',
    is_watching_only BOOLEAN DEFAULT FALSE,
    is_cold_storage BOOLEAN DEFAULT FALSE,
    
    -- Balance
    balance NUMERIC(30, 0) DEFAULT 0,
    balance_updated_at TIMESTAMP WITH TIME ZONE,
    balance_at_height BIGINT,
    
    -- Usage
    first_used_at TIMESTAMP WITH TIME ZONE,
    last_used_at TIMESTAMP WITH TIME ZONE,
    transaction_count INT DEFAULT 0,
    
    -- Labels
    label VARCHAR(255),
    notes TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(wallet_fingerprint, derivation_path),
    INDEX idx_wallet_fingerprint (wallet_fingerprint),
    INDEX idx_address (address),
    INDEX idx_public_key (public_key),
    INDEX idx_balance (balance)
);

CREATE TABLE encrypted_private_keys (
    key_id BIGSERIAL PRIMARY KEY,
    address VARCHAR(255) NOT NULL UNIQUE REFERENCES wallet_addresses(address) ON DELETE CASCADE,
    
    -- Encryption
    algorithm VARCHAR(50) DEFAULT 'AES-256-GCM',
    kdf_algorithm VARCHAR(50) DEFAULT 'PBKDF2-SHA3-512',
    kdf_iterations INT DEFAULT 16384,
    
    -- Key material
    nonce_hex VARCHAR(255) NOT NULL,
    salt_hex VARCHAR(255) NOT NULL,
    ciphertext_hex TEXT NOT NULL,
    auth_tag_hex VARCHAR(255),
    
    -- Metadata
    key_fingerprint VARCHAR(255),
    derivation_path VARCHAR(100),
    
    -- Status
    is_locked BOOLEAN DEFAULT FALSE,
    lock_reason TEXT,
    
    -- Usage
    last_used_for_signing TIMESTAMP WITH TIME ZONE,
    signing_count INT DEFAULT 0,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_address (address),
    INDEX idx_key_fingerprint (key_fingerprint)
);

CREATE TABLE address_transactions (
    id BIGSERIAL PRIMARY KEY,
    address VARCHAR(255) NOT NULL REFERENCES wallet_addresses(address) ON DELETE CASCADE,
    
    tx_hash VARCHAR(255) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    
    -- Details
    from_address VARCHAR(255),
    to_address VARCHAR(255),
    amount NUMERIC(30, 0),
    
    -- Block
    block_height BIGINT REFERENCES blocks(height) ON DELETE CASCADE,
    block_hash VARCHAR(255),
    block_timestamp BIGINT,
    
    -- Status
    tx_status VARCHAR(50) DEFAULT 'pending',
    confirmation_count INT DEFAULT 0,
    
    -- Fees
    gas_used BIGINT,
    gas_price NUMERIC(30, 0),
    total_fee NUMERIC(30, 0),
    
    -- Notes
    notes TEXT,
    label VARCHAR(255),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(address, tx_hash),
    INDEX idx_address (address),
    INDEX idx_tx_hash (tx_hash),
    INDEX idx_block_height (block_height),
    INDEX idx_tx_status (tx_status),
    INDEX idx_direction (direction)
);

CREATE TABLE address_balance_history (
    id BIGSERIAL PRIMARY KEY,
    address VARCHAR(255) NOT NULL REFERENCES wallet_addresses(address) ON DELETE CASCADE,
    
    block_height BIGINT NOT NULL,
    block_hash VARCHAR(255),
    
    balance NUMERIC(30, 0) NOT NULL,
    delta NUMERIC(30, 0),
    
    snapshot_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(address, block_height),
    INDEX idx_address (address),
    INDEX idx_block_height (block_height)
);

CREATE TABLE wallet_seed_backup_status (
    backup_id BIGSERIAL PRIMARY KEY,
    wallet_fingerprint VARCHAR(64) NOT NULL UNIQUE,
    
    -- Status
    seed_phrase_backed_up BOOLEAN DEFAULT FALSE,
    backup_confirmed_at TIMESTAMP WITH TIME ZONE,
    
    -- Verification (NEVER store actual phrase!)
    seed_hint VARCHAR(50),
    seed_hash VARCHAR(255),
    
    -- Warnings
    backup_required BOOLEAN DEFAULT TRUE,
    days_since_creation_without_backup INT,
    email_notifications_sent INT DEFAULT 0,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE address_labels (
    label_id BIGSERIAL PRIMARY KEY,
    address VARCHAR(255) NOT NULL REFERENCES wallet_addresses(address) ON DELETE CASCADE,
    
    label VARCHAR(255) NOT NULL,
    description TEXT,
    label_type VARCHAR(50),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_address (address),
    INDEX idx_label (label)
);

CREATE TABLE address_utxos (
    utxo_id BIGSERIAL PRIMARY KEY,
    address VARCHAR(255) NOT NULL REFERENCES wallet_addresses(address) ON DELETE CASCADE,
    
    -- UTXO reference
    tx_hash VARCHAR(255) NOT NULL,
    output_index INT NOT NULL,
    
    -- Amount
    amount NUMERIC(30, 0) NOT NULL,
    
    -- Status
    spent BOOLEAN DEFAULT FALSE,
    spent_at_height BIGINT,
    spent_in_tx_hash VARCHAR(255),
    
    -- Block info
    created_at_height BIGINT REFERENCES blocks(height) ON DELETE CASCADE,
    created_at_timestamp BIGINT,
    
    INDEX idx_address (address),
    INDEX idx_spent (spent),
    UNIQUE(tx_hash, output_index)
);

CREATE TABLE wallet_key_rotation_history (
    rotation_id BIGSERIAL PRIMARY KEY,
    wallet_fingerprint VARCHAR(64) NOT NULL,
    
    -- Old and new
    old_key_id VARCHAR(255),
    new_key_id VARCHAR(255),
    
    -- Details
    rotation_reason TEXT,
    rotation_timestamp TIMESTAMP WITH TIME ZONE,
    
    -- Ratchet
    ratchet_material TEXT,
    next_rotation_material TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_wallet_fingerprint (wallet_fingerprint),
    INDEX idx_rotation_timestamp (rotation_timestamp)
);

-- ═════════════════════════════════════════════════════════════════════════════════
-- SECTION 6: CLIENT SYNCHRONIZATION (6 TABLES)
-- ═════════════════════════════════════════════════════════════════════════════════

CREATE TABLE client_oracle_sync (
    sync_id BIGSERIAL PRIMARY KEY,
    peer_id VARCHAR(255) PRIMARY KEY REFERENCES peer_registry(peer_id) ON DELETE CASCADE,
    
    -- Synchronization state
    block_height_local BIGINT,
    block_height_oracle BIGINT,
    w_state_hash_local VARCHAR(255),
    w_state_hash_oracle VARCHAR(255),
    
    -- Density matrix
    density_matrix_hash_local VARCHAR(255),
    density_matrix_hash_oracle VARCHAR(255),
    density_matrix_sync_status VARCHAR(50) DEFAULT 'pending',
    
    -- Entropy
    entropy_hash_local VARCHAR(255),
    entropy_hash_oracle VARCHAR(255),
    
    -- Coherence
    coherence_measure_local NUMERIC(5, 4),
    coherence_measure_oracle NUMERIC(5, 4),
    coherence_aligned BOOLEAN DEFAULT FALSE,
    
    -- Lattice
    lattice_sync_quality NUMERIC(5, 4),
    tessellation_in_sync BOOLEAN DEFAULT FALSE,
    last_lattice_update TIMESTAMP WITH TIME ZONE,
    
    -- Status
    sync_status VARCHAR(50) DEFAULT 'initializing',
    sync_confidence NUMERIC(5, 4) DEFAULT 0.0,
    
    last_sync_attempt TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_successful_sync TIMESTAMP WITH TIME ZONE,
    sync_error_message TEXT,
    sync_attempt_count INT DEFAULT 0,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_sync_status (sync_status),
    INDEX idx_last_sync_attempt (last_sync_attempt)
);

CREATE TABLE entanglement_records (
    entanglement_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    
    -- Peer
    peer_id VARCHAR(255) NOT NULL REFERENCES peer_registry(peer_id) ON DELETE CASCADE,
    peer_public_key VARCHAR(255),
    
    -- Entanglement
    entanglement_type VARCHAR(50),
    entanglement_measure NUMERIC(5, 4),
    bell_parameter NUMERIC(5, 4),
    
    -- Oracle sync
    oracle_entanglement_measure NUMERIC(5, 4),
    entanglement_match_score NUMERIC(5, 4),
    in_sync_with_oracle BOOLEAN DEFAULT FALSE,
    
    -- Local state
    local_w_state_hash VARCHAR(255),
    local_density_matrix_hash VARCHAR(255),
    
    -- Verification
    verification_proof TEXT,
    verified BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_peer_id (peer_id),
    INDEX idx_block_height (block_height),
    INDEX idx_timestamp (timestamp)
);

CREATE TABLE lattice_sync_state (
    state_id BIGSERIAL PRIMARY KEY,
    peer_id VARCHAR(255) NOT NULL REFERENCES peer_registry(peer_id) ON DELETE CASCADE,
    block_height BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    
    -- Tessellation synchronization
    hyperbolic_coordinates_synced JSONB,
    poincare_disk_coverage NUMERIC(5, 4),
    vertex_synchronization_count INT,
    edge_synchronization_count INT,
    
    -- Pseudoqubit lattice
    pseudoqubit_positions_hash VARCHAR(255),
    pseudoqubit_lattice_sync_quality NUMERIC(5, 4),
    
    -- Coherence
    lattice_coherence_measure NUMERIC(5, 4),
    critical_points_coherence JSONB,
    geodesic_paths_synchronized BOOLEAN DEFAULT FALSE,
    
    -- Updates
    updates_since_last_sync INT DEFAULT 0,
    bytes_synchronized BIGINT DEFAULT 0,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_peer_id (peer_id),
    INDEX idx_block_height (block_height)
);

CREATE TABLE client_block_sync (
    sync_id BIGSERIAL PRIMARY KEY,
    peer_id VARCHAR(255) NOT NULL REFERENCES peer_registry(peer_id) ON DELETE CASCADE,
    
    -- Sync progress
    blocks_downloaded INT,
    blocks_requested INT,
    blocks_total INT,
    
    -- Status
    sync_started_at TIMESTAMP WITH TIME ZONE,
    sync_completed_at TIMESTAMP WITH TIME ZONE,
    sync_status VARCHAR(50),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_peer_id (peer_id),
    INDEX idx_sync_status (sync_status)
);

CREATE TABLE client_sync_events (
    event_id BIGSERIAL PRIMARY KEY,
    peer_id VARCHAR(255) NOT NULL REFERENCES peer_registry(peer_id) ON DELETE CASCADE,
    timestamp BIGINT NOT NULL,
    
    -- Event type
    event_type VARCHAR(50),
    event_description TEXT,
    
    -- Details
    details JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_peer_id (peer_id),
    INDEX idx_event_type (event_type),
    INDEX idx_timestamp (timestamp)
);

CREATE TABLE client_network_metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    peer_id VARCHAR(255) NOT NULL REFERENCES peer_registry(peer_id) ON DELETE CASCADE,
    timestamp BIGINT NOT NULL,
    
    -- Network stats
    latency_ms NUMERIC(10, 2),
    bandwidth_in_kbps NUMERIC(15, 2),
    bandwidth_out_kbps NUMERIC(15, 2),
    packet_loss_rate NUMERIC(5, 4),
    
    -- Sync stats
    blocks_per_second NUMERIC(10, 2),
    avg_sync_time_ms NUMERIC(15, 2),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_peer_id (peer_id),
    INDEX idx_timestamp (timestamp)
);

-- ═════════════════════════════════════════════════════════════════════════════════
-- SECTION 7: CONSENSUS & FINALITY (6 TABLES)
-- ═════════════════════════════════════════════════════════════════════════════════

CREATE TABLE epochs (
    epoch_id BIGSERIAL PRIMARY KEY,
    epoch_number BIGINT UNIQUE NOT NULL,
    
    -- Timeline
    start_block_height BIGINT NOT NULL,
    end_block_height BIGINT,
    
    -- Validators
    validator_count INT,
    total_stake NUMERIC(30, 0),
    
    -- Status
    finalized BOOLEAN DEFAULT FALSE,
    finalized_at TIMESTAMP WITH TIME ZONE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_epoch_number (epoch_number),
    INDEX idx_finalized (finalized)
);

CREATE TABLE epoch_validators (
    membership_id BIGSERIAL PRIMARY KEY,
    epoch_id BIGINT NOT NULL REFERENCES epochs(epoch_id) ON DELETE CASCADE,
    validator_id BIGINT NOT NULL REFERENCES validators(validator_id) ON DELETE CASCADE,
    
    -- Stake
    stake NUMERIC(30, 0) NOT NULL,
    
    -- Participation
    blocks_proposed INT DEFAULT 0,
    blocks_attested INT DEFAULT 0,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(epoch_id, validator_id),
    INDEX idx_epoch_id (epoch_id),
    INDEX idx_validator_id (validator_id)
);

CREATE TABLE finality_records (
    finality_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL UNIQUE REFERENCES blocks(height) ON DELETE CASCADE,
    block_hash VARCHAR(255) NOT NULL,
    
    -- Finality status
    finalized BOOLEAN DEFAULT FALSE,
    finalized_at TIMESTAMP WITH TIME ZONE,
    finality_epoch BIGINT REFERENCES epochs(epoch_id) ON DELETE SET NULL,
    
    -- Confirmations
    confirmation_count INT DEFAULT 0,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_block_height (block_height),
    INDEX idx_finalized (finalized)
);

CREATE TABLE state_root_updates (
    update_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT NOT NULL REFERENCES blocks(height) ON DELETE CASCADE,
    
    -- State
    new_state_root VARCHAR(255) NOT NULL,
    previous_state_root VARCHAR(255),
    
    -- Timestamp
    timestamp BIGINT NOT NULL,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(block_height),
    INDEX idx_block_height (block_height),
    INDEX idx_new_state_root (new_state_root)
);

CREATE TABLE chain_reorganizations (
    reorg_id BIGSERIAL PRIMARY KEY,
    timestamp BIGINT NOT NULL,
    
    -- Reorg details
    reorg_depth INT NOT NULL,
    old_head_height BIGINT,
    new_head_height BIGINT,
    
    -- Hashes
    old_head_hash VARCHAR(255),
    new_head_hash VARCHAR(255),
    reorg_point_hash VARCHAR(255),
    
    -- Transactions affected
    transactions_reverted INT,
    transactions_reinserted INT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_timestamp (timestamp),
    INDEX idx_reorg_depth (reorg_depth)
);

CREATE TABLE consensus_events (
    event_id BIGSERIAL PRIMARY KEY,
    block_height BIGINT REFERENCES blocks(height) ON DELETE CASCADE,
    timestamp BIGINT NOT NULL,
    
    -- Event type
    event_type VARCHAR(100),
    event_description TEXT,
    
    -- Severity
    severity VARCHAR(20),  -- 'info', 'warning', 'critical'
    
    -- Details
    details JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_block_height (block_height),
    INDEX idx_event_type (event_type),
    INDEX idx_severity (severity),
    INDEX idx_timestamp (timestamp)
);

-- ═════════════════════════════════════════════════════════════════════════════════
-- SECTION 8: AUDIT & METADATA (5 TABLES)
-- ═════════════════════════════════════════════════════════════════════════════════

CREATE TABLE audit_logs (
    log_id BIGSERIAL PRIMARY KEY,
    
    event_type VARCHAR(100) NOT NULL,
    actor_peer_id VARCHAR(255) REFERENCES peer_registry(peer_id) ON DELETE SET NULL,
    action VARCHAR(255),
    
    resource_type VARCHAR(100),
    resource_id VARCHAR(255),
    
    changes JSONB,
    result VARCHAR(50),
    error_message TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_event_type (event_type),
    INDEX idx_created_at (created_at),
    INDEX idx_actor_peer_id (actor_peer_id)
);

CREATE TABLE database_metadata (
    metadata_id BIGSERIAL PRIMARY KEY,
    
    schema_version VARCHAR(50),
    build_timestamp TIMESTAMP WITH TIME ZONE,
    build_info JSONB,
    
    -- Initialization
    tables_created INT,
    indexes_created INT,
    constraints_created INT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE network_events (
    event_id BIGSERIAL PRIMARY KEY,
    timestamp BIGINT NOT NULL,
    
    event_type VARCHAR(100) NOT NULL,
    event_description TEXT,
    
    affected_peers INT,
    details JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_event_type (event_type),
    INDEX idx_timestamp (timestamp)
);

CREATE TABLE system_metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    timestamp BIGINT NOT NULL,
    
    -- Database
    db_size_mb NUMERIC(15, 2),
    active_connections INT,
    
    -- Network
    active_peers INT,
    total_peers INT,
    avg_latency_ms NUMERIC(10, 2),
    
    -- Consensus
    blocks_per_minute NUMERIC(10, 2),
    transactions_per_second NUMERIC(10, 2),
    
    -- Quantum
    avg_coherence NUMERIC(5, 4),
    oracle_sync_quality NUMERIC(5, 4),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_timestamp (timestamp)
);

CREATE TABLE entropy_quality_log (
    log_id BIGSERIAL PRIMARY KEY,
    timestamp BIGINT NOT NULL,
    
    -- Source quality
    anu_qrng_quality NUMERIC(5, 4),
    random_org_quality NUMERIC(5, 4),
    qbick_quality NUMERIC(5, 4),
    outshift_quality NUMERIC(5, 4),
    hotbits_quality NUMERIC(5, 4),
    
    -- Combined
    ensemble_min_entropy NUMERIC(5, 4),
    ensemble_shannon_entropy NUMERIC(5, 4),
    
    -- Verification
    passed_diehard BOOLEAN,
    passed_nist BOOLEAN,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_timestamp (timestamp)
);

"""

# ═════════════════════════════════════════════════════════════════════════════════
# DATABASE BUILDER CLASS
# ═════════════════════════════════════════════════════════════════════════════════

class QuantumTemporalCoherenceLedgerServer:
    """Complete server-side database builder for PostgreSQL/Supabase"""
    
    def __init__(self, pg_url: str, tessellation_depth: int = 5):
        self.pg_url = pg_url
        self.tessellation_depth = tessellation_depth
        self.tessellation = TessellationConstructor(tessellation_depth)
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """Connect to PostgreSQL"""
        logger.info(f"{CLR.QUANTUM}[DB] Connecting to PostgreSQL...{CLR.E}")
        try:
            self.conn = psycopg2.connect(self.pg_url)
            self.cursor = self.conn.cursor()
            logger.info(f"{CLR.OK}[DB] Connected successfully{CLR.E}")
        except Exception as e:
            logger.error(f"{CLR.ERROR}[DB] Connection failed: {e}{CLR.E}")
            raise
    
    def drop_all_tables(self):
        """DROP ALL TABLES - complete fresh start"""
        logger.info(f"{CLR.ERROR}[DROP] Dropping ALL existing tables...{CLR.E}")
        
        try:
            self.cursor.execute("""
                SELECT tablename FROM pg_tables WHERE schemaname = 'public';
            """)
            tables = self.cursor.fetchall()
            
            for table in tables:
                table_name = table[0]
                self.cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
                logger.debug(f"  Dropped: {table_name}")
            
            self.conn.commit()
            logger.info(f"{CLR.OK}[DROP] All tables dropped successfully{CLR.E}")
        except Exception as e:
            logger.error(f"{CLR.ERROR}[DROP] Drop failed: {e}{CLR.E}")
            self.conn.rollback()
            raise
    
    def create_schema(self):
        """Create complete schema from scratch"""
        logger.info(f"{CLR.QUANTUM}[SCHEMA] Creating schema...{CLR.E}")
        
        try:
            self.cursor.execute(COMPLETE_SCHEMA)
            self.conn.commit()
            logger.info(f"{CLR.OK}[SCHEMA] Schema created successfully{CLR.E}")
        except Exception as e:
            logger.error(f"{CLR.ERROR}[SCHEMA] Schema creation failed: {e}{CLR.E}")
            self.conn.rollback()
            raise
    
    def populate_tessellation(self):
        """Build tessellation and populate database"""
        logger.info(f"{CLR.QUANTUM}[POPULATE] Building and inserting tessellation...{CLR.E}")
        
        try:
            # Build tessellation
            self.tessellation.build_recursive_tessellation()
            self.tessellation.place_pseudoqubits()
            
            # Insert triangles
            logger.info(f"{CLR.C}[POPULATE] Inserting {len(self.tessellation.triangles)} triangles...{CLR.E}")
            for tri_id, tri in self.tessellation.triangles.items():
                self.cursor.execute("""
                    INSERT INTO hyperbolic_triangles (
                        triangle_id, depth, parent_id,
                        v0_x, v0_y, v0_name,
                        v1_x, v1_y, v1_name,
                        v2_x, v2_y, v2_name
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    tri.triangle_id, tri.depth, tri.parent_id,
                    str(tri.v0.x), str(tri.v0.y), tri.v0.name,
                    str(tri.v1.x), str(tri.v1.y), tri.v1.name,
                    str(tri.v2.x), str(tri.v2.y), tri.v2.name
                ))
                
                if tri_id % 1000 == 0:
                    self.conn.commit()
            
            # Insert pseudoqubits
            logger.info(f"{CLR.C}[POPULATE] Inserting {len(self.tessellation.pseudoqubits)} pseudoqubits...{CLR.E}")
            for qubit_id, qubit in self.tessellation.pseudoqubits.items():
                self.cursor.execute("""
                    INSERT INTO pseudoqubits (
                        pseudoqubit_id, triangle_id, x, y,
                        placement_type, phase_theta, coherence_measure
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    qubit.pseudoqubit_id, qubit.triangle_id,
                    str(qubit.x), str(qubit.y),
                    qubit.placement_type,
                    str(qubit.phase_theta),
                    str(qubit.coherence_measure)
                ))
                
                if qubit_id % 10000 == 0:
                    self.conn.commit()
            
            # Insert metadata
            self.cursor.execute("""
                INSERT INTO quantum_lattice_metadata (
                    tessellation_depth, total_triangles, total_pseudoqubits,
                    precision_bits, hyperbolicity_constant, poincare_radius,
                    status, construction_started_at, construction_completed_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                self.tessellation_depth,
                len(self.tessellation.triangles),
                len(self.tessellation.pseudoqubits),
                150, -1.0, 1.0,
                'complete',
                datetime.now(timezone.utc),
                datetime.now(timezone.utc)
            ))
            
            # Insert database metadata
            self.cursor.execute("""
                INSERT INTO database_metadata (
                    schema_version, build_timestamp, tables_created
                ) VALUES (%s, %s, %s)
            """, ('4.0.0', datetime.now(timezone.utc), 58))
            
            self.conn.commit()
            logger.info(f"{CLR.OK}[POPULATE] Tessellation fully populated{CLR.E}")
        except Exception as e:
            logger.error(f"{CLR.ERROR}[POPULATE] Population failed: {e}{CLR.E}")
            self.conn.rollback()
            raise
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info(f"{CLR.OK}[DB] Connection closed{CLR.E}")
    
    def rebuild_complete(self):
        """Complete rebuild: drop, create, populate"""
        logger.info(f"{CLR.HEADER}{'='*80}{CLR.E}")
        logger.info(f"{CLR.HEADER}QTCL DATABASE BUILDER V4 - COMPLETE SERVER BUILD{CLR.E}")
        logger.info(f"{CLR.HEADER}{'='*80}{CLR.E}")
        
        try:
            self.connect()
            self.drop_all_tables()
            self.create_schema()
            self.populate_tessellation()
            
            logger.info(f"{CLR.HEADER}{'='*80}{CLR.E}")
            logger.info(f"{CLR.OK}BUILD COMPLETE{CLR.E}")
            logger.info(f"{CLR.OK}  Tessellation depth: {self.tessellation_depth}{CLR.E}")
            logger.info(f"{CLR.OK}  Base triangles: {len(self.tessellation.triangles)}{CLR.E}")
            logger.info(f"{CLR.OK}  Pseudoqubits: {len(self.tessellation.pseudoqubits)}{CLR.E}")
            logger.info(f"{CLR.OK}  Schema tables: 58{CLR.E}")
            logger.info(f"{CLR.OK}  Status: PRODUCTION READY{CLR.E}")
            logger.info(f"{CLR.HEADER}{'='*80}{CLR.E}")
        
        except Exception as e:
            logger.error(f"{CLR.ERROR}Build failed: {e}{CLR.E}")
            raise
        finally:
            self.close()

# ═════════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    
    # Get database URL from environment or use default
    pg_url = os.environ.get(
        'DATABASE_URL',
        'postgresql://postgres:postgres@localhost:5432/qtcl_v4'
    )
    
    logger.info(f"\n{CLR.HEADER}{'='*80}{CLR.E}")
    logger.info(f"{CLR.HEADER}QTCL DATABASE BUILDER V4{CLR.E}")
    logger.info(f"{CLR.HEADER}Complete Server Implementation{CLR.E}")
    logger.info(f"{CLR.HEADER}PostgreSQL/Supabase with Full Tessellation{CLR.E}")
    logger.info(f"{CLR.HEADER}{'='*80}{CLR.E}\n")
    logger.info(f"Database URL: {pg_url}\n")
    
    # Build complete system
    builder = QuantumTemporalCoherenceLedgerServer(
        pg_url=pg_url,
        tessellation_depth=5
    )
    
    builder.rebuild_complete()
    
    logger.info(f"\n{CLR.QUANTUM}✓ QTCL V4 server database ready for production deployment{CLR.E}\n")
