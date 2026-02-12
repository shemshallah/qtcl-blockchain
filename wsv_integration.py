#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  WSV INTEGRATION v2.0 — Thin bridge: legacy → quantum_engine               ║
║                                                                              ║
║  This module exists for backwards compatibility.                             ║
║  All real work now lives in quantum_engine.QuantumTXExecutor                ║
║  This wrapper adapts the old WStateTransactionExecutor interface.            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import logging
from typing import Dict, Optional, Any

from quantum_engine import get_quantum_executor, QuantumFinalityProof

logger = logging.getLogger(__name__)


class WStateTransactionExecutor:
    """
    Backward-compatible facade over the new QuantumTXExecutor.
    Drop-in replacement — same interface, real quantum underneath.
    """

    def __init__(self, database_connection=None):
        self._executor = get_quantum_executor()
        self.db_connection = database_connection
        logger.info("[WSV] WStateTransactionExecutor facade initialized -> quantum_engine")

    def execute_transaction(self, tx_id: str, from_user: str, to_user: str,
                            amount: float, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute transaction — returns dict matching old WSV interface."""
        try:
            proof: QuantumFinalityProof = self._executor.execute_transaction(
                tx_id=tx_id, user_id=from_user, target_id=to_user,
                amount=amount, metadata=metadata)

            # If database connection provided, attempt measurement persistence
            if self.db_connection:
                try:
                    self.db_connection.execute_update(
                        """UPDATE transactions SET
                               quantum_state_hash  = %s,
                               commitment_hash     = %s,
                               entropy_score       = %s,
                               validator_agreement = %s
                           WHERE tx_id = %s""",
                        (proof.state_hash, proof.commitment_hash,
                         proof.entropy_normalized * 100,
                         proof.validator_agreement_score, tx_id))
                except Exception as db_exc:
                    logger.warning(f"[WSV] DB update skipped: {db_exc}")

            return {
                'status': 'success',
                'tx_id': tx_id,
                'message': 'Quantum finality achieved (gas-free)',
                'quantum_results': {
                    'circuit_name':            f"qtcl_{tx_id[:12]}",
                    'num_qubits':              8,
                    'circuit_depth':           proof.circuit_depth,
                    'circuit_size':            proof.circuit_size,
                    'execution_time_ms':       proof.execution_time_ms,
                    'entropy_percent':         proof.entropy_normalized * 100,
                    'ghz_fidelity':            proof.ghz_fidelity,
                    'validator_consensus':     proof.validator_consensus,
                    'validator_agreement_score': proof.validator_agreement_score,
                    'user_signature':          proof.user_signature_bit,
                    'target_signature':        proof.target_signature_bit,
                    'oracle_collapse_bit':     proof.oracle_collapse_bit,
                    'state_hash':              proof.state_hash,
                    'commitment_hash':         proof.commitment_hash,
                    'dominant_bitstring':      proof.dominant_bitstring,
                    'mev_proof_score':         proof.mev_proof_score,
                    'noise_source':            proof.noise_source,
                    'is_valid_finality':       proof.is_valid_finality,
                }
            }
        except Exception as exc:
            logger.error(f"[WSV] execute_transaction failed for {tx_id}: {exc}")
            return {'status': 'error', 'tx_id': tx_id, 'message': str(exc)}

    def get_stats(self) -> Dict[str, Any]:
        return self._executor.get_stats()


class WStateTransactionProcessorAdapter:
    """Adapter for legacy transaction_processor integration."""

    def __init__(self, wsv_executor: WStateTransactionExecutor):
        self.wsv_executor = wsv_executor

    def execute_transaction_with_fallback(self, tx: Dict[str, Any]) -> Dict[str, Any]:
        meta = {}
        try:
            meta = json.loads(tx.get('metadata') or '{}')
        except Exception:
            pass
        return self.wsv_executor.execute_transaction(
            tx_id=tx['tx_id'],
            from_user=tx['from_user_id'],
            to_user=tx['to_user_id'],
            amount=tx['amount'],
            metadata=meta)


# ─────────────────────────────────────────────────────────────────────────────
# Singleton — for any code that still uses get_wsv_executor()
# ─────────────────────────────────────────────────────────────────────────────
_wsv_executor_instance = None

def get_wsv_executor(db_connection=None) -> WStateTransactionExecutor:
    global _wsv_executor_instance
    if _wsv_executor_instance is None:
        _wsv_executor_instance = WStateTransactionExecutor(db_connection)
    return _wsv_executor_instance


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    ex = get_wsv_executor()
    result = ex.execute_transaction('tx_wsv_test_001', 'alice', 'bob', 42.0)
    import json
    print(json.dumps(result, indent=2, default=str))
