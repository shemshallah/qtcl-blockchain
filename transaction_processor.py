#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  QTCL TRANSACTION PROCESSOR v2.0                                             ║
║  Gas-Free | W-State Consensus | GHZ-8 Quantum Finality                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import uuid, time, json, logging, threading
from datetime import datetime
from typing import Dict, Optional, Any

from db_config import DatabaseConnection
from quantum_engine import get_quantum_executor, QuantumFinalityProof

logger = logging.getLogger(__name__)


class TransactionProcessor:
    WORKER_POLL_INTERVAL = 2
    WORKER_ERROR_SLEEP   = 5
    WORKER_BATCH_SIZE    = 5
    LOCAL_CACHE_MAX      = 200

    def __init__(self):
        self.running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._local_cache: Dict[str, Dict] = {}
        self._cache_lock = threading.RLock()
        self._executor = None

    def _get_executor(self):
        if self._executor is None:
            self._executor = get_quantum_executor()
        return self._executor

    def start(self) -> None:
        if not self.running:
            self.running = True
            self._worker_thread = threading.Thread(
                target=self._worker_loop, daemon=True, name='QTCLWorker')
            self._worker_thread.start()
            logger.info("[TXN] Quantum transaction processor started (gas-free)")

    def stop(self) -> None:
        self.running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=10)
        logger.info("[TXN] Transaction processor stopped")

    def submit_transaction(self, from_user: str, to_user: str, amount: float,
                           tx_type: str = 'transfer',
                           metadata: Optional[Dict] = None) -> Dict[str, Any]:
        tx_id = f"tx_{uuid.uuid4().hex[:16]}"
        now = datetime.utcnow().isoformat()
        meta_json = json.dumps(metadata or {})
        try:
            DatabaseConnection.execute_update(
                """INSERT INTO transactions
                   (tx_id, from_user_id, to_user_id, amount, tx_type,
                    status, created_at, metadata)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (tx_id, from_user, to_user, float(amount),
                 tx_type, 'pending', now, meta_json))
            with self._cache_lock:
                self._local_cache[tx_id] = {
                    'status': 'pending', 'submitted_at': datetime.utcnow(),
                    'from_user': from_user, 'to_user': to_user,
                    'amount': amount, 'type': tx_type}
            logger.info(f"[TXN] Submitted {tx_id}: {from_user} -> {to_user} ({amount})")
            return {'status': 'success', 'tx_id': tx_id,
                    'message': 'Transaction queued for quantum finality (gas-free)'}
        except Exception as exc:
            logger.error(f"[TXN] Submit failed: {exc}")
            return {'status': 'error', 'message': str(exc)}

    def get_transaction_status(self, tx_id: str) -> Dict[str, Any]:
        try:
            rows = DatabaseConnection.execute(
                """SELECT tx_id, status, created_at,
                          quantum_state_hash, commitment_hash,
                          entropy_score, validator_agreement,
                          circuit_depth, execution_time_ms
                   FROM transactions WHERE tx_id = %s""", (tx_id,))
            if not rows:
                return {'status': 'not_found', 'tx_id': tx_id}
            tx = rows[0]
            return {
                'status': 'found', 'tx_id': tx['tx_id'],
                'tx_status': tx['status'],
                'quantum_state_hash': tx['quantum_state_hash'],
                'commitment_hash': tx['commitment_hash'],
                'entropy_score': float(tx['entropy_score']) if tx['entropy_score'] else None,
                'validator_agreement': float(tx['validator_agreement']) if tx['validator_agreement'] else None,
                'circuit_depth': tx['circuit_depth'],
                'execution_time_ms': float(tx['execution_time_ms']) if tx['execution_time_ms'] else None,
                'created_at': tx['created_at'].isoformat() if tx['created_at'] else None,
                'gas': None,
                'gas_price': None,
            }
        except Exception as exc:
            logger.error(f"[TXN] Status check error for {tx_id}: {exc}")
            return {'status': 'error', 'message': str(exc)}

    def _worker_loop(self) -> None:
        logger.info("[TXN] Worker loop online")
        while self.running:
            try:
                pending = DatabaseConnection.execute(
                    """SELECT tx_id, from_user_id, to_user_id, amount,
                              tx_type, created_at, metadata
                       FROM transactions
                       WHERE status = 'pending'
                       ORDER BY created_at ASC
                       LIMIT %s""", (self.WORKER_BATCH_SIZE,))
                if pending:
                    logger.info(f"[TXN] Processing {len(pending)} transactions")
                    for tx in pending:
                        self._execute_transaction(tx)
                self._trim_local_cache()
                time.sleep(self.WORKER_POLL_INTERVAL)
            except Exception as exc:
                logger.error(f"[TXN] Worker loop error: {exc}", exc_info=True)
                time.sleep(self.WORKER_ERROR_SLEEP)
        logger.info("[TXN] Worker loop stopped")

    def _execute_transaction(self, tx: Dict) -> None:
        tx_id = tx['tx_id']
        try:
            logger.info(f"[TXN] Executing {tx_id} ({tx.get('tx_type', 'transfer')})")
            DatabaseConnection.execute_update(
                "UPDATE transactions SET status = 'processing' WHERE tx_id = %s",
                (tx_id,))
            meta = {}
            try:
                meta = json.loads(tx.get('metadata') or '{}')
            except (json.JSONDecodeError, TypeError):
                pass
            proof: QuantumFinalityProof = self._get_executor().execute_transaction(
                tx_id=tx_id,
                user_id=tx['from_user_id'],
                target_id=tx['to_user_id'],
                amount=float(tx['amount']),
                metadata=meta)
            DatabaseConnection.execute_update(
                """UPDATE transactions SET
                       status              = 'finalized',
                       quantum_state_hash  = %s,
                       commitment_hash     = %s,
                       entropy_score       = %s,
                       validator_agreement = %s,
                       circuit_depth       = %s,
                       circuit_size        = %s,
                       execution_time_ms   = %s,
                       finalized_at        = %s
                   WHERE tx_id = %s""",
                (proof.state_hash, proof.commitment_hash,
                 proof.entropy_normalized * 100,
                 proof.validator_agreement_score,
                 proof.circuit_depth, proof.circuit_size,
                 proof.execution_time_ms,
                 datetime.utcnow().isoformat(), tx_id))
            self._persist_quantum_measurement(tx_id, proof)
            with self._cache_lock:
                if tx_id in self._local_cache:
                    self._local_cache[tx_id].update({
                        'status': 'finalized',
                        'commitment_hash': proof.commitment_hash,
                        'ghz_fidelity': proof.ghz_fidelity,
                        'entropy': proof.entropy_normalized})
            logger.info(
                f"[TXN] Finalized {tx_id} | ghz={proof.ghz_fidelity:.4f} | "
                f"entropy={proof.entropy_normalized:.3f} | "
                f"valid={proof.is_valid_finality}")
        except Exception as exc:
            logger.error(f"[TXN] Execution failed for {tx_id}: {exc}", exc_info=True)
            try:
                DatabaseConnection.execute_update(
                    "UPDATE transactions SET status = 'failed' WHERE tx_id = %s",
                    (tx_id,))
            except Exception as db_exc:
                logger.error(f"[TXN] Could not mark {tx_id} as failed: {db_exc}")

    def _persist_quantum_measurement(self, tx_id: str, proof: QuantumFinalityProof) -> None:
        try:
            DatabaseConnection.execute_update(
                """INSERT INTO quantum_measurements
                   (tx_id, measurement_result_json, validator_consensus_json,
                    entropy_score, ghz_fidelity, commitment_hash, noise_source,
                    is_valid_finality, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (tx_id) DO UPDATE SET
                       measurement_result_json = EXCLUDED.measurement_result_json,
                       ghz_fidelity            = EXCLUDED.ghz_fidelity,
                       commitment_hash         = EXCLUDED.commitment_hash""",
                (tx_id, json.dumps(proof.to_dict(), default=str),
                 json.dumps(proof.validator_consensus),
                 proof.entropy_normalized * 100, proof.ghz_fidelity,
                 proof.commitment_hash, proof.noise_source,
                 proof.is_valid_finality, datetime.utcnow().isoformat()))
        except Exception as exc:
            logger.warning(f"[TXN] Could not persist quantum measurement for {tx_id}: {exc}")

    def _trim_local_cache(self) -> None:
        with self._cache_lock:
            if len(self._local_cache) > self.LOCAL_CACHE_MAX:
                oldest = sorted(
                    self._local_cache.items(),
                    key=lambda kv: kv[1].get('submitted_at', datetime.min))
                for tx_id, _ in oldest[:len(self._local_cache) - self.LOCAL_CACHE_MAX]:
                    del self._local_cache[tx_id]


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────────────────────────────────────────
processor = TransactionProcessor()


def register_transaction_routes(app) -> None:
    from flask import request, jsonify

    @app.route('/api/transactions', methods=['POST'])
    def submit_transaction():
        try:
            data = request.get_json(force=True) or {}
            required = ['from_user', 'to_user', 'amount']
            missing = [f for f in required if f not in data]
            if missing:
                return jsonify({'status': 'error',
                                'message': f"Missing: {missing}"}), 400
            result = processor.submit_transaction(
                from_user=data['from_user'], to_user=data['to_user'],
                amount=float(data['amount']),
                tx_type=data.get('tx_type', 'transfer'),
                metadata=data.get('metadata', {}))
            return jsonify(result), 202 if result['status'] == 'success' else 400
        except Exception as exc:
            logger.error(f"[API] POST /transactions: {exc}", exc_info=True)
            return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

    @app.route('/api/transactions/<tx_id>', methods=['GET'])
    def get_transaction(tx_id):
        try:
            return jsonify(processor.get_transaction_status(tx_id)), 200
        except Exception as exc:
            return jsonify({'status': 'error', 'message': str(exc)}), 500

    @app.route('/api/transactions', methods=['GET'])
    def list_transactions():
        try:
            limit = min(request.args.get('limit', 50, type=int), 500)
            status_filter = request.args.get('status')
            sql = ("SELECT tx_id, from_user_id, to_user_id, amount, tx_type, status, "
                   "created_at, entropy_score, validator_agreement, commitment_hash "
                   "FROM transactions")
            params = []
            if status_filter:
                sql += " WHERE status = %s"
                params.append(status_filter)
            sql += " ORDER BY created_at DESC LIMIT %s"
            params.append(limit)
            rows = DatabaseConnection.execute(sql, tuple(params))
            return jsonify({
                'status': 'success', 'count': len(rows),
                'transactions': [{
                    'tx_id': t['tx_id'], 'from': t['from_user_id'],
                    'to': t['to_user_id'], 'amount': float(t['amount']),
                    'type': t['tx_type'], 'status': t['status'],
                    'entropy_score': float(t['entropy_score']) if t['entropy_score'] else None,
                    'validator_agreement': float(t['validator_agreement']) if t['validator_agreement'] else None,
                    'commitment_hash': t['commitment_hash'],
                    'created_at': t['created_at'].isoformat() if t['created_at'] else None,
                    'gas': None,
                } for t in rows]
            }), 200
        except Exception as exc:
            logger.error(f"[API] GET /transactions: {exc}", exc_info=True)
            return jsonify({'status': 'error', 'message': str(exc)}), 500

    @app.route('/api/quantum/stats', methods=['GET'])
    def quantum_stats():
        try:
            stats = processor._get_executor().get_stats()
            w = processor._get_executor().w_bus.get_current_state()
            return jsonify({
                'status': 'success', 'quantum_stats': stats,
                'w_state_bus': {
                    'validators': w.validator_ids,
                    'cycle_count': w.cycle_count,
                    'cumulative_agreement': w.cumulative_agreement,
                    'last_collapse': w.last_collapse_outcome,
                }}), 200
        except Exception as exc:
            return jsonify({'status': 'error', 'message': str(exc)}), 500

    logger.info("[TXN] Transaction routes registered (gas-free quantum finality)")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] [%(name)s] %(levelname)s: %(message)s')
    processor.start()
    time.sleep(0.5)
    result = processor.submit_transaction('alice', 'bob', 99.0)
    print(f"Submit: {result}")
    if result.get('tx_id'):
        time.sleep(1)
        print(f"Status: {processor.get_transaction_status(result['tx_id'])}")
    processor.stop()
