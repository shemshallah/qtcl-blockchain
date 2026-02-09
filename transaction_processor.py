#!/usr/bin/env python3
"""
transaction_processor.py
Bridges API layer with quantum execution
"""

import uuid
import time
import logging
import threading
import json
from datetime import datetime
from typing import Dict, Optional
from db_config import DatabaseConnection, Config

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# TRANSACTION PROCESSOR - Handles submission and execution
# ═══════════════════════════════════════════════════════════════════════════

class TransactionProcessor:
    """Process quantum transactions from submission to finality"""
    
    def __init__(self):
        self.running = False
        self.worker_thread = None
        self.tx_queue = {}
    
    def start(self):
        """Start background worker thread"""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            logger.info("[TXN] Transaction processor started")
    
    def stop(self):
        """Stop background worker"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("[TXN] Transaction processor stopped")
    
    def submit_transaction(
        self,
        from_user: str,
        to_user: str,
        amount: float,
        tx_type: str = 'transfer',
        metadata: Optional[Dict] = None
    ) -> Dict:
        """Submit new transaction for quantum processing"""
        
        tx_id = f"tx_{uuid.uuid4().hex[:16]}"
        timestamp = datetime.utcnow().isoformat()
        
        try:
            # Insert into database
            DatabaseConnection.execute_update(
                """INSERT INTO transactions 
                   (tx_id, from_user_id, to_user_id, amount, tx_type, status, 
                    created_at, metadata)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    tx_id,
                    from_user,
                    to_user,
                    amount,
                    tx_type,
                    'pending',
                    timestamp,
                    json.dumps(metadata or {})
                )
            )
            
            # Track locally
            self.tx_queue[tx_id] = {
                'status': 'pending',
                'submitted_at': datetime.utcnow(),
                'from_user': from_user,
                'to_user': to_user,
                'amount': amount,
                'type': tx_type
            }
            
            logger.info(f"[TXN] Submitted {tx_id}: {from_user} → {to_user} ({amount})")
            
            return {
                'status': 'success',
                'tx_id': tx_id,
                'message': 'Transaction submitted for quantum processing'
            }
        
        except Exception as e:
            logger.error(f"[TXN] Submit failed: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def get_transaction_status(self, tx_id: str) -> Dict:
        """Get current status of a transaction"""
        try:
            result = DatabaseConnection.execute(
                "SELECT tx_id, status, created_at, quantum_state_hash, entropy_score "
                "FROM transactions WHERE tx_id = %s",
                (tx_id,)
            )
            
            if result:
                tx = result[0]
                return {
                    'status': 'found',
                    'tx_id': tx['tx_id'],
                    'tx_status': tx['status'],
                    'quantum_state_hash': tx['quantum_state_hash'],
                    'entropy_score': tx['entropy_score'],
                    'created_at': tx['created_at'].isoformat() if tx['created_at'] else None
                }
            else:
                return {
                    'status': 'not_found',
                    'tx_id': tx_id
                }
        
        except Exception as e:
            logger.error(f"[TXN] Status check failed for {tx_id}: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _worker_loop(self):
        """Background worker: poll for pending transactions and execute"""
        logger.info("[TXN] Worker loop started")
        
        while self.running:
            try:
                # Get pending transactions
                pending = DatabaseConnection.execute(
                    """SELECT tx_id, from_user_id, to_user_id, amount, tx_type, 
                              created_at, metadata
                       FROM transactions 
                       WHERE status = 'pending' 
                       ORDER BY created_at ASC 
                       LIMIT 5"""
                )
                
                if pending:
                    logger.info(f"[TXN] Processing {len(pending)} transactions")
                    for tx in pending:
                        self._execute_transaction(tx)
                
                # Update local queue
                self._cleanup_old_transactions()
                
                # Sleep before next poll
                time.sleep(2)
            
            except Exception as e:
                logger.error(f"[TXN] Worker error: {e}")
                time.sleep(5)
        
        logger.info("[TXN] Worker loop stopped")
    
    def _execute_transaction(self, tx: Dict):
    """Execute transaction with W-state validator consensus + GHZ-8"""
    tx_id = tx['tx_id']
    
    try:
        logger.info(f"[TXN] Executing {tx_id} ({tx['tx_type']})")
        
        # Mark as processing
        DatabaseConnection.execute_update(
            "UPDATE transactions SET status = %s WHERE tx_id = %s",
            ('processing', tx_id)
        )
        
        # Import W-state executor
        from wsv_integration import get_wsv_executor
        
        # Execute with W-state validator consensus
        wsv_executor = get_wsv_executor(DatabaseConnection)
        result = wsv_executor.execute_transaction(
            tx_id=tx_id,
            from_user=tx['from_user_id'],
            to_user=tx['to_user_id'],
            amount=tx['amount'],
            metadata=json.loads(tx.get('metadata', '{}'))

    def _cleanup_old_transactions(self):
        """Remove old transactions from local queue (keep last 100)"""
        if len(self.tx_queue) > 100:
            # Sort by submission time
            sorted_txs = sorted(
                self.tx_queue.items(),
                key=lambda x: x[1]['submitted_at']
            )
            # Keep only newest 100
            self.tx_queue = dict(sorted_txs[-100:])

# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL PROCESSOR INSTANCE
# ═══════════════════════════════════════════════════════════════════════════

processor = TransactionProcessor()

# ═══════════════════════════════════════════════════════════════════════════
# FLASK API ROUTES
# ═══════════════════════════════════════════════════════════════════════════

def register_transaction_routes(app):
    """Register transaction endpoints with Flask app"""
    from flask import request, jsonify
    
    @app.route('/api/transactions', methods=['POST'])
    def submit_transaction():
        """Submit new transaction for processing"""
        try:
            data = request.get_json()
            
            # Validate required fields
            required = ['from_user', 'to_user', 'amount']
            if not all(k in data for k in required):
                return jsonify({
                    'status': 'error',
                    'message': f'Missing required fields: {required}'
                }), 400
            
            # Submit transaction
            result = processor.submit_transaction(
                from_user=data['from_user'],
                to_user=data['to_user'],
                amount=float(data['amount']),
                tx_type=data.get('tx_type', 'transfer'),
                metadata=data.get('metadata', {})
            )
            
            if result['status'] == 'success':
                return jsonify(result), 202  # Accepted for async processing
            else:
                return jsonify(result), 400
        
        except Exception as e:
            logger.error(f"[API] Submit error: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    @app.route('/api/transactions/<tx_id>', methods=['GET'])
    def get_transaction(tx_id):
        """Get transaction status and details"""
        try:
            result = processor.get_transaction_status(tx_id)
            return jsonify(result), 200
        except Exception as e:
            logger.error(f"[API] Get error: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    @app.route('/api/transactions', methods=['GET'])
    def list_transactions():
        """List recent transactions"""
        try:
            limit = request.args.get('limit', 50, type=int)
            
            results = DatabaseConnection.execute(
                """SELECT tx_id, from_user_id, to_user_id, amount, tx_type, status, 
                          created_at, entropy_score
                   FROM transactions
                   ORDER BY created_at DESC
                   LIMIT %s""",
                (limit,)
            )
            
            return jsonify({
                'status': 'success',
                'count': len(results),
                'transactions': [
                    {
                        'tx_id': t['tx_id'],
                        'from': t['from_user_id'],
                        'to': t['to_user_id'],
                        'amount': t['amount'],
                        'type': t['tx_type'],
                        'status': t['status'],
                        'entropy': t['entropy_score'],
                        'created_at': t['created_at'].isoformat() if t['created_at'] else None
                    }
                    for t in results
                ]
            }), 200
        
        except Exception as e:
            logger.error(f"[API] List error: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Test locally
    processor.start()
    time.sleep(1)
    
    # Submit test transaction
    result = processor.submit_transaction(
        from_user='user1',
        to_user='user2',
        amount=100.0,
        tx_type='transfer'
    )
    print(f"Submitted: {result}")
    
    # Check status
    tx_id = result.get('tx_id')
    if tx_id:
        time.sleep(3)
        status = processor.get_transaction_status(tx_id)
        print(f"Status: {status}")
    
    processor.stop()
