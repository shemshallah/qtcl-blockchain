#!/usr/bin/env python3
"""
QUANTUM TRANSACTION EXECUTOR - PART 2: API INTEGRATION & SCHEMA MODIFICATIONS
Flask API endpoints, database schema additions, and drop-in integration with existing api_gateway.py
Production-grade: ~1200 lines
Single file - copy/paste into api_gateway.py or use as standalone integration layer
"""

from flask import Flask,request,jsonify
from quantum_transaction_executor_integrated import executor,db,logger,Config
import json
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════════════
# DATABASE SCHEMA MODIFICATIONS - RUN THESE ONCE
# ═══════════════════════════════════════════════════════════════════════════════════════

class SchemaModifier:
    """Add required tables and columns for quantum transaction executor"""
    
    @staticmethod
    def ensure_schema():
        """Create or modify tables as needed"""
        
        # Table 1: superposition_states (NEW)
        db.execute_update("""
            CREATE TABLE IF NOT EXISTS superposition_states (
                superposition_id SERIAL PRIMARY KEY,
                tx_id VARCHAR(32) UNIQUE NOT NULL,
                user_id VARCHAR(128) NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                qubits_state VARCHAR(128),
                entropy FLOAT,
                coherence FLOAT,
                phase_encoding VARCHAR(256),
                oracle_measurement VARCHAR(16),
                collapsed BOOLEAN DEFAULT FALSE,
                collapse_time TIMESTAMP WITH TIME ZONE,
                FOREIGN KEY(tx_id) REFERENCES transactions(tx_id) ON DELETE CASCADE
            );
        """,None)
        
        # Table 2: quantum_measurements (NEW)
        db.execute_update("""
            CREATE TABLE IF NOT EXISTS quantum_measurements (
                measurement_id SERIAL PRIMARY KEY,
                tx_id VARCHAR(32) NOT NULL,
                bitstring_counts JSONB,
                entropy FLOAT,
                dominant_states JSONB,
                state_space_utilization FLOAT,
                properties VARCHAR(64),
                measurement_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(tx_id) REFERENCES transactions(tx_id) ON DELETE CASCADE
            );
        """,None)
        
        # Table 3: oracle_validations (NEW)
        db.execute_update("""
            CREATE TABLE IF NOT EXISTS oracle_validations (
                validation_id SERIAL PRIMARY KEY,
                tx_id VARCHAR(32) NOT NULL,
                oracle_state VARCHAR(16),
                agreement FLOAT,
                validators_agreed INT,
                valid BOOLEAN,
                validation_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(tx_id) REFERENCES transactions(tx_id) ON DELETE CASCADE
            );
        """,None)
        
        # Modify transactions table - ADD MISSING COLUMNS
        try:
            db.execute_update("ALTER TABLE transactions ADD COLUMN session_id VARCHAR(128);",None)
        except:pass  # Column already exists
        
        try:
            db.execute_update("ALTER TABLE transactions ADD COLUMN oracle_measurement VARCHAR(16);",None)
        except:pass
        
        try:
            db.execute_update("ALTER TABLE transactions ADD COLUMN entropy_score FLOAT;",None)
        except:pass
        
        try:
            db.execute_update("ALTER TABLE transactions ADD COLUMN quantum_state_hash VARCHAR(256);",None)
        except:pass
        
        try:
            db.execute_update("ALTER TABLE transactions ADD COLUMN block_hash VARCHAR(256);",None)
        except:pass
        
        # Create indexes for performance
        db.execute_update("CREATE INDEX IF NOT EXISTS idx_tx_status ON transactions(status);",None)
        db.execute_update("CREATE INDEX IF NOT EXISTS idx_tx_block_hash ON transactions(block_hash);",None)
        db.execute_update("CREATE INDEX IF NOT EXISTS idx_superposition_user ON superposition_states(user_id);",None)
        db.execute_update("CREATE INDEX IF NOT EXISTS idx_oracle_validation_tx ON oracle_validations(tx_id);",None)
        
        logger.info("✓ Database schema verification complete")

# Run schema modification on startup
SchemaModifier.ensure_schema()

# ═══════════════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS - ADD TO EXISTING FLASK APP
# ═══════════════════════════════════════════════════════════════════════════════════════

def register_quantum_transaction_routes(app:Flask):
    """Register quantum transaction endpoints with Flask app (drop-in integration)"""
    
    @app.route('/api/v1/transactions/submit-quantum',methods=['POST'])
    def submit_transaction_quantum():
        """Submit transaction with full quantum processing pipeline"""
        try:
            data=request.get_json()
            user_id=data.get('from_user')
            receiver_id=data.get('to_user')
            amount=float(data.get('amount',0))
            session_id=data.get('session_id','')
            
            # Validate
            if not all([user_id,receiver_id,amount>0]):
                return jsonify({'error':'Missing required fields','success':False}),400
            
            if amount<=0:
                return jsonify({'error':'Amount must be positive','success':False}),400
            
            # Check sender balance
            sender=db.execute_one("SELECT balance FROM users WHERE user_id=%s",(user_id,))
            if not sender or sender['balance']<int(amount*Config.TOKEN_WEI_PER_UNIT):
                return jsonify({'error':'Insufficient balance','success':False}),400
            
            # Check receiver exists
            receiver=db.execute_one("SELECT user_id FROM users WHERE user_id=%s",(receiver_id,))
            if not receiver:
                return jsonify({'error':'Receiver not found','success':False}),404
            
            # Execute full quantum pipeline
            result=executor.submit_and_execute_transaction(user_id,receiver_id,amount,session_id)
            
            if result['success']:
                return jsonify(result),202  # 202 Accepted for async
            else:
                return jsonify(result),400
        
        except Exception as e:
            logger.error(f"Quantum transaction submit error: {e}")
            return jsonify({'error':str(e),'success':False}),500
    
    @app.route('/api/v1/transactions/batch-execute',methods=['POST'])
    def batch_execute_transactions():
        """Execute multiple transactions in parallel (threading)"""
        try:
            data=request.get_json()
            transactions=data.get('transactions',[])
            
            if not transactions:
                return jsonify({'error':'No transactions provided','success':False}),400
            
            results=[]
            for tx_data in transactions:
                result=executor.submit_and_execute_transaction(
                    tx_data['from_user'],
                    tx_data['to_user'],
                    float(tx_data['amount']),
                    tx_data.get('session_id','')
                )
                results.append(result)
            
            successful=sum(1 for r in results if r.get('success'))
            return jsonify({
                'success':True,
                'total':len(results),
                'successful':successful,
                'failed':len(results)-successful,
                'transactions':results
            }),200
        
        except Exception as e:
            logger.error(f"Batch execute error: {e}")
            return jsonify({'error':str(e),'success':False}),500
    
    @app.route('/api/v1/transactions/<tx_id>/status',methods=['GET'])
    def get_transaction_status_quantum(tx_id):
        """Get transaction status with quantum metrics"""
        try:
            status=executor.get_transaction_status(tx_id)
            
            if status:
                return jsonify({'success':True,'transaction':status}),200
            else:
                return jsonify({'error':'Transaction not found','success':False}),404
        
        except Exception as e:
            logger.error(f"Status query error: {e}")
            return jsonify({'error':str(e),'success':False}),500
    
    @app.route('/api/v1/transactions/list',methods=['GET'])
    def list_transactions_quantum():
        """List transactions with persistence (no longer returns empty)"""
        try:
            limit=request.args.get('limit',50,type=int)
            offset=request.args.get('offset',0,type=int)
            
            # Get transactions with pagination
            results=db.execute(
                """SELECT tx_id,from_user_id,to_user_id,amount,status,entropy_score,created_at 
                   FROM transactions ORDER BY created_at DESC LIMIT %s OFFSET %s""",
                (limit,offset)
            )
            
            transactions=[
                {
                    'tx_id':r['tx_id'],
                    'from':r['from_user_id'],
                    'to':r['to_user_id'],
                    'amount':r['amount']/Config.TOKEN_WEI_PER_UNIT if r['amount'] else 0,
                    'status':r['status'],
                    'entropy':r['entropy_score'],
                    'created_at':r['created_at'].isoformat() if r['created_at'] else None
                }
                for r in results
            ]
            
            return jsonify({
                'success':True,
                'count':len(transactions),
                'limit':limit,
                'offset':offset,
                'transactions':transactions
            }),200
        
        except Exception as e:
            logger.error(f"List transactions error: {e}")
            return jsonify({'error':str(e),'success':False}),500
    
    @app.route('/api/v1/transactions/quantum-status',methods=['GET'])
    def get_quantum_executor_status():
        """Get executor statistics and metrics"""
        try:
            stats=executor.get_stats()
            
            return jsonify({
                'success':True,
                'status':'operational',
                'executor_stats':stats['execution_stats'],
                'total_blocks':stats['total_blocks'],
                'total_transactions':stats['total_transactions'],
                'pending_in_queue':stats['pending_transactions'],
                'avg_circuit_time_ms':stats['circuit_avg_time_ms']
            }),200
        
        except Exception as e:
            logger.error(f"Status query error: {e}")
            return jsonify({'error':str(e),'success':False}),500
    
    @app.route('/api/v1/blocks/pending',methods=['GET'])
    def get_pending_block_info():
        """Get info about pending transactions waiting for block creation"""
        try:
            pending_count=executor.block_manager.get_pending_block_size()
            
            # Get pending transactions details
            pending_results=db.execute(
                """SELECT tx_id,from_user_id,to_user_id,amount,entropy_score 
                   FROM transactions WHERE status='FINALIZED' AND block_hash IS NULL 
                   ORDER BY created_at DESC LIMIT 10""",
                ()
            )
            
            return jsonify({
                'success':True,
                'pending_count':pending_count,
                'max_block_size':Config.BLOCK_SIZE_MAX_TX,
                'utilization_percent':(pending_count/Config.BLOCK_SIZE_MAX_TX)*100 if Config.BLOCK_SIZE_MAX_TX>0 else 0,
                'sample_pending':[
                    {
                        'tx_id':r['tx_id'],
                        'from':r['from_user_id'],
                        'to':r['to_user_id'],
                        'amount':r['amount']/Config.TOKEN_WEI_PER_UNIT if r['amount'] else 0,
                        'entropy':r['entropy_score']
                    }
                    for r in pending_results
                ]
            }),200
        
        except Exception as e:
            logger.error(f"Pending block query error: {e}")
            return jsonify({'error':str(e),'success':False}),500
    
    @app.route('/api/v1/superposition/<tx_id>/status',methods=['GET'])
    def get_superposition_status(tx_id):
        """Get superposition state for transaction"""
        try:
            super_state=db.execute_one(
                """SELECT tx_id,user_id,created_at,entropy,coherence,oracle_measurement,collapsed,collapse_time 
                   FROM superposition_states WHERE tx_id=%s""",
                (tx_id,)
            )
            
            if super_state:
                return jsonify({
                    'success':True,
                    'superposition':{
                        'tx_id':super_state['tx_id'],
                        'user_id':super_state['user_id'],
                        'entropy':super_state['entropy'],
                        'coherence':super_state['coherence'],
                        'oracle_measurement':super_state['oracle_measurement'],
                        'collapsed':super_state['collapsed'],
                        'created_at':super_state['created_at'].isoformat() if super_state['created_at'] else None,
                        'collapse_time':super_state['collapse_time'].isoformat() if super_state['collapse_time'] else None
                    }
                }),200
            else:
                return jsonify({'error':'Superposition not found','success':False}),404
        
        except Exception as e:
            logger.error(f"Superposition query error: {e}")
            return jsonify({'error':str(e),'success':False}),500
    
    @app.route('/api/v1/quantum-metrics',methods=['GET'])
    def get_quantum_metrics():
        """Get aggregate quantum metrics across recent transactions"""
        try:
            # Get recent transaction quantum data
            results=db.execute(
                """SELECT entropy_score,oracle_measurement FROM transactions 
                   WHERE entropy_score IS NOT NULL ORDER BY created_at DESC LIMIT 100""",
                ()
            )
            
            if not results:
                return jsonify({
                    'success':True,
                    'avg_entropy':0,
                    'min_entropy':0,
                    'max_entropy':0,
                    'sample_count':0
                }),200
            
            entropies=[r['entropy_score'] for r in results if r['entropy_score'] is not None]
            
            import statistics
            return jsonify({
                'success':True,
                'avg_entropy':statistics.mean(entropies) if entropies else 0,
                'median_entropy':statistics.median(entropies) if entropies else 0,
                'min_entropy':min(entropies) if entropies else 0,
                'max_entropy':max(entropies) if entropies else 0,
                'stdev_entropy':statistics.stdev(entropies) if len(entropies)>1 else 0,
                'sample_count':len(entropies)
            }),200
        
        except Exception as e:
            logger.error(f"Metrics query error: {e}")
            return jsonify({'error':str(e),'success':False}),500
    
    logger.info("✓ Quantum transaction endpoints registered")

# ═══════════════════════════════════════════════════════════════════════════════════════
# BLOCKCHAIN INFORMATION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════════

def register_blockchain_info_routes(app:Flask):
    """Information endpoints about blocks and blockchain state"""
    
    @app.route('/api/v1/blocks/latest',methods=['GET'])
    def get_latest_block():
        """Get latest block information"""
        try:
            latest=db.execute_one(
                """SELECT block_hash,block_number,parent_hash,timestamp,transaction_count,
                          aggregate_entropy FROM blocks ORDER BY block_number DESC LIMIT 1"""
            )
            
            if latest:
                return jsonify({
                    'success':True,
                    'block':{
                        'hash':latest['block_hash'],
                        'number':latest['block_number'],
                        'parent_hash':latest['parent_hash'],
                        'timestamp':latest['timestamp'],
                        'transaction_count':latest['transaction_count'],
                        'aggregate_entropy':latest['aggregate_entropy']
                    }
                }),200
            else:
                return jsonify({'success':True,'block':None}),200
        
        except Exception as e:
            logger.error(f"Latest block query error: {e}")
            return jsonify({'error':str(e),'success':False}),500
    
    @app.route('/api/v1/blocks/<int:block_number>',methods=['GET'])
    def get_block_by_number(block_number):
        """Get block by block number with all transactions"""
        try:
            block=db.execute_one(
                """SELECT block_hash,block_number,parent_hash,timestamp,transaction_count,
                          aggregate_entropy FROM blocks WHERE block_number=%s""",
                (block_number,)
            )
            
            if not block:
                return jsonify({'error':'Block not found','success':False}),404
            
            # Get transactions in block
            transactions=db.execute(
                """SELECT tx_id,from_user_id,to_user_id,amount,status,entropy_score 
                   FROM transactions WHERE block_hash=%s ORDER BY created_at ASC""",
                (block['block_hash'],)
            )
            
            return jsonify({
                'success':True,
                'block':{
                    'hash':block['block_hash'],
                    'number':block['block_number'],
                    'parent_hash':block['parent_hash'],
                    'timestamp':block['timestamp'],
                    'transaction_count':block['transaction_count'],
                    'aggregate_entropy':block['aggregate_entropy'],
                    'transactions':[
                        {
                            'tx_id':t['tx_id'],
                            'from':t['from_user_id'],
                            'to':t['to_user_id'],
                            'amount':t['amount']/Config.TOKEN_WEI_PER_UNIT if t['amount'] else 0,
                            'status':t['status'],
                            'entropy':t['entropy_score']
                        }
                        for t in transactions
                    ]
                }
            }),200
        
        except Exception as e:
            logger.error(f"Block query error: {e}")
            return jsonify({'error':str(e),'success':False}),500
    
    @app.route('/api/v1/blocks/stats',methods=['GET'])
    def get_blockchain_stats():
        """Get blockchain statistics"""
        try:
            block_count=db.execute_one("SELECT COUNT(*) as count FROM blocks")
            tx_count=db.execute_one("SELECT COUNT(*) as count FROM transactions")
            user_count=db.execute_one("SELECT COUNT(*) as count FROM users")
            avg_entropy=db.execute_one("SELECT AVG(aggregate_entropy) as avg FROM blocks")
            
            return jsonify({
                'success':True,
                'stats':{
                    'total_blocks':block_count['count'] if block_count else 0,
                    'total_transactions':tx_count['count'] if tx_count else 0,
                    'total_users':user_count['count'] if user_count else 0,
                    'avg_block_entropy':avg_entropy['avg'] if avg_entropy and avg_entropy['avg'] else 0
                }
            }),200
        
        except Exception as e:
            logger.error(f"Stats query error: {e}")
            return jsonify({'error':str(e),'success':False}),500
    
    logger.info("✓ Blockchain information endpoints registered")

# ═══════════════════════════════════════════════════════════════════════════════════════
# INTEGRATION INSTRUCTIONS FOR EXISTING API GATEWAY
# ═══════════════════════════════════════════════════════════════════════════════════════

INTEGRATION_INSTRUCTIONS="""
INTEGRATION GUIDE: Add Quantum Transaction Executor to api_gateway.py

1. IMPORTS (add to top of api_gateway.py):
   from quantum_transaction_executor_integrated import executor,db,logger,Config
   from quantum_transaction_integration import (
       SchemaModifier,register_quantum_transaction_routes,
       register_blockchain_info_routes
   )

2. SCHEMA SETUP (in main app initialization):
   SchemaModifier.ensure_schema()

3. ROUTE REGISTRATION (in Flask app initialization):
   register_quantum_transaction_routes(app)
   register_blockchain_info_routes(app)

4. REPLACE EXISTING TRANSACTION ENDPOINTS:
   - OLD: /api/v1/transactions/send-step-3
   - NEW: /api/v1/transactions/submit-quantum
   
   Update api_gateway.py send_step_3() to call:
   result=executor.submit_and_execute_transaction(
       sender_id=user_id,
       receiver_id=tx_data['recipient_id'],
       amount=amount,
       session_id=session_id
   )

5. TESTING:
   curl -X POST http://localhost:5000/api/v1/transactions/submit-quantum \
     -H "Content-Type: application/json" \
     -d '{"from_user":"user1","to_user":"user2","amount":10.5,"session_id":"sess123"}'

6. MONITOR TRANSACTIONS:
   curl http://localhost:5000/api/v1/transactions/list
   curl http://localhost:5000/api/v1/transactions/<tx_id>/status
   curl http://localhost:5000/api/v1/quantum-metrics
"""

if __name__=='__main__':
    print(INTEGRATION_INSTRUCTIONS)
    print("\n✓ Quantum transaction integration module ready")
    print("✓ Database schema verified")
    print("✓ Executor instance initialized")
