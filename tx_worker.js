
/**
 * TransactionWorker: Ledger Transaction Stream Processor
 * Processes high-frequency transaction updates and block inclusions.
 */
self.onmessage = function(e) {
    const { type, txs, block } = e.data;
    if (type !== 'UPDATE_TXS') return;

    const transactions = txs || [];
    
    // Sort transactions by timestamp descending
    const processedTxs = transactions.map(tx => ({
        id: tx.txid || tx.id,
        from: tx.sender || 'Unknown',
        to: tx.receiver || 'Unknown',
        amount: tx.amount || 0,
        fee: tx.fee || 0,
        time: tx.timestamp || Date.now(),
        status: tx.status || 'pending'
    })).sort((a, b) => b.time - a.time);

    self.postMessage({
        type: 'TX_STATE_READY',
        transactions: processedTxs,
        latestBlock: block || null,
        count: processedTxs.length,
        timestamp: Date.now()
    });
};
