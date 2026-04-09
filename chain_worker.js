
/**
 * ChainStateWorker: Blockchain Telemetry Processor
 * Maintains a lightweight state machine for Blocks, Transactions, and Peers.
 */
self.onmessage = function(e) {
    const { type, blockField, stats } = e.data;
    if (type !== 'UPDATE_CHAIN') return;

    const bf = blockField || {};
    const s = stats || {};

    // Extract critical chain metrics
    const state = {
        pq_curr: bf.pq_curr || 0,
        pq_last: bf.pq_last || 0,
        pq0: bf.pq0 || 0,
        block_height: bf.height || 0,
        tx_count: s.tx_count || 0,
        peer_count: s.peer_count || 0,
        network_status: s.status || 'stable',
        timestamp: Date.now()
    };

    // Only emit the processed state to the UI
    self.postMessage({
        type: 'CHAIN_STATE_READY',
        state
    });
};
