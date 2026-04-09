/**
 * StreamNexus: The High-Performance Event Orchestrator
 * Designed for Zero-Lag Quantum Data Distribution
 */
class StreamNexus {
    constructor(endpoint) {
        this.endpoint = endpoint;
        this.eventSource = null;
        this.workers = new Map();
        this.reconnectInterval = 1000;
        this.maxReconnectInterval = 30000;
        this.currentReconnectDelay = this.reconnectInterval;
        this.pollInterval = null;
    }

    registerWorker(id, worker) {
        this.workers.set(id, worker);
    }

    connect() {
        console.log(`[StreamNexus] Initializing Quantum Stream: ${this.endpoint}`);
        
        // Try EventSource (SSE) first
        try {
            this.eventSource = new EventSource(this.endpoint);
            this.eventSource.onopen = () => {
                console.log('[StreamNexus] SSE Stream Synchronized.');
                this.currentReconnectDelay = this.reconnectInterval;
            };
            this.eventSource.onmessage = (e) => this.route(e.data);
            this.eventSource.onerror = (err) => {
                console.warn('[StreamNexus] SSE Divergence. Switching to RPC Polling...');
                this.disconnect();
                this.startPolling();
            };
        } catch (e) {
            console.error('[StreamNexus] SSE Connection Failed:', e);
            this.startPolling();
        }
    }

    startPolling() {
        if (this.pollInterval) return;
        console.log(`[StreamNexus] Transitioning to Aggressive RPC Polling: ${this.endpoint}`);
        this.pollInterval = setInterval(async () => {
            try {
                const res = await fetch(this.endpoint);
                if (res.ok) {
                    const data = await res.json();
                    // The server returns the result inside a 'result' key for RPC snapshots
                    const wrappedData = data.result ? JSON.stringify({ result: data.result }) : JSON.stringify(data);
                    this.route(wrappedData);
                }
            } catch (e) {
                console.debug('[StreamNexus] Poll error:', e);
            }
        }, 300); // 300ms as specified in qtcl_client.py
    }

    disconnect() {
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
    }

    route(rawData) {
        try {
            const decoded = JSON.parse(rawData);
            const result = decoded.result;
            if (!result) return;

            // Matrix Routing (High Volume)
            if (result.density_matrix_hex && this.workers.has('matrix')) {
                this.workers.get('matrix').postMessage({
                    type: 'UPDATE_MATRIX',
                    hex: result.density_matrix_hex,
                    dim: result.dm_dim || 64,
                    fidelity: result.w_state_fidelity,
                    purity: result.purity
                });
            }

            // Mermin Routing - Handle both 'mermin_test' and 'mermin_data'
            const merminData = result.mermin_test || result.mermin_data;
            if (merminData && this.workers.has('mermin')) {
                this.workers.get('mermin').postMessage({
                    type: 'UPDATE_MERMIN',
                    data: merminData
                });
            }

            // Chain State Routing - Handle a variety of block field shapes
            const blockField = (result.block_field || result.block_height !== undefined) ? {
                height: result.block_height || result.height,
                pq_curr: result.pq_curr,
                pq_last: result.pq_last
            } : null;

            if (blockField || result.chain_stats) {
                if (this.workers.has('chain')) {
                    this.workers.get('chain').postMessage({
                        type: 'UPDATE_CHAIN',
                        blockField: blockField,
                        stats: result.chain_stats || {}
                    });
                }
            }

            // Telemetry Routing
            if (result.lattice_refresh_counter !== undefined && this.workers.has('telemetry')) {
                this.workers.get('telemetry').postMessage({
                    type: 'UPDATE_METRICS',
                    cycle: result.lattice_refresh_counter,
                    timestamp: Date.now()
                });
            }

            // Peer Routing
            if (result.peers && this.workers.has('peers')) {
                this.workers.get('peers').postMessage({
                    type: 'UPDATE_PEERS',
                    peers: result.peers,
                    status: result.network_status
                });
            }

            // Transaction Routing
            if (result.transactions && this.workers.has('tx')) {
                this.workers.get('tx').postMessage({
                    type: 'UPDATE_TXS',
                    txs: result.transactions,
                    block: result.latest_block
                });
            }

        } catch (e) {
            console.warn('[StreamNexus] Packet Corruption:', e);
        }
    }
}