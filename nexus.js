
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
    }

    registerWorker(id, worker) {
        this.workers.set(id, worker);
    }

    connect() {
        console.log(`[StreamNexus] Initializing Quantum Stream: ${this.endpoint}`);
        this.eventSource = new EventSource(this.endpoint);

        this.eventSource.onopen = () => {
            console.log('[StreamNexus] Stream Synchronized.');
            this.currentReconnectDelay = this.reconnectInterval;
        };

        this.eventSource.onmessage = (e) => {
            this.route(e.data);
        };

        this.eventSource.onerror = (err) => {
            console.error('[StreamNexus] Stream Divergence Detected. Attempting Re-coherence...');
            this.disconnect();
            setTimeout(() => {
                this.currentReconnectDelay = Math.min(this.currentReconnectDelay * 2, this.maxReconnectInterval);
                this.connect();
            }, this.currentReconnectDelay);
        };
    }

    disconnect() {
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
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

            // Mermin Routing
            if (result.mermin_data && this.workers.has('mermin')) {
                this.workers.get('mermin').postMessage({
                    type: 'UPDATE_MERMIN',
                    data: result.mermin_data
                });
            }

            // Chain State Routing
            if (result.block_field || result.chain_stats) {
                if (this.workers.has('chain')) {
                    this.workers.get('chain').postMessage({
                        type: 'UPDATE_CHAIN',
                        blockField: result.block_field,
                        stats: result.chain_stats
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

        } catch (e) {
            console.warn('[StreamNexus] Packet Corruption:', e);
        }
    }
}
