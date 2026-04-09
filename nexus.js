/**
 * StreamNexus: Worker Orchestrator
 */
class StreamNexus {
    constructor(endpoint) {
        this.endpoint = endpoint;
        this.workers = new Map();
    }

    registerWorker(id, worker) {
        this.workers.set(id, worker);
        console.log('[Nexus] Registered:', id);
    }

    connect() {
        console.log('[Nexus] Connecting to:', this.endpoint);
        
        this.pollInterval = setInterval(async () => {
            try {
                const res = await fetch(this.endpoint);
                const text = await res.text();
                const match = text.match(/data:\s*(\{.*\})/);
                if (match) {
                    const data = JSON.parse(match[1]);
                    this.route(data.result || {});
                }
            } catch (e) {
                console.log('[Nexus] Poll error:', e.message);
            }
        }, 500);
    }

    route(result) {
        // Matrix Worker
        if (result.density_matrix_hex && this.workers.has('matrix')) {
            this.workers.get('matrix').postMessage({
                type: 'UPDATE_MATRIX',
                hex: result.density_matrix_hex,
                dim: result.dm_dim || 64,
                fidelity: result.w_state_fidelity,
                purity: result.purity
            });
        }
        // Peer Worker
        if (result.peers && this.workers.has('peers')) {
            this.workers.get('peers').postMessage({
                type: 'UPDATE_PEERS',
                peers: result.peers,
                status: result.status || 'stable'
            });
        }
        // Transaction Worker
        if (result.transactions && this.workers.has('txs')) {
            this.workers.get('txs').postMessage({
                type: 'UPDATE_TXS',
                txs: result.transactions,
                block: result.latest_block || null
            });
        }
    }
}