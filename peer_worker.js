
/**
 * PeerNetworkWorker: P2P Connectivity and Discovery Processor
 * Handles peer lists, heartbeats, and DHT-based node discovery.
 */
self.onmessage = function(e) {
    const { type, peers, status, heartbeat } = e.data;
    if (type !== 'UPDATE_PEERS') return;

    const peerList = peers || [];
    const systemStatus = status || 'stable';
    
    // Process peer data: extract IPs, roles, and health
    const processedPeers = peerList.map(p => ({
        id: p.id || 'unknown',
        addr: p.external_addr || p.host,
        port: p.port || 9091,
        role: p.role || 'miner',
        verified: !!p.verified,
        latency: p.latency || 0,
        lastSeen: p.last_seen || new Date().toISOString()
    }));

    self.postMessage({
        type: 'PEER_STATE_READY',
        peers: processedPeers,
        status: systemStatus,
        count: processedPeers.length,
        timestamp: Date.now()
    });
};
