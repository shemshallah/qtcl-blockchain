
/**
 * TelemetryWorker: Lattice and Coherence Metrics Tracker
 * Monitors timing, refresh cycles, and coherence drift.
 */
self.onmessage = function(e) {
    const { type, cycle, timestamp } = e.data;
    if (type !== 'UPDATE_METRICS') return;

    // We track the delta between cycles to detect jitter in the quantum oracle
    const now = timestamp || Date.now();
    
    // Simple state for this worker
    if (!self.lastCycleTime) self.lastCycleTime = now;
    const delta = now - self.lastCycleTime;
    self.lastCycleTime = now;

    self.postMessage({
        type: 'TELEMETRY_READY',
        cycle,
        jitter: delta,
        timestamp: now
    });
};
