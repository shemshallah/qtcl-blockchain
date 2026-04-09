
/**
 * QuantumViolationWorker: Real-time Mermin Inequality Processor
 * Analyzes oracle measurements to detect quantum violations of classical bounds.
 */
self.onmessage = function(e) {
    const { type, data } = e.data;
    if (type !== 'UPDATE_MERMIN') return;

    const mData = data || {};
    const M = parseFloat(mData.M_value || mData.M || mData.mermin_value || 0);
    const verdict = mData.verdict || mData.interpretation || (M > 2.0 ? 'Quantum Violation Detected' : 'Classical Bound');
    const cycle = mData.cycle || 0;
    const angles = mData.angle_degrees || [];
    const labels = mData.angle_labels || [];

    // We perform a "Sanity Check" on the violation
    // M > 2 is the Bell/Mermin limit. M > 3 is strong GHZ-state evidence.
    let severity = 'normal';
    if (M > 3.0) severity = 'strong';
    else if (M <= 2.0) severity = 'classical';

    self.postMessage({
        type: 'MERMIN_RESULT',
        M,
        verdict,
        cycle,
        angles,
        labels,
        severity
    });
};
