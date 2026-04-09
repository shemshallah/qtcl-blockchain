
/**
 * QuantumMatrixWorker: High-Performance Density Matrix Processor
 * Handles 64x64 Complex128 matrices with zero-copy transferables.
 */
self.onmessage = function(e) {
    const { type, hex, dim, fidelity, purity } = e.data;
    if (type !== 'UPDATE_MATRIX') return;

    const N = dim || 64;
    const n2 = N * N;
    
    // Fast hex to float64 conversion
    const re = new Float64Array(n2);
    const im = new Float64Array(n2);
    
    const HEX_MAP = new Uint8Array(256);
    for (let i = 0; i < 10; i++) HEX_MAP[48 + i] = i;
    for (let i = 0; i < 6; i++) { HEX_MAP[65 + i] = 10 + i; HEX_MAP[97 + i] = 10 + i; }

    const dv = new DataView(new ArrayBuffer(8));
    const u8 = new Uint8Array(dv.buffer);

    for (let i = 0; i < n2; i++) {
        const offset = i * 32;
        if (offset + 31 >= hex.length) break;

        // Real part
        for (let b = 0; b < 8; b++) {
            u8[b] = (HEX_MAP[hex.charCodeAt(offset + b * 2)] << 4) | HEX_MAP[hex.charCodeAt(offset + b * 2 + 1)];
        }
        re[i] = dv.getFloat64(0, true);

        // Imaginary part
        for (let b = 0; b < 8; b++) {
            u8[b] = (HEX_MAP[hex.charCodeAt(offset + 16 + b * 2)] << 4) | HEX_MAP[hex.charCodeAt(offset + 16 + b * 2 + 1)];
        }
        im[i] = dv.getFloat64(0, true);
    }

    // Compute Matrix Metrics and Generate Pixel Buffer
    const mag = new Float64Array(n2);
    const ph = new Float64Array(n2);
    let maxMag = 0;

    for (let i = 0; i < n2; i++) {
        const m = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
        mag[i] = m;
        ph[i] = Math.atan2(im[i], re[i]);
        if (m > maxMag) maxMag = m;
    }

    // Generate Color Map (RGB)
    // We create a Uint8ClampedArray for direct canvas manipulation
    const pixelBuffer = new Uint8ClampedArray(n2 * 4);
    
    for (let i = 0; i < n2; i++) {
        const rgb = phaseToRgb(ph[i], mag[i], maxMag);
        const p = i * 4;
        pixelBuffer[p] = rgb[0];
        pixelBuffer[p + 1] = rgb[1];
        pixelBuffer[p + 2] = rgb[2];
        pixelBuffer[p + 3] = 255;
    }

    // Transfer results back to main thread
    self.postMessage({
        type: 'MATRIX_RENDER_READY',
        pixelBuffer,
        fidelity,
        purity,
        maxMag,
        N
    }, [pixelBuffer.buffer]);
};

function phaseToRgb(phase, mag, maxMag) {
    const normalizedPhase = (phase + Math.PI) / (2 * Math.PI);
    const hue = 160 + normalizedPhase * 160;
    const lightness = Math.min(0.9, 0.1 + (mag / (maxMag || 1)) * 0.8);
    const saturation = 0.8;

    const c = (1 - Math.abs(2 * lightness - 1)) * saturation;
    const x = c * (1 - Math.abs((hue / 60) % 2 - 1));
    const m = lightness - c / 2;

    let r, g, b;
    if (hue < 60) { r = c; g = x; b = 0; }
    else if (hue < 120) { r = x; g = c; b = 0; }
    else if (hue < 180) { r = 0; g = c; b = x; }
    else if (hue < 240) { r = 0; g = x; b = c; }
    else if (hue < 300) { r = x; g = 0; b = c; }
    else { r = c; g = 0; b = x; }

    return [
        (r + m) * 255 | 0,
        (g + m) * 255 | 0,
        (b + m) * 255 | 0
    ];
}
