-- Migration: Create quantum_metrics table for storing quantum statistics
-- Date: 2026-02-23
-- Purpose: Store all quantum-stats command output to database for persistence and analysis

CREATE TABLE IF NOT EXISTS quantum_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    engine VARCHAR(50) DEFAULT 'QTCL-QE v8.0',
    
    -- Heartbeat metrics
    heartbeat_running BOOLEAN DEFAULT FALSE,
    heartbeat_pulse_count INTEGER DEFAULT 0,
    heartbeat_frequency_hz FLOAT DEFAULT 1.0,
    
    -- Lattice metrics
    lattice_operations INTEGER DEFAULT 0,
    lattice_tx_processed INTEGER DEFAULT 0,
    
    -- Neural network metrics
    neural_convergence VARCHAR(50) DEFAULT 'unknown',
    neural_iterations INTEGER DEFAULT 0,
    
    -- W-state metrics
    w_state_coherence_avg FLOAT DEFAULT 0.0,
    w_state_fidelity_avg FLOAT DEFAULT 0.0,
    w_state_entanglement FLOAT DEFAULT 0.0,
    w_state_superposition_count INTEGER DEFAULT 5,
    w_state_tx_validations INTEGER DEFAULT 0,
    
    -- Noise bath metrics
    noise_kappa FLOAT DEFAULT 0.08,
    noise_fidelity_preservation FLOAT DEFAULT 0.99,
    noise_decoherence_events INTEGER DEFAULT 0,
    noise_non_markovian_order INTEGER DEFAULT 5,
    
    -- v8 Revival metrics
    v8_initialized BOOLEAN DEFAULT FALSE,
    v8_total_pulses INTEGER DEFAULT 0,
    v8_floor_violations INTEGER DEFAULT 0,
    v8_maintainer_hz FLOAT DEFAULT 0.0,
    v8_maintainer_running BOOLEAN DEFAULT FALSE,
    v8_coherence_floor FLOAT DEFAULT 0.89,
    v8_w_state_target FLOAT DEFAULT 0.9997,
    
    -- Bell boundary metrics
    bell_quantum_fraction FLOAT DEFAULT 0.0,
    bell_chsh_violations INTEGER DEFAULT 0,
    bell_boundary_kappa_est FLOAT,
    bell_s_chsh_mean FLOAT DEFAULT 0.0,
    
    -- MI trend metrics
    mi_trend_slope FLOAT DEFAULT 0.0,
    mi_trend_mean FLOAT DEFAULT 0.0,
    mi_trend_status VARCHAR(50) DEFAULT 'insufficient_data',
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_quantum_metrics_timestamp 
    ON quantum_metrics(timestamp DESC);

-- Create index for efficient latest metric queries
CREATE INDEX IF NOT EXISTS idx_quantum_metrics_created 
    ON quantum_metrics(created_at DESC);
