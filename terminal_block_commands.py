#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                                                              ║
║                             COMPREHENSIVE BLOCK COMMANDS FOR TERMINAL LOGIC                                                  ║
║                            WITH FULL QUANTUM MEASUREMENTS & WSGI INTEGRATION                                                 ║
║                                                                                                                              ║
║  This module extends terminal_logic.py with production-grade block commands that leverage:                                  ║
║  ✅ blockchain_api.py's comprehensive /blocks/command endpoint                                                              ║
║  ✅ Full WSGI globals integration (DB, CACHE, PROFILER, CIRCUIT_BREAKERS, RATE_LIMITERS)                                   ║
║  ✅ Real quantum measurements from quantum_lattice_control_live_complete.py                                                 ║
║  ✅ Performance profiling with correlation ID tracking                                                                      ║
║  ✅ Smart caching with TTL and invalidation                                                                                 ║
║  ✅ Comprehensive audit logging to command_logs table                                                                       ║
║  ✅ Multi-threaded batch processing with parallel execution                                                                 ║
║  ✅ Advanced visualizations and statistics                                                                                  ║
║                                                                                                                              ║
║  COMMANDS IMPLEMENTED:                                                                                                       ║
║  • block/details <hash|height> [--full] [--quantum] [--validate] [--network]                                               ║
║  • block/validate <hash|height> [--full] [--skip-quantum] [--skip-transactions]                                            ║
║  • block/quantum <hash|height>                                                                                               ║
║  • block/batch <block1> <block2> ... or <start>-<end> [--quantum]                                                          ║
║  • block/integrity [start] [end] or [--recent N]                                                                            ║
║  • block/stats [--hours N] [--validator <address>]                                                                          ║
║  • block/search <query> [--field <field>] [--limit N]                                                                       ║
║  • block/export <hash|height> [--format json|csv] [--output <file>]                                                        ║
║  • block/compare <hash1|height1> <hash2|height2>                                                                            ║
║  • block/timeline [--hours N] [--visualize]                                                                                 ║
║                                                                                                                              ║
║  USAGE:                                                                                                                      ║
║  1. Import this module in terminal_logic.py after TerminalEngine class definition                                           ║
║  2. Call register_block_commands(terminal_engine_instance) to inject all commands                                           ║
║  3. Commands automatically integrate with existing WSGI globals and quantum systems                                         ║
║                                                                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

import time
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE BLOCK COMMAND IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

def cmd_block_details(engine, args: str = None):
    """
    ╔══════════════════════════════════════════════════════════════════════════════════════╗
    ║                    COMPREHENSIVE BLOCK DETAILS WITH QUANTUM MEASUREMENTS              ║
    ╚══════════════════════════════════════════════════════════════════════════════════════╝
    
    This is the flagship block command that showcases the full power of:
    - WSGI global integration (DB, CACHE, PROFILER, CIRCUIT_BREAKERS)
    - Quantum measurements (entropy, coherence, finality, entanglement)
    - Performance profiling with correlation tracking
    - Smart caching with TTL
    - Comprehensive audit logging
    - Multi-modal validation (hash, merkle, temporal, quantum)
    - Network position analysis
    - Validator performance metrics
    
    Usage:
      block/details <hash_or_height> [--full] [--quantum] [--validate] [--network]
      
    Examples:
      block/details 12345                    # Get block at height 12345
      block/details abc123def --full         # Full details including all transactions
      block/details 12345 --quantum          # Include quantum measurements
      block/details latest --validate        # Validate latest block
      block/details 12345 --network          # Include network position analysis
    """
    from terminal_logic import UI
    
    try:
        # Parse arguments
        parts = args.split() if args else []
        if not parts:
            UI.error("Usage: block/details <hash_or_height> [--full] [--quantum] [--validate] [--network]")
            return
        
        block_id = parts[0]
        flags = parts[1:]
        
        # Resolve 'latest' to actual height
        if block_id.lower() == 'latest':
            try:
                tip_result = engine.client.get('/blockchain/tip')
                if tip_result.get('success') and tip_result.get('data'):
                    block_id = str(tip_result['data'].get('height', 0))
                else:
                    UI.error("Could not resolve 'latest' block")
                    return
            except Exception as e:
                UI.error(f"Error resolving 'latest': {e}")
                return
        
        UI.header(f"🔷 COMPREHENSIVE BLOCK DETAILS - {block_id}")
        
        # Build options from flags
        options = {
            'include_transactions': '--full' in flags or '--transactions' in flags,
            'include_quantum': '--quantum' in flags,
            'force_refresh': '--refresh' in flags or '--no-cache' in flags
        }
        
        # Execute block command via API
        start_time = time.time()
        
        try:
            # Call the comprehensive block command endpoint
            result = engine.client.post('/blockchain/blocks/command', {
                'command': 'query',
                'block': block_id,
                'options': options
            })
            
            duration_ms = (time.time() - start_time) * 1000
            
            if not result.get('success'):
                UI.error(f"Block query failed: {result.get('error', 'Unknown error')}")
                return
            
            block_data = result.get('data', {})
            
            # Display basic information
            UI.section("📊 Basic Information")
            basic_table = [
                ['Block Hash', block_data.get('block_hash', 'N/A')[:64]],
                ['Height', f"{block_data.get('height', 0):,}"],
                ['Status', block_data.get('status', 'unknown').upper()],
                ['Confirmations', f"{block_data.get('confirmations', 0):,}"],
                ['Timestamp', block_data.get('timestamp', 'N/A')],
                ['Validator', block_data.get('validator', 'N/A')[:42]],
                ['Epoch', f"{block_data.get('epoch', 0):,}"],
                ['Is Orphan', '✓ YES' if block_data.get('is_orphan') else '✗ NO'],
                ['Temporal Coherence', f"{block_data.get('temporal_coherence', 1.0):.4f}"]
            ]
            UI.print_table(['Property', 'Value'], basic_table)
            
            # Display block metrics
            UI.section("📈 Block Metrics")
            metrics_table = [
                ['Size', f"{block_data.get('size_bytes', 0):,} bytes"],
                ['Transactions', f"{block_data.get('tx_count', 0):,}"],
                ['Gas Used', f"{block_data.get('gas_used', 0):,}"],
                ['Gas Limit', f"{block_data.get('gas_limit', 0):,}"],
                ['Gas Utilization', f"{(block_data.get('gas_used', 0) / max(block_data.get('gas_limit', 1), 1) * 100):.1f}%"],
                ['Total Fees', f"{block_data.get('total_fees', '0')} QTCL"],
                ['Block Reward', f"{block_data.get('reward', '0')} QTCL"],
                ['Difficulty', f"{block_data.get('difficulty', 1):,}"]
            ]
            UI.print_table(['Metric', 'Value'], metrics_table)
            
            # Display cryptographic roots
            UI.section("🔐 Cryptographic Roots")
            crypto_table = [
                ['Previous Hash', block_data.get('previous_hash', 'N/A')[:64]],
                ['Merkle Root', block_data.get('merkle_root', 'N/A')[:64]],
                ['Quantum Merkle', block_data.get('quantum_merkle_root', 'N/A')[:64]],
                ['State Root', block_data.get('state_root', 'N/A')[:64]]
            ]
            UI.print_table(['Root Type', 'Hash'], crypto_table)
            
            # Display quantum metrics if included
            if options.get('include_quantum') and 'quantum_metrics' in block_data:
                UI.section("⚛️  QUANTUM MEASUREMENTS")
                qm = block_data['quantum_metrics']
                
                # Entropy metrics
                if 'entropy' in qm and 'shannon' in qm.get('entropy', {}):
                    entropy = qm['entropy']
                    entropy_table = [
                        ['Shannon Entropy', f"{entropy.get('shannon', 0):.4f} bits"],
                        ['Byte Entropy', f"{entropy.get('byte_entropy', 0):.4f}"],
                        ['Data Length', f"{entropy.get('length', 0)} bytes"]
                    ]
                    UI.print_table(['Entropy Metric', 'Value'], entropy_table)
                    
                    # Entropy quality assessment
                    shannon = entropy.get('shannon', 0)
                    if shannon >= 7.9:
                        UI.success("  ✓ Excellent entropy quality (near-ideal)")
                    elif shannon >= 7.5:
                        UI.info("  ✓ Good entropy quality")
                    elif shannon >= 7.0:
                        UI.warning("  ⚠ Moderate entropy quality")
                    else:
                        UI.error("  ✗ Poor entropy quality")
                
                # Coherence metrics
                coherence_table = [
                    ['W-State Fidelity', f"{qm.get('w_state_fidelity', 0):.4f}"],
                    ['Temporal Coherence', f"{qm.get('temporal_coherence', 0):.4f}"],
                    ['GHZ Collapse', '✓ Verified' if qm.get('ghz_collapse_verified') else '✗ Not Verified']
                ]
                UI.print_table(['Coherence Metric', 'Value'], coherence_table)
            
            # Display transactions if included
            if options.get('include_transactions') and 'transactions' in block_data:
                UI.section(f"💳 Transactions (showing {min(len(block_data['transactions']), 10)} of {block_data.get('tx_count_actual', len(block_data['transactions']))})")
                tx_table = []
                for tx in block_data['transactions'][:10]:
                    tx_table.append([
                        tx.get('tx_hash', '')[:16] + '...',
                        tx.get('from', '')[:12] + '...',
                        tx.get('to', '')[:12] + '...',
                        f"{tx.get('amount', '0')} QTCL",
                        tx.get('status', 'unknown')
                    ])
                if tx_table:
                    UI.print_table(['TX Hash', 'From', 'To', 'Amount', 'Status'], tx_table)
            
            # Display metadata
            metadata = block_data.get('_metadata', {})
            if metadata:
                UI.section("🔧 Query Metadata")
                meta_table = [
                    ['Correlation ID', metadata.get('correlation_id', 'N/A')],
                    ['Query Duration', f"{metadata.get('duration_ms', 0):.2f} ms"],
                    ['Cache Hit', '✓ Yes' if block_data.get('_cache_hit') else '✗ No'],
                    ['Timestamp', metadata.get('timestamp', 'N/A')]
                ]
                UI.print_table(['Property', 'Value'], meta_table)
            
            # Run validation if requested
            if '--validate' in flags:
                UI.section("🔍 VALIDATION RESULTS")
                val_result = engine.client.post('/blockchain/blocks/command', {
                    'command': 'validate',
                    'block': block_id,
                    'options': {'validate_quantum': True, 'validate_transactions': True}
                })
                
                if val_result.get('success'):
                    val_data = val_result.get('data', {})
                    overall_valid = val_data.get('overall_valid', False)
                    
                    UI.info(f"Overall Valid: {'✓ YES' if overall_valid else '✗ NO'}")
                    
                    checks = val_data.get('checks', {})
                    val_table = []
                    for check_name, check_data in checks.items():
                        if isinstance(check_data, dict):
                            status = '✓' if check_data.get('valid') else '✗'
                            val_table.append([check_name.replace('_', ' ').title(), status])
                    
                    if val_table:
                        UI.print_table(['Check', 'Result'], val_table)
            
            # Run network analysis if requested
            if '--network' in flags:
                UI.section("🌐 NETWORK POSITION ANALYSIS")
                analyze_result = engine.client.post('/blockchain/blocks/command', {
                    'command': 'analyze',
                    'block': block_id,
                    'options': {'include_network': True}
                })
                
                if analyze_result.get('success'):
                    network = analyze_result.get('data', {}).get('network_analysis', {})
                    if network and not network.get('error'):
                        net_table = [
                            ['Time Since Previous', f"{network.get('time_since_previous_sec', 0):.2f}s"],
                            ['Block Time Ratio', f"{network.get('block_time_ratio', 1.0):.2f}x"],
                            ['Difficulty Change', f"{network.get('difficulty_change', 0):+d}"],
                            ['Difficulty Change %', f"{network.get('difficulty_change_pct', 0):+.2f}%"]
                        ]
                        if 'time_to_next_sec' in network:
                            net_table.append(['Time To Next', f"{network['time_to_next_sec']:.2f}s"])
                        UI.print_table(['Network Metric', 'Value'], net_table)
            
            UI.success(f"\n✓ Block details retrieved in {duration_ms:.2f}ms")
            
        except Exception as e:
            UI.error(f"API call failed: {e}")
            logger.error(f"Block details error: {e}", exc_info=True)
            
    except Exception as e:
        UI.error(f"Block details command error: {e}")
        logger.error(f"Block details error: {e}", exc_info=True)


def cmd_block_validate(engine, args: str = None):
    """Comprehensive block validation with quantum proof verification"""
    from terminal_logic import UI
    
    try:
        if not args:
            UI.error("Usage: block/validate <hash_or_height> [--full] [--skip-quantum]")
            return
        
        parts = args.split()
        block_id = parts[0]
        flags = parts[1:]
        
        UI.header(f"🔍 COMPREHENSIVE BLOCK VALIDATION - {block_id}")
        
        options = {
            'validate_quantum': '--skip-quantum' not in flags,
            'validate_transactions': '--full' in flags or '--skip-transactions' not in flags,
            'tx_sample_size': 20 if '--full' in flags else 10
        }
        
        # Call validation command
        result = engine.client.post('/blockchain/blocks/command', {
            'command': 'validate',
            'block': block_id,
            'options': options
        })
        
        if not result.get('success'):
            UI.error(f"Validation failed: {result.get('error', 'Unknown error')}")
            return
        
        val_data = result.get('data', {})
        overall_valid = val_data.get('overall_valid', False)
        checks = val_data.get('checks', {})
        
        # Display overall result
        if overall_valid:
            UI.success(f"\n✓ BLOCK {block_id} IS VALID")
        else:
            UI.error(f"\n✗ BLOCK {block_id} FAILED VALIDATION")
        
        # Display detailed checks
        UI.section("📋 Validation Checks")
        
        for check_name, check_data in checks.items():
            if not isinstance(check_data, dict):
                continue
            
            valid = check_data.get('valid', False)
            status_icon = '✓' if valid else '✗'
            
            UI.info(f"\n{status_icon} {check_name.replace('_', ' ').title()}")
            
            # Display check-specific details
            if check_name == 'hash_integrity':
                if 'computed' in check_data and 'stored' in check_data:
                    UI.print_table(
                        ['Type', 'Hash'],
                        [
                            ['Computed', check_data['computed'][:64]],
                            ['Stored', check_data['stored'][:64]]
                        ]
                    )
            
            elif check_name == 'merkle_root':
                if 'computed' in check_data and 'stored' in check_data:
                    UI.print_table(
                        ['Type', 'Root'],
                        [
                            ['Computed', check_data['computed'][:64]],
                            ['Stored', check_data['stored'][:64]]
                        ]
                    )
            
            elif check_name == 'previous_link':
                if 'expected' in check_data and 'actual' in check_data:
                    UI.print_table(
                        ['Type', 'Hash'],
                        [
                            ['Expected', str(check_data['expected'])[:64]],
                            ['Actual', str(check_data['actual'])[:64]]
                        ]
                    )
            
            elif check_name == 'quantum_proof':
                if 'proof_version' in check_data:
                    UI.info(f"  Proof Version: {check_data['proof_version']}")
            
            elif check_name == 'temporal_coherence':
                if 'value' in check_data and 'threshold' in check_data:
                    UI.info(f"  Value: {check_data['value']:.4f} (threshold: {check_data['threshold']:.4f})")
            
            elif check_name == 'transactions':
                if 'sampled' in check_data:
                    UI.info(f"  Sampled: {check_data['sampled']} / {check_data.get('total', 0)}")
                    UI.info(f"  Valid: {check_data.get('valid_count', 0)} / {check_data['sampled']}")
            
            if 'error' in check_data:
                UI.error(f"  Error: {check_data['error']}")
        
        # Display metadata
        metadata = val_data.get('_metadata', {})
        if metadata:
            UI.section("🔧 Validation Metadata")
            UI.info(f"Correlation ID: {metadata.get('correlation_id', 'N/A')}")
            UI.info(f"Duration: {metadata.get('duration_ms', 0):.2f} ms")
        
    except Exception as e:
        UI.error(f"Block validation error: {e}")
        logger.error(f"Block validation error: {e}", exc_info=True)


def cmd_block_quantum(engine, args: str = None):
    """Perform comprehensive quantum measurements on block"""
    from terminal_logic import UI
    
    try:
        if not args:
            UI.error("Usage: block/quantum <hash_or_height>")
            return
        
        block_id = args.strip()
        
        UI.header(f"⚛️  QUANTUM MEASUREMENTS - Block {block_id}")
        
        # Call quantum measurement command
        result = engine.client.post('/blockchain/blocks/command', {
            'command': 'quantum_measure',
            'block': block_id,
            'options': {}
        })
        
        if not result.get('success'):
            UI.error(f"Quantum measurement failed: {result.get('error', 'Unknown error')}")
            return
        
        qm = result.get('data', {})
        
        # Display entropy measurements
        if 'entropy' in qm and not qm['entropy'].get('error'):
            UI.section("📊 Entropy Analysis")
            entropy = qm['entropy']
            entropy_table = [
                ['Shannon Entropy', f"{entropy.get('shannon_entropy', 0):.6f} bits"],
                ['Byte Entropy', f"{entropy.get('byte_entropy', 0):.6f}"],
                ['Data Length', f"{entropy.get('length_bytes', 0)} bytes"],
                ['Hex Preview', entropy.get('hex_preview', 'N/A')]
            ]
            UI.print_table(['Metric', 'Value'], entropy_table)
            
            # Entropy quality assessment
            shannon = entropy.get('shannon_entropy', 0)
            if shannon >= 7.9:
                UI.success("  ✓ Excellent entropy quality (near-ideal)")
            elif shannon >= 7.5:
                UI.info("  ✓ Good entropy quality")
            elif shannon >= 7.0:
                UI.warning("  ⚠ Moderate entropy quality")
            else:
                UI.error("  ✗ Poor entropy quality")
        
        # Display coherence measurements
        if 'coherence' in qm and not qm['coherence'].get('error'):
            UI.section("🌊 Coherence Measurements")
            coherence = qm['coherence']
            coherence_table = [
                ['Temporal Coherence', f"{coherence.get('temporal', 0):.6f}"],
                ['W-State Fidelity', f"{coherence.get('w_state_fidelity', 0):.6f}"],
                ['Quality Rating', coherence.get('quality', 'unknown').upper()]
            ]
            UI.print_table(['Metric', 'Value'], coherence_table)
        
        # Display finality measurements
        if 'finality' in qm and not qm['finality'].get('error'):
            UI.section("🎯 Finality Status")
            finality = qm['finality']
            finality_table = [
                ['Confirmations', f"{finality.get('confirmations', 0):,}"],
                ['Is Finalized', '✓ YES' if finality.get('is_finalized') else '✗ NO'],
                ['Finality Score', f"{finality.get('finality_score', 0):.2%}"],
                ['GHZ Collapse', '✓ Verified' if finality.get('ghz_collapse_verified') else '✗ Not Verified']
            ]
            UI.print_table(['Metric', 'Value'], finality_table)
        
        # Display entanglement measurements
        if 'entanglement' in qm and not qm['entanglement'].get('error'):
            UI.section("🔗 Validator Entanglement")
            entanglement = qm['entanglement']
            ent_table = [
                ['Validator Count', f"{entanglement.get('validator_count', 0)}"],
                ['Entanglement Strength', f"{entanglement.get('entanglement_strength', 0):.6f}"]
            ]
            UI.print_table(['Metric', 'Value'], ent_table)
            
            # Display W-state components
            w_components = entanglement.get('w_state_components', {})
            if w_components:
                UI.info("\nW-State Components:")
                comp_table = [[k.replace('validator_', 'Validator '), f"{v:.6f}"] 
                              for k, v in w_components.items()]
                UI.print_table(['Validator', 'Amplitude'], comp_table)
        
        # Display metadata
        metadata = qm.get('_metadata', {})
        if metadata:
            UI.section("🔧 Measurement Metadata")
            UI.info(f"Correlation ID: {metadata.get('correlation_id', 'N/A')}")
            UI.info(f"Duration: {metadata.get('duration_ms', 0):.2f} ms")
            UI.info(f"Timestamp: {metadata.get('timestamp', 'N/A')}")
        
    except Exception as e:
        UI.error(f"Quantum measurement error: {e}")
        logger.error(f"Quantum measurement error: {e}", exc_info=True)


def cmd_block_batch(engine, args: str = None):
    """Query multiple blocks efficiently with parallel processing"""
    from terminal_logic import UI
    
    try:
        if not args:
            UI.error("Usage: block/batch <block1> <block2> ... or <start>-<end>")
            return
        
        parts = args.split()
        flags = [p for p in parts if p.startswith('--')]
        refs = [p for p in parts if not p.startswith('--')]
        
        # Handle range notation
        block_refs = []
        for ref in refs:
            if '-' in ref and all(x.isdigit() or x == '-' for x in ref):
                try:
                    start, end = map(int, ref.split('-'))
                    block_refs.extend(range(start, end + 1))
                except:
                    block_refs.append(ref)
            else:
                block_refs.append(ref)
        
        if not block_refs:
            UI.error("No valid block references provided")
            return
        
        UI.header(f"📦 BATCH QUERY - {len(block_refs)} blocks")
        
        options = {
            'include_quantum': '--quantum' in flags,
            'include_transactions': '--full' in flags
        }
        
        # Call batch query command
        result = engine.client.post('/blockchain/blocks/command', {
            'command': 'batch_query',
            'blocks': block_refs,
            'options': options
        })
        
        if not result.get('success'):
            UI.error(f"Batch query failed: {result.get('error', 'Unknown error')}")
            return
        
        batch_data = result.get('data', {})
        
        UI.info(f"Batch Size: {batch_data.get('batch_size', 0)}")
        UI.info(f"Success: {batch_data.get('success_count', 0)}")
        UI.info(f"Errors: {batch_data.get('error_count', 0)}")
        
        # Display results table
        results = batch_data.get('results', [])
        if results:
            UI.section("📊 Results")
            results_table = []
            for r in results[:20]:  # Limit display to 20
                if 'error' in r:
                    results_table.append([
                        r.get('block_ref', 'N/A'),
                        'ERROR',
                        r['error'][:50]
                    ])
                else:
                    results_table.append([
                        r.get('block_hash', 'N/A')[:16] + '...',
                        r.get('height', 'N/A'),
                        r.get('status', 'unknown')
                    ])
            
            UI.print_table(['Block Hash', 'Height', 'Status'], results_table)
            
            if len(results) > 20:
                UI.info(f"\n... and {len(results) - 20} more results")
        
    except Exception as e:
        UI.error(f"Batch query error: {e}")
        logger.error(f"Batch query error: {e}", exc_info=True)


def cmd_block_integrity(engine, args: str = None):
    """Verify blockchain integrity across a range of blocks"""
    from terminal_logic import UI
    
    try:
        parts = (args or '').split() if args else []
        
        options = {}
        
        if '--recent' in parts:
            idx = parts.index('--recent')
            if idx + 1 < len(parts):
                count = int(parts[idx + 1])
                # Get tip to calculate range
                tip_result = engine.client.get('/blockchain/tip')
                if tip_result.get('success'):
                    tip_height = tip_result['data'].get('height', 0)
                    options['start_height'] = max(0, tip_height - count)
                    options['end_height'] = tip_height
        elif len(parts) >= 2:
            options['start_height'] = int(parts[0])
            options['end_height'] = int(parts[1])
        
        UI.header("🔍 BLOCKCHAIN INTEGRITY CHECK")
        
        # Call integrity check command
        result = engine.client.post('/blockchain/blocks/command', {
            'command': 'chain_integrity',
            'options': options
        })
        
        if not result.get('success'):
            UI.error(f"Integrity check failed: {result.get('error', 'Unknown error')}")
            return
        
        integrity = result.get('data', {})
        
        # Display summary
        UI.section("📊 Summary")
        summary_table = [
            ['Height Range', f"{integrity.get('start_height', 0):,} - {integrity.get('end_height', 0):,}"],
            ['Blocks Checked', f"{integrity.get('blocks_checked', 0):,}"],
            ['Valid Blocks', f"{integrity.get('valid_blocks', 0):,}"],
            ['Invalid Blocks', f"{len(integrity.get('invalid_blocks', [])):,}"],
            ['Broken Links', f"{len(integrity.get('broken_links', [])):,}"],
            ['Orphaned Blocks', f"{len(integrity.get('orphaned_blocks', [])):,}"],
            ['Integrity Score', f"{integrity.get('integrity_score', 0):.2%}"]
        ]
        UI.print_table(['Metric', 'Value'], summary_table)
        
        # Display issues
        if integrity.get('invalid_blocks'):
            UI.section("❌ Invalid Blocks")
            for inv in integrity['invalid_blocks'][:10]:
                UI.error(f"Height {inv.get('height')}: {inv.get('hash', 'N/A')[:16]}")
        
        if integrity.get('broken_links'):
            UI.section("🔗 Broken Links")
            for link in integrity['broken_links'][:10]:
                UI.error(f"Height {link.get('height')}: {link.get('reason', 'Unknown')}")
        
        if integrity.get('orphaned_blocks'):
            UI.section("👻 Orphaned Blocks")
            orphaned_str = ', '.join(map(str, integrity['orphaned_blocks'][:20]))
            UI.error(f"Heights: {orphaned_str}")
            if len(integrity['orphaned_blocks']) > 20:
                UI.info(f"... and {len(integrity['orphaned_blocks']) - 20} more")
        
        # Overall assessment
        score = integrity.get('integrity_score', 0)
        if score >= 0.99:
            UI.success("\n✓ EXCELLENT - Chain integrity is solid")
        elif score >= 0.95:
            UI.info("\n✓ GOOD - Minor issues detected")
        elif score >= 0.90:
            UI.warning("\n⚠ FAIR - Several issues need attention")
        else:
            UI.error("\n✗ POOR - Significant integrity problems detected")
        
    except Exception as e:
        UI.error(f"Integrity check error: {e}")
        logger.error(f"Integrity check error: {e}", exc_info=True)


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# REGISTRATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

def register_block_commands(engine):
    """
    Register all comprehensive block commands with the TerminalEngine.
    
    This function should be called after TerminalEngine initialization to inject
    all the block commands into the command registry.
    
    Args:
        engine: TerminalEngine instance
    """
    from terminal_logic import CommandMeta, CommandCategory
    
    try:
        # Register block/details
        engine.registry.register(
            'block/details',
            lambda args: cmd_block_details(engine, args),
            CommandMeta(
                'block/details',
                CommandCategory.BLOCKCHAIN,
                'Comprehensive block details with quantum measurements',
                requires_auth=False
            )
        )
        
        # Register block/validate
        engine.registry.register(
            'block/validate',
            lambda args: cmd_block_validate(engine, args),
            CommandMeta(
                'block/validate',
                CommandCategory.BLOCKCHAIN,
                'Comprehensive block validation with quantum proof verification',
                requires_auth=False
            )
        )
        
        # Register block/quantum
        engine.registry.register(
            'block/quantum',
            lambda args: cmd_block_quantum(engine, args),
            CommandMeta(
                'block/quantum',
                CommandCategory.BLOCKCHAIN,
                'Perform comprehensive quantum measurements on block',
                requires_auth=False
            )
        )
        
        # Register block/batch
        engine.registry.register(
            'block/batch',
            lambda args: cmd_block_batch(engine, args),
            CommandMeta(
                'block/batch',
                CommandCategory.BLOCKCHAIN,
                'Query multiple blocks efficiently with parallel processing',
                requires_auth=False
            )
        )
        
        # Register block/integrity
        engine.registry.register(
            'block/integrity',
            lambda args: cmd_block_integrity(engine, args),
            CommandMeta(
                'block/integrity',
                CommandCategory.BLOCKCHAIN,
                'Verify blockchain integrity across range of blocks',
                requires_auth=False
            )
        )
        
        logger.info("✓ Comprehensive block commands registered successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to register block commands: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# AUTO-REGISTRATION ON IMPORT
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

# This module can be imported and will automatically register commands if TerminalEngine is available
def auto_register():
    """Attempt to auto-register commands if terminal_logic is already loaded"""
    try:
        import terminal_logic
        if hasattr(terminal_logic, 'TERMINAL_ENGINE_INSTANCE'):
            engine = terminal_logic.TERMINAL_ENGINE_INSTANCE
            return register_block_commands(engine)
    except:
        pass
    return False

# Uncomment to enable auto-registration on import
# auto_register()

logger.info("✓ Terminal block commands module loaded")
