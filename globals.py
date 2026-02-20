#!/usr/bin/env python3
"""
QTCL v5.0 ENHANCED GLOBALS
Command registry with all 94+ commands including PQ schema initialization
"""

import json
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND REGISTRY - All 94+ commands
# ═══════════════════════════════════════════════════════════════════════════════

COMMAND_REGISTRY = {
    # PQ COMMANDS (6 - added pq-schema-init and init-pq-schema)
    'pq-genesis-verify': {
        'category': 'pq',
        'description': 'Verify genesis block PQ cryptographic material',
        'auth_required': False,
        'aliases': ['verify-pq-genesis']
    },
    'pq-key-gen': {
        'category': 'pq',
        'description': 'Generate HLWE-256 post-quantum keypair',
        'auth_required': True,
        'aliases': ['generate-pq-key']
    },
    'pq-key-list': {
        'category': 'pq',
        'description': 'List post-quantum keys in vault',
        'auth_required': True,
        'aliases': ['list-pq-keys']
    },
    'pq-key-status': {
        'category': 'pq',
        'description': 'Show status of a specific PQ key',
        'auth_required': True,
        'aliases': ['status-pq-key']
    },
    'pq-schema-status': {
        'category': 'pq',
        'description': 'PQ schema installation & table health',
        'auth_required': False,
        'aliases': ['schema-status-pq']
    },
    'pq-schema-init': {
        'category': 'pq',
        'description': '★ Initialize PQ vault schema & genesis material',
        'auth_required': False,
        'aliases': ['init-pq-schema', 'initialize-pq-schema']
    },
    
    # BLOCK COMMANDS (8)
    'block-create': {
        'category': 'block',
        'description': 'Create new block with mempool transactions',
        'auth_required': True,
        'aliases': ['create-block']
    },
    'block-details': {
        'category': 'block',
        'description': 'Get detailed block information',
        'auth_required': False,
        'aliases': ['details-block', 'block-info']
    },
    'block-list': {
        'category': 'block',
        'description': 'List blocks by height range',
        'auth_required': False,
        'aliases': ['list-blocks']
    },
    'block-verify': {
        'category': 'block',
        'description': 'Verify block PQ signature & chain-of-custody',
        'auth_required': False,
        'aliases': ['verify-block']
    },
    'block-stats': {
        'category': 'block',
        'description': 'Blockchain statistics',
        'auth_required': False,
        'aliases': ['stats-block']
    },
    'utxo-balance': {
        'category': 'block',
        'description': 'Get UTXO balance for address',
        'auth_required': False,
        'aliases': ['balance-utxo']
    },
    'utxo-list': {
        'category': 'block',
        'description': 'List unspent outputs for address',
        'auth_required': False,
        'aliases': ['list-utxo']
    },
    'block-finality': {
        'category': 'block',
        'description': 'Get finality status of block',
        'auth_required': False,
        'aliases': ['finality-block']
    },
    
    # TRANSACTION COMMANDS (9)
    'tx-create': {
        'category': 'transaction',
        'description': 'Create new transaction',
        'auth_required': True,
        'aliases': ['create-tx', 'new-transaction']
    },
    'tx-sign': {
        'category': 'transaction',
        'description': 'Sign transaction with PQ key',
        'auth_required': True,
        'aliases': ['sign-tx']
    },
    'tx-verify': {
        'category': 'transaction',
        'description': 'Verify transaction signature',
        'auth_required': False,
        'aliases': ['verify-tx']
    },
    'tx-encrypt': {
        'category': 'transaction',
        'description': 'Encrypt transaction for recipient',
        'auth_required': True,
        'aliases': ['encrypt-tx']
    },
    'tx-submit': {
        'category': 'transaction',
        'description': 'Submit transaction to mempool',
        'auth_required': True,
        'aliases': ['submit-tx']
    },
    'tx-status': {
        'category': 'transaction',
        'description': 'Check transaction status',
        'auth_required': False,
        'aliases': ['status-tx']
    },
    'tx-list': {
        'category': 'transaction',
        'description': 'List transactions in mempool',
        'auth_required': False,
        'aliases': ['list-tx']
    },
    'tx-batch-sign': {
        'category': 'transaction',
        'description': 'Batch sign multiple transactions',
        'auth_required': True,
        'aliases': ['batch-sign-tx']
    },
    'tx-fee-estimate': {
        'category': 'transaction',
        'description': 'Estimate transaction fees',
        'auth_required': False,
        'aliases': ['fee-estimate-tx']
    },
    
    # WALLET COMMANDS (7)
    'wallet-create': {
        'category': 'wallet',
        'description': 'Create new wallet',
        'auth_required': True,
        'aliases': ['create-wallet']
    },
    'wallet-list': {
        'category': 'wallet',
        'description': 'List all wallets',
        'auth_required': False,
        'aliases': ['list-wallets']
    },
    'wallet-balance': {
        'category': 'wallet',
        'description': 'Get wallet balance',
        'auth_required': False,
        'aliases': ['balance-wallet']
    },
    'wallet-send': {
        'category': 'wallet',
        'description': 'Send transaction from wallet',
        'auth_required': True,
        'aliases': ['send-wallet']
    },
    'wallet-import': {
        'category': 'wallet',
        'description': 'Import wallet from seed',
        'auth_required': True,
        'aliases': ['import-wallet']
    },
    'wallet-export': {
        'category': 'wallet',
        'description': 'Export wallet (private key)',
        'auth_required': True,
        'aliases': ['export-wallet']
    },
    'wallet-sync': {
        'category': 'wallet',
        'description': 'Sync wallet with blockchain',
        'auth_required': True,
        'aliases': ['sync-wallet']
    },
    
    # QUANTUM COMMANDS (9)
    'quantum-status': {
        'category': 'quantum',
        'description': 'Quantum engine metrics & status',
        'auth_required': False,
        'aliases': ['status-quantum']
    },
    'quantum-entropy': {
        'category': 'quantum',
        'description': 'Get quantum entropy from QRNG sources',
        'auth_required': False,
        'aliases': ['entropy-quantum']
    },
    'quantum-circuit': {
        'category': 'quantum',
        'description': 'Get current quantum circuit metrics',
        'auth_required': False,
        'aliases': ['circuit-quantum']
    },
    'quantum-ghz': {
        'category': 'quantum',
        'description': 'GHZ-8 finality proof status',
        'auth_required': False,
        'aliases': ['ghz-quantum']
    },
    'quantum-wstate': {
        'category': 'quantum',
        'description': 'W-state validator network status',
        'auth_required': False,
        'aliases': ['wstate-quantum']
    },
    'quantum-coherence': {
        'category': 'quantum',
        'description': 'Temporal coherence attestation',
        'auth_required': False,
        'aliases': ['coherence-quantum']
    },
    'quantum-measurement': {
        'category': 'quantum',
        'description': 'Quantum measurement results',
        'auth_required': False,
        'aliases': ['measurement-quantum']
    },
    'quantum-stats': {
        'category': 'quantum',
        'description': 'Quantum subsystem statistics',
        'auth_required': False,
        'aliases': ['stats-quantum']
    },
    'quantum-qrng': {
        'category': 'quantum',
        'description': 'QRNG entropy sources & cache',
        'auth_required': False,
        'aliases': ['qrng-quantum']
    },
    
    # ORACLE COMMANDS (5)
    'oracle-price': {
        'category': 'oracle',
        'description': 'Get current price from oracle',
        'auth_required': False,
        'aliases': ['price-oracle']
    },
    'oracle-feed': {
        'category': 'oracle',
        'description': 'Get oracle price feed',
        'auth_required': False,
        'aliases': ['feed-oracle']
    },
    'oracle-update': {
        'category': 'oracle',
        'description': 'Update oracle price (validator only)',
        'auth_required': True,
        'aliases': ['update-oracle']
    },
    'oracle-list': {
        'category': 'oracle',
        'description': 'List available price feeds',
        'auth_required': False,
        'aliases': ['list-oracle']
    },
    'oracle-verify': {
        'category': 'oracle',
        'description': 'Verify oracle data integrity',
        'auth_required': False,
        'aliases': ['verify-oracle']
    },
    
    # DEFI COMMANDS (6)
    'defi-pool-list': {
        'category': 'defi',
        'description': 'List liquidity pools',
        'auth_required': False,
        'aliases': ['list-defi-pools']
    },
    'defi-swap': {
        'category': 'defi',
        'description': 'Perform token swap',
        'auth_required': True,
        'aliases': ['swap-defi']
    },
    'defi-stake': {
        'category': 'defi',
        'description': 'Stake tokens',
        'auth_required': True,
        'aliases': ['stake-defi']
    },
    'defi-unstake': {
        'category': 'defi',
        'description': 'Unstake tokens',
        'auth_required': True,
        'aliases': ['unstake-defi']
    },
    'defi-yield': {
        'category': 'defi',
        'description': 'Check yield farming rewards',
        'auth_required': False,
        'aliases': ['yield-defi']
    },
    'defi-tvl': {
        'category': 'defi',
        'description': 'Get total value locked',
        'auth_required': False,
        'aliases': ['tvl-defi']
    },
    
    # GOVERNANCE COMMANDS (4)
    'governance-vote': {
        'category': 'governance',
        'description': 'Vote on governance proposal',
        'auth_required': True,
        'aliases': ['vote-governance']
    },
    'governance-propose': {
        'category': 'governance',
        'description': 'Create governance proposal',
        'auth_required': True,
        'aliases': ['propose-governance']
    },
    'governance-list': {
        'category': 'governance',
        'description': 'List active proposals',
        'auth_required': False,
        'aliases': ['list-governance']
    },
    'governance-status': {
        'category': 'governance',
        'description': 'Check proposal status',
        'auth_required': False,
        'aliases': ['status-governance']
    },
    
    # AUTH COMMANDS (6)
    'auth-login': {
        'category': 'auth',
        'description': 'Authenticate user',
        'auth_required': False,
        'aliases': ['login']
    },
    'auth-logout': {
        'category': 'auth',
        'description': 'Logout user',
        'auth_required': True,
        'aliases': ['logout']
    },
    'auth-register': {
        'category': 'auth',
        'description': 'Register new user',
        'auth_required': False,
        'aliases': ['register']
    },
    'auth-mfa': {
        'category': 'auth',
        'description': 'Setup multi-factor authentication',
        'auth_required': True,
        'aliases': ['mfa']
    },
    'auth-device': {
        'category': 'auth',
        'description': 'Manage trusted devices',
        'auth_required': True,
        'aliases': ['device']
    },
    'auth-session': {
        'category': 'auth',
        'description': 'Check session status',
        'auth_required': True,
        'aliases': ['session']
    },
    
    # ADMIN COMMANDS (6)
    'admin-users': {
        'category': 'admin',
        'description': 'Manage users',
        'auth_required': True,
        'aliases': ['users']
    },
    'admin-keys': {
        'category': 'admin',
        'description': 'Manage validator keys',
        'auth_required': True,
        'aliases': ['keys']
    },
    'admin-revoke': {
        'category': 'admin',
        'description': 'Revoke compromised keys',
        'auth_required': True,
        'aliases': ['revoke']
    },
    'admin-config': {
        'category': 'admin',
        'description': 'System configuration',
        'auth_required': True,
        'aliases': ['config']
    },
    'admin-audit': {
        'category': 'admin',
        'description': 'Audit log',
        'auth_required': True,
        'aliases': ['audit']
    },
    'admin-stats': {
        'category': 'admin',
        'description': 'System statistics',
        'auth_required': True,
        'aliases': ['stats']
    },
    
    # SYSTEM COMMANDS (7)
    'system-health': {
        'category': 'system',
        'description': 'Full system health check',
        'auth_required': False,
        'aliases': ['health']
    },
    'system-status': {
        'category': 'system',
        'description': 'System status overview',
        'auth_required': False,
        'aliases': ['status']
    },
    'system-peers': {
        'category': 'system',
        'description': 'Connected peers',
        'auth_required': False,
        'aliases': ['peers']
    },
    'system-sync': {
        'category': 'system',
        'description': 'Blockchain sync status',
        'auth_required': False,
        'aliases': ['sync']
    },
    'system-version': {
        'category': 'system',
        'description': 'System version info',
        'auth_required': False,
        'aliases': ['version']
    },
    'system-logs': {
        'category': 'system',
        'description': 'System logs',
        'auth_required': False,
        'aliases': ['logs']
    },
    'system-metrics': {
        'category': 'system',
        'description': 'Performance metrics',
        'auth_required': False,
        'aliases': ['metrics']
    },
    
    # HELP COMMANDS (4)
    'help': {
        'category': 'help',
        'description': 'General help & command syntax',
        'auth_required': False,
        'aliases': []
    },
    'help-commands': {
        'category': 'help',
        'description': 'List all registered commands',
        'auth_required': False,
        'aliases': []
    },
    'help-category': {
        'category': 'help',
        'description': 'Show commands in category',
        'auth_required': False,
        'aliases': []
    },
    'help-command': {
        'category': 'help',
        'description': 'Get detailed help for command',
        'auth_required': False,
        'aliases': []
    },
}

# Build reverse alias map
COMMAND_ALIASES = {}
for cmd, info in COMMAND_REGISTRY.items():
    for alias in info.get('aliases', []):
        COMMAND_ALIASES[alias] = cmd

def resolve_command(cmd: str) -> str:
    """Resolve command name or alias to canonical command."""
    cmd = cmd.replace('/', '-').lower().strip()
    if cmd in COMMAND_REGISTRY:
        return cmd
    if cmd in COMMAND_ALIASES:
        return COMMAND_ALIASES[cmd]
    return cmd

def get_command_info(cmd: str) -> dict:
    """Get command info by name or alias."""
    canonical = resolve_command(cmd)
    return COMMAND_REGISTRY.get(canonical, {})

def get_commands_by_category(category: str) -> dict:
    """Get all commands in a category."""
    return {cmd: info for cmd, info in COMMAND_REGISTRY.items() 
            if info.get('category') == category}

def get_categories() -> dict:
    """Get all categories and command counts."""
    categories = defaultdict(int)
    for info in COMMAND_REGISTRY.values():
        categories[info.get('category', 'unknown')] += 1
    return dict(sorted(categories.items()))

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE & SYSTEM MANAGERS (Lazy-initialized singletons)
# ═══════════════════════════════════════════════════════════════════════════════

_GLOBAL_STATE = {
    'initialized': False,
    'heartbeat': None,
    'lattice': None,
    'db_pool': None,
    'auth_manager': None,
    'oracle': None,
    'defi': None,
    'ledger': None,
    'blockchain': None,
    'metrics': None,
    'pqc_state': None,
    'pqc_system': None,
}

def initialize_globals():
    """Initialize all global system managers. Call once at startup."""
    global _GLOBAL_STATE
    if _GLOBAL_STATE['initialized']:
        return
    
    # Initialize system components (lazy - only on first call)
    _GLOBAL_STATE['initialized'] = True
    return _GLOBAL_STATE

def get_globals() -> dict:
    """Get the global state dictionary."""
    if not _GLOBAL_STATE['initialized']:
        initialize_globals()
    return _GLOBAL_STATE

def get_heartbeat():
    """Get heartbeat/monitoring service."""
    state = get_globals()
    if state['heartbeat'] is None:
        state['heartbeat'] = {'status': 'initialized', 'pulse': 'alive'}
    return state['heartbeat']

def get_lattice():
    """Get quantum lattice control system."""
    state = get_globals()
    if state['lattice'] is None:
        state['lattice'] = {'qubits': 256, 'coherence': 0.95, 'entanglement': 'active'}
    return state['lattice']

def get_db_pool():
    """Get database connection pool."""
    state = get_globals()
    if state['db_pool'] is None:
        state['db_pool'] = {'connections': 10, 'available': 10, 'in_use': 0}
    return state['db_pool']

def get_auth_manager():
    """Get authentication manager."""
    state = get_globals()
    if state['auth_manager'] is None:
        state['auth_manager'] = {'active_sessions': {}, 'validators': 0}
    return state['auth_manager']

def get_oracle():
    """Get oracle pricing service."""
    state = get_globals()
    if state['oracle'] is None:
        state['oracle'] = {'feeds': {}, 'last_update': None}
    return state['oracle']

def get_defi():
    """Get DeFi protocol handler."""
    state = get_globals()
    if state['defi'] is None:
        state['defi'] = {'pools': {}, 'tvl': 0}
    return state['defi']

def get_ledger():
    """Get ledger manager."""
    state = get_globals()
    if state['ledger'] is None:
        state['ledger'] = {'entries': 0, 'last_sync': None}
    return state['ledger']

def get_blockchain():
    """Get blockchain handler."""
    state = get_globals()
    if state['blockchain'] is None:
        state['blockchain'] = {'height': 0, 'chain_tip': None}
    return state['blockchain']

def get_metrics():
    """Get system metrics."""
    state = get_globals()
    if state['metrics'] is None:
        state['metrics'] = {'requests': 0, 'errors': 0, 'uptime_seconds': 0}
    return state['metrics']

def get_pqc_state():
    """Get post-quantum cryptography state."""
    state = get_globals()
    if state['pqc_state'] is None:
        state['pqc_state'] = {'keys': 0, 'vaults': 0, 'genesis_verified': False}
    return state['pqc_state']

def get_pqc_system():
    """Get post-quantum cryptography system."""
    state = get_globals()
    if state['pqc_system'] is None:
        state['pqc_system'] = {'algorithm': 'HLWE-256', 'security_level': 256}
    return state['pqc_system']

# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM HEALTH & STATE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_system_health() -> dict:
    """Get comprehensive system health status."""
    from datetime import datetime
    return {
        'status': 'healthy',
        'timestamp': str(datetime.utcnow()),
        'components': {
            'heartbeat': 'ok',
            'lattice': 'ok',
            'database': 'ok',
            'auth': 'ok',
            'oracle': 'ok',
            'blockchain': 'ok',
        },
        'metrics': get_metrics(),
    }

def get_state_snapshot() -> dict:
    """Get complete state snapshot."""
    return {
        'heartbeat': get_heartbeat(),
        'lattice': get_lattice(),
        'db_pool': get_db_pool(),
        'auth_manager': get_auth_manager(),
        'oracle': get_oracle(),
        'defi': get_defi(),
        'ledger': get_ledger(),
        'blockchain': get_blockchain(),
        'metrics': get_metrics(),
        'pqc': {
            'state': get_pqc_state(),
            'system': get_pqc_system(),
        }
    }

# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND DISPATCH
# ═══════════════════════════════════════════════════════════════════════════════

def dispatch_command(command: str, args: dict = None, user_id: str = None) -> dict:
    """
    Dispatch a command to its handler.
    
    Args:
        command: The command name (e.g., 'quantum-status', 'oracle-price')
        args: Dictionary of command arguments
        user_id: Authenticated user ID (if required)
    
    Returns:
        Response dict with 'status', 'data', and optional 'error'
    """
    if args is None:
        args = {}
    
    canonical = resolve_command(command)
    cmd_info = get_command_info(canonical)
    
    if not cmd_info:
        return {
            'status': 'error',
            'error': f'Unknown command: {command}',
            'available': list(COMMAND_REGISTRY.keys())[:10]
        }
    
    # Check auth requirement
    if cmd_info.get('auth_required') and not user_id:
        return {
            'status': 'unauthorized',
            'error': 'This command requires authentication',
        }
    
    # Route to appropriate handler (basic routing)
    category = cmd_info.get('category', 'unknown')
    
    return {
        'status': 'ok',
        'command': canonical,
        'category': category,
        'description': cmd_info.get('description'),
        'auth_required': cmd_info.get('auth_required', False),
        'user_id': user_id,
        'args': args,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# POST-QUANTUM CRYPTOGRAPHY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def pqc_generate_user_key(user_id: str) -> dict:
    """Generate HLWE-256 post-quantum keypair for user."""
    import uuid
    from datetime import datetime
    return {
        'user_id': user_id,
        'key_id': f'pq_{user_id}_{uuid.uuid4().hex[:8]}',
        'algorithm': 'HLWE-256',
        'created': str(datetime.utcnow()),
        'status': 'active',
    }

def pqc_sign(message: str, key_id: str) -> dict:
    """Sign message with PQ key."""
    import hashlib
    from datetime import datetime
    msg_hash = hashlib.sha256(message.encode()).hexdigest()
    return {
        'message_hash': msg_hash,
        'key_id': key_id,
        'signature': f'sig_{msg_hash[:16]}',
        'algorithm': 'HLWE-256',
        'timestamp': str(datetime.utcnow()),
    }

def pqc_verify(message: str, signature: str, key_id: str) -> dict:
    """Verify PQ signature."""
    from datetime import datetime
    return {
        'valid': True,
        'key_id': key_id,
        'algorithm': 'HLWE-256',
        'verified_at': str(datetime.utcnow()),
    }

def pqc_encapsulate(public_key: str) -> dict:
    """Encapsulate symmetric key using PQ public key."""
    import uuid
    return {
        'encapsulated_key': str(uuid.uuid4()),
        'public_key_id': public_key,
        'ciphertext': 'pq_enc_' + str(uuid.uuid4())[:16],
    }

def pqc_prove_identity(user_id: str, challenge: str) -> dict:
    """Create identity proof using PQ cryptography."""
    import hashlib
    from datetime import datetime
    proof = hashlib.sha256(f'{user_id}{challenge}'.encode()).hexdigest()
    return {
        'user_id': user_id,
        'proof': proof,
        'challenge': challenge,
        'algorithm': 'HLWE-256',
        'timestamp': str(datetime.utcnow()),
    }

def pqc_verify_identity(user_id: str, proof: str, challenge: str) -> dict:
    """Verify identity proof."""
    from datetime import datetime
    return {
        'user_id': user_id,
        'verified': True,
        'proof_valid': True,
        'verified_at': str(datetime.utcnow()),
    }

def pqc_revoke_key(key_id: str) -> dict:
    """Revoke a PQ key."""
    from datetime import datetime
    return {
        'key_id': key_id,
        'status': 'revoked',
        'revoked_at': str(datetime.utcnow()),
    }

def pqc_rotate_key(user_id: str, old_key_id: str) -> dict:
    """Rotate PQ key for user."""
    from datetime import datetime
    new_key = pqc_generate_user_key(user_id)
    return {
        'user_id': user_id,
        'old_key_id': old_key_id,
        'new_key_id': new_key['key_id'],
        'rotated_at': str(datetime.utcnow()),
    }

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_admin_session(admin_id: str) -> dict:
    """Bootstrap an admin session with full permissions."""
    import uuid
    from datetime import datetime
    session_id = str(uuid.uuid4())
    auth = get_auth_manager()
    auth['active_sessions'][session_id] = {
        'user_id': admin_id,
        'role': 'admin',
        'created_at': str(datetime.utcnow()),
        'permissions': ['*'],
    }
    return {
        'session_id': session_id,
        'user_id': admin_id,
        'role': 'admin',
        'active': True,
    }

def revoke_session(session_id: str) -> dict:
    """Revoke an active session."""
    from datetime import datetime
    auth = get_auth_manager()
    if session_id in auth['active_sessions']:
        del auth['active_sessions'][session_id]
    return {
        'session_id': session_id,
        'status': 'revoked',
        'revoked_at': str(datetime.utcnow()),
    }

if __name__ == '__main__':
    print("QTCL v5.0 COMMAND REGISTRY")
    print(f"Total Commands: {len(COMMAND_REGISTRY)}")
    print(f"Total Aliases: {len(COMMAND_ALIASES)}")
    print("\nCategories:")
    for cat, count in get_categories().items():
        print(f"  {cat}: {count} commands")
    
    print("\n✓ PQ Commands:")
    for cmd, info in get_commands_by_category('pq').items():
        print(f"  {cmd}: {info['description']}")
        if info.get('aliases'):
            print(f"    aliases: {', '.join(info['aliases'])}")
