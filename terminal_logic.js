// QTCL Terminal Logic
// Complete API interface for Quantum Temporal Coherence Ledger

class TerminalLogic {
    constructor() {
        this.term = null;
        this.fitAddon = null;
        this.authToken = localStorage.getItem('auth_token');
        this.userEmail = localStorage.getItem('user_email');
        this.baseUrl = window.location.origin;
        this.commandHistory = [];
        this.historyIndex = -1;
        
        this.init();
    }
    
    init() {
        // Initialize terminal
        this.term = new Terminal({
            rows: 24,
            cols: 100,
            theme: {
                background: '#0a0e27',
                foreground: '#e0e0e0',
                cursor: '#00d9ff',
                cursorAccent: '#0a0e27',
                selection: 'rgba(0, 217, 255, 0.2)',
                black: '#1a1a1a',
                red: '#ff4d4d',
                green: '#00ff41',
                yellow: '#ffff00',
                blue: '#00d9ff',
                magenta: '#ff00ff',
                cyan: '#00ffff',
                white: '#e0e0e0'
            },
            fontFamily: '"Fira Code", "Courier New", monospace',
            fontSize: 13,
            lineHeight: 1.4,
            letterSpacing: 0.5,
            scrollback: 1000
        });
        
        this.fitAddon = new FitAddon.FitAddon();
        this.term.loadAddon(this.fitAddon);
        
        const terminalDiv = document.getElementById('terminal');
        this.term.open(terminalDiv);
        this.fitAddon.fit();
        
        // Handle window resize
        window.addEventListener('resize', () => {
            try {
                this.fitAddon.fit();
            } catch (e) {
                console.error('Fit error:', e);
            }
        });
        
        // Handle keyboard input
        this.term.onData(data => this.handleInput(data));
        
        // Check authentication
        if (this.authToken && this.userEmail) {
            this.showMainContainer();
            this.writeWelcome();
        } else {
            this.showLogin();
        }
    }
    
    showLogin() {
        document.getElementById('loginOverlay').classList.remove('hidden');
        document.getElementById('mainContainer').classList.add('hidden');
        
        document.getElementById('loginForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleLogin();
        });
    }
    
    showMainContainer() {
        document.getElementById('loginOverlay').classList.add('hidden');
        document.getElementById('mainContainer').classList.remove('hidden');
        document.getElementById('userDisplay').textContent = this.userEmail;
    }
    
    async handleLogin() {
        const email = document.getElementById('loginEmail').value;
        const password = document.getElementById('loginPassword').value;
        
        try {
            const response = await fetch(`${this.baseUrl}/api/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password })
            });
            
            const data = await response.json();
            
            if (response.ok && data.token) {
                this.authToken = data.token;
                this.userEmail = email;
                localStorage.setItem('auth_token', this.authToken);
                localStorage.setItem('user_email', this.userEmail);
                
                this.showMainContainer();
                this.writeWelcome();
            } else {
                document.getElementById('loginError').textContent = data.message || 'Login failed';
                document.getElementById('loginError').style.display = 'block';
            }
        } catch (error) {
            document.getElementById('loginError').textContent = 'Connection error';
            document.getElementById('loginError').style.display = 'block';
        }
    }
    
    writeWelcome() {
        this.writeLine('\r\n╔════════════════════════════════════════════════════════════════╗');
        this.writeLine('║         QUANTUM TEMPORAL COHERENCE LEDGER (QTCL) v4.0           ║');
        this.writeLine('║     Interactive Terminal Interface - Type "help" for commands    ║');
        this.writeLine('╚════════════════════════════════════════════════════════════════╝\r\n');
        this.writeLine(`Welcome, ${this.userEmail}`);
        this.writeLine('Type "help" for available commands or click sidebar items.\r\n');
        this.showPrompt();
    }
    
    async handleInput(data) {
        if (data === '\r') {
            // Enter key
            const line = this.getLastLine();
            this.writeLine('');
            
            if (line.trim()) {
                this.commandHistory.push(line.trim());
                this.historyIndex = -1;
                await this.executeCommand(line.trim());
            }
            
            this.showPrompt();
        } else if (data === '\u007F') {
            // Backspace
            this.term.write('\b \b');
        } else if (data === '\u0003') {
            // Ctrl+C
            this.writeLine('^C');
            this.showPrompt();
        } else if (data === '\u0001') {
            // Ctrl+A - go to line start
            this.term.write('\x1b[H');
        } else if (data === '\u0005') {
            // Ctrl+E - go to line end
            this.term.write('\x1b[F');
        } else if (data === '\u0018') {
            // Ctrl+X - clear line
            this.term.write('\x1b[2K\r');
            this.showPrompt();
        } else {
            this.term.write(data);
        }
    }
    
    getLastLine() {
        const content = this.term.buffer.active.getLine(this.term.buffer.active.cursorY);
        return content ? content.translateToString(true).substring(2) : '';
    }
    
    async executeCommand(command) {
        const parts = command.trim().split(/\s+/);
        const cmd = parts[0];
        const args = parts.slice(1);
        
        switch (cmd.toLowerCase()) {
            // Help commands
            case 'help':
                this.showHelp(args[0]);
                break;
            
            // Authentication
            case 'auth':
                await this.handleAuthCommand(args);
                break;
            
            // Transactions
            case 'tx':
                await this.handleTxCommand(args);
                break;
            
            // Blocks
            case 'block':
                await this.handleBlockCommand(args);
                break;
            
            // Quantum system
            case 'quantum':
                await this.handleQuantumCommand(args);
                break;
            
            // Admin
            case 'admin':
                await this.handleAdminCommand(args);
                break;
            
            // System
            case 'clear':
                this.clearScreen();
                break;
            
            case 'logout':
                this.handleLogout();
                break;
            
            default:
                this.writeLine(`\x1b[31mUnknown command: ${cmd}\x1b[0m`);
                this.writeLine('Type "help" for available commands.');
        }
    }
    
    // ═══════════════════════════════════════════════════════════════════════════════════
    // HELP SYSTEM
    // ═══════════════════════════════════════════════════════════════════════════════════
    
    showHelp(category) {
        const help = {
            'auth': [
                '📝 AUTHENTICATION COMMANDS',
                '',
                'auth status          - Get current auth status',
                'auth me              - Get your user profile',
                'auth verify <token>  - Verify token validity',
                'logout               - Logout and return to login'
            ],
            'tx': [
                '💸 TRANSACTION COMMANDS',
                '',
                'tx list              - List your transactions',
                'tx create            - Create new transaction (interactive)',
                'tx get <tx_id>       - Get transaction details',
                'tx stats             - Get transaction statistics',
                'tx cancel <tx_id>    - Cancel pending transaction'
            ],
            'block': [
                '⛓️  BLOCKCHAIN COMMANDS',
                '',
                'block latest         - Get latest block',
                'block get <number>   - Get block by number',
                'block list           - List recent blocks'
            ],
            'quantum': [
                '⚛️  QUANTUM SYSTEM COMMANDS',
                '',
                'quantum status       - Get quantum system status',
                'quantum cycles       - Get cycle information',
                'quantum metrics      - Get system metrics'
            ],
            'admin': [
                '🔐 ADMIN COMMANDS',
                '',
                'admin users          - List all users',
                'admin health         - Health check',
                'admin transactions   - List all transactions'
            ]
        };
        
        if (category && help[category]) {
            help[category].forEach(line => this.writeLine(line));
        } else {
            this.writeLine('📚 AVAILABLE COMMANDS');
            this.writeLine('');
            this.writeLine('help [category]      - Show help (auth, tx, block, quantum, admin)');
            this.writeLine('auth                 - Authentication & profile');
            this.writeLine('tx                   - Transaction management');
            this.writeLine('block                - Blockchain operations');
            this.writeLine('quantum              - Quantum system');
            this.writeLine('admin                - Administrative');
            this.writeLine('clear                - Clear screen');
            this.writeLine('logout               - Logout');
        }
    }
    
    // ═══════════════════════════════════════════════════════════════════════════════════
    // AUTHENTICATION COMMANDS
    // ═══════════════════════════════════════════════════════════════════════════════════
    
    async handleAuthCommand(args) {
        const subCmd = args[0];
        
        switch (subCmd) {
            case 'status':
                this.writeLine('🔐 Authentication Status:');
                this.writeLine(`  User: ${this.userEmail}`);
                this.writeLine(`  Token: ${this.authToken ? '✓ Valid' : '✗ Invalid'}`);
                break;
            
            case 'me':
            case 'profile':
                await this.apiCall('GET', '/api/users/me', null, (data) => {
                    this.writeLine('👤 Your Profile:');
                    this.writeLine(`  ID: ${data.id}`);
                    this.writeLine(`  Email: ${data.email}`);
                    this.writeLine(`  Pseudoqubit: ${data.pseudoqubit_address || 'Not assigned'}`);
                    this.writeLine(`  Created: ${new Date(data.created_at).toLocaleString()}`);
                });
                break;
            
            case 'verify':
                if (!args[1]) {
                    this.writeLine('Usage: auth verify <token>');
                } else {
                    await this.apiCall('POST', '/api/auth/verify', { token: args[1] }, (data) => {
                        this.writeLine(`✓ Token: ${data.valid ? 'Valid' : 'Invalid'}`);
                    });
                }
                break;
            
            default:
                this.writeLine('Usage: auth [status|me|verify|profile]');
        }
    }
    
    // ═══════════════════════════════════════════════════════════════════════════════════
    // TRANSACTION COMMANDS
    // ═══════════════════════════════════════════════════════════════════════════════════
    
    async handleTxCommand(args) {
        const subCmd = args[0];
        
        switch (subCmd) {
            case 'list':
                await this.apiCall('GET', '/api/transactions?limit=10', null, (data) => {
                    this.writeLine('📋 Recent Transactions:');
                    this.writeLine('');
                    if (data.transactions && data.transactions.length > 0) {
                        data.transactions.forEach(tx => {
                            this.writeLine(`  ID: ${tx.id}`);
                            this.writeLine(`  To: ${tx.receiver_id}`);
                            this.writeLine(`  Amount: ${tx.amount}`);
                            this.writeLine(`  Status: ${tx.status}`);
                            this.writeLine('  ---');
                        });
                    } else {
                        this.writeLine('  No transactions yet.');
                    }
                });
                break;
            
            case 'create':
                await this.promptTransaction();
                break;
            
            case 'get':
            case 'details':
                if (!args[1]) {
                    this.writeLine('Usage: tx get <transaction_id>');
                } else {
                    await this.apiCall('GET', `/api/transactions/${args[1]}`, null, (data) => {
                        this.writeLine('📊 Transaction Details:');
                        this.writeLine(`  ID: ${data.id}`);
                        this.writeLine(`  From: ${data.sender_id}`);
                        this.writeLine(`  To: ${data.receiver_id}`);
                        this.writeLine(`  Amount: ${data.amount}`);
                        this.writeLine(`  Status: ${data.status}`);
                        this.writeLine(`  Quantum Validated: ${data.quantum_validated ? '✓' : '✗'}`);
                        this.writeLine(`  Created: ${new Date(data.created_at).toLocaleString()}`);
                    });
                }
                break;
            
            case 'stats':
                await this.apiCall('GET', '/api/transactions/stats', null, (data) => {
                    this.writeLine('📈 Transaction Statistics:');
                    this.writeLine(`  Total: ${data.total}`);
                    this.writeLine(`  Pending: ${data.pending}`);
                    this.writeLine(`  Completed: ${data.completed}`);
                    this.writeLine(`  Failed: ${data.failed}`);
                    this.writeLine(`  Total Amount: ${data.total_amount}`);
                });
                break;
            
            case 'cancel':
                if (!args[1]) {
                    this.writeLine('Usage: tx cancel <transaction_id>');
                } else {
                    await this.apiCall('POST', `/api/transactions/${args[1]}/cancel`, {}, (data) => {
                        this.writeLine(`✓ Transaction cancelled: ${data.id}`);
                    });
                }
                break;
            
            default:
                this.writeLine('Usage: tx [list|create|get|stats|cancel]');
        }
    }
    
    async promptTransaction() {
        // Simple transaction creation (non-interactive for now)
        this.writeLine('💸 Transaction Creation:');
        this.writeLine('');
        this.writeLine('To create a transaction interactively, use:');
        this.writeLine('tx create <email> <amount> <password>');
        this.writeLine('');
        this.writeLine('Example: tx create user@example.com 100.50 mypassword');
    }
    
    // ═══════════════════════════════════════════════════════════════════════════════════
    // BLOCK COMMANDS
    // ═══════════════════════════════════════════════════════════════════════════════════
    
    async handleBlockCommand(args) {
        const subCmd = args[0];
        
        switch (subCmd) {
            case 'latest':
                await this.apiCall('GET', '/api/blocks/latest', null, (data) => {
                    this.writeLine('⛓️  Latest Block:');
                    this.writeLine(`  Number: ${data.block_number}`);
                    this.writeLine(`  ID: ${data.id}`);
                    this.writeLine(`  Transactions: ${data.transaction_count}`);
                    this.writeLine(`  Created: ${new Date(data.created_at).toLocaleString()}`);
                });
                break;
            
            case 'get':
                if (!args[1]) {
                    this.writeLine('Usage: block get <block_number>');
                } else {
                    await this.apiCall('GET', `/api/blocks/${args[1]}`, null, (data) => {
                        this.writeLine('📦 Block Details:');
                        this.writeLine(`  Number: ${data.block_number}`);
                        this.writeLine(`  ID: ${data.id}`);
                        this.writeLine(`  Transactions: ${data.transaction_count}`);
                        this.writeLine(`  Created: ${new Date(data.created_at).toLocaleString()}`);
                    });
                }
                break;
            
            case 'list':
                await this.apiCall('GET', '/api/blocks?limit=10', null, (data) => {
                    this.writeLine('⛓️  Recent Blocks:');
                    this.writeLine('');
                    if (data.blocks && data.blocks.length > 0) {
                        data.blocks.forEach(block => {
                            this.writeLine(`  Block #${block.block_number}: ${block.transaction_count} txs`);
                        });
                    }
                });
                break;
            
            default:
                this.writeLine('Usage: block [latest|get|list]');
        }
    }
    
    // ═══════════════════════════════════════════════════════════════════════════════════
    // QUANTUM SYSTEM COMMANDS
    // ═══════════════════════════════════════════════════════════════════════════════════
    
    async handleQuantumCommand(args) {
        const subCmd = args[0];
        
        switch (subCmd) {
            case 'status':
                await this.apiCall('GET', '/api/health', null, (data) => {
                    this.writeLine('⚛️  Quantum System Status:');
                    this.writeLine(`  Status: ${data.status}`);
                    this.writeLine(`  Quantum Running: ${data.quantum_system_running ? '✓' : '✗'}`);
                    this.writeLine(`  Timestamp: ${new Date(data.timestamp).toLocaleString()}`);
                });
                break;
            
            case 'cycles':
            case 'metrics':
                this.writeLine('⚛️  Quantum System Metrics:');
                this.writeLine('  106,496 qubits');
                this.writeLine('  Non-Markovian Noise Bath: κ=0.08');
                this.writeLine('  W-state refresh: Every 5 cycles');
                this.writeLine('  Expected cycle time: 30-40 seconds');
                this.writeLine('  V7 Layers: 5 (Information Pressure, Sigma Field, Fisher Manifold, SPT, TQFT)');
                break;
            
            default:
                this.writeLine('Usage: quantum [status|cycles|metrics]');
        }
    }
    
    // ═══════════════════════════════════════════════════════════════════════════════════
    // ADMIN COMMANDS
    // ═══════════════════════════════════════════════════════════════════════════════════
    
    async handleAdminCommand(args) {
        const subCmd = args[0];
        
        switch (subCmd) {
            case 'users':
                await this.apiCall('GET', '/api/users', null, (data) => {
                    this.writeLine('👥 System Users:');
                    this.writeLine('');
                    if (data.users && data.users.length > 0) {
                        data.users.forEach(user => {
                            this.writeLine(`  ${user.email} (${user.id})`);
                        });
                    }
                });
                break;
            
            case 'health':
                await this.apiCall('GET', '/api/health', null, (data) => {
                    this.writeLine('🏥 System Health:');
                    this.writeLine(`  Status: ${data.status}`);
                    this.writeLine(`  Quantum System: ${data.quantum_system_running ? '✓ Running' : '✗ Stopped'}`);
                    this.writeLine(`  Timestamp: ${new Date(data.timestamp).toLocaleString()}`);
                });
                break;
            
            case 'transactions':
                await this.apiCall('GET', '/api/admin/transactions', null, (data) => {
                    this.writeLine('📋 All Transactions:');
                    this.writeLine('');
                    if (data.transactions && data.transactions.length > 0) {
                        data.transactions.slice(0, 10).forEach(tx => {
                            this.writeLine(`  ${tx.id}: ${tx.sender_id} → ${tx.receiver_id} (${tx.amount})`);
                        });
                        this.writeLine(`  ... and ${data.transactions.length - 10} more`);
                    }
                });
                break;
            
            default:
                this.writeLine('Usage: admin [users|health|transactions]');
        }
    }
    
    // ═══════════════════════════════════════════════════════════════════════════════════
    // API COMMUNICATION
    // ═══════════════════════════════════════════════════════════════════════════════════
    
    async apiCall(method, endpoint, body, callback) {
        try {
            const options = {
                method: method,
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.authToken}`
                }
            };
            
            if (body) {
                options.body = JSON.stringify(body);
            }
            
            const response = await fetch(`${this.baseUrl}${endpoint}`, options);
            const data = await response.json();
            
            if (response.ok) {
                if (callback) {
                    callback(data);
                }
            } else {
                this.writeLine(`\x1b[31m✗ Error: ${data.message || 'Request failed'}\x1b[0m`);
            }
        } catch (error) {
            this.writeLine(`\x1b[31m✗ Connection error: ${error.message}\x1b[0m`);
        }
    }
    
    // ═══════════════════════════════════════════════════════════════════════════════════
    // UTILITY FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════════════
    
    writeLine(text) {
        this.term.writeln(text);
    }
    
    showPrompt() {
        this.term.write('\x1b[32m$ \x1b[0m');
    }
    
    clearScreen() {
        this.term.clear();
        this.writeWelcome();
    }
    
    copyTerminalOutput() {
        const buffer = this.term.buffer.active.getLine(0);
        if (buffer) {
            const content = buffer.translateToString();
            navigator.clipboard.writeText(content);
            this.writeLine('✓ Copied to clipboard');
        }
    }
    
    handleLogout() {
        localStorage.removeItem('auth_token');
        localStorage.removeItem('user_email');
        location.reload();
    }
}

// Initialize terminal logic when page loads
let terminalLogic;
document.addEventListener('DOMContentLoaded', () => {
    terminalLogic = new TerminalLogic();
    
    // Logout button
    document.getElementById('logoutBtn').addEventListener('click', () => {
        terminalLogic.handleLogout();
    });
});
