/**
 * ╔═════════════════════════════════════════════════════════════════════════════════════════╗
 * ║                                                                                         ║
 * ║   🚀 QTCL COMMAND EXECUTION BRIDGE - JavaScript Client v5.0 🚀                         ║
 * ║                                                                                         ║
 * ║   Real-time command execution interface for index.html                                 ║
 * ║   WebSocket streaming | Command history | Flag parsing | Real-time feedback           ║
 * ║                                                                                         ║
 * ║   THIS CONNECTS index.html TO THE BACKEND COMMAND ENGINE:                              ║
 * ║   ✅ Execute commands in real-time                                                     ║
 * ║   ✅ Stream responses via WebSocket                                                    ║
 * ║   ✅ Parse flags and variables                                                         ║
 * ║   ✅ Compound command support                                                          ║
 * ║   ✅ Command history tracking                                                          ║
 * ║   ✅ Error handling & retry                                                            ║
 * ║   ✅ Performance monitoring                                                            ║
 * ║   ✅ Autocomplete support                                                              ║
 * ║                                                                                         ║
 * ╚═════════════════════════════════════════════════════════════════════════════════════════╝
 */

class QTCLCommandExecutor {
    constructor(config = {}) {
        this.config = {
            apiUrl: config.apiUrl || 'http://localhost:5000',
            wsUrl: config.wsUrl || 'ws://localhost:5000/socket.io',
            timeout: config.timeout || 30000,
            maxHistory: config.maxHistory || 100,
            ...config
        };
        
        this.history = [];
        this.results = new Map();
        this.ws = null;
        this.isConnected = false;
        this.commandQueue = [];
        this.listeners = {
            'commandExecuted': [],
            'commandError': [],
            'commandStarted': [],
            'connectionChanged': []
        };
        
        this.init();
    }
    
    /**
     * Initialize the executor
     */
    init() {
        this.connectWebSocket();
        this.setupEventListeners();
        logger.info('[CommandExecutor] Initialized');
    }
    
    /**
     * Connect to WebSocket server
     */
    connectWebSocket() {
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/socket.io`;
            
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                this.isConnected = true;
                logger.info('[WebSocket] Connected');
                this._emit('connectionChanged', { connected: true });
                this._flushCommandQueue();
            };
            
            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this._handleWebSocketMessage(data);
            };
            
            this.ws.onerror = (error) => {
                logger.error('[WebSocket] Error:', error);
                this._emit('commandError', { message: 'WebSocket error' });
            };
            
            this.ws.onclose = () => {
                this.isConnected = false;
                logger.warn('[WebSocket] Disconnected');
                this._emit('connectionChanged', { connected: false });
                setTimeout(() => this.connectWebSocket(), 5000); // Reconnect
            };
        } catch (error) {
            logger.error('[WebSocket] Connection failed:', error);
        }
    }
    
    /**
     * Execute a command
     */
    async execute(command, options = {}) {
        const commandId = this._generateId();
        const startTime = performance.now();
        
        try {
            // Parse command
            const parsed = this.parseCommand(command);
            
            // Emit start event
            this._emit('commandStarted', { command, commandId, parsed });
            
            // Add to history
            this.history.unshift({
                id: commandId,
                command,
                parsed,
                status: 'executing',
                startTime,
                timestamp: new Date()
            });
            if (this.history.length > this.config.maxHistory) {
                this.history.pop();
            }
            
            // Execute via HTTP API
            const result = await this._executeViaAPI(command, commandId, options);
            
            const duration = performance.now() - startTime;
            
            // Update history
            this._updateHistory(commandId, {
                status: result.status,
                output: result.output,
                error: result.error,
                duration,
                endTime: startTime + duration
            });
            
            // Store result
            this.results.set(commandId, result);
            
            // Emit event
            this._emit('commandExecuted', {
                commandId,
                command,
                result,
                duration
            });
            
            return result;
        } catch (error) {
            const duration = performance.now() - startTime;
            
            this._updateHistory(commandId, {
                status: 'error',
                error: error.message,
                duration
            });
            
            this._emit('commandError', {
                commandId,
                command,
                error: error.message,
                duration
            });
            
            throw error;
        }
    }
    
    /**
     * Execute compound command
     */
    async executeCompound(compoundCommand, options = {}) {
        const commandId = this._generateId();
        const startTime = performance.now();
        
        try {
            const response = await fetch(
                `${this.config.apiUrl}/api/execute/compound`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        command: compoundCommand,
                        ...options
                    }),
                    timeout: this.config.timeout
                }
            );
            
            if (!response.ok) {
                throw new Error(`API error: ${response.statusText}`);
            }
            
            const data = await response.json();
            const duration = performance.now() - startTime;
            
            this._emit('commandExecuted', {
                commandId,
                command: compoundCommand,
                result: data,
                duration
            });
            
            return data;
        } catch (error) {
            const duration = performance.now() - startTime;
            
            this._emit('commandError', {
                commandId,
                command: compoundCommand,
                error: error.message,
                duration
            });
            
            throw error;
        }
    }
    
    /**
     * Execute batch of commands
     */
    async executeBatch(commands, options = {}) {
        const commandId = this._generateId();
        const startTime = performance.now();
        
        try {
            const response = await fetch(
                `${this.config.apiUrl}/api/execute/batch`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        commands,
                        ...options
                    }),
                    timeout: this.config.timeout
                }
            );
            
            if (!response.ok) {
                throw new Error(`API error: ${response.statusText}`);
            }
            
            const data = await response.json();
            const duration = performance.now() - startTime;
            
            this._emit('commandExecuted', {
                commandId,
                command: `batch(${commands.length})`,
                result: data,
                duration
            });
            
            return data;
        } catch (error) {
            const duration = performance.now() - startTime;
            
            this._emit('commandError', {
                commandId,
                command: `batch(${commands.length})`,
                error: error.message,
                duration
            });
            
            throw error;
        }
    }
    
    /**
     * Parse command into structure
     */
    parseCommand(command) {
        const parsed = {
            raw: command,
            flags: {},
            variables: {},
            args: [],
            category: null,
            action: null
        };
        
        // Parse flags
        const flagRegex = /--([a-z0-9\-]+)(?:=([^\s]+))?|-([a-z0-9])\s*([^\s]+)?/gi;
        let match;
        while ((match = flagRegex.exec(command)) !== null) {
            const flagName = (match[1] || match[3]).replace(/-/g, '_');
            const flagValue = match[2] || match[4] || true;
            parsed.flags[flagName] = flagValue;
        }
        
        // Extract base command
        const baseParts = command.replace(/--[^\s]+(=[^\s]+)?/g, '').replace(/-[a-z]\s*[^\s]+?/g, '').split(/\s+/).filter(p => p);
        if (baseParts.length > 0) {
            const baseCmd = baseParts[0];
            if (baseCmd.includes('/')) {
                [parsed.category, parsed.action] = baseCmd.split('/');
            } else {
                parsed.category = baseCmd;
                parsed.action = baseParts[1] || 'default';
            }
            parsed.args = baseParts.slice(parsed.action ? 2 : 1);
        }
        
        return parsed;
    }
    
    /**
     * Get available commands
     */
    async getCommands() {
        try {
            const response = await fetch(`${this.config.apiUrl}/api/commands`);
            if (!response.ok) throw new Error('Failed to fetch commands');
            return await response.json();
        } catch (error) {
            logger.error('[CommandExecutor] Error fetching commands:', error);
            return null;
        }
    }
    
    /**
     * Get command help
     */
    async getHelp(command) {
        try {
            const response = await fetch(
                `${this.config.apiUrl}/api/commands/help?command=${encodeURIComponent(command)}`
            );
            if (!response.ok) throw new Error('Failed to fetch help');
            return await response.json();
        } catch (error) {
            logger.error('[CommandExecutor] Error fetching help:', error);
            return null;
        }
    }
    
    /**
     * Get execution history
     */
    getHistory(limit = 50) {
        return this.history.slice(0, limit);
    }
    
    /**
     * Get execution statistics
     */
    async getStats() {
        try {
            const response = await fetch(`${this.config.apiUrl}/api/execute/stats`);
            if (!response.ok) throw new Error('Failed to fetch stats');
            return await response.json();
        } catch (error) {
            logger.error('[CommandExecutor] Error fetching stats:', error);
            return null;
        }
    }
    
    /**
     * Register event listener
     */
    on(event, callback) {
        if (this.listeners[event]) {
            this.listeners[event].push(callback);
        }
    }
    
    /**
     * Remove event listener
     */
    off(event, callback) {
        if (this.listeners[event]) {
            const index = this.listeners[event].indexOf(callback);
            if (index > -1) {
                this.listeners[event].splice(index, 1);
            }
        }
    }
    
    /**
     * Private methods
     */
    
    async _executeViaAPI(command, commandId, options) {
        const response = await fetch(
            `${this.config.apiUrl}/api/execute`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    command,
                    ...options
                }),
                timeout: this.config.timeout
            }
        );
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || `API error: ${response.statusText}`);
        }
        
        return await response.json();
    }
    
    _handleWebSocketMessage(data) {
        if (data.type === 'command_result') {
            this._emit('commandExecuted', data);
        } else if (data.type === 'command_error') {
            this._emit('commandError', data);
        }
    }
    
    _emit(event, data) {
        if (this.listeners[event]) {
            this.listeners[event].forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    logger.error(`[CommandExecutor] Error in ${event} listener:`, error);
                }
            });
        }
    }
    
    _updateHistory(commandId, updates) {
        const entry = this.history.find(h => h.id === commandId);
        if (entry) {
            Object.assign(entry, updates);
        }
    }
    
    _generateId() {
        return 'cmd_' + Math.random().toString(36).substr(2, 9);
    }
    
    _flushCommandQueue() {
        while (this.commandQueue.length > 0) {
            const cmd = this.commandQueue.shift();
            this.execute(cmd.command, cmd.options).catch(error => {
                logger.error('[CommandExecutor] Queued command error:', error);
            });
        }
    }
    
    setupEventListeners() {
        // Can be extended in subclass
    }
}

/**
 * Global logger (simple implementation)
 */
const logger = {
    info: (msg) => console.log(`[INFO] ${msg}`),
    warn: (msg) => console.warn(`[WARN] ${msg}`),
    error: (msg, err) => console.error(`[ERROR] ${msg}`, err),
    debug: (msg) => console.log(`[DEBUG] ${msg}`)
};

/**
 * Initialize and export
 */
window.CommandExecutor = QTCLCommandExecutor;

// Auto-initialize
document.addEventListener('DOMContentLoaded', () => {
    if (!window.commandExecutor) {
        window.commandExecutor = new QTCLCommandExecutor({
            apiUrl: `http://${window.location.hostname}:${window.location.port || 5000}`,
            wsUrl: `ws://${window.location.hostname}:${window.location.port || 5000}/socket.io`
        });
        
        logger.info('[CommandExecutor] Auto-initialized');
    }
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = QTCLCommandExecutor;
}
