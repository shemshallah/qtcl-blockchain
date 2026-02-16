/**
 * COMMAND EXECUTOR - FULLY INTEGRATED WITH GLOBALS SYSTEM
 * Executes commands via /api/command endpoint
 */

class QTCLCommandExecutor {
    constructor() {
        this.history = [];
        this.isConnected = true;
        this.listeners = {};
        console.log('[Executor] ✓ Initialized - ready to execute commands');
    }
    
    async execute(command, args=[], kwargs={}) {
        console.log('[Executor] Executing:', command, {args, kwargs});
        
        try {
            const response = await fetch('/api/command', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('authToken') || ''}`
                },
                body: JSON.stringify({ 
                    command,
                    args: args || [],
                    kwargs: kwargs || {}
                })
            });
            
            const data = await response.json();
            console.log('[Executor] Response:', data);
            
            // Handle successful response
            if (response.ok) {
                this.history.push({
                    command,
                    timestamp: new Date(),
                    result: data
                });
                
                return {
                    status: data.status || 'success',
                    result: data.result || data.output,
                    command: command,
                    error: data.error || null,
                    raw: data
                };
            } else {
                // Handle error response
                return {
                    status: 'error',
                    error: data.error || 'Command execution failed',
                    command: command,
                    raw: data
                };
            }
        } catch (error) {
            console.error('[Executor] Error:', error);
            return {
                status: 'error',
                error: error.message,
                command: command
            };
        }
    }
    
    async executeRaw(commandString) {
        // Parse command string like "quantum/status arg1 arg2"
        const parts = commandString.trim().split(/\s+/);
        const command = parts[0];
        const args = parts.slice(1);
        return this.execute(command, args, {});
    }
    
    on(event, callback) {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].push(callback);
    }
    
    emit(event, data) {
        if (this.listeners[event]) {
            this.listeners[event].forEach(cb => cb(data));
        }
    }
    
    getHistory(limit=100) {
        return this.history.slice(-limit);
    }
    
    clearHistory() {
        this.history = [];
    }
}

// Create global instance
window.qtclExecutor = new QTCLCommandExecutor();
console.log('[CommandExecutor] ✓ Global executor ready at window.qtclExecutor');
                error: error.message,
                command: command,
                output: null
            };
        }
    }
    
    on(event, callback) {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].push(callback);
    }
    
    emit(event, data) {
        if (this.listeners[event]) {
            this.listeners[event].forEach(cb => {
                try { cb(data); } catch (e) { console.error(e); }
            });
        }
    }
}

// Initialize immediately
console.log('[Main] Initializing executor...');
window.commandExecutor = new QTCLCommandExecutor();
console.log('[Main] ✓ window.commandExecutor is ready');