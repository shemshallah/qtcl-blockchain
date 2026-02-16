/**
 * ULTRA-SIMPLE COMMAND EXECUTOR - WITH INPUT PROMPT DETECTION
 * Just send commands to /api/execute and display output
 */

class QTCLCommandExecutor {
    constructor() {
        this.history = [];
        this.isConnected = true;
        this.listeners = {};
        console.log('[Executor] ✓ Initialized');
    }
    
    async execute(command) {
        console.log('[Executor] Executing:', command);
        
        try {
            const response = await fetch('/api/execute', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('authToken') || ''}`
                },
                body: JSON.stringify({ command })
            });
            
            const data = await response.json();
            console.log('[Executor] Response:', data);
            
            // Check if this is an interactive input prompt
            if (data.input_prompt && data.status === 'collecting_input') {
                console.log('[Executor] Input prompt detected');
                return {
                    status: 'input_prompt',
                    input_prompt: data.input_prompt,
                    progress: data.progress,
                    command: command
                };
            }
            
            return {
                status: data.status || 'success',
                output: data.output || JSON.stringify(data),
                command: command,
                error: data.error || null
            };
        } catch (error) {
            console.error('[Executor] Error:', error);
            return {
                status: 'error',
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