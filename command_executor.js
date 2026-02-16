/**
 * ULTRA-SIMPLE COMMAND EXECUTOR - WITH INTERACTIVE PROMPT SUPPORT
 * Sends commands to /api/execute and displays output
 * Handles interactive input_prompt responses by rendering input boxes
 */

class QTCLCommandExecutor {
    constructor() {
        this.history = [];
        this.isConnected = true;
        this.listeners = {};
        this.currentInputPrompt = null;
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
                console.log('[Executor] Interactive prompt detected:', data.input_prompt);
                this.currentInputPrompt = { command, data };
                return {
                    status: 'input_prompt',
                    input_prompt: data.input_prompt,
                    progress: data.progress,
                    command: command,
                    interactive: true
                };
            }
            
            return {
                status: data.status || 'success',
                output: data.output || JSON.stringify(data),
                command: command,
                error: data.error || null,
                result: data.result || data
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
    
    submitPromptInput(value) {
        if (!this.currentInputPrompt) return;
        
        const { command, data } = this.currentInputPrompt;
        const prompt = data.input_prompt;
        
        // Build new command with the submitted value
        const newParams = [...prompt.current_params, `--${prompt.field_name}=${value}`];
        const newCommand = `${command.split(' ')[0]} ${newParams.join(' ')}`;
        
        console.log('[Executor] Submitting prompt response:', newCommand);
        return this.execute(newCommand);
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