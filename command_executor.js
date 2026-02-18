/**
 * ╔══════════════════════════════════════════════════════════════════════╗
 * ║  QTCL COMMAND EXECUTOR v2.0 — HYPHEN-NATIVE, FLAG-AWARE            ║
 * ║                                                                      ║
 * ║  Standard command format: category-command --flag=value --bool-flag  ║
 * ║  Examples:                                                           ║
 * ║    quantum-status                                                     ║
 * ║    oracle-price --symbol=BTCUSD                                       ║
 * ║    admin-users --limit=20 --role=admin                               ║
 * ║    login --email=user@x.com --password=secret                        ║
 * ║    block-details --block=42                                           ║
 * ║                                                                      ║
 * ║  Slash notation auto-converted: quantum/status → quantum-status      ║
 * ║  Positional args supported:    block-details 42  (becomes --block=42)║
 * ╚══════════════════════════════════════════════════════════════════════╝
 */

class QTCLCommandExecutor {
    constructor() {
        this.history    = [];          // { command, timestamp, result, duration }
        this.authToken  = null;        // set via setToken() after login
        this.baseUrl    = '';          // same origin by default
        this.maxHistory = 500;
        this._listeners = {};
        console.log('[Executor] ✓ QTCL CommandExecutor v2.0 — hyphen-native, flag-aware');
    }

    // ── AUTH ─────────────────────────────────────────────────────────────────

    setToken(token) {
        this.authToken = token;
        try { localStorage.setItem('qtcl_auth_token', token); } catch {}
    }

    getToken() {
        if (this.authToken) return this.authToken;
        try { return localStorage.getItem('qtcl_auth_token') || ''; } catch { return ''; }
    }

    clearToken() {
        this.authToken = null;
        try { localStorage.removeItem('qtcl_auth_token'); } catch {}
    }

    // ── PARSE ────────────────────────────────────────────────────────────────

    /**
     * Parse a raw command string into { command, args, kwargs }.
     *
     * Handles:
     *   category/command      → category-command   (slash → hyphen)
     *   category-command      → as-is
     *   --key=value           → kwargs.key = value
     *   --bool-flag           → kwargs.bool_flag = true
     *   positionalArg         → args array
     *
     * @param {string} raw  e.g. "oracle-price --symbol=BTCUSD --verbose"
     * @returns {{ command: string, args: string[], kwargs: Object }}
     */
    parse(raw) {
        const tokens = raw.trim().split(/\s+/).filter(Boolean);
        if (!tokens.length) return { command: '', args: [], kwargs: {} };

        // Normalise slash → hyphen in command name only
        const command = tokens[0].toLowerCase().replace(/\//g, '-');
        const args    = [];
        const kwargs  = {};

        for (let i = 1; i < tokens.length; i++) {
            const tok = tokens[i];
            // --key=value  or  --key (boolean flag)
            const flagMatch = tok.match(/^--([a-zA-Z0-9_-]+)(?:=(.+))?$/);
            if (flagMatch) {
                // Normalise key: hyphens → underscores to match Python flag names
                const key = flagMatch[1].replace(/-/g, '_');
                kwargs[key] = flagMatch[2] !== undefined ? flagMatch[2] : true;
            } else {
                args.push(tok);
            }
        }

        return { command, args, kwargs };
    }

    /**
     * Build the final command string to send to /api/command.
     * Backend dispatch_command() parses inline flags, so we embed them all
     * back into one string — this is the most robust approach regardless of
     * which kwargs the specific handler reads.
     *
     * @param {string} command  hyphen-command name
     * @param {string[]} args   positional args
     * @param {Object} kwargs   flag map
     * @returns {string}        e.g. "oracle-price --symbol=BTCUSD"
     */
    buildCommandString(command, args, kwargs) {
        let parts = [command];
        // Positional args first (handlers read them from args[])
        parts = parts.concat(args);
        // Then flags
        for (const [k, v] of Object.entries(kwargs)) {
            if (v === true || v === '') {
                parts.push(`--${k}`);
            } else if (v !== false && v !== null && v !== undefined) {
                parts.push(`--${k}=${v}`);
            }
        }
        return parts.join(' ');
    }

    // ── EXECUTE ──────────────────────────────────────────────────────────────

    /**
     * Execute a parsed command.
     *
     * @param {string} command     Hyphen-separated command name
     * @param {string[]} args      Positional arguments
     * @param {Object} kwargs      Flag key→value map
     * @returns {Promise<Object>}  { status, result?, error?, suggestions?, raw }
     */
    async execute(command, args = [], kwargs = {}) {
        // Build the canonical command string (backend parses inline flags)
        const cmdStr = this.buildCommandString(command, args, kwargs);
        const t0 = performance.now();

        console.log(`[Executor] → ${cmdStr}`, { args, kwargs });
        this.emit('before-execute', { command, args, kwargs, cmdStr });

        const token = this.getToken();
        const headers = { 'Content-Type': 'application/json' };
        if (token) headers['Authorization'] = `Bearer ${token}`;

        let result;
        try {
            const resp = await fetch(`${this.baseUrl}/api/command`, {
                method:  'POST',
                headers,
                body: JSON.stringify({
                    command: cmdStr,
                    args:    [],     // args are embedded in cmdStr above
                    kwargs:  {}      // kwargs are embedded in cmdStr above
                })
            });

            const data = await resp.json().catch(() => ({
                status: 'error', error: 'Invalid JSON from server'
            }));

            const duration = Math.round(performance.now() - t0);

            if (resp.ok && (data.status === 'success' || data.result !== undefined)) {
                result = {
                    status:   'success',
                    result:   data.result ?? data.output ?? null,
                    command:  cmdStr,
                    duration,
                    raw:      data
                };
                // Detect auth token in login responses
                const tok = data.result?.token || data.result?.access_token || data.token;
                if (tok) this.setToken(tok);
            } else {
                result = {
                    status:      'error',
                    error:       data.error || `HTTP ${resp.status}`,
                    suggestions: data.suggestions || [],
                    command:     cmdStr,
                    duration,
                    raw:         data
                };
            }
        } catch (err) {
            const duration = Math.round(performance.now() - t0);
            console.error('[Executor] Network error:', err);
            result = {
                status:   'error',
                error:    `Network error: ${err.message}`,
                command:  cmdStr,
                duration,
                raw:      null
            };
        }

        // Record history
        this.history.unshift({ command: cmdStr, timestamp: new Date(), result, duration: result.duration });
        if (this.history.length > this.maxHistory) this.history.length = this.maxHistory;

        console.log(`[Executor] ← ${cmdStr} (${result.duration}ms) ${result.status}`, result);
        this.emit('after-execute', result);
        return result;
    }

    /**
     * Parse a raw string then execute — the main entry point for the terminal.
     *
     * @param {string} rawString  e.g. "oracle-price --symbol=BTCUSD"
     * @returns {Promise<Object>}
     */
    async executeRaw(rawString) {
        const { command, args, kwargs } = this.parse(rawString);
        if (!command) {
            return { status: 'error', error: 'Empty command', command: '' };
        }
        return this.execute(command, args, kwargs);
    }

    /**
     * Execute many commands in parallel.
     *
     * @param {string[]} rawStrings
     * @returns {Promise<Object[]>}
     */
    async executeBatch(rawStrings) {
        return Promise.all(rawStrings.map(s => this.executeRaw(s)));
    }

    // ── HELPERS ──────────────────────────────────────────────────────────────

    /**
     * Fetch the full command registry from the backend.
     * Returns array of { command, category, description, requires_auth, requires_admin }
     */
    async fetchRegistry() {
        try {
            const r = await fetch(`${this.baseUrl}/api/commands`);
            const d = await r.json();
            return d.commands || [];
        } catch {
            return [];
        }
    }

    /** Get the last N history entries. */
    getHistory(limit = 50) {
        return this.history.slice(0, limit);
    }

    clearHistory() {
        this.history = [];
    }

    // ── EVENTS ───────────────────────────────────────────────────────────────

    on(event, cb)   { (this._listeners[event] = this._listeners[event] || []).push(cb); }
    off(event, cb)  { this._listeners[event] = (this._listeners[event] || []).filter(f => f !== cb); }
    emit(event, data) { (this._listeners[event] || []).forEach(cb => { try { cb(data); } catch {} }); }
}

// ── Singleton ─────────────────────────────────────────────────────────────────
window.qtclExecutor = new QTCLCommandExecutor();
console.log('[CommandExecutor] ✓ window.qtclExecutor ready');
