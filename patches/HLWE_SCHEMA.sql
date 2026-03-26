-- HLWE-ONLY POSTGRES SCHEMA - MINIMAL & CLEAN
-- ═══════════════════════════════════════════════════════════════════════════════════════════════

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Drop old tables if they exist (clean slate)
DROP TABLE IF EXISTS user_private_keys CASCADE;
DROP TABLE IF EXISTS users CASCADE;
DROP TABLE IF EXISTS hlwe_audit_log CASCADE;
DROP TABLE IF EXISTS login_attempts CASCADE;
DROP TABLE IF EXISTS user_sessions CASCADE;
DROP TABLE IF EXISTS encryption_keys CASCADE;

-- ═══════════════════════════════════════════════════════════════════════════════════════════════
-- 1. USERS TABLE
-- ═══════════════════════════════════════════════════════════════════════════════════════════════

CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(255) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    password_method VARCHAR(50) NOT NULL DEFAULT 'hlwe' CHECK (password_method = 'hlwe'),
    role VARCHAR(50) DEFAULT 'user',
    status VARCHAR(50) DEFAULT 'active',
    pseudoqubit_id INT,
    pseudoqubit_registered BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMP WITH TIME ZONE,
    last_password_change_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_status ON users(status);

-- ═══════════════════════════════════════════════════════════════════════════════════════════════
-- 2. USER PRIVATE KEYS
-- ═══════════════════════════════════════════════════════════════════════════════════════════════

CREATE TABLE user_private_keys (
    key_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    private_key_json TEXT NOT NULL,
    private_key_encrypted BYTEA,
    encryption_key_id UUID,
    key_version INT DEFAULT 1,
    status VARCHAR(50) DEFAULT 'active',
    algorithm VARCHAR(50) DEFAULT 'hlwe_128',
    key_strength INT DEFAULT 256,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    rotated_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    revoked_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_user_private_keys_user_id ON user_private_keys(user_id);
CREATE INDEX idx_user_private_keys_status ON user_private_keys(status);

-- ═══════════════════════════════════════════════════════════════════════════════════════════════
-- 3. AUDIT LOG
-- ═══════════════════════════════════════════════════════════════════════════════════════════════

CREATE TABLE hlwe_audit_log (
    audit_id VARCHAR(16) PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    event_type VARCHAR(50) NOT NULL,
    user_id UUID REFERENCES users(user_id) ON DELETE SET NULL,
    username VARCHAR(255),
    details JSONB,
    severity VARCHAR(20),
    previous_hash VARCHAR(64),
    entry_hash VARCHAR(64) NOT NULL UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_hlwe_audit_log_timestamp ON hlwe_audit_log(timestamp);
CREATE INDEX idx_hlwe_audit_log_user_id ON hlwe_audit_log(user_id);
CREATE INDEX idx_hlwe_audit_log_event_type ON hlwe_audit_log(event_type);

-- ═══════════════════════════════════════════════════════════════════════════════════════════════
-- 4. LOGIN ATTEMPTS
-- ═══════════════════════════════════════════════════════════════════════════════════════════════

CREATE TABLE login_attempts (
    attempt_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
    username VARCHAR(255),
    attempt_type VARCHAR(50),
    success BOOLEAN DEFAULT FALSE,
    ip_address INET,
    user_agent VARCHAR(500),
    attempted_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_login_attempts_user_id ON login_attempts(user_id);
CREATE INDEX idx_login_attempts_username ON login_attempts(username);

-- ═══════════════════════════════════════════════════════════════════════════════════════════════
-- 5. USER SESSIONS
-- ═══════════════════════════════════════════════════════════════════════════════════════════════

CREATE TABLE user_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    token TEXT,
    refresh_token TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);

-- ═══════════════════════════════════════════════════════════════════════════════════════════════
-- 6. ENCRYPTION KEYS
-- ═══════════════════════════════════════════════════════════════════════════════════════════════

CREATE TABLE encryption_keys (
    key_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_name VARCHAR(255) NOT NULL UNIQUE,
    key_version INT DEFAULT 1,
    key_material BYTEA NOT NULL,
    algorithm VARCHAR(50) DEFAULT 'aes-256-gcm',
    key_strength INT DEFAULT 256,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    rotated_at TIMESTAMP WITH TIME ZONE
);

-- ═══════════════════════════════════════════════════════════════════════════════════════════════
-- 7. TRIGGER
-- ═══════════════════════════════════════════════════════════════════════════════════════════════

CREATE OR REPLACE FUNCTION validate_hlwe_envelope()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.password_method != 'hlwe' THEN
        RAISE EXCEPTION 'SECURITY VIOLATION: Non-HLWE password method detected';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_validate_hlwe_envelope ON users;

CREATE TRIGGER trigger_validate_hlwe_envelope
BEFORE INSERT OR UPDATE ON users
FOR EACH ROW
EXECUTE FUNCTION validate_hlwe_envelope();

-- ═══════════════════════════════════════════════════════════════════════════════════════════════
-- 8. VERIFICATION
-- ═══════════════════════════════════════════════════════════════════════════════════════════════

-- SELECT 'Schema created successfully!' as status;
-- SELECT * FROM information_schema.tables WHERE table_name IN ('users', 'user_private_keys', 'hlwe_audit_log');
