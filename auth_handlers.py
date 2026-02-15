"""
AUTH COMMAND HANDLERS - PRODUCTION-GRADE AUTHENTICATION SYSTEM
Implements login, logout, register, verify, refresh with JWT, hashing, validation
"""

import os
import sys
import jwt
import bcrypt
import secrets
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════════════════════════════

JWT_SECRET = os.getenv('JWT_SECRET', secrets.token_urlsafe(64))
JWT_ALGORITHM = 'HS512'
JWT_EXPIRATION_HOURS = int(os.getenv('JWT_EXPIRATION_HOURS', '24'))
TOKEN_REFRESH_WINDOW = 1  # Hours before expiry to allow refresh
PASSWORD_MIN_LENGTH = 8
PASSWORD_REGEX = r'^(?=.*[A-Za-z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!%*#?&]{8,}$'
EMAIL_REGEX = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

# ════════════════════════════════════════════════════════════════════════════════════════════════════════
# DATABASE CONNECTION
# ════════════════════════════════════════════════════════════════════════════════════════════════════════

class AuthDB:
    """Database operations for authentication"""
    
    _conn = None
    
    @classmethod
    def get_connection(cls):
        """Get or create database connection"""
        if cls._conn is None:
            try:
                cls._conn = psycopg2.connect(
                    host=os.getenv('SUPABASE_HOST'),
                    user=os.getenv('SUPABASE_USER'),
                    password=os.getenv('SUPABASE_PASSWORD'),
                    database=os.getenv('SUPABASE_DB'),
                    port=int(os.getenv('SUPABASE_PORT', 5432))
                )
                logger.info("[AuthDB] ✓ Connection established")
            except Exception as e:
                logger.error(f"[AuthDB] ✗ Connection failed: {e}")
                raise
        return cls._conn
    
    @classmethod
    def execute(cls, query: str, params: tuple = ()) -> Optional[list]:
        """Execute query safely"""
        try:
            conn = cls.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(query, params)
            result = cur.fetchall() if cur.description else None
            conn.commit()
            cur.close()
            return result
        except Exception as e:
            logger.error(f"[AuthDB] Query error: {e}")
            if cls._conn:
                cls._conn.rollback()
            raise
    
    @classmethod
    def fetch_one(cls, query: str, params: tuple = ()) -> Optional[dict]:
        """Fetch single row"""
        result = cls.execute(query, params)
        return result[0] if result else None
    
    @classmethod
    def fetch_all(cls, query: str, params: tuple = ()) -> list:
        """Fetch all rows"""
        result = cls.execute(query, params)
        return result or []

# ════════════════════════════════════════════════════════════════════════════════════════════════════════
# VALIDATION HELPERS
# ════════════════════════════════════════════════════════════════════════════════════════════════════════

class ValidationError(Exception):
    """Validation error"""
    pass

def validate_email(email: str) -> str:
    """Validate and normalize email"""
    if not email:
        raise ValidationError("Email is required")
    
    email = email.strip().lower()
    if not re.match(EMAIL_REGEX, email):
        raise ValidationError(f"Invalid email format: {email}")
    
    if len(email) > 255:
        raise ValidationError("Email is too long")
    
    return email

def validate_password(password: str) -> str:
    """Validate password strength"""
    if not password:
        raise ValidationError("Password is required")
    
    if len(password) < PASSWORD_MIN_LENGTH:
        raise ValidationError(f"Password must be at least {PASSWORD_MIN_LENGTH} characters")
    
    if not re.match(PASSWORD_REGEX, password):
        raise ValidationError(
            "Password must contain: letters, numbers, and special characters (@$!%*#?&)"
        )
    
    return password

def validate_username(username: str) -> str:
    """Validate username"""
    if not username:
        raise ValidationError("Username is required")
    
    username = username.strip()
    
    if len(username) < 3:
        raise ValidationError("Username must be at least 3 characters")
    
    if len(username) > 50:
        raise ValidationError("Username must be less than 50 characters")
    
    if not re.match(r'^[a-zA-Z0-9_-]+$', username):
        raise ValidationError("Username can only contain letters, numbers, hyphens, and underscores")
    
    return username

def hash_password(password: str) -> str:
    """Hash password with bcrypt"""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against hash"""
    return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))

# ════════════════════════════════════════════════════════════════════════════════════════════════════════
# JWT TOKEN MANAGEMENT
# ════════════════════════════════════════════════════════════════════════════════════════════════════════

class TokenManager:
    """JWT token creation and validation"""
    
    @staticmethod
    def create_token(user_id: str, email: str, username: str) -> str:
        """Create JWT token"""
        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=JWT_EXPIRATION_HOURS)
        
        payload = {
            'user_id': user_id,
            'email': email,
            'username': username,
            'iat': now.timestamp(),
            'exp': expires.timestamp(),
            'type': 'access'
        }
        
        token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
        logger.info(f"[TokenManager] ✓ Token created for {email}")
        return token
    
    @staticmethod
    def verify_token(token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            logger.info(f"[TokenManager] ✓ Token verified for {payload.get('email')}")
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("[TokenManager] Token expired")
            raise ValidationError("Token expired")
        except jwt.InvalidTokenError as e:
            logger.warning(f"[TokenManager] Invalid token: {e}")
            raise ValidationError("Invalid token")
    
    @staticmethod
    def get_token_expiry(token: str) -> Optional[datetime]:
        """Get token expiry time"""
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM], options={"verify_exp": False})
            exp_timestamp = payload.get('exp')
            if exp_timestamp:
                return datetime.fromtimestamp(exp_timestamp, tz=timezone.utc)
        except Exception as e:
            logger.error(f"[TokenManager] Error getting expiry: {e}")
        return None

# ════════════════════════════════════════════════════════════════════════════════════════════════════════
# USER MANAGEMENT
# ════════════════════════════════════════════════════════════════════════════════════════════════════════

class UserManager:
    """User lookup and creation"""
    
    @staticmethod
    def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
        """Get user by email"""
        query = "SELECT id, email, username, password_hash, created_at, is_verified FROM users WHERE email = %s"
        return AuthDB.fetch_one(query, (email,))
    
    @staticmethod
    def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        query = "SELECT id, email, username, created_at, is_verified FROM users WHERE id = %s"
        return AuthDB.fetch_one(query, (user_id,))
    
    @staticmethod
    def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
        """Get user by username"""
        query = "SELECT id, email, username, created_at FROM users WHERE username = %s"
        return AuthDB.fetch_one(query, (username,))
    
    @staticmethod
    def create_user(email: str, username: str, password_hash: str) -> Dict[str, Any]:
        """Create new user"""
        user_id = secrets.token_urlsafe(16)
        now = datetime.now(timezone.utc)
        
        query = """
            INSERT INTO users (id, email, username, password_hash, created_at, is_verified)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id, email, username, created_at
        """
        
        result = AuthDB.fetch_one(query, (user_id, email, username, password_hash, now, False))
        logger.info(f"[UserManager] ✓ User created: {email}")
        return result
    
    @staticmethod
    def list_users(limit: int = 100, offset: int = 0) -> Tuple[list, int]:
        """List all users with pagination"""
        query = "SELECT id, email, username, created_at, is_verified FROM users ORDER BY created_at DESC LIMIT %s OFFSET %s"
        users = AuthDB.fetch_all(query, (limit, offset))
        
        count_query = "SELECT COUNT(*) as total FROM users"
        count_result = AuthDB.fetch_one(count_query)
        total = count_result['total'] if count_result else 0
        
        return users, total
    
    @staticmethod
    def verify_user(user_id: str) -> bool:
        """Mark user as verified"""
        query = "UPDATE users SET is_verified = TRUE WHERE id = %s RETURNING id"
        result = AuthDB.fetch_one(query, (user_id,))
        return result is not None

# ════════════════════════════════════════════════════════════════════════════════════════════════════════
# SESSION MANAGEMENT
# ════════════════════════════════════════════════════════════════════════════════════════════════════════

class SessionManager:
    """User session tracking"""
    
    @staticmethod
    def create_session(user_id: str, ip_address: str = None, user_agent: str = None) -> str:
        """Create new session"""
        session_id = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=JWT_EXPIRATION_HOURS)
        
        query = """
            INSERT INTO sessions (id, user_id, ip_address, user_agent, created_at, expires_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        
        result = AuthDB.fetch_one(query, (session_id, user_id, ip_address, user_agent, now, expires))
        logger.info(f"[SessionManager] ✓ Session created: {session_id}")
        return session_id if result else None
    
    @staticmethod
    def get_session(session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID"""
        query = "SELECT id, user_id, created_at, expires_at FROM sessions WHERE id = %s AND expires_at > NOW()"
        return AuthDB.fetch_one(query, (session_id,))
    
    @staticmethod
    def invalidate_session(session_id: str) -> bool:
        """Invalidate session (logout)"""
        query = "UPDATE sessions SET expires_at = NOW() WHERE id = %s RETURNING id"
        result = AuthDB.fetch_one(query, (session_id,))
        logger.info(f"[SessionManager] ✓ Session invalidated: {session_id}")
        return result is not None
    
    @staticmethod
    def list_user_sessions(user_id: str) -> list:
        """List all active sessions for user"""
        query = "SELECT id, created_at, ip_address FROM sessions WHERE user_id = %s AND expires_at > NOW() ORDER BY created_at DESC"
        return AuthDB.fetch_all(query, (user_id,))

# ════════════════════════════════════════════════════════════════════════════════════════════════════════
# AUTH COMMAND HANDLERS
# ════════════════════════════════════════════════════════════════════════════════════════════════════════

class AuthCommandHandlers:
    """Production-grade auth command implementations"""
    
    @staticmethod
    def auth_login(email: str = None, password: str = None, **kwargs) -> Dict[str, Any]:
        """
        LOGIN: Authenticate user with email and password
        Returns: {status, token, user_id, email, message}
        """
        logger.info(f"[Auth/Login] Attempting login for: {email}")
        
        try:
            # Validation
            if not email or not password:
                raise ValidationError("Email and password required")
            
            email = validate_email(email)
            
            # Get user
            user = UserManager.get_user_by_email(email)
            if not user:
                logger.warning(f"[Auth/Login] User not found: {email}")
                return {
                    'status': 'error',
                    'error': 'Invalid email or password',
                    'code': 'AUTH_FAILED'
                }
            
            # Verify password
            if not verify_password(password, user['password_hash']):
                logger.warning(f"[Auth/Login] Invalid password for: {email}")
                return {
                    'status': 'error',
                    'error': 'Invalid email or password',
                    'code': 'AUTH_FAILED'
                }
            
            # Create token
            token = TokenManager.create_token(user['id'], user['email'], user['username'])
            
            # Create session
            session_id = SessionManager.create_session(user['id'])
            
            logger.info(f"[Auth/Login] ✓ Login successful: {email}")
            
            return {
                'status': 'success',
                'result': f'Login successful for {user["username"]}',
                'token': token,
                'session_id': session_id,
                'user_id': user['id'],
                'email': user['email'],
                'username': user['username'],
                'verified': user['is_verified']
            }
        
        except ValidationError as e:
            logger.warning(f"[Auth/Login] Validation error: {e}")
            return {'status': 'error', 'error': str(e), 'code': 'VALIDATION_ERROR'}
        except Exception as e:
            logger.error(f"[Auth/Login] Error: {e}", exc_info=True)
            return {'status': 'error', 'error': 'Login failed', 'code': 'SERVER_ERROR'}
    
    @staticmethod
    def auth_register(email: str = None, username: str = None, password: str = None, **kwargs) -> Dict[str, Any]:
        """
        REGISTER: Create new user account
        Returns: {status, user_id, email, message}
        """
        logger.info(f"[Auth/Register] Registration attempt for: {email}")
        
        try:
            # Validate inputs
            email = validate_email(email)
            username = validate_username(username)
            password = validate_password(password)
            
            # Check if email exists
            existing_user = UserManager.get_user_by_email(email)
            if existing_user:
                logger.warning(f"[Auth/Register] Email already registered: {email}")
                return {
                    'status': 'error',
                    'error': 'Email already registered',
                    'code': 'EMAIL_EXISTS'
                }
            
            # Check if username exists
            existing_username = UserManager.get_user_by_username(username)
            if existing_username:
                logger.warning(f"[Auth/Register] Username already taken: {username}")
                return {
                    'status': 'error',
                    'error': 'Username already taken',
                    'code': 'USERNAME_EXISTS'
                }
            
            # Hash password
            password_hash = hash_password(password)
            
            # Create user
            user = UserManager.create_user(email, username, password_hash)
            
            # Create verification token
            verify_token = TokenManager.create_token(user['id'], user['email'], user['username'])
            
            logger.info(f"[Auth/Register] ✓ Registration successful: {email}")
            
            return {
                'status': 'success',
                'result': f'Registration successful. Welcome {username}!',
                'user_id': user['id'],
                'email': user['email'],
                'username': user['username'],
                'verification_token': verify_token,
                'message': 'Please verify your email to activate your account'
            }
        
        except ValidationError as e:
            logger.warning(f"[Auth/Register] Validation error: {e}")
            return {'status': 'error', 'error': str(e), 'code': 'VALIDATION_ERROR'}
        except Exception as e:
            logger.error(f"[Auth/Register] Error: {e}", exc_info=True)
            return {'status': 'error', 'error': 'Registration failed', 'code': 'SERVER_ERROR'}
    
    @staticmethod
    def auth_logout(token: str = None, session_id: str = None, **kwargs) -> Dict[str, Any]:
        """
        LOGOUT: Invalidate session and token
        Returns: {status, message}
        """
        logger.info("[Auth/Logout] Logout attempt")
        
        try:
            if not session_id:
                raise ValidationError("Session ID required")
            
            # Invalidate session
            SessionManager.invalidate_session(session_id)
            
            logger.info(f"[Auth/Logout] ✓ Logout successful: {session_id}")
            
            return {
                'status': 'success',
                'result': 'Logout successful. Session invalidated.'
            }
        
        except ValidationError as e:
            logger.warning(f"[Auth/Logout] Validation error: {e}")
            return {'status': 'error', 'error': str(e), 'code': 'VALIDATION_ERROR'}
        except Exception as e:
            logger.error(f"[Auth/Logout] Error: {e}", exc_info=True)
            return {'status': 'error', 'error': 'Logout failed', 'code': 'SERVER_ERROR'}
    
    @staticmethod
    def auth_verify(token: str = None, user_id: str = None, **kwargs) -> Dict[str, Any]:
        """
        VERIFY: Verify user email and activate account
        Returns: {status, verified, message}
        """
        logger.info("[Auth/Verify] Email verification attempt")
        
        try:
            if not token:
                raise ValidationError("Verification token required")
            
            # Verify token
            payload = TokenManager.verify_token(token)
            user_id = payload.get('user_id')
            
            # Get user
            user = UserManager.get_user_by_id(user_id)
            if not user:
                raise ValidationError("User not found")
            
            if user['is_verified']:
                return {
                    'status': 'success',
                    'result': 'Email already verified',
                    'verified': True
                }
            
            # Mark as verified
            UserManager.verify_user(user_id)
            
            logger.info(f"[Auth/Verify] ✓ Email verified: {user['email']}")
            
            return {
                'status': 'success',
                'result': f'Email verified successfully. Account activated for {user["username"]}',
                'verified': True,
                'email': user['email'],
                'username': user['username']
            }
        
        except ValidationError as e:
            logger.warning(f"[Auth/Verify] Validation error: {e}")
            return {'status': 'error', 'error': str(e), 'code': 'VALIDATION_ERROR'}
        except Exception as e:
            logger.error(f"[Auth/Verify] Error: {e}", exc_info=True)
            return {'status': 'error', 'error': 'Verification failed', 'code': 'SERVER_ERROR'}
    
    @staticmethod
    def auth_refresh(token: str = None, **kwargs) -> Dict[str, Any]:
        """
        REFRESH: Refresh access token before expiry
        Returns: {status, new_token, expires_in}
        """
        logger.info("[Auth/Refresh] Token refresh attempt")
        
        try:
            if not token:
                raise ValidationError("Token required")
            
            # Verify current token
            payload = TokenManager.verify_token(token)
            
            # Check if token is close to expiry
            exp_time = datetime.fromtimestamp(payload['exp'], tz=timezone.utc)
            now = datetime.now(timezone.utc)
            time_to_expiry = (exp_time - now).total_seconds() / 3600
            
            if time_to_expiry > TOKEN_REFRESH_WINDOW:
                return {
                    'status': 'error',
                    'error': f'Token not yet eligible for refresh. Refresh in {time_to_expiry - TOKEN_REFRESH_WINDOW:.1f} hours',
                    'code': 'TOKEN_NOT_ELIGIBLE'
                }
            
            # Create new token
            new_token = TokenManager.create_token(
                payload['user_id'],
                payload['email'],
                payload['username']
            )
            
            logger.info(f"[Auth/Refresh] ✓ Token refreshed for: {payload['email']}")
            
            return {
                'status': 'success',
                'result': 'Token refreshed successfully',
                'new_token': new_token,
                'expires_in': f'{JWT_EXPIRATION_HOURS} hours'
            }
        
        except ValidationError as e:
            logger.warning(f"[Auth/Refresh] Validation error: {e}")
            return {'status': 'error', 'error': str(e), 'code': 'VALIDATION_ERROR'}
        except Exception as e:
            logger.error(f"[Auth/Refresh] Error: {e}", exc_info=True)
            return {'status': 'error', 'error': 'Token refresh failed', 'code': 'SERVER_ERROR'}

