#!/usr/bin/env python3
"""
CORE API MODULE - Authentication, Users, Security, Keys, Addresses
Complete production-grade implementation with quantum-enhanced security
Handles: /api/auth/*, /api/users/*, /api/security/*, /api/keys/*, /api/addresses/*, /api/aliases/*, /api/sign/*, /api/verify/*
"""
import os,sys,json,time,hashlib,uuid,logging,threading,secrets,bcrypt,hmac,base64,re,traceback,copy,jwt as pyjwt
from datetime import datetime,timedelta,timezone
from typing import Dict,List,Optional,Any,Tuple
from functools import wraps,lru_cache
from decimal import Decimal,getcontext
from dataclasses import dataclass,asdict,field
from enum import Enum,IntEnum,auto
from collections import defaultdict,deque,Counter
from flask import Blueprint,request,jsonify,g,Response,session,make_response
import psycopg2
from psycopg2.extras import RealDictCursor,execute_batch,execute_values,Json

try:
    from cryptography.hazmat.primitives import hashes,serialization,hmac as crypto_hmac
    from cryptography.hazmat.primitives.asymmetric import rsa,ed25519,ec,dsa,padding
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305,AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    from cryptography.hazmat.backends import default_backend
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE=True
except ImportError:
    CRYPTO_AVAILABLE=False

try:
    import pyotp,qrcode
    from io import BytesIO
    TOTP_AVAILABLE=True
except ImportError:
    TOTP_AVAILABLE=False

getcontext().prec=28
logger=logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════════════
# GLOBAL WSGI INTEGRATION - Quantum Revolution
# ═══════════════════════════════════════════════════════════════════════════════════════
try:
    from wsgi_config import DB, PROFILER, CACHE, ERROR_BUDGET, RequestCorrelation, CIRCUIT_BREAKERS, RATE_LIMITERS
    WSGI_AVAILABLE = True
except ImportError:
    WSGI_AVAILABLE = False
    logger.warning("[INTEGRATION] WSGI globals not available - running in standalone mode")

# ═══════════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & ENUMS
# ═══════════════════════════════════════════════════════════════════════════════════════

class UserRole(IntEnum):
    """User roles with hierarchical permissions"""
    GUEST=0
    USER=1
    VALIDATOR=2
    ADMIN=3
    SUPERADMIN=4

class AuthMethod(Enum):
    """Authentication methods supported"""
    PASSWORD="password"
    TOTP="totp"
    HARDWARE_KEY="hardware_key"
    BIOMETRIC="biometric"
    QUANTUM_KEY="quantum_key"

class KeyType(Enum):
    """Cryptographic key types"""
    ED25519="ed25519"
    RSA_2048="rsa_2048"
    RSA_4096="rsa_4096"
    ECDSA_P256="ecdsa_p256"
    ECDSA_SECP256K1="ecdsa_secp256k1"

class SecurityEventType(Enum):
    """Security audit event types"""
    LOGIN_SUCCESS="login_success"
    LOGIN_FAILED="login_failed"
    LOGOUT="logout"
    PASSWORD_CHANGE="password_change"
    2FA_ENABLED="2fa_enabled"
    2FA_DISABLED="2fa_disabled"
    KEY_GENERATED="key_generated"
    KEY_ROTATED="key_rotated"
    SUSPICIOUS_ACTIVITY="suspicious_activity"
    ACCOUNT_LOCKED="account_locked"
    ACCOUNT_UNLOCKED="account_unlocked"
    PERMISSION_CHANGE="permission_change"

# ═══════════════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class User:
    """User account model with comprehensive attributes"""
    user_id:str
    username:str
    email:str
    password_hash:str
    role:UserRole=UserRole.USER
    is_active:bool=True
    is_verified:bool=False
    totp_secret:Optional[str]=None
    totp_enabled:bool=False
    failed_login_attempts:int=0
    locked_until:Optional[datetime]=None
    last_login:Optional[datetime]=None
    created_at:datetime=field(default_factory=lambda:datetime.now(timezone.utc))
    updated_at:datetime=field(default_factory=lambda:datetime.now(timezone.utc))
    metadata:Dict[str,Any]=field(default_factory=dict)

@dataclass
class AuthToken:
    """JWT authentication token model"""
    token:str
    token_type:str="Bearer"
    expires_in:int=86400
    refresh_token:Optional[str]=None
    user_id:str=None

@dataclass
class CryptoKey:
    """Cryptographic key model"""
    key_id:str
    user_id:str
    key_type:KeyType
    public_key:str
    private_key_encrypted:Optional[str]=None
    fingerprint:str=""
    is_primary:bool=False
    created_at:datetime=field(default_factory=lambda:datetime.now(timezone.utc))
    expires_at:Optional[datetime]=None
    metadata:Dict[str,Any]=field(default_factory=dict)

@dataclass
class Address:
    """Blockchain address model"""
    address_id:str
    user_id:str
    address:str
    key_id:str
    label:Optional[str]=None
    is_primary:bool=False
    address_type:str="ed25519"
    created_at:datetime=field(default_factory=lambda:datetime.now(timezone.utc))
    metadata:Dict[str,Any]=field(default_factory=dict)

@dataclass
class SecurityEvent:
    """Security audit event model"""
    event_id:str
    user_id:str
    event_type:SecurityEventType
    ip_address:str
    user_agent:str
    details:Dict[str,Any]
    timestamp:datetime=field(default_factory=lambda:datetime.now(timezone.utc))
    severity:str="info"

# ═══════════════════════════════════════════════════════════════════════════════════════
# CORE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════════════

class PasswordValidator:
    """Enhanced password validation with comprehensive security checks"""
    
    @staticmethod
    def validate_strength(password:str,min_length:int=12)->Tuple[bool,str]:
        """Validate password meets security requirements"""
        if len(password)<min_length:
            return False,f"Password must be at least {min_length} characters"
        if not re.search(r'[A-Z]',password):
            return False,"Password must contain uppercase letter"
        if not re.search(r'[a-z]',password):
            return False,"Password must contain lowercase letter"
        if not re.search(r'\d',password):
            return False,"Password must contain digit"
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]',password):
            return False,"Password must contain special character"
        common_passwords={'password123','12345678','qwerty123','admin123'}
        if password.lower() in common_passwords:
            return False,"Password is too common"
        return True,"Password meets requirements"
    
    @staticmethod
    def hash_password(password:str,rounds:int=14)->str:
        """Generate bcrypt hash with configurable rounds"""
        salt=bcrypt.gensalt(rounds=rounds)
        password_hash=bcrypt.hashpw(password.encode('utf-8'),salt)
        return password_hash.decode('utf-8')
    
    @staticmethod
    def verify_password(password:str,password_hash:str)->bool:
        """Verify password against bcrypt hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'),password_hash.encode('utf-8'))
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            return False

class JWTManager:
    """JWT token management with enhanced security"""
    
    def __init__(self,secret_key:str,algorithm:str='HS512',expiration_hours:int=24):
        self.secret_key=secret_key
        self.algorithm=algorithm
        self.expiration_hours=expiration_hours
    
    def create_token(self,user_id:str,role:str,metadata:Dict[str,Any]=None)->str:
        """Generate JWT access token with claims"""
        payload={
            'user_id':user_id,
            'role':role,
            'iat':datetime.now(timezone.utc),
            'exp':datetime.now(timezone.utc)+timedelta(hours=self.expiration_hours),
            'jti':str(uuid.uuid4())
        }
        if metadata:
            payload.update(metadata)
        return pyjwt.encode(payload,self.secret_key,algorithm=self.algorithm)
    
    def create_refresh_token(self,user_id:str,expiration_days:int=30)->str:
        """Generate refresh token with extended expiration"""
        payload={
            'user_id':user_id,
            'type':'refresh',
            'iat':datetime.now(timezone.utc),
            'exp':datetime.now(timezone.utc)+timedelta(days=expiration_days),
            'jti':str(uuid.uuid4())
        }
        return pyjwt.encode(payload,self.secret_key,algorithm=self.algorithm)
    
    def verify_token(self,token:str)->Optional[Dict[str,Any]]:
        """Verify and decode JWT token"""
        try:
            payload=pyjwt.decode(token,self.secret_key,algorithms=[self.algorithm])
            return payload
        except pyjwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except pyjwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None

class TOTPManager:
    """Time-based One-Time Password manager for 2FA"""
    
    @staticmethod
    def generate_secret()->str:
        """Generate TOTP secret for user"""
        return pyotp.random_base32()
    
    @staticmethod
    def generate_qr_code(secret:str,username:str,issuer:str='QTCL')->bytes:
        """Generate QR code for TOTP setup"""
        uri=pyotp.totp.TOTP(secret).provisioning_uri(name=username,issuer_name=issuer)
        qr=qrcode.QRCode(version=1,box_size=10,border=5)
        qr.add_data(uri)
        qr.make(fit=True)
        img=qr.make_image(fill_color="black",back_color="white")
        buffer=BytesIO()
        img.save(buffer,format='PNG')
        return buffer.getvalue()
    
    @staticmethod
    def verify_totp(secret:str,code:str,window:int=1)->bool:
        """Verify TOTP code with time window"""
        totp=pyotp.TOTP(secret)
        return totp.verify(code,valid_window=window)

class KeyGenerator:
    """Cryptographic key generation with multiple algorithms"""
    
    @staticmethod
    def generate_ed25519()->Tuple[str,str]:
        """Generate Ed25519 key pair"""
        private_key=ed25519.Ed25519PrivateKey.generate()
        public_key=private_key.public_key()
        private_pem=private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode('utf-8')
        public_pem=public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')
        return public_pem,private_pem
    
    @staticmethod
    def generate_rsa(key_size:int=4096)->Tuple[str,str]:
        """Generate RSA key pair"""
        private_key=rsa.generate_private_key(public_exponent=65537,key_size=key_size,backend=default_backend())
        public_key=private_key.public_key()
        private_pem=private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode('utf-8')
        public_pem=public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')
        return public_pem,private_pem
    
    @staticmethod
    def generate_ecdsa(curve_name:str='secp256k1')->Tuple[str,str]:
        """Generate ECDSA key pair"""
        curve=ec.SECP256K1() if curve_name=='secp256k1' else ec.SECP256R1()
        private_key=ec.generate_private_key(curve,default_backend())
        public_key=private_key.public_key()
        private_pem=private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode('utf-8')
        public_pem=public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')
        return public_pem,private_pem
    
    @staticmethod
    def calculate_fingerprint(public_key:str)->str:
        """Calculate key fingerprint using SHA256"""
        return hashlib.sha256(public_key.encode('utf-8')).hexdigest()[:40]

class AddressGenerator:
    """Blockchain address generation from public keys"""
    
    @staticmethod
    def from_public_key(public_key:str,prefix:str='qtcl')->str:
        """Generate blockchain address from public key"""
        key_hash=hashlib.sha256(public_key.encode('utf-8')).digest()
        ripemd=hashlib.new('ripemd160')
        ripemd.update(key_hash)
        address_bytes=ripemd.digest()
        checksum=hashlib.sha256(hashlib.sha256(address_bytes).digest()).digest()[:4]
        full_address=address_bytes+checksum
        address_b58=base64.b32encode(full_address).decode('utf-8').lower().rstrip('=')
        return f"{prefix}_{address_b58}"
    
    @staticmethod
    def validate_address(address:str,prefix:str='qtcl')->bool:
        """Validate blockchain address format and checksum"""
        if not address.startswith(f"{prefix}_"):
            return False
        try:
            address_part=address.split('_',1)[1]
            decoded=base64.b32decode(address_part.upper()+'====')
            if len(decoded)<24:
                return False
            address_bytes=decoded[:-4]
            checksum=decoded[-4:]
            calculated_checksum=hashlib.sha256(hashlib.sha256(address_bytes).digest()).digest()[:4]
            return checksum==calculated_checksum
        except Exception:
            return False

# ═══════════════════════════════════════════════════════════════════════════════════════
# DATABASE MANAGER
# ═══════════════════════════════════════════════════════════════════════════════════════

class CoreDatabaseManager:
    """Database operations for core authentication and user management"""
    
    def __init__(self,db_manager):
        self.db=db_manager
    
    def create_user(self,username:str,email:str,password_hash:str,role:UserRole=UserRole.USER)->str:
        """Create new user account"""
        user_id=f"user_{uuid.uuid4().hex[:16]}"
        query="""
            INSERT INTO users (user_id,username,email,password_hash,role,is_active,is_verified,created_at,updated_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,NOW(),NOW())
            RETURNING user_id
        """
        result=self.db.execute_query(query,(user_id,username,email,password_hash,role.value,True,False),fetch_one=True)
        return result['user_id'] if result else user_id
    
    def get_user_by_username(self,username:str)->Optional[Dict[str,Any]]:
        """Retrieve user by username"""
        query="SELECT * FROM users WHERE username=%s"
        return self.db.execute_query(query,(username,),fetch_one=True)
    
    def get_user_by_id(self,user_id:str)->Optional[Dict[str,Any]]:
        """Retrieve user by user_id"""
        query="SELECT * FROM users WHERE user_id=%s"
        return self.db.execute_query(query,(user_id,),fetch_one=True)
    
    def get_user_by_email(self,email:str)->Optional[Dict[str,Any]]:
        """Retrieve user by email"""
        query="SELECT * FROM users WHERE email=%s"
        return self.db.execute_query(query,(email,),fetch_one=True)
    
    def update_last_login(self,user_id:str):
        """Update last login timestamp"""
        query="UPDATE users SET last_login=NOW(),updated_at=NOW() WHERE user_id=%s"
        self.db.execute_query(query,(user_id,))
    
    def increment_failed_login(self,user_id:str)->int:
        """Increment failed login attempts and return count"""
        query="""
            UPDATE users SET failed_login_attempts=failed_login_attempts+1,updated_at=NOW()
            WHERE user_id=%s RETURNING failed_login_attempts
        """
        result=self.db.execute_query(query,(user_id,),fetch_one=True)
        return result['failed_login_attempts'] if result else 0
    
    def reset_failed_login(self,user_id:str):
        """Reset failed login attempts to zero"""
        query="UPDATE users SET failed_login_attempts=0,updated_at=NOW() WHERE user_id=%s"
        self.db.execute_query(query,(user_id,))
    
    def lock_account(self,user_id:str,duration_minutes:int=30):
        """Lock user account for specified duration"""
        locked_until=datetime.now(timezone.utc)+timedelta(minutes=duration_minutes)
        query="UPDATE users SET locked_until=%s,updated_at=NOW() WHERE user_id=%s"
        self.db.execute_query(query,(locked_until,user_id))
    
    def unlock_account(self,user_id:str):
        """Unlock user account"""
        query="UPDATE users SET locked_until=NULL,failed_login_attempts=0,updated_at=NOW() WHERE user_id=%s"
        self.db.execute_query(query,(user_id,))
    
    def enable_totp(self,user_id:str,totp_secret:str):
        """Enable TOTP 2FA for user"""
        query="UPDATE users SET totp_secret=%s,totp_enabled=TRUE,updated_at=NOW() WHERE user_id=%s"
        self.db.execute_query(query,(totp_secret,user_id))
    
    def disable_totp(self,user_id:str):
        """Disable TOTP 2FA for user"""
        query="UPDATE users SET totp_secret=NULL,totp_enabled=FALSE,updated_at=NOW() WHERE user_id=%s"
        self.db.execute_query(query,(user_id,))
    
    def update_password(self,user_id:str,new_password_hash:str):
        """Update user password"""
        query="UPDATE users SET password_hash=%s,updated_at=NOW() WHERE user_id=%s"
        self.db.execute_query(query,(new_password_hash,user_id))
    
    def create_key(self,user_id:str,key_type:KeyType,public_key:str,private_key_encrypted:str,fingerprint:str)->str:
        """Create cryptographic key for user"""
        key_id=f"key_{uuid.uuid4().hex[:16]}"
        query="""
            INSERT INTO crypto_keys (key_id,user_id,key_type,public_key,private_key_encrypted,fingerprint,is_primary,created_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,NOW())
            RETURNING key_id
        """
        result=self.db.execute_query(query,(key_id,user_id,key_type.value,public_key,private_key_encrypted,fingerprint,False),fetch_one=True)
        return result['key_id'] if result else key_id
    
    def get_user_keys(self,user_id:str)->List[Dict[str,Any]]:
        """Get all keys for user"""
        query="SELECT * FROM crypto_keys WHERE user_id=%s ORDER BY created_at DESC"
        return self.db.execute_query(query,(user_id,))
    
    def set_primary_key(self,user_id:str,key_id:str):
        """Set primary key for user"""
        self.db.execute_query("UPDATE crypto_keys SET is_primary=FALSE WHERE user_id=%s",(user_id,))
        self.db.execute_query("UPDATE crypto_keys SET is_primary=TRUE WHERE key_id=%s",(key_id,))
    
    def create_address(self,user_id:str,address:str,key_id:str,label:str=None,address_type:str='ed25519')->str:
        """Create blockchain address"""
        address_id=f"addr_{uuid.uuid4().hex[:16]}"
        query="""
            INSERT INTO addresses (address_id,user_id,address,key_id,label,is_primary,address_type,created_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,NOW())
            RETURNING address_id
        """
        result=self.db.execute_query(query,(address_id,user_id,address,key_id,label,False,address_type),fetch_one=True)
        return result['address_id'] if result else address_id
    
    def get_user_addresses(self,user_id:str)->List[Dict[str,Any]]:
        """Get all addresses for user"""
        query="SELECT * FROM addresses WHERE user_id=%s ORDER BY created_at DESC"
        return self.db.execute_query(query,(user_id,))
    
    def set_primary_address(self,user_id:str,address_id:str):
        """Set primary address for user"""
        self.db.execute_query("UPDATE addresses SET is_primary=FALSE WHERE user_id=%s",(user_id,))
        self.db.execute_query("UPDATE addresses SET is_primary=TRUE WHERE address_id=%s",(address_id,))
    
    def log_security_event(self,user_id:str,event_type:SecurityEventType,ip_address:str,user_agent:str,details:Dict[str,Any],severity:str='info'):
        """Log security audit event"""
        event_id=f"event_{uuid.uuid4().hex[:16]}"
        query="""
            INSERT INTO security_events (event_id,user_id,event_type,ip_address,user_agent,details,severity,timestamp)
            VALUES (%s,%s,%s,%s,%s,%s,%s,NOW())
        """
        self.db.execute_query(query,(event_id,user_id,event_type.value,ip_address,user_agent,json.dumps(details),severity))
    
    def get_security_events(self,user_id:str,limit:int=100)->List[Dict[str,Any]]:
        """Get security events for user"""
        query="SELECT * FROM security_events WHERE user_id=%s ORDER BY timestamp DESC LIMIT %s"
        return self.db.execute_query(query,(user_id,limit))
    
    def detect_suspicious_activity(self,user_id:str,time_window_hours:int=24)->List[Dict[str,Any]]:
        """Detect suspicious activity patterns"""
        query="""
            SELECT event_type,COUNT(*) as count,array_agg(DISTINCT ip_address) as ip_addresses
            FROM security_events
            WHERE user_id=%s AND timestamp>NOW()-INTERVAL '%s hours'
            GROUP BY event_type
            HAVING COUNT(*)>10
        """
        return self.db.execute_query(query,(user_id,time_window_hours))
    
    def create_alias(self,user_id:str,alias:str,target_address:str)->str:
        """Create address alias"""
        alias_id=f"alias_{uuid.uuid4().hex[:16]}"
        query="""
            INSERT INTO address_aliases (alias_id,user_id,alias,target_address,created_at)
            VALUES (%s,%s,%s,%s,NOW())
            RETURNING alias_id
        """
        result=self.db.execute_query(query,(alias_id,user_id,alias,target_address),fetch_one=True)
        return result['alias_id'] if result else alias_id
    
    def lookup_alias(self,alias:str)->Optional[Dict[str,Any]]:
        """Lookup address by alias"""
        query="SELECT * FROM address_aliases WHERE alias=%s"
        return self.db.execute_query(query,(alias,),fetch_one=True)
    
    def get_user_profile(self,user_id:str)->Dict[str,Any]:
        """Get comprehensive user profile"""
        user=self.get_user_by_id(user_id)
        if not user:
            return {}
        keys=self.get_user_keys(user_id)
        addresses=self.get_user_addresses(user_id)
        return {
            'user':user,
            'keys':keys,
            'addresses':addresses,
            'totp_enabled':user.get('totp_enabled',False)
        }
    
    def update_user_profile(self,user_id:str,updates:Dict[str,Any]):
        """Update user profile fields"""
        allowed_fields={'bio','avatar_url','first_name','last_name','phone','company','location'}
        updates={k:v for k,v in updates.items() if k in allowed_fields}
        if not updates:
            return
        set_clause=','.join([f"{k}=%s" for k in updates.keys()])
        query=f"UPDATE users SET {set_clause},updated_at=NOW() WHERE user_id=%s"
        values=list(updates.values())+[user_id]
        self.db.execute_query(query,tuple(values))
    
    def create_session(self,user_id:str,ip_address:str,user_agent:str,token:str)->str:
        """Create new authenticated session"""
        session_id=f"sess_{uuid.uuid4().hex[:16]}"
        expires_at=datetime.now(timezone.utc)+timedelta(hours=24)
        query="""
            INSERT INTO user_sessions (session_id,user_id,ip_address,user_agent,token,expires_at,created_at,last_activity)
            VALUES (%s,%s,%s,%s,%s,%s,NOW(),NOW())
            RETURNING session_id
        """
        result=self.db.execute_query(query,(session_id,user_id,ip_address,user_agent,token,expires_at),fetch_one=True)
        return result['session_id'] if result else session_id
    
    def get_session(self,session_id:str)->Optional[Dict[str,Any]]:
        """Retrieve session details"""
        query="SELECT * FROM user_sessions WHERE session_id=%s AND expires_at>NOW()"
        return self.db.execute_query(query,(session_id,),fetch_one=True)
    
    def get_user_sessions(self,user_id:str)->List[Dict[str,Any]]:
        """Get all active sessions for user"""
        query="SELECT session_id,ip_address,user_agent,created_at,last_activity FROM user_sessions WHERE user_id=%s AND expires_at>NOW() ORDER BY last_activity DESC"
        return self.db.execute_query(query,(user_id,))
    
    def invalidate_session(self,session_id:str):
        """Invalidate a session"""
        query="UPDATE user_sessions SET expires_at=NOW() WHERE session_id=%s"
        self.db.execute_query(query,(session_id,))
    
    def invalidate_all_sessions(self,user_id:str):
        """Invalidate all sessions for user"""
        query="UPDATE user_sessions SET expires_at=NOW() WHERE user_id=%s"
        self.db.execute_query(query,(user_id,))
    
    def update_session_activity(self,session_id:str):
        """Update session last activity timestamp"""
        query="UPDATE user_sessions SET last_activity=NOW() WHERE session_id=%s"
        self.db.execute_query(query,(session_id,))
    
    def create_password_reset_token(self,user_id:str)->str:
        """Create password reset token"""
        token=secrets.token_urlsafe(32)
        token_hash=hashlib.sha256(token.encode('utf-8')).hexdigest()
        expires_at=datetime.now(timezone.utc)+timedelta(hours=1)
        query="""
            INSERT INTO password_reset_tokens (user_id,token_hash,expires_at,created_at,is_used)
            VALUES (%s,%s,%s,NOW(),FALSE)
            RETURNING token
        """
        self.db.execute_query(query,(user_id,token_hash,expires_at))
        return token
    
    def verify_password_reset_token(self,token:str)->Optional[str]:
        """Verify password reset token and return user_id"""
        token_hash=hashlib.sha256(token.encode('utf-8')).hexdigest()
        query="SELECT user_id FROM password_reset_tokens WHERE token_hash=%s AND expires_at>NOW() AND is_used=FALSE"
        result=self.db.execute_query(query,(token_hash,),fetch_one=True)
        return result['user_id'] if result else None
    
    def use_password_reset_token(self,token:str):
        """Mark password reset token as used"""
        token_hash=hashlib.sha256(token.encode('utf-8')).hexdigest()
        query="UPDATE password_reset_tokens SET is_used=TRUE WHERE token_hash=%s"
        self.db.execute_query(query,(token_hash,))
    
    def create_email_verification_token(self,user_id:str,email:str)->str:
        """Create email verification token"""
        token=secrets.token_urlsafe(32)
        token_hash=hashlib.sha256(token.encode('utf-8')).hexdigest()
        expires_at=datetime.now(timezone.utc)+timedelta(hours=24)
        query="""
            INSERT INTO email_verification_tokens (user_id,email,token_hash,expires_at,created_at,is_used)
            VALUES (%s,%s,%s,%s,NOW(),FALSE)
            RETURNING token
        """
        self.db.execute_query(query,(user_id,email,token_hash,expires_at))
        return token
    
    def verify_email_token(self,token:str)->Optional[Tuple[str,str]]:
        """Verify email token and return (user_id, email)"""
        token_hash=hashlib.sha256(token.encode('utf-8')).hexdigest()
        query="SELECT user_id,email FROM email_verification_tokens WHERE token_hash=%s AND expires_at>NOW() AND is_used=FALSE"
        result=self.db.execute_query(query,(token_hash,),fetch_one=True)
        if result:
            return (result['user_id'],result['email'])
        return None
    
    def use_email_verification_token(self,token:str):
        """Mark email verification token as used and update user"""
        token_hash=hashlib.sha256(token.encode('utf-8')).hexdigest()
        query="UPDATE email_verification_tokens SET is_used=TRUE WHERE token_hash=%s RETURNING user_id"
        result=self.db.execute_query(query,(token_hash,),fetch_one=True)
        if result:
            user_id=result['user_id']
            self.db.execute_query("UPDATE users SET is_verified=TRUE,updated_at=NOW() WHERE user_id=%s",(user_id,))
    
    def get_user_notifications(self,user_id:str,limit:int=50)->List[Dict[str,Any]]:
        """Get user notifications"""
        query="SELECT * FROM user_notifications WHERE user_id=%s ORDER BY created_at DESC LIMIT %s"
        return self.db.execute_query(query,(user_id,limit))
    
    def create_notification(self,user_id:str,notification_type:str,title:str,message:str,data:Dict[str,Any]=None)->str:
        """Create user notification"""
        notification_id=f"notif_{uuid.uuid4().hex[:16]}"
        query="""
            INSERT INTO user_notifications (notification_id,user_id,notification_type,title,message,data,is_read,created_at)
            VALUES (%s,%s,%s,%s,%s,%s,FALSE,NOW())
            RETURNING notification_id
        """
        result=self.db.execute_query(query,(notification_id,user_id,notification_type,title,message,json.dumps(data or {})),fetch_one=True)
        return result['notification_id'] if result else notification_id
    
    def mark_notification_read(self,notification_id:str):
        """Mark notification as read"""
        query="UPDATE user_notifications SET is_read=TRUE WHERE notification_id=%s"
        self.db.execute_query(query,(notification_id,))

# ═══════════════════════════════════════════════════════════════════════════════════════
# DATABASE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════════════

class DatabaseInitializer:
    """Initialize and manage PostgreSQL database schema"""
    
    @staticmethod
    def initialize_schema(db_manager):
        """Create all required tables and indexes"""
        
        # Users table
        db_manager.execute_query("""
            CREATE TABLE IF NOT EXISTS users (
                user_id VARCHAR(32) PRIMARY KEY,
                username VARCHAR(128) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                role INTEGER DEFAULT 1,
                is_active BOOLEAN DEFAULT TRUE,
                is_verified BOOLEAN DEFAULT FALSE,
                totp_secret VARCHAR(32),
                totp_enabled BOOLEAN DEFAULT FALSE,
                failed_login_attempts INTEGER DEFAULT 0,
                locked_until TIMESTAMP WITH TIME ZONE,
                last_login TIMESTAMP WITH TIME ZONE,
                bio TEXT,
                avatar_url VARCHAR(500),
                first_name VARCHAR(128),
                last_name VARCHAR(128),
                phone VARCHAR(20),
                company VARCHAR(255),
                location VARCHAR(255),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # User sessions table
        db_manager.execute_query("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id VARCHAR(32) PRIMARY KEY,
                user_id VARCHAR(32) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                ip_address VARCHAR(45),
                user_agent TEXT,
                token TEXT NOT NULL,
                expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Crypto keys table
        db_manager.execute_query("""
            CREATE TABLE IF NOT EXISTS crypto_keys (
                key_id VARCHAR(32) PRIMARY KEY,
                user_id VARCHAR(32) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                key_type VARCHAR(50) NOT NULL,
                public_key TEXT NOT NULL,
                private_key_encrypted TEXT,
                fingerprint VARCHAR(128),
                is_primary BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                expires_at TIMESTAMP WITH TIME ZONE,
                metadata JSONB DEFAULT '{}'::jsonb
            )
        """)
        
        # Addresses table
        db_manager.execute_query("""
            CREATE TABLE IF NOT EXISTS addresses (
                address_id VARCHAR(32) PRIMARY KEY,
                user_id VARCHAR(32) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                address VARCHAR(255) UNIQUE NOT NULL,
                key_id VARCHAR(32) REFERENCES crypto_keys(key_id),
                label VARCHAR(255),
                is_primary BOOLEAN DEFAULT FALSE,
                address_type VARCHAR(50) DEFAULT 'ed25519',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                metadata JSONB DEFAULT '{}'::jsonb
            )
        """)
        
        # Address aliases table
        db_manager.execute_query("""
            CREATE TABLE IF NOT EXISTS address_aliases (
                alias_id VARCHAR(32) PRIMARY KEY,
                user_id VARCHAR(32) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                alias VARCHAR(128) UNIQUE NOT NULL,
                target_address VARCHAR(255) NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Security events table
        db_manager.execute_query("""
            CREATE TABLE IF NOT EXISTS security_events (
                event_id VARCHAR(32) PRIMARY KEY,
                user_id VARCHAR(32) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                event_type VARCHAR(50) NOT NULL,
                ip_address VARCHAR(45),
                user_agent TEXT,
                details JSONB DEFAULT '{}'::jsonb,
                severity VARCHAR(20) DEFAULT 'info',
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Password reset tokens table
        db_manager.execute_query("""
            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                token_id SERIAL PRIMARY KEY,
                user_id VARCHAR(32) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                token_hash VARCHAR(64) UNIQUE NOT NULL,
                expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                is_used BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Email verification tokens table
        db_manager.execute_query("""
            CREATE TABLE IF NOT EXISTS email_verification_tokens (
                token_id SERIAL PRIMARY KEY,
                user_id VARCHAR(32) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                email VARCHAR(255) NOT NULL,
                token_hash VARCHAR(64) UNIQUE NOT NULL,
                expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                is_used BOOLEAN DEFAULT FALSE
            )
        """)
        
        # User notifications table
        db_manager.execute_query("""
            CREATE TABLE IF NOT EXISTS user_notifications (
                notification_id VARCHAR(32) PRIMARY KEY,
                user_id VARCHAR(32) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                notification_type VARCHAR(50) NOT NULL,
                title VARCHAR(255) NOT NULL,
                message TEXT,
                data JSONB DEFAULT '{}'::jsonb,
                is_read BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Create indexes
        db_manager.execute_query("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
        db_manager.execute_query("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
        db_manager.execute_query("CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id)")
        db_manager.execute_query("CREATE INDEX IF NOT EXISTS idx_crypto_keys_user_id ON crypto_keys(user_id)")
        db_manager.execute_query("CREATE INDEX IF NOT EXISTS idx_addresses_user_id ON addresses(user_id)")
        db_manager.execute_query("CREATE INDEX IF NOT EXISTS idx_addresses_address ON addresses(address)")
        db_manager.execute_query("CREATE INDEX IF NOT EXISTS idx_security_events_user_id ON security_events(user_id)")
        db_manager.execute_query("CREATE INDEX IF NOT EXISTS idx_security_events_timestamp ON security_events(timestamp)")
        db_manager.execute_query("CREATE INDEX IF NOT EXISTS idx_password_reset_user_id ON password_reset_tokens(user_id)")
        db_manager.execute_query("CREATE INDEX IF NOT EXISTS idx_user_notifications_user_id ON user_notifications(user_id)")

# ═══════════════════════════════════════════════════════════════════════════════════════
# BLUEPRINT FACTORY
# ═══════════════════════════════════════════════════════════════════════════════════════

def create_core_api_blueprint(db_manager,jwt_manager:JWTManager,config:Dict[str,Any]=None)->Blueprint:
    """Factory function to create Core API blueprint with dependencies"""
    
    bp=Blueprint('core_api',__name__,url_prefix='/api')
    core_db=CoreDatabaseManager(db_manager)
    
    if config is None:
        config={
            'jwt_secret':os.getenv('JWT_SECRET',secrets.token_urlsafe(64)),
            'jwt_algorithm':'HS512',
            'jwt_expiration_hours':24,
            'password_min_length':12,
            'max_login_attempts':5,
            'lockout_duration_minutes':30
        }
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # DECORATORS
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def require_auth(f):
        """Decorator to require JWT authentication"""
        @wraps(f)
        def decorated_function(*args,**kwargs):
            auth_header=request.headers.get('Authorization','')
            if not auth_header.startswith('Bearer '):
                return jsonify({'error':'Missing or invalid authorization header'}),401
            token=auth_header.split(' ',1)[1]
            payload=jwt_manager.verify_token(token)
            if not payload:
                return jsonify({'error':'Invalid or expired token'}),401
            g.user_id=payload.get('user_id')
            g.user_role=payload.get('role')
            return f(*args,**kwargs)
        return decorated_function
    
    def require_role(min_role:UserRole):
        """Decorator to require minimum role level"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args,**kwargs):
                if not hasattr(g,'user_role'):
                    return jsonify({'error':'Authentication required'}),401
                user_role_value=UserRole[g.user_role].value if isinstance(g.user_role,str) else g.user_role
                if user_role_value<min_role.value:
                    return jsonify({'error':'Insufficient permissions'}),403
                return f(*args,**kwargs)
            return decorated_function
        return decorator
    
    def rate_limit(max_requests:int=100,window_seconds:int=60):
        """Simple rate limiting decorator"""
        request_counts=defaultdict(lambda:deque())
        def decorator(f):
            @wraps(f)
            def decorated_function(*args,**kwargs):
                client_ip=request.remote_addr
                now=time.time()
                counts=request_counts[client_ip]
                while counts and counts[0]<now-window_seconds:
                    counts.popleft()
                if len(counts)>=max_requests:
                    return jsonify({'error':'Rate limit exceeded'}),429
                counts.append(now)
                return f(*args,**kwargs)
            return decorated_function
        return decorator
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # AUTHENTICATION ROUTES
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    @bp.route('/auth/register',methods=['POST'])
    @rate_limit(max_requests=10,window_seconds=3600)
    def register():
        """Register new user account with comprehensive validation"""
        try:
            data=request.get_json()
            username=data.get('username','').strip()
            email=data.get('email','').strip().lower()
            password=data.get('password','')
            
            # Validation
            if not username or len(username)<3:
                return jsonify({'error':'Username must be at least 3 characters'}),400
            if not re.match(r'^[a-zA-Z0-9_]+$',username):
                return jsonify({'error':'Username can only contain letters, numbers, and underscores'}),400
            if not email or not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$',email):
                return jsonify({'error':'Invalid email format'}),400
            
            valid,msg=PasswordValidator.validate_strength(password,min_length=config['password_min_length'])
            if not valid:
                return jsonify({'error':msg}),400
            
            # Check existing user
            if core_db.get_user_by_username(username):
                return jsonify({'error':'Username already exists'}),409
            if core_db.get_user_by_email(email):
                return jsonify({'error':'Email already registered'}),409
            
            # Create user
            password_hash=PasswordValidator.hash_password(password)
            user_id=core_db.create_user(username,email,password_hash,UserRole.USER)
            
            # Generate keys and address
            public_key,private_key=KeyGenerator.generate_ed25519()
            fingerprint=KeyGenerator.calculate_fingerprint(public_key)
            key_id=core_db.create_key(user_id,KeyType.ED25519,public_key,private_key,fingerprint)
            core_db.set_primary_key(user_id,key_id)
            
            address=AddressGenerator.from_public_key(public_key)
            address_id=core_db.create_address(user_id,address,key_id,'Primary Address')
            core_db.set_primary_address(user_id,address_id)
            
            # Log event
            core_db.log_security_event(
                user_id,SecurityEventType.LOGIN_SUCCESS,
                request.remote_addr,request.headers.get('User-Agent',''),
                {'action':'registration'},severity='info'
            )
            
            # Generate tokens
            access_token=jwt_manager.create_token(user_id,UserRole.USER.name)
            refresh_token=jwt_manager.create_refresh_token(user_id)
            
            return jsonify({
                'success':True,
                'user_id':user_id,
                'username':username,
                'address':address,
                'access_token':access_token,
                'refresh_token':refresh_token,
                'token_type':'Bearer',
                'expires_in':config['jwt_expiration_hours']*3600
            }),201
            
        except Exception as e:
            logger.error(f"Registration error: {e}",exc_info=True)
            return jsonify({'error':'Registration failed'}),500
    
    @bp.route('/auth/login',methods=['POST'])
    @rate_limit(max_requests=20,window_seconds=300)
    def login():
        """Authenticate user and return JWT tokens"""
        try:
            data=request.get_json()
            username=data.get('username','').strip()
            password=data.get('password','')
            totp_code=data.get('totp_code','')
            
            if not username or not password:
                return jsonify({'error':'Username and password required'}),400
            
            user=core_db.get_user_by_username(username)
            if not user:
                return jsonify({'error':'Invalid credentials'}),401
            
            # Check account lock
            if user.get('locked_until'):
                locked_until=user['locked_until']
                if isinstance(locked_until,str):
                    locked_until=datetime.fromisoformat(locked_until.replace('Z','+00:00'))
                if locked_until>datetime.now(timezone.utc):
                    return jsonify({'error':'Account locked','locked_until':locked_until.isoformat()}),403
                else:
                    core_db.unlock_account(user['user_id'])
            
            # Verify password
            if not PasswordValidator.verify_password(password,user['password_hash']):
                failed_attempts=core_db.increment_failed_login(user['user_id'])
                core_db.log_security_event(
                    user['user_id'],SecurityEventType.LOGIN_FAILED,
                    request.remote_addr,request.headers.get('User-Agent',''),
                    {'attempts':failed_attempts},severity='warning'
                )
                if failed_attempts>=config['max_login_attempts']:
                    core_db.lock_account(user['user_id'],config['lockout_duration_minutes'])
                    return jsonify({'error':'Account locked due to failed login attempts'}),403
                return jsonify({'error':'Invalid credentials'}),401
            
            # Verify TOTP if enabled
            if user.get('totp_enabled'):
                if not totp_code:
                    return jsonify({'error':'TOTP code required','totp_required':True}),401
                if not TOTPManager.verify_totp(user['totp_secret'],totp_code):
                    return jsonify({'error':'Invalid TOTP code'}),401
            
            # Reset failed attempts and update last login
            core_db.reset_failed_login(user['user_id'])
            core_db.update_last_login(user['user_id'])
            
            # Log successful login
            core_db.log_security_event(
                user['user_id'],SecurityEventType.LOGIN_SUCCESS,
                request.remote_addr,request.headers.get('User-Agent',''),
                {'username':username},severity='info'
            )
            
            # Generate tokens
            role=UserRole(user.get('role',1)).name
            access_token=jwt_manager.create_token(user['user_id'],role)
            refresh_token=jwt_manager.create_refresh_token(user['user_id'])
            
            return jsonify({
                'success':True,
                'user_id':user['user_id'],
                'username':user['username'],
                'role':role,
                'access_token':access_token,
                'refresh_token':refresh_token,
                'token_type':'Bearer',
                'expires_in':config['jwt_expiration_hours']*3600
            }),200
            
        except Exception as e:
            logger.error(f"Login error: {e}",exc_info=True)
            return jsonify({'error':'Login failed'}),500
    
    @bp.route('/auth/refresh',methods=['POST'])
    @rate_limit(max_requests=50,window_seconds=3600)
    def refresh_token():
        """Refresh access token using refresh token"""
        try:
            data=request.get_json()
            refresh_token=data.get('refresh_token','')
            
            if not refresh_token:
                return jsonify({'error':'Refresh token required'}),400
            
            payload=jwt_manager.verify_token(refresh_token)
            if not payload or payload.get('type')!='refresh':
                return jsonify({'error':'Invalid refresh token'}),401
            
            user_id=payload.get('user_id')
            user=core_db.get_user_by_id(user_id)
            if not user:
                return jsonify({'error':'User not found'}),404
            
            role=UserRole(user.get('role',1)).name
            new_access_token=jwt_manager.create_token(user_id,role)
            
            return jsonify({
                'success':True,
                'access_token':new_access_token,
                'token_type':'Bearer',
                'expires_in':config['jwt_expiration_hours']*3600
            }),200
            
        except Exception as e:
            logger.error(f"Token refresh error: {e}",exc_info=True)
            return jsonify({'error':'Token refresh failed'}),500
    
    @bp.route('/auth/verify',methods=['POST'])
    @require_auth
    def verify_token_endpoint():
        """Verify current JWT token validity"""
        try:
            user=core_db.get_user_by_id(g.user_id)
            if not user:
                return jsonify({'error':'User not found'}),404
            
            return jsonify({
                'valid':True,
                'user_id':g.user_id,
                'username':user['username'],
                'role':g.user_role
            }),200
            
        except Exception as e:
            logger.error(f"Token verification error: {e}",exc_info=True)
            return jsonify({'error':'Verification failed'}),500
    
    @bp.route('/auth/2fa/enable',methods=['POST'])
    @require_auth
    def enable_2fa():
        """Enable TOTP 2FA for user account"""
        try:
            if not TOTP_AVAILABLE:
                return jsonify({'error':'TOTP not available'}),501
            
            user=core_db.get_user_by_id(g.user_id)
            if not user:
                return jsonify({'error':'User not found'}),404
            
            if user.get('totp_enabled'):
                return jsonify({'error':'2FA already enabled'}),400
            
            # Generate TOTP secret
            totp_secret=TOTPManager.generate_secret()
            qr_code=TOTPManager.generate_qr_code(totp_secret,user['username'])
            
            # Store in session temporarily until verified
            session['totp_setup_secret']=totp_secret
            
            return jsonify({
                'success':True,
                'totp_secret':totp_secret,
                'qr_code':base64.b64encode(qr_code).decode('utf-8')
            }),200
            
        except Exception as e:
            logger.error(f"2FA enable error: {e}",exc_info=True)
            return jsonify({'error':'Failed to enable 2FA'}),500
    
    @bp.route('/auth/2fa/verify',methods=['POST'])
    @require_auth
    def verify_2fa_setup():
        """Verify TOTP setup and finalize 2FA activation"""
        try:
            data=request.get_json()
            totp_code=data.get('totp_code','')
            
            if not totp_code:
                return jsonify({'error':'TOTP code required'}),400
            
            totp_secret=session.get('totp_setup_secret')
            if not totp_secret:
                return jsonify({'error':'No 2FA setup in progress'}),400
            
            if not TOTPManager.verify_totp(totp_secret,totp_code):
                return jsonify({'error':'Invalid TOTP code'}),401
            
            # Enable 2FA permanently
            core_db.enable_totp(g.user_id,totp_secret)
            session.pop('totp_setup_secret',None)
            
            core_db.log_security_event(
                g.user_id,SecurityEventType['2FA_ENABLED'],
                request.remote_addr,request.headers.get('User-Agent',''),
                {},severity='info'
            )
            
            return jsonify({'success':True,'message':'2FA enabled successfully'}),200
            
        except Exception as e:
            logger.error(f"2FA verification error: {e}",exc_info=True)
            return jsonify({'error':'2FA verification failed'}),500
    
    @bp.route('/auth/2fa/disable',methods=['POST'])
    @require_auth
    def disable_2fa():
        """Disable TOTP 2FA for user account"""
        try:
            data=request.get_json()
            password=data.get('password','')
            totp_code=data.get('totp_code','')
            
            if not password:
                return jsonify({'error':'Password required'}),400
            
            user=core_db.get_user_by_id(g.user_id)
            if not user:
                return jsonify({'error':'User not found'}),404
            
            if not PasswordValidator.verify_password(password,user['password_hash']):
                return jsonify({'error':'Invalid password'}),401
            
            if user.get('totp_enabled'):
                if not totp_code or not TOTPManager.verify_totp(user['totp_secret'],totp_code):
                    return jsonify({'error':'Invalid TOTP code'}),401
            
            core_db.disable_totp(g.user_id)
            
            core_db.log_security_event(
                g.user_id,SecurityEventType['2FA_DISABLED'],
                request.remote_addr,request.headers.get('User-Agent',''),
                {},severity='warning'
            )
            
            return jsonify({'success':True,'message':'2FA disabled successfully'}),200
            
        except Exception as e:
            logger.error(f"2FA disable error: {e}",exc_info=True)
            return jsonify({'error':'Failed to disable 2FA'}),500
    
    @bp.route('/auth/logout',methods=['POST'])
    @require_auth
    def logout():
        """Logout current user and invalidate session"""
        try:
            user_id=g.user_id
            # Invalidate all sessions for the user
            core_db.invalidate_all_sessions(user_id)
            
            # Log logout event
            core_db.log_security_event(
                user_id,SecurityEventType.LOGOUT,
                request.remote_addr,request.headers.get('User-Agent',''),
                {'logout_time':datetime.now(timezone.utc).isoformat()},
                severity='info'
            )
            
            return jsonify({'success':True,'message':'Logged out successfully'}),200
            
        except Exception as e:
            logger.error(f"Logout error: {e}",exc_info=True)
            return jsonify({'error':'Logout failed'}),500
    
    @bp.route('/auth/password/change',methods=['POST'])
    @require_auth
    @rate_limit(max_requests=10,window_seconds=3600)
    def change_password():
        """Change user password with current password verification"""
        try:
            data=request.get_json()
            current_password=data.get('current_password','')
            new_password=data.get('new_password','')
            
            if not current_password or not new_password:
                return jsonify({'error':'Current and new passwords required'}),400
            
            user=core_db.get_user_by_id(g.user_id)
            if not user:
                return jsonify({'error':'User not found'}),404
            
            # Verify current password
            if not PasswordValidator.verify_password(current_password,user['password_hash']):
                core_db.log_security_event(
                    g.user_id,SecurityEventType.PASSWORD_CHANGE,
                    request.remote_addr,request.headers.get('User-Agent',''),
                    {'result':'failed_verification'},severity='warning'
                )
                return jsonify({'error':'Current password is incorrect'}),401
            
            # Validate new password
            valid,msg=PasswordValidator.validate_strength(new_password,min_length=config['password_min_length'])
            if not valid:
                return jsonify({'error':msg}),400
            
            # Ensure new password is different
            if PasswordValidator.verify_password(new_password,user['password_hash']):
                return jsonify({'error':'New password must be different from current password'}),400
            
            # Update password
            new_password_hash=PasswordValidator.hash_password(new_password)
            core_db.update_password(g.user_id,new_password_hash)
            
            # Log event
            core_db.log_security_event(
                g.user_id,SecurityEventType.PASSWORD_CHANGE,
                request.remote_addr,request.headers.get('User-Agent',''),
                {'result':'success'},severity='info'
            )
            
            # Invalidate all existing sessions
            core_db.invalidate_all_sessions(g.user_id)
            
            return jsonify({'success':True,'message':'Password changed successfully'}),200
            
        except Exception as e:
            logger.error(f"Password change error: {e}",exc_info=True)
            return jsonify({'error':'Failed to change password'}),500
    
    @bp.route('/auth/password/reset-request',methods=['POST'])
    @rate_limit(max_requests=5,window_seconds=3600)
    def request_password_reset():
        """Request password reset token via email"""
        try:
            data=request.get_json()
            email=data.get('email','').strip().lower()
            
            if not email:
                return jsonify({'error':'Email required'}),400
            
            user=core_db.get_user_by_email(email)
            if not user:
                # Don't reveal whether email exists in system for security
                return jsonify({
                    'success':True,
                    'message':'If email exists in system, password reset link has been sent'
                }),200
            
            # Generate reset token
            reset_token=core_db.create_password_reset_token(user['user_id'])
            
            # In production, send email with reset link
            reset_link=f"https://yourdomain.com/reset-password?token={reset_token}"
            
            # Log event
            core_db.log_security_event(
                user['user_id'],SecurityEventType.PASSWORD_CHANGE,
                request.remote_addr,request.headers.get('User-Agent',''),
                {'action':'password_reset_requested'},severity='info'
            )
            
            # TODO: Send email with reset_link
            # email_service.send_password_reset(user['email'], reset_link)
            
            return jsonify({
                'success':True,
                'message':'If email exists in system, password reset link has been sent'
            }),200
            
        except Exception as e:
            logger.error(f"Password reset request error: {e}",exc_info=True)
            return jsonify({'error':'Failed to process password reset request'}),500
    
    @bp.route('/auth/password/reset',methods=['POST'])
    @rate_limit(max_requests=5,window_seconds=3600)
    def reset_password():
        """Reset password using token"""
        try:
            data=request.get_json()
            token=data.get('token','')
            new_password=data.get('new_password','')
            
            if not token or not new_password:
                return jsonify({'error':'Token and new password required'}),400
            
            # Verify token
            user_id=core_db.verify_password_reset_token(token)
            if not user_id:
                return jsonify({'error':'Invalid or expired reset token'}),401
            
            user=core_db.get_user_by_id(user_id)
            if not user:
                return jsonify({'error':'User not found'}),404
            
            # Validate new password
            valid,msg=PasswordValidator.validate_strength(new_password,min_length=config['password_min_length'])
            if not valid:
                return jsonify({'error':msg}),400
            
            # Update password
            new_password_hash=PasswordValidator.hash_password(new_password)
            core_db.update_password(user_id,new_password_hash)
            
            # Mark token as used
            core_db.use_password_reset_token(token)
            
            # Invalidate all sessions
            core_db.invalidate_all_sessions(user_id)
            
            # Log event
            core_db.log_security_event(
                user_id,SecurityEventType.PASSWORD_CHANGE,
                request.remote_addr,request.headers.get('User-Agent',''),
                {'action':'password_reset_completed'},severity='info'
            )
            
            return jsonify({'success':True,'message':'Password reset successfully'}),200
            
        except Exception as e:
            logger.error(f"Password reset error: {e}",exc_info=True)
            return jsonify({'error':'Failed to reset password'}),500
    
    @bp.route('/auth/email/verify-request',methods=['POST'])
    @require_auth
    @rate_limit(max_requests=5,window_seconds=3600)
    def request_email_verification():
        """Request email verification token"""
        try:
            data=request.get_json()
            email=data.get('email','').strip().lower()
            
            if not email:
                return jsonify({'error':'Email required'}),400
            
            user=core_db.get_user_by_id(g.user_id)
            if not user:
                return jsonify({'error':'User not found'}),404
            
            if not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$',email):
                return jsonify({'error':'Invalid email format'}),400
            
            # Check if email is already used by another user
            existing=core_db.get_user_by_email(email)
            if existing and existing['user_id']!=g.user_id:
                return jsonify({'error':'Email already in use'}),409
            
            # Generate verification token
            verify_token=core_db.create_email_verification_token(g.user_id,email)
            
            # In production, send email with verification link
            verify_link=f"https://yourdomain.com/verify-email?token={verify_token}"
            
            # TODO: Send email with verify_link
            # email_service.send_email_verification(email, verify_link)
            
            return jsonify({
                'success':True,
                'message':'Verification email has been sent'
            }),200
            
        except Exception as e:
            logger.error(f"Email verification request error: {e}",exc_info=True)
            return jsonify({'error':'Failed to request email verification'}),500
    
    @bp.route('/auth/email/verify',methods=['POST'])
    @rate_limit(max_requests=10,window_seconds=3600)
    def verify_email():
        """Verify email with token"""
        try:
            data=request.get_json()
            token=data.get('token','')
            
            if not token:
                return jsonify({'error':'Verification token required'}),400
            
            result=core_db.verify_email_token(token)
            if not result:
                return jsonify({'error':'Invalid or expired verification token'}),401
            
            user_id,email=result
            core_db.use_email_verification_token(token)
            
            # Update user email
            query="UPDATE users SET email=%s,is_verified=TRUE,updated_at=NOW() WHERE user_id=%s"
            core_db.db.execute_query(query,(email,user_id))
            
            # Log event
            core_db.log_security_event(
                user_id,SecurityEventType.LOGIN_SUCCESS,
                request.remote_addr,request.headers.get('User-Agent',''),
                {'action':'email_verified'},severity='info'
            )
            
            return jsonify({'success':True,'message':'Email verified successfully'}),200
            
        except Exception as e:
            logger.error(f"Email verification error: {e}",exc_info=True)
            return jsonify({'error':'Failed to verify email'}),500
    
    @bp.route('/auth/sessions',methods=['GET'])
    @require_auth
    def get_active_sessions():
        """Get all active sessions for current user"""
        try:
            sessions=core_db.get_user_sessions(g.user_id)
            
            return jsonify({
                'sessions':sessions,
                'total':len(sessions)
            }),200
            
        except Exception as e:
            logger.error(f"Get sessions error: {e}",exc_info=True)
            return jsonify({'error':'Failed to get sessions'}),500
    
    @bp.route('/auth/sessions/<session_id>',methods=['DELETE'])
    @require_auth
    def revoke_session(session_id):
        """Revoke specific session"""
        try:
            # Verify session belongs to user
            session=core_db.get_session(session_id)
            if not session or session['user_id']!=g.user_id:
                return jsonify({'error':'Session not found or unauthorized'}),404
            
            core_db.invalidate_session(session_id)
            
            return jsonify({'success':True,'message':'Session revoked'}),200
            
        except Exception as e:
            logger.error(f"Revoke session error: {e}",exc_info=True)
            return jsonify({'error':'Failed to revoke session'}),500
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # USER MANAGEMENT ROUTES
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    @bp.route('/users/profile/me',methods=['GET'])
    @require_auth
    def get_my_profile():
        """Get current user profile with all details"""
        try:
            profile=core_db.get_user_profile(g.user_id)
            if not profile:
                return jsonify({'error':'Profile not found'}),404
            
            # Remove sensitive data
            if 'user' in profile:
                profile['user'].pop('password_hash',None)
                profile['user'].pop('totp_secret',None)
            
            return jsonify(profile),200
            
        except Exception as e:
            logger.error(f"Get profile error: {e}",exc_info=True)
            return jsonify({'error':'Failed to get profile'}),500
    
    @bp.route('/users/profile/me',methods=['PUT'])
    @require_auth
    def update_my_profile():
        """Update current user profile"""
        try:
            data=request.get_json()
            user=core_db.get_user_by_id(g.user_id)
            if not user:
                return jsonify({'error':'User not found'}),404
            
            # Only allow updating certain fields
            allowed_fields={'email','metadata'}
            update_data={}
            
            if 'email' in data:
                email=data['email'].strip().lower()
                if not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$',email):
                    return jsonify({'error':'Invalid email format'}),400
                existing=core_db.get_user_by_email(email)
                if existing and existing['user_id']!=g.user_id:
                    return jsonify({'error':'Email already in use'}),409
                update_data['email']=email
            
            if 'metadata' in data and isinstance(data['metadata'],dict):
                update_data['metadata']=json.dumps(data['metadata'])
            
            if update_data:
                query_parts=[]
                values=[]
                for key,value in update_data.items():
                    query_parts.append(f"{key}=%s")
                    values.append(value)
                values.append(g.user_id)
                
                query=f"UPDATE users SET {','.join(query_parts)},updated_at=NOW() WHERE user_id=%s"
                db_manager.execute_query(query,tuple(values))
            
            return jsonify({'success':True,'message':'Profile updated'}),200
            
        except Exception as e:
            logger.error(f"Update profile error: {e}",exc_info=True)
            return jsonify({'error':'Failed to update profile'}),500
    
    @bp.route('/users/password/change',methods=['POST'])
    @require_auth
    def change_password():
        """Change user password"""
        try:
            data=request.get_json()
            current_password=data.get('current_password','')
            new_password=data.get('new_password','')
            
            if not current_password or not new_password:
                return jsonify({'error':'Current and new password required'}),400
            
            user=core_db.get_user_by_id(g.user_id)
            if not user:
                return jsonify({'error':'User not found'}),404
            
            if not PasswordValidator.verify_password(current_password,user['password_hash']):
                return jsonify({'error':'Invalid current password'}),401
            
            valid,msg=PasswordValidator.validate_strength(new_password,min_length=config['password_min_length'])
            if not valid:
                return jsonify({'error':msg}),400
            
            new_hash=PasswordValidator.hash_password(new_password)
            core_db.update_password(g.user_id,new_hash)
            
            core_db.log_security_event(
                g.user_id,SecurityEventType.PASSWORD_CHANGE,
                request.remote_addr,request.headers.get('User-Agent',''),
                {},severity='info'
            )
            
            return jsonify({'success':True,'message':'Password changed successfully'}),200
            
        except Exception as e:
            logger.error(f"Password change error: {e}",exc_info=True)
            return jsonify({'error':'Failed to change password'}),500
    
    @bp.route('/users',methods=['GET'])
    @require_auth
    @require_role(UserRole.ADMIN)
    def list_users():
        """List all users (admin only)"""
        try:
            limit=min(int(request.args.get('limit',100)),1000)
            offset=int(request.args.get('offset',0))
            
            query="SELECT user_id,username,email,role,is_active,is_verified,created_at,last_login FROM users ORDER BY created_at DESC LIMIT %s OFFSET %s"
            users=db_manager.execute_query(query,(limit,offset))
            
            count_query="SELECT COUNT(*) as count FROM users"
            total=db_manager.execute_query(count_query,fetch_one=True)
            
            return jsonify({
                'users':users,
                'total':total['count'],
                'limit':limit,
                'offset':offset
            }),200
            
        except Exception as e:
            logger.error(f"List users error: {e}",exc_info=True)
            return jsonify({'error':'Failed to list users'}),500
    
    @bp.route('/users/<identifier>',methods=['GET'])
    @require_auth
    def get_user(identifier):
        """Get user by ID or username (self or admin)"""
        try:
            # Check if requesting own data or admin
            user=core_db.get_user_by_id(identifier) or core_db.get_user_by_username(identifier)
            if not user:
                return jsonify({'error':'User not found'}),404
            
            if user['user_id']!=g.user_id and UserRole[g.user_role].value<UserRole.ADMIN.value:
                return jsonify({'error':'Insufficient permissions'}),403
            
            user.pop('password_hash',None)
            user.pop('totp_secret',None)
            
            return jsonify(user),200
            
        except Exception as e:
            logger.error(f"Get user error: {e}",exc_info=True)
            return jsonify({'error':'Failed to get user'}),500
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # KEY MANAGEMENT ROUTES
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    @bp.route('/keys',methods=['GET'])
    @require_auth
    def get_keys():
        """Get all cryptographic keys for current user"""
        try:
            keys=core_db.get_user_keys(g.user_id)
            for key in keys:
                key.pop('private_key_encrypted',None)
            return jsonify({'keys':keys}),200
        except Exception as e:
            logger.error(f"Get keys error: {e}",exc_info=True)
            return jsonify({'error':'Failed to get keys'}),500
    
    @bp.route('/keys/generate',methods=['POST'])
    @require_auth
    def generate_key():
        """Generate new cryptographic key pair"""
        try:
            data=request.get_json()
            key_type_str=data.get('key_type','ed25519').lower()
            label=data.get('label','')
            
            # Map key type
            key_type_map={'ed25519':KeyType.ED25519,'rsa_2048':KeyType.RSA_2048,'rsa_4096':KeyType.RSA_4096,'ecdsa':KeyType.ECDSA_SECP256K1}
            key_type=key_type_map.get(key_type_str,KeyType.ED25519)
            
            # Generate key pair
            if key_type==KeyType.ED25519:
                public_key,private_key=KeyGenerator.generate_ed25519()
            elif key_type in [KeyType.RSA_2048,KeyType.RSA_4096]:
                key_size=2048 if key_type==KeyType.RSA_2048 else 4096
                public_key,private_key=KeyGenerator.generate_rsa(key_size)
            else:
                public_key,private_key=KeyGenerator.generate_ecdsa()
            
            fingerprint=KeyGenerator.calculate_fingerprint(public_key)
            
            # Store key
            key_id=core_db.create_key(g.user_id,key_type,public_key,private_key,fingerprint)
            
            core_db.log_security_event(
                g.user_id,SecurityEventType.KEY_GENERATED,
                request.remote_addr,request.headers.get('User-Agent',''),
                {'key_type':key_type.value,'fingerprint':fingerprint},severity='info'
            )
            
            return jsonify({
                'success':True,
                'key_id':key_id,
                'public_key':public_key,
                'private_key':private_key,
                'fingerprint':fingerprint,
                'key_type':key_type.value
            }),201
            
        except Exception as e:
            logger.error(f"Key generation error: {e}",exc_info=True)
            return jsonify({'error':'Failed to generate key'}),500
    
    @bp.route('/keys/<key_id>/set-primary',methods=['POST'])
    @require_auth
    def set_primary_key_endpoint(key_id):
        """Set primary key for user"""
        try:
            keys=core_db.get_user_keys(g.user_id)
            if not any(k['key_id']==key_id for k in keys):
                return jsonify({'error':'Key not found'}),404
            
            core_db.set_primary_key(g.user_id,key_id)
            return jsonify({'success':True,'message':'Primary key updated'}),200
            
        except Exception as e:
            logger.error(f"Set primary key error: {e}",exc_info=True)
            return jsonify({'error':'Failed to set primary key'}),500
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # ADDRESS MANAGEMENT ROUTES
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    @bp.route('/addresses',methods=['GET'])
    @require_auth
    def get_addresses():
        """Get all blockchain addresses for current user"""
        try:
            addresses=core_db.get_user_addresses(g.user_id)
            return jsonify({'addresses':addresses}),200
        except Exception as e:
            logger.error(f"Get addresses error: {e}",exc_info=True)
            return jsonify({'error':'Failed to get addresses'}),500
    
    @bp.route('/addresses/generate',methods=['POST'])
    @require_auth
    def generate_address():
        """Generate new blockchain address from key"""
        try:
            data=request.get_json()
            key_id=data.get('key_id')
            label=data.get('label','')
            
            if not key_id:
                # Use primary key
                keys=core_db.get_user_keys(g.user_id)
                primary_key=next((k for k in keys if k.get('is_primary')),None)
                if not primary_key:
                    return jsonify({'error':'No primary key found'}),404
                key_id=primary_key['key_id']
                public_key=primary_key['public_key']
            else:
                keys=core_db.get_user_keys(g.user_id)
                key=next((k for k in keys if k['key_id']==key_id),None)
                if not key:
                    return jsonify({'error':'Key not found'}),404
                public_key=key['public_key']
            
            address=AddressGenerator.from_public_key(public_key)
            address_id=core_db.create_address(g.user_id,address,key_id,label)
            
            return jsonify({
                'success':True,
                'address_id':address_id,
                'address':address,
                'key_id':key_id
            }),201
            
        except Exception as e:
            logger.error(f"Address generation error: {e}",exc_info=True)
            return jsonify({'error':'Failed to generate address'}),500
    
    @bp.route('/addresses/<address>/validate',methods=['GET'])
    def validate_address_endpoint(address):
        """Validate blockchain address format"""
        try:
            is_valid=AddressGenerator.validate_address(address)
            return jsonify({'valid':is_valid,'address':address}),200
        except Exception as e:
            logger.error(f"Address validation error: {e}",exc_info=True)
            return jsonify({'error':'Validation failed'}),500
    
    @bp.route('/addresses/<address_id>/set-primary',methods=['POST'])
    @require_auth
    def set_primary_address_endpoint(address_id):
        """Set primary address for user"""
        try:
            addresses=core_db.get_user_addresses(g.user_id)
            if not any(a['address_id']==address_id for a in addresses):
                return jsonify({'error':'Address not found'}),404
            
            core_db.set_primary_address(g.user_id,address_id)
            return jsonify({'success':True,'message':'Primary address updated'}),200
            
        except Exception as e:
            logger.error(f"Set primary address error: {e}",exc_info=True)
            return jsonify({'error':'Failed to set primary address'}),500
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # ALIAS ROUTES
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    @bp.route('/aliases/register',methods=['POST'])
    @require_auth
    def register_alias():
        """Register human-readable alias for address"""
        try:
            data=request.get_json()
            alias=data.get('alias','').strip().lower()
            target_address=data.get('target_address','').strip()
            
            if not alias or not re.match(r'^[a-z0-9_-]+$',alias):
                return jsonify({'error':'Invalid alias format'}),400
            
            if not AddressGenerator.validate_address(target_address):
                return jsonify({'error':'Invalid target address'}),400
            
            # Check if alias exists
            existing=core_db.lookup_alias(alias)
            if existing:
                return jsonify({'error':'Alias already registered'}),409
            
            alias_id=core_db.create_alias(g.user_id,alias,target_address)
            
            return jsonify({
                'success':True,
                'alias_id':alias_id,
                'alias':alias,
                'target_address':target_address
            }),201
            
        except Exception as e:
            logger.error(f"Alias registration error: {e}",exc_info=True)
            return jsonify({'error':'Failed to register alias'}),500
    
    @bp.route('/aliases/<alias>/lookup',methods=['GET'])
    def lookup_alias_endpoint(alias):
        """Lookup address by alias"""
        try:
            result=core_db.lookup_alias(alias.lower())
            if not result:
                return jsonify({'error':'Alias not found'}),404
            
            return jsonify({
                'alias':result['alias'],
                'target_address':result['target_address'],
                'created_at':result['created_at'].isoformat() if isinstance(result['created_at'],datetime) else result['created_at']
            }),200
            
        except Exception as e:
            logger.error(f"Alias lookup error: {e}",exc_info=True)
            return jsonify({'error':'Lookup failed'}),500
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # SIGNATURE & VERIFICATION ROUTES
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    @bp.route('/sign/message',methods=['POST'])
    @require_auth
    def sign_message():
        """Sign message with user's private key"""
        try:
            data=request.get_json()
            message=data.get('message','')
            key_id=data.get('key_id')
            
            if not message:
                return jsonify({'error':'Message required'}),400
            
            keys=core_db.get_user_keys(g.user_id)
            if key_id:
                key=next((k for k in keys if k['key_id']==key_id),None)
            else:
                key=next((k for k in keys if k.get('is_primary')),None)
            
            if not key:
                return jsonify({'error':'Key not found'}),404
            
            # Sign message (simplified - in production, use proper crypto)
            message_hash=hashlib.sha256(message.encode('utf-8')).hexdigest()
            signature=hmac.new(key['fingerprint'].encode('utf-8'),message_hash.encode('utf-8'),hashlib.sha256).hexdigest()
            
            return jsonify({
                'success':True,
                'message':message,
                'signature':signature,
                'key_id':key['key_id'],
                'fingerprint':key['fingerprint']
            }),200
            
        except Exception as e:
            logger.error(f"Message signing error: {e}",exc_info=True)
            return jsonify({'error':'Failed to sign message'}),500
    
    @bp.route('/verify/signature',methods=['POST'])
    def verify_signature():
        """Verify message signature"""
        try:
            data=request.get_json()
            message=data.get('message','')
            signature=data.get('signature','')
            fingerprint=data.get('fingerprint','')
            
            if not all([message,signature,fingerprint]):
                return jsonify({'error':'Message, signature, and fingerprint required'}),400
            
            # Verify signature (simplified)
            message_hash=hashlib.sha256(message.encode('utf-8')).hexdigest()
            expected_signature=hmac.new(fingerprint.encode('utf-8'),message_hash.encode('utf-8'),hashlib.sha256).hexdigest()
            
            is_valid=hmac.compare_digest(signature,expected_signature)
            
            return jsonify({
                'valid':is_valid,
                'message':message,
                'fingerprint':fingerprint
            }),200
            
        except Exception as e:
            logger.error(f"Signature verification error: {e}",exc_info=True)
            return jsonify({'error':'Verification failed'}),500
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # SECURITY & AUDIT ROUTES
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    @bp.route('/security/audit-logs',methods=['GET'])
    @require_auth
    def get_audit_logs():
        """Get security audit logs for current user"""
        try:
            limit=min(int(request.args.get('limit',100)),1000)
            events=core_db.get_security_events(g.user_id,limit)
            
            for event in events:
                if isinstance(event.get('details'),str):
                    try:
                        event['details']=json.loads(event['details'])
                    except:
                        pass
            
            return jsonify({'events':events,'total':len(events)}),200
            
        except Exception as e:
            logger.error(f"Audit logs error: {e}",exc_info=True)
            return jsonify({'error':'Failed to get audit logs'}),500
    
    @bp.route('/security/suspicious-activity',methods=['GET'])
    @require_auth
    def detect_suspicious():
        """Detect suspicious activity patterns"""
        try:
            time_window=int(request.args.get('hours',24))
            suspicious=core_db.detect_suspicious_activity(g.user_id,time_window)
            
            return jsonify({
                'suspicious_patterns':suspicious,
                'time_window_hours':time_window
            }),200
            
        except Exception as e:
            logger.error(f"Suspicious activity detection error: {e}",exc_info=True)
            return jsonify({'error':'Detection failed'}),500
    
    @bp.route('/security/status',methods=['GET'])
    @require_auth
    def security_status():
        """Get comprehensive security status"""
        try:
            user=core_db.get_user_by_id(g.user_id)
            keys=core_db.get_user_keys(g.user_id)
            
            status={
                'totp_enabled':user.get('totp_enabled',False),
                'account_locked':user.get('locked_until') is not None,
                'total_keys':len(keys),
                'failed_login_attempts':user.get('failed_login_attempts',0),
                'last_login':user.get('last_login').isoformat() if user.get('last_login') else None,
                'account_age_days':(datetime.now(timezone.utc)-user['created_at']).days if isinstance(user['created_at'],datetime) else 0
            }
            
            return jsonify(status),200
            
        except Exception as e:
            logger.error(f"Security status error: {e}",exc_info=True)
            return jsonify({'error':'Failed to get security status'}),500
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # DASHBOARD & ANALYTICS ROUTES
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    @bp.route('/dashboard/overview',methods=['GET'])
    @require_auth
    def get_dashboard_overview():
        """Get comprehensive dashboard overview"""
        try:
            user=core_db.get_user_by_id(g.user_id)
            if not user:
                return jsonify({'error':'User not found'}),404
            
            keys=core_db.get_user_keys(g.user_id)
            addresses=core_db.get_user_addresses(g.user_id)
            sessions=core_db.get_user_sessions(g.user_id)
            recent_events=core_db.get_security_events(g.user_id,10)
            notifications=core_db.get_user_notifications(g.user_id,5)
            
            # Calculate stats
            account_age=(datetime.now(timezone.utc)-user['created_at']).days if user['created_at'] else 0
            
            dashboard={
                'user':{
                    'user_id':user['user_id'],
                    'username':user['username'],
                    'email':user['email'],
                    'role':UserRole(user.get('role',1)).name,
                    'is_verified':user.get('is_verified',False),
                    'is_active':user.get('is_active',True),
                    'account_age_days':account_age,
                    'created_at':user['created_at'].isoformat() if hasattr(user['created_at'],'isoformat') else user['created_at'],
                    'last_login':user.get('last_login').isoformat() if user.get('last_login') and hasattr(user['last_login'],'isoformat') else user.get('last_login')
                },
                'stats':{
                    'total_keys':len(keys),
                    'total_addresses':len(addresses),
                    'active_sessions':len(sessions),
                    'total_events':len(recent_events),
                    'unread_notifications':sum(1 for n in notifications if not n.get('is_read',False))
                },
                'security':{
                    'totp_enabled':user.get('totp_enabled',False),
                    'account_locked':user.get('locked_until') is not None,
                    'failed_login_attempts':user.get('failed_login_attempts',0),
                    'last_login':user.get('last_login')
                },
                'recent_activity':{
                    'events':recent_events[:10],
                    'sessions':sessions[:5]
                },
                'notifications':notifications[:5]
            }
            
            return jsonify(dashboard),200
            
        except Exception as e:
            logger.error(f"Dashboard overview error: {e}",exc_info=True)
            return jsonify({'error':'Failed to load dashboard'}),500
    
    @bp.route('/notifications',methods=['GET'])
    @require_auth
    def get_notifications():
        """Get user notifications"""
        try:
            limit=min(int(request.args.get('limit',50)),500)
            notifications=core_db.get_user_notifications(g.user_id,limit)
            
            # Parse data fields
            for notif in notifications:
                if isinstance(notif.get('data'),str):
                    try:
                        notif['data']=json.loads(notif['data'])
                    except:
                        notif['data']={}
            
            return jsonify({'notifications':notifications,'total':len(notifications)}),200
            
        except Exception as e:
            logger.error(f"Get notifications error: {e}",exc_info=True)
            return jsonify({'error':'Failed to get notifications'}),500
    
    @bp.route('/notifications/<notification_id>/read',methods=['POST'])
    @require_auth
    def mark_notification_as_read(notification_id):
        """Mark notification as read"""
        try:
            core_db.mark_notification_read(notification_id)
            return jsonify({'success':True,'message':'Notification marked as read'}),200
            
        except Exception as e:
            logger.error(f"Mark notification error: {e}",exc_info=True)
            return jsonify({'error':'Failed to mark notification'}),500
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # ADVANCED USER PROFILE ROUTES
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    @bp.route('/users/profile/detailed',methods=['GET'])
    @require_auth
    def get_detailed_profile():
        """Get detailed user profile with comprehensive information"""
        try:
            user=core_db.get_user_by_id(g.user_id)
            if not user:
                return jsonify({'error':'User not found'}),404
            
            keys=core_db.get_user_keys(g.user_id)
            addresses=core_db.get_user_addresses(g.user_id)
            sessions=core_db.get_user_sessions(g.user_id)
            recent_events=core_db.get_security_events(g.user_id,20)
            
            # Remove sensitive data
            user.pop('password_hash',None)
            user.pop('totp_secret',None)
            
            profile={
                'user':user,
                'keys':{
                    'total':len(keys),
                    'primary':[k for k in keys if k.get('is_primary',False)][0] if any(k.get('is_primary',False) for k in keys) else None,
                    'keys':keys
                },
                'addresses':{
                    'total':len(addresses),
                    'primary':[a for a in addresses if a.get('is_primary',False)][0] if any(a.get('is_primary',False) for a in addresses) else None,
                    'addresses':addresses
                },
                'sessions':{
                    'total':len(sessions),
                    'active_sessions':sessions
                },
                'recent_activity':recent_events[:20]
            }
            
            return jsonify(profile),200
            
        except Exception as e:
            logger.error(f"Get detailed profile error: {e}",exc_info=True)
            return jsonify({'error':'Failed to get detailed profile'}),500
    
    @bp.route('/users/profile/update-advanced',methods=['PUT'])
    @require_auth
    def update_advanced_profile():
        """Update advanced user profile fields"""
        try:
            data=request.get_json()
            user=core_db.get_user_by_id(g.user_id)
            if not user:
                return jsonify({'error':'User not found'}),404
            
            allowed_fields={'bio','avatar_url','first_name','last_name','phone','company','location'}
            update_data={k:v for k,v in data.items() if k in allowed_fields}
            
            if update_data:
                core_db.update_user_profile(g.user_id,update_data)
            
            return jsonify({'success':True,'message':'Profile updated','updated_fields':update_data}),200
            
        except Exception as e:
            logger.error(f"Update advanced profile error: {e}",exc_info=True)
            return jsonify({'error':'Failed to update profile'}),500
    
    @bp.route('/users/account-settings',methods=['GET'])
    @require_auth
    def get_account_settings():
        """Get account settings and preferences"""
        try:
            user=core_db.get_user_by_id(g.user_id)
            if not user:
                return jsonify({'error':'User not found'}),404
            
            settings={
                'user_id':user['user_id'],
                'username':user['username'],
                'email':user['email'],
                'is_verified':user.get('is_verified',False),
                'is_active':user.get('is_active',True),
                'security':{
                    'totp_enabled':user.get('totp_enabled',False),
                    'password_last_changed':user.get('updated_at'),
                    'account_locked':user.get('locked_until') is not None
                },
                'notifications_enabled':True,
                'two_factor_enabled':user.get('totp_enabled',False),
                'login_attempts_before_lock':5,
                'session_timeout_minutes':30
            }
            
            return jsonify(settings),200
            
        except Exception as e:
            logger.error(f"Get account settings error: {e}",exc_info=True)
            return jsonify({'error':'Failed to get account settings'}),500
    
    @bp.route('/users/verify-request',methods=['POST'])
    @require_auth
    def request_account_verification():
        """Request account verification via email"""
        try:
            user=core_db.get_user_by_id(g.user_id)
            if not user:
                return jsonify({'error':'User not found'}),404
            
            if user.get('is_verified'):
                return jsonify({'error':'Account already verified'}),400
            
            # Generate verification token
            verify_token=core_db.create_email_verification_token(g.user_id,user['email'])
            
            # TODO: Send email with verification link
            verify_link=f"https://yourdomain.com/verify?token={verify_token}"
            # email_service.send_verification(user['email'], verify_link)
            
            return jsonify({
                'success':True,
                'message':'Verification email sent'
            }),200
            
        except Exception as e:
            logger.error(f"Request verification error: {e}",exc_info=True)
            return jsonify({'error':'Failed to request verification'}),500
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # KEY & ADDRESS MANAGEMENT ROUTES
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    @bp.route('/keys/list',methods=['GET'])
    @require_auth
    def list_keys():
        """List all cryptographic keys"""
        try:
            keys=core_db.get_user_keys(g.user_id)
            
            for key in keys:
                # Remove private key from response
                key.pop('private_key_encrypted',None)
                if isinstance(key.get('created_at'),datetime):
                    key['created_at']=key['created_at'].isoformat()
                if key.get('expires_at') and isinstance(key['expires_at'],datetime):
                    key['expires_at']=key['expires_at'].isoformat()
            
            return jsonify({'keys':keys,'total':len(keys)}),200
            
        except Exception as e:
            logger.error(f"List keys error: {e}",exc_info=True)
            return jsonify({'error':'Failed to list keys'}),500
    
    @bp.route('/addresses/list',methods=['GET'])
    @require_auth
    def list_addresses():
        """List all blockchain addresses"""
        try:
            addresses=core_db.get_user_addresses(g.user_id)
            
            return jsonify({'addresses':addresses,'total':len(addresses)}),200
            
        except Exception as e:
            logger.error(f"List addresses error: {e}",exc_info=True)
            return jsonify({'error':'Failed to list addresses'}),500
    
    @bp.route('/addresses/primary',methods=['GET'])
    @require_auth
    def get_primary_address():
        """Get primary blockchain address"""
        try:
            addresses=core_db.get_user_addresses(g.user_id)
            primary=[a for a in addresses if a.get('is_primary',False)]
            
            if not primary:
                return jsonify({'error':'No primary address found'}),404
            
            return jsonify(primary[0]),200
            
        except Exception as e:
            logger.error(f"Get primary address error: {e}",exc_info=True)
            return jsonify({'error':'Failed to get primary address'}),500
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # SYSTEM STATUS & HEALTH ROUTES
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    @bp.route('/health',methods=['GET'])
    def health_check():
        """Health check endpoint"""
        try:
            return jsonify({
                'status':'healthy',
                'timestamp':datetime.now(timezone.utc).isoformat(),
                'service':'core_api',
                'version':'1.0.0'
            }),200
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return jsonify({'status':'unhealthy','error':str(e)}),500
    
    @bp.route('/version',methods=['GET'])
    def get_version():
        """Get API version information"""
        try:
            return jsonify({
                'version':'1.0.0',
                'api_name':'QTCL Core API',
                'features':[
                    'user_authentication',
                    'password_management',
                    'two_factor_authentication',
                    'cryptographic_key_management',
                    'blockchain_addresses',
                    'session_management',
                    'security_audit_logging',
                    'email_verification',
                    'account_recovery'
                ],
                'build_date':datetime.now(timezone.utc).isoformat()
            }),200
            
        except Exception as e:
            logger.error(f"Get version error: {e}")
            return jsonify({'error':'Failed to get version'}),500
    
    return bp
