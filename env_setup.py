#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
QTCL ENVIRONMENT SETUP (WITH RESILIENT DB CONNECTION)
Initialize all environment variables, database, and directories
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import secrets
import logging
import time
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.resolve()
ENV_FILE = PROJECT_ROOT / '.env'
LOG_DIR = Path.home() / 'QTCL_logs'
DATA_DIR = PROJECT_ROOT / 'data'

# ═══════════════════════════════════════════════════════════════════════════════
# DIRECTORY SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def setup_directories():
    """Create necessary directories"""
    logger.info("Setting up directories...")
    
    for directory in [LOG_DIR, DATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"  ✓ {directory}")
    
    logger.info("Directories ready")

# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT VARIABLES
# ═══════════════════════════════════════════════════════════════════════════════

def generate_env_file():
    """Generate .env file with required variables"""
    logger.info("Generating environment configuration...")
    
    env_content = f"""# ═══════════════════════════════════════════════════════════════════════════════
# QTCL ENVIRONMENT VARIABLES
# Generated: {datetime.now().isoformat()}
# ═══════════════════════════════════════════════════════════════════════════════

# FLASK CONFIGURATION
FLASK_ENV=production
FLASK_SECRET_KEY={secrets.token_hex(32)}
DEBUG=False

# SECURITY
JWT_SECRET={secrets.token_hex(32)}
JWT_ALGORITHM=HS256
TOKEN_EXPIRY_HOURS=24

# DATABASE - SUPABASE PostgreSQL
# Update these with your actual Supabase credentials
SUPABASE_HOST=aws-0-us-west-2.pooler.supabase.com
SUPABASE_PORT=5432
SUPABASE_USER=postgres.rslvlsqwkfmdtebqsvtw
SUPABASE_PASSWORD=your_secure_password_here_change_me
SUPABASE_DB=postgres

# DATABASE CONNECTION POOL
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=10
DB_POOL_TIMEOUT=30
DB_POOL_RECYCLE=3600
DB_CONNECTION_TIMEOUT=30
DB_RETRIES=3

# QUANTUM CONFIGURATION
QUANTUM_SHOTS=1024
QUANTUM_BACKEND=aer_simulator
QUANTUM_OPTIMIZATION_LEVEL=2

# ORACLE CONFIGURATION
ORACLE_POLL_INTERVAL=5
ORACLE_TIMEOUT=30
ORACLE_RETRIES=3

# LOGGING
LOG_LEVEL=INFO
LOG_DIR={LOG_DIR}

# API CONFIGURATION
API_HOST=0.0.0.0
API_PORT=5000
API_WORKERS=4

# FEATURES
ENABLE_QUANTUM_CACHE=True
ENABLE_BLOCK_FINALITY=True
ENABLE_ORACLE_COLLAPSING=True
SUPERPOSITION_DECAY_RATE=0.95
"""
    
    if ENV_FILE.exists():
        logger.warning(f"⚠️  {ENV_FILE} already exists - backing up")
        backup = ENV_FILE.with_suffix('.env.backup')
        ENV_FILE.rename(backup)
        logger.info(f"  → Backup saved to {backup}")
    
    ENV_FILE.write_text(env_content)
    logger.info(f"✓ Environment file created: {ENV_FILE}")

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════

def load_env_file():
    """Load .env file into os.environ"""
    logger.info("Loading environment variables...")
    
    if not ENV_FILE.exists():
        logger.error(f"✗ {ENV_FILE} not found!")
        return False
    
    try:
        with open(ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        
        logger.info("✓ Environment variables loaded")
        return True
    
    except Exception as e:
        logger.error(f"✗ Error loading .env: {e}")
        return False

# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE SETUP (WITH RETRY LOGIC)
# ═══════════════════════════════════════════════════════════════════════════════

def test_database_connection_with_retry(retries=3, delay=2):
    """Test connection to Supabase PostgreSQL with retry logic"""
    logger.info("Testing database connection...")
    
    try:
        import psycopg2
    except ImportError:
        logger.warning("✗ psycopg2 not installed - skipping database test")
        return False
    
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(
                host=os.getenv('SUPABASE_HOST'),
                port=int(os.getenv('SUPABASE_PORT', 5432)),
                user=os.getenv('SUPABASE_USER'),
                password=os.getenv('SUPABASE_PASSWORD'),
                database=os.getenv('SUPABASE_DB'),
                connect_timeout=10
            )
            
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            logger.info(f"✓ Database connected (attempt {attempt + 1}/{retries})")
            logger.info(f"  → {version[0][:60]}...")
            return True
        
        except psycopg2.OperationalError as e:
            if attempt < retries - 1:
                logger.warning(f"✗ Connection attempt {attempt + 1}/{retries} failed: {str(e)[:60]}...")
                logger.info(f"  → Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.warning(f"✗ Database connection failed after {retries} attempts")
                logger.warning("  → This is OK - you can test later with: psql -h... -U... -d...")
                return False
        
        except Exception as e:
            logger.error(f"✗ Unexpected error: {e}")
            return False
    
    return False

# ═══════════════════════════════════════════════════════════════════════════════
# DEPENDENCY CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def check_dependencies():
    """Verify all required packages installed"""
    logger.info("Checking dependencies...")
    
    required = {
        'flask': 'Flask',
        'flask_cors': 'Flask-CORS',
        'psycopg2': 'psycopg2-binary',
        'numpy': 'numpy',
        'qiskit': 'qiskit',
        'cryptography': 'cryptography',
        'jwt': 'PyJWT',
        'requests': 'requests',
        'dotenv': 'python-dotenv',
    }
    
    missing = []
    
    for module, package in required.items():
        try:
            __import__(module)
            logger.info(f"  ✓ {package}")
        except ImportError:
            logger.warning(f"  ✗ {package} missing")
            missing.append(package)
    
    if missing:
        logger.error(f"\n✗ Missing packages: {', '.join(missing)}")
        logger.info("\nInstall with:")
        logger.info(f"  pip install {' '.join(missing)}")
        return False
    
    logger.info("✓ All dependencies present")
    return True

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Run complete setup"""
    print("\n" + "=" * 80)
    print("QTCL ENVIRONMENT SETUP")
    print("=" * 80 + "\n")
    
    logger.info(f"Project root: {PROJECT_ROOT}")
    
    # Step 1: Create directories
    setup_directories()
    
    # Step 2: Generate .env file
    generate_env_file()
    
    # Step 3: Load environment
    if not load_env_file():
        logger.error("Failed to load environment variables")
        return False
    
    # Step 4: Check dependencies
    if not check_dependencies():
        logger.error("Some dependencies missing - install before running")
        return False
    
    # Step 5: Test database (with retry, non-blocking)
    db_ok = test_database_connection_with_retry(retries=3, delay=2)
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ SETUP COMPLETE!")
    
    if db_ok:
        print("\nDatabase: CONNECTED ✓")
    else:
        print("\nDatabase: UNREACHABLE (will retry on app start)")
    
    print("\nNext steps:")
    print("  1. python wsgi_config.py          (test locally)")
    print("  2. curl http://127.0.0.1:5000/health")
    print("  3. Deploy to PythonAnywhere")
    print("=" * 80 + "\n")
    
    return True

# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)