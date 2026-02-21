#!/usr/bin/env python3
"""
DEBUG DATABASE INITIALIZATION

This script will show EXACTLY what's happening with environment variables
and database connection attempts. Run this to diagnose credential issues.

Usage:
    python debug_db_init.py
"""

import os
import sys
from pathlib import Path

print("\n" + "="*80)
print("🔧 DATABASE INITIALIZATION DEBUG")
print("="*80 + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Show Python Environment
# ─────────────────────────────────────────────────────────────────────────────

print("📌 STEP 1: Python Environment")
print("─" * 80)
print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")
print(f"CWD: {os.getcwd()}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Check Raw Environment Variables
# ─────────────────────────────────────────────────────────────────────────────

print("📌 STEP 2: Raw Environment Variables")
print("─" * 80)

all_env_vars = os.environ.copy()
print(f"Total environment variables: {len(all_env_vars)}\n")

# Look for Supabase-related variables
supabase_vars = {k: v for k, v in all_env_vars.items() if 'SUPABASE' in k or 'DB' in k}

if supabase_vars:
    print("Found Supabase/Database variables:")
    for k, v in supabase_vars.items():
        if 'PASSWORD' in k or 'TOKEN' in k:
            masked = v[:3] + '*'*(len(v)-6) + v[-3:] if len(v) > 6 else '***'
        else:
            masked = v
        print(f"  {k:30} = {masked}")
else:
    print("⚠️  No Supabase/Database environment variables found!")

print()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Simulate Credential Loading (as db_builder_v2 does it)
# ─────────────────────────────────────────────────────────────────────────────

print("📌 STEP 3: Credential Loading (simulating db_builder_v2)")
print("─" * 80 + "\n")

admin_email = os.getenv('ADMIN_EMAIL', 'shemshallah@gmail.com')
print(f"Admin email: {admin_email}\n")

# Try to load from environment
print("Checking environment variables:")
password = os.getenv('SUPABASE_PASSWORD')
host = os.getenv('SUPABASE_HOST')
user = os.getenv('SUPABASE_USER')
port = os.getenv('SUPABASE_PORT', '5432')
database = os.getenv('SUPABASE_DB', 'postgres')

print(f"  SUPABASE_PASSWORD: {password if password else '(NOT FOUND)'}")
print(f"  SUPABASE_HOST: {host if host else '(NOT FOUND)'}")
print(f"  SUPABASE_USER: {user if user else '(NOT FOUND)'}")
print(f"  SUPABASE_PORT: {port if port else '(NOT FOUND)'}")
print(f"  SUPABASE_DB: {database if database else '(NOT FOUND)'}")
print()

# Try to load from .env file
print("Checking .env file:")
env_path = Path('.env')
if env_path.exists():
    print(f"  ✓ Found .env at {env_path.absolute()}\n")
    
    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith('SUPABASE_PASSWORD='):
                    env_password = line.split('=', 1)[1].strip().strip("'\"")
                    password = password or env_password
                    print(f"    → Loaded SUPABASE_PASSWORD from .env")
                elif line.startswith('SUPABASE_HOST='):
                    env_host = line.split('=', 1)[1].strip().strip("'\"")
                    host = host or env_host
                    print(f"    → Loaded SUPABASE_HOST from .env: {env_host}")
                elif line.startswith('SUPABASE_USER='):
                    env_user = line.split('=', 1)[1].strip().strip("'\"")
                    user = user or env_user
                    print(f"    → Loaded SUPABASE_USER from .env: {env_user}")
    except Exception as e:
        print(f"  ✗ Error reading .env: {e}\n")
else:
    print(f"  ✗ No .env file found at {env_path.absolute()}\n")

# Apply safe defaults if still missing
print("\nApplying safe defaults:")
if not password:
    password = "$h10j1r1H0w4rd"
    print(f"  → password (using safe default)")
if not host:
    host = "aws-0-us-west-2.pooler.supabase.com"
    print(f"  → host (using safe default)")
if not user:
    user = "postgres.rslvlsqwkfmdtebqsvtw"
    print(f"  → user (using safe default)")

print()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Final Credentials to Be Used
# ─────────────────────────────────────────────────────────────────────────────

print("📌 STEP 4: Final Credentials to Be Used")
print("─" * 80 + "\n")

print("Database connection parameters:")
print(f"  Host:     {host}")
print(f"  User:     {user}")
print(f"  Password: {'***' * (len(password)//3) if password else '(NONE)'}")
print(f"  Port:     {port}")
print(f"  Database: {database}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Test Connection
# ─────────────────────────────────────────────────────────────────────────────

print("📌 STEP 5: Testing Connection")
print("─" * 80 + "\n")

try:
    import psycopg2
    from psycopg2 import sql, errors as psycopg2_errors
    
    print(f"Attempting to connect to {host}:{port}...")
    
    conn = psycopg2.connect(
        host=host,
        user=user,
        password=password,
        port=int(port),
        database=database,
        connect_timeout=15
    )
    
    print("✅ CONNECTION SUCCESSFUL!\n")
    
    # Get connection info
    cursor = conn.cursor()
    cursor.execute("SELECT VERSION();")
    version = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';")
    table_count = cursor.fetchone()[0]
    
    cursor.close()
    conn.close()
    
    print(f"PostgreSQL: {version[:60]}...")
    print(f"Tables in database: {table_count}")
    print()
    
except ImportError:
    print("❌ psycopg2 not installed")
    print("   Install: pip install psycopg2-binary\n")
    
except psycopg2_errors.OperationalError as e:
    print(f"❌ CONNECTION FAILED: {e}\n")
    
    error_str = str(e).lower()
    
    if 'authentication' in error_str or 'password' in error_str:
        print("DIAGNOSIS: Authentication error (wrong password or user)")
        print("ACTION: Verify credentials are correct")
    elif 'could not translate' in error_str or 'name resolution' in error_str:
        print("DIAGNOSIS: Host not found (DNS issue)")
        print("ACTION: Verify host is correct and network is accessible")
    elif 'connection refused' in error_str:
        print("DIAGNOSIS: Connection refused (port not listening)")
        print("ACTION: Verify port is correct and database is running")
    elif 'timeout' in error_str or 'timed out' in error_str:
        print("DIAGNOSIS: Connection timeout")
        print("ACTION: Check network connectivity and firewall")
    else:
        print("DIAGNOSIS: Unknown connection error")
        print("ACTION: Check all credentials and network connectivity")
    print()
    
except Exception as e:
    print(f"❌ UNEXPECTED ERROR: {type(e).__name__}: {e}\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Test ThreadedConnectionPool Creation
# ─────────────────────────────────────────────────────────────────────────────

print("📌 STEP 6: Testing ThreadedConnectionPool Creation")
print("─" * 80 + "\n")

try:
    from psycopg2.pool import ThreadedConnectionPool
    
    print(f"Creating ThreadedConnectionPool with size 5-50...")
    
    pool = ThreadedConnectionPool(
        5, 50,  # minconns, maxconns
        host=host,
        user=user,
        password=password,
        port=int(port),
        database=database,
        connect_timeout=15
    )
    
    print("✅ POOL CREATION SUCCESSFUL!\n")
    
    # Test getting a connection from pool
    print("Testing connection from pool...")
    conn = pool.getconn()
    
    cursor = conn.cursor()
    cursor.execute("SELECT 1;")
    result = cursor.fetchone()[0]
    cursor.close()
    
    pool.putconn(conn)
    pool.closeall()
    
    print("✅ POOL OPERATION SUCCESSFUL!\n")
    
except Exception as e:
    print(f"❌ POOL CREATION FAILED: {e}\n")
    print("This is the error that db_builder_v2 encountered.")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY & RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────

print("📌 SUMMARY & RECOMMENDATIONS")
print("─" * 80 + "\n")

checks = {
    "Environment variables set": bool(os.getenv('SUPABASE_PASSWORD')),
    ".env file exists": env_path.exists(),
    "Credentials complete": all([host, user, password]),
}

print("Status:")
for check, passed in checks.items():
    symbol = "✅" if passed else "❌"
    print(f"  {symbol} {check}")

print()

if not all(checks.values()):
    print("RECOMMENDED ACTIONS:")
    if not os.getenv('SUPABASE_PASSWORD'):
        print("  1. Set environment variables:")
        print("     export SUPABASE_HOST='...'")
        print("     export SUPABASE_USER='...'")
        print("     export SUPABASE_PASSWORD='...'")
        print()
    
    if not env_path.exists():
        print("  2. OR create .env file:")
        print("     cat > .env << 'EOF'")
        print("     SUPABASE_HOST=aws-0-us-west-2.pooler.supabase.com")
        print("     SUPABASE_USER=postgres.xxxxx")
        print("     SUPABASE_PASSWORD=your-password")
        print("     SUPABASE_PORT=5432")
        print("     SUPABASE_DB=postgres")
        print("     EOF")
        print()

print("="*80 + "\n")
