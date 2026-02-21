#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║        QTCL DIAGNOSTIC TOOL — Environment & Database Connection Debug       ║
║                                                                              ║
║  This script diagnoses why environment variables aren't being detected      ║
║  and why database connections are failing.                                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
from pathlib import Path

print("\n" + "="*80)
print("🔍 QTCL DIAGNOSTIC REPORT")
print("="*80 + "\n")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Check Environment Variables
# ═══════════════════════════════════════════════════════════════════════════════

print("📋 STEP 1: Environment Variable Detection")
print("─" * 80)

env_vars = {
    'SUPABASE_HOST': os.getenv('SUPABASE_HOST'),
    'SUPABASE_USER': os.getenv('SUPABASE_USER'),
    'SUPABASE_PASSWORD': os.getenv('SUPABASE_PASSWORD'),
    'SUPABASE_PORT': os.getenv('SUPABASE_PORT'),
    'SUPABASE_DB': os.getenv('SUPABASE_DB'),
    'SUPABASE_URL': os.getenv('SUPABASE_URL'),
    'SUPABASE_AUTH_TOKEN': os.getenv('SUPABASE_AUTH_TOKEN'),
}

has_required_vars = all([
    env_vars['SUPABASE_HOST'],
    env_vars['SUPABASE_USER'],
    env_vars['SUPABASE_PASSWORD']
])

print(f"\nRequired variables found: {'✅ YES' if has_required_vars else '❌ NO'}\n")

for var_name, value in env_vars.items():
    if value:
        # Mask sensitive data
        if 'PASSWORD' in var_name or 'TOKEN' in var_name:
            masked = value[:3] + '*' * (len(value)-6) + value[-3:]
        else:
            masked = value
        print(f"  ✓ {var_name:25} = {masked}")
    else:
        print(f"  ✗ {var_name:25} = (NOT SET)")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Check .env File
# ═══════════════════════════════════════════════════════════════════════════════

print("\n📋 STEP 2: .env File Detection")
print("─" * 80)

env_file = Path('.env')
if env_file.exists():
    print(f"\n  ✓ Found .env file: {env_file.absolute()}")
    print(f"  Size: {env_file.stat().st_size} bytes\n")
    
    try:
        with open(env_file) as f:
            env_content = {}
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    env_content[key] = val.strip('\'"')
        
        print("  Variables in .env file:")
        for key, value in env_content.items():
            if any(x in key for x in ['PASSWORD', 'TOKEN', 'SECRET']):
                masked = value[:3] + '*' * (len(value)-6) + value[-3:] if len(value) > 6 else '***'
            else:
                masked = value
            print(f"    • {key:25} = {masked}")
    except Exception as e:
        print(f"  ⚠️  Could not read .env file: {e}")
else:
    print("\n  ✗ No .env file found in current directory")
    print(f"  Current directory: {Path('.').absolute()}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Check Current Working Directory
# ═══════════════════════════════════════════════════════════════════════════════

print("\n📋 STEP 3: Current Working Directory")
print("─" * 80)

cwd = Path.cwd()
print(f"\n  Current directory: {cwd}")
print(f"  Files in current directory:")

for item in sorted(cwd.glob('*'))[:20]:  # First 20 items
    if item.is_file():
        size = item.stat().st_size
        print(f"    • {item.name:40} ({size:,} bytes)")
    else:
        print(f"    📁 {item.name}/")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: Test Database Connection
# ═══════════════════════════════════════════════════════════════════════════════

print("\n📋 STEP 4: Database Connection Test")
print("─" * 80)

if has_required_vars:
    print("\n  Testing PostgreSQL connection...")
    
    try:
        import psycopg2
        
        conn = psycopg2.connect(
            host=env_vars['SUPABASE_HOST'],
            user=env_vars['SUPABASE_USER'],
            password=env_vars['SUPABASE_PASSWORD'],
            port=int(env_vars['SUPABASE_PORT'] or 5432),
            database=env_vars['SUPABASE_DB'] or 'postgres',
            connect_timeout=10
        )
        
        cursor = conn.cursor()
        cursor.execute("SELECT VERSION();")
        version = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        
        print(f"  ✅ Connection successful!")
        print(f"  PostgreSQL: {version[:50]}...")
        
    except ImportError:
        print(f"  ⚠️  psycopg2 not installed")
        print(f"     Install with: pip install psycopg2-binary")
    except Exception as e:
        print(f"  ❌ Connection failed!")
        print(f"  Error: {e}")
        print(f"\n  Troubleshooting:")
        print(f"    1. Check credentials are correct")
        print(f"    2. Verify network connectivity to {env_vars['SUPABASE_HOST']}")
        print(f"    3. Check if firewall is blocking port {env_vars['SUPABASE_PORT'] or 5432}")
        print(f"    4. Verify Supabase project is active")
else:
    print("\n  ❌ Required environment variables not set")
    print("  Cannot test database connection without credentials")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: Python Import Test
# ═══════════════════════════════════════════════════════════════════════════════

print("\n📋 STEP 5: Python Module Import Test")
print("─" * 80)

print("\n  Testing imports...")

modules = {
    'flask': 'Flask web framework',
    'psycopg2': 'PostgreSQL driver',
    'requests': 'HTTP library',
    'jwt': 'JWT tokens',
}

for module_name, description in modules.items():
    try:
        __import__(module_name)
        print(f"  ✓ {module_name:15} — {description}")
    except ImportError:
        print(f"  ✗ {module_name:15} — {description} (NOT INSTALLED)")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: Database Builder Initialization Test
# ═══════════════════════════════════════════════════════════════════════════════

print("\n📋 STEP 6: Database Builder Initialization Test")
print("─" * 80)

print("\n  Attempting to import db_builder_v2...")

try:
    sys.path.insert(0, str(Path.cwd()))
    from db_builder_v2 import db_manager, POOLER_HOST, POOLER_USER, POOLER_PASSWORD
    
    print(f"  ✓ Successfully imported db_builder_v2")
    print(f"\n  Credentials used by db_manager:")
    print(f"    Host:     {POOLER_HOST}")
    print(f"    User:     {POOLER_USER}")
    print(f"    Password: {'*' * len(POOLER_PASSWORD) if POOLER_PASSWORD else '(NOT SET)'}")
    
    if db_manager:
        print(f"\n  Database Manager Status:")
        print(f"    pool:          {db_manager.pool}")
        print(f"    pool_error:    {db_manager.pool_error}")
        print(f"    initialized:   {db_manager.initialized}")
        
        if db_manager.pool is None and db_manager.pool_error:
            print(f"\n  ⚠️  Pool creation failed!")
            print(f"  Error: {db_manager.pool_error}")
    else:
        print(f"\n  ⚠️  db_manager is None")
        
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
except Exception as e:
    print(f"  ✗ Error during import: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7: Recommendations
# ═══════════════════════════════════════════════════════════════════════════════

print("\n📋 STEP 7: Recommendations")
print("─" * 80)

recommendations = []

if not has_required_vars:
    recommendations.append(
        "Set missing environment variables:\n"
        "  export SUPABASE_HOST='aws-0-us-west-2.pooler.supabase.com'\n"
        "  export SUPABASE_USER='postgres.xxxxx'\n"
        "  export SUPABASE_PASSWORD='your-password'\n"
        "  export SUPABASE_PORT='5432'\n"
        "  export SUPABASE_DB='postgres'"
    )

if not env_file.exists():
    recommendations.append(
        "Create .env file in project root:\n"
        "  cat > .env << 'EOF'\n"
        "  SUPABASE_HOST=aws-0-us-west-2.pooler.supabase.com\n"
        "  SUPABASE_USER=postgres.xxxxx\n"
        "  SUPABASE_PASSWORD=your-password\n"
        "  SUPABASE_PORT=5432\n"
        "  SUPABASE_DB=postgres\n"
        "  EOF"
    )

if recommendations:
    for i, rec in enumerate(recommendations, 1):
        print(f"\n  {i}. {rec}")
else:
    print("\n  ✅ All checks passed! System appears to be configured correctly.")

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("📊 DIAGNOSTIC SUMMARY")
print("="*80)

summary = {
    "env_vars_set": has_required_vars,
    "env_file_exists": env_file.exists(),
    "current_dir": str(Path.cwd()),
    "required_vars": {k: "✓" if v else "✗" for k, v in {
        'SUPABASE_HOST': env_vars['SUPABASE_HOST'],
        'SUPABASE_USER': env_vars['SUPABASE_USER'],
        'SUPABASE_PASSWORD': env_vars['SUPABASE_PASSWORD'],
    }.items()}
}

print(json.dumps(summary, indent=2))

print("\n" + "="*80)
print("Next: Review recommendations above and apply fixes")
print("Then: python main_app.py")
print("="*80 + "\n")
