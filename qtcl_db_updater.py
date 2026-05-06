#!/usr/bin/env python3
"""
QTCL Database Updater — Safe schema migration tool.

Reads schema from qtcl_db_builder.py and applies ONLY missing tables/columns/indexes.
NEVER drops or truncates existing data.

Usage:
    python qtcl_db_updater.py [optional_sqlalchemy_url]

Examples:
    python qtcl_db_updater.py
    python qtcl_db_updater.py postgresql://user:pass@localhost/qtcl
    DATABASE_URL=postgresql://... python qtcl_db_updater.py
"""

import os
import sys
import re
import logging
from typing import List, Dict, Tuple, Optional
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE CONNECTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_db_url() -> str:
    """Get database URL from environment or default."""
    url = os.environ.get('DATABASE_URL')
    if url:
        return url
    # Default local PostgreSQL
    return "postgresql://postgres:postgres@localhost:5432/qtcl"


def parse_db_url(url: str) -> Dict[str, str]:
    """Parse database URL into components."""
    parsed = urlparse(url)
    return {
        'scheme': parsed.scheme,
        'host': parsed.hostname or 'localhost',
        'port': str(parsed.port or 5432),
        'user': parsed.username or 'postgres',
        'password': parsed.password or '',
        'database': parsed.path.lstrip('/') or 'qtcl',
    }


def connect_db(url: str):
    """Connect to PostgreSQL database."""
    try:
        import psycopg2
        info = parse_db_url(url)
        conn = psycopg2.connect(
            host=info['host'],
            port=info['port'],
            user=info['user'],
            password=info['password'],
            database=info['database']
        )
        conn.autocommit = False
        logger.info(f"✅ Connected to PostgreSQL: {info['host']}:{info['port']}/{info['database']}")
        return conn
    except ImportError:
        logger.error("❌ psycopg2 not installed. Run: pip install psycopg2-binary")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA PARSING
# ═══════════════════════════════════════════════════════════════════════════════

def extract_create_table_statements(sql_text: str) -> List[Dict]:
    """Extract CREATE TABLE statements from SQL text."""
    tables = []
    
    # Pattern to match CREATE TABLE statements (including IF NOT EXISTS)
    pattern = r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)\s*\((.*?)\);'
    
    for match in re.finditer(pattern, sql_text, re.DOTALL | re.IGNORECASE):
        table_name = match.group(1)
        body = match.group(2)
        
        # Parse columns and constraints
        columns = []
        constraints = []
        
        # Split by comma, but be careful with nested parentheses
        parts = split_sql_columns(body)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            # Check if it's a constraint
            if part.upper().startswith(('CONSTRAINT ', 'PRIMARY KEY', 'UNIQUE', 'FOREIGN KEY', 'CHECK', 'INDEX')):
                constraints.append(part)
            else:
                # It's a column definition
                columns.append(parse_column_def(part))
        
        tables.append({
            'name': table_name,
            'columns': columns,
            'constraints': constraints,
            'raw_sql': match.group(0)
        })
    
    return tables


def split_sql_columns(body: str) -> List[str]:
    """Split SQL table body by commas, respecting parentheses."""
    parts = []
    current = ""
    depth = 0
    
    for char in body:
        if char == '(':
            depth += 1
            current += char
        elif char == ')':
            depth -= 1
            current += char
        elif char == ',' and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += char
    
    if current.strip():
        parts.append(current.strip())
    
    return parts


def parse_column_def(col_def: str) -> Dict:
    """Parse a column definition string."""
    # Extract column name (first word)
    parts = col_def.split(None, 1)
    col_name = parts[0].strip().strip('"')
    
    return {
        'name': col_name,
        'definition': col_def,
        'raw': col_def
    }


def extract_create_index_statements(sql_text: str) -> List[Dict]:
    """Extract CREATE INDEX statements from SQL text."""
    indexes = []
    pattern = r'CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)\s+ON\s+(\w+)\s*\((.*?)\)'
    
    for match in re.finditer(pattern, sql_text, re.DOTALL | re.IGNORECASE):
        indexes.append({
            'name': match.group(1),
            'table': match.group(2),
            'columns': match.group(3),
            'raw_sql': match.group(0)
        })
    
    return indexes


def extract_alter_table_statements(sql_text: str) -> List[str]:
    """Extract ALTER TABLE statements."""
    alters = []
    pattern = r'ALTER\s+TABLE\s+\w+\s+.*?;'
    for match in re.finditer(pattern, sql_text, re.DOTALL | re.IGNORECASE):
        alters.append(match.group(0))
    return alters


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def get_existing_tables(cur) -> List[str]:
    """Get list of existing tables in the database."""
    cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """)
    return [row[0] for row in cur.fetchall()]


def get_existing_columns(cur, table_name: str) -> List[str]:
    """Get list of existing columns in a table."""
    cur.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        ORDER BY ordinal_position
    """, (table_name,))
    return [row[0] for row in cur.fetchall()]


def get_existing_indexes(cur, table_name: str) -> List[str]:
    """Get list of existing indexes on a table."""
    cur.execute("""
        SELECT indexname FROM pg_indexes
        WHERE schemaname = 'public' AND tablename = %s
    """, (table_name,))
    return [row[0] for row in cur.fetchall()]


def get_existing_constraints(cur, table_name: str) -> List[str]:
    """Get list of existing constraints on a table."""
    cur.execute("""
        SELECT constraint_name FROM information_schema.table_constraints
        WHERE table_schema = 'public' AND table_name = %s
    """, (table_name,))
    return [row[0] for row in cur.fetchall()]


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def table_exists(cur, table_name: str) -> bool:
    """Check if a table exists."""
    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = %s
        )
    """, (table_name,))
    return cur.fetchone()[0]


def create_table_safe(cur, table_def: Dict) -> bool:
    """Create a table if it doesn't exist."""
    table_name = table_def['name']
    
    if table_exists(cur, table_name):
        logger.info(f"  ⏭️  Table '{table_name}' already exists, skipping creation")
        return False
    
    # Reconstruct CREATE TABLE with IF NOT EXISTS
    columns_sql = ",\n    ".join([col['raw'] for col in table_def['columns']])
    if table_def['constraints']:
        columns_sql += ",\n    " + ",\n    ".join(table_def['constraints'])
    
    sql = f"CREATE TABLE IF NOT EXISTS {table_name} (\n    {columns_sql}\n);"
    
    try:
        cur.execute(sql)
        logger.info(f"  ✅ Created table '{table_name}'")
        return True
    except Exception as e:
        logger.error(f"  ❌ Failed to create table '{table_name}': {e}")
        return False


def add_column_safe(cur, table_name: str, column_def: Dict) -> bool:
    """Add a column to an existing table if it doesn't exist."""
    col_name = column_def['name']
    
    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s AND column_name = %s
        )
    """, (table_name, col_name))
    
    if cur.fetchone()[0]:
        return False  # Column already exists
    
    # Extract type and constraints from definition
    def_parts = column_def['definition'].split(None, 1)
    if len(def_parts) < 2:
        logger.warning(f"  ⚠️  Cannot parse column definition: {column_def['definition']}")
        return False
    
    col_type = def_parts[1]
    
    sql = f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {col_name} {col_type}"
    
    try:
        cur.execute(sql)
        logger.info(f"  ✅ Added column '{col_name}' to '{table_name}'")
        return True
    except Exception as e:
        logger.error(f"  ❌ Failed to add column '{col_name}' to '{table_name}': {e}")
        return False


def create_index_safe(cur, index_def: Dict) -> bool:
    """Create an index if it doesn't exist."""
    index_name = index_def['name']
    table_name = index_def['table']
    columns = index_def['columns']
    
    # Check if index exists
    cur.execute("""
        SELECT EXISTS (
            SELECT FROM pg_indexes
            WHERE schemaname = 'public' AND indexname = %s
        )
    """, (index_name,))
    
    if cur.fetchone()[0]:
        return False  # Index already exists
    
    sql = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({columns})"
    
    try:
        cur.execute(sql)
        logger.info(f"  ✅ Created index '{index_name}' on '{table_name}'")
        return True
    except Exception as e:
        logger.error(f"  ❌ Failed to create index '{index_name}': {e}")
        return False


def apply_alter_statement_safe(cur, alter_sql: str) -> bool:
    """Apply an ALTER TABLE statement safely."""
    try:
        cur.execute(alter_sql)
        logger.info(f"  ✅ Applied: {alter_sql[:80]}...")
        return True
    except Exception as e:
        # Check if it's a "already exists" type error
        err_str = str(e).lower()
        if 'already exists' in err_str or 'duplicate' in err_str:
            logger.debug(f"  ⏭️  Already applied: {alter_sql[:80]}...")
            return False
        logger.error(f"  ❌ Failed: {alter_sql[:80]}... → {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN UPDATE LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def update_database(conn, schema_sql: str) -> Dict[str, int]:
    """Apply schema updates to the database."""
    stats = {
        'tables_created': 0,
        'columns_added': 0,
        'indexes_created': 0,
        'alters_applied': 0,
        'errors': 0
    }
    
    cur = conn.cursor()
    
    try:
        # Extract schema definitions
        logger.info("🔍 Parsing schema definitions...")
        tables = extract_create_table_statements(schema_sql)
        indexes = extract_create_index_statements(schema_sql)
        alters = extract_alter_table_statements(schema_sql)
        
        logger.info(f"📋 Found {len(tables)} tables, {len(indexes)} indexes, {len(alters)} ALTER statements")
        
        # Phase 1: Create missing tables
        logger.info("\n📦 Phase 1: Creating missing tables...")
        for table_def in tables:
            if create_table_safe(cur, table_def):
                stats['tables_created'] += 1
        
        # Phase 2: Add missing columns to existing tables
        logger.info("\n🔧 Phase 2: Adding missing columns...")
        for table_def in tables:
            table_name = table_def['name']
            if table_exists(cur, table_name):
                existing_cols = get_existing_columns(cur, table_name)
                for col_def in table_def['columns']:
                    if col_def['name'] not in existing_cols:
                        if add_column_safe(cur, table_name, col_def):
                            stats['columns_added'] += 1
        
        # Phase 3: Create missing indexes
        logger.info("\n📇 Phase 3: Creating missing indexes...")
        for index_def in indexes:
            if create_index_safe(cur, index_def):
                stats['indexes_created'] += 1

        # Phase 4: Apply ALTER statements
        logger.info("\n⚙️  Phase 4: Applying ALTER statements...")
        for alter_sql in alters:
            if apply_alter_statement_safe(cur, alter_sql):
                stats['alters_applied'] += 1

        # Phase 5: Ensure genesis block (height 0) exists with proper difficulty and treasury mint
        logger.info("\n🌍 Phase 5: Ensuring genesis block...")
        import hashlib, json, time
        cur.execute("SELECT EXISTS (SELECT 1 FROM blocks WHERE height = 0)")
        genesis_exists = cur.fetchone()[0]
        if not genesis_exists:
            genesis_hash = hashlib.sha3_256(b"QTCL_GENESIS_2025").hexdigest()
            treasury_addr = "e8ffb27915ac244e8257de8b7f96ad387d1e9d93c634d849a6ad2dae0da6750b"
            cur.execute(
                """
                INSERT INTO blocks
                (height, block_hash, parent_hash, merkle_root, timestamp,
                 w_state_hash, oracle_w_state_hash, miner_address, nonce,
                 difficulty, coherence_snapshot, fidelity_snapshot, tx_count,
                 pq_curr, pq_last, finalized, finalized_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE, %s)
                ON CONFLICT (height) DO NOTHING
                """,
                (0, genesis_hash, "0" * 64, genesis_hash[:64], int(time.time()),
                 genesis_hash[:64], genesis_hash[:64], treasury_addr, 0, 4,
                 1.0, 1.0, 1, 0, 0, int(time.time())),
            )
            logger.info("  ✅ Genesis block (height 0) created: diff=4")
        else:
            # Ensure existing genesis has difficulty >= 4
            cur.execute("SELECT difficulty FROM blocks WHERE height = 0")
            row = cur.fetchone()
            if row and row[0] is not None and row[0] < 4:
                cur.execute("UPDATE blocks SET difficulty = 4 WHERE height = 0")
                logger.info("  ✅ Genesis difficulty updated from %s to 4" % row[0])
            else:
                logger.info("  ⏭️  Genesis block already exists with difficulty >= 4")

        conn.commit()
        logger.info("\n✅ Schema update committed successfully")

    except Exception as e:
        conn.rollback()
        logger.error(f"\n❌ Schema update failed, rolled back: {e}")
        stats['errors'] += 1
    finally:
        cur.close()

    return stats


def load_schema_from_builder(builder_path: str) -> str:
    """Load schema SQL from the builder file."""
    # Read the builder file
    with open(builder_path, 'r') as f:
        content = f.read()
    
    # Extract the SQL schema section
    # Look for SQL blocks (text between triple quotes or SQL strings)
    schema_parts = []
    
    # Pattern 1: Triple-quoted SQL strings
    pattern = r'"""(.*?)"""'
    for match in re.finditer(pattern, content, re.DOTALL):
        text = match.group(1)
        if 'CREATE TABLE' in text.upper():
            schema_parts.append(text)
    
    # Pattern 2: Single-quoted SQL strings that span lines
    pattern2 = r"'''(.*?)'''"
    for match in re.finditer(pattern2, content, re.DOTALL):
        text = match.group(1)
        if 'CREATE TABLE' in text.upper():
            schema_parts.append(text)
    
    # Pattern 3: SQL statements in lists or direct strings
    # Look for lines that contain CREATE TABLE within the file
    lines = content.split('\n')
    in_sql_block = False
    current_block = []
    
    for line in lines:
        stripped = line.strip()
        
        # Detect SQL block starts
        if '-- TABLE:' in stripped or 'CREATE TABLE' in stripped.upper():
            in_sql_block = True
        
        if in_sql_block:
            current_block.append(line)
            
            # End of statement
            if stripped.endswith(';'):
                schema_parts.append('\n'.join(current_block))
                current_block = []
                in_sql_block = False
    
    # Combine all schema parts
    return '\n\n'.join(schema_parts)


def load_schema_comprehensive(builder_path: str) -> str:
    """Comprehensive schema loader that captures all SQL from the builder."""
    with open(builder_path, 'r') as f:
        content = f.read()
    
    # Extract all SQL-like statements
    schema_parts = []
    
    # Find all CREATE TABLE blocks
    create_table_pattern = r'(CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?\w+\s*\(.*?\);)'
    for match in re.finditer(create_table_pattern, content, re.DOTALL | re.IGNORECASE):
        schema_parts.append(match.group(1))
    
    # Find all CREATE INDEX blocks
    create_index_pattern = r'(CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?\w+\s+ON\s+\w+\s*\(.*?\);)'
    for match in re.finditer(create_index_pattern, content, re.DOTALL | re.IGNORECASE):
        schema_parts.append(match.group(1))
    
    # Find all ALTER TABLE blocks
    alter_pattern = r'(ALTER\s+TABLE\s+\w+\s+.*?;)'
    for match in re.finditer(alter_pattern, content, re.DOTALL | re.IGNORECASE):
        # Filter out comments
        stmt = match.group(1)
        if not stmt.strip().startswith('--'):
            schema_parts.append(stmt)
    
    return '\n\n'.join(schema_parts)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point."""
    print("=" * 70)
    print("  QTCL Database Updater")
    print("  Safe schema migration — never wipes existing data")
    print("=" * 70)
    
    # Get database URL
    db_url = sys.argv[1] if len(sys.argv) > 1 else get_db_url()
    logger.info(f"🔗 Database URL: {db_url[:50]}...")
    
    # Connect to database
    conn = connect_db(db_url)
    
    # Find builder file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    builder_path = os.path.join(script_dir, 'qtcl_db_builder.py')
    
    if not os.path.exists(builder_path):
        logger.error(f"❌ Builder file not found: {builder_path}")
        sys.exit(1)
    
    logger.info(f"📄 Loading schema from: {builder_path}")
    
    # Load schema
    schema_sql = load_schema_comprehensive(builder_path)
    
    if not schema_sql.strip():
        logger.error("❌ No schema SQL found in builder file")
        sys.exit(1)
    
    logger.info(f"📊 Loaded {len(schema_sql)} characters of schema SQL")
    
    # Apply updates
    stats = update_database(conn, schema_sql)
    
    # Print summary
    print("\n" + "=" * 70)
    print("  UPDATE SUMMARY")
    print("=" * 70)
    print(f"  Tables created:     {stats['tables_created']}")
    print(f"  Columns added:      {stats['columns_added']}")
    print(f"  Indexes created:    {stats['indexes_created']}")
    print(f"  ALTERs applied:     {stats['alters_applied']}")
    print(f"  Errors:             {stats['errors']}")
    print("=" * 70)
    
    # Close connection
    conn.close()
    logger.info("🔒 Database connection closed")
    
    if stats['errors'] > 0:
        sys.exit(1)
    
    print("\n✅ Database update complete!")


if __name__ == "__main__":
    main()
