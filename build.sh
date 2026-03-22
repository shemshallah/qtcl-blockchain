#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
# QTCL Koyeb build script
# Koyeb calls this before starting the web process if it exists.
# Forces cffi downgrade and purges stale C artifacts.
# ══════════════════════════════════════════════════════════════════
set -e

echo "[BUILD] Purging stale cffi C artifacts..."
find /workspace -name '_cffi__*.c'  -delete 2>/dev/null || true
find /workspace -name '_cffi__*.so' -delete 2>/dev/null || true
find .          -name '_cffi__*.c'  -delete 2>/dev/null || true
find .          -name '_cffi__*.so' -delete 2>/dev/null || true
find /tmp       -name '_cffi__*.c'  -delete 2>/dev/null || true
find /tmp       -name 'qtcl_oracle_accel*' -delete 2>/dev/null || true

echo "[BUILD] Enforcing cffi < 2.0.0 (pre-built wheel, no OpenSSL headers needed)..."
pip install --quiet --upgrade "cffi>=1.15.1,<2.0.0"

echo "[BUILD] Installing full requirements..."
pip install --quiet -r requirements.txt

echo "[BUILD] cffi version: $(python3 -c 'import cffi; print(cffi.__version__)')"
echo "[BUILD] ✅ Build complete"
