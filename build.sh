#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
# QTCL Koyeb build script — optimized for fast container boot
# ══════════════════════════════════════════════════════════════════
set -e

echo "[BUILD] Purging stale cffi C artifacts..."
find /workspace -name '_cffi__*.c'  -delete 2>/dev/null || true
find /workspace -name '_cffi__*.so' -delete 2>/dev/null || true
find .          -name '_cffi__*.c'  -delete 2>/dev/null || true
find .          -name '_cffi__*.so' -delete 2>/dev/null || true
find /tmp       -name '_cffi__*.c'  -delete 2>/dev/null || true
find /tmp       -name 'qtcl_oracle_accel*' -delete 2>/dev/null || true

echo "[BUILD] Pre-installing cffi < 2.0.0 (no OpenSSL headers needed)..."
pip install --quiet --no-cache-dir "cffi>=1.15.1,<2.0.0" || true

echo "[BUILD] Installing requirements.txt (timeout: 180s)..."
timeout 180 pip install --quiet --no-cache-dir -r requirements.txt || {
  echo "[BUILD] ⚠️  First install attempt hit timeout — retrying with --no-deps..."
  pip install --quiet --no-cache-dir --no-deps -r requirements.txt || true
}

echo "[BUILD] cffi version: $(python3 -c 'import cffi; print(cffi.__version__)')"
echo "[BUILD] ✅ Build complete (gunicorn will start now)"
