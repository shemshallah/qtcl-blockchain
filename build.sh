#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
# QTCL Koyeb build script
# Koyeb runs this during the BUILD phase before image creation.
# ══════════════════════════════════════════════════════════════════

# NO set -e — cffi pin failure must not abort the entire build

echo "[BUILD] Purging stale cffi C artifacts..."
find /workspace -name '_cffi__*.c'  -delete 2>/dev/null || true
find /workspace -name '_cffi__*.so' -delete 2>/dev/null || true
find .          -name '_cffi__*.c'  -delete 2>/dev/null || true
find .          -name '_cffi__*.so' -delete 2>/dev/null || true
find /tmp       -name '_cffi__*.c'  -delete 2>/dev/null || true
find /tmp       -name 'qtcl_oracle_accel*' -delete 2>/dev/null || true

echo "[BUILD] Enforcing cffi < 2.0.0..."
pip install --quiet --upgrade "cffi>=1.15.1,<2.0.0" || {
    echo "[BUILD] WARNING: cffi pin failed, continuing anyway"
}

echo "[BUILD] Installing requirements..."
pip install --quiet -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[BUILD] ERROR: pip install requirements.txt failed"
    exit 1
fi

# Re-enforce cffi pin AFTER all packages installed
# (qiskit-aer or other deps may have upgraded cffi during install)
echo "[BUILD] Re-enforcing cffi < 2.0.0 post-install..."
pip install --quiet --force-reinstall "cffi>=1.15.1,<2.0.0" || {
    echo "[BUILD] WARNING: cffi re-pin failed"
}

echo "Installing secp256k1..." 
pip install secp256k1 || {
    echo "[BUILD} WARNING: secp256k1 build failed."

echo "[BUILD] cffi version: $(python3 -c 'import cffi; print(cffi.__version__)')"
echo "[BUILD] ✅ Build complete"
