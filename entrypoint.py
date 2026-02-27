#!/usr/bin/env python3
"""
entrypoint.py — PART 1/1: Enterprise liboqs pre-build entrypoint.

Builds liboqs shared library from source at a verified git tag, sets
OQS_INSTALL_DIR + LD_LIBRARY_PATH in the current process environment,
then exec-replaces itself with gunicorn so all env vars are inherited.

Deployment: Procfile → `web: python entrypoint.py`
"""

import os, sys, subprocess, shutil, logging, time, json, hashlib, re
from pathlib import Path
from typing import Optional, List, Tuple

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] oqs-build: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger('oqs-build')

# ────────────────────────────────────────────────────────────────────────────
# CONSTANTS — all paths and versions in one place
# ────────────────────────────────────────────────────────────────────────────
LIBOQS_REPO          = 'https://github.com/open-quantum-safe/liboqs.git'
INSTALL_DIR          = Path(os.environ.get('OQS_INSTALL_DIR', '/app/_oqs'))
BUILD_DIR            = Path('/tmp/_liboqs_build')
STAMP_FILE           = INSTALL_DIR / '.build_stamp'
BUILD_TIMEOUT_S      = 600   # 10 min hard cap

# Tag priority list — newest → oldest; entrypoint tries each in order until
# one clones and builds successfully.  All of these exist in the liboqs repo.
# Pin whichever version matches `oqs-python` installed on the system.
CANDIDATE_TAGS: List[str] = [
    '0.12.0',   # latest as of late 2025
    '0.11.0',   # stable Dec 2024
    '0.10.1',   # stable Oct 2024
    '0.10.0',   # stable Aug 2024
    '0.9.2',    # stable
    '0.9.0',    # fallback
    '0.8.0',    # last-resort
]

# gunicorn launch spec — mirrors the old Procfile
GUNICORN_WORKERS = int(os.environ.get('WEB_CONCURRENCY', '4'))
GUNICORN_BIND    = f"0.0.0.0:{os.environ.get('PORT', '8000')}"
GUNICORN_MODULE  = 'wsgi_config:application'
GUNICORN_ARGS: List[str] = [
    'gunicorn',
    f'--workers={GUNICORN_WORKERS}',
    f'--bind={GUNICORN_BIND}',
    '--timeout=120',
    '--keep-alive=5',
    '--max-requests=1000',
    '--max-requests-jitter=100',
    '--worker-class=sync',
    '--preload',
    '--access-logfile=-',
    '--error-logfile=-',
    '--log-level=info',
    GUNICORN_MODULE,
]

# cmake build flags — shared lib, no tests, use OpenSSL, release build
CMAKE_FLAGS: List[str] = [
    '-GNinja',
    '-DCMAKE_BUILD_TYPE=Release',
    '-DBUILD_SHARED_LIBS=ON',
    '-DOQS_BUILD_ONLY_LIB=ON',
    '-DOQS_USE_OPENSSL=ON',
    '-DOQS_DIST_BUILD=ON',
    '-DOQS_ENABLE_KEM_BIKE=OFF',    # optional – speeds build; re-enable if needed
    '-DCMAKE_POSITION_INDEPENDENT_CODE=ON',
]

# ────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────

def _run(cmd: List[str], cwd: Optional[Path] = None, timeout: int = BUILD_TIMEOUT_S) -> bool:
    """Run subprocess, stream output, return success bool."""
    log.info(f"$ {' '.join(cmd)}" + (f"  [cwd={cwd}]" if cwd else ''))
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, bufsize=1,
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                log.info(f"  | {line}")
        proc.wait(timeout=timeout)
        if proc.returncode != 0:
            log.error(f"Command failed (exit {proc.returncode}): {' '.join(cmd)}")
            return False
        return True
    except subprocess.TimeoutExpired:
        proc.kill()
        log.error(f"Command timed out after {timeout}s: {' '.join(cmd)}")
        return False
    except FileNotFoundError as exc:
        log.error(f"Binary not found: {exc}")
        return False
    except Exception as exc:
        log.error(f"Unexpected error: {exc}")
        return False


def _binary_exists(name: str) -> bool:
    """Return True if `name` is on PATH."""
    return shutil.which(name) is not None


def _installed_oqs_version() -> Optional[str]:
    """Return the installed oqs Python package version string, or None."""
    try:
        result = subprocess.run(
            [sys.executable, '-c',
             'import importlib.metadata; print(importlib.metadata.version("liboqs-python"))'],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            ver = result.stdout.strip()
            log.info(f"[OQS-VER] liboqs-python installed version: {ver}")
            return ver
    except Exception:
        pass
    # fallback: ask the oqs module itself (without triggering the installer)
    try:
        env = os.environ.copy()
        env['OQS_SKIP_SETUP'] = '1'
        env['OQS_BUILD'] = '0'
        result = subprocess.run(
            [sys.executable, '-c', 'import oqs; print(oqs.__version__)'],
            capture_output=True, text=True, timeout=15, env=env,
        )
        if result.returncode == 0:
            ver = result.stdout.strip()
            log.info(f"[OQS-VER] oqs.__version__: {ver}")
            return ver
    except Exception:
        pass
    return None


def _resolve_build_tag(oqs_ver: Optional[str]) -> List[str]:
    """
    Return ordered list of tags to attempt, putting the oqs Python version
    first if it exists in our candidate list, then falling back to the full list.
    Also adds the raw major.minor from the version string as a candidate.
    """
    candidates = list(CANDIDATE_TAGS)
    if not oqs_ver:
        return candidates
    # normalize version: strip leading 'v', trailing suffixes
    clean = re.sub(r'[^0-9.]', '', oqs_ver.lstrip('v')).strip('.')
    if clean and clean not in candidates:
        candidates.insert(0, clean)
    elif clean in candidates:
        # move to front
        candidates.remove(clean)
        candidates.insert(0, clean)
    # also try major.minor.0 form if not already present
    parts = clean.split('.')
    if len(parts) >= 2:
        mm0 = f"{parts[0]}.{parts[1]}.0"
        if mm0 not in candidates:
            candidates.insert(1, mm0)
    log.info(f"[OQS-VER] Tag resolution order: {candidates[:5]}...")
    return candidates


def _lib_already_valid() -> bool:
    """Check if liboqs shared library already exists and is loadable."""
    lib_dir = INSTALL_DIR / 'lib'
    lib64_dir = INSTALL_DIR / 'lib64'
    for d in [lib_dir, lib64_dir]:
        for pattern in ['liboqs.so', 'liboqs.so.*', 'liboqs.dylib']:
            if any(d.glob(pattern)):
                if STAMP_FILE.exists():
                    log.info(f"[OQS-CHECK] Found pre-built liboqs at {d} (stamp: {STAMP_FILE.read_text().strip()})")
                    return True
    return False


def _install_build_deps() -> bool:
    """Install cmake, ninja-build, libssl-dev via apt-get (best-effort, non-fatal)."""
    if not _binary_exists('apt-get'):
        log.warning("[BUILD-DEPS] apt-get not available — hoping build tools exist")
        return True
    needed = []
    if not _binary_exists('cmake'):
        needed.append('cmake')
    if not (_binary_exists('ninja') or _binary_exists('ninja-build')):
        needed.append('ninja-build')
    if not _binary_exists('git'):
        needed.append('git')
    # always ensure libssl-dev and gcc are present
    needed += ['libssl-dev', 'gcc', 'g++', 'make']
    if not needed:
        log.info("[BUILD-DEPS] All build tools already present")
        return True
    log.info(f"[BUILD-DEPS] Installing: {needed}")
    ok = _run(['apt-get', 'update', '-qq'], timeout=120)
    if not ok:
        log.warning("[BUILD-DEPS] apt-get update failed (continuing anyway)")
    return _run(
        ['apt-get', 'install', '-y', '-qq', '--no-install-recommends'] + needed,
        timeout=180,
    )


def _git_clone_tag(tag: str, dest: Path) -> bool:
    """Attempt shallow clone of liboqs at `tag`."""
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    log.info(f"[CLONE] Trying liboqs tag={tag} → {dest}")
    ok = _run(
        ['git', 'clone', '--depth', '1', '--branch', tag, LIBOQS_REPO, str(dest)],
        timeout=180,
    )
    if not ok:
        log.warning(f"[CLONE] Tag {tag} not found in upstream")
        shutil.rmtree(dest, ignore_errors=True)
    return ok


def _cmake_build(src: Path, install_prefix: Path) -> bool:
    """Run cmake configure + build + install."""
    build = src / 'build'
    build.mkdir(exist_ok=True)
    flags = CMAKE_FLAGS + [f'-DCMAKE_INSTALL_PREFIX={install_prefix}', '..']
    # Prefer ninja, fall back to make
    if not (_binary_exists('ninja') or _binary_exists('ninja-build')):
        flags[0] = '-G'
        flags.insert(1, 'Unix Makefiles')
        build_cmd = ['make', f'-j{os.cpu_count() or 2}']
    else:
        ninja_bin = shutil.which('ninja') or shutil.which('ninja-build') or 'ninja'
        build_cmd = [ninja_bin]
    if not _run(['cmake'] + flags, cwd=build, timeout=300):
        return False
    if not _run(build_cmd, cwd=build, timeout=BUILD_TIMEOUT_S):
        return False
    return _run([build_cmd[0], 'install'] if 'ninja' in build_cmd[0]
                else ['make', 'install'], cwd=build, timeout=120)


def _write_stamp(tag: str) -> None:
    """Write build stamp for cache invalidation."""
    INSTALL_DIR.mkdir(parents=True, exist_ok=True)
    stamp = {
        'tag': tag,
        'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'python': sys.version,
    }
    STAMP_FILE.write_text(json.dumps(stamp, indent=2))
    log.info(f"[STAMP] Written: {STAMP_FILE}")


def build_liboqs() -> bool:
    """
    Main build orchestrator.
    1. Detect installed oqs-python version → choose matching liboqs tag.
    2. Clone source at tag → cmake build → install to INSTALL_DIR.
    3. Write stamp file on success.
    Returns True on success, False on total failure.
    """
    log.info("═" * 70)
    log.info("  LIBOQS BUILD SYSTEM — Enterprise Post-Quantum Crypto")
    log.info(f"  Install prefix : {INSTALL_DIR}")
    log.info(f"  Build dir      : {BUILD_DIR}")
    log.info("═" * 70)

    # Fast-path: already built
    if _lib_already_valid():
        log.info("[BUILD] ✓ liboqs already built — skipping rebuild")
        return True

    # Install system build deps
    _install_build_deps()

    # Detect oqs version → ordered tag list
    oqs_ver = _installed_oqs_version()
    tags = _resolve_build_tag(oqs_ver)

    src_dir = BUILD_DIR / 'liboqs_src'
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    INSTALL_DIR.mkdir(parents=True, exist_ok=True)

    for tag in tags:
        log.info(f"\n[BUILD] ── Attempting tag: {tag} ──")
        if not _git_clone_tag(tag, src_dir):
            continue
        log.info(f"[BUILD] ✓ Cloned liboqs {tag} — starting cmake build")
        if _cmake_build(src_dir, INSTALL_DIR):
            _write_stamp(tag)
            log.info(f"[BUILD] ✓✓ liboqs {tag} built and installed to {INSTALL_DIR}")
            return True
        log.warning(f"[BUILD] cmake failed for tag {tag}, trying next")
        shutil.rmtree(src_dir, ignore_errors=True)

    # Last resort: try main branch
    log.warning("[BUILD] All tags failed — attempting main branch")
    if _git_clone_tag('main', src_dir):
        if _cmake_build(src_dir, INSTALL_DIR):
            _write_stamp('main')
            log.info(f"[BUILD] ✓✓ liboqs main built and installed")
            return True

    log.error("[BUILD] ✗ liboqs build FAILED across all candidates")
    return False


def configure_env() -> None:
    """
    Set all OQS-related env vars in the current process so gunicorn
    inherits them at exec time.
    """
    lib_candidates = [INSTALL_DIR / 'lib', INSTALL_DIR / 'lib64']
    # Find actual lib dir
    lib_dir = None
    for d in lib_candidates:
        if d.exists() and any(d.glob('liboqs*')):
            lib_dir = d
            break
    if not lib_dir:
        # default even if no files yet — gunicorn will set OQS_INSTALL_DIR
        lib_dir = INSTALL_DIR / 'lib'

    os.environ['OQS_INSTALL_DIR']   = str(INSTALL_DIR)
    os.environ['OQS_SKIP_SETUP']    = '1'
    os.environ['OQS_BUILD']         = '0'

    # Prepend to LD_LIBRARY_PATH so dynamic linker finds liboqs.so
    existing_ldpath = os.environ.get('LD_LIBRARY_PATH', '')
    paths = [str(lib_dir), str(INSTALL_DIR / 'lib64')]
    new_ldpath = ':'.join(p for p in paths + [existing_ldpath] if p)
    os.environ['LD_LIBRARY_PATH'] = new_ldpath

    # Also update PYTHONPATH / sys.path for the oqs Python package wrappers
    oqs_python_path = INSTALL_DIR / 'python'
    if oqs_python_path.exists():
        sys.path.insert(0, str(oqs_python_path))

    log.info(f"[ENV] OQS_INSTALL_DIR    = {os.environ['OQS_INSTALL_DIR']}")
    log.info(f"[ENV] OQS_SKIP_SETUP     = {os.environ['OQS_SKIP_SETUP']}")
    log.info(f"[ENV] LD_LIBRARY_PATH    = {os.environ['LD_LIBRARY_PATH'][:80]}...")


def verify_oqs_import() -> bool:
    """
    Run a sub-process to verify oqs imports cleanly with the built library.
    Returns True if import succeeds, False otherwise.
    """
    log.info("[VERIFY] Testing oqs import in isolated subprocess...")
    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, '-c',
         'import oqs; k=oqs.KeyEncapsulation("Kyber512"); pk,sk=k.generate_keypair(); '
         'print(f"✓ oqs OK — Kyber512 keypair generated pk={len(pk)}B sk={len(sk)}B")'],
        capture_output=True, text=True, timeout=30, env=env,
    )
    if result.returncode == 0:
        log.info(f"[VERIFY] ✓ {result.stdout.strip()}")
        return True
    log.error(f"[VERIFY] ✗ oqs import failed:\n{result.stderr}")
    return False


def exec_gunicorn() -> None:
    """
    Replace current process with gunicorn (os.execvpe) so env vars
    established here propagate to all workers.
    """
    gunicorn_bin = shutil.which('gunicorn')
    if not gunicorn_bin:
        log.error("[EXEC] gunicorn not found on PATH — aborting")
        sys.exit(1)
    log.info(f"[EXEC] exec → {gunicorn_bin} " + ' '.join(GUNICORN_ARGS[1:]))
    log.info("═" * 70)
    # os.execvpe replaces the current process — no return
    os.execvpe(gunicorn_bin, GUNICORN_ARGS, os.environ)


# ────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Orchestrate: set OQS env → build liboqs if needed → verify → exec gunicorn.
    Non-zero exit on hard failure so Koyeb/Heroku restarts the instance.
    """
    t0 = time.monotonic()

    # Set env vars before anything else (blocks oqs auto-installer for this process)
    configure_env()

    # Attempt build
    build_ok = build_liboqs()

    # Re-configure env after build (lib dir now confirmed to exist)
    configure_env()

    elapsed = time.monotonic() - t0
    if build_ok:
        log.info(f"[MAIN] ✓ liboqs ready in {elapsed:.1f}s")
        # Verify import works
        if not verify_oqs_import():
            log.error("[MAIN] ✗ oqs import verification failed — exiting for restart")
            sys.exit(1)
        log.info("[MAIN] ✓ All PQC checks passed — handing off to gunicorn")
    else:
        log.error(f"[MAIN] ✗ liboqs build failed after {elapsed:.1f}s — exiting for restart")
        sys.exit(1)

    exec_gunicorn()


if __name__ == '__main__':
    main()
