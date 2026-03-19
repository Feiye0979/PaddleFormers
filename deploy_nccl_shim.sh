#!/bin/bash
# NCCL Shim Deployment Script - Conda Environment Only
# Usage: ./deploy_nccl_shim.sh [conda_env_name] [path_to_nccl_2.21.5]
#
# Default source NCCL: /usr/local/nccl/libnccl.so
#
# Example:
#   ./deploy_nccl_shim.sh why_pd
#   ./deploy_nccl_shim.sh why_pd /usr/local/nccl/libnccl.so

set -e

CONDA_ENV=${1:-why_pd}
NCCL_2215_SRC=${2:-/usr/local/nccl/libnccl.so}

CONDA_BASE=$(conda info --base 2>/dev/null || echo "/root/miniconda3")
NCCL_LIB_DIR="$CONDA_BASE/envs/$CONDA_ENV/lib/python3.10/site-packages/nvidia/nccl/lib"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SHIM_SRC="$SCRIPT_DIR/nccl_shim.c"

echo "[INFO] Conda env:  $CONDA_ENV"
echo "[INFO] Target dir: $NCCL_LIB_DIR"
echo "[INFO] NCCL 2.21.5: $NCCL_2215_SRC"

# --- pre-checks ---
if [ ! -d "$NCCL_LIB_DIR" ]; then
    echo "[ERROR] nvidia/nccl/lib not found: $NCCL_LIB_DIR"
    exit 1
fi
if [ ! -f "$NCCL_2215_SRC" ]; then
    echo "[ERROR] NCCL 2.21.5 not found: $NCCL_2215_SRC"
    echo "[HINT]  Expected: /usr/local/nccl/libnccl.so"
    exit 1
fi
if [ ! -f "$SHIM_SRC" ]; then
    echo "[ERROR] Shim source not found: $SHIM_SRC"
    exit 1
fi

cd "$NCCL_LIB_DIR"

# -----------------------------------------------------------------------
# Step 1: Backup original NCCL 2.27.5 (only if no backup yet)
# -----------------------------------------------------------------------
if [ ! -f "libnccl_227.so.2" ]; then
    cp libnccl.so.2 libnccl_227.so.2
    echo "[OK] Backed up NCCL 2.27.5 -> libnccl_227.so.2"
else
    echo "[SKIP] Backup already exists (libnccl_227.so.2)"
fi

# -----------------------------------------------------------------------
# Step 2: Install NCCL 2.21.5 as libnccl_real.so
#         ALSO ensure SONAME is set correctly even if file already exists.
#         Wrong SONAME causes the shim's DT_NEEDED to point to libnccl.so.2
#         (itself), creating a circular dependency that breaks symbol resolution.
# -----------------------------------------------------------------------
if [ ! -f "libnccl_real.so" ]; then
    cp "$NCCL_2215_SRC" libnccl_real.so
    echo "[OK] NCCL 2.21.5 copied -> libnccl_real.so"
else
    echo "[SKIP] libnccl_real.so file already exists"
fi

# Always verify (and fix) the SONAME regardless of whether we just copied it
CURRENT_SONAME=$(readelf -d libnccl_real.so 2>/dev/null | grep SONAME | sed 's/.*\[\(.*\)\]/\1/')
echo "[INFO] libnccl_real.so current SONAME: '${CURRENT_SONAME}'"

WHOLE_ARCHIVE=0
if [ "$CURRENT_SONAME" != "libnccl_real.so" ]; then
    echo "[WARN] SONAME is wrong (should be 'libnccl_real.so'). Fixing..."
    if command -v patchelf &>/dev/null; then
        patchelf --set-soname libnccl_real.so libnccl_real.so
        echo "[OK] SONAME fixed via patchelf"
    else
        echo "[WARN] patchelf not found, trying to install..."
        # Try common install methods
        conda install -n "$CONDA_ENV" patchelf -y -q 2>/dev/null || \
        apt-get install -y -q patchelf 2>/dev/null || true

        if command -v patchelf &>/dev/null; then
            patchelf --set-soname libnccl_real.so libnccl_real.so
            echo "[OK] SONAME fixed via patchelf (newly installed)"
        else
            echo "[WARN] Cannot install patchelf, will use --whole-archive fallback"
            echo "       (shim will be ~108MB instead of ~16KB, but functionally identical)"
            WHOLE_ARCHIVE=1
        fi
    fi
else
    echo "[OK] SONAME is already correct"
fi

# -----------------------------------------------------------------------
# Step 3: Build shim (always rebuild to pick up any SONAME/source changes)
# -----------------------------------------------------------------------
CUDA_INC=""
NCCL_H=$(find /usr/local -path "*/cuda*/include/nccl.h" 2>/dev/null | head -1)
if [ -n "$NCCL_H" ]; then
    CUDA_INC=$(dirname "$NCCL_H")
fi
if [ -z "$CUDA_INC" ]; then
    CUDA_INC=$(find /usr -path "*/cuda-*/include" -type d 2>/dev/null | head -1)
fi
if [ -z "$CUDA_INC" ]; then
    CUDA_INC="/usr/local/cuda/include"
fi
echo "[INFO] CUDA include: $CUDA_INC"

if [ "$WHOLE_ARCHIVE" = "1" ]; then
    # Fallback: embed ALL symbols from libnccl_real.so directly into the shim.
    # This avoids any SONAME/DT_NEEDED issues at the cost of a larger binary.
    gcc -shared -fPIC -o libnccl.so.2 "$SHIM_SRC" \
        -I"$CUDA_INC" \
        -Wl,--whole-archive ./libnccl_real.so -Wl,--no-whole-archive \
        -Wl,-soname,libnccl.so.2
    echo "[OK] Shim built (whole-archive mode): $(stat -c%s libnccl.so.2) bytes"
else
    gcc -shared -fPIC -o libnccl.so.2 "$SHIM_SRC" \
        -I"$CUDA_INC" \
        ./libnccl_real.so \
        -Wl,--disable-new-dtags \
        -Wl,-rpath,\$ORIGIN \
        -Wl,-soname,libnccl.so.2
    echo "[OK] Shim built (normal mode): $(stat -c%s libnccl.so.2) bytes"
fi

# -----------------------------------------------------------------------
# Step 4: Final verification
# -----------------------------------------------------------------------
echo ""
echo "=== DT_NEEDED / RPATH of libnccl.so.2 ==="
readelf -d libnccl.so.2 | grep -E "NEEDED|RPATH|RUNPATH"
echo ""
echo "=== Files ==="
ls -lh libnccl*.so*
echo ""
echo "[DONE] Verify with:"
echo "  conda activate $CONDA_ENV"
echo "  python -c 'import paddle; import torch; print(\"OK\")'"
