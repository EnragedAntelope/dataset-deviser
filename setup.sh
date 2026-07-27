#!/usr/bin/env bash
# LoRA Dataset Studio - one-time setup (Linux / macOS). Safe to re-run: use it to
# install dependencies added by a `git pull`, or to change an API key.
set -euo pipefail
cd "$(dirname "$0")"

echo "=== LoRA Dataset Studio setup ==="
echo

# --- Python present and new enough? ---
PY=python3
command -v "$PY" >/dev/null 2>&1 || {
    echo "[ERROR] python3 not found — install Python 3.10 or newer first."
    exit 1
}
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)'; then
    echo "[ERROR] $($PY --version) is too old — this project needs Python 3.10+."
    echo "        Install a newer Python, delete the .venv folder, then re-run ./setup.sh"
    exit 1
fi
echo "Found $($PY --version)"
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info[:2] <= (3, 13) else 1)'; then
    echo "[note] That is newer than the versions this project is tested against"
    echo "       (3.10-3.13). It usually works, but if pip cannot find a torch or"
    echo "       onnxruntime wheel, use Python 3.13 instead."
    echo
fi

# --- venv ---
if [ ! -d .venv ]; then
    echo "Creating virtual environment..."
    "$PY" -m venv .venv
fi
PIP=.venv/bin/pip
VENV_PY=.venv/bin/python

# --- torch + ONNX Runtime: CUDA if an NVIDIA GPU is present, else platform default ---
if command -v nvidia-smi >/dev/null 2>&1; then
    WANT=gpu
    echo "NVIDIA GPU detected - installing CUDA build of PyTorch..."
    "$PIP" install torch torchvision --index-url https://download.pytorch.org/whl/cu128
else
    WANT=cpu
    echo "No NVIDIA GPU detected - installing default PyTorch build."
    echo "NOTE: local captioning/isolation models are very slow without a GPU"
    echo "      (Apple Silicon uses MPS and is usable). Cloud captioners"
    echo "      (Gemini/Groq) work fine without a GPU."
    "$PIP" install torch torchvision
fi

echo
echo "Installing dependencies..."
"$PIP" install -r requirements.txt

# ONNX Runtime for the WD/e621 taggers (③), matched to the chosen build. Kept
# out of requirements.txt so the CPU vs CUDA variant tracks the torch install.
echo
echo "Installing ONNX Runtime ($WANT) for the taggers..."
"$PIP" uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
if [ "$WANT" = gpu ]; then
    "$PIP" install onnxruntime-gpu \
        || echo "[warn] onnxruntime-gpu failed - the taggers can still use the CPU build: pip install onnxruntime"
else
    "$PIP" install onnxruntime \
        || echo "[warn] onnxruntime failed - the taggers need it; install it later with: pip install onnxruntime"
fi

# --- optional API keys -> .env ---
# Handled by cli.py, NOT here: this script used to append keys unconditionally, so
# every re-run duplicated them, and it could not show or change an existing value.
# One tested Python implementation instead (studio/env_keys.py).
echo
"$VENV_PY" cli.py keys --setup \
    || echo "[warn] Key setup was interrupted - run 'python cli.py keys' any time to set them."

# --- final report: dependencies, keys, optional backends ---
echo
echo "=== Checking the finished install ==="
if ! "$VENV_PY" cli.py doctor; then
    echo
    echo "[ERROR] The install check above found a problem. Fix what it lists, then"
    echo "        re-run ./setup.sh. Re-run the check on its own with:"
    echo "            .venv/bin/python cli.py doctor"
    exit 1
fi

echo
echo "=== Setup complete ($WANT build). Run ./start.sh to launch. ==="
echo "    Re-run ./setup.sh any time to install new dependencies after a git pull."
echo "    Change an API key with:  .venv/bin/python cli.py keys"
