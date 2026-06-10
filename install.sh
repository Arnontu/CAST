#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/cast_venv"

echo "Creating virtual environment at $VENV_DIR..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "Installing requirements..."
pip install --upgrade pip
pip install -r "$SCRIPT_DIR/requirements.txt"

echo "Installing WhiStress from source..."
# WhiStress has no PyPI package — clone and install deps manually
WHISTRESS_DIR="$SCRIPT_DIR/third_party/WhiStress"
if [ ! -d "$WHISTRESS_DIR" ]; then
    mkdir -p "$SCRIPT_DIR/third_party"
    git clone https://github.com/slp-rl/WhiStress.git "$WHISTRESS_DIR"
else
    echo "WhiStress already cloned, pulling latest..."
    git -C "$WHISTRESS_DIR" pull
fi
pip install -r "$WHISTRESS_DIR/requirements.txt"
# WhiStress has no setup.py — add it to the venv's path via a .pth file
SITE_PACKAGES="$("$VENV_DIR/bin/python" -c "import site; print(site.getsitepackages()[0])")"
echo "$WHISTRESS_DIR" > "$SITE_PACKAGES/whistress.pth"
python "$WHISTRESS_DIR/download_weights.py"

echo ""
echo "Done. Activate the environment with:"
echo "  source cast_venv/bin/activate"
