#!/bin/bash
# Sync H200Research code to remote H200 machine for testing
# Run this from your local machine: bash sync_to_h200.sh

REMOTE="yigit@192.168.2.9"
REMOTE_DIR="/home/yigit/LargeCorporaTranslation"
LOCAL_DIR="$HOME/Documents/ComputerScience/Projects/H200Research"

echo "=== Syncing code to $REMOTE ==="

# Sync benchmark/ directory
echo "--- benchmark/ ---"
rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='.pytest_cache' --exclude='*.egg-info' \
  "$LOCAL_DIR/benchmark/" "$REMOTE:$REMOTE_DIR/benchmark/"

# Sync quantization/ directory
echo "--- quantization/ ---"
rsync -avz --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='.pytest_cache' \
  "$LOCAL_DIR/quantization/" "$REMOTE:$REMOTE_DIR/quantization/"

# Sync tests/ directory
echo "--- tests/ ---"
rsync -avz --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='.pytest_cache' \
  "$LOCAL_DIR/tests/" "$REMOTE:$REMOTE_DIR/tests/"

# Sync root config files and README
echo "--- config files and README ---"
rsync -avz "$LOCAL_DIR/pyproject.toml" "$REMOTE:$REMOTE_DIR/pyproject.toml"
rsync -avz "$LOCAL_DIR/README.md" "$REMOTE:$REMOTE_DIR/README.md"

# Sync docs/ directory
echo "--- docs/ ---"
rsync -avz "$LOCAL_DIR/docs/" "$REMOTE:$REMOTE_DIR/docs/"

echo ""
echo "=== Sync complete ==="
echo ""
echo "To run tests on remote:"
echo "  ssh $REMOTE 'cd $REMOTE_DIR && .venv/bin/python -m pytest tests/ -v --timeout=60 2>&1 | tail -50'"
echo ""
echo "To run a quick benchmark:"
echo "  ssh $REMOTE 'cd $REMOTE_DIR && .venv/bin/python -m benchmark --quick 2>&1'"
