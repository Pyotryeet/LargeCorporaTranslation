#!/bin/bash
# Sync evaluation results, logs, and output files from remote H200 machine to local machine
# Run this from your local machine: bash sync_from_h200.sh

REMOTE="yigit@192.168.2.9"
REMOTE_DIR="/home/yigit/LargeCorporaTranslation"
LOCAL_DIR="$HOME/Documents/ComputerScience/Projects/H200Research"

echo "=== Syncing results from $REMOTE ==="

# Sync data/output/ directory
echo "--- data/output/ ---"
mkdir -p "$LOCAL_DIR/data/output/"
rsync -avz "$REMOTE:$REMOTE_DIR/data/output/" "$LOCAL_DIR/data/output/"

# Sync logs if any
echo "--- logs/ ---"
mkdir -p "$LOCAL_DIR/logs/"
rsync -avz --include='*.log' --exclude='*' "$REMOTE:$REMOTE_DIR/" "$LOCAL_DIR/logs/"

echo ""
echo "=== Sync complete ==="
