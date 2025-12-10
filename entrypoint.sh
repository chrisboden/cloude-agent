#!/bin/bash
# Ensure workspace directory exists and is writable
# Railway volumes mount as root, so we need to fix permissions

WORKSPACE_DIR="${WORKSPACE_DIR:-/app/workspace}"

# Create workspace if it doesn't exist
mkdir -p "$WORKSPACE_DIR"

# If we're root, change ownership to appuser
if [ "$(id -u)" = "0" ]; then
    chown -R appuser:appuser "$WORKSPACE_DIR"
    # Run the app as appuser
    exec su appuser -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"
else
    # Already running as appuser
    exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
fi

