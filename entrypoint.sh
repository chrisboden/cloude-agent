#!/bin/bash
# Ensure workspace, skills, and commands directories exist and are writable
# Railway volumes mount as root, so we need to fix permissions

WORKSPACE_DIR="${WORKSPACE_DIR:-/app/workspace}"
SKILLS_DIR="${SKILLS_DIR:-$WORKSPACE_DIR/.claude/skills}"
COMMANDS_DIR="${COMMANDS_DIR:-$WORKSPACE_DIR/.claude/commands}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-$WORKSPACE_DIR/artifacts}"
CLAUDE_CONFIG_DIR="${WORKSPACE_DIR}/.claude-home"

# Create directories if they don't exist
mkdir -p "$WORKSPACE_DIR"
mkdir -p "$SKILLS_DIR"
mkdir -p "$COMMANDS_DIR"
mkdir -p "$ARTIFACTS_DIR"
mkdir -p "$CLAUDE_CONFIG_DIR"

# Make all skill scripts executable
find "$SKILLS_DIR" -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true
find "$SKILLS_DIR" -name "*.py" -exec chmod +x {} \; 2>/dev/null || true

# If we're root, change ownership to appuser
if [ "$(id -u)" = "0" ]; then
    chown -R appuser:appuser "$WORKSPACE_DIR"

    # Symlink ~/.claude to persistent volume storage
    # This ensures sessions persist across deployments
    APPUSER_HOME=$(eval echo ~appuser)
    if [ ! -L "$APPUSER_HOME/.claude" ]; then
        rm -rf "$APPUSER_HOME/.claude" 2>/dev/null || true
        ln -sf "$CLAUDE_CONFIG_DIR" "$APPUSER_HOME/.claude"
        chown -h appuser:appuser "$APPUSER_HOME/.claude"
    fi

    # Run the app as appuser
    exec su appuser -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"
else
    # Already running as appuser - symlink ~/.claude
    if [ ! -L "$HOME/.claude" ]; then
        rm -rf "$HOME/.claude" 2>/dev/null || true
        ln -sf "$CLAUDE_CONFIG_DIR" "$HOME/.claude"
    fi

    exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
fi




