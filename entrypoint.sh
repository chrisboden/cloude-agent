#!/bin/bash
# Ensure workspace, skills, and commands directories exist and are writable
# Railway volumes mount as root, so we need to fix permissions

WORKSPACE_DIR="${WORKSPACE_DIR:-/app/workspace}"
SKILLS_DIR="${SKILLS_DIR:-$WORKSPACE_DIR/.claude/skills}"
COMMANDS_DIR="${COMMANDS_DIR:-$WORKSPACE_DIR/.claude/commands}"
SCRIPTS_DIR="${SCRIPTS_DIR:-$WORKSPACE_DIR/.claude/scripts}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-$WORKSPACE_DIR/artifacts}"
CLAUDE_CONFIG_DIR="${CLAUDE_CONFIG_DIR:-$WORKSPACE_DIR/.claude-home}"
CLAUDE_MEM_DATA_DIR="${CLAUDE_MEM_DATA_DIR:-$WORKSPACE_DIR/.claude-mem}"
CLAUDE_MEM_WORKER_PORT="${CLAUDE_MEM_WORKER_PORT:-37777}"
CLAUDE_MEM_WORKER_HOST="${CLAUDE_MEM_WORKER_HOST:-127.0.0.1}"
CLAUDE_MEM_ENABLED="${CLAUDE_MEM_ENABLED:-1}"

# Create directories if they don't exist
mkdir -p "$WORKSPACE_DIR"
mkdir -p "$SKILLS_DIR"
mkdir -p "$COMMANDS_DIR"
mkdir -p "$SCRIPTS_DIR"
mkdir -p "$ARTIFACTS_DIR"
mkdir -p "$CLAUDE_CONFIG_DIR"
mkdir -p "$CLAUDE_MEM_DATA_DIR"

# Seed default commands/skills from the image into the volume on first run.
# This keeps "one-click deploy" usable while still allowing runtime edits on the volume.
DEFAULT_COMMANDS_SRC="/app/.claude/commands"
DEFAULT_SCRIPTS_SRC="/app/.claude/scripts"
DEFAULT_SKILLS_SRC="/app/.claude/skills"
DEFAULT_SETTINGS_SRC="/app/.claude/settings.json"
DEFAULT_CLAUDE_MD_SRC="/app/.claude/CLAUDE.md"

# Always copy new command files (but don't overwrite existing ones with -n)
if [ -d "$DEFAULT_COMMANDS_SRC" ]; then
    cp -n "$DEFAULT_COMMANDS_SRC"/*.md "$COMMANDS_DIR"/ 2>/dev/null || true
fi

# Also seed helper scripts (non-destructive: only copies missing files).
if [ -d "$DEFAULT_SCRIPTS_SRC" ]; then
    mkdir -p "$SCRIPTS_DIR"
    cp -R -n "$DEFAULT_SCRIPTS_SRC/"* "$SCRIPTS_DIR/" 2>/dev/null || true
fi

if [ -d "$DEFAULT_SKILLS_SRC" ] && [ -z "$(ls -A "$SKILLS_DIR" 2>/dev/null || true)" ]; then
    cp -R -n "$DEFAULT_SKILLS_SRC"/* "$SKILLS_DIR"/ 2>/dev/null || true
fi

# Seed a minimal project settings.json for non-interactive runs (webhook mode can't approve prompts).
if [ -f "$DEFAULT_SETTINGS_SRC" ]; then
    mkdir -p "$WORKSPACE_DIR/.claude"
    cp -n "$DEFAULT_SETTINGS_SRC" "$WORKSPACE_DIR/.claude/settings.json" 2>/dev/null || true
    cp -n "$DEFAULT_SETTINGS_SRC" "$CLAUDE_CONFIG_DIR/settings.json" 2>/dev/null || true
fi

# Seed CLAUDE.md (project context) into the workspace volume (non-destructive).
if [ -f "$DEFAULT_CLAUDE_MD_SRC" ]; then
    mkdir -p "$WORKSPACE_DIR/.claude"
    cp -n "$DEFAULT_CLAUDE_MD_SRC" "$WORKSPACE_DIR/.claude/CLAUDE.md" 2>/dev/null || true
fi

# Make all skill scripts executable
find "$SKILLS_DIR" -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true
find "$SKILLS_DIR" -name "*.py" -exec chmod +x {} \; 2>/dev/null || true
find "$SCRIPTS_DIR" -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true
find "$SCRIPTS_DIR" -name "*.py" -exec chmod +x {} \; 2>/dev/null || true

# If we're root, change ownership to appuser
if [ "$(id -u)" = "0" ]; then
    chown -R appuser:appuser "$WORKSPACE_DIR"
    chown -R appuser:appuser "$CLAUDE_MEM_DATA_DIR"

    # Symlink ~/.claude to persistent volume storage
    # This ensures sessions persist across deployments
    APPUSER_HOME=$(eval echo ~appuser)
    if [ ! -L "$APPUSER_HOME/.claude" ]; then
        rm -rf "$APPUSER_HOME/.claude" 2>/dev/null || true
        ln -sf "$CLAUDE_CONFIG_DIR" "$APPUSER_HOME/.claude"
        chown -h appuser:appuser "$APPUSER_HOME/.claude"
    fi

    export CLAUDE_CONFIG_DIR
    export CLAUDE_MEM_DATA_DIR
    export CLAUDE_MEM_WORKER_PORT
    export CLAUDE_MEM_WORKER_HOST

    if [ "$CLAUDE_MEM_ENABLED" = "1" ] && command -v claude >/dev/null 2>&1; then
        INSTALL_FILE="$CLAUDE_CONFIG_DIR/plugins/installed_plugins.json"
        if ! grep -q '"claude-mem@thedotmack"' "$INSTALL_FILE" 2>/dev/null; then
            su appuser -c "CLAUDE_CONFIG_DIR='$CLAUDE_CONFIG_DIR' CLAUDE_MEM_DATA_DIR='$CLAUDE_MEM_DATA_DIR' CLAUDE_MEM_WORKER_PORT='$CLAUDE_MEM_WORKER_PORT' CLAUDE_MEM_WORKER_HOST='$CLAUDE_MEM_WORKER_HOST' claude plugin marketplace add thedotmack/claude-mem" || true
            su appuser -c "CLAUDE_CONFIG_DIR='$CLAUDE_CONFIG_DIR' CLAUDE_MEM_DATA_DIR='$CLAUDE_MEM_DATA_DIR' CLAUDE_MEM_WORKER_PORT='$CLAUDE_MEM_WORKER_PORT' CLAUDE_MEM_WORKER_HOST='$CLAUDE_MEM_WORKER_HOST' claude plugin install claude-mem@thedotmack || claude plugin install claude-mem" || true
        fi
    fi

    # Run the app as appuser
    exec su appuser -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"
else
    # Already running as appuser - symlink ~/.claude
    if [ ! -L "$HOME/.claude" ]; then
        rm -rf "$HOME/.claude" 2>/dev/null || true
        ln -sf "$CLAUDE_CONFIG_DIR" "$HOME/.claude"
    fi

    export CLAUDE_CONFIG_DIR
    export CLAUDE_MEM_DATA_DIR
    export CLAUDE_MEM_WORKER_PORT
    export CLAUDE_MEM_WORKER_HOST

    if [ "$CLAUDE_MEM_ENABLED" = "1" ] && command -v claude >/dev/null 2>&1; then
        INSTALL_FILE="$CLAUDE_CONFIG_DIR/plugins/installed_plugins.json"
        if ! grep -q '"claude-mem@thedotmack"' "$INSTALL_FILE" 2>/dev/null; then
            CLAUDE_CONFIG_DIR="$CLAUDE_CONFIG_DIR" CLAUDE_MEM_DATA_DIR="$CLAUDE_MEM_DATA_DIR" CLAUDE_MEM_WORKER_PORT="$CLAUDE_MEM_WORKER_PORT" CLAUDE_MEM_WORKER_HOST="$CLAUDE_MEM_WORKER_HOST" claude plugin marketplace add thedotmack/claude-mem || true
            CLAUDE_CONFIG_DIR="$CLAUDE_CONFIG_DIR" CLAUDE_MEM_DATA_DIR="$CLAUDE_MEM_DATA_DIR" CLAUDE_MEM_WORKER_PORT="$CLAUDE_MEM_WORKER_PORT" CLAUDE_MEM_WORKER_HOST="$CLAUDE_MEM_WORKER_HOST" claude plugin install claude-mem@thedotmack || claude plugin install claude-mem || true
        fi
    fi

    exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
fi
