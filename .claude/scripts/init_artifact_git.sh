#!/bin/bash
set -euo pipefail

ARTIFACTS_DIR="/app/workspace/artifacts"

usage() {
  echo "Usage: $0 <artifact_dir_or_name>"
  echo
  echo "Examples:"
  echo "  $0 demo-project"
  echo "  $0 /app/workspace/artifacts/demo-project"
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ $# -lt 1 ]; then
  usage >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "Error: git is not installed in this environment." >&2
  exit 1
fi

target_input="$1"
if [[ "$target_input" = /* ]]; then
  target_dir="$target_input"
else
  target_dir="$ARTIFACTS_DIR/$target_input"
fi

mkdir -p "$target_dir"

resolved_root=$(realpath -m "$ARTIFACTS_DIR")
resolved_target=$(realpath -m "$target_dir")

case "$resolved_target" in
  "$resolved_root"|"$resolved_root"/*) ;;
  *)
    echo "Error: target must be under $ARTIFACTS_DIR" >&2
    exit 1
    ;;
esac

if git -C "$resolved_target" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Git repo already initialized at $resolved_target"
else
  git -C "$resolved_target" init >/dev/null
  echo "Initialized git repo at $resolved_target"
fi

if [ ! -f "$resolved_target/.gitignore" ]; then
  cat > "$resolved_target/.gitignore" <<'EOF'
.env
.env.*
*.log
.DS_Store

node_modules/
dist/
build/

.venv/
__pycache__/
*.pyc
EOF
  echo "Created .gitignore at $resolved_target/.gitignore"
fi

if [ -d "$resolved_target/.git" ]; then
  if ! git -C "$resolved_target" config --get user.name >/dev/null; then
    git -C "$resolved_target" config user.name "Cloude Agent"
  fi
  if ! git -C "$resolved_target" config --get user.email >/dev/null; then
    git -C "$resolved_target" config user.email "agent@cloude.local"
  fi
fi
