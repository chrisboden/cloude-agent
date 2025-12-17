#!/bin/bash

# Artifacts management utility
# Usage:
#   ./list_artifacts.sh                    # List all artifact subdirectories with stats
#   ./list_artifacts.sh -d <subdir>        # List files in a specific subdirectory
#   ./list_artifacts.sh -s                 # Show only summary statistics
#   ./list_artifacts.sh -f <pattern>       # Search for files by name pattern
#   ./list_artifacts.sh -c <text>          # Search for files containing text
#   ./list_artifacts.sh -r [days]          # Show recent activity (default: 7 days)
#   ./list_artifacts.sh --stats            # Show detailed statistics

ARTIFACTS_DIR="/app/workspace/artifacts"

show_summary() {
  echo "Artifact Subdirectories Summary:"
  echo "================================"
  echo

  printf "%-30s %10s %15s\n" "Directory" "Files" "Total Size"
  printf "%-30s %10s %15s\n" "---------" "-----" "----------"

  for dir in "$ARTIFACTS_DIR"/*/; do
    if [ -d "$dir" ]; then
      dirname=$(basename "$dir")
      filecount=$(find "$dir" -maxdepth 1 -type f | wc -l)
      dirsize=$(du -sh "$dir" 2>/dev/null | cut -f1)
      printf "%-30s %10d %15s\n" "$dirname" "$filecount" "$dirsize"
    fi
  done

  echo
  echo "Root-level files:"
  filecount=$(find "$ARTIFACTS_DIR" -maxdepth 1 -type f | wc -l)
  echo "  Count: $filecount"
}

list_directory() {
  local subdir="$1"
  local dir="$ARTIFACTS_DIR/$subdir"

  if [ ! -d "$dir" ]; then
    echo "Error: Directory not found: $subdir" >&2
    echo "Available directories:" >&2
    ls -d "$ARTIFACTS_DIR"/*/ 2>/dev/null | xargs -I {} basename {} >&2
    exit 1
  fi

  echo "Contents of $subdir:"
  echo "==================="
  echo

  ls -lh "$dir" | tail -n +2 | awk '{print $9, "(" $5 ")"}'
}

search_files() {
  local pattern="$1"
  echo "Searching for files matching: $pattern"
  echo "======================================="
  echo

  find "$ARTIFACTS_DIR" -type f -name "*${pattern}*" -printf "%T+ %p\n" | sort -r | while read timestamp path; do
    size=$(du -h "$path" 2>/dev/null | cut -f1)
    relpath=${path#$ARTIFACTS_DIR/}
    echo "$relpath ($size)"
  done
}

search_content() {
  local text="$1"
  echo "Searching for content: $text"
  echo "============================"
  echo

  grep -r -l -i "$text" "$ARTIFACTS_DIR" 2>/dev/null | while read path; do
    size=$(du -h "$path" 2>/dev/null | cut -f1)
    relpath=${path#$ARTIFACTS_DIR/}
    # Show matching line
    match=$(grep -i -n "$text" "$path" | head -1)
    echo "$relpath ($size)"
    echo "  → $match"
  done
}

show_recent() {
  local days="${1:-7}"
  echo "Files modified in last $days days:"
  echo "=================================="
  echo

  find "$ARTIFACTS_DIR" -type f -mtime -"$days" -printf "%T+ %p\n" | sort -r | while read timestamp path; do
    size=$(du -h "$path" 2>/dev/null | cut -f1)
    relpath=${path#$ARTIFACTS_DIR/}
    date=$(echo "$timestamp" | cut -d+ -f1)
    echo "[$date] $relpath ($size)"
  done
}

show_stats() {
  echo "Artifact Statistics:"
  echo "==================="
  echo

  # Total counts
  total_files=$(find "$ARTIFACTS_DIR" -type f | wc -l)
  total_dirs=$(find "$ARTIFACTS_DIR" -type d | wc -l)
  total_size=$(du -sh "$ARTIFACTS_DIR" 2>/dev/null | cut -f1)

  echo "Total files: $total_files"
  echo "Total directories: $total_dirs"
  echo "Total size: $total_size"
  echo

  # File types
  echo "File types:"
  find "$ARTIFACTS_DIR" -type f -name "*.*" | sed 's/.*\.//' | sort | uniq -c | sort -rn | head -10 | while read count ext; do
    printf "  %-10s %5d files\n" ".$ext" "$count"
  done
  echo

  # Age distribution
  echo "Age distribution:"
  echo "  Last 24 hours:  $(find "$ARTIFACTS_DIR" -type f -mtime -1 | wc -l) files"
  echo "  Last 7 days:    $(find "$ARTIFACTS_DIR" -type f -mtime -7 | wc -l) files"
  echo "  Last 30 days:   $(find "$ARTIFACTS_DIR" -type f -mtime -30 | wc -l) files"
  echo "  Older:          $(find "$ARTIFACTS_DIR" -type f -mtime +30 | wc -l) files"
  echo

  # Largest files
  echo "Largest files:"
  find "$ARTIFACTS_DIR" -type f -printf "%s %p\n" | sort -rn | head -5 | while read size path; do
    human_size=$(numfmt --to=iec-i --suffix=B "$size" 2>/dev/null || echo "$size bytes")
    relpath=${path#$ARTIFACTS_DIR/}
    echo "  $human_size - $relpath"
  done
}

# Main logic
case "$1" in
  -s)
    show_summary
    ;;
  -d)
    if [ -z "$2" ]; then
      echo "Error: -d requires a subdirectory name" >&2
      exit 1
    fi
    list_directory "$2"
    ;;
  -f)
    if [ -z "$2" ]; then
      echo "Error: -f requires a search pattern" >&2
      exit 1
    fi
    search_files "$2"
    ;;
  -c)
    if [ -z "$2" ]; then
      echo "Error: -c requires search text" >&2
      exit 1
    fi
    search_content "$2"
    ;;
  -r)
    show_recent "${2:-7}"
    ;;
  --stats)
    show_stats
    ;;
  -h|--help)
    echo "Usage:"
    echo "  $0                    # List all artifact subdirectories with stats"
    echo "  $0 -d <subdir>        # List files in a specific subdirectory"
    echo "  $0 -s                 # Show only summary statistics"
    echo "  $0 -f <pattern>       # Search for files by name pattern"
    echo "  $0 -c <text>          # Search for files containing text"
    echo "  $0 -r [days]          # Show recent activity (default: 7 days)"
    echo "  $0 --stats            # Show detailed statistics"
    echo "  $0 -h                 # Show this help"
    ;;
  *)
    show_summary
    ;;
esac
