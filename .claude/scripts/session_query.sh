#!/bin/bash

# Session query script - retrieves session metadata or full session data
# Usage:
#   ./session_query.sh -n <count>     # Get N most recent sessions
#   ./session_query.sh -s <sessionid>  # Get full session data by ID

PROJECTS_DIR="/app/workspace/.claude-home/projects/-app-workspace"

# Helper function to truncate text to N words
truncate_words() {
  local text="$1"
  local max_words="$2"
  echo "$text" | tr '\n' ' ' | awk -v m="$max_words" '{
    words = split($0, arr, " ");
    if (words > m) {
      for (i=1; i<=m; i++) printf "%s ", arr[i];
      printf "...";
    } else {
      print $0;
    }
  }'
}

# Helper function to extract first user message content
get_first_user_message() {
  local file="$1"
  grep '"type":"user"' "$file" | head -1 | grep -o '"content":"[^"]*"' | head -1 | sed 's/"content":"//' | sed 's/"$//'
}

# Helper function to get session metadata
get_session_metadata() {
  local session_file="$1"
  local session_id="$2"

  if [ ! -f "$session_file" ]; then
    return 1
  fi

  # Get first and last timestamps
  local first_timestamp=$(grep '"timestamp"' "$session_file" | head -1 | grep -o '"timestamp":"[^"]*"' | head -1 | sed 's/"timestamp":"//' | sed 's/"$//')
  local last_timestamp=$(grep '"timestamp"' "$session_file" | tail -1 | grep -o '"timestamp":"[^"]*"' | sed 's/"timestamp":"//' | sed 's/"$//')

  # Count user and assistant messages
  local user_count=$(grep -c '"type":"user"' "$session_file")
  local assistant_count=$(grep -c '"type":"assistant"' "$session_file")
  local total_messages=$((user_count + assistant_count))

  # Get first user message
  local first_user_msg=$(get_first_user_message "$session_file")
  local truncated_msg=$(truncate_words "$first_user_msg" 100)

  # Get model used (from first assistant message)
  local model=$(grep '"type":"assistant"' "$session_file" | grep '"model"' | head -1 | grep -o '"model":"[^"]*"' | sed 's/"model":"//' | sed 's/"$//')

  # Output as JSON
  printf '{"sessionId":"%s","firstUserMessage":"%s","totalMessages":%d,"firstMessageTimestamp":"%s","lastMessageTimestamp":"%s","model":"%s"}' \
    "$session_id" \
    "$(echo "$truncated_msg" | sed 's/"/\\"/g')" \
    "$total_messages" \
    "$first_timestamp" \
    "$last_timestamp" \
    "$model"
}

# Main logic
if [[ "$1" == "-n" && -n "$2" ]]; then
  # Get N most recent sessions
  count="$2"

  # Get most recent session files by modification time
  echo "["
  first=true
  ls -t "$PROJECTS_DIR"/*.jsonl 2>/dev/null | head -n "$count" | while read session_file; do
    session_id=$(basename "$session_file" .jsonl)

    if [ "$first" != "true" ]; then
      echo ","
    fi

    get_session_metadata "$session_file" "$session_id"
    first=false
  done
  echo
  echo "]"

elif [[ "$1" == "-s" && -n "$2" ]]; then
  # Get full session data by ID
  session_id="$2"
  session_file="$PROJECTS_DIR/${session_id}.jsonl"

  if [ ! -f "$session_file" ]; then
    echo "{\"error\":\"Session not found: $session_id\"}" >&2
    exit 1
  fi

  cat "$session_file"

else
  echo "Usage:" >&2
  echo "  $0 -n <count>      # Get N most recent sessions" >&2
  echo "  $0 -s <sessionid>  # Get full session data by ID" >&2
  exit 1
fi
