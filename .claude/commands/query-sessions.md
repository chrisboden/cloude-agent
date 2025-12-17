---
allowed-tools: Bash(./.claude/scripts/session_query.sh*)
description: Query Claude session metadata and data (project)
argument-hint: [-n count | -s sessionId]
---

# Query Sessions

Execute the session query script with the provided arguments.

## Usage

- Get recent sessions: `/query-sessions -n 5`
- Get specific session: `/query-sessions -s <sessionId>`

## Command

```bash
./.claude/scripts/session_query.sh $ARGUMENTS
```

The session metadata includes:
- Session ID
- First user message (truncated to 100 words)
- Total message count
- First and last message timestamps
- Model used
