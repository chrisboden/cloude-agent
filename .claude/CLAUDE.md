# Project Context

You are the Cloude Agent.
Under the hood, you are an instance of Claude Agent running in the cloud.
You are deployed on Railway.com at https://cloude-agent-production.up.railway.app
Your cwd is a Railway volume, where you read and write files.

**Top-level structure:**
- **.claude/** - Claude Code project configuration
  - `CLAUDE.md` - Project context documentation
  - `commands/` - Slash command definitions
  - `scripts/` - Utility scripts
  - `settings.json` - Settings
  - `skills/` - Custom skills (eg artifacts-builder, canvas-design, skill-creator)

- **.claude-home/** - Claude Agent runtime home directory
  - `debug/` - Debug files for sessions
  - `plans/` - Planning files
  - `plugins/` - Plugin configuration
  - `projects/` - Session data organized by project
  - `shell-snapshots/` - Shell state snapshots
  - `statsig/` - Feature flag evaluations
  - `session-env/` - Session environment (excluded from tree)
  - `todos/` - Todo items storage (excluded from tree)

- **artifacts/** - Outputs and working files (Public)
- **lost+found/** - System directory
- **prompts/** - System instructions to override CLAUDE.md

## Git Hygiene

- Treat `/app/workspace` as the **platform repo**: track changes to `.claude/**`, `prompts/**`, and other global config/docs. Do **not** commit `artifacts/` or `.claude-home/`.
- Each project under `/app/workspace/artifacts/<project>` should have its **own git repo**. If missing, initialize it with `/.claude/scripts/init_artifact_git.sh <project>`.
- When making changes:
  - Use `git -C /app/workspace …` for platform config changes.
  - Use `git -C /app/workspace/artifacts/<project> …` for project changes.
- Commit only the files you touched; keep commits small and descriptive. One logical change per commit.
- Never commit secrets (`.env`, tokens) or large binaries unless explicitly asked.
- Do not add remotes or push unless the user explicitly requests it.

If git commands are blocked or unavailable, report it and continue without committing.


## Management Tools

### Session Analysis

**Query Sessions** - `/query-sessions` slash command
Inspect session data to debug agent behaviour: tool calls, execution errors, missing files, etc.

**Usage:**
- `/query-sessions -n 5` - Get 5 most recent sessions with metadata
- `/query-sessions -s <id>` - Get full session data by ID

**Script:** `.claude/scripts/session_query.sh`

**Session metadata includes:**
- sessionId - Unique session identifier
- firstUserMessage - First user message (truncated to 100 words)
- totalMessages - Total message count (user + assistant)
- firstMessageTimestamp - Session start timestamp
- lastMessageTimestamp - Session end timestamp
- model - Claude model used in the session

### Artifact Management

**List Artifacts** - `/list-artifacts` slash command
Comprehensive utility for managing and exploring artifact files.

**Usage:**
- `/list-artifacts` - List all subdirectories with stats
- `/list-artifacts -d <subdir>` - List files in specific subdirectory
- `/list-artifacts -f <pattern>` - Search for files by name pattern
- `/list-artifacts -c <text>` - Search for files containing text
- `/list-artifacts -r [days]` - Show recent activity (default: 7 days)
- `/list-artifacts --stats` - Show detailed statistics

**Script:** `.claude/scripts/list_artifacts.sh`

**Features:**
- Directory listing with file counts and sizes
- File name pattern search
- Content search across all files
- Recent activity tracking (modified files)
- Statistics: file types, age distribution, largest files, total counts