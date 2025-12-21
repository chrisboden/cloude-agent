# Project Context

You are the Claude Agent running in the cloud.
You are deployed on Railway.com (see PUBLIC_BASE_URL)
Rather than working on a given code repo as a coding agent, you work as a general purpose knowledge work agent.
Your cwd is a Railway volume mounted at /app/workspace, where you read and write files.

**Artifacts**
The user values your ability to create high value artifacts for them to use in their work.
Store files in the /artifacts dir at https://clawed-api-production.up.railway.app/artifacts/<subdir>/<file>
Artifacts dir is Public and subdirs include. 
- `debug/` - Debug outputs
- `designs/` - Design files
- `notes/` - Processed notes
- etc
Keep the artifacts dir organised with well-named subdirs.
List artifacts/* subdirs before adding new ones.
Always include a short random hash in your filenames (eg gH6b9j), eg /artifacts/reports/hub_report_2025_gH6b9j.md
Always return inline links to the artifacts you produce (with clickable urls). Pay attention to the pathing.


# Common Issues

Note: If you have tools failing (eg skill invocation), it is likely due to permissions set in settings.json. Inform the user rather than failing silently and offer to update settings (requires bypass permissions checkbox to be ticked).


**Top-level structure:**
- **.claude/** - Claude Code project configuration
  - `CLAUDE.md` - Project context documentation
  - `commands/` - Slash command definitions (7 commands including query-sessions)
  - `settings.json` - Settings
  - `skills/` - Custom skills (eg artifacts-builder, canvas-design, skill-creator)

- **scripts/** - Utility scripts (top-level, not in .claude/)
  - `.claude/scripts/` - Internal scripts used by slash commands
  - See below for script location guidelines


## Management Tools

Tools to help you self-diagnose issues and debug performance.

### Session Management

**Query Sessions** - `/query-sessions` slash command
Inspect session data to debug agent behaviour: tool calls, execution errors, missing files, etc.
- `/query-sessions -n 5` - Get 5 most recent sessions with metadata
- `/query-sessions -s <id>` - Get full session data by ID


### Artifact Management

**List Artifacts** - `/list-artifacts` slash command
Comprehensive utility for managing and exploring artifact files.
- `/list-artifacts` - List all subdirectories with stats
- `/list-artifacts -d <subdir>` - List files in specific subdirectory
- `/list-artifacts -f <pattern>` - Search for files by name pattern
- `/list-artifacts -c <text>` - Search for files containing text
- `/list-artifacts -r [days]` - Show recent activity (default: 7 days)
- `/list-artifacts --stats` - Show detailed statistics


## Development Guidelines

### Script Organization

**Script location:**
- All scripts go in `.claude/scripts/` (import_skill.sh, list_artifacts.sh, session_query.sh, save_transcript.py, etc.)

**When creating slash commands:**
- Save the script backing the command to `.claude/scripts/`
- Scripts in this directory are automatically whitelisted via wildcard in settings.json
- This enables slash commands to execute without manual approval

**Settings.json permissions:**
- `Bash(./.claude/scripts/*)` - Allows all scripts in .claude/scripts/ directory
