---
allowed-tools: Bash(./.claude/scripts/list_artifacts.sh*), Bash(/app/workspace/.claude/scripts/list_artifacts.sh*), Bash(bash .claude/scripts/list_artifacts.sh*), Bash(bash /app/workspace/.claude/scripts/list_artifacts.sh*), Bash(bash:./.claude/scripts/list_artifacts.sh*), Bash(bash:.claude/scripts/list_artifacts.sh*), Bash(bash:/app/workspace/.claude/scripts/list_artifacts.sh*)
description: List and explore artifact subdirectories (project)
---

Comprehensive artifact management utility for listing, searching, and analyzing artifact files.

**List all artifact subdirectories:**
```bash
./.claude/scripts/list_artifacts.sh
```

**List files in a specific subdirectory:**
```bash
./.claude/scripts/list_artifacts.sh -d <subdir>
```

**Search for files by name:**
```bash
./.claude/scripts/list_artifacts.sh -f <pattern>
```

**Search for files containing text:**
```bash
./.claude/scripts/list_artifacts.sh -c <text>
```

**Show recent activity:**
```bash
./.claude/scripts/list_artifacts.sh -r [days]
```

**Show detailed statistics:**
```bash
./.claude/scripts/list_artifacts.sh --stats
```

**Examples:**
- List all subdirectories: `./.claude/scripts/list_artifacts.sh`
- List notes directory: `./.claude/scripts/list_artifacts.sh -d notes`
- Find files with "contract" in name: `./.claude/scripts/list_artifacts.sh -f contract`
- Find files containing "claude": `./.claude/scripts/list_artifacts.sh -c claude`
- Show files from last 3 days: `./.claude/scripts/list_artifacts.sh -r 3`
- Show statistics: `./.claude/scripts/list_artifacts.sh --stats`

**Features:**
- Directory listing with file counts and sizes
- File name pattern search
- Content search across all files
- Recent activity tracking (modified files)
- Statistics: file types, age distribution, largest files, total counts
