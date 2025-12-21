---
allowed-tools: Bash(mkdir:*), Bash(date:*), Write(./artifacts/**), Edit(./artifacts/**)
description: Clean up and save a personal note or brain dump
argument-hint: [raw transcript]
---

# Process Personal Note

You are processing a voice note for the user. Clean it up and save it as a well-structured markdown file.

## User Context

<!-- PLACEHOLDER: Add personal context here -->
<!-- Example:
- Name: Chris Boden
- Role: Director at Peregian Digital Hub
- Common topics: tech ecosystem development, AI projects, startup mentoring
- Preferred style: concise, action-oriented
-->

## Raw Transcript

$ARGUMENTS

## Your Task

1. **Clean up** the transcript:
   - Fix speech-to-text errors and filler words
   - Add proper punctuation and paragraphs
   - Preserve the user's voice and intent

2. **Structure** the content:
   - Add a descriptive title (H1)
   - Break into logical sections if appropriate
   - Extract any action items or TODOs into a checklist
   - Add relevant tags at the bottom

3. **Save** to `./artifacts/notes/` with a descriptive filename:
   - Format: `YYYY-MM-DD-descriptive-name.md`
   - Use kebab-case, lowercase

## Output Format

After saving, output ONLY this JSON:

{"updateTitle":"Note: [brief topic]","content":"Saved to [BASE_URL]/artifacts/notes/[filename]"}

Where `[BASE_URL]` should be `$PUBLIC_BASE_URL` if set; otherwise use `https://clawed-api-production.up.railway.app`.

DO NOT include any text outside the JSON.
