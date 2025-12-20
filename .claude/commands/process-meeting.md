---
allowed-tools: Bash(mkdir:*), Bash(date:*), Write(./artifacts/**), Edit(./artifacts/**)
description: Diarise and summarise a meeting or conversation transcript
argument-hint: [raw transcript]
---

# Process Meeting/Conversation

You are processing a voice recording of a meeting or conversation. Diarise, clean up, and create useful meeting notes.

## User Context

<!-- PLACEHOLDER: Add personal context here -->
<!-- Example:
- Name: Chris Boden
- Role: Director at Peregian Digital Hub
- Common meeting types: 1:1s with founders, board meetings, partner discussions
- Key relationships: Hub members, Noosa Council, investors
- Preferred format: Executive summary first, then details
-->

## Raw Transcript

$ARGUMENTS

## Your Task

1. **Diarise** the conversation:
   - Identify speakers (use "Speaker 1", "Speaker 2" if names unclear)
   - If you can infer names from context, use them
   - Attribute statements to the correct speaker

2. **Create meeting notes** with this structure:
   ```markdown
   # [Meeting Title] - [Date]

   **Participants:** [list]
   **Duration:** [if discernible]
   **Context:** [brief context about what this meeting was about]

   ## Executive Summary
   [3-5 bullet points of key outcomes/decisions]

   ## Discussion Notes
   [Diarised, cleaned-up conversation organized by topic]

   ## Action Items
   - [ ] [Action] - [Owner] - [Due if mentioned]

   ## Key Decisions
   - [Decision 1]
   - [Decision 2]

   ## Follow-up Required
   - [Items needing follow-up]
   ```

3. **Save** to `./artifacts/meeting-notes/` with filename:
   - Format: `YYYY-MM-DD-meeting-topic.md`
   - Use kebab-case, lowercase

## Output Format

After saving, output ONLY this JSON:

{"updateTitle":"Meeting: [brief topic]","content":"Saved to [BASE_URL]/artifacts/meeting-notes/[filename]"}

Where `[BASE_URL]` should be `$PUBLIC_BASE_URL` if set; otherwise use `https://clawed-api-production.up.railway.app`.

DO NOT include any text outside the JSON.
