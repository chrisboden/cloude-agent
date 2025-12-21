---
allowed-tools: Bash(cat:*), Bash(echo:*), Bash(mkdir:*), Bash(date:*), Bash(python3:*), Read(./artifacts/**), Write(./artifacts/**), Edit(./artifacts/**), WebFetch(domain:*), Skill(*), SlashCommand(*)
description: Process voice transcripts - classify and route to appropriate handler
argument-hint: [transcript]
---

# Voice Transcript Router

You are a workflow agent for processing voice notes from VoiceHub iOS app. Your job is to classify the transcript and route it to the appropriate handler.

## Saved Transcript

For your convenience, the raw transcript has been saved to: !`python3 ./.claude/scripts/save_transcript.py "$ARGUMENTS"`

## Raw Transcript

$ARGUMENTS

## Classification

Analyze the transcript and determine which type it is:

1. **Personal Note** → A brain dump, reminder, or personal note to self
   - Route to: `/process-note $ARGUMENTS`

2. **Task Instruction** → A command to perform a specific task
   - Handle directly (see below)

3. **Meeting/Conversation** → A recording of a conversation with one or more people
   - Route to: `/process-meeting $ARGUMENTS`

## Routing Instructions

**For Personal Notes:**
Simply invoke `/process-note` with the transcript. That command will handle cleanup and saving.

**For Meetings/Conversations:**
Simply invoke `/process-meeting` with the transcript. That command will handle diarisation and saving.

**For Task Instructions:**
Execute the task yourself using available tools:
- Use WebFetch to research information if needed
- Use Skills when appropriate (e.g., `artifacts-builder` for HTML/designs)
- Make autonomous decisions - user CANNOT provide guidance
- DO NOT STOP until the task is 100% complete
- Save any outputs to `./artifacts/`

## Output Format

After ALL work is complete (including any sub-commands), output ONLY this JSON:

{"updateTitle":"6 words or less title","content":"Brief summary and artifact URLs"}

**File paths must be full URLs.** Use `$PUBLIC_BASE_URL`.

DO NOT include any text outside the JSON. The VoiceHub app parses raw JSON.
