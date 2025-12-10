# Clawed (Claude Agent SDK API + Web Chat)

A FastAPI-based API plus single-page chat UI for a Claude Code agent that can chat, use tools, manage skills, and read/write files on a Railway-mounted workspace volume with Redis volume for chat persistence.

## Features
- Chat API with streaming (`/chat/stream`) and model selection.
- File-aware agent: reads/writes in `/app/workspace` (Railway volume-backed).
- Skill management: list/get/add/delete/upload/download skills.
- Web chat UI (`chat.html`): sidebar for model choice, skills, workspace files; drag/drop skill zips; streaming responses; one-click open/download/curl for files.
- Permissions modes: default is `acceptEdits`; toggle `bypassPermissions` in UI.

## API
- `POST /chat` — non-streaming chat
- `POST /chat/stream` — SSE streaming chat
- `GET /workspace` — list workspace files
- `GET /workspace/{path}` — download workspace file (requires `X-API-Key`)
- `DELETE /workspace/{path}` — delete workspace file
- `GET /skills` — list skills
- `GET /skills/{id}` — get SKILL.md + files
- `POST /skills` — add/update a SKILL.md
- `POST /skills/upload` — upload skill zip
- `GET /skills/{id}/download` — download skill zip
- `DELETE /skills/{id}` — delete skill
- `GET /health` — health check

## Auth
- All endpoints except `/health` require `X-API-Key` (set via `API_KEY` env).

## Permissions Modes
- `default` — standard checks
- `acceptEdits` (default) — auto-approve file edits/fs ops
- `bypassPermissions` — auto-approve all tools (requires non-root; we run as `appuser`)
Pass via `context.permission_mode` in `/chat` or toggle in the UI.

## Workspace Files (volume)
- Mounted at `/app/workspace` (Railway volume).
- Agent works in this directory (SDK `cwd` set).
- UI “Workspace Files” shows new files after each run with buttons: Open, Download, Copy link, Copy curl.

## Skills
- Skills live under `.claude/skills/{skill_id}/SKILL.md` (+ supporting files).
- Manage via API or UI (drag/drop zip).

## Running Locally
1) `python -m venv .venv && source .venv/bin/activate`
2) `pip install -r requirements.txt`
3) `export API_KEY=dev-key` (or your key)
4) `uvicorn main:app --reload`
5) Open `chat.html` in a browser (set `API_URL` at top if needed).

## Env Vars
- `API_KEY` (required)
- `REDIS_URL` (default `redis://localhost:6379`)
- `PORT` (default `8080`)
- `WORKSPACE_DIR` (default `/app/workspace`)

## Deployment (Railway)
- Dockerfile installs node + Claude CLI, creates `appuser`, uses entrypoint to `chown` `/app/workspace` before dropping privileges.
- Attach a volume to `/app/workspace` for persistence.
- Run `railway up --detach` to deploy.

## Notable Files
- `main.py` — FastAPI app and endpoints
- `agent_manager.py` — chat logic, streaming, skills, workspace helpers
- `chat.html` — single-page UI
- `Dockerfile`, `entrypoint.sh` — image and permissions fix


## Current Limitations
- Workspace download URLs require API key (by design). UI provides one-click open/download/curl to ease this.
- SDK streams messages (not tokens); long tool runs may show status but not partial text.