# Cloude ☁️ Agent

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.com/deploy/cloude-agent?referralCode=P5pe6R&utm_medium=integration&utm_source=template&utm_campaign=generic)

Deploy the Claude Code agent to the cloud. Give it a workspace to work with files. Load up skills and commands to extend its capabilities. Invoke via API or webhooks.

## Quick Start

### 1. Deploy on Railway

Click the deploy button above. You'll be prompted for two environment variables:

- **`ANTHROPIC_API_KEY`** — Your Anthropic API key (get one at [console.anthropic.com](https://console.anthropic.com))
- **`API_KEY`** — Create your own secret key for authenticating with your deployed app (e.g. a random string)

### 2. Enable Public URL

After deployment, generate a public URL for your service:

1. Open your service in the Railway dashboard
2. Click on Architecture tab in the top navbar
3. Double click on the cloude-agent service to open the config panel
4. Click on **Settings** and scroll down to **Networking**
5. Click **Generate Domain** under "Public Networking"

You'll get a URL like `https://your-app-name.up.railway.app`

### 3. Open the Chat UI

Open the chat interface in your browser:

```
https://your-app-name.up.railway.app/chat.html
```

Alternatively, you can open `chat.html` from this repo locally — it works as a standalone file too.

### 4. Configure Settings

Click the **Settings** button (gear icon) in the chat UI and enter:

- **API URL** — Your Railway app URL (e.g. `https://your-app-name.up.railway.app`)
- **API Key** — The `API_KEY` you set during deployment

The status indicator will show "Connected" when configured correctly.

> **Tip:** Settings are saved in your browser's localStorage, so you only need to configure once per browser.

### 5. Start Chatting

You're ready to go! The agent can read/write files in its workspace, run commands, and use any skills you've installed.

---

## Test with cURL

Verify your deployment with these commands (replace the URL and API key):

**Health check:**
```bash
curl https://your-app-name.up.railway.app/health
```

**Send a message:**
```bash
curl -X POST https://your-app-name.up.railway.app/chat \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test", "message": "Hello! What can you do?"}'
```

**Streaming response:**
```bash
curl -X POST https://your-app-name.up.railway.app/chat/stream \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test", "message": "Write a haiku about clouds"}'
```

---

## Background

Claude Code is an amazing coding agent and, iykyk, it's actually a fantastic general purpose agent harness. With the Claude Agent SDK, we now have programmatic access. This little project deploys the SDK to the cloud and makes it available via API and webhook.

Simply add your preferred skills and slash commands to the workspace and you have a custom agent available 24x7 in the cloud.

## Features

- **Chat API** with streaming (`/chat/stream`) via SSE and model selection
- **Persistent workspace** — agent reads/writes files in `/app/workspace` (Railway volume-backed)
- **Skill management** — list, add, delete, upload/download skills as zip files
- **Slash commands** — define reusable prompt templates for consistent workflows
- **Web chat UI** (`chat.html`) — Claude.com-style interface with file explorer and searchable model picker
- **Session persistence** — Redis-backed conversation history
- **Webhook support** — trigger agent runs from external services
- **OpenRouter support** - allows you to use any LLM model from OpenRouter, manage keys for others, track usage, etc.

## API

### Chat
- `POST /chat` — non-streaming chat (requires `X-API-Key` header)
- `POST /chat/stream` — SSE streaming chat (requires `X-API-Key` header)
- `POST /webhook` — webhook endpoint with flexible field mapping (supports `X-API-Key` header OR `api_key` query param)

### Models
- `GET /models` — list available models (OpenRouter or Anthropic, based on env) and return server defaults

### Workspace
- `GET /workspace` — list workspace files
- `GET /workspace/{path}` — download workspace file
- `PUT /workspace/{path}` — create/update a text file
- `DELETE /workspace/{path}` — delete workspace file
- `POST /workspace/move` — move/rename a file or directory

### Skills
- `GET /skills` — list skills
- `GET /skills/{id}` — get SKILL.md + files
- `POST /skills` — add/update a SKILL.md
- `POST /skills/upload` — upload skill zip
- `GET /skills/{id}/download` — download skill zip
- `DELETE /skills/{id}` — delete skill

### Commands
- `GET /commands` — list commands
- `GET /commands/{id}` — get command template
- `POST /commands` — add/update a command template
- `DELETE /commands/{id}` — delete a command

### Sessions
- `GET /sessions` — list all sessions (newest first)
- `GET /sessions/{id}` — get session content (add `?raw=true` for raw JSONL)

### Artifacts (Public)
- `GET /artifacts/{path}` — public file access (**no auth required**, no directory listing)
- `POST /artifacts/upload` — upload files to artifacts directory

### System
- `GET /health` — health check
- `GET /chat.html` — chat UI
- `GET /` — API info and endpoint list

## Auth
- Set your ANTHROPIC_API_KEY in the Railway environment variables - this enables the Claude Agent to use the Anthropic LLM models.
- Create your own API KEY for authentication - this is used to authenticate requests to the API.

## OpenRouter + Model Routing
You can route Claude Code through OpenRouter by setting these environment variables (Railway or local):

```bash
ANTHROPIC_BASE_URL=https://openrouter.ai/api
ANTHROPIC_AUTH_TOKEN=sk-or-...
ANTHROPIC_API_KEY=""   # important: explicitly blank
```

To override the default Claude Code aliases (Opus/Sonnet/Haiku) with OpenRouter model IDs:

```bash
ANTHROPIC_DEFAULT_OPUS_MODEL=openai/gpt-5.1-codex
ANTHROPIC_DEFAULT_SONNET_MODEL=z-ai/glm-4.6
ANTHROPIC_DEFAULT_HAIKU_MODEL=x-ai/grok-4.1-fast
```

The chat UI uses `GET /models` to populate the searchable model picker and displays the server default in Settings and the sidebar.

## Workspace Files (volume)
- Mounted at `/app/workspace` (Railway volume).
- Agent works in this directory (SDK `cwd` set).
- Chat UI (chat.html) has fully functional file explorer for browsing the workspace and editing files, etc (by making calls to the API).

## Skills
- Skills live on the volume at `$WORKSPACE_DIR/.claude/skills/{skill_id}/SKILL.md` (+ supporting files).
- Skills are **not** baked into the Docker build - they persist on the volume and can be added/modified without redeployment.
- Manage via API or UI (drag/drop zip).
- After uploading skills with scripts, they are automatically made executable on container start.

## Commands (Slash Commands)
- Commands are prompt templates stored at `$WORKSPACE_DIR/.claude/commands/{command_id}.md`
- Invoke via `command` parameter in `/chat`: `{"command": "voice-transcript", "message": "the transcript text..."}`
- Commands use Claude Code’s markdown format (YAML frontmatter + prompt body). Frontmatter `allowed-tools` controls what the command is allowed to run.
- Commands can optionally pin a model via frontmatter (use OpenRouter IDs if routing through OpenRouter):
  ```yaml
  ---
  model: z-ai/glm-4.6
  ---
  ```
- If `model` is omitted, the command uses the server default (or the UI-selected override for chat requests).
- Inside command markdown, use `$ARGUMENTS` (or positional `$1`, `$2`, etc.) to consume the arguments passed after `/{command}`.
- Manage via API: `GET/POST/DELETE /commands` (and/or edit the files on the volume).
- Useful for webhooks that need consistent prompt formatting and reliable routing.

## Webhooks

The `/webhook` endpoint accepts arbitrary JSON payloads and maps them to chat requests. Useful for integrating with external services (transcription services, form submissions, etc.).

**Query parameters:**
- `api_key` — API key (alternative to `X-API-Key` header)
- `command` — slash command to invoke (e.g., `voice-transcript`)
- `session_id` — field name in body to use as session ID (default: looks for `session_id`, `id`)
- `message` — field name in body to use as message (default: looks for `message`, `transcript`, `text`)
- `raw_response` — if `true`, returns Claude's response directly without wrapper

**Example:**
```bash
# Webhook with slash command
curl -X POST "https://your-app.up.railway.app/webhook?api_key=YOUR_KEY&command=voice-transcript" \
  -H "Content-Type: application/json" \
  -d '{"id": "session-123", "transcript": "Hello, this is a voice transcript..."}'
```

**Permissions:** Webhook-triggered runs can't click "approve", so **any tool that would normally prompt must be pre-approved** or you'll see errors like "This command requires approval".

- Recommended: keep permissions locked down and add explicit allow rules in `.claude/settings.json` (seeded into the volume by `entrypoint.sh`).
- Avoid relying on `bypassPermissions` for production webhooks unless you fully trust the deployment and isolation.

## Running Locally
1) `python -m venv .venv && source .venv/bin/activate`
2) `pip install -r requirements.txt`
3) `export API_KEY="$(openssl rand -hex 32)"` (or any strong random key)
4) Start Redis (required for `/chat` session/history storage):
   - Homebrew: `brew install redis && brew services start redis`
   - Docker: `docker run -d --name clawed-redis -p 6379:6379 redis:7`
5) `export REDIS_URL="redis://localhost:6379"` (default, but explicit is clearer)
6) `uvicorn main:app --reload`
7) Open `chat.html` in a browser, click Settings, and configure:
   - **API URL**: `http://127.0.0.1:8000`
   - **API Key**: The key you exported in step 3

### Local Smoke Tests

Health:
```bash
curl -fsS http://127.0.0.1:8080/health
```

Chat (non-streaming):
```bash
curl -sS -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"session_id":"smoke","message":"Say hello world and nothing else.","context":{"permission_mode":"acceptEdits"}}' \
  http://127.0.0.1:8080/chat | python3 -m json.tool
```

If `/chat` returns a 500, run uvicorn with debug logs and re-run the curl without `-f` to see the error:
```bash
uvicorn main:app --reload --log-level debug
curl -sS -D - -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"session_id":"smoke","message":"Hello","context":{"permission_mode":"acceptEdits"}}' \
  http://127.0.0.1:8080/chat
```

## Env Vars
- `API_KEY` (required)
- `REDIS_URL` (default `redis://localhost:6379`)
- `PORT` (default `8080`)
- `WORKSPACE_DIR` (default `/app/workspace`)
- `SKILLS_DIR` (default `$WORKSPACE_DIR/.claude/skills`)
- `COMMANDS_DIR` (default `$WORKSPACE_DIR/.claude/commands`)
- `PROJECT_CONTEXT_PATH` (default `$WORKSPACE_DIR/.claude/CLAUDE.md`)
- `MAX_PROJECT_CONTEXT_CHARS` (default `50000`)
- `ALLOW_BYPASS_PERMISSIONS` (default `0`) — set to `1` to allow `permission_mode=bypassPermissions`
- `PUBLIC_BASE_URL` (optional) — used by slash commands to generate fully-qualified artifact URLs (defaults to the production Railway URL in the included commands).

## Deployment (Railway)
- Dockerfile installs node + Claude CLI, creates `appuser`, uses entrypoint to `chown` `/app/workspace` before dropping privileges.
- Attach a volume to `/app/workspace` for persistence.
- Run `railway up --detach` to deploy.
- `entrypoint.sh` seeds default `.claude/commands/`, `.claude/scripts/`, `.claude/skills/`, `.claude/settings.json`, and `.claude/CLAUDE.md` into the volume **only if missing** (non-destructive). To pick up image updates, edit the volume files (preferred) or delete the specific file(s) from the volume so they can be re-seeded.

## Notable Files
- `main.py` — FastAPI app and endpoints
- `agent_manager.py` — chat logic, streaming, skills, workspace helpers
- `chat.html` — single-page UI
- `Dockerfile`, `entrypoint.sh` — image and permissions fix
- `.claude/settings.json` — project permission rules used for non-interactive runs (webhooks)
- `.claude/CLAUDE.md` — project context (volume-managed) appended to the system prompt - editable via the chat UI
- `.claude/scripts/` — helper scripts invoked by commands
