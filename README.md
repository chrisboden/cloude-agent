# Cloude ☁️ Agent

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.com/deploy/cloude-agent?referralCode=P5pe6R&utm_medium=integration&utm_source=template&utm_campaign=generic)

Cloude is the Claude Agent SDK packaged as a one-click Railway deploy. You get a cloud-native agent with a powerful API, so you can build anything you would with Claude Code, but natively in the cloud.

Instead of working on files on your computer, the agent reads and writes files on its own hard drive in the cloud (the Railway volume). The included chat UI is one example app: enter your app API URL and key, then start working with the agent. You can build many other apps on top of the same API, and the agent can even build those apps for you and host them in its own workspace.

## Quick start (template deploy)

1) Click the deploy button above.
2) Set these environment variables:
   - `OPENROUTER_API_KEY` (recommended) — defaults the provider to OpenRouter
   - `API_KEY` — your app auth key (any strong random string)
   - `ANTHROPIC_API_KEY` (optional) — only if you want Anthropic directly
3) In Railway, generate a public domain for the service. See [this Loom](https://www.loom.com/share/50fa2f5aa41045d7b64f5d9a489f9687) for instructions.
4) Open the chat UI:
   - `https://your-app-name.up.railway.app/chat.html`
5) Click Settings (gear icon) and enter:
   - API URL: `https://your-app-name.up.railway.app`
   - API Key: the `API_KEY` you set above

You are ready to chat and use the agent.

## What you can build

- A custom chat UI or inbox that talks to the agent
- Webhook-driven automation (forms, transcripts, CRM, etc.)
- File-based workflows that read/write the agent's cloud workspace
- Agent-built apps hosted in its own workspace or published from `/artifacts`

## How the cloud workspace works

- The agent works inside `/app/workspace` (persistent Railway volume).
- Skills and commands live under `/app/workspace/.claude`.
- Files placed in `/app/workspace/artifacts` are publicly served at `/artifacts/{path}`.

## API overview

All API calls require `X-API-Key: <API_KEY>` unless noted. `/webhook` also accepts `?api_key=` for clients that can’t set headers.

Public endpoints (no API key):
- `GET /health` — health check
- `GET /chat.html` — chat UI
- `GET /` — API info
- `GET /artifacts/{path}` — serve public files
- `POST /realtime` — OpenAI realtime webhook (validated via `OPENAI_WEBHOOK_SECRET`)

Core agent endpoints:
- `POST /chat` — non-streaming chat (supports `command` for slash command wrapper)
- `POST /chat/stream` — streaming chat (SSE, supports `command`)
- `POST /chat/interrupt` — interrupt an active streaming session
- `GET /models` — list available models and server defaults
- `POST /webhook` — map external payloads to chat requests

Workspace endpoints:
- `GET /workspace` — list files in the workspace
- `GET /workspace/{path}` — download file from workspace
- `PUT /workspace/{path}` — create/update a text file
- `DELETE /workspace/{path}` — delete a file or directory
- `POST /workspace/move` — move/rename a file or directory

Artifacts:
- `POST /artifacts/upload` — upload public files
- `GET /artifacts/{path}` — serve public files (no auth)

Skills:
- `GET /skills` — list skills
- `GET /skills/{id}` — get skill content and file listing
- `POST /skills` — create/update a simple skill (SKILL.md only)
- `POST /skills/upload` — upload a full skill zip
- `GET /skills/{id}/download` — download a skill zip
- `DELETE /skills/{id}` — delete a skill

Commands and prompts:
- `GET /commands` — list commands
- `GET /commands/{id}` — get a command
- `POST /commands` — create/update a command
- `DELETE /commands/{id}` — delete a command
- `GET /prompts` — list prompt overrides

Sessions and debug:
- `GET /sessions` — list sessions (newest first)
- `GET /sessions/{id}` — get session content (`?raw=true` for JSONL)
- `POST /debug/permissions` — inspect effective settings + permission resolution
- `GET /debug/permission-log` — fetch recent permission decisions

## Model defaults and providers

- `OPENROUTER_API_KEY` makes OpenRouter the default provider.
- `ANTHROPIC_API_KEY` is optional for direct Anthropic use.
- Default model aliases can be set via env vars or in `.claude/settings.json` (no restart needed).

Example `.claude/settings.json` override:
```json
{
  "env": {
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "openai/gpt-5.1-codex",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "z-ai/glm-4.7",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "openai/gpt-5.1-codex-mini"
  }
}
```

## Local development (optional)

1) `python -m venv .venv && source .venv/bin/activate`
2) `pip install -r requirements.txt`
3) `export API_KEY="$(openssl rand -hex 32)"`
4) Start Redis (required for sessions):
   - Homebrew: `brew install redis && brew services start redis`
   - Docker: `docker run -d --name clawed-redis -p 6379:6379 redis:7`
5) `uvicorn main:app --reload`
6) Open `chat.html` and set:
   - API URL: `http://127.0.0.1:8000`
   - API Key: the key you exported in step 3

## Notable files

- `main.py` — FastAPI app and endpoints
- `agent_manager.py` — chat logic, streaming, skills, workspace helpers
- `chat.html` — single-page UI
- `entrypoint.sh` — volume seeding for `.claude` and workspace
- `.claude/settings.json` — permissions and shared config
