import os
import json
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Header, UploadFile, File, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional
from agent_manager import AgentManager, WORKSPACE_DIR
import httpx


# Config
logger = logging.getLogger(__name__)

API_KEY = os.environ.get("API_KEY")
ALLOW_BYPASS_PERMISSIONS = os.environ.get("ALLOW_BYPASS_PERMISSIONS", "0") == "1"

if not API_KEY:
    raise RuntimeError("API_KEY is required.")

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")

agent_manager: Optional[AgentManager] = None
_models_cache: dict = {"fetched_at": 0.0, "models": None, "error": None}


def _is_openrouter_enabled() -> bool:
    base_url = (os.environ.get("ANTHROPIC_BASE_URL") or "").lower().strip()
    return "openrouter.ai" in base_url


def _get_model_defaults() -> tuple[dict, Optional[str], str]:
    defaults = {
        "opus": os.environ.get("ANTHROPIC_DEFAULT_OPUS_MODEL") or "",
        "sonnet": os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL") or "",
        "haiku": os.environ.get("ANTHROPIC_DEFAULT_HAIKU_MODEL") or "",
    }
    cleaned = {key: value for key, value in defaults.items() if value}
    default_alias = "sonnet"
    default_model_id = cleaned.get(default_alias)
    return cleaned, default_model_id, default_alias


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_manager
    agent_manager = AgentManager(redis_url=REDIS_URL)
    yield
    await agent_manager.close()


app = FastAPI(
    title="Cloude ☁️ Agent",
    description="Claude Agent SDK endpoint for invoking Claude in the cloud",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for browser clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    """Verify API key from header only."""
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


async def verify_api_key_webhook(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    api_key: Optional[str] = None,  # Query parameter fallback for webhooks
):
    """Verify API key from header (preferred) or query parameter (fallback for webhooks)."""
    key = x_api_key or api_key
    if not key or key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


class ChatContext(BaseModel):
    source: str = "api"
    user_name: Optional[str] = None
    permission_mode: str = Field(
        default="acceptEdits",
        description="Permission mode: 'default', 'acceptEdits' (auto-approve file edits), or 'bypassPermissions' (approve all tools)"
    )
    metadata: dict = Field(default_factory=dict)


class ImageAttachment(BaseModel):
    data: str = Field(..., description="Base64-encoded image data")
    media_type: str = Field(default="image/jpeg", description="MIME type (image/jpeg, image/png, image/gif, image/webp)")


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique conversation identifier")
    message: str = Field(..., description="User message to the agent")
    command: Optional[str] = Field(default=None, description="Slash command to invoke (e.g., 'voice-transcript'). The message becomes the command argument.")
    images: Optional[list[ImageAttachment]] = Field(default=None, description="List of base64-encoded images")
    context: Optional[ChatContext] = None
    model: Optional[str] = Field(default=None, description="Model to use")


class ChatResponse(BaseModel):
    session_id: str
    response: str
    tools_used: list[str]
    usage: dict


class SkillCreate(BaseModel):
    id: str = Field(..., description="Unique skill identifier (alphanumeric, dashes, underscores)")
    content: str = Field(..., description="SKILL.md content with YAML frontmatter")


class CommandCreate(BaseModel):
    id: str = Field(..., description="Command identifier (alphanumeric, dashes, underscores)")
    template: str = Field(
        ...,
        description="Full markdown content for the command file (supports $ARGUMENTS and positional args like $1, $2).",
    )


class WorkspaceFileUpdate(BaseModel):
    content: str = Field(..., description="Full text content to write to the file")


class WorkspaceMoveRequest(BaseModel):
    src: str = Field(..., description="Workspace-relative source path")
    dst: str = Field(..., description="Workspace-relative destination path")
    overwrite: bool = Field(default=False, description="Whether to overwrite destination if it exists")


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
async def chat(req: ChatRequest):
    """
    Send a message to the Claude agent.

    Sessions persist across requests - use the same session_id to continue a conversation.
    Supports image attachments via base64-encoded data.

    If `command` is specified, the message is passed through the command template before sending.
    """
    try:
        if (req.context and req.context.permission_mode == "bypassPermissions") and not ALLOW_BYPASS_PERMISSIONS:
            raise HTTPException(status_code=403, detail="permission_mode=bypassPermissions is disabled on this server")

        # Process command if specified - send as slash command to get !` bash execution
        message = req.message
        if req.command:
            try:
                command_template = agent_manager.get_command(req.command)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            if not command_template:
                raise HTTPException(status_code=404, detail=f"Command '{req.command}' not found")
            # Format as slash command: /{command} {message}
            message = f"/{req.command} {req.message}"

        # Convert images to list of dicts if provided
        images = None
        if req.images:
            images = [{"data": img.data, "media_type": img.media_type} for img in req.images]

        result = await agent_manager.chat(
            user_session_id=req.session_id,
            message=message,
            images=images,
            context=req.context.model_dump() if req.context else None,
            model=req.model
        )
        return ChatResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled /chat error")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/webhook", dependencies=[Depends(verify_api_key_webhook)])
async def webhook(
    request: Request,
    command: Optional[str] = None,
    session_id: Optional[str] = None,  # Maps to field in body, e.g., session_id=id
    message: Optional[str] = None,     # Maps to field in body, e.g., message=transcript
    raw_response: bool = False,        # Return Claude's response directly without wrapper
):
    """
    Generic webhook endpoint with field mapping via query params.

    Use query params to map incoming payload fields to expected fields:
    - `session_id=<field>`: Map body field to session_id (default: "id" or "session_id")
    - `message=<field>`: Map body field to message (default: "message" or "transcript")
    - `command=<cmd>`: Slash command to invoke
    - `raw_response=true`: Return Claude's response directly (no ChatResponse wrapper)

    Example:
        POST /webhook?api_key=xxx&command=voice-transcript&session_id=id&message=transcript&raw_response=true

    With body:
        {"id": "abc123", "transcript": "Hello world", "title": "My Note"}

    Maps to internal:
        session_id = "abc123", message = "Hello world", command = "voice-transcript"
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Map session_id from body using query param as field name
    session_id_field = session_id or "session_id"
    actual_session_id = body.get(session_id_field) or body.get("id") or body.get("session_id")
    if not actual_session_id:
        actual_session_id = f"webhook-{int(time.time() * 1000)}"

    # Map message from body using query param as field name
    message_field = message or "message"
    actual_message = body.get(message_field) or body.get("transcript") or body.get("message") or body.get("text")

    if not actual_message:
        raise HTTPException(status_code=400, detail="No message content found in body")

    # Process command if specified - send as slash command to get !` bash execution
    if command:
        # Verify command exists
        try:
            command_template = agent_manager.get_command(command)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not command_template:
            raise HTTPException(status_code=404, detail=f"Command '{command}' not found")

        # Format as slash command: /{command} {argument}
        actual_message = f"/{command} {actual_message}"

    result = await agent_manager.chat(
        user_session_id=str(actual_session_id),
        message=actual_message,
        images=None,
        context={"source": "webhook", "permission_mode": "acceptEdits"},
        model=None
    )

    # Return raw response if requested (for clients expecting specific JSON format)
    if raw_response:
        return Response(
            content=result["response"],
            media_type="application/json"
        )

    return ChatResponse(**result)


@app.post("/chat/stream", dependencies=[Depends(verify_api_key)])
async def chat_stream(req: ChatRequest):
    """
    Stream a response from the Claude agent using Server-Sent Events.

    Returns a stream of SSE events with the following types:
    - text: A chunk of response text
    - tool: A tool that was used
    - done: Final message with session info
    - error: An error occurred

    If `command` is specified, the message is passed through the command template before sending.
    """
    if (req.context and req.context.permission_mode == "bypassPermissions") and not ALLOW_BYPASS_PERMISSIONS:
        return StreamingResponse(
            iter([f"data: {json.dumps({'type': 'error', 'error': 'permission_mode=bypassPermissions is disabled on this server'})}\n\n"]),
            media_type="text/event-stream"
        )

    # Process command if specified - send as slash command to get !` bash execution
    message = req.message
    if req.command:
        try:
            command_template = agent_manager.get_command(req.command)
        except ValueError as e:
            return StreamingResponse(
                iter([f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"]),
                media_type="text/event-stream"
            )
        if not command_template:
            return StreamingResponse(
                iter([f"data: {json.dumps({'type': 'error', 'error': f'Command {req.command} not found'})}\n\n"]),
                media_type="text/event-stream"
            )
        # Format as slash command: /{command} {message}
        message = f"/{req.command} {req.message}"

    # Convert images to list of dicts if provided
    images = None
    if req.images:
        images = [{"data": img.data, "media_type": img.media_type} for img in req.images]

    async def event_generator():
        try:
            async for event in agent_manager.chat_stream(
                user_session_id=req.session_id,
                message=message,
                images=images,
                context=req.context.model_dump() if req.context else None,
                model=req.model
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.exception("Unhandled /chat/stream error")
            yield f"data: {json.dumps({'type': 'error', 'error': 'Internal server error'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/models", dependencies=[Depends(verify_api_key)])
async def list_models(refresh: bool = False):
    """
    List available models for the current deployment.

    If OpenRouter is configured, this queries `GET https://openrouter.ai/api/v1/models`.
    If `ANTHROPIC_API_KEY` is available, this queries `GET https://api.anthropic.com/v1/models`.
    Otherwise returns a small fallback list based on defaults or known good IDs.
    """
    cache_ttl_s = 60 * 60
    now = time.time()
    defaults, default_model_id, default_alias = _get_model_defaults()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    openrouter_enabled = _is_openrouter_enabled()
    cache_source = _models_cache.get("source")
    desired_source = "openrouter" if openrouter_enabled else ("anthropic" if api_key else "fallback")

    if (
        not refresh
        and _models_cache.get("models")
        and now - float(_models_cache.get("fetched_at", 0)) < cache_ttl_s
        and cache_source == desired_source
    ):
        return {
            "models": _models_cache["models"],
            "source": "cache",
            "defaults": defaults,
            "default_model_id": default_model_id,
            "default_model_alias": default_alias,
        }

    fallback: list[dict] = []
    for key in ("sonnet", "opus", "haiku"):
        model_id = defaults.get(key)
        if model_id and model_id not in {m["id"] for m in fallback}:
            fallback.append({"id": model_id, "display_name": model_id})
    if not fallback:
        fallback = [
            {"id": "claude-sonnet-4-5-20250929", "display_name": "Claude Sonnet 4.5"},
            {"id": "claude-3-5-haiku-20241022", "display_name": "Claude 3.5 Haiku"},
        ]

    if openrouter_enabled:
        auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
        headers = {}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "https://openrouter.ai/api/v1/models",
                    headers=headers or None,
                )
            resp.raise_for_status()
            data = resp.json()
            models = [
                {"id": m.get("id"), "display_name": m.get("name") or m.get("display_name") or m.get("id")}
                for m in (data.get("data") or [])
                if m.get("id")
            ]
            _models_cache.update(
                {"fetched_at": now, "models": models, "error": None, "source": "openrouter"}
            )
            return {
                "models": models,
                "source": "openrouter",
                "defaults": defaults,
                "default_model_id": default_model_id,
                "default_model_alias": default_alias,
            }
        except Exception as e:
            logger.exception("Failed to fetch models from OpenRouter")
            _models_cache.update(
                {"fetched_at": now, "models": fallback, "error": str(e), "source": "fallback"}
            )
            return {
                "models": fallback,
                "source": "fallback",
                "warning": "Failed to fetch models from OpenRouter; returning fallback list",
                "defaults": defaults,
                "default_model_id": default_model_id,
                "default_model_alias": default_alias,
            }

    if not api_key:
        _models_cache.update(
            {"fetched_at": now, "models": fallback, "error": "ANTHROPIC_API_KEY not set", "source": "fallback"}
        )
        return {
            "models": fallback,
            "source": "fallback",
            "warning": "ANTHROPIC_API_KEY not set; returning fallback list",
            "defaults": defaults,
            "default_model_id": default_model_id,
            "default_model_alias": default_alias,
        }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "X-Api-Key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                params={"limit": 1000},
            )
        resp.raise_for_status()
        data = resp.json()
        models = [
            {"id": m.get("id"), "display_name": m.get("display_name")}
            for m in (data.get("data") or [])
            if m.get("id")
        ]
        _models_cache.update(
            {"fetched_at": now, "models": models, "error": None, "source": "anthropic"}
        )
        return {
            "models": models,
            "source": "anthropic",
            "defaults": defaults,
            "default_model_id": default_model_id,
            "default_model_alias": default_alias,
        }
    except Exception as e:
        logger.exception("Failed to fetch models from Anthropic")
        _models_cache.update(
            {"fetched_at": now, "models": fallback, "error": str(e), "source": "fallback"}
        )
        return {
            "models": fallback,
            "source": "fallback",
            "warning": "Failed to fetch models from Anthropic; returning fallback list",
            "defaults": defaults,
            "default_model_id": default_model_id,
            "default_model_alias": default_alias,
        }


@app.get("/health")
async def health():
    """Health check endpoint for Railway."""
    return {"status": "ok"}


# Public artifacts endpoint (no auth required for file access, but no directory listing)
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", str(WORKSPACE_DIR / "artifacts"))


@app.get("/artifacts/{file_path:path}")
async def get_artifact(file_path: str):
    """
    Serve a file from the public artifacts directory (no authentication required).

    Files in /artifacts/ are publicly accessible. Directory listing is not allowed.
    Claude can save files here when it needs to share them publicly.

    URL format: /artifacts/{session_id}/{filename}
    Example: /artifacts/abc123/report.html
    """
    import mimetypes
    from pathlib import Path

    artifacts_path = Path(ARTIFACTS_DIR)
    artifacts_root = artifacts_path.resolve()
    full_path = (artifacts_root / file_path).resolve()

    # Security: ensure path is within artifacts directory
    try:
        full_path.relative_to(artifacts_root)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    # Don't allow directory listing
    if full_path.is_dir():
        raise HTTPException(status_code=403, detail="Directory listing not allowed")

    if not full_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Read and serve the file
    content = full_path.read_bytes()
    content_type, _ = mimetypes.guess_type(str(full_path))
    if not content_type:
        content_type = "application/octet-stream"

    return Response(
        content=content,
        media_type=content_type
    )


@app.post("/artifacts/upload", dependencies=[Depends(verify_api_key)])
async def upload_artifact_files(
    target_dir: str = Form(""),
    files: list[UploadFile] = File(...),
):
    """Upload one or more files into a subdirectory under /artifacts/."""
    if agent_manager is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    artifacts_root = Path(ARTIFACTS_DIR).resolve()
    if not artifacts_root.exists():
        artifacts_root.mkdir(parents=True, exist_ok=True)

    normalized = (target_dir or "").strip().lstrip("./")
    if normalized in ("", "artifacts"):
        rel_target = ""
    elif normalized.startswith("artifacts/"):
        rel_target = normalized.removeprefix("artifacts/").lstrip("/")
    else:
        rel_target = normalized

    dest_dir = (artifacts_root / rel_target).resolve()
    try:
        dest_dir.relative_to(artifacts_root)
    except ValueError:
        raise HTTPException(status_code=400, detail="target_dir must stay within artifacts")

    dest_dir.mkdir(parents=True, exist_ok=True)

    uploaded: list[dict] = []
    for f in files:
        if not f.filename:
            continue
        safe_name = Path(f.filename).name
        if safe_name in ("", ".", ".."):
            continue
        data = await f.read()
        dest_path = (dest_dir / safe_name).resolve()
        try:
            dest_path.relative_to(artifacts_root)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid filename")
        dest_path.write_bytes(data)
        uploaded.append(
            {
                "name": safe_name,
                "path": str(dest_path.relative_to(artifacts_root)),
                "size": len(data),
            }
        )

    return {
        "status": "uploaded",
        "target_dir": f"artifacts/{rel_target}".rstrip("/"),
        "files": uploaded,
    }


# Skill management endpoints
@app.get("/skills", dependencies=[Depends(verify_api_key)])
async def list_skills():
    """List all installed skills."""
    return {"skills": agent_manager.list_skills()}


@app.get("/skills/{skill_id}", dependencies=[Depends(verify_api_key)])
async def get_skill(skill_id: str):
    """Get a specific skill's content."""
    try:
        skill = agent_manager.get_skill(skill_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    return skill


@app.post("/skills", dependencies=[Depends(verify_api_key)])
async def create_skill(skill: SkillCreate):
    """
    Create or update a skill.
    
    The skill will be immediately available to the agent without redeployment.
    
    Example SKILL.md content:
    ```
    ---
    name: my-skill
    description: Does something useful when asked about X
    ---
    
    # My Skill
    
    Instructions for Claude on how to use this skill...
    ```
    """
    try:
        result = agent_manager.add_skill(skill.id, skill.content)
        return {"status": "created", **result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/skills/{skill_id}", dependencies=[Depends(verify_api_key)])
async def delete_skill(skill_id: str):
    """Delete a skill."""
    try:
        if agent_manager.delete_skill(skill_id):
            return {"status": "deleted", "id": skill_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    raise HTTPException(status_code=404, detail="Skill not found")


@app.post("/skills/upload", dependencies=[Depends(verify_api_key)])
async def upload_skill(file: UploadFile = File(...)):
    """
    Upload a skill as a zip file.
    
    The zip should contain:
    - A directory with SKILL.md at its root, OR
    - SKILL.md directly at the zip root
    
    Supporting files (scripts, templates, data) will be preserved.
    The skill ID is derived from the directory name or the 'name' field in SKILL.md frontmatter.
    """
    if not file.filename or not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="File must be a .zip file")
    
    try:
        zip_data = await file.read()
        result = agent_manager.add_skill_from_zip(zip_data)
        return {"status": "uploaded", **result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Skill zip upload failed")
        raise HTTPException(status_code=500, detail="Failed to process zip")


@app.get("/skills/{skill_id}/download", dependencies=[Depends(verify_api_key)])
async def download_skill(skill_id: str):
    """Download a skill as a zip file."""
    try:
        zip_data = agent_manager.export_skill_zip(skill_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not zip_data:
        raise HTTPException(status_code=404, detail="Skill not found")
    
    return Response(
        content=zip_data,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={skill_id}.zip"}
    )


# Command management endpoints
@app.get("/commands", dependencies=[Depends(verify_api_key)])
async def list_commands():
    """List all available commands."""
    return {"commands": agent_manager.list_commands()}


@app.get("/commands/{command_id}", dependencies=[Depends(verify_api_key)])
async def get_command(command_id: str):
    """Get a specific command's template."""
    try:
        template = agent_manager.get_command(command_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not template:
        raise HTTPException(status_code=404, detail="Command not found")
    return {"id": command_id, "template": template}


@app.post("/commands", dependencies=[Depends(verify_api_key)])
async def create_command(cmd: CommandCreate):
    """
    Create or update a command.

    Commands are prompt templates that can be invoked via the `command` parameter in /chat.
    Pass arguments by invoking the slash command (e.g. `/{id} ...args`), then use `$ARGUMENTS` and/or `$1`, `$2` inside the command markdown.

    Example:
    ```
    ---
    argument-hint: [optional args]
    ---

    Analyze and summarize:
    $ARGUMENTS
    ```
    """
    try:
        result = agent_manager.add_command(cmd.id, cmd.template)
        return {"status": "created", **result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/commands/{command_id}", dependencies=[Depends(verify_api_key)])
async def delete_command(command_id: str):
    """Delete a command."""
    try:
        if agent_manager.delete_command(command_id):
            return {"status": "deleted", "id": command_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    raise HTTPException(status_code=404, detail="Command not found")


# Workspace file management endpoints
@app.get("/workspace", dependencies=[Depends(verify_api_key)])
async def list_workspace_files(path: str = ""):
    """List files in the agent's workspace directory."""
    files = agent_manager.list_workspace_files(path)
    return {
        "path": path or "/",
        "files": files
    }


@app.get("/workspace/{file_path:path}", dependencies=[Depends(verify_api_key)])
async def get_workspace_file(file_path: str):
    """Download a file from the workspace."""
    result = agent_manager.get_workspace_file(file_path)
    if not result:
        raise HTTPException(status_code=404, detail="File not found")
    
    content, filename = result
    
    # Determine content type
    import mimetypes
    content_type, _ = mimetypes.guess_type(filename)
    if not content_type:
        content_type = "application/octet-stream"
    
    return Response(
        content=content,
        media_type=content_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.delete("/workspace/{file_path:path}", dependencies=[Depends(verify_api_key)])
async def delete_workspace_file(file_path: str):
    """Delete a file or directory from the workspace."""
    if agent_manager.delete_workspace_file(file_path):
        return {"status": "deleted", "path": file_path}
    raise HTTPException(status_code=404, detail="File not found")


@app.put("/workspace/{file_path:path}", dependencies=[Depends(verify_api_key)])
async def put_workspace_file(file_path: str, payload: WorkspaceFileUpdate):
    """Create or update a text file in the workspace."""
    try:
        result = agent_manager.write_workspace_file(file_path, payload.content)
        return {"status": "saved", **result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/workspace/move", dependencies=[Depends(verify_api_key)])
async def move_workspace_item(payload: WorkspaceMoveRequest):
    """Move or rename a file/directory within the workspace."""
    if agent_manager is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    try:
        result = agent_manager.move_workspace_item(payload.src, payload.dst, overwrite=payload.overwrite)
        return {"status": "moved", **result}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Source not found")
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e) or "Destination already exists")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Workspace move failed")
        raise HTTPException(status_code=500, detail="Failed to move item")


# Session management endpoints
@app.get("/sessions", dependencies=[Depends(verify_api_key)])
async def list_sessions():
    """List all Claude sessions ordered by modified date (newest first)."""
    sessions = agent_manager.list_sessions()
    return {"sessions": sessions}


@app.get("/sessions/{session_id}", dependencies=[Depends(verify_api_key)])
async def get_session(session_id: str, raw: bool = False):
    """
    Get a session's content.

    - If raw=false (default): Returns parsed JSONL entries as structured data
    - If raw=true: Returns raw JSONL text content
    """
    if raw:
        content = agent_manager.get_session_raw(session_id)
        if content is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return Response(content=content, media_type="text/plain")

    session = agent_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.get("/chat.html")
async def serve_chat_ui():
    """Serve the chat UI."""
    chat_html_path = Path(__file__).parent / "chat.html"
    if not chat_html_path.exists():
        raise HTTPException(status_code=404, detail="chat.html not found")
    return FileResponse(chat_html_path, media_type="text/html")


@app.get("/")
async def root():
    """API information."""
    return {
        "name": "Cloude ☁️ Agent",
        "version": "1.0.0",
        "endpoints": {
            "POST /chat": "Send message to agent (supports `command` param for slash commands)",
            "POST /chat/stream": "Stream response from agent (SSE)",
            "GET /commands": "List available commands",
            "GET /commands/{id}": "Get command template",
            "POST /commands": "Create/update a command",
            "DELETE /commands/{id}": "Delete a command",
            "GET /workspace": "List files in workspace",
            "GET /workspace/{path}": "Download file from workspace",
            "DELETE /workspace/{path}": "Delete file from workspace",
            "POST /workspace/move": "Move/rename a file or directory in workspace",
            "GET /sessions": "List Claude sessions (newest first)",
            "GET /sessions/{id}": "Get session content (add ?raw=true for raw JSONL)",
            "GET /artifacts/{path}": "Public file access (no auth, no directory listing)",
            "GET /skills": "List installed skills",
            "POST /skills": "Create/update a simple skill (SKILL.md only)",
            "POST /skills/upload": "Upload a skill zip file (with supporting files)",
            "GET /skills/{id}": "Get skill content and file listing",
            "GET /skills/{id}/download": "Download skill as zip",
            "DELETE /skills/{id}": "Delete a skill",
            "GET /health": "Health check",
            "GET /chat.html": "Chat UI"
        }
    }
