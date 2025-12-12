import os
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Header, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
from agent_manager import AgentManager


# Config
API_KEY = os.environ.get("API_KEY", "dev-key")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")

agent_manager: Optional[AgentManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_manager
    agent_manager = AgentManager(redis_url=REDIS_URL)
    yield
    await agent_manager.close()


app = FastAPI(
    title="Clawed",
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
    api_key: Optional[str] = None  # Query parameter fallback
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
    template: str = Field(..., description="Prompt template with {{argument}} placeholder for the message")


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
async def chat(req: ChatRequest):
    """
    Send a message to the Claude agent.

    Sessions persist across requests - use the same session_id to continue a conversation.
    Supports image attachments via base64-encoded data.

    If `command` is specified, the message is passed through the command template before sending.
    """
    try:
        # Process command if specified
        message = req.message
        if req.command:
            command_template = agent_manager.get_command(req.command)
            if not command_template:
                raise HTTPException(status_code=404, detail=f"Command '{req.command}' not found")
            # Replace {{argument}} placeholder with the message
            message = command_template.replace("{{argument}}", req.message)

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
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/webhook", dependencies=[Depends(verify_api_key)])
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
    actual_session_id = body.get(session_id_field) or body.get("id") or body.get("session_id") or "webhook-session"

    # Map message from body using query param as field name
    message_field = message or "message"
    actual_message = body.get(message_field) or body.get("transcript") or body.get("message") or body.get("text")

    if not actual_message:
        raise HTTPException(status_code=400, detail="No message content found in body")

    # Process command if specified
    if command:
        command_template = agent_manager.get_command(command)
        if not command_template:
            raise HTTPException(status_code=404, detail=f"Command '{command}' not found")
        actual_message = command_template.replace("{{argument}}", actual_message)

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
    # Process command if specified
    message = req.message
    if req.command:
        command_template = agent_manager.get_command(req.command)
        if not command_template:
            return StreamingResponse(
                iter([f"data: {json.dumps({'type': 'error', 'error': f'Command {req.command} not found'})}\n\n"]),
                media_type="text/event-stream"
            )
        message = command_template.replace("{{argument}}", req.message)

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
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/health")
async def health():
    """Health check endpoint for Railway."""
    return {"status": "ok"}


# Public artifacts endpoint (no auth required for file access, but no directory listing)
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "/app/workspace/artifacts")


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
    full_path = (artifacts_path / file_path).resolve()

    # Security: ensure path is within artifacts directory
    if not str(full_path).startswith(str(artifacts_path.resolve())):
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


# Skill management endpoints
@app.get("/skills", dependencies=[Depends(verify_api_key)])
async def list_skills():
    """List all installed skills."""
    return {"skills": agent_manager.list_skills()}


@app.get("/skills/{skill_id}", dependencies=[Depends(verify_api_key)])
async def get_skill(skill_id: str):
    """Get a specific skill's content."""
    skill = agent_manager.get_skill(skill_id)
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
    if agent_manager.delete_skill(skill_id):
        return {"status": "deleted", "id": skill_id}
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
        raise HTTPException(status_code=500, detail=f"Failed to process zip: {str(e)}")


@app.get("/skills/{skill_id}/download", dependencies=[Depends(verify_api_key)])
async def download_skill(skill_id: str):
    """Download a skill as a zip file."""
    zip_data = agent_manager.export_skill_zip(skill_id)
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
    template = agent_manager.get_command(command_id)
    if not template:
        raise HTTPException(status_code=404, detail="Command not found")
    return {"id": command_id, "template": template}


@app.post("/commands", dependencies=[Depends(verify_api_key)])
async def create_command(cmd: CommandCreate):
    """
    Create or update a command.

    Commands are prompt templates that can be invoked via the `command` parameter in /chat.
    Use {{argument}} as a placeholder for the message content.

    Example:
    ```
    Analyze this transcript and summarize:

    {{argument}}

    Respond in JSON format.
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
    if agent_manager.delete_command(command_id):
        return {"status": "deleted", "id": command_id}
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


@app.get("/")
async def root():
    """API information."""
    return {
        "name": "Clawed - Claude Agent API",
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
            "GET /sessions": "List Claude sessions (newest first)",
            "GET /sessions/{id}": "Get session content (add ?raw=true for raw JSONL)",
            "GET /artifacts/{path}": "Public file access (no auth, no directory listing)",
            "GET /skills": "List installed skills",
            "POST /skills": "Create/update a simple skill (SKILL.md only)",
            "POST /skills/upload": "Upload a skill zip file (with supporting files)",
            "GET /skills/{id}": "Get skill content and file listing",
            "GET /skills/{id}/download": "Download skill as zip",
            "DELETE /skills/{id}": "Delete a skill",
            "GET /health": "Health check"
        }
    }

