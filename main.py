import os
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Header, UploadFile, File
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


async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != API_KEY:
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


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
async def chat(req: ChatRequest):
    """
    Send a message to the Claude agent.
    
    Sessions persist across requests - use the same session_id to continue a conversation.
    Supports image attachments via base64-encoded data.
    """
    try:
        # Convert images to list of dicts if provided
        images = None
        if req.images:
            images = [{"data": img.data, "media_type": img.media_type} for img in req.images]
        
        result = await agent_manager.chat(
            user_session_id=req.session_id,
            message=req.message,
            images=images,
            context=req.context.model_dump() if req.context else None,
            model=req.model
        )
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream", dependencies=[Depends(verify_api_key)])
async def chat_stream(req: ChatRequest):
    """
    Stream a response from the Claude agent using Server-Sent Events.
    
    Returns a stream of SSE events with the following types:
    - text: A chunk of response text
    - tool: A tool that was used
    - done: Final message with session info
    - error: An error occurred
    """
    # Convert images to list of dicts if provided
    images = None
    if req.images:
        images = [{"data": img.data, "media_type": img.media_type} for img in req.images]
    
    async def event_generator():
        try:
            async for event in agent_manager.chat_stream(
                user_session_id=req.session_id,
                message=req.message,
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


@app.get("/")
async def root():
    """API information."""
    return {
        "name": "Clawed - Claude Agent API",
        "version": "1.0.0",
        "endpoints": {
            "POST /chat": "Send message to agent",
            "POST /chat/stream": "Stream response from agent (SSE)",
            "GET /workspace": "List files in workspace",
            "GET /workspace/{path}": "Download file from workspace",
            "DELETE /workspace/{path}": "Delete file from workspace",
            "GET /skills": "List installed skills",
            "POST /skills": "Create/update a simple skill (SKILL.md only)",
            "POST /skills/upload": "Upload a skill zip file (with supporting files)",
            "GET /skills/{id}": "Get skill content and file listing",
            "GET /skills/{id}/download": "Download skill as zip",
            "DELETE /skills/{id}": "Delete a skill",
            "GET /health": "Health check"
        }
    }

