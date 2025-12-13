import json
import os
import shutil
import zipfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from claude_code_sdk import ClaudeSDKClient, ClaudeCodeOptions
from claude_code_sdk.types import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock, SystemMessage, UserMessage
import redis.asyncio as redis

# Workspace directory for agent file operations
# Can be overridden via WORKSPACE_DIR env var (for Railway volume mount)
WORKSPACE_DIR = Path(os.environ.get("WORKSPACE_DIR", "/app/workspace"))
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

# Skills directory - on the volume for runtime management
# Can be overridden via SKILLS_DIR env var
SKILLS_DIR = Path(os.environ.get("SKILLS_DIR", str(WORKSPACE_DIR / ".claude" / "skills")))

# Commands directory - prompt templates on the volume
# Can be overridden via COMMANDS_DIR env var
COMMANDS_DIR = Path(os.environ.get("COMMANDS_DIR", str(WORKSPACE_DIR / ".claude" / "commands")))


class AgentManager:
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
        self.conversation_histories: dict[str, list[dict]] = {}
        # Ensure skills and commands directories exist
        SKILLS_DIR.mkdir(parents=True, exist_ok=True)
        COMMANDS_DIR.mkdir(parents=True, exist_ok=True)
    
    async def _get_stored_session(self, user_session_id: str) -> Optional[dict]:
        data = await self.redis.get(f"session:{user_session_id}")
        if data:
            return json.loads(data)
        return None
    
    async def _store_session(self, user_session_id: str, conversation_summary: str = ""):
        await self.redis.set(
            f"session:{user_session_id}",
            json.dumps({
                "created": datetime.utcnow().isoformat(),
                "last_active": datetime.utcnow().isoformat(),
                "summary": conversation_summary
            }),
            ex=86400 * 7  # 7 day expiry
        )
    
    async def _update_session_activity(self, user_session_id: str):
        data = await self._get_stored_session(user_session_id)
        if data:
            data["last_active"] = datetime.utcnow().isoformat()
            await self.redis.set(
                f"session:{user_session_id}",
                json.dumps(data),
                ex=86400 * 7
            )
    
    async def _get_conversation_history(self, user_session_id: str) -> list[dict]:
        """Get conversation history from Redis."""
        data = await self.redis.get(f"history:{user_session_id}")
        if data:
            return json.loads(data)
        return []
    
    async def _store_conversation_history(self, user_session_id: str, history: list[dict]):
        """Store conversation history in Redis."""
        # Keep last 20 exchanges to avoid context limits
        trimmed = history[-40:] if len(history) > 40 else history
        await self.redis.set(
            f"history:{user_session_id}",
            json.dumps(trimmed),
            ex=86400 * 7
        )
    
    async def chat(
        self, 
        user_session_id: str, 
        message: str,
        images: Optional[list[dict]] = None,
        context: Optional[dict] = None,
        model: Optional[str] = None
    ) -> dict:
        # Check for existing session
        stored = await self._get_stored_session(user_session_id)
        history = await self._get_conversation_history(user_session_id)
        
        # Build the prompt with context
        text_content = message
        if context:
            source = context.get("source", "unknown")
            user_name = context.get("user_name", "User")
            text_content = f"[Context: {user_name} via {source}]\n\n{message}"
        
        # If we have history, include it as context
        if history:
            history_text = "\n".join([
                f"{'User' if h['role'] == 'user' else 'Assistant'}: {h['content']}" 
                for h in history[-10:]  # Last 10 messages for context
            ])
            text_content = f"Previous conversation:\n{history_text}\n\nNew message: {text_content}"
        
        # Build message content - either string or list with images
        if images:
            # Build content array with text and images
            content: Any = [{"type": "text", "text": text_content}]
            for img in images:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img.get("media_type", "image/jpeg"),
                        "data": img["data"]
                    }
                })
        else:
            content = text_content
        
        tools_used = []
        response_parts = []
        
        # Message generator for ClaudeSDKClient
        async def message_generator():
            yield {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": content
                }
            }
        
        # Use ClaudeSDKClient for proper message handling including images
        # Default to acceptEdits (safer), can override to bypassPermissions via API
        permission_mode = context.get("permission_mode", "acceptEdits") if context else "acceptEdits"
        
        # Set working directory to workspace for file operations
        options = ClaudeCodeOptions(
            permission_mode=permission_mode,
            cwd=str(WORKSPACE_DIR)
        )
        
        async with ClaudeSDKClient(options) as client:
            # Send the message via generator
            await client.query(message_generator())
            
            # Process responses
            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            response_parts.append(block.text)
                        elif isinstance(block, ToolUseBlock):
                            tools_used.append(block.name)
        
        response_text = "".join(response_parts)
        
        # Update conversation history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response_text})
        await self._store_conversation_history(user_session_id, history)
        
        # Update session
        await self._store_session(user_session_id)
        await self._update_session_activity(user_session_id)
        
        return {
            "session_id": user_session_id,
            "response": response_text,
            "tools_used": list(set(tools_used)),
            "usage": {
                "num_turns": len(history) // 2
            }
        }
    
    async def chat_stream(
        self, 
        user_session_id: str, 
        message: str,
        images: Optional[list[dict]] = None,
        context: Optional[dict] = None,
        model: Optional[str] = None
    ):
        """Stream chat responses as they're generated."""
        # Check for existing session
        stored = await self._get_stored_session(user_session_id)
        history = await self._get_conversation_history(user_session_id)
        
        # Build the prompt with context
        text_content = message
        if context:
            source = context.get("source", "unknown")
            user_name = context.get("user_name", "User")
            text_content = f"[Context: {user_name} via {source}]\n\n{message}"
        
        # If we have history, include it as context
        if history:
            history_text = "\n".join([
                f"{'User' if h['role'] == 'user' else 'Assistant'}: {h['content']}" 
                for h in history[-10:]
            ])
            text_content = f"Previous conversation:\n{history_text}\n\nNew message: {text_content}"
        
        # Build message content
        if images:
            content: Any = [{"type": "text", "text": text_content}]
            for img in images:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img.get("media_type", "image/jpeg"),
                        "data": img["data"]
                    }
                })
        else:
            content = text_content
        
        # Message generator
        async def message_generator():
            yield {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": content
                }
            }
        
        tools_used = []
        response_parts = []
        
        # Default to acceptEdits (safer), can override to bypassPermissions via API
        permission_mode = context.get("permission_mode", "acceptEdits") if context else "acceptEdits"
        
        # Set working directory to workspace for file operations
        options = ClaudeCodeOptions(
            permission_mode=permission_mode,
            cwd=str(WORKSPACE_DIR)
        )
        
        # Signal that we're starting
        yield {"type": "status", "status": "connecting"}
        
        async with ClaudeSDKClient(options) as client:
            yield {"type": "status", "status": "sending"}
            await client.query(message_generator())
            yield {"type": "status", "status": "processing"}
            
            async for msg in client.receive_response():
                if isinstance(msg, SystemMessage):
                    # System is ready
                    yield {"type": "status", "status": "ready"}
                elif isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            response_parts.append(block.text)
                            # Yield text chunk
                            yield {"type": "text", "text": block.text}
                        elif isinstance(block, ToolUseBlock):
                            tools_used.append(block.name)
                            # Yield tool use with more detail
                            yield {"type": "tool", "name": block.name, "status": "started"}
                elif isinstance(msg, UserMessage):
                    # Tool result came back
                    if tools_used:
                        yield {"type": "tool", "name": tools_used[-1], "status": "completed"}
        
        response_text = "".join(response_parts)
        
        # Update conversation history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response_text})
        await self._store_conversation_history(user_session_id, history)
        
        # Update session
        await self._store_session(user_session_id)
        await self._update_session_activity(user_session_id)
        
        # Yield final done event
        yield {
            "type": "done",
            "session_id": user_session_id,
            "tools_used": list(set(tools_used)),
            "usage": {"num_turns": len(history) // 2}
        }
    
    # Skill management methods
    def _count_files(self, directory: Path) -> int:
        """Count all files in a directory recursively."""
        count = 0
        for item in directory.rglob("*"):
            if item.is_file():
                count += 1
        return count

    def list_skills(self) -> list[dict]:
        """List all installed skills."""
        skills = []
        if SKILLS_DIR.exists():
            for skill_dir in SKILLS_DIR.iterdir():
                if skill_dir.is_dir():
                    skill_file = skill_dir / "SKILL.md"
                    if skill_file.exists():
                        content = skill_file.read_text()
                        # Parse frontmatter
                        name = skill_dir.name
                        description = ""
                        if content.startswith("---"):
                            parts = content.split("---", 2)
                            if len(parts) >= 3:
                                frontmatter = parts[1]
                                for line in frontmatter.strip().split("\n"):
                                    if line.startswith("name:"):
                                        name = line.split(":", 1)[1].strip()
                                    elif line.startswith("description:"):
                                        description = line.split(":", 1)[1].strip()
                        
                        file_count = self._count_files(skill_dir)
                        skills.append({
                            "id": skill_dir.name,
                            "name": name,
                            "description": description,
                            "path": str(skill_file),
                            "file_count": file_count
                        })
        return skills

    def get_skill(self, skill_id: str) -> Optional[dict]:
        """Get a specific skill's content and file listing."""
        skill_dir = SKILLS_DIR / skill_id
        skill_file = skill_dir / "SKILL.md"
        if skill_file.exists():
            # List all files in the skill directory
            files = []
            for item in skill_dir.rglob("*"):
                if item.is_file():
                    rel_path = item.relative_to(skill_dir)
                    files.append({
                        "path": str(rel_path),
                        "size": item.stat().st_size
                    })
            
            return {
                "id": skill_id,
                "content": skill_file.read_text(),
                "path": str(skill_file),
                "files": files
            }
        return None

    def add_skill(self, skill_id: str, content: str) -> dict:
        """Add or update a simple skill (SKILL.md only)."""
        # Sanitize skill_id
        skill_id = "".join(c for c in skill_id if c.isalnum() or c in "-_").lower()
        if not skill_id:
            raise ValueError("Invalid skill ID")
        
        skill_dir = SKILLS_DIR / skill_id
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file = skill_dir / "SKILL.md"
        existed = skill_file.exists()
        skill_file.write_text(content)
        
        return {
            "id": skill_id,
            "path": str(skill_file),
            "created": not existed
        }

    def add_skill_from_zip(self, zip_data: bytes) -> dict:
        """
        Add a skill from a zip file.
        
        The zip should contain a skill directory with SKILL.md at its root.
        Can be structured as:
        - skill-name/SKILL.md (directory at root)
        - SKILL.md (files at root, skill ID derived from zip name or frontmatter)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            zip_path = tmp_path / "skill.zip"
            
            # Write zip data
            zip_path.write_bytes(zip_data)
            
            # Extract
            extract_dir = tmp_path / "extracted"
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_dir)
            
            # Find SKILL.md - could be at root or in a subdirectory
            skill_md_files = list(extract_dir.rglob("SKILL.md"))
            
            if not skill_md_files:
                raise ValueError("No SKILL.md found in zip file")
            
            # Use the first SKILL.md found
            skill_md = skill_md_files[0]
            skill_source_dir = skill_md.parent
            
            # Determine skill ID from directory name or frontmatter
            content = skill_md.read_text()
            skill_id = skill_source_dir.name
            
            # Try to get name from frontmatter
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = parts[1]
                    for line in frontmatter.strip().split("\n"):
                        if line.startswith("name:"):
                            potential_id = line.split(":", 1)[1].strip()
                            # Sanitize for use as directory name
                            potential_id = "".join(c for c in potential_id if c.isalnum() or c in "-_ ").lower()
                            potential_id = potential_id.replace(" ", "-")
                            if potential_id:
                                skill_id = potential_id
                            break
            
            # If skill_source_dir is extract_dir itself (files at root), use skill_id
            if skill_source_dir == extract_dir:
                skill_id = skill_id if skill_id != "extracted" else "imported-skill"
            
            # Sanitize skill_id
            skill_id = "".join(c for c in skill_id if c.isalnum() or c in "-_").lower()
            if not skill_id:
                skill_id = "imported-skill"
            
            # Target directory
            target_dir = SKILLS_DIR / skill_id
            
            # Remove existing if present
            if target_dir.exists():
                shutil.rmtree(target_dir)
            
            # Copy the skill directory
            shutil.copytree(skill_source_dir, target_dir)
            
            file_count = self._count_files(target_dir)
            
            return {
                "id": skill_id,
                "path": str(target_dir),
                "file_count": file_count
            }

    def delete_skill(self, skill_id: str) -> bool:
        """Delete a skill."""
        skill_dir = SKILLS_DIR / skill_id
        if skill_dir.exists() and skill_dir.is_dir():
            shutil.rmtree(skill_dir)
            return True
        return False
    
    def export_skill_zip(self, skill_id: str) -> Optional[bytes]:
        """Export a skill as a zip file."""
        skill_dir = SKILLS_DIR / skill_id
        if not skill_dir.exists():
            return None
        
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / f"{skill_id}.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file_path in skill_dir.rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to(skill_dir.parent)
                        zf.write(file_path, arcname)
            return zip_path.read_bytes()

    # Workspace file management
    def list_workspace_files(self, subdir: str = "") -> list[dict]:
        """List files in workspace directory."""
        target_dir = WORKSPACE_DIR / subdir if subdir else WORKSPACE_DIR
        if not target_dir.exists():
            return []
        
        files = []
        for item in sorted(target_dir.iterdir()):
            rel_path = item.relative_to(WORKSPACE_DIR)
            files.append({
                "name": item.name,
                "path": str(rel_path),
                "is_dir": item.is_dir(),
                "size": item.stat().st_size if item.is_file() else None,
                "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
            })
        return files
    
    def get_workspace_file(self, file_path: str) -> Optional[tuple[bytes, str]]:
        """Get a file from workspace. Returns (content, filename) or None."""
        # Prevent directory traversal
        safe_path = Path(file_path).name if ".." in file_path else file_path
        full_path = WORKSPACE_DIR / safe_path
        
        if not full_path.exists() or not full_path.is_file():
            return None
        
        # Check it's within workspace
        try:
            full_path.resolve().relative_to(WORKSPACE_DIR.resolve())
        except ValueError:
            return None
        
        return (full_path.read_bytes(), full_path.name)
    
    def delete_workspace_file(self, file_path: str) -> bool:
        """Delete a file or directory from workspace."""
        safe_path = Path(file_path).name if ".." in file_path else file_path
        full_path = WORKSPACE_DIR / safe_path
        
        try:
            full_path.resolve().relative_to(WORKSPACE_DIR.resolve())
        except ValueError:
            return False
        
        if full_path.exists():
            if full_path.is_dir():
                shutil.rmtree(full_path)
            else:
                full_path.unlink()
            return True
        return False

    # Command management methods
    def list_commands(self) -> list[dict]:
        """List all available commands."""
        commands = []
        if COMMANDS_DIR.exists():
            for cmd_file in COMMANDS_DIR.glob("*.md"):
                commands.append({
                    "id": cmd_file.stem,
                    "path": str(cmd_file)
                })
        return commands

    def get_command(self, command_id: str) -> Optional[str]:
        """Get a command template by ID. Returns the template string or None."""
        cmd_file = COMMANDS_DIR / f"{command_id}.md"
        if cmd_file.exists():
            return cmd_file.read_text()
        return None

    def add_command(self, command_id: str, template: str) -> dict:
        """Add or update a command template."""
        # Sanitize command_id
        command_id = "".join(c for c in command_id if c.isalnum() or c in "-_").lower()
        if not command_id:
            raise ValueError("Invalid command ID")

        cmd_file = COMMANDS_DIR / f"{command_id}.md"
        existed = cmd_file.exists()
        cmd_file.write_text(template)

        return {
            "id": command_id,
            "path": str(cmd_file),
            "created": not existed
        }

    def delete_command(self, command_id: str) -> bool:
        """Delete a command."""
        cmd_file = COMMANDS_DIR / f"{command_id}.md"
        if cmd_file.exists():
            cmd_file.unlink()
            return True
        return False

    # Session management methods
    def _get_sessions_dir(self) -> Path:
        """Get the Claude sessions directory path."""
        # Claude stores sessions at ~/.claude/projects/{project-path-hash}/
        # For workspace at /app/workspace, Claude uses -app-workspace as the hash
        home = Path.home()
        sessions_base = home / ".claude" / "projects"

        # Look for the workspace project directory
        # The path encoding replaces / with - (keeping the leading dash)
        workspace_path = str(WORKSPACE_DIR.resolve())
        # Convert /app/workspace to -app-workspace (keep leading dash!)
        encoded_path = workspace_path.replace("/", "-")

        project_dir = sessions_base / encoded_path
        return project_dir

    def list_sessions(self) -> list[dict]:
        """List all Claude sessions ordered by modified date."""
        sessions = []
        sessions_dir = self._get_sessions_dir()

        if not sessions_dir.exists():
            return sessions

        for session_file in sessions_dir.glob("*.jsonl"):
            stat = session_file.stat()
            sessions.append({
                "id": session_file.stem,
                "filename": session_file.name,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "modified_iso": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })

        # Sort by modified date, newest first
        sessions.sort(key=lambda x: x["modified"], reverse=True)
        return sessions

    def get_session(self, session_id: str) -> Optional[dict]:
        """Get a session's JSONL content parsed into structured data."""
        sessions_dir = self._get_sessions_dir()
        session_file = sessions_dir / f"{session_id}.jsonl"

        if not session_file.exists():
            return None

        entries = []
        try:
            with open(session_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            entries.append(entry)
                        except json.JSONDecodeError:
                            entries.append({"_parse_error": True, "_line": line_num, "_raw": line[:200]})
        except Exception as e:
            return {"error": str(e)}

        stat = session_file.stat()
        return {
            "id": session_id,
            "filename": session_file.name,
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "modified_iso": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "entry_count": len(entries),
            "entries": entries
        }

    def get_session_raw(self, session_id: str) -> Optional[str]:
        """Get a session's raw JSONL content."""
        sessions_dir = self._get_sessions_dir()
        session_file = sessions_dir / f"{session_id}.jsonl"

        if not session_file.exists():
            return None

        return session_file.read_text()

    async def close(self):
        await self.redis.close()

