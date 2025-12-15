import json
import os
import logging
import re
import shutil
import stat
import zipfile
import tempfile
import dataclasses
import io
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from claude_code_sdk import ClaudeCodeOptions, query
from claude_code_sdk.types import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock, SystemMessage, UserMessage
import redis.asyncio as redis

logger = logging.getLogger(__name__)

# Workspace directory for agent file operations
# Can be overridden via WORKSPACE_DIR env var (for Railway volume mount)
def _resolve_workspace_dir() -> Path:
    configured = os.environ.get("WORKSPACE_DIR")
    if configured:
        workspace_dir = Path(configured)
        workspace_dir.mkdir(parents=True, exist_ok=True)
        return workspace_dir

    # Railway/Docker default.
    if Path("/app").exists():
        workspace_dir = Path("/app/workspace")
        workspace_dir.mkdir(parents=True, exist_ok=True)
        return workspace_dir

    # Local dev default: repo root (this file's directory), which also contains `.claude/`.
    return Path(__file__).resolve().parent


WORKSPACE_DIR = _resolve_workspace_dir()

# Skills directory - on the volume for runtime management
# Can be overridden via SKILLS_DIR env var
SKILLS_DIR = Path(os.environ.get("SKILLS_DIR", str(WORKSPACE_DIR / ".claude" / "skills")))

# Commands directory - prompt templates on the volume
# Can be overridden via COMMANDS_DIR env var
COMMANDS_DIR = Path(os.environ.get("COMMANDS_DIR", str(WORKSPACE_DIR / ".claude" / "commands")))

_IDENTIFIER_RE = re.compile(r"^[a-z0-9_-]+$")

_WEBHOOK_REQUIRED_ALLOW_RULES: list[str] = [
    # Keep webhook runs non-interactive by allowing a small set of bash commands used by volume-managed commands.
    "Bash(cat:*)",
    "Bash(echo:*)",
    "Bash(date:*)",
    "Bash(mkdir:*)",
    # python3 script helpers (two parsing styles: `python3:<args>` vs `python3 <script>:<args>`)
    "Bash(python3:./.claude/commands/scripts/*)",
    "Bash(python3:.claude/commands/scripts/*)",
    "Bash(python3:./.claude/commands/scripts/save_transcript.py*)",
    "Bash(python3:.claude/commands/scripts/save_transcript.py*)",
    # Some versions of the bash parser treat "python3 <script>" as the command+subcommand.
    "Bash(python3 ./.claude/commands/scripts/save_transcript.py:*)",
    "Bash(python3 .claude/commands/scripts/save_transcript.py:*)",
    "Bash(python3 ./.claude/commands/scripts/*:*)",
    "Bash(python3 .claude/commands/scripts/*:*)",
]

def _format_query_error(*, stderr_text: str, exc: Exception) -> RuntimeError:
    stderr_text = (stderr_text or "").strip()
    if stderr_text:
        return RuntimeError(stderr_text)
    return RuntimeError(str(exc))

def _normalize_identifier(raw: str, *, kind: str) -> str:
    value = (raw or "").strip().lower()
    if not value or not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"Invalid {kind} ID")
    return value


def _resolve_under(base_dir: Path, user_path: str) -> Path:
    base_resolved = base_dir.resolve()
    rel = Path(user_path or "")
    if rel.is_absolute():
        raise ValueError("Absolute paths are not allowed")
    full_path = (base_resolved / rel).resolve()
    full_path.relative_to(base_resolved)
    return full_path


async def _collect_query_events(
    *,
    prompt: str | Any,
    options: ClaudeCodeOptions,
) -> tuple[list[Any], Optional[RuntimeError]]:
    stderr_buf = io.StringIO()
    opts = dataclasses.replace(options, debug_stderr=stderr_buf, model=(options.model or None))
    events: list[Any] = []
    try:
        async for msg in query(prompt=prompt, options=opts):
            events.append(msg)
    except Exception as e:
        return events, _format_query_error(stderr_text=stderr_buf.getvalue(), exc=e)
    return events, None


class AgentManager:
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
        self.conversation_histories: dict[str, list[dict]] = {}
        # Ensure skills and commands directories exist
        SKILLS_DIR.mkdir(parents=True, exist_ok=True)
        COMMANDS_DIR.mkdir(parents=True, exist_ok=True)

    def _resolve_permission_mode(self, context: Optional[dict]) -> str:
        mode = context.get("permission_mode", "acceptEdits") if context else "acceptEdits"
        if mode == "bypassPermissions" and os.environ.get("ALLOW_BYPASS_PERMISSIONS", "0") != "1":
            raise PermissionError("permission_mode=bypassPermissions is disabled on this server")
        if (context or {}).get("source") == "webhook" and mode == "bypassPermissions":
            raise PermissionError("permission_mode=bypassPermissions is not allowed for webhook runs")
        return mode

    def _build_webhook_settings(self) -> str:
        candidate = WORKSPACE_DIR / ".claude" / "settings.json"
        settings_obj: dict[str, Any] = {}
        if candidate.is_file():
            try:
                settings_obj = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                settings_obj = {}

        permissions = settings_obj.get("permissions") if isinstance(settings_obj.get("permissions"), dict) else {}
        allow_list = permissions.get("allow") if isinstance(permissions.get("allow"), list) else []
        allow_list = list(allow_list)

        for rule in _WEBHOOK_REQUIRED_ALLOW_RULES:
            if rule not in allow_list:
                allow_list.append(rule)

        permissions["allow"] = allow_list
        settings_obj["permissions"] = permissions
        return json.dumps(settings_obj)
    
    async def _get_stored_session(self, user_session_id: str) -> Optional[dict]:
        data = await self.redis.get(f"session:{user_session_id}")
        if data:
            return json.loads(data)
        return None
    
    async def _store_session(
        self,
        user_session_id: str,
        *,
        claude_session_id: Optional[str] = None,
        conversation_summary: str = "",
    ):
        existing = await self._get_stored_session(user_session_id)
        created = existing.get("created") if existing else None
        if not created:
            created = datetime.utcnow().isoformat()

        summary = conversation_summary or (existing.get("summary") if existing else "") or ""

        record: dict[str, Any] = {
            "created": created,
            "last_active": datetime.utcnow().isoformat(),
            "summary": summary,
        }
        existing_claude_session_id = (existing or {}).get("claude_session_id")
        record["claude_session_id"] = claude_session_id or existing_claude_session_id

        await self.redis.set(
            f"session:{user_session_id}",
            json.dumps(record),
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
        stored = await self._get_stored_session(user_session_id)

        raw_message = message.strip()
        is_slash_command = raw_message.startswith("/")

        # Build the prompt with per-request context, but don't break slash command preprocessing.
        text_content = message
        if context and not is_slash_command:
            source = context.get("source", "unknown")
            user_name = context.get("user_name", "User")
            text_content = f"[Context: {user_name} via {source}]\n\n{message}"
        
        # Build message content - either string or list with images.
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

        # Preserve Claude Code session for interactive chat, but avoid resuming for webhook calls
        # (webhooks are typically stateless and should always pick up latest volume commands/cwd).
        resume_session_id: Optional[str] = None
        if (context or {}).get("source") != "webhook":
            resume_session_id = (stored or {}).get("claude_session_id")

        # Default to acceptEdits (safer), can override to bypassPermissions via API
        permission_mode = self._resolve_permission_mode(context)

        # Webhook runs are non-interactive; ensure required permission rules are present so we
        # don't hang on approval prompts (e.g., command helpers like save_transcript.py).
        settings: Optional[str] = None
        if (context or {}).get("source") == "webhook":
            settings = self._build_webhook_settings()
        
        # Set working directory to workspace for file operations and for discovering .claude/commands/ etc.
        options = ClaudeCodeOptions(
            permission_mode=permission_mode,
            cwd=str(WORKSPACE_DIR),
            model=(model or None),
            resume=resume_session_id,
            settings=settings,
        )
        
        # query() enables Claude Code preprocessing for slash commands and !` bash execution.
        prompt: str | Any
        if images:
            async def message_generator():
                yield {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": content
                    }
                }
            prompt = message_generator()
        else:
            prompt = text_content

        claude_session_id: Optional[str] = None
        usage: dict[str, Any] = {}

        events, err = await _collect_query_events(prompt=prompt, options=options)
        if err and options.model and not events:
            fallback_options = dataclasses.replace(options, model=None)
            events, err = await _collect_query_events(prompt=prompt, options=fallback_options)
        if err:
            raise err

        for msg in events:
            if isinstance(msg, SystemMessage):
                if msg.subtype == "init":
                    claude_session_id = msg.data.get("session_id") or claude_session_id
            elif isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        response_parts.append(block.text)
                    elif isinstance(block, ToolUseBlock):
                        tools_used.append(block.name)
            elif isinstance(msg, ResultMessage):
                claude_session_id = msg.session_id or claude_session_id
                usage = msg.usage or {"num_turns": msg.num_turns}
                if usage.get("num_turns") is None:
                    usage["num_turns"] = msg.num_turns
        
        response_text = "".join(response_parts)
        
        # Update server-side metadata (and keep a lightweight transcript for UI/debugging).
        history = await self._get_conversation_history(user_session_id)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response_text})
        await self._store_conversation_history(user_session_id, history)

        # If user explicitly cleared context, also clear our local transcript.
        if raw_message.startswith("/clear"):
            await self.redis.delete(f"history:{user_session_id}")

        await self._store_session(user_session_id, claude_session_id=claude_session_id)
        await self._update_session_activity(user_session_id)
        
        return {
            "session_id": user_session_id,
            "response": response_text,
            "tools_used": list(set(tools_used)),
            "usage": usage or {"num_turns": len(history) // 2},
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
        stored = await self._get_stored_session(user_session_id)

        raw_message = message.strip()
        is_slash_command = raw_message.startswith("/")

        # Build the prompt with per-request context, but don't break slash command preprocessing.
        text_content = message
        if context and not is_slash_command:
            source = context.get("source", "unknown")
            user_name = context.get("user_name", "User")
            text_content = f"[Context: {user_name} via {source}]\n\n{message}"
        
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
        
        tools_used = []
        response_parts = []

        # Preserve Claude Code session for interactive chat, but avoid resuming for webhook calls
        # (webhooks are typically stateless and should always pick up latest volume commands/cwd).
        resume_session_id: Optional[str] = None
        if (context or {}).get("source") != "webhook":
            resume_session_id = (stored or {}).get("claude_session_id")
        
        # Default to acceptEdits (safer), can override to bypassPermissions via API
        permission_mode = self._resolve_permission_mode(context)

        settings: Optional[str] = None
        if (context or {}).get("source") == "webhook":
            settings = self._build_webhook_settings()
        
        # Set working directory to workspace for file operations and for discovering .claude/commands/ etc.
        options = ClaudeCodeOptions(
            permission_mode=permission_mode,
            cwd=str(WORKSPACE_DIR),
            model=(model or None),
            resume=resume_session_id,
            settings=settings,
        )
        
        # Signal that we're starting
        yield {"type": "status", "status": "connecting"}

        yield {"type": "status", "status": "sending"}
        yield {"type": "status", "status": "processing"}

        # query() enables Claude Code preprocessing for slash commands and !` bash execution.
        prompt: str | Any
        if images:
            async def message_generator():
                yield {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": content
                    }
                }
            prompt = message_generator()
        else:
            prompt = text_content

        async def run_stream(current_options: ClaudeCodeOptions):
            nonlocal claude_session_id, usage

            stderr_buf = io.StringIO()
            opts = dataclasses.replace(
                current_options,
                debug_stderr=stderr_buf,
                model=(current_options.model or None),
            )
            emitted_any_output = False
            try:
                async for msg in query(prompt=prompt, options=opts):
                    if isinstance(msg, SystemMessage):
                        if msg.subtype == "init":
                            claude_session_id = msg.data.get("session_id") or claude_session_id
                            yield {"type": "status", "status": "ready"}
                    elif isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if isinstance(block, TextBlock):
                                emitted_any_output = True
                                response_parts.append(block.text)
                                yield {"type": "text", "text": block.text}
                            elif isinstance(block, ToolUseBlock):
                                emitted_any_output = True
                                tools_used.append(block.name)
                                yield {"type": "tool", "name": block.name, "status": "started"}
                    elif isinstance(msg, UserMessage):
                        if tools_used:
                            yield {"type": "tool", "name": tools_used[-1], "status": "completed"}
                    elif isinstance(msg, ResultMessage):
                        claude_session_id = msg.session_id or claude_session_id
                        usage = msg.usage or {"num_turns": msg.num_turns}
                        if usage.get("num_turns") is None:
                            usage["num_turns"] = msg.num_turns
            except Exception as e:
                stderr_text = stderr_buf.getvalue()
                if not emitted_any_output and current_options.model:
                    fallback_options = dataclasses.replace(current_options, model=None)
                    async for ev in run_stream(fallback_options):
                        yield ev
                    return
                if stderr_text.strip():
                    raise RuntimeError(stderr_text.strip()) from e
                raise

        claude_session_id: Optional[str] = None
        usage: dict[str, Any] = {}

        async for ev in run_stream(options):
            yield ev
        
        response_text = "".join(response_parts)
        
        # Update server-side metadata (and keep a lightweight transcript for UI/debugging).
        history = await self._get_conversation_history(user_session_id)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response_text})
        await self._store_conversation_history(user_session_id, history)

        # If user explicitly cleared context, also clear our local transcript.
        if raw_message.startswith("/clear"):
            await self.redis.delete(f"history:{user_session_id}")

        await self._store_session(user_session_id, claude_session_id=claude_session_id)
        await self._update_session_activity(user_session_id)
        
        # Yield final done event
        yield {
            "type": "done",
            "session_id": user_session_id,
            "tools_used": list(set(tools_used)),
            "usage": usage or {"num_turns": len(history) // 2},
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
        skill_id = _normalize_identifier(skill_id, kind="skill")
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
        skill_id = _normalize_identifier(skill_id, kind="skill")
        
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
        max_files = int(os.environ.get("MAX_SKILL_ZIP_FILES", "200"))
        max_total_bytes = int(os.environ.get("MAX_SKILL_ZIP_TOTAL_UNCOMPRESSED_BYTES", str(50 * 1024 * 1024)))
        max_file_bytes = int(os.environ.get("MAX_SKILL_ZIP_FILE_UNCOMPRESSED_BYTES", str(10 * 1024 * 1024)))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            zip_path = tmp_path / "skill.zip"
            
            # Write zip data
            zip_path.write_bytes(zip_data)
            
            # Extract
            extract_dir = tmp_path / "extracted"
            with zipfile.ZipFile(zip_path, 'r') as zf:
                members = zf.infolist()
                if len(members) > max_files:
                    raise ValueError(f"Zip contains too many files (max {max_files})")

                total_uncompressed = 0
                extract_base = extract_dir.resolve()

                for info in members:
                    name = (info.filename or "").replace("\\", "/")
                    if not name or name.endswith("/"):
                        continue

                    member_path = Path(name)
                    if member_path.is_absolute() or ".." in member_path.parts:
                        raise ValueError("Zip contains unsafe paths")

                    mode = (info.external_attr >> 16) & 0o777777
                    if stat.S_ISLNK(mode):
                        raise ValueError("Zip contains symlinks, which are not allowed")

                    if info.file_size > max_file_bytes:
                        raise ValueError(f"Zip member '{name}' exceeds max size ({max_file_bytes} bytes)")
                    total_uncompressed += int(info.file_size)
                    if total_uncompressed > max_total_bytes:
                        raise ValueError(f"Zip exceeds max uncompressed size ({max_total_bytes} bytes)")

                    dest_path = (extract_base / member_path).resolve()
                    dest_path.relative_to(extract_base)
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info, "r") as src, open(dest_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
            
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
        skill_id = _normalize_identifier(skill_id, kind="skill")
        skill_dir = SKILLS_DIR / skill_id
        if skill_dir.exists() and skill_dir.is_dir():
            shutil.rmtree(skill_dir)
            return True
        return False
    
    def export_skill_zip(self, skill_id: str) -> Optional[bytes]:
        """Export a skill as a zip file."""
        skill_id = _normalize_identifier(skill_id, kind="skill")
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
        try:
            target_dir = _resolve_under(WORKSPACE_DIR, subdir) if subdir else WORKSPACE_DIR.resolve()
        except ValueError:
            return []
        if not target_dir.exists():
            return []
        if not target_dir.is_dir():
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
        try:
            full_path = _resolve_under(WORKSPACE_DIR, file_path)
        except ValueError:
            return None
        
        if not full_path.exists() or not full_path.is_file():
            return None

        return (full_path.read_bytes(), full_path.name)
    
    def delete_workspace_file(self, file_path: str) -> bool:
        """Delete a file or directory from workspace."""
        try:
            full_path = _resolve_under(WORKSPACE_DIR, file_path)
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
                rel_path = cmd_file.relative_to(WORKSPACE_DIR) if cmd_file.is_relative_to(WORKSPACE_DIR) else None
                commands.append({
                    "id": cmd_file.stem,
                    "path": str(cmd_file),
                    "relative_path": str(rel_path) if rel_path else None
                })
        return commands

    def get_command(self, command_id: str) -> Optional[str]:
        """Get a command template by ID. Returns the template string or None."""
        command_id = _normalize_identifier(command_id, kind="command")
        cmd_file = COMMANDS_DIR / f"{command_id}.md"
        if cmd_file.exists():
            return cmd_file.read_text()
        return None

    def add_command(self, command_id: str, template: str) -> dict:
        """Add or update a command template."""
        command_id = _normalize_identifier(command_id, kind="command")

        cmd_file = COMMANDS_DIR / f"{command_id}.md"
        existed = cmd_file.exists()
        cmd_file.write_text(template)

        return {
            "id": command_id,
            "path": str(cmd_file),
            "created": not existed
        }

    def write_workspace_file(self, file_path: str, content: str) -> dict:
        """Create or update a text file in the workspace."""
        try:
            full_path = _resolve_under(WORKSPACE_DIR, file_path)
        except ValueError:
            raise ValueError("Path must stay within workspace")

        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)

        stat = full_path.stat()
        return {
            "path": str(full_path.relative_to(WORKSPACE_DIR)),
            "size": stat.st_size,
            "modified": stat.st_mtime
        }

    def delete_command(self, command_id: str) -> bool:
        """Delete a command."""
        command_id = _normalize_identifier(command_id, kind="command")
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
        except Exception:
            logger.exception("Failed to read session %s", session_id)
            return {"error": "Failed to read session"}

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

        try:
            return session_file.read_text()
        except Exception:
            logger.exception("Failed to read raw session %s", session_id)
            return None

    async def close(self):
        await self.redis.close()
