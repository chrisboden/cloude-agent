"""
This module implements the AgentManager class, which is used to manage the agent's state and interactions with the user.
It provides methods for:
- Chatting with the agent
- Streaming chat responses
- Managing skills and commands
- Managing workspace files
"""
import asyncio
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
import hashlib
import subprocess
import platform
import sys
import importlib.metadata as importlib_metadata
import fnmatch
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, query
from claude_agent_sdk.types import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    SystemMessage,
    UserMessage,
    SystemPromptPreset,
    PermissionResultAllow,
    PermissionResultDeny,
)
import redis.asyncio as redis

logger = logging.getLogger(__name__)
_PERMISSION_LOGGING_ENABLED = os.environ.get("PERMISSION_LOGGING", "1") == "1"
_permission_logger = logging.getLogger("cloude.permissions")
_PERMISSION_LOG_MAX = int(os.environ.get("PERMISSION_LOG_MAX", "200"))
_PERMISSION_LOG_GLOBAL_KEY = "permission_log:global"
_PERMISSION_LOG_SESSION_KEY_PREFIX = "permission_log:session:"


def _configure_permission_logger() -> logging.Logger:
    if not _PERMISSION_LOGGING_ENABLED:
        return _permission_logger
    if _permission_logger.handlers:
        return _permission_logger
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "PERMISSION %(asctime)s %(levelname)s session=%(session)s tool=%(tool)s decision=%(decision)s reason=%(reason)s rule=%(rule)s input=%(input)s"
    )
    handler.setFormatter(formatter)
    _permission_logger.addHandler(handler)
    _permission_logger.setLevel(logging.INFO)
    _permission_logger.propagate = False
    return _permission_logger

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
PROJECT_CONTEXT_PATH = Path(
    os.environ.get("PROJECT_CONTEXT_PATH", str(WORKSPACE_DIR / ".claude" / "CLAUDE.md"))
)
PROMPTS_DIR = Path(os.environ.get("PROMPTS_DIR", str(WORKSPACE_DIR / "prompts")))
_IMAGE_PROJECT_CONTEXT_FALLBACK = Path("/app/.claude/CLAUDE.md")

_IDENTIFIER_RE = re.compile(r"^[a-z0-9_-]+$")

_WEBHOOK_REQUIRED_ALLOW_RULES: list[str] = [
    # Keep webhook runs non-interactive by allowing a small set of bash commands used by volume-managed commands.
    "Bash(cat:*)",
    "Bash(echo:*)",
    "Bash(date:*)",
    "Bash(mkdir:*)",
    # python3 script helpers (two parsing styles: `python3:<args>` vs `python3 <script>:<args>`)
    "Bash(python3:./.claude/scripts/*)",
    "Bash(python3:.claude/scripts/*)",
    # Some versions of the bash parser treat "python3 <script>" as the command+subcommand.
    "Bash(python3 ./.claude/scripts/*:*)",
    "Bash(python3 .claude/scripts/*:*)",
]
_DEFAULT_SETTING_SOURCES = ("project",)
_INTERRUPT_TTL_S = 86400 * 7

def _format_query_error(*, stderr_text: str, exc: Exception) -> RuntimeError:
    stderr_text = (stderr_text or "").strip()
    if stderr_text:
        return RuntimeError(stderr_text)
    return RuntimeError(str(exc))


class _StreamError(RuntimeError):
    def __init__(self, *, stderr_text: str, exc: Exception, emitted_any_output: bool):
        super().__init__(str(exc))
        self.stderr_text = (stderr_text or "").strip()
        self.emitted_any_output = emitted_any_output

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


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    lines = (text or "").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            metadata: dict[str, str] = {}
            for line in lines[1:idx]:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or ":" not in stripped:
                    continue
                key, value = stripped.split(":", 1)
                metadata[key.strip().lower()] = value.strip().strip('"').strip("'")
            body = "\n".join(lines[idx + 1 :])
            return metadata, body

    return {}, text


def _merge_settings(base: Any, overlay: Any) -> Any:
    if isinstance(base, dict) and isinstance(overlay, dict):
        merged = dict(base)
        for key, value in overlay.items():
            if key in merged:
                merged[key] = _merge_settings(merged[key], value)
            else:
                merged[key] = value
        return merged
    if isinstance(base, list) and isinstance(overlay, list):
        combined = list(base)
        for item in overlay:
            if item not in combined:
                combined.append(item)
        return combined
    return overlay


_MAX_SETTINGS_BYTES = 1024 * 1024


def _is_sensitive_env_key(key: str) -> bool:
    if not key:
        return False
    upper = key.upper()
    if upper.endswith(("_KEY", "_TOKEN", "_SECRET", "_PASSWORD")):
        return True
    if "API_KEY" in upper or "AUTH_TOKEN" in upper or "ACCESS_TOKEN" in upper:
        return True
    if "SECRET" in upper or "PASSWORD" in upper:
        return True
    return False


def _is_sensitive_key(key: str) -> bool:
    if not key:
        return False
    upper = key.upper()
    if _is_sensitive_env_key(upper):
        return True
    if "OAUTH" in upper or "AUTH_TOKEN" in upper:
        return True
    if "ACCESS_TOKEN" in upper or "REFRESH_TOKEN" in upper:
        return True
    return False


def _redact_settings_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        redacted: dict[str, Any] = {}
        for key, value in payload.items():
            if key == "env" and isinstance(value, dict):
                env_redacted: dict[str, Any] = {}
                for env_key, env_value in value.items():
                    if _is_sensitive_env_key(env_key):
                        env_redacted[env_key] = "<redacted>"
                    else:
                        env_redacted[env_key] = env_value
                redacted[key] = env_redacted
                continue
            redacted[key] = _redact_settings_payload(value)
        return redacted
    if isinstance(payload, list):
        return [_redact_settings_payload(item) for item in payload]
    return payload


def _redact_sensitive_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        redacted: dict[str, Any] = {}
        for key, value in payload.items():
            if _is_sensitive_key(key):
                redacted[key] = "<redacted>"
            else:
                redacted[key] = _redact_sensitive_payload(value)
        return redacted
    if isinstance(payload, list):
        return [_redact_sensitive_payload(item) for item in payload]
    return payload


def _path_debug_info(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "is_symlink": path.is_symlink(),
    }
    try:
        info["resolved"] = str(path.resolve())
    except Exception:
        pass
    if info["is_symlink"]:
        try:
            info["symlink_target"] = str(path.resolve())
        except Exception:
            pass
    return info


def _settings_file_debug(path: Path, *, redactor=_redact_settings_payload) -> dict[str, Any]:
    info: dict[str, Any] = {"path": str(path)}
    try:
        stat_info = path.stat()
    except FileNotFoundError:
        info["exists"] = False
        return info
    except Exception as exc:
        info["exists"] = False
        info["error"] = str(exc)
        return info

    info.update(
        {
            "exists": True,
            "is_file": path.is_file(),
            "is_dir": path.is_dir(),
            "is_symlink": path.is_symlink(),
            "size": stat_info.st_size,
            "mtime": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
            "readable": os.access(path, os.R_OK),
        }
    )

    if info["is_symlink"]:
        try:
            info["symlink_target"] = str(path.resolve())
        except Exception:
            pass

    if info["is_file"] and info["readable"]:
        if stat_info.st_size > _MAX_SETTINGS_BYTES:
            info["skipped_read"] = f"size>{_MAX_SETTINGS_BYTES}"
        else:
            try:
                raw_bytes = path.read_bytes()
                info["sha256"] = hashlib.sha256(raw_bytes).hexdigest()
                try:
                    payload = json.loads(raw_bytes.decode("utf-8"))
                    info["parse_ok"] = True
                    info["payload"] = redactor(payload)
                except Exception as exc:
                    info["parse_ok"] = False
                    info["parse_error"] = str(exc)
            except Exception as exc:
                info["read_error"] = str(exc)
    return info


def _extract_bash_rule_prefix(rule: str) -> Optional[str]:
    if not rule.startswith("Bash(") or not rule.endswith(")"):
        return None
    inner = rule[len("Bash("):-1]
    star_idx = inner.find("*")
    if star_idx != -1:
        inner = inner[:star_idx]
    return inner


def _match_bash_rules(rules: list[str], command: str) -> list[dict[str, str]]:
    matches: list[dict[str, str]] = []
    for rule in rules:
        prefix = _extract_bash_rule_prefix(rule)
        if prefix is None:
            continue
        if command.startswith(prefix):
            matches.append({"rule": rule, "prefix": prefix})
    return matches


def _path_candidates(raw_path: Optional[str]) -> list[str]:
    if not raw_path:
        return []
    candidates = [raw_path]
    try:
        resolved = str(Path(raw_path).expanduser().resolve())
        if resolved not in candidates:
            candidates.append(resolved)
        try:
            rel = str(Path(resolved).relative_to(WORKSPACE_DIR.resolve()))
            if rel not in candidates:
                candidates.append(rel)
            dotted = f"./{rel}"
            if dotted not in candidates:
                candidates.append(dotted)
        except Exception:
            pass
    except Exception:
        pass
    return candidates


def _match_rule(tool: str, input_data: dict, rule: str) -> bool:
    if rule == tool:
        return True
    prefix = f"{tool}("
    if not rule.startswith(prefix) or not rule.endswith(")"):
        return False
    pattern = rule[len(prefix):-1]
    if tool == "Bash":
        command = input_data.get("command") or ""
        return command.startswith(pattern)
    if tool in {"Read", "Write", "Edit"}:
        file_path = input_data.get("file_path") or input_data.get("path") or ""
        if not file_path:
            return False
        patterns = [pattern]
        if pattern.startswith("./"):
            patterns.append(pattern[2:])
        for candidate in _path_candidates(file_path):
            for candidate_pattern in patterns:
                if fnmatch.fnmatch(candidate, candidate_pattern):
                    return True
        return False
    return False


def _summarize_tool_input(tool: str, input_data: dict) -> str:
    if not isinstance(input_data, dict):
        return ""
    if tool == "Bash":
        return str(input_data.get("command") or "")
    for key in ("file_path", "path", "url", "command"):
        if key in input_data:
            return str(input_data.get(key) or "")
    return ""


def _redact_sensitive_text(text: str) -> str:
    if not text:
        return text
    redacted = text
    redacted = re.sub(
        r"(?i)\b(bearer)\s+([a-z0-9._-]+)",
        r"\1 <redacted>",
        redacted,
    )
    redacted = re.sub(
        r"(?i)\b(api[_-]?key|token|secret|password)\s*[:=]\s*([^\s]+)",
        r"\1=<redacted>",
        redacted,
    )
    redacted = re.sub(
        r"(?i)--(api[_-]?key|token|secret|password)\s+([^\s]+)",
        r"--\1 <redacted>",
        redacted,
    )
    return redacted


def _sanitize_permission_input(text: str, *, max_len: int = 400) -> str:
    if not text:
        return ""
    cleaned = _redact_sensitive_text(text)
    if len(cleaned) > max_len:
        return cleaned[: max_len - 3] + "..."
    return cleaned


async def _collect_query_events(
    *,
    prompt: str | Any,
    options: ClaudeAgentOptions,
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
        self._active_streams: dict[str, ClaudeSDKClient] = {}
        self._active_streams_lock = asyncio.Lock()
        self._settings_cache_paths = {
            "default": Path(
                os.environ.get(
                    "CLAUDE_SETTINGS_CACHE_PATH",
                    f"/tmp/claude-settings-{os.getpid()}.json",
                )
            ),
            "webhook": Path(
                os.environ.get(
                    "CLAUDE_WEBHOOK_SETTINGS_CACHE_PATH",
                    f"/tmp/claude-settings-webhook-{os.getpid()}.json",
                )
            ),
        }
        # Ensure skills and commands directories exist
        SKILLS_DIR.mkdir(parents=True, exist_ok=True)
        COMMANDS_DIR.mkdir(parents=True, exist_ok=True)
        PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        self._ensure_project_context_file()

    def _ensure_project_context_file(self) -> None:
        """Ensure the project context file exists on the workspace volume (non-destructive)."""
        try:
            target = PROJECT_CONTEXT_PATH
            if target.is_file():
                return

            # Only auto-create when the target lives under the workspace.
            workspace_root = WORKSPACE_DIR.resolve()
            try:
                target.resolve().relative_to(workspace_root)
            except Exception:
                return

            source_text: Optional[str] = None
            if _IMAGE_PROJECT_CONTEXT_FALLBACK.is_file():
                source_text = _IMAGE_PROJECT_CONTEXT_FALLBACK.read_text(encoding="utf-8")

            if not source_text:
                source_text = "# Project Context\n\n(Write project-specific instructions here.)\n"

            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source_text, encoding="utf-8")
        except Exception:
            logger.exception("Failed to ensure project context file at %s", PROJECT_CONTEXT_PATH)

    def _resolve_permission_mode(self, context: Optional[dict]) -> Optional[str]:
        mode = context.get("permission_mode") if context else None
        if not mode or mode == "default":
            return None
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

    def _load_project_context(self) -> Optional[str]:
        try:
            path = PROJECT_CONTEXT_PATH
            if not path.is_file():
                return None
            content = path.read_text(encoding="utf-8")
        except Exception:
            logger.exception("Failed to read project context from %s", PROJECT_CONTEXT_PATH)
            return None

        content = (content or "").strip()
        if not content:
            return None
        max_chars = int(os.environ.get("MAX_PROJECT_CONTEXT_CHARS", "50000"))
        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n[...truncated...]"
        return content

    def _load_prompt_override(self, prompt_id: Optional[str]) -> Optional[str]:
        if not prompt_id:
            return None
        prompt_id = _normalize_identifier(prompt_id, kind="prompt")
        prompt_file = PROMPTS_DIR / f"{prompt_id}.md"
        if not prompt_file.is_file():
            raise ValueError(f"Prompt '{prompt_id}' not found")
        content = prompt_file.read_text(encoding="utf-8")
        _, body = _parse_frontmatter(content)
        body = (body or "").strip()
        if not body:
            raise ValueError(f"Prompt '{prompt_id}' is empty")
        return body

    def _load_project_settings_payload(self, *, include_local: bool = True) -> dict[str, Any]:
        settings_path = WORKSPACE_DIR / ".claude" / "settings.json"
        local_path = WORKSPACE_DIR / ".claude" / "settings.local.json"
        merged: dict[str, Any] = {}

        candidates = [settings_path]
        if include_local:
            candidates.append(local_path)

        for candidate in candidates:
            if not candidate.is_file():
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                logger.exception("Failed to read settings from %s", candidate)
                continue
            if isinstance(payload, dict):
                merged = _merge_settings(merged, payload)

        return merged

    def _load_project_settings(self, *, include_local: bool = True) -> Optional[str]:
        merged = self._load_project_settings_payload(include_local=include_local)
        if not merged:
            return None
        return json.dumps(merged)

    def _write_settings_cache(self, settings_json: Optional[str], *, cache_key: str) -> Optional[str]:
        if not settings_json:
            return None
        target = self._settings_cache_paths.get(cache_key)
        if not target:
            return None
        try:
            target.write_text(settings_json, encoding="utf-8")
        except Exception:
            logger.exception("Failed to write settings cache %s", target)
            return None
        return str(target)

    async def _record_permission_decision(
        self,
        *,
        session_id: str,
        source: str,
        tool: str,
        decision: str,
        reason: str,
        rule: Optional[str],
        input_summary: str,
    ) -> None:
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "session_id": session_id,
            "source": source,
            "tool": tool,
            "decision": decision,
            "reason": reason,
            "rule": rule,
            "input": input_summary,
        }
        data = json.dumps(record)
        session_key = f"{_PERMISSION_LOG_SESSION_KEY_PREFIX}{session_id}"
        try:
            pipe = self.redis.pipeline()
            for key in (_PERMISSION_LOG_GLOBAL_KEY, session_key):
                pipe.lpush(key, data)
                pipe.ltrim(key, 0, max(_PERMISSION_LOG_MAX - 1, 0))
            await pipe.execute()
        except Exception:
            logger.exception("Failed to store permission decision log")

    async def get_permission_log(
        self,
        *,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 100), 500))
        key = (
            f"{_PERMISSION_LOG_SESSION_KEY_PREFIX}{session_id}"
            if session_id
            else _PERMISSION_LOG_GLOBAL_KEY
        )
        try:
            raw_items = await self.redis.lrange(key, 0, limit - 1)
        except Exception:
            logger.exception("Failed to read permission log")
            return []
        entries: list[dict[str, Any]] = []
        for raw in raw_items:
            try:
                if isinstance(raw, (bytes, bytearray)):
                    raw = raw.decode("utf-8")
                entries.append(json.loads(raw))
            except Exception:
                continue
        return entries

    def _build_permission_handler(
        self,
        *,
        user_session_id: str,
        context: Optional[dict],
        permission_mode: Optional[str],
        settings_payload: Optional[dict],
    ):
        permissions = {}
        if isinstance(settings_payload, dict):
            permissions = settings_payload.get("permissions") if isinstance(settings_payload.get("permissions"), dict) else {}
        allow_list = permissions.get("allow") if isinstance(permissions.get("allow"), list) else []
        ask_list = permissions.get("ask") if isinstance(permissions.get("ask"), list) else []
        deny_list = permissions.get("deny") if isinstance(permissions.get("deny"), list) else []
        source = (context or {}).get("source") or "unknown"
        permission_logger = _configure_permission_logger()

        def _allow(updated_input: Optional[dict]) -> PermissionResultAllow:
            result = PermissionResultAllow()
            result.updated_input = updated_input
            return result

        def _deny(message: str) -> PermissionResultDeny:
            return PermissionResultDeny(message=message)

        async def can_use_tool(
            tool: str,
            input_data: dict,
            *_args,
            **_kwargs,
        ) -> PermissionResultAllow | PermissionResultDeny:
            summary = _sanitize_permission_input(_summarize_tool_input(tool, input_data or {}))
            if permission_mode == "bypassPermissions":
                permission_logger.info(
                    "",
                    extra={
                        "session": user_session_id,
                        "tool": tool,
                        "decision": "allow",
                        "reason": "bypassPermissions",
                        "rule": "",
                        "input": summary,
                    },
                )
                await self._record_permission_decision(
                    session_id=user_session_id,
                    source=source,
                    tool=tool,
                    decision="allow",
                    reason="bypassPermissions",
                    rule=None,
                    input_summary=summary,
                )
                return _allow(input_data)

            matched_rule = None
            decision = "deny"
            reason = "no_match"

            for rule in deny_list:
                if _match_rule(tool, input_data or {}, rule):
                    matched_rule = rule
                    decision = "deny"
                    reason = "deny_rule"
                    break

            if matched_rule is None:
                for rule in allow_list:
                    if _match_rule(tool, input_data or {}, rule):
                        matched_rule = rule
                        decision = "allow"
                        reason = "allow_rule"
                        break

            if matched_rule is None:
                for rule in ask_list:
                    if _match_rule(tool, input_data or {}, rule):
                        matched_rule = rule
                        decision = "deny"
                        reason = "ask_rule"
                        break

            permission_logger.info(
                "",
                extra={
                    "session": user_session_id,
                    "tool": tool,
                    "decision": decision,
                    "reason": reason,
                    "rule": matched_rule or "",
                    "input": summary,
                },
            )
            await self._record_permission_decision(
                session_id=user_session_id,
                source=source,
                tool=tool,
                decision=decision,
                reason=reason,
                rule=matched_rule,
                input_summary=summary,
            )
            if decision == "allow":
                return _allow(input_data)
            return _deny(f"Blocked by permissions ({reason})")

        return can_use_tool

    def get_debug_info(
        self,
        *,
        context: Optional[dict] = None,
        commands: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        ctx = context or {}
        debug: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "runtime": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "cwd": os.getcwd(),
            },
        }

        sdk_info: dict[str, Any] = {}
        try:
            sdk_info["claude_agent_sdk_version"] = importlib_metadata.version("claude-agent-sdk")
        except importlib_metadata.PackageNotFoundError:
            sdk_info["claude_agent_sdk_version"] = None
            sdk_info["claude_agent_sdk_error"] = "package not found"
        except Exception as exc:
            sdk_info["claude_agent_sdk_version"] = None
            sdk_info["claude_agent_sdk_error"] = str(exc)

        claude_cli_path = shutil.which("claude")
        claude_cli: dict[str, Any] = {"path": claude_cli_path}
        if claude_cli_path:
            try:
                result = subprocess.run(
                    [claude_cli_path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                claude_cli["returncode"] = result.returncode
                claude_cli["stdout"] = result.stdout.strip()
                claude_cli["stderr"] = result.stderr.strip()
            except Exception as exc:
                claude_cli["error"] = str(exc)

        debug["runtime"]["sdk"] = sdk_info
        debug["runtime"]["claude_cli"] = claude_cli

        debug["paths"] = {
            "workspace_dir": _path_debug_info(WORKSPACE_DIR),
            "claude_dir": _path_debug_info(WORKSPACE_DIR / ".claude"),
            "claude_home": _path_debug_info(Path.home() / ".claude"),
            "workspace_claude_home": _path_debug_info(WORKSPACE_DIR / ".claude-home"),
            "skills_dir": _path_debug_info(SKILLS_DIR),
            "commands_dir": _path_debug_info(COMMANDS_DIR),
            "prompts_dir": _path_debug_info(PROMPTS_DIR),
            "project_context": _path_debug_info(PROJECT_CONTEXT_PATH),
            "artifacts_dir": _path_debug_info(WORKSPACE_DIR / "artifacts"),
        }

        settings_path = WORKSPACE_DIR / ".claude" / "settings.json"
        local_path = WORKSPACE_DIR / ".claude" / "settings.local.json"
        user_path = Path.home() / ".claude" / "settings.json"
        user_state_path = Path.home() / ".claude.json"
        managed_paths = [
            Path("/etc/claude-code/managed-settings.json"),
            Path("/Library/Application Support/ClaudeCode/managed-settings.json"),
            Path("C:/Program Files/ClaudeCode/managed-settings.json"),
        ]

        settings_info: dict[str, Any] = {
            "setting_sources": list(_DEFAULT_SETTING_SOURCES),
            "project": _settings_file_debug(settings_path),
            "local": _settings_file_debug(local_path),
            "user": _settings_file_debug(user_path),
            "user_state": _settings_file_debug(user_state_path, redactor=_redact_sensitive_payload),
            "managed": [_settings_file_debug(path) for path in managed_paths],
            "settings_cache_paths": {key: str(path) for key, path in self._settings_cache_paths.items()},
        }
        settings_info["settings_cache"] = {
            key: _settings_file_debug(path) for key, path in self._settings_cache_paths.items()
        }

        merged_settings_raw = self._load_project_settings()
        merged_settings_payload: Optional[dict[str, Any]] = None
        if merged_settings_raw:
            try:
                merged_settings_payload = json.loads(merged_settings_raw)
                settings_info["merged_project_local"] = _redact_settings_payload(merged_settings_payload)
            except Exception as exc:
                settings_info["merged_project_local_error"] = str(exc)
        else:
            settings_info["merged_project_local"] = None

        project_only_raw = self._load_project_settings(include_local=False)
        if project_only_raw:
            try:
                project_only_payload = json.loads(project_only_raw)
                settings_info["project_only"] = _redact_settings_payload(project_only_payload)
            except Exception as exc:
                settings_info["project_only_error"] = str(exc)
        else:
            settings_info["project_only"] = None

        permissions_summary: dict[str, Any] = {}
        if isinstance(merged_settings_payload, dict):
            permissions = merged_settings_payload.get("permissions")
            if isinstance(permissions, dict):
                allow_list = permissions.get("allow") if isinstance(permissions.get("allow"), list) else []
                ask_list = permissions.get("ask") if isinstance(permissions.get("ask"), list) else []
                deny_list = permissions.get("deny") if isinstance(permissions.get("deny"), list) else []
                permissions_summary = {
                    "defaultMode": permissions.get("defaultMode"),
                    "disableBypassPermissionsMode": permissions.get("disableBypassPermissionsMode"),
                    "allow_count": len(allow_list),
                    "ask_count": len(ask_list),
                    "deny_count": len(deny_list),
                }

        settings_info["permissions_summary"] = permissions_summary
        debug["settings"] = settings_info

        permission_mode_input = ctx.get("permission_mode")
        resolved_permission_mode: Optional[str] = None
        permission_mode_error: Optional[str] = None
        try:
            resolved_permission_mode = self._resolve_permission_mode(ctx)
        except Exception as exc:
            permission_mode_error = str(exc)

        permissions_debug = {
            "context_permission_mode": permission_mode_input,
            "context_source": ctx.get("source"),
            "resolved_permission_mode": resolved_permission_mode,
            "resolve_error": permission_mode_error,
            "allow_bypass_permissions_env": os.environ.get("ALLOW_BYPASS_PERMISSIONS"),
            "allow_bypass_permissions_enabled": os.environ.get("ALLOW_BYPASS_PERMISSIONS", "0") == "1",
            "settings_default_mode": permissions_summary.get("defaultMode"),
        }

        options_preview: dict[str, Any] = {
            "cwd": str(WORKSPACE_DIR),
            "permission_mode": resolved_permission_mode,
            "setting_sources": list(_DEFAULT_SETTING_SOURCES),
        }

        if ctx.get("source") == "webhook":
            webhook_settings_raw = self._build_webhook_settings()
            try:
                webhook_payload = json.loads(webhook_settings_raw)
                options_preview["settings_override"] = _redact_settings_payload(webhook_payload)
            except Exception as exc:
                options_preview["settings_override_error"] = str(exc)
        else:
            options_preview["settings_override"] = settings_info.get("project_only")

        debug["permissions"] = permissions_debug
        debug["options_preview"] = options_preview

        env_summary = {
            "ALLOW_BYPASS_PERMISSIONS": os.environ.get("ALLOW_BYPASS_PERMISSIONS"),
            "WORKSPACE_DIR": os.environ.get("WORKSPACE_DIR"),
            "SKILLS_DIR": os.environ.get("SKILLS_DIR"),
            "COMMANDS_DIR": os.environ.get("COMMANDS_DIR"),
            "PROMPTS_DIR": os.environ.get("PROMPTS_DIR"),
            "PROJECT_CONTEXT_PATH": os.environ.get("PROJECT_CONTEXT_PATH"),
            "ANTHROPIC_BASE_URL": os.environ.get("ANTHROPIC_BASE_URL"),
            "API_KEY_SET": bool(os.environ.get("API_KEY")),
            "OPENAI_API_KEY_SET": bool(os.environ.get("OPENAI_API_KEY")),
            "OPENROUTER_API_KEY_SET": bool(os.environ.get("OPENROUTER_API_KEY")),
            "ANTHROPIC_AUTH_TOKEN_SET": bool(os.environ.get("ANTHROPIC_AUTH_TOKEN")),
            "ANTHROPIC_API_KEY_SET": bool(os.environ.get("ANTHROPIC_API_KEY")),
        }
        debug["env_summary"] = env_summary
        debug["permission_logging"] = {
            "enabled": _PERMISSION_LOGGING_ENABLED,
            "logger_handlers": len(_permission_logger.handlers),
            "logger_level": _permission_logger.level,
            "log_max": _PERMISSION_LOG_MAX,
        }

        if commands and isinstance(merged_settings_payload, dict):
            permissions = merged_settings_payload.get("permissions")
            if isinstance(permissions, dict):
                allow_list = permissions.get("allow") if isinstance(permissions.get("allow"), list) else []
                ask_list = permissions.get("ask") if isinstance(permissions.get("ask"), list) else []
                deny_list = permissions.get("deny") if isinstance(permissions.get("deny"), list) else []
                previews: list[dict[str, Any]] = []
                for command in commands:
                    if not command:
                        continue
                    previews.append(
                        {
                            "command": command,
                            "allow_prefix_matches": _match_bash_rules(allow_list, command),
                            "ask_prefix_matches": _match_bash_rules(ask_list, command),
                            "deny_prefix_matches": _match_bash_rules(deny_list, command),
                            "note": "Prefix matching is heuristic and only checks the portion of Bash(...) before the first '*'.",
                        }
                    )
                debug["command_rule_preview"] = previews

        warnings: list[str] = []
        if permission_mode_input and permission_mode_input != "default":
            warnings.append("Context permission_mode overrides permissions.defaultMode in settings.")
        if permission_mode_input == "bypassPermissions" and os.environ.get("ALLOW_BYPASS_PERMISSIONS", "0") != "1":
            warnings.append("permission_mode=bypassPermissions requested but ALLOW_BYPASS_PERMISSIONS is not enabled.")
        if "project" not in _DEFAULT_SETTING_SOURCES and settings_info["project"].get("exists"):
            warnings.append("Project settings exist but setting_sources does not include 'project'.")
        if "user" not in _DEFAULT_SETTING_SOURCES and settings_info["user"].get("exists"):
            warnings.append("User settings exist but setting_sources excludes 'user'.")
        if settings_info["project"].get("parse_ok") is False:
            warnings.append("Project settings.json failed to parse.")
        if settings_info["local"].get("parse_ok") is False:
            warnings.append("Local settings.local.json failed to parse.")
        if permissions_summary.get("defaultMode") and permission_mode_input:
            warnings.append("permissions.defaultMode is set but a context permission_mode was also supplied.")
        debug["warnings"] = warnings

        return debug

    async def _register_active_stream(self, user_session_id: str, client: ClaudeSDKClient) -> None:
        async with self._active_streams_lock:
            self._active_streams[user_session_id] = client

    async def _unregister_active_stream(self, user_session_id: str, client: ClaudeSDKClient) -> None:
        async with self._active_streams_lock:
            if self._active_streams.get(user_session_id) is client:
                self._active_streams.pop(user_session_id, None)

    async def interrupt_stream(self, user_session_id: str) -> bool:
        async with self._active_streams_lock:
            client = self._active_streams.get(user_session_id)
        if not client:
            return False
        try:
            await client.interrupt()
        except Exception:
            logger.exception("Failed to interrupt session %s", user_session_id)
            return False
        try:
            await self._mark_interrupted(user_session_id)
        except Exception:
            logger.exception("Failed to record interrupt for session %s", user_session_id)
        return True

    async def _get_stored_session(self, user_session_id: str) -> Optional[dict]:
        data = await self.redis.get(f"session:{user_session_id}")
        if data:
            return json.loads(data)
        return None

    async def _mark_interrupted(self, user_session_id: str) -> None:
        await self.redis.set(f"interrupt:{user_session_id}", "1", ex=_INTERRUPT_TTL_S)

    async def _consume_interrupt_note(self, user_session_id: str) -> Optional[str]:
        key = f"interrupt:{user_session_id}"
        data = await self.redis.get(key)
        if not data:
            return None
        await self.redis.delete(key)
        return "Note: The previous response was interrupted by the user."
    
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

    def _compose_user_text(
        self,
        message: str,
        *,
        context: Optional[dict],
        is_slash_command: bool,
    ) -> str:
        if context and not is_slash_command:
            source = context.get("source", "unknown")
            user_name = context.get("user_name", "User")
            return f"[Context: {user_name} via {source}]\n\n{message}"
        return message

    def _build_message_content(self, text_content: str, images: Optional[list[dict]]) -> Any:
        if not images:
            return text_content
        content: Any = [{"type": "text", "text": text_content}]
        for img in images:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img.get("media_type", "image/jpeg"),
                        "data": img["data"],
                    },
                }
            )
        return content

    def _build_prompt_factory(
        self,
        *,
        text_content: str,
        content: Any,
        images: Optional[list[dict]],
    ):
        if not images:
            return lambda: text_content

        async def message_generator():
            yield {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": content,
                },
            }

        return lambda: message_generator()

    def _prepare_settings(
        self,
        context: Optional[dict],
    ) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        settings_json: Optional[str] = None
        settings_path: Optional[str] = None
        settings_payload: Optional[dict[str, Any]] = None

        if (context or {}).get("source") == "webhook":
            settings_json = self._build_webhook_settings()
            try:
                settings_payload = json.loads(settings_json)
            except Exception:
                logger.exception("Failed to parse webhook settings override")
            settings_path = self._write_settings_cache(settings_json, cache_key="webhook")
        else:
            settings_payload = self._load_project_settings_payload(include_local=False)
            if settings_payload:
                settings_json = json.dumps(settings_payload)
            settings_path = self._write_settings_cache(settings_json, cache_key="default")

        return settings_payload, settings_path

    async def _build_system_prompt(
        self,
        *,
        user_session_id: str,
        context: Optional[dict],
    ) -> SystemPromptPreset:
        interrupt_note = await self._consume_interrupt_note(user_session_id)
        prompt_override = self._load_prompt_override((context or {}).get("prompt_id"))
        project_context = prompt_override or self._load_project_context()
        append_parts = [part for part in (project_context, interrupt_note) if part]
        append_text = "\n\n".join(append_parts) if append_parts else None
        if append_text:
            return {
                "type": "preset",
                "preset": "claude_code",
                "append": append_text,
            }
        return {"type": "preset", "preset": "claude_code"}

    def _build_agent_options(
        self,
        *,
        user_session_id: str,
        context: Optional[dict],
        permission_mode: Optional[str],
        settings_payload: Optional[dict[str, Any]],
        settings_path: Optional[str],
        model: Optional[str],
        resume_session_id: Optional[str],
        system_prompt: SystemPromptPreset,
    ) -> ClaudeAgentOptions:
        permission_handler = self._build_permission_handler(
            user_session_id=user_session_id,
            context=context,
            permission_mode=permission_mode,
            settings_payload=settings_payload,
        )
        return ClaudeAgentOptions(
            permission_mode=permission_mode,
            cwd=str(WORKSPACE_DIR),
            model=(model or None),
            resume=resume_session_id,
            settings=settings_path,
            system_prompt=system_prompt,
            setting_sources=list(_DEFAULT_SETTING_SOURCES),
            can_use_tool=permission_handler,
        )

    async def _record_conversation(
        self,
        *,
        user_session_id: str,
        raw_message: str,
        user_message: str,
        response_text: str,
        claude_session_id: Optional[str],
    ) -> list[dict]:
        history = await self._get_conversation_history(user_session_id)
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": response_text})
        await self._store_conversation_history(user_session_id, history)

        if raw_message.startswith("/clear"):
            await self.redis.delete(f"history:{user_session_id}")

        await self._store_session(user_session_id, claude_session_id=claude_session_id)
        await self._update_session_activity(user_session_id)
        return history
    
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
        text_content = self._compose_user_text(
            message,
            context=context,
            is_slash_command=is_slash_command,
        )
        
        # Build message content - either string or list with images.
        content = self._build_message_content(text_content, images)
        
        tools_used = []
        response_parts = []

        # Preserve Claude Code session for interactive chat, but avoid resuming for webhook calls
        # (webhooks are typically stateless and should always pick up latest volume commands/cwd).
        resume_session_id: Optional[str] = None
        if (context or {}).get("source") != "webhook":
            resume_session_id = (stored or {}).get("claude_session_id")

        # Resolve explicit permission override; default None means use settings.json defaultMode.
        permission_mode = self._resolve_permission_mode(context)

        # Webhook runs are non-interactive; ensure required permission rules are present so we
        # don't hang on approval prompts (e.g., command helpers like save_transcript.py).
        settings_payload, settings_path = self._prepare_settings(context)
        
        system_prompt = await self._build_system_prompt(
            user_session_id=user_session_id,
            context=context,
        )
        options = self._build_agent_options(
            user_session_id=user_session_id,
            context=context,
            permission_mode=permission_mode,
            settings_payload=settings_payload,
            settings_path=settings_path,
            model=model,
            resume_session_id=resume_session_id,
            system_prompt=system_prompt,
        )
        
        # query() enables Claude Code preprocessing for slash commands and !` bash execution.
        prompt_factory = self._build_prompt_factory(
            text_content=text_content,
            content=content,
            images=images,
        )
        prompt: str | Any = prompt_factory()

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
        history = await self._record_conversation(
            user_session_id=user_session_id,
            raw_message=raw_message,
            user_message=message,
            response_text=response_text,
            claude_session_id=claude_session_id,
        )
        
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
        text_content = self._compose_user_text(
            message,
            context=context,
            is_slash_command=is_slash_command,
        )
        
        # Build message content
        content = self._build_message_content(text_content, images)
        
        tools_used = []
        response_parts = []

        # Preserve Claude Code session for interactive chat, but avoid resuming for webhook calls
        # (webhooks are typically stateless and should always pick up latest volume commands/cwd).
        resume_session_id: Optional[str] = None
        if (context or {}).get("source") != "webhook":
            resume_session_id = (stored or {}).get("claude_session_id")
        
        # Resolve explicit permission override; default None means use settings.json defaultMode.
        permission_mode = self._resolve_permission_mode(context)

        settings_payload, settings_path = self._prepare_settings(context)
        
        system_prompt = await self._build_system_prompt(
            user_session_id=user_session_id,
            context=context,
        )
        options = self._build_agent_options(
            user_session_id=user_session_id,
            context=context,
            permission_mode=permission_mode,
            settings_payload=settings_payload,
            settings_path=settings_path,
            model=model,
            resume_session_id=resume_session_id,
            system_prompt=system_prompt,
        )
        
        # Signal that we're starting
        yield {"type": "status", "status": "connecting"}

        yield {"type": "status", "status": "sending"}
        yield {"type": "status", "status": "processing"}

        # query() enables Claude Code preprocessing for slash commands and !` bash execution.
        prompt_factory = self._build_prompt_factory(
            text_content=text_content,
            content=content,
            images=images,
        )

        stream_session_id = resume_session_id or user_session_id

        async def run_stream(current_options: ClaudeAgentOptions):
            nonlocal claude_session_id, usage, session_stored

            stderr_buf = io.StringIO()
            opts = dataclasses.replace(
                current_options,
                debug_stderr=stderr_buf,
                model=(current_options.model or None),
            )
            emitted_any_output = False
            client = ClaudeSDKClient(opts)
            try:
                await client.connect()
                await self._register_active_stream(user_session_id, client)
                await client.query(prompt=prompt_factory(), session_id=stream_session_id)
                async for msg in client.receive_response():
                    if isinstance(msg, SystemMessage):
                        if msg.subtype == "init":
                            claude_session_id = msg.data.get("session_id") or claude_session_id
                            if claude_session_id and not session_stored:
                                await self._store_session(user_session_id, claude_session_id=claude_session_id)
                                await self._update_session_activity(user_session_id)
                                session_stored = True
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
            except asyncio.CancelledError:
                raise
            except Exception as e:
                raise _StreamError(
                    stderr_text=stderr_buf.getvalue(),
                    exc=e,
                    emitted_any_output=emitted_any_output,
                ) from e
            finally:
                await self._unregister_active_stream(user_session_id, client)
                await client.disconnect()

        claude_session_id: Optional[str] = None
        usage: dict[str, Any] = {}
        session_stored = False

        try:
            async for ev in run_stream(options):
                yield ev
        except _StreamError as e:
            if options.model and not e.emitted_any_output:
                fallback_options = dataclasses.replace(options, model=None)
                async for ev in run_stream(fallback_options):
                    yield ev
            else:
                if e.stderr_text:
                    raise RuntimeError(e.stderr_text) from e
                raise
        
        response_text = "".join(response_parts)
        
        # Update server-side metadata (and keep a lightweight transcript for UI/debugging).
        history = await self._record_conversation(
            user_session_id=user_session_id,
            raw_message=raw_message,
            user_message=message,
            response_text=response_text,
            claude_session_id=claude_session_id,
        )
        
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
            requested = Path(file_path or "").as_posix().lstrip("./")
        except Exception:
            requested = file_path or ""
        if requested == ".claude/CLAUDE.md":
            self._ensure_project_context_file()
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

    def move_workspace_item(self, src_path: str, dst_path: str, *, overwrite: bool = False) -> dict:
        """Move or rename a file/directory within the workspace."""
        try:
            src_full = _resolve_under(WORKSPACE_DIR, src_path)
            dst_full = _resolve_under(WORKSPACE_DIR, dst_path)
        except ValueError:
            raise ValueError("Path must stay within workspace")

        if not src_full.exists():
            raise FileNotFoundError("Source not found")

        if src_full.resolve() == dst_full.resolve():
            return {
                "from": str(src_full.relative_to(WORKSPACE_DIR)),
                "to": str(dst_full.relative_to(WORKSPACE_DIR)),
                "moved": False,
            }

        if src_full.is_dir():
            try:
                dst_full.resolve().relative_to(src_full.resolve())
            except ValueError:
                pass
            else:
                raise ValueError("Cannot move a directory into itself")

        if dst_full.exists():
            if not overwrite:
                raise FileExistsError("Destination already exists")
            if dst_full.is_dir():
                shutil.rmtree(dst_full)
            else:
                dst_full.unlink()

        dst_full.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_full), str(dst_full))

        return {
            "from": str(src_full.relative_to(WORKSPACE_DIR)),
            "to": str(dst_full.relative_to(WORKSPACE_DIR)),
            "moved": True,
        }

    # Prompt management methods
    def list_prompts(self) -> list[dict]:
        """List all available prompt overrides."""
        prompts = []
        if PROMPTS_DIR.exists():
            for prompt_file in sorted(PROMPTS_DIR.glob("*.md")):
                try:
                    prompt_id = _normalize_identifier(prompt_file.stem, kind="prompt")
                except ValueError:
                    continue
                if prompt_id != prompt_file.stem:
                    continue
                try:
                    content = prompt_file.read_text(encoding="utf-8")
                except Exception:
                    logger.exception("Failed to read prompt %s", prompt_file)
                    continue
                metadata, _ = _parse_frontmatter(content)
                prompts.append({
                    "id": prompt_id,
                    "name": metadata.get("name") or prompt_id,
                    "description": metadata.get("description") or "",
                })
        return prompts

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
        for client in list(self._active_streams.values()):
            try:
                await client.disconnect()
            except Exception:
                logger.exception("Failed to disconnect active stream client")
        self._active_streams.clear()
        await self.redis.close()
