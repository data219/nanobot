"""Session management for conversation history."""

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.config.paths import get_legacy_sessions_dir
from nanobot.utils.helpers import ensure_dir, safe_filename


@dataclass
class Session:
    """
    A conversation session.

    Stores messages in JSONL format for easy reading and persistence.

    Important: Messages are append-only for LLM cache efficiency.
    The consolidation process writes summaries to MEMORY.md/HISTORY.md
    but does NOT modify the messages list or get_history() output.
    """

    key: str  # channel:chat_id
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_consolidated: int = 0  # Number of messages already consolidated to files

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the session."""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.messages.append(msg)
        self.updated_at = datetime.now()

    @staticmethod
    def _find_legal_start(messages: list[dict[str, Any]]) -> int:
        """Find first index where every tool result has a matching assistant tool_call."""
        declared: set[str] = set()
        start = 0
        for i, msg in enumerate(messages):
            role = msg.get("role")
            if role == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict) and tc.get("id"):
                        declared.add(str(tc["id"]))
            elif role == "tool":
                tid = msg.get("tool_call_id")
                if tid and str(tid) not in declared:
                    start = i + 1
                    declared.clear()
                    for prev in messages[start:i + 1]:
                        if prev.get("role") == "assistant":
                            for tc in prev.get("tool_calls") or []:
                                if isinstance(tc, dict) and tc.get("id"):
                                    declared.add(str(tc["id"]))
        return start

    def get_history(self, max_messages: int = 500) -> list[dict[str, Any]]:
        """Return unconsolidated messages for LLM input, aligned to a legal tool-call boundary."""
        unconsolidated = self.messages[self.last_consolidated:]
        sliced = unconsolidated[-max_messages:]

        # Drop leading non-user messages to avoid starting mid-turn when possible.
        for i, message in enumerate(sliced):
            if message.get("role") == "user":
                sliced = sliced[i:]
                break

        # Some providers reject orphan tool results if the matching assistant
        # tool_calls message fell outside the fixed-size history window.
        start = self._find_legal_start(sliced)
        if start:
            sliced = sliced[start:]

        out: list[dict[str, Any]] = []
        for message in sliced:
            entry: dict[str, Any] = {"role": message["role"], "content": message.get("content", "")}
            for key in ("tool_calls", "tool_call_id", "name"):
                if key in message:
                    entry[key] = message[key]
            out.append(entry)
        return out

    def clear(self) -> None:
        """Clear all messages and reset session to initial state."""
        self.messages = []
        self.last_consolidated = 0
        self.updated_at = datetime.now()

    def retain_recent_legal_suffix(self, max_messages: int) -> None:
        """Keep a legal recent suffix, mirroring get_history boundary rules."""
        if max_messages <= 0:
            self.clear()
            return
        if len(self.messages) <= max_messages:
            return

        start_idx = max(0, len(self.messages) - max_messages)

        # If the cutoff lands mid-turn, extend backward to the nearest user turn.
        while start_idx > 0 and self.messages[start_idx].get("role") != "user":
            start_idx -= 1

        retained = self.messages[start_idx:]

        # Mirror get_history(): avoid persisting orphan tool results at the front.
        start = self._find_legal_start(retained)
        if start:
            retained = retained[start:]

        dropped = len(self.messages) - len(retained)
        self.messages = retained
        self.last_consolidated = max(0, self.last_consolidated - dropped)
        self.updated_at = datetime.now()


class SessionManager:
    """
    Manages conversation sessions.

    Sessions are stored as JSONL files in the sessions directory.
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.sessions_dir = ensure_dir(self.workspace / "sessions")
        self.legacy_sessions_dir = get_legacy_sessions_dir()
        self._cache: dict[str, Session] = {}

    def _get_session_path(self, key: str) -> Path:
        """Get the file path for a session."""
        safe_key = safe_filename(key.replace(":", "_"))
        return self.sessions_dir / f"{safe_key}.jsonl"

    def _get_legacy_session_path(self, key: str) -> Path:
        """Legacy global session path (~/.nanobot/sessions/)."""
        safe_key = safe_filename(key.replace(":", "_"))
        return self.legacy_sessions_dir / f"{safe_key}.jsonl"

    def get_or_create(self, key: str) -> Session:
        """
        Get an existing session or create a new one.

        Args:
            key: Session key (usually channel:chat_id).

        Returns:
            The session.
        """
        if key in self._cache:
            return self._cache[key]

        session = self._load(key)
        if session is None:
            session = Session(key=key)

        self._cache[key] = session
        return session

    def _load(self, key: str) -> Session | None:
        """Load a session from disk, recovering partial data from corrupt files."""
        path = self._get_session_path(key)
        if not path.exists():
            legacy_path = self._get_legacy_session_path(key)
            if legacy_path.exists():
                try:
                    shutil.move(str(legacy_path), str(path))
                    logger.info("Migrated session {} from legacy path", key)
                except Exception:
                    logger.exception("Failed to migrate session {}", key)

        if not path.exists():
            return None

        try:
            messages: list[dict[str, Any]] = []
            metadata: dict[str, Any] = {}
            created_at: datetime | None = None
            last_consolidated: int = 0
            last_consolidated_untrustworthy = False
            metadata_parsed = False
            recovered = False
            skipped_count = 0
            total_lines = 0

            # Position-aware index-shift tracking:
            # If a corrupt line is skipped BEFORE the known consolidation boundary,
            # subsequent messages shift to lower indices, making the boundary unreliable.
            # Known limitation: if metadata appears after messages (non-standard format
            # not produced by save()), pre-metadata skips won't be caught here.
            # The fallback via `not metadata_parsed` still covers that extreme case.
            msg_index = 0
            skipped_before_boundary = False

            with open(path, encoding="utf-8-sig") as f:
                for line_num, raw in enumerate(f, 1):
                    stripped = raw.strip()
                    if not stripped:
                        continue
                    total_lines += 1

                    try:
                        data = json.loads(stripped)
                    except (json.JSONDecodeError, RecursionError, MemoryError):
                        logger.warning(
                            "Corrupt line {} in session {} at {} — skipping",
                            line_num, key, path,
                        )
                        skipped_count += 1
                        # Check if this skip is before the consolidation boundary.
                        # Only corrupt MESSAGE lines (parse failures) can cause index shifts.
                        # Non-dict values are not messages and don't occupy message slots.
                        if metadata_parsed and msg_index < last_consolidated:
                            skipped_before_boundary = True
                        continue

                    if not isinstance(data, dict):
                        logger.warning(
                            "Non-dict JSON on line {} in session {} at {} — skipping",
                            line_num, key, path,
                        )
                        skipped_count += 1
                        # Non-dict values were never messages — no index shift.
                        # Do NOT set skipped_before_boundary.
                        continue

                    if data.get("_type") == "metadata":
                        # Parse metadata fields individually with safe defaults
                        raw_meta = data.get("metadata", {})
                        if not isinstance(raw_meta, dict):
                            logger.warning("Non-dict metadata in session {} at {}", key, path)
                            raw_meta = {}
                        metadata = raw_meta

                        try:
                            created_at = (
                                datetime.fromisoformat(data["created_at"])
                                if data.get("created_at")
                                else None
                            )
                        except (ValueError, TypeError):
                            logger.warning("Invalid created_at in session {} at {}", key, path)

                        if "last_consolidated" not in data:
                            last_consolidated_untrustworthy = True
                        else:
                            try:
                                last_consolidated = int(data["last_consolidated"])
                            except (ValueError, TypeError, OverflowError):
                                logger.warning(
                                    "Invalid last_consolidated in session {} at {}, falling back to len(messages)",
                                    key, path,
                                )
                                last_consolidated_untrustworthy = True

                        metadata_parsed = True
                        recovered = True
                    else:
                        messages.append(data)
                        msg_index += 1
                        recovered = True

            if not recovered:
                return None

            # Consolidation safety: when last_consolidated is untrustworthy,
            # metadata line missing, or a corrupt line was skipped before the
            # consolidation boundary (index-shift protection), assume all loaded
            # messages are already consolidated.
            # No `and messages` guard — when messages is empty, len(messages)=0
            # is the correct fallback (prevents high lc from making new messages invisible).
            if last_consolidated_untrustworthy or not metadata_parsed or skipped_before_boundary:
                last_consolidated = len(messages)
                logger.warning(
                    "Consolidation boundary uncertain in session {} (skipped={}, "
                    "untrusted={}, no_metadata={}) — assuming all {} loaded messages "
                    "are consolidated",
                    key, skipped_count, last_consolidated_untrustworthy,
                    not metadata_parsed, len(messages),
                )

            # Lower-bound clamping for negative values.
            # Upper-bound clamp intentionally omitted: all consumers (get_history(),
            # pick_consolidation_boundary, retain_recent_legal_suffix) handle
            # last_consolidated > len(messages) correctly via Python slice semantics.
            if last_consolidated < 0:
                logger.warning(
                    "Negative last_consolidated ({}) in session {} — clamping to 0",
                    last_consolidated, key,
                )
                last_consolidated = 0

            if skipped_count > 0:
                logger.info(
                    "Session {} partially recovered: {}/{} lines loaded, {} skipped (excludes duplicate metadata)",
                    key, len(messages) + (1 if metadata_parsed else 0), total_lines, skipped_count,
                )

            return Session(
                key=key,
                messages=messages,
                created_at=created_at or datetime.now(),
                metadata=metadata,
                last_consolidated=last_consolidated,
            )
        except (OSError, UnicodeDecodeError) as e:
            # Outer catch: I/O failures and encoding errors that prevent
            # reading the file at all. All parse errors are handled per-line above.
            logger.warning("Failed to load session {} at {}: {}", key, path, e)
            return None

    def save(self, session: Session) -> None:
        """Save a session to disk."""
        path = self._get_session_path(session.key)

        with open(path, "w", encoding="utf-8") as f:
            metadata_line = {
                "_type": "metadata",
                "key": session.key,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "metadata": session.metadata,
                "last_consolidated": session.last_consolidated
            }
            f.write(json.dumps(metadata_line, ensure_ascii=False) + "\n")
            for msg in session.messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")

        self._cache[session.key] = session

    def invalidate(self, key: str) -> None:
        """Remove a session from the in-memory cache."""
        self._cache.pop(key, None)

    def list_sessions(self) -> list[dict[str, Any]]:
        """
        List all sessions.

        Returns:
            List of session info dicts.
        """
        sessions = []

        for path in self.sessions_dir.glob("*.jsonl"):
            try:
                # Read just the metadata line
                with open(path, encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        data = json.loads(first_line)
                        if data.get("_type") == "metadata":
                            key = data.get("key") or path.stem.replace("_", ":", 1)
                            sessions.append({
                                "key": key,
                                "created_at": data.get("created_at"),
                                "updated_at": data.get("updated_at"),
                                "path": str(path)
                            })
            except Exception:
                continue

        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)
