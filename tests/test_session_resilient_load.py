"""Tests for resilient session loading from corrupt JSONL files."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from nanobot.session.manager import Session, SessionManager


@pytest.fixture
def tmp_session_manager(tmp_path: Path) -> SessionManager:
    return SessionManager(workspace=tmp_path)


def _write_session_file(path: Path, lines: list[str]) -> None:
    """Helper: write raw lines to a session JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_session_file_bytes(path: Path, data: bytes) -> None:
    """Helper: write raw bytes to a session JSONL file (for encoding tests)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _make_metadata_line(**overrides) -> str:
    """Helper: build a metadata JSONL line."""
    data = {
        "_type": "metadata",
        "key": "telegram:12345",
        "created_at": "2026-03-23T10:00:00",
        "updated_at": "2026-03-23T22:00:00",
        "metadata": {},
        "last_consolidated": 0,
    }
    data.update(overrides)
    return json.dumps(data, ensure_ascii=False)


def _make_message_line(role: str, content: str, **kwargs) -> str:
    """Helper: build a message JSONL line."""
    data = {"role": role, "content": content, "timestamp": "2026-03-23T12:00:00", **kwargs}
    return json.dumps(data, ensure_ascii=False)


class TestCorruptMessageLines:
    def test_load_truncated_last_line(self, tmp_session_manager: SessionManager):
        """Truncated JSON on last line: all previous messages should load."""
        path = tmp_session_manager._get_session_path("telegram:12345")
        lines = [
            _make_metadata_line(),
            _make_message_line("user", "hello"),
            _make_message_line("assistant", "hi there"),
            '{"role": "assistant", "content": "I was writ',
        ]
        _write_session_file(path, lines)

        session = tmp_session_manager._load("telegram:12345")

        assert session is not None
        assert len(session.messages) == 2
        assert session.messages[0]["content"] == "hello"
        assert session.messages[1]["content"] == "hi there"
        # skipped_count > 0 but skip is AFTER boundary (lc=0) → no fallback
        assert session.last_consolidated == 0

    def test_load_corrupt_middle_line(self, tmp_session_manager: SessionManager):
        """Invalid JSON in middle: that line skipped, before and after load."""
        path = tmp_session_manager._get_session_path("telegram:12345")
        lines = [
            _make_metadata_line(),
            _make_message_line("user", "before"),
            "THIS IS NOT JSON {{{",
            _make_message_line("user", "after"),
            _make_message_line("assistant", "response"),
        ]
        _write_session_file(path, lines)

        session = tmp_session_manager._load("telegram:12345")

        assert session is not None
        assert len(session.messages) == 3
        assert session.messages[0]["content"] == "before"
        assert session.messages[1]["content"] == "after"
        assert session.messages[2]["content"] == "response"
        # skip at msg_index=1, lc=0 → not before boundary → no fallback
        assert session.last_consolidated == 0

    def test_load_all_lines_corrupt_returns_none(self, tmp_session_manager: SessionManager):
        """All lines invalid: returns None (fresh session)."""
        path = tmp_session_manager._get_session_path("telegram:12345")
        lines = ["NOT JSON", "ALSO NOT {{{", "STILL BAD"]
        _write_session_file(path, lines)

        session = tmp_session_manager._load("telegram:12345")

        assert session is None

    def test_load_completely_empty_file_returns_none(self, tmp_session_manager: SessionManager):
        """Empty file: returns None."""
        path = tmp_session_manager._get_session_path("telegram:12345")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

        session = tmp_session_manager._load("telegram:12345")

        assert session is None

    def test_load_non_dict_line_skipped(self, tmp_session_manager: SessionManager):
        """Non-dict JSON value (e.g. a string) is skipped — does NOT cause index-shift."""
        path = tmp_session_manager._get_session_path("telegram:12345")
        lines = [
            _make_metadata_line(),
            _make_message_line("user", "before"),
            '"just a string value"',
            _make_message_line("user", "after"),
        ]
        _write_session_file(path, lines)

        session = tmp_session_manager._load("telegram:12345")

        assert session is not None
        assert len(session.messages) == 2
        assert session.messages[0]["content"] == "before"
        assert session.messages[1]["content"] == "after"
        # Non-dict skips do NOT trigger skipped_before_boundary — no fallback
        assert session.last_consolidated == 0

    def test_load_bom_file(self, tmp_session_manager: SessionManager):
        """File with UTF-8 BOM is parsed correctly."""
        path = tmp_session_manager._get_session_path("telegram:12345")
        content = (
            "\ufeff"
            + "\n".join(
                [
                    _make_metadata_line(),
                    _make_message_line("user", "hello"),
                ]
            )
            + "\n"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

        session = tmp_session_manager._load("telegram:12345")

        assert session is not None
        assert len(session.messages) == 1
        assert session.messages[0]["content"] == "hello"

    def test_load_recursion_error_line_skipped(self, tmp_session_manager: SessionManager):
        """Deeply nested JSON triggers RecursionError — line is skipped."""
        path = tmp_session_manager._get_session_path("telegram:12345")
        deep_json = '{"a":' * 50000 + '"b"' + "}" * 50000
        lines = [
            _make_metadata_line(),
            _make_message_line("user", "before"),
            deep_json,
            _make_message_line("user", "after"),
        ]
        _write_session_file(path, lines)

        session = tmp_session_manager._load("telegram:12345")

        assert session is not None
        assert len(session.messages) == 2
        assert session.messages[0]["content"] == "before"
        assert session.messages[1]["content"] == "after"

    def test_load_unicode_decode_error_mid_file_recovers(self, tmp_session_manager: SessionManager):
        """Truncation mid-multi-byte UTF-8 mid-file: garbled line skipped, surrounding data recovered."""
        path = tmp_session_manager._get_session_path("telegram:12345")
        # Write bytes: valid metadata + valid msg + truncated mid-é (0xc3 without 0xa9) + valid msg
        data = (
            _make_metadata_line().encode("utf-8")
            + b"\n"
            + _make_message_line("user", "before").encode("utf-8")
            + b"\n"
            + b'{"role":"user","content":"caf\xc3'
            + b"\n"
            + _make_message_line("assistant", "after").encode("utf-8")
        )
        _write_session_file_bytes(path, data)

        session = tmp_session_manager._load("telegram:12345")

        assert session is not None
        assert len(session.messages) == 2
        assert session.messages[0]["content"] == "before"
        assert session.messages[1]["content"] == "after"


class TestCorruptMetadata:
    def test_load_metadata_only_returns_session(self, tmp_session_manager: SessionManager):
        """Metadata-only file with empty metadata dict: returns Session (not None)."""
        path = tmp_session_manager._get_session_path("telegram:12345")
        lines = [_make_metadata_line()]
        _write_session_file(path, lines)

        session = tmp_session_manager._load("telegram:12345")

        assert session is not None
        assert session.messages == []
        assert session.metadata == {}
        assert session.last_consolidated == 0

    def test_load_corrupt_metadata_created_at(self, tmp_session_manager: SessionManager):
        """Invalid created_at in metadata: defaults to None, messages still load."""
        path = tmp_session_manager._get_session_path("telegram:12345")
        lines = [
            _make_metadata_line(created_at="not-a-date"),
            _make_message_line("user", "hello"),
        ]
        _write_session_file(path, lines)

        session = tmp_session_manager._load("telegram:12345")

        assert session is not None
        assert len(session.messages) == 1
        assert isinstance(session.created_at, datetime)  # falls back to datetime.now()

    def test_load_corrupt_metadata_last_consolidated(self, tmp_session_manager: SessionManager):
        """Invalid last_consolidated: falls back to len(messages) to prevent re-consolidation."""
        path = tmp_session_manager._get_session_path("telegram:12345")
        lines = [
            _make_metadata_line(last_consolidated="abc"),
            _make_message_line("user", "hello"),
        ]
        _write_session_file(path, lines)

        session = tmp_session_manager._load("telegram:12345")

        assert session is not None
        assert session.last_consolidated == 1  # len(messages)
        assert len(session.messages) == 1

    def test_load_metadata_missing_fields(self, tmp_session_manager: SessionManager):
        """Metadata line with only _type: all fields get defaults, last_consolidated = len(messages)."""
        path = tmp_session_manager._get_session_path("telegram:12345")
        lines = [
            json.dumps({"_type": "metadata"}),
            _make_message_line("user", "hello"),
        ]
        _write_session_file(path, lines)

        session = tmp_session_manager._load("telegram:12345")

        assert session is not None
        assert session.metadata == {}
        assert session.last_consolidated == 1  # len(messages)
        assert len(session.messages) == 1

    def test_load_corrupt_metadata_json_with_valid_messages(
        self, tmp_session_manager: SessionManager
    ):
        """Entire metadata line is invalid JSON, but message lines are valid — recovered."""
        path = tmp_session_manager._get_session_path("telegram:12345")
        lines = [
            "{_type: metadata BROKEN JSON {{{",
            _make_message_line("user", "hello"),
            _make_message_line("assistant", "world"),
        ]
        _write_session_file(path, lines)

        session = tmp_session_manager._load("telegram:12345")

        assert session is not None
        assert len(session.messages) == 2
        assert session.last_consolidated == 2  # len(messages) — no metadata parsed
        assert session.metadata == {}  # default when metadata line not parsed

    def test_load_non_dict_metadata_defaults_to_empty(self, tmp_session_manager: SessionManager):
        """Non-dict metadata field: defaults to empty dict, messages still load."""
        path = tmp_session_manager._get_session_path("telegram:12345")
        raw_metadata = (
            '{"_type":"metadata","key":"telegram:12345",'
            '"created_at":"2026-03-23T10:00:00","updated_at":"2026-03-23T22:00:00",'
            '"metadata":"not a dict","last_consolidated":0}'
        )
        lines = [raw_metadata, _make_message_line("user", "hello")]
        _write_session_file(path, lines)

        session = tmp_session_manager._load("telegram:12345")

        assert session is not None
        assert session.metadata == {}
        assert session.last_consolidated == 0  # valid lc parsed, no fallback
        assert len(session.messages) == 1


class TestLastConsolidatedBounds:
    def test_load_negative_last_consolidated_clamped(self, tmp_session_manager: SessionManager):
        """Negative last_consolidated is clamped to 0."""
        path = tmp_session_manager._get_session_path("telegram:12345")
        lines = [
            _make_metadata_line(last_consolidated=-1),
            _make_message_line("user", "hello"),
        ]
        _write_session_file(path, lines)

        session = tmp_session_manager._load("telegram:12345")

        assert session is not None
        assert session.last_consolidated == 0
        assert len(session.messages) == 1

    def test_load_overflow_last_consolidated_fallback(self, tmp_session_manager: SessionManager):
        """JSON float 1e999 triggers OverflowError on int() → fallback to len(messages)."""
        path = tmp_session_manager._get_session_path("telegram:12345")
        # 1e999 → json.loads → float('inf') → int(float('inf')) → OverflowError
        raw_metadata = (
            '{"_type":"metadata","key":"telegram:12345",'
            '"created_at":"2026-03-23T10:00:00","updated_at":"2026-03-23T22:00:00",'
            '"metadata":{},"last_consolidated":1e999}'
        )
        lines = [raw_metadata, _make_message_line("user", "hello")]
        _write_session_file(path, lines)

        session = tmp_session_manager._load("telegram:12345")

        assert session is not None
        assert session.last_consolidated == 1  # len(messages)
        assert len(session.messages) == 1


class TestIndexShiftProtection:
    def test_load_skipped_line_before_consolidation_boundary(
        self, tmp_session_manager: SessionManager
    ):
        """Corrupt line before last_consolidated boundary triggers index-shift protection."""
        path = tmp_session_manager._get_session_path("telegram:12345")
        lines = [
            _make_metadata_line(last_consolidated=3),
            _make_message_line("user", "msg0"),  # index 0
            "CORRUPT LINE {{{",  # index 1 — skipped, BEFORE boundary (lc=3)
            _make_message_line("user", "msg2"),  # index 2 (shifted)
            _make_message_line("user", "msg3"),  # index 3 (shifted)
            _make_message_line("user", "msg4"),  # index 4 (shifted)
        ]
        _write_session_file(path, lines)

        session = tmp_session_manager._load("telegram:12345")

        assert session is not None
        assert len(session.messages) == 4  # 5 minus 1 corrupt
        # skipped_before_boundary=True → fallback: last_consolidated = len(messages) = 4
        assert session.last_consolidated == 4

    def test_load_skipped_line_after_consolidation_boundary(
        self, tmp_session_manager: SessionManager
    ):
        """Corrupt line after boundary: last_consolidated stays unchanged."""
        path = tmp_session_manager._get_session_path("telegram:12345")
        lines = [
            _make_metadata_line(last_consolidated=2),
            _make_message_line("user", "msg0"),  # index 0 — consolidated
            _make_message_line("user", "msg1"),  # index 1 — consolidated
            _make_message_line("user", "msg2"),  # index 2 — unconsolidated
            "CORRUPT LINE {{{",  # index 3 — skipped, AFTER boundary (lc=2)
        ]
        _write_session_file(path, lines)

        session = tmp_session_manager._load("telegram:12345")

        assert session is not None
        assert len(session.messages) == 3  # 4 minus 1 corrupt
        # skipped_before_boundary=False → NO fallback → last_consolidated stays at 2
        assert session.last_consolidated == 2

    def test_load_non_dict_line_before_boundary_no_over_consolidation(
        self, tmp_session_manager: SessionManager
    ):
        """Non-dict JSON before consolidation boundary does NOT trigger over-consolidation."""
        path = tmp_session_manager._get_session_path("telegram:12345")
        lines = [
            _make_metadata_line(last_consolidated=2),
            _make_message_line("user", "msg0"),  # index 0 — consolidated
            '"garbage string"',  # non-dict — NOT a message, no index-shift
            _make_message_line("user", "msg1"),  # index 1 — consolidated
            _make_message_line("user", "msg2"),  # index 2 — unconsolidated
        ]
        _write_session_file(path, lines)

        session = tmp_session_manager._load("telegram:12345")

        assert session is not None
        assert len(session.messages) == 3  # 4 minus 1 non-dict
        # Non-dict skips do NOT set skipped_before_boundary → lc stays at 2
        assert session.last_consolidated == 2


class TestMessagesOnlyNoMetadata:
    def test_load_messages_only_no_metadata(self, tmp_session_manager: SessionManager):
        """File with only message lines (no metadata): recovered with len(messages) fallback."""
        path = tmp_session_manager._get_session_path("telegram:12345")
        lines = [
            _make_message_line("user", "hello"),
            _make_message_line("assistant", "world"),
        ]
        _write_session_file(path, lines)

        session = tmp_session_manager._load("telegram:12345")

        assert session is not None
        assert len(session.messages) == 2
        # metadata_parsed=False → fallback: last_consolidated = len(messages) = 2
        assert session.last_consolidated == 2
        assert session.metadata == {}  # default when no metadata line


class TestFallbackWithEmptyMessages:
    def test_load_valid_metadata_high_lc_all_messages_corrupt(
        self, tmp_session_manager: SessionManager
    ):
        """Metadata says lc=100 but all message lines are corrupt → lc corrected to 0.
        Regression: when messages is empty, the fallback must still set lc=len(messages)
        to prevent a high lc from making new user messages invisible to the LLM."""
        path = tmp_session_manager._get_session_path("telegram:12345")
        lines = [
            _make_metadata_line(last_consolidated=100),
            "CORRUPT{{{",
            "ALSO BAD{{{",
        ]
        _write_session_file(path, lines)

        session = tmp_session_manager._load("telegram:12345")

        assert session is not None
        assert session.messages == []
        # Both corrupt lines parse-fail at msg_index=0 < lc=100 → skipped_before_boundary=True.
        # Fallback sets lc = len([]) = 0 (no `and messages` guard).
        assert session.last_consolidated == 0


class TestLastConsolidatedNoUpperClamp:
    def test_load_last_consolidated_exceeds_message_count(
        self, tmp_session_manager: SessionManager
    ):
        """Metadata says lc=10 but only 5 messages in file — lc preserved (no upper clamp)."""
        path = tmp_session_manager._get_session_path("telegram:12345")
        lines = [_make_metadata_line(last_consolidated=10)]
        for i in range(5):
            lines.append(_make_message_line("user", f"msg{i}"))
        _write_session_file(path, lines)

        session = tmp_session_manager._load("telegram:12345")

        assert session is not None
        assert len(session.messages) == 5
        # No upper-bound clamp — all consumers handle via slice semantics
        assert session.last_consolidated == 10


class TestValidFileRoundtrip:
    def test_load_valid_file_roundtrip(self, tmp_session_manager: SessionManager):
        """Valid file saved by save() loads back via fresh SessionManager with key, messages,
        metadata, last_consolidated, and created_at intact."""
        fixed_time = datetime(2026, 1, 15, 12, 0, 0)
        session = Session(key="telegram:12345", metadata={"lang": "en"}, created_at=fixed_time)
        for i in range(10):
            session.add_message("user", f"msg{i}")
        session.last_consolidated = 7
        tmp_session_manager.save(session)

        # Fresh SessionManager with empty cache — forces _load() from disk
        fresh_manager = SessionManager(workspace=tmp_session_manager.workspace)
        loaded = fresh_manager._load("telegram:12345")

        assert loaded is not None
        assert loaded.key == "telegram:12345"
        assert len(loaded.messages) == 10
        assert loaded.messages[0]["content"] == "msg0"
        assert loaded.messages[9]["content"] == "msg9"
        assert loaded.metadata == {"lang": "en"}
        assert loaded.last_consolidated == 7
        assert loaded.created_at == fixed_time


class TestOSErrorCatch:
    def test_load_permission_error_returns_none(
        self, tmp_session_manager: SessionManager, tmp_path
    ):
        """OSError (e.g. PermissionError) on open → returns None."""
        path = tmp_session_manager._get_session_path("telegram:12345")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_make_metadata_line() + "\n", encoding="utf-8")
        # Remove read permission
        path.chmod(0o000)

        try:
            session = tmp_session_manager._load("telegram:12345")
        finally:
            path.chmod(0o644)  # restore for cleanup

        # On systems where root ignores permissions (e.g. Docker), this may not fail.
        # Only assert None if we're not root.
        import os

        if os.geteuid() != 0:
            assert session is None
        else:
            # Root can read anything — at least verify it loads without error
            assert session is not None


class TestDuplicateMetadata:
    def test_load_duplicate_metadata_last_wins(self, tmp_session_manager: SessionManager):
        """Two metadata lines: second overwrites first, untrustworthy flag is sticky."""
        path = tmp_session_manager._get_session_path("telegram:12345")
        lines = [
            _make_metadata_line(last_consolidated="invalid"),
            _make_message_line("user", "msg0"),
            _make_metadata_line(last_consolidated=1),
            _make_message_line("user", "msg1"),
        ]
        _write_session_file(path, lines)

        session = tmp_session_manager._load("telegram:12345")

        assert session is not None
        assert len(session.messages) == 2
        # First metadata sets untrustworthy=True (sticky), second metadata parsed but
        # untrustworthy was never reset → fallback triggers
        assert session.last_consolidated == 2  # len(messages)


class TestRecursionErrorIndexShift:
    def test_load_recursion_error_before_boundary_triggers_fallback(
        self, tmp_session_manager: SessionManager
    ):
        """RecursionError (deeply nested JSON) before consolidation boundary → index-shift protection."""
        path = tmp_session_manager._get_session_path("telegram:12345")
        deep_json = '{"a":' * 50000 + '"b"' + "}" * 50000
        lines = [
            _make_metadata_line(last_consolidated=2),
            _make_message_line("user", "msg0"),  # index 0
            deep_json,  # RecursionError at index 1, BEFORE boundary (lc=2)
            _make_message_line("user", "msg2"),  # index 2 (shifted)
        ]
        _write_session_file(path, lines)

        session = tmp_session_manager._load("telegram:12345")

        assert session is not None
        assert len(session.messages) == 2
        # skipped_before_boundary=True → fallback: last_consolidated = len(messages)
        assert session.last_consolidated == 2


class TestCorruptLineAtBoundary:
    def test_load_corrupt_line_at_exact_boundary_no_fallback(
        self, tmp_session_manager: SessionManager
    ):
        """Corrupt line at exactly msg_index == last_consolidated: no fallback triggered."""
        path = tmp_session_manager._get_session_path("telegram:12345")
        lines = [
            _make_metadata_line(last_consolidated=2),
            _make_message_line("user", "msg0"),  # index 0 — consolidated
            _make_message_line("user", "msg1"),  # index 1 — consolidated
            "CORRUPT{{{",  # index 2 == lc — first unconsolidated, NOT before boundary
            _make_message_line("user", "msg3"),  # index 3 — unconsolidated
        ]
        _write_session_file(path, lines)

        session = tmp_session_manager._load("telegram:12345")

        assert session is not None
        assert len(session.messages) == 3
        # msg_index=2, lc=2 → 2 < 2 is False → skipped_before_boundary stays False
        assert session.last_consolidated == 2


class TestAllNonDictFile:
    def test_load_all_non_dict_returns_none(self, tmp_session_manager: SessionManager):
        """File with only non-dict JSON values (strings): recovered=False → None."""
        path = tmp_session_manager._get_session_path("telegram:12345")
        lines = ['"hello"', '"world"', '"test"']
        _write_session_file(path, lines)

        session = tmp_session_manager._load("telegram:12345")

        assert session is None
