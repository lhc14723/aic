"""Small, auditable filesystem helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def atomic_write_text(path: str | Path, text: str) -> None:
    """Atomically write UTF-8 text without exposing a partial destination."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def write_json(path: str | Path, value: Any) -> None:
    """Write deterministic, human-readable JSON."""
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON document."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write one JSON object per line."""
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    atomic_write_text(path, text)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON Lines file."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    return rows
