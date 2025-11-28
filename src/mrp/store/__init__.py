"""Behavior storage and retrieval for the MRP pipeline.

This module provides:
- Immutable data models (frozen Pydantic models) for behaviors and records
- Protocol-based storage abstraction with both sync and async interfaces
- In-memory and file-backed implementations
- Functional search/scoring using comprehensions and sorted()
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import aiofiles
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


# ---------------------------------------------------------------------------
# Immutable Data Models (frozen Pydantic models)
# ---------------------------------------------------------------------------


class Behavior(BaseModel):
    """A reusable behavior extracted from a math reasoning trace (immutable)."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(
        description=(
            "Short, machine-usable behavior name starting with 'behavior_' "
            "that can be referenced explicitly in reasoning."
        )
    )
    instruction: str = Field(
        description=(
            "Actionable instruction describing when and how to apply this behavior "
            "while solving math problems."
        )
    )


class BehaviorRecord(BaseModel):
    """A behavior plus provenance metadata (immutable)."""

    model_config = ConfigDict(frozen=True)

    behavior: Behavior = Field(description="The behavior itself (name + instruction).")
    source_question: str | None = Field(
        default=None,
        description=(
            "Optional problem statement from which this behavior was extracted."
        ),
    )
    source_solution: str | None = Field(
        default=None,
        description=(
            "Optional solution trace that motivated this behavior, if available."
        ),
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when this behavior was first added to the store.",
    )


# ---------------------------------------------------------------------------
# Type Aliases (Python 3.12+ syntax)
# ---------------------------------------------------------------------------

type RecordMap = dict[str, BehaviorRecord]
type ScoredRecord = tuple[int, BehaviorRecord]


# ---------------------------------------------------------------------------
# Pure Functions (scoring and search logic)
# ---------------------------------------------------------------------------


def _score_record(record: BehaviorRecord, query_lower: str) -> int:
    """Compute relevance score for a record against a query (pure function)."""
    searchable_text = " ".join(
        [
            record.behavior.name,
            record.behavior.instruction,
            record.source_question or "",
        ]
    ).lower()
    return searchable_text.count(query_lower)


def _search_records(
    records: Sequence[BehaviorRecord],
    query: str,
    k: int | None = None,
) -> list[BehaviorRecord]:
    """Search records by relevance score using functional patterns (pure function).

    Uses comprehension for scoring, sorted() for ordering, and slicing for limit.
    """
    query_lower = query.lower()

    # Score all records using comprehension (functional transformation)
    scored: list[ScoredRecord] = [
        (score, record)
        for record in records
        if (score := _score_record(record, query_lower)) > 0
    ]

    # Sort by score descending using sorted() (functional, returns new list)
    sorted_scored = sorted(scored, key=lambda item: item[0], reverse=True)

    # Extract records and apply limit
    results = [record for _, record in sorted_scored]
    return results[:k] if k is not None else results


def _merge_records(base: RecordMap, new_records: Sequence[BehaviorRecord]) -> RecordMap:
    """Merge new records into existing map, deduplicating by name (pure function).

    Returns a new dict rather than mutating the input.
    """
    return {**base, **{r.behavior.name: r for r in new_records}}


# ---------------------------------------------------------------------------
# Storage Protocol (sync interface for DSPy compatibility)
# ---------------------------------------------------------------------------


class BehaviorStore(Protocol):
    """Abstract interface for persisting and retrieving behaviors.

    Uses synchronous methods for DSPy module compatibility.
    Store operations are fast (in-memory or local file I/O).
    """

    def add(self, record: BehaviorRecord) -> None:
        """Add or update a single behavior record in the store."""
        ...

    def add_many(self, records: Sequence[BehaviorRecord]) -> None:
        """Add or update multiple behavior records in the store."""
        ...

    def get_all(self) -> list[BehaviorRecord]:
        """Return all behavior records currently stored."""
        ...

    def search(self, query: str, k: int | None = None) -> list[BehaviorRecord]:
        """Return up to k behavior records that are most relevant to the query."""
        ...


# ---------------------------------------------------------------------------
# In-Memory Implementation (functional internal operations)
# ---------------------------------------------------------------------------


class InMemoryBehaviorStore:
    """In-memory behavior store with deduplication by name.

    Internal state is a dict mapping behavior name -> record.
    Mutations are localized to add/add_many; all reads are pure.
    """

    __slots__ = ("_records_by_name",)

    def __init__(self, initial_records: Sequence[BehaviorRecord] = ()) -> None:
        """Initialize store, optionally with initial records."""
        self._records_by_name: RecordMap = {r.behavior.name: r for r in initial_records}

    def add(self, record: BehaviorRecord) -> None:
        """Add or update a single record (localized mutation)."""
        self._records_by_name[record.behavior.name] = record

    def add_many(self, records: Sequence[BehaviorRecord]) -> None:
        """Merge multiple records using functional dict merge."""
        self._records_by_name = _merge_records(self._records_by_name, records)

    def get_all(self) -> list[BehaviorRecord]:
        """Return all records as a list (pure read)."""
        return list(self._records_by_name.values())

    def search(self, query: str, k: int | None = None) -> list[BehaviorRecord]:
        """Search records by relevance (delegates to pure function)."""
        return _search_records(self.get_all(), query, k)


# ---------------------------------------------------------------------------
# File-Backed Implementation (async I/O with sync facade)
# ---------------------------------------------------------------------------

# Type adapter for JSON serialization (created once, reused)
_RECORD_LIST_ADAPTER: TypeAdapter[list[BehaviorRecord]] = TypeAdapter(
    list[BehaviorRecord]
)


def _serialize_records(records: Sequence[BehaviorRecord]) -> str:
    """Serialize records to JSON string (pure function)."""
    return _RECORD_LIST_ADAPTER.dump_json(
        list(records), indent=2, ensure_ascii=False
    ).decode("utf-8")


def _deserialize_records(text: str) -> list[BehaviorRecord]:
    """Deserialize JSON string to records (pure function)."""
    if not text.strip():
        return []
    return _RECORD_LIST_ADAPTER.validate_json(text)


class JsonFileBehaviorStore:
    """File-backed behavior store that persists behaviors as JSON.

    Provides both sync and async interfaces:
    - Sync methods (add, add_many) use blocking I/O for DSPy compatibility
    - Async methods (aload, aflush) use aiofiles for non-blocking I/O

    The on-disk representation is a JSON list of BehaviorRecord objects.
    """

    __slots__ = ("_path", "_memory")

    def __init__(self, path: str | Path, *, auto_load: bool = True) -> None:
        """Initialize store with optional auto-loading from disk.

        Args:
            path: Path to the JSON file for persistence.
            auto_load: If True (default), load existing records from disk on init.
        """
        self._path = Path(path)
        self._memory = InMemoryBehaviorStore()
        if auto_load:
            self._load_from_disk()

    @property
    def path(self) -> Path:
        """Location of the underlying JSON file."""
        return self._path

    # --- Sync I/O (for DSPy module compatibility) ---

    def _load_from_disk(self) -> None:
        """Load records from disk synchronously."""
        if not self._path.exists():
            return
        text = self._path.read_text(encoding="utf-8")
        records = _deserialize_records(text)
        if records:
            self._memory.add_many(records)

    def _flush_to_disk(self) -> None:
        """Flush records to disk synchronously."""
        text = _serialize_records(self._memory.get_all())
        if not self._path.parent.exists():
            self._path.parent.mkdir(parents=True)
        self._path.write_text(text, encoding="utf-8")

    # --- Async I/O (for concurrent pipelines) ---

    async def aload(self) -> None:
        """Load records from disk asynchronously using aiofiles."""
        if not self._path.exists():
            return
        async with aiofiles.open(self._path, mode="r", encoding="utf-8") as f:
            text = await f.read()
        records = _deserialize_records(text)
        if records:
            self._memory.add_many(records)

    async def aflush(self) -> None:
        """Flush records to disk asynchronously using aiofiles."""
        text = _serialize_records(self._memory.get_all())
        if not self._path.parent.exists():
            self._path.parent.mkdir(parents=True)
        async with aiofiles.open(self._path, mode="w", encoding="utf-8") as f:
            await f.write(text)

    # --- BehaviorStore interface (sync, delegates to in-memory store) ---

    def add(self, record: BehaviorRecord) -> None:
        """Add a record and persist to disk (sync)."""
        self._memory.add(record)
        self._flush_to_disk()

    def add_many(self, records: Sequence[BehaviorRecord]) -> None:
        """Add multiple records and persist to disk (sync)."""
        self._memory.add_many(records)
        self._flush_to_disk()

    def get_all(self) -> list[BehaviorRecord]:
        """Return all records (pure read from memory)."""
        return self._memory.get_all()

    def search(self, query: str, k: int | None = None) -> list[BehaviorRecord]:
        """Search records by relevance (delegates to in-memory store)."""
        return self._memory.search(query=query, k=k)
