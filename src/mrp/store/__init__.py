from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence, Protocol

from pydantic import BaseModel, Field, TypeAdapter


class Behavior(BaseModel):
    """A reusable behavior extracted from a math reasoning trace."""

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
    """A behavior plus metadata about where it came from and when it was added."""

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


class BehaviorStore(Protocol):
    """Abstract interface for persisting and retrieving behaviors."""

    def add(self, record: BehaviorRecord) -> None:
        """Add or update a single behavior record in the store."""

    def add_many(self, records: Sequence[BehaviorRecord]) -> None:
        """Add or update multiple behavior records in the store."""

    def get_all(self) -> list[BehaviorRecord]:
        """Return all behavior records currently stored."""

    def search(self, query: str, k: int | None = None) -> list[BehaviorRecord]:
        """Return up to k behavior records that are most relevant to the query."""


class InMemoryBehaviorStore(BehaviorStore):
    """Simple in-memory behavior store with deduplication by name."""

    def __init__(self) -> None:
        self._records_by_name: dict[str, BehaviorRecord] = {}

    def add(self, record: BehaviorRecord) -> None:
        self._records_by_name[record.behavior.name] = record

    def add_many(self, records: Sequence[BehaviorRecord]) -> None:
        for record in records:
            self.add(record)

    def get_all(self) -> list[BehaviorRecord]:
        return list(self._records_by_name.values())

    def search(self, query: str, k: int | None = None) -> list[BehaviorRecord]:
        query_lower = query.lower()
        scored: list[tuple[int, BehaviorRecord]] = []
        for record in self._records_by_name.values():
            searchable_text = " ".join(
                [
                    record.behavior.name,
                    record.behavior.instruction,
                    record.source_question or "",
                ]
            ).lower()
            score = searchable_text.count(query_lower)
            if score > 0:
                scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        results = [record for _, record in scored]
        if k is not None:
            return results[:k]
        return results


class JsonFileBehaviorStore(BehaviorStore):
    """File-backed behavior store that persists behaviors as JSON.

    The on-disk representation is a JSON list of BehaviorRecord.model_dump() objects.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._memory = InMemoryBehaviorStore()
        self._load_from_disk()

    @property
    def path(self) -> Path:
        """Location of the underlying JSON file."""
        return self._path

    def _load_from_disk(self) -> None:
        if not self._path.exists():
            return

        text = self._path.read_text(encoding="utf-8")
        if not text.strip():
            return

        adapter = TypeAdapter(list[BehaviorRecord])
        records = adapter.validate_json(text)
        self._memory.add_many(records)

    def _flush_to_disk(self) -> None:
        records = self._memory.get_all()
        adapter = TypeAdapter(list[BehaviorRecord])
        text = adapter.dump_json(records, indent=2, ensure_ascii=False).decode("utf-8")
        if not self._path.parent.exists():
            self._path.parent.mkdir(parents=True)
        self._path.write_text(text, encoding="utf-8")

    def add(self, record: BehaviorRecord) -> None:
        self._memory.add(record)
        self._flush_to_disk()

    def add_many(self, records: Sequence[BehaviorRecord]) -> None:
        self._memory.add_many(records)
        self._flush_to_disk()

    def get_all(self) -> list[BehaviorRecord]:
        return self._memory.get_all()

    def search(self, query: str, k: int | None = None) -> list[BehaviorRecord]:
        return self._memory.search(query=query, k=k)
