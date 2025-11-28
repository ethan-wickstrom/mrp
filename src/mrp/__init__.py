"""
mrp - a Python package for metacognitive reuse.
"""

# MRP - the full four-stage metacognitive reuse pipeline.
from .pipeline import MRP

# Behavior - the core behavior type (name + instruction).
# BehaviorRecord - behavior plus metadata (source question, solution, timestamp).
# BehaviorStore - a protocol describing the store interface.
# InMemoryBehaviorStore - a deduplicating in-memory implementation.
# JsonFileBehaviorStore - a JSON file-backed store layered on top of the in-memory one.
from .store import (
    Behavior,
    BehaviorRecord,
    BehaviorStore,
    InMemoryBehaviorStore,
    JsonFileBehaviorStore,
)

__all__ = [
    "Behavior",
    "BehaviorRecord",
    "BehaviorStore",
    "InMemoryBehaviorStore",
    "JsonFileBehaviorStore",
    "MRP",
]
