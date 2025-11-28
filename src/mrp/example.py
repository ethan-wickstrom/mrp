from __future__ import annotations

import asyncio
import os
from pathlib import Path

import dspy

from mrp.store import JsonFileBehaviorStore
from mrp.pipeline import MRP


def build_pipeline_with_store(
    path: str | Path,
) -> tuple[MRP, JsonFileBehaviorStore]:
    """Construct a MRP wired to a JSON-backed behavior store.

    This uses a smaller model for solving/behaving and a larger model for
    reflection/behavior extraction. Replace model names if you prefer other providers.
    """
    store = JsonFileBehaviorStore(path=path)

    small_lm = dspy.LM("openai/gpt-4.1-mini")
    large_lm = dspy.LM(
        "openai/gpt-5.1",
        temperature=1.0,
        max_tokens=32000,
        reasoning_effort="high",
    )

    # Set a global default LM so ad-hoc modules also work as expected.
    dspy.configure(lm=small_lm)

    pipeline = MRP(
        solve_lm=small_lm,
        reflection_lm=large_lm,
        behavior_lm=large_lm,
        behave_lm=small_lm,
        behavior_store=store,
        max_behaviors_for_step4=8,
    )
    return pipeline, store


async def run_one_session_async(
    store_path: str | Path,
    initial_question: str,
    new_question: str,
) -> None:
    """Run a single pipeline session using async execution."""
    pipeline, store = build_pipeline_with_store(store_path)

    print("=== Running pipeline (async) ===")
    # Use native DSPy async via acall() which invokes aforward() on the pipeline.
    result = await pipeline.acall(
        initial_question=initial_question,
        new_question=new_question,
        existing_behaviors=None,
    )

    _print_result(result, store)


def run_one_session(
    store_path: str | Path,
    initial_question: str,
    new_question: str,
) -> None:
    """Run a single pipeline session using synchronous execution."""
    pipeline, store = build_pipeline_with_store(store_path)

    print("=== Running pipeline ===")
    result = pipeline(
        initial_question=initial_question,
        new_question=new_question,
        existing_behaviors=None,
    )

    _print_result(result, store)


def _print_result(result: dspy.Prediction, store: JsonFileBehaviorStore) -> None:
    """Print pipeline results and store state."""
    print("=== STEP 1: Initial Solution ===")
    print(result.initial_solution.reasoning)
    print("Final answer:", result.initial_solution.solution)
    print()

    print("=== STEP 2: Reflection ===")
    print("Correctness analysis:")
    print(result.reflection.correctness_analysis)
    print()
    print("Missing behaviors analysis:")
    print(result.reflection.missing_behaviors_analysis)
    print()
    print("Suggested new behaviors:")
    for behavior in result.reflection.new_behaviors:
        print(f"- {behavior.name}: {behavior.instruction}")
    print()

    print("=== STEP 3: Extracted Behaviors (Handbook contribution) ===")
    for behavior in result.extracted_behaviors:
        print(f"- {behavior.name}: {behavior.instruction}")
    print()

    print("=== STEP 4: Behavior-Conditioned Solution on New Problem ===")
    print(result.behaved_solution.reasoning)
    print("Final answer:", result.behaved_solution.solution)
    print()

    all_records = store.get_all()
    print(f"Store now contains {len(all_records)} behavior records.")
    for record in all_records:
        print(f"* {record.behavior.name} (from: {record.source_question!r})")
    print()


async def main_async() -> None:
    """Async entry point demonstrating native DSPy async patterns."""
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError(
            "Please set the OPENAI_API_KEY environment variable before running this example."
        )

    store_path = Path("behavior_handbook.json")

    # First session: one pair of problems.
    first_initial = "A circle has radius 4. What is its area?"
    first_new = "A circle has radius 7. What is its area?"
    await run_one_session_async(
        store_path=store_path,
        initial_question=first_initial,
        new_question=first_new,
    )

    # Second session: new pair of problems, reusing the same persisted behavior store.
    second_initial = (
        "A fair six-sided die is rolled twice. "
        "What is the probability that the sum of the two rolls is at least 10?"
    )
    second_new = (
        "A fair six-sided die is rolled three times. "
        "What is the probability that the sum of the three rolls is at least 15?"
    )
    await run_one_session_async(
        store_path=store_path,
        initial_question=second_initial,
        new_question=second_new,
    )


def main() -> None:
    """Synchronous entry point (delegates to async main)."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
