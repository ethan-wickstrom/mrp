"""Evaluation harness for the MRP pipeline using functional programming patterns.

This module uses declarative, side-effect-minimizing constructs:
- Pure functions that transform data without external mutation
- Result types for error handling instead of try/catch with continue
- itertools/functools for data transformations
- Closure-based dependency injection instead of global mutable state
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
import itertools
import math
import random
import re
import sys
from statistics import mean, stdev
from typing import Awaitable, Iterable, Sequence

import dspy
from scipy import stats

from mrp.pipeline import MRP
from mrp.store import InMemoryBehaviorStore


# ---------------------------------------------------------------------------
# Data Types (immutable, frozen dataclasses)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalExample:
    name: str
    initial_question: str
    new_question: str
    expected_answer: str


@dataclass(frozen=True)
class ConditionOutcome:
    """Result of running a single condition (with or without store)."""

    tokens_step4: int
    correct: bool


@dataclass(frozen=True)
class Observation:
    """Paired observation comparing both conditions for a single example."""

    repetition: int
    example_name: str
    tokens_a: int
    tokens_b: int
    correct_a: bool
    correct_b: bool


@dataclass(frozen=True)
class Success:
    """Successful evaluation result containing an Observation."""

    value: Observation


@dataclass(frozen=True)
class Failure:
    """Failed evaluation with error context."""

    example_name: str
    error: str


# Result type: either Success or Failure (sum type / tagged union)
type EvalResult = Success | Failure


def default_dataset() -> list[EvalExample]:
    """A small, diverse set of math pairs for metacognitive reuse evaluation."""
    return [
        EvalExample(
            name="circle_area_r7",
            initial_question="A circle has radius 4. What is its area?",
            new_question="A circle has radius 7. What is its area?",
            expected_answer="49*pi",
        ),
        EvalExample(
            name="dice_sum_three",
            initial_question=(
                "A fair six-sided die is rolled twice. "
                "What is the probability that the sum of the two rolls is at least 10?"
            ),
            new_question=(
                "A fair six-sided die is rolled three times. "
                "What is the probability that the sum of the three rolls is at least 15?"
            ),
            expected_answer="5/54",
        ),
        EvalExample(
            name="rectangle_area",
            initial_question="A rectangle has length 5 and width 7. What is its area?",
            new_question="A rectangle has length 8 and width 12. What is its area?",
            expected_answer="96",
        ),
        EvalExample(
            name="percent_increase",
            initial_question="Increase 100 by 10%. What is the result?",
            new_question="Increase 80 by 25%. What is the result?",
            expected_answer="100",
        ),
        EvalExample(
            name="distance_travel",
            initial_question="A car travels at 60 mph for 2 hours. How far does it go?",
            new_question="A car travels at 45 mph for 3 hours. How far does it go?",
            expected_answer="135",
        ),
        EvalExample(
            name="linear_equation",
            initial_question="Solve for x: 2x + 5 = 17",
            new_question="Solve for x: 3x - 4 = 20",
            expected_answer="8",
        ),
        EvalExample(
            name="simple_interest",
            initial_question="Simple interest on $500 at 4% for 3 years (final amount)?",
            new_question="Simple interest on $800 at 5% for 2 years (final amount)?",
            expected_answer="880",
        ),
        EvalExample(
            name="average_value",
            initial_question="What is the average of 4, 6, and 10?",
            new_question="What is the average of 3, 7, 11, and 13?",
            expected_answer="8.5",
        ),
        EvalExample(
            name="triangle_area",
            initial_question="A triangle with base 10 and height 6 has what area?",
            new_question="A triangle with base 14 and height 9 has what area?",
            expected_answer="63",
        ),
        EvalExample(
            name="coin_heads",
            initial_question="Probability of getting exactly two heads in three coin flips?",
            new_question="Probability of getting at least two heads in four coin flips?",
            expected_answer="11/16",
        ),
        EvalExample(
            name="work_rate",
            initial_question=(
                "Person A paints a room in 6 hours, B in 8 hours. "
                "How long working together?"
            ),
            new_question=(
                "Person A paints a room in 3 hours, B in 4 hours. "
                "How long working together?"
            ),
            expected_answer="12/7",
        ),
        EvalExample(
            name="arithmetic_sequence",
            initial_question=(
                "In an arithmetic sequence with first term 3 and common difference 5, "
                "what is the 10th term?"
            ),
            new_question=(
                "In an arithmetic sequence with first term 7 and common difference 4, "
                "what is the 15th term?"
            ),
            expected_answer="63",
        ),
    ]


# ---------------------------------------------------------------------------
# Pure Functions (no side effects, output depends only on input)
# ---------------------------------------------------------------------------


def _strip_text(value: str) -> str:
    """Remove whitespace and lowercase a string."""
    return re.sub(r"\s+", "", value).lower()


def _parse_numeric(text: str) -> float | None:
    """Extract a numeric value from text, handling pi expressions, fractions, and decimals.

    Searches for patterns anywhere in the text (not just exact matches) to handle
    model outputs like "The answer is 5/54" or "Area = 49*pi".
    Priority: pi expressions > fractions > plain decimals.
    """
    normalized = _strip_text(text).replace(",", "").replace("π", "pi")

    # Search for pi expressions anywhere (e.g., "49*pi", "49pi")
    if pi_match := re.search(r"([+-]?\d+(?:\.\d+)?)\*?pi", normalized):
        return float(pi_match.group(1)) * math.pi

    # Search for fractions anywhere (e.g., "5/54", "11/16")
    if fraction_match := re.search(
        r"([+-]?\d+(?:\.\d+)?)/([+-]?\d+(?:\.\d+)?)", normalized
    ):
        numerator, denominator = (
            float(fraction_match.group(1)),
            float(fraction_match.group(2)),
        )
        return numerator / denominator if denominator != 0.0 else None

    # Search for plain decimals (e.g., "96", "8.5")
    if decimal_match := re.search(r"[+-]?\d+(?:\.\d+)?", normalized):
        return float(decimal_match.group(0))

    return None


def _extract_tokens_from_entry(entry: dict) -> int:
    """Extract token count from a single history entry (pure function)."""
    usage = entry.get("usage")
    if usage is None:
        return 0
    if "total_tokens" in usage:
        return int(usage["total_tokens"])
    return int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0))


def _sum_usage_tokens(history: Iterable[dict]) -> int:
    """Sum token usage across history entries using functional reduce."""
    return reduce(
        lambda acc, entry: acc + _extract_tokens_from_entry(entry), history, 0
    )


def accuracy_metric(
    example: EvalExample,
    pred: dspy.Prediction,
    trace: object = None,
) -> float:
    """Compute accuracy (1.0 or 0.0) by comparing predicted vs expected answer."""
    predicted_answer: str = pred.behaved_solution.solution
    predicted_value = _parse_numeric(predicted_answer)
    expected_value = _parse_numeric(example.expected_answer)

    # Numeric comparison with tolerance
    if predicted_value is not None and expected_value is not None:
        return (
            1.0
            if math.isclose(predicted_value, expected_value, rel_tol=1e-3, abs_tol=1e-3)
            else 0.0
        )

    # Fallback to exact text match
    return (
        1.0
        if _strip_text(predicted_answer) == _strip_text(example.expected_answer)
        else 0.0
    )


# ---------------------------------------------------------------------------
# Pipeline Construction (factory functions)
# ---------------------------------------------------------------------------


def _build_pipeline(
    with_store: bool, shared_store: InMemoryBehaviorStore | None
) -> MRP:
    """Construct an MRP pipeline with the specified store configuration."""
    solve_lm = dspy.LM("openai/gpt-4.1-mini", cache=False)
    behave_lm = dspy.LM("openai/gpt-4.1-mini", cache=False)
    reflection_lm = dspy.LM(
        "openai/gpt-5.1",
        temperature=1.0,
        max_tokens=32000,
        reasoning_effort="high",
        cache=False,
    )
    return MRP(
        solve_lm=solve_lm,
        reflection_lm=reflection_lm,
        behavior_lm=reflection_lm,
        behave_lm=behave_lm,
        behavior_store=shared_store if with_store else None,
        max_behaviors_for_step4=8,
    )


def _shuffle_copy(
    examples: Sequence[EvalExample], rng: random.Random
) -> list[EvalExample]:
    """Create a shuffled copy of examples (pure aside from RNG state)."""
    shuffled = list(examples)
    rng.shuffle(shuffled)
    return shuffled


def _prepare_shuffles(
    examples: Sequence[EvalExample],
    repetitions: int,
    seed: int,
) -> list[list[EvalExample]]:
    """Generate shuffled example orderings for each repetition.

    Uses a seeded RNG for reproducibility. Expressed as a list comprehension
    over range(repetitions) with a helper function for the shuffle operation.
    """
    rng = random.Random(seed)
    return [_shuffle_copy(examples, rng) for _ in range(repetitions)]


# ---------------------------------------------------------------------------
# Async Evaluation (functional patterns with dependency injection)
# ---------------------------------------------------------------------------


# Type alias for the example runner function signature
type ExampleRunner = Callable[
    [EvalExample, bool, InMemoryBehaviorStore | None],
    Awaitable[ConditionOutcome],
]


def _make_example_runner(semaphore: asyncio.Semaphore) -> ExampleRunner:
    """Factory that creates an example runner with injected semaphore (closure pattern).

    This eliminates global mutable state by capturing the semaphore in a closure.
    The returned async function is pure in its data flow (inputs -> outputs).
    """

    async def run_example(
        example: EvalExample,
        with_store: bool,
        shared_store: InMemoryBehaviorStore | None,
    ) -> ConditionOutcome:
        pipeline = _build_pipeline(with_store=with_store, shared_store=shared_store)
        behave_lm = pipeline.behave_lm
        behave_lm.history.clear()

        async with semaphore:
            result = await pipeline.acall(
                initial_question=example.initial_question,
                new_question=example.new_question,
                existing_behaviors=None,
            )

        return ConditionOutcome(
            tokens_step4=_sum_usage_tokens(behave_lm.history),
            correct=bool(accuracy_metric(example, result)),
        )

    return run_example


def _outcomes_to_observation(
    repetition: int,
    example: EvalExample,
    outcome_a: ConditionOutcome,
    outcome_b: ConditionOutcome,
) -> Observation:
    """Pure function to combine outcomes into an Observation."""
    return Observation(
        repetition=repetition,
        example_name=example.name,
        tokens_a=outcome_a.tokens_step4,
        tokens_b=outcome_b.tokens_step4,
        correct_a=outcome_a.correct,
        correct_b=outcome_b.correct,
    )


async def _evaluate_example_pair(
    run_example: ExampleRunner,
    repetition: int,
    example: EvalExample,
    store: InMemoryBehaviorStore,
) -> EvalResult:
    """Evaluate a single example under both conditions, returning Success or Failure.

    Uses Result type pattern for functional error handling instead of try/catch with continue.
    """
    try:
        # Positional args required for Callable type compatibility
        outcome_a, outcome_b = await asyncio.gather(
            run_example(example, False, None),  # without store
            run_example(example, True, store),  # with store
        )
        return Success(
            _outcomes_to_observation(repetition, example, outcome_a, outcome_b)
        )
    except Exception as e:
        return Failure(example_name=example.name, error=str(e))


async def _evaluate_single_repetition(
    run_example: ExampleRunner,
    repetition: int,
    examples: Sequence[EvalExample],
    verbose: bool = True,
) -> list[EvalResult]:
    """Evaluate all examples in a repetition sequentially (store accumulates).

    Returns a list of EvalResult (Success | Failure) for functional filtering downstream.
    Sequential processing is required because the behavior store accumulates across examples.
    """
    store = InMemoryBehaviorStore()
    total = len(examples)

    async def evaluate_with_progress(idx: int, example: EvalExample) -> EvalResult:
        result = await _evaluate_example_pair(run_example, repetition, example, store)
        if verbose:
            status = "✓" if isinstance(result, Success) else "✗"
            print(
                f"[rep {repetition}] {idx + 1}/{total} {example.name} {status}",
                file=sys.stderr,
            )
        return result

    # Sequential evaluation (store must accumulate) expressed as async comprehension
    return [await evaluate_with_progress(idx, ex) for idx, ex in enumerate(examples)]


def _extract_successes(results: Iterable[EvalResult]) -> list[Observation]:
    """Filter successful results and extract observations (pure function)."""
    return [r.value for r in results if isinstance(r, Success)]


def _log_failures(results: Iterable[EvalResult], repetition: int) -> None:
    """Log failed evaluations to stderr (isolated side effect)."""
    for r in results:
        if isinstance(r, Failure):
            print(
                f"[rep {repetition}] Error on {r.example_name}: {r.error}",
                file=sys.stderr,
            )


async def evaluate_repetitions_async(
    examples: Sequence[EvalExample],
    repetitions: int,
    seed: int,
    max_concurrent: int = 10,
) -> list[Observation]:
    """Run evaluation across multiple repetitions with functional orchestration.

    - Semaphore is created once and injected via closure (no global state)
    - Repetitions run in parallel via asyncio.gather
    - Results flattened via itertools.chain.from_iterable
    - Failures filtered out functionally
    """
    dspy.configure(track_usage=True)

    # Inject semaphore via factory (eliminates global mutable state)
    semaphore = asyncio.Semaphore(max_concurrent)
    run_example = _make_example_runner(semaphore)

    shuffled_runs = _prepare_shuffles(examples, repetitions, seed)

    # Parallel repetition evaluation (each repetition is independent)
    gathered = await asyncio.gather(
        *[
            _evaluate_single_repetition(run_example, rep_idx, run_examples)
            for rep_idx, run_examples in enumerate(shuffled_runs)
        ]
    )
    all_results: list[list[EvalResult]] = list(gathered)

    # Log failures for each repetition (isolated side effect)
    for rep_idx, results in enumerate(all_results):
        _log_failures(results, rep_idx)

    # Flatten results and extract successes using itertools.chain.from_iterable
    flattened_results = itertools.chain.from_iterable(all_results)
    return _extract_successes(flattened_results)


def evaluate_repetitions(
    examples: Sequence[EvalExample],
    repetitions: int,
    seed: int,
) -> list[Observation]:
    """Synchronous entry point that delegates to async implementation."""
    return asyncio.run(
        evaluate_repetitions_async(
            examples=examples,
            repetitions=repetitions,
            seed=seed,
        )
    )


# ---------------------------------------------------------------------------
# Reporting (functional data extraction + isolated print side effect)
# ---------------------------------------------------------------------------


def _format_observation_csv(obs: Observation) -> str:
    """Format a single observation as CSV row (pure function)."""
    return (
        f"{obs.repetition},{obs.example_name},{obs.tokens_a},{obs.tokens_b},"
        f"{int(obs.correct_a)},{int(obs.correct_b)}"
    )


def _compute_summary_stats(
    observations: Sequence[Observation],
    alternative: str,
    alpha: float,
) -> dict[str, float | bool | str]:
    """Compute all summary statistics from observations (pure function).

    Returns a dictionary of computed values for downstream formatting.
    """
    # Extract data series using comprehensions (functional transformation)
    tokens_a = [float(obs.tokens_a) for obs in observations]
    tokens_b = [float(obs.tokens_b) for obs in observations]
    token_deltas = [a - b for a, b in zip(tokens_a, tokens_b, strict=True)]
    acc_a = [1.0 if obs.correct_a else 0.0 for obs in observations]
    acc_b = [1.0 if obs.correct_b else 0.0 for obs in observations]

    # Compute aggregate statistics
    mean_tokens_a = mean(tokens_a)
    mean_tokens_b = mean(tokens_b)
    reduction_pct = ((mean_tokens_a - mean_tokens_b) / mean_tokens_a) * 100.0

    ttest_result = stats.ttest_rel(tokens_a, tokens_b, alternative=alternative)
    ci = ttest_result.confidence_interval(confidence_level=1.0 - alpha)

    return {
        "mean_tokens_a": mean_tokens_a,
        "mean_tokens_b": mean_tokens_b,
        "reduction_pct": reduction_pct,
        "accuracy_a": mean(acc_a),
        "accuracy_b": mean(acc_b),
        "mean_delta": mean(token_deltas),
        "std_delta": stdev(token_deltas),
        "t_stat": ttest_result.statistic,
        "p_value": ttest_result.pvalue,
        "df": ttest_result.df,
        "ci_low": ci.low,
        "ci_high": ci.high,
        "significant": ttest_result.pvalue < alpha,
        "alternative": alternative,
        "alpha": alpha,
    }


def summarize(
    observations: list[Observation],
    repetitions: int,
    alternative: str,
    alpha: float = 0.05,
) -> None:
    """Print evaluation summary (isolated side effect with pure data computation).

    Data extraction and statistics computation are pure functions.
    Only the final print statements produce side effects.
    """
    stats_dict = _compute_summary_stats(observations, alternative, alpha)

    # Generate CSV rows using generator expression (functional)
    csv_header = (
        "repetition,example,tokens_no_store,tokens_with_store,"
        "correct_no_store,correct_with_store"
    )
    csv_rows = "\n".join(_format_observation_csv(obs) for obs in observations)

    # Isolated side effect: print output
    print(f"\nPer-observation results (Step 4 only):\n{csv_header}\n{csv_rows}")
    print("\nAggregate across repetitions:")
    print(f"repetitions: {repetitions}")
    print(f"observations: {len(observations)}")
    print(f"mean_tokens_no_store: {stats_dict['mean_tokens_a']:.2f}")
    print(f"mean_tokens_with_store: {stats_dict['mean_tokens_b']:.2f}")
    print(f"token_reduction_pct: {stats_dict['reduction_pct']:.2f}%")
    print(f"accuracy_no_store: {stats_dict['accuracy_a']:.2f}")
    print(f"accuracy_with_store: {stats_dict['accuracy_b']:.2f}")
    print(f"mean_delta_tokens: {stats_dict['mean_delta']:.2f}")
    print(f"std_delta_tokens: {stats_dict['std_delta']:.2f}")
    print(
        f"paired_t_test (alternative={stats_dict['alternative']}): "
        f"t={stats_dict['t_stat']:.4f}, p={stats_dict['p_value']:.4g}, df={stats_dict['df']}"
    )
    print(
        f"{100 * (1 - float(stats_dict['alpha'])):.0f}% CI for mean delta: "
        f"[{stats_dict['ci_low']:.2f}, {stats_dict['ci_high']:.2f}]"
    )
    print(f"significant_at_alpha_{stats_dict['alpha']}: {stats_dict['significant']}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate metacognitive reuse token effects."
    )
    parser.add_argument(
        "--repetitions", type=int, default=5, help="Number of shuffled runs to perform."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling."
    )
    parser.add_argument(
        "--alternative",
        choices=["two-sided", "greater", "less"],
        default="greater",
        help="Hypothesis for paired t-test (delta = tokens_A - tokens_B).",
    )
    args = parser.parse_args()

    examples = default_dataset()
    observations = evaluate_repetitions(
        examples, repetitions=args.repetitions, seed=args.seed
    )
    summarize(observations, repetitions=args.repetitions, alternative=args.alternative)


if __name__ == "__main__":
    main()
