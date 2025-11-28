from __future__ import annotations

from dataclasses import dataclass
import math
import random
import re
from statistics import mean, stdev
from typing import Iterable, List

import dspy
from scipy import stats

from mrp.pipeline import MRP
from mrp.store import InMemoryBehaviorStore


@dataclass(frozen=True)
class EvalExample:
    name: str
    initial_question: str
    new_question: str
    expected_answer: str


@dataclass(frozen=True)
class ConditionOutcome:
    tokens_step4: int
    correct: bool


@dataclass(frozen=True)
class Observation:
    repetition: int
    example_name: str
    tokens_a: int
    tokens_b: int
    correct_a: bool
    correct_b: bool


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


def _strip_text(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def _parse_numeric(text: str) -> float | None:
    normalized = _strip_text(text).replace(",", "").replace("π", "pi")
    pi_match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\*?pi", normalized)
    if pi_match:
        return float(pi_match.group(1)) * math.pi

    fraction_match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)/([+-]?\d+(?:\.\d+)?)", normalized)
    if fraction_match:
        numerator = float(fraction_match.group(1))
        denominator = float(fraction_match.group(2))
        if denominator != 0.0:
            return numerator / denominator
        return None

    decimal_match = re.fullmatch(r"[+-]?\d+(?:\.\d+)?", normalized)
    if decimal_match:
        return float(decimal_match.group(0))

    number_in_text = re.search(r"[+-]?\d+(?:\.\d+)?", normalized)
    if number_in_text is not None:
        return float(number_in_text.group(0))

    return None


def accuracy_metric(
    example: EvalExample,
    pred: dspy.Prediction,
    trace=None,
) -> float:
    predicted_answer = pred.behaved_solution.solution
    predicted_value = _parse_numeric(predicted_answer)
    expected_value = _parse_numeric(example.expected_answer)

    if predicted_value is not None and expected_value is not None:
        if math.isclose(predicted_value, expected_value, rel_tol=1e-3, abs_tol=1e-3):
            return 1.0
        return 0.0

    predicted_text = _strip_text(predicted_answer)
    expected_text = _strip_text(example.expected_answer)
    return 1.0 if predicted_text == expected_text else 0.0


def _sum_usage_tokens(history: Iterable[dict]) -> int:
    total = 0
    for entry in history:
        usage = entry.get("usage")
        if usage is None:
            continue
        if "total_tokens" in usage:
            total += int(usage["total_tokens"])
            continue
        prompt = int(usage.get("prompt_tokens", 0))
        completion = int(usage.get("completion_tokens", 0))
        total += prompt + completion
    return total


def _build_pipeline(with_store: bool, shared_store: InMemoryBehaviorStore | None) -> MRP:
    solve_lm = dspy.LM("openai/gpt-4.1-mini", cache=False)
    behave_lm = dspy.LM("openai/gpt-4.1-mini", cache=False)
    reflection_lm = dspy.LM(
        "openai/gpt-5.1",
        temperature=1.0,
        max_tokens=32000,
        reasoning_effort="high",
        cache=False,
    )

    behavior_store = shared_store if with_store else None

    return MRP(
        solve_lm=solve_lm,
        reflection_lm=reflection_lm,
        behavior_lm=reflection_lm,
        behave_lm=behave_lm,
        behavior_store=behavior_store,
        max_behaviors_for_step4=8,
    )


def _run_example(
    example: EvalExample,
    with_store: bool,
    shared_store: InMemoryBehaviorStore | None,
) -> ConditionOutcome:
    pipeline = _build_pipeline(with_store=with_store, shared_store=shared_store)
    behave_lm = pipeline.behave_lm
    behave_lm.history.clear()

    result = pipeline(
        initial_question=example.initial_question,
        new_question=example.new_question,
        existing_behaviors=None,
    )

    tokens_step4 = _sum_usage_tokens(behave_lm.history)
    is_correct = bool(accuracy_metric(example, result))
    return ConditionOutcome(tokens_step4=tokens_step4, correct=is_correct)


def evaluate_repetitions(
    examples: List[EvalExample],
    repetitions: int,
    seed: int,
) -> list[Observation]:
    dspy.configure(track_usage=True)
    rng = random.Random(seed)

    observations: list[Observation] = []

    for repetition in range(repetitions):
        shuffled = list(examples)
        rng.shuffle(shuffled)
        persistent_store = InMemoryBehaviorStore()

        for example in shuffled:
            outcome_a = _run_example(example, with_store=False, shared_store=None)
            outcome_b = _run_example(
                example,
                with_store=True,
                shared_store=persistent_store,
            )

            observations.append(
                Observation(
                    repetition=repetition,
                    example_name=example.name,
                    tokens_a=outcome_a.tokens_step4,
                    tokens_b=outcome_b.tokens_step4,
                    correct_a=outcome_a.correct,
                    correct_b=outcome_b.correct,
                )
            )

    return observations


def summarize(
    observations: list[Observation],
    repetitions: int,
    alternative: str,
    alpha: float = 0.05,
) -> None:
    tokens_a = [float(obs.tokens_a) for obs in observations]
    tokens_b = [float(obs.tokens_b) for obs in observations]
    token_deltas = [a - b for a, b in zip(tokens_a, tokens_b)]

    acc_a = [1.0 if obs.correct_a else 0.0 for obs in observations]
    acc_b = [1.0 if obs.correct_b else 0.0 for obs in observations]

    mean_tokens_a = mean(tokens_a)
    mean_tokens_b = mean(tokens_b)
    reduction_pct = ((mean_tokens_a - mean_tokens_b) / mean_tokens_a) * 100.0

    ttest_result = stats.ttest_rel(tokens_a, tokens_b, alternative=alternative)
    ci = ttest_result.confidence_interval(confidence_level=0.95, alternative=alternative)
    significant = ttest_result.pvalue < alpha

    print("\nPer-observation results (Step 4 only):")
    print(
        "repetition,example,tokens_no_store,tokens_with_store,"
        "correct_no_store,correct_with_store"
    )
    for obs in observations:
        print(
            f"{obs.repetition},"
            f"{obs.example_name},"
            f"{obs.tokens_a},"
            f"{obs.tokens_b},"
            f"{int(obs.correct_a)},"
            f"{int(obs.correct_b)}"
        )

    print("\nAggregate across repetitions:")
    print(f"repetitions: {repetitions}")
    print(f"observations: {len(observations)}")
    print(f"mean_tokens_no_store: {mean_tokens_a:.2f}")
    print(f"mean_tokens_with_store: {mean_tokens_b:.2f}")
    print(f"token_reduction_pct: {reduction_pct:.2f}%")
    print(f"accuracy_no_store: {mean(acc_a):.2f}")
    print(f"accuracy_with_store: {mean(acc_b):.2f}")
    print(f"mean_delta_tokens: {mean(token_deltas):.2f}")
    print(f"std_delta_tokens: {stdev(token_deltas):.2f}")
    print(
        f"paired_t_test (alternative={alternative}): "
        f"t={ttest_result.statistic:.4f}, p={ttest_result.pvalue:.4g}, df={ttest_result.df}"
    )
    print(
        "95% CI for mean delta: "
        f"[{ci.low:.2f}, {ci.high:.2f}]"
    )
    print(f"significant_at_alpha_{alpha}: {significant}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate metacognitive reuse token effects.")
    parser.add_argument("--repetitions", type=int, default=5, help="Number of shuffled runs to perform.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    parser.add_argument(
        "--alternative",
        choices=["two-sided", "greater", "less"],
        default="greater",
        help="Hypothesis for paired t-test (delta = tokens_A - tokens_B).",
    )
    args = parser.parse_args()

    examples = default_dataset()
    observations = evaluate_repetitions(examples, repetitions=args.repetitions, seed=args.seed)
    summarize(observations, repetitions=args.repetitions, alternative=args.alternative)


if __name__ == "__main__":
    main()
