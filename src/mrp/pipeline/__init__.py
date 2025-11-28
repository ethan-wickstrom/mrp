from __future__ import annotations

from pydantic import BaseModel, Field

import dspy

from mrp.store import Behavior, BehaviorRecord, BehaviorStore


class Reflection(BaseModel):
    """Structured reflection on a math solution."""

    correctness_analysis: str = Field(
        description=(
            "Analysis of mathematical correctness, including any calculation errors, "
            "logical gaps, or unjustified steps."
        )
    )
    missing_behaviors_analysis: str = Field(
        description=(
            "Analysis of behaviors that should have been used but were not, including "
            "how they would shorten the solution or prevent errors."
        )
    )
    new_behaviors: list[Behavior] = Field(
        default_factory=list,
        description=(
            "Specific new behaviors that would help with similar problems. Each "
            "behavior's instruction should be self-contained and generalizable."
        ),
    )


class SolveOutput(BaseModel):
    """Normalized representation of a model's solution to a math problem."""

    question: str = Field(description="The math problem that was solved.")
    reasoning: str = Field(
        description="The model's chain-of-thought reasoning used to reach the answer."
    )
    solution: str = Field(description="The final answer stated succinctly.")


class MetacognitiveReuseResult(BaseModel):
    """Container for the full four-stage metacognitive reuse pipeline output."""

    initial_solution: SolveOutput = Field(
        description="Solution produced in Step 1 (Solve)."
    )
    reflection: Reflection = Field(
        description="Structured reflection produced in Step 2 (Reflect)."
    )
    extracted_behaviors: list[Behavior] = Field(
        description=(
            "Final behavior handbook entries derived from the current example and "
            "any pre-existing behaviors."
        )
    )
    behaved_solution: SolveOutput = Field(
        description=(
            "Solution to the new problem produced in Step 4 (Behave) using the "
            "behavior handbook."
        )
    )


def format_solution_for_reflection(solved: SolveOutput) -> str:
    """Render a SolveOutput as a single text block for reflection / behavior extraction."""
    return f"Reasoning:\n{solved.reasoning}\n\nFinal answer:\n{solved.solution}"


class SolveMathProblem(dspy.Signature):
    """Solve a math problem step by step and return the final answer."""

    question: str = dspy.InputField(desc="The math problem to solve.")
    solution: str = dspy.OutputField(
        desc=(
            "The final numeric or algebraic answer. The reasoning used to obtain the "
            "answer is exposed via the 'reasoning' field when using dspy.ChainOfThought."
        )
    )


class ReflectOnSolution(dspy.Signature):
    """Reflect on a solved math problem to assess correctness and propose behaviors."""

    question: str = dspy.InputField(desc="The original math problem.")
    solution: str = dspy.InputField(
        desc="The full solution, including reasoning steps and final answer."
    )
    reflection: Reflection = dspy.OutputField(
        desc=(
            "Structured reflection including correctness analysis, missing behaviors "
            "analysis, and suggested new behaviors."
        )
    )


class GenerateBehaviorsFromReflection(dspy.Signature):
    """Distill a reflection into concrete, reusable behaviors."""

    question: str = dspy.InputField(desc="The original math problem.")
    solution: str = dspy.InputField(
        desc="The full solution, including reasoning steps and final answer."
    )
    reflection: Reflection = dspy.InputField(
        desc=(
            "Structured reflection on the solution, including candidate new behaviors."
        )
    )
    behaviors: list[Behavior] = dspy.OutputField(
        desc=(
            "List of reusable behaviors (name + instruction) distilled from the "
            "reflection and solution."
        )
    )


class BehaveWithBehaviors(dspy.Signature):
    """Solve a new math problem using a provided behavior handbook."""

    question: str = dspy.InputField(desc="A new math problem to solve.")
    behaviors: list[Behavior] = dspy.InputField(
        desc=(
            "List of behaviors that may be useful for this problem. When a behavior is "
            "applied, the reasoning should explicitly reference it by name."
        )
    )
    solution: str = dspy.OutputField(
        desc=(
            "A step-by-step solution that explicitly cites behavior names when they "
            "are applied, followed by the final answer."
        )
    )


class SolveModule(dspy.Module):
    """STEP 1: Solve – generate an initial solution trace for a math problem."""

    def __init__(self) -> None:
        super().__init__()
        # ChainOfThought adds a 'reasoning' field to this signature.
        self._solver = dspy.ChainOfThought(SolveMathProblem)

    def forward(self, question: str) -> dspy.Prediction:
        prediction = self._solver(question=question)
        solve_output = SolveOutput(
            question=question,
            reasoning=str(prediction.reasoning),
            solution=str(prediction.solution),
        )
        return dspy.Prediction(solve_output=solve_output)


class ReflectModule(dspy.Module):
    """STEP 2: Reflect – critique a solution and suggest new behaviors."""

    def __init__(self) -> None:
        super().__init__()
        self._reflect = dspy.ChainOfThought(ReflectOnSolution)

    def forward(self, question: str, solved: SolveOutput) -> dspy.Prediction:
        solution_text = format_solution_for_reflection(solved)
        prediction = self._reflect(question=question, solution=solution_text)
        return dspy.Prediction(reflection=prediction.reflection)


class GenerateBehaviorsModule(dspy.Module):
    """STEP 3: Generate Behaviors – distill reflection into reusable behaviors."""

    def __init__(self) -> None:
        super().__init__()
        # Behavior extraction is a relatively direct transformation, so Predict suffices.
        self._generator = dspy.Predict(GenerateBehaviorsFromReflection)

    def forward(
        self,
        question: str,
        solved: SolveOutput,
        reflection: Reflection,
    ) -> dspy.Prediction:
        solution_text = format_solution_for_reflection(solved)
        prediction = self._generator(
            question=question,
            solution=solution_text,
            reflection=reflection,
        )
        return dspy.Prediction(behaviors=prediction.behaviors)


class BehaveModule(dspy.Module):
    """STEP 4: Behave – behavior-conditioned inference on a new problem."""

    def __init__(self) -> None:
        super().__init__()
        self._behave = dspy.ChainOfThought(BehaveWithBehaviors)

    def forward(self, question: str, behaviors: list[Behavior]) -> dspy.Prediction:
        prediction = self._behave(question=question, behaviors=behaviors)
        solve_output = SolveOutput(
            question=question,
            reasoning=str(prediction.reasoning),
            solution=str(prediction.solution),
        )
        return dspy.Prediction(solve_output=solve_output)


class MRP(dspy.Module):
    """Full four-stage Metacognitive Reuse pipeline.

    The pipeline wires together the four DSPy modules as follows:

    1. Solve:      Use a smaller LM to solve the initial problem and emit a reasoning trace.
    2. Reflect:    Use a stronger LM to critique the solution and suggest behaviors.
    3. Behaviors:  Use a strong LM to distill reflection into a behavior handbook.
    4. Behave:     Use a smaller LM to solve a new problem conditioned on the behaviors.

    Optionally, a BehaviorStore can be provided to persist behaviors across sessions.
    """

    def __init__(
        self,
        solve_lm: dspy.LM,
        reflection_lm: dspy.LM,
        behavior_lm: dspy.LM,
        behave_lm: dspy.LM,
        behavior_store: BehaviorStore | None = None,
        max_behaviors_for_step4: int | None = None,
    ) -> None:
        super().__init__()
        self.solve_lm = solve_lm
        self.reflection_lm = reflection_lm
        self.behavior_lm = behavior_lm
        self.behave_lm = behave_lm
        self.behavior_store = behavior_store
        self.max_behaviors_for_step4 = max_behaviors_for_step4

        self.solve_module = SolveModule()
        self.reflect_module = ReflectModule()
        self.generate_behaviors_module = GenerateBehaviorsModule()
        self.behave_module = BehaveModule()

    def _records_from_behaviors(
        self,
        behaviors: list[Behavior],
        source_question: str,
        source_solution: SolveOutput,
    ) -> list[BehaviorRecord]:
        """Wrap plain behaviors in BehaviorRecord objects with provenance."""
        solution_text = format_solution_for_reflection(source_solution)
        return [
            BehaviorRecord(
                behavior=behavior,
                source_question=source_question,
                source_solution=solution_text,
            )
            for behavior in behaviors
        ]

    def forward(
        self,
        initial_question: str,
        new_question: str,
        existing_behaviors: list[Behavior] | None = None,
    ) -> dspy.Prediction:
        # STEP 1: Solve with a smaller LM.
        with dspy.context(lm=self.solve_lm):
            solve_prediction = self.solve_module(question=initial_question)
        initial_solution = solve_prediction.solve_output

        # STEP 2: Reflect with a stronger LM.
        with dspy.context(lm=self.reflection_lm):
            reflection_prediction = self.reflect_module(
                question=initial_question,
                solved=initial_solution,
            )
        reflection = reflection_prediction.reflection

        # STEP 3: Generate behaviors with a stronger LM.
        with dspy.context(lm=self.behavior_lm):
            behaviors_prediction = self.generate_behaviors_module(
                question=initial_question,
                solved=initial_solution,
                reflection=reflection,
            )
        behaviors_from_example = behaviors_prediction.behaviors

        # Merge behaviors from reflection, STEP 3, and any explicit existing behaviors.
        behavior_by_name: dict[str, Behavior] = {
            behavior.name: behavior for behavior in reflection.new_behaviors
        }
        for behavior in behaviors_from_example:
            behavior_by_name[behavior.name] = behavior

        if existing_behaviors is not None:
            for behavior in existing_behaviors:
                behavior_by_name[behavior.name] = behavior

        current_behaviors = list(behavior_by_name.values())

        # If a behavior store is configured, persist current behaviors and build
        # the handbook from the accumulated store contents.
        if self.behavior_store is not None:
            records = self._records_from_behaviors(
                behaviors=current_behaviors,
                source_question=initial_question,
                source_solution=initial_solution,
            )
            self.behavior_store.add_many(records)

            if self.max_behaviors_for_step4 is None:
                stored_records = self.behavior_store.get_all()
            else:
                stored_records = self.behavior_store.search(
                    query=new_question,
                    k=self.max_behaviors_for_step4,
                )
            behavior_handbook = [record.behavior for record in stored_records]
        else:
            behavior_handbook = current_behaviors

        # STEP 4: Behave on a new question with the final handbook.
        with dspy.context(lm=self.behave_lm):
            behave_prediction = self.behave_module(
                question=new_question,
                behaviors=behavior_handbook,
            )
        behaved_solution = behave_prediction.solve_output

        return dspy.Prediction(
            initial_solution=initial_solution,
            reflection=reflection,
            extracted_behaviors=behavior_handbook,
            behaved_solution=behaved_solution,
        )
