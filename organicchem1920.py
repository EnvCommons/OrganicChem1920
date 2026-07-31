"""
OrganicChem1920 - 1920 Organic Chemistry Textbook Environment

Questions testing procedural understanding and conceptual comprehension
from "A Text-book of Organic Chemistry" by Arnold Frederik Holleman (1920).

LLM-based grading handles alternative nomenclature and explanation quality.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd
import openai
from pydantic import BaseModel, Field

from openreward.environments import (
    Environment,
    JSONObject,
    TextBlock,
    ToolOutput,
    terminal,
    tool,
    Split
)


# ============================================================================
# Path Configuration
# ============================================================================

# Production: /orwd_data; Development: local directory
if os.path.exists("/orwd_data"):
    DATA_PATH = Path("/orwd_data")
else:
    DATA_PATH = Path(__file__).parent


# ============================================================================
# Grader Template
# ============================================================================

GRADER_TEMPLATE = """You are grading a student's answer to a question from a 1920 organic chemistry textbook (Holleman's "A Text-book of Organic Chemistry").

The answer should be evaluated on CONCEPTUAL CORRECTNESS and UNDERSTANDING, not exact wording.

Key considerations for chemistry answers:
1. **Chemical nomenclature**: Accept both IUPAC names, common names, and historical 1920s terminology
   - Example: "ethyl alcohol" = "ethanol" = "C2H5OH"
2. **Procedural understanding**: Check if the student grasps WHY steps are performed, not just WHAT
3. **No partial credit**: This is STRICT grading. Mark CORRECT only if the answer contains every key point of the reference answer, with no chemical errors and no material omissions. If anything essential is missing, vague, or wrong, do NOT mark CORRECT — use PARTIALLY_CORRECT for a partially sound answer and INCORRECT otherwise.
4. **Safety awareness**: Value safety considerations even if not in reference answer

Question: {question}
Reference Answer: {reference_answer}
Student Answer: {student_answer}

Provide your analysis in this format:

**Analysis:**
[2-3 sentences explaining what the student got right/wrong, addressing chemical accuracy and conceptual understanding]

**Score:** [0.0 to 1.0 — your confidence the answer is fully correct]

**Grade:** [CORRECT | PARTIALLY_CORRECT | INCORRECT]

**Feedback:**
[1-2 sentences of constructive feedback for the student]
"""

# The grader endpoint occasionally returns finish_reason=stop with empty content.
# A blank or fieldless reply carries no verdict, so retry it before giving up.
GRADER_MAX_ATTEMPTS = 3

# The template asks for "**Grade:** [CORRECT | PARTIALLY_CORRECT | INCORRECT]".
# Match the field itself so the grade is read from where it was requested;
# longest alternative first, since "INCORRECT" contains "CORRECT".
GRADE_FIELD_RE = re.compile(
    r"\*\*Grade:\*\*\s*\[?\s*(PARTIALLY[_ ]CORRECT|INCORRECT|CORRECT)",
    re.IGNORECASE,
)
SCORE_FIELD_RE = re.compile(r"\*\*Score:\*\*\s*([0-9.]+)", re.IGNORECASE)
SCORE_LOOSE_RE = re.compile(r"Score[:\s]+([0-9.]+)", re.IGNORECASE)


# ============================================================================
# Global Data Loading
# ============================================================================

QUESTIONS_DF: pd.DataFrame | None = None
TASKS_BY_SPLIT: dict[str, list[JSONObject]] = {}


def load_questions_data() -> None:
    """Load questions from parquet at module import time."""
    global QUESTIONS_DF, TASKS_BY_SPLIT

    parquet_path = DATA_PATH / "organicchem1920_questions.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {parquet_path}. "
            f"Please run 'python generate_dataset.py' to create the dataset, "
            f"or refer to DATA_UPLOAD.md for upload instructions."
        )

    QUESTIONS_DF = pd.read_parquet(parquet_path)

    # Group by split for efficient list_tasks
    for split in ["train", "validation", "test"]:
        split_df = QUESTIONS_DF[QUESTIONS_DF["split"] == split]
        TASKS_BY_SPLIT[split] = [
            {"uuid": str(row["uuid"])}
            for _, row in split_df.iterrows()
        ]

    print(f"Loaded {len(QUESTIONS_DF)} questions across {len(TASKS_BY_SPLIT)} splits")


# Load at module import
load_questions_data()


# ============================================================================
# Pydantic Models
# ============================================================================

class TaskSpec(BaseModel):
    """Lightweight task specification with UUID only."""
    uuid: str


class AnswerInput(BaseModel, extra="forbid"):
    """Input schema for answer submission."""
    answer: str = Field(
        ...,
        description="Your answer to the chemistry question. Provide detailed reasoning and explanation."
    )


# ============================================================================
# Environment Class
# ============================================================================

class OrganicChem1920(Environment):
    """
    1920 Organic Chemistry Textbook Question Environment.

    Tests procedural understanding and conceptual comprehension of organic
    chemistry from Holleman's "A Text-book of Organic Chemistry" (1920).
    Questions focus on reasoning about procedures, mechanisms, and chemical
    principles rather than rote recall.

    Grading uses LLM evaluation to handle:
    - Alternative chemical nomenclature (IUPAC vs. common vs. historical)
    - Varied explanation styles

    Grading is strict and the reward is binary: an answer earns 1.0 only if the
    grader marks it CORRECT, meaning every key point is present with no
    material omissions. A partially sound answer earns 0.0.
    """

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.validated = TaskSpec.model_validate(task_spec)

        # Load question from global DataFrame
        question_row = QUESTIONS_DF[QUESTIONS_DF["uuid"] == self.validated.uuid].iloc[0]

        # Extract fields
        self.uuid = str(question_row["uuid"])
        self.question = str(question_row["question"])
        self.reference_answer = str(question_row["answer"])
        self.category = str(question_row["category"])
        self.difficulty = str(question_row["difficulty"])
        self.page_reference = int(question_row["page_reference"])
        self.context_snippet = str(question_row["context_snippet"])
        self.chapter = str(question_row["chapter"])

        # Initialize OpenAI client with required API key
        api_key = secrets.get("openai_api_key")
        if not api_key:
            raise ValueError(
                "OpenAI API key required for LLM grading. "
                "Pass secrets={'openai_api_key': 'your-key'} when creating session."
            )
        self.client = openai.AsyncClient(api_key=api_key)

    @classmethod
    def list_splits(cls) -> list[str]:
        """Return available splits."""
        return [Split(name="train", type="train"), Split(name="validation", type="validation"), Split(name="test", type="test")]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        """Return tasks for the given split."""
        if split not in TASKS_BY_SPLIT:
            raise ValueError(
                f"Unknown split: {split}. "
                f"Available splits: {list(TASKS_BY_SPLIT.keys())}"
            )
        return TASKS_BY_SPLIT[split].copy()

    async def get_prompt(self) -> List[TextBlock]:
        """Return the question prompt with metadata."""
        prompt_text = f"""{self.question}"""

        return [TextBlock(text=prompt_text)]

    async def _grade_answer(self, student_answer: str) -> dict:
        """
        Grade student answer using GPT-5-mini.

        Returns dict with:
        - grading_response: full LLM response
        - score: the grader's numeric score, or None if it gave none
        - grade: CORRECT | PARTIALLY_CORRECT | INCORRECT (this sets the reward)
        """
        grader_prompt = GRADER_TEMPLATE.format(
            question=self.question,
            reference_answer=self.reference_answer,
            student_answer=student_answer
        )

        # Use gpt-5-mini for grading (per framework requirements)
        # NO temperature parameter
        # API exceptions propagate: the platform retries them and treats a
        # persistent failure as terminal. Only a 2xx reply carrying no usable
        # verdict is retried here, since that fails silently otherwise.
        for _ in range(GRADER_MAX_ATTEMPTS):
            response = await self.client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": grader_prompt}]
            )

            grading_response = response.choices[0].message.content or ""

            if self._has_verdict(grading_response):
                break
        else:
            raise RuntimeError(
                f"Grader returned no usable verdict after {GRADER_MAX_ATTEMPTS} "
                f"attempts (last reply: {grading_response!r}). Refusing to score "
                f"this answer, since a blank grader reply is not evidence that "
                f"the answer was wrong."
            )

        return {
            "grading_response": grading_response,
            "score": self._parse_score(grading_response),
            "grade": self._parse_grade(grading_response),
        }

    def _has_verdict(self, response: str) -> bool:
        """Whether a grader reply carries a grade the reward can be read from.

        The reward comes from the grade alone, so a reply with a score but no
        grade is not usable — treating it as usable would let _parse_grade fall
        through to INCORRECT and fabricate a 0.0.
        """
        if GRADE_FIELD_RE.search(response):
            return True
        # No Grade field, but a bare keyword still parses. "CORRECT" is a
        # substring of INCORRECT and PARTIALLY_CORRECT, so this covers all three.
        return "CORRECT" in response.upper()

    def _parse_grade(self, response: str) -> str:
        """Extract final grade from grading response."""
        field = GRADE_FIELD_RE.search(response)
        if field:
            return field.group(1).upper().replace(" ", "_")

        # No Grade field — fall back to scanning for a bare keyword. Check
        # PARTIALLY and INCORRECT first: both contain "CORRECT" as a substring.
        upper = response.upper()
        if "PARTIALLY_CORRECT" in upper or "PARTIALLY CORRECT" in upper:
            return "PARTIALLY_CORRECT"
        elif "CORRECT" in upper and "INCORRECT" not in upper:
            return "CORRECT"
        else:
            return "INCORRECT"

    def _parse_score(self, response: str) -> Optional[float]:
        """Extract the grader's numeric score, if it supplied one.

        Recorded for analysis only — the reward comes from the grade. Returns
        None when the reply carries no score rather than inventing one.
        """
        # Look for "Score: X.XX" or "**Score:** X.XX"
        match = SCORE_FIELD_RE.search(response) or SCORE_LOOSE_RE.search(response)
        if not match:
            return None

        try:
            return max(0.0, min(1.0, float(match.group(1))))  # Clamp to [0, 1]
        except ValueError:
            return None

    @terminal
    @tool
    async def answer(self, params: AnswerInput) -> ToolOutput:
        """
        Submit your answer to the chemistry question.

        Your answer will be graded on conceptual understanding and reasoning,
        not exact wording. Alternative chemical nomenclature is accepted.
        Grading is strict: the answer must cover every key point to score at
        all, and a partially correct answer earns nothing.
        """
        # Handle empty answers
        if not params.answer.strip():
            return ToolOutput(
                blocks=[TextBlock(text="Please provide an answer before submitting.")],
                metadata={"error": "empty_answer"},
                reward=0.0,
                finished=True
            )

        # Grade the answer
        grading = await self._grade_answer(params.answer)

        # Strict binary reward: only a CORRECT verdict earns credit.
        # PARTIALLY_CORRECT and INCORRECT both score 0.0. The grader's numeric
        # score is kept in metadata for analysis but does not shape the reward,
        # since the LLM picks it freely and it is not calibrated between runs.
        reward = 1.0 if grading['grade'] == "CORRECT" else 0.0

        # This is a @terminal tool, so the harness routes the model's final
        # plain message here and the rollout is already over — no turn remains
        # in which anything could read a reply. Emit a one-line receipt for the
        # logs and keep the detail in metadata: the reference answer and the
        # full grading must not be echoed into the recorded transcript, which
        # would put this task's answer key in every trajectory.
        return ToolOutput(
            blocks=[TextBlock(text=f"Answer graded: {grading['grade']} (reward {reward:.1f}).")],
            metadata={
                "uuid": self.uuid,
                "student_answer": params.answer,
                "reference_answer": self.reference_answer,
                "grade": grading['grade'],
                "score": grading['score'],
                "reward": reward,
                "category": self.category,
                "difficulty": self.difficulty,
                "full_grading": grading['grading_response']
            },
            reward=reward,
            finished=True
        )
