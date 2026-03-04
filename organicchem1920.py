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
from typing import List

import pandas as pd
import openai
from pydantic import BaseModel, Field

from openreward.environments import (
    Environment,
    JSONObject,
    TextBlock,
    ToolOutput,
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
3. **Partial credit**: Award partial credit for partially correct reasoning
4. **Safety awareness**: Value safety considerations even if not in reference answer

Question: {question}
Reference Answer: {reference_answer}
Student Answer: {student_answer}

Provide your analysis in this format:

**Analysis:**
[2-3 sentences explaining what the student got right/wrong, addressing chemical accuracy and conceptual understanding]

**Score:** [0.0 to 1.0, where 0.7+ is partially correct, 0.85+ is correct]

**Grade:** [CORRECT | PARTIALLY_CORRECT | INCORRECT]

**Feedback:**
[1-2 sentences of constructive feedback for the student]
"""


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
    - Partial credit for incomplete but conceptually sound answers
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
        - score: numeric 0.0-1.0
        - grade: CORRECT | PARTIALLY_CORRECT | INCORRECT
        - feedback: student-facing explanation
        """
        grader_prompt = GRADER_TEMPLATE.format(
            question=self.question,
            reference_answer=self.reference_answer,
            student_answer=student_answer
        )

        # Use gpt-5-mini for grading (per framework requirements)
        # NO temperature parameter
        response = await self.client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": grader_prompt}]
        )

        grading_response = response.choices[0].message.content or ""

        # Parse grade and score
        grade = self._parse_grade(grading_response)
        score = self._parse_score(grading_response)
        feedback = self._extract_feedback(grading_response)

        return {
            "grading_response": grading_response,
            "score": score,
            "grade": grade,
            "feedback": feedback
        }

    def _parse_grade(self, response: str) -> str:
        """Extract final grade from grading response."""
        upper = response.upper()
        if "PARTIALLY_CORRECT" in upper or "PARTIALLY CORRECT" in upper:
            return "PARTIALLY_CORRECT"
        elif "CORRECT" in upper and "INCORRECT" not in upper:
            return "CORRECT"
        else:
            return "INCORRECT"

    def _parse_score(self, response: str) -> float:
        """Extract numeric score from grading response."""
        # Look for "Score: X.XX" or "**Score:** X.XX"
        match = re.search(r"\*\*Score:\*\*\s*([0-9.]+)", response, re.IGNORECASE)
        if not match:
            match = re.search(r"Score[:\s]+([0-9.]+)", response, re.IGNORECASE)

        if match:
            try:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))  # Clamp to [0, 1]
            except ValueError:
                pass

        # Fallback based on grade keyword
        if "CORRECT" in response.upper() and "INCORRECT" not in response.upper():
            return 1.0
        elif "PARTIALLY" in response.upper():
            return 0.5
        else:
            return 0.0

    def _extract_feedback(self, response: str) -> str:
        """Extract student feedback from grading response."""
        # Look for "Feedback:" section
        match = re.search(
            r"\*\*Feedback:\*\*\s*(.+?)(?:\n\n|\Z)",
            response,
            re.IGNORECASE | re.DOTALL
        )
        if not match:
            match = re.search(
                r"Feedback[:\s]+(.+?)(?:\n\n|\Z)",
                response,
                re.IGNORECASE | re.DOTALL
            )

        if match:
            return match.group(1).strip()

        # Fallback: return last paragraph
        paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
        return paragraphs[-1] if paragraphs else "See grading analysis above."

    @tool
    async def answer(self, params: AnswerInput) -> ToolOutput:
        """
        Submit your answer to the chemistry question.

        Your answer will be graded on conceptual understanding and reasoning,
        not exact wording. Alternative chemical nomenclature is accepted.
        Partial credit is awarded for incomplete but conceptually sound reasoning.
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

        # Construct feedback message
        feedback_text = f"""**Grade:** {grading['grade']} (Score: {grading['score']:.2f})

**Your Answer:**
{params.answer}

**Reference Answer:**
{self.reference_answer}

**Feedback:**
{grading['feedback']}

---
**Source:** Page {self.page_reference}, {self.chapter}
"""

        return ToolOutput(
            blocks=[TextBlock(text=feedback_text)],
            metadata={
                "uuid": self.uuid,
                "student_answer": params.answer,
                "reference_answer": self.reference_answer,
                "grade": grading['grade'],
                "score": grading['score'],
                "category": self.category,
                "difficulty": self.difficulty,
                "full_grading": grading['grading_response']
            },
            reward=grading['score'],
            finished=True
        )
