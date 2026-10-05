"""Unit tests for OrganicChem1920 grading.

The LLM grader is replaced by a fake client that returns a fixed reply, so the
tests check how a verdict becomes a reward and what the result shows the
agent. The question parquet is not in the repo; when it is absent, the module
is loaded next to a small fixture parquet instead. Run with:
    uv run --no-project --with-requirements requirements.txt --with pytest \
        --with pytest-asyncio python -m pytest tests.py
"""

import atexit
import importlib.util
import json
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

MODULE = "organicchem1920"
PARQUET = "organicchem1920_questions.parquet"
HERE = Path(__file__).parent

FIXTURE_QUESTIONS = [
    {
        "uuid": "00000000-0000-4000-8000-000000000001",
        "question": "Why is the crude ester washed with sodium carbonate solution before drying?",
        "answer": "Sodium carbonate neutralises and removes the residual mineral acid and "
                  "unchanged carboxylic acid as water-soluble salts, so they do not "
                  "contaminate the ester or catalyse its hydrolysis during distillation.",
    },
    {
        "uuid": "00000000-0000-4000-8000-000000000002",
        "question": "What is the purpose of adding a few pieces of porous pot to the distilling flask?",
        "answer": "The porous fragments release small air bubbles that act as nuclei for "
                  "boiling, which prevents superheating and violent bumping of the liquid.",
    },
    {
        "uuid": "00000000-0000-4000-8000-000000000003",
        "question": "Why is the bromination of benzene carried out in the presence of iron filings?",
        "answer": "Iron reacts with bromine to form ferric bromide, a halogen carrier that "
                  "polarises the bromine molecule and allows substitution of the aromatic "
                  "ring, which bromine alone attacks only very slowly.",
    },
]


def load_env_module():
    if (HERE / PARQUET).exists() or os.path.exists("/orwd_data"):
        import organicchem1920
        return organicchem1920
    tmp = Path(tempfile.mkdtemp())
    atexit.register(shutil.rmtree, tmp, True)
    shutil.copy(HERE / f"{MODULE}.py", tmp / f"{MODULE}.py")
    rows = [
        {**q, "category": "reasoning", "difficulty": "intermediate", "page_reference": 1,
         "context_snippet": "", "chapter": "1", "split": "test"}
        for q in FIXTURE_QUESTIONS
    ]
    pd.DataFrame(rows).to_parquet(tmp / PARQUET)
    spec = importlib.util.spec_from_file_location(MODULE, tmp / f"{MODULE}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


env_module = load_env_module()
TASKS = [
    {"uuid": row["uuid"], "reference": row["answer"]}
    for _, row in env_module.QUESTIONS_DF.head(3).iterrows()
]
SECRETS = {"openai_api_key": "test-openai-key"}
RESULT_METADATA_KEYS = {"uuid", "student_answer", "grade", "score", "reward", "category", "difficulty"}


class FakeCompletions:
    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[str] = []

    async def create(self, model, messages):
        self.prompts.append(messages[0]["content"])
        message = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def grader_reply(task: dict, grade: str, score: float) -> str:
    # Real grader replies quote or paraphrase the reference in their analysis.
    return (
        f"**Analysis:**\nThe reference answer states: {task['reference']} "
        f"The student's answer is compared with this.\n\n"
        f"**Score:** {score}\n\n**Grade:** {grade}\n\n"
        f"**Feedback:**\nRecall that {task['reference']}"
    )


def make_env(task: dict, reply: str):
    env = env_module.OrganicChem1920(task_spec={"uuid": task["uuid"]}, secrets=SECRETS)
    completions = FakeCompletions(reply)
    env.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return env, completions


def visible_to_agent(result) -> str:
    """Everything the agent can see in a result, minus its own submitted answer."""
    metadata = {k: v for k, v in (result.metadata or {}).items() if k != "student_answer"}
    return "\n".join(b.text for b in result.blocks) + "\n" + json.dumps(metadata, ensure_ascii=False)


async def submit(task: dict, answer: str, grade: str, score: float):
    env, completions = make_env(task, grader_reply(task, grade, score))
    result = await env.answer(env_module.AnswerInput(answer=answer))
    return result, completions


@pytest.mark.asyncio
@pytest.mark.parametrize("task", TASKS, ids=lambda t: t["uuid"])
async def test_correct_verdict_scores_one(task):
    result, completions = await submit(task, task["reference"], "CORRECT", 0.95)
    assert result.reward == 1.0
    assert result.finished is True
    assert result.metadata["grade"] == "CORRECT"
    assert result.metadata["score"] == 0.95
    # The grader itself still sees the reference answer.
    assert task["reference"] in completions.prompts[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("task", TASKS, ids=lambda t: t["uuid"])
@pytest.mark.parametrize("grade", ["PARTIALLY_CORRECT", "INCORRECT"])
async def test_other_verdicts_score_zero(task, grade):
    result, _ = await submit(task, "It removes water.", grade, 0.3)
    assert result.reward == 0.0
    assert result.finished is True
    assert result.metadata["grade"] == grade


@pytest.mark.asyncio
@pytest.mark.parametrize("task", TASKS, ids=lambda t: t["uuid"])
@pytest.mark.parametrize("kind", ["gold", "wrong"])
async def test_result_does_not_reveal_reference(task, kind):
    if kind == "gold":
        result, _ = await submit(task, task["reference"], "CORRECT", 1.0)
    else:
        result, _ = await submit(task, "It removes water.", "INCORRECT", 0.0)

    assert set(result.metadata) == RESULT_METADATA_KEYS
    visible = visible_to_agent(result)
    assert task["reference"] not in visible
    assert "The reference answer states" not in visible


@pytest.mark.asyncio
@pytest.mark.parametrize("task", TASKS, ids=lambda t: t["uuid"])
async def test_grader_failure_message_does_not_reveal_reference(task):
    # No verdict keyword anywhere, so every attempt is unusable and grading raises.
    env, completions = make_env(task, f"Unable to decide. The reference says: {task['reference']}")
    with pytest.raises(RuntimeError) as excinfo:
        await env.answer(env_module.AnswerInput(answer="It removes water."))
    assert len(completions.prompts) == env_module.GRADER_MAX_ATTEMPTS
    assert task["reference"] not in str(excinfo.value)


@pytest.mark.asyncio
async def test_empty_answer_scores_zero_without_grader():
    task = TASKS[0]
    env, completions = make_env(task, grader_reply(task, "CORRECT", 1.0))
    result = await env.answer(env_module.AnswerInput(answer="   "))
    assert result.reward == 0.0
    assert completions.prompts == []


@pytest.mark.asyncio
async def test_repeat_submission_is_penalised_and_not_regraded():
    task = TASKS[0]
    env, completions = make_env(task, grader_reply(task, "CORRECT", 1.0))
    first = await env.answer(env_module.AnswerInput(answer=task["reference"]))
    second = await env.answer(env_module.AnswerInput(answer=task["reference"]))
    assert first.reward == 1.0
    assert second.reward == env_module.REPEAT_SUBMISSION_PENALTY
    assert second.metadata == {"already_submitted": True, "submission_count": 1}
    assert len(completions.prompts) == 1
