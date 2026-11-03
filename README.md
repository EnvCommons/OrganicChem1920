# OrganicChem1920

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/OrganicChem1920)

## Description

**OrganicChem1920** is an environment for evaluating agents on organic chemistry questions derived from Arnold Frederik Holleman's "A Text-book of Organic Chemistry" (1920). Questions test procedural understanding and conceptual comprehension of organic chemistry, including historical 1920s terminology and practices. An LLM grader evaluates answers for conceptual correctness, accepting alternative nomenclature but grading strictly: only a fully correct answer scores.

## Capabilities

- Answering organic chemistry questions requiring procedural understanding
- Reasoning about chemical mechanisms and principles
- Understanding historical chemistry nomenclature (1920s conventions)
- Demonstrating conceptual comprehension rather than rote recall

## Compute Requirements

OrganicChem1920 does not require a sandbox. It has minimal compute requirements.

## License

[MIT](https://opensource.org/licenses/MIT).

## Tasks

There are 279 tasks in the `train` split, 60 in `validation` and 60 in `test` (399 questions in total). Questions are loaded from a parquet file and span multiple categories (procedural, conceptual, reasoning, safety) and difficulty levels. Each question includes metadata: chapter, page reference, category, difficulty, and a context snippet from the source textbook.

## Reward Structure

This is a sparse reward environment with a **binary** reward. The agent replies with its answer as an ordinary assistant message (no tool call); the message text is then graded by an LLM grader (gpt-5-mini), which returns one of three verdicts. Only the first earns anything:

- **CORRECT** → reward **1.0**. Every key point of the reference answer is present, with no chemical errors and no material omissions.
- **PARTIALLY_CORRECT** → reward **0.0**. Partially sound reasoning earns nothing; there is no partial credit.
- **INCORRECT** → reward **0.0**. The answer is wrong or missing key information.

An empty answer scores 0.0 without invoking the grader.

The grader also emits a numeric confidence score. It is recorded in the tool result's `metadata` for analysis but does **not** shape the reward: it is an uncalibrated free-choice number, so the discrete verdict is used instead.

Grading rules:
- Both IUPAC names, common names, and historical 1920s terminology are accepted.
- Evaluation focuses on conceptual correctness and understanding, not exact wording.
- Grading is strict: an answer that is incomplete, vague, or partially correct is not marked CORRECT.
- Safety awareness is valued even if not in the reference answer.

## Data

Questions are sourced from Holleman's "A Text-book of Organic Chemistry" (1920), a public domain textbook. Question and reference-answer pairs were generated from chunks of the textbook text by an OpenAI model (`generate_dataset.py`). The dataset is stored as a parquet file on the OpenReward platform.

## Tools

OrganicChem1920 exposes no tools to the agent. The rollout ends as soon as the model emits a plain assistant message; that message is graded by the LLM judge against the reference answer.

Because the whole assistant message becomes the graded text, avoid any preamble or commentary that isn't part of the answer.

## Time Horizon

OrganicChem1920 is a single-turn environment. The agent receives a question and replies with its answer as an ordinary message.

## Environment Difficulty

Questions span three difficulty levels: intermediate (238 tasks, 60%), advanced (130 tasks, 33%), and basic (31 tasks, 8%). Categories include conceptual (140), reasoning (121), procedural (121), and safety (17) questions.

## Other Environment Requirements

OrganicChem1920 requires an OpenAI API key for LLM-based grading of answers, passed in session secrets as `openai_api_key`. Pass the key via the secrets parameter when creating a session:

```python
async with environment.session(task=task, secrets={"openai_api_key": OPENAI_API_KEY}) as session:
```

## Safety

Agents in OrganicChem1920 are asked to answer chemistry questions from a historical textbook. The environment does not present direct safety risks, as agents only provide text answers with no access to external systems, tools, or the internet.

## Citations

```bibtex
@dataset{GROrganicChem1920,
  author    = {General Reasoning Inc. Team},
  title     = {OrganicChem1920},
  year      = {2026},
  publisher = {OpenReward},
  url       = {https://openreward.ai/GeneralReasoning/OrganicChem1920}
}
```

```bibtex
@book{holleman1920textbook,
  title={A Text-book of Organic Chemistry},
  author={Holleman, Arnold Frederik},
  year={1920},
  publisher={John Wiley \& Sons}
}
```
