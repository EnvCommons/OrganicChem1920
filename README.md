# OrganicChem1920

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/OrganicChem1920)

## Description

**OrganicChem1920** is an environment for evaluating agents on organic chemistry questions derived from Arnold Frederik Holleman's "A Text-book of Organic Chemistry" (1920). Questions test procedural understanding and conceptual comprehension of organic chemistry, including historical 1920s terminology and practices. An LLM grader evaluates answers for conceptual correctness, accepting alternative nomenclature and awarding partial credit.

## Capabilities

- Answering organic chemistry questions requiring procedural understanding
- Reasoning about chemical mechanisms and principles
- Understanding historical chemistry nomenclature (1920s conventions)
- Demonstrating conceptual comprehension rather than rote recall

## Compute Requirements

OrganicChem1920 does not require a sandbox. It has minimal compute requirements.

## License

[MIT](https://opensource.org/license/mit).

## Tasks

There are 399 questions across three splits: train (279 tasks, 70%), validation (60 tasks, 15%), and test (60 tasks, 15%). Questions are loaded from a parquet file and span multiple categories (procedural, conceptual, reasoning, safety) and difficulty levels. Each question includes metadata: chapter, page reference, category, difficulty, and a context snippet from the source textbook.

## Reward Structure

This is a sparse reward environment with continuous scoring. The agent calls the `answer` tool once with its response, and the environment grades it using an LLM grader (gpt-5-mini). The grader assigns a score from 0.0 to 1.0 and a grade:

- **CORRECT** (score 0.85+): The answer demonstrates full conceptual understanding. Reward: the grader's score (0.85-1.0).
- **PARTIALLY_CORRECT** (score 0.7+): The answer shows partial but conceptually sound reasoning. Reward: the grader's score (0.7-0.85).
- **INCORRECT** (score < 0.7): The answer is conceptually wrong or missing key information. Reward: the grader's score (0.0-0.7).

Grading rules:
- Both IUPAC names, common names, and historical 1920s terminology are accepted.
- Evaluation focuses on conceptual correctness and understanding, not exact wording.
- Partial credit is awarded for incomplete but conceptually sound answers.
- Safety awareness is valued even if not in the reference answer.

## Data

Questions are sourced from Holleman's "A Text-book of Organic Chemistry" (1920), a public domain textbook. The dataset is stored as a parquet file on the OpenReward platform.

## Tools

Agents are given a single tool:

- `answer`: Submit an answer to the chemistry question. The answer is graded by the LLM grader against the reference answer. Returns the grade, score, and feedback. This tool can only be called once per task.

## Time Horizon

OrganicChem1920 is a single-turn environment. The agent receives a question and submits one answer. Each task requires exactly one tool call.

## Environment Difficulty

Questions span three difficulty levels: intermediate (238 tasks, 60%), advanced (130 tasks, 33%), and basic (31 tasks, 8%). Categories include conceptual (140), reasoning (121), procedural (121), and safety (17) questions.

## Other Environment Requirements

OrganicChem1920 requires an OpenAI API key (`OPENAI_API_KEY` secret) for LLM-based grading of answers.

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
