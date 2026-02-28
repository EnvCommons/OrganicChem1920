# OrganicChem1920

OpenReward environment for testing organic chemistry knowledge based on Holleman's "A Text-book of Organic Chemistry" (5th English ed., 1920).

## Source Textbook

**"A Text-book of Organic Chemistry"** by Arnold Frederik Holleman, translated by Andrew Jamieson Walker and Owen E. Mott. 5th English edition, published by John Wiley & Sons, 1920.

Available on Internet Archive: https://archive.org/details/atextbookorgani04hollgoog

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate dataset (requires OPENAI_API_KEY)
export OPENAI_API_KEY="your-key"
python generate_dataset.py

# Run server
python server.py

# Test with agent (in another terminal)
python test_agent.py
```

## Environment Details

- **Type:** Single-turn Q&A with LLM grading
- **Tool:** `answer` - submit answer to chemistry question
- **Grading:** Semantic evaluation using gpt-5-mini
- **Splits:** train (70%), validation (15%), test (15%)

## Question Categories

- **Procedural:** Understanding of laboratory procedures and techniques
- **Conceptual:** Explanation of chemical principles and mechanisms
- **Reasoning:** Prediction and justification of chemical outcomes
- **Safety:** Awareness of hazards and precautions
