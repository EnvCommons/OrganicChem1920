# Data Upload Requirements for OrganicChem1920

## Overview

This environment requires a parquet dataset generated from Holleman's "A Text-book of Organic Chemistry" (1920).

## File Required

**organicchem1920_questions.parquet**
- Size: ~2-5 MB (approximately 400-500 questions)
- Format: Apache Parquet
- Location: `/orwd_data/` (production) or local directory (development)

## Dataset Schema

The parquet file must contain the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `uuid` | string | Unique question identifier (UUID format) |
| `question` | string | The question text |
| `answer` | string | Reference answer with reasoning (2-4 sentences) |
| `category` | string | One of: procedural, conceptual, reasoning, safety |
| `difficulty` | string | One of: basic, intermediate, advanced |
| `page_reference` | int64 | Original textbook page number |
| `context_snippet` | string | Short excerpt from textbook (context) |
| `chapter` | string | Chapter title or number |
| `split` | string | One of: train, validation, test |

## Generation Instructions

### Local Development

1. **Install dependencies:**
   ```bash
   pip install PyMuPDF openai pandas pyarrow scikit-learn requests
   ```

2. **Set OpenAI API key:**
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

3. **Run generation script:**
   ```bash
   python generate_dataset.py
   ```

   This will:
   - Download Holleman's 1920 textbook PDF from Internet Archive
   - Extract text by chapters
   - Generate ~400-500 questions using GPT-5.2
   - Create stratified train/val/test splits (70/15/15)
   - Save to `organicchem1920_questions.parquet`

4. **Verify output:**
   ```bash
   python -c "import pandas as pd; df = pd.read_parquet('organicchem1920_questions.parquet'); print(df.info()); print(df.head())"
   ```

### Production Upload

Once generated locally, the parquet file needs to be uploaded to OpenReward cloud storage:

1. **Go to OpenReward Storage:**
   - Visit https://openreward.ai/storage
   - Navigate to your namespace

2. **Upload the file:**
   - Upload `organicchem1920_questions.parquet` to the root directory
   - Verify the file appears at `/orwd_data/organicchem1920_questions.parquet`

3. **Verify upload:**
   - File size should be 2-5 MB
   - Contains 400-500 rows
   - All required columns present

## Dataset Statistics

**Expected distribution:**
- **Total questions:** 400-500
- **Train split:** ~280-350 (70%)
- **Validation split:** ~60-75 (15%)
- **Test split:** ~60-75 (15%)

**Category breakdown:**
- Procedural: ~35%
- Conceptual: ~35%
- Reasoning: ~20%
- Safety: ~10%

**Difficulty levels:**
- Basic: ~30%
- Intermediate: ~50%
- Advanced: ~20%

## Troubleshooting

**FileNotFoundError when running server.py:**
- Ensure the parquet file exists at `./organicchem1920_questions.parquet` (local) or `/orwd_data/organicchem1920_questions.parquet` (production)
- Run `python generate_dataset.py` if file is missing locally

**Generation script fails:**
- Check that `OPENAI_API_KEY` is set
- Ensure PDF downloads successfully from Internet Archive
- Check API rate limits and quota

**Invalid schema errors:**
- Verify all required columns are present
- Check data types match schema above
- Ensure no null values in critical columns

## Questions?

If you encounter issues with dataset generation or upload, please check:
1. The generation script output for errors
2. Parquet file schema matches requirements
3. File permissions and path accessibility
