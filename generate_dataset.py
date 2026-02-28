"""
generate_dataset.py - Question Generation Script for OrganicChem1920

This script:
1. Downloads Holleman's "A Text-book of Organic Chemistry" (1920) PDF
2. Extracts text using PyMuPDF
3. Generates procedural/conceptual questions using GPT-4.1
4. Creates stratified train/val/test splits
5. Saves to organicchem1920_questions.parquet

Usage:
    export OPENAI_API_KEY="your-key-here"
    python generate_dataset.py
"""

import asyncio
import json
import os
import re
import uuid
from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF
import pandas as pd
import requests
from openai import AsyncOpenAI
from pydantic import BaseModel
from sklearn.model_selection import train_test_split


# ============================================================================
# Configuration
# ============================================================================

PDF_URL = "https://archive.org/download/atextbookorgani04hollgoog/atextbookorgani04hollgoog_text.pdf"
LOCAL_PDF = "holleman_organic_chemistry_1920.pdf"
OUTPUT_PARQUET = "organicchem1920_questions.parquet"
PROGRESS_PARQUET = "organicchem1920_progress.parquet"  # Incremental save file

# Question generation targets
TARGET_TOTAL_QUESTIONS = 400
QUESTIONS_PER_CHUNK = 3  # Generate 3-5 questions per text chunk


# ============================================================================
# Question Generation Prompt
# ============================================================================

QUESTION_GENERATION_PROMPT = """You are an expert organic chemistry educator analyzing Holleman's "A Text-book of Organic Chemistry" (1920).

Generate {num_questions} high-quality questions from this text that test PROCEDURAL UNDERSTANDING and CONCEPTUAL COMPREHENSION, NOT just factual recall.

Text excerpt (Page {page_num}, Chapter "{chapter}"):
```
{text_chunk}
```

For each question, provide:
1. **Question**: Clear, specific question requiring understanding of procedures, mechanisms, or concepts
2. **Answer**: Detailed answer with reasoning (2-4 sentences)
3. **Category**: One of [procedural, conceptual, reasoning, safety]
4. **Difficulty**: One of [basic, intermediate, advanced]

Question types to favor:
- **Procedural**: "Why is it necessary to..." "What would happen if..." "How does the procedure ensure..."
- **Conceptual**: "Explain why..." "What is the chemical basis for..." "Compare and contrast..."
- **Reasoning**: "Predict the outcome if..." "Justify the choice of..." "What alternative approach..."
- **Safety**: "What precautions are necessary when..." "Why is this procedure potentially hazardous..."

Please use the language of the text instead of anachronistic modern English.

Avoid simple factual recall like "What temperature is used for..." or "Name the compound..."

Please use the language of the text instead of anachronistic modern English.

Questions should be self-contained, i.e. not rely on the answers of previous questions or any other outside content.

Return as a JSON object with a "questions" array:
{{
  "questions": [
    {{
      "question": "...",
      "answer": "...",
      "category": "procedural",
      "difficulty": "intermediate"
    }},
    ...
  ]
}}

Generate exactly {num_questions} questions."""


# ============================================================================
# Pydantic Models
# ============================================================================

class Question(BaseModel):
    question: str
    answer: str
    category: str
    difficulty: str


class QuestionSet(BaseModel):
    questions: List[Question]


# ============================================================================
# PDF Download and Text Extraction
# ============================================================================

def download_pdf() -> None:
    """Download PDF if not exists."""
    if Path(LOCAL_PDF).exists():
        print(f"PDF already exists at {LOCAL_PDF}")
        return

    print(f"Downloading PDF from {PDF_URL}...")
    response = requests.get(PDF_URL, stream=True)
    response.raise_for_status()

    with open(LOCAL_PDF, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"PDF downloaded to {LOCAL_PDF}")


def extract_text_by_chapters() -> List[Dict[str, Any]]:
    """
    Extract text organized by chapters.

    Returns:
        List of dicts with: {chapter, page_start, page_end, text}
    """
    print("Extracting text from PDF...")
    doc = fitz.open(LOCAL_PDF)

    chapters = []
    current_chapter = {
        "chapter": "Introduction",
        "page_start": 1,
        "pages": []
    }

    # Simple heuristic: detect chapter headings (all caps, short lines)
    # For a production system, you'd want more sophisticated detection
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        # Store page text
        current_chapter["pages"].append({
            "page_num": page_num + 1,
            "text": text
        })

        # Simple chapter detection: look for "CHAPTER" keyword or numbered chapters
        # This is a heuristic and may need adjustment based on actual PDF structure
        lines = text.split('\n')
        for line in lines[:10]:  # Check first 10 lines of page
            if re.match(r'^\s*(CHAPTER|Chapter)\s+[IVXLCDM0-9]+', line, re.IGNORECASE):
                # Found new chapter
                if len(current_chapter["pages"]) > 0:
                    current_chapter["page_end"] = page_num
                    current_chapter["text"] = "\n\n".join(
                        p["text"] for p in current_chapter["pages"]
                    )
                    chapters.append(current_chapter)

                # Start new chapter
                current_chapter = {
                    "chapter": line.strip(),
                    "page_start": page_num + 1,
                    "pages": []
                }
                break

    # Add last chapter
    if len(current_chapter["pages"]) > 0:
        current_chapter["page_end"] = len(doc)
        current_chapter["text"] = "\n\n".join(
            p["text"] for p in current_chapter["pages"]
        )
        chapters.append(current_chapter)

    doc.close()

    print(f"Extracted {len(chapters)} chapters")
    return chapters


def chunk_text(text: str, chunk_size: int = 3000) -> List[str]:
    """
    Split text into chunks of approximately chunk_size characters.
    Tries to break on paragraph boundaries.
    """
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_size = 0

    for para in paragraphs:
        para_size = len(para)
        if current_size + para_size > chunk_size and current_chunk:
            # Finalize current chunk
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_size = para_size
        else:
            current_chunk.append(para)
            current_size += para_size

    # Add last chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks


# ============================================================================
# Progress Management
# ============================================================================

def load_progress() -> tuple[pd.DataFrame, set[str]]:
    """
    Load existing progress from parquet file.

    Returns:
        Tuple of (DataFrame of existing questions, set of processed chunk_ids)
    """
    if Path(PROGRESS_PARQUET).exists():
        df = pd.read_parquet(PROGRESS_PARQUET)
        processed_chunks = set(df['chunk_id'].unique())
        print(f"Loaded {len(df)} existing questions from {len(processed_chunks)} chunks")
        return df, processed_chunks
    return pd.DataFrame(), set()


def save_progress(questions: List[Dict[str, Any]]) -> None:
    """Save current progress to parquet file."""
    if not questions:
        return
    df = pd.DataFrame(questions)
    df.to_parquet(PROGRESS_PARQUET, index=False)
    print(f"  [Saved {len(questions)} questions to {PROGRESS_PARQUET}]")


def make_chunk_id(chapter: str, chunk_index: int) -> str:
    """Create a unique identifier for a chunk."""
    # Sanitize chapter name for consistent IDs
    sanitized_chapter = re.sub(r'[^\w\s]', '', chapter).strip().lower()[:50]
    return f"{sanitized_chapter}_chunk{chunk_index}"


# ============================================================================
# Question Generation
# ============================================================================

async def generate_questions_for_chunk(
    client: AsyncOpenAI,
    text_chunk: str,
    chapter: str,
    page_num: int,
    chunk_id: str,
    num_questions: int = QUESTIONS_PER_CHUNK
) -> List[Dict[str, Any]]:
    """
    Generate questions for a text chunk using GPT-4.1.

    Returns:
        List of question dicts
    """
    prompt = QUESTION_GENERATION_PROMPT.format(
        num_questions=num_questions,
        text_chunk=text_chunk[:3000],  # Limit to 3000 chars
        page_num=page_num,
        chapter=chapter
    )

    try:
        response = await client.chat.completions.create(
            model="gpt-5.2",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        response_text = response.choices[0].message.content or "{}"
        data = json.loads(response_text)

        # Validate with Pydantic
        question_set = QuestionSet.model_validate(data)

        # Add metadata
        questions = []
        for q in question_set.questions:
            questions.append({
                "uuid": str(uuid.uuid4()),
                "question": q.question,
                "answer": q.answer,
                "category": q.category,
                "difficulty": q.difficulty,
                "page_reference": page_num,
                "context_snippet": text_chunk[:200].strip(),
                "chapter": chapter,
                "chunk_id": chunk_id
            })

        return questions

    except Exception as e:
        print(f"Error generating questions for page {page_num}: {e}")
        return []


async def generate_all_questions(chapters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate questions for all chapters, with incremental saving and resume support.

    Returns:
        List of all generated questions
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable must be set")

    client = AsyncOpenAI(api_key=api_key)

    # Load existing progress
    existing_df, processed_chunks = load_progress()
    all_questions = existing_df.to_dict('records') if len(existing_df) > 0 else []

    # Calculate how many chunks to process
    total_chunks_needed = TARGET_TOTAL_QUESTIONS // QUESTIONS_PER_CHUNK

    print(f"Generating approximately {TARGET_TOTAL_QUESTIONS} questions...")
    print(f"Target: {total_chunks_needed} text chunks across {len(chapters)} chapters")
    if processed_chunks:
        print(f"Resuming: {len(processed_chunks)} chunks already processed, {len(all_questions)} questions so far")

    chunks_processed = len(processed_chunks)
    new_questions_count = 0

    for chapter in chapters:
        if chunks_processed >= total_chunks_needed:
            break

        print(f"\nProcessing chapter: {chapter['chapter']}")

        # Chunk the chapter text
        chunks = chunk_text(chapter["text"])

        for i, text_chunk in enumerate(chunks):
            if chunks_processed >= total_chunks_needed:
                break

            # Skip very short chunks
            if len(text_chunk.strip()) < 500:
                continue

            # Create chunk ID and check if already processed
            chunk_id = make_chunk_id(chapter["chapter"], i)
            if chunk_id in processed_chunks:
                print(f"  Chunk {i+1}/{len(chunks)} - already processed, skipping")
                continue

            page_num = chapter["page_start"] + (i * len(chunks) // len(chapter["pages"]))

            print(f"  Chunk {i+1}/{len(chunks)} (page ~{page_num})...", end=" ")

            questions = await generate_questions_for_chunk(
                client,
                text_chunk,
                chapter["chapter"],
                page_num,
                chunk_id,
                num_questions=QUESTIONS_PER_CHUNK
            )

            all_questions.extend(questions)
            processed_chunks.add(chunk_id)
            chunks_processed += 1
            new_questions_count += len(questions)

            print(f"Generated {len(questions)} questions (total: {len(all_questions)})")

            # Save progress after each successful chunk
            save_progress(all_questions)

            # Rate limiting: small delay between requests
            await asyncio.sleep(1)

    print(f"\nGeneration complete!")
    print(f"  New questions this run: {new_questions_count}")
    print(f"  Total questions: {len(all_questions)}")
    return all_questions


# ============================================================================
# Data Validation and Splitting
# ============================================================================

def create_splits(questions: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create stratified train/val/test splits.

    Falls back to simpler stratification if some strata have too few samples.

    Returns:
        DataFrame with split column added
    """
    df = pd.DataFrame(questions)

    # Drop chunk_id column if present (only used for progress tracking)
    if 'chunk_id' in df.columns:
        df = df.drop(columns=['chunk_id'])

    # Try stratification strategies in order of preference
    stratify_options = [
        ('category + difficulty', df['category'] + '_' + df['difficulty']),
        ('category only', df['category']),
        ('none', None),
    ]

    train_df = None
    temp_df = None

    for strat_name, strat_col in stratify_options:
        # Check if all strata have at least 2 members (needed for split)
        if strat_col is not None:
            min_count = strat_col.value_counts().min()
            if min_count < 2:
                print(f"Stratification by {strat_name}: skipping (min group size = {min_count})")
                continue

        try:
            print(f"Stratification by {strat_name}...")
            # First split: 70% train, 30% temp
            train_df, temp_df = train_test_split(
                df,
                test_size=0.3,
                stratify=strat_col,
                random_state=42
            )
            break  # Success
        except ValueError as e:
            print(f"Stratification by {strat_name}: failed ({e})")
            continue

    if train_df is None:
        raise ValueError("Could not create splits - dataset too small or imbalanced")

    # Second split: 50/50 of temp = 15% val, 15% test
    # Try progressively simpler stratification for smaller subset
    second_split_options = [
        ('category + difficulty', (df['category'] + '_' + df['difficulty']).loc[temp_df.index]),
        ('category only', df['category'].loc[temp_df.index]),
        ('none', None),
    ]

    val_df = None
    for strat_name, temp_strat in second_split_options:
        if temp_strat is not None:
            min_count = temp_strat.value_counts().min()
            if min_count < 2:
                continue
        try:
            val_df, test_df = train_test_split(
                temp_df,
                test_size=0.5,
                stratify=temp_strat,
                random_state=42
            )
            print(f"Second split stratification: {strat_name}")
            break
        except ValueError:
            continue

    if val_df is None:
        # Final fallback - no stratification
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            stratify=None,
            random_state=42
        )

    # Assign split labels
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df['split'] = 'train'
    val_df['split'] = 'validation'
    test_df['split'] = 'test'

    # Combine
    result_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    print(f"\nSplit distribution:")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

    print(f"\nCategory distribution:")
    print(result_df.groupby(['split', 'category']).size().unstack(fill_value=0))

    print(f"\nDifficulty distribution:")
    print(result_df.groupby(['split', 'difficulty']).size().unstack(fill_value=0))

    return result_df


def validate_dataset(df: pd.DataFrame) -> None:
    """Validate dataset quality."""
    print("\n" + "="*60)
    print("DATASET VALIDATION")
    print("="*60)

    # Check required columns
    required_columns = [
        'uuid', 'question', 'answer', 'category', 'difficulty',
        'page_reference', 'context_snippet', 'chapter', 'split'
    ]

    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print("All required columns present")

    # Check for nulls
    null_counts = df.isnull().sum()
    if null_counts.any():
        print(f"Warning: Null values found:\n{null_counts[null_counts > 0]}")
    else:
        print("No null values")

    # Check answer length
    short_answers = df[df['answer'].str.len() < 50]
    if len(short_answers) > 0:
        print(f"Warning: {len(short_answers)} answers are shorter than 50 characters")
    else:
        print("All answers have adequate length")

    # Check for duplicates
    duplicates = df.duplicated(subset=['question'], keep=False)
    if duplicates.any():
        print(f"Warning: {duplicates.sum()} duplicate questions found")
    else:
        print("No duplicate questions")

    print(f"\nTotal questions: {len(df)}")
    print(f"Categories: {df['category'].unique().tolist()}")
    print(f"Difficulties: {df['difficulty'].unique().tolist()}")
    print(f"Page range: {df['page_reference'].min()} - {df['page_reference'].max()}")


# ============================================================================
# Main Execution
# ============================================================================

async def main():
    """Main execution flow."""
    print("="*60)
    print("OrganicChem1920 Dataset Generation")
    print("Holleman's 'A Text-book of Organic Chemistry' (1920)")
    print("="*60)

    # Step 1: Download PDF
    download_pdf()

    # Step 2: Extract text
    chapters = extract_text_by_chapters()

    # Step 3: Generate questions
    questions = await generate_all_questions(chapters)

    if len(questions) == 0:
        print("ERROR: No questions generated. Check PDF extraction and API key.")
        return

    # Step 4: Create splits
    df = create_splits(questions)

    # Step 5: Validate
    validate_dataset(df)

    # Step 6: Save to parquet
    print(f"\nSaving to {OUTPUT_PARQUET}...")
    df.to_parquet(OUTPUT_PARQUET, index=False)

    file_size = Path(OUTPUT_PARQUET).stat().st_size / (1024 * 1024)
    print(f"Saved {len(df)} questions ({file_size:.2f} MB)")

    # Step 7: Display sample questions
    print("\n" + "="*60)
    print("SAMPLE QUESTIONS")
    print("="*60)

    for i, row in df.sample(3).iterrows():
        print(f"\nQuestion #{i+1} ({row['category']}, {row['difficulty']}):")
        print(f"  Q: {row['question']}")
        print(f"  A: {row['answer'][:150]}...")
        print(f"  Page: {row['page_reference']}, Chapter: {row['chapter']}")

    print("\n" + "="*60)
    print("GENERATION COMPLETE!")
    print("="*60)
    print(f"Output file: {OUTPUT_PARQUET}")
    print("\nNext steps:")
    print("1. Review sample questions for quality")
    print("2. Test the environment with: python server.py")
    print("3. Upload parquet file per DATA_UPLOAD.md instructions")


if __name__ == "__main__":
    asyncio.run(main())
