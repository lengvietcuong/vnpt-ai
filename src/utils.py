import asyncio
import json
import logging
import re
import tempfile
from pathlib import Path
from textwrap import shorten

import aiohttp
import numpy as np

from src.config import (
    CALCULATION_PATTERN,
    CAPITAL_A_ASCII_CODE,
    CHEMISTRY_MOL_PATTERN,
    CHEMISTRY_PH_PATTERN,
    CHEMISTRY_TERMS_PATTERN,
    CURRENCY_PATTERN,
    DIGIT_PATTERN,
    ECONOMICS_TERMS_PATTERN,
    KNOWLEDGE_TERMS_PATTERN,
    LATEX_PATTERN,
    LAW_REFERENCE_PATTERN,
    MATH_TERMS_PATTERN,
    MAX_LOG_MESSAGE_LENGTH,
    MAX_READING_CHUNK_LENGTH,
    MIN_READING_CHUNK_LENGTH,
    PERCENTAGE_PATTERN,
    PHYSICS_TERMS_PATTERN,
    PHYSICS_UNITS_PATTERN,
    QUESTION_INDICATOR,
    READING_KEYWORD,
    READING_MIN_LENGTH,
    READING_PASSAGE_PATTERN,
    READING_SIMILARITY_THRESHOLD,
    READING_TOP_K,
    VARIABLE_PATTERN,
)
from src.models import Question, QuestionType, APICredentials


logger = logging.getLogger(__name__)


def load_questions(file_path: Path) -> list[Question]:
    """
    Load questions from a JSON file.

    Args:
        file_path (Path): Path to the JSON file

    Returns:
        list[Question]: List of Question objects
    """
    logger.info(f"Reading input data from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    logger.info(f"Loaded {len(data)} questions")
    return [Question(**item) for item in data]


def format_question(question: Question) -> str:
    """
    Format a question and choices into a prompt string.

    Args:
        question (Question): The question object
        choices (List[str]): List of answer choices

    Returns:
        str: The formatted prompt string
    """
    choices = "\n".join(
        f"{chr(CAPITAL_A_ASCII_CODE + i)}. {choice}"
        for i, choice in enumerate(question["choices"])
    )
    prompt = f"""Câu hỏi: {question["question"]}

Các đáp án:
{choices}"""

    return prompt


def classify_question(question: str) -> QuestionType:
    """
    Classify a question as MATH, READING, or KNOWLEDGE based on its content.

    Args:
        question: The question text to classify

    Returns:
        QuestionType enum value (MATH, READING, or KNOWLEDGE)
    """
    lowercase_question = question.lower()

    # Check READING
    if len(question) >= READING_MIN_LENGTH or READING_KEYWORD in lowercase_question:
        return QuestionType.READING

    # Check for math-specific patterns
    if any(
        re.search(pattern, question, re.IGNORECASE)
        for pattern in [
            LATEX_PATTERN,
            PHYSICS_UNITS_PATTERN,
            VARIABLE_PATTERN,
            CHEMISTRY_PH_PATTERN,
            CHEMISTRY_MOL_PATTERN,
        ]
    ):
        return QuestionType.MATH

    # Check for numbers combined with math indicators
    has_numbers = re.search(DIGIT_PATTERN, question)
    has_percentage = re.search(PERCENTAGE_PATTERN, lowercase_question)
    has_currency = re.search(CURRENCY_PATTERN, lowercase_question)
    has_calculation = re.search(CALCULATION_PATTERN, lowercase_question, re.IGNORECASE)
    has_domain_term = any(
        re.search(pattern, lowercase_question, re.IGNORECASE)
        for pattern in [
            PHYSICS_TERMS_PATTERN,
            CHEMISTRY_TERMS_PATTERN,
            ECONOMICS_TERMS_PATTERN,
            MATH_TERMS_PATTERN,
        ]
    )

    has_knowledge_context = re.search(
        KNOWLEDGE_TERMS_PATTERN, lowercase_question, re.IGNORECASE
    )
    has_law_reference = re.search(
        LAW_REFERENCE_PATTERN, lowercase_question, re.IGNORECASE
    )
    if (
        has_numbers
        and (has_percentage or has_currency or has_calculation or has_domain_term)
        and not (has_knowledge_context or has_law_reference)
    ):
        return QuestionType.MATH

    return QuestionType.KNOWLEDGE


def extract_answer(text: str, number_of_choices: int) -> str:
    """
    Extract the answer letter from the LLM response text (wrapped in double curly braces, e.g., {{A}}).
    If no braces found, returns the last capital letter.

    Args:
        text (str): The raw response text from the LLM
        number_of_choices (int): Number of answer choices (determines valid letters)

    Returns:
        str: The extracted answer letter (A, B, C, D, E, etc.)
    """
    if not text:
        raise ValueError("Cannot extract answer from empty model response")

    valid_letters = [chr(CAPITAL_A_ASCII_CODE + i) for i in range(number_of_choices)]
    valid_pattern = f"[{''.join(valid_letters)}]"

    matches = list(re.finditer(rf"\{{\{{({valid_pattern})\}}\}}", text))
    if matches:
        return matches[-1].group(1)

    letter_matches = re.findall(valid_pattern, text)
    if letter_matches:
        return letter_matches[-1]

    raise ValueError("Answer not found in model's response")


async def execute_python_code(code: str) -> str:
    """
    Execute Python code and return the result.

    Args:
        code (str): Python code to execute

    Returns:
        str: The result of the code execution.

    Raises:
        RuntimeError: If the Python code execution fails with an error.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
        temp_file.write(code)
        temp_file_path = Path(temp_file.name)

    logger.debug(f"Executing Python code:\n{format_log_message(code)}")

    try:
        process = await asyncio.create_subprocess_exec(
            "python3",
            str(temp_file_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error = stderr.decode().strip()
            raise RuntimeError(f"Python execution failed: {error}")

        result = stdout.decode()
        logger.debug(f"Python code output:\n{result}")

        return result
    finally:
        temp_file_path.unlink()


def format_log_message(message: str) -> str:
    return shorten(message, width=MAX_LOG_MESSAGE_LENGTH, placeholder="...")


def split_reading_context_and_question(question: str) -> tuple[str, str]:
    """
    Split a reading question into context and question parts.

    Args:
        question (str): The full reading question string

    Returns:
        tuple[str, str]: A tuple of (context, question) where context is everything before 'Câu hỏi: ' and question is everything after it
    """
    parts = question.split(QUESTION_INDICATOR, 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return "", question.strip()


async def extract_best_reading_passages(
    context: str,
    question: str,
    http_session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    api_credentials: APICredentials,
) -> str:
    """
    Extract the most relevant passages from a reading context based on semantic similarity.

    Args:
        context (str): The reading context containing multiple passages
        question (str): The question to compare passages against
        http_session (aiohttp.ClientSession): aiohttp ClientSession for making HTTP requests
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent requests
        api_credentials (APICredentials): API credentials for authentication

    Returns:
        str: The joined relevant passages
    """
    from src.vnpt_ai_api import embed

    passages = _chunk_reading_context(context)

    if not passages:
        return context

    # Embed all passages and the question concurrently
    embed_tasks = [
        embed(passage, http_session, semaphore, api_credentials) for passage in passages
    ] + [embed(question, http_session, semaphore, api_credentials)]

    embeddings = await asyncio.gather(*embed_tasks)
    passage_embeddings = embeddings[:-1]
    question_embedding = embeddings[-1]

    # Filter passages by similarity threshold
    relevant_passages = _filter_passages_by_similarity(
        passages, passage_embeddings, question_embedding
    )

    return "\n\n".join(relevant_passages)


def _chunk_reading_context(context: str) -> list[str]:
    """
    Chunk a reading context into passages using pattern matching or smart chunking.

    Args:
        context (str): The reading context to chunk

    Returns:
        list[str]: List of passage chunks
    """
    # Try to chunk by the "-- Đoạn văn x --" or "-- Document x --" pattern
    passages = re.split(READING_PASSAGE_PATTERN, context, flags=re.IGNORECASE)
    passages = [p.strip() for p in passages if p.strip()]

    # If pattern splitting worked well, return those passages
    if len(passages) > 1:
        return passages

    # Fall back to smart chunking
    return _smart_chunk_context(context)


def _smart_chunk_context(context: str) -> list[str]:
    """
    Perform smart chunking on context using double newlines and length constraints.

    Args:
        context (str): The context to chunk

    Returns:
        list[str]: List of smartly chunked passages
    """
    passages = context.split("\n\n")
    passages = [p.strip() for p in passages if p.strip()]

    filtered_passages = []
    for passage in passages:
        if len(passage) < MIN_READING_CHUNK_LENGTH:
            # Merge with previous if available
            if filtered_passages:
                filtered_passages[-1] = filtered_passages[-1] + "\n\n" + passage
            else:
                filtered_passages.append(passage)
        elif len(passage) > MAX_READING_CHUNK_LENGTH:
            # Split long passages by sentence boundaries
            filtered_passages.extend(_split_long_passage(passage))
        else:
            filtered_passages.append(passage)

    return filtered_passages


def _split_long_passage(passage: str) -> list[str]:
    """
    Split a long passage into smaller chunks at sentence boundaries.

    Args:
        passage (str): The long passage to split

    Returns:
        list[str]: List of smaller passage chunks
    """
    sentences = re.split(r"([.!?]\s+)", passage)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if not current_chunk:
            current_chunk = sentence
            continue

        if len(current_chunk) + len(sentence) > MAX_READING_CHUNK_LENGTH:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1 (list[float]): First vector
        vec2 (list[float]): Second vector

    Returns:
        float: Cosine similarity score
    """
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    return float(
        np.dot(vec1_np, vec2_np) / (np.linalg.norm(vec1_np) * np.linalg.norm(vec2_np))
    )


def _filter_passages_by_similarity(
    passages: list[str],
    passage_embeddings: list[list[float]],
    question_embedding: list[float],
) -> list[str]:
    """
    Filter passages based on cosine similarity threshold, then return top K.

    Args:
        passages (list[str]): List of passage texts
        passage_embeddings (list[list[float]]): List of passage embeddings
        question_embedding (list[float]): Question embedding

    Returns:
        list[str]: List of relevant passages (up to K passages)
    """
    passage_similarities = [
        (passage, _cosine_similarity(passage_embedding, question_embedding))
        for passage, passage_embedding in zip(passages, passage_embeddings)
    ]

    filtered_passages = [
        (passage, similarity)
        for passage, similarity in passage_similarities
        if similarity >= READING_SIMILARITY_THRESHOLD
    ]

    filtered_passages.sort(key=lambda x: x[1], reverse=True)
    top_passages = filtered_passages[:READING_TOP_K]

    relevant_passages = [passage for passage, _ in top_passages]

    return relevant_passages
