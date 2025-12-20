import asyncio
import csv
import logging
import re
from typing import List, TextIO

import aiohttp
from tqdm.asyncio import tqdm

from src.config import (
    FALLBACK_ANSWER,
    LLM_SETTINGS,
    QUESTION_INDICATOR,
    READING_CONTEXT_PREFIX,
    REFUSE_PATTERN,
)
from src.models import APICredentials, Question, QuestionType
from src.rag.chunk_retrieval import semantic_search
from src.utils import (
    classify_question,
    extract_answer,
    extract_best_reading_passages,
    format_question,
    split_reading_context_and_question,
)
import src.vnpt_ai_api as vnpt_ai_api
from src.vnpt_ai_api import embed


logger = logging.getLogger(__name__)


CAPITAL_A_INDEX = 65


async def _retry_with_filtered_passages(
    question: Question,
    llm_settings,
    http_session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    api_credentials: APICredentials,
) -> str | None:
    """
    Retry processing a reading question with filtered passages to remove potentially toxic content.

    Args:
        question (Question): Question data dictionary
        llm_settings: LLM configuration settings
        http_session (aiohttp.ClientSession): aiohttp ClientSession
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent requests
        api_credentials (APICredentials): API credentials for authentication

    Returns:
        str | None: The raw response from LLM or None if retry fails
    """
    logger.info(
        f"Attempting to filter passages for reading question {question['qid']} after error"
    )

    try:
        # Split into context and question
        context, question_text = split_reading_context_and_question(
            question["question"]
        )

        if not context:
            return None

        # Extract best passages based on similarity
        filtered_context = await extract_best_reading_passages(
            context=context,
            question=question_text,
            http_session=http_session,
            semaphore=semaphore,
            api_credentials=api_credentials,
        )

        # Reconstruct the reading question with filtered passages
        new_question_text = f"{READING_CONTEXT_PREFIX}\n{filtered_context}\n\n{QUESTION_INDICATOR}{question_text}"

        # Create a new prompt with filtered content
        new_question = question.copy()
        new_question["question"] = new_question_text
        new_prompt = format_question(new_question)

        logger.info(f"Retrying question {question['qid']} with filtered passages")

        # Retry with filtered content
        messages = await vnpt_ai_api.get_llm_response(
            prompt=new_prompt,
            http_session=http_session,
            semaphore=semaphore,
            llm_settings=llm_settings,
            api_credentials=api_credentials,
            query_embedding=None,
        )
        return messages[-1].get("content")

    except Exception as retry_error:
        logger.error(
            f"Error during retry with filtered passages for question {question['qid']}: {retry_error}.",
            exc_info=True,
        )
        return None


async def process_question(
    question: Question,
    csv_writer: csv.DictWriter,
    csv_file: TextIO,
    http_session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    api_credentials: APICredentials,
) -> None:
    """
    Answer a question and write the result to a CSV file.

    Args:
        question (Question): Question data dictionary
        csv_writer (csv.DictWriter): CSV writer object
        csv_file (TextIO): File object for the CSV output
        http_session (aiohttp.ClientSession): aiohttp ClientSession
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent requests
        api_credentials (APICredentials): API credentials for authentication
    """
    logger.debug(f"Processing question {question['qid']}")

    question_type = classify_question(question["question"])
    logger.debug(f"Classified question {question['qid']} as {question_type.name}")

    prompt = format_question(question)
    llm_settings = LLM_SETTINGS[question_type]

    # Perform initial semantic search when faced with knowledge-based questions
    query_embedding = None
    if question_type == QuestionType.KNOWLEDGE:
        query_embedding = await embed(
            text=prompt,
            http_session=http_session,
            semaphore=semaphore,
            api_credentials=api_credentials,
        )
        relevant_info = await semantic_search(query_embedding=query_embedding)
        if relevant_info:
            prompt += f"\n\n---\n\nThông tin có thể liên quan:\n{relevant_info}"

    try:
        messages = await vnpt_ai_api.get_llm_response(
            prompt=prompt,
            http_session=http_session,
            semaphore=semaphore,
            llm_settings=llm_settings,
            api_credentials=api_credentials,
            query_embedding=query_embedding,
        )
        raw_response = messages[-1].get("content")
    except Exception as error:
        logger.error(
            f"Error getting LLM response for question {question['qid']}: {error}.",
            exc_info=True,
        )
        if question_type != QuestionType.READING:
            raw_response = None
        else:
            raw_response = await _retry_with_filtered_passages(
                question=question,
                llm_settings=llm_settings,
                http_session=http_session,
                semaphore=semaphore,
                api_credentials=api_credentials,
            )

    try:
        answer = extract_answer(raw_response, len(question["choices"]))
    except ValueError as error:
        logger.error(
            f"Error extracting answer for question {question['qid']}: {error}. Using fallback answer: {FALLBACK_ANSWER}.",
            exc_info=True,
        )
        answer = FALLBACK_ANSWER
        for i, choice in enumerate(
            question["choices"], start=CAPITAL_A_INDEX
        ):  # Fallback to refusal
            if re.search(REFUSE_PATTERN, choice, re.IGNORECASE):
                answer = chr(i)
                break

    logger.debug(f"Saving answer for question {question['qid']}: {answer}")

    csv_writer.writerow({"qid": question["qid"], "answer": answer})
    csv_file.flush()

    logger.debug(f"Processed question {question['qid']}")


async def process_questions(
    questions: List[Question],
    csv_file: TextIO,
    csv_writer: csv.DictWriter,
    http_session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    api_credentials: APICredentials,
) -> None:
    """
    Answer all questions concurrently write results continuously.

    Args:
        questions (List[Question]): List of Question objects
        csv_file (TextIO): File object for the CSV output
        csv_writer (csv.DictWriter): CSV writer object
        http_session (aiohttp.ClientSession): aiohttp ClientSession
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent requests
        api_credentials (APICredentials): API credentials for authentication
    """
    logger.info(f"Processing questions...")

    tasks = [
        process_question(
            question=question,
            csv_writer=csv_writer,
            csv_file=csv_file,
            http_session=http_session,
            semaphore=semaphore,
            api_credentials=api_credentials,
        )
        for question in questions
    ]
    await tqdm.gather(*tasks, desc="Processing questions")

    logger.info(f"All questions processed!")
