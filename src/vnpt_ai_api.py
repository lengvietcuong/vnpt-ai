import asyncio
import json
import logging
import re
from http import HTTPStatus
from typing import List

import aiohttp

from src.config import (
    VNPT_AI_BASE_URL,
    LLM_SMALL_ENDPOINT,
    LLM_LARGE_ENDPOINT,
    LLM_SMALL_MODEL,
    LLM_LARGE_MODEL,
    EMBEDDING_ENDPOINT,
    EMBEDDING_MODEL,
    REQUEST_TIMEOUT_SECONDS,
    TRY_AGAIN_DELAY_SECONDS,
    PYTHON_EXECUTOR_TIMEOUT_SECONDS,
    PYTHON_PATTERN,
    EMBEDDING_DIMENSIONS,
    TEMPERATURE,
    MAX_TOKENS,
    MAX_ITERATIONS,
)
from src.models import LLMSettings, APICredentials, Message
from src.rag.chunk_retrieval import keyword_search
from src.rag.document_retrieval import (
    list_documents,
    list_sections,
    read_sections,
)
from src.utils import format_log_message, execute_python_code


logger = logging.getLogger(__name__)


async def _call_api_with_retry(
    url: str,
    headers: dict,
    payload: dict,
    http_session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    retry_delay_seconds: float,
) -> dict:
    """
    Make an HTTP POST request to VNPT AI API with retry logic.

    Args:
        url (str): The API endpoint URL
        headers (dict): Request headers
        payload (dict): Request payload
        http_session (aiohttp.ClientSession): aiohttp ClientSession for making HTTP requests
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent requests
        retry_delay_seconds (float): Delay before retrying the request

    Returns:
        dict: The JSON response from the API
    """

    async def _handle_api_response(
        response: aiohttp.ClientResponse, retry_delay_seconds: float
    ) -> dict | None:
        """
        Handle API response and return JSON data if successful, or None if retry is needed.
        """
        if response.status in (
            HTTPStatus.UNAUTHORIZED,
            HTTPStatus.TOO_MANY_REQUESTS,
        ):
            logger.warning(
                f"Received {response.status} error, retrying in {retry_delay_seconds}s..."
            )
            await asyncio.sleep(retry_delay_seconds)
            return None

        response.raise_for_status()
        return await response.json()

    while True:
        async with semaphore:
            async with http_session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS),
            ) as response:
                result = await _handle_api_response(response, retry_delay_seconds)
                if result is not None:
                    return result


async def _call_vnpt_llm(
    messages: List[Message],
    http_session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    llm_settings: LLMSettings,
    api_credentials: APICredentials,
) -> Message:
    """
    Internal function to call VNPT AI API with the model size specified in llm_settings.

    Args:
        messages (List[Message]): The conversation messages
        http_session (aiohttp.ClientSession): aiohttp ClientSession for making HTTP requests
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent requests
        llm_settings (LLMSettings): LLM configuration settings
        api_credentials (APICredentials): API credentials for authentication

    Returns:
        Message: The assistant's message object
    """
    if llm_settings.llm_size == "small":
        endpoint = LLM_SMALL_ENDPOINT
        model_name = LLM_SMALL_MODEL
        token_id = api_credentials.llm_small_token_id
        token_key = api_credentials.llm_small_token_key
    else:
        endpoint = LLM_LARGE_ENDPOINT
        model_name = LLM_LARGE_MODEL
        token_id = api_credentials.llm_large_token_id
        token_key = api_credentials.llm_large_token_key

    headers = {
        "Authorization": f"Bearer {api_credentials.vnpt_ai_authorization_token}",
        "Token-id": token_id,
        "Token-key": token_key,
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_completion_tokens": MAX_TOKENS,
        "tools": llm_settings.tools,
        "tool_choice": "auto",
    }

    url = f"{VNPT_AI_BASE_URL}{endpoint}"

    result = await _call_api_with_retry(
        url, headers, payload, http_session, semaphore, TRY_AGAIN_DELAY_SECONDS
    )
    return result["choices"][0]["message"]


async def get_tool_call_result(
    function_name: str,
    function_args: dict,
    query: str | List[float],
    http_session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    api_credentials: APICredentials,
) -> str:
    """
    Execute a single tool call and return its result.

    Args:
        function_name (str): Name of the function to call
        function_args (dict): Arguments for the function
        query (str | List[float]): Original user query or its embedding for context
        http_session (aiohttp.ClientSession): aiohttp ClientSession for making HTTP requests
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent requests
        api_credentials (APICredentials): API credentials for authentication

    Returns:
        str: The result of the tool call
    """
    if function_name == "keyword_search":
        if not isinstance(query, list):
            query = await embed(
                text=query,
                http_session=http_session,
                semaphore=semaphore,
                api_credentials=api_credentials,
            )
        return await keyword_search(
            query_embedding=query,
            keywords=function_args["keywords"],
            target_documents=function_args.get("target_documents"),
        )

    if function_name == "list_documents":
        return await list_documents(
            topics=function_args["topics"],
        )

    if function_name == "list_sections":
        return await list_sections(
            documents=function_args["documents"],
        )

    if function_name == "read_sections":
        return await read_sections(
            sections=function_args["sections"],
            documents=function_args["documents"],
        )

    raise ValueError(f"Invalid tool '{function_name}'")


async def handle_tool_calls(
    assistant_message: Message,
    http_session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    api_credentials: APICredentials,
    query: str | List[float],
) -> List[Message]:
    """
    Handle tool calls from the assistant's message and return results as messages.

    Args:
        assistant_message (Message): The assistant's message containing tool calls
        http_session (aiohttp.ClientSession): aiohttp ClientSession for making HTTP requests
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent requests
        api_credentials (APICredentials): API credentials for authentication
        query (str | List[float]): Original user query or its embedding for context

    Returns:
        List[Message]: List of tool result messages to append to the conversation
    """
    tool_messages: List[Message] = []

    for tool_call in assistant_message["tool_calls"]:
        function_name = tool_call["function"]["name"]
        function_args = json.loads(tool_call["function"]["arguments"])
        logger.debug(f"Model called {function_name} with args {function_args}")

        try:
            tool_result = await get_tool_call_result(
                function_name=function_name,
                function_args=function_args,
                query=query,
                http_session=http_session,
                semaphore=semaphore,
                api_credentials=api_credentials,
            )
        except Exception as error:
            logger.error(f"Error with tool '{function_name}': {error}")
            tool_result = f"Error: {str(error)}\nPlease try again."

        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": tool_result,
            }
        )

    return tool_messages


async def get_llm_response(
    prompt: str,
    http_session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    llm_settings: LLMSettings,
    api_credentials: APICredentials,
    query_embedding: List[float] = None,
) -> List[Message]:
    """
    Call the VNPT AI API to get a response for a given prompt.

    Args:
        prompt (str): The formatted prompt string
        http_session (aiohttp.ClientSession): aiohttp ClientSession for making HTTP requests
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent requests
        llm_settings (LLMSettings): LLM configuration settings
        api_credentials (APICredentials): API credentials for authentication
        query_embedding (List[float]): Optional query embedding for semantic search

    Returns:
        List[Message]: The entire conversation messages array
    """
    messages: List[Message] = []
    if llm_settings.system_prompt:
        messages.append({"role": "system", "content": llm_settings.system_prompt})
    messages.append({"role": "user", "content": prompt})
    if llm_settings.prefill:
        messages.append(
            {"role": "assistant", "content": llm_settings.prefill}
        )
        
    for i in range(1, MAX_ITERATIONS + 1):
        assistant_message = await _call_vnpt_llm(
            messages=messages,
            http_session=http_session,
            semaphore=semaphore,
            llm_settings=llm_settings,
            api_credentials=api_credentials,
        )
        messages.append(assistant_message)
        content = assistant_message["content"]

        if assistant_message.get("tool_calls"):
            tool_messages = await handle_tool_calls(
                assistant_message=assistant_message,
                http_session=http_session,
                semaphore=semaphore,
                api_credentials=api_credentials,
                query=query_embedding or prompt,
            )
            messages.extend(tool_messages)
            continue
        
        code = re.search(PYTHON_PATTERN, content, re.DOTALL)
        if code:
            try:
                code_result = await asyncio.wait_for(
                    execute_python_code(code.group(1)), timeout=PYTHON_EXECUTOR_TIMEOUT_SECONDS
                )
                messages.append({"role": "user", "content": f"Output: {code_result}"})
            except Exception as error:
                logger.error(f"Error executing Python code: {error}")
                messages.append(
                    {
                        "role": "user",
                        "content": f"Error: {str(error)}\nPlease try again.",
                    }
                )
            continue

        has_answer = re.search(r"\{\{[A-Z]\}\}", content)
        if has_answer:
            logger.debug(
                f"No more tool calls for prompt {format_log_message(prompt)}, returning response"
            )
            return messages

        logger.warning(
            f"Model response does not contain an answer or a tool call: {format_log_message(content)}"
        )
        if i == MAX_ITERATIONS - 1:  # Second last iteration
            messages.append(
                {
                    "role": "user",
                    "content": "Make an educated guess and wrap the answer letter in double curly braces (e.g., {{A}}).",
                }
            )
            continue
        messages.append(
            {
                "role": "user",
                "content": "If you are trying to call a tool, your syntax is invalid, so try again and do it properly. If you are done, wrap the answer letter in double curly braces (e.g., {{A}}). If you cannot find the answer, make an educated guess.",
            }
        )


async def embed(
    text: str,
    http_session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    api_credentials: APICredentials,
) -> List[float]:
    """
    Call the VNPT AI Embedding API to get a vector embedding for a given text.

    Args:
        text (str): The text to be vectorized
        http_session (aiohttp.ClientSession): aiohttp ClientSession for making HTTP requests
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent requests
        api_credentials (APICredentials): API credentials for authentication

    Returns:
        List[float]: The embedding vector with EMBEDDING_DIMENSIONS elements
    """
    headers = {
        "Authorization": f"Bearer {api_credentials.vnpt_ai_authorization_token}",
        "Token-id": api_credentials.embedding_token_id,
        "Token-key": api_credentials.embedding_token_key,
        "Content-Type": "application/json",
    }
    payload = {
        "model": EMBEDDING_MODEL,
        "input": text,
        "encoding_format": "float",
    }
    url = f"{VNPT_AI_BASE_URL}{EMBEDDING_ENDPOINT}"

    result = await _call_api_with_retry(
        url,
        headers,
        payload,
        http_session,
        semaphore,
        TRY_AGAIN_DELAY_SECONDS,
    )
    embedding = result["data"][0]["embedding"]

    if len(embedding) != EMBEDDING_DIMENSIONS:
        logger.warning(
            f"Expected embedding dimension {EMBEDDING_DIMENSIONS}, but got {len(embedding)}"
        )

    return embedding
