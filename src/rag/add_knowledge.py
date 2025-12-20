"""
Knowledge base initialization for RAG system.
Functions for loading Markdown documents, CSV legal data, chunking, and inserting into database.
"""
import asyncio
import csv
import hashlib
import logging
from pathlib import Path
from typing import List, Set

import aiohttp
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from sqlalchemy import select
from tqdm.asyncio import tqdm

from src.config import (
    DOCUMENT_TYPE_ABBREVIATIONS,
    KNOWLEDGE_DATA_DIRECTORY,
    LAW_CSV_PATH,
    MARKDOWN_CHUNK_OVERLAP,
    MARKDOWN_CHUNK_SIZE,
    MARKDOWN_HEADERS_TO_SPLIT_ON,
    MINIMUM_CHUNK_CHARACTERS,
    MINISTRY_ABBREVIATIONS,
    ORGANIZATION_ABBREVIATIONS,
    MAX_EMBEDDING_REQUESTS_PER_SECOND,
    api_credentials,
)
from src.models import APICredentials
from src.rag.add_knowledge import (
    add_law_articles,
    add_markdown_documents,
    add_markdown_document_chunks,
)
from src.rag.database import AsyncSessionLocal, Document, DocumentChunk, ensure_database_schema
from src.vnpt_ai_api import embed


logger = logging.getLogger(__name__)


def _expand_legal_issuers(issuer_part: str, index: str) -> str:
    """
    Expand issuer abbreviations to full text.

    Args:
        issuer_part (str): The issuer part from the legal document index
        index (str): The full index for logging purposes

    Returns:
        str: Expanded issuer text
    """
    issuers = issuer_part.split("-")
    expanded_issuers = []

    for issuer in issuers:
        if issuer in MINISTRY_ABBREVIATIONS:
            expanded_issuers.append(MINISTRY_ABBREVIATIONS[issuer])
        elif issuer in ORGANIZATION_ABBREVIATIONS:
            expanded_issuers.append(ORGANIZATION_ABBREVIATIONS[issuer])
        else:
            logger.warning(f"Unknown issuer abbreviation: {issuer} in {index}")
            expanded_issuers.append(issuer)

    return (
        " và ".join(expanded_issuers)
        if len(expanded_issuers) > 1
        else expanded_issuers[0]
    )


def _parse_legal_document_type_and_issuer(
    type_and_issuer: str, index: str
) -> tuple[str, str | None]:
    """
    Parse document type and issuer from the type_and_issuer part.

    Args:
        type_and_issuer (str): The combined type and issuer string
        index (str): The full index for logging purposes

    Returns:
        tuple[str, str | None]: Document type and optional issuer part
    """
    sorted_abbreviations = sorted(
        DOCUMENT_TYPE_ABBREVIATIONS.items(), key=lambda x: len(x[0]), reverse=True
    )

    for abbreviation, full in sorted_abbreviations:
        if not type_and_issuer.startswith(abbreviation):
            continue

        doc_type = full
        remaining = type_and_issuer[len(abbreviation) :]

        issuer_part = None
        if remaining.startswith(("/", "-")):
            issuer_part = remaining[1:]

        return doc_type, issuer_part

    logger.warning(f"Unknown document type in: {index}")
    doc_type = type_and_issuer.split("/")[0].split("-")[0]
    return doc_type, None


def _expand_legal_document_index(index: str) -> str:
    """
    Expand Vietnamese legal document abbreviations to full text.

    Example:
        "01/2014/NQLT/CP-UBTƯMTTQVN" -> "Nghị quyết liên tịch số 01 năm 2014 của Chính phủ và Ủy ban Trung ương Mặt trận Tổ quốc Việt Nam"

    Args:
        index (str): The abbreviated legal document index

    Returns:
        str: Full Vietnamese text describing the legal document
    """

    if index.startswith("HP"):
        year = index[2:]
        return f"Hiến pháp năm {year}."

    if "/" not in index:
        return index

    parts = index.split("/")
    if len(parts) < 2:
        return index

    number = parts[0]

    # Format without year: number/type-issuer (e.g., "595/QĐ-BHXH")
    if len(parts) == 2:
        type_and_issuer = parts[1]
        doc_type, issuer_part = _parse_legal_document_type_and_issuer(
            type_and_issuer, index
        )

        result = f"{doc_type} số {number}"
        if issuer_part:
            issuer_text = _expand_legal_issuers(issuer_part, index)
            result += f" của {issuer_text}"

        return result

    # Format with year: number/year/type-issuer
    year = parts[1]
    type_and_issuer = "/".join(parts[2:])

    # Special case: Quốc hội with session number
    if (
        type_and_issuer.startswith("QH")
        and len(type_and_issuer) > 2
        and type_and_issuer[2].isdigit()
    ):
        session = type_and_issuer[2:]
        return f"Nghị quyết số {number} năm {year} của Quốc hội khóa {session}."

    doc_type, issuer_part = _parse_legal_document_type_and_issuer(
        type_and_issuer, index
    )

    result = f"{doc_type} số {number} năm {year}"
    if issuer_part:
        issuer_text = _expand_legal_issuers(issuer_part, index)
        result += f" của {issuer_text}"

    return result


def _chunk_markdown_text(
    markdown_text: str,
    chunk_size: int = MARKDOWN_CHUNK_SIZE,
    chunk_overlap: int = MARKDOWN_CHUNK_OVERLAP,
) -> List[str]:
    """
    Perform hierarchical Markdown-aware chunking with full header preservation.

    Steps:
      1. Split Markdown into sections by headers (H1–H6), preserving hierarchy.
      2. For each section, further split with RecursiveCharacterTextSplitter.
      3. Prepend all parent headers to each chunk's content.

    Args:
        markdown_text (str): The Markdown text to chunk
        chunk_size (int): Maximum size of each chunk in characters
        chunk_overlap (int): Number of characters to overlap between chunks

    Returns:
        List[str]: List of chunk contents
    """
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=MARKDOWN_HEADERS_TO_SPLIT_ON,
        return_each_line=False,
        strip_headers=True,
    )
    header_docs = splitter.split_text(markdown_text)
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", "。 ", "。", ". ", "!", "?", " ", ""],
    )

    chunks: List[str] = []
    for doc in header_docs:
        content = getattr(doc, "page_content", str(doc)).strip()
        metadata = getattr(doc, "metadata", {}) or {}
        if not content:
            continue

        header_lines = []
        for level in range(1, 7):
            key = f"h{level}"
            if key in metadata and metadata[key]:
                header_text = metadata[key].strip()
                header_line = "#" * level + " " + header_text
                header_lines.append(header_line)

        header_prefix = "\n".join(header_lines) + "\n" if header_lines else ""
        sub_texts = char_splitter.split_text(content)
        for sub_text in sub_texts:
            text = sub_text.strip()
            if len(text) < MINIMUM_CHUNK_CHARACTERS:
                continue

            full_content = header_prefix + "\n" + text if header_prefix else text
            chunks.append(full_content)

    return chunks


async def _get_existing_document_sources() -> Set[str]:
    """
    Load all existing document source files from the database.

    Returns:
        Set[str]: Set of document source files
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Document.source_file))
        return set(row[0] for row in result.fetchall())


async def _get_existing_document_hashes() -> Set[bytes]:
    """
    Load all existing document hashes from the database.

    Returns:
        Set[bytes]: Set of document SHA256 hashes
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Document.sha256_hash))
        return set(row[0] for row in result.fetchall())


async def _get_existing_chunk_hashes() -> Set[bytes]:
    """
    Load all existing chunk hashes from the database.

    Returns:
        Set[bytes]: Set of chunk SHA256 hashes
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(DocumentChunk.sha256_hash))
        return set(row[0] for row in result.fetchall())


async def _add_markdown_document(
    content: str,
    topic: str,
    source_file: str,
) -> None:
    """
    Add a single Markdown document to the documents table.

    Args:
        content (str): The content of the Markdown file
        topic (str): The topic/category of the document
        source_file (str): The source file name

    Returns:
        None
    """
    async with AsyncSessionLocal() as session:
        document = Document(
            content=content,
            topic=topic,
            source_file=source_file,
        )
        session.add(document)
        await session.commit()


async def add_markdown_documents(
    data_directory: Path,
) -> None:
    """
    Load all Markdown documents into the documents table.

    Args:
        data_directory (Path): Base directory for knowledge files

    Returns:
        None
    """
    existing_documents, existing_document_hashes = await asyncio.gather(
        _get_existing_document_sources(),
        _get_existing_document_hashes(),
    )

    markdown_files = []
    for topic_directory in data_directory.iterdir():
        if not topic_directory.is_dir():
            continue

        topic = topic_directory.name
        for markdown_file in topic_directory.rglob("*.md"):
            source_file = markdown_file.name
            if source_file in existing_documents:
                logger.debug(f"Skipping duplicate source file: {source_file}")
                continue
            existing_documents.add(source_file)

            content = markdown_file.read_text(encoding="utf-8")
            document_hash = hashlib.sha256(content.encode("utf-8")).digest()
            if document_hash in existing_document_hashes:
                logger.debug(
                    f"Skipping duplicate document content: {source_file} (hash: {document_hash.hex()[:8]}...)"
                )
                continue
            existing_document_hashes.add(document_hash)

            markdown_files.append((content, topic, source_file))

    logger.info(f"Found {len(markdown_files)} Markdown files to add")
    tasks = [
        _add_markdown_document(content=content, topic=topic, source_file=source_file)
        for content, topic, source_file in markdown_files
    ]

    await tqdm.gather(*tasks, desc="Adding Markdown documents", total=len(tasks))


async def _add_markdown_document_chunk(
    chunk: str,
    source_file: str,
    chunk_index: int,
    http_session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    api_credentials: APICredentials,
) -> None:
    """
    Embed chunks and add them to the document_chunks table.

    Args:
        chunk (str): The chunk content
        source_file (str): The source file name
        chunk_index (int): The index of the chunk within the document
        http_session (aiohttp.ClientSession): aiohttp session for API calls
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent requests
        api_credentials (APICredentials): API credentials for authentication

    Returns:
        None
    """
    embedding = await embed(chunk, http_session, semaphore, api_credentials)

    async with AsyncSessionLocal() as session:
        chunk_obj = DocumentChunk(
            content=chunk,
            embedding=embedding,
            source_file=source_file,
            chunk_index=chunk_index,
        )
        session.add(chunk_obj)
        await session.commit()


async def add_markdown_document_chunks(
    data_directory: Path,
    http_session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    api_credentials: APICredentials,
) -> None:
    """
    Chunk all Markdown documents and add chunks to the document_chunks table.

    Args:
        data_directory (Path): Base directory for knowledge files
        http_session (aiohttp.ClientSession): aiohttp session for API calls
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent requests
        api_credentials (APICredentials): API credentials for authentication

    Returns:
        None
    """
    existing_chunk_hashes = await _get_existing_chunk_hashes()

    chunks_to_add = []
    for topic_directory in data_directory.iterdir():
        if not topic_directory.is_dir():
            continue

        for markdown_file in topic_directory.rglob("*.md"):
            content = markdown_file.read_text(encoding="utf-8")
            chunks = _chunk_markdown_text(content)

            for i, chunk_content in enumerate(chunks):
                chunk_hash = hashlib.sha256(chunk_content.encode("utf-8")).digest()

                if chunk_hash in existing_chunk_hashes:
                    continue
                existing_chunk_hashes.add(chunk_hash)

                chunks_to_add.append([chunk_content, markdown_file.name, i])

    logger.info(f"Found {len(chunks_to_add)} Markdown chunks to add")
    tasks = [
        _add_markdown_document_chunk(
            chunk, source_file, chunk_index, http_session, semaphore, api_credentials
        )
        for chunk, source_file, chunk_index in chunks_to_add
    ]
    await tqdm.gather(*tasks, desc="Adding Markdown document chunks", total=len(tasks))


async def add_law_article(
    index: str,
    article: str,
    content: str,
    row_index: int,
    source_file: str,
    http_session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    api_credentials: APICredentials,
) -> None:
    """
    Process a single row from the legal CSV file.

    Args:
        index (str): The legal document index (abbreviated)
        article (str): The article number
        content (str): The article content
        row_index (int): The row index for chunk_index
        source_file (str): The source file name
        http_session (aiohttp.ClientSession): aiohttp session for API calls
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent requests
        api_credentials (APICredentials): API credentials for authentication

    Returns:
        None
    """
    expanded_index = _expand_legal_document_index(index)
    chunk_content = f"{expanded_index}\nĐiều {article}: {content}"

    embedding = await embed(chunk_content, http_session, semaphore, api_credentials)

    async with AsyncSessionLocal() as session:
        chunk = DocumentChunk(
            content=chunk_content,
            embedding=embedding,
            source_file=source_file,
            chunk_index=row_index,
        )
        session.add(chunk)
        await session.commit()


async def add_law_articles(
    file_path: Path,
    http_session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    api_credentials: APICredentials,
) -> None:
    """
    Load and process the legal CSV file.

    Args:
        file_path (Path): Path to the legal CSV file
        http_session (aiohttp.ClientSession): aiohttp session for API calls
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent requests
        api_credentials (APICredentials): API credentials for authentication

    Returns:
        None
    """
    existing_chunk_hashes = await _get_existing_chunk_hashes()

    csv.field_size_limit(10 * 1024 * 1024)
    rows = []
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            expanded_index = _expand_legal_document_index(row["index"])
            chunk_content = f"{expanded_index}\nĐiều {row['article']}: {row['content']}"
            chunk_hash = hashlib.sha256(chunk_content.encode("utf-8")).digest()

            if chunk_hash in existing_chunk_hashes:
                logger.debug(
                    f"Skipping duplicate law article: {row['index']} Điều {row['article']} (hash: {chunk_hash.hex()[:8]}...)"
                )
                continue

            existing_chunk_hashes.add(chunk_hash)
            rows.append((row["index"], row["article"], row["content"], i))

    logger.info(f"Found {len(rows)} legal articles to add")
    tasks = [
        add_law_article(
            index=index,
            article=article,
            content=content,
            row_index=row_index,
            source_file=file_path.name,
            http_session=http_session,
            semaphore=semaphore,
            api_credentials=api_credentials,
        )
        for index, article, content, row_index in rows
    ]
    await tqdm.gather(*tasks, desc="Adding legal articles", total=len(tasks))


async def main():
    logger.info("Ensuring database is ready...")
    await ensure_database_schema()
    logger.info("Database ready!")

    semaphore = asyncio.Semaphore(MAX_EMBEDDING_REQUESTS_PER_SECOND)
    data_directory = Path(KNOWLEDGE_DATA_DIRECTORY)
    law_csv_file = Path(LAW_CSV_PATH)

    logger.info("Initializing knowledge base...")

    logger.info("Step 1: Adding full Markdown documents...")
    await add_markdown_documents(data_directory)
    logger.info("Markdown documents added!")

    async with aiohttp.ClientSession() as http_session:
        logger.info("Step 2: Adding Markdown document chunks...")
        await add_markdown_document_chunks(
            data_directory, http_session, semaphore, api_credentials
        )
        logger.info("Markdown document chunks added!")

        logger.info("Step 3: Adding law articles...")
        await add_law_articles(law_csv_file, http_session, semaphore, api_credentials)
        logger.info("Law articles added!")

    logger.info("Knowledge base initialization complete!")


if __name__ == "__main__":
    asyncio.run(main())
