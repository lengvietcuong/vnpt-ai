"""
Database configuration and models for RAG knowledge base.
Defines SQLAlchemy models and connection pooling for PostgreSQL with pgvector.
"""

import logging

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    Column,
    Computed,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    TIMESTAMP,
    func,
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.schema import DDL

from src.config import (
    DB_HOST,
    DB_NAME,
    DB_PASSWORD,
    DB_PORT,
    DB_USER,
    EMBEDDING_DIMENSIONS,
)


logger = logging.getLogger(__name__)

Base = declarative_base()


class DocumentChunk(Base):
    """SQLAlchemy model for document chunks with embeddings."""

    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True)
    content = Column(
        Text,
        CheckConstraint("length(content) > 0"),
        nullable=False,
    )
    embedding = Column(Vector(EMBEDDING_DIMENSIONS), nullable=False)
    source_file = Column(
        String(255),
        CheckConstraint("length(source_file) > 0"),
        nullable=False,
    )
    chunk_index = Column(Integer, nullable=False)
    sha256_hash = Column(
        LargeBinary(32),
        Computed("digest(content, 'sha256')", persisted=True),
        nullable=False,
        unique=True,
    )
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(
        TIMESTAMP,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        Index(
            "idx_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class Document(Base):
    """SQLAlchemy model for full documents."""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    content = Column(
        Text,
        CheckConstraint("length(content) > 0"),
        nullable=False,
    )
    topic = Column(
        Text,
        CheckConstraint("length(topic) > 0"),
        nullable=False,
    )
    source_file = Column(
        String(255),
        CheckConstraint("length(source_file) > 0"),
        nullable=False,
        unique=True,
        index=True,
    )
    sha256_hash = Column(
        LargeBinary(32),
        Computed("digest(content, 'sha256')", persisted=True),
        nullable=False,
        unique=True,
    )
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(
        TIMESTAMP,
        server_default=func.now(),
        onupdate=func.now(),
    )


DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
ASYNC_DATABASE_URL = (
    f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    pool_pre_ping=True,
    echo=False,
)
AsyncSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def ensure_database_schema() -> None:
    """Create database schema and indexes if they don't exist."""

    logger.info("Ensuring database schema exists...")

    async with async_engine.begin() as connection:
        await connection.execute(DDL("CREATE EXTENSION IF NOT EXISTS vector"))
        await connection.execute(DDL("CREATE EXTENSION IF NOT EXISTS pgcrypto"))
        await connection.run_sync(Base.metadata.create_all)

    logger.info("Database schema ready")
