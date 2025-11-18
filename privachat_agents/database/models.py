"""Database models for PrivaChat agent system."""

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector  # type: ignore[import-untyped]
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


class ResearchSession(Base):
    """Research session tracking user queries and results."""

    __tablename__ = "research_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    mode: Mapped[str] = mapped_column(
        String(20), nullable=False, index=True
    )  # 'search' or 'research'
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, index=True
    )  # 'pending', 'processing', 'completed', 'failed'
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    execution_time_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Relationships
    documents: Mapped[list["SessionDocument"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class RAGDocument(Base):
    """Documents stored in vector database for RAG retrieval."""

    __tablename__ = "rag_documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True, unique=True)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_type: Mapped[str] = mapped_column(
        String(20), nullable=False, index=True
    )  # 'web', 'pdf', 'excel', 'word'
    doc_metadata: Mapped[dict[str, Any] | None] = mapped_column("doc_metadata", JSON, nullable=True)
    embedding: Mapped[Any] = mapped_column(Vector(384), nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    last_accessed: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    access_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Relationships
    sessions: Mapped[list["SessionDocument"]] = relationship(back_populates="document")


class SessionDocument(Base):
    """Association table between sessions and documents."""

    __tablename__ = "session_documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("research_sessions.id"), nullable=False
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("rag_documents.id"), nullable=False
    )
    relevance_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    used_in_synthesis: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    session: Mapped[ResearchSession] = relationship(back_populates="documents")
    document: Mapped[RAGDocument] = relationship(back_populates="sessions")


class DocumentEmbedding(Base):
    """Session-specific document embeddings for vector similarity search."""

    __tablename__ = "document_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("research_sessions.id"), nullable=False, index=True
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Any] = mapped_column(Vector(384), nullable=False)
    doc_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


class ExcludedDomain(Base):
    r"""Domains excluded from web crawling to avoid unnecessary data fetching.
    
    Supports multiple pattern types:
    - exact: Exact domain match (e.g., 'youtube.com')
    - wildcard: Shell-style wildcard patterns (e.g., '*.youtube.*')
    - regex: Regular expression patterns (e.g., r'^.*\.youtube\..*$')
    """

    __tablename__ = "excluded_domains"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    domain_pattern: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    pattern_type: Mapped[str] = mapped_column(
        String(20), nullable=False, default="exact"
    )  # 'exact', 'wildcard', 'regex'
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    created_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
