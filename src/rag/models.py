from dataclasses import dataclass
from enum import Enum
from pathlib import Path


@dataclass
class DocumentChunk:
    id: str
    text: str
    source: str
    page: int
    #embedding: list[float] | None = None

@dataclass
class IndexResult:
    success: bool
    chunk_count: int
    error: str | None = None

@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int | None = None
    distance: float | None = None
    source_type: str | None = None
    category: str | None = None
    document_hash: str | None = None
    chunk: int | None = None


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]

@dataclass
class PdfPage:
    page: int
    text: str

@dataclass
class KnowledgeDocument:
    """
    Represents a document discovered by a knowledge source.
    """

    path: Path | None
    name: str
    category: str
    size: int
    hash: str

    indexed: bool = False
    chunk_count: int = 0
    indexed_at: str | None = None


class ActionType(Enum):
    SKIP = "skip"
    INDEX_NEW = "index_new"
    REINDEX = "reindex"
    DELETE = "delete"


@dataclass
class DocumentAction:
    """
    Describes an action determined by knowledge comparison.
    """

    document: KnowledgeDocument
    action: ActionType
