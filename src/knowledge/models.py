from dataclasses import dataclass
from enum import Enum
from pathlib import Path


@dataclass
class KnowledgeDocument:
    """
    Represents a document known to Bridge AI.

    This model is independent of the vector database and represents
    the application-level state of a knowledge document.
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
    """
    Actions that may be performed during knowledge synchronization.
    """

    SKIP = "skip"
    INDEX_NEW = "index_new"
    REINDEX = "reindex"
    DELETE = "delete"


@dataclass
class DocumentAction:
    """
    Represents one synchronization decision for a document.
    """

    document: KnowledgeDocument
    action: ActionType
