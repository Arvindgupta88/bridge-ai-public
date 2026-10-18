from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from src.core.logger import logger
from src.knowledge.scanner import scan_library

from src.knowledge.metadata_store import MetadataStore

from src.knowledge.models import (
    ActionType,
    DocumentAction,
    KnowledgeDocument
)

if TYPE_CHECKING:
    from src.rag.pipeline import RagPipeline

class KnowledgeManager:
    """
    Coordinates the local knowledge library.

    Responsibilities:
    - Discover documents
    - Track document metadata
    - Determine which documents require indexing
    """

    def __init__(self) -> None:

        self.documents: list[KnowledgeDocument] = []
        self.metadata = MetadataStore()
        self.index = self.metadata.load()

    def refresh(self) -> None:

        self.documents = scan_library()

        for document in self.documents:
            metadata = self.index.get(document.hash)
            if metadata:
                document.indexed = metadata.get(
                    "indexed",
                    False,
                )
                document.chunk_count = metadata.get(
                    "chunk_count",
                    0,
                )
                document.indexed_at = metadata.get(
                    "indexed_at",
                    None,
                )

    def compare(self) -> list[DocumentAction]:
        """
        Compare the current filesystem state with stored metadata.
        This method determines required actions but performs no indexing,
        deletion, or metadata writes.
        """

        actions: list[DocumentAction] = []
        current_hashes = {
            document.hash: document
            for document in self.documents
        }
        stored_hashes = set(self.index.keys())
        #
        # Documents currently present in the knowledge library.
        #
        for document in self.documents:
            stored_metadata = self.index.get(document.hash)
            if (
                stored_metadata is not None
                and stored_metadata.get("indexed", False)
            ):
                action_type = ActionType.SKIP
            else:
                action_type = ActionType.INDEX_NEW

            actions.append(
                DocumentAction(
                    document=document,
                    action=action_type,
                )
            )

        #
        # Documents stored in metadata but no longer present on disk.
        #
        for stored_hash in stored_hashes:
            if stored_hash not in current_hashes:
                stored_metadata = self.index[stored_hash]
                deleted_document = KnowledgeDocument(
                    path=None,
                    name=stored_metadata["name"],
                    category=stored_metadata["category"],
                    size=0,
                    hash=stored_hash,
                    indexed=stored_metadata.get("indexed", False),
                    chunk_count=stored_metadata.get("chunk_count", 0),
                    indexed_at=stored_metadata.get("indexed_at"),
                )
                actions.append(
                    DocumentAction(
                        document=deleted_document,
                        action=ActionType.DELETE,
                    )
                )
        return actions

    def _execute_action(
        self,
        action: DocumentAction,
        pipeline: RagPipeline,
    ) -> tuple[bool, str | None]:
        """
        Execute one synchronization action.
        Returns
        -------
        tuple[bool, str | None]
            Success flag and an optional error message.
        """
        document = action.document

        if action.action == ActionType.SKIP:
            return True, None

        if action.action == ActionType.INDEX_NEW:
            result = pipeline.index_library_document(document)
            if not result.success:
                return (
                    False,
                    result.error or "Document indexing failed.",
                )
            document.indexed = True
            document.chunk_count = result.chunk_count
            document.indexed_at = datetime.now(timezone.utc).isoformat()
            return True, None

        if action.action == ActionType.DELETE:
            pipeline.delete_document(document.hash)
            return True, None

        if action.action == ActionType.REINDEX:
            result = pipeline.index_library_document(document)
            if not result.success:
                return (
                    False,
                    result.error or "Document reindexing failed.",
                )
            document.indexed = True
            document.chunk_count = result.chunk_count
            document.indexed_at = datetime.now(timezone.utc).isoformat()
            return True, None

        return (
            False,
            f"Unsupported action type: {action.action}",
        )

    def execute(
            self,
            pipeline: RagPipeline,
    ) -> dict[str, object]:
        """
        Compare, execute and persist knowledge-library changes.
        Metadata is synchronized only when every required action succeeds.
        """
        logger.info(
            "Knowledge synchronization started"
        )

        actions = self.compare()
        summary: dict[str, object] = {
            "skip": 0,
            "indexed": 0,
            "deleted": 0,
            "reindexed": 0,
            "failed": 0,
            "metadata_synced": False,
            "errors": [],
        }

        for action in actions:
            logger.info("Knowledge action started (action=%s)", action.action.value)
            try:
                success, _error = self._execute_action(
                    action=action,
                    pipeline=pipeline,
                )
            except Exception as exception:
                logger.error(
                    "Knowledge action failed (action=%s error_type=%s)",
                    action.action.value,
                    type(exception).__name__,
                )
                success = False

            if not success:
                summary["failed"] += 1
                errors = summary["errors"]
                if isinstance(errors, list):
                    errors.append(
                        {
                            "action": action.action.value,
                            "error": "Knowledge action failed; retry synchronization.",
                        }
                    )

                continue

            if action.action == ActionType.SKIP:
                summary["skip"] += 1

            elif action.action == ActionType.INDEX_NEW:
                summary["indexed"] += 1

            elif action.action == ActionType.DELETE:
                summary["deleted"] += 1

            elif action.action == ActionType.REINDEX:
                summary["reindexed"] += 1

        if summary["failed"] == 0:
            try:
                self.sync_metadata()
                summary["metadata_synced"] = True
            except Exception as exception:
                summary["failed"] = int(summary["failed"]) + 1
                errors = summary["errors"]
                if isinstance(errors, list):
                    errors.append({
                        "action": "save_metadata",
                        "error": "Knowledge metadata could not be saved; retry synchronization.",
                    })
                logger.error(
                    "Knowledge metadata save failed (error_type=%s)",
                    type(exception).__name__,
                )

            if summary["metadata_synced"]:
                logger.info("Knowledge metadata synchronized")
        else:
            logger.error(
                "Knowledge metadata was not synchronized "
                "because %d action(s) failed",
                summary["failed"],
            )

        logger.info(
            "Knowledge synchronization complete "
            "(skip=%d indexed=%d deleted=%d "
            "reindexed=%d failed=%d)",
            summary["skip"],
            summary["indexed"],
            summary["deleted"],
            summary["reindexed"],
            summary["failed"],
        )

        return summary

    def get_documents(
            self,
    ) -> list[KnowledgeDocument]:

        return self.documents

    def sync_metadata(self) -> None:

        updated = {}

        for document in self.documents:

            updated[document.hash] = {

                "name": document.name,

                "category": document.category,

                "path": (
                    str(document.path)
                    if document.path is not None
                    else None
                ),

                "indexed": document.indexed,

                "chunk_count": document.chunk_count,

                "indexed_at": document.indexed_at,

            }

        self.metadata.save(updated)

        self.index = updated
