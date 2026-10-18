from __future__ import annotations
from typing import TYPE_CHECKING

from src.core.logger import logger
from src.knowledge.knowledge_manager import KnowledgeManager
if TYPE_CHECKING:
    from src.rag.pipeline import RagPipeline
class StartupService:
    """
    Coordinates application startup activities and knowledge synchronization.
    The service keeps orchestration logic outside the Streamlit UI.
    """

    def __init__(
            self,
            pipeline: RagPipeline,
    ) -> None:
        """
        Initialize the startup service.
        Args:
            pipeline: Shared Rag Pileline used for indexing and deletion.
        """
        self.pipeline = pipeline
        self.knowledge_manager = KnowledgeManager()

    def initialize(self) -> dict[str, object]:
        """
        Run application startup activities.
        Returns:
            Summary of the knowledge synchronization operation.
        """
        logger.info("Application initialization started")
        summary = self._synchronize_knowledge_library()
        logger.info(
            "Application initialization complete "
            "(knowledge_sync=%s)",
            summary,
        )

        return summary

    def refresh_knowledge_library(self) -> dict[str, object]:
        """
        Synchronize the knowledge library after application startup.

        Returns:
            Summary of the knowledge synchronization operation.
        """
        logger.info("Manual knowledge-library refresh started")
        summary = self._synchronize_knowledge_library()
        logger.info(
            "Knowledge library refresh complete "
            "(knowledge_sync=%s)",
            summary,
        )

        return summary

    def _synchronize_knowledge_library(self) -> dict[str, object]:
        """
        Rescan, compare, execute and persist knowledge-library changes.
        """
        self.knowledge_manager.refresh()
        return self.knowledge_manager.execute(
            pipeline=self.pipeline,
        )
