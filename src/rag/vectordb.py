from chromadb import PersistentClient

from src.core.settings import (
    CHROMA_COLLECTION,
    CHROMA_PATH,
)
from src.core.logger import logger


class VectorDatabase:

    def __init__(self):

        self.client = PersistentClient(path=CHROMA_PATH)

        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION
        )

    def add_document(
        self,
        chunk_id: str,
        text: str,
        embedding: list[float],
        metadata: dict,
    ):

        self.collection.add(
            ids=[chunk_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata],
        )

    def search(
        self,
        embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
    ):
        """
        Search for vector chunks matching the query embedding.

        Documents, metadata, and raw Chroma distances are returned for
        retrieval diagnostics. An optional metadata filter restricts the
        search to a specific document, category, or source type.
        """

        include = [
            "documents",
            "metadatas",
            "distances",
        ]

        if where is None:
            return self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                include=include,
            )

        return self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where,
            include=include,
        )

    def delete_document(
        self,
        document_hash: str,
    ) -> None:
        """
        Delete every chunk belonging to a document.
        Args:
            document_hash: SHA-256 hash used to identify the source document.
        Raises:
            ValueError: If document_hash is empty.
        """
        if not document_hash:
            raise ValueError("document_hash must not be empty")

        self.collection.delete(
            where={
                "document_hash": document_hash
            }
        )
        logger.info(
            "Deleted vector chunks for one document"
        )

    def get_document_records(self, document_hash: str) -> list[dict]:
        """Return a document's stored chunks for safe replacement recovery."""
        result = self.collection.get(
            where={"document_hash": document_hash},
            include=["documents", "metadatas", "embeddings"],
        )
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        embeddings = result.get("embeddings")
        return [
            {
                "chunk_id": ids[index],
                "text": documents[index],
                "embedding": embeddings[index].tolist()
                if hasattr(embeddings[index], "tolist")
                else list(embeddings[index]),
                "metadata": metadatas[index],
            }
            for index in range(len(ids))
        ]

    def reset(self):
        """
        Delete the current collection and recreate it.

        Raises:
            RuntimeError: If deletion or recreation fails, leaving the database
                         in a potentially inconsistent state
        """

        try:
            # Delete existing collection
            try:
                self.client.delete_collection(CHROMA_COLLECTION)
                logger.info(f"Successfully deleted collection {CHROMA_COLLECTION}")
            except Exception as e:
                logger.warning(f"Collection deletion failed: {str(e)}")
                # Some implementations may raise if collection doesn't exist
                # Try to proceed with recreation

            # Recreate the collection
            self.collection = self.client.get_or_create_collection(
                name=CHROMA_COLLECTION
            )

            # Verify collection is properly initialized
            if self.collection is None:
                raise RuntimeError(
                    f"Failed to create collection {CHROMA_COLLECTION}: "
                    "get_or_create_collection returned None"
                )

            logger.info(f"Successfully recreated collection {CHROMA_COLLECTION}")

        except Exception as e:
            logger.exception(f"Database reset failed: {str(e)}")
            raise RuntimeError(
                f"Failed to reset vector database: {str(e)}. "
                "Database may be in an inconsistent state."
            ) from e
