import hashlib
from uuid import uuid4

from src.core.logger import logger
from src.knowledge.models import KnowledgeDocument
from src.rag.chunker import chunk_text
from src.rag.embeddings import embed_text
from src.rag.loader import load_pdf
from src.rag.models import (
    IndexResult,
    RetrievedChunk,
    RetrievalResult,
)
from src.rag.vectordb import VectorDatabase
from src.rag.reranker import rerank_chunks
from src.rag.relevance import filter_relevant_chunks


RETRIEVAL_CANDIDATE_MULTIPLIER = 4
class RagPipeline:
    """
    Coordinates document indexing and semantic retrieval.

    Public indexing entry points distinguish temporary uploads from
    persistent knowledge-library documents. Both delegate to one private
    indexing implementation.
    """

    def __init__(self) -> None:
        self.db = VectorDatabase()

    def index_uploaded_document(
        self,
        uploaded_file,
    ) -> IndexResult:
        """
        Index a temporary document uploaded through the UI.
        """

        document_hash = hashlib.sha256(
            uploaded_file.getvalue()
        ).hexdigest()

        return self._index_document(
            document_source=uploaded_file,
            document_hash=document_hash,
            source_name=uploaded_file.name,
            category="temporary",
            source_type="upload",
        )

    def index_library_document(
        self,
        document: KnowledgeDocument,
    ) -> IndexResult:
        """
        Index a persistent document discovered in the knowledge library.
        """

        if document.path is None:
            return IndexResult(
                success=False,
                chunk_count=0,
                error="Knowledge document has no file path.",
            )

        if not document.hash:
            return IndexResult(
                success=False,
                chunk_count=0,
                error="Knowledge document has no content hash.",
            )

        return self._index_document(
            document_source=document.path,
            document_hash=document.hash,
            source_name=document.name,
            category=document.category,
            source_type="library",
        )

    def _index_document(
        self,
        document_source,
        document_hash: str,
        source_name: str,
        category: str,
        source_type: str,
    ) -> IndexResult:
        """
        Perform the common PDF loading, chunking, embedding and storage flow.
        """

        try:
            pages = load_pdf(document_source)

            chunk_records: list[dict] = []

            for page in pages:
                if not page.text.strip():
                    continue

                page_chunks = chunk_text(page.text)

                for chunk_number, chunk in enumerate(
                    page_chunks,
                    start=1,
                ):
                    embedding = embed_text(chunk)

                    chunk_records.append(
                        {
                            "text": chunk,
                            "embedding": embedding,
                            "page": page.page,
                            "chunk": chunk_number,
                        }
                    )

            if not chunk_records:
                return IndexResult(
                    success=False,
                    chunk_count=0,
                    error="No extractable text was found in this PDF.",
                )

            # Prepare replacement chunks before touching the current index.
            # Chroma has no transaction spanning delete and multiple adds, so
            # restore the old document if a replacement write partially fails.
            previous_records = self.db.get_document_records(document_hash)
            self.db.delete_document(document_hash)
            for record in chunk_records:
                self.db.add_document(
                    chunk_id=str(uuid4()),
                    text=record["text"],
                    embedding=record["embedding"],
                    metadata={
                        "document_hash": document_hash,
                        "source": source_name,
                        "category": category,
                        "source_type": source_type,
                        "page": record["page"],
                        "chunk": record["chunk"],
                    },
                )

            # All adds completed. The replacement is now authoritative.

            logger.info(
                "Indexed document source_type=%s chunks=%d",
                source_type,
                len(chunk_records),
            )

            return IndexResult(
                success=True,
                chunk_count=len(chunk_records),
            )

        except ValueError as error:
            self._restore_previous_document(
                document_hash=document_hash,
                records=locals().get("previous_records"),
                source_type=source_type,
            )
            logger.warning(
                "Document indexing failed (source_type=%s error_type=%s)",
                source_type,
                type(error).__name__,
            )

            return IndexResult(
                success=False,
                chunk_count=0,
                error="The document could not be read or indexed.",
            )

        except Exception as error:
            # Restore prior vectors if replacement writes failed after delete.
            self._restore_previous_document(
                document_hash=document_hash,
                records=locals().get("previous_records"),
                source_type=source_type,
            )
            logger.error(
                "Unexpected indexing failure "
                "(source_type=%s error_type=%s)",
                source_type,
                type(error).__name__,
            )

            return IndexResult(
                success=False,
                chunk_count=0,
                error="An unexpected error occurred while indexing the document.",
            )

    def _restore_previous_document(
        self,
        *,
        document_hash: str,
        records: list[dict] | None,
        source_type: str,
    ) -> None:
        """Best-effort rollback after a partially written replacement."""
        if records is None:
            return
        try:
            self.db.delete_document(document_hash)
            for record in records:
                self.db.add_document(**record)
        except Exception as error:
            logger.error(
                "Index rollback failed (source_type=%s error_type=%s)",
                source_type,
                type(error).__name__,
            )

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        where: dict | None = None,
    ) -> RetrievalResult:
        """
        Retrieve the chunks most relevant to a question.

        Chroma distance is preserved as a raw diagnostic value. Lower
        distances represent closer vector matches, but the value is not
        treated as a probability or confidence percentage.
        """

        embedding = embed_text(question)

        candidate_count = (
            top_k
            * RETRIEVAL_CANDIDATE_MULTIPLIER
        )

        results = self.db.search(
            embedding=embedding,
            top_k=candidate_count,
            where=where,
        )

        document_groups = results.get("documents") or []
        metadata_groups = results.get("metadatas") or []
        distance_groups = results.get("distances") or []

        documents = (
            document_groups[0]
            if document_groups
            else []
        )
        metadatas = (
            metadata_groups[0]
            if metadata_groups
            else []
        )
        distances = (
            distance_groups[0]
            if distance_groups
            else []
        )

        chunks: list[RetrievedChunk] = []

        for index, document in enumerate(documents):
            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else None
            )
            if not isinstance(metadata, dict):
                metadata = {}
            distance = (
                distances[index]
                if index < len(distances)
                else None
            )

            chunks.append(
                RetrievedChunk(
                    text=document,
                    source=(
                        metadata.get("source", "Unknown")
                        if metadata
                        else "Unknown"
                    ),
                    page=(
                        metadata.get("page")
                        if metadata
                        else None
                    ),
                    distance=(
                        float(distance)
                        if distance is not None
                        else None
                    ),
                    source_type=metadata.get("source_type"),
                    category=metadata.get("category"),
                    document_hash=metadata.get("document_hash"),
                    chunk=metadata.get("chunk"),
                )
            )

        relevant_chunks = filter_relevant_chunks(chunks)
        if not relevant_chunks:
            return RetrievalResult(
                chunks=[]
            )
        reranked_chunks = rerank_chunks(
            question=question,
            chunks=relevant_chunks,
        )

        return RetrievalResult(
            chunks=reranked_chunks[:top_k]
        )

    def delete_document(
        self,
        document_hash: str,
    ) -> None:
        """
        Delete one document from the vector index.
        """

        self.db.delete_document(document_hash)

    def reset_index(self) -> None:
        """
        Administratively clear the complete vector collection.
        """

        self.db.reset()
