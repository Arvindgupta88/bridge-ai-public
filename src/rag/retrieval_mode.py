from enum import Enum


class RetrievalMode(Enum):
    """
    Determines which knowledge sources should be used when
    answering a user question.
    """
    GENERAL = "general"
    LIBRARY = "library"
    UPLOADED = "uploaded"
    HYBRID = "hybrid"


def build_retrieval_filter(
    retrieval_mode: RetrievalMode,
    document_hash: str | None,
) -> dict | None:
    """
    Convert an application retrieval mode into a vector-store filter.
    """

    if retrieval_mode == RetrievalMode.GENERAL:
        return None

    if retrieval_mode == RetrievalMode.LIBRARY:
        return {
            "source_type": "library",
        }

    if retrieval_mode == RetrievalMode.UPLOADED:
        if not document_hash:
            raise ValueError(
                "Uploaded-document mode requires an active document."
            )

        return {
            "document_hash": document_hash,
        }

    if retrieval_mode == RetrievalMode.HYBRID:
        if document_hash:
            return {
                "$or": [
                    {
                        "source_type": "library",
                    },
                    {
                        "document_hash": document_hash,
                    },
                ]
            }

        return {
            "source_type": "library",
        }

    raise ValueError(
        f"Unsupported retrieval mode: {retrieval_mode}"
    )
