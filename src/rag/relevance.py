from collections.abc import Sequence

from src.rag.models import RetrievedChunk


DEFAULT_MAX_RETRIEVAL_DISTANCE = 0.85


def filter_relevant_chunks(
    chunks: Sequence[RetrievedChunk],
    max_distance: float = DEFAULT_MAX_RETRIEVAL_DISTANCE,
) -> list[RetrievedChunk]:
    """
    Retain chunks with acceptable semantic distance.

    Chroma distance is lower for closer vector matches. Chunks without
    a distance are excluded because their semantic relevance cannot be
    established.
    """

    if max_distance < 0:
        raise ValueError(
            "max_distance must not be negative."
        )

    return [
        chunk
        for chunk in chunks
        if (
            chunk.distance is not None
            and chunk.distance <= max_distance
        )
    ]
