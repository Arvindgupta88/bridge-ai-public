from typing import List

from src.core.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: The text to chunk
        chunk_size: Size of each chunk (default from settings)
        overlap: Number of overlapping characters between chunks (default from settings)

    Returns:
        List of text chunks

    Raises:
        ValueError: If chunk_size is invalid or overlap >= chunk_size
    """

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")

    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size}). "
            "This would cause infinite loops or backwards iteration."
        )

    if not text.strip():
        return []

    chunks = []

    start = 0

    step = chunk_size - overlap

    while start < len(text):

        end = min(
            start+chunk_size,
            len(text),
        )

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        if end >= len(text):
            break

        start += step

    return chunks
