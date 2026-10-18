from src.rag.models import RetrievalResult


def build_source_references(
    retrieval: RetrievalResult | None,
) -> list[dict[str, str | int | None]]:
    """
    Build unique source references in retrieval-ranking order.

    A source reference is unique by document name and page number.
    """

    if retrieval is None:
        return []

    references: list[dict[str, str | int | None]] = []
    seen: set[tuple[str, int | None]] = set()

    for chunk in retrieval.chunks:
        key = (
            chunk.source,
            chunk.page,
        )

        if key in seen:
            continue

        seen.add(key)

        references.append(
            {
                "source": chunk.source,
                "page": chunk.page,
            }
        )

    return references
