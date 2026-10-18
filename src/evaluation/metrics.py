from collections.abc import Collection, Sequence

from src.rag.models import RetrievedChunk


def find_first_relevant_rank(
    chunks: Sequence[RetrievedChunk],
    expected_sources: Collection[str],
    expected_pages: Collection[int] = (),
    top_k: int = 5,
) -> int | None:
    """
    Return the one-based rank of the first relevant retrieved chunk.

    A chunk is relevant when its source matches an expected source. When
    expected pages are supplied, its page must also match one of them.
    """

    if top_k < 1:
        raise ValueError("top_k must be greater than zero.")

    source_set = set(expected_sources)
    page_set = set(expected_pages)

    for rank, chunk in enumerate(
        chunks[:top_k],
        start=1,
    ):
        if chunk.source not in source_set:
            continue

        if page_set and chunk.page not in page_set:
            continue

        return rank

    return None


def calculate_hit_rate(
    hits: Sequence[bool],
) -> float:
    """
    Calculate the proportion of evaluation cases with a relevant result.
    """

    if not hits:
        raise ValueError(
            "At least one evaluation result is required."
        )

    return sum(hits) / len(hits)


def calculate_mean_reciprocal_rank(
    first_relevant_ranks: Sequence[int | None],
) -> float:
    """
    Calculate mean reciprocal rank across evaluation cases.

    Cases without a relevant result contribute zero.
    """

    if not first_relevant_ranks:
        raise ValueError(
            "At least one evaluation result is required."
        )

    reciprocal_rank_total = 0.0

    for rank in first_relevant_ranks:
        if rank is None:
            continue

        if rank < 1:
            raise ValueError(
                "Relevant-result ranks must be greater than zero."
            )

        reciprocal_rank_total += 1 / rank

    return (
        reciprocal_rank_total
        / len(first_relevant_ranks)
    )
