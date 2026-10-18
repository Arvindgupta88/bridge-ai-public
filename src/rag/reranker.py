import math
import re
from collections import Counter
from collections.abc import Sequence

from src.rag.models import RetrievedChunk


DEFAULT_LEXICAL_WEIGHT = 0.50
DEFAULT_RRF_CONSTANT = 30

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "under",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "why",
    "will",
    "with",
    "without",
}


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in _TOKEN_PATTERN.findall(
            text.casefold()
        )
        if (
            len(token) > 1
            and token not in _STOP_WORDS
        )
    ]


def _calculate_bm25_scores(
    question: str,
    chunks: Sequence[RetrievedChunk],
) -> list[float]:
    documents = [
        _tokenize(chunk.text)
        for chunk in chunks
    ]

    document_count = len(documents)

    if document_count == 0:
        return []

    average_document_length = (
        sum(len(document) for document in documents)
        / document_count
    )

    document_frequency: Counter[str] = Counter()

    for document in documents:
        document_frequency.update(set(document))

    query_terms = set(_tokenize(question))
    scores: list[float] = []

    k1 = 1.5
    b = 0.75

    for document in documents:
        term_frequency = Counter(document)
        document_length = len(document)
        score = 0.0

        for term in query_terms:
            frequency = term_frequency[term]

            if frequency == 0:
                continue

            containing_documents = (
                document_frequency[term]
            )

            inverse_document_frequency = math.log(
                1
                + (
                    document_count
                    - containing_documents
                    + 0.5
                )
                / (
                    containing_documents
                    + 0.5
                )
            )

            normalization = (
                frequency
                + k1
                * (
                    1
                    - b
                    + b
                    * document_length
                    / max(
                        average_document_length,
                        1.0,
                    )
                )
            )

            score += (
                inverse_document_frequency
                * frequency
                * (k1 + 1)
                / normalization
            )

        scores.append(score)

    return scores


def rerank_chunks(
    question: str,
    chunks: Sequence[RetrievedChunk],
    lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
    rrf_constant: int = DEFAULT_RRF_CONSTANT,
) -> list[RetrievedChunk]:
    """
    Rerank vector candidates using BM25 and reciprocal-rank fusion.

    The original vector rank and a lexical BM25 rank contribute equally
    by default. Existing chunk diagnostics, including raw distance, are
    preserved.
    """

    if not 0 <= lexical_weight <= 1:
        raise ValueError(
            "lexical_weight must be between 0 and 1."
        )

    if rrf_constant < 0:
        raise ValueError(
            "rrf_constant must not be negative."
        )

    candidates = list(chunks)

    if not candidates:
        return []

    lexical_scores = _calculate_bm25_scores(
        question=question,
        chunks=candidates,
    )

    lexical_order = sorted(
        range(len(candidates)),
        key=lambda index: (
            -lexical_scores[index],
            index,
        ),
    )

    lexical_ranks = {
        index: rank
        for rank, index in enumerate(
            lexical_order,
            start=1,
        )
    }

    combined_scores: list[float] = []

    for index in range(len(candidates)):
        vector_rank = index + 1
        lexical_rank = lexical_ranks[index]

        vector_score = 1 / (
            rrf_constant + vector_rank
        )
        lexical_score = 1 / (
            rrf_constant + lexical_rank
        )

        combined_scores.append(
            (
                (1 - lexical_weight)
                * vector_score
            )
            + (
                lexical_weight
                * lexical_score
            )
        )

    reranked_indexes = sorted(
        range(len(candidates)),
        key=lambda index: (
            -combined_scores[index],
            index,
        ),
    )

    return [
        candidates[index]
        for index in reranked_indexes
    ]
