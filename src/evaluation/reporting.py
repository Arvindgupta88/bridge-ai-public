from src.evaluation.models import (
    RetrievalEvaluationResult,
    RetrievalEvaluationSummary,
)


_EXCERPT_LENGTH = 160


def build_report(
    summary: RetrievalEvaluationSummary,
    generated_at: str,
    embedding_model: str,
) -> dict[str, object]:
    """
    Build a machine-readable retrieval evaluation report.
    """

    if not generated_at.strip():
        raise ValueError(
            "Report generation timestamp must not be blank."
        )

    if not embedding_model.strip():
        raise ValueError(
            "Report embedding model must not be blank."
        )

    return {
        "generated_at": generated_at,
        "embedding_model": embedding_model,
        "aggregate": {
            "total_cases": summary.total_cases,
            "scored_cases": summary.scored_cases,
            "diagnostic_cases": summary.diagnostic_cases,
            "passed_cases": summary.passed_cases,
            "failed_cases": summary.failed_cases,
            "top_k_values": list(
                summary.top_k_values
            ),
            "hit_rate_at_k": summary.hit_rate_at_k,
            "mean_reciprocal_rank": (
                summary.mean_reciprocal_rank
            ),
            "mean_latency_ms": summary.mean_latency_ms,
        },
        "cases": [
            _build_case_report(result)
            for result in summary.results
        ],
    }


def format_console_report(
    summary: RetrievalEvaluationSummary,
    embedding_model: str,
) -> str:
    """
    Format a human-readable retrieval evaluation report.
    """

    if not embedding_model.strip():
        raise ValueError(
            "Report embedding model must not be blank."
        )

    hit_rate = (
        "n/a"
        if summary.hit_rate_at_k is None
        else f"{summary.hit_rate_at_k:.2%}"
    )
    mean_reciprocal_rank = (
        "n/a"
        if summary.mean_reciprocal_rank is None
        else f"{summary.mean_reciprocal_rank:.4f}"
    )

    lines = [
        "Bridge AI Retrieval Evaluation",
        f"Embedding model: {embedding_model}",
        (
            "Raw distance: lower values indicate "
            "closer vector matches."
        ),
        "",
        "Aggregate",
        f"  Total cases: {summary.total_cases}",
        f"  Scored cases: {summary.scored_cases}",
        (
            "  Diagnostic cases: "
            f"{summary.diagnostic_cases}"
        ),
        f"  Passed cases: {summary.passed_cases}",
        f"  Failed cases: {summary.failed_cases}",
        (
            "  Top-K values: "
            + ", ".join(
                str(value)
                for value in summary.top_k_values
            )
        ),
        f"  Hit Rate@K: {hit_rate}",
        (
            "  Mean Reciprocal Rank: "
            f"{mean_reciprocal_rank}"
        ),
        (
            "  Mean retrieval latency: "
            f"{summary.mean_latency_ms:.2f} ms"
        ),
        "",
        "Cases",
    ]

    for result in summary.results:
        lines.extend(
            _format_case(result)
        )

    return "\n".join(lines)


def _build_case_report(
    result: RetrievalEvaluationResult,
) -> dict[str, object]:
    return {
        "id": result.case.case_id,
        "question": result.case.question,
        "status": _case_status(result),
        "retrieval_mode": (
            result.case.retrieval_mode.value
        ),
        "top_k": result.case.top_k,
        "expected_sources": list(
            result.case.expected_sources
        ),
        "expected_pages": list(
            result.case.expected_pages
        ),
        "first_relevant_rank": (
            result.first_relevant_rank
        ),
        "reciprocal_rank": result.reciprocal_rank,
        "latency_ms": result.latency_ms,
        "retrieved": [
            {
                "rank": rank,
                "source": chunk.source,
                "page": chunk.page,
                "distance": chunk.distance,
                "excerpt": _build_excerpt(
                    chunk.text
                ),
            }
            for rank, chunk in enumerate(
                result.chunks,
                start=1,
            )
        ],
    }


def _format_case(
    result: RetrievalEvaluationResult,
) -> list[str]:
    status = _case_status(result).upper()

    lines = [
        "",
        f"[{status}] {result.case.case_id}",
        f"  Question: {result.case.question}",
        (
            "  Retrieval mode: "
            f"{result.case.retrieval_mode.value}"
        ),
        f"  Top-K: {result.case.top_k}",
        (
            "  Latency: "
            f"{result.latency_ms:.2f} ms"
        ),
    ]

    if result.case.is_scored:
        lines.append(
            "  Expected sources: "
            + ", ".join(
                result.case.expected_sources
            )
        )

        if result.case.expected_pages:
            lines.append(
                "  Expected pages: "
                + ", ".join(
                    str(page)
                    for page
                    in result.case.expected_pages
                )
            )

        rank_label = (
            "not found"
            if result.first_relevant_rank is None
            else str(result.first_relevant_rank)
        )
        lines.append(
            "  First relevant rank: "
            f"{rank_label}"
        )
    else:
        lines.append(
            "  Diagnostic-only unrelated question"
        )

    if not result.chunks:
        lines.append(
            "  No chunks retrieved."
        )
        return lines

    lines.append(
        "  Retrieved chunks:"
    )

    for rank, chunk in enumerate(
        result.chunks,
        start=1,
    ):
        page_label = (
            "n/a"
            if chunk.page is None
            else str(chunk.page)
        )
        distance_label = (
            "n/a"
            if chunk.distance is None
            else f"{chunk.distance:.6f}"
        )

        lines.append(
            f"    {rank}. {chunk.source} "
            f"page={page_label} "
            f"distance={distance_label}"
        )
        lines.append(
            "       "
            + _build_excerpt(chunk.text)
        )

    return lines


def _case_status(
    result: RetrievalEvaluationResult,
) -> str:
    if result.hit is None:
        return "diagnostic"

    if result.hit:
        return "pass"

    return "fail"


def _build_excerpt(
    text: str,
) -> str:
    normalized_text = " ".join(
        text.split()
    )

    if len(normalized_text) <= _EXCERPT_LENGTH:
        return normalized_text

    return (
        normalized_text[
            : _EXCERPT_LENGTH - 3
        ].rstrip()
        + "..."
    )
