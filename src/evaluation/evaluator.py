from collections.abc import Sequence
from time import perf_counter

from src.evaluation.metrics import (
    find_first_relevant_rank,
)
from src.evaluation.models import (
    RetrievalEvaluationCase,
    RetrievalEvaluationResult,
    RetrievalEvaluationSummary,
)
from src.rag.pipeline import RagPipeline
from src.rag.retrieval_mode import (
    build_retrieval_filter,
)


class RetrievalEvaluator:
    """
    Executes repeatable retrieval evaluation cases.
    """

    def __init__(
        self,
        pipeline: RagPipeline,
    ) -> None:
        self.pipeline = pipeline

    def evaluate_case(
        self,
        evaluation_case: RetrievalEvaluationCase,
    ) -> RetrievalEvaluationResult:
        """
        Execute one evaluation case and measure retrieval latency.
        """

        where = build_retrieval_filter(
            retrieval_mode=evaluation_case.retrieval_mode,
            document_hash=evaluation_case.document_hash,
        )

        started_at = perf_counter()

        retrieval_result = self.pipeline.retrieve(
            question=evaluation_case.question,
            top_k=evaluation_case.top_k,
            where=where,
        )

        latency_ms = (
            perf_counter() - started_at
        ) * 1000

        chunks = tuple(retrieval_result.chunks)

        first_relevant_rank = None

        if evaluation_case.is_scored:
            first_relevant_rank = (
                find_first_relevant_rank(
                    chunks=chunks,
                    expected_sources=(
                        evaluation_case.expected_sources
                    ),
                    expected_pages=(
                        evaluation_case.expected_pages
                    ),
                    top_k=evaluation_case.top_k,
                )
            )

        return RetrievalEvaluationResult(
            case=evaluation_case,
            chunks=chunks,
            latency_ms=latency_ms,
            first_relevant_rank=first_relevant_rank,
        )

    def evaluate_cases(
        self,
        evaluation_cases: Sequence[
            RetrievalEvaluationCase
        ],
    ) -> tuple[RetrievalEvaluationResult, ...]:
        """
        Execute evaluation cases in their declared order.
        """

        return tuple(
            self.evaluate_case(evaluation_case)
            for evaluation_case in evaluation_cases
        )

    def evaluate_suite(
        self,
        evaluation_cases: Sequence[
            RetrievalEvaluationCase
        ],
    ) -> RetrievalEvaluationSummary:
        """
        Execute and summarize a complete evaluation suite.
        """

        return RetrievalEvaluationSummary(
            results=self.evaluate_cases(
                evaluation_cases
            )
        )
