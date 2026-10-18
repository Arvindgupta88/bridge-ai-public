from dataclasses import dataclass

from src.rag.models import RetrievedChunk
from src.rag.retrieval_mode import RetrievalMode

from src.evaluation.metrics import (
    calculate_hit_rate,
    calculate_mean_reciprocal_rank,
)

@dataclass(frozen=True)
class RetrievalEvaluationCase:
    """
    Defines one repeatable retrieval evaluation question.
    """

    case_id: str
    question: str
    retrieval_mode: RetrievalMode
    expected_sources: tuple[str, ...]
    expected_pages: tuple[int, ...] = ()
    top_k: int = 5
    document_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError(
                "Evaluation case ID must not be blank."
            )

        if not self.question.strip():
            raise ValueError(
                "Evaluation question must not be blank."
            )

        if not isinstance(
            self.retrieval_mode,
            RetrievalMode,
        ):
            raise ValueError(
                "Evaluation retrieval mode is invalid."
            )

        if self.retrieval_mode == RetrievalMode.GENERAL:
            raise ValueError(
                "General mode does not perform retrieval."
            )

        if (
            self.retrieval_mode == RetrievalMode.UPLOADED
            and (
                self.document_hash is None
                or not self.document_hash.strip()
            )
        ):
            raise ValueError(
                "Uploaded evaluation cases require a document hash."
            )

        if (
            self.document_hash is not None
            and not self.document_hash.strip()
        ):
            raise ValueError(
                "Evaluation document hash must not be blank."
            )

        if (
            not isinstance(self.top_k, int)
            or isinstance(self.top_k, bool)
            or self.top_k < 1
        ):
            raise ValueError(
                "Evaluation top_k must be greater than zero."
            )

        for source in self.expected_sources:
            if not isinstance(source, str) or not source.strip():
                raise ValueError(
                    "Expected source names must not be blank."
                )

        for page in self.expected_pages:
            if (
                not isinstance(page, int)
                or isinstance(page, bool)
                or page < 1
            ):
                raise ValueError(
                    "Expected page numbers must be positive integers."
                )

        if (
            self.expected_pages
            and not self.expected_sources
        ):
            raise ValueError(
                "Expected pages require at least one expected source."
            )

    @property
    def is_scored(self) -> bool:
        """
        Whether this case contributes to retrieval-quality metrics.
        """

        return bool(self.expected_sources)


@dataclass(frozen=True)
class RetrievalEvaluationResult:
    """
    Contains the measured outcome of one evaluation case.
    """

    case: RetrievalEvaluationCase
    chunks: tuple[RetrievedChunk, ...]
    latency_ms: float
    first_relevant_rank: int | None = None

    def __post_init__(self) -> None:
        if self.latency_ms < 0:
            raise ValueError(
                "Retrieval latency must not be negative."
            )

        rank = self.first_relevant_rank

        if rank is not None:
            if (
                not isinstance(rank, int)
                or isinstance(rank, bool)
                or rank < 1
            ):
                raise ValueError(
                    "First relevant rank must be a positive integer."
                )

            if rank > self.case.top_k:
                raise ValueError(
                    "First relevant rank cannot exceed top_k."
                )

            if not self.case.is_scored:
                raise ValueError(
                    "Diagnostic-only cases cannot have a relevant rank."
                )

    @property
    def hit(self) -> bool | None:
        """
        Return pass/fail for scored cases and None for diagnostic cases.
        """

        if not self.case.is_scored:
            return None

        return self.first_relevant_rank is not None

    @property
    def reciprocal_rank(self) -> float | None:
        """
        Return reciprocal rank for scored cases.

        A scored miss contributes zero. Diagnostic cases are excluded.
        """

        if not self.case.is_scored:
            return None

        if self.first_relevant_rank is None:
            return 0.0

        return 1 / self.first_relevant_rank

@dataclass(frozen=True)
class RetrievalEvaluationSummary:
    """
    Aggregates the results of one retrieval evaluation run.
    """

    results: tuple[RetrievalEvaluationResult, ...]

    def __post_init__(self) -> None:
        if not self.results:
            raise ValueError(
                "At least one evaluation result is required."
            )

        case_ids = [
            result.case.case_id
            for result in self.results
        ]

        if len(case_ids) != len(set(case_ids)):
            raise ValueError(
                "Evaluation result case IDs must be unique."
            )

    @property
    def scored_results(
        self,
    ) -> tuple[RetrievalEvaluationResult, ...]:
        return tuple(
            result
            for result in self.results
            if result.case.is_scored
        )

    @property
    def total_cases(self) -> int:
        return len(self.results)

    @property
    def scored_cases(self) -> int:
        return len(self.scored_results)

    @property
    def diagnostic_cases(self) -> int:
        return self.total_cases - self.scored_cases

    @property
    def passed_cases(self) -> int:
        return sum(
            result.hit is True
            for result in self.results
        )

    @property
    def failed_cases(self) -> int:
        return sum(
            result.hit is False
            for result in self.results
        )

    @property
    def top_k_values(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    result.case.top_k
                    for result in self.results
                }
            )
        )

    @property
    def hit_rate_at_k(self) -> float | None:
        scored_results = self.scored_results

        if not scored_results:
            return None

        return calculate_hit_rate(
            [
                result.hit is True
                for result in scored_results
            ]
        )

    @property
    def mean_reciprocal_rank(
        self,
    ) -> float | None:
        scored_results = self.scored_results

        if not scored_results:
            return None

        return calculate_mean_reciprocal_rank(
            [
                result.first_relevant_rank
                for result in scored_results
            ]
        )

    @property
    def mean_latency_ms(self) -> float:
        return (
            sum(
                result.latency_ms
                for result in self.results
            )
            / self.total_cases
        )
