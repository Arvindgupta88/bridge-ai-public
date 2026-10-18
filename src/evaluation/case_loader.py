import json
from pathlib import Path

from src.evaluation.models import (
    RetrievalEvaluationCase,
)
from src.rag.retrieval_mode import RetrievalMode


SCHEMA_VERSION = 1

_REQUIRED_CASE_FIELDS = {
    "id",
    "question",
    "retrieval_mode",
    "expected_sources",
}

_ALLOWED_CASE_FIELDS = _REQUIRED_CASE_FIELDS | {
    "expected_pages",
    "top_k",
    "document_hash",
}


def load_evaluation_cases(
    path: Path,
) -> tuple[RetrievalEvaluationCase, ...]:
    """
    Load and validate retrieval evaluation cases from JSON.
    """

    try:
        raw_content = path.read_text(
            encoding="utf-8"
        )
    except OSError as error:
        raise ValueError(
            f"Unable to read evaluation cases from '{path}'."
        ) from error

    try:
        payload = json.loads(raw_content)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Evaluation case file '{path}' is not valid JSON."
        ) from error

    if not isinstance(payload, dict):
        raise ValueError(
            "Evaluation case file must contain a JSON object."
        )

    schema_version = payload.get("schema_version")

    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != SCHEMA_VERSION
    ):
        raise ValueError(
            "Unsupported evaluation-case schema version."
        )

    raw_cases = payload.get("cases")

    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(
            "Evaluation case file must contain at least one case."
        )

    evaluation_cases = []

    for position, raw_case in enumerate(
        raw_cases,
        start=1,
    ):
        try:
            evaluation_case = _parse_case(
                raw_case=raw_case,
            )
        except ValueError as error:
            raise ValueError(
                f"Invalid evaluation case {position}: {error}"
            ) from error

        evaluation_cases.append(
            evaluation_case
        )

    case_ids = [
        evaluation_case.case_id
        for evaluation_case in evaluation_cases
    ]

    if len(case_ids) != len(set(case_ids)):
        raise ValueError(
            "Evaluation case IDs must be unique."
        )

    return tuple(evaluation_cases)


def _parse_case(
    raw_case: object,
) -> RetrievalEvaluationCase:
    if not isinstance(raw_case, dict):
        raise ValueError(
            "Case must be a JSON object."
        )

    missing_fields = (
        _REQUIRED_CASE_FIELDS
        - raw_case.keys()
    )

    if missing_fields:
        raise ValueError(
            "Missing required field(s): "
            + ", ".join(sorted(missing_fields))
            + "."
        )

    unknown_fields = (
        raw_case.keys()
        - _ALLOWED_CASE_FIELDS
    )

    if unknown_fields:
        raise ValueError(
            "Unknown field(s): "
            + ", ".join(sorted(unknown_fields))
            + "."
        )

    case_id = raw_case["id"]
    question = raw_case["question"]
    raw_mode = raw_case["retrieval_mode"]
    expected_sources = raw_case["expected_sources"]
    expected_pages = raw_case.get(
        "expected_pages",
        [],
    )
    top_k = raw_case.get(
        "top_k",
        5,
    )
    document_hash = raw_case.get(
        "document_hash"
    )

    if not isinstance(case_id, str):
        raise ValueError(
            "Field 'id' must be a string."
        )

    if not isinstance(question, str):
        raise ValueError(
            "Field 'question' must be a string."
        )

    if not isinstance(raw_mode, str):
        raise ValueError(
            "Field 'retrieval_mode' must be a string."
        )

    try:
        retrieval_mode = RetrievalMode(
            raw_mode
        )
    except ValueError as error:
        raise ValueError(
            f"Unsupported retrieval mode '{raw_mode}'."
        ) from error

    if (
        not isinstance(expected_sources, list)
        or any(
            not isinstance(source, str)
            for source in expected_sources
        )
    ):
        raise ValueError(
            "Field 'expected_sources' must be a list of strings."
        )

    if (
        not isinstance(expected_pages, list)
        or any(
            not isinstance(page, int)
            or isinstance(page, bool)
            for page in expected_pages
        )
    ):
        raise ValueError(
            "Field 'expected_pages' must be a list of integers."
        )

    if (
        document_hash is not None
        and not isinstance(document_hash, str)
    ):
        raise ValueError(
            "Field 'document_hash' must be a string or null."
        )

    return RetrievalEvaluationCase(
        case_id=case_id,
        question=question,
        retrieval_mode=retrieval_mode,
        expected_sources=tuple(
            expected_sources
        ),
        expected_pages=tuple(
            expected_pages
        ),
        top_k=top_k,
        document_hash=document_hash,
    )
