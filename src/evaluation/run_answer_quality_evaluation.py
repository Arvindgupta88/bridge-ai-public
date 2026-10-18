"""Offline grading of saved answers, independent of retrieval/model startup."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from src.evaluation.answer_quality import (
    DEFAULT_CASES_PATH,
    evaluate,
    format_report,
    load_candidates,
    load_cases,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grade saved answers against the offline answer-quality cases."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument(
        "--answers",
        type=Path,
        required=True,
        help="JSON file containing saved candidate answers.",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = build_argument_parser().parse_args(arguments)
    try:
        cases = load_cases(parsed.cases)
        candidates = load_candidates(
            parsed.answers,
            expected_case_ids={case.case_id for case in cases},
        )
        results = evaluate(cases, candidates)
        print(format_report(results))
    except (OSError, ValueError) as error:
        print(f"Answer-quality evaluation failed: {error}", file=sys.stderr)
        return 2
    return 1 if any(result.status != "pass" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
