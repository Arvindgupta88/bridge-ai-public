import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from src.core.settings import (
    EMBED_MODEL,
    PROJECT_ROOT,
)
from src.evaluation.case_loader import (
    load_evaluation_cases,
)
from src.evaluation.evaluator import (
    RetrievalEvaluator,
)
from src.evaluation.models import (
    RetrievalEvaluationSummary,
)
from src.evaluation.reporting import (
    build_report,
    format_console_report,
)
from src.rag.pipeline import RagPipeline
from src.services.startup_service import (
    StartupService,
)


DEFAULT_CASES_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "retrieval_cases.json"
)


def run_evaluation(
    cases_path: Path,
) -> RetrievalEvaluationSummary:
    """
    Synchronize the knowledge library and run its evaluation suite.
    """

    evaluation_cases = load_evaluation_cases(
        cases_path
    )

    pipeline = RagPipeline()

    startup_service = StartupService(
        pipeline=pipeline
    )
    synchronization_summary = (
        startup_service.initialize()
    )

    failed_documents = (
        synchronization_summary.get(
            "failed",
            0,
        )
    )

    if failed_documents:
        raise RuntimeError(
            "Knowledge synchronization failed for "
            f"{failed_documents} document(s)."
        )

    evaluator = RetrievalEvaluator(
        pipeline=pipeline
    )

    return evaluator.evaluate_suite(
        evaluation_cases
    )


def write_json_report(
    report: dict[str, object],
    output_path: Path,
) -> None:
    """
    Write a machine-readable evaluation report.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Bridge AI retrieval quality "
            "against version-controlled cases."
        )
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help=(
            "Path to the retrieval evaluation case file."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional path for a JSON diagnostic report."
        ),
    )

    return parser


def main(
    arguments: Sequence[str] | None = None,
) -> int:
    parser = build_argument_parser()
    parsed_arguments = parser.parse_args(
        arguments
    )

    try:
        summary = run_evaluation(
            cases_path=parsed_arguments.cases
        )

        console_report = format_console_report(
            summary=summary,
            embedding_model=EMBED_MODEL,
        )
        print(console_report)

        if parsed_arguments.output is not None:
            generated_at = datetime.now(
                timezone.utc
            ).isoformat()

            report = build_report(
                summary=summary,
                generated_at=generated_at,
                embedding_model=EMBED_MODEL,
            )
            report["cases_file"] = str(
                parsed_arguments.cases
            )

            write_json_report(
                report=report,
                output_path=parsed_arguments.output,
            )

            print()
            print(
                "JSON report written to: "
                f"{parsed_arguments.output}"
            )

    except (
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(
            f"Evaluation failed: {error}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
