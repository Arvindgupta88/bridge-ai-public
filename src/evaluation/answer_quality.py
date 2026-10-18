"""Deterministic grading for saved answers; imports no runtime RAG/provider code."""

import json
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CASES_PATH = Path(__file__).resolve().parents[2] / "evaluation" / "answer_quality_cases.json"
_FINDING_CATEGORIES = {
    "incorrect_answer",
    "unsupported_claim",
    "attribution_failure",
    "provider_failure",
    "retrieval_failure",
    "unverified",
}
_OPERATIONAL_ERRORS = {"provider_failure", "retrieval_failure"}
_RETRIEVAL_MODES = {"general", "library", "uploaded", "hybrid"}


@dataclass(frozen=True)
class Finding:
    category: str
    detail: str


@dataclass(frozen=True)
class Case:
    case_id: str
    question: str
    retrieval_mode: str
    evidence: tuple[dict, ...]
    expected_facts: tuple[dict, ...]
    disallowed_claims: tuple[dict, ...]
    sources: tuple[dict, ...]
    abstention_signals: tuple[str, ...] = ()


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    status: str
    findings: tuple[Finding, ...]


def load_cases(path: Path = DEFAULT_CASES_PATH) -> tuple[Case, ...]:
    """Load and validate the existing five-case answer-quality dataset."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to load answer-quality cases from '{path}'.") from error

    if not isinstance(payload, dict):
        raise ValueError("Answer-quality case file must contain a JSON object.")
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError("Unsupported answer-quality case schema.")

    rows = payload.get("cases")
    if not isinstance(rows, list) or len(rows) != 5:
        raise ValueError("Answer-quality dataset must contain exactly five cases.")

    cases = []
    for position, row in enumerate(rows, start=1):
        cases.append(_parse_case(row, position))

    _require_unique_ids([case.case_id for case in cases], "case")
    return tuple(cases)


def _parse_case(row: object, position: int) -> Case:
    required = (
        "id", "question", "retrieval_mode", "evidence",
        "expected_facts", "disallowed_claims", "sources",
    )
    if not isinstance(row, dict) or not all(key in row for key in required):
        raise ValueError(f"Case {position} must contain all required fields.")

    case_id = row["id"]
    question = row["question"]
    mode = row["retrieval_mode"]
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError(f"Case {position} ID must be a nonempty string.")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Case '{case_id}' question must be a nonempty string.")
    if not isinstance(mode, str) or mode not in _RETRIEVAL_MODES:
        raise ValueError(f"Case '{case_id}' has an unsupported retrieval mode.")

    list_fields = ("evidence", "expected_facts", "disallowed_claims", "sources")
    for field in list_fields:
        if not isinstance(row[field], list):
            raise ValueError(f"Case '{case_id}' field '{field}' must be a list.")

    evidence = row["evidence"]
    facts = row["expected_facts"]
    claims = row["disallowed_claims"]
    sources = row["sources"]
    _validate_evidence(case_id, evidence)
    _validate_facts(case_id, facts)
    _validate_claims(case_id, claims)
    _validate_sources(case_id, sources)

    source_ids = {source["id"] for source in sources}
    evidence_by_id = {record["id"]: record for record in evidence}
    if set(evidence_by_id) != source_ids:
        raise ValueError(f"Case '{case_id}' source and evidence IDs must match.")
    for source in sources:
        metadata = evidence_by_id[source["id"]]["metadata"]
        if metadata["source"] not in source["aliases"]:
            raise ValueError(f"Case '{case_id}' source filename must match a source alias.")
        required_type = source.get("required_source_type")
        if required_type is not None and metadata["source_type"] != required_type:
            raise ValueError(f"Case '{case_id}' required_source_type must match evidence metadata.")
        expected_type = {"library": "library", "uploaded": "upload"}.get(mode)
        if expected_type is not None and metadata["source_type"] != expected_type:
            raise ValueError(f"Case '{case_id}' source_type is inconsistent with retrieval mode.")
    for fact in facts:
        unknown_sources = set(fact["source_ids"]) - source_ids
        if unknown_sources:
            raise ValueError(
                f"Case '{case_id}' fact '{fact['id']}' references unknown source ID(s): "
                + ", ".join(sorted(unknown_sources))
            )

    abstention_signals = row.get("abstention_signals", [])
    if abstention_signals and not _nonempty_string_list(abstention_signals):
        raise ValueError(f"Case '{case_id}' abstention_signals must be a list of nonempty strings.")
    if not isinstance(abstention_signals, list):
        raise ValueError(f"Case '{case_id}' abstention_signals must be a list of nonempty strings.")

    return Case(
        case_id=case_id,
        question=question,
        retrieval_mode=mode,
        evidence=tuple(evidence),
        expected_facts=tuple(facts),
        disallowed_claims=tuple(claims),
        sources=tuple(sources),
        abstention_signals=tuple(abstention_signals),
    )


def _validate_evidence(case_id: str, evidence: list) -> None:
    ids = []
    for record in evidence:
        if not isinstance(record, dict):
            raise ValueError(f"Case '{case_id}' evidence records must be objects.")
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"Case '{case_id}' evidence record IDs must be nonempty strings.")
        if not isinstance(record.get("text"), str) or not record["text"].strip():
            raise ValueError(f"Case '{case_id}' evidence '{record_id}' text must be nonempty.")
        if not isinstance(record.get("metadata"), dict):
            raise ValueError(f"Case '{case_id}' evidence '{record_id}' metadata must be an object.")
        metadata = record["metadata"]
        if not isinstance(metadata.get("source"), str) or not metadata["source"].strip():
            raise ValueError(f"Case '{case_id}' evidence '{record_id}' source must be a nonempty string.")
        if metadata.get("source_type") not in ("library", "upload"):
            raise ValueError(f"Case '{case_id}' evidence '{record_id}' source_type must be library or upload.")
        ids.append(record_id)
    _require_unique_ids(ids, f"evidence record in case '{case_id}'")


def _validate_facts(case_id: str, facts: list) -> None:
    ids = []
    for fact in facts:
        if not isinstance(fact, dict):
            raise ValueError(f"Case '{case_id}' expected facts must be objects.")
        fact_id = fact.get("id")
        if not isinstance(fact_id, str) or not fact_id.strip():
            raise ValueError(f"Case '{case_id}' expected-fact IDs must be nonempty strings.")
        if not _nonempty_string_list(fact.get("signals")):
            raise ValueError(f"Case '{case_id}' fact '{fact_id}' signals must be a nonempty list of strings.")
        source_ids = fact.get("source_ids")
        if not isinstance(source_ids, list) or any(not isinstance(item, str) or not item.strip() for item in source_ids):
            raise ValueError(f"Case '{case_id}' fact '{fact_id}' source_ids must be a list of nonempty strings.")
        _require_unique_ids(source_ids, f"source reference in fact '{fact_id}'")
        ids.append(fact_id)
    _require_unique_ids(ids, f"expected fact in case '{case_id}'")


def _validate_claims(case_id: str, claims: list) -> None:
    ids = []
    for claim in claims:
        if not isinstance(claim, dict):
            raise ValueError(f"Case '{case_id}' disallowed claims must be objects.")
        claim_id = claim.get("id")
        if not isinstance(claim_id, str) or not claim_id.strip():
            raise ValueError(f"Case '{case_id}' disallowed-claim IDs must be nonempty strings.")
        if not _nonempty_string_list(claim.get("signals")):
            raise ValueError(f"Case '{case_id}' claim '{claim_id}' signals must be a nonempty list of strings.")
        ids.append(claim_id)
    _require_unique_ids(ids, f"disallowed claim in case '{case_id}'")


def _validate_sources(case_id: str, sources: list) -> None:
    ids = []
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError(f"Case '{case_id}' sources must be objects.")
        source_id = source.get("id")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError(f"Case '{case_id}' source IDs must be nonempty strings.")
        if not _nonempty_string_list(source.get("aliases")):
            raise ValueError(f"Case '{case_id}' source '{source_id}' aliases must be a nonempty list of strings.")
        source_type = source.get("required_source_type")
        if source_type is not None and (not isinstance(source_type, str) or not source_type.strip()):
            raise ValueError(f"Case '{case_id}' source '{source_id}' required_source_type must be a nonempty string.")
        if source_type is not None and source_type not in ("library", "upload"):
            raise ValueError(f"Case '{case_id}' required_source_type must be library or upload.")
        ids.append(source_id)
    _require_unique_ids(ids, f"source in case '{case_id}'")


def _nonempty_string_list(value: object) -> bool:
    return isinstance(value, list) and bool(value) and all(
        isinstance(item, str) and bool(item.strip()) for item in value
    )


def _require_unique_ids(values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {label} IDs are not allowed.")


def load_candidates(path: Path, expected_case_ids: set[str]) -> dict[str, dict]:
    """Load and validate saved candidate answers keyed by known case ID."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to load saved candidate answers from '{path}'.") from error
    rows = payload.get("answers") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("Candidate file must contain an 'answers' list.")

    result = {}
    for position, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Candidate record {position} must be an object.")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"Candidate record {position} case_id must be a nonempty string.")
        if case_id in result:
            raise ValueError(f"Duplicate candidate case_id '{case_id}'.")
        if case_id not in expected_case_ids:
            raise ValueError(f"Candidate references unknown case_id '{case_id}'.")
        if "answer" in row and not isinstance(row["answer"], str):
            raise ValueError(f"Candidate '{case_id}' answer must be a string when provided.")
        operational_error = row.get("operational_error")
        if operational_error is not None and (
            not isinstance(operational_error, str)
            or operational_error not in _OPERATIONAL_ERRORS
        ):
            raise ValueError(
                f"Candidate '{case_id}' operational_error must be provider_failure or retrieval_failure."
            )
        result[case_id] = row
    return result


def grade_answer(case: Case, candidate: dict) -> CaseResult:
    answer = candidate.get("answer", "")
    findings: list[Finding] = []
    operational_error = candidate.get("operational_error")
    if operational_error is not None:
        findings.append(Finding(operational_error, "Candidate generation/retrieval reported an operational failure."))
    if not answer.strip():
        if not findings:
            findings.append(Finding("incorrect_answer", "No answer or operational failure was supplied."))
        return CaseResult(case.case_id, "fail", tuple(findings))

    normalized = _normalize(answer)
    abstained = any(_contains(normalized, signal) for signal in case.abstention_signals)
    for claim in case.disallowed_claims:
        if _matches_any(normalized, claim["signals"]):
            category = "unsupported_claim" if not case.expected_facts else "incorrect_answer"
            findings.append(Finding(category, f"Disallowed claim present: {claim['id']}"))

    if case.expected_facts and not abstained:
        for fact in case.expected_facts:
            matching_sentences = _matching_fact_sentences(answer, fact["signals"])
            if not matching_sentences:
                findings.append(Finding("incorrect_answer", f"Expected fact missing: {fact['id']}"))
                continue
            required_sources = set(fact["source_ids"])
            if required_sources and not any(
                required_sources.issubset(_cited_sources(sentence, case.sources))
                for sentence in matching_sentences
            ):
                findings.append(Finding("attribution_failure", f"Missing or incorrect source attribution for fact: {fact['id']}"))
    elif case.abstention_signals:
        if not abstained and not any(f.category == "unsupported_claim" for f in findings):
            findings.append(Finding(
                "unverified",
                "No configured unsupported assertion or recognized abstention; expert review is required.",
            ))

    if any(f.category != "unverified" for f in findings):
        status = "fail"
    elif any(f.category == "unverified" for f in findings):
        status = "unverified"
    else:
        status = "fail" if findings else "pass"
    return CaseResult(case.case_id, status, tuple(findings))


def evaluate(cases: tuple[Case, ...], candidates: dict[str, dict]) -> tuple[CaseResult, ...]:
    """Grade candidates already saved to disk; omitted expected cases fail."""
    case_ids = {case.case_id for case in cases}
    unknown_ids = set(candidates) - case_ids
    if unknown_ids:
        raise ValueError("Candidate file contains unknown case ID(s): " + ", ".join(sorted(unknown_ids)))
    return tuple(grade_answer(case, candidates.get(case.case_id, {})) for case in cases)


def format_report(results: tuple[CaseResult, ...]) -> str:
    totals = {category: 0 for category in sorted(_FINDING_CATEGORIES)}
    lines = [
        "Bridge AI Offline Answer-Quality Evaluation",
        "Candidate answers are saved inputs; no model or retrieval is run.",
        "",
    ]
    for result in results:
        lines.append(f"[{result.status.upper()}] {result.case_id}")
        for finding in result.findings:
            totals[finding.category] = totals.get(finding.category, 0) + 1
            lines.append(f"  {finding.category}: {finding.detail}")
    lines.extend([
        "",
        f"Cases: {len(results)}",
        f"Passed: {sum(result.status == 'pass' for result in results)}",
        f"Failed: {sum(result.status == 'fail' for result in results)}",
        f"Unverified (review required): {sum(result.status == 'unverified' for result in results)}",
        "Findings by category:",
    ])
    lines.extend(f"  {category}: {count}" for category, count in sorted(totals.items()))
    return "\n".join(lines)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold().replace("–", "-").replace("—", "-")).strip()


def _normalize_proposition(text: str) -> str:
    normalized = _normalize(text)
    # Bounded surface variants for the configured Zero Trust propositions.
    normalized = normalized.replace("**", "").replace("`", "")
    normalized = re.sub(r"\s*\((?:pe|pep|pa)\)", "", normalized)
    normalized = re.sub(r"\bpep\b", "policy enforcement point", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


_CLOUD_CHARACTERISTICS = (
    "on-demand self-service", "broad network access", "resource pooling",
    "rapid elasticity", "measured service",
)


def _contains(normalized_text: str, signal: str) -> bool:
    # Keep role surface normalization out of citation alias matching.
    normalized_text = _normalize_proposition(normalized_text)
    normalized_signal = _normalize_proposition(signal)
    start = 0
    while (position := normalized_text.find(normalized_signal, start)) >= 0:
        end = position + len(normalized_signal)
        # Numeric fact signals must not match prefixes of a different number.
        if normalized_signal[-1].isdigit() and end < len(normalized_text) and normalized_text[end].isdigit():
            start = end
            continue
        if normalized_signal in _CLOUD_CHARACTERISTICS and re.match(
            r"\s+(?:is|are)\s+(?:not|never)\b|\s+isn't\b|\s+aren't\b",
            normalized_text[end:],
        ):
            start = end
            continue
        prefix = re.sub(r"[^a-z0-9%]+", " ", normalized_text[max(0, position - 40):position]).strip()
        if not re.search(r"(?:\bnot|\bnever|\bno|\bisnt|\bdoesnt|\bwithout)(?:\s+\w+){0,2}$", prefix):
            return True
        start = position + len(normalized_signal)
    return False


def _matches_any(normalized_text: str, signals: list[str]) -> bool:
    return any(_contains(normalized_text, signal) for signal in signals)


def _matching_fact_sentences(answer: str, signals: list[str]) -> tuple[str, ...]:
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    return tuple(
        sentence for sentence in sentences
        if _matches_any(_normalize(sentence), signals)
    )


def _cited_sources(answer: str, sources: tuple[dict, ...]) -> set[str]:
    normalized = _normalize(answer)
    cited = set()
    for source in sources:
        for alias in source["aliases"]:
            normalized_alias = _normalize(alias)
            for match in re.finditer(re.escape(normalized_alias), normalized):
                required_type = source.get("required_source_type")
                if required_type is not None:
                    # The label must immediately follow its own alias, before
                    # punctuation or another citation; neighboring labels cannot count.
                    citation = normalized[match.end():]
                    if not re.match(
                        rf"\s+source[_ -]?type\s*[:=]\s*{re.escape(required_type.casefold())}\b",
                        citation,
                    ):
                        continue
                cited.add(source["id"])
                break
            if source["id"] in cited:
                break
    return cited
