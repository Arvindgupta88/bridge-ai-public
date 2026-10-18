from pathlib import Path
import re

from src.core.logger import logger
from src.core.settings import (
    CHAT_PROVIDER,
    MAX_HISTORY,
    OLLAMA_CHAT_MODEL,
)
from src.llm.provider_factory import create_chat_provider
from src.llm.providers.exceptions import ChatProviderError
from src.prompts.prompt_profile import PromptProfile

PROMPT_DIRECTORY = Path(__file__).resolve().parent.parent / "prompts"

PROMPT_FILES = {
    PromptProfile.BANKING_MODERNIZATION: "bridge_ai_system.txt",
    PromptProfile.IBM_I_APPLICATION_MODERNIZATION:
        "ibm_i_application_modernization.txt",
    PromptProfile.CLOUD_ARCHITECTURE: "cloud_architecture.txt",
    PromptProfile.GENERAL_ASSISTANT: "general_assistant.txt",
}


def load_system_prompt(
    prompt_profile: PromptProfile = PromptProfile.BANKING_MODERNIZATION,
) -> str:
    """Loads the system prompt associated with the selected prompt profile.

    Invalid profile values safely fall back to General Assistant.
    Missing configured prompt files will raise a clear error.
    """
    if not isinstance(prompt_profile, PromptProfile):
        logger.warning(
            "Invalid prompt profile '%s', falling back to General Assistant",
            prompt_profile,
        )
        prompt_profile = PromptProfile.GENERAL_ASSISTANT

    prompt_file = PROMPT_DIRECTORY / PROMPT_FILES[prompt_profile]

    try:
        return prompt_file.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise FileNotFoundError(
            "Prompt file not found for profile "
            f"'{prompt_profile.value}': {prompt_file}"
        ) from error


MODEL_NAME = OLLAMA_CHAT_MODEL

GENERATION_OPTIONS = {
    "temperature": 0,
    "seed": 42,
    "num_predict": 800,
}


def _has_known_incomplete_finish_reason(
    provider_name: str,
    finish_reason: str | None,
) -> bool:
    """Return true only for provider-specific, known incomplete reasons."""
    if not isinstance(finish_reason, str):
        return False

    incomplete_reasons = {
        "anthropic": {"max_tokens", "model_context_window_exceeded"},
        "ollama": {"length"},
    }
    provider_reasons = incomplete_reasons.get(provider_name.strip().lower(), set())
    return finish_reason in provider_reasons


def _has_pathological_repetition(content: str) -> bool:
    """Detect only long, dominant sentence loops in unstructured prose.

    Five exact repeats of a sentence (60+ characters) or paragraph (100+
    characters) must make up 75% of eligible prose. Code, tables, and list
    items are excluded so repeated syntax or checklist entries are not rejected.
    """
    prose_paragraphs = []
    current_paragraph = []
    in_code_block = False
    lines = content.splitlines()
    table_row_indices = set()

    def is_table_separator(line: str) -> bool:
        cells = line.strip().strip("|").split("|")
        return (
            len(cells) >= 2
            and all(
                re.fullmatch(r":?-{3,}:?", cell.strip())
                for cell in cells
            )
        )

    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            table_row_indices.add(index)
        if is_table_separator(line):
            table_row_indices.add(index)
            for direction in (-1, 1):
                row_index = index + direction
                while 0 <= row_index < len(lines):
                    row = lines[row_index].strip()
                    if not row or "|" not in row:
                        break
                    table_row_indices.add(row_index)
                    row_index += direction

    def finish_paragraph() -> None:
        if current_paragraph:
            prose_paragraphs.append(" ".join(current_paragraph))
            current_paragraph.clear()

    for index, line in enumerate(lines):
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            finish_paragraph()
            continue
        if in_code_block:
            continue
        stripped = line.strip()
        if (
            not stripped
            or index in table_row_indices
            or re.match(r"^(?:[-*+]\s+|\d+[.)]\s+)", stripped)
        ):
            finish_paragraph()
            continue
        current_paragraph.append(stripped)
    finish_paragraph()

    sentences = [
        sentence.strip()
        for paragraph in prose_paragraphs
        for sentence in re.split(r"(?<=[.!?])\s+", paragraph)
        if sentence.strip()
    ]

    def is_dominantly_repeated(units: list[str], minimum_length: int) -> bool:
        eligible = [unit for unit in units if len(unit) >= minimum_length]
        if not eligible:
            return False

        counts: dict[str, int] = {}
        for unit in eligible:
            normalized = " ".join(unit.casefold().split())
            counts[normalized] = counts.get(normalized, 0) + 1

        repeated_characters = sum(
            len(unit) * count
            for unit, count in counts.items()
            if count >= 5
        )
        eligible_characters = sum(len(unit) for unit in eligible)
        return (
            repeated_characters >= 5 * minimum_length
            and repeated_characters / eligible_characters >= 0.75
        )

    return (
        is_dominantly_repeated(sentences, minimum_length=60)
        or is_dominantly_repeated(prose_paragraphs, minimum_length=100)
    )


def ask_llm(
    messages: list,
    context: str | None = None,
    grounding_instructions: str | None = None,
    prompt_profile: PromptProfile = PromptProfile.BANKING_MODERNIZATION,
    provider_name: str = CHAT_PROVIDER,
    model: str = MODEL_NAME,
) -> str:
    """Generate a response using the configured chat provider."""
    logger.info("Conversation length=%s", len(messages))
    logger.info(
        "Sending request to chat provider '%s' model='%s'",
        provider_name,
        model,
    )

    system_prompt = load_system_prompt(prompt_profile)

    if grounding_instructions:
        system_prompt += f"""

        Answering policy:
        {grounding_instructions}
        """

    if context:
        system_prompt += f"""
        Use the following document context when answering.
        If the answer is not contained in the document context,
        say so instead of inventing information.

        Document Context:
        {context}
        """

    if grounding_instructions:
        system_prompt += (
            "\nFinal response rules: Answer briefly and directly. Do not add "
            "code or examples unless requested and supported. Do not guess "
            "technical behavior. When using evidence, use only established "
            "facts and put its exact filename citation before the sentence's "
            "final period. Never infer a page, scope, or other citation field; "
            "omit fields absent from that excerpt's metadata."
        )

    prepared_messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        *(messages[-MAX_HISTORY:] if MAX_HISTORY else []),
    ]

    try:
        provider = create_chat_provider(
            provider_name=provider_name,
            model=model,
            options=GENERATION_OPTIONS,
        )

        response = provider.generate(prepared_messages)

        logger.info(
            "Response received "
            "(finish_reason=%s input_tokens=%s output_tokens=%s)",
            response.finish_reason,
            response.input_tokens,
            response.output_tokens,
        )

        if _has_known_incomplete_finish_reason(
            provider_name,
            response.finish_reason,
        ):
            raise ChatProviderError(
                "The model response was incomplete because a token or context limit was reached."
            )

        if _has_pathological_repetition(response.content):
            raise ChatProviderError(
                "The model response contained excessive repetition and could not be completed."
            )

        return response.content

    except ChatProviderError as error:
        # Provider adapters attach the SDK exception as the cause. Log only
        # the sanitized application error; SDK messages can contain request
        # details or credentials.
        logger.error(
            "LLM request failed (provider=%s model=%s): %s",
            provider_name,
            model,
            error,
        )
        raise
