from dataclasses import dataclass
from pathlib import Path

from src.core.settings import (
    CHAT_PROVIDER,
    CHROMA_PATH,
    CHROMA_COLLECTION,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_MODEL,
    KNOWLEDGE_INDEX_FILE,
    KNOWLEDGE_PATH,
    MAX_HISTORY,
    OLLAMA_CHAT_MODEL,
)
from src.llm.provider_catalog import (
    PROVIDER_CATALOG,
    ProviderName,
)


@dataclass(frozen=True)
class ConfigIssue:
    """One configuration validation finding."""

    level: str
    message: str


@dataclass(frozen=True)
class ConfigValidationResult:
    """Result of validating Bridge AI configuration."""

    issues: tuple[ConfigIssue, ...]

    @property
    def is_valid(self) -> bool:
        """Return True when no configuration errors exist."""
        return not any(
            issue.level == "ERROR"
            for issue in self.issues
        )


def validate_configuration(
    provider_name: str = CHAT_PROVIDER,
    model: str = OLLAMA_CHAT_MODEL,
) -> ConfigValidationResult:
    """Validate the current Bridge AI configuration."""

    issues: list[ConfigIssue] = []

    _validate_provider_and_model(
        provider_name=provider_name,
        model=model,
        issues=issues,
    )

    _validate_embedding_configuration(issues)
    _validate_paths(issues)
    _validate_retrieval_configuration(issues)

    return ConfigValidationResult(
        issues=tuple(issues)
    )


def _validate_provider_and_model(
    provider_name: str,
    model: str,
    issues: list[ConfigIssue],
) -> None:
    """Validate provider, model and provider credentials."""

    if not isinstance(provider_name, str) or not provider_name.strip():
        issues.append(
            ConfigIssue(
                level="ERROR",
                message="Chat provider must be a non-empty string.",
            )
        )
        return

    try:
        provider_id = ProviderName(
            provider_name.strip().lower()
        )
    except ValueError:
        issues.append(
            ConfigIssue(
                level="ERROR",
                message=f"Unknown chat provider '{provider_name.strip()}'.",
            )
        )
        return

    definition = PROVIDER_CATALOG[provider_id]

    if not definition.adapter_implemented:
        issues.append(
            ConfigIssue(
                level="ERROR",
                message=(
                    f"Chat provider '{definition.display_name}' "
                    "is not implemented."
                ),
            )
        )
        return

    if not isinstance(model, str) or not model.strip():
        issues.append(
            ConfigIssue(
                level="ERROR",
                message="Chat model must be a non-empty string.",
            )
        )
        return

    normalized_model = model.strip()

    if normalized_model not in definition.models:
        configured_models = ", ".join(definition.models)

        issues.append(
            ConfigIssue(
                level="ERROR",
                message=(
                    f"Model '{normalized_model}' is not configured "
                    f"for provider '{definition.display_name}'. "
                    f"Configured models: {configured_models}."
                ),
            )
        )
        return

    unavailable_reason = definition.unavailable_reason()

    if unavailable_reason is not None:
        issues.append(
            ConfigIssue(
                level="ERROR",
                message=(
                    f"Chat provider '{definition.display_name}' "
                    f"is unavailable: {unavailable_reason}."
                ),
            )
        )


def _validate_embedding_configuration(
    issues: list[ConfigIssue],
) -> None:
    """Validate embedding configuration."""

    if not isinstance(EMBED_MODEL, str) or not EMBED_MODEL.strip():
        issues.append(
            ConfigIssue(
                level="ERROR",
                message="Embedding model must be a non-empty string.",
            )
        )


def _validate_paths(
    issues: list[ConfigIssue],
) -> None:
    """Validate configured filesystem locations."""

    configured_paths = {
        "knowledge path": KNOWLEDGE_PATH,
        "Chroma path": CHROMA_PATH,
        "knowledge metadata file": KNOWLEDGE_INDEX_FILE,
    }

    for name, path in configured_paths.items():
        if isinstance(path, Path):
            valid = bool(str(path))
        elif isinstance(path, str):
            valid = bool(path.strip())
        else:
            valid = False
        if not valid:
            issues.append(
                ConfigIssue(
                    level="ERROR",
                    message=f"Configured {name} is not a valid Path.",
                )
            )


def _validate_retrieval_configuration(
    issues: list[ConfigIssue],
) -> None:
    """Validate retrieval-related numeric and structural settings."""

    if (
        isinstance(CHUNK_SIZE, bool)
        or not isinstance(CHUNK_SIZE, int)
        or CHUNK_SIZE <= 0
    ):
        issues.append(
            ConfigIssue(
                level="ERROR",
                message="Chunk size must be a positive integer.",
            )
        )

    if (
        isinstance(CHUNK_OVERLAP, bool)
        or not isinstance(CHUNK_OVERLAP, int)
        or CHUNK_OVERLAP < 0
    ):
        issues.append(
            ConfigIssue(
                level="ERROR",
                message="Chunk overlap must be a non-negative integer.",
            )
        )

    if (
        isinstance(CHUNK_SIZE, int)
        and isinstance(CHUNK_OVERLAP, int)
        and CHUNK_OVERLAP >= CHUNK_SIZE
    ):
        issues.append(
            ConfigIssue(
                level="ERROR",
                message="Chunk overlap must be smaller than chunk size.",
            )
        )

    if (
        isinstance(MAX_HISTORY, bool)
        or not isinstance(MAX_HISTORY, int)
        or MAX_HISTORY < 0
    ):
        issues.append(
            ConfigIssue(
                level="ERROR",
                message="Maximum history must be a non-negative integer.",
            )
        )

    if not isinstance(CHROMA_COLLECTION, str) or not CHROMA_COLLECTION.strip():
        issues.append(
            ConfigIssue(
                level="ERROR",
                message="Chroma collection name must be non-empty.",
            )
        )
