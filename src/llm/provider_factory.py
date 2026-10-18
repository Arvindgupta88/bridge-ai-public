import os

from src.llm.provider_catalog import (
    PROVIDER_CATALOG,
    ProviderName,
)
from src.llm.providers.anthropic_provider import AnthropicProvider
from src.llm.providers.base import ChatProvider
from src.llm.providers.exceptions import (
    ChatProviderConfigurationError,
)
from src.llm.providers.ollama_provider import OllamaProvider


def create_chat_provider(
    provider_name: str,
    model: str,
    options: dict[str, object] | None = None,
) -> ChatProvider:
    """Create the configured chat-provider implementation."""

    if not isinstance(provider_name, str) or not provider_name.strip():
        raise ChatProviderConfigurationError(
            "Chat-provider name must be a non-empty string."
        )

    normalized_name = provider_name.strip().lower()

    try:
        provider_id = ProviderName(normalized_name)
    except ValueError as error:
        known_providers = ", ".join(
            provider.value
            for provider in ProviderName
        )
        raise ChatProviderConfigurationError(
            f"Unknown chat provider '{provider_name}'. "
            f"Known providers: {known_providers}."
        ) from error

    definition = PROVIDER_CATALOG[provider_id]
    unavailable_reason = definition.unavailable_reason()

    if unavailable_reason is not None:
        raise ChatProviderConfigurationError(
            f"Chat provider '{definition.display_name}' "
            f"is unavailable: {unavailable_reason}."
        )

    if not isinstance(model, str) or not model.strip():
        raise ChatProviderConfigurationError(
            "Chat-model name must be a non-empty string."
        )
    normalized_model = model.strip()
    if normalized_model not in definition.models:
        configured_models = ", ".join(
            definition.models
        )
        raise ChatProviderConfigurationError(
            f"Model '{normalized_model}' is not configured for "
            f"provider '{definition.display_name}'. "
            f"Configured models: {configured_models}."
        )

    if provider_id == ProviderName.OLLAMA:
        return OllamaProvider(
            model=normalized_model,
            options=options,
        )

    if provider_id == ProviderName.ANTHROPIC:
        credential_name = (
            definition.credential_environment_variable
        )

        if credential_name is None:
            raise ChatProviderConfigurationError(
                "Anthropic credential configuration is missing."
            )

        return AnthropicProvider(
            model=normalized_model,
            api_key=os.getenv(credential_name),
            options=options,
        )

    raise ChatProviderConfigurationError(
        f"Chat-provider adapter is not implemented for "
        f"'{provider_id.value}'."
    )
