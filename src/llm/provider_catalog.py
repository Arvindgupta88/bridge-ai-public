import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from src.core.settings import OLLAMA_CHAT_MODEL

class ProviderName(str, Enum):
    """Provider identifiers used by configuration and the UI."""

    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"

@dataclass(frozen=True)
class ProviderDefinition:
    """Configuration and availability information for one provider."""

    name: ProviderName
    display_name: str
    models: tuple[str, ...]
    adapter_implemented: bool
    credential_environment_variable: str | None = None

    def is_available(
            self,
            environment: Mapping[str, str] | None = None,
    ) -> bool:
        """Return whether the provider can be selected."""

        if not self.adapter_implemented:
            return False

        if self.credential_environment_variable is None:
            return True

        environment_values = (
            os.environ
            if environment is None
            else environment
        )

        credential = environment_values.get(
            self.credential_environment_variable
        )

        return bool(
            isinstance(credential, str)
            and credential.strip()
        )

    def unavailable_reason(
            self,
            environment: Mapping[str, str] | None = None,
    ) -> str | None:
        """Explain why the provider cannot be currently selected."""

        if not self.adapter_implemented:
            return "Adapter not yet implemented"

        if not self.is_available(environment):
            return (
                f"{self.credential_environment_variable} "
                "is not configured"
            )

        return None

PROVIDER_CATALOG = {
    ProviderName.OLLAMA: ProviderDefinition(
        name=ProviderName.OLLAMA,
        display_name="Ollama",
        models=(OLLAMA_CHAT_MODEL, "qwen2.5:3b"),
        adapter_implemented=True,
    ),
    ProviderName.ANTHROPIC: ProviderDefinition(
        name=ProviderName.ANTHROPIC,
        display_name="Anthropic",
        models=(
            "claude-haiku-4-5",
            "claude-sonnet-5",
            "claude-opus-5",
        ),
        adapter_implemented=True,
        credential_environment_variable="ANTHROPIC_API_KEY",
    ),
    ProviderName.OPENAI: ProviderDefinition(
        name=ProviderName.OPENAI,
        display_name="OpenAI",
        models=(),
        adapter_implemented=False,
        credential_environment_variable="OPENAI_API_KEY",
    ),
    ProviderName.AZURE_OPENAI: ProviderDefinition(
        name=ProviderName.AZURE_OPENAI,
        display_name="Azure OpenAI",
        models=(),
        adapter_implemented=False,
        credential_environment_variable="AZURE_OPENAI_API_KEY",
    ),
}
