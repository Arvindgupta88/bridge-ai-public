from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass(frozen=True)
class ChatProviderResponse:
    """Provider-independent response returned by a chat provider."""

    content: str
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

class ChatProvider(ABC):
    """Contract implemented by every chat model provider."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
    ) -> ChatProviderResponse:
        """Generate a response from prepared conversation messages."""
