import ollama

from src.llm.providers.base import ChatProvider, ChatProviderResponse
from src.llm.providers.exceptions import (
    ChatProviderConfigurationError,
    ChatProviderError,
)

class OllamaProvider(ChatProvider):
    """Chat-provider implementation backed by the local Ollama service."""

    def __init__(
            self,
            model: str,
            options: dict[str, object] | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ChatProviderConfigurationError(
                "Ollama model name must be a non-empty string."
            )

        self._model = model
        self._options = dict(options or {})

    def generate(
            self,
            messages: list[dict[str, str]],
    ) -> ChatProviderResponse:
        try:
            response = ollama.chat(
                model=self._model,
                messages=messages,
                options=self._options,
            )
            content = response.message.content
        except Exception as error:
            raise ChatProviderError(
                f"Ollama failed to generate a response "
                f"using model '{self._model}'."
            ) from error

        if not isinstance(content, str) or not content.strip():
            raise ChatProviderError(
                "Ollama returned an empty chat response."
            )

        return ChatProviderResponse(
            content=content,
            finish_reason=getattr(response, "done_reason", None),
            input_tokens=getattr(response, "prompt_eval_count", None),
            output_tokens=getattr(response, "eval_count", None)
        )