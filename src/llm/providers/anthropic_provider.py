import anthropic

from src.llm.providers.base import (
    ChatProvider,
    ChatProviderResponse,
)
from src.llm.providers.exceptions import (
    ChatProviderConfigurationError,
    ChatProviderError,
)


class AnthropicProvider(ChatProvider):
    """Chat-provider implementation backed by the Anthropic API."""

    DEFAULT_MAX_TOKENS = 800

    def __init__(
        self,
        model: str,
        api_key: str | None,
        options: dict[str, object] | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ChatProviderConfigurationError(
                "Anthropic model name must be a non-empty string."
            )

        if not isinstance(api_key, str) or not api_key.strip():
            raise ChatProviderConfigurationError(
                "Anthropic API key must be a non-empty string."
            )

        self._model = model.strip()
        self._options = dict(options or {})
        self._max_tokens = self._get_max_tokens()
        self._temperature = self._get_temperature()
        self._client = anthropic.Anthropic(
            api_key=api_key.strip(),
        )

    def generate(
        self,
        messages: list[dict[str, str]],
    ) -> ChatProviderResponse:
        system_prompt, conversation_messages = (
            self._prepare_messages(messages)
        )

        request: dict[str, object] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": conversation_messages,
        }

        if system_prompt is not None:
            request["system"] = system_prompt

        if self._temperature is not None:
            request["temperature"] = self._temperature

        try:
            response = self._client.messages.create(**request)
        except anthropic.AnthropicError as error:
            raise ChatProviderError(
                "Anthropic failed to generate a response "
                f"using model '{self._model}'."
            ) from error

        content = self._extract_text_content(response.content)

        if not content:
            raise ChatProviderError(
                "Anthropic returned an empty chat response."
            )

        usage = getattr(response, "usage", None)

        return ChatProviderResponse(
            content=content,
            finish_reason=getattr(response, "stop_reason", None),
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
        )

    def _get_max_tokens(self) -> int:
        max_tokens = self._options.get(
            "num_predict",
            self.DEFAULT_MAX_TOKENS,
        )

        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens <= 0
        ):
            raise ChatProviderConfigurationError(
                "Anthropic max tokens must be a positive integer."
            )

        return max_tokens

    def _get_temperature(self) -> float | None:
        temperature = self._options.get("temperature")

        if temperature is None:
            return None

        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not 0 <= temperature <= 1
        ):
            raise ChatProviderConfigurationError(
                "Anthropic temperature must be between 0 and 1."
            )

        return float(temperature)

    @staticmethod
    def _prepare_messages(
        messages: list[dict[str, str]],
    ) -> tuple[str | None, list[dict[str, str]]]:
        if not isinstance(messages, list) or not messages:
            raise ChatProviderConfigurationError(
                "Anthropic requires at least one message."
            )

        system_parts: list[str] = []
        conversation_messages: list[dict[str, str]] = []

        for message in messages:
            role = message.get("role")
            content = message.get("content")

            if role not in {"system", "user", "assistant"}:
                raise ChatProviderConfigurationError(
                    f"Unsupported Anthropic message role: {role}"
                )

            if not isinstance(content, str) or not content.strip():
                raise ChatProviderConfigurationError(
                    "Anthropic message content must be a non-empty string."
                )

            if role == "system":
                system_parts.append(content)
            else:
                conversation_messages.append(
                    {
                        "role": role,
                        "content": content,
                    }
                )

        if not conversation_messages:
            raise ChatProviderConfigurationError(
                "Anthropic requires a user or assistant message."
            )

        system_prompt = (
            "\n\n".join(system_parts)
            if system_parts
            else None
        )

        return system_prompt, conversation_messages

    @staticmethod
    def _extract_text_content(content_blocks: list) -> str:
        text_parts = [
            block.text
            for block in content_blocks
            if (
                getattr(block, "type", None) == "text"
                and isinstance(getattr(block, "text", None), str)
                and block.text.strip()
            )
        ]

        return "\n".join(text_parts).strip()
