class ChatProviderError(RuntimeError):
    """Raised when a configured chat provider cannot generate a response."""

class ChatProviderConfigurationError(ChatProviderError):
    """Raised when chat-provider configuration is invalid."""
