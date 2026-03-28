from .base import BaseLLMProvider
from config import settings


def get_provider() -> BaseLLMProvider:
    provider = settings.LLM_PROVIDER.lower().strip()

    if provider == "anthropic":
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider()

    elif provider == "ollama":
        from .ollama_provider import OllamaProvider
        return OllamaProvider()

    elif provider == "openai":
        from .openai_provider import OpenAIProvider
        return OpenAIProvider()

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'. "
            "Valid options: anthropic, ollama, openai"
        )