import anthropic
from loguru import logger
from .base import BaseLLMProvider
from config import settings


class AnthropicProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: str = settings.LLM_API_KEY,
        model: str = settings.LLM_MODEL,
        max_tokens: int = settings.LLM_MAX_TOKENS,
    ):
        if not api_key or not api_key.startswith("sk-ant"):
            raise ValueError(
                "LLM_API_KEY is missing or invalid for Anthropic provider. "
                "Set it in your .env file."
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        logger.info(f"Anthropic provider ready — model: {model}")

    def complete(self, system: str, user: str) -> tuple[str, dict]:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        usage = {
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
        }
        return message.content[0].text, usage

    def health_check(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False