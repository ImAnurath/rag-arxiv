from openai import OpenAI
from loguru import logger
from .base import BaseLLMProvider
from config import settings


class OpenAIProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: str = settings.LLM_API_KEY,
        base_url: str = settings.OPENAI_BASE_URL,
        model: str = settings.LLM_MODEL,
        max_tokens: int = settings.LLM_MAX_TOKENS,
    ):
        self.client = OpenAI(api_key=api_key or "none", base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens
        logger.info(f"OpenAI-compatible provider ready — model: {model} @ {base_url}")

    def complete(self, system: str, user: str) -> tuple[str, dict]:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        text = response.choices[0].message.content
        usage = {
            "input_tokens":  response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
        }
        return text, usage

    def health_check(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False