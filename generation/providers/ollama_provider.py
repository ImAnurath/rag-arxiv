import httpx
from loguru import logger
from .base import BaseLLMProvider
from config import settings


class OllamaProvider(BaseLLMProvider):
    def __init__(
        self,
        base_url: str = settings.OLLAMA_BASE_URL,
        model: str = settings.LLM_MODEL,
        max_tokens: int = settings.LLM_MAX_TOKENS,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        logger.info(f"Ollama provider ready — model: {model} @ {base_url}")

    def complete(self, system: str, user: str) -> tuple[str, dict]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "stream": False,
            "options": {"num_predict": self.max_tokens},
        }
        with httpx.Client(timeout=120) as client:
            response = client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

        text = data["message"]["content"]
        # Ollama reports token counts in eval_count / prompt_eval_count
        usage = {
            "input_tokens":  data.get("prompt_eval_count", 0),
            "output_tokens": data.get("eval_count", 0),
        }
        return text, usage

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=5) as client:
                r = client.get(f"{self.base_url}/api/tags")
                return r.status_code == 200
        except Exception:
            return False