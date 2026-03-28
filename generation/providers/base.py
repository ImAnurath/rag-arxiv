from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    """
    Every provider implements this interface.
    The generator only ever calls complete() — it never knows
    which backend is running underneath.
    """

    @abstractmethod
    def complete(self, system: str, user: str) -> tuple[str, dict]:
        """
        Returns (answer_text, usage_dict).
        usage_dict should have 'input_tokens' and 'output_tokens' keys.
        Providers that don't report token counts return zeros.
        """
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """Returns True if the provider is reachable."""
        ...