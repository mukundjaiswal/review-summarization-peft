"""Loading and driving the base and tuned variants.

Both variants share this module and therefore share prompt construction and
decoding settings. That is deliberate: if the two paths differed, the measured
gap would include the difference in scaffolding and be reported as the effect of
fine-tuning.

Greedy decoding (``do_sample=False``) for the same reason — sampling noise
between two runs is indistinguishable from a real effect at these sample sizes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import lru_cache
from typing import Any, Final, Literal

from review_peft.prompts import strip_prompt
from review_peft.settings import Settings, get_settings

Variant = Literal["base", "tuned"]

GENERATION_KWARGS: Final[dict[str, Any]] = {
    "max_new_tokens": 128,
    "do_sample": False,
}


@lru_cache(maxsize=2)
def _load(variant: Variant, settings: Settings | None = None) -> Any:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = settings or get_settings()
    name = resolved.base_model if variant == "base" else str(resolved.require_adapter())
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, device_map="auto")
    model.eval()
    return model, tokenizer


class HuggingFaceGenerator:
    """Adapter around a causal LM, for either variant."""

    def __init__(self, variant: Variant, settings: Settings | None = None) -> None:
        """Load the requested variant."""
        self.variant = variant
        self._settings = settings or get_settings()
        self._model, self._tokenizer = _load(variant, self._settings)

    def generate(self, prompt: str) -> str:
        """Return the completion only, with the prompt stripped."""
        inputs = self._tokenizer([prompt], return_tensors="pt").to(self._model.device)
        output = self._model.generate(**inputs, **GENERATION_KWARGS)
        decoded = self._tokenizer.decode(output[0], skip_special_tokens=True)
        return strip_prompt(decoded)


class ScriptedGenerator:
    """Deterministic generator for tests."""

    def __init__(
        self,
        replies: Iterable[str] | None = None,
        *,
        responder: Callable[[str], str] | None = None,
    ) -> None:
        """Configure a fixed reply sequence or a prompt-driven responder."""
        self._replies = list(replies or [])
        self._responder = responder
        self._index = 0
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        """Record ``prompt`` and return the next scripted reply."""
        self.prompts.append(prompt)
        if self._responder is not None:
            return self._responder(prompt)
        if not self._replies:
            return ""
        reply = self._replies[min(self._index, len(self._replies) - 1)]
        self._index += 1
        return reply
