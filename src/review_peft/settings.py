"""Typed, validated, environment-driven configuration.

``seed`` and ``eval_size`` are load-bearing: together they define the held-out
slice. Changing either between the base and tuned runs silently destroys the
control, and the resulting comparison measures a different dataset rather than
a different model.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from review_peft.exceptions import AdapterNotFoundError


class Settings(BaseSettings):
    """Runtime configuration, loaded from the environment or a ``.env`` file."""

    model_config = SettingsConfigDict(
        env_prefix="REVIEW_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    hf_token: SecretStr | None = None
    base_model: str = "unsloth/llama-2-7b-bnb-4bit"
    chat_model: str = "meta-llama/Llama-2-13b-chat-hf"
    output_dir: Path = Path("./outputs")
    max_seq_length: int = Field(default=2048, ge=128, le=32768)

    lora_r: int = Field(default=16, ge=1, le=256)
    lora_alpha: int = Field(default=16, ge=1)
    lora_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)

    learning_rate: float = Field(default=2e-4, gt=0.0)
    max_steps: int = Field(default=60, ge=1)
    batch_size: int = Field(default=2, ge=1)
    grad_accum: int = Field(default=4, ge=1)

    seed: int = 42
    eval_size: int = Field(default=100, ge=1)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @property
    def adapter_dir(self) -> Path:
        """Where the trained LoRA adapter is written."""
        return self.output_dir / "lora"

    def require_adapter(self) -> Path:
        """Return the adapter path, or fail with an actionable message."""
        if not self.adapter_dir.exists():
            message = (
                f"No trained adapter at {self.adapter_dir}. Run "
                "`review-peft finetune` before evaluating the tuned variant."
            )
            raise AdapterNotFoundError(message)
        return self.adapter_dir


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
