"""Structural interfaces.

A :class:`typing.Protocol`, not a base class. It is what lets the evaluation
path run in tests against a scripted fake, with no GPU, no weights and no
download — and, more importantly, it is what guarantees the base and tuned
variants are driven through **identical** code.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TextGenerator(Protocol):
    """A prompt-in, completion-out model."""

    def generate(self, prompt: str) -> str:
        """Return the model's completion for ``prompt``."""
        ...
