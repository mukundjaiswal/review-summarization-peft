"""Customer review classification and summarization, with QLoRA fine-tuning.

Two questions answered in order:

1. Can prompting alone do the job? Few-shot classification with an
   instruction-tuned chat model, scored with Micro-F1.
2. Does fine-tuning beat it, and by how much? QLoRA on a 4-bit base model,
   with the **same base model scored before and after** on the **same** held-out
   slice.

The second question is the reason the repository exists. A fine-tuned model
that scores well proves nothing on its own; the claim "fine-tuning helped"
requires the untuned model measured on identical data with an identical scorer.

Registration of this project's metrics is an explicit call, not an import side
effect: see :func:`review_peft.eval_metrics.register_task_metrics`.
"""

from review_peft.eval_metrics import register_task_metrics
from review_peft.settings import Settings, get_settings

__all__ = ["Settings", "__version__", "get_settings", "register_task_metrics"]
__version__ = "0.1.0"
