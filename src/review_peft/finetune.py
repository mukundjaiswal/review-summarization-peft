"""QLoRA fine-tuning.

QLoRA rather than full fine-tuning for a practical reason: a 4-bit base plus
low-rank adapters fits a 7B model on a single consumer GPU, where full
fine-tuning does not. For a task this narrow, full fine-tuning would cost more
than the task is worth.

Heavy imports are local so this module can be imported, and its dataset
construction tested, without a GPU or the training stack installed.
"""

from __future__ import annotations

from typing import Any

from review_peft.data import Record, load_records, train_eval_split
from review_peft.settings import Settings, get_settings

TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def build_training_texts(
    records: list[Record] | None = None, settings: Settings | None = None
) -> list[str]:
    """Return the training strings, excluding the held-out slice.

    Separated from :func:`train` so the thing most likely to be wrong — whether
    the eval rows leaked into training — is testable without a GPU.
    """
    resolved = settings or get_settings()
    train_records, _ = train_eval_split(records or load_records(), resolved)
    return [r.to_sft_text() for r in train_records]


def train(settings: Settings | None = None, *, max_steps: int | None = None) -> Any:
    """Run the QLoRA fine-tune and write the adapter."""
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer

    resolved = settings or get_settings()
    dataset = Dataset.from_dict({"text": build_training_texts(settings=resolved)})

    tokenizer = AutoTokenizer.from_pretrained(resolved.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        resolved.base_model,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        device_map="auto",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=resolved.max_seq_length,
        peft_config=LoraConfig(
            r=resolved.lora_r,
            lora_alpha=resolved.lora_alpha,
            lora_dropout=resolved.lora_dropout,
            target_modules=list(TARGET_MODULES),
            bias="none",
            task_type="CAUSAL_LM",
        ),
        args=TrainingArguments(
            output_dir=str(resolved.output_dir),
            per_device_train_batch_size=resolved.batch_size,
            gradient_accumulation_steps=resolved.grad_accum,
            warmup_steps=5,
            max_steps=max_steps or resolved.max_steps,
            learning_rate=resolved.learning_rate,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=resolved.seed,
        ),
    )

    trainer.train()
    trainer.model.save_pretrained(str(resolved.adapter_dir))
    tokenizer.save_pretrained(str(resolved.adapter_dir))
    return trainer
