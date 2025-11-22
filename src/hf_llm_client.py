from typing import List, Dict
from functools import lru_cache

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from input_loader import ModelDef


@lru_cache(maxsize=4)
def _get_pipeline(model_id: str):
    """
    Φορτώνει και κάνει cache ένα text-generation pipeline
    για το δοσμένο Hugging Face model id (π.χ. TinyLlama, Phi-3-mini).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=64,
    )
    return pipe


def _messages_to_prompt(messages: List[Dict]) -> str:
    """
    Πολύ απλή μετατροπή chat-messages -> plain prompt.
    Δεν είναι τέλειο, αλλά αρκεί για το BiasMind prototype.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"[SYSTEM] {content}\n")
        elif role == "assistant":
            parts.append(f"[ASSISTANT] {content}\n")
        else:
            parts.append(f"[USER] {content}\n")
    parts.append("[ASSISTANT] ")
    return "".join(parts)


def call_hf_local_chat(
    model: ModelDef,
    messages: List[Dict],
    temperature: float = 0.7,
) -> str:
    """
    Καλεί ένα τοπικό Hugging Face μοντέλο (π.χ. TinyLlama, Phi-3-mini)
    χρησιμοποιώντας transformers.

    - model.api_name: HF model id, π.χ.
        - "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        - "microsoft/Phi-3-mini-4k-instruct"
    """
    prompt = _messages_to_prompt(messages)

    pipe = _get_pipeline(model.api_name)

    outputs = pipe(
        prompt,
        do_sample=True,
        temperature=temperature,
        num_return_sequences=1,
    )

    full_text = outputs[0]["generated_text"]
    reply = full_text[len(prompt):].strip()
    return reply
