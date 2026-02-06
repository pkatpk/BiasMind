# hf_llm_client.py
from typing import List, Dict
from functools import lru_cache
import os

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from input_loader import ModelDef

DEBUG = os.getenv("BIASMIND_DEBUG") == "1"


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
    )
    return pipe


def _messages_to_prompt(messages: List[Dict]) -> str:
    """
    Minimal, NON-chat prompt builder (avoids [USER]/[ASSISTANT] artifacts).
    Keeps item text untouched.
    """
    system_parts = []
    user_parts = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            user_parts.append(content)
        # ignore assistant history for now (local HF is single-turn here)

    system_text = "\n".join(system_parts).strip()
    user_text = "\n".join(user_parts).strip()

    # Plain instruction + item text + explicit answer slot
    prompt = (
        f"{system_text}\n\n"
        f"Statement:\n{user_text}\n\n"
        f"Answer (single integer only): "
    )
    return prompt


def call_hf_local_chat(
    model: ModelDef,
    messages: List[Dict],
    temperature: float = 0.7,
) -> str:
    """
    Καλεί ένα τοπικό Hugging Face μοντέλο χρησιμοποιώντας transformers pipeline.

    Fixes:
    - Avoids fake chat transcript tags that trigger roleplay.
    - Uses return_full_text=False so we get only the completion (no slicing).
    - Uses short generation by default to reduce rambling.
    """
    prompt = _messages_to_prompt(messages)

    if DEBUG:
        print("\n=== HF DEBUG ===", flush=True)
        print("PROMPT repr:", repr(prompt), flush=True)

    pipe = _get_pipeline(model.api_name)

    # Keep it short; you can tweak max_new_tokens if needed.
    # If you want fully deterministic outputs, set do_sample=False and temperature=0.0.
    outputs = pipe(
        prompt,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=8,
        num_return_sequences=1,
        return_full_text=False,
    )

    reply = (outputs[0].get("generated_text") or "").strip()

    if DEBUG:
        print("REPLY repr:", repr(reply), flush=True)
        print("=== END HF DEBUG ===\n", flush=True)

    return reply
