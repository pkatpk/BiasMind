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
    για το δοσμένο Hugging Face model id.
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
    Plain instruction-style prompt.
    Αποφεύγει chat transcript / role tags.
    Δεν αλλοιώνει το item text.
    """
    system_text = ""
    user_text = ""

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            system_text = content
        elif role == "user":
            user_text = content

    prompt = (
        f"{system_text}\n\n"
        f"{user_text}\n\n"
        f"Answer with a single integer only: "
    )
    return prompt


def call_hf_local_chat(
    model: ModelDef,
    messages: List[Dict],
    temperature: float = 0.7,
) -> str:
    """
    Καλεί τοπικό HF μοντέλο.

    Fixes:
    - No fake chat tags → no roleplay
    - Deterministic generation
    - 1-token completion → καθαρό digit
    - return_full_text=False → no slicing artifacts
    """
    prompt = _messages_to_prompt(messages)

    if DEBUG:
        print("\n=== HF DEBUG ===", flush=True)
        print("PROMPT repr:", repr(prompt), flush=True)

    pipe = _get_pipeline(model.api_name)

    outputs = pipe(
        prompt,
        do_sample=False,
        temperature=0.0,
        max_new_tokens=1,
        num_return_sequences=1,
        return_full_text=False,
    )

    reply = (outputs[0].get("generated_text") or "").strip()

    if DEBUG:
        print("REPLY repr:", repr(reply), flush=True)
        print("=== END HF DEBUG ===\n", flush=True)

    return reply
