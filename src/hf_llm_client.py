# hf_llm_client.py
from typing import List, Dict, Optional, Tuple
from functools import lru_cache
import re

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from input_loader import ModelDef


@lru_cache(maxsize=4)
def _get_pipeline(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def _extract_scale_from_system(messages: List[Dict]) -> Optional[Tuple[int, int]]:
    """
    Extracts (min,max) from system prompt like:
    "Always answer ONLY with a single integer number from 1 to 5."
    Dynamic per test (no hardcoded 1–5).
    """
    sys = ""
    for m in messages:
        if m.get("role") == "system":
            sys = m.get("content", "") or ""
            break

    matches = re.findall(r"from\s+(-?\d+)\s+to\s+(-?\d+)", sys, flags=re.IGNORECASE)
    if not matches:
        return None

    a, b = matches[-1]  # last match, in case persona text contains numbers
    try:
        mn, mx = int(a), int(b)
        if mn > mx:
            mn, mx = mx, mn
        return mn, mx
    except Exception:
        return None


def _messages_to_prompt(messages: List[Dict]) -> str:
    """
    Plain prompt builder (NO [USER]/[ASSISTANT] tags).
    Keeps item text exactly as-is.
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

    return f"{system_text}\n\n{user_text}\n\nOutput: "


def call_hf_local_chat(
    model: ModelDef,
    messages: List[Dict],
    temperature: float = 0.7,
) -> str:
    prompt = _messages_to_prompt(messages)
    scale = _extract_scale_from_system(messages)  # (min,max) or None

    # ALWAYS-ON DISPLAYS
    print("\n=== HF PROMPT ===", flush=True)
    print(prompt, flush=True)
    print("PROMPT repr:", repr(prompt), flush=True)
    print("SCALE:", scale, flush=True)
    print("=== END HF PROMPT ===\n", flush=True)

    pipe = _get_pipeline(model.api_name)

    outputs = pipe(
        prompt,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        max_new_tokens=2,
        num_return_sequences=1,
        return_full_text=False,
    )

    gen = (outputs[0].get("generated_text") or "")
    print("GEN repr:", repr(gen), flush=True)

    # take first non-space char as the intended digit
    first = gen.lstrip()[:1].strip()

    # dynamic validation using extracted scale (no hardcoded 1–5)
    if scale is not None:
        mn, mx = scale
        try:
            v = int(first)
            if not (mn <= v <= mx):
                first = ""
        except Exception:
            first = ""

    print("REPLY (final) repr:", repr(first), flush=True)

    return first
