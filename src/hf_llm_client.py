# hf_llm_client.py
from typing import List, Dict, Optional, Tuple
from functools import lru_cache
import os
import re

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from input_loader import ModelDef

DEBUG = os.getenv("BIASMIND_DEBUG") == "1"
SHOW_PROMPT = os.getenv("BIASMIND_SHOW_PROMPT") == "1"  # αν το θες πάντα on: βάλε "1"


@lru_cache(maxsize=4)
def _get_pipeline(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def _extract_scale_from_system(messages: List[Dict]) -> Optional[Tuple[int, int]]:
    """
    Βγάζει (min,max) από το system prompt, π.χ.
    "Always answer ONLY with a single integer number from 1 to 5."
    """
    sys = ""
    for m in messages:
        if m.get("role") == "system":
            sys = m.get("content", "") or ""
            break

    # παίρνουμε το LAST match για να μην μας μπερδέψουν άλλα νούμερα στο persona text
    matches = re.findall(r"from\s+(-?\d+)\s+to\s+(-?\d+)", sys, flags=re.IGNORECASE)
    if not matches:
        return None
    a, b = matches[-1]
    try:
        mn, mx = int(a), int(b)
        if mn > mx:
            mn, mx = mx, mn
        return mn, mx
    except Exception:
        return None


def _messages_to_prompt(messages: List[Dict]) -> str:
    """
    Plain prompt builder (no fake chat tags).
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

    prompt = f"{system_text}\n\n{user_text}\n\nOutput: "
    return prompt


def call_hf_local_chat(
    model: ModelDef,
    messages: List[Dict],
    temperature: float = 0.7,
) -> str:
    prompt = _messages_to_prompt(messages)
    scale = _extract_scale_from_system(messages)  # (min,max) ή None

    if DEBUG or SHOW_PROMPT:
        print("\n=== HF PROMPT ===", flush=True)
        print(prompt, flush=True)            # human-readable
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

    gen = (outputs[0].get("generated_text") or "").lstrip()
    reply = (gen[:1] if gen else "").strip()  # παίρνουμε 1 char

    # dynamic validation με βάση min/max από system prompt
    if scale is not None:
        mn, mx = scale
        try:
            v = int(reply)
            if not (mn <= v <= mx):
                reply = ""  # invalid → empty (ή βάλε fallback αν θες)
        except Exception:
            reply = ""

    if DEBUG:
        print("GEN repr:", repr(gen), flush=True)
        print("REPLY repr:", repr(reply), flush=True)

    return reply
