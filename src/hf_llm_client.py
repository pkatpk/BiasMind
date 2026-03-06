from typing import List, Dict, Optional, Tuple
from functools import lru_cache
import os
import re

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers.utils import logging as hf_logging

from input_loader import ModelDef

hf_logging.set_verbosity_error()


@lru_cache(maxsize=4)
def _get_pipeline(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def _extract_scale_from_system(messages: List[Dict]) -> Optional[Tuple[int, int]]:
    sys = ""
    for m in messages:
        if m.get("role") == "system":
            sys = m.get("content", "") or ""
            break

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
    Preserve the FULL conversation history in plain text.
    No extra answer instruction is appended here anymore.
    The instruction now lives only in the initial system prompt.
    """
    system_text = ""
    conversation_parts: List[str] = []

    for msg in messages:
        role = msg.get("role")
        content = (msg.get("content", "") or "").strip()

        if not content:
            continue

        if role == "system":
            system_text = content
        elif role in ("user", "assistant"):
            conversation_parts.append(content)

    body = "\n\n".join(conversation_parts).strip()

    if system_text and body:
        return f"{system_text}\n\n{body}"
    elif system_text:
        return system_text
    else:
        return body


def _parse_first_int_in_range(text: str, mn: int, mx: int) -> Optional[int]:
    for tok in re.findall(r"-?\d+", text):
        try:
            v = int(tok)
        except Exception:
            continue
        if mn <= v <= mx:
            return v
    return None


def call_hf_local_chat(
    model: ModelDef,
    messages: List[Dict],
    temperature: float = 0.7,
) -> str:
    debug = (os.getenv("BIASMIND_DEBUG_LLM") or "").strip().lower() in ("1", "true", "yes", "on")

    prompt = _messages_to_prompt(messages)
    scale = _extract_scale_from_system(messages)

    pipe = _get_pipeline(model.api_name)

    outputs = pipe(
        prompt,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        max_new_tokens=12,
        num_return_sequences=1,
        return_full_text=False,
    )

    gen = (outputs[0].get("generated_text") or "")

    parsed: str = ""

    if scale is not None:
        mn, mx = scale
        v = _parse_first_int_in_range(gen, mn, mx)
        parsed = "" if v is None else str(v)
    else:
        m = re.search(r"-?\d+", gen)
        parsed = "" if not m else m.group(0)

    if debug:
        print("\n" + "=" * 90)
        print("[BiasMind DEBUG] hf_local_chat")
        print(
            f"model.id={getattr(model, 'id', None)} "
            f"api_name={getattr(model, 'api_name', None)} "
            f"provider={getattr(model, 'provider', None)}"
        )
        print(f"extracted_scale={scale[0]}..{scale[1]}" if scale else "extracted_scale=None")
        print("-" * 90)
        print("[PROMPT SENT TO MODEL]")
        print(prompt)
        print("-" * 90)
        print("[RAW MODEL OUTPUT]")
        print(gen)
        print("-" * 90)
        print("[PARSED RETURN]")
        print(repr(parsed))
        print("=" * 90 + "\n")

    return parsed