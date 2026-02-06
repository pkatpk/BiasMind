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
    Cached HF text-generation pipeline.
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
    Plain prompt builder (no fake chat tags).
    Keeps item text untouched.
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

    # Avoid "Statement:" label that the model may echo
    prompt = (
        f"{system_text}\n\n"
        f"{user_text}\n\n"
        f"Output: "
    )
    return prompt


def call_hf_local_chat(
    model: ModelDef,
    messages: List[Dict],
    temperature: float = 0.7,
) -> str:
    """
    Local HF call.

    - No fake chat tags → prevents roleplay artifacts
    - Sampling enabled → avoids collapsing to constant "1"
    - return_full_text=False → no prompt slicing artifacts
    - Keep completion tiny → reduces rambling
    - Post-trim to first char → usually the Likert digit
    """
    prompt = _messages_to_prompt(messages)

    if DEBUG:
        print("\n=== HF DEBUG ===", flush=True)
        print("PROMPT repr:", repr(prompt), flush=True)

    pipe = _get_pipeline(model.api_name)

    outputs = pipe(
        prompt,
        do_sample=True,
        temperature=temperature,  # controlled by caller (experiment_runner uses 0.2)
        top_p=0.9,
        max_new_tokens=2,
        num_return_sequences=1,
        return_full_text=False,
    )

    gen = (outputs[0].get("generated_text") or "").lstrip()
    reply = gen[:1].strip()  # keep only first char (expected to be 1-5)

    if DEBUG:
        print("GEN repr:", repr(gen), flush=True)
        print("REPLY repr:", repr(reply), flush=True)
        print("=== END HF DEBUG ===\n", flush=True)

    return reply
