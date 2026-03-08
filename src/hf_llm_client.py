import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


DEBUG_LLM = os.getenv("BIASMIND_DEBUG_LLM") == "1"

_CLIENT_CACHE = {}


def _get_cached_client(model_def):

    key = model_def.id

    if key in _CLIENT_CACHE:
        return _CLIENT_CACHE[key]

    tokenizer = AutoTokenizer.from_pretrained(model_def.api_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_def.api_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    client = (tokenizer, model)

    _CLIENT_CACHE[key] = client

    return client


def call_hf_local_chat(model_def, messages, temperature=0.2):

    tokenizer, model = _get_cached_client(model_def)

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    if DEBUG_LLM:
        print("\n" + "="*100)
        print("PROMPT SENT TO MODEL")
        print("="*100)
        print(prompt)
        print("\n")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=True,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    reply = text[len(prompt):].strip()

    if DEBUG_LLM:
        print("="*100)
        print("RAW MODEL OUTPUT")
        print("="*100)
        print(reply)
        print("\n")

    first_line = reply.split("\n")[0].strip()

    if DEBUG_LLM:
        print("="*100)
        print("PARSED ANSWER TEXT")
        print("="*100)
        print(first_line)
        print("="*100 + "\n")

    return first_line