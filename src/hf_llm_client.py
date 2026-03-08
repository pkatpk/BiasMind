import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


DEBUG_LLM = os.getenv("BIASMIND_DEBUG_LLM", "0") == "1"

_client_cache = {}


def get_client(model_name):

    if model_name not in _client_cache:

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

        model.eval()

        _client_cache[model_name] = (tokenizer, model)

    return _client_cache[model_name]


def build_prompt(messages):

    system_prompt = ""
    history = []
    question = ""

    for m in messages:

        if m["role"] == "system":
            system_prompt = m["content"]

        elif m["role"] == "user":
            question = m["content"]

        elif m["role"] == "assistant":
            history.append(m["content"])

    parts = []

    if system_prompt:
        parts.append(system_prompt)
        parts.append("")

    if history:
        parts.append("Previous answers:")
        for a in history:
            parts.append(a)
        parts.append("")

    parts.append("Statement:")
    parts.append(question)
    parts.append("")
    parts.append("Answer with ONE number (1-9):")

    return "\n".join(parts)


def parse_answer(text):

    if not text:
        return ""

    m = re.search(r"\b([1-9])\b", text)

    if m:
        return m.group(1)

    return text.strip()


def call_hf_local_chat(model, messages, temperature=0.0):

    model_name = model.model_name

    tokenizer, hf_model = get_client(model_name)

    prompt = build_prompt(messages)

    if DEBUG_LLM:
        print("\n================ PROMPT =================")
        print(prompt)

    inputs = tokenizer(prompt, return_tensors="pt").to(hf_model.device)

    with torch.no_grad():

        outputs = hf_model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    generated = text[len(prompt):].strip()

    if DEBUG_LLM:
        print("\n================ RAW OUTPUT =============")
        print(generated)

    return parse_answer(generated)