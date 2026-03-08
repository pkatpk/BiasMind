import os
import re
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEBUG_LLM = os.getenv("BIASMIND_DEBUG_LLM", "0") == "1"


class HFLLMClient:

    def __init__(self, model_name: str):

        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

        self.model.eval()

    # -------------------------------------------------------------
    # PROMPT BUILDER
    # -------------------------------------------------------------

    def build_prompt(self, system_prompt: str, history: List[Dict], question: str):

        parts = []

        if system_prompt:
            parts.append(system_prompt.strip())
            parts.append("")

        if history:
            parts.append("Previous answers:")

            for turn in history:
                q = turn["question"]
                a = turn["answer"]

                parts.append(f"Q: {q}")
                parts.append(f"A: {a}")

            parts.append("")

        parts.append("Statement:")
        parts.append(question)
        parts.append("")
        parts.append("Answer with ONE number (1-9):")

        return "\n".join(parts)

    # -------------------------------------------------------------
    # GENERATION
    # -------------------------------------------------------------

    def generate(self, system_prompt: str, history: List[Dict], question: str):

        prompt = self.build_prompt(system_prompt, history, question)

        if DEBUG_LLM:
            print("\n")
            print("=" * 100)
            print("PROMPT SENT TO MODEL")
            print("=" * 100)
            print(prompt)
            print("\n")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id
            )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        generated = text[len(prompt):].strip()

        if DEBUG_LLM:
            print("\n")
            print("=" * 100)
            print("RAW MODEL OUTPUT")
            print("=" * 100)
            print(generated)
            print("\n")

        parsed = self.parse_answer(generated)

        if DEBUG_LLM:
            print("\n")
            print("=" * 100)
            print("PARSED ANSWER TEXT")
            print("=" * 100)
            print(parsed)
            print("=" * 100)
            print("\n")

        return parsed

    # -------------------------------------------------------------
    # PARSER
    # -------------------------------------------------------------

    def parse_answer(self, text: str):

        if not text:
            return ""

        match = re.search(r"\b([1-9])\b", text)

        if match:
            return match.group(1)

        return text.strip()


# -------------------------------------------------------------
# ROUTER COMPATIBILITY FUNCTION
# -------------------------------------------------------------

_client_cache = {}

def call_hf_local_chat(model_name, messages, temperature=0.0):

    if model_name not in _client_cache:
        _client_cache[model_name] = HFLLMClient(model_name)

    client = _client_cache[model_name]

    # messages format from runner
    system_prompt = ""
    history = []
    question = ""

    for m in messages:

        if m["role"] == "system":
            system_prompt = m["content"]

        elif m["role"] == "user":
            question = m["content"]

        elif m["role"] == "assistant":
            history.append({
                "question": question,
                "answer": m["content"]
            })

    return client.generate(system_prompt, history, question)