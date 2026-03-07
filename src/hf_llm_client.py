# hf_llm_client.py
from __future__ import annotations

import os
import re
from typing import List, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEBUG_LLM = (os.getenv("BIASMIND_DEBUG_LLM") or "").strip().lower() in ("1", "true", "yes", "on")

# Απλό cache για να μη φορτώνεται το model από την αρχή σε κάθε κλήση
_CLIENT_CACHE: Dict[Tuple[str, Optional[str], int], "HuggingFaceLLMClient"] = {}


class HuggingFaceLLMClient:
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        max_new_tokens: int = 16,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(self.device)

        self.model.eval()

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Μετατρέπει chat-style messages σε plain instruction prompt.

        Υποστηρίζει:
        - system message(s)
        - πολλαπλά user / assistant turns
        - σωστό history injection για continuous mode

        Format:

        [system]

        User: ...
        Assistant: ...
        User: ...
        Assistant:
        """
        system_parts: List[str] = []
        convo_parts: List[str] = []

        for msg in messages:
            role = (msg.get("role") or "").strip().lower()
            content = (msg.get("content") or "").strip()

            if not content:
                continue

            if role == "system":
                system_parts.append(content)
            elif role == "user":
                convo_parts.append(f"User: {content}")
            elif role == "assistant":
                convo_parts.append(f"Assistant: {content}")
            else:
                convo_parts.append(content)

        system_text = "\n\n".join(system_parts).strip()
        convo_text = "\n".join(convo_parts).strip()

        if system_text and convo_text:
            prompt = f"{system_text}\n\n{convo_text}\nAssistant:"
        elif system_text:
            prompt = f"{system_text}\n\nAssistant:"
        elif convo_text:
            prompt = f"{convo_text}\nAssistant:"
        else:
            prompt = "Assistant:"

        return prompt

    def _extract_answer_text(self, generated_text: str, prompt_text: str) -> str:
        """
        Αφαιρεί το prompt prefix από το generated text και επιστρέφει μόνο το νέο generated τμήμα.
        """
        if generated_text.startswith(prompt_text):
            return generated_text[len(prompt_text):].strip()

        return generated_text.strip()

    def _coerce_single_integer(self, text: str) -> str:
        """
        Κρατά το τελευταίο integer από την απάντηση.
        Αν δεν βρει τίποτα, επιστρέφει το stripped raw text.
        """
        text = (text or "").strip()
        nums = re.findall(r"-?\d+", text)
        if nums:
            return nums[-1]
        return text

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
    ) -> str:
        prompt = self._messages_to_prompt(messages)

        if DEBUG_LLM:
            print("\n" + "=" * 100)
            print("PROMPT SENT TO MODEL")
            print("=" * 100)
            print(prompt)
            print("=" * 100 + "\n")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        do_sample = temperature > 0

        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if do_sample:
            generate_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **generate_kwargs,
            )

        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        answer_text = self._extract_answer_text(generated_text, prompt)
        answer_text = self._coerce_single_integer(answer_text)

        if DEBUG_LLM:
            print("\n" + "=" * 100)
            print("RAW MODEL OUTPUT")
            print("=" * 100)
            print(generated_text)
            print("=" * 100)
            print("PARSED ANSWER TEXT")
            print("=" * 100)
            print(answer_text)
            print("=" * 100 + "\n")

        return answer_text


def _get_cached_client(
    model_name: str,
    device: Optional[str] = None,
    max_new_tokens: int = 16,
) -> HuggingFaceLLMClient:
    key = (model_name, device, max_new_tokens)
    client = _CLIENT_CACHE.get(key)

    if client is None:
        client = HuggingFaceLLMClient(
            model_name=model_name,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        _CLIENT_CACHE[key] = client

    return client


def call_hf_local_chat(
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    device: Optional[str] = None,
    max_new_tokens: int = 16,
) -> str:
    """
    Compatibility wrapper για το llm_router.py.

    Χρήση:
        reply = call_hf_local_chat(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            messages=[...],
            temperature=0.2,
        )
    """
    client = _get_cached_client(
        model_name=model_name,
        device=device,
        max_new_tokens=max_new_tokens,
    )
    return client.generate(messages=messages, temperature=temperature)