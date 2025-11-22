import os
from typing import List, Dict

from openai import OpenAI

from input_loader import ModelDef


def get_openai_client() -> OpenAI:
    """
    Δημιουργεί OpenAI client χρησιμοποιώντας το OPENAI_API_KEY
    από τα environment variables.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Set it in your environment or Colab before running."
        )

    # Ο client διαβάζει το api_key από το περιβάλλον,
    # αλλά το περνάμε ρητά για να είμαστε σίγουροι.
    client = OpenAI(api_key=api_key)
    return client


def call_openai_chat(
    model: ModelDef,
    messages: List[Dict],
    temperature: float = 0.7,
) -> str:
    """
    Καλεί το OpenAI chat API για ένα δοσμένο μοντέλο (ModelDef).

    - model.api_name: π.χ. "gpt-4o" ή "gpt-4o-mini"
    - messages: λίστα από dicts τύπου
      {"role": "user" | "system" | "assistant", "content": "..."}
    Επιστρέφει το text της απάντησης.
    """
    client = get_openai_client()

    response = client.chat.completions.create(
        model=model.api_name,
        messages=messages,
        temperature=temperature,
    )

    return response.choices[0].message.content
