from typing import List, Dict

from input_loader import ModelDef
from hf_llm_client import call_hf_local_chat

# Προαιρετικό: αν υπάρχει OpenAI client, τον φορτώνουμε, αλλιώς αφήνουμε placeholder.
try:
    from llm_client import call_openai_chat  # για μελλοντική χρήση με OpenAI API
except ImportError:
    call_openai_chat = None


def call_model(
    model: ModelDef,
    messages: List[Dict],
    temperature: float = 0.7,
) -> str:
    """
    Ενιαίο entry point για ΟΛΑ τα μοντέλα.

    - Διαβάζει το model.provider
    - Καλεί τον κατάλληλο client (HF local, OpenAI, κλπ.)

    ΤΩΡΑ:
      - Υποστηρίζουμε μόνο provider == "huggingface_local"
      - Έχουμε placeholders για "openai" και "anthropic"
    """

    if model.provider == "huggingface_local":
        # Τοπικά HuggingFace models (TinyLlama, Phi-3, κλπ.)
        return call_hf_local_chat(model, messages, temperature=temperature)

    if model.provider == "openai":
        # Placeholder για μελλοντική χρήση
        if call_openai_chat is None:
            raise RuntimeError(
                "OpenAI provider ζητήθηκε αλλά δεν είναι διαθέσιμος "
                "(λείπει ο llm_client ή δεν θέλουμε ακόμα OpenAI API)."
            )
        return call_openai_chat(model, messages, temperature=temperature)

    if model.provider == "anthropic":
        # Placeholder για μελλοντική χρήση (Claude)
        raise NotImplementedError(
            "Anthropic/Claude provider δεν έχει υλοποιηθεί ακόμα."
        )

    raise ValueError(f"Άγνωστος provider: {model.provider}")
