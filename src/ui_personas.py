# src/ui_personas.py
import os
import json
import re
import gradio as gr

PERSONAS_DIR = "data/personas"


def _load_personas():
    if not os.path.exists(PERSONAS_DIR):
        return []
    personas = []
    for fname in os.listdir(PERSONAS_DIR):
        if fname.endswith(".json"):
            personas.append(fname.replace(".json", ""))
    return sorted(personas)


def _refresh_dropdown():
    choices = _load_personas()
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)


def _normalize_persona_id(name: str) -> str:
    name = (name or "").strip().lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^a-z0-9_\-]", "", name)
    return name


def _persona_path(persona_id: str) -> str:
    return os.path.join(PERSONAS_DIR, f"{persona_id}.json")


def _load_persona_prompt(persona_id: str):
    if not persona_id:
        return ""
    path = _persona_path(persona_id)
    if not os.path.exists(path):
        return "(file not found)"
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj.get("prompt_prefix", "")
    except Exception as e:
        return f"(error reading json: {e})"


def _create_persona(persona_name: str, prompt_prefix: str):
    persona_id = _normalize_persona_id(persona_name)
    prompt_prefix = (prompt_prefix or "").strip()

    if not persona_id:
        return "❌ Δώσε ένα έγκυρο Persona name.", _refresh_dropdown(), ""
    if not prompt_prefix:
        return "❌ Το prompt δεν μπορεί να είναι κενό.", _refresh_dropdown(), ""

    os.makedirs(PERSONAS_DIR, exist_ok=True)
    out_path = _persona_path(persona_id)

    if os.path.exists(out_path):
        return f"❌ Υπάρχει ήδη persona με όνομα '{persona_id}'.", _refresh_dropdown(), _load_persona_prompt(persona_id)

    obj = {"id": persona_id, "prompt_prefix": prompt_prefix}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

    dd = _refresh_dropdown()
    dd.value = persona_id
    return f"✅ Persona δημιουργήθηκε: {out_path}", dd, _load_persona_prompt(persona_id)


def build_personas_ui():
    """Returns a Gradio Blocks UI for persona management (no launch here)."""
    with gr.Blocks() as personas_ui:
        gr.Markdown("## Personas")

        persona_dropdown = gr.Dropdown(
            choices=_load_personas(),
            value=_load_personas()[0] if _load_personas() else None,
            label="Select Persona",
            interactive=True
        )

        selected_prompt = gr.Textbox(label="Prompt prefix", lines=6, interactive=False)

        persona_dropdown.change(
            fn=_load_persona_prompt,
            inputs=persona_dropdown,
            outputs=selected_prompt
        )

        refresh_btn = gr.Button("Refresh personas")
        refresh_btn.click(fn=_refresh_dropdown, outputs=persona_dropdown)

        gr.Markdown("### Create new persona")
        new_name = gr.Textbox(label="Persona name")
        new_prompt = gr.Textbox(label="Prompt prefix", lines=4, placeholder="You are a ... Answer as ...")
        create_btn = gr.Button("Create persona")
        status = gr.Markdown("")

        create_btn.click(
            fn=_create_persona,
            inputs=[new_name, new_prompt],
            outputs=[status, persona_dropdown, selected_prompt]
        )

        personas_ui.load(
            fn=_load_persona_prompt,
            inputs=persona_dropdown,
            outputs=selected_prompt
        )

    return personas_ui


if __name__ == "__main__":
    # Standalone run (optional)
    app = build_personas_ui()
    app.launch(share=True)
