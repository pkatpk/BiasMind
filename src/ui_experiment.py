# src/ui_experiment.py
import os
import json
import sys
import shlex
import subprocess
from pathlib import Path

import gradio as gr

PERSONAS_DIR = Path("data/personas")
TESTS_DIR = Path("data/tests")


def _list_persona_ids():
    if not PERSONAS_DIR.exists():
        return []
    return sorted([p.stem for p in PERSONAS_DIR.glob("*.json")])


def _list_test_files():
    if not TESTS_DIR.exists():
        return []
    return sorted([str(p) for p in TESTS_DIR.glob("*.json")])


def _load_persona_prompt(persona_id: str) -> str:
    if not persona_id:
        return ""
    p = PERSONAS_DIR / f"{persona_id}.json"
    if not p.exists():
        return "(file not found)"
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj.get("prompt_prefix", "")
    except Exception as e:
        return f"(error reading json: {e})"


def _rows_from_selection(selected, existing_rows):
    """
    rows: [persona_id, runs, memory_within]
    """
    selected = selected or []
    existing_rows = existing_rows or []

    by_id = {}
    for row in existing_rows:
        if not row or len(row) < 3:
            continue
        pid = str(row[0]).strip()
        if pid:
            by_id[pid] = [pid, str(row[1]), str(row[2])]

    rows = []
    for pid in selected:
        if pid in by_id:
            rows.append(by_id[pid])
        else:
            rows.append([pid, "1", "fresh"])
    return rows


def _build_cmd(test_file, model_ids_csv, memory_between, persona_rows):
    model_ids = [m.strip() for m in (model_ids_csv or "").split(",") if m.strip()]
    if not model_ids:
        raise ValueError("Δώσε τουλάχιστον ένα model id (comma-separated), π.χ. tinyllama-chat")

    rows = persona_rows or []
    if not rows:
        raise ValueError("Επίλεξε τουλάχιστον μία persona.")

    argv = [sys.executable, "src/run_experiment.py", "--test-file", test_file]

    for mid in model_ids:
        argv += ["--model", mid]

    for row in rows:
        if len(row) < 3:
            continue
        pid = str(row[0]).strip()
        runs = str(row[1]).strip()
        mem = str(row[2]).strip()

        if not pid:
            continue

        try:
            int(runs)
        except Exception:
            raise ValueError(f"runs για '{pid}' πρέπει να είναι ακέραιος (π.χ. 1,2,3...).")

        if mem not in ("fresh", "continuous"):
            raise ValueError(f"memory_within για '{pid}' πρέπει να είναι fresh ή continuous.")

        argv += ["--persona", f"{pid}:{runs}:{mem}"]

    if memory_between not in ("reset", "carry_over"):
        raise ValueError("memory-between πρέπει να είναι reset ή carry_over.")

    argv += ["--memory-between", memory_between]

    pretty = " ".join(shlex.quote(a) for a in argv)
    return argv, pretty


def _run_experiment(test_file, model_ids_csv, memory_between, persona_rows):
    argv, pretty = _build_cmd(test_file, model_ids_csv, memory_between, persona_rows)

    proc = subprocess.run(argv, capture_output=True, text=True)
    out = [f"$ {pretty}\n\n"]

    if proc.stdout:
        out.append("---- STDOUT ----\n")
        out.append(proc.stdout)

    if proc.stderr:
        out.append("\n---- STDERR ----\n")
        out.append(proc.stderr)

    out.append(f"\n\n(exit code: {proc.returncode})")
    return pretty, "".join(out)


def build_experiment_ui():
    persona_ids = _list_persona_ids()
    test_files = _list_test_files()

    with gr.Blocks() as experiment_ui:
        gr.Markdown("## Experiment Runner")

        with gr.Row():
            test_file = gr.Dropdown(
                choices=test_files,
                value=test_files[0] if test_files else None,
                allow_custom_value=True,
                label="Test file (data/tests/*.json)",
            )
            model_ids_csv = gr.Textbox(
                value="tinyllama-chat",
                label="Models (comma-separated)",
                placeholder="π.χ. tinyllama-chat, llama3 ...",
            )
            memory_between = gr.Dropdown(
                choices=["reset", "carry_over"],
                value="reset",
                label="memory-between personas",
            )

        gr.Markdown("### Personas")

        with gr.Row():
            persona_select = gr.CheckboxGroup(
                choices=persona_ids,
                label="Select personas (multi)",
            )
            persona_preview = gr.Textbox(
                label="Prompt prefix preview (last selected)",
                lines=8,
                interactive=False,
            )

        persona_cfg = gr.Dataframe(
            headers=["persona_id", "runs", "memory_within (fresh|continuous)"],
            datatype=["str", "str", "str"],
            row_count=(0, "dynamic"),
            col_count=(3, "fixed"),
            label="Per-persona config",
            interactive=True,
        )

        with gr.Row():
            btn_run = gr.Button("Run experiment", variant="primary")
            cmd_preview = gr.Textbox(label="Command preview", interactive=False)

        output = gr.Textbox(label="Output", lines=18, interactive=False)

        # events
        persona_select.change(
            fn=_rows_from_selection,
            inputs=[persona_select, persona_cfg],
            outputs=[persona_cfg],
        )

        def _preview(selected):
            if not selected:
                return ""
            return _load_persona_prompt(selected[-1])

        persona_select.change(fn=_preview, inputs=[persona_select], outputs=[persona_preview])

        btn_run.click(
            fn=_run_experiment,
            inputs=[test_file, model_ids_csv, memory_between, persona_cfg],
            outputs=[cmd_preview, output],
        )

    return experiment_ui


if __name__ == "__main__":
    app = build_experiment_ui()
    app.launch(share=True)
