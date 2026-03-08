# src/ui_experiment.py
import csv
import json
import sys
import shlex
import subprocess
from pathlib import Path

import gradio as gr

PERSONAS_DIR = Path("data/personas")
TESTS_DIR = Path("data/tests")
MODELS_DIR = Path("data/models")
RESULTS_DIR = Path("results")
SCORED_DIR = RESULTS_DIR / "scored"


# ---------- helpers ----------

def _list_persona_ids():
    if not PERSONAS_DIR.exists():
        return []
    return sorted([p.stem for p in PERSONAS_DIR.glob("*.json")])


def _list_test_files():
    if not TESTS_DIR.exists():
        return []
    paths = sorted(TESTS_DIR.glob("*.json"))
    return [(p.name, str(p)) for p in paths]


def _list_model_ids():
    if not MODELS_DIR.exists():
        return []
    ids = set()
    for p in MODELS_DIR.glob("*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            mid = obj.get("id", p.stem)
            if isinstance(mid, str) and mid.strip():
                ids.add(mid.strip())
        except Exception:
            ids.add(p.stem)
    return sorted(ids)


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


def _build_cmd(test_file, model_id, persona_id, runs, memory_within):
    if not test_file:
        raise ValueError("Επίλεξε test file.")

    model_id = (model_id or "").strip()
    if not model_id:
        raise ValueError("Επίλεξε model.")

    persona_id = (persona_id or "").strip()
    if not persona_id:
        raise ValueError("Επίλεξε persona.")

    try:
        runs = max(1, int(runs))
    except Exception:
        raise ValueError("Τα runs πρέπει να είναι ακέραιος >= 1.")

    memory_within = (memory_within or "").strip().lower()
    if memory_within not in ("fresh", "continuous"):
        raise ValueError("memory_within πρέπει να είναι fresh ή continuous.")

    argv = [sys.executable, "src/run_experiment.py", "--test-file", test_file]
    argv += ["--model", model_id]
    argv += ["--persona", f"{persona_id}:{runs}:{memory_within}"]

    pretty = " ".join(shlex.quote(a) for a in argv)
    return argv, pretty


def _preview_command(test_file, model_id, persona_id, runs, memory_within):
    _, pretty = _build_cmd(test_file, model_id, persona_id, runs, memory_within)
    pretty_ml = pretty.replace(" --", "\n  --")
    return f"$ {pretty_ml}"


def _run_experiment(test_file, model_id, persona_id, runs, memory_within):
    argv, _pretty = _build_cmd(test_file, model_id, persona_id, runs, memory_within)

    proc = subprocess.run(argv, capture_output=True, text=True)
    out = []

    if proc.stdout:
        out.append("---- STDOUT ----\n")
        out.append(proc.stdout)

    if proc.stderr:
        out.append("\n---- STDERR ----\n")
        out.append(proc.stderr)

    out.append(f"\n(exit code: {proc.returncode})")
    return "".join(out)


def _format_summary_table(rows):
    columns = ["model", "provider", "persona_id", "test_name", "trait", "n_runs", "mean", "std", "min", "max", "sem"]

    clean_rows = []
    for row in rows:
        clean_rows.append({col: str(row.get(col, "")) for col in columns})

    widths = {}
    for col in columns:
        max_len = len(col)
        for row in clean_rows:
            max_len = max(max_len, len(row[col]))
        widths[col] = max_len + 3

    header = "".join(col.ljust(widths[col]) for col in columns).rstrip()
    data_lines = [
        "".join(row[col].ljust(widths[col]) for col in columns).rstrip()
        for row in clean_rows
    ]

    return "\n".join([header] + data_lines)


def _run_summary(experiment_id: str):
    experiment_id = (experiment_id or "").strip()
    if not experiment_id:
        return "❌ Δώσε experiment id."

    # Τρέχουμε κανονικά το analyze script όπως ζήτησες
    argv = [
        sys.executable,
        "src/analyze_experiment.py",
        "--experiment-id",
        experiment_id,
        "--results-dir",
        "results",
    ]
    proc = subprocess.run(argv, capture_output=True, text=True)

    scored_file = SCORED_DIR / f"scored_{experiment_id}.csv"
    if not scored_file.exists():
        out = []
        if proc.stdout:
            out.append(proc.stdout)
        if proc.stderr:
            out.append("\n---- STDERR ----\n")
            out.append(proc.stderr)
        if not out:
            out.append(f"❌ Δεν βρέθηκε το αρχείο: {scored_file}")
        return "".join(out).strip()

    # Διαβάζουμε το CSV και κρατάμε μόνο τα summary rows
    summary_rows = []
    try:
        with scored_file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # κρατά summary γραμμές, ή fallback σε όλες αν δεν υπάρχει summary_label
                summary_label = str(row.get("summary_label", "") or "").strip()
                if summary_label:
                    summary_rows.append(
                        {
                            "model": row.get("model", ""),
                            "provider": row.get("provider", ""),
                            "persona_id": row.get("persona_id", ""),
                            "test_name": row.get("test_name", ""),
                            "trait": row.get("trait", row.get("score_name", "")),
                            "n_runs": row.get("n_runs", ""),
                            "mean": row.get("mean", ""),
                            "std": row.get("std", ""),
                            "min": row.get("min", ""),
                            "max": row.get("max", ""),
                            "sem": row.get("sem", ""),
                        }
                    )
    except Exception as e:
        return f"❌ Σφάλμα στο διάβασμα του summary CSV:\n{e}"

    # Αν δεν βρέθηκαν explicit summary rows, fallback:
    # προσπαθούμε να διαβάσουμε το analyze_experiment stdout ως έχει
    if not summary_rows:
        stdout = (proc.stdout or "").strip()
        if stdout:
            return stdout
        return f"❌ Δεν βρέθηκαν summary rows στο {scored_file}"

    table = _format_summary_table(summary_rows)

    parts = [
        f"=== SUMMARY for experiment {experiment_id} ===",
        f"Source: {scored_file.as_posix()}",
        "",
        table,
        "=== END SUMMARY ===",
    ]

    if proc.stderr:
        parts.append("\n---- STDERR ----")
        parts.append(proc.stderr.strip())

    return "\n".join(parts).strip()


# ---------- UI ----------

def build_experiment_ui():
    persona_ids = _list_persona_ids()
    test_files = _list_test_files()
    model_ids = _list_model_ids()

    with gr.Blocks(css="""
    .monospace textarea {
        font-family: monospace !important;
    }
    """) as experiment_ui:

        gr.Markdown("## Experiment Runner")

        with gr.Row():
            test_file = gr.Dropdown(
                choices=test_files,
                value=None,
                label="Test file",
            )

            model_id = gr.Dropdown(
                choices=model_ids,
                value=None,
                label="Model",
            )

        gr.Markdown("### Persona")

        with gr.Row():
            persona_id = gr.Dropdown(
                choices=persona_ids,
                value=None,
                label="Persona",
            )

            persona_preview = gr.Textbox(
                label="Prompt prefix preview",
                lines=8,
                interactive=False,
            )

        persona_id.change(
            fn=_load_persona_prompt,
            inputs=[persona_id],
            outputs=[persona_preview],
        )

        with gr.Row():
            runs = gr.Number(
                value=1,
                precision=0,
                minimum=1,
                label="Runs",
            )

            memory_within = gr.Radio(
                choices=["fresh", "continuous"],
                value="fresh",
                label="Memory mode",
            )

        gr.Markdown("### Run")

        with gr.Row():
            btn_preview = gr.Button("Command preview (optional)")
            cmd_preview = gr.Textbox(
                label="CLI command",
                lines=8,
                interactive=False,
                placeholder="Press 'Command preview' to generate the CLI command...",
            )

        with gr.Row():
            btn_run = gr.Button("Run experiment", variant="primary")

        output = gr.Textbox(
            label="Output",
            lines=10,
            interactive=False,
        )

        gr.Markdown("### Summary")

        with gr.Row():
            summary_experiment_id = gr.Textbox(
                label="Experiment ID",
                placeholder="e.g. 20260307T142849",
            )

            btn_summary = gr.Button(
                "Summary",
                variant="primary",
            )

        summary_output = gr.Textbox(
            label="Summary output",
            lines=12,
            interactive=False,
            elem_classes="monospace",
        )

        btn_preview.click(
            fn=_preview_command,
            inputs=[test_file, model_id, persona_id, runs, memory_within],
            outputs=[cmd_preview],
        )

        btn_run.click(
            fn=_run_experiment,
            inputs=[test_file, model_id, persona_id, runs, memory_within],
            outputs=[output],
        )

        btn_summary.click(
            fn=_run_summary,
            inputs=[summary_experiment_id],
            outputs=[summary_output],
        )

    return experiment_ui


if __name__ == "__main__":
    app = build_experiment_ui()
    app.launch(share=True)