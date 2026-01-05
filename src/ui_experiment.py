# src/ui_experiment.py
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


def _merge_selection_into_cfg(selected, cfg):
    """
    cfg: dict[persona_id] = {"runs": int, "memory_within": "fresh|continuous"}
    """
    selected = selected or []
    cfg = cfg or {}

    # keep only selected
    new_cfg = {}
    for pid in selected:
        if pid in cfg:
            new_cfg[pid] = cfg[pid]
        else:
            new_cfg[pid] = {"runs": 1, "memory_within": "fresh"}

    return new_cfg


def _preview_from_selected(selected):
    if not selected:
        return ""
    # last selected for preview
    pid = selected[-1]
    return _load_persona_prompt(pid)


def _build_cmd(test_file, model_ids_csv, memory_between, cfg_dict):
    model_ids = [m.strip() for m in (model_ids_csv or "").split(",") if m.strip()]
    if not model_ids:
        raise ValueError("Δώσε τουλάχιστον ένα model id (comma-separated), π.χ. tinyllama-chat")

    cfg_dict = cfg_dict or {}
    if not cfg_dict:
        raise ValueError("Επίλεξε τουλάχιστον μία persona.")

    argv = [sys.executable, "src/run_experiment.py", "--test-file", test_file]

    for mid in model_ids:
        argv += ["--model", mid]

    for pid, cfg in cfg_dict.items():
        runs = int(cfg.get("runs", 1))
        mem = str(cfg.get("memory_within", "fresh")).strip()
        if mem not in ("fresh", "continuous"):
            raise ValueError(f"memory_within για '{pid}' πρέπει να είναι fresh ή continuous.")
        argv += ["--persona", f"{pid}:{runs}:{mem}"]

    if memory_between not in ("reset", "carry_over"):
        raise ValueError("memory-between πρέπει να είναι reset ή carry_over.")
    argv += ["--memory-between", memory_between]

    pretty = " ".join(shlex.quote(a) for a in argv)
    return argv, pretty


def _run_experiment(test_file, model_ids_csv, memory_between, cfg_dict):
    argv, pretty = _build_cmd(test_file, model_ids_csv, memory_between, cfg_dict)

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

        gr.Markdown("### Personas")

        cfg_state = gr.State({})  # dict config per persona

        with gr.Row():
            persona_select = gr.Dropdown(
                choices=persona_ids,
                multiselect=True,
                label="Select personas (multi)",
            )
            persona_preview = gr.Textbox(
                label="Prompt prefix preview (last selected)",
                lines=8,
                interactive=False,
            )

        # update cfg_state on selection
        def _on_select(selected, cfg):
            return _merge_selection_into_cfg(selected, cfg)

        persona_select.change(
            fn=_on_select,
            inputs=[persona_select, cfg_state],
            outputs=[cfg_state],
        )

        # update preview
        persona_select.change(
            fn=_preview_from_selected,
            inputs=[persona_select],
            outputs=[persona_preview],
        )

        # Render per-persona rows
        @gr.render(inputs=[cfg_state])
        def _render_rows(cfg):
            cfg = cfg or {}
            if not cfg:
                gr.Markdown("_No personas selected._")
                return

            gr.Markdown("#### Per-persona settings")

            # Helper: update functions
            def _set_runs(pid, val, cfg_in):
                cfg_in = cfg_in or {}
                if pid in cfg_in:
                    try:
                        cfg_in[pid]["runs"] = int(val)
                    except Exception:
                        cfg_in[pid]["runs"] = 1
                return cfg_in

            def _set_mem(pid, val, cfg_in):
                cfg_in = cfg_in or {}
                if pid in cfg_in:
                    cfg_in[pid]["memory_within"] = val
                return cfg_in

            for pid in cfg.keys():
                with gr.Row():
                    gr.Markdown(f"**{pid}**")
                    runs = gr.Number(
                        value=int(cfg[pid].get("runs", 1)),
                        precision=0,
                        minimum=1,
                        label="runs",
                    )
                    mem = gr.Dropdown(
                        choices=["fresh", "continuous"],
                        value=str(cfg[pid].get("memory_within", "fresh")),
                        label="memory_within",
                    )

                runs.change(
                    fn=lambda v, s, _pid=pid: _set_runs(_pid, v, s),
                    inputs=[runs, cfg_state],
                    outputs=[cfg_state],
                )
                mem.change(
                    fn=lambda v, s, _pid=pid: _set_mem(_pid, v, s),
                    inputs=[mem, cfg_state],
                    outputs=[cfg_state],
                )

        # ✅ memory-between moved here (after personas, before Run)
        memory_between = gr.Dropdown(
            choices=["reset", "carry_over"],
            value="reset",
            label="memory-between personas",
        )

        with gr.Row():
            btn_run = gr.Button("Run experiment", variant="primary")
            cmd_preview = gr.Textbox(label="Command preview", interactive=False)

        output = gr.Textbox(label="Output", lines=18, interactive=False)

        btn_run.click(
            fn=_run_experiment,
            inputs=[test_file, model_ids_csv, memory_between, cfg_state],
            outputs=[cmd_preview, output],
        )

    return experiment_ui


if __name__ == "__main__":
    app = build_experiment_ui()
    app.launch(share=True)
