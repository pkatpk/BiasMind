# src/ui_experiment.py
import json
import sys
import shlex
import subprocess
from pathlib import Path

import gradio as gr

PERSONAS_DIR = Path("data/personas")
TESTS_DIR = Path("data/tests")
MODELS_DIR = Path("data/models")


# ---------- helpers ----------

def _list_persona_ids():
    if not PERSONAS_DIR.exists():
        return []
    return sorted([p.stem for p in PERSONAS_DIR.glob("*.json")])


def _list_test_files():
    if not TESTS_DIR.exists():
        return []
    return sorted([str(p) for p in TESTS_DIR.glob("*.json")])


def _list_model_ids():
    """
    data/models/<model_id>.json
    """
    if not MODELS_DIR.exists():
        return []
    ids = []
    for p in MODELS_DIR.glob("*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            mid = obj.get("id", p.stem)
            if isinstance(mid, str) and mid.strip():
                ids.append(mid.strip())
        except Exception:
            ids.append(p.stem)
    return sorted(set(ids))


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
    selected = selected or []
    cfg = cfg or {}

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
    return _load_persona_prompt(selected[-1])


def _build_cmd(test_file, model_ids, memory_between, cfg_dict):
    model_ids = [m for m in (model_ids or []) if isinstance(m, str) and m.strip()]
    if not model_ids:
        raise ValueError("Επίλεξε τουλάχιστον ένα model.")

    cfg_dict = cfg_dict or {}
    if not cfg_dict:
        raise ValueError("Επίλεξε τουλάχιστον μία persona.")

    argv = [sys.executable, "src/run_experiment.py", "--test-file", test_file]

    for mid in model_ids:
        argv += ["--model", mid]

    for pid, cfg in cfg_dict.items():
        runs = int(cfg.get("runs", 1))
        mem = cfg.get("memory_within", "fresh")
        if mem not in ("fresh", "continuous"):
            raise ValueError(f"memory_within για '{pid}' πρέπει να είναι fresh ή continuous.")
        argv += ["--persona", f"{pid}:{runs}:{mem}"]

    if memory_between not in ("reset", "carry_over"):
        raise ValueError("memory-between πρέπει να είναι reset ή carry_over.")
    argv += ["--memory-between", memory_between]

    pretty = " ".join(shlex.quote(a) for a in argv)
    return argv, pretty


def _run_experiment(test_file, model_ids, memory_between, cfg_dict):
    argv, pretty = _build_cmd(test_file, model_ids, memory_between, cfg_dict)

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


# ---------- UI ----------

def build_experiment_ui():
    persona_ids = _list_persona_ids()
    test_files = _list_test_files()
    model_ids = _list_model_ids()

    with gr.Blocks() as experiment_ui:
        gr.Markdown("## Experiment Runner")

        # ---- global ----
        with gr.Row():
            test_file = gr.Dropdown(
                choices=test_files,
                value=test_files[0] if test_files else None,
                allow_custom_value=True,
                label="Test file (data/tests/*.json)",
            )

            models = gr.Dropdown(
                choices=model_ids,
                value=[model_ids[0]] if model_ids else [],
                multiselect=True,
                label="Models",
            )

        # ---- personas ----
        gr.Markdown("### Personas")

        cfg_state = gr.State({})

        with gr.Row():
            persona_select = gr.Dropdown(
                choices=persona_ids,
                multiselect=True,
                label="Select personas",
            )
            persona_preview = gr.Textbox(
                label="Prompt prefix preview (last selected)",
                lines=8,
                interactive=False,
            )

        persona_select.change(
            fn=_merge_selection_into_cfg,
            inputs=[persona_select, cfg_state],
            outputs=[cfg_state],
        )

        persona_select.change(
            fn=_preview_from_selected,
            inputs=[persona_select],
            outputs=[persona_preview],
        )

        @gr.render(inputs=[cfg_state])
        def _render_persona_rows(cfg):
            cfg = cfg or {}
            if not cfg:
                gr.Markdown("_No personas selected._")
                return

            gr.Markdown("#### Per-persona settings")

            def _set_runs(pid, val, cfg_in):
                if pid in cfg_in:
                    cfg_in[pid]["runs"] = max(1, int(val))
                return cfg_in

            def _set_mem(pid, val, cfg_in):
                if pid in cfg_in:
                    cfg_in[pid]["memory_within"] = val
                return cfg_in

            for pid in cfg.keys():
                with gr.Row():
                    gr.Markdown(f"**{pid}**")
                    runs = gr.Number(
                        value=int(cfg[pid]["runs"]),
                        precision=0,
                        minimum=1,
                        label="runs",
                    )
                    mem = gr.Dropdown(
                        choices=["fresh", "continuous"],
                        value=cfg[pid]["memory_within"],
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

        # ---- memory between ----
        memory_between = gr.Dropdown(
            choices=["reset", "carry_over"],
            value="reset",
            label="memory-between personas",
        )

        # ---- run ----
        with gr.Row():
            btn_run = gr.Button("Run experiment", variant="primary")
            cmd_preview = gr.Textbox(label="Command preview", interactive=False)

        output = gr.Textbox(label="Output", lines=18, interactive=False)

        btn_run.click(
            fn=_run_experiment,
            inputs=[test_file, models, memory_between, cfg_state],
            outputs=[cmd_preview, output],
        )

    return experiment_ui


if __name__ == "__main__":
    app = build_experiment_ui()
    app.launch(share=True)
