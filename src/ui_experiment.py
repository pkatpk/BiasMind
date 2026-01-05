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


def _add_persona(selected_pid, cfg, order):
    cfg = cfg or {}
    order = order or []
    pid = (selected_pid or "").strip()

    if not pid:
        return cfg, order, "❌ Διάλεξε persona πρώτα."

    if pid in cfg:
        return cfg, order, f"ℹ️ Η persona **{pid}** είναι ήδη στη λίστα."

    cfg[pid] = {"runs": 1, "memory_within": "fresh"}
    order = order + [pid]
    return cfg, order, f"✅ Προστέθηκε η persona **{pid}**."


def _remove_persona(pid, cfg, order):
    cfg = cfg or {}
    order = order or []
    if pid in cfg:
        del cfg[pid]
    order = [x for x in order if x != pid]
    return cfg, order


def _move_persona(pid, direction, order):
    order = order or []
    if pid not in order:
        return order

    i = order.index(pid)
    j = i + int(direction)
    if j < 0 or j >= len(order):
        return order

    new_order = order[:]
    new_order[i], new_order[j] = new_order[j], new_order[i]
    return new_order


def _memory_between_ui(order):
    n = len(order or [])
    if n < 2:
        return gr.update(value="reset", interactive=False)
    return gr.update(interactive=True)


def _build_cmd(test_file, model_id, memory_between, cfg_dict, order_list):
    if not test_file:
        raise ValueError("Επίλεξε test file.")

    model_id = (model_id or "").strip()
    if not model_id:
        raise ValueError("Επίλεξε model.")

    cfg_dict = cfg_dict or {}
    order_list = order_list or []
    ordered_personas = [pid for pid in order_list if pid in cfg_dict]

    if not ordered_personas:
        raise ValueError("Πρόσθεσε τουλάχιστον μία persona (Add).")

    if len(ordered_personas) < 2:
        memory_between = "reset"

    argv = [sys.executable, "src/run_experiment.py", "--test-file", test_file]

    # single model
    argv += ["--model", model_id]

    for pid in ordered_personas:
        cfg = cfg_dict[pid]
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


def _run_experiment(test_file, model_id, memory_between, cfg_dict, order_list):
    argv, pretty = _build_cmd(test_file, model_id, memory_between, cfg_dict, order_list)

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

        with gr.Row():
            test_file = gr.Dropdown(
                choices=test_files,
                value=None,
                label="Test file (data/tests/*.json)",
            )

            # ✅ single-select model (closes like test dropdown)
            model_id = gr.Dropdown(
                choices=model_ids,
                value=None,
                label="Model",
            )

        gr.Markdown("### Personas")

        cfg_state = gr.State({})
        order_state = gr.State([])

        with gr.Row():
            persona_select = gr.Dropdown(
                choices=persona_ids,
                value=None,
                label="Select persona",
            )
            persona_preview = gr.Textbox(
                label="Prompt prefix preview",
                lines=8,
                interactive=False,
            )

        with gr.Row():
            btn_add_persona = gr.Button("Add persona", variant="primary")
            add_status = gr.Markdown("")

        persona_select.change(
            fn=_load_persona_prompt,
            inputs=[persona_select],
            outputs=[persona_preview],
        )

        btn_add_persona.click(
            fn=_add_persona,
            inputs=[persona_select, cfg_state, order_state],
            outputs=[cfg_state, order_state, add_status],
        )

        @gr.render(inputs=[cfg_state, order_state])
        def _render_persona_rows(cfg, order):
            cfg = cfg or {}
            order = order or []
            ordered = [pid for pid in order if pid in cfg]

            if not ordered:
                gr.Markdown("_No personas added yet._")
                return

            gr.Markdown("#### Per-persona settings (ordered)")

            def _set_runs(pid, val, cfg_in):
                cfg_in = cfg_in or {}
                if pid in cfg_in:
                    try:
                        cfg_in[pid]["runs"] = max(1, int(val))
                    except Exception:
                        cfg_in[pid]["runs"] = 1
                return cfg_in

            def _set_mem(pid, val, cfg_in):
                cfg_in = cfg_in or {}
                if pid in cfg_in:
                    cfg_in[pid]["memory_within"] = val
                return cfg_in

            for pid in ordered:
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

                    with gr.Column(scale=0):
                        btn_up = gr.Button("↑", size="sm")
                        btn_down = gr.Button("↓", size="sm")

                    btn_remove = gr.Button("Remove", variant="secondary")

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

                btn_up.click(
                    fn=lambda s_order, _pid=pid: _move_persona(_pid, -1, s_order),
                    inputs=[order_state],
                    outputs=[order_state],
                )

                btn_down.click(
                    fn=lambda s_order, _pid=pid: _move_persona(_pid, +1, s_order),
                    inputs=[order_state],
                    outputs=[order_state],
                )

                btn_remove.click(
                    fn=lambda s_cfg, s_order, _pid=pid: _remove_persona(_pid, s_cfg, s_order),
                    inputs=[cfg_state, order_state],
                    outputs=[cfg_state, order_state],
                )

        memory_between = gr.Dropdown(
            choices=["reset", "carry_over"],
            value="reset",
            label="memory-between personas",
            interactive=False,
        )

        order_state.change(
            fn=_memory_between_ui,
            inputs=[order_state],
            outputs=[memory_between],
        )

        with gr.Row():
            btn_run = gr.Button("Run experiment", variant="primary")
            cmd_preview = gr.Textbox(label="Command preview", interactive=False)

        output = gr.Textbox(label="Output", lines=18, interactive=False)

        btn_run.click(
            fn=_run_experiment,
            inputs=[test_file, model_id, memory_between, cfg_state, order_state],
            outputs=[cmd_preview, output],
        )

    return experiment_ui


if __name__ == "__main__":
    app = build_experiment_ui()
    app.launch(share=True)
