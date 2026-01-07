# src/ui_biasmind.py
import gradio as gr
from ui_personas import build_personas_ui
from ui_experiment import build_experiment_ui
from ui_results import build_results_ui


APP_CSS = r"""
/* =========================
   Cognitive / Bias Theme
   ========================= */

.gradio-container {
  background:
    radial-gradient(800px 500px at 15% 10%, rgba(99,102,241,0.10), transparent 40%),
    radial-gradient(900px 600px at 85% 20%, rgba(139,92,246,0.10), transparent 45%),
    radial-gradient(700px 500px at 50% 90%, rgba(79,70,229,0.08), transparent 50%),
    #f7f7fb;
}

.bm-card {
  background: rgba(255, 255, 255, 0.88);
  backdrop-filter: blur(6px);
  border: 1px solid rgba(99, 102, 241, 0.18);
  border-radius: 18px;
  padding: 22px;
  box-shadow:
    0 10px 30px rgba(0, 0, 0, 0.06),
    0 1px 0 rgba(255,255,255,0.6) inset;
}

/* ✅ FIX: dropdown popups clipped when Results is embedded */
.bm-card,
.gradio-container .block,
.gradio-container .form,
.gradio-container .gr-box,
.gradio-container .gr-panel,
.gradio-container .wrap,
.gradio-container .prose {
  overflow: visible !important;
}

/* Ensure dropdown menu appears above other elements */
.gradio-container [role="listbox"],
.gradio-container .dropdown,
.gradio-container .dropdown-menu {
  z-index: 5000 !important;
}

.bm-title h1,
.bm-title h2 {
  letter-spacing: -0.025em;
  color: #1f2937;
}

.bm-title h2::after {
  content: "";
  display: block;
  width: 52px;
  height: 3px;
  margin-top: 6px;
  border-radius: 2px;
  background: linear-gradient(90deg, #6366f1, #8b5cf6);
}

.gr-button-primary,
button.primary {
  background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
  border: none !important;
  color: white !important;
  border-radius: 14px !important;
  box-shadow: 0 6px 16px rgba(99,102,241,0.35);
}

.gr-button-primary:hover {
  filter: brightness(1.05);
}

.gr-button {
  border-radius: 12px;
}

label span[aria-hidden="true"] {
  display: none;
}
"""


def build_main_ui():
    # IMPORTANT: return the actual button objects (no elem_id tricks)
    with gr.Blocks() as main_ui:
        gr.Markdown("## Bias Mind", elem_classes=["bm-title"])
        gr.Markdown("Persona-based experiment runner for studying cognitive bias in language models.")

        go_personas = gr.Button("Manage Personas →")
        go_experiment = gr.Button("Run Experiment →")
        go_results = gr.Button("Results →")

    return main_ui, go_personas, go_experiment, go_results


def build_app():
    main_ui, go_personas, go_experiment, go_results = build_main_ui()
    personas_ui = build_personas_ui()
    experiment_ui = build_experiment_ui()
    results_ui = build_results_ui()

    with gr.Blocks(theme=gr.themes.Soft(), css=APP_CSS) as app:
        # MAIN
        with gr.Column(visible=True, elem_classes=["bm-card"]) as page_main:
            main_ui.render()

        # PERSONAS
        with gr.Column(visible=False, elem_classes=["bm-card"]) as page_personas:
            back_btn_1 = gr.Button("← Back")
            personas_ui.render()

        # EXPERIMENT
        with gr.Column(visible=False, elem_classes=["bm-card"]) as page_experiment:
            back_btn_2 = gr.Button("← Back")
            experiment_ui.render()

        # RESULTS
        with gr.Column(visible=False, elem_classes=["bm-card"]) as page_results:
            back_btn_3 = gr.Button("← Back")
            results_ui.render()

        # ---------- Navigation ----------
        def show_personas():
            return (
                gr.Column(visible=False),
                gr.Column(visible=True),
                gr.Column(visible=False),
                gr.Column(visible=False),
            )

        def show_experiment():
            return (
                gr.Column(visible=False),
                gr.Column(visible=False),
                gr.Column(visible=True),
                gr.Column(visible=False),
            )

        def show_results():
            return (
                gr.Column(visible=False),
                gr.Column(visible=False),
                gr.Column(visible=False),
                gr.Column(visible=True),
            )

        def show_main():
            return (
                gr.Column(visible=True),
                gr.Column(visible=False),
                gr.Column(visible=False),
                gr.Column(visible=False),
            )

        go_personas.click(fn=show_personas, outputs=[page_main, page_personas, page_experiment, page_results])
        go_experiment.click(fn=show_experiment, outputs=[page_main, page_personas, page_experiment, page_results])
        go_results.click(fn=show_results, outputs=[page_main, page_personas, page_experiment, page_results])

        back_btn_1.click(fn=show_main, outputs=[page_main, page_personas, page_experiment, page_results])
        back_btn_2.click(fn=show_main, outputs=[page_main, page_personas, page_experiment, page_results])
        back_btn_3.click(fn=show_main, outputs=[page_main, page_personas, page_experiment, page_results])

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(share=True)
