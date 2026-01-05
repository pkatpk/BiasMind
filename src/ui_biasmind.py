# src/ui_biasmind.py
import gradio as gr
from ui_personas import build_personas_ui
from ui_experiment import build_experiment_ui


APP_CSS = """
/* ---------- App background ---------- */
.gradio-container {
  background: radial-gradient(
    1200px 800px at 10% 0%,
    #f3f7ff 0%,
    #ffffff 55%,
    #f7f7fb 100%
  );
}

/* ---------- Card look ---------- */
.bm-card {
  background: rgba(255, 255, 255, 0.88);
  border: 1px solid rgba(20, 20, 20, 0.08);
  border-radius: 16px;
  padding: 20px;
  box-shadow: 0 8px 28px rgba(0, 0, 0, 0.06);
}

/* ---------- Headings ---------- */
.bm-title h1,
.bm-title h2 {
  letter-spacing: -0.02em;
}

/* ---------- Buttons ---------- */
button.primary,
.gr-button-primary {
  border-radius: 12px !important;
}

/* Make secondary buttons a bit softer */
.gr-button {
  border-radius: 10px;
}
"""


def build_main_ui():
    with gr.Blocks() as main_ui:
        gr.Markdown("## Bias Mind", elem_classes=["bm-title"])
        gr.Markdown(
            "Demo UI / experiment runner for persona-based bias testing."
        )

        go_personas = gr.Button("Manage Personas →")
        go_experiment = gr.Button("Run Experiment →")
        go_results = gr.Button("Results →")

    return main_ui, go_personas, go_experiment, go_results


def build_app():
    main_ui, go_personas, go_experiment, go_results = build_main_ui()
    personas_ui = build_personas_ui()
    experiment_ui = build_experiment_ui()

    with gr.Blocks(theme=gr.themes.Soft(), css=APP_CSS) as app:

        # ---------- MAIN ----------
        with gr.Column(visible=True, elem_classes=["bm-card"]) as page_main:
            main_ui.render()

        # ---------- PERSONAS ----------
        with gr.Column(visible=False, elem_classes=["bm-card"]) as page_personas:
            back_btn_1 = gr.Button("← Back")
            personas_ui.render()

        # ---------- EXPERIMENT ----------
        with gr.Column(visible=False, elem_classes=["bm-card"]) as page_experiment:
            back_btn_2 = gr.Button("← Back")
            experiment_ui.render()

        # ---------- RESULTS (placeholder) ----------
        with gr.Column(visible=False, elem_classes=["bm-card"]) as page_results:
            back_btn_3 = gr.Button("← Back")
            gr.Markdown("## Results")
            gr.Markdown("Results UI will be implemented later.")

        # ---------- Navigation logic ----------
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

        go_personas.click(
            fn=show_personas,
            outputs=[page_main, page_personas, page_experiment, page_results],
        )
        go_experiment.click(
            fn=show_experiment,
            outputs=[page_main, page_personas, page_experiment, page_results],
        )
        go_results.click(
            fn=show_results,
            outputs=[page_main, page_personas, page_experiment, page_results],
        )

        back_btn_1.click(
            fn=show_main,
            outputs=[page_main, page_personas, page_experiment, page_results],
        )
        back_btn_2.click(
            fn=show_main,
            outputs=[page_main, page_personas, page_experiment, page_results],
        )
        back_btn_3.click(
            fn=show_main,
            outputs=[page_main, page_personas, page_experiment, page_results],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(share=True)
