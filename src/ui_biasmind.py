# src/ui_biasmind.py
import gradio as gr
from ui_personas import build_personas_ui
from ui_experiment import build_experiment_ui


def build_main_ui():
    with gr.Blocks() as main_ui:
        gr.Markdown("## Bias Mind")

        go_personas = gr.Button("Manage Personas →")
        go_experiment = gr.Button("Run Experiment →")
        go_results = gr.Button("Results →")

    return main_ui, go_personas, go_experiment, go_results


def build_app():
    main_ui, go_personas, go_experiment, go_results = build_main_ui()
    personas_ui = build_personas_ui()
    experiment_ui = build_experiment_ui()

    with gr.Blocks() as app:
        with gr.Column(visible=True) as page_main:
            main_ui.render()

        with gr.Column(visible=False) as page_personas:
            back_btn_1 = gr.Button("← Back")
            personas_ui.render()

        with gr.Column(visible=False) as page_experiment:
            back_btn_2 = gr.Button("← Back")
            experiment_ui.render()

        with gr.Column(visible=False) as page_results:
            back_btn_3 = gr.Button("← Back")
            gr.Markdown("## Results")
            gr.Markdown("Results UI will be implemented later.")

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
