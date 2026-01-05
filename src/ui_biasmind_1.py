# src/ui_biasmind.py
import gradio as gr
from ui_personas import build_personas_ui


def build_main_ui():
    with gr.Blocks() as main_ui:
        gr.Markdown("## Bias Mind")

        go_personas = gr.Button("Manage Personas →")

    return main_ui, go_personas


def build_app():
    main_ui, go_personas = build_main_ui()
    personas_ui = build_personas_ui()

    with gr.Blocks() as app:
        with gr.Column(visible=True) as page_main:
            main_ui.render()

        with gr.Column(visible=False) as page_personas:
            back_btn = gr.Button("← Back")
            personas_ui.render()

        def show_personas():
            return gr.Column(visible=False), gr.Column(visible=True)

        def show_main():
            return gr.Column(visible=True), gr.Column(visible=False)

        go_personas.click(fn=show_personas, outputs=[page_main, page_personas])
        back_btn.click(fn=show_main, outputs=[page_main, page_personas])

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(share=True)
