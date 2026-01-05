# src/ui_experiment.py
import gradio as gr


def build_experiment_ui():
    with gr.Blocks() as experiment_ui:
        gr.Markdown("## Experiment Runner")

        gr.Markdown(
            "⚠️ Experiment UI – work in progress.\n\n"
            "Εδώ θα μπουν:\n"
            "- επιλογή test\n"
            "- επιλογή model\n"
            "- επιλογή personas (multi)\n"
            "- per-persona runs & memory\n"
            "- Run experiment\n"
        )

    return experiment_ui


if __name__ == "__main__":
    app = build_experiment_ui()
    app.launch(share=True)
