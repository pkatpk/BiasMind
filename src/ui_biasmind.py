# src/ui_biasmind.py
import gradio as gr
from ui_personas import build_personas_ui
from ui_experiment import build_experiment_ui


APP_CSS = r"""
/* =========================
   Cognitive / Bias Theme
   + Abstract brain/network background (SVG)
   ========================= */

/* ---- Global background ---- */
.gradio-container {
  /* brain/network SVG as first layer (very subtle) */
  background:
    url("data:image/svg+xml,%3Csvg%20xmlns%3D%27http%3A//www.w3.org/2000/svg%27%20width%3D%27800%27%20height%3D%27600%27%20viewBox%3D%270%200%20800%20600%27%3E%3Cdefs%3E%3CradialGradient%20id%3D%27g%27%20cx%3D%2750%25%27%20cy%3D%2750%25%27%20r%3D%2765%25%27%3E%3Cstop%20offset%3D%270%25%27%20stop-color%3D%27%236366f1%27%20stop-opacity%3D%270.10%27/%3E%3Cstop%20offset%3D%27100%25%27%20stop-color%3D%27%238b5cf6%27%20stop-opacity%3D%270.00%27/%3E%3C/radialGradient%3E%3C/defs%3E%3C!--%20soft%20blob%20--%3E%3Cellipse%20cx%3D%27510%27%20cy%3D%27270%27%20rx%3D%27240%27%20ry%3D%27185%27%20fill%3D%27url(%23g)%27%20opacity%3D%270.45%27/%3E%3C!--%20network%20lines%20--%3E%3Cg%20stroke%3D%27%236366f1%27%20stroke-opacity%3D%270.22%27%20stroke-width%3D%272%27%20fill%3D%27none%27%3E%3Cpath%20d%3D%27M360%20240%20L440%20170%20L540%20195%20L615%20265%20L585%20355%20L495%20395%20L405%20355%20L360%20240%27/%3E%3Cpath%20d%3D%27M420%20210%20L470%20285%20L560%20295%27/%3E%3Cpath%20d%3D%27M430%20350%20L490%20285%20L600%20260%27/%3E%3Cpath%20d%3D%27M440%20170%20L470%20285%20L405%20355%27/%3E%3Cpath%20d%3D%27M540%20195%20L490%20285%20L585%20355%27/%3E%3C/g%3E%3C!--%20nodes%20--%3E%3Cg%20fill%3D%27%238b5cf6%27%20fill-opacity%3D%270.22%27%3E%3Ccircle%20cx%3D%27360%27%20cy%3D%27240%27%20r%3D%276%27/%3E%3Ccircle%20cx%3D%27440%27%20cy%3D%27170%27%20r%3D%276%27/%3E%3Ccircle%20cx%3D%27540%27%20cy%3D%27195%27%20r%3D%276%27/%3E%3Ccircle%20cx%3D%27615%27%20cy%3D%27265%27%20r%3D%276%27/%3E%3Ccircle%20cx%3D%27585%27%20cy%3D%27355%27%20r%3D%276%27/%3E%3Ccircle%20cx%3D%27495%27%20cy%3D%27395%27%20r%3D%276%27/%3E%3Ccircle%20cx%3D%27405%27%20cy%3D%27355%27%20r%3D%276%27/%3E%3Ccircle%20cx%3D%27470%27%20cy%3D%27285%27%20r%3D%277%27/%3E%3C/g%3E%3C!--%20outer%20hint%20--%3E%3Cg%20stroke%3D%27%238b5cf6%27%20stroke-opacity%3D%270.12%27%20stroke-width%3D%272%27%20fill%3D%27none%27%3E%3Cellipse%20cx%3D%27510%27%20cy%3D%27270%27%20rx%3D%27260%27%20ry%3D%27200%27/%3E%3C/g%3E%3C/svg%3E")
    no-repeat right -80px top -60px / 620px auto,

    radial-gradient(800px 500px at 15% 10%, rgba(99,102,241,0.10), transparent 40%),
    radial-gradient(900px 600px at 85% 20%, rgba(139,92,246,0.10), transparent 45%),
    radial-gradient(700px 500px at 50% 90%, rgba(79,70,229,0.08), transparent 50%),
    #f7f7fb;
}

/* ---- Card containers ---- */
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

/* ---- Titles ---- */
.bm-title h1,
.bm-title h2 {
  letter-spacing: -0.025em;
  color: #1f2937;
}

/* Accent line under title */
.bm-title h2::after {
  content: "";
  display: block;
  width: 52px;
  height: 3px;
  margin-top: 6px;
  border-radius: 2px;
  background: linear-gradient(90deg, #6366f1, #8b5cf6);
}

/* ---- Buttons ---- */
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

/* ---- Hide required asterisks ---- */
label span[aria-hidden="true"] {
  display: none;
}

/* Optional: make markdown text slightly softer */
.md, .prose {
  color: rgba(31, 41, 55, 0.92);
}
"""


def build_main_ui():
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

    with gr.Blocks(theme=gr.themes.Soft(), css=APP_CSS) as app:
        with gr.Column(visible=True, elem_classes=["bm-card"]) as page_main:
            main_ui.render()

        with gr.Column(visible=False, elem_classes=["bm-card"]) as page_personas:
            back_btn_1 = gr.Button("← Back")
            personas_ui.render()

        with gr.Column(visible=False, elem_classes=["bm-card"]) as page_experiment:
            back_btn_2 = gr.Button("← Back")
            experiment_ui.render()

        with gr.Column(visible=False, elem_classes=["bm-card"]) as page_results:
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
