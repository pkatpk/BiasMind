# src/ui_results.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import datetime

import gradio as gr


RESULTS_METADATA_DIR = Path("results/metadata")
RESULTS_RAW_DIR = Path("results/raw")
RESULTS_SCORED_DIR = Path("results/scored")

TEXT_EXTS = {".txt", ".md", ".json", ".jsonl", ".csv", ".tsv", ".log", ".yaml", ".yml"}


@dataclass
class FileItem:
    label: str      # what user sees
    value: str      # full path for download
    path: Path


def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.1f} {u}" if u != "B" else f"{int(x)} {u}"
        x /= 1024.0
    return f"{int(n)} B"


def _mtime_str(p: Path) -> str:
    try:
        ts = p.stat().st_mtime
        return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "unknown"


def _list_files(dir_path: Path) -> List[FileItem]:
    if not dir_path.exists():
        return []
    files = [p for p in dir_path.rglob("*") if p.is_file()]
    # Sort by modified time (newest first)
    files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)

    items: List[FileItem] = []
    for p in files:
        # Show relative to the results folder for readability
        try:
            rel = p.relative_to(Path("results"))
            label = str(rel).replace("\\", "/")
        except Exception:
            label = p.name
        items.append(FileItem(label=label, value=str(p), path=p))
    return items


def _choices_for_dir(dir_path: Path) -> List[Tuple[str, str]]:
    items = _list_files(dir_path)
    return [(it.label, it.value) for it in items]


def _file_info(file_path: Optional[str]) -> str:
    if not file_path:
        return "_No file selected._"
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        return f"❌ Not found: `{file_path}`"
    size = _human_bytes(p.stat().st_size)
    mtime = _mtime_str(p)
    return f"**File:** `{p.as_posix()}`  \n**Size:** {size}  \n**Modified:** {mtime}"


def _preview_file(file_path: Optional[str]) -> str:
    if not file_path:
        return ""
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        return ""
    if p.suffix.lower() not in TEXT_EXTS:
        return "(Preview not available for this file type.)"

    try:
        # Read a limited amount (safe for large files)
        with p.open("r", encoding="utf-8", errors="replace") as f:
            lines = []
            for i, line in enumerate(f):
                lines.append(line.rstrip("\n"))
                if i >= 199:  # 200 lines max
                    lines.append("… (truncated)")
                    break
        return "\n".join(lines)
    except Exception as e:
        return f"(Could not preview file: {e})"


def _download_selected(file_path: Optional[str]):
    # gr.File expects a path-like string (or list of strings)
    if not file_path:
        return None
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        return None
    return str(p)


def _refresh_dir(dir_path: Path):
    # returns: choices, clears selection, clears info, clears preview, clears download
    return (
        gr.update(choices=_choices_for_dir(dir_path), value=None),
        gr.update(value="_No file selected._"),
        gr.update(value=""),
        None,
    )


def _build_dir_tab(title: str, dir_path: Path):
    gr.Markdown(f"### {title}")
    if not dir_path.exists():
        gr.Markdown(f"⚠️ Folder not found: `{dir_path.as_posix()}`")

    with gr.Row():
        dd = gr.Dropdown(
            choices=_choices_for_dir(dir_path),
            value=None,
            label="Select a file",
        )
        btn_refresh = gr.Button("Refresh", variant="secondary")

    info = gr.Markdown("_No file selected._")
    preview = gr.Textbox(label="Preview", lines=12, interactive=False)
    btn_dl = gr.Button("Download selected", variant="primary")
    dl = gr.File(label="Download")

    dd.change(fn=_file_info, inputs=[dd], outputs=[info])
    dd.change(fn=_preview_file, inputs=[dd], outputs=[preview])
    btn_dl.click(fn=_download_selected, inputs=[dd], outputs=[dl])

    btn_refresh.click(
        fn=lambda: _refresh_dir(dir_path),
        inputs=[],
        outputs=[dd, info, preview, dl],
    )


def build_results_ui():
    with gr.Blocks() as results_ui:
        gr.Markdown("## Results")

        # Order as requested: metadata -> raw -> scored
        with gr.Tabs():
            with gr.Tab("metadata"):
                _build_dir_tab("Metadata results", RESULTS_METADATA_DIR)

            with gr.Tab("raw"):
                _build_dir_tab("Raw results", RESULTS_RAW_DIR)

            with gr.Tab("scored"):
                _build_dir_tab("Scored results", RESULTS_SCORED_DIR)

    return results_ui


if __name__ == "__main__":
    app = build_results_ui()
    app.launch(share=True)
