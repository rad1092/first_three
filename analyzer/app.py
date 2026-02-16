from __future__ import annotations

from pathlib import Path

import webview

from .backend import check_bitnetd_health


class AnalyzerApi:
    def install_build_engine(self) -> str:
        return "not implemented yet"

    def download_model(self) -> str:
        return "not implemented yet"


def _render_html(status_text: str, is_connected: bool) -> str:
    assets_dir = Path(__file__).parent / "assets"
    template = (assets_dir / "index.html").read_text(encoding="utf-8")

    return (
        template.replace("__STATUS_TEXT__", status_text)
        .replace("__STATUS_CLASS__", "ok" if is_connected else "fail")
        .replace("__FAILURE_ACTIONS_DISPLAY__", "none" if is_connected else "flex")
    )


def run_app() -> None:
    is_connected, status_text = check_bitnetd_health()
    html = _render_html(status_text=status_text, is_connected=is_connected)

    api = AnalyzerApi()
    window = webview.create_window(
        "Analyzer",
        html=html,
        js_api=api,
        width=1080,
        height=720,
    )
    webview.start(debug=False)


if __name__ == "__main__":
    run_app()
