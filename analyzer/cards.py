from __future__ import annotations

import html
from typing import Any


TextCard = dict[str, Any]
TableCard = dict[str, Any]
ChartCard = dict[str, Any]
Card = dict[str, Any]


def make_text_card(title: str, text: str, meta: dict[str, Any] | None = None) -> TextCard:
    return {"type": "text", "title": title, "text": text, "meta": meta or {}}


def dataframe_to_html(df: Any) -> str:
    headers = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns.tolist())
    rows: list[str] = []
    for row in df.itertuples(index=False, name=None):
        cells = "".join(f"<td>{html.escape(str(v))}</td>" for v in row)
        rows.append(f"<tr>{cells}</tr>")
    body = "".join(rows)
    return (
        '<table class="result-table">'
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
    )


def make_table_card(title: str, df: Any, meta: dict[str, Any] | None = None) -> TableCard:
    return {
        "type": "table",
        "title": title,
        "html": dataframe_to_html(df),
        "meta": meta or {},
    }


def make_chart_card(title: str, image_data_uri: str, meta: dict[str, Any] | None = None) -> ChartCard:
    return {
        "type": "chart",
        "title": title,
        "image_data_uri": image_data_uri,
        "meta": meta or {},
    }


def _render_meta(meta: dict[str, Any]) -> str:
    if not meta:
        return ""
    items = [f"{html.escape(str(k))}: {html.escape(str(v))}" for k, v in meta.items() if v is not None]
    if not items:
        return ""
    return f'<div class="result-meta">{" | ".join(items)}</div>'


def _render_text(text: str) -> str:
    if "```" in text:
        return f'<pre class="result-pre">{html.escape(text)}</pre>'
    escaped = html.escape(text).replace("\n", "<br>")
    return f'<div class="result-text">{escaped}</div>'


def render_cards_to_html(cards: list[Card]) -> str:
    blocks: list[str] = ['<div class="result-cards">']
    for idx, card in enumerate(cards, start=1):
        title = html.escape(str(card.get("title", "결과")))
        meta = card.get("meta", {})
        trace_id_attr = ""
        trace_id = meta.get("trace_id") if isinstance(meta, dict) else None
        if trace_id:
            trace_id_attr = f' data-trace-id="{html.escape(str(trace_id))}"'

        blocks.append(f'<section class="result-card"{trace_id_attr}><div class="result-card-title">#{idx} {title}</div>')
        card_type = card.get("type")
        if card_type == "table":
            blocks.append(card.get("html", ""))
        elif card_type == "chart":
            src = card.get("image_data_uri", "")
            blocks.append(f'<img class="result-img" src="{src}" alt="{title}">')
        else:
            blocks.append(_render_text(str(card.get("text", ""))))

        blocks.append(_render_meta(meta if isinstance(meta, dict) else {}))
        blocks.append("</section>")
    blocks.append("</div>")
    return "".join(blocks)
