"""Render the report dict produced by `_collect_report` into PDF bytes.

Uses ReportLab platypus for layout. No native deps. Charts are drawn with the
built-in `reportlab.graphics.charts` so we stay pure-Python.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone

from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ACCENT = colors.HexColor("#0a7d2c")
SAVED = colors.HexColor("#1f6feb")
MUTED = colors.HexColor("#6b6b6b")
BORDER = colors.HexColor("#e8e8e8")
HEADER_BG = colors.HexColor("#f4f4f4")


def _fmt_int(n: float | int | None) -> str:
    return f"{int(n or 0):,}"


def _fmt_usd(n: float | None) -> str:
    return f"${(n or 0.0):,.4f}"


def _fmt_usd_2(n: float | None) -> str:
    return f"${(n or 0.0):,.2f}"


def _fmt_ts(ts: float, bucket_seconds: int) -> str:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d") if bucket_seconds >= 86400 else dt.strftime("%m-%d %H:%M")


def _fmt_dt(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontSize=20, leading=24, spaceAfter=4
        ),
        "sub": ParagraphStyle(
            "sub", parent=base["Normal"], fontSize=10, textColor=MUTED, spaceAfter=14
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontSize=12,
            leading=14,
            spaceBefore=14,
            spaceAfter=6,
            textColor=colors.black,
        ),
        "p": ParagraphStyle("p", parent=base["Normal"], fontSize=10, leading=13),
        "muted": ParagraphStyle(
            "muted", parent=base["Normal"], fontSize=9, leading=12, textColor=MUTED
        ),
        "kpi_v": ParagraphStyle(
            "kpi_v", parent=base["Normal"], fontSize=16, leading=18, alignment=1
        ),
        "kpi_l": ParagraphStyle(
            "kpi_l",
            parent=base["Normal"],
            fontSize=8,
            leading=10,
            textColor=MUTED,
            alignment=1,
        ),
    }


def _kpi_row(items: list[tuple[str, str]], styles: dict[str, ParagraphStyle]) -> Table:
    """Render KPI cards across a single row."""
    rows = [
        [Paragraph(v, styles["kpi_v"]) for v, _ in items],
        [Paragraph(label, styles["kpi_l"]) for _, label in items],
    ]
    col_w = (LETTER[0] - 1.5 * inch) / len(items)
    t = Table(rows, colWidths=[col_w] * len(items))
    t.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER),
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return t


def _table(
    header: list[str],
    rows: list[list[str]],
    col_widths: list[float] | None = None,
    align_right: list[int] | None = None,
) -> Table:
    align_right = align_right or []
    data = [header] + rows
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, BORDER),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for col in align_right:
        style.append(("ALIGN", (col, 0), (col, -1), "RIGHT"))
    t.setStyle(TableStyle(style))
    return t


def _timeseries_chart(rows: list[dict], bucket_seconds: int) -> Drawing:
    """Two-series line chart: spend (USD) and saved tokens, over the window."""
    d = Drawing(LETTER[0] - 1.5 * inch, 200)
    if not rows:
        d.add(String(20, 90, "No traffic in this window.", fontSize=10, fillColor=MUTED))
        return d

    spend_data = [(r["bucket"], (r.get("spent_usd") or 0.0)) for r in rows]
    saved_data = [(r["bucket"], (r.get("saved_tokens") or 0.0) / 1000.0) for r in rows]

    plot = LinePlot()
    plot.x = 50
    plot.y = 35
    plot.height = 145
    plot.width = LETTER[0] - 1.5 * inch - 80
    plot.data = [spend_data, saved_data]
    plot.lines[0].strokeColor = ACCENT
    plot.lines[0].strokeWidth = 1.6
    plot.lines[1].strokeColor = SAVED
    plot.lines[1].strokeWidth = 1.6
    plot.xValueAxis.valueMin = rows[0]["bucket"]
    plot.xValueAxis.valueMax = rows[-1]["bucket"] + bucket_seconds
    plot.xValueAxis.labelTextFormat = lambda v: _fmt_ts(v, bucket_seconds)
    plot.xValueAxis.labels.fontSize = 7
    plot.xValueAxis.labels.angle = 30
    plot.xValueAxis.labels.dy = -6
    plot.yValueAxis.labelTextFormat = "%.4f"
    plot.yValueAxis.labels.fontSize = 7
    d.add(plot)

    legend = Legend()
    legend.x = 60
    legend.y = 195
    legend.deltax = 110
    legend.fontSize = 8
    legend.alignment = "right"
    legend.colorNamePairs = [(ACCENT, "spend (USD)"), (SAVED, "saved tokens (k)")]
    legend.boxAnchor = "nw"
    d.add(legend)
    return d


def _comparison_block(payload: dict, styles: dict[str, ParagraphStyle]) -> list:
    """The headline section: actual vs counterfactual."""
    cmp = payload["comparison"]
    saved_pct = cmp["saved_pct"]
    return [
        Paragraph("Bottom line", styles["h2"]),
        _kpi_row(
            [
                (_fmt_usd_2(cmp["without_tokenq_usd"]), "without tokenq"),
                (_fmt_usd_2(cmp["with_tokenq_usd"]), "with tokenq (actual)"),
                (_fmt_usd_2(cmp["saved_usd"]), "saved"),
                (f"{saved_pct:.1f}%", "savings rate"),
            ],
            styles,
        ),
        Spacer(1, 6),
        Paragraph(
            "Counterfactual cost is computed from <i>saved_tokens</i> at each model's base "
            "input rate; cache-read and cache-creation discounts already captured upstream "
            "are reflected in the actual spend.",
            styles["muted"],
        ),
    ]


def _totals_block(payload: dict, styles: dict[str, ParagraphStyle]) -> list:
    t = payload["totals"]
    return [
        Paragraph("Usage totals", styles["h2"]),
        _kpi_row(
            [
                (_fmt_int(t.get("requests")), "requests"),
                (_fmt_int((t.get("input_tokens") or 0) + (t.get("output_tokens") or 0)), "tokens"),
                (_fmt_int(t.get("cache_read_tokens")), "cache reads"),
                (_fmt_int(t.get("saved_tokens")), "saved tokens"),
                (f"{int(t.get('avg_latency_ms') or 0)}ms", "avg latency"),
            ],
            styles,
        ),
    ]


def _stage_breakdown_block(payload: dict, styles: dict[str, ParagraphStyle]) -> list:
    t = payload["totals"]
    total_saved = t.get("saved_tokens") or 0
    rows = [
        ("cache", "full short-circuit · no upstream call",
         t.get("saved_by_cache") or 0, t.get("reqs_cache") or 0),
        ("dedup", "duplicate tool_results stubbed",
         t.get("saved_by_dedup") or 0, t.get("reqs_dedup") or 0),
        ("compress", "tool_result trim + ANSI strip",
         t.get("saved_by_compress") or 0, t.get("reqs_compress") or 0),
        ("skills", "irrelevant skill descriptions trimmed",
         t.get("saved_by_skills") or 0, t.get("reqs_skills") or 0),
        ("compaction", "transcript rollover (per-turn)",
         payload["compactions"]["saved_per_turn"], len(payload["compactions"]["events"])),
    ]
    body = []
    for source, desc, tokens, reqs in rows:
        pct = (tokens / total_saved * 100.0) if total_saved else 0.0
        body.append([source, desc, _fmt_int(reqs), _fmt_int(tokens), f"{pct:.0f}%"])

    return [
        Paragraph("Savings by stage", styles["h2"]),
        _table(
            ["source", "description", "requests", "tokens skipped", "share"],
            body,
            col_widths=[0.9 * inch, 2.6 * inch, 0.9 * inch, 1.1 * inch, 0.6 * inch],
            align_right=[2, 3, 4],
        ),
    ]


def _by_model_block(payload: dict, styles: dict[str, ParagraphStyle]) -> list:
    rows = []
    for r in payload["by_model"]:
        tokens = (r.get("input_tokens") or 0) + (r.get("output_tokens") or 0)
        rows.append(
            [
                (r.get("model") or "(other)").replace("claude-", ""),
                _fmt_int(r.get("requests")),
                _fmt_int(tokens),
                _fmt_int(r.get("saved_tokens")),
                _fmt_usd(r.get("spent_usd")),
                _fmt_usd(r.get("saved_usd")),
            ]
        )
    if not rows:
        return [Paragraph("By model", styles["h2"]), Paragraph("No traffic.", styles["muted"])]
    return [
        Paragraph("By model", styles["h2"]),
        _table(
            ["model", "reqs", "tokens", "saved tokens", "spent", "saved $"],
            rows,
            col_widths=[1.6 * inch, 0.7 * inch, 1.0 * inch, 1.1 * inch, 0.9 * inch, 0.9 * inch],
            align_right=[1, 2, 3, 4, 5],
        ),
    ]


def _timeseries_block(
    payload: dict, styles: dict[str, ParagraphStyle]
) -> list:
    bucket_seconds = payload["window"]["bucket_seconds"]
    rows = payload["timeseries"]
    label = "hourly" if bucket_seconds == 3600 else "daily"
    out: list = [
        Paragraph(f"Activity over time ({label})", styles["h2"]),
        _timeseries_chart(rows, bucket_seconds),
    ]
    if rows:
        body = [
            [
                _fmt_ts(r["bucket"], bucket_seconds),
                _fmt_int(r.get("requests")),
                _fmt_int(r.get("tokens")),
                _fmt_int(r.get("saved_tokens")),
                _fmt_usd(r.get("spent_usd")),
            ]
            for r in rows
        ]
        out.append(Spacer(1, 6))
        out.append(
            _table(
                ["bucket", "reqs", "tokens", "saved tokens", "spent"],
                body,
                col_widths=[1.6 * inch, 0.8 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch],
                align_right=[1, 2, 3, 4],
            )
        )
    return out


def _expensive_block(payload: dict, styles: dict[str, ParagraphStyle]) -> list:
    rows = []
    for r in payload["expensive"]:
        rows.append(
            [
                _fmt_dt(r["ts"]),
                (r.get("model") or "—").replace("claude-", ""),
                _fmt_int(r.get("input_tokens")),
                _fmt_int(r.get("output_tokens")),
                f"{r.get('latency_ms') or 0}ms",
                _fmt_usd(r.get("estimated_cost_usd")),
            ]
        )
    if not rows:
        return [
            Paragraph("Top expensive requests", styles["h2"]),
            Paragraph("No billable requests in this window.", styles["muted"]),
        ]
    return [
        Paragraph("Top expensive requests", styles["h2"]),
        _table(
            ["time", "model", "in", "out", "latency", "cost"],
            rows,
            col_widths=[1.6 * inch, 1.4 * inch, 0.7 * inch, 0.7 * inch, 0.7 * inch, 0.9 * inch],
            align_right=[2, 3, 4, 5],
        ),
    ]


def _events_block(payload: dict, styles: dict[str, ParagraphStyle]) -> list:
    out: list = [Paragraph("Compactions & skill compressions", styles["h2"])]
    cps = payload["compactions"]["events"]
    if cps:
        rows = [
            [
                _fmt_dt(r["ts"]),
                (r.get("model") or "—").replace("claude-", ""),
                _fmt_int(r.get("dropped_messages")),
                _fmt_int(r.get("dropped_tokens")),
                _fmt_int(r.get("summary_tokens")),
                _fmt_int(r.get("saved_per_turn")),
            ]
            for r in cps
        ]
        out.append(Paragraph("transcript rollovers", styles["p"]))
        out.append(
            _table(
                ["time", "model", "msgs dropped", "tokens dropped", "summary", "saved/turn"],
                rows,
                col_widths=[
                    1.5 * inch,
                    1.2 * inch,
                    0.9 * inch,
                    1.0 * inch,
                    0.8 * inch,
                    0.8 * inch,
                ],
                align_right=[2, 3, 4, 5],
            )
        )
        out.append(Spacer(1, 8))
    skills = payload["skill_compressions"]
    if skills:
        rows = [
            [
                _fmt_dt(r["ts"]),
                (r.get("path") or "")[-40:],
                (r.get("model") or "—").replace("claude-", ""),
                _fmt_int(r.get("before_tokens")),
                _fmt_int(r.get("after_tokens")),
                _fmt_int(r.get("saved_tokens")),
            ]
            for r in skills
        ]
        out.append(Paragraph("skill rewrites (offline)", styles["p"]))
        out.append(
            _table(
                ["time", "path", "model", "before", "after", "saved"],
                rows,
                col_widths=[
                    1.5 * inch,
                    2.0 * inch,
                    1.0 * inch,
                    0.7 * inch,
                    0.7 * inch,
                    0.7 * inch,
                ],
                align_right=[3, 4, 5],
            )
        )
    if not cps and not skills:
        out.append(Paragraph("No compaction or skill-compression events.", styles["muted"]))
    return out


def render_report_pdf(payload: dict) -> bytes:
    """Render the report payload to PDF bytes."""
    styles = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="tokenq audit report",
        author="tokenq",
    )

    w = payload["window"]
    story: list = [
        Paragraph("tokenq audit report", styles["title"]),
        Paragraph(
            f"Window: {_fmt_dt(w['from'])}  →  {_fmt_dt(w['to'])} "
            f"&nbsp;·&nbsp; generated {_fmt_dt(w['generated_at'])}",
            styles["sub"],
        ),
    ]
    story += _comparison_block(payload, styles)
    story += _totals_block(payload, styles)
    story += _stage_breakdown_block(payload, styles)
    story.append(PageBreak())
    story += _timeseries_block(payload, styles)
    story.append(PageBreak())
    story += _by_model_block(payload, styles)
    story += _expensive_block(payload, styles)
    story.append(PageBreak())
    story += _events_block(payload, styles)

    doc.build(story)
    return buf.getvalue()
