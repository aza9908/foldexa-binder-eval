"""Rendering: terminal table, JSON, CSV, standalone HTML."""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import METRICS, TIERS, Evaluation, MetricValue
from .thresholds import ThresholdProfile

__all__ = [
    "render_table",
    "evaluations_to_frame",
    "write_csv",
    "write_json",
    "write_html",
]

_TIER_LABEL = {
    "confidence": "Prediction confidence",
    "interface": "Interface quality",
    "developability": "Developability",
}

_MARK = {"pass": "PASS", "fail": "FAIL", "missing": "  - ", "unfiltered": "  . "}


def _fmt(value: float | None, units: str) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    if units == "count":
        return f"{value:.0f}"
    if units == "A^2":
        return f"{value:.0f}"
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.3f}"


def _threshold_text(mv: MetricValue) -> str:
    if mv.threshold is None:
        return "not gated"
    arrow = ">=" if mv.spec.direction == "higher" else "<="
    return f"{arrow} {_fmt(mv.threshold, mv.spec.units)}"


def render_table(evaluation: Evaluation, profile: ThresholdProfile | None = None) -> str:
    """Human-readable single-candidate report."""
    lines: list[str] = []
    name = evaluation.design_id
    lines.append("=" * 78)
    lines.append(f"  {name}")
    sub = f"  profile: {evaluation.profile}"
    if profile and profile.lineage:
        sub += f"  (extends {' -> '.join(profile.lineage)})"
    lines.append(sub)
    lines.append("=" * 78)

    for tier in TIERS:
        specs = [m for m in METRICS if m.tier == tier]
        if not specs:
            continue
        lines.append("")
        lines.append(f"  {_TIER_LABEL[tier]}")
        lines.append(f"  {'-' * 74}")
        lines.append(
            f"  {'':<4} {'metric':<28} {'value':>12}  {'threshold':>14}"
        )
        for spec in specs:
            mv = evaluation.values[spec.key]
            note = f"   ({mv.note})" if mv.note else ""
            lines.append(
                f"  {_MARK[mv.status]:<4} {spec.label:<28} "
                f"{_fmt(mv.value, spec.units):>12}  {_threshold_text(mv):>14}{note}"
            )

    lines.append("")
    lines.append("  " + "-" * 74)
    verdict = evaluation.verdict
    lines.append(
        f"  VERDICT: {verdict}   "
        f"({evaluation.n_pass}/{evaluation.n_evaluated} gated metrics passed)"
    )
    if evaluation.failures:
        lines.append(
            "  failed: " + ", ".join(v.spec.label for v in evaluation.failures)
        )
    blocking = [v for v in evaluation.missing if v.threshold is not None]
    if blocking:
        lines.append(
            "  not computed: " + ", ".join(v.spec.label for v in blocking)
        )
    lines.append("=" * 78)
    return "\n".join(lines)


def evaluations_to_frame(evaluations: Sequence[Evaluation]) -> pd.DataFrame:
    """One row per candidate, metric values and per-metric pass flags."""
    return pd.DataFrame([e.to_row() for e in evaluations])


def write_csv(evaluations: Sequence[Evaluation], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    evaluations_to_frame(evaluations).to_csv(path, index=False)
    return path


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _evaluation_payload(evaluation: Evaluation) -> dict[str, Any]:
    return {
        "design_id": evaluation.design_id,
        "verdict": evaluation.verdict,
        "profile": evaluation.profile,
        "n_pass": evaluation.n_pass,
        "n_evaluated": evaluation.n_evaluated,
        "metrics": {
            key: {
                "label": mv.spec.label,
                "tier": mv.spec.tier,
                "units": mv.spec.units,
                "direction": mv.spec.direction,
                "value": _jsonable(mv.value),
                "threshold": mv.threshold,
                "passed": mv.passed,
                "status": mv.status,
                "note": mv.note,
                "source": mv.spec.source,
            }
            for key, mv in evaluation.values.items()
        },
        "extras": _jsonable(evaluation.extras),
    }


def write_json(
    evaluations: Sequence[Evaluation],
    path: str | Path,
    profile: ThresholdProfile | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "profile": profile.as_dict() if profile else None,
        "designs": [_evaluation_payload(e) for e in evaluations],
    }
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    return path


_HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Binder evaluation &mdash; {title}</title>
<style>
:root {{
  --bg:#ffffff; --fg:#16181d; --muted:#6b7280; --line:#e5e7eb;
  --pass:#0f766e; --fail:#b91c1c; --na:#9ca3af; --card:#f9fafb;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg:#0f1115; --fg:#e8eaed; --muted:#9aa0a6; --line:#2a2f3a;
    --pass:#2dd4bf; --fail:#f87171; --na:#6b7280; --card:#171a21;
  }}
}}
:root[data-theme="dark"] {{
  --bg:#0f1115; --fg:#e8eaed; --muted:#9aa0a6; --line:#2a2f3a;
  --pass:#2dd4bf; --fail:#f87171; --na:#6b7280; --card:#171a21;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif; }}
.wrap {{ max-width:960px; margin:0 auto; padding:40px 16px 72px; }}
h1 {{ font-size:1.5rem; margin:0 0 4px; letter-spacing:-0.01em; }}
.sub {{ color:var(--muted); font-size:.875rem; margin-bottom:28px; }}
.design {{ border:1px solid var(--line); border-radius:10px;
  margin-bottom:22px; overflow:hidden; background:var(--card); }}
.dhead {{ display:flex; justify-content:space-between; align-items:center;
  gap:12px; padding:14px 18px; border-bottom:1px solid var(--line); }}
.dname {{ font-weight:600; font-family:ui-monospace,monospace; }}
.badge {{ font-size:.75rem; font-weight:700; letter-spacing:.06em;
  padding:4px 10px; border-radius:999px; }}
.badge.PASS {{ color:var(--pass); border:1px solid var(--pass); }}
.badge.FAIL {{ color:var(--fail); border:1px solid var(--fail); }}
.badge.INCOMPLETE {{ color:var(--na); border:1px solid var(--na); }}
table {{ width:100%; border-collapse:collapse; }}
th, td {{ padding:8px 18px; text-align:left; font-size:.875rem;
  border-bottom:1px solid var(--line); }}
th {{ color:var(--muted); font-weight:500; font-size:.75rem;
  text-transform:uppercase; letter-spacing:.05em; }}
td.num {{ text-align:right; font-family:ui-monospace,monospace; }}
tr.tier td {{ background:transparent; color:var(--muted);
  font-size:.72rem; text-transform:uppercase; letter-spacing:.07em;
  padding-top:14px; font-weight:600; }}
.st {{ font-weight:700; font-size:.72rem; letter-spacing:.05em; }}
.st.pass {{ color:var(--pass); }}
.st.fail {{ color:var(--fail); }}
.st.missing, .st.unfiltered {{ color:var(--na); }}
.note {{ color:var(--muted); font-size:.78rem; }}
footer {{ color:var(--muted); font-size:.78rem; margin-top:32px;
  border-top:1px solid var(--line); padding-top:14px; }}
@media (max-width:620px) {{ th:nth-child(5), td:nth-child(5) {{ display:none; }} }}
</style></head><body><div class="wrap">
<h1>Binder candidate evaluation</h1>
<div class="sub">{subtitle}</div>
{body}
<footer>{footer}</footer>
</div></body></html>
"""


def write_html(
    evaluations: Sequence[Evaluation],
    path: str | Path,
    profile: ThresholdProfile | None = None,
) -> Path:
    """Standalone HTML report, readable offline and in dark mode."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    blocks: list[str] = []
    for ev in evaluations:
        rows: list[str] = []
        for tier in TIERS:
            specs = [m for m in METRICS if m.tier == tier]
            if not specs:
                continue
            rows.append(f'<tr class="tier"><td colspan="5">{_TIER_LABEL[tier]}</td></tr>')
            for spec in specs:
                mv = ev.values[spec.key]
                rows.append(
                    "<tr>"
                    f'<td class="st {mv.status}">{mv.status.upper()}</td>'
                    f"<td>{html.escape(spec.label)}</td>"
                    f'<td class="num">{_fmt(mv.value, spec.units)}</td>'
                    f'<td class="num">{html.escape(_threshold_text(mv))}</td>'
                    f'<td class="note">{html.escape(mv.note or spec.source)}</td>'
                    "</tr>"
                )
        blocks.append(
            '<div class="design"><div class="dhead">'
            f'<span class="dname">{html.escape(ev.design_id)}</span>'
            f'<span class="badge {ev.verdict}">{ev.verdict}</span></div>'
            "<table><thead><tr><th></th><th>Metric</th><th>Value</th>"
            "<th>Threshold</th><th>Source</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>"
        )

    n_pass = sum(1 for e in evaluations if e.verdict == "PASS")
    profile_name = profile.name if profile else (evaluations[0].profile if evaluations else "-")
    subtitle = (
        f"{len(evaluations)} candidate(s) &middot; {n_pass} passing &middot; "
        f"profile <strong>{html.escape(str(profile_name))}</strong>"
    )
    footer = (
        "Generated by foldexa-binder-eval on "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. "
        "Thresholds and metric definitions are documented in docs/thresholds.md."
    )
    path.write_text(
        _HTML_TEMPLATE.format(
            title=html.escape(str(profile_name)),
            subtitle=subtitle,
            body="\n".join(blocks),
            footer=footer,
        )
    )
    return path
