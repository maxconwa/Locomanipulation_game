"""The scorecard format every eval writes: a list of rows (one metric each) inside the eval's own result dict.

A row is (section, key, label, value, reference, target, sense): `reference` is the baseline the plots draw dashed
(sport mode for the legs, the IK alone for the arms), `target` and `sense` ("ge" / "le") the success criterion, if the
metric has one; Met is value against target. A new metric is one more row: the json, the markdown table and the
plots (which find rows by key) pick it up. Units are in the label.

<name>.json = the eval's result dict + "rows" + the eval conditions (conditions.stamp); <name>.md = the rows as one
table per section, plus whatever free text the eval adds.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass


@dataclass
class Row:
    section: str
    key: str
    label: str
    value: float | None
    reference: float | str | None = None
    target: float | None = None
    sense: str | None = None  # "ge" | "le"
    digits: int = 3

    @property
    def met(self) -> bool | None:
        if self.target is None or self.sense is None:
            return None
        v = self.value
        if v is None or not math.isfinite(v):
            return False
        return v >= self.target if self.sense == "ge" else v <= self.target


def num(v) -> float:
    """A float, NaN for anything missing."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def fmt(v, digits: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    return "—" if not math.isfinite(v) else f"{v:.{digits}f}"


def rows_table(rows: list[Row]) -> str:
    """Markdown: one table per section, in the rows' order."""
    out, section = [], None
    for r in rows:
        if r.section != section:
            section = r.section
            out += ["", f"### {section}", "", "| Metric | Value | Reference | Target | Met |", "| --- | --- | --- | --- | --- |"]
        target = "" if r.target is None else f"{'≥' if r.sense == 'ge' else '≤'} {r.target:g}"
        met = "" if r.met is None else ("yes" if r.met else "**no**")
        ref = fmt(r.reference, r.digits) if not isinstance(r.reference, str) else r.reference
        out.append(f"| {r.label} | {fmt(r.value, r.digits)} | {ref if r.reference is not None else ''} | {target} | {met} |")
    return "\n".join(out)


def rows_to_json(rows: list[Row]) -> list[dict]:
    return [dict(asdict(r), met=r.met) for r in rows]


def write(out_base: str, result: dict, rows: list[Row], header: str, footer: str = "") -> None:
    """<out_base>.json and <out_base>.md."""
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    result = dict(result, rows=rows_to_json(rows))
    met = [r.met for r in rows if r.met is not None]
    result["targets_met"], result["targets"] = int(sum(met)), len(met)
    with open(out_base + ".json", "w") as f:
        json.dump(result, f, indent=1, default=lambda o: float(o) if hasattr(o, "__float__") else str(o))
    with open(out_base + ".md", "w") as f:
        f.write(header.rstrip() + "\n" + rows_table(rows) + "\n" + (("\n" + footer.rstrip() + "\n") if footer else ""))
