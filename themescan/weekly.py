"""Weekly report: one run that logs the picks and writes a markdown summary of what changed.

Sections fail soft (a network hiccup in one source never blocks the rest). Pushing to a phone via ntfy.sh is opt-in (topic given by you).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import forward

OUT = Path("reports")


def previous_snapshot(universe: str, asof: pd.Timestamp) -> pd.DataFrame | None:
    if not forward.LOG.exists():
        return None
    log = pd.read_csv(forward.LOG, parse_dates=["asof"])
    old = log[(log["universe"] == universe) & (log["asof"] < asof)]
    if old.empty:
        return None
    d = old["asof"].max()
    return old[old["asof"] == d].set_index("theme")[["conviction", "verdict"]].assign(date=d)


def changes_section(universe: str, df: pd.DataFrame, prev: pd.DataFrame | None, top=6) -> list[str]:
    lines = [f"### {universe.replace('.yaml', '')}"]
    strong = df[df.verdict.isin(["Strong", "Positive"])]
    lines.append("**Strong / Positive now:** " + (", ".join(f"{k} ({r.conviction:.0f})" for k, r in strong.iterrows()) or "none"))
    lines.append("**Avoid now:** " + (", ".join(f"{k} ({r.conviction:.0f})" for k, r in df[df.verdict == "Avoid"].iterrows()) or "none"))
    if prev is None:
        lines.append("_No earlier snapshot in the log yet, so no week-on-week changes._")
        return lines
    j = df[["conviction", "verdict"]].join(prev, rsuffix="_prev", how="inner")
    up = j[j.verdict.isin(["Strong", "Positive"]) & ~j.verdict_prev.isin(["Strong", "Positive"])]
    down = j[j.verdict.isin(["Avoid"]) & (j.verdict_prev != "Avoid")]
    out = j[~j.verdict.isin(["Strong", "Positive"]) & j.verdict_prev.isin(["Strong", "Positive"])]
    lines.append(f"**Since {pd.Timestamp(prev.date.iloc[0]).date()}:**")
    lines.append(f"- Newly Strong/Positive: " + (", ".join(f"{k} ({r.verdict_prev} -> {r.verdict})" for k, r in up.iterrows()) or "none"))
    lines.append(f"- Dropped out of Strong/Positive: " + (", ".join(f"{k} ({r.verdict_prev} -> {r.verdict})" for k, r in out.iterrows()) or "none"))
    lines.append(f"- Newly Avoid: " + (", ".join(f"{k} ({r.verdict_prev} -> Avoid)" for k, r in down.iterrows()) or "none"))
    mv = (j.conviction - j.conviction_prev).sort_values()
    lines.append("- Biggest conviction moves: " + ", ".join(f"{k} {v:+.0f}" for k, v in list(mv.tail(3).items())[::-1] + list(mv.head(3).items())))
    return lines


def compact(sections: dict[str, list[str]], max_chars=900) -> str:
    """Short plain-text summary for a phone push."""
    parts = []
    for name, lines in sections.items():
        key = [l.replace("**", "").replace("_", "") for l in lines if l.startswith(("**Strong", "- Newly", "- Dropped", "Fuel", "AI source"))]
        if key:
            parts.append(name + ": " + " | ".join(key))
    return "\n".join(parts)[:max_chars]


def push_ntfy(topic: str, title: str, body: str) -> bool:
    import requests

    try:
        r = requests.post(f"https://ntfy.sh/{topic}", data=body.encode("utf-8"), headers={"Title": title, "Tags": "chart_with_upwards_trend"}, timeout=20)
        return r.status_code == 200
    except Exception:
        return False
