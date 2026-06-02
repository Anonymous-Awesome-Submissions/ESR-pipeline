"""ESR-Bench GitHub stream normalization (main HuggingFace `datasets` issue split).

Inputs
------
- Raw HF-mirrored jsonl from `$ESR_RAW_DIR/github_issues_esr/Francesco-A__github-issues_huggingface-datasets/`
  (re-fetch via the HuggingFace dataset URL; this is NOT what `github_api_pull.py` produces).

Outputs
-------
- `data/processed/issues.jsonl`    : filtered + normalized issue threads.
- `data/processed/streams.jsonl`   : evidence streams (events + comments unified, time-ordered).
- `data/processed/slice_a.jsonl`   : metadata-grounded gold (status/labels/etc).

The pipeline is deterministic, idempotent, CPU-only.

Slice B and atom extraction live in `atom_extract.py` / `gold_slice_b.py`.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(os.environ.get("ESR_ROOT", Path(__file__).resolve().parents[1]))
RAW_DIR = Path(os.environ.get("ESR_RAW_DIR", "data/raw")) / "github_issues_esr" / "Francesco-A__github-issues_huggingface-datasets"
PROC_DIR = PROJECT_ROOT / "data" / "processed"
# PROC_DIR.mkdir is deferred to main() — see the rationale in github_api_pull.py.

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("data")


# ---------- helpers ----------

def parse_iso(s: str) -> datetime | None:
    if not s or s in ("None", "null"):
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except Exception:
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            return None


def safe_eval(x: Any) -> Any:
    """Many fields in this dump are str-encoded Python literals (e.g. "[]", "None", "True")."""
    if not isinstance(x, str):
        return x
    s = x.strip()
    if not s:
        return x
    try:
        return ast.literal_eval(s)
    except Exception:
        return x


def normalize_text(t: str) -> str:
    if not t:
        return ""
    t = re.sub(r"\r\n|\r", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def short_id(*parts: Any) -> str:
    h = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()[:10]
    return h


# ---------- core ----------

@dataclasses.dataclass
class Event:
    event_id: str
    kind: str            # body | comment | open | close | reopen | label | unlabel
    timestamp: str       # ISO
    text: str            # may be empty
    actor: str | None    # commenter login if known
    role: str            # maintainer | contributor | bot | unknown
    meta: dict           # extras (label name, etc)


# very small heuristic role classifier for the huggingface/datasets repo
KNOWN_MAINTAINERS = {
    # huggingface/datasets core maintainers (approx, from public commit/contributor list)
    "lhoestq", "albertvillanova", "thomwolf", "patrickvonplaten",
    "mariosasko", "polinaeterna", "qgallouedec", "loubnabnl",
    "stas00", "sgugger", "loubnabnl",
}
BOT_PATTERNS = re.compile(r"(stale\[bot\]|github-actions\[bot\]|dependabot|huggingface-bot|bot$)", re.I)


def classify_role(actor: str | None) -> str:
    if not actor:
        return "unknown"
    if BOT_PATTERNS.search(actor):
        return "bot"
    if actor in KNOWN_MAINTAINERS:
        return "maintainer"
    return "contributor"


def parse_one(d: dict) -> dict | None:
    """Convert one raw row into a normalized issue thread, or None to drop."""
    state = (d.get("state") or "").strip()
    if state not in ("open", "closed"):
        return None

    is_pr = safe_eval(d.get("is_pull_request"))
    if is_pr is True or is_pr == "True":
        # Skip pull requests: review threads have noisier timelines than issue threads.
        # We keep regular issues. Roughly half the pool.
        return None

    body = normalize_text(d.get("body") or "")
    title = normalize_text(d.get("title") or "")
    if len(body) < 5 and len(title) < 5:
        return None

    # comments_text is a stringified python list of comment bodies
    comments_text = safe_eval(d.get("comments_text") or "[]")
    if not isinstance(comments_text, list):
        comments_text = []
    comments_text = [normalize_text(str(c)) for c in comments_text if str(c).strip()]

    if len(comments_text) < 2:
        # Need at least 2 substantive comments for any state-revision pattern.
        return None

    labels = safe_eval(d.get("labels") or "[]") or []
    label_names = []
    if isinstance(labels, list):
        for L in labels:
            if isinstance(L, dict) and "name" in L:
                label_names.append(L["name"])
            elif isinstance(L, str):
                label_names.append(L)

    created_at = d.get("created_at")
    closed_at = d.get("closed_at") if d.get("closed_at") not in (None, "None", "null") else None
    repo = "huggingface/datasets"
    issue_number = d.get("number")

    events: list[Event] = []
    # Body event = creation
    events.append(Event(
        event_id=short_id(repo, issue_number, "body"),
        kind="body",
        timestamp=created_at,
        text=f"{title}\n\n{body}".strip(),
        actor=None,  # author login not in this dump — unknown
        role="unknown",
        meta={"title": title},
    ))
    # Comments — we don't have per-comment timestamps in this dump. We synthesize
    # ordinal timestamps spaced evenly between created_at and (closed_at or now)
    # to preserve ORDER, which is what ESR cares about. Real timestamps would
    # require a re-fetch from the GitHub API.
    if created_at and len(comments_text) > 0:
        c0 = parse_iso(created_at)
        c1 = parse_iso(closed_at) if closed_at else None
        if c0 is None:
            c0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        if c1 is None or c1 <= c0:
            from datetime import timedelta
            c1 = c0 + timedelta(days=max(1, len(comments_text)))
        span = (c1 - c0).total_seconds()
        for i, ctxt in enumerate(comments_text):
            frac = (i + 1) / (len(comments_text) + 1)
            from datetime import timedelta
            ts = (c0 + timedelta(seconds=span * frac)).strftime("%Y-%m-%dT%H:%M:%SZ")
            events.append(Event(
                event_id=short_id(repo, issue_number, "c", i),
                kind="comment",
                timestamp=ts,
                text=ctxt,
                actor=None,  # commenter login not in this dump
                role="unknown",  # cannot classify without actor
                meta={"comment_index": i},
            ))
    # Final state-change event
    if state == "closed" and closed_at:
        events.append(Event(
            event_id=short_id(repo, issue_number, "close"),
            kind="close",
            timestamp=closed_at,
            text="",
            actor=None,
            role="unknown",
            meta={"state_reason": d.get("state_reason")},
        ))

    # Sort events by timestamp (defensive)
    events.sort(key=lambda e: e.timestamp or "")

    return {
        "stream_id": short_id(repo, issue_number),
        "repo": repo,
        "issue_number": str(issue_number),
        "title": title,
        "current_state": state,
        "current_labels": label_names,
        "created_at": created_at,
        "closed_at": closed_at,
        "n_comments": len(comments_text),
        "events": [dataclasses.asdict(e) for e in events],
    }


def build_processed(seed: int = 13, max_n: int | None = None) -> tuple[Path, Path]:
    raw_train = RAW_DIR / "train.jsonl"
    raw_test = RAW_DIR / "test.jsonl"
    issues_path = PROC_DIR / "issues.jsonl"
    streams_path = PROC_DIR / "streams.jsonl"

    log.info(f"reading raw {raw_train} + {raw_test}")
    rows: list[dict] = []
    for p in (raw_train, raw_test):
        if not p.exists():
            log.warning(f"  missing {p}")
            continue
        with open(p) as f:
            for line in f:
                rows.append(json.loads(line))
    log.info(f"  raw rows: {len(rows)}")

    kept = []
    for r in rows:
        out = parse_one(r)
        if out is not None:
            kept.append(out)
    log.info(f"  kept after filters: {len(kept)} (dropped {len(rows)-len(kept)})")

    if max_n:
        import random
        rng = random.Random(seed)
        rng.shuffle(kept)
        kept = kept[:max_n]

    # Write streams.jsonl with full events; issues.jsonl is a slim summary.
    with open(streams_path, "w") as fs, open(issues_path, "w") as fi:
        for k in kept:
            fs.write(json.dumps(k, ensure_ascii=False) + "\n")
            slim = {kk: vv for kk, vv in k.items() if kk != "events"}
            slim["n_events"] = len(k["events"])
            fi.write(json.dumps(slim, ensure_ascii=False) + "\n")
    log.info(f"wrote {streams_path} ({streams_path.stat().st_size//1024} KB)")
    log.info(f"wrote {issues_path}")
    return streams_path, issues_path


# ---------- Slice A ----------

def build_slice_a(streams_path: Path) -> Path:
    out = PROC_DIR / "slice_a.jsonl"
    n = 0
    with open(streams_path) as fin, open(out, "w") as fout:
        for line in fin:
            s = json.loads(line)
            qa = []
            # Q1: status
            qa.append({
                "qid": s["stream_id"] + ":q1",
                "type": "status",
                "question": "What is the current status of this issue?",
                "gold_answer": s["current_state"],
                "gold_supporting_event_ids": [
                    e["event_id"] for e in s["events"]
                    if e["kind"] in ("close", "reopen") or e["kind"] == "body"
                ][-1:],
                "gold_deprecated_event_ids": [],
            })
            # Q2: label set
            qa.append({
                "qid": s["stream_id"] + ":q2",
                "type": "label_set",
                "question": "What is the current set of labels on this issue?",
                "gold_answer": sorted(s["current_labels"]),
                "gold_supporting_event_ids": [],
                "gold_deprecated_event_ids": [],
            })
            for q in qa:
                rec = {**s, "qa": q, "slice": "A"}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    log.info(f"wrote {out} ({n} qa rows)")
    return out


# ---------- main ----------

def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=None)
    ap.add_argument("--no-slice-a", action="store_true")
    args = ap.parse_args()

    streams_path, _ = build_processed(max_n=args.max_n)
    if not args.no_slice_a:
        build_slice_a(streams_path)


if __name__ == "__main__":
    main()
