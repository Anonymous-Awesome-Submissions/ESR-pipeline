"""Atom extraction from issue streams.

For each event in each stream, ask the LLM to produce a list of atoms:
    {entity, attribute, value, polarity, timestamp, source_event_id, source_role,
     source_modality, extractor_confidence, span}

Attributes are constrained to the closed enum defined below in `ATTRIBUTE_ENUM`
(eight types: issue_status, issue_label_set, reproduction_condition, active_blocker,
proposed_workaround, assigned_owner, severity, claim_validity).

Outputs `data/processed/atoms.jsonl` keyed by stream_id; idempotent.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("ESR_ROOT", Path(__file__).resolve().parents[1]))
PROC_DIR = PROJECT_ROOT / "data" / "processed"

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("atom")

ATTRIBUTE_ENUM = [
    "issue_status",
    "issue_label_set",
    "reproduction_condition",
    "active_blocker",
    "proposed_workaround",
    "assigned_owner",
    "severity",
    "claim_validity",
]

PROMPT_HEADER = """You extract structured "evidence atoms" from a single GitHub issue event.

An atom captures one attribute claim about the issue. Output a JSON array (possibly empty) of atom objects with these fields ONLY:
  - entity: a string id of the issue (use the provided "issue_id")
  - attribute: ONE OF {attrs}
  - value: a short string capturing the claim (e.g. "open", "blocked-by-#1234", "fails on macOS Big Sur with Python 3.9")
  - polarity: "+" if the attribute claim is asserted, "-" if explicitly negated
  - confidence: a float in [0,1] for how confident the writer sounds
  - span: a short verbatim quote (≤160 chars) from the event text supporting the atom

Rules:
- Only emit atoms for attributes in the closed enum above.
- Do NOT invent timestamps or actors; the harness adds those.
- Multiple atoms allowed per event when distinct attributes are claimed.
- If the event makes no atomic claim (e.g., a thank-you, a reaction-only comment), output [].
- Output ONLY the JSON array, no prose, no markdown fences.
"""


def build_prompt(issue_id: str, event_kind: str, event_text: str) -> list[dict]:
    sys_msg = PROMPT_HEADER.format(attrs=", ".join(ATTRIBUTE_ENUM))
    user_msg = (
        f"issue_id: {issue_id}\n"
        f"event_kind: {event_kind}\n"
        f"event_text:\n---\n{event_text[:3500]}\n---\n"
        "Atoms (JSON array only):"
    )
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", default=str(PROC_DIR / "streams.jsonl"))
    ap.add_argument("--out", default=str(PROC_DIR / "atoms.jsonl"))
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N streams (for pilot)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--model", default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-event-chars", type=int, default=3500)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from llm_runner import LLMRunner

    runner = LLMRunner(model=args.model) if args.model else LLMRunner()

    streams = []
    with open(args.streams) as f:
        for i, line in enumerate(f):
            if i % args.num_shards != args.shard:
                continue
            streams.append(json.loads(line))
    if args.limit:
        streams = streams[: args.limit]
    log.info(f"shard {args.shard}/{args.num_shards}: {len(streams)} streams to process")

    out_path = Path(args.out)
    if args.num_shards > 1:
        out_path = out_path.with_suffix(f".s{args.shard}of{args.num_shards}.jsonl")

    # Build flat list of (stream_idx, event_idx, prompt) work items
    work = []  # (sid, eid, prompt_msgs, event_kind, event_id)
    for si, s in enumerate(streams):
        for ei, e in enumerate(s["events"]):
            text = e.get("text", "") or ""
            if not text.strip():
                continue
            text = text[: args.max_event_chars]
            prompt = build_prompt(
                issue_id=f"{s['repo']}#{s['issue_number']}",
                event_kind=e["kind"],
                event_text=text,
            )
            work.append((si, ei, prompt, e["kind"], e["event_id"]))
    log.info(f"total LLM calls: {len(work)}")

    # Pre-allocate per-stream atom lists
    stream_atoms: dict[int, list[dict]] = {i: [] for i in range(len(streams))}

    t0 = time.time()
    for chunk_start in range(0, len(work), args.batch):
        chunk = work[chunk_start : chunk_start + args.batch]
        chats = [w[2] for w in chunk]
        responses = runner.chat(
            chats, max_tokens=args.max_tokens, temperature=0.0, seed=args.seed,
        )
        for (si, ei, _msgs, ek, eid), resp in zip(chunk, responses):
            parsed = LLMRunner.parse_json(resp)
            if not isinstance(parsed, list):
                # try to recover by wrapping a single object
                if isinstance(parsed, dict):
                    parsed = [parsed]
                else:
                    parsed = []
            ev = streams[si]["events"][ei]
            for ai, atom in enumerate(parsed):
                if not isinstance(atom, dict):
                    continue
                attr = atom.get("attribute")
                if attr not in ATTRIBUTE_ENUM:
                    continue
                stream_atoms[si].append({
                    "atom_id": f"{streams[si]['stream_id']}_{eid}_{ai}",
                    "stream_id": streams[si]["stream_id"],
                    "event_id": eid,
                    "event_index": ei,
                    "event_kind": ek,
                    "timestamp": ev.get("timestamp"),
                    "actor": ev.get("actor"),
                    "role": ev.get("role"),
                    "modality": "text",
                    "entity": str(atom.get("entity") or "")[:200],
                    "attribute": attr,
                    "value": str(atom.get("value") or "")[:300],
                    "polarity": str(atom.get("polarity") or "+")[:1],
                    "confidence": float(atom.get("confidence", 0.5))
                                  if isinstance(atom.get("confidence"), (int, float, str)) else 0.5,
                    "span": str(atom.get("span") or "")[:300],
                })
        elapsed = time.time() - t0
        done = min(chunk_start + args.batch, len(work))
        log.info(f"  progress {done}/{len(work)}  ({done/elapsed:.1f} calls/s, {elapsed:.0f}s)")

    log.info(f"finished in {time.time()-t0:.0f}s")

    # Write per-stream rows
    with open(out_path, "w") as f:
        for si, s in enumerate(streams):
            atoms = stream_atoms[si]
            # Force each atom's `entity` to a stable id (the issue id)
            issue_id = f"{s['repo']}#{s['issue_number']}"
            for a in atoms:
                a["entity"] = issue_id
            row = {"stream_id": s["stream_id"], "n_atoms": len(atoms), "atoms": atoms}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info(f"wrote {out_path} ({out_path.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
