import os
"""Rule-detect csc + refines candidates across all 4 datasets, then build
DeepSeek workspecs to verify each candidate and generate a clean QA.

CSC (cross_source_conflict) pattern:
  Same (entity, attribute) key has ≥2 atoms with different values, from
  DIFFERENT actors/roles, no later atom restores consistency. We treat the
  LATEST 2 disagreeing atoms as candidates (they bound the unresolved conflict).

Refines pattern:
  Same (entity, attribute) key has ≥2 atoms where a LATER atom's value
  contains the earlier value (substring) AND is meaningfully longer
  (e.g., "Python 3.10" -> "Python 3.10.4"; "fails on Colab" -> "fails on
  Colab with Python 3.7.14"). The pair must NOT be a flip-flop (i.e., no
  third atom flipping back).
"""
import json
from collections import defaultdict
from pathlib import Path

P = Path(os.environ.get("ESR_ROOT", Path(__file__).resolve().parents[1]))


CSC_SYS = """You verify and clean up cross_source_conflict (CSC) QA candidates.

A CSC item satisfies: ≥2 sources give conflicting answers within overlapping time windows for the same entity-attribute key, AND no later event resolves the disagreement. The correct gold_answer should be ABSTAIN.

You are given:
  - The (entity, attribute) key
  - 2 conflicting atoms from different actors at overlapping times
  - The surrounding event stream

Verify:
  - Both atoms are about the same entity-attribute
  - Their values genuinely contradict each other (not refinements)
  - The atoms come from DIFFERENT sources (different actors / different roles)
  - NO later event in the stream resolves the conflict by supersession

Output JSON exactly:
{
  "is_csc": true | false,
  "rejection_reason": "<short>" or null,
  "question": "<natural-language question asking for the current value, required if true>",
  "gold_answer": "ABSTAIN"  (required if true),
  "supporting_event_ids": [<event_ids of the conflicting atoms>],
  "deprecated_event_ids": []
}
JSON only, no prose."""


REFINES_SYS = """You verify and clean up refines QA candidates.

A refines item satisfies: a later event provides a MORE PRECISE / FINER-GRAINED version of an earlier event's answer. The same core claim stays; the later version just adds precision (longer string, narrower condition).

Key test: if you removed the later event, the earlier answer would still stand (just coarser).

You are given:
  - The (entity, attribute) key
  - 2 atoms: earlier (coarser) and later (finer)
  - The surrounding event stream

Verify:
  - The later atom's value adds precision (NOT a different method/value).
  - There is NO third event that flips back to the earlier value (would be rev_rev, not refines).
  - The later atom is the LATEST event for this key.

Output JSON exactly:
{
  "is_refines": true | false,
  "rejection_reason": "<short>" or null,
  "question": "<natural-language question, required if true>",
  "gold_answer": "<the more precise value, required if true>",
  "supporting_event_ids": [<latest finer atom event_id>],
  "deprecated_event_ids": []
}
JSON only, no prose."""


def fmt_events(events, focus_ids, cap=15):
    focus = [e for e in events if e.get("event_id") in focus_ids]
    rest = [e for e in events if e.get("event_id") not in focus_ids][: cap - len(focus)]
    all_ev = sorted(focus + rest, key=lambda e: e.get("timestamp") or "")
    lines = []
    for e in all_ev:
        eid = e.get("event_id", "")
        kind = (e.get("kind") or "?")[:8].ljust(8)
        ts = (e.get("timestamp") or "")[:10]
        text = (e.get("text") or "").replace("\n", " ").strip()[:300]
        mark = "★" if eid in focus_ids else " "
        lines.append(f"  {mark} {eid}  [{kind} {ts}] {text}")
    return "\n".join(lines)


def detect_csc(atoms_file, streams_file):
    """Atoms grouped by (entity, attribute). Find groups with ≥2 atoms of different
    values from different events (look up actor via stream), with no later atom
    matching either value."""
    atoms_by_stream = {}
    for l in open(P / f"data/processed/{atoms_file}"):
        d = json.loads(l); atoms_by_stream[d["stream_id"]] = d.get("atoms", [])
    # Build event_id → actor lookup via streams file
    eid_actor = {}
    for l in open(P / f"data/processed/{streams_file}"):
        s = json.loads(l)
        for e in s.get("events", []):
            eid_actor[e.get("event_id")] = (e.get("actor") or e.get("role") or e.get("kind") or "?")
    candidates = []
    for sid, atoms in atoms_by_stream.items():
        by_key = defaultdict(list)
        for a in atoms: by_key[(a.get("entity"), a.get("attribute"))].append(a)
        for key, group in by_key.items():
            if len(group) < 2: continue
            g = sorted(group, key=lambda a: a.get("event_index", 0))
            vals = [str(a.get("value", "")).strip().lower() for a in g]
            actors = [eid_actor.get(a.get("event_id"), "?") for a in g]
            for i in range(len(g) - 1):
                for j in range(i + 1, len(g)):
                    if vals[i] == vals[j]: continue
                    if actors[i] == actors[j]: continue
                    later = g[j+1:]
                    later_vals = [str(a.get("value","")).strip().lower() for a in later]
                    if any(v == vals[i] or v == vals[j] for v in later_vals): continue
                    candidates.append({
                        "stream_id": sid, "key": key,
                        "pair": [(g[i]["event_id"], vals[i], actors[i]),
                                  (g[j]["event_id"], vals[j], actors[j])],
                    })
                    break
                else: continue
                break
    return candidates


def detect_refines(atoms_file):
    atoms_by_stream = {}
    for l in open(P / f"data/processed/{atoms_file}"):
        d = json.loads(l); atoms_by_stream[d["stream_id"]] = d.get("atoms", [])
    candidates = []
    for sid, atoms in atoms_by_stream.items():
        by_key = defaultdict(list)
        for a in atoms: by_key[(a.get("entity"), a.get("attribute"))].append(a)
        for key, group in by_key.items():
            if len(group) < 2: continue
            g = sorted(group, key=lambda a: a.get("event_index", 0))
            vals = [str(a.get("value", "")).strip().lower() for a in g]
            # find pair i < j where vals[j] strictly contains vals[i] AND vals[j] much longer
            for i in range(len(g) - 1):
                for j in range(i + 1, len(g)):
                    vi, vj = vals[i], vals[j]
                    if vi and vj and vi != vj and vi in vj and len(vj) >= len(vi) + 4:
                        # check no later flip-back: no k>j with vals[k] == vi
                        if any(vals[k] == vi for k in range(j + 1, len(g))): continue
                        candidates.append({
                            "stream_id": sid, "key": key,
                            "pair": [(g[i]["event_id"], vi), (g[j]["event_id"], vj)],
                        })
                        break
                else: continue
                break
    return candidates


def build_csc_workspec(atoms_file, streams_file, tag, out_path):
    cands = detect_csc(atoms_file, streams_file)
    print(f"  {tag} CSC: {len(cands)} candidates")
    streams_map = {}
    for l in open(P / f"data/processed/{streams_file}"):
        s = json.loads(l); streams_map[s.get("stream_id") or s.get("id")] = s
    out = open(out_path, "w")
    written = 0
    for c in cands:
        s = streams_map.get(c["stream_id"])
        if not s: continue
        ent, attr = c["key"]
        focus_ids = [t[0] for t in c["pair"]]
        user_msg = (
            f"Candidate cross_source_conflict pattern:\n"
            f"  Entity: {ent}\n  Attribute: {attr}\n"
            f"  Conflicting atoms from different sources:\n"
            f"    eid={c['pair'][0][0]}  value={c['pair'][0][1]!r}  actor={c['pair'][0][2]}\n"
            f"    eid={c['pair'][1][0]}  value={c['pair'][1][1]!r}  actor={c['pair'][1][2]}\n\n"
            f"Surrounding event stream (★ = focus events):\n"
            f"{fmt_events(s.get('events', []), focus_ids)}\n\n"
            f"Verify and produce clean QA. Output JSON."
        )
        out.write(json.dumps({
            "task_type": "csc_verify", "task_subtype": tag,
            "work_id": f"csc_verify::{tag}::{c['stream_id']}::{ent}::{attr}",
            "meta": {"stream_id": c["stream_id"], "dataset": tag,
                     "entity": ent, "attribute": attr, "triple_ids": focus_ids},
            "messages": [{"role": "system", "content": CSC_SYS},
                         {"role": "user", "content": user_msg}],
        }) + "\n")
        written += 1
    out.close()
    print(f"  {tag} CSC: wrote {written} → {out_path.name}")


def build_refines_workspec(atoms_file, streams_file, tag, out_path):
    cands = detect_refines(atoms_file)
    print(f"  {tag} refines: {len(cands)} candidates")
    streams_map = {}
    for l in open(P / f"data/processed/{streams_file}"):
        s = json.loads(l); streams_map[s.get("stream_id") or s.get("id")] = s
    out = open(out_path, "w")
    written = 0
    for c in cands:
        s = streams_map.get(c["stream_id"])
        if not s: continue
        ent, attr = c["key"]
        focus_ids = [t[0] for t in c["pair"]]
        user_msg = (
            f"Candidate refines pattern:\n"
            f"  Entity: {ent}\n  Attribute: {attr}\n"
            f"  Pair (earlier coarser → later finer):\n"
            f"    eid={c['pair'][0][0]}  value={c['pair'][0][1]!r}\n"
            f"    eid={c['pair'][1][0]}  value={c['pair'][1][1]!r}\n\n"
            f"Surrounding event stream (★ = focus events):\n"
            f"{fmt_events(s.get('events', []), focus_ids)}\n\n"
            f"Verify and produce clean QA. Output JSON."
        )
        out.write(json.dumps({
            "task_type": "refines_verify", "task_subtype": tag,
            "work_id": f"refines_verify::{tag}::{c['stream_id']}::{ent}::{attr}",
            "meta": {"stream_id": c["stream_id"], "dataset": tag,
                     "entity": ent, "attribute": attr, "triple_ids": focus_ids},
            "messages": [{"role": "system", "content": REFINES_SYS},
                         {"role": "user", "content": user_msg}],
        }) + "\n")
        written += 1
    out.close()
    print(f"  {tag} refines: wrote {written} → {out_path.name}")


def main():
    datasets = [
        ("atoms.pilot.jsonl", "streams.jsonl", "pilot"),
        ("atoms.multi_repo.jsonl", "streams_multi_repo.jsonl", "multi_repo"),
        ("atoms.wiki_v4_closed.jsonl", "streams_wiki_v4.jsonl", "wiki_v4"),
        ("atoms.dyknow.jsonl", "streams_dyknow.jsonl", "dyknow"),
    ]
    print("=== CSC mining ===")
    for af, sf, tag in datasets:
        out = P / f"data/processed/api_work_spec_csc_mine_{tag}.jsonl"
        build_csc_workspec(af, sf, tag, out)
    print("\n=== refines mining ===")
    for af, sf, tag in datasets:
        out = P / f"data/processed/api_work_spec_refines_mine_{tag}.jsonl"
        build_refines_workspec(af, sf, tag, out)


if __name__ == "__main__":
    main()
