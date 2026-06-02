import os
"""Rule-detect strict flip-flop patterns across all 4 datasets, then build
DeepSeek workspecs to (a) verify each candidate is a real strict rev_rev and
(b) generate a clean QA for it.

Pattern: for each (entity, attribute) group with ≥3 atoms, find i<j<k with
vals[i]≠vals[j] AND vals[k]==vals[i] (or substring match). This is the
operational form of the paper's "e_1 = A, e_2 contradicts A, e_3 supersedes e_2,
final ≈ A" definition.
"""
import json
from collections import defaultdict
from pathlib import Path

P = Path(os.environ.get("ESR_ROOT", Path(__file__).resolve().parents[1]))


VERIFY_SYS = """You verify and clean up strict reverted_revert (flip-flop) QA candidates.

A strict flip-flop has THREE events with the same entity-attribute key:
- Early event e_A1 states answer A
- Middle event e_B contradicts A (states ¬A or B that conflicts with A)
- Later event e_A2 restores A or a close variant of A (overrides e_B)

The candidate is given as a triple (atom@e_A1=A, atom@e_B=B, atom@e_A2=A'). You must check whether the surrounding event texts actually support this flip-flop:
  - Are e_A1 and e_A2 really claiming the same thing (A ≈ A')?
  - Does e_B really contradict A?
  - Is e_A2 the LATEST event for this key (no later event undoes it)?

Output JSON exactly:
{
  "is_strict_flip_flop": true | false,
  "rejection_reason": "<short reason if false>" or null,
  "question": "<natural-language question asking for the current value, e.g. 'What is the current CEO of X?' — required if true>",
  "gold_answer": "<short final answer, ideally <= 10 words — required if true>",
  "supporting_event_ids": [<event_ids of e_A1 and e_A2 that establish the surviving value>],
  "deprecated_event_ids": [<event_id of e_B that was reverted>]
}

JSON only, no prose."""


def fmt_events(events, focus_ids, cap=15):
    """Show focus events first, then surrounding context."""
    focus = [e for e in events if e.get("event_id") in focus_ids]
    rest = [e for e in events if e.get("event_id") not in focus_ids][:cap - len(focus)]
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


def detect(atoms_file):
    atoms_by_stream = {}
    for l in open(P / f"data/processed/{atoms_file}"):
        d = json.loads(l); atoms_by_stream[d["stream_id"]] = d.get("atoms", [])
    flip = []
    for sid, atoms in atoms_by_stream.items():
        by_key = defaultdict(list)
        for a in atoms: by_key[(a.get("entity"), a.get("attribute"))].append(a)
        for key, group in by_key.items():
            if len(group) < 3: continue
            g = sorted(group, key=lambda a: a.get("event_index", 0))
            vals = [str(a.get("value", "")).strip().lower() for a in g]
            ids = [a.get("event_id") for a in g]
            n = len(g)
            found = False
            for i in range(n - 2):
                for j in range(i + 1, n - 1):
                    if vals[j] == vals[i]: continue
                    for k in range(j + 1, n):
                        if vals[k] == vals[i] or (
                            vals[i] and vals[k] and (vals[i] in vals[k] or vals[k] in vals[i])
                        ):
                            flip.append({
                                "stream_id": sid,
                                "key": key,
                                "triple": [(ids[i], vals[i]), (ids[j], vals[j]), (ids[k], vals[k])],
                                "all_atoms": g,
                            })
                            found = True; break
                    if found: break
                if found: break
    return flip


def build_workspec(atoms_file, streams_file, tag, out_path):
    flip = detect(atoms_file)
    print(f"  {tag}: {len(flip)} candidate patterns")
    streams_map = {}
    for l in open(P / f"data/processed/{streams_file}"):
        s = json.loads(l); streams_map[s.get("stream_id") or s.get("id")] = s
    out = open(out_path, "w")
    written = 0
    for f in flip:
        s = streams_map.get(f["stream_id"])
        if not s: continue
        ent, attr = f["key"]
        triple = f["triple"]
        focus_ids = [tr[0] for tr in triple]
        user_msg = (
            f"Candidate strict flip-flop pattern:\n"
            f"  Entity: {ent}\n"
            f"  Attribute: {attr}\n"
            f"  Time-ordered atom values:\n"
            f"    e_A1 = {triple[0][0]} : value={triple[0][1]!r}\n"
            f"    e_B  = {triple[1][0]} : value={triple[1][1]!r}\n"
            f"    e_A2 = {triple[2][0]} : value={triple[2][1]!r}\n\n"
            f"Surrounding event stream (★ = the three focus events):\n"
            f"{fmt_events(s.get('events', []), focus_ids)}\n\n"
            f"Verify and produce clean QA. Output JSON."
        )
        out.write(json.dumps({
            "task_type": "strict_revrev_verify",
            "task_subtype": tag,
            "work_id": f"verify::{tag}::{f['stream_id']}::{ent}::{attr}",
            "meta": {"stream_id": f["stream_id"], "dataset": tag,
                     "entity": ent, "attribute": attr,
                     "triple_ids": focus_ids},
            "messages": [{"role": "system", "content": VERIFY_SYS},
                         {"role": "user", "content": user_msg}],
        }) + "\n")
        written += 1
    out.close()
    print(f"  {tag}: wrote {written} verify-work-items → {out_path.name}")


def main():
    print("=== building strict-rev_rev verify workspecs ===")
    for af, sf, tag in [
        ("atoms.pilot.jsonl", "streams.jsonl", "pilot"),
        ("atoms.multi_repo.jsonl", "streams_multi_repo.jsonl", "multi_repo"),
        ("atoms.wiki_v4_closed.jsonl", "streams_wiki_v4.jsonl", "wiki_v4"),
        ("atoms.dyknow.jsonl", "streams_dyknow.jsonl", "dyknow"),
    ]:
        out = P / f"data/processed/api_work_spec_revrev_mine_{tag}.jsonl"
        build_workspec(af, sf, tag, out)


if __name__ == "__main__":
    main()
