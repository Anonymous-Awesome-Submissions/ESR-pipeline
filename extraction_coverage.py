"""Proxy precision/recall-style coverage diagnostics for the atom extractor + co-keyer,
using the benchmark's own gold supporting/deprecated event ids (no human labels).

Two questions a reviewer asks ("how good is the extractor, and how sensitive are the
gains to extraction errors?"), answered with what's available:

  (R1) EVENT RECALL  — of the gold-relevant events (supporting ∪ deprecated) in a QA,
       what fraction produced >=1 atom? (an event with no atom can never enter the ledger)
  (R2) CO-KEY RECALL — among QAs that have BOTH a gold supporting and a gold deprecated
       event AND both produced atoms, what fraction land on a *shared* (entity,attribute)
       key? (supersession can only fire if the new and old value sit on the same ledger key;
       this is the single most load-bearing co-keying behaviour)
  (D)  density — atoms/event, distinct keys/stream (sanity that the extractor isn't
       collapsing everything onto one key, which would inflate R2 trivially).

Run: python code/extraction_coverage.py            # prints a table
     python code/extraction_coverage.py --json     # writes results/extraction_coverage.json
"""
import json, sys
from collections import defaultdict, Counter
from pathlib import Path

P = Path(__file__).resolve().parents[1]


def load_atoms(fn):
    by_stream = {}
    for l in open(P / "data/processed" / fn):
        d = json.loads(l)
        by_stream[d["stream_id"]] = d.get("atoms", [])
    return by_stream


def load_qas(fn):
    return [json.loads(l) for l in open(P / "data/processed" / fn)]


def coverage(qa_fn, atom_fn, name):
    qas = load_qas(qa_fn)
    atoms = load_atoms(atom_fn)
    # event -> set of (entity,attribute) keys it produced, per stream
    ev_keys = defaultdict(lambda: defaultdict(set))
    n_atoms_per_ev = defaultdict(Counter)
    keys_per_stream = defaultdict(set)
    for sid, ats in atoms.items():
        for a in ats:
            ev = a.get("event_id")
            k = (a.get("entity"), a.get("attribute"))
            ev_keys[sid][ev].add(k)
            n_atoms_per_ev[sid][ev] += 1
            keys_per_stream[sid].add(k)

    # R1: event recall over gold-relevant events
    gold_ev_total = gold_ev_hit = 0
    # R2: co-key recall
    cokey_elig = cokey_hit = 0
    for q in qas:
        sid = q["stream_id"]
        sup = [e for e in (q.get("gold_supporting_event_ids") or [])]
        dep = [e for e in (q.get("gold_deprecated_event_ids") or [])]
        for e in set(sup) | set(dep):
            gold_ev_total += 1
            if e in ev_keys.get(sid, {}):
                gold_ev_hit += 1
        # co-key: need a supporting and a deprecated event, both with atoms
        sup_hit = [e for e in sup if e in ev_keys.get(sid, {})]
        dep_hit = [e for e in dep if e in ev_keys.get(sid, {})]
        if sup_hit and dep_hit:
            cokey_elig += 1
            sk = set().union(*[ev_keys[sid][e] for e in sup_hit])
            dk = set().union(*[ev_keys[sid][e] for e in dep_hit])
            if sk & dk:
                cokey_hit += 1

    # density
    apes = [c for sid in n_atoms_per_ev for c in n_atoms_per_ev[sid].values()]
    kps = [len(v) for v in keys_per_stream.values()]
    res = {
        "name": name, "n_qas": len(qas), "n_streams": len(atoms),
        "event_recall": round(gold_ev_hit / max(gold_ev_total, 1), 3),
        "event_recall_n": [gold_ev_hit, gold_ev_total],
        "cokey_recall": round(cokey_hit / max(cokey_elig, 1), 3),
        "cokey_recall_n": [cokey_hit, cokey_elig],
        "atoms_per_event_mean": round(sum(apes) / max(len(apes), 1), 2),
        "distinct_keys_per_stream_mean": round(sum(kps) / max(len(kps), 1), 2),
    }
    return res


def main():
    rows = [
        coverage("slice_b_validated.pilot.jsonl", "atoms.pilot.jsonl", "ESR-Bench-GitHub"),
        coverage("slice_b_validated.multi_repo.jsonl", "atoms.multi_repo.jsonl", "ESR-MultiRepo"),
    ]
    # wiki: atoms file name varies; try the validated wiki QA + atoms.wiki_v4_closed
    try:
        rows.append(coverage("slice_b_validated.wiki_v4.jsonl", "atoms.wiki_v4_closed.jsonl", "ESR-Wiki"))
    except FileNotFoundError:
        pass
    w = max(len(r["name"]) for r in rows)
    print(f"{'split'.ljust(w)}  event_recall   co-key_recall   atoms/ev  keys/stream")
    for r in rows:
        er = f'{r["event_recall"]:.3f} ({r["event_recall_n"][0]}/{r["event_recall_n"][1]})'
        ck = f'{r["cokey_recall"]:.3f} ({r["cokey_recall_n"][0]}/{r["cokey_recall_n"][1]})'
        print(f'{r["name"].ljust(w)}  {er:>13}   {ck:>14}   {r["atoms_per_event_mean"]:>7}   {r["distinct_keys_per_stream_mean"]:>10}')
    if "--json" in sys.argv:
        (P / "results/extraction_coverage.json").write_text(json.dumps(rows, indent=2))
        print("\nwrote results/extraction_coverage.json")


if __name__ == "__main__":
    main()
