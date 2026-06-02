"""Full Option E: replace all Llama-8B-proposed QAs with DeepSeek-V4-flash proposals
import os
using Llama's original ESR-Bench construction prompt (verified 0% abstain on probe).

Pipeline (per dataset):
  1. For each stream, build_chat from gold_slice_b.py (REUSE Llama's prompt verbatim).
  2. Run DS twice (seed=1 and seed=7) — both non-thinking — to enable 2-pass agreement filter.
  3. Parse JSON array of QAs from each pass.
  4. Agreement filter: keep QAs where the two passes propose semantically-equivalent
     gold_answer on the same/similar question and same phenomenon class. (We use a
     simple lexical-overlap test mirroring the spirit of the original 2-pass filter.)
  5. Validate event_ids are real (in the stream).
  6. Write to data/processed/slice_b_validated.<dataset>.jsonl, OVERWRITING.

Provenance metadata in each kept QA:
  qid          : "<stream_id>:dsv4_<idx>"
  slice        : "B_dsv4_propose"
  agreement_status : "ds_v4_2pass_agreed"
  question, gold_answer, gold_supporting_event_ids, gold_deprecated_event_ids
  phenomenon   : as proposed (will be re-labelled by phenv3 in a separate pass)
  phenomenon_source : "ds_v4_flash_initial_propose"
"""
import json
import sys
import re
from pathlib import Path
from collections import defaultdict

P = Path(os.environ.get("ESR_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(P / "code"))
from gold_slice_b import build_chat, PHENOMENA


DS_KEY = os.environ.get("DS_API_KEY", "")


def build_workspecs():
    """Generate 2-pass work-specs for each dataset."""
    datasets = [
        ("streams.jsonl", "atoms.pilot.jsonl", "pilot"),
        ("streams_multi_repo.jsonl", "atoms.multi_repo.jsonl", "multi_repo"),
        ("streams_wiki_v4.jsonl", "atoms.wiki_v4_closed.jsonl", "wiki_v4"),
        ("streams_dyknow.jsonl", "atoms.dyknow.jsonl", "dyknow"),
    ]
    for streams_file, atoms_file, tag in datasets:
        streams = {}
        for l in open(P / "data/processed" / streams_file):
            s = json.loads(l); streams[s.get("stream_id") or s.get("id")] = s
        atoms_idx = {}
        ap = P / "data/processed" / atoms_file
        if ap.exists():
            for l in open(ap):
                d = json.loads(l); atoms_idx[d["stream_id"]] = d.get("atoms", [])
        for pass_idx, seed in enumerate([1, 7]):
            out = open(P / f"data/processed/api_work_spec_dsE_{tag}_pass{pass_idx}.jsonl", "w")
            n = 0
            for sid, s in streams.items():
                atoms = atoms_idx.get(sid, [])
                chat = build_chat(s, atoms)
                # Tweak: append seed hint to user message to diversify the two passes
                if pass_idx == 1:
                    chat[-1]["content"] += "\n\nReasoning seed: alternative perspective."
                out.write(json.dumps({
                    "task_type": "ds_propose_qa",
                    "task_subtype": tag,
                    "work_id": f"dsE::{tag}::pass{pass_idx}::{sid}",
                    "meta": {"stream_id": sid, "dataset": tag, "pass": pass_idx, "seed": seed},
                    "messages": chat,
                }) + "\n")
                n += 1
            out.close()
            print(f"  {tag} pass{pass_idx}: wrote {n} → api_work_spec_dsE_{tag}_pass{pass_idx}.jsonl")


def parse_qas(content, valid_event_ids):
    """Parse the DS response (json_object mode) → list of QAs."""
    try:
        obj = json.loads(content)
    except Exception:
        return []
    if isinstance(obj, list): qas = obj
    elif isinstance(obj, dict):
        for k in ["qas", "questions", "items", "result", "data"]:
            if k in obj and isinstance(obj[k], list): qas = obj[k]; break
        else: qas = [obj] if "gold_answer" in obj else []
    else: qas = []
    valid = set(valid_event_ids)
    out = []
    for qa in qas:
        if not isinstance(qa, dict): continue
        gq = (qa.get("question") or "").strip()
        gg = (qa.get("gold_answer") or "").strip()
        if not (gq and gg): continue
        sup = [e for e in (qa.get("gold_supporting_event_ids") or []) if e in valid]
        dep = [e for e in (qa.get("gold_deprecated_event_ids") or []) if e in valid]
        out.append({
            "question": gq, "gold_answer": gg,
            "gold_supporting_event_ids": sup,
            "gold_deprecated_event_ids": dep,
            "phenomenon": qa.get("phenomenon") if qa.get("phenomenon") in PHENOMENA else "monotonic",
            "confidence": qa.get("confidence", "medium"),
            "rationale": (qa.get("rationale") or "")[:300],
        })
    return out


def norm_text(s):
    return re.sub(r"[\W_]+", " ", str(s or "").lower()).strip()


def jaccard(a, b):
    sa = set(norm_text(a).split()); sb = set(norm_text(b).split())
    return len(sa & sb) / max(len(sa | sb), 1)


def agreement_filter(qas_p0, qas_p1, jaccard_thresh=0.5):
    """Keep p0 QAs that have a counterpart in p1 with (a) similar question (jaccard >= thresh)
    AND (b) semantically-equivalent gold_answer (jaccard >= thresh OR identical normalized)."""
    kept = []
    used_p1 = set()
    for q0 in qas_p0:
        for i, q1 in enumerate(qas_p1):
            if i in used_p1: continue
            qj = jaccard(q0["question"], q1["question"])
            gj = jaccard(q0["gold_answer"], q1["gold_answer"])
            # ABSTAIN must match ABSTAIN; otherwise allow gold disagreement if questions match closely
            ngj = norm_text(q0["gold_answer"]) == norm_text(q1["gold_answer"])
            if qj >= jaccard_thresh and (ngj or gj >= jaccard_thresh):
                kept.append(q0)
                used_p1.add(i)
                break
    return kept


def build_datasets():
    """After both passes are done, apply agreement filter + write final dataset files."""
    streams_for = {
        "pilot": "streams.jsonl",
        "multi_repo": "streams_multi_repo.jsonl",
        "wiki_v4": "streams_wiki_v4.jsonl",
        "dyknow": "streams_dyknow.jsonl",
    }
    # Also preserve existing mined items (DS-generated, already in current datasets)
    for tag, streams_file in streams_for.items():
        streams = {}
        for l in open(P / "data/processed" / streams_file):
            s = json.loads(l); streams[s.get("stream_id") or s.get("id")] = s

        # Load existing mined items (slice startswith "B_mined_")
        cur_path = P / "data/processed" / f"slice_b_validated.{tag}.jsonl"
        existing_mined = []
        if cur_path.exists():
            for l in open(cur_path):
                q = json.loads(l)
                if str(q.get("slice", "")).startswith("B_mined_"):
                    existing_mined.append(q)

        # Load DS two-pass responses
        by_stream = defaultdict(lambda: {0: [], 1: []})
        for pass_idx in [0, 1]:
            fn = P / f"results/api_work_spec_dsE_{tag}_pass{pass_idx}.jsonl"
            if not fn.exists(): continue
            for l in open(fn):
                r = json.loads(l); m = r["meta"]
                sid = m["stream_id"]
                content = r.get("response_content") or ""
                if not content: continue
                events = streams.get(sid, {}).get("events") or []
                valid_eids = [e.get("event_id") for e in events]
                by_stream[sid][pass_idx] = parse_qas(content, valid_eids)

        # Agreement filter
        new_qas = []
        stream_counter = defaultdict(int)
        n_p0 = n_p1 = n_kept = 0
        for sid, passes in by_stream.items():
            p0, p1 = passes[0], passes[1]
            n_p0 += len(p0); n_p1 += len(p1)
            kept = agreement_filter(p0, p1)
            n_kept += len(kept)
            for qa in kept:
                stream_counter[sid] += 1
                idx = stream_counter[sid]
                qid = f"{sid}:dsv4_{idx}"
                new_qas.append({
                    "qid": qid, "stream_id": sid,
                    "slice": "B_dsv4_propose", "agreement_status": "ds_v4_2pass_agreed",
                    "question": qa["question"],
                    "gold_answer": qa["gold_answer"],
                    "gold_supporting_event_ids": qa["gold_supporting_event_ids"],
                    "gold_deprecated_event_ids": qa["gold_deprecated_event_ids"],
                    "phenomenon": qa["phenomenon"],
                    "phenomenon_source": "ds_v4_flash_initial_propose",
                    "confidence": qa["confidence"], "rationale": qa["rationale"],
                })
        print(f"  {tag}: pass0_n={n_p0}, pass1_n={n_p1}, kept after agreement={n_kept}; existing mined={len(existing_mined)}")

        # Write the new dataset: agreement-filtered DS proposals + existing DS-mined items
        out_path = cur_path
        with open(out_path, "w") as f:
            for q in new_qas: f.write(json.dumps(q, ensure_ascii=False) + "\n")
            for q in existing_mined: f.write(json.dumps(q, ensure_ascii=False) + "\n")
        print(f"  {tag}: wrote {len(new_qas) + len(existing_mined)} total → {out_path.name}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["build_workspec", "build_datasets"], required=True)
    args = ap.parse_args()
    if args.stage == "build_workspec":
        build_workspecs()
    elif args.stage == "build_datasets":
        build_datasets()
