"""Slice-B (content-grounded) gold proposer.

Given (stream, atoms), the LLM proposes:
  - questions: 2-3 content-grounded questions
  - per question: gold_answer (or "ABSTAIN"), gold_supporting_event_ids, gold_deprecated_event_ids
  - phenomenon ∈ {monotonic, refines, reverted_revert, low_credibility_latest, cross_source_conflict}
  - confidence

We run TWO different settings (temperature 0.0 and 0.7) and KEEP only items where the two
agree on (phenomenon, gold_answer string-normalized) — the "agreement filter" from
specs/gold-construction-spec.md §3.

Outputs `data/processed/slice_b_proposals.jsonl` (raw both-pass output) and
`data/processed/slice_b_validated.jsonl` (agreed subset).
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

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger("goldB")

PHENOMENA = ["monotonic", "refines", "reverted_revert",
             "low_credibility_latest", "cross_source_conflict"]

PROMPT_GITHUB = """You construct EVALUATION QA items for the Evidence-State Revision (ESR) benchmark.

You will see a chronological GitHub issue thread plus pre-extracted atoms. Propose 2-3
content-grounded questions about the issue's CURRENT EFFECTIVE STATE.

CRITICAL CONSTRAINTS — violations DISQUALIFY the question:
1. NO metadata questions. NEVER ask about: issue status (open/closed/locked/blocked),
   labels, assignee, milestone, draft, lock state. The benchmark's Slice A already
   covers these — do not duplicate them here. Questions whose answer comes from the
   issue's metadata fields are REJECTED.
2. Each gold_supporting_event_ids and gold_deprecated_event_ids entry MUST be a real
   event_id from the provided event list. NEVER invent event_ids. NEVER use comment
   indices, dates, or anything else.
3. The answer must come from the BODY TEXT of the events — not from labels.

PREFERRED QUESTION TYPES (these probe non-monotonic reasoning):
- "What is the latest valid reproduction condition?"
- "Which earlier claimed cause / workaround / fix has been superseded?"
- "Has the originally proposed workaround been replaced? If so, by what?"
- "Has any earlier claim been contradicted and then re-instated?"
- "Among conflicting comments, which represents the current valid state — or is it unresolved?"

Phenomenon classification (be honest — do not over-label as `monotonic`):
- monotonic: the latest comment alone gives the answer; no earlier event was modified.
- refines: a later comment narrows an earlier claim without invalidating it.
- reverted_revert: a claim was made, contradicted, then reinstated.
- low_credibility_latest: latest commenter contradicts an earlier maintainer; gold should
  trust the maintainer.
- cross_source_conflict: two equally-credible sources disagree; no later event resolves.
  Gold answer = "ABSTAIN".

For each question output a JSON object with EXACTLY these keys:
  - "question": string
  - "gold_answer": short string, or the literal "ABSTAIN" for cross_source_conflict
  - "gold_supporting_event_ids": list of REAL event_ids
  - "gold_deprecated_event_ids": list of REAL event_ids whose claims are superseded; may be []
  - "phenomenon": one of {phenomena}
  - "confidence": "high" | "medium" | "low"
  - "rationale": ≤2 sentences

PRIORITY: produce non-monotonic questions when the thread supports it. If the thread is
purely monotonic (no supersession / no contradiction / no refinement anywhere), return
just ONE question. If the thread shows clear non-monotonic structure, prefer NM questions.

Output ONLY the JSON array, no prose, no markdown fences.
"""


PROMPT_WIKIPEDIA = """You construct EVALUATION QA items for the Evidence-State Revision (ESR) benchmark on Wikipedia.

You will see a chronological sequence of Wikipedia article revisions for one page,
plus pre-extracted atoms. Propose 1-3 content-grounded questions about the
CURRENT EFFECTIVE STATE of a fact on the page (current CEO, current version, current
office holder, current status, current member set, etc.).

CRITICAL CONSTRAINTS:
1. The question must be about a CONTENT FACT in the article body, not about
   metadata of the revision itself (do NOT ask "who edited last?", "when was it
   reverted?", or "what was the edit comment?").
2. Each gold_supporting_event_ids / gold_deprecated_event_ids entry MUST be a real
   event_id from the provided event list. NEVER invent event_ids.
3. Each question must have an answer that depends on which revision was applied
   most recently — i.e. the question is meaningful BECAUSE the article changed.

PREFERRED QUESTION SHAPES (Wikipedia-flavoured):
- "Who is the current CEO of the page subject?"
- "What is the current stable version of the page subject?"
- "Who is the current Prime Minister / President / leader / manager of the page subject?"
- "Was an earlier value (e.g. an earlier office-holder, earlier version) superseded? If so by what?"
- "Has any earlier claim about the page subject been removed and then restored?"
- "Are two revisions in conflict about the page subject, with no later revision resolving it?"

Phenomenon classification (be honest — do NOT over-label as `monotonic`):
- monotonic: the latest revision states the answer directly; earlier revisions had different/no value but no contradiction.
- refines: a later revision narrows an earlier claim (e.g. specifies year of tenure).
- reverted_revert: a value was set, removed/changed, then re-instated.
- low_credibility_latest: latest editor is anonymous/IP user contradicting an earlier sourced edit; trust the sourced one.
- cross_source_conflict: two well-sourced revisions disagree, no later revision resolves. Gold = "ABSTAIN".

For each question output JSON object with EXACTLY these keys:
  - "question": string
  - "gold_answer": short string, or "ABSTAIN" for cross_source_conflict
  - "gold_supporting_event_ids": list of REAL event_ids
  - "gold_deprecated_event_ids": list of REAL event_ids whose claims are superseded; may be []
  - "phenomenon": one of {phenomena}
  - "confidence": "high" | "medium" | "low"
  - "rationale": ≤2 sentences

PRIORITY: produce questions that EXERCISE the revision history. If the page is
purely additive (every revision just adds info, never changes the answer), return ONE monotonic question only.

Output ONLY the JSON array, no prose, no markdown fences.
"""


# Forbidden question patterns (post-LLM reject filter, GitHub-specific)
import re as _re
FORBIDDEN_QUESTION = _re.compile(
    r"\b(open|closed|locked|blocked|state|status|label(?:s|ed)?|milestone|"
    r"assigned|assignee|draft|merged)\b",
    _re.IGNORECASE,
)

# Wiki-specific forbidden: questions about edit metadata
FORBIDDEN_QUESTION_WIKI = _re.compile(
    r"\b(reverted|edit comment|edit summary|who (?:edited|edits)|when was|"
    r"latest valid reproduction|workaround|superseded fix)\b",
    _re.IGNORECASE,
)


def build_chat(stream: dict, atoms: list[dict]) -> list[dict]:
    is_wiki = stream.get("repo") == "wikipedia"
    if is_wiki:
        sys_msg = PROMPT_WIKIPEDIA.format(phenomena=", ".join(PHENOMENA))
    else:
        sys_msg = PROMPT_GITHUB.format(phenomena=", ".join(PHENOMENA))
    # Compact the events for the model. Adaptive truncation: with many events, shorten each.
    n_events = len(stream["events"])
    per_event_chars = 600 if n_events <= 12 else (300 if n_events <= 25 else 200)
    ev_text = []
    valid_event_ids = []
    for i, e in enumerate(stream["events"]):
        text = (e.get("text") or "").strip().replace("\n", " ")
        text = text[:per_event_chars]
        ev_text.append(f"[{i:02d} {e['kind']} @ {e['timestamp']} (id={e['event_id']})] {text}")
        valid_event_ids.append(e["event_id"])
    atom_text = []
    for a in atoms[:30]:
        atom_text.append(
            f"  - atom={a['atom_id']} ev={a['event_id']} attr={a['attribute']} "
            f"val={(a.get('value') or '')[:100]} pol={a['polarity']}"
        )
    if is_wiki:
        # Wikipedia: include the article title so the model can pin questions to the entity
        entity_id = stream.get("title") or stream.get("issue_number")
        attribute_hint = stream.get("wiki_attribute_hint") or ""
        user_msg = (
            f"page: {entity_id}\n"
            f"target_attribute_hint: {attribute_hint}\n"
            f"valid event_ids (use these EXACTLY): {valid_event_ids}\n\n"
            f"revisions (chronological, with comment + content):\n" + "\n".join(ev_text) + "\n\n"
            f"pre-extracted atoms:\n" + "\n".join(atom_text) + "\n\n"
            f"Output 1-3 content-grounded QA items about the CURRENT VALUE of a fact "
            f"about {entity_id} (e.g. {attribute_hint}). PREFER non-monotonic items when "
            f"the revision history shows supersession / refinement / restoration. "
            f"NEVER ask about edit metadata or revert events themselves."
        )
    else:
        user_msg = (
            f"issue_id: {stream['repo']}#{stream['issue_number']}\n"
            f"valid event_ids (use these EXACTLY): {valid_event_ids}\n\n"
            f"events (chronological, with text):\n" + "\n".join(ev_text) + "\n\n"
            f"pre-extracted atoms:\n" + "\n".join(atom_text) + "\n\n"
            "Output up to 3 content-grounded QA items as a JSON array. PREFER non-monotonic "
            "items when the thread shows supersession / refinement / contradiction. NEVER ask "
            "about issue status / labels / assignee / milestone."
        )
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]


def is_clean_qa(q: dict, valid_event_ids: set[str], domain: str = "github") -> tuple[bool, str | None]:
    """Validate a single QA dict; return (ok, reason_if_rejected). domain ∈ {github, wikipedia}."""
    if not isinstance(q, dict):
        return False, "not_dict"
    qtext = (q.get("question") or "").strip()
    if not qtext:
        return False, "empty_question"
    if domain == "github" and FORBIDDEN_QUESTION.search(qtext):
        return False, "metadata_question"
    if domain == "wikipedia" and FORBIDDEN_QUESTION_WIKI.search(qtext):
        return False, "wiki_metadata_question"
    ans = q.get("gold_answer")
    if ans is None or (isinstance(ans, str) and not ans.strip()):
        return False, "empty_answer"
    sup = q.get("gold_supporting_event_ids") or []
    dep = q.get("gold_deprecated_event_ids") or []
    if not isinstance(sup, list) or not isinstance(dep, list):
        return False, "non_list_ids"
    bad = [e for e in (list(sup) + list(dep)) if e not in valid_event_ids]
    if bad:
        return False, f"hallucinated_event_ids:{bad[:2]}"
    if set(sup) & set(dep):
        return False, "supporting_deprecated_overlap"
    if q.get("phenomenon") not in PHENOMENA:
        return False, f"bad_phenomenon:{q.get('phenomenon')}"
    return True, None


def normalize_answer(a: str) -> str:
    if a is None:
        return ""
    return " ".join(str(a).strip().lower().split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", default=str(PROC_DIR / "streams.jsonl"))
    ap.add_argument("--atoms", default=str(PROC_DIR / "atoms.jsonl"))
    ap.add_argument("--out", default=str(PROC_DIR / "slice_b_proposals.jsonl"))
    ap.add_argument("--out-validated", default=str(PROC_DIR / "slice_b_validated.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-tokens", type=int, default=900)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from llm_runner import LLMRunner

    runner = LLMRunner(model=args.model) if args.model else LLMRunner()

    atom_index: dict[str, list[dict]] = {}
    with open(args.atoms) as f:
        for line in f:
            d = json.loads(line)
            atom_index[d["stream_id"]] = d.get("atoms", [])
    log.info(f"atoms for {len(atom_index)} streams loaded")

    streams = []
    with open(args.streams) as f:
        for i, line in enumerate(f):
            if i % args.num_shards != args.shard:
                continue
            streams.append(json.loads(line))
    if args.limit:
        streams = streams[: args.limit]
    log.info(f"shard {args.shard}/{args.num_shards}: {len(streams)} streams")

    out_path = Path(args.out)
    if args.num_shards > 1:
        out_path = out_path.with_suffix(f".s{args.shard}of{args.num_shards}.jsonl")

    chats = [build_chat(s, atom_index.get(s["stream_id"], [])) for s in streams]

    t0 = time.time()
    log.info("running pass A (temp=0.0, seed=1)")
    pass_a = runner.chat(chats, max_tokens=args.max_tokens, temperature=0.0, seed=1)
    log.info(f"pass A done in {time.time()-t0:.0f}s")
    t1 = time.time()
    log.info("running pass B (temp=0.7, seed=7)")
    pass_b = runner.chat(chats, max_tokens=args.max_tokens, temperature=0.7, top_p=0.95, seed=7)
    log.info(f"pass B done in {time.time()-t1:.0f}s")

    proposals = []
    validated = []
    for s, a_text, b_text in zip(streams, pass_a, pass_b):
        a_qas = LLMRunner.parse_json(a_text) or []
        b_qas = LLMRunner.parse_json(b_text) or []
        if not isinstance(a_qas, list):
            a_qas = []
        if not isinstance(b_qas, list):
            b_qas = []

        rec = {"stream_id": s["stream_id"], "pass_a": a_qas, "pass_b": b_qas}
        proposals.append(rec)

        # Agreement filter: a question from pass A is "validated" if pass B has a question
        # whose normalized answer + phenomenon matches.
        b_index = {(normalize_answer(q.get("gold_answer", "")), q.get("phenomenon")): q
                   for q in b_qas if isinstance(q, dict)}
        for q in a_qas:
            if not isinstance(q, dict):
                continue
            key = (normalize_answer(q.get("gold_answer", "")), q.get("phenomenon"))
            if key in b_index and key[0] != "":
                qid = f"{s['stream_id']}:b{len([v for v in validated if v['stream_id']==s['stream_id']])}"
                validated.append({
                    "qid": qid,
                    "stream_id": s["stream_id"],
                    "slice": "B",
                    "agreement_status": "agree",
                    "question": q.get("question"),
                    "gold_answer": q.get("gold_answer"),
                    "gold_supporting_event_ids": q.get("gold_supporting_event_ids", []),
                    "gold_deprecated_event_ids": q.get("gold_deprecated_event_ids", []),
                    "phenomenon": q.get("phenomenon"),
                    "confidence": q.get("confidence"),
                    "rationale": q.get("rationale"),
                })

    with open(out_path, "w") as f:
        for r in proposals:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"wrote {out_path} ({len(proposals)} proposal records)")

    out_v = Path(args.out_validated)
    if args.num_shards > 1:
        out_v = out_v.with_suffix(f".s{args.shard}of{args.num_shards}.jsonl")
    with open(out_v, "w") as f:
        for v in validated:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    log.info(f"wrote {out_v} ({len(validated)} validated QAs)")

    # Phenomenon distribution in validated set
    from collections import Counter
    counts = Counter(v["phenomenon"] for v in validated)
    log.info(f"phenomenon distribution: {dict(counts)}")


if __name__ == "__main__":
    main()
