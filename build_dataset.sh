#!/usr/bin/env bash
# ESR-Bench dataset construction — reference pipeline.
# NOT one-click: requires network (GitHub/Wikipedia), an extraction LLM, and a
# DeepSeek API key. LLM stages are non-deterministic, so this reproduces the
# CONSTRUCTION, not the exact released rows. See README.md.
set -euo pipefail
export ESR_ROOT="${ESR_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"   # repo root
export ESR_RAW_DIR="${ESR_RAW_DIR:-$ESR_ROOT/data/raw}"            # fetched raw streams
: "${DS_API_KEY:?set DS_API_KEY (DeepSeek) before running stages 3-4}"
PY="${PY:-python3}"

echo "[0] fetch raw public streams (GitHub issues + Wikipedia revisions)"
# NOTE: the GitHub side has two PARALLEL sub-pipelines, not a single chain:
#   (a) data_pipeline.py normalizes the main HuggingFace `datasets` issue split
#       (legacy `pilot` filename), which is pulled separately as an HF-mirrored
#       JSONL into $ESR_RAW_DIR/github_issues_esr/ — re-fetch instructions are in
#       data_pipeline.py's docstring.
#   (b) github_api_pull.py + multi-repo normalizer produce the `multi_repo` split
#       under $ESR_RAW_DIR/github_issues_esr_v2/ for held-out repos
#       (pytorch, tensorflow, rust-lang, huggingface/transformers).
# They feed separate downstream splits; running (a) does not depend on (b).
$PY github_api_pull.py            # multi-repo issues -> $ESR_RAW_DIR/github_issues_esr_v2/ (rate-limited; GH_TOKEN optional)
$PY wikipedia_split.py            # Wikipedia revisions -> data/processed/streams_wiki_v4.jsonl

echo "[1] normalize main HF GitHub split into evidence streams"
$PY data_pipeline.py              # reads $ESR_RAW_DIR/github_issues_esr/ -> data/processed/{issues,streams,slice_a}.jsonl

echo "[2] atom extraction (needs an extraction LLM; configure inside atom_extract.py)"
$PY atom_extract.py --streams data/processed/streams.jsonl --out data/processed/atoms.jsonl

echo "[3] QA proposal + 2-pass agreement filter (DeepSeek)"
$PY ds_propose_qa_pipeline.py     # -> data/processed/slice_b_validated.<split>.jsonl

echo "[4] mine phenomenon strata (DeepSeek verify)"
$PY mine_strict_revrev.py
$PY mine_csc_refines.py

echo "[QC] extraction coverage"
$PY extraction_coverage.py
echo "done. NOTE: regenerated labels will differ from the released set (LLM drift)."
