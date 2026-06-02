# ESR-Bench construction pipeline (reference)

Reproduces how ESR-Bench is built from public GitHub issue threads and
Wikipedia revisions. This is a reference pipeline, not a turnkey one-click
script: it needs network access, an extraction LLM, and a DeepSeek API key,
and the LLM stages are non-deterministic — it reproduces the construction
method and distribution, not the exact released rows.

## Stages (see `build_dataset.sh`)
0. `github_api_pull.py`, `wikipedia_split.py` — fetch raw public streams.
1. `data_pipeline.py` — normalize into time-ordered evidence streams (CPU, deterministic).
2. `atom_extract.py` — LLM atom extraction `(entity, attribute, value, polarity, time, ids)`.
3. `ds_propose_qa_pipeline.py` / `gold_slice_b.py` — DeepSeek proposes QAs; 2-pass agreement filter.
4. `mine_strict_revrev.py`, `mine_csc_refines.py` — mine reverted-revert / csc / refines strata.
QC: `extraction_coverage.py`.

### GitHub data has two parallel sub-pipelines (not one chain)
The GitHub side combines two upstream sources that produce different splits and
flow through different normalizers; running one does not feed the other.
- Main HF GitHub split (legacy filename `pilot`, 1{,}698 QAs — the "1,698
  GitHub" row of the paper's primary surface): pulled separately from a
  HuggingFace-mirrored JSONL of the `huggingface/datasets` issue stream into
  `$ESR_RAW_DIR/github_issues_esr/`, then normalized by `data_pipeline.py`.
  `github_api_pull.py` does not populate this directory; see the docstring of
  `data_pipeline.py` for the re-fetch source.
- Multi-repo split (`multi_repo`, 252 QAs): pulled by `github_api_pull.py` over
  the GitHub REST API into `$ESR_RAW_DIR/github_issues_esr_v2/<repo>/` for the
  held-out repos (pytorch, tensorflow, rust-lang, huggingface/transformers),
  then normalized by a separate multi-repo path (not `data_pipeline.py`).

## Requirements
- Python 3.10+, `requests`; an extraction LLM (local vLLM or API — configured in `atom_extract.py`).
- `DS_API_KEY` — DeepSeek API key (stages 3–4). `GH_TOKEN` optional (raises GitHub rate limit).

## Configuration (env vars, no hardcoded paths)
- `ESR_ROOT`  — repo root (default: parent of this dir).
- `ESR_RAW_DIR` — where fetched raw streams land (default: `$ESR_ROOT/data/raw`).

## Privacy
Fetched raw streams stay local; the public data release does not include raw
thread/revision text (see the data release's README). Re-fetch raw text only from
the `source_manifest.jsonl` URLs.

The pipeline's `data/processed` output is NOT a publishable product as-is. The
extraction stages emit `actor` and verbatim `span` fields and leave `@handle`
mentions and personal URLs in free-text, so running this pipeline reproduces
identifiable material locally. Before any redistribution you must (1) drop the
`actor` / `span` fields and (2) scrub residual `@handle` mentions, bare contributor
names, and personal URL paths from free-text and extracted fields. The accompanying
data release ships the already-minimized result and its `PRIVACY_AUDIT.md`
documents the cumulative scrub counts and the zero-scans that returned clean.
