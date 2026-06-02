"""Pull GitHub issues + comments from multiple repos via the GitHub REST API.

Strategy: if `GH_TOKEN` is set in the environment, the request is authenticated and
the per-request sleep is short (auth quota is 5000/hr); otherwise the puller falls
back to the unauthenticated 60/hr quota and sleeps ~60s between requests. Resumes
incrementally from existing output.

Output: per-repo JSONL files at
  $ESR_RAW_DIR/github_issues_esr_v2/<repo>/issues.jsonl
This is the `multi_repo` raw source; the main HuggingFace `datasets` split is
fetched separately and normalised by `data_pipeline.py`, not by this script.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

OUT_DIR = Path(os.environ.get("ESR_RAW_DIR", "data/raw")) / "github_issues_esr_v2"
# NOTE: filesystem side effects (mkdir, FileHandler) are deferred to main() so that
# `import github_api_pull` does NOT touch the disk. This matters for release hygiene:
# importing the module for type-checking or test discovery must not create empty raw/
# directories that then leak into the release archive.

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO,
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("ghpull")

REPOS = [
    "pytorch/pytorch",
    "tensorflow/tensorflow",
    "rust-lang/rust",
    "huggingface/transformers",  # similar style, but different repo
]
PER_REPO_TARGET = 30  # 30 issues per repo × 4 = 120 issues

UA = "ESR-Bench/0.1 (research)"
GH_TOKEN = os.environ.get("GH_TOKEN", "").strip()           # optional: raises quota 60/hr -> 5000/hr
PER_REQUEST_SLEEP = 1.0 if GH_TOKEN else 60.5               # adaptive to auth quota


def gh_get(url: str, attempts: int = 3) -> dict | list | None:
    headers = {
        "User-Agent": UA,
        "Accept": "application/vnd.github+json",
    }
    if GH_TOKEN:
        headers["Authorization"] = f"Bearer {GH_TOKEN}"
    for _ in range(attempts):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                rl_remain = resp.headers.get("X-RateLimit-Remaining")
                rl_reset = resp.headers.get("X-RateLimit-Reset")
                if rl_remain is not None:
                    rl_remain = int(rl_remain)
                    log.info(f"  rate-limit remaining: {rl_remain}")
                    if rl_remain <= 1 and rl_reset:
                        sleep_until = int(rl_reset) + 5
                        wait = max(1, sleep_until - int(time.time()))
                        log.info(f"  sleeping {wait}s until rate-limit reset")
                        time.sleep(wait)
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code == 403:
                # Rate-limited. Sleep until reset.
                rl_reset = e.headers.get("X-RateLimit-Reset")
                if rl_reset:
                    wait = max(60, int(rl_reset) + 5 - int(time.time()))
                    log.warning(f"  403 rate-limited, sleeping {wait}s")
                    time.sleep(wait)
                else:
                    time.sleep(60)
            elif e.code == 404:
                return None
            else:
                log.warning(f"  HTTP {e.code} on {url}; retrying after 30s")
                time.sleep(30)
        except Exception as e:
            log.warning(f"  exception {type(e).__name__} on {url}: {e}; retrying")
            time.sleep(30)
    return None


def fetch_repo(repo: str, target: int) -> list[dict]:
    """Fetch up to `target` closed issues with ≥3 comments from the repo."""
    out_dir = OUT_DIR / repo.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "issues.jsonl"
    seen = set()
    if out_file.exists():
        with open(out_file) as f:
            for line in f:
                d = json.loads(line)
                seen.add(d.get("number"))
    log.info(f"[{repo}] starting; already have {len(seen)} issues")
    if len(seen) >= target:
        log.info(f"[{repo}] target met")
        return []

    new_items = []
    fout = open(out_file, "a")
    page = 1
    while len(seen) + len(new_items) < target and page <= 10:
        list_url = (
            f"https://api.github.com/repos/{repo}/issues"
            f"?state=closed&per_page=100&page={page}"
        )
        log.info(f"[{repo}] fetching list page {page}")
        items = gh_get(list_url)
        if not items:
            break
        for item in items:
            if len(seen) + len(new_items) >= target:
                break
            if item.get("pull_request"):
                continue
            if item.get("number") in seen:
                continue
            n_comments = item.get("comments", 0)
            if n_comments < 3:
                continue
            # Fetch comments
            comments_url = item.get("comments_url")
            log.info(f"[{repo}] fetching #{item['number']} ({n_comments} comments)")
            time.sleep(PER_REQUEST_SLEEP)  # 60.5s unauth (60/hr) / 1.0s with GH_TOKEN (5000/hr)
            cs = gh_get(comments_url)
            if cs is None:
                continue
            comments_text = [c.get("body") or "" for c in cs]
            comments_users = [(c.get("user") or {}).get("login") for c in cs]
            comments_at = [c.get("created_at") for c in cs]
            d = {
                "url": item.get("url"),
                "html_url": item.get("html_url"),
                "id": item.get("id"),
                "number": item.get("number"),
                "title": item.get("title"),
                "labels": [l.get("name") for l in (item.get("labels") or [])],
                "state": item.get("state"),
                "locked": item.get("locked"),
                "comments": n_comments,
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
                "closed_at": item.get("closed_at"),
                "body": item.get("body") or "",
                "state_reason": item.get("state_reason"),
                "is_pull_request": False,
                "comments_text": comments_text,
                "comments_users": comments_users,
                "comments_at": comments_at,
                "user": (item.get("user") or {}).get("login"),
            }
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")
            fout.flush()
            new_items.append(d)
            time.sleep(0.1)
        page += 1
    fout.close()
    log.info(f"[{repo}] done: +{len(new_items)} (total {len(seen) + len(new_items)})")
    return new_items


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.getLogger().addHandler(logging.FileHandler(OUT_DIR / "pull.log", mode="a"))
    for repo in REPOS:
        try:
            fetch_repo(repo, PER_REPO_TARGET)
        except Exception as e:
            log.error(f"[{repo}] failed: {e}")
        # Brief inter-repo pause
        time.sleep(5)
    log.info("DONE")


if __name__ == "__main__":
    main()
