"""Upload run artifacts (per-series predictions + summaries) to a HF dataset repo.

The prediction parquets are the reproducibility payload the paper promises to release:
every offline analysis (repair arm, calibration diagnostics, any future metric) is
recomputed from them without re-running a model. They are deliberately untracked in git
(~6 GB), so this script is both the machine-to-machine transfer path and, later, the
artifact release.

Defaults to a PRIVATE repo: the paper is an anonymous submission, and a public upload
under a personal account would de-anonymize it. Flip with --public only after the
review decision.

Usage:
    uv run huggingface-cli login          # once, interactively, in your own terminal
    uv run python scripts/upload_artifacts.py <user-or-org>/chronosquant-artifacts
    # resume after an interruption: re-run the exact same command
"""

import argparse
import sys
from pathlib import Path

from chronosquant.utils import REPO_ROOT, ensure_truststore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", help="target dataset repo, e.g. someuser/chronosquant-artifacts")
    parser.add_argument(
        "--tier",
        choices=["full", "dev", "both"],
        default="both",
        help="which run tier(s) to upload (default: both; 'full' is the paper-critical one)",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="create a PUBLIC repo (default private; only after the review decision)",
    )
    parser.add_argument("--dry-run", action="store_true", help="list what would upload, then exit")
    args = parser.parse_args()

    ensure_truststore()  # corporate TLS proxy
    from huggingface_hub import HfApi

    raw = REPO_ROOT / "results" / "raw"
    if not raw.is_dir():
        print(f"error: {raw} not found", file=sys.stderr)
        return 1

    suffixes = {"full": ["_full"], "dev": ["_dev"], "both": ["_full", "_dev"]}[args.tier]
    run_dirs = sorted(d for d in raw.iterdir() if d.is_dir() and d.name.endswith(tuple(suffixes)))
    allow = [f"{d.name}/**" for d in run_dirs]

    n_files = sum(1 for d in run_dirs for _ in d.rglob("*") if _.is_file())
    n_bytes = sum(f.stat().st_size for d in run_dirs for f in d.rglob("*") if f.is_file())
    print(f"{len(run_dirs)} run dirs | {n_files} files | {n_bytes / 1e9:.2f} GB")
    print(f"target: {args.repo_id} ({'PUBLIC' if args.public else 'private'})")
    for d in run_dirs:
        print(f"  {d.name}")
    if args.dry_run:
        return 0

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=not args.public,
        exist_ok=True,
    )
    # upload_large_folder: chunked, multi-threaded, and resumable across interruptions —
    # the right primitive for a multi-GB payload over a flaky/proxied link.
    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(raw),
        allow_patterns=allow,
        print_report=True,
    )
    print(f"\ndone: https://huggingface.co/datasets/{args.repo_id}")
    print("pull on the other machine with:")
    print(
        f"  uv run huggingface-cli download {args.repo_id} --repo-type dataset "
        f"--local-dir results/raw"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
