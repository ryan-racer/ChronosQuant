"""Live sweep progress dashboard. Run in your own terminal:

    uv run python scripts/watch_progress.py

Refreshes every few seconds; tab-independent (reads result files on disk, does not
touch the running sweep). Ctrl-C to quit.
"""

import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "results" / "raw"
TOTAL_TASKS = 27  # full-tier Benchmark II


def count_tasks(summaries: Path) -> int:
    if not summaries.exists():
        return 0
    with open(summaries) as f:
        return max(sum(1 for _ in f) - 1, 0)  # minus header


def snapshot() -> list[tuple[str, int, str]]:
    rows = []
    for run_dir in sorted(RAW.glob("*_full")):
        summaries = run_dir / "summaries.csv"
        meta = run_dir / "run_metadata.yaml"
        n = count_tasks(summaries)
        status = "running"
        if meta.exists():
            try:
                m = yaml.safe_load(meta.read_text()) or {}
                if m.get("num_completed"):
                    status = "done"
                    n = m["num_completed"]
            except Exception:
                pass
        rows.append((run_dir.name.replace("_full", ""), n, status))
    return rows


def main() -> None:
    try:
        while True:
            rows = snapshot()
            done = sum(1 for _, _, s in rows if s == "done")
            lines = ["\033[2J\033[H"]  # clear screen, home cursor
            lines.append(f"  ChronosQuant sweep — {done}/{len(rows)} runs complete")
            lines.append("  " + "-" * 44)
            for name, n, status in rows:
                if status == "done":
                    bar = "#" * 20
                    mark = "OK "
                else:
                    filled = int(20 * n / TOTAL_TASKS)
                    bar = "#" * filled + "." * (20 - filled)
                    mark = ">> "
                lines.append(f"  {mark}{name:26s} [{bar}] {n:2d}/{TOTAL_TASKS}")
            lines.append("")
            lines.append("  refreshing every 5s — Ctrl-C to quit")
            print("\n".join(lines), flush=True)
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nstopped watching (sweep keeps running).")


if __name__ == "__main__":
    main()
