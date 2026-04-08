#!/usr/bin/env python3
"""Audit repo paths against the conservative Windows length policy.

Purpose:
- Quantify where historical long names still exist.
- Freeze one repo-local rule set for new artifacts and CLI invocations.
- Provide a single reproducible report before future quantum branches switch
  to compact stems and response files.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.windows_length_policy import (  # noqa: E402
    WINDOWS_SAFE_NAME_LIMIT,
    WINDOWS_SAFE_PATH_LIMIT,
    build_metrics_paths,
    build_compact_artifact_stem,
    summarize_longest_paths,
)


OUT_DIR = ROOT / "output" / "private" / "summary"
DEFAULT_OUT_STEM = "windows_length_policy"


# 関数: `now_iso` の入出力契約と処理意図を定義する。
def now_iso() -> str:
    """Return the current UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


# 関数: `iter_repo_paths` の入出力契約と処理意図を定義する。

def iter_repo_paths(roots: list[Path]) -> list[Path]:
    """Collect every file and directory under the selected roots."""
    collected: list[Path] = []
    for base in roots:
        # 条件分岐: `not base.exists()` を満たす経路を評価する。
        if not base.exists():
            continue

        collected.append(base)
        if base.is_dir():
            collected.extend(sorted(base.rglob("*")))

    return collected


# 関数: `classify_paths` の入出力契約と処理意図を定義する。

def classify_paths(paths: list[Path], max_path_len: int, max_name_len: int) -> tuple[list[dict], dict]:
    """Classify scanned paths against the configured budgets."""
    rows: list[dict] = []
    max_path = 0
    max_name = 0
    over_path = 0
    over_name = 0
    output_over_path = 0
    quantum_over_path = 0

    for candidate in paths:
        path_text = str(candidate)
        path_len = len(path_text)
        name_len = len(candidate.name)
        max_path = max(max_path, path_len)
        max_name = max(max_name, name_len)
        path_status = "pass"
        name_status = "pass"

        # 条件分岐: `path_len > max_path_len` を満たす経路を評価する。
        if path_len > max_path_len:
            path_status = "reject"
            over_path += 1
            candidate_text = path_text.replace("\\", "/")
            if "/output/" in candidate_text:
                output_over_path += 1

            if "/output/public/quantum/" in candidate_text or "/output/private/quantum/" in candidate_text:
                quantum_over_path += 1

        # 条件分岐: `name_len > max_name_len` を満たす経路を評価する。

        if name_len > max_name_len:
            name_status = "reject"
            over_name += 1

        rows.append(
            {
                "path": path_text,
                "path_len": path_len,
                "name_len": name_len,
                "path_status": path_status,
                "name_status": name_status,
                "kind": "dir" if candidate.is_dir() else "file",
            }
        )

    summary = {
        "scanned_path_count": len(paths),
        "windows_safe_path_limit": int(max_path_len),
        "windows_safe_name_limit": int(max_name_len),
        "max_path_len_observed": max_path,
        "max_name_len_observed": max_name,
        "paths_over_limit": over_path,
        "names_over_limit": over_name,
        "output_paths_over_limit": output_over_path,
        "quantum_output_paths_over_limit": quantum_over_path,
    }
    return rows, summary


# 関数: `write_outputs` の入出力契約と処理意図を定義する。

def write_outputs(out_stem: str, rows: list[dict], summary: dict) -> dict[str, Path]:
    """Write the audit CSV and JSON outputs."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_paths = build_metrics_paths(OUT_DIR, out_stem, "audit")
    out_json = metrics_paths["json"]
    out_csv = metrics_paths["csv"]

    out_json.write_text(
        json.dumps(
            {
                "generated_utc": now_iso(),
                "summary": summary,
                "rows": rows[:250],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["kind", "path_status", "name_status", "path_len", "name_len", "path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return {"json": out_json, "csv": out_csv}


# 関数: `parse_args` の入出力契約と処理意図を定義する。

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Audit repo paths against the Windows length policy.")
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["doc", "scripts", "output/public", "output/private/quantum"],
        help="Directories to scan (default: doc scripts output/public output/private/quantum).",
    )
    parser.add_argument(
        "--max-path-len",
        type=int,
        default=WINDOWS_SAFE_PATH_LIMIT,
        help=f"Conservative Windows-safe full-path limit (default: {WINDOWS_SAFE_PATH_LIMIT}).",
    )
    parser.add_argument(
        "--max-name-len",
        type=int,
        default=WINDOWS_SAFE_NAME_LIMIT,
        help=f"Conservative Windows-safe filename limit (default: {WINDOWS_SAFE_NAME_LIMIT}).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of longest paths to print in the console summary (default: 20).",
    )
    return parser.parse_args()


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    """Run the Windows length audit."""
    args = parse_args()
    scan_roots = [(ROOT / root_text).resolve() for root_text in args.roots]
    all_paths = iter_repo_paths(scan_roots)
    rows, summary = classify_paths(all_paths, args.max_path_len, args.max_name_len)
    compact_example = build_compact_artifact_stem(
        "8.7.56.1471",
        "exact_action_level_ell0_operator_derivation",
    )
    summary["recommended_new_artifact_stem_example"] = compact_example
    summary["recommended_new_artifact_example_json"] = (
        f"output/public/quantum/{compact_example}_source_inventory_metrics.json"
    )

    outputs = write_outputs(DEFAULT_OUT_STEM, rows, summary)

    print(
        "[summary] scanned={scanned} max_path={max_path} max_name={max_name} "
        "paths_over={paths_over} names_over={names_over}".format(
            scanned=summary["scanned_path_count"],
            max_path=summary["max_path_len_observed"],
            max_name=summary["max_name_len_observed"],
            paths_over=summary["paths_over_limit"],
            names_over=summary["names_over_limit"],
        )
    )
    print(f"[summary] suggested_new_stem={compact_example}")
    print(f"[summary] metrics_json={outputs['json']}")
    print(f"[summary] rows_csv={outputs['csv']}")
    for path_len, name_len, path_text in summarize_longest_paths(all_paths, top_n=args.top):
        print(f"[top] path_len={path_len} name_len={name_len} path={path_text}")

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
