#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
part3b_rebuild_source_figures.py

Part III-B manuscript が参照する図の source script を inventory から抽出し、
共通 baseline / 日本語化 / fixed-width canvas ルールで順次再実行する。

目的:
- Part III-B の graph creation processing を source 側から再構築する。
- 手動 figsize / bbox_inches="tight" が残る script でも、
  現在の共通 profile で canonical public PDF を更新する。

入力:
- output/private/summary/part3b_stem_script_inventory.json

出力:
- output/private/summary/part3b_source_rebuild_audit.json
- output/private/summary/part3b_source_rebuild_audit.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = ROOT / "output" / "private" / "summary" / "part3b_stem_script_inventory.json"
DEFAULT_OUT_JSON = ROOT / "output" / "private" / "summary" / "part3b_source_rebuild_audit.json"
DEFAULT_OUT_CSV = ROOT / "output" / "private" / "summary" / "part3b_source_rebuild_audit.csv"
SKIP_SCRIPT_NAMES = {
    "figure_japanese_localizer.py",
    "frozen_parameters_quantum.py",
}
SKIP_PREFIXES = (
    "fetch_",
)


# 関数: UTC 現在時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: inventory JSON を読み込む。

def _load_inventory(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: inventory から実行対象 script の一意リストを抽出する。

def _collect_candidate_scripts(payload: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()
    for row in payload.get("rows", []):
        for hit in row.get("hits", []):
            rel = str(hit).split(":", 2)[0].replace("\\", "/")
            if not rel.endswith(".py"):
                continue

            parts = Path(rel).parts
            if len(parts) < 3 or parts[0] != "scripts" or parts[1] != "quantum":
                continue

            name = Path(rel).name
            if name in SKIP_SCRIPT_NAMES or any(name.startswith(prefix) for prefix in SKIP_PREFIXES):
                continue

            abs_path = ROOT / Path(rel)
            if not abs_path.exists():
                continue

            try:
                text = abs_path.read_text(encoding="utf-8")
            except Exception:
                continue

            has_plotting = (
                "savefig(" in text
                or "plt.subplots" in text
                or "plt.figure" in text
                or "matplotlib.pyplot" in text
            )
            if not has_plotting:
                continue

            key = str(abs_path.resolve()).lower()
            if key in seen:
                continue

            seen.add(key)
            candidates.append(abs_path)

    return sorted(candidates)


# 関数: 単一 script を Part III-B 共通環境で実行して結果行を返す。

def _run_script(path: Path) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["MPLBACKEND"] = "Agg"
    env["WAVEP_MPL_FONT_PROFILE"] = "part3b_quantum_verification"
    env["WAVEP_MPL_FONT_SCALE"] = "1.0"
    env["WAVEP_MPL_CJK_FONT"] = "Noto Sans CJK JP"
    env["WAVEP_FIGURE_LANG"] = "ja"

    started = time.perf_counter()
    completed_utc = None
    try:
        cp = subprocess.run(
            [sys.executable, "-B", str(path)],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        completed_utc = _utc_now_iso()
        elapsed_s = time.perf_counter() - started
        return {
            "script": str(path.relative_to(ROOT)).replace("\\", "/"),
            "returncode": int(cp.returncode),
            "elapsed_s": float(elapsed_s),
            "completed_utc": completed_utc,
            "stdout_tail": cp.stdout[-1200:],
            "stderr_tail": cp.stderr[-1200:],
        }
    except Exception as exc:
        completed_utc = _utc_now_iso()
        elapsed_s = time.perf_counter() - started
        return {
            "script": str(path.relative_to(ROOT)).replace("\\", "/"),
            "returncode": -1,
            "elapsed_s": float(elapsed_s),
            "completed_utc": completed_utc,
            "stdout_tail": "",
            "stderr_tail": repr(exc),
        }


# 関数: 実行結果を JSON / CSV に書き出す。

def _write_audit(rows: list[dict[str, Any]], out_json: Path, out_csv: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_utc": _utc_now_iso(),
        "row_count": len(rows),
        "failure_count": int(sum(1 for row in rows if int(row["returncode"]) != 0)),
        "rows": rows,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = ["script", "returncode", "elapsed_s", "completed_utc", "stdout_tail", "stderr_tail"]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


# 関数: CLI 引数を解釈して Part III-B source figure rebuild を実行する。

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild Part III-B source figures from the script inventory.")
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY, help="Inventory JSON path.")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON, help="Audit JSON output path.")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV, help="Audit CSV output path.")
    args = parser.parse_args(argv)

    payload = _load_inventory(args.inventory)
    candidates = _collect_candidate_scripts(payload)
    rows: list[dict[str, Any]] = []
    for script_path in candidates:
        row = _run_script(script_path)
        rows.append(row)
        print(f"[{row['returncode']}] {row['script']}")

    _write_audit(rows, args.out_json, args.out_csv)
    failures = sum(1 for row in rows if int(row["returncode"]) != 0)
    print(f"[done] rebuilt {len(rows)} script(s); failures={failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
