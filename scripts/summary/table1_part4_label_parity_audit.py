from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402

MARK_START = "<!-- DELTA_UNIT:START table1_part4_label_parity_82351 -->"
MARK_END = "<!-- DELTA_UNIT:END table1_part4_label_parity_82351 -->"
SECTION_HEADER = "### 2.1 Table 1 行ラベル整合（8.2.35.1）"


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _escape_cell(text: str) -> str:
    return str(text).replace("|", "&#124;").replace("\n", " ").strip()


def _unescape_cell(text: str) -> str:
    return str(text).replace("&#124;", "|").strip()


def _part4_section_for_topic(topic: str) -> str:
    t = str(topic)
    if t.startswith("LLR"):
        return "7"
    if t.startswith("Cassini") or t.startswith("Viking") or t.startswith("Mercury") or t.startswith("GPS"):
        return "9"
    if t.startswith("光偏向") or t.startswith("重力赤方偏移"):
        return "9"
    if t.startswith("宇宙論") or t.startswith("JWST/MAST"):
        return "11"
    if t.startswith("XRISM"):
        return "12"
    if t.startswith("回転"):
        return "10 / 12"
    if t.startswith("EHT") or t.startswith("連星パルサー") or t.startswith("重力波") or t.startswith("強場"):
        return "12"
    if t.startswith("速度飽和"):
        return "10"
    return "-"


def _build_marker_block(rows: List[Dict[str, Any]], *, public_png_ref: str) -> str:
    lines: List[str] = [
        MARK_START,
        "| No. | Table 1 topic | observable | Part IV section |",
        "|---|---|---|---|",
    ]
    for idx, row in enumerate(rows, start=1):
        topic = _escape_cell(str(row.get("topic", "")))
        observable = _escape_cell(str(row.get("observable", "")))
        part4_section = _escape_cell(_part4_section_for_topic(topic))
        lines.append(f"| {idx} | {topic} | {observable} | {part4_section} |")
    lines.append("")
    lines.append(f"行数固定: {len(rows)}（Table 1 行と 1:1 対応）")
    lines.append("")
    lines.append(f"`{public_png_ref}`")
    lines.append("")
    lines.append("**反証条件**：")
    lines.append("- Table 1 の全行（topic/observable）が Part IV の固定ラベル表に 1:1 で存在しない場合は棄却する。")
    lines.append("- Part IV 側に Table 1 に存在しない余分な行ラベルがある場合は棄却する。")
    lines.append("- 行数固定（39）と監査JSONの `n_missing_rows=0` / `n_extra_rows=0` が同時に満たされない場合は棄却する。")
    lines.append(MARK_END)
    return "\n".join(lines)


def _sync_part4(path: Path, block: str) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    start_idx = text.find(MARK_START)
    end_idx = text.find(MARK_END)
    mode = "inserted"
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        end_pos = end_idx + len(MARK_END)
        new_text = text[:start_idx] + block + text[end_pos:]
        mode = "replaced"
    else:
        section = (
            f"{SECTION_HEADER}\n\n"
            "Table 1 の行ラベル（topic / observable）を Part IV 側へ固定し、"
            "本文参照と集計行名のずれを防止する。\n\n"
            f"{block}\n\n"
        )
        insert_pos = text.find("## 3.")
        if insert_pos == -1:
            new_text = text.rstrip() + "\n\n" + section
        else:
            new_text = text[:insert_pos] + section + text[insert_pos:]
    path.write_text(new_text, encoding="utf-8")
    return {"mode": mode}


def _extract_block_pairs(text: str) -> List[Tuple[str, str]]:
    start_idx = text.find(MARK_START)
    end_idx = text.find(MARK_END)
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return []
    block = text[start_idx + len(MARK_START):end_idx]
    pairs: List[Tuple[str, str]] = []
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        if line.startswith("|---"):
            continue
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) < 4:
            continue
        if cols[0].lower() in {"no.", "no"}:
            continue
        topic = _unescape_cell(cols[1])
        observable = _unescape_cell(cols[2])
        if not topic or not observable:
            continue
        pairs.append((topic, observable))
    return pairs


def _plot_summary(*, out_png: Path, n_rows: int, n_ok: int, n_missing: int, n_extra: int) -> None:
    labels = ["table1 rows", "matched", "missing", "extra"]
    values = np.asarray([float(n_rows), float(n_ok), float(n_missing), float(n_extra)], dtype=float)
    colors = ["#4c78a8", "#54a24b", "#e45756", "#f58518"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(labels), dtype=float)
    ax.bar(x, values, color=colors, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("count")
    ax.set_title("Table 1 vs Part IV label parity audit")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Step 8.2.35.1: enforce exact row-label parity between Table 1 and Part IV summary labels."
    )
    ap.add_argument(
        "--table1-json",
        default=str(_ROOT / "output" / "private" / "summary" / "paper_table1_results.json"),
    )
    ap.add_argument(
        "--part4-md",
        default=str(_ROOT / "doc" / "paper" / "13_part4_verification.md"),
    )
    ap.add_argument("--outdir", default=str(_ROOT / "output" / "private" / "summary"))
    ap.add_argument("--public-outdir", default=str(_ROOT / "output" / "public" / "summary"))
    ap.add_argument("--prefix", default="table1_part4_label_parity_audit")
    ap.add_argument("--step-tag", default="8.2.35.1")
    ap.add_argument("--no-sync", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    table1_json = Path(args.table1_json)
    part4_md = Path(args.part4_md)
    outdir = Path(args.outdir)
    public_outdir = Path(args.public_outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    public_outdir.mkdir(parents=True, exist_ok=True)

    payload = _read_json(table1_json)
    table1 = payload.get("table1") if isinstance(payload.get("table1"), dict) else {}
    rows = table1.get("rows") if isinstance(table1.get("rows"), list) else []
    rows_norm: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        rows_norm.append(
            {
                "topic": str(row.get("topic", "")).strip(),
                "observable": str(row.get("observable", "")).strip(),
            }
        )

    sync_mode = "skipped"
    if not args.no_sync:
        public_png_ref = f"output/public/summary/{args.prefix}.png"
        block = _build_marker_block(rows_norm, public_png_ref=public_png_ref)
        sync_info = _sync_part4(part4_md, block)
        sync_mode = str(sync_info.get("mode", "unknown"))

    part4_text = part4_md.read_text(encoding="utf-8")
    part4_pairs = _extract_block_pairs(part4_text)

    table_counter = Counter((r["topic"], r["observable"]) for r in rows_norm)
    part4_counter = Counter(part4_pairs)

    missing_pairs: List[Tuple[str, str]] = []
    for pair, n in table_counter.items():
        diff = n - int(part4_counter.get(pair, 0))
        if diff > 0:
            missing_pairs.extend([pair] * diff)

    extra_pairs: List[Tuple[str, str]] = []
    for pair, n in part4_counter.items():
        diff = n - int(table_counter.get(pair, 0))
        if diff > 0:
            extra_pairs.extend([pair] * diff)

    seen = Counter()
    csv_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows_norm, start=1):
        pair = (row["topic"], row["observable"])
        seen[pair] += 1
        present = int(part4_counter.get(pair, 0) >= seen[pair])
        csv_rows.append(
            {
                "row_no": idx,
                "topic": row["topic"],
                "observable": row["observable"],
                "row_label": f"{row['topic']}｜{row['observable']}",
                "part4_section": _part4_section_for_topic(row["topic"]),
                "present_in_part4_label_block": present,
            }
        )

    for idx, (topic, observable) in enumerate(extra_pairs, start=1):
        csv_rows.append(
            {
                "row_no": f"extra_{idx}",
                "topic": topic,
                "observable": observable,
                "row_label": f"{topic}｜{observable}",
                "part4_section": "-",
                "present_in_part4_label_block": 1,
            }
        )

    n_rows = len(rows_norm)
    n_missing = len(missing_pairs)
    n_extra = len(extra_pairs)
    n_ok = max(0, n_rows - n_missing)
    overall_status = "pass" if (n_missing == 0 and n_extra == 0) else "reject"
    decision = (
        "table1_part4_label_parity_pass"
        if overall_status == "pass"
        else "table1_part4_label_parity_reject"
    )
    next_action = (
        "parity_locked_keep_table1_and_part4_in_sync"
        if overall_status == "pass"
        else "fix_mismatched_labels_then_rerun_step_8_2_35_1"
    )

    out_json = outdir / f"{args.prefix}.json"
    out_csv = outdir / f"{args.prefix}.csv"
    out_png = outdir / f"{args.prefix}.png"

    _write_csv(out_csv, csv_rows)
    _plot_summary(
        out_png=out_png,
        n_rows=n_rows,
        n_ok=n_ok,
        n_missing=n_missing,
        n_extra=n_extra,
    )

    result = {
        "generated_utc": _iso_utc_now(),
        "schema": "wavep.summary.table1_part4_label_parity.v1",
        "phase": 8,
        "step": str(args.step_tag),
        "inputs": {
            "table1_json": str(table1_json).replace("\\", "/"),
            "part4_markdown": str(part4_md).replace("\\", "/"),
            "sync_part4": bool(not args.no_sync),
            "sync_mode": sync_mode,
        },
        "summary": {
            "overall_status": overall_status,
            "decision": decision,
            "next_action": next_action,
            "n_table1_rows": n_rows,
            "n_part4_label_rows": len(part4_pairs),
            "n_matched_rows": n_ok,
            "n_missing_rows": n_missing,
            "n_extra_rows": n_extra,
        },
        "missing_pairs": [
            {"topic": topic, "observable": observable, "row_label": f"{topic}｜{observable}"}
            for topic, observable in missing_pairs
        ],
        "extra_pairs": [
            {"topic": topic, "observable": observable, "row_label": f"{topic}｜{observable}"}
            for topic, observable in extra_pairs
        ],
        "outputs": {
            "audit_json": str(out_json).replace("\\", "/"),
            "audit_csv": str(out_csv).replace("\\", "/"),
            "audit_png": str(out_png).replace("\\", "/"),
        },
        "falsification": {
            "hard_pass_if": [
                "all Table 1 (topic, observable) pairs appear in Part IV label block with exact text equality",
                "no extra (topic, observable) pair exists in Part IV label block",
            ],
            "reject_if": [
                "any missing pair exists",
                "any extra pair exists",
            ],
        },
    }
    _write_json(out_json, result)

    public_copies: List[str] = []
    for src in [out_json, out_csv, out_png]:
        dst = public_outdir / src.name
        shutil.copy2(src, dst)
        public_copies.append(str(dst).replace("\\", "/"))
    result["outputs"]["public_copies"] = public_copies
    _write_json(out_json, result)
    shutil.copy2(out_json, public_outdir / out_json.name)

    try:
        worklog.append_event(
            {
                "event_type": "table1_part4_label_parity_audit",
                "argv": list(sys.argv),
                "outputs": {
                    "audit_json": out_json,
                    "audit_csv": out_csv,
                    "audit_png": out_png,
                },
                "metrics": {
                    "overall_status": overall_status,
                    "decision": decision,
                    "n_table1_rows": n_rows,
                    "n_part4_label_rows": len(part4_pairs),
                    "n_missing_rows": n_missing,
                    "n_extra_rows": n_extra,
                    "sync_mode": sync_mode,
                },
            }
        )
    except Exception:
        pass

    print(f"[ok] status={overall_status} decision={decision}")
    print(
        "[ok] rows table1={t} part4={p} matched={m} missing={miss} extra={extra}".format(
            t=n_rows,
            p=len(part4_pairs),
            m=n_ok,
            miss=n_missing,
            extra=n_extra,
        )
    )
    print(f"[ok] json: {out_json}")
    print(f"[ok] csv : {out_csv}")
    print(f"[ok] png : {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
