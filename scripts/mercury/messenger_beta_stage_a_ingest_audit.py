#!/usr/bin/env python3
"""
messenger_beta_stage_a_ingest_audit.py

Roadmap Step 8.7.48.1 (Stage A ingestion/unit test) の実装。

目的:
- MESSENGER theory-native β推定の入口である ODF/TNF と補助データの可用性を固定する。
- Stage A の単体監査（時刻系/局情報/range-doppler符号/2way-3way/transponder）を
  parser-independent に実行し、Stage B へ進める前提を明示する。

入力（既定）:
- data/mercury/messenger/
  - data-odf/
  - data-tnf/
  - calib/{ant,ion,tro,wea,ltf,mdm,mpd,sff}/
  - external/{naif,iers}/

出力（既定）:
- output/private/mercury/messenger_beta_stage_a_ingest_inventory.csv
- output/private/mercury/messenger_beta_stage_a_unit_tests.csv
- output/private/mercury/messenger_beta_stage_a_ingest_metrics.json
- output/private/mercury/messenger_beta_stage_a_ingest_status.pdf/.png
- output/private/mercury/messenger_beta_stage_a_unit_tests_status.pdf/.png
- 上記を output/public/mercury/ へ同期
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from scripts.summary.worklog import append_event


# クラス: `DataGate` の責務と境界条件を定義する。
@dataclass
class DataGate:
    gate_id: str
    rel_path: str
    required_level: str  # required / optional
    exists: bool
    file_count: int
    status: str
    note: str


# クラス: `UnitTestResult` の責務と境界条件を定義する。

@dataclass
class UnitTestResult:
    test_id: str
    status: str
    evidence_files: int
    note: str
    keywords: str


# 関数: `_safe_rel` の入出力契約と処理意図を定義する。

def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


# 関数: `_resolve_path` の入出力契約と処理意図を定義する。

def _resolve_path(path_str: str, root: Path) -> Path:
    p = Path(str(path_str))
    if p.is_absolute():
        return p

    return (root / p).resolve()


# 関数: `_count_files` の入出力契約と処理意図を定義する。

def _count_files(path: Path) -> int:
    if not path.exists():
        return 0

    if path.is_file():
        return 1

    return sum(1 for p in path.rglob("*") if p.is_file())


# 関数: `_status_for_gate` の入出力契約と処理意図を定義する。

def _status_for_gate(required_level: str, file_count: int) -> str:
    if file_count > 0:
        return "pass"

    if required_level == "required":
        return "reject"

    return "watch"


# 関数: `_build_gates` の入出力契約と処理意図を定義する。

def _build_gates(data_root: Path) -> List[DataGate]:
    expected: Sequence[Tuple[str, str, str, str]] = (
        ("primary_odf", "data-odf", "required", "主入口（range + Doppler）"),
        ("secondary_tnf", "data-tnf", "optional", "再走査枝（TNF replay）"),
        ("calib_ant", "calib/ant", "optional", "antenna selection"),
        ("calib_ion", "calib/ion", "optional", "ionosphere calibration"),
        ("calib_tro", "calib/tro", "optional", "troposphere calibration"),
        ("calib_wea", "calib/wea", "optional", "DSN weather"),
        ("calib_ltf", "calib/ltf", "optional", "light-time file"),
        ("calib_mdm", "calib/mdm", "optional", "momentum dump summary"),
        ("calib_mpd", "calib/mpd", "optional", "maneuver performance"),
        ("calib_sff", "calib/sff", "optional", "small forces"),
        ("external_naif", "external/naif", "optional", "SPICE kernels"),
        ("external_iers", "external/iers", "optional", "Earth orientation"),
    )
    rows: List[DataGate] = []
    for gate_id, rel, req, note in expected:
        p = data_root / rel
        count = _count_files(p)
        rows.append(
            DataGate(
                gate_id=gate_id,
                rel_path=rel,
                required_level=req,
                exists=p.exists(),
                file_count=count,
                status=_status_for_gate(req, count),
                note=note,
            )
        )

    return rows


# 関数: `_collect_candidate_text_files` の入出力契約と処理意図を定義する。

def _collect_candidate_text_files(
    roots: Sequence[Path],
    max_files: int,
) -> List[Path]:
    allowed_ext = {
        ".lbl",
        ".txt",
        ".xml",
        ".json",
        ".csv",
        ".tab",
        ".fmt",
        ".dat",
        ".log",
        ".rpt",
        ".md",
    }
    candidates: List[Path] = []
    for root in roots:
        if not root.exists():
            continue

        if root.is_file():
            if root.suffix.lower() in allowed_ext:
                candidates.append(root)

            continue

        for p in root.rglob("*"):
            if not p.is_file():
                continue

            if p.suffix.lower() in allowed_ext:
                candidates.append(p)

            if len(candidates) >= max_files:
                return candidates

    return candidates


# 関数: `_read_text_sample` の入出力契約と処理意図を定義する。

def _read_text_sample(path: Path, max_bytes: int) -> str:
    try:
        with path.open("rb") as f:
            raw = f.read(max_bytes)
    except Exception:
        return ""

    if not raw:
        return ""

    try:
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""


# 関数: `_keyword_file_hits` の入出力契約と処理意図を定義する。

def _keyword_file_hits(
    files: Sequence[Path],
    keywords: Sequence[str],
    max_bytes: int,
) -> Tuple[int, List[str]]:
    escaped = [re.escape(k) for k in keywords]
    pattern = re.compile("|".join(escaped), flags=re.IGNORECASE)
    hit_files: List[str] = []
    for path in files:
        text = _read_text_sample(path, max_bytes=max_bytes)
        if not text:
            continue

        if pattern.search(text) is not None:
            hit_files.append(_safe_rel(path, _ROOT))

    return len(hit_files), hit_files


# 関数: `_unit_status` の入出力契約と処理意図を定義する。

def _unit_status(primary_odf_count: int, evidence_files: int, pass_threshold: int = 2) -> str:
    if primary_odf_count <= 0:
        return "reject"

    if evidence_files >= pass_threshold:
        return "pass"

    if evidence_files >= 1:
        return "watch"

    return "reject"


# 関数: `_build_unit_tests` の入出力契約と処理意図を定義する。

def _build_unit_tests(
    data_root: Path,
    gates: Sequence[DataGate],
    max_scan_files: int,
    max_scan_bytes: int,
) -> Tuple[List[UnitTestResult], Dict[str, List[str]]]:
    odf_dir = data_root / "data-odf"
    tnf_dir = data_root / "data-tnf"
    aux_dirs = [
        data_root / "calib",
        data_root / "external",
    ]
    scan_roots: List[Path] = [odf_dir, tnf_dir]
    scan_roots.extend(aux_dirs)
    text_files = _collect_candidate_text_files(scan_roots, max_files=max_scan_files)
    primary_odf = next((g for g in gates if g.gate_id == "primary_odf"), None)
    primary_count = int(primary_odf.file_count if primary_odf is not None else 0)

    specs: Sequence[Tuple[str, Sequence[str], str]] = (
        (
            "time_system_conversion",
            ("UTC", "TDB", "ET", "SCLK", "TIME SYSTEM", "START_TIME", "STOP_TIME"),
            "時刻系変換（UTC/TDB/SCLK）を確認する",
        ),
        (
            "dsn_station_metadata",
            ("DSS", "DEEP SPACE STATION", "STATION", "COMPLEX", "ANTENNA"),
            "DSN 局メタデータを確認する",
        ),
        (
            "range_doppler_sign_convention",
            ("DOPPLER", "RANGE", "SIGN", "COHERENT", "FREQ", "FREQUENCY"),
            "range/doppler と符号規約の語彙を確認する",
        ),
        (
            "two_three_way_flag",
            ("TWO-WAY", "THREE-WAY", "2-WAY", "3-WAY", "ONE-WAY"),
            "two-way/three-way フラグ語彙を確認する",
        ),
        (
            "transponder_delay",
            ("TRANSPONDER", "TURNAROUND", "DELAY"),
            "transponder delay 語彙を確認する",
        ),
    )
    results: List[UnitTestResult] = []
    evidence_map: Dict[str, List[str]] = {}
    for test_id, keywords, note in specs:
        hits, hit_files = _keyword_file_hits(
            files=text_files,
            keywords=keywords,
            max_bytes=max_scan_bytes,
        )
        status = _unit_status(primary_count, evidence_files=hits, pass_threshold=2)
        results.append(
            UnitTestResult(
                test_id=test_id,
                status=status,
                evidence_files=hits,
                note=note,
                keywords=";".join(keywords),
            )
        )
        evidence_map[test_id] = hit_files[:20]

    return results, evidence_map


# 関数: `_write_inventory_csv` の入出力契約と処理意図を定義する。

def _write_inventory_csv(path: Path, gates: Sequence[DataGate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["gate_id", "rel_path", "required_level", "exists", "file_count", "status", "note"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for g in gates:
            writer.writerow(
                {
                    "gate_id": g.gate_id,
                    "rel_path": g.rel_path,
                    "required_level": g.required_level,
                    "exists": int(g.exists),
                    "file_count": g.file_count,
                    "status": g.status,
                    "note": g.note,
                }
            )


# 関数: `_write_unit_tests_csv` の入出力契約と処理意図を定義する。

def _write_unit_tests_csv(path: Path, tests: Sequence[UnitTestResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["test_id", "status", "evidence_files", "note", "keywords"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for t in tests:
            writer.writerow(
                {
                    "test_id": t.test_id,
                    "status": t.status,
                    "evidence_files": t.evidence_files,
                    "note": t.note,
                    "keywords": t.keywords,
                }
            )


# 関数: `_status_counts` の入出力契約と処理意図を定義する。

def _status_counts(values: Sequence[str]) -> Dict[str, int]:
    counts = {"pass": 0, "watch": 0, "reject": 0}
    for value in values:
        counts[value] = int(counts.get(value, 0)) + 1

    return counts


# 関数: `_overall_status` の入出力契約と処理意図を定義する。

def _overall_status(gates: Sequence[DataGate], tests: Sequence[UnitTestResult]) -> str:
    gate_statuses = [g.status for g in gates]
    test_statuses = [t.status for t in tests]
    combined = gate_statuses + test_statuses
    if "reject" in combined:
        return "reject"

    if "watch" in combined:
        return "watch"

    return "pass"


# 関数: `_make_bar_plot` の入出力契約と処理意図を定義する。

def _make_bar_plot(
    labels: Sequence[str],
    statuses: Sequence[str],
    subtitle: Sequence[str],
    title: str,
    out_pdf: Path,
    out_png: Path,
) -> Optional[str]:
    if plt is None:
        return "matplotlib_unavailable"

    color_map = {"pass": "#2ca02c", "watch": "#f1c232", "reject": "#d62728"}
    score_map = {"pass": 0.6, "watch": 1.6, "reject": 2.8}
    values = [score_map[s] for s in statuses]
    colors = [color_map[s] for s in statuses]

    fig, ax = plt.subplots(figsize=(11.8, 6.8))
    y = list(range(len(labels)))
    ax.barh(y, values, color=colors, alpha=0.92)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.2)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 3.0)
    ax.axvline(1.0, color="#999999", linestyle="--", linewidth=1.0)
    ax.axvline(2.0, color="#999999", linestyle="--", linewidth=1.0)
    ax.grid(axis="x", alpha=0.22)
    ax.set_xlabel("status score (pass=0.6, watch=1.6, reject=2.8)")
    ax.set_title(title)
    for i, line in enumerate(subtitle):
        ax.text(0.02, i, line, va="center", ha="left", fontsize=8.6)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return None


# 関数: `_sync_to_public` の入出力契約と処理意図を定義する。

def _sync_to_public(paths: Iterable[Path], private_root: Path, public_root: Path) -> List[Path]:
    public_root.mkdir(parents=True, exist_ok=True)
    synced: List[Path] = []
    for src in paths:
        try:
            rel = src.resolve().relative_to(private_root.resolve())
        except Exception:
            rel = Path(src.name)

        dst = public_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        synced.append(dst)

    return synced


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Roadmap 8.7.48.1: audit MESSENGER ODF/TNF ingestion and parser-unit gates."
    )
    ap.add_argument(
        "--data-root",
        type=str,
        default=str(_ROOT / "data" / "mercury" / "messenger"),
        help="Root directory containing data-odf/data-tnf/calib/external subtrees.",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(_ROOT / "output" / "private" / "mercury"),
        help="Private output directory.",
    )
    ap.add_argument(
        "--public-dir",
        type=str,
        default=str(_ROOT / "output" / "public" / "mercury"),
        help="Public sync directory.",
    )
    ap.add_argument(
        "--text-scan-max-files",
        type=int,
        default=5000,
        help="Max number of text-like files to scan for Stage A keyword-based unit tests.",
    )
    ap.add_argument(
        "--text-scan-max-bytes",
        type=int,
        default=1024 * 1024,
        help="Max bytes per file to scan.",
    )
    args = ap.parse_args()

    data_root = _resolve_path(args.data_root, _ROOT)
    out_dir = _resolve_path(args.out_dir, _ROOT)
    public_dir = _resolve_path(args.public_dir, _ROOT)

    gates = _build_gates(data_root)
    unit_tests, unit_evidence = _build_unit_tests(
        data_root=data_root,
        gates=gates,
        max_scan_files=int(args.text_scan_max_files),
        max_scan_bytes=int(args.text_scan_max_bytes),
    )
    overall = _overall_status(gates, unit_tests)
    gate_counts = _status_counts([g.status for g in gates])
    unit_counts = _status_counts([t.status for t in unit_tests])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_inventory_csv = out_dir / "messenger_beta_stage_a_ingest_inventory.csv"
    out_unit_csv = out_dir / "messenger_beta_stage_a_unit_tests.csv"
    out_json = out_dir / "messenger_beta_stage_a_ingest_metrics.json"
    out_gate_pdf = out_dir / "messenger_beta_stage_a_ingest_status.pdf"
    out_gate_png = out_dir / "messenger_beta_stage_a_ingest_status.png"
    out_unit_pdf = out_dir / "messenger_beta_stage_a_unit_tests_status.pdf"
    out_unit_png = out_dir / "messenger_beta_stage_a_unit_tests_status.png"

    _write_inventory_csv(out_inventory_csv, gates)
    _write_unit_tests_csv(out_unit_csv, unit_tests)
    gate_plot_note = _make_bar_plot(
        labels=[g.gate_id for g in gates],
        statuses=[g.status for g in gates],
        subtitle=[f"{g.rel_path} files={g.file_count}" for g in gates],
        title="Roadmap 8.7.48.1: MESSENGER Stage A ingestion gates",
        out_pdf=out_gate_pdf,
        out_png=out_gate_png,
    )
    unit_plot_note = _make_bar_plot(
        labels=[t.test_id for t in unit_tests],
        statuses=[t.status for t in unit_tests],
        subtitle=[f"evidence_files={t.evidence_files}" for t in unit_tests],
        title="Roadmap 8.7.48.1: MESSENGER Stage A parser unit-test gates",
        out_pdf=out_unit_pdf,
        out_png=out_unit_png,
    )

    produced: List[Path] = [out_inventory_csv, out_unit_csv, out_json]
    if gate_plot_note is None:
        produced.extend([out_gate_pdf, out_gate_png])

    if unit_plot_note is None:
        produced.extend([out_unit_pdf, out_unit_png])

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase_step": "8.7.48.1",
        "data_root": _safe_rel(data_root, _ROOT),
        "overall_status": overall,
        "stage_a_ready_for_stage_b": bool(overall != "reject"),
        "ingestion_gate_counts": gate_counts,
        "unit_test_counts": unit_counts,
        "stage_a_gates": [
            {
                "gate_id": g.gate_id,
                "rel_path": g.rel_path,
                "required_level": g.required_level,
                "exists": g.exists,
                "file_count": g.file_count,
                "status": g.status,
                "note": g.note,
            }
            for g in gates
        ],
        "unit_tests": [
            {
                "test_id": t.test_id,
                "status": t.status,
                "evidence_files": t.evidence_files,
                "note": t.note,
                "keywords": t.keywords.split(";"),
                "evidence_sample": unit_evidence.get(t.test_id, []),
            }
            for t in unit_tests
        ],
        "notes": {
            "ingestion_plot": "generated" if gate_plot_note is None else gate_plot_note,
            "unit_plot": "generated" if unit_plot_note is None else unit_plot_note,
            "policy": "beta_extraction_is_theory_native_no_ppn_summary_inputs",
        },
        "outputs_private": [_safe_rel(p, _ROOT) for p in produced if p != out_json],
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    synced = _sync_to_public(produced, private_root=out_dir, public_root=public_dir)
    payload["outputs_public"] = [
        _safe_rel(p, _ROOT)
        for p in synced
        if p.name != out_json.name
    ]
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    synced_json = _sync_to_public([out_json], private_root=out_dir, public_root=public_dir)
    synced.extend(synced_json)

    append_event(
        {
            "event": "run_script",
            "script": "scripts/mercury/messenger_beta_stage_a_ingest_audit.py",
            "phase_step": "8.7.48.1",
            "status": overall,
            "input": _safe_rel(data_root, _ROOT),
            "outputs": [_safe_rel(p, _ROOT) for p in produced],
            "metrics": {
                "overall_status": overall,
                "ingestion_pass": gate_counts.get("pass", 0),
                "ingestion_watch": gate_counts.get("watch", 0),
                "ingestion_reject": gate_counts.get("reject", 0),
                "unit_pass": unit_counts.get("pass", 0),
                "unit_watch": unit_counts.get("watch", 0),
                "unit_reject": unit_counts.get("reject", 0),
            },
        }
    )

    print(f"[ok] stage_a_overall={overall}")
    print(f"[ok] wrote: {out_inventory_csv}")
    print(f"[ok] wrote: {out_unit_csv}")
    print(f"[ok] wrote: {out_json}")
    if gate_plot_note is None:
        print(f"[ok] wrote: {out_gate_pdf}")
        print(f"[ok] wrote: {out_gate_png}")
    else:
        print(f"[warn] ingestion plot skipped: {gate_plot_note}")

    if unit_plot_note is None:
        print(f"[ok] wrote: {out_unit_pdf}")
        print(f"[ok] wrote: {out_unit_png}")
    else:
        print(f"[warn] unit plot skipped: {unit_plot_note}")

    print(f"[ok] synced_to_public={len(synced)}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
