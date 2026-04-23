#!/usr/bin/env python3
"""
cassini_odf_media_state_join_from_ancillary.py

PDS SCE1 ancillary（ION/TRO）から抽出した command window を用いて、
ODF raw の各観測行に media state（ion/tro active）を結合し、
record-level media_correction_state が再構成可能かを監査する。

目的:
- 8.7.43.2: 時刻・局ID・band を使った record-level media state 再構成フックを固定する。
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# 関数: `_repo_root` の入出力契約と処理意図を定義する。
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# 関数: `_to_epoch` の入出力契約と処理意図を定義する。

def _to_epoch(iso_text: str) -> Optional[float]:
    s = str(iso_text).strip()
    if not s:
        return None

    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    return float(dt.timestamp())


# 関数: `_safe_int` の入出力契約と処理意図を定義する。

def _safe_int(value: object) -> Optional[int]:
    try:
        return int(str(value).strip())
    except Exception:
        return None


# 関数: `_station_to_complex` の入出力契約と処理意図を定義する。

def _station_to_complex(station_id: Optional[int]) -> Optional[int]:
    if station_id is None:
        return None

    # DSN station-to-complex routing used in SCE1 era (minimal mapping for media commands C10/C40/C60).

    mapping = {
        10: {12, 13, 14, 15, 16, 17, 18, 19, 24, 25, 26},
        40: {33, 34, 35, 36, 43, 45},
        60: {53, 54, 55, 63, 65},
    }
    for complex_id, stations in mapping.items():
        if int(station_id) in stations:
            return int(complex_id)

    return None


# 関数: `_load_windows` の入出力契約と処理意図を定義する。

def _load_windows(path: Path) -> Dict[Tuple[str, int], Dict[str, List[float]]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing command windows CSV: {path}")

    grouped: Dict[Tuple[str, int], List[Tuple[float, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            medium = str(r.get("medium") or "").strip().lower()
            complex_id = _safe_int(r.get("dsn_complex"))
            ts0 = _to_epoch(str(r.get("start_utc") or ""))
            ts1 = _to_epoch(str(r.get("stop_utc") or ""))
            if medium not in ("ion", "tro") or complex_id is None or ts0 is None or ts1 is None:
                continue

            if ts1 < ts0:
                ts0, ts1 = ts1, ts0

            key = (medium, int(complex_id))
            grouped.setdefault(key, []).append((float(ts0), float(ts1)))

    indexed: Dict[Tuple[str, int], Dict[str, List[float]]] = {}
    for key, spans in grouped.items():
        spans = sorted(spans, key=lambda x: (x[0], x[1]))
        starts = [s[0] for s in spans]
        stops = [s[1] for s in spans]
        indexed[key] = {"starts": starts, "stops": stops}

    return indexed


# 関数: `_covered` の入出力契約と処理意図を定義する。

def _covered(ts: float, starts: Sequence[float], stops: Sequence[float]) -> bool:
    if not starts or not stops:
        return False

    pos = bisect.bisect_right(starts, ts) - 1
    if pos < 0:
        return False

    return bool(ts <= stops[pos])


# 関数: `_write_summary_csv` の入出力契約と処理意図を定義する。

def _write_summary_csv(path: Path, payload: Dict[str, object]) -> None:
    rows = [
        ("status", str(payload.get("status") or "")),
        ("record_level_media_state_extractable", str(bool(payload.get("record_level_media_state_extractable"))).lower()),
        ("total_rows", str(int(payload.get("total_rows") or 0))),
        ("station_mapped_rows", str(int(payload.get("station_mapped_rows") or 0))),
        ("ion_covered_rows", str(int(payload.get("ion_covered_rows") or 0))),
        ("tro_covered_rows", str(int(payload.get("tro_covered_rows") or 0))),
        ("both_covered_rows", str(int(payload.get("both_covered_rows") or 0))),
        ("any_covered_rows", str(int(payload.get("any_covered_rows") or 0))),
        ("coverage_any_ratio", str(payload.get("coverage_any_ratio"))),
        ("coverage_both_ratio", str(payload.get("coverage_both_ratio"))),
        ("coverage_on_mapped_ratio", str(payload.get("coverage_on_mapped_ratio"))),
        ("threshold_coverage_any", str(payload.get("threshold_coverage_any"))),
        ("threshold_coverage_on_mapped", str(payload.get("threshold_coverage_on_mapped"))),
        ("watch_reason", str(payload.get("watch_reason") or "")),
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for k, v in rows:
            writer.writerow([k, v])


# 関数: `_sync_public` の入出力契約と処理意図を定義する。

def _sync_public(root: Path, names: Sequence[str]) -> None:
    src_dir = root / "output" / "cassini"
    dst_dir = root / "output" / "public" / "cassini"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for n in names:
        src = src_dir / n
        if src.exists():
            shutil.copy2(src, dst_dir / n)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Join ODF rows with ancillary media command windows.")
    ap.add_argument(
        "--odf-csv",
        type=Path,
        default=root / "output" / "cassini" / "cassini_sce1_odf_observed_raw.csv",
        help="ODF observed raw CSV.",
    )
    ap.add_argument(
        "--windows-csv",
        type=Path,
        default=root / "output" / "cassini" / "cassini_csp_media_command_windows.csv",
        help="Command windows CSV generated from ancillary files.",
    )
    ap.add_argument(
        "--coverage-any-threshold",
        type=float,
        default=0.95,
        help="Threshold for any-coverage ratio (ion or tro).",
    )
    ap.add_argument(
        "--coverage-mapped-threshold",
        type=float,
        default=0.95,
        help="Threshold for coverage ratio on rows that can be station->complex mapped.",
    )
    args = ap.parse_args()

    odf_csv = Path(args.odf_csv)
    windows_csv = Path(args.windows_csv)
    if not odf_csv.exists():
        raise FileNotFoundError(f"Missing ODF observed CSV: {odf_csv}")

    windows = _load_windows(windows_csv)

    total_rows = 0
    station_mapped_rows = 0
    ion_covered_rows = 0
    tro_covered_rows = 0
    both_covered_rows = 0
    any_covered_rows = 0
    state_counts: Dict[str, int] = {}
    with odf_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            total_rows += 1
            ts = _to_epoch(str(r.get("time_utc") or ""))
            station_rx = _safe_int(r.get("station_rx"))
            complex_id = _station_to_complex(station_rx)
            if ts is None or complex_id is None:
                state = "unmapped_station_or_time"
                state_counts[state] = state_counts.get(state, 0) + 1
                continue

            station_mapped_rows += 1
            ion_idx = windows.get(("ion", int(complex_id)), {})
            tro_idx = windows.get(("tro", int(complex_id)), {})
            ion_active = _covered(
                float(ts),
                ion_idx.get("starts", []) if isinstance(ion_idx, dict) else [],
                ion_idx.get("stops", []) if isinstance(ion_idx, dict) else [],
            )
            tro_active = _covered(
                float(ts),
                tro_idx.get("starts", []) if isinstance(tro_idx, dict) else [],
                tro_idx.get("stops", []) if isinstance(tro_idx, dict) else [],
            )
            ion_covered_rows += int(bool(ion_active))
            tro_covered_rows += int(bool(tro_active))
            both_covered_rows += int(bool(ion_active and tro_active))
            any_covered_rows += int(bool(ion_active or tro_active))

            if ion_active and tro_active:
                state = "ion_tro"
            elif ion_active:
                state = "ion_only"
            elif tro_active:
                state = "tro_only"
            else:
                state = "none"

            state_counts[state] = state_counts.get(state, 0) + 1

    cov_any = float(any_covered_rows / total_rows) if total_rows > 0 else math.nan
    cov_both = float(both_covered_rows / total_rows) if total_rows > 0 else math.nan
    cov_on_mapped = (
        float(any_covered_rows / station_mapped_rows) if station_mapped_rows > 0 else math.nan
    )
    th_any = float(args.coverage_any_threshold)
    th_mapped = float(args.coverage_mapped_threshold)
    extractable = bool(
        total_rows > 0
        and math.isfinite(cov_any)
        and math.isfinite(cov_on_mapped)
        and cov_any >= th_any
        and cov_on_mapped >= th_mapped
    )
    status = "pass" if extractable else "watch"
    watch_reason = ""
    if not extractable:
        watch_reason = (
            f"record-level media join coverage below threshold: any={cov_any:.6f} (th={th_any:.2f}), "
            f"mapped={cov_on_mapped:.6f} (th={th_mapped:.2f})"
        )

    payload: Dict[str, object] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "odf_csv": str(odf_csv),
            "windows_csv": str(windows_csv),
        },
        "status": status,
        "record_level_media_state_extractable": bool(extractable),
        "total_rows": int(total_rows),
        "station_mapped_rows": int(station_mapped_rows),
        "ion_covered_rows": int(ion_covered_rows),
        "tro_covered_rows": int(tro_covered_rows),
        "both_covered_rows": int(both_covered_rows),
        "any_covered_rows": int(any_covered_rows),
        "coverage_any_ratio": float(cov_any) if math.isfinite(cov_any) else None,
        "coverage_both_ratio": float(cov_both) if math.isfinite(cov_both) else None,
        "coverage_on_mapped_ratio": float(cov_on_mapped) if math.isfinite(cov_on_mapped) else None,
        "threshold_coverage_any": float(th_any),
        "threshold_coverage_on_mapped": float(th_mapped),
        "state_counts": state_counts,
        "watch_reason": watch_reason,
    }

    out_dir = root / "output" / "cassini"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "cassini_odf_media_state_join_metrics.json"
    out_csv = out_dir / "cassini_odf_media_state_join_summary.csv"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_summary_csv(out_csv, payload)
    _sync_public(root, [out_json.name, out_csv.name])
    print("Wrote:", out_json)
    print("Wrote:", out_csv)
    print("Synced:", root / "output" / "public" / "cassini")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
