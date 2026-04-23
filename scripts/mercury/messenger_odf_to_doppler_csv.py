#!/usr/bin/env python3
"""
messenger_odf_to_doppler_csv.py

Roadmap Step 8.7.48.2/8.7.48.3 の補助I/F:
ODF 由来ファイル群から Stage B/Stage C 入力の正規化観測CSVを生成する。

目的:
- ODF parser 実装の前段として、観測テーブル抽出の入出力契約を固定する。
- observable mode（doppler/range/all）を切り替え、同一I/Fで再利用可能にする。
- 実データ未配置時も reject 理由を機械可読で残し、次作業を明確化する。
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary.worklog import append_event


# クラス: `ScanRow` の責務と境界条件を定義する。
@dataclass
class ScanRow:
    source_file: str
    parse_status: str
    n_rows_raw: int
    n_rows_valid: int
    epoch_col: str
    doppler_col: str
    note: str


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


# 関数: `_collect_candidates` の入出力契約と処理意図を定義する。

def _collect_candidates(odf_root: Path, max_files: int) -> List[Path]:
    if not odf_root.exists():
        return []

    if odf_root.is_file():
        return [odf_root]

    allowed_ext = {".csv", ".tab", ".tsv", ".txt", ".dat"}
    rows: List[Path] = []
    for p in odf_root.rglob("*"):
        if not p.is_file():
            continue

        if p.suffix.lower() not in allowed_ext:
            continue

        rows.append(p)
        if len(rows) >= max_files:
            break

    return rows


# 関数: `_parse_yyyymmdd_hhmmss` の入出力契約と処理意図を定義する。

def _parse_yyyymmdd_hhmmss(date_int: int, time_int: int) -> Optional[datetime]:
    d = f"{int(date_int):08d}"
    t = f"{int(time_int):06d}"
    try:
        return datetime(
            year=int(d[0:4]),
            month=int(d[4:6]),
            day=int(d[6:8]),
            hour=int(t[0:2]),
            minute=int(t[2:4]),
            second=int(t[4:6]),
            tzinfo=timezone.utc,
        )
    except Exception:
        return None


# 関数: `_extract_bits_msb` の入出力契約と処理意図を定義する。

def _extract_bits_msb(word: int, start_bit: int, stop_bit: int, total_bits: int = 32) -> int:
    width = int(stop_bit) - int(start_bit) + 1
    shift = int(total_bits) - int(stop_bit)
    mask = (1 << width) - 1
    return (int(word) >> shift) & mask


# 関数: `_find_odf_table` の入出力契約と処理意図を定義する。

def _find_odf_table(root: ET.Element, name: str) -> Optional[Tuple[int, int, int]]:
    ns = {"pds": "http://pds.nasa.gov/pds4/pds/v1"}
    for table in root.findall(".//pds:Table_Binary", ns):
        table_name = str(table.findtext("pds:name", default="", namespaces=ns)).strip()
        if table_name != name:
            continue

        offset_text = str(table.findtext("pds:offset", default="", namespaces=ns)).strip()
        records_text = str(table.findtext("pds:records", default="", namespaces=ns)).strip()
        rec_len_text = str(
            table.findtext("pds:Record_Binary/pds:record_length", default="", namespaces=ns)
        ).strip()
        try:
            offset = int(offset_text)
            records = int(records_text)
            rec_len = int(rec_len_text)
        except Exception:
            return None

        return (offset, records, rec_len)

    return None


# 関数: `_link_type_from_dtype` の入出力契約と処理意図を定義する。

def _link_type_from_dtype(dtype_id: int) -> str:
    if int(dtype_id) == 11:
        return "one-way"

    if int(dtype_id) == 12:
        return "two-way"

    if int(dtype_id) == 13:
        return "three-way"

    return "unknown"


# 関数: `_dtype_mode_kind` の入出力契約と処理意図を定義する。

def _dtype_mode_kind(dtype_id: int) -> str:
    if int(dtype_id) in (11, 12, 13):
        return "doppler"

    if int(dtype_id) in (37, 41):
        return "range"

    return "unknown"


# 関数: `_dtype_mode_unit` の入出力契約と処理意図を定義する。

def _dtype_mode_unit(dtype_id: int) -> str:
    if int(dtype_id) in (11, 12, 13):
        return "Hz"

    if int(dtype_id) == 37:
        return "RU"

    if int(dtype_id) == 41:
        return "ns"

    return "unknown"


# 関数: `_mode_accept_dtype` の入出力契約と処理意図を定義する。

def _mode_accept_dtype(dtype_id: int, observable_mode: str) -> bool:
    mode = str(observable_mode).strip().lower()
    kind = _dtype_mode_kind(dtype_id)
    if mode == "all":
        return kind in ("doppler", "range")

    return kind == mode


# 関数: `_mode_missing_reason` の入出力契約と処理意図を定義する。

def _mode_missing_reason(observable_mode: str) -> str:
    mode = str(observable_mode).strip().lower()
    if mode == "range":
        return "no_parseable_odf_range_files"

    if mode == "all":
        return "no_parseable_odf_observable_files"

    return "no_parseable_odf_doppler_files"


# 関数: `_normalize_single_odf_binary` の入出力契約と処理意図を定義する。

def _normalize_single_odf_binary(path: Path, observable_mode: str) -> Tuple[pd.DataFrame, ScanRow]:
    xml_path = path.with_suffix(".xml")
    if not xml_path.exists():
        return (
            pd.DataFrame(),
            ScanRow(
                source_file=_safe_rel(path, _ROOT),
                parse_status="watch",
                n_rows_raw=0,
                n_rows_valid=0,
                epoch_col="",
                doppler_col="",
                note="odf_label_xml_missing",
            ),
        )

    try:
        xml_root = ET.parse(xml_path).getroot()
    except Exception:
        return (
            pd.DataFrame(),
            ScanRow(
                source_file=_safe_rel(path, _ROOT),
                parse_status="reject",
                n_rows_raw=0,
                n_rows_valid=0,
                epoch_col="",
                doppler_col="",
                note="odf_label_parse_failed",
            ),
        )

    file_label_table = _find_odf_table(xml_root, "ODF File Label Group Data")
    orbit_data_table = _find_odf_table(xml_root, "ODF Orbit Data Group Data")
    if file_label_table is None or orbit_data_table is None:
        return (
            pd.DataFrame(),
            ScanRow(
                source_file=_safe_rel(path, _ROOT),
                parse_status="reject",
                n_rows_raw=0,
                n_rows_valid=0,
                epoch_col="",
                doppler_col="",
                note="odf_tables_not_found",
            ),
        )

    try:
        raw = path.read_bytes()
    except Exception:
        return (
            pd.DataFrame(),
            ScanRow(
                source_file=_safe_rel(path, _ROOT),
                parse_status="reject",
                n_rows_raw=0,
                n_rows_valid=0,
                epoch_col="",
                doppler_col="",
                note="odf_binary_read_failed",
            ),
        )

    file_offset, _file_records, file_rec_len = file_label_table
    if len(raw) < file_offset + file_rec_len:
        return (
            pd.DataFrame(),
            ScanRow(
                source_file=_safe_rel(path, _ROOT),
                parse_status="reject",
                n_rows_raw=0,
                n_rows_valid=0,
                epoch_col="",
                doppler_col="",
                note="odf_file_label_out_of_range",
            ),
        )

    file_rec = raw[file_offset : file_offset + file_rec_len]
    ref_date = int.from_bytes(file_rec[28:32], byteorder="big", signed=False)
    ref_time = int.from_bytes(file_rec[32:36], byteorder="big", signed=False)
    ref_dt = _parse_yyyymmdd_hhmmss(ref_date, ref_time)
    if ref_dt is None:
        return (
            pd.DataFrame(),
            ScanRow(
                source_file=_safe_rel(path, _ROOT),
                parse_status="reject",
                n_rows_raw=0,
                n_rows_valid=0,
                epoch_col="",
                doppler_col="",
                note="odf_reference_datetime_invalid",
            ),
        )

    orbit_offset, orbit_records, orbit_rec_len = orbit_data_table
    max_rows = min(int(orbit_records), max(0, (len(raw) - orbit_offset) // max(1, orbit_rec_len)))
    rows: List[Dict[str, object]] = []
    for i in range(max_rows):
        start = orbit_offset + i * orbit_rec_len
        rec = raw[start : start + orbit_rec_len]
        if len(rec) < orbit_rec_len:
            continue

        sec_int = int.from_bytes(rec[0:4], byteorder="big", signed=False)
        item2_3 = int.from_bytes(rec[4:8], byteorder="big", signed=False)
        observable_i = int.from_bytes(rec[8:12], byteorder="big", signed=True)
        observable_f = int.from_bytes(rec[12:16], byteorder="big", signed=True)
        item6_14 = int.from_bytes(rec[16:20], byteorder="big", signed=False)

        ms_frac = _extract_bits_msb(item2_3, 1, 10)
        rx_station_id = _extract_bits_msb(item6_14, 4, 10)
        dtype_id = _extract_bits_msb(item6_14, 20, 25)
        valid_flag = _extract_bits_msb(item6_14, 32, 32)
        if int(valid_flag) != 0:
            continue

        mode_kind = _dtype_mode_kind(dtype_id)
        if mode_kind == "unknown":
            continue

        if not _mode_accept_dtype(dtype_id, observable_mode=observable_mode):
            continue

        link_type = _link_type_from_dtype(dtype_id)
        epoch = ref_dt + timedelta(seconds=float(sec_int), milliseconds=float(ms_frac))
        obs_value = float(observable_i) + float(observable_f) * 1e-9
        obs_unit = _dtype_mode_unit(dtype_id)
        row: Dict[str, object] = {
            "epoch_utc": epoch,
            "observable_value": obs_value,
            "observable_kind": mode_kind,
            "observable_unit": obs_unit,
            "station_id": str(int(rx_station_id)),
            "link_type": link_type,
            "source_file": _safe_rel(path, _ROOT),
            "dtype_id": int(dtype_id),
        }
        if mode_kind == "doppler":
            row["doppler_hz"] = obs_value

        if mode_kind == "range":
            row["range_value"] = obs_value

        rows.append(
            row
        )

    work = pd.DataFrame(rows)
    if len(work) <= 0:
        return (
            pd.DataFrame(),
            ScanRow(
                source_file=_safe_rel(path, _ROOT),
                parse_status="watch",
                n_rows_raw=max_rows,
                n_rows_valid=0,
                epoch_col="odf_binary_time_tag",
                doppler_col="odf_binary_observable",
                note=f"odf_binary_no_valid_rows_mode={str(observable_mode).strip().lower()}",
            ),
        )

    drop_cols = ["epoch_utc", "observable_value"]
    keep_cols = [c for c in drop_cols if c in work.columns]
    work = work.dropna(subset=keep_cols).reset_index(drop=True)
    return (
        work,
        ScanRow(
            source_file=_safe_rel(path, _ROOT),
            parse_status="pass",
            n_rows_raw=max_rows,
            n_rows_valid=int(len(work)),
            epoch_col="odf_binary_time_tag",
            doppler_col="odf_binary_observable",
            note=f"odf_binary_ok_mode={str(observable_mode).strip().lower()}_ref={ref_date:08d}_{ref_time:06d}",
        ),
    )


# 関数: `_detect_epoch_column` の入出力契約と処理意図を定義する。

def _detect_epoch_column(columns: Sequence[str]) -> Optional[str]:
    lowers = {c.lower(): c for c in columns}
    for key in ("epoch_utc", "time_utc", "utc", "epoch", "time", "timestamp", "date_time"):
        if key in lowers:
            return lowers[key]

    return None


# 関数: `_detect_doppler_column` の入出力契約と処理意図を定義する。

def _detect_doppler_column(columns: Sequence[str]) -> Optional[str]:
    lowers = {c.lower(): c for c in columns}
    keys = (
        "doppler_hz",
        "doppler",
        "doppler_residual_hz",
        "freq_hz",
        "frequency_hz",
        "frequency",
    )
    for key in keys:
        if key in lowers:
            return lowers[key]

    return None


# 関数: `_detect_station_column` の入出力契約と処理意図を定義する。

def _detect_station_column(columns: Sequence[str]) -> Optional[str]:
    lowers = {c.lower(): c for c in columns}
    for key in ("station_id", "station", "dss", "antenna", "complex"):
        if key in lowers:
            return lowers[key]

    return None


# 関数: `_detect_link_column` の入出力契約と処理意図を定義する。

def _detect_link_column(columns: Sequence[str]) -> Optional[str]:
    lowers = {c.lower(): c for c in columns}
    for key in ("link_type", "way", "two_three_way", "mode"):
        if key in lowers:
            return lowers[key]

    return None


# 関数: `_read_candidate_table` の入出力契約と処理意図を定義する。

def _read_candidate_table(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        return None

    if len(df.columns) <= 1:
        return None

    return df


# 関数: `_parse_link_value` の入出力契約と処理意図を定義する。

def _parse_link_value(value: object) -> str:
    text = str(value or "").strip().lower()
    if "3" in text and "way" in text:
        return "three-way"

    if "2" in text and "way" in text:
        return "two-way"

    if "three" in text:
        return "three-way"

    if "two" in text:
        return "two-way"

    return "unknown"


# 関数: `_normalize_single_file` の入出力契約と処理意図を定義する。

def _normalize_single_file(path: Path, observable_mode: str) -> Tuple[pd.DataFrame, ScanRow]:
    if path.suffix.lower() == ".dat":
        return _normalize_single_odf_binary(path, observable_mode=observable_mode)

    df = _read_candidate_table(path)
    if df is None:
        return (
            pd.DataFrame(),
            ScanRow(
                source_file=_safe_rel(path, _ROOT),
                parse_status="reject",
                n_rows_raw=0,
                n_rows_valid=0,
                epoch_col="",
                doppler_col="",
                note="table_read_failed_or_not_delimited",
            ),
        )

    epoch_col = _detect_epoch_column(df.columns.tolist())
    doppler_col = _detect_doppler_column(df.columns.tolist())
    if epoch_col is None or doppler_col is None:
        return (
            pd.DataFrame(),
            ScanRow(
                source_file=_safe_rel(path, _ROOT),
                parse_status="watch",
                n_rows_raw=int(len(df)),
                n_rows_valid=0,
                epoch_col=str(epoch_col or ""),
                doppler_col=str(doppler_col or ""),
                note="required_columns_not_found",
            ),
        )

    station_col = _detect_station_column(df.columns.tolist())
    link_col = _detect_link_column(df.columns.tolist())

    work = pd.DataFrame()
    work["epoch_utc"] = pd.to_datetime(df[epoch_col], utc=True, errors="coerce")
    work["doppler_hz"] = pd.to_numeric(df[doppler_col], errors="coerce")
    if station_col is not None:
        work["station_id"] = df[station_col].astype(str)
    else:
        work["station_id"] = "unknown"

    if link_col is not None:
        work["link_type"] = df[link_col].map(_parse_link_value)
    else:
        work["link_type"] = "unknown"

    work["source_file"] = _safe_rel(path, _ROOT)
    work["observable_value"] = work["doppler_hz"]
    work["observable_kind"] = "doppler"
    work["observable_unit"] = "Hz"
    work["dtype_id"] = -1
    work = work.dropna(subset=["epoch_utc", "doppler_hz"]).reset_index(drop=True)
    note = "ok"
    status = "pass"
    if len(work) <= 0:
        note = "no_valid_rows_after_parse"
        status = "reject"

    return (
        work,
        ScanRow(
            source_file=_safe_rel(path, _ROOT),
            parse_status=status,
            n_rows_raw=int(len(df)),
            n_rows_valid=int(len(work)),
            epoch_col=epoch_col,
            doppler_col=doppler_col,
            note=note,
        ),
    )


# 関数: `_write_scan_csv` の入出力契約と処理意図を定義する。

def _write_scan_csv(path: Path, rows: Sequence[ScanRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_file",
        "parse_status",
        "n_rows_raw",
        "n_rows_valid",
        "epoch_col",
        "doppler_col",
        "note",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "source_file": row.source_file,
                    "parse_status": row.parse_status,
                    "n_rows_raw": row.n_rows_raw,
                    "n_rows_valid": row.n_rows_valid,
                    "epoch_col": row.epoch_col,
                    "doppler_col": row.doppler_col,
                    "note": row.note,
                }
            )


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
        description="Roadmap 8.7.48.2 helper: normalize ODF-like tabular files to Stage-B Doppler CSV."
    )
    ap.add_argument(
        "--data-root",
        type=str,
        default=str(_ROOT / "data" / "mercury" / "messenger"),
        help="MESSENGER data root.",
    )
    ap.add_argument(
        "--odf-root",
        type=str,
        default="",
        help="ODF root directory; defaults to <data-root>/data-odf.",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default="",
        help=(
            "Normalized observable CSV output path; defaults to "
            "<data-root>/derived/odf_{mode}_observations.csv."
        ),
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(_ROOT / "output" / "private" / "mercury"),
        help="Private output directory for metrics.",
    )
    ap.add_argument(
        "--public-dir",
        type=str,
        default=str(_ROOT / "output" / "public" / "mercury"),
        help="Public sync directory for metrics.",
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=2000,
        help="Maximum number of candidate files to scan.",
    )
    ap.add_argument(
        "--observable-mode",
        type=str,
        default="doppler",
        choices=["doppler", "range", "all"],
        help="Observable extraction mode.",
    )
    args = ap.parse_args()

    data_root = _resolve_path(args.data_root, _ROOT)
    odf_root = _resolve_path(args.odf_root, _ROOT) if str(args.odf_root).strip() else (data_root / "data-odf")
    mode = str(args.observable_mode).strip().lower()
    out_csv = _resolve_path(args.out_csv, _ROOT) if str(args.out_csv).strip() else (
        data_root / "derived" / f"odf_{mode}_observations.csv"
    )
    out_dir = _resolve_path(args.out_dir, _ROOT)
    public_dir = _resolve_path(args.public_dir, _ROOT)

    out_dir.mkdir(parents=True, exist_ok=True)
    if mode == "range":
        out_scan_csv = out_dir / "messenger_odf_to_range_file_scan.csv"
        out_metrics_json = out_dir / "messenger_odf_to_range_metrics.json"
    elif mode == "all":
        out_scan_csv = out_dir / "messenger_odf_to_observable_file_scan.csv"
        out_metrics_json = out_dir / "messenger_odf_to_observable_metrics.json"
    else:
        out_scan_csv = out_dir / "messenger_odf_to_doppler_file_scan.csv"
        out_metrics_json = out_dir / "messenger_odf_to_doppler_metrics.json"

    candidates = _collect_candidates(odf_root, max_files=int(args.max_files))
    scan_rows: List[ScanRow] = []
    normalized_parts: List[pd.DataFrame] = []
    for candidate in candidates:
        norm, scan = _normalize_single_file(candidate, observable_mode=mode)
        scan_rows.append(scan)
        if len(norm) > 0:
            normalized_parts.append(norm)

    _write_scan_csv(out_scan_csv, scan_rows)
    status = "reject"
    reason = "odf_input_missing"
    rows_out = 0
    if len(candidates) > 0:
        reason = _mode_missing_reason(mode)
        if len(normalized_parts) > 0:
            merged = pd.concat(normalized_parts, ignore_index=True)
            dedup_cols = [c for c in ("epoch_utc", "source_file", "dtype_id", "observable_value") if c in merged.columns]
            merged = merged.drop_duplicates(subset=dedup_cols).sort_values("epoch_utc")
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            merged.to_csv(out_csv, index=False)
            rows_out = int(len(merged))
            if rows_out > 0:
                status = "pass"
                reason = "ok"

    metrics: Dict[str, object] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase_step": "8.7.48.2",
        "status": status,
        "reason": reason,
        "observable_mode": mode,
        "data_root": _safe_rel(data_root, _ROOT),
        "odf_root": _safe_rel(odf_root, _ROOT),
        "normalized_csv": _safe_rel(out_csv, _ROOT),
        "n_candidate_files": int(len(candidates)),
        "n_scan_rows": int(len(scan_rows)),
        "n_rows_out": rows_out,
        "scan_status_counts": {
            "pass": int(sum(1 for r in scan_rows if r.parse_status == "pass")),
            "watch": int(sum(1 for r in scan_rows if r.parse_status == "watch")),
            "reject": int(sum(1 for r in scan_rows if r.parse_status == "reject")),
        },
        "outputs_private": [
            _safe_rel(out_scan_csv, _ROOT),
            _safe_rel(out_metrics_json, _ROOT),
        ],
    }
    out_metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    produced = [out_scan_csv, out_metrics_json]
    synced = _sync_to_public(produced, private_root=out_dir, public_root=public_dir)
    metrics["outputs_public"] = [_safe_rel(p, _ROOT) for p in synced]
    out_metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    _sync_to_public([out_metrics_json], private_root=out_dir, public_root=public_dir)

    append_event(
        {
            "event": "run_script",
            "script": "scripts/mercury/messenger_odf_to_doppler_csv.py",
            "phase_step": "8.7.48.2",
            "status": status,
            "input": _safe_rel(odf_root, _ROOT),
            "outputs": [_safe_rel(p, _ROOT) for p in produced],
            "metrics": {
                "reason": reason,
                "observable_mode": mode,
                "n_candidate_files": int(len(candidates)),
                "n_rows_out": rows_out,
            },
        }
    )
    print(f"[ok] status={status} reason={reason}")
    print(f"[ok] wrote: {out_scan_csv}")
    print(f"[ok] wrote: {out_metrics_json}")
    print(f"[ok] synced_to_public={len(synced)}")
    if status == "pass":
        print(f"[ok] wrote normalized csv: {out_csv}")

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
