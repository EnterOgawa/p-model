#!/usr/bin/env python3
"""
messenger_beta_stage_e_tnf_replay.py

Roadmap Step 8.7.48.5 (Stage E TNF replay) の実装。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.mercury.messenger_beta_stage_d_joint_fit import (
    _aggregate_channel,
    _build_design_matrix,
    _fit_joint,
    _load_channel_csv,
    _make_plot,
    _sync_to_public,
)
from scripts.summary.worklog import append_event


# クラス: `FieldSpec` の責務と境界条件を定義する。
@dataclass
class FieldSpec:
    name: str
    abs_offset0: int
    length: int
    data_type: str


# クラス: `TableSpec` の責務と境界条件を定義する。

@dataclass
class TableSpec:
    dtype_id: int
    table_name: str
    offset: int
    records: int
    record_length: int
    fields: Dict[str, FieldSpec]


# クラス: `ScanRow` の責務と境界条件を定義する。

@dataclass
class ScanRow:
    source_file: str
    parse_status: str
    n_rows_raw: int
    n_rows_valid: int
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

def _collect_candidates(tnf_root: Path, max_files: int) -> List[Path]:
    if not tnf_root.exists():
        return []

    if tnf_root.is_file():
        return [tnf_root]

    rows: List[Path] = []
    for p in sorted(tnf_root.rglob("*_tnf.dat")):
        if not p.is_file():
            continue

        rows.append(p)
        if len(rows) >= int(max_files):
            break

    return rows


# 関数: `_parse_dtype_id` の入出力契約と処理意図を定義する。

def _parse_dtype_id(name: str) -> Optional[int]:
    text = str(name)
    key = "Data Type "
    i = text.find(key)
    if i < 0:
        return None

    j = i + len(key)
    digits: List[str] = []
    while j < len(text) and text[j].isdigit():
        digits.append(text[j])
        j += 1

    if len(digits) <= 0:
        return None

    try:
        return int("".join(digits))
    except Exception:
        return None


# 関数: `_build_table_specs` の入出力契約と処理意図を定義する。

def _build_table_specs(xml_root: ET.Element) -> Dict[int, TableSpec]:
    ns = {"pds": "http://pds.nasa.gov/pds4/pds/v1"}
    out: Dict[int, TableSpec] = {}
    for table in xml_root.findall(".//pds:Table_Binary", ns):
        table_name = str(table.findtext("pds:name", default="", namespaces=ns)).strip()
        dtype_id = _parse_dtype_id(table_name)
        if dtype_id is None:
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
            continue

        fields: Dict[str, FieldSpec] = {}
        for group in table.findall("pds:Record_Binary/pds:Group_Field_Binary", ns):
            group_loc_text = str(group.findtext("pds:group_location", default="", namespaces=ns)).strip()
            try:
                group_loc = int(group_loc_text)
            except Exception:
                continue

            for field in group.findall("pds:Field_Binary", ns):
                fname = str(field.findtext("pds:name", default="", namespaces=ns)).strip()
                floc_text = str(field.findtext("pds:field_location", default="", namespaces=ns)).strip()
                flen_text = str(field.findtext("pds:field_length", default="", namespaces=ns)).strip()
                ftype = str(field.findtext("pds:data_type", default="", namespaces=ns)).strip()
                if len(fname) <= 0:
                    continue

                try:
                    field_loc = int(floc_text)
                    field_len = int(flen_text)
                except Exception:
                    continue

                abs_offset0 = int(group_loc + field_loc - 2)
                fields[str(fname)] = FieldSpec(
                    name=str(fname),
                    abs_offset0=int(abs_offset0),
                    length=int(field_len),
                    data_type=str(ftype),
                )

        out[int(dtype_id)] = TableSpec(
            dtype_id=int(dtype_id),
            table_name=str(table_name),
            offset=int(offset),
            records=int(records),
            record_length=int(rec_len),
            fields=fields,
        )

    return out


# 関数: `_decode_field_value` の入出力契約と処理意図を定義する。

def _decode_field_value(rec: bytes, spec: FieldSpec) -> Optional[object]:
    b0 = int(spec.abs_offset0)
    b1 = int(spec.abs_offset0 + spec.length)
    if b0 < 0 or b1 > len(rec) or b0 >= b1:
        return None

    raw = rec[b0:b1]
    dt = str(spec.data_type)
    try:
        if dt == "ASCII_String":
            return raw.decode("ascii", errors="ignore").strip("\x00 ").strip()

        if dt == "UnsignedByte":
            return int(raw[0])

        if dt == "UnsignedMSB2":
            return int(int.from_bytes(raw, byteorder="big", signed=False))

        if dt == "UnsignedMSB4":
            return int(int.from_bytes(raw, byteorder="big", signed=False))

        if dt == "UnsignedMSB8":
            return int(int.from_bytes(raw, byteorder="big", signed=False))

        if dt == "SignedMSB2":
            return int(int.from_bytes(raw, byteorder="big", signed=True))

        if dt == "SignedMSB4":
            return int(int.from_bytes(raw, byteorder="big", signed=True))

        if dt == "SignedMSB8":
            return int(int.from_bytes(raw, byteorder="big", signed=True))

        if dt == "IEEE754MSBSingle":
            return float(struct.unpack(">f", raw)[0])

        if dt == "IEEE754MSBDouble":
            return float(struct.unpack(">d", raw)[0])
    except Exception:
        return None

    return None


# 関数: `_read_field` の入出力契約と処理意図を定義する。

def _read_field(rec: bytes, fields: Dict[str, FieldSpec], name: str) -> Optional[object]:
    spec = fields.get(str(name))
    if spec is None:
        return None

    return _decode_field_value(rec, spec)


# 関数: `_build_epoch` の入出力契約と処理意図を定義する。

def _build_epoch(year: object, doy: object, sec: object) -> Optional[datetime]:
    if year is None or doy is None or sec is None:
        return None

    try:
        y = int(year)
        d = int(doy)
        s = float(sec)
        dt0 = datetime(year=y, month=1, day=1, tzinfo=timezone.utc)
        return dt0 + timedelta(days=(d - 1), seconds=s)
    except Exception:
        return None


# 関数: `_decode_tnf_table_rows` の入出力契約と処理意図を定義する。

def _decode_tnf_table_rows(
    dat_bytes: bytes,
    table: TableSpec,
    source_file: str,
    observable_kind: str,
    doppler_abs_max_hz: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    max_rows = min(
        int(table.records),
        max(0, (len(dat_bytes) - int(table.offset)) // max(1, int(table.record_length))),
    )
    for i in range(max_rows):
        start = int(table.offset + i * table.record_length)
        rec = dat_bytes[start : start + int(table.record_length)]
        if len(rec) < int(table.record_length):
            continue

        year = _read_field(rec, table.fields, "year")
        doy = _read_field(rec, table.fields, "doy")
        sec = _read_field(rec, table.fields, "sec")
        epoch = _build_epoch(year, doy, sec)
        if epoch is None:
            continue

        dl_dss = _read_field(rec, table.fields, "dl_dss_id")
        ul_dss = _read_field(rec, table.fields, "ul_dss_id")
        station_id_obj = dl_dss if dl_dss is not None else ul_dss
        if station_id_obj is None:
            station_id = "unknown"
        else:
            try:
                station_id = str(int(station_id_obj))
            except Exception:
                station_id = str(station_id_obj)

        if str(observable_kind) == "doppler":
            raw_value = _read_field(rec, table.fields, "dop_resid")
            if raw_value is None:
                continue

            value = float(raw_value)
            if not math.isfinite(value):
                continue

            if abs(value) > float(doppler_abs_max_hz):
                continue

            row: Dict[str, object] = {
                "epoch_utc": epoch,
                "observable_value": value,
                "observable_kind": "doppler",
                "observable_unit": "Hz",
                "station_id": station_id,
                "link_type": "one-way",
                "source_file": source_file,
                "dtype_id": 1,
                "doppler_hz": value,
            }
            rows.append(row)
            continue

        raw_value = _read_field(rec, table.fields, "rng_obs")
        if raw_value is None:
            continue

        value = float(raw_value)
        if (not math.isfinite(value)) or value <= 0.0:
            continue

        row = {
            "epoch_utc": epoch,
            "observable_value": value,
            "observable_kind": "range",
            "observable_unit": "RU",
            "station_id": station_id,
            "link_type": "unknown",
            "source_file": source_file,
            "dtype_id": 7,
            "range_value": value,
        }
        rows.append(row)

    return rows


# 関数: `_normalize_single_tnf_binary` の入出力契約と処理意図を定義する。

def _normalize_single_tnf_binary(path: Path, doppler_abs_max_hz: float) -> Tuple[pd.DataFrame, pd.DataFrame, ScanRow]:
    xml_path = path.with_suffix(".xml")
    if not xml_path.exists():
        scan = ScanRow(
            source_file=_safe_rel(path, _ROOT),
            parse_status="watch",
            n_rows_raw=0,
            n_rows_valid=0,
            note="tnf_label_xml_missing",
        )
        return (pd.DataFrame(), pd.DataFrame(), scan)

    try:
        xml_root = ET.parse(xml_path).getroot()
    except Exception:
        scan = ScanRow(
            source_file=_safe_rel(path, _ROOT),
            parse_status="reject",
            n_rows_raw=0,
            n_rows_valid=0,
            note="tnf_label_parse_failed",
        )
        return (pd.DataFrame(), pd.DataFrame(), scan)

    table_specs = _build_table_specs(xml_root)
    table_dop = table_specs.get(1)
    table_rng = table_specs.get(7)
    if table_dop is None and table_rng is None:
        scan = ScanRow(
            source_file=_safe_rel(path, _ROOT),
            parse_status="watch",
            n_rows_raw=0,
            n_rows_valid=0,
            note="tnf_required_dtypes_missing",
        )
        return (pd.DataFrame(), pd.DataFrame(), scan)

    try:
        dat_bytes = path.read_bytes()
    except Exception:
        scan = ScanRow(
            source_file=_safe_rel(path, _ROOT),
            parse_status="reject",
            n_rows_raw=0,
            n_rows_valid=0,
            note="tnf_binary_read_failed",
        )
        return (pd.DataFrame(), pd.DataFrame(), scan)

    rel = _safe_rel(path, _ROOT)
    dop_rows: List[Dict[str, object]] = []
    rng_rows: List[Dict[str, object]] = []
    n_rows_raw = 0
    if table_dop is not None:
        n_rows_raw += int(table_dop.records)
        dop_rows = _decode_tnf_table_rows(
            dat_bytes=dat_bytes,
            table=table_dop,
            source_file=rel,
            observable_kind="doppler",
            doppler_abs_max_hz=float(doppler_abs_max_hz),
        )

    if table_rng is not None:
        n_rows_raw += int(table_rng.records)
        rng_rows = _decode_tnf_table_rows(
            dat_bytes=dat_bytes,
            table=table_rng,
            source_file=rel,
            observable_kind="range",
            doppler_abs_max_hz=float(doppler_abs_max_hz),
        )

    dop_df = pd.DataFrame(dop_rows)
    rng_df = pd.DataFrame(rng_rows)
    n_valid = int(len(dop_df) + len(rng_df))
    status = "pass" if n_valid > 0 else "watch"
    note = (
        f"tnf_decode_ok_dop={len(dop_df)}_rng={len(rng_df)}"
        if n_valid > 0
        else "tnf_decode_no_valid_rows"
    )
    scan = ScanRow(
        source_file=rel,
        parse_status=status,
        n_rows_raw=int(n_rows_raw),
        n_rows_valid=n_valid,
        note=note,
    )
    return (dop_df, rng_df, scan)


# 関数: `_write_scan_csv` の入出力契約と処理意図を定義する。

def _write_scan_csv(path: Path, rows: Sequence[ScanRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_file",
                "parse_status",
                "n_rows_raw",
                "n_rows_valid",
                "note",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "source_file": str(r.source_file),
                    "parse_status": str(r.parse_status),
                    "n_rows_raw": int(r.n_rows_raw),
                    "n_rows_valid": int(r.n_rows_valid),
                    "note": str(r.note),
                }
            )


# 関数: `_status_counts` の入出力契約と処理意図を定義する。

def _status_counts(rows: Sequence[ScanRow]) -> Dict[str, int]:
    out = {"pass": 0, "watch": 0, "reject": 0}
    for r in rows:
        key = str(r.parse_status)
        if key not in out:
            continue

        out[key] += 1

    return out


# 関数: `_write_coeff_csv` の入出力契約と処理意図を定義する。

def _write_coeff_csv(path: Path, labels: Sequence[str], coef: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["parameter", "value"])
        writer.writeheader()
        for i, name in enumerate(labels):
            writer.writerow({"parameter": str(name), "value": float(coef[i])})


# 関数: `_load_json_if_exists` の入出力契約と処理意図を定義する。

def _load_json_if_exists(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}

    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return {}

    return {}


# 関数: `_compare_with_odf` の入出力契約と処理意図を定義する。

def _compare_single_beta(
    *,
    component_name: str,
    odf_beta: object,
    odf_sigma: object,
    tnf_beta: object,
    tnf_sigma: object,
) -> Dict[str, object]:
    try:
        b0 = float(odf_beta)
        s0 = float(odf_sigma)
        bt = float(tnf_beta)
        st = float(tnf_sigma)
    except Exception:
        return {
            "status": "watch",
            "reason": "metrics_unavailable",
            "component": str(component_name),
            "odf_beta": odf_beta,
            "odf_sigma": odf_sigma,
            "tnf_beta": tnf_beta,
            "tnf_sigma": tnf_sigma,
        }

    denom = float(math.sqrt(max(0.0, s0 * s0 + st * st)))
    if denom <= 0.0:
        z = float("inf")
    else:
        z = float(abs(bt - b0) / denom)

    if not math.isfinite(z):
        status = "reject"
    elif z <= 2.0:
        status = "pass"
    elif z <= 5.0:
        status = "watch"
    else:
        status = "reject"

    return {
        "status": status,
        "reason": "z_delta_beta",
        "component": str(component_name),
        "z_delta_beta": z,
        "delta_beta": float(bt - b0),
        "odf_beta": b0,
        "odf_sigma": s0,
        "tnf_beta": bt,
        "tnf_sigma": st,
    }


# 関数: `_compare_with_odf` の入出力契約と処理意図を定義する。

def _compare_with_odf(
    *,
    tnf_beta_dyn: object,
    tnf_sigma_dyn: object,
    tnf_beta_lt: object,
    tnf_sigma_lt: object,
    beta_split_mode: str,
    odf_metrics: Dict[str, object],
) -> Dict[str, object]:
    cmp_dyn = _compare_single_beta(
        component_name="beta_dyn",
        odf_beta=odf_metrics.get("beta_dyn_estimate"),
        odf_sigma=odf_metrics.get("beta_sigma"),
        tnf_beta=tnf_beta_dyn,
        tnf_sigma=tnf_sigma_dyn,
    )
    if str(beta_split_mode).strip().lower() == "split":
        cmp_lt = _compare_single_beta(
            component_name="beta_lt",
            odf_beta=odf_metrics.get("beta_lt_estimate"),
            odf_sigma=odf_metrics.get("beta_lt_sigma"),
            tnf_beta=tnf_beta_lt,
            tnf_sigma=tnf_sigma_lt,
        )
    else:
        cmp_lt = {
            "status": "pass",
            "reason": "not_applicable_coupled_mode",
            "component": "beta_lt",
        }

    status = "pass"
    for cmp_obj in (cmp_dyn, cmp_lt):
        st = str(cmp_obj.get("status", "watch"))
        if st == "reject":
            status = "reject"
            break

        if st == "watch":
            status = "watch"

    return {
        "status": status,
        "reason": "componentwise_z_delta_beta",
        "z_delta_beta": cmp_dyn.get("z_delta_beta"),
        "delta_beta": cmp_dyn.get("delta_beta"),
        "replay_vs_odf_beta_dyn": cmp_dyn,
        "replay_vs_odf_beta_lt": cmp_lt,
    }


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    ap = argparse.ArgumentParser(description="Roadmap 8.7.48.5: TNF replay for Stage D joint-fit I/F.")
    ap.add_argument("--data-root", type=str, default=str(_ROOT / "data" / "mercury" / "messenger"))
    ap.add_argument("--tnf-root", type=str, default="")
    ap.add_argument("--max-files", type=int, default=200)
    ap.add_argument("--doppler-abs-max-hz", type=float, default=1.0e6)
    ap.add_argument("--tnf-doppler-csv", type=str, default="")
    ap.add_argument("--tnf-range-csv", type=str, default="")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument(
        "--odf-stage-d-metrics",
        type=str,
        default=str(_ROOT / "output" / "public" / "mercury" / "messenger_beta_stage_d_joint_metrics.json"),
    )
    ap.add_argument("--out-dir", type=str, default=str(_ROOT / "output" / "private" / "mercury"))
    ap.add_argument("--public-dir", type=str, default=str(_ROOT / "output" / "public" / "mercury"))
    ap.add_argument("--doppler-bin-minutes", type=int, default=30)
    ap.add_argument("--range-bin-minutes", type=int, default=30)
    ap.add_argument("--min-joint-rows", type=int, default=300)
    ap.add_argument("--max-station-bias-per-channel", type=int, default=8)
    ap.add_argument("--orbital-period-days", type=float, default=87.9691)
    ap.add_argument("--sigma-watch-threshold", type=float, default=0.1)
    ap.add_argument(
        "--beta-split-mode",
        type=str,
        choices=("auto", "coupled", "split"),
        default="auto",
        help="Beta parameterization mode for TNF replay. 'auto' follows ODF Stage D metrics.",
    )
    args = ap.parse_args()

    data_root = _resolve_path(args.data_root, _ROOT)
    tnf_root = _resolve_path(args.tnf_root, _ROOT) if str(args.tnf_root).strip() else (data_root / "data-tnf")
    derived_root = data_root / "derived"
    derived_root.mkdir(parents=True, exist_ok=True)
    tnf_doppler_csv = _resolve_path(args.tnf_doppler_csv, _ROOT) if str(args.tnf_doppler_csv).strip() else (
        derived_root / "tnf_doppler_observations.csv"
    )
    tnf_range_csv = _resolve_path(args.tnf_range_csv, _ROOT) if str(args.tnf_range_csv).strip() else (
        derived_root / "tnf_range_observations.csv"
    )
    odf_metrics_json = _resolve_path(args.odf_stage_d_metrics, _ROOT)
    odf_metrics_obj = _load_json_if_exists(odf_metrics_json)
    out_dir = _resolve_path(args.out_dir, _ROOT)
    public_dir = _resolve_path(args.public_dir, _ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_scan_csv = out_dir / "messenger_tnf_extract_file_scan.csv"
    out_extract_metrics_json = out_dir / "messenger_tnf_extract_metrics.json"
    out_summary_csv = out_dir / "messenger_beta_stage_e_tnf_replay_summary.csv"
    out_coeff_csv = out_dir / "messenger_beta_stage_e_tnf_replay_coefficients.csv"
    out_resid_csv = out_dir / "messenger_beta_stage_e_tnf_replay_residuals.csv"
    out_scale_csv = out_dir / "messenger_beta_stage_e_tnf_replay_channel_scales.csv"
    out_compare_csv = out_dir / "messenger_beta_stage_e_tnf_replay_vs_odf.csv"
    out_metrics_json = out_dir / "messenger_beta_stage_e_tnf_replay_metrics.json"
    out_plot_pdf = out_dir / "messenger_beta_stage_e_tnf_replay_fit.pdf"
    out_plot_png = out_dir / "messenger_beta_stage_e_tnf_replay_fit.png"

    scan_rows: List[ScanRow] = []
    if not bool(args.skip_extract):
        candidates = _collect_candidates(tnf_root=tnf_root, max_files=int(args.max_files))
        dop_frames: List[pd.DataFrame] = []
        rng_frames: List[pd.DataFrame] = []
        for p in candidates:
            dop_df, rng_df, scan = _normalize_single_tnf_binary(path=p, doppler_abs_max_hz=float(args.doppler_abs_max_hz))
            scan_rows.append(scan)
            if len(dop_df) > 0:
                dop_frames.append(dop_df)

            if len(rng_df) > 0:
                rng_frames.append(rng_df)

        if len(dop_frames) > 0:
            dop_all = pd.concat(dop_frames, ignore_index=True).sort_values("epoch_utc").reset_index(drop=True)
        else:
            dop_all = pd.DataFrame(columns=["epoch_utc", "observable_value", "observable_kind", "observable_unit", "station_id", "link_type", "source_file", "dtype_id", "doppler_hz"])

        if len(rng_frames) > 0:
            rng_all = pd.concat(rng_frames, ignore_index=True).sort_values("epoch_utc").reset_index(drop=True)
        else:
            rng_all = pd.DataFrame(columns=["epoch_utc", "observable_value", "observable_kind", "observable_unit", "station_id", "link_type", "source_file", "dtype_id", "range_value"])

        dop_all.to_csv(tnf_doppler_csv, index=False)
        rng_all.to_csv(tnf_range_csv, index=False)
        _write_scan_csv(out_scan_csv, scan_rows)
        extract_payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "phase_step": "8.7.48.5_extract",
            "overall_status": "pass" if int(len(dop_all) + len(rng_all)) > 0 else "watch",
            "tnf_root": _safe_rel(tnf_root, _ROOT),
            "n_candidate_files": int(len(candidates)),
            "n_scan_rows": int(len(scan_rows)),
            "scan_status_counts": _status_counts(scan_rows),
            "n_rows_doppler": int(len(dop_all)),
            "n_rows_range": int(len(rng_all)),
            "tnf_doppler_csv": _safe_rel(tnf_doppler_csv, _ROOT),
            "tnf_range_csv": _safe_rel(tnf_range_csv, _ROOT),
            "scan_csv": _safe_rel(out_scan_csv, _ROOT),
            "doppler_abs_max_hz": float(args.doppler_abs_max_hz),
        }
        out_extract_metrics_json.write_text(json.dumps(extract_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        extract_payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "phase_step": "8.7.48.5_extract",
            "overall_status": "watch",
            "reason": "skip_extract",
            "tnf_doppler_csv": _safe_rel(tnf_doppler_csv, _ROOT),
            "tnf_range_csv": _safe_rel(tnf_range_csv, _ROOT),
        }
        out_extract_metrics_json.write_text(json.dumps(extract_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if (not tnf_doppler_csv.exists()) or (not tnf_range_csv.exists()):
        payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "phase_step": "8.7.48.5",
            "overall_status": "reject",
            "reason": "tnf_input_missing",
            "tnf_doppler_csv": _safe_rel(tnf_doppler_csv, _ROOT),
            "tnf_range_csv": _safe_rel(tnf_range_csv, _ROOT),
            "extract_metrics": _safe_rel(out_extract_metrics_json, _ROOT),
        }
        out_metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        produced = [out_extract_metrics_json, out_metrics_json]
        if out_scan_csv.exists():
            produced.append(out_scan_csv)

        synced = _sync_to_public(produced, private_root=out_dir, public_root=public_dir)
        append_event(
            {
                "event": "run_script",
                "script": "scripts/mercury/messenger_beta_stage_e_tnf_replay.py",
                "phase_step": "8.7.48.5",
                "status": "reject",
                "input": f"{_safe_rel(tnf_doppler_csv, _ROOT)}|{_safe_rel(tnf_range_csv, _ROOT)}",
                "outputs": [_safe_rel(p, _ROOT) for p in produced],
                "metrics": {"reason": "tnf_input_missing"},
            }
        )
        print("[warn] Stage E skipped: TNF input CSV missing.")
        print(f"[ok] wrote: {out_metrics_json}")
        print(f"[ok] synced_to_public={len(synced)}")
        return 0

    doppler_df = _load_channel_csv(tnf_doppler_csv, channel="doppler")
    range_df = _load_channel_csv(tnf_range_csv, channel="range")
    doppler_agg = _aggregate_channel(doppler_df, bin_minutes=int(args.doppler_bin_minutes))
    range_agg = _aggregate_channel(range_df, bin_minutes=int(args.range_bin_minutes))
    joint_df = pd.concat([range_agg, doppler_agg], ignore_index=True).sort_values("epoch_utc").reset_index(drop=True)

    beta_split_mode_req = str(args.beta_split_mode).strip().lower()
    if beta_split_mode_req == "auto":
        odf_split_mode = str(odf_metrics_obj.get("beta_split_mode", "coupled")).strip().lower()
        if odf_split_mode not in {"coupled", "split"}:
            odf_split_mode = "coupled"

        beta_split_mode = odf_split_mode
    else:
        beta_split_mode = beta_split_mode_req

    X, y_norm, y_obs, labels, meta, work = _build_design_matrix(
        joint_df,
        orbital_period_days=float(args.orbital_period_days),
        max_station_bias_per_channel=int(args.max_station_bias_per_channel),
        split_beta_lt=(beta_split_mode == "split"),
    )
    channels = work["channel"].astype(str).to_numpy()
    fit, coef, fit_norm, residual_norm = _fit_joint(
        X=X,
        y_norm=y_norm,
        y_obs=y_obs,
        scale_by_row=work["scale_by_row"].to_numpy(dtype=float),
        labels=labels,
        channels=channels,
        min_rows=int(args.min_joint_rows),
        sigma_watch_threshold=float(args.sigma_watch_threshold),
    )

    work["fit_scaled"] = fit_norm
    work["value_scaled"] = y_norm
    work["residual_norm"] = residual_norm
    work.to_csv(out_resid_csv, index=False)
    _write_coeff_csv(out_coeff_csv, labels=labels, coef=coef)

    scale_df = pd.DataFrame([{"channel": "range", "scale": float(meta.get("scale_range", 1.0))}, {"channel": "doppler", "scale": float(meta.get("scale_doppler", 1.0))}])
    scale_df.to_csv(out_scale_csv, index=False)

    replay_cmp = _compare_with_odf(
        tnf_beta_dyn=float(fit.beta_dyn),
        tnf_sigma_dyn=float(fit.beta_sigma),
        tnf_beta_lt=float(fit.beta_lt),
        tnf_sigma_lt=float(fit.beta_lt_sigma),
        beta_split_mode=str(fit.beta_split_mode),
        odf_metrics=odf_metrics_obj,
    )
    compare_row = {
        "status": replay_cmp.get("status"),
        "reason": replay_cmp.get("reason"),
        "z_delta_beta_dyn": replay_cmp.get("replay_vs_odf_beta_dyn", {}).get("z_delta_beta"),
        "delta_beta_dyn": replay_cmp.get("replay_vs_odf_beta_dyn", {}).get("delta_beta"),
        "status_beta_dyn": replay_cmp.get("replay_vs_odf_beta_dyn", {}).get("status"),
        "z_delta_beta_lt": replay_cmp.get("replay_vs_odf_beta_lt", {}).get("z_delta_beta"),
        "delta_beta_lt": replay_cmp.get("replay_vs_odf_beta_lt", {}).get("delta_beta"),
        "status_beta_lt": replay_cmp.get("replay_vs_odf_beta_lt", {}).get("status"),
    }
    pd.DataFrame([compare_row]).to_csv(out_compare_csv, index=False)

    if fit.status_data == "reject":
        overall = "reject"
    else:
        cmp_status = str(replay_cmp.get("status", "watch"))
        if cmp_status == "reject":
            overall = "reject"
        elif fit.status_sigma == "watch" or cmp_status == "watch":
            overall = "watch"
        else:
            overall = "watch"

    summary = pd.DataFrame(
        [
            {
                "phase_step": "8.7.48.5",
                "overall_status": overall,
                "beta_dyn": fit.beta_dyn,
                "beta_sigma": fit.beta_sigma,
                "beta_z_from_1": fit.beta_z_from_1,
                "beta_lt": fit.beta_lt,
                "beta_lt_sigma": fit.beta_lt_sigma,
                "beta_lt_z_from_1": fit.beta_lt_z_from_1,
                "beta_split_mode": fit.beta_split_mode,
                "beta_dyn_lt_delta": fit.beta_dyn_lt_delta,
                "beta_dyn_lt_consistency_status": fit.beta_dyn_lt_consistency_status,
                "rss_norm": fit.rss_norm,
                "dof": fit.dof,
                "n_rows": fit.n_rows,
                "n_range_rows": fit.n_range_rows,
                "n_doppler_rows": fit.n_doppler_rows,
                "rms_range": fit.rms_range,
                "rms_doppler": fit.rms_doppler,
                "status_data": fit.status_data,
                "status_sigma": fit.status_sigma,
                "status_replay_vs_odf": replay_cmp.get("status", "watch"),
                "doppler_bin_minutes": int(args.doppler_bin_minutes),
                "range_bin_minutes": int(args.range_bin_minutes),
            }
        ]
    )
    summary.to_csv(out_summary_csv, index=False)

    plot_note = _make_plot(work, out_pdf=out_plot_pdf, out_png=out_plot_png)
    produced: List[Path] = [out_extract_metrics_json, out_summary_csv, out_coeff_csv, out_resid_csv, out_scale_csv, out_compare_csv, out_metrics_json]
    if out_scan_csv.exists():
        produced.append(out_scan_csv)

    if plot_note is None:
        produced.extend([out_plot_pdf, out_plot_png])

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase_step": "8.7.48.5",
        "overall_status": overall,
        "data_root": _safe_rel(data_root, _ROOT),
        "tnf_root": _safe_rel(tnf_root, _ROOT),
        "tnf_doppler_csv": _safe_rel(tnf_doppler_csv, _ROOT),
        "tnf_range_csv": _safe_rel(tnf_range_csv, _ROOT),
        "n_rows_joint": fit.n_rows,
        "n_rows_range": fit.n_range_rows,
        "n_rows_doppler": fit.n_doppler_rows,
        "beta_dyn_estimate": fit.beta_dyn,
        "beta_sigma": fit.beta_sigma,
        "beta_z_from_1": fit.beta_z_from_1,
        "beta_lt_estimate": fit.beta_lt,
        "beta_lt_sigma": fit.beta_lt_sigma,
        "beta_lt_z_from_1": fit.beta_lt_z_from_1,
        "beta_split_mode": fit.beta_split_mode,
        "beta_dyn_lt_delta": fit.beta_dyn_lt_delta,
        "beta_dyn_lt_consistency_z": fit.beta_dyn_lt_consistency_z,
        "beta_dyn_lt_consistency_status": fit.beta_dyn_lt_consistency_status,
        "rss_norm": fit.rss_norm,
        "dof": int(fit.dof),
        "rms_range": fit.rms_range,
        "rms_doppler": fit.rms_doppler,
        "status_components": {
            "data": fit.status_data,
            "sigma": fit.status_sigma,
            "model": "watch",
            "replay_vs_odf": replay_cmp.get("status", "watch"),
            "replay_vs_odf_beta_dyn": replay_cmp.get("replay_vs_odf_beta_dyn", {}).get("status", "watch"),
            "replay_vs_odf_beta_lt": replay_cmp.get("replay_vs_odf_beta_lt", {}).get("status", "watch"),
        },
        "replay_vs_odf": replay_cmp,
        "joint_meta": meta,
        "extract_metrics": _load_json_if_exists(out_extract_metrics_json),
        "gating_policy": {
            "min_joint_rows": int(args.min_joint_rows),
            "sigma_watch_threshold": float(args.sigma_watch_threshold),
            "replay_pass_z_threshold": 2.0,
            "replay_watch_z_threshold": 5.0,
            "model_status_cap": "watch_until_stage_f",
            "beta_split_mode": fit.beta_split_mode,
            "beta_split_mode_request": str(args.beta_split_mode),
        },
        "plot": "generated" if plot_note is None else plot_note,
        "outputs_private": [_safe_rel(p, _ROOT) for p in produced if p != out_metrics_json],
    }
    out_metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    synced = _sync_to_public(produced, private_root=out_dir, public_root=public_dir)
    payload["outputs_public"] = [_safe_rel(p, _ROOT) for p in synced if p.name != out_metrics_json.name]
    out_metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _sync_to_public([out_metrics_json], private_root=out_dir, public_root=public_dir)

    append_event(
        {
            "event": "run_script",
            "script": "scripts/mercury/messenger_beta_stage_e_tnf_replay.py",
            "phase_step": "8.7.48.5",
            "status": overall,
            "input": f"{_safe_rel(tnf_doppler_csv, _ROOT)}|{_safe_rel(tnf_range_csv, _ROOT)}",
            "outputs": [_safe_rel(p, _ROOT) for p in produced],
            "metrics": {"n_rows_joint": fit.n_rows, "beta_dyn": fit.beta_dyn, "beta_sigma": fit.beta_sigma, "beta_z_from_1": fit.beta_z_from_1, "status_replay_vs_odf": replay_cmp.get("status", "watch"), "z_delta_beta": replay_cmp.get("z_delta_beta")},
        }
    )

    print(f"[ok] stage_e_overall={overall}")
    print(f"[ok] n_rows_joint={fit.n_rows} (range={fit.n_range_rows}, doppler={fit.n_doppler_rows})")
    print(f"[ok] beta_dyn={fit.beta_dyn:.8f} sigma={fit.beta_sigma:.8f} z={fit.beta_z_from_1:.4f}")
    if fit.beta_split_mode == "split":
        print(
            f"[ok] beta_lt={fit.beta_lt:.8f} sigma={fit.beta_lt_sigma:.8f} "
            f"z={fit.beta_lt_z_from_1:.4f} dyn_minus_lt={fit.beta_dyn_lt_delta:.8f} "
            f"status={fit.beta_dyn_lt_consistency_status}"
        )

    if "z_delta_beta" in replay_cmp:
        print(f"[ok] replay_z_delta_beta={float(replay_cmp['z_delta_beta']):.4f} status={replay_cmp.get('status')}")
    else:
        print(f"[warn] replay compare note={replay_cmp.get('reason', 'n/a')}")

    print(f"[ok] wrote: {out_summary_csv}")
    print(f"[ok] wrote: {out_metrics_json}")
    if plot_note is None:
        print(f"[ok] wrote: {out_plot_pdf}")
        print(f"[ok] wrote: {out_plot_png}")
    else:
        print(f"[warn] plot skipped: {plot_note}")

    print(f"[ok] synced_to_public={len(synced)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
