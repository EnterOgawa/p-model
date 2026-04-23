#!/usr/bin/env python3
"""Build ODF-raw normalization interface manifest for absolute-beta promotion.

Purpose:
- Freeze real-data interface values for ODF raw direct-beta fit.
- Provide an explicit manifest for the four keys introduced in Part II/IV:
  1) band_id
  2) doppler_sign_convention
  3) media_correction_state
  4) time_scale_id

Inputs:
- output/cassini/cassini_sce1_odf_observed_raw.csv
- data/cassini/pds_sce1/**/odf/*.lbl

Outputs:
- output/cassini/cassini_odf_raw_if_manifest.json
- output/cassini/cassini_odf_raw_if_manifest.csv
- output/cassini/cassini_odf_media_correction_field_map.json
- output/cassini/cassini_odf_media_correction_field_map.csv
- output/cassini/cassini_odf_external_csp_availability.json
- output/cassini/cassini_odf_external_csp_availability.csv
- synced copies in output/public/cassini/
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# 関数: `_repo_root` の入出力契約と処理意図を定義する。
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# 関数: `_safe_int` の入出力契約と処理意図を定義する。

def _safe_int(value: object) -> Optional[int]:
    try:
        return int(float(value))
    except Exception:
        return None


# 関数: `_first_match_int` の入出力契約と処理意図を定義する。

def _first_match_int(pattern: str, text: str) -> Optional[int]:
    m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not m:
        return None

    return _safe_int(m.group(1))


# 関数: `_first_match_text` の入出力契約と処理意図を定義する。

def _first_match_text(pattern: str, text: str) -> str:
    m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not m:
        return ""

    return str(m.group(1)).strip()


# 関数: `_read_label_meta` の入出力契約と処理意図を定義する。

# 関数: `_read_odf1b_reference_datetime` の入出力契約と処理意図を定義する。

def _read_odf1b_reference_datetime(odf_path: Path, odf1b_record_1based: Optional[int]) -> Tuple[Optional[int], Optional[int]]:
    if odf1b_record_1based is None or odf1b_record_1based < 1:
        return None, None

    rec_bytes = 36
    data = odf_path.read_bytes()
    idx = int(odf1b_record_1based) - 1
    start = idx * rec_bytes
    stop = start + rec_bytes
    if start < 0 or stop > len(data):
        return None, None

    rec = data[start:stop]
    # ODF1B layout: Item20 (FILE REFERENCE DATE) starts at byte 29, Item21 at byte 33 (1-based).
    ref_date = int.from_bytes(rec[28:32], "big", signed=False)
    ref_time = int.from_bytes(rec[32:36], "big", signed=False)
    return int(ref_date), int(ref_time)


# 関数: `_read_label_meta` の入出力契約と処理意図を定義する。

def _read_label_meta(label_path: Path, odf_path: Path) -> Dict[str, object]:
    txt = label_path.read_text(encoding="utf-8", errors="replace")
    odf1b_record = _first_match_int(
        r"^\s*\^ODF1B_TABLE\s*=\s*\(\"[^\"]+\"\s*,\s*([0-9]+)\s*\)\s*$",
        txt,
    )
    ref_date, ref_time = _read_odf1b_reference_datetime(odf_path, odf1b_record)
    return {
        "label_path": str(label_path),
        "odf1b_record_1based": odf1b_record,
        "file_reference_date": ref_date,
        "file_reference_time": ref_time,
        "start_time": _first_match_text(r"^\s*START_TIME\s*=\s*([0-9T:\-\.]+)\s*$", txt),
        "stop_time": _first_match_text(r"^\s*STOP_TIME\s*=\s*([0-9T:\-\.]+)\s*$", txt),
    }


# 関数: `_sorted_unique_ints` の入出力契約と処理意図を定義する。

def _sorted_unique_ints(values: Iterable[object]) -> List[int]:
    out: set[int] = set()
    for v in values:
        n = _safe_int(v)
        if n is not None:
            out.add(n)

    return sorted(out)


# 関数: `_status_from_bool` の入出力契約と処理意図を定義する。

def _status_from_bool(flag: bool) -> str:
    return "pass" if bool(flag) else "watch"


# 関数: `_format_list` の入出力契約と処理意図を定義する。

def _format_list(values: Sequence[object]) -> str:
    return ",".join([str(v) for v in values]) if values else "-"


# 関数: `_load_observed_rows` の入出力契約と処理意図を定義する。

def _load_observed_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing observed CSV: {path}. Run cassini_fig2_overlay.py with --source pds_odf_raw first."
        )

    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({str(k): ("" if v is None else str(v)) for k, v in dict(r).items()})

    if not rows:
        raise RuntimeError(f"Observed CSV has no rows: {path}")

    return rows


# 関数: `_build_label_meta_map` の入出力契約と処理意図を定義する。

def _build_label_meta_map(pds_root: Path, source_files: Sequence[str]) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for rel in sorted(set(source_files)):
        if not rel:
            continue

        odf_path = pds_root / Path(*rel.split("/"))
        lbl_lower = odf_path.with_suffix(".lbl")
        lbl_upper = odf_path.with_suffix(".LBL")
        label_path = lbl_lower if lbl_lower.exists() else lbl_upper if lbl_upper.exists() else None
        if label_path is None:
            out[rel] = {
                "label_path": "",
                "file_reference_date": None,
                "file_reference_time": None,
                "start_time": "",
                "stop_time": "",
                "status": "missing_label",
            }
            continue

        meta = _read_label_meta(label_path, odf_path)
        meta["status"] = "ok"
        out[rel] = meta

    return out


# 関数: `_read_text_safe` の入出力契約と処理意図を定義する。

def _read_text_safe(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


# 関数: `_collect_doc_paths` の入出力契約と処理意図を定義する。

def _collect_doc_paths(pds_root: Path, pattern: str) -> List[str]:
    return sorted([str(p) for p in pds_root.rglob(pattern)])


# 関数: `_read_json_if_exists` の入出力契約と処理意図を定義する。

def _read_json_if_exists(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    return payload if isinstance(payload, dict) else {}


# 関数: `_read_sign_recordlevel_reaudit` の入出力契約と処理意図を定義する。

def _read_sign_recordlevel_reaudit(path: Path) -> Dict[str, object]:
    payload = _read_json_if_exists(path)
    record_level = payload.get("record_level_sign_audit") if isinstance(payload.get("record_level_sign_audit"), dict) else {}
    arc_gate = (
        record_level.get("arc_stability_terminal_gate")
        if isinstance(record_level.get("arc_stability_terminal_gate"), dict)
        else {}
    )
    selected = arc_gate.get("selected_interval") if isinstance(arc_gate.get("selected_interval"), dict) else {}
    cfg = arc_gate.get("config") if isinstance(arc_gate.get("config"), dict) else {}
    reasons = arc_gate.get("reasons") if isinstance(arc_gate.get("reasons"), list) else []
    window_reassess = (
        payload.get("window_quality_reassessment")
        if isinstance(payload.get("window_quality_reassessment"), dict)
        else {}
    )
    promotion = payload.get("promotion_gates") if isinstance(payload.get("promotion_gates"), dict) else {}
    return {
        "path": str(path),
        "available": bool(payload),
        "status": str(arc_gate.get("status") or ""),
        "gate_pass": bool(arc_gate.get("gate_pass")),
        "reasons": [str(v) for v in reasons if str(v).strip()],
        "selected_interval": selected,
        "selected_interval_coverage_ratio_points": selected.get("coverage_ratio_points"),
        "selected_interval_n_arcs": selected.get("n_arcs"),
        "min_contiguous_arcs": cfg.get("min_contiguous_arcs"),
        "min_coverage_ratio_points": cfg.get("min_coverage_ratio_points"),
        "sign_closed_best_window_quality": bool(promotion.get("sign_closed_best_window_quality")),
        "sign_closed_all_record_levels": bool(promotion.get("sign_closed_all_record_levels")),
        "window_best_scenario": (
            window_reassess.get("best_scenario") if isinstance(window_reassess.get("best_scenario"), dict) else {}
        ),
    }


# 関数: `_extract_csp_name_tokens` の入出力契約と処理意図を定義する。

def _extract_csp_name_tokens(text: str) -> List[str]:
    pat = r"\b(?:IONCAL|TROPCAL|PLSMCAL)_[A-Z0-9_]+\.CSP\b"
    hits = re.findall(pat, text.upper(), flags=re.IGNORECASE)
    return sorted(set([str(h).upper() for h in hits if str(h).strip()]))


# 関数: `_scan_external_csp_availability` の入出力契約と処理意図を定義する。

def _scan_external_csp_availability(pds_root: Path, root: Path) -> Dict[str, object]:
    csp_files = _collect_doc_paths(pds_root, "*.csp")
    csp_files.extend(_collect_doc_paths(pds_root, "*.CSP"))
    csp_files = sorted(set(csp_files))
    medium_counts_local = {"IONCAL": 0, "TROPCAL": 0, "PLSMCAL": 0}
    for rel in csp_files:
        up = Path(rel).name.upper()
        for medium in medium_counts_local:
            if medium in up:
                medium_counts_local[medium] += 1

    trk_paths = _collect_doc_paths(pds_root, "trk_2_23_*.txt")
    casrssis_paths = _collect_doc_paths(pds_root, "casrssis.txt")
    dors_paths = _collect_doc_paths(pds_root, "dors_002_020501.txt")
    doc_paths = sorted(set(trk_paths + casrssis_paths + dors_paths))
    route_paths: List[str] = []
    csp_name_hits: List[str] = []
    medium_mentions_doc = {"IONCAL": 0, "TROPCAL": 0, "PLSMCAL": 0}
    for rel in doc_paths:
        txt = _read_text_safe(Path(rel))
        up = txt.upper()
        has_tsac = "TSAC" in up
        has_ftp = "FTP" in up or "FILE TRANSFER PROTOCOL" in up
        if has_tsac and has_ftp:
            route_paths.append(rel)

        csp_name_hits.extend(_extract_csp_name_tokens(up))
        for medium in medium_mentions_doc:
            medium_mentions_doc[medium] += int(up.count(medium))

    ancillary_manifest_path = root / "data" / "cassini" / "sources" / "csp_media_ancillary" / "media_ancillary_manifest.json"
    ancillary = _read_json_if_exists(ancillary_manifest_path)
    ancillary_mediums = ancillary.get("medium_availability") if isinstance(ancillary.get("medium_availability"), dict) else {}
    ancillary_all_required = bool(ancillary.get("all_required_mediums_local")) if ancillary else False
    ancillary_windows_n = int(ancillary.get("command_windows_n") or 0) if ancillary else 0
    for medium in medium_counts_local:
        if bool(ancillary_mediums.get(medium)):
            medium_counts_local[medium] += 1

    required_mediums = ["IONCAL", "TROPCAL", "PLSMCAL"]
    local_mediums_observed = sorted([k for k, v in medium_counts_local.items() if int(v) > 0])
    all_required_mediums_local = all(int(medium_counts_local[m]) > 0 for m in required_mediums)
    local_payload_complete = bool(all_required_mediums_local)
    hard_watch_required = not local_payload_complete
    public_state = "route_documented_but_payload_missing"
    if local_payload_complete and csp_files:
        public_state = "local_csp_payload_complete"
    elif local_payload_complete and ancillary_all_required:
        public_state = "pds_ancillary_payload_complete"

    hard_watch_reason = ""
    if hard_watch_required:
        hard_watch_reason = (
            "IONCAL/TROPCAL/PLSMCAL payload is incomplete in local/public bundle; "
            "only TSAC/FTP route and partial media artifacts are available."
        )

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "required_mediums": required_mediums,
        "local_csp_file_count": int(len(csp_files)),
        "local_csp_paths": csp_files,
        "local_csp_medium_counts": medium_counts_local,
        "local_mediums_observed": local_mediums_observed,
        "all_required_mediums_local": bool(all_required_mediums_local),
        "ancillary_manifest_path": str(ancillary_manifest_path),
        "ancillary_manifest_available": bool(bool(ancillary)),
        "ancillary_medium_availability": ancillary_mediums,
        "ancillary_all_required_mediums_local": bool(ancillary_all_required),
        "ancillary_command_windows_n": int(ancillary_windows_n),
        "source_documents": {
            "trk_2_23_paths": trk_paths,
            "casrssis_paths": casrssis_paths,
            "dors_002_020501_paths": dors_paths,
            "route_confirmed_paths": sorted(set(route_paths)),
        },
        "doc_csp_name_samples": sorted(set(csp_name_hits))[:20],
        "doc_medium_mentions": medium_mentions_doc,
        "delivery_route_documented": bool(route_paths),
        "public_reproducibility_state": public_state,
        "hard_watch_required": bool(hard_watch_required),
        "hard_watch_reason": hard_watch_reason,
        "next_action": (
            "Acquire CSP payload files from primary source, cache to data/cassini/sources, and rerun ODF media join hook."
            if hard_watch_required
            else "Local CSP payload is complete; run record-level media join hook."
        ),
    }


# 関数: `_build_external_csp_join_hook` の入出力契約と処理意図を定義する。

def _build_external_csp_join_hook(
    observed_rows: Sequence[Dict[str, str]],
    external_csp: Dict[str, object],
    join_metrics: Dict[str, object],
    join_metrics_path: Path,
) -> Dict[str, object]:
    required_join_keys = [
        "time_utc",
        "station_rx",
        "station_tx",
        "downlink_band_id",
        "uplink_band_id",
        "data_type_id",
        "source_file",
    ]
    available_keys = sorted(set(observed_rows[0].keys())) if observed_rows else []
    missing_keys = sorted([k for k in required_join_keys if k not in available_keys])
    payload_ready = bool(external_csp.get("all_required_mediums_local"))
    join_metrics_available = bool(join_metrics)
    join_metrics_extractable = bool(join_metrics.get("record_level_media_state_extractable")) if join_metrics else False
    join_ready_candidate = bool(join_metrics_extractable) if join_metrics_available else bool(payload_ready)
    join_ready = bool(payload_ready and not missing_keys and join_ready_candidate)
    hook_mode = "external_csp_join_waiting_payload"
    if join_ready:
        hook_mode = "external_csp_join_ready"
    elif payload_ready and missing_keys:
        hook_mode = "external_csp_join_blocked_missing_keys"
    elif payload_ready and join_metrics_available and not join_metrics_extractable:
        hook_mode = "external_csp_join_insufficient_coverage"
    elif payload_ready and not join_metrics_available:
        hook_mode = "external_csp_join_pending_metrics"

    watch_reason = ""
    if not join_ready:
        if not payload_ready:
            watch_reason = str(external_csp.get("hard_watch_reason") or "")
        elif join_metrics_available and not join_metrics_extractable:
            watch_reason = str(join_metrics.get("watch_reason") or "record-level media join coverage is below threshold")
        else:
            watch_reason = f"Required join keys are missing from observed CSV: {','.join(missing_keys)}"

    return {
        "required_join_keys": required_join_keys,
        "available_join_keys": available_keys,
        "missing_join_keys": missing_keys,
        "payload_ready_for_join": bool(payload_ready),
        "record_level_media_state_extractable_via_csp": bool(join_ready),
        "hook_mode": hook_mode,
        "join_metrics_available": bool(join_metrics_available),
        "join_metrics_json": str(join_metrics_path) if join_metrics_available else "",
        "join_metrics_record_level_extractable": bool(join_metrics_extractable),
        "join_metrics_coverage_any_ratio": join_metrics.get("coverage_any_ratio") if join_metrics_available else None,
        "join_metrics_coverage_on_mapped_ratio": join_metrics.get("coverage_on_mapped_ratio") if join_metrics_available else None,
        "watch_reason": watch_reason,
    }


# 関数: `_extract_table_blocks` の入出力契約と処理意図を定義する。

def _extract_table_blocks(label_text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    pattern = r"OBJECT\s*=\s*(ODF3C_TABLE|ODF4[AB][0-9]{2}_TABLE)\s*(.*?)END_OBJECT\s*=\s*\1"
    for m in re.finditer(pattern, label_text, flags=re.IGNORECASE | re.DOTALL):
        table = str(m.group(1)).upper()
        block = str(m.group(2))
        prev = out.get(table, "")
        out[table] = prev if len(prev) >= len(block) else block

    return out


# 関数: `_extract_name_fields` の入出力契約と処理意図を定義する。

def _extract_name_fields(table_block: str) -> List[str]:
    names = []
    for m in re.finditer(r'^\s*NAME\s*=\s*"([^"]+)"\s*$', table_block, flags=re.IGNORECASE | re.MULTILINE):
        name = str(m.group(1)).strip()
        if not name:
            continue

        names.append(name)

    return sorted(set(names))


# 関数: `_classify_media_field` の入出力契約と処理意図を定義する。

def _classify_media_field(field_name: str) -> str:
    up = field_name.upper()
    direct_tokens = ["MEDIA", "CALIBRATION", "IONO", "TROPO", "PLASMA", "CHPART", "NUPART", "DRVID", "MODEL"]
    proxy_tokens = ["DELAY", "FREQUENCY", "BAND", "STATION", "VALIDITY", "TIME TAG"]

    for token in direct_tokens:
        if token in up:
            return "direct"

    for token in proxy_tokens:
        if token in up:
            return "proxy"

    return "none"


# 関数: `_build_media_crosswalk` の入出力契約と処理意図を定義する。

def _build_media_crosswalk(
    observed_rows: Sequence[Dict[str, str]],
    label_meta_map: Dict[str, Dict[str, object]],
    pds_root: Path,
    root: Path,
) -> Tuple[Dict[str, object], List[Dict[str, str]]]:
    table_fields: Dict[str, set[str]] = {}
    trk_2_18_reference_count = 0
    label_scanned = 0
    label_missing = 0

    for meta in label_meta_map.values():
        label_path = Path(str(meta.get("label_path") or ""))
        if not label_path.exists():
            label_missing += 1
            continue

        label_scanned += 1
        txt = _read_text_safe(label_path)
        trk_2_18_reference_count += txt.upper().count("TRK-2-18")
        blocks = _extract_table_blocks(txt)
        for table, block in blocks.items():
            fields = _extract_name_fields(block)
            bucket = table_fields.setdefault(table, set())
            bucket.update(fields)

    trk_2_23_paths = _collect_doc_paths(pds_root, "trk_2_23_*.txt")
    trk_2_18_paths = []
    trk_2_18_paths.extend(_collect_doc_paths(pds_root, "trk_2_18*.txt"))
    trk_2_18_paths.extend(_collect_doc_paths(pds_root, "trk-2-18*.txt"))
    trk_2_18_paths = sorted(set(trk_2_18_paths))

    trk_2_23_text = ""
    if trk_2_23_paths:
        trk_2_23_text = _read_text_safe(Path(trk_2_23_paths[0])).upper()

    token_patterns = {
        "adjust": "ADJUST",
        "delete": "DELETE",
        "model_chpart": "MODEL(CHPART)",
        "model_wet_nupart": "MODEL(WET NUPART)",
        "model_dry_nupart": "MODEL(DRY NUPART)",
        "model_drvid": "MODEL(DRVID)",
        "downlink_specifier": "DOWNLINK(",
        "doppler_range_data_type": "DOPRNG",
    }
    trk_tokens = {
        k: (trk_2_23_text.count(v) > 0 if trk_2_23_text else False)
        for k, v in token_patterns.items()
    }

    rows: List[Dict[str, str]] = []
    direct_fields: List[str] = []
    proxy_fields: List[str] = []
    table_proxy_counts: Dict[str, int] = {}
    for table in sorted(table_fields):
        for field in sorted(table_fields.get(table, set())):
            klass = _classify_media_field(field)
            if klass == "none":
                continue

            if klass == "direct":
                direct_fields.append(f"{table}:{field}")
                rows.append(
                    {
                        "table": table,
                        "field_name": field,
                        "mapping_role": "direct_media_flag_candidate",
                        "extractability": "candidate",
                        "note": "Field name directly contains media/calibration token.",
                    }
                )
                continue

            proxy_fields.append(f"{table}:{field}")
            table_proxy_counts[table] = table_proxy_counts.get(table, 0) + 1
            rows.append(
                {
                    "table": table,
                    "field_name": field,
                    "mapping_role": "proxy_only",
                    "extractability": "not_record_level_media_flag",
                    "note": "Useful for timing/frequency context but not explicit media-model state.",
                }
            )

    join_metrics_path = root / "output" / "cassini" / "cassini_odf_media_state_join_metrics.json"
    join_metrics = _read_json_if_exists(join_metrics_path)
    external_csp = _scan_external_csp_availability(pds_root, root)
    external_csp_join = _build_external_csp_join_hook(observed_rows, external_csp, join_metrics, join_metrics_path)
    trk_csp_confirmed = all(bool(v) for v in trk_tokens.values()) if trk_tokens else False
    direct_extractable = len(direct_fields) > 0
    csp_extractable = bool(external_csp_join.get("record_level_media_state_extractable_via_csp"))
    record_level_extractable = bool(direct_extractable or csp_extractable)
    parser_mode = "non_extractable_confirmed"
    if direct_extractable:
        parser_mode = "extractable_candidate"

    if csp_extractable:
        parser_mode = "external_csp_join_ready"

    status = "pass" if record_level_extractable else "watch"
    watch_reason = ""
    if not record_level_extractable:
        watch_reason = (
            "TRK media calibrations are external CSP adjustments and ODF3C/ODF4 fields expose only proxy context; "
            "record-level media-correction state is not explicitly encoded."
        )
        if bool(external_csp.get("hard_watch_required")):
            hard = str(external_csp.get("hard_watch_reason") or "")
            watch_reason = f"{watch_reason} {hard}".strip()

    crosswalk = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "label_count_scanned": int(label_scanned),
            "label_count_missing": int(label_missing),
        },
        "source_documents": {
            "trk_2_23_paths": trk_2_23_paths,
            "trk_2_18_paths_local": trk_2_18_paths,
            "trk_2_18_reference_count_in_labels": int(trk_2_18_reference_count),
            "trk_2_18_local_available": bool(trk_2_18_paths),
            "trk_2_18_note": (
                "TRK-2-18 is referenced in ODF FORMAT ID=1 notes, but local SCE1 bundle did not contain TRK-2-18."
                if not trk_2_18_paths
                else "TRK-2-18 local copy available."
            ),
        },
        "trk_2_23_csp_tokens": trk_tokens,
        "trk_2_23_csp_context_confirmed": bool(trk_csp_confirmed),
        "odf_tables_scanned": sorted(table_fields.keys()),
        "odf_proxy_field_counts": table_proxy_counts,
        "odf_direct_media_fields": sorted(set(direct_fields)),
        "odf_proxy_fields": sorted(set(proxy_fields)),
        "record_level_media_state_extractable": bool(record_level_extractable),
        "parser_media_hook_mode": parser_mode,
        "external_csp_availability": external_csp,
        "external_csp_join_hook": external_csp_join,
        "status": status,
        "watch_reason": watch_reason,
    }
    return crosswalk, rows


# 関数: `_write_media_crosswalk_csv` の入出力契約と処理意図を定義する。

def _write_media_crosswalk_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "table",
                "field_name",
                "mapping_role",
                "extractability",
                "note",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# 関数: `_write_external_csp_csv` の入出力契約と処理意図を定義する。

def _write_external_csp_csv(path: Path, payload: Dict[str, object]) -> None:
    rows: List[Tuple[str, str]] = []
    for k in [
        "public_reproducibility_state",
        "local_csp_file_count",
        "all_required_mediums_local",
        "delivery_route_documented",
        "hard_watch_required",
        "hard_watch_reason",
        "next_action",
    ]:
        rows.append((k, str(payload.get(k))))

    medium_counts = payload.get("local_csp_medium_counts") if isinstance(payload.get("local_csp_medium_counts"), dict) else {}
    for k, v in medium_counts.items():
        rows.append((f"local_csp_medium_counts.{k}", str(v)))

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for k, v in rows:
            writer.writerow([k, v])


# 関数: `_build_manifest` の入出力契約と処理意図を定義する。

def _build_manifest(
    observed_rows: Sequence[Dict[str, str]],
    label_meta_map: Dict[str, Dict[str, object]],
    media_crosswalk: Dict[str, object],
    media_crosswalk_json: Path,
    media_crosswalk_csv: Path,
    external_csp_json: Path,
    external_csp_csv: Path,
    sign_recordlevel_reaudit_json: Path,
    observed_csv: Path,
    pds_manifest_odf: Path,
) -> Tuple[Dict[str, object], List[Dict[str, str]]]:
    downlink_unique = _sorted_unique_ints([r.get("downlink_band_id") for r in observed_rows])
    uplink_unique = _sorted_unique_ints([r.get("uplink_band_id") for r in observed_rows])
    dtype_unique = _sorted_unique_ints([r.get("data_type_id") for r in observed_rows])
    station_rx_unique = _sorted_unique_ints([r.get("station_rx") for r in observed_rows])
    station_tx_unique = _sorted_unique_ints([r.get("station_tx") for r in observed_rows])

    y_pos = 0
    y_neg = 0
    for r in observed_rows:
        try:
            y = float(r.get("y_obs") or "nan")
        except Exception:
            continue

        if y > 0:
            y_pos += 1
        elif y < 0:
            y_neg += 1

    label_ref_date = sorted(
        set(
            int(v)
            for v in [m.get("file_reference_date") for m in label_meta_map.values()]
            if isinstance(v, int)
        )
    )
    label_ref_time = sorted(
        set(
            int(v)
            for v in [m.get("file_reference_time") for m in label_meta_map.values()]
            if isinstance(v, int)
        )
    )

    source_files = sorted(set([str(r.get("source_file") or "") for r in observed_rows if str(r.get("source_file") or "")]))
    label_missing_count = int(
        sum(
            1
            for rel in source_files
            if str((label_meta_map.get(rel) or {}).get("status") or "") != "ok"
        )
    )

    band_pass = len(downlink_unique) == 1 and len(uplink_unique) == 1
    time_pass = (
        len(label_ref_date) == 1
        and len(label_ref_time) == 1
        and label_missing_count == 0
        and int(label_ref_date[0]) in (0, 19500101)
    )
    sign_reaudit = _read_sign_recordlevel_reaudit(sign_recordlevel_reaudit_json)
    sign_terminal_gate_pass = bool(sign_reaudit.get("gate_pass"))
    sign_terminal_gate_status = str(sign_reaudit.get("status") or "")
    sign_watch_codes = [str(v) for v in (sign_reaudit.get("reasons") or []) if str(v).strip()]
    if not sign_terminal_gate_pass:
        sign_watch_codes.insert(0, "sign_terminal_gate_not_closed")

    # ODF3C formula defines a signed observable, and promotion requires record-level terminal gate closure.

    sign_pass = bool(sign_terminal_gate_pass)
    # Media extractability is determined from TRK/SCE1 crosswalk (direct field presence vs proxy-only fields).
    media_pass = bool(media_crosswalk.get("record_level_media_state_extractable"))
    media_status = str(media_crosswalk.get("status") or _status_from_bool(media_pass)).lower()
    parser_media_mode = str(media_crosswalk.get("parser_media_hook_mode") or "non_extractable_confirmed")
    media_watch_reason = str(media_crosswalk.get("watch_reason") or "")
    external_csp = media_crosswalk.get("external_csp_availability") if isinstance(media_crosswalk.get("external_csp_availability"), dict) else {}
    external_csp_join = media_crosswalk.get("external_csp_join_hook") if isinstance(media_crosswalk.get("external_csp_join_hook"), dict) else {}
    external_csp_state = str(external_csp.get("public_reproducibility_state") or "")
    external_csp_hard_watch = bool(external_csp.get("hard_watch_required"))
    external_csp_hard_watch_reason = str(external_csp.get("hard_watch_reason") or "")
    external_csp_file_count = int(external_csp.get("local_csp_file_count") or 0)
    external_csp_local_mediums = external_csp.get("local_mediums_observed") if isinstance(external_csp.get("local_mediums_observed"), list) else []
    external_csp_required_mediums = external_csp.get("required_mediums") if isinstance(external_csp.get("required_mediums"), list) else []
    external_csp_route_documented = bool(external_csp.get("delivery_route_documented"))
    external_csp_join_mode = str(external_csp_join.get("hook_mode") or "")
    external_csp_join_extractable = bool(external_csp_join.get("record_level_media_state_extractable_via_csp"))
    external_csp_join_required = external_csp_join.get("required_join_keys") if isinstance(external_csp_join.get("required_join_keys"), list) else []
    external_csp_join_missing = external_csp_join.get("missing_join_keys") if isinstance(external_csp_join.get("missing_join_keys"), list) else []
    external_csp_join_metrics_json = str(external_csp_join.get("join_metrics_json") or "")
    external_csp_join_coverage_any = external_csp_join.get("join_metrics_coverage_any_ratio")
    external_csp_join_coverage_mapped = external_csp_join.get("join_metrics_coverage_on_mapped_ratio")
    trk_2_18_local_available = bool(media_crosswalk.get("source_documents", {}).get("trk_2_18_local_available")) if isinstance(media_crosswalk.get("source_documents"), dict) else False
    trk_2_18_label_refs = int(
        (
            media_crosswalk.get("source_documents", {}).get("trk_2_18_reference_count_in_labels")
            if isinstance(media_crosswalk.get("source_documents"), dict)
            else 0
        )
        or 0
    )

    key_band = {
        "key": "band_id",
        "status": _status_from_bool(band_pass),
        "downlink_band_id_unique": downlink_unique,
        "uplink_band_id_unique": uplink_unique,
        "data_type_id_unique": dtype_unique,
        "pass_condition": "single downlink/uplink band across fitted ODF rows",
        "evidence": "observed CSV unique-value census",
    }
    key_sign = {
        "key": "doppler_sign_convention",
        "status": _status_from_bool(sign_pass),
        "sign_formula_basis": "ODF3C Item 4: Observable=[B/|B|]*[(Nj-Ni)/(tj-ti)-|Fb*K+B|]",
        "residual_definition_basis": "ODF3C Item 4: residual = observed Doppler - predicted Doppler",
        "parser_mapping": "y_obs = doppler_hz / carrier_hz (no parser-side sign inversion)",
        "observed_sign_counts": {"positive": int(y_pos), "negative": int(y_neg)},
        "recordlevel_reaudit_json": str(sign_reaudit.get("path") or ""),
        "recordlevel_reaudit_available": bool(sign_reaudit.get("available")),
        "terminal_gate_status": sign_terminal_gate_status,
        "terminal_gate_pass": bool(sign_terminal_gate_pass),
        "terminal_gate_selected_interval": sign_reaudit.get("selected_interval"),
        "terminal_gate_selected_coverage_ratio_points": sign_reaudit.get("selected_interval_coverage_ratio_points"),
        "terminal_gate_selected_n_arcs": sign_reaudit.get("selected_interval_n_arcs"),
        "terminal_gate_min_contiguous_arcs": sign_reaudit.get("min_contiguous_arcs"),
        "terminal_gate_min_coverage_ratio_points": sign_reaudit.get("min_coverage_ratio_points"),
        "watch_reason_codes": sign_watch_codes,
        "watch_reason": "record-level terminal gate for ODF sign convention is not closed" if not sign_pass else "",
    }
    key_media = {
        "key": "media_correction_state",
        "status": media_status,
        "parser_state": "raw ODF observable path; no additional plasma/tropo correction in parser",
        "label_note": "ODF labels expose ODF3C/ODF4 proxy context but no explicit per-record media-model flag.",
        "record_level_media_state_extractable": bool(media_pass),
        "parser_media_hook_mode": parser_media_mode,
        "trk_2_18_local_available": bool(trk_2_18_local_available),
        "trk_2_18_reference_count_in_labels": int(trk_2_18_label_refs),
        "crosswalk_json": str(media_crosswalk_json),
        "crosswalk_csv": str(media_crosswalk_csv),
        "external_csp_public_reproducibility_state": external_csp_state,
        "external_csp_hard_watch_required": bool(external_csp_hard_watch),
        "external_csp_hard_watch_reason": external_csp_hard_watch_reason,
        "external_csp_local_csp_file_count": int(external_csp_file_count),
        "external_csp_local_mediums_observed": external_csp_local_mediums,
        "external_csp_required_mediums": external_csp_required_mediums,
        "external_csp_delivery_route_documented": bool(external_csp_route_documented),
        "external_csp_join_hook_mode": external_csp_join_mode,
        "external_csp_join_extractable": bool(external_csp_join_extractable),
        "external_csp_join_required_keys": external_csp_join_required,
        "external_csp_join_missing_keys": external_csp_join_missing,
        "external_csp_join_metrics_json": external_csp_join_metrics_json,
        "external_csp_join_coverage_any_ratio": external_csp_join_coverage_any,
        "external_csp_join_coverage_on_mapped_ratio": external_csp_join_coverage_mapped,
        "external_csp_availability_json": str(external_csp_json),
        "external_csp_availability_csv": str(external_csp_csv),
        "watch_reason": media_watch_reason or "record-level media-correction state is not explicitly extractable from current ODF fields",
    }
    key_time = {
        "key": "time_scale_id",
        "status": _status_from_bool(time_pass),
        "file_reference_date_unique": label_ref_date,
        "file_reference_time_unique": label_ref_time,
        "parser_mapping": "time_utc = EPOCH_1950 + seconds + frac_ms",
        "time_scale_inference": "EME50 epoch anchor (19500101 or legacy 0 treated as EME50)",
        "pass_condition": "single ODF reference date/time across used label set (EME50-compatible)",
        "evidence": "ODF1B Item 20/21 (FILE REFERENCE DATE/TIME)",
    }

    keys = {
        "band_id": key_band,
        "doppler_sign_convention": key_sign,
        "media_correction_state": key_media,
        "time_scale_id": key_time,
    }

    all_pass = all(str((v or {}).get("status", "")).lower() == "pass" for v in keys.values())
    overall_status = "pass" if all_pass else "watch"

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "observed_csv": str(observed_csv),
            "pds_manifest_odf": str(pds_manifest_odf),
        },
        "counts": {
            "n_observed_rows": int(len(observed_rows)),
            "n_source_files": int(len(source_files)),
            "n_label_missing": int(label_missing_count),
            "station_rx_unique": station_rx_unique,
            "station_tx_unique": station_tx_unique,
        },
        "keys": keys,
        "overall_status": overall_status,
        "overall_if_ready_for_absolute_beta": bool(all_pass),
        "promotion_gate": "all four keys must be pass",
        "source_files": source_files,
        "label_meta_by_source_file": label_meta_map,
        "media_correction_field_crosswalk": media_crosswalk,
        "sign_recordlevel_terminal_assessment": sign_reaudit,
    }

    rows_csv: List[Dict[str, str]] = []
    rows_csv.append(
        {
            "key": "band_id",
            "status": key_band["status"],
            "summary": f"downlink={_format_list(downlink_unique)}; uplink={_format_list(uplink_unique)}; dtype={_format_list(dtype_unique)}",
            "pass_condition": str(key_band["pass_condition"]),
            "evidence": str(key_band["evidence"]),
        }
    )
    rows_csv.append(
        {
            "key": "doppler_sign_convention",
            "status": key_sign["status"],
            "summary": (
                f"parser=no_sign_inversion; y_sign(+/-)={int(y_pos)}/{int(y_neg)}; "
                f"basis=ODF3C item4; terminal_gate={str(key_sign['terminal_gate_status'])}; "
                f"coverage={str(key_sign['terminal_gate_selected_coverage_ratio_points'])}"
            ),
            "pass_condition": (
                "cross-source sign convention lock is closed "
                "+ stable contiguous arc interval satisfies minimum coverage gate"
            ),
            "evidence": (
                f"{str(key_sign['residual_definition_basis'])}; "
                f"recordlevel_reaudit={str(key_sign['recordlevel_reaudit_json'])}; "
                f"watch_codes={','.join([str(v) for v in key_sign.get('watch_reason_codes', [])])}"
            ),
        }
    )
    rows_csv.append(
        {
            "key": "media_correction_state",
            "status": key_media["status"],
            "summary": (
                f"{key_media['parser_state']}; "
                f"mode={key_media['parser_media_hook_mode']}; "
                f"extractable={str(bool(key_media['record_level_media_state_extractable'])).lower()}; "
                f"external_csp_state={str(key_media['external_csp_public_reproducibility_state'])}; "
                f"external_csp_hard_watch={str(bool(key_media['external_csp_hard_watch_required'])).lower()}; "
                f"join_cov_any={str(key_media['external_csp_join_coverage_any_ratio'])}"
            ),
            "pass_condition": "record-level media correction state is explicitly fixed",
            "evidence": (
                f"{str(key_media['label_note'])}; "
                f"crosswalk_json={str(key_media['crosswalk_json'])}; "
                f"external_csp_json={str(key_media['external_csp_availability_json'])}; "
                f"join_metrics_json={str(key_media['external_csp_join_metrics_json'])}"
            ),
        }
    )
    rows_csv.append(
        {
            "key": "time_scale_id",
            "status": key_time["status"],
            "summary": (
                f"ref_date={_format_list(label_ref_date)}; "
                f"ref_time={_format_list(label_ref_time)}; parser=EPOCH_1950+sec+frac_ms"
            ),
            "pass_condition": str(key_time["pass_condition"]),
            "evidence": str(key_time["evidence"]),
        }
    )

    return manifest, rows_csv


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "key",
                "status",
                "summary",
                "pass_condition",
                "evidence",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


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

def main() -> None:
    root = _repo_root()
    out_dir = root / "output" / "cassini"
    out_dir.mkdir(parents=True, exist_ok=True)
    observed_csv = out_dir / "cassini_sce1_odf_observed_raw.csv"
    pds_root = root / "data" / "cassini" / "pds_sce1"
    pds_manifest_odf = pds_root / "manifest_odf.json"
    media_crosswalk_json = out_dir / "cassini_odf_media_correction_field_map.json"
    media_crosswalk_csv = out_dir / "cassini_odf_media_correction_field_map.csv"
    external_csp_json = out_dir / "cassini_odf_external_csp_availability.json"
    external_csp_csv = out_dir / "cassini_odf_external_csp_availability.csv"
    sign_recordlevel_reaudit_json = out_dir / "cassini_odf_sign_recordlevel_reaudit.json"

    observed_rows = _load_observed_rows(observed_csv)
    source_files = sorted(set([str(r.get("source_file") or "") for r in observed_rows if str(r.get("source_file") or "")]))
    label_meta_map = _build_label_meta_map(pds_root, source_files)
    media_crosswalk, media_rows = _build_media_crosswalk(observed_rows, label_meta_map, pds_root, root)
    external_csp = media_crosswalk.get("external_csp_availability") if isinstance(media_crosswalk.get("external_csp_availability"), dict) else {}
    media_crosswalk_json.write_text(json.dumps(media_crosswalk, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_media_crosswalk_csv(media_crosswalk_csv, media_rows)
    external_csp_json.write_text(json.dumps(external_csp, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_external_csp_csv(external_csp_csv, external_csp)

    manifest, rows_csv = _build_manifest(
        observed_rows=observed_rows,
        label_meta_map=label_meta_map,
        media_crosswalk=media_crosswalk,
        media_crosswalk_json=media_crosswalk_json,
        media_crosswalk_csv=media_crosswalk_csv,
        external_csp_json=external_csp_json,
        external_csp_csv=external_csp_csv,
        sign_recordlevel_reaudit_json=sign_recordlevel_reaudit_json,
        observed_csv=observed_csv,
        pds_manifest_odf=pds_manifest_odf,
    )

    out_json = out_dir / "cassini_odf_raw_if_manifest.json"
    out_csv = out_dir / "cassini_odf_raw_if_manifest.csv"
    out_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(out_csv, rows_csv)
    _sync_public(
        root,
        [
            out_json.name,
            out_csv.name,
            media_crosswalk_json.name,
            media_crosswalk_csv.name,
            external_csp_json.name,
            external_csp_csv.name,
        ],
    )
    print("Wrote:", media_crosswalk_json)
    print("Wrote:", media_crosswalk_csv)
    print("Wrote:", external_csp_json)
    print("Wrote:", external_csp_csv)
    print("Wrote:", out_json)
    print("Wrote:", out_csv)
    print("Synced:", root / "output" / "public" / "cassini")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
