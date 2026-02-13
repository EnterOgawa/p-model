#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llr_falsification_pack.py

Phase 2 / Step 2.1（LLR）:
LLR（月レーザー測距）のバッチ監査結果（time-tag/外れ値/補正の寄与）を、
公開用の “falsification pack” として `output/public/llr/` に固定する。

このスクリプトは計算の再実行を行わない（既存の private 出力を public に昇格する）。

入力（既存の固定出力）:
- output/private/llr/batch/llr_batch_summary.json
- output/private/llr/batch/llr_time_tag_best_by_station.json
- output/private/llr/batch/llr_outliers_diagnosis_summary.json

出力（固定; Git tracked）:
- output/public/llr/llr_falsification_pack.json
- output/public/llr/（主要PNG/JSON/CSVをコピー）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from scripts.summary import worklog  # type: ignore # noqa: E402
except Exception:  # pragma: no cover
    worklog = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path: Path) -> str:
    try:
        return path.relative_to(_ROOT).as_posix()
    except Exception:
        return path.as_posix()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_file(src: Path, dst: Path, *, overwrite: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(str(src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return
    shutil.copy2(src, dst)


def _artifact_record(path: Path) -> Dict[str, Any]:
    return {
        "path": _rel(path),
        "sha256": _sha256(path) if path.exists() else None,
        "bytes": int(path.stat().st_size) if path.exists() else None,
    }


def _as_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    return v if v == v else None


def _build_pack(
    *,
    batch_summary: Dict[str, Any],
    time_tag_best: Dict[str, Any],
    outliers_diag: Dict[str, Any],
    out_dir: Path,
    copied: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    med = batch_summary.get("median_rms_ns") if isinstance(batch_summary.get("median_rms_ns"), dict) else {}
    rms_tide = _as_float(med.get("station_reflector_tropo_tide") if isinstance(med, dict) else None)
    rms_no_shapiro = _as_float(med.get("station_reflector_tropo_no_shapiro") if isinstance(med, dict) else None)
    ratio = None
    if rms_tide is not None and rms_no_shapiro not in (None, 0.0):
        ratio = rms_tide / float(rms_no_shapiro)

    # Minimal operational gate: Shapiro term should improve the median RMS.
    shapiro_threshold = 0.95
    shapiro_pass = None if ratio is None else (ratio <= shapiro_threshold)

    pack: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "domain": "llr",
        "version": 1,
        "intent": "Promote LLR batch audit outputs into tracked public artifacts and fix a minimal falsification gate.",
        "inputs": {
            "batch_summary": {
                "generated_utc": batch_summary.get("generated_utc"),
                "n_files": batch_summary.get("n_files"),
                "n_points_total": batch_summary.get("n_points_total"),
                "stations": batch_summary.get("stations"),
                "targets": batch_summary.get("targets"),
                "beta": batch_summary.get("beta"),
                "time_tag_mode": batch_summary.get("time_tag_mode"),
                "station_coords_mode": batch_summary.get("station_coords_mode"),
                "outlier_gate": {
                    "clip_sigma": batch_summary.get("outlier_clip_sigma"),
                    "clip_min_ns": batch_summary.get("outlier_clip_ns"),
                },
            },
            "time_tag_best_by_station": {
                "selection_metric": time_tag_best.get("selection_metric"),
                "best_mode_by_station": time_tag_best.get("best_mode_by_station"),
            },
            "outliers_diagnosis_summary": {
                "n_outliers": outliers_diag.get("n_outliers"),
                "by_cause_hint": outliers_diag.get("by_cause_hint"),
                "time_tag_sensitivity": outliers_diag.get("time_tag_sensitivity"),
                "target_mixing_sensitivity": outliers_diag.get("target_mixing_sensitivity"),
            },
        },
        "metrics": {
            "median_rms_ns": {
                "station_reflector": _as_float(med.get("station_reflector") if isinstance(med, dict) else None),
                "station_reflector_tropo": _as_float(med.get("station_reflector_tropo") if isinstance(med, dict) else None),
                "station_reflector_tropo_tide": rms_tide,
                "station_reflector_tropo_no_shapiro": rms_no_shapiro,
            },
            "shapiro_rms_ratio": {
                "value": ratio,
                "unit": "(tropo+tide)/(tropo+no_shapiro)",
            },
        },
        "criteria": [
            {
                "id": "llr_shapiro_improves",
                "title": "Shapiro項を除去するとRMSが悪化すること（中央値）",
                "value": ratio,
                "op": "<=",
                "threshold": shapiro_threshold,
                "pass": shapiro_pass,
                "gate": True,
                "unit": "(tropo+tide)/(tropo+no_shapiro)",
                "rationale": "Shapiro項の符号/係数が逆だと、RMS改善が消えるため、最小の反証条件として採用する。",
            }
        ],
        "artifacts": {
            "dir": _rel(out_dir),
            "copied": copied,
        },
    }
    return pack


def run(*, in_dir: Path, out_dir: Path, overwrite: bool) -> Tuple[Path, List[str]]:
    warnings: List[str] = []

    required = {
        "llr_batch_summary.json": in_dir / "llr_batch_summary.json",
        "llr_time_tag_best_by_station.json": in_dir / "llr_time_tag_best_by_station.json",
        "llr_outliers_diagnosis_summary.json": in_dir / "llr_outliers_diagnosis_summary.json",
    }
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"missing required input: {name} ({_rel(path)})")

    batch_summary = _read_json(required["llr_batch_summary.json"])
    time_tag_best = _read_json(required["llr_time_tag_best_by_station.json"])
    outliers_diag = _read_json(required["llr_outliers_diagnosis_summary.json"])

    # Copy a minimal set of artifacts used as audit evidence.
    out_dir.mkdir(parents=True, exist_ok=True)
    copy_specs: List[Tuple[Path, Path]] = []

    # Small tables (optional but useful)
    for fname in (
        "llr_batch_summary.json",
        "llr_batch_metrics.csv",
        "llr_time_tag_best_by_station.json",
        "llr_outliers_summary.json",
        "llr_outliers_diagnosis_summary.json",
        "llr_station_diagnostics.json",
    ):
        src = in_dir / fname
        if src.exists():
            copy_specs.append((src, out_dir / fname))
        else:
            warnings.append(f"missing optional input: {_rel(src)}")

    # Key figures (keep small; omit the huge per-point CSV)
    for fname in (
        "llr_residual_distribution.png",
        "llr_rms_improvement_overall.png",
        "llr_rms_ablations_overall.png",
        "llr_shapiro_ablations_overall.png",
        "llr_tide_ablations_overall.png",
        "llr_time_tag_selection_by_station.png",
        "llr_outliers_overview.png",
        "llr_outliers_time_tag_sensitivity.png",
        "llr_station_coord_delta_pos_eop.png",
    ):
        src = in_dir / fname
        if src.exists():
            copy_specs.append((src, out_dir / fname))
        else:
            warnings.append(f"missing optional figure: {_rel(src)}")

    copied: Dict[str, Dict[str, Any]] = {}
    for src, dst in copy_specs:
        _copy_file(src, dst, overwrite=overwrite)
        copied[str(dst.name)] = _artifact_record(dst)

    pack = _build_pack(
        batch_summary=batch_summary,
        time_tag_best=time_tag_best,
        outliers_diag=outliers_diag,
        out_dir=out_dir,
        copied=copied,
    )
    pack_path = out_dir / "llr_falsification_pack.json"
    pack_path.write_text(
        json.dumps(pack, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if worklog is not None:
        try:
            worklog.append_event(
                {
                    "event_type": "llr_falsification_pack_public",
                    "inputs": {k: _rel(v) for k, v in required.items()},
                    "outputs": {"llr_falsification_pack_json": _rel(pack_path), "public_llr_dir": _rel(out_dir)},
                    "warnings": warnings,
                }
            )
        except Exception:
            pass

    return pack_path, warnings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Promote LLR batch audit outputs into output/public/llr.")
    parser.add_argument(
        "--in-dir",
        default="output/private/llr/batch",
        help="Input directory (default: output/private/llr/batch).",
    )
    parser.add_argument(
        "--out-dir",
        default="output/public/llr",
        help="Output directory (default: output/public/llr).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    in_dir = Path(str(args.in_dir))
    if not in_dir.is_absolute():
        in_dir = _ROOT / in_dir
    out_dir = Path(str(args.out_dir))
    if not out_dir.is_absolute():
        out_dir = _ROOT / out_dir

    try:
        pack_path, warnings = run(in_dir=in_dir, out_dir=out_dir, overwrite=bool(args.overwrite))
    except FileNotFoundError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2

    for w in warnings:
        print(f"[warn] {w}", file=sys.stderr)
    print(f"[ok] wrote: {_rel(pack_path)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

