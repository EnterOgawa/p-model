#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_xrism_target_catalog.py

Phase 4 / Step 4.8（XRISM）:
DARTS の Resolve metadata（一次）から、P-model 検証に使うターゲット候補を選定し、
解析スクリプトが参照できる形で `data/xrism/sources/xrism_target_catalog.json` を生成する。

本スクリプトは「選定の I/F を固定」する位置づけであり、z_opt や参照（論文/arXiv）は
将来の追加・更新を前提として明示的にフィールドとして保持する。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


# 関数: `_utc_now` の入出力契約と処理意図を定義する。
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_rel` の入出力契約と処理意図を定義する。

def _rel(path: Path) -> str:
    try:
        return path.relative_to(_ROOT).as_posix()
    except Exception:
        return path.as_posix()


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# 関数: `_read_csv_rows` の入出力契約と処理意図を定義する。

def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # 条件分岐: `not isinstance(r, dict)` を満たす経路を評価する。
            if not isinstance(r, dict):
                continue

            rows.append({str(k): (v or "").strip() for k, v in r.items() if k is not None})

    return rows


# 関数: `_parse_iso_date` の入出力契約と処理意図を定義する。

def _parse_iso_date(s: str) -> Optional[str]:
    s = (s or "").strip()
    # 条件分岐: `not s` を満たす経路を評価する。
    if not s:
        return None

    m = re.match(r"^(\d{4}-\d{2}-\d{2})", s)
    return m.group(1) if m else None


# 関数: `_is_public` の入出力契約と処理意図を定義する。

def _is_public(row: Dict[str, str], *, today_ymd: str) -> bool:
    d = _parse_iso_date(row.get("public_date") or "")
    # 条件分岐: `not d` を満たす経路を評価する。
    if not d:
        return False

    return d <= today_ymd


# 関数: `_maybe_float` の入出力契約と処理意図を定義する。

def _maybe_float(x: str) -> Optional[float]:
    s = (x or "").strip()
    # 条件分岐: `not s` を満たす経路を評価する。
    if not s:
        return None

    try:
        return float(s)
    except Exception:
        return None


# クラス: `TargetPreset` の責務と境界条件を定義する。

@dataclass(frozen=True)
class TargetPreset:
    object_name: str
    role: str
    line_type: str
    z_opt: Optional[float]
    z_opt_ref: str
    expected_lines: List[str]
    arxiv_refs: List[str]
    notes: str


# 関数: `_default_presets` の入出力契約と処理意図を定義する。

def _default_presets() -> List[TargetPreset]:
    # Minimal seeded presets (expand as data / refs accumulate).
    # z_opt_ref は現時点では “凍結した値の由来” の台帳。将来、一次論文/DBを PRIMARY_SOURCES へ追加して更新する。
    return [
        TargetPreset(
            object_name="CENTAURUS_A",
            role="bh_agn",
            line_type="agn_fek_velocity",
            z_opt=0.00183,
            z_opt_ref="curated (TBD: literature ref)",
            expected_lines=["Fe Kα (6.404/6.391 keV)", "Fe XXV (6.700 keV)", "Fe XXVI (6.966 keV)"],
            arxiv_refs=["arXiv:2507.02195"],
            notes="優先度A（AGN）。公開済み・既存解析あり（arXiv）。",
        ),
        TargetPreset(
            object_name="NGC4151",
            role="bh_agn",
            line_type="agn_fek_velocity",
            z_opt=0.00332,
            z_opt_ref="curated (TBD: literature ref)",
            expected_lines=["Fe K band absorption/emission (Fe XXV/XXVI)"],
            arxiv_refs=[],
            notes="既存I/Fの継続（Step 4.8.2 初版）。複数obsidが存在するため detected_obsids>=2 の候補。",
        ),
        TargetPreset(
            object_name="PDS456",
            role="bh_agn",
            line_type="agn_ufo_velocity",
            z_opt=0.184,
            z_opt_ref="curated (TBD: literature ref)",
            expected_lines=["Fe XXV (6.700 keV)", "Fe XXVI (6.966 keV)"],
            arxiv_refs=[],
            notes="UFO候補。公開済み。",
        ),
        TargetPreset(
            object_name="Centaurus_Cluster",
            role="cluster",
            line_type="cluster_fek_z_xray",
            z_opt=0.0104,
            z_opt_ref="curated (TBD: literature ref; Abell 3526)",
            expected_lines=["Fe XXV (6.700 keV)", "Fe XXVI (6.966 keV)"],
            arxiv_refs=[],
            notes="優先度B（銀河団）。z_xray を距離指標非依存で固定。",
        ),
        TargetPreset(
            object_name="Perseus_C1",
            role="cluster",
            line_type="cluster_fek_z_xray",
            z_opt=0.0179,
            z_opt_ref="curated (TBD: literature ref)",
            expected_lines=["Fe XXV (6.700 keV)", "Fe XXVI (6.966 keV)"],
            arxiv_refs=[],
            notes="既存I/Fの継続（Step 4.8.3 初版）。Hitomi比較の入口。",
        ),
        TargetPreset(
            object_name="COMA_CENTER",
            role="cluster",
            line_type="cluster_fek_z_xray",
            z_opt=0.0231,
            z_opt_ref="curated (TBD: literature ref)",
            expected_lines=["Fe XXV (6.700 keV)", "Fe XXVI (6.966 keV)"],
            arxiv_refs=[],
            notes="銀河団（turbulence）。複数pointingがあるため将来のdetected増に有利。",
        ),
        TargetPreset(
            object_name="N132D",
            role="snr_qc",
            line_type="snr_qc_energy_scale",
            z_opt=None,
            z_opt_ref="n/a",
            expected_lines=["Fe-K band (composition/QC)"],
            arxiv_refs=[],
            notes="優先度C（SNR）。energy-scale/QCの基準候補。",
        ),
    ]


# 関数: `_pick_obsids_for_object` の入出力契約と処理意図を定義する。

def _pick_obsids_for_object(
    rows: List[Dict[str, str]],
    *,
    object_name: str,
    today_ymd: str,
    max_obsids: int,
) -> List[Dict[str, str]]:
    cand: List[Dict[str, str]] = []
    for r in rows:
        # 条件分岐: `not _is_public(r, today_ymd=today_ymd)` を満たす経路を評価する。
        if not _is_public(r, today_ymd=today_ymd):
            continue

        # 条件分岐: `(r.get("object_name") or "").strip() != object_name` を満たす経路を評価する。

        if (r.get("object_name") or "").strip() != object_name:
            continue

        obsid = (r.get("observation_id") or "").strip()
        # 条件分岐: `not obsid` を満たす経路を評価する。
        if not obsid:
            continue

        cand.append(r)
    # Prefer longer Resolve exposure.

    cand.sort(key=lambda r: -(float(r.get("resolve_exposure") or 0.0)))
    return cand[: max(0, int(max_obsids))]


# 関数: `build_catalog` の入出力契約と処理意図を定義する。

def build_catalog(
    *,
    metadata_csv: Path,
    out_json: Path,
    presets: Sequence[TargetPreset],
    max_obsids_per_object: int,
) -> Dict[str, Any]:
    rows = _read_csv_rows(metadata_csv)
    today_ymd = datetime.now(timezone.utc).date().isoformat()

    targets: List[Dict[str, Any]] = []
    missing: List[str] = []
    for p in presets:
        picked = _pick_obsids_for_object(
            rows,
            object_name=p.object_name,
            today_ymd=today_ymd,
            max_obsids=max_obsids_per_object,
        )
        # 条件分岐: `not picked` を満たす経路を評価する。
        if not picked:
            missing.append(p.object_name)
            continue

        for r in picked:
            obsid = (r.get("observation_id") or "").strip()
            targets.append(
                {
                    "obsid": obsid,
                    "object_name": p.object_name,
                    "role": p.role,
                    "line_type": p.line_type,
                    "public_date": _parse_iso_date(r.get("public_date") or "") or "",
                    "resolve_exposure": _maybe_float(r.get("resolve_exposure") or ""),
                    "processing_version": (r.get("processing_version") or "").strip(),
                    "data_access_url": (r.get("data_access_url") or "").strip(),
                    "ql_access_url": (r.get("ql_access_url") or "").strip(),
                    "z_opt": p.z_opt,
                    "z_opt_ref": p.z_opt_ref,
                    "expected_lines": list(p.expected_lines),
                    "arxiv_refs": list(p.arxiv_refs),
                    "notes": p.notes,
                }
            )

    targets.sort(key=lambda t: (str(t.get("role") or ""), str(t.get("object_name") or ""), str(t.get("obsid") or "")))

    out = {
        "generated_utc": _utc_now(),
        "inputs": {"resolve_metadata_csv": _rel(metadata_csv)},
        "policy": {
            "public_filter": "public_date <= today(UTC)",
            "selection": "preset-driven (seeded list; expand as refs accumulate)",
            "max_obsids_per_object": int(max_obsids_per_object),
        },
        "targets": targets,
        "missing_object_names": missing,
    }
    _write_json(out_json, out)
    return out


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--metadata-csv",
        default=str(_ROOT / "data" / "xrism" / "sources" / "darts" / "xrism_resolve_data.csv"),
    )
    p.add_argument(
        "--out-json",
        default=str(_ROOT / "data" / "xrism" / "sources" / "xrism_target_catalog.json"),
    )
    p.add_argument("--max-obsids-per-object", type=int, default=2)
    args = p.parse_args(list(argv) if argv is not None else None)

    catalog = build_catalog(
        metadata_csv=Path(args.metadata_csv),
        out_json=Path(args.out_json),
        presets=_default_presets(),
        max_obsids_per_object=int(args.max_obsids_per_object),
    )

    worklog.append_event(
        {
            "task": "xrism_target_catalog_build",
            "inputs": {"metadata_csv": Path(args.metadata_csv)},
            "outputs": {"target_catalog_json": Path(args.out_json)},
            "metrics": {
                "n_targets": len(catalog.get("targets") or []),
                "missing_object_names": list(catalog.get("missing_object_names") or []),
            },
        }
    )

    print(f"[ok] wrote: {args.out_json}")
    # 条件分岐: `catalog.get("missing_object_names")` を満たす経路を評価する。
    if catalog.get("missing_object_names"):
        print(f"[warn] missing objects: {', '.join(catalog['missing_object_names'])}")

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())

