#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xrism_event_level_qc.py

Phase 4 / Step 4.8（XRISM）:
event_cl（event-level）から PI histogram を再構成し、products（*_src.pi.gz）と比較する QC を固定出力化する。

主な目的:
- Pixel除外（例：Pixel 27）/GTI適用の効果を「手続き差（系統）」として定量化する入口を作る。
- 将来、HEASoft/xrism-pipeline による厳密抽出へ移行しても比較できる I/F（入力・出力・指標）を先に閉じる。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402
from scripts.xrism.xrism_bh_outflow_velocity import _choose_pi_rmf_pair, _load_pi_spectrum, _read_ebounds_table  # noqa: E402
from scripts.xrism.xrism_event_utils import (  # noqa: E402
    choose_event_file,
    compute_spectrum_diff_metrics,
    extract_pi_spectrum_from_event,
    load_gti_intervals,
    load_gti_intervals_from_event,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sanitize_variant(s: str) -> str:
    s = str(s or "").strip()
    if not s:
        return ""
    # Keep filenames simple and stable across platforms.
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", s)
    s = s.strip("._-")
    return s


def _parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for part in str(s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _maybe_float(x: object) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        return v if np.isfinite(v) else None
    s = str(x).strip()
    if not s:
        return None
    try:
        v = float(s)
    except Exception:
        return None
    return v if np.isfinite(v) else None


def _align_counts_to_rmf(
    *,
    rmf_channels: np.ndarray,
    src_channels: np.ndarray,
    src_counts: np.ndarray,
) -> np.ndarray:
    ch = np.asarray(rmf_channels, dtype=int)
    src_ch = np.asarray(src_channels, dtype=int)
    src_ct = np.asarray(src_counts, dtype=float)
    max_ch = int(np.max(ch)) if ch.size else -1
    if max_ch < 0:
        return np.zeros(0, dtype=float)
    idx = -np.ones(max_ch + 1, dtype=int)
    ok = (ch >= 0) & (ch <= max_ch)
    idx[ch[ok]] = np.arange(int(ch.size), dtype=int)[ok]
    out = np.zeros(int(ch.size), dtype=float)
    m = (src_ch >= 0) & (src_ch <= max_ch)
    if np.any(m):
        ii = idx[src_ch[m]]
        ok2 = ii >= 0
        if np.any(ok2):
            out[ii[ok2]] = src_ct[m][ok2]
    return out


def _read_obsids_from_catalog(path: Path) -> List[str]:
    if not path.exists():
        return []
    d = json.loads(path.read_text(encoding="utf-8"))
    obsids: List[str] = []
    for t in (d.get("targets") or []):
        if not isinstance(t, dict):
            continue
        o = str(t.get("obsid") or "").strip()
        if len(o) == 9 and o.isdigit():
            obsids.append(o)
    return sorted(set(obsids))


def _run_qc(
    *,
    catalog: str,
    obsids: List[str],
    data_root: Path,
    out_dir: Path,
    apply_gti: bool,
    pixel_exclude: List[int],
    chunk_rows: int,
    fek_band: tuple[float, float],
    variant: str,
) -> Dict[str, Any]:
    v = _sanitize_variant(variant)
    suffix = f"__{v}" if v else ""

    rows: List[Dict[str, Any]] = []
    per_obsid: Dict[str, Any] = {}

    for obsid in obsids:
        local_root = data_root / obsid
        products_dir = local_root / "resolve" / "products"
        event_dir = local_root / "resolve" / "event_cl"
        auxil_dir = local_root / "auxil"

        rec: Dict[str, Any] = {
            "obsid": obsid,
            "generated_utc": _utc_now(),
            "inputs": {
                "products_dir": str(products_dir),
                "event_dir": str(event_dir),
                "auxil_dir": str(auxil_dir),
                "apply_gti": bool(apply_gti),
                "pixel_exclude": [int(x) for x in pixel_exclude],
                "chunk_rows": int(chunk_rows),
            },
            "variant": v or None,
        }

        if not products_dir.is_dir():
            rec["ok"] = False
            rec["reason"] = "missing products_dir"
            per_obsid[obsid] = rec
            continue

        pi_path, rmf_path = _choose_pi_rmf_pair(products_dir)
        rmf_ch, rmf_E = _read_ebounds_table(rmf_path)
        src_ch, src_counts_raw = _load_pi_spectrum(pi_path)
        src_counts = _align_counts_to_rmf(rmf_channels=rmf_ch, src_channels=src_ch, src_counts=src_counts_raw)
        rec["products"] = {
            "pi_path": str(pi_path),
            "rmf_path": str(rmf_path),
            "rmf_n_channels": int(rmf_ch.size),
            "src_counts_sum": float(np.nansum(src_counts)),
        }

        event_ok = event_dir.is_dir()
        gti_path_auxil = auxil_dir / f"xa{obsid}_gen.gti.gz"
        gti_start = None
        gti_stop = None

        if event_ok:
            try:
                event_path = choose_event_file(event_dir)
                if bool(apply_gti):
                    # Prefer GTI embedded in the event file; fallback to auxil if available.
                    try:
                        gti_start, gti_stop = load_gti_intervals_from_event(event_path)
                        rec["gti"] = {"source": "event", "n": int(np.asarray(gti_start).size)}
                    except Exception as e_evt:
                        if gti_path_auxil.exists():
                            try:
                                gti_start, gti_stop = load_gti_intervals(gti_path_auxil)
                                rec["gti"] = {"source": "auxil", "path": str(gti_path_auxil), "n": int(np.asarray(gti_start).size)}
                            except Exception as e_aux:
                                rec["gti"] = {
                                    "source": "failed",
                                    "event_error": str(e_evt),
                                    "auxil_path": str(gti_path_auxil),
                                    "auxil_error": str(e_aux),
                                }
                                gti_start, gti_stop = None, None
                        else:
                            rec["gti"] = {"source": "event_failed", "event_error": str(e_evt), "auxil_present": False}
                            gti_start, gti_stop = None, None
                else:
                    rec["gti"] = {"source": "not_applied", "auxil_present": bool(gti_path_auxil.exists())}

                ev = extract_pi_spectrum_from_event(
                    event_path,
                    channels=rmf_ch,
                    gti_start=gti_start,
                    gti_stop=gti_stop,
                    pixel_exclude=pixel_exclude,
                    chunk_rows=int(chunk_rows),
                )
                rec["event"] = {
                    "event_path": str(event_path),
                    "counts_sum": float(np.nansum(ev.counts)),
                    "qc": ev.qc,
                }
                diff = compute_spectrum_diff_metrics(
                    channels=rmf_ch,
                    energy_keV=rmf_E,
                    counts_a=src_counts,
                    counts_b=ev.counts,
                    fek_band=fek_band,
                )
                rec["compare_products_vs_event"] = diff

                out_qc = out_dir / f"{obsid}__event_level_qc{suffix}.json"
                _write_json(out_qc, rec)

                l1_norm_a = diff.get("l1_norm_a")
                mean_shift_keV = diff.get("mean_energy_shift_keV_b_minus_a")
                rows.append(
                    {
                        "obsid": obsid,
                        "pi_path": str(pi_path.name),
                        "event_path": str(event_path.name),
                        "event_rows": int(ev.qc.get("event_n_rows") or 0),
                        "event_counts_sum": float(ev.qc.get("event_counts_sum") or 0.0),
                        "l1_norm_a": float(l1_norm_a) if diff.get("ok") and l1_norm_a is not None else float("nan"),
                        "mean_shift_keV": float(mean_shift_keV) if diff.get("ok") and mean_shift_keV is not None else float("nan"),
                        "pixel_exclude": ",".join(str(x) for x in pixel_exclude),
                        "apply_gti": bool(apply_gti),
                        "gti_n": int(np.asarray(gti_start).size) if gti_start is not None else 0,
                        "variant": v,
                    }
                )
            except Exception as e:
                rec["ok"] = False
                rec["reason"] = f"event_qc_failed: {e}"
                per_obsid[obsid] = rec
                _write_json(out_dir / f"{obsid}__event_level_qc{suffix}.json", rec)
                continue
        else:
            rec["ok"] = False
            rec["reason"] = "missing event_dir"
            per_obsid[obsid] = rec
            _write_json(out_dir / f"{obsid}__event_level_qc{suffix}.json", rec)

    # Summary CSV
    out_csv = out_dir / f"xrism_event_level_qc_summary{suffix}.csv"
    if rows:
        cols = [
            "obsid",
            "pi_path",
            "event_path",
            "event_rows",
            "event_counts_sum",
            "l1_norm_a",
            "mean_shift_keV",
            "pixel_exclude",
            "apply_gti",
            "gti_n",
            "variant",
        ]
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in cols})

    metrics = {
        "generated_utc": _utc_now(),
        "inputs": {
            "catalog": str(catalog),
            "data_root": str(data_root),
            "out_dir": str(out_dir),
            "obsids": obsids,
            "apply_gti": bool(apply_gti),
            "pixel_exclude": [int(x) for x in pixel_exclude],
            "chunk_rows": int(chunk_rows),
            "fek_band_keV": [float(fek_band[0]), float(fek_band[1])],
            "variant": v,
        },
        "outputs": {
            "summary_csv": str(out_csv),
            "per_obsid_qc": [str(out_dir / f"{o}__event_level_qc{suffix}.json") for o in obsids],
        },
        "result": {
            "n_obsids": int(len(obsids)),
            "n_rows": int(len(rows)),
        },
    }
    out_metrics = out_dir / f"xrism_event_level_qc_summary_metrics{suffix}.json"
    _write_json(out_metrics, metrics)

    worklog.append_event(
        {
            "task": "xrism_event_level_qc",
            "inputs": metrics["inputs"],
            "outputs": {
                "summary_csv": out_csv,
                "summary_metrics": out_metrics,
                "per_obsid_qc": [Path(p) for p in metrics["outputs"]["per_obsid_qc"]],
            },
            "metrics": metrics["result"],
        },
    )

    print(f"[ok] wrote: {out_csv}")
    print(f"[ok] wrote: {out_metrics}")
    return {"summary_csv": str(out_csv), "summary_metrics": str(out_metrics), "variant": v, "n_rows": int(len(rows))}


def _summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    l1 = [float(v) for r in rows if (v := _maybe_float(r.get("l1_norm_a"))) is not None]
    ms = [float(v) for r in rows if (v := _maybe_float(r.get("mean_shift_keV"))) is not None]
    ms_eV = [1000.0 * float(x) for x in ms]
    return {
        "n_rows": int(len(rows)),
        "l1_norm_a_range": [float(min(l1)), float(max(l1))] if l1 else None,
        "mean_shift_eV_event_minus_products_range": [float(min(ms_eV)), float(max(ms_eV))] if ms_eV else None,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", default=str(_ROOT / "data" / "xrism" / "sources" / "xrism_target_catalog.json"))
    p.add_argument("--obsid", action="append", default=[], help="override: obsid(s) to QC (repeatable)")
    p.add_argument("--data-root", default=str(_ROOT / "data" / "xrism" / "raw"))
    p.add_argument("--out-dir", default=str(_ROOT / "output" / "xrism"))
    p.add_argument("--apply-gti", action="store_true", help="apply auxil/*_gen.gti.gz if available")
    p.add_argument("--pixel-exclude", default="27", help="comma-separated pixel IDs to exclude (default: 27)")
    p.add_argument("--chunk-rows", type=int, default=200_000)
    p.add_argument("--fek-lo", type=float, default=5.5)
    p.add_argument("--fek-hi", type=float, default=7.5)
    p.add_argument("--variant", default="", help="output suffix tag (default: none)")
    p.add_argument(
        "--sweep-procedure",
        action="store_true",
        help="run 2x2 sweep over {apply_gti on/off} x {pixel_exclude none/27} and write per-variant outputs",
    )
    args = p.parse_args(argv)

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    obsids = [str(o).strip() for o in args.obsid if str(o).strip()]
    if not obsids:
        obsids = _read_obsids_from_catalog(Path(args.catalog))
    if not obsids:
        raise SystemExit("no obsid provided and catalog is empty/missing")

    fek_band = (float(args.fek_lo), float(args.fek_hi))

    if bool(args.sweep_procedure):
        sweep_defs = [
            {"variant": "gti0_px0", "apply_gti": False, "pixel_exclude": []},
            {"variant": "gti0_px27", "apply_gti": False, "pixel_exclude": [27]},
            {"variant": "gti1_px0", "apply_gti": True, "pixel_exclude": []},
            {"variant": "gti1_px27", "apply_gti": True, "pixel_exclude": [27]},
        ]
        sweep_results: List[Dict[str, Any]] = []
        for spec in sweep_defs:
            res = _run_qc(
                catalog=str(args.catalog),
                obsids=obsids,
                data_root=data_root,
                out_dir=out_dir,
                apply_gti=bool(spec["apply_gti"]),
                pixel_exclude=[int(x) for x in (spec["pixel_exclude"] or [])],
                chunk_rows=int(args.chunk_rows),
                fek_band=fek_band,
                variant=str(spec["variant"]),
            )
            try:
                rows = list(csv.DictReader(Path(res["summary_csv"]).open("r", encoding="utf-8", newline="")))
            except Exception:
                rows = []
            sweep_results.append(
                {
                    "variant": str(res.get("variant") or ""),
                    "apply_gti": bool(spec["apply_gti"]),
                    "pixel_exclude": [int(x) for x in (spec["pixel_exclude"] or [])],
                    "summary": _summarize_rows(rows),
                    "summary_csv": str(res.get("summary_csv") or ""),
                    "summary_metrics": str(res.get("summary_metrics") or ""),
                }
            )

        out_sweep = out_dir / "xrism_event_level_qc_sweep_metrics.json"
        _write_json(
            out_sweep,
            {
                "generated_utc": _utc_now(),
                "inputs": {
                    "catalog": str(args.catalog),
                    "obsids": obsids,
                    "data_root": str(data_root),
                    "out_dir": str(out_dir),
                    "chunk_rows": int(args.chunk_rows),
                    "fek_band_keV": [float(fek_band[0]), float(fek_band[1])],
                },
                "variants": sweep_results,
                "note": "手続き差（apply_gti/pixel_exclude）の sweep を保存可能にするための集計。各variantは `xrism_event_level_qc_summary__<variant>.csv` と per-obs JSON を生成する。",
            },
        )
        print(f"[ok] wrote: {out_sweep}")
        return 0

    pixel_exclude = _parse_int_list(str(args.pixel_exclude))
    _ = _run_qc(
        catalog=str(args.catalog),
        obsids=obsids,
        data_root=data_root,
        out_dir=out_dir,
        apply_gti=bool(args.apply_gti),
        pixel_exclude=pixel_exclude,
        chunk_rows=int(args.chunk_rows),
        fek_band=fek_band,
        variant=str(args.variant),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
