#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weak_field_public_pack.py

Optional publication helper (Phase 8):
Promote weak-field verification artifacts (Cassini/Viking/Mercury/GPS + theory freeze) into
`output/public/` and emit a citable integrated "falsification pack" JSON under
`output/public/weak_field/`.

This script does NOT re-run upstream computations. It copies existing fixed outputs from
`output/private/` (or already-public LLR pack) and writes sanitized summary JSONs that
reference only `output/public/...` paths (public repo should not depend on `output/private` junctions).
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
    from scripts.summary import worklog  # type: ignore
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


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact_record(path: Path) -> Dict[str, Any]:
    return {
        "path": _rel(path),
        "sha256": _sha256(path) if path.exists() else None,
        "bytes": int(path.stat().st_size) if path.exists() else None,
    }


def _copy_file(src: Path, dst: Path, *, overwrite: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(str(src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return
    shutil.copy2(src, dst)


def _rewrite_paths(obj: Any) -> Any:
    """
    Rewrite internal repo-relative paths so the public materials only reference `output/public/...`.
    Applies to string values (recursively).
    """

    replacements: Sequence[Tuple[str, str]] = (
        # Prefer explicit private paths (these appear in some summary JSONs).
        ("output/private/llr/batch/", "output/public/llr/"),
        ("output/private/llr/", "output/public/llr/"),
        ("output/private/summary/", "output/public/weak_field/"),
        ("output/private/theory/", "output/public/theory/"),
        ("output/private/cassini/", "output/public/cassini/"),
        ("output/private/viking/", "output/public/viking/"),
        ("output/private/mercury/", "output/public/mercury/"),
        ("output/private/gps/", "output/public/gps/"),
        # Also rewrite the legacy (junction) paths used by other components.
        ("output/llr/batch/", "output/public/llr/"),
        ("output/llr/", "output/public/llr/"),
        ("output/summary/", "output/public/weak_field/"),
        ("output/theory/", "output/public/theory/"),
        ("output/cassini/", "output/public/cassini/"),
        ("output/viking/", "output/public/viking/"),
        ("output/mercury/", "output/public/mercury/"),
        ("output/gps/", "output/public/gps/"),
    )

    def _rw_str(s: str) -> str:
        if "output/public/" in s:
            return s
        out = s
        for a, b in replacements:
            out = out.replace(a, b)
        return out

    if isinstance(obj, dict):
        return {k: _rewrite_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rewrite_paths(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_rewrite_paths(v) for v in obj)
    if isinstance(obj, str):
        return _rw_str(obj)
    return obj


def _sanitize_test_matrix(matrix: Dict[str, Any]) -> Dict[str, Any]:
    """
    Public version of weak_field_test_matrix:
    - drop doc/* and cache directory inputs/refs
    - drop PPTX outputs (not cited; keep PNG/CSV/JSON)
    - replace LLR outputs with the public LLR pack reference
    - rewrite output paths to output/public/*
    """

    m = json.loads(json.dumps(matrix, ensure_ascii=False))

    # Common frozen parameters path (ensure public).
    try:
        common = m.get("common")
        if isinstance(common, dict):
            fp = common.get("frozen_parameters")
            if isinstance(fp, dict) and isinstance(fp.get("path"), str):
                fp["path"] = fp["path"].replace("output/theory/", "output/public/theory/")
            if isinstance(fp, dict) and isinstance(fp.get("policy"), dict):
                pol = fp["policy"]
                if isinstance(pol.get("delta_source"), str):
                    pol["delta_source"] = pol["delta_source"].replace("output/theory/", "output/public/theory/")
    except Exception:
        pass

    tests = m.get("tests")
    if not isinstance(tests, list):
        return _rewrite_paths(m)

    for t in tests:
        if not isinstance(t, dict):
            continue
        tid = t.get("id")

        def _filter_items(items: Any) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            if not isinstance(items, list):
                return out
            for it in items:
                if not isinstance(it, dict):
                    continue
                p = it.get("path")
                if not isinstance(p, str):
                    continue
                if p.startswith("doc/"):
                    continue
                if p.startswith("output/viking/horizons_cache"):
                    continue
                out.append(it)
            return out

        if "inputs" in t:
            t["inputs"] = _filter_items(t.get("inputs"))
        if "refs" in t:
            t["refs"] = _filter_items(t.get("refs"))

        # Drop PPTX outputs.
        outs = t.get("outputs")
        if isinstance(outs, list):
            kept: List[Dict[str, Any]] = []
            for it in outs:
                if not isinstance(it, dict):
                    continue
                p = it.get("path")
                if isinstance(p, str) and p.lower().endswith(".pptx"):
                    continue
                kept.append(it)
            t["outputs"] = kept

        # LLR: prefer the public pack as the citable entry.
        if tid == "llr_batch":
            t["outputs"] = [
                {"path": "output/public/llr/llr_falsification_pack.json", "exists": True, "note": "public pack (citable)"},
                {"path": "output/public/llr/llr_shapiro_ablations_overall.png", "exists": True, "note": "key audit figure"},
            ]

    return _rewrite_paths(m)


def _copy_topic(
    *,
    topic: str,
    src_dir: Path,
    dst_dir: Path,
    filenames: Sequence[str],
    overwrite: bool,
) -> Dict[str, Dict[str, Any]]:
    copied: Dict[str, Dict[str, Any]] = {}
    for fname in filenames:
        src = src_dir / fname
        dst = dst_dir / fname
        _copy_file(src, dst, overwrite=overwrite)
        copied[fname] = _artifact_record(dst)
    return copied


def run(*, overwrite: bool) -> Tuple[Path, List[str]]:
    warnings: List[str] = []

    # Inputs (private summaries)
    priv_summary = _ROOT / "output" / "private" / "summary"
    priv_matrix = priv_summary / "weak_field_test_matrix.json"
    priv_templates = priv_summary / "weak_field_systematics_templates.json"
    priv_consistency = priv_summary / "weak_field_longterm_consistency.json"
    priv_consistency_png = priv_summary / "weak_field_longterm_consistency.png"
    priv_falsification = priv_summary / "weak_field_falsification.json"

    for p in (priv_matrix, priv_templates, priv_consistency, priv_falsification):
        if not p.exists():
            raise FileNotFoundError(f"missing required input: {_rel(p)}")

    # Public output dirs
    out_public = _ROOT / "output" / "public"
    out_weak = out_public / "weak_field"
    out_weak.mkdir(parents=True, exist_ok=True)

    copied_by_topic: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # Promote theory freeze (required for consistency across weak-field tests)
    copied_by_topic["theory"] = _copy_topic(
        topic="theory",
        src_dir=_ROOT / "output" / "private" / "theory",
        dst_dir=out_public / "theory",
        filenames=(
            "frozen_parameters.json",
            "delta_saturation_constraints.json",
            "delta_saturation_constraints.png",
        ),
        overwrite=overwrite,
    )

    # Promote Cassini (Doppler / Shapiro)
    copied_by_topic["cassini"] = _copy_topic(
        topic="cassini",
        src_dir=_ROOT / "output" / "private" / "cassini",
        dst_dir=out_public / "cassini",
        filenames=(
            "cassini_shapiro_y_full.csv",
            "cassini_fig2_overlay_full.png",
            "cassini_fig2_overlay_zoom10d.png",
            "cassini_fig2_residuals.png",
            "cassini_fig2_metrics.csv",
            "cassini_fig2_run_metadata.json",
            "cassini_beta_sweep_rmse.csv",
            "cassini_beta_sweep_rmse.png",
        ),
        overwrite=overwrite,
    )

    # Promote Viking (Shapiro peak scan)
    copied_by_topic["viking"] = _copy_topic(
        topic="viking",
        src_dir=_ROOT / "output" / "private" / "viking",
        dst_dir=out_public / "viking",
        filenames=(
            "viking_shapiro_result.csv",
            "viking_p_model_vs_measured_no_arrow.png",
        ),
        overwrite=overwrite,
    )

    # Promote Mercury (perihelion precession)
    copied_by_topic["mercury"] = _copy_topic(
        topic="mercury",
        src_dir=_ROOT / "output" / "private" / "mercury",
        dst_dir=out_public / "mercury",
        filenames=(
            "mercury_orbit.png",
            "mercury_precession_metrics.json",
            "mercury_perihelion_shifts.csv",
        ),
        overwrite=overwrite,
    )

    # Promote GPS (satellite clock residuals)
    copied_by_topic["gps"] = _copy_topic(
        topic="gps",
        src_dir=_ROOT / "output" / "private" / "gps",
        dst_dir=out_public / "gps",
        filenames=(
            "summary_batch.csv",
            "gps_clock_residuals_all_31.png",
            "gps_residual_compare_G01.png",
            "gps_rms_compare.png",
            "gps_relativistic_correction_G02.png",
            "gps_compare_metrics.json",
        ),
        overwrite=overwrite,
    )

    # Public LLR pack (already published; reference as input)
    llr_pack = _ROOT / "output" / "public" / "llr" / "llr_falsification_pack.json"
    if not llr_pack.exists():
        warnings.append(f"missing public LLR pack (expected): {_rel(llr_pack)}")
        llr_pack_rec = None
    else:
        llr_pack_rec = _artifact_record(llr_pack)

    # Build sanitized public summary JSONs
    priv_matrix_obj = _read_json(priv_matrix)
    pub_matrix_obj = _sanitize_test_matrix(priv_matrix_obj)
    pub_templates_obj = _rewrite_paths(_read_json(priv_templates))
    pub_consistency_obj = _rewrite_paths(_read_json(priv_consistency))
    pub_falsification_obj = _rewrite_paths(_read_json(priv_falsification))

    pub_matrix_path = out_weak / "weak_field_test_matrix.json"
    pub_templates_path = out_weak / "weak_field_systematics_templates.json"
    pub_consistency_path = out_weak / "weak_field_longterm_consistency.json"
    pub_falsification_path = out_weak / "weak_field_falsification.json"
    _write_json(pub_matrix_path, pub_matrix_obj)
    _write_json(pub_templates_path, pub_templates_obj)
    _write_json(pub_consistency_path, pub_consistency_obj)
    _write_json(pub_falsification_path, pub_falsification_obj)

    copied_by_topic["weak_field"] = {
        "weak_field_test_matrix.json": _artifact_record(pub_matrix_path),
        "weak_field_systematics_templates.json": _artifact_record(pub_templates_path),
        "weak_field_longterm_consistency.json": _artifact_record(pub_consistency_path),
        "weak_field_falsification.json": _artifact_record(pub_falsification_path),
    }

    # Copy the consistency PNG if present.
    if priv_consistency_png.exists():
        pub_png = out_weak / "weak_field_longterm_consistency.png"
        _copy_file(priv_consistency_png, pub_png, overwrite=overwrite)
        copied_by_topic["weak_field"]["weak_field_longterm_consistency.png"] = _artifact_record(pub_png)
    else:
        warnings.append(f"missing weak_field_longterm_consistency.png (optional): {_rel(priv_consistency_png)}")

    # Read frozen parameter values from public artifact (for pack summary)
    frozen_pub = out_public / "theory" / "frozen_parameters.json"
    frozen_vals: Dict[str, Any] = {"path": _rel(frozen_pub), "exists": frozen_pub.exists()}
    if frozen_pub.exists():
        try:
            fz = _read_json(frozen_pub)
            frozen_vals.update({k: fz.get(k) for k in ("beta", "beta_sigma", "gamma_pmodel", "gamma_pmodel_sigma", "delta")})
        except Exception:
            warnings.append(f"failed to parse frozen parameters: {_rel(frozen_pub)}")

    # Integrated public pack
    pack_path = out_weak / "weak_field_falsification_pack.json"
    pack: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "domain": "weak_field",
        "version": 1,
        "intent": "Promote weak-field verification outputs into tracked public artifacts and provide an integrated falsification pack.",
        "inputs": {
            "frozen_parameters": frozen_vals,
            "llr_public_pack": llr_pack_rec,
            "public_summaries": {
                "weak_field_test_matrix_json": _artifact_record(pub_matrix_path),
                "weak_field_systematics_templates_json": _artifact_record(pub_templates_path),
                "weak_field_longterm_consistency_json": _artifact_record(pub_consistency_path),
                "weak_field_falsification_json": _artifact_record(pub_falsification_path),
            },
        },
        "policy": pub_falsification_obj.get("policy"),
        "criteria": pub_falsification_obj.get("criteria"),
        "summary": pub_falsification_obj.get("summary"),
        "consistency_results": pub_consistency_obj.get("results"),
        "artifacts": {"dir": _rel(out_public), "copied": copied_by_topic},
        "warnings": warnings,
    }

    _write_json(pack_path, pack)

    if worklog is not None:
        try:
            worklog.append_event(
                {
                    "event": "weak_field_public_pack",
                    "script": _rel(Path(__file__)),
                    "outputs": [
                        pack_path,
                        pub_matrix_path,
                        pub_templates_path,
                        pub_consistency_path,
                        pub_falsification_path,
                    ],
                    "public_topics": sorted([k for k in copied_by_topic.keys() if k not in ("weak_field",)]),
                    "warnings_n": len(warnings),
                }
            )
        except Exception:
            pass

    return pack_path, warnings


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Promote weak-field artifacts into output/public and build an integrated pack.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing public artifacts.")
    args = ap.parse_args(argv)

    pack_path, warnings = run(overwrite=bool(args.overwrite))
    print(f"[ok] wrote: {_rel(pack_path)}")
    for w in warnings:
        print(f"[warn] {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
