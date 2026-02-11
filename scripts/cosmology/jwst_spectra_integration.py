#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jwst_spectra_integration.py

Phase 4 / Step 4.6（JWST/MAST：スペクトル一次データ）:
JWST x1d（MAST）パイプラインの固定出力（manifest_all / release_waitlist / z_confirmed 等）を集約し、
Table 1 への扱い（参考/screening）と、公開待ち（proprietary）の次回解放日時を「出力として」固定する。

注意:
- JWST x1d は距離指標と独立な一次データ入口だが、現時点では GN-z11 が未公開であり、
  解析の completeness が将来に依存するため、Table 1 では σ評価の対象にしない（info/screening）。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rel(path: Path) -> str:
    try:
        return path.relative_to(_ROOT).as_posix()
    except Exception:
        return path.as_posix()


def _maybe_float(x: object) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    s = str(x).strip()
    if not s:
        return None
    try:
        v = float(s)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def build_metrics(root: Path) -> Dict[str, Any]:
    manifest_all = root / "data" / "cosmology" / "mast" / "jwst_spectra" / "manifest_all.json"
    waitlist_path = root / "output" / "cosmology" / "jwst_spectra_release_waitlist.json"

    j = _read_json(manifest_all) if manifest_all.exists() else {}
    items = j.get("items") if isinstance(j.get("items"), dict) else {}
    targets = j.get("targets") if isinstance(j.get("targets"), list) else []
    n_targets = len(targets) if targets else len(items)

    qc_ok = 0
    z_est_ok = 0
    z_conf_ok = 0
    z_conf_ok_multi_spectra = 0
    z_conf_chi2_gt1 = 0
    z_conf_chi2_gt1_targets: List[str] = []
    missing_local = 0
    for key, it in items.items():
        if not isinstance(it, dict):
            continue
        qc = it.get("qc")
        if isinstance(qc, dict) and bool(qc.get("ok")):
            qc_ok += 1
        z = it.get("z_estimate")
        if isinstance(z, dict) and bool(z.get("ok")):
            z_est_ok += 1
        if isinstance(z, dict) and str(z.get("reason") or "") == "no_local_x1d":
            missing_local += 1
        zc = it.get("z_confirmed")
        if isinstance(zc, dict) and bool(zc.get("ok")):
            z_conf_ok += 1
            try:
                ok_spectra_n = int(zc.get("ok_spectra_n") or 0)
            except Exception:
                ok_spectra_n = 0
            if ok_spectra_n >= 2:
                z_conf_ok_multi_spectra += 1

            chi2_dof = zc.get("chi2_dof")
            try:
                chi2_dof_f = float(chi2_dof) if chi2_dof is not None else None
            except Exception:
                chi2_dof_f = None
            if chi2_dof_f is not None and math.isfinite(chi2_dof_f) and chi2_dof_f > 1.0:
                z_conf_chi2_gt1 += 1
                z_conf_chi2_gt1_targets.append(str(key))

    blocked_targets: List[Dict[str, Any]] = []
    blocked_n = 0
    next_release_utc = ""
    if waitlist_path.exists():
        wl = _read_json(waitlist_path)
        summ = wl.get("summary") if isinstance(wl.get("summary"), dict) else {}
        try:
            blocked_n = int(summ.get("blocked_targets_n") or 0)
        except Exception:
            blocked_n = 0
        for b in wl.get("blocked_targets") or []:
            if not isinstance(b, dict):
                continue
            blocked_targets.append(
                {
                    "target": b.get("target"),
                    "obs_id": b.get("obs_id"),
                    "proposal_id": b.get("proposal_id"),
                    "next_release_utc": b.get("next_release_utc"),
                }
            )
        rels = []
        for b in blocked_targets:
            s = str(b.get("next_release_utc") or "").strip()
            if s:
                rels.append(s)
        next_release_utc = min(rels) if rels else ""

    reasons: List[str] = []
    if blocked_n > 0:
        reasons.append("proprietary_release_wait（公開待ちが残るため completeness が将来に依存）")
    if z_conf_ok <= 0:
        reasons.append("no_z_confirmed_targets")

    table1_status = "info_only"
    if reasons:
        table1_status = "not_adopted_yet"

    return {
        "generated_utc": _utc_now(),
        "inputs": {
            "manifest_all_json": _rel(manifest_all) if manifest_all.exists() else None,
            "release_waitlist_json": _rel(waitlist_path) if waitlist_path.exists() else None,
        },
        "jwst_mast": {
            "targets_n": n_targets,
            "qc_ok_n": qc_ok,
            "z_estimate_ok_n": z_est_ok,
            "z_confirmed_ok_n": z_conf_ok,
            "z_confirmed_multi_spectra_ok_n": z_conf_ok_multi_spectra,
            "z_confirmed_chi2_dof_gt1_n": z_conf_chi2_gt1,
            "z_confirmed_chi2_dof_gt1_targets": sorted(set(z_conf_chi2_gt1_targets)) if z_conf_chi2_gt1_targets else [],
            "missing_local_x1d_n": missing_local,
            "blocked_targets_n": blocked_n,
            "next_release_utc": next_release_utc or None,
            "blocked_targets": blocked_targets,
        },
        "table1": {
            "status": table1_status,
            "policy": "JWST/MAST は距離指標と独立な一次データ入口だが、現時点は公開待ち（例：GN-z11）が残るため Table 1 では参考（σ評価は除外）として扱う。",
            "reasons": reasons,
        },
        "outputs": {
            "integration_metrics_json": _rel(root / "output" / "cosmology" / "jwst_spectra_integration_metrics.json"),
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Integrate JWST/MAST x1d pipeline outputs and freeze Table 1 policy.")
    ap.add_argument("--out-path", default="", help="Override output path (default: output/cosmology/jwst_spectra_integration_metrics.json).")
    args = ap.parse_args(list(argv) if argv is not None else None)

    root = _ROOT
    out_path = Path(str(args.out_path)).expanduser() if str(args.out_path).strip() else (root / "output" / "cosmology" / "jwst_spectra_integration_metrics.json")

    payload = build_metrics(root)
    _write_json(out_path, payload)

    try:
        worklog.append_event(
            {
                "event_type": "jwst_spectra_integration",
                "argv": list(argv) if argv is not None else None,
                "outputs": {"integration_metrics_json": str(out_path)},
            }
        )
    except Exception:
        pass

    print(f"[ok] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
