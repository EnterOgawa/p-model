#!/usr/bin/env python3
"""Freeze ODF-raw terminal Watch policy for Cassini absolute-beta promotion.

Purpose:
- Implement roadmap step 8.7.45.3 as machine-readable policy output.
- If ODF-raw promotion gates remain unmet, fix ODF raw as Reference/Watch.
- Explicitly mark the primary beta judgement route as TDF + scalar-limit.

Inputs:
- output/cassini/cassini_odf_raw_if_manifest.json
- output/cassini/cassini_odf_sign_recordlevel_reaudit.json
- output/cassini/cassini_beta_direct_fit_cross_source_metrics.json
- output/cassini/cassini_odf_sign_media_closure_audit.json

Outputs:
- output/cassini/cassini_odf_raw_watch_terminal_gate.json
- output/cassini/cassini_odf_raw_watch_terminal_gate.csv
- synced copies in output/public/cassini/
"""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence


# 関数: `_repo_root` の入出力契約と処理意図を定義する。
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    return payload if isinstance(payload, dict) else {}


# 関数: `_status_of_gate` の入出力契約と処理意図を定義する。

def _status_of_gate(gates: Dict[str, object], key: str) -> str:
    gate = gates.get(key) if isinstance(gates.get(key), dict) else {}
    return str(gate.get("status") or "").strip().lower()


# 関数: `_append_unique` の入出力契約と処理意図を定義する。

def _append_unique(dst: List[str], src: Sequence[object]) -> None:
    for one in src:
        text = str(one).strip()
        if text and text not in dst:
            dst.append(text)


# 関数: `_sync_public` の入出力契約と処理意図を定義する。

def _sync_public(root: Path, names: Sequence[str]) -> None:
    src_dir = root / "output" / "cassini"
    dst_dir = root / "output" / "public" / "cassini"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    root = _repo_root()
    out_dir = root / "output" / "cassini"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "cassini_odf_raw_if_manifest.json"
    sign_reaudit_path = out_dir / "cassini_odf_sign_recordlevel_reaudit.json"
    cross_metrics_path = out_dir / "cassini_beta_direct_fit_cross_source_metrics.json"
    sign_media_path = out_dir / "cassini_odf_sign_media_closure_audit.json"

    manifest = _read_json(manifest_path)
    sign_reaudit = _read_json(sign_reaudit_path)
    cross_metrics = _read_json(cross_metrics_path)
    sign_media = _read_json(sign_media_path)

    keys0 = manifest.get("keys") if isinstance(manifest.get("keys"), dict) else {}
    sign_key = keys0.get("doppler_sign_convention") if isinstance(keys0.get("doppler_sign_convention"), dict) else {}
    sign_terminal_gate_pass = bool(sign_key.get("terminal_gate_pass"))
    sign_terminal_status = str(sign_key.get("terminal_gate_status") or "").strip().lower()
    sign_terminal_cov = sign_key.get("terminal_gate_selected_coverage_ratio_points")
    sign_terminal_cov_gate = sign_key.get("terminal_gate_min_coverage_ratio_points")
    sign_watch_codes = sign_key.get("watch_reason_codes") if isinstance(sign_key.get("watch_reason_codes"), list) else []

    convergence = cross_metrics.get("convergence_audit") if isinstance(cross_metrics.get("convergence_audit"), dict) else {}
    convergence_status = str(convergence.get("recommended_status") or "").strip().lower()
    convergence_reasons = convergence.get("reasons") if isinstance(convergence.get("reasons"), list) else []
    convergence_gates = convergence.get("gates") if isinstance(convergence.get("gates"), dict) else {}
    gate_odf_z = _status_of_gate(convergence_gates, "odf_vs_tdf_z_best")
    gate_odf_corr = _status_of_gate(convergence_gates, "odf_corr_floor")
    gate_sign_terminal = _status_of_gate(convergence_gates, "odf_sign_terminal_gate")

    sign_media_status = str(sign_media.get("recommended_status") or "").strip().lower()
    sign_media_reasons = sign_media.get("reasons") if isinstance(sign_media.get("reasons"), list) else []

    promotion_checks = [
        {
            "gate": "manifest_sign_terminal_gate",
            "status": "pass" if sign_terminal_gate_pass else "watch",
            "pass_condition": "terminal_gate_pass == true",
            "value": bool(sign_terminal_gate_pass),
        },
        {
            "gate": "cross_odf_vs_tdf_z_best",
            "status": gate_odf_z if gate_odf_z in {"pass", "watch", "reject"} else "watch",
            "pass_condition": "odf_vs_tdf_z_best.status == pass",
            "value": gate_odf_z,
        },
        {
            "gate": "cross_odf_corr_floor",
            "status": gate_odf_corr if gate_odf_corr in {"pass", "watch", "reject"} else "watch",
            "pass_condition": "odf_corr_floor.status == pass",
            "value": gate_odf_corr,
        },
        {
            "gate": "sign_media_closure",
            "status": "pass" if sign_media_status == "pass_candidate" else "watch",
            "pass_condition": "sign_media.recommended_status == pass_candidate",
            "value": sign_media_status,
        },
    ]
    promotion_ready = all(str(one.get("status") or "").lower() == "pass" for one in promotion_checks)

    terminal_status = "pass_candidate" if promotion_ready else "watch"
    odf_raw_role = "promotion_candidate" if promotion_ready else "reference_watch_fixed"
    primary_route = "tdf_plus_scalar_limit"
    policy_statement = (
        "ODF raw remains Reference/Watch; primary beta judgement uses TDF + scalar-limit."
        if not promotion_ready
        else "ODF raw promotion candidate is available; keep TDF + scalar-limit as cross-check."
    )

    reasons: List[str] = []
    _append_unique(reasons, sign_watch_codes)
    _append_unique(reasons, convergence_reasons)
    _append_unique(reasons, sign_media_reasons)
    if not promotion_ready:
        _append_unique(reasons, ["odf_raw_reference_watch_fixed", "primary_beta_route_tdf_plus_scalar_limit"])

    payload: Dict[str, object] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "manifest_json": str(manifest_path),
            "sign_recordlevel_reaudit_json": str(sign_reaudit_path),
            "cross_source_metrics_json": str(cross_metrics_path),
            "sign_media_closure_json": str(sign_media_path),
        },
        "terminal_policy": {
            "status": terminal_status,
            "odf_raw_role": odf_raw_role,
            "primary_beta_route": primary_route,
            "policy_statement": policy_statement,
            "promotion_ready": bool(promotion_ready),
        },
        "promotion_checks": promotion_checks,
        "sign_terminal_snapshot": {
            "terminal_gate_status": sign_terminal_status,
            "terminal_gate_pass": bool(sign_terminal_gate_pass),
            "terminal_gate_selected_coverage_ratio_points": sign_terminal_cov,
            "terminal_gate_min_coverage_ratio_points": sign_terminal_cov_gate,
            "cross_gate_odf_sign_terminal": gate_sign_terminal,
        },
        "convergence_snapshot": {
            "recommended_status": convergence_status,
            "gate_odf_vs_tdf_z_best": gate_odf_z,
            "gate_odf_corr_floor": gate_odf_corr,
        },
        "sign_media_snapshot": {
            "recommended_status": sign_media_status,
        },
        "reasons": reasons,
        "re_promotion_conditions": [
            "manifest.keys.doppler_sign_convention.terminal_gate_pass == true",
            "cross_source.convergence_audit.gates.odf_vs_tdf_z_best.status == pass",
            "cross_source.convergence_audit.gates.odf_corr_floor.status == pass",
            "sign_media_closure.recommended_status == pass_candidate",
        ],
    }

    out_json = out_dir / "cassini_odf_raw_watch_terminal_gate.json"
    out_csv = out_dir / "cassini_odf_raw_watch_terminal_gate.csv"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        writer.writerow(["terminal_status", terminal_status])
        writer.writerow(["odf_raw_role", odf_raw_role])
        writer.writerow(["primary_beta_route", primary_route])
        writer.writerow(["promotion_ready", str(bool(promotion_ready)).lower()])
        writer.writerow(["policy_statement", policy_statement])
        writer.writerow(["sign_terminal_gate_status", sign_terminal_status])
        writer.writerow(["sign_terminal_gate_pass", str(bool(sign_terminal_gate_pass)).lower()])
        writer.writerow(["sign_terminal_gate_selected_coverage_ratio_points", str(sign_terminal_cov)])
        writer.writerow(["sign_terminal_gate_min_coverage_ratio_points", str(sign_terminal_cov_gate)])
        writer.writerow(["cross_gate_odf_sign_terminal", gate_sign_terminal])
        writer.writerow(["cross_gate_odf_vs_tdf_z_best", gate_odf_z])
        writer.writerow(["cross_gate_odf_corr_floor", gate_odf_corr])
        writer.writerow(["sign_media_recommended_status", sign_media_status])
        writer.writerow(["reasons", ",".join(reasons)])

    _sync_public(root, [out_json.name, out_csv.name])
    print("Wrote:", out_json)
    print("Wrote:", out_csv)
    print("Synced:", root / "output" / "public" / "cassini")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
