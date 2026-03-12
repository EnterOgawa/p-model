#!/usr/bin/env python3
"""
llr_kappa_llr_beta_promotion_gate.py

Step 8.7.47.17:
- LLR Step1-5 prerequisite gates to decide whether kappa->beta promotion is allowed.
- Promote beta_LLR only when all prerequisite gates are pass.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


# 関数: `_safe_rel` の入出力契約と処理意図を定義する。
def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_norm_status` の入出力契約と処理意図を定義する。

def _norm_status(value: Any) -> str:
    s = str(value or "").strip().lower()
    if s in {"pass", "ok"}:
        return "pass"

    if s in {"watch", "pending", "mixed"}:
        return "watch"

    if s in {"reject", "ng", "fail", "failed"}:
        return "reject"

    return "reject"


# 関数: `_combine_status` の入出力契約と処理意図を定義する。

def _combine_status(values: Iterable[str]) -> str:
    norm = [_norm_status(v) for v in values if str(v or "").strip()]
    if not norm:
        return "reject"

    if any(v == "reject" for v in norm):
        return "reject"

    if all(v == "pass" for v in norm):
        return "pass"

    return "watch"


# 関数: `_to_float` の入出力契約と処理意図を定義する。

def _to_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except Exception:
        return None

    if not np.isfinite(x):
        return None

    return x


# 関数: `_sync_outputs` の入出力契約と処理意図を定義する。

def _sync_outputs(paths: Iterable[Path], *, private_root: Path, public_root: Path) -> List[str]:
    out: List[str] = []
    for src in paths:
        rel = src.resolve().relative_to(private_root.resolve())
        dst = public_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        out.append(str(dst))

    return out


# 関数: `_load_paths` の入出力契約と処理意図を定義する。

def _load_paths(root: Path, args: argparse.Namespace) -> Dict[str, Path]:
    # 関数: `_abs` の入出力契約と処理意図を定義する。
    def _abs(p: str) -> Path:
        q = Path(str(p))
        return q if q.is_absolute() else (root / q).resolve()

    return {
        "llr_metrics": _abs(args.llr_metrics),
        "injection_metrics": _abs(args.injection_metrics),
        "cluster_metrics": _abs(args.cluster_metrics),
        "hardware_metrics": _abs(args.hardware_metrics),
        "homogeneous_metrics": _abs(args.homogeneous_metrics),
        "crd_flag_metrics": _abs(args.crd_flag_metrics),
    }


# 関数: `_extract_prereq_rows` の入出力契約と処理意図を定義する。

def _extract_prereq_rows(payloads: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    inj = payloads["injection_metrics"]
    cl = payloads["cluster_metrics"]
    hw = payloads["hardware_metrics"]
    hm = payloads["homogeneous_metrics"]
    crd = payloads["crd_flag_metrics"]

    s1 = _norm_status(((inj.get("recovery") or {}).get("overall_status")))
    s2 = _norm_status(((cl.get("cluster_robust") or {}).get("overall_status")))
    s3 = _norm_status(((hw.get("gate_status") or {}).get("overall_status")))
    s4 = _norm_status(((hm.get("subset_summary") or {}).get("overall_status")))
    s5 = _norm_status(((crd.get("gate_status") or {}).get("overall_status")))
    return [
        {
            "prereq_id": "step1_injection_recovery",
            "label": "Step1 injection/recovery",
            "status": s1,
            "metric": str((inj.get("recovery") or {}).get("linearity_fit") or {}),
            "note": "slope/intercept consistency of synthetic recovery",
        },
        {
            "prereq_id": "step2_cluster_robust",
            "label": "Step2 cluster robust",
            "status": s2,
            "metric": f"overall={((cl.get('cluster_robust') or {}).get('overall_status'))}",
            "note": "cluster-robust station/target/policy consistency",
        },
        {
            "prereq_id": "step3_hardware_period",
            "label": "Step3 hardware-period",
            "status": s3,
            "metric": f"overall={((hw.get('gate_status') or {}).get('overall_status'))}",
            "note": "station_target_hardware_period and pre/post boundary continuity",
        },
        {
            "prereq_id": "step4_homogeneous_subset",
            "label": "Step4 homogeneous subset",
            "status": s4,
            "metric": f"overall={((hm.get('subset_summary') or {}).get('overall_status'))}",
            "note": "subset internal consistency and stability vs reference",
        },
        {
            "prereq_id": "step5_crd_flag_unit_test",
            "label": "Step5 CRD flag test",
            "status": s5,
            "metric": f"overall={((crd.get('gate_status') or {}).get('overall_status'))}",
            "note": "range/system-delay/COM/refraction + CRD-MeritII inversion",
        },
    ]


# 関数: `_write_plot` の入出力契約と処理意図を定義する。

def _write_plot(rows_df: pd.DataFrame, promotion_status: str, out_pdf: Path, out_png: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10.0, 4.8))
    labels = rows_df["label"].astype(str).tolist()
    score_map = {"pass": 2.0, "watch": 1.0, "reject": 0.0}
    color_map = {"pass": "#2ca02c", "watch": "#ffbf00", "reject": "#d62728"}
    scores = [score_map.get(str(v), 0.0) for v in rows_df["status"].astype(str).tolist()]
    colors = [color_map.get(str(v), "#d62728") for v in rows_df["status"].astype(str).tolist()]
    x = np.arange(len(labels), dtype=float)
    ax.bar(x, scores, color=colors, width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks([0.0, 1.0, 2.0])
    ax.set_yticklabels(["reject", "watch", "pass"])
    ax.set_ylabel("status")
    ax.set_title("kappa to beta promotion prerequisites (Step 8.7.47.17)")
    ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"Promotion gate status: {promotion_status}", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    ap = argparse.ArgumentParser(description="LLR kappa->beta promotion gate (Step 8.7.47.17).")
    ap.add_argument("--llr-metrics", type=str, default=str(ROOT / "output" / "public" / "llr" / "llr_kappa_llr_metrics.json"))
    ap.add_argument("--injection-metrics", type=str, default=str(ROOT / "output" / "public" / "llr" / "llr_kappa_llr_injection_recovery_metrics.json"))
    ap.add_argument("--cluster-metrics", type=str, default=str(ROOT / "output" / "public" / "llr" / "llr_kappa_llr_cluster_robust_metrics.json"))
    ap.add_argument("--hardware-metrics", type=str, default=str(ROOT / "output" / "public" / "llr" / "llr_kappa_llr_hardware_period_metrics.json"))
    ap.add_argument("--homogeneous-metrics", type=str, default=str(ROOT / "output" / "public" / "llr" / "llr_kappa_llr_homogeneous_subset_metrics.json"))
    ap.add_argument("--crd-flag-metrics", type=str, default=str(ROOT / "output" / "public" / "llr" / "llr_kappa_llr_crd_flag_unit_test_metrics.json"))
    ap.add_argument("--out-dir", type=str, default=str(ROOT / "output" / "private" / "llr"))
    ap.add_argument("--public-dir", type=str, default=str(ROOT / "output" / "public" / "llr"))
    args = ap.parse_args()

    paths = _load_paths(ROOT, args)
    out_dir = Path(str(args.out_dir))
    public_dir = Path(str(args.public_dir))
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()

    if not public_dir.is_absolute():
        public_dir = (ROOT / public_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)

    payloads: Dict[str, Dict[str, Any]] = {}
    for key, path in paths.items():
        if not path.exists():
            raise RuntimeError(f"missing prerequisite metrics for {key}: {path}")

        payloads[key] = _read_json(path)

    prereq_rows = _extract_prereq_rows(payloads)
    prereq_df = pd.DataFrame(prereq_rows).sort_values(["prereq_id"]).reset_index(drop=True)
    prereq_statuses = prereq_df["status"].astype(str).tolist()
    prereq_overall = _combine_status(prereq_statuses)
    all_pass = bool(all(s == "pass" for s in prereq_statuses))
    promotion_status = "pass" if all_pass else ("watch" if prereq_overall == "watch" else "reject")

    llr_metrics = payloads["llr_metrics"]
    fit = llr_metrics.get("fit") if isinstance(llr_metrics.get("fit"), dict) else {}
    beta_mapping = fit.get("beta_mapping") if isinstance(fit.get("beta_mapping"), dict) else {}
    beta_est = _to_float(beta_mapping.get("beta_est", fit.get("selected_kappa_est")))
    beta_sigma = _to_float(beta_mapping.get("beta_sigma", fit.get("selected_kappa_sigma")))
    beta_abs_z = _to_float(beta_mapping.get("abs_z_beta_minus_1", fit.get("selected_abs_z")))
    beta_source = str(beta_mapping.get("source", "selected_kappa"))
    blocking = prereq_df[prereq_df["status"].astype(str) != "pass"]["prereq_id"].astype(str).tolist()

    decision = {
        "promoted": bool(all_pass),
        "status": promotion_status,
        "rule": "promote only if all Step1-5 prerequisite gates are pass",
        "prereq_overall_status": prereq_overall,
        "blocking_prerequisites": blocking,
        "beta_source_if_promoted": beta_source,
        "beta_est_candidate": beta_est,
        "beta_sigma_candidate": beta_sigma,
        "beta_abs_z_minus_1_candidate": beta_abs_z,
        "beta_est_effective": beta_est if all_pass else None,
        "beta_sigma_effective": beta_sigma if all_pass else None,
        "beta_abs_z_minus_1_effective": beta_abs_z if all_pass else None,
        "hold_mode_effective": "dataset_specific_kappa_amplitude" if not all_pass else "",
    }

    prereq_csv = out_dir / "llr_kappa_llr_beta_promotion_gate_prereq.csv"
    metrics_json = out_dir / "llr_kappa_llr_beta_promotion_gate_metrics.json"
    plot_pdf = out_dir / "llr_kappa_llr_beta_promotion_gate_audit.pdf"
    plot_png = out_dir / "llr_kappa_llr_beta_promotion_gate_audit.png"
    prereq_df.to_csv(prereq_csv, index=False)
    _write_plot(prereq_df, promotion_status=promotion_status, out_pdf=plot_pdf, out_png=plot_png)

    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": {"step": "8.7.47.17"},
        "inputs": {k: _safe_rel(v, ROOT) for k, v in paths.items()},
        "prerequisites": prereq_rows,
        "promotion_decision": decision,
        "gate_status": {"overall_status": promotion_status},
        "outputs": {
            "prereq_csv": _safe_rel(prereq_csv, ROOT),
            "metrics_json": _safe_rel(metrics_json, ROOT),
            "plot_pdf": _safe_rel(plot_pdf, ROOT),
            "plot_png": _safe_rel(plot_png, ROOT),
        },
    }
    metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    produced = [prereq_csv, metrics_json, plot_pdf, plot_png]
    synced = _sync_outputs(produced, private_root=out_dir, public_root=public_dir)
    print(f"Wrote: {prereq_csv}")
    print(f"Wrote: {metrics_json}")
    print(f"Wrote: {plot_pdf}")
    print(f"Wrote: {plot_png}")
    print(f"Synced: {len(synced)} files")
    print(f"Promotion status: {promotion_status}")
    print(f"Promoted: {all_pass}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
