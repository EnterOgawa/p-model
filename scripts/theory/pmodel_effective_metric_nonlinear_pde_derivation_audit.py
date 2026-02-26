#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 8.7.32.10
有効計量 g_{μν}(P) 下の非線形連立PDE導出鎖を固定し、
Part I / Part II / Part IV の記法同期と閉包判定の追跡可能性を監査する。

固定出力:
- output/public/theory/pmodel_effective_metric_nonlinear_pde_derivation_audit.json
- output/public/theory/pmodel_effective_metric_nonlinear_pde_derivation_audit.csv
- output/public/theory/pmodel_effective_metric_nonlinear_pde_derivation_audit.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(ROOT) not in sys.path` を満たす経路を評価する。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.summary import worklog  # type: ignore
except Exception:  # pragma: no cover
    worklog = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None


# 関数: `_utc_now` の入出力契約と処理意図を定義する。

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_rel` の入出力契約と処理意図を定義する。

def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return path.resolve().as_posix()


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


# 関数: `_extract_section` の入出力契約と処理意図を定義する。

def _extract_section(path: Path, heading_fragment: str) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = None
    level = None
    for idx, line in enumerate(lines):
        s = line.strip()
        # 条件分岐: `s.startswith("#") and heading_fragment in s` を満たす経路を評価する。
        if s.startswith("#") and heading_fragment in s:
            start = idx
            level = len(s) - len(s.lstrip("#"))
            break

    # 条件分岐: `start is None or level is None` を満たす経路を評価する。

    if start is None or level is None:
        return ""

    end = len(lines)
    for idx in range(start + 1, len(lines)):
        s = lines[idx].strip()
        # 条件分岐: `not s.startswith("#")` を満たす経路を評価する。
        if not s.startswith("#"):
            continue

        lv = len(s) - len(s.lstrip("#"))
        # 条件分岐: `lv <= level` を満たす経路を評価する。
        if lv <= level:
            end = idx
            break

    return "\n".join(lines[start:end])


# 関数: `_contains_all` の入出力契約と処理意図を定義する。

def _contains_all(text: str, patterns: Sequence[str]) -> bool:
    return all(re.search(pat, text, flags=re.MULTILINE) is not None for pat in patterns)


# 関数: `_status_bool` の入出力契約と処理意図を定義する。

def _status_bool(v: bool) -> str:
    return "pass" if bool(v) else "reject"


# 関数: `_plot` の入出力契約と処理意図を定義する。

def _plot(doc_ratios: Dict[str, float], flux_caseb: float, flux_direct: float, out_png: Path) -> None:
    # 条件分岐: `plt is None` を満たす経路を評価する。
    if plt is None:
        return

    labels = list(doc_ratios.keys())
    values = [float(doc_ratios[k]) for k in labels]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10.5, 8.0))
    fig.suptitle("Step 8.7.32.10: effective-metric nonlinear PDE derivation audit")

    x = np.arange(len(labels), dtype=float)
    ax0.bar(x, values, color="#1f77b4", alpha=0.9)
    ax0.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
    ax0.set_ylim(0.0, 1.08)
    ax0.set_ylabel("section completeness")
    ax0.set_title("Required derivation blocks present")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)
    ax0.grid(True, axis="y", alpha=0.25)

    fx = np.array([max(float(flux_caseb), 1.0e-30), max(float(flux_direct), 1.0e-30)], dtype=float)
    ax1.bar(np.arange(2), fx, color=["#2ca02c", "#ff7f0e"], alpha=0.9)
    ax1.axhline(1.0e-3, color="#333333", linestyle="--", linewidth=1.0, label="closure threshold")
    ax1.set_yscale("log")
    ax1.set_ylabel("max_flux_rel_std")
    ax1.set_title("Flux-closure consistency (caseB vs direct audit)")
    ax1.set_xticks(np.arange(2))
    ax1.set_xticklabels(["caseB_effective", "direct_ring"])
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(loc="best")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    parser = argparse.ArgumentParser(description="Audit nonlinear effective-metric PDE derivation closure.")
    parser.add_argument("--step-tag", default="8.7.32.10")
    parser.add_argument(
        "--caseb-json",
        type=Path,
        default=ROOT / "output/public/theory/pmodel_vector_metric_choice_audit_caseB_effective.json",
    )
    parser.add_argument(
        "--direct-json",
        type=Path,
        default=ROOT / "output/public/theory/pmodel_rotating_bh_photon_ring_direct_audit.json",
    )
    parser.add_argument(
        "--part1-md",
        type=Path,
        default=ROOT / "doc/paper/10_part1_core_theory.md",
    )
    parser.add_argument(
        "--part2-md",
        type=Path,
        default=ROOT / "doc/paper/11_part2_astrophysics.md",
    )
    parser.add_argument(
        "--part4-md",
        type=Path,
        default=ROOT / "doc/paper/13_part4_verification.md",
    )
    parser.add_argument("--flux-consistency-tol", type=float, default=1.0e-18)
    parser.add_argument("--outdir", type=Path, default=ROOT / "output/public/theory")
    args = parser.parse_args()

    caseb_payload = _read_json(args.caseb_json)
    direct_payload = _read_json(args.direct_json)

    caseb_result = caseb_payload.get("case_result") if isinstance(caseb_payload.get("case_result"), dict) else {}
    caseb_derived = caseb_result.get("derived") if isinstance(caseb_result.get("derived"), dict) else {}
    caseb_summary = caseb_result.get("summary") if isinstance(caseb_result.get("summary"), dict) else {}
    direct_axis = (
        direct_payload.get("axisymmetric_pde_block")
        if isinstance(direct_payload.get("axisymmetric_pde_block"), dict)
        else {}
    )
    direct_diag = (
        direct_axis.get("boundary_diagnostics") if isinstance(direct_axis.get("boundary_diagnostics"), dict) else {}
    )

    caseb_flux = float(caseb_derived.get("max_flux_rel_std") or float("nan"))
    direct_flux = float(direct_diag.get("max_flux_rel_std") or float("nan"))
    flux_delta = abs(caseb_flux - direct_flux) if np.isfinite(caseb_flux) and np.isfinite(direct_flux) else float("inf")

    part1_sec = _extract_section(args.part1_md, "2.7.4")
    part2_sec = _extract_section(args.part2_md, "4.16")
    part4_sec = _extract_section(args.part4_md, "## 12.")

    req_part1 = {
        "nonlinear_pde": _contains_all(
            part1_sec,
            [
                r"(\\nabla|∇)\^\{\(g\(P\)\)\}_\\mu",
                r"\\sqrt\{\|g\(P\)\|\}",
                r"e\^\{-2u\}",
                r"\\mathcal\{N\}_\{0\}\^\{\(2\)\}",
                r"\\mathcal\{N\}_\{\\phi\}\^\{\(2\)\}",
            ],
        ),
        "approx_order": _contains_all(
            part1_sec,
            [
                r"O\(\s*\\epsilon\^2\s*\)",
                r"O\(\s*\\epsilon\^3\s*\)",
                r"g\(P\)\\to\\eta",
            ],
        ),
        "boundary_conditions": _contains_all(
            part1_sec,
            [
                r"r\\to\\infty",
                r"\\theta=0,\\pi",
                r"r\\to r_H\^\+",
            ],
        ),
        "closure_gate": _contains_all(
            part1_sec,
            [
                r"boundary(_|\\_)closure(_|\\_)pass",
                r"max(_|\\_)flux(_|\\_)rel(_|\\_)std",
                r"1\.2527e[−-]16",
            ],
        ),
    }
    req_part2 = {
        "nonlinear_pde": _contains_all(
            part2_sec,
            [
                r"(\\nabla|∇)\^\{\(g\(P\)\)\}_\\mu",
                r"e\^\{-2u\}",
                r"\\mathcal\{N\}_\{0\}\^\{\(2\)\}",
                r"\\mathcal\{N\}_\{\\phi\}\^\{\(2\)\}",
            ],
        ),
        "approx_order": _contains_all(
            part2_sec,
            [
                r"O\(\s*\\epsilon\^2\s*\)",
                r"O\(\s*\\epsilon\^3\s*\)",
                r"g\(P\)\\to\\eta",
            ],
        ),
        "boundary_conditions": _contains_all(
            part2_sec,
            [
                r"r\\to\\infty",
                r"(\\theta|θ)=0,(\\pi|π)",
                r"r\\to r_H\^\+",
            ],
        ),
        "closure_gate": _contains_all(
            part2_sec,
            [
                r"boundary(_|\\_)closure(_|\\_)pass",
                r"max(_|\\_)flux(_|\\_)rel(_|\\_)std",
                r"1\.2527e[−-]16",
            ],
        ),
    }
    req_part4 = {
        "artifact_links": _contains_all(
            part4_sec,
            [
                r"pmodel_effective_metric_nonlinear_pde_derivation_audit\.json",
                r"pmodel_effective_metric_nonlinear_pde_derivation_audit\.csv",
                r"pmodel_effective_metric_nonlinear_pde_derivation_audit\.png",
            ],
        ),
        "script_link": re.search(
            r"scripts/theory/pmodel_effective_metric_nonlinear_pde_derivation_audit\.py", part4_sec
        )
        is not None,
    }

    rows: List[Dict[str, Any]] = []
    for doc_id, checks in (
        ("part1_2_7_4", req_part1),
        ("part2_4_16", req_part2),
        ("part4_12", req_part4),
    ):
        for check_id, passed in checks.items():
            rows.append(
                {
                    "doc_id": doc_id,
                    "check_id": check_id,
                    "value": 1 if passed else 0,
                    "reference": 1,
                    "residual": 0 if passed else -1,
                    "status": _status_bool(bool(passed)),
                }
            )

    gate_rows: List[Dict[str, Any]] = [
        {
            "doc_id": "cross",
            "check_id": "caseb_nonlinear_pde_closure_pass",
            "value": 1 if bool(caseb_derived.get("nonlinear_pde_closure_pass")) else 0,
            "reference": 1,
            "residual": 0 if bool(caseb_derived.get("nonlinear_pde_closure_pass")) else -1,
            "status": _status_bool(bool(caseb_derived.get("nonlinear_pde_closure_pass"))),
        },
        {
            "doc_id": "cross",
            "check_id": "caseb_boundary_closure_pass",
            "value": 1 if bool(caseb_derived.get("boundary_closure_pass")) else 0,
            "reference": 1,
            "residual": 0 if bool(caseb_derived.get("boundary_closure_pass")) else -1,
            "status": _status_bool(bool(caseb_derived.get("boundary_closure_pass"))),
        },
        {
            "doc_id": "cross",
            "check_id": "direct_boundary_closure_pass",
            "value": 1 if bool(direct_axis.get("boundary_closure_pass")) else 0,
            "reference": 1,
            "residual": 0 if bool(direct_axis.get("boundary_closure_pass")) else -1,
            "status": _status_bool(bool(direct_axis.get("boundary_closure_pass"))),
        },
        {
            "doc_id": "cross",
            "check_id": "flux_value_consistency",
            "value": flux_delta,
            "reference": 0.0,
            "residual": flux_delta,
            "status": _status_bool(bool(np.isfinite(flux_delta) and flux_delta <= float(args.flux_consistency_tol))),
        },
    ]
    rows.extend(gate_rows)

    hard_reject_n = sum(1 for r in rows if str(r["status"]) == "reject")
    overall_status = "pass" if hard_reject_n == 0 else "reject"
    decision = (
        "effective_metric_nonlinear_pde_chain_fixed" if overall_status == "pass" else "effective_metric_nonlinear_pde_chain_incomplete"
    )

    part1_ratio = float(sum(1 for v in req_part1.values() if v) / max(1, len(req_part1)))
    part2_ratio = float(sum(1 for v in req_part2.values() if v) / max(1, len(req_part2)))
    part4_ratio = float(sum(1 for v in req_part4.values() if v) / max(1, len(req_part4)))

    outdir = args.outdir
    out_json = outdir / "pmodel_effective_metric_nonlinear_pde_derivation_audit.json"
    out_csv = outdir / "pmodel_effective_metric_nonlinear_pde_derivation_audit.csv"
    out_png = outdir / "pmodel_effective_metric_nonlinear_pde_derivation_audit.png"

    payload: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "schema": "wavep.theory.effective_metric_nonlinear_pde_derivation_audit.v1",
        "phase": {"phase": 8, "step": str(args.step_tag), "name": "effective-metric nonlinear PDE derivation audit"},
        "intent": "Lock the derivation chain for nonlinear axisymmetric PDEs under g_{mu nu}(P) and synchronize Part I/II/IV notation.",
        "inputs": {
            "caseb_json": _rel(args.caseb_json),
            "direct_json": _rel(args.direct_json),
            "part1_md": _rel(args.part1_md),
            "part2_md": _rel(args.part2_md),
            "part4_md": _rel(args.part4_md),
            "flux_consistency_tol": float(args.flux_consistency_tol),
        },
        "equations": {
            "master_eq": "∇^{(g(P))}_μ F_{(P)}^{μν}=0,  F_{(P)}^{μν}=g^{μα}(P)g^{νβ}(P)(∂_αP_β-∂_βP_α)",
            "nonlinear_order": "keep O(ε^2), drop O(ε^3) in e^{-2u} expansion",
            "closure_gate": "boundary_closure_pass && nonlinear_pde_closure_pass && max_flux_rel_std<=1e-3",
        },
        "derived": {
            "caseb": {
                "metric_choice_decision": caseb_payload.get("metric_choice_decision"),
                "overall_status": caseb_summary.get("overall_status"),
                "decision": caseb_summary.get("decision"),
                "nonlinear_pde_closure_pass": bool(caseb_derived.get("nonlinear_pde_closure_pass")),
                "boundary_closure_pass": bool(caseb_derived.get("boundary_closure_pass")),
                "max_flux_rel_std": caseb_flux,
            },
            "direct_axisymmetric": {
                "boundary_closure_pass": bool(direct_axis.get("boundary_closure_pass")),
                "formulation_complete": bool(direct_axis.get("formulation_complete")),
                "max_flux_rel_std": direct_flux,
            },
            "cross_consistency": {
                "flux_abs_delta": flux_delta,
                "flux_consistent": bool(np.isfinite(flux_delta) and flux_delta <= float(args.flux_consistency_tol)),
            },
            "document_completeness": {
                "part1_2_7_4_ratio": part1_ratio,
                "part2_4_16_ratio": part2_ratio,
                "part4_12_ratio": part4_ratio,
            },
        },
        "document_checks": {
            "part1_2_7_4": req_part1,
            "part2_4_16": req_part2,
            "part4_12": req_part4,
        },
        "rows": rows,
        "decision": {
            "overall_status": overall_status,
            "decision": decision,
            "hard_reject_n": int(hard_reject_n),
            "watch_n": 0,
        },
        "outputs": {"audit_json": _rel(out_json), "audit_csv": _rel(out_csv), "audit_png": _rel(out_png)},
        "falsification_gate": {
            "reject_if": [
                "Any required derivation block is missing in Part I 2.7.4 / Part II 4.16 / Part IV 12.",
                "caseB nonlinear_pde_closure_pass is false.",
                "boundary_closure_pass is false in either caseB or direct axisymmetric audit.",
                "abs(caseB.max_flux_rel_std - direct.max_flux_rel_std) > flux_consistency_tol.",
            ],
            "watch_if": [],
        },
    }

    _write_json(out_json, payload)
    _write_csv(out_csv, rows, fieldnames=["doc_id", "check_id", "value", "reference", "residual", "status"])
    _plot(
        {
            "Part I 2.7.4": part1_ratio,
            "Part II 4.16": part2_ratio,
            "Part IV 12": part4_ratio,
        },
        caseb_flux,
        direct_flux,
        out_png,
    )

    # 条件分岐: `worklog is not None` を満たす経路を評価する。
    if worklog is not None:
        try:
            worklog.append_event(
                {
                    "event_type": "theory_effective_metric_nonlinear_pde_derivation_audit",
                    "phase": str(args.step_tag),
                    "overall_status": overall_status,
                    "decision": decision,
                    "hard_reject_n": int(hard_reject_n),
                    "caseb_max_flux_rel_std": caseb_flux,
                    "direct_max_flux_rel_std": direct_flux,
                    "flux_abs_delta": flux_delta,
                    "outputs": {"json": _rel(out_json), "csv": _rel(out_csv), "png": _rel(out_png)},
                }
            )
        except Exception:
            pass

    print(f"[ok] wrote {out_json}")
    print(f"[ok] wrote {out_csv}")
    # 条件分岐: `plt is not None` を満たす経路を評価する。
    if plt is not None:
        print(f"[ok] wrote {out_png}")
    else:
        print("[warn] matplotlib not available; png not generated")

    print(f"[done] overall_status={overall_status} decision={decision} hard_reject_n={hard_reject_n}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
