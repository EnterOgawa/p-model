#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
action_principle_el_derivation_audit.py

Step 8.7.9:
最小作用
  L = |D_mu P|^2 - V(|P|) - (1/4) F_munu F^munu
に対して、導出で使う最小監査項目（局所位相変換下の共変性・不変性）を
固定出力する。

出力:
  - output/public/quantum/action_principle_el_derivation_audit.json
  - output/public/quantum/action_principle_el_derivation_audit.csv
  - output/public/quantum/action_principle_el_derivation_audit.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(ROOT) not in sys.path` を満たす経路を評価する。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summary import worklog  # noqa: E402


# 関数: `_iso_utc_now` の入出力契約と処理意図を定義する。
def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_rel` の入出力契約と処理意図を定義する。

def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


# 関数: `_d_dx` の入出力契約と処理意図を定義する。

def _d_dx(values: np.ndarray, dx: float) -> np.ndarray:
    n = int(values.size)
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    return np.fft.ifft(1j * k * np.fft.fft(values))


# 関数: `_criterion` の入出力契約と処理意図を定義する。

def _criterion(
    *,
    cid: str,
    metric: str,
    value: float,
    threshold: float,
    operator: str = "<=",
    note: str,
) -> Dict[str, Any]:
    passed = bool(value <= threshold) if operator == "<=" else bool(value >= threshold)
    return {
        "id": cid,
        "metric": metric,
        "value": float(value),
        "threshold": float(threshold),
        "operator": operator,
        "pass": passed,
        "note": note,
    }


# 関数: `build_pack` の入出力契約と処理意図を定義する。

def build_pack() -> Dict[str, Any]:
    n = 1024
    x = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    dx = float(x[1] - x[0])
    q = 0.37
    eps = 1.0e-15

    amp = 1.0 + 0.06 * np.cos(3.0 * x)
    phase = 0.20 * np.sin(2.0 * x) + 0.03 * np.cos(5.0 * x)
    p_field = amp * np.exp(1j * phase)
    a_field = 0.08 * np.sin(1.7 * x) + 0.03 * np.cos(3.2 * x)
    chi = 0.05 * np.sin(4.0 * x) + 0.02 * np.cos(7.0 * x)

    dp = _d_dx(p_field, dx)
    dchi = _d_dx(chi, dx)
    cov_d = dp + 1j * q * a_field * p_field

    p_g = p_field * np.exp(1j * chi)
    a_g = a_field - dchi / q
    dp_g = _d_dx(p_g, dx)
    cov_d_g = dp_g + 1j * q * a_g * p_g

    transport = np.exp(1j * chi) * cov_d
    cov_residual = float(np.max(np.abs(cov_d_g - transport)) / max(float(np.max(np.abs(cov_d))), eps))

    l_kin = np.abs(cov_d) ** 2
    l_kin_g = np.abs(cov_d_g) ** 2
    lagrangian_residual = float(np.max(np.abs(l_kin_g - l_kin)) / max(float(np.max(np.abs(l_kin))), eps))

    j_spatial = 1j * q * (np.conj(p_field) * cov_d - np.conj(cov_d) * p_field)
    j_spatial_g = 1j * q * (np.conj(p_g) * cov_d_g - np.conj(cov_d_g) * p_g)
    current_residual = float(np.max(np.abs(j_spatial_g - j_spatial)) / max(float(np.max(np.abs(j_spatial))), eps))
    current_imag_ratio = float(np.max(np.abs(np.imag(j_spatial))) / max(float(np.max(np.abs(np.real(j_spatial)))), eps))

    criteria: List[Dict[str, Any]] = [
        _criterion(
            cid="covariant_derivative_gauge_covariance",
            metric="max_rel_error(D_mu P)",
            value=cov_residual,
            threshold=5.0e-8,
            note="D_mu P のゲージ共変性（D_mu'P' = exp(iχ) D_mu P）を有限差分で監査。",
        ),
        _criterion(
            cid="kinetic_density_gauge_invariance",
            metric="max_rel_error(|D_mu P|^2)",
            value=lagrangian_residual,
            threshold=5.0e-8,
            note="運動項 |D_mu P|^2 のゲージ不変性を有限差分で監査。",
        ),
        _criterion(
            cid="noether_current_gauge_invariance",
            metric="max_rel_error(j_mu)",
            value=current_residual,
            threshold=5.0e-8,
            note="Noether 電流 j_mu のゲージ不変性を有限差分で監査。",
        ),
        _criterion(
            cid="noether_current_realness",
            metric="max_abs_imag_over_real(j_spatial)",
            value=current_imag_ratio,
            threshold=5.0e-10,
            note="電流密度の実数性（数値誤差レベル）を確認。",
        ),
    ]

    fail_ids = [str(c["id"]) for c in criteria if not bool(c.get("pass"))]
    status = "pass" if not fail_ids else "watch"

    return {
        "generated_utc": _iso_utc_now(),
        "phase": {"phase": 8, "step": "8.7.9", "name": "Action-principle EL derivation audit"},
        "equations": {
            "lagrangian_density": "L = |D_mu P|^2 - V(|P|) - (1/4) F_munu F^munu",
            "covariant_derivative": "D_mu = partial_mu + i q A_mu",
            "field_strength": "F_munu = partial_mu A_nu - partial_nu A_mu",
            "el_for_P_conjugate": "D_mu D^mu P + dV/dP* = 0",
            "el_for_A_nu": "partial_mu F^munu = j^nu,  j^nu = i q [P* D^nu P - (D^nu P)* P]",
            "continuity": "partial_nu j^nu = 0 (from antisymmetry of F_munu and EL consistency)",
        },
        "assumptions": [
            "場 P, A_mu は境界を含めて十分滑らか（最小で C1）",
            "変分は境界で消える（境界項は 0）",
            "高次導関数項は最小モデルでは採用しない",
        ],
        "numerical_audit": {
            "grid_points": int(n),
            "periodic_boundary": True,
            "charge_q": float(q),
            "criteria": criteria,
            "status": status,
            "fail_ids": fail_ids,
        },
        "decision": {
            "route_a_el_derivation_gate": status,
            "rule": "All invariance checks pass => pass; otherwise watch.",
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, criteria: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "metric", "value", "threshold", "operator", "pass", "note"],
        )
        writer.writeheader()
        for row in criteria:
            writer.writerow(row)


# 関数: `_plot` の入出力契約と処理意図を定義する。

def _plot(path: Path, payload: Dict[str, Any]) -> None:
    audit = payload.get("numerical_audit") if isinstance(payload.get("numerical_audit"), dict) else {}
    criteria = audit.get("criteria") if isinstance(audit.get("criteria"), list) else []

    labels: List[str] = []
    scores: List[float] = []
    colors: List[str] = []
    for row in criteria:
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        value = float(row.get("value", math.nan))
        threshold = float(row.get("threshold", math.nan))
        score = value / threshold if math.isfinite(value) and math.isfinite(threshold) and threshold != 0.0 else math.nan
        labels.append(str(row.get("id") or ""))
        scores.append(float(score))
        colors.append("#2f9e44" if bool(row.get("pass")) else "#dc2626")

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11.8, 4.8), dpi=180)
    ax.barh(y, scores, color=colors)
    ax.axvline(1.0, linestyle="--", color="#6b7280", linewidth=1.2)
    ax.set_yticks(y, labels)
    ax.set_xlabel("normalized error (<=1 is pass)")
    ax.set_title("Action-principle EL derivation audit (gauge covariance / invariance)")
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate action-principle EL derivation audit pack.")
    parser.add_argument(
        "--out-json",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "action_principle_el_derivation_audit.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "action_principle_el_derivation_audit.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-png",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "action_principle_el_derivation_audit.png"),
        help="Output PNG path.",
    )
    args = parser.parse_args(argv)

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_png = Path(args.out_png)
    # 条件分岐: `not out_json.is_absolute()` を満たす経路を評価する。
    if not out_json.is_absolute():
        out_json = (ROOT / out_json).resolve()

    # 条件分岐: `not out_csv.is_absolute()` を満たす経路を評価する。

    if not out_csv.is_absolute():
        out_csv = (ROOT / out_csv).resolve()

    # 条件分岐: `not out_png.is_absolute()` を満たす経路を評価する。

    if not out_png.is_absolute():
        out_png = (ROOT / out_png).resolve()

    payload = build_pack()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    criteria = payload.get("numerical_audit", {}).get("criteria") if isinstance(payload.get("numerical_audit"), dict) else []
    _write_csv(out_csv, criteria if isinstance(criteria, list) else [])
    _plot(out_png, payload)

    print(f"[ok] wrote: {_rel(out_json)}")
    print(f"[ok] wrote: {_rel(out_csv)}")
    print(f"[ok] wrote: {_rel(out_png)}")

    try:
        worklog.append_event(
            {
                "event_type": "quantum_action_principle_el_derivation_audit",
                "phase": "8.7.9",
                "outputs": {
                    "action_principle_el_derivation_audit_json": _rel(out_json),
                    "action_principle_el_derivation_audit_csv": _rel(out_csv),
                    "action_principle_el_derivation_audit_png": _rel(out_png),
                },
                "decision": payload.get("decision"),
            }
        )
    except Exception:
        pass

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
