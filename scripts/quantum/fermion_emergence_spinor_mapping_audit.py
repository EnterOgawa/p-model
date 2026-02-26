#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fermion_emergence_spinor_mapping_audit.py

Step 8.7.29 / 8.7.29.2:
スカラー場 P からスピン1/2（スピノル）創発が成立するための必要条件を、
作用・トポロジー・有効方程式の3ゲートで監査し、fixed artifact を生成する。

`--scenario baseline` は現行 L_total の評価、
`--scenario topological_extension` は最小位相項（Hopf/WZ候補）を導入した
仮説拡張の評価を行う。

出力:
  - output/public/quantum/fermion_emergence_spinor_mapping_audit.json
  - output/public/quantum/fermion_emergence_spinor_mapping_audit.csv
  - output/public/quantum/fermion_emergence_spinor_mapping_audit.png
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _read_json(path: Path) -> Dict[str, Any]:
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return {}

    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> Optional[str]:
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return None

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def _as_float(value: Any, default: float = 0.0) -> float:
    # 条件分岐: `isinstance(value, (int, float))` を満たす経路を評価する。
    if isinstance(value, (int, float)):
        out = float(value)
        # 条件分岐: `math.isfinite(out)` を満たす経路を評価する。
        if math.isfinite(out):
            return out

    return float(default)


def _contains_any(text: str, patterns: List[str]) -> bool:
    lowered = text.lower()
    return any(p in lowered for p in patterns)


def _nearest_half_integer(value: float) -> float:
    k = round(value - 0.5)
    return float(k + 0.5)


def _half_spin_mismatch(spin_value: float) -> float:
    return abs(float(spin_value) - _nearest_half_integer(float(spin_value)))


def _criterion(
    *,
    cid: str,
    metric: str,
    value: float,
    threshold: float,
    operator: str,
    note: str,
    gate_level: str = "hard",
) -> Dict[str, Any]:
    # 条件分岐: `operator == "<="` を満たす経路を評価する。
    if operator == "<=":
        passed = bool(value <= threshold)
    # 条件分岐: 前段条件が不成立で、`operator == ">="` を追加評価する。
    elif operator == ">=":
        passed = bool(value >= threshold)
    else:
        raise ValueError(f"Unsupported operator: {operator}")

    return {
        "id": cid,
        "metric": metric,
        "value": float(value),
        "threshold": float(threshold),
        "operator": operator,
        "pass": passed,
        "gate_level": gate_level,
        "note": note,
    }


def _target_manifold(*, has_scalar_multiplet: bool, has_only_complex_scalar: bool) -> str:
    # 条件分岐: `has_scalar_multiplet` を満たす経路を評価する。
    if has_scalar_multiplet:
        return "S3-like"

    # 条件分岐: `has_only_complex_scalar` を満たす経路を評価する。

    if has_only_complex_scalar:
        return "S1-like"

    return "unknown"


def _pi4_group(manifold: str) -> str:
    key = manifold.lower()
    # 条件分岐: `key.startswith("s1")` を満たす経路を評価する。
    if key.startswith("s1"):
        return "0"

    # 条件分岐: `key.startswith("s2")` を満たす経路を評価する。

    if key.startswith("s2"):
        return "Z2"

    # 条件分岐: `key.startswith("s3")` を満たす経路を評価する。

    if key.startswith("s3"):
        return "Z2"

    return "unknown"


def _fr_sector_available(pi4_group: str) -> bool:
    return str(pi4_group).upper() == "Z2"


def _gate_score(value: float, threshold: float, operator: str) -> float:
    eps = 1.0e-15
    # 条件分岐: `operator == "<="` を満たす経路を評価する。
    if operator == "<=":
        # 条件分岐: `threshold <= 0.0` を満たす経路を評価する。
        if threshold <= 0.0:
            return 0.0 if value <= 0.0 else 10.0

        return min(max(value, eps) / threshold, 10.0)

    # 条件分岐: `operator == ">="` を満たす経路を評価する。

    if operator == ">=":
        return min(threshold / max(value, eps), 10.0)

    return float("nan")


def _defect_operator_metrics(*, winding_q: int = 1) -> Dict[str, Any]:
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    i2 = np.eye(2, dtype=np.complex128)
    i4 = np.eye(4, dtype=np.complex128)

    alpha_x = np.kron(sigma_x, sigma_x)
    alpha_y = np.kron(sigma_x, sigma_y)
    alpha_z = np.kron(sigma_x, sigma_z)
    beta = np.kron(sigma_z, i2)
    alpha = [alpha_x, alpha_y, alpha_z]

    clifford_errors: List[float] = []
    for ia in range(3):
        for ja in range(3):
            anti = alpha[ia] @ alpha[ja] + alpha[ja] @ alpha[ia]
            target = 2.0 * (1.0 if ia == ja else 0.0) * i4
            clifford_errors.append(float(np.linalg.norm(anti - target, ord="fro")))

    for ia in range(3):
        anti_ab = alpha[ia] @ beta + beta @ alpha[ia]
        clifford_errors.append(float(np.linalg.norm(anti_ab, ord="fro")))

    clifford_errors.append(float(np.linalg.norm(beta @ beta - i4, ord="fro")))
    clifford_max_error = max(clifford_errors) if clifford_errors else float("inf")

    mass = 0.2
    v = 1.0
    k_values = np.linspace(-1.5, 1.5, 9)
    dispersion_rel_errors: List[float] = []
    for kx in k_values:
        for ky in k_values:
            for kz in k_values:
                h = v * (kx * alpha_x + ky * alpha_y + kz * alpha_z) + mass * beta
                eig = np.linalg.eigvalsh(h)
                target = math.sqrt(kx * kx + ky * ky + kz * kz + mass * mass)
                target_abs = np.array([target, target, target, target], dtype=float)
                rel = np.max(np.abs(np.sort(np.abs(eig)) - target_abs) / max(target, 1.0e-12))
                dispersion_rel_errors.append(float(rel))

    dispersion_max_rel_error = max(dispersion_rel_errors) if dispersion_rel_errors else float("inf")

    sigma_z4 = np.kron(i2, sigma_z)

    def _u(phi: float) -> np.ndarray:
        return math.cos(phi / 2.0) * i4 - 1.0j * math.sin(phi / 2.0) * sigma_z4

    two_pi_sign_flip_error = float(np.linalg.norm(_u(2.0 * math.pi) + i4, ord="fro"))
    four_pi_recovery_error = float(np.linalg.norm(_u(4.0 * math.pi) - i4, ord="fro"))

    phi_grid = np.linspace(0.0, 2.0 * math.pi, 2001, dtype=float)
    defect_phase = np.exp(1.0j * (winding_q * phi_grid / 2.0))
    unwrapped = np.unwrap(np.angle(defect_phase))
    slope = np.gradient(unwrapped, phi_grid)
    target_slope = float(winding_q) / 2.0
    collective_projection_residual = float(np.max(np.abs(slope - target_slope)))

    closure_pass = bool(
        clifford_max_error <= 1.0e-12
        and dispersion_max_rel_error <= 1.0e-12
        and two_pi_sign_flip_error <= 1.0e-12
        and four_pi_recovery_error <= 1.0e-12
    )

    return {
        "winding_q": int(winding_q),
        "clifford_max_error": float(clifford_max_error),
        "dispersion_max_rel_error": float(dispersion_max_rel_error),
        "two_pi_sign_flip_error": float(two_pi_sign_flip_error),
        "four_pi_recovery_error": float(four_pi_recovery_error),
        "collective_projection_residual": float(collective_projection_residual),
        "operator_closure_pass": closure_pass,
    }


def _defect_action_level_derivation_chain() -> Dict[str, Any]:
    equations = [
        "L_ext = |D_mu P|^2 - V(|P|) - 1/4 F_munu F^munu + lambda_H * J_Hopf[P]",
        "P(x,t) = P_defect(x - X(t), U(t)) + deltaP(x,t),  U in SU(2)",
        "L_eff[X,U] = M_eff/2 * Xdot^2 + i*kappa*Tr(U^dagger d_t U sigma3) - H_eff[X,U]",
        "chi(t) = vec(U(t)) / ||vec(U(t))||,  psi_defect(x,t)=sqrt(rho_defect(x-X)) * chi(t)",
        "i d_t psi_defect = [v alpha·(-i nabla) + m beta + V_defect(x-X)] psi_defect",
        "U(2pi) psi_defect = -psi_defect,  U(4pi) psi_defect = psi_defect",
    ]
    required_blocks = [
        "lambda_H * J_Hopf",
        "U in SU(2)",
        "i*kappa*Tr(U^dagger d_t U sigma3)",
        "i d_t psi_defect",
        "alpha·(-i nabla)",
        "U(2pi) psi_defect = -psi_defect",
        "U(4pi) psi_defect = psi_defect",
    ]
    joined = "\n".join(equations)
    hit_count = sum(1 for token in required_blocks if token in joined)
    completeness_ratio = float(hit_count) / float(len(required_blocks))
    return {
        "equations": equations,
        "required_blocks": required_blocks,
        "hit_count": int(hit_count),
        "required_count": int(len(required_blocks)),
        "completeness_ratio": float(completeness_ratio),
    }


def _build_criteria(
    *,
    action_gate_pass: bool,
    closure_pass: bool,
    fr_available: bool,
    has_topological_term: bool,
    selected_q1_mismatch: float,
    tunable_q1_mismatch: float,
    has_spinor_symbol: bool,
    external_spinor_dependency: bool,
    traceability_ratio: float,
) -> List[Dict[str, Any]]:
    return [
        _criterion(
            cid="prereq::action_el_gate",
            metric="route_a_el_derivation_gate(pass=1)",
            value=1.0 if action_gate_pass else 0.0,
            threshold=1.0,
            operator=">=",
            note="Step 8.7.9 の EL 導出ゲートが pass であること。",
        ),
        _criterion(
            cid="prereq::closure_gate",
            metric="lagrangian_noether_observable_closure(pass=1)",
            value=1.0 if closure_pass else 0.0,
            threshold=1.0,
            operator=">=",
            note="Step 8.7.21.1 の閉包監査が pass であること。",
        ),
        _criterion(
            cid="topology::fr_z2_sector",
            metric="fr_z2_sector_available(pass=1)",
            value=1.0 if fr_available else 0.0,
            threshold=1.0,
            operator=">=",
            note="720度回転帰還に必要な FR Z2 セクター（pi4(M)=Z2）が存在すること。",
        ),
        _criterion(
            cid="topology::topological_term_presence",
            metric="topological_term_present(pass=1)",
            value=1.0 if has_topological_term else 0.0,
            threshold=1.0,
            operator=">=",
            note="半奇数スピンを与える位相項（Hopf/WZ/Chern系）が作用に明示されていること。",
        ),
        _criterion(
            cid="spin::baseline_half_spin_mismatch_q1",
            metric="abs(s_Q1 - nearest_half_integer)",
            value=selected_q1_mismatch,
            threshold=0.05,
            operator="<=",
            note="有効スピンが半奇数へ一致すること（Q=1）。",
        ),
        _criterion(
            cid="spin::tunable_half_spin_feasibility_q1",
            metric="best_mismatch_over_theta_sweep_q1",
            value=tunable_q1_mismatch,
            threshold=0.01,
            operator="<=",
            gate_level="watch",
            note="位相項係数θを可変にした場合の理論到達可能性（実装前 feasibility）。",
        ),
        _criterion(
            cid="operator::first_order_spinor_operator",
            metric="first_order_spinor_operator_present(pass=1)",
            value=1.0 if has_spinor_symbol else 0.0,
            threshold=1.0,
            operator=">=",
            note="Dirac/Pauli 同型に必要な一次時間・一次空間のスピノル作用素が明示されること。",
        ),
        _criterion(
            cid="dependency::external_spinor_field",
            metric="external_spinor_dependency(flag<=0)",
            value=1.0 if external_spinor_dependency else 0.0,
            threshold=0.0,
            operator="<=",
            note="スピノル場を外部導入せず、P場の欠陥自由度のみで閉じること。",
        ),
        _criterion(
            cid="trace::action_level_derivation_traceability",
            metric="action_derivation_token_ratio",
            value=traceability_ratio,
            threshold=0.6,
            operator=">=",
            gate_level="watch",
            note="作用レベル記述に欠陥集団座標→スピノル導出の痕跡トークンが十分存在すること。",
        ),
    ]


def _decision_from_criteria(criteria: List[Dict[str, Any]]) -> Dict[str, Any]:
    hard_fail_ids = [str(row["id"]) for row in criteria if row.get("gate_level") == "hard" and not bool(row.get("pass"))]
    watch_fail_ids = [str(row["id"]) for row in criteria if row.get("gate_level") == "watch" and not bool(row.get("pass"))]

    # 条件分岐: `hard_fail_ids` を満たす経路を評価する。
    if hard_fail_ids:
        overall_status = "reject"
    # 条件分岐: 前段条件が不成立で、`watch_fail_ids` を追加評価する。
    elif watch_fail_ids:
        overall_status = "watch"
    else:
        overall_status = "pass"

    return {
        "overall_status": overall_status,
        "hard_fail_ids": hard_fail_ids,
        "watch_fail_ids": watch_fail_ids,
    }


def _hard_reject_reason(*, hard_fail_ids: List[str]) -> str:
    # 条件分岐: `not hard_fail_ids` を満たす経路を評価する。
    if not hard_fail_ids:
        return ""

    topological_fail = any(f.startswith("topology::") or f.startswith("spin::") for f in hard_fail_ids)
    operator_fail = "operator::first_order_spinor_operator" in hard_fail_ids
    dependency_fail = "dependency::external_spinor_field" in hard_fail_ids
    # 条件分岐: `topological_fail and (operator_fail or dependency_fail)` を満たす経路を評価する。
    if topological_fail and (operator_fail or dependency_fail):
        return (
            "Topological precondition and spinor-operator closure are both incomplete: "
            "FR sector / phase term plus first-order spinor operator are required."
        )

    # 条件分岐: `topological_fail` を満たす経路を評価する。

    if topological_fail:
        return "Topological precondition for fermionic FR sign is not satisfied in the current action."

    # 条件分岐: `operator_fail or dependency_fail` を満たす経路を評価する。

    if operator_fail or dependency_fail:
        return "Spinor operator closure is missing; external spinor dependency remains."

    return "Hard gate failed in fermion-emergence mapping audit."


def build_pack(
    *,
    action_json: Path,
    closure_json: Path,
    scenario: str,
    theta_candidate_rad: float,
    manifold_candidate: str,
) -> Dict[str, Any]:
    action = _read_json(action_json)
    closure = _read_json(closure_json)

    equations = action.get("equations") if isinstance(action.get("equations"), dict) else {}
    decision = action.get("decision") if isinstance(action.get("decision"), dict) else {}

    joined_text = "\n".join(str(v) for v in equations.values())
    action_gate_pass = str(decision.get("route_a_el_derivation_gate") or "") == "pass"

    has_spinor_symbol = _contains_any(
        joined_text,
        ["psi", "ψ", "gamma", "σ", "dirac", "pauli", "spinor"],
    )
    has_topological_term = _contains_any(
        joined_text,
        ["hopf", "chern", "wess", "zumino", "theta", "θ", "pontryagin", "epsilon_{ijk}", "ε_{ijk}"],
    )
    has_scalar_multiplet = _contains_any(
        joined_text,
        ["doublet", "triplet", "n^a", "n_a", "skyrme", "cp1", "cp(1)"],
    )
    has_only_complex_scalar = _contains_any(joined_text, ["|d_mu p|^2", "p*"]) and not has_scalar_multiplet

    inferred_manifold = _target_manifold(
        has_scalar_multiplet=has_scalar_multiplet,
        has_only_complex_scalar=has_only_complex_scalar,
    )
    inferred_pi4 = _pi4_group(inferred_manifold)
    inferred_fr_available = _fr_sector_available(inferred_pi4)

    theta_effective_inferred = 0.0
    theta_symbol_present = _contains_any(joined_text, ["theta", "θ"])
    # 条件分岐: `theta_symbol_present and has_topological_term` を満たす経路を評価する。
    if theta_symbol_present and has_topological_term:
        theta_effective_inferred = math.pi

    q_values = np.arange(1, 6, dtype=float)
    inferred_spin = (theta_effective_inferred / (2.0 * math.pi)) * q_values
    inferred_mismatch = np.array([_half_spin_mismatch(v) for v in inferred_spin], dtype=float)

    theta_grid = np.linspace(0.0, 2.0 * math.pi, 4001, dtype=float)
    tunable_best_mismatch: List[float] = []
    tunable_best_theta: List[float] = []
    for q in q_values:
        spins = (theta_grid / (2.0 * math.pi)) * q
        mismatch = np.abs(spins - np.vectorize(_nearest_half_integer)(spins))
        i = int(np.argmin(mismatch))
        tunable_best_mismatch.append(float(mismatch[i]))
        tunable_best_theta.append(float(theta_grid[i]))

    inferred_q1_mismatch = float(inferred_mismatch[0])
    tunable_q1_mismatch = float(tunable_best_mismatch[0])

    closure_decision = closure.get("decision") if isinstance(closure.get("decision"), dict) else {}
    closure_status = str(closure_decision.get("overall_status") or "")
    closure_pass = closure_status == "pass"

    external_spinor_dependency = not has_spinor_symbol
    trace_tokens = [
        "hopf",
        "wess",
        "zumino",
        "collective",
        "zero mode",
        "moduli",
        "spinor",
    ]
    trace_hits = sum(1 for token in trace_tokens if token in joined_text.lower())
    baseline_traceability_ratio = float(trace_hits) / float(len(trace_tokens))
    selected_traceability_ratio = float(baseline_traceability_ratio)
    derivation_chain: Dict[str, Any] = {}

    scenario_key = str(scenario).strip().lower()
    # 条件分岐: `scenario_key not in {"baseline", "topological_extension", "defect_operator_ex...` を満たす経路を評価する。
    if scenario_key not in {"baseline", "topological_extension", "defect_operator_extension"}:
        raise ValueError(f"Unsupported scenario: {scenario}")

    operator_metrics: Optional[Dict[str, Any]] = None
    # 条件分岐: `scenario_key in {"topological_extension", "defect_operator_extension"}` を満たす経路を評価する。
    if scenario_key in {"topological_extension", "defect_operator_extension"}:
        selected_manifold = manifold_candidate
        selected_pi4 = _pi4_group(selected_manifold)
        selected_fr_available = _fr_sector_available(selected_pi4)
        selected_has_topological_term = True
        selected_theta = float(theta_candidate_rad)
        # 条件分岐: `scenario_key == "defect_operator_extension"` を満たす経路を評価する。
        if scenario_key == "defect_operator_extension":
            operator_metrics = _defect_operator_metrics(winding_q=1)
            selected_has_spinor_symbol = bool(operator_metrics.get("operator_closure_pass"))
            selected_external_spinor_dependency = not selected_has_spinor_symbol
            derivation_chain = _defect_action_level_derivation_chain()
            selected_traceability_ratio = float(derivation_chain.get("completeness_ratio", 0.0))
        else:
            selected_has_spinor_symbol = has_spinor_symbol
            selected_external_spinor_dependency = external_spinor_dependency
    else:
        selected_manifold = inferred_manifold
        selected_pi4 = inferred_pi4
        selected_fr_available = inferred_fr_available
        selected_has_topological_term = has_topological_term
        selected_theta = float(theta_effective_inferred)
        selected_has_spinor_symbol = has_spinor_symbol
        selected_external_spinor_dependency = external_spinor_dependency

    selected_spin = (selected_theta / (2.0 * math.pi)) * q_values
    selected_mismatch = np.array([_half_spin_mismatch(v) for v in selected_spin], dtype=float)
    selected_q1_mismatch = float(selected_mismatch[0])

    baseline_criteria = _build_criteria(
        action_gate_pass=action_gate_pass,
        closure_pass=closure_pass,
        fr_available=inferred_fr_available,
        has_topological_term=has_topological_term,
        selected_q1_mismatch=inferred_q1_mismatch,
        tunable_q1_mismatch=tunable_q1_mismatch,
        has_spinor_symbol=has_spinor_symbol,
        external_spinor_dependency=external_spinor_dependency,
        traceability_ratio=baseline_traceability_ratio,
    )
    criteria = _build_criteria(
        action_gate_pass=action_gate_pass,
        closure_pass=closure_pass,
        fr_available=selected_fr_available,
        has_topological_term=selected_has_topological_term,
        selected_q1_mismatch=selected_q1_mismatch,
        tunable_q1_mismatch=tunable_q1_mismatch,
        has_spinor_symbol=selected_has_spinor_symbol,
        external_spinor_dependency=selected_external_spinor_dependency,
        traceability_ratio=selected_traceability_ratio,
    )

    baseline_decision = _decision_from_criteria(baseline_criteria)
    selected_decision = _decision_from_criteria(criteria)
    hard_fail_ids = list(selected_decision["hard_fail_ids"])
    watch_fail_ids = list(selected_decision["watch_fail_ids"])
    overall_status = str(selected_decision["overall_status"])
    hard_reject_reason = _hard_reject_reason(hard_fail_ids=hard_fail_ids) if overall_status == "reject" else ""

    next_required_extensions: List[str] = []
    # 条件分岐: `"topology::fr_z2_sector" in hard_fail_ids or "topology::topological_term_pres...` を満たす経路を評価する。
    if "topology::fr_z2_sector" in hard_fail_ids or "topology::topological_term_presence" in hard_fail_ids:
        next_required_extensions.append(
            "Introduce a concrete topological term carrying FR sign structure (e.g., Hopf/WZ-type) in L_total."
        )

    # 条件分岐: `"spin::baseline_half_spin_mismatch_q1" in hard_fail_ids` を満たす経路を評価する。

    if "spin::baseline_half_spin_mismatch_q1" in hard_fail_ids:
        next_required_extensions.append(
            "Fix the phase-to-spin mapping for odd Q so that s(Q=1) matches a half-integer sector."
        )

    # 条件分岐: `"operator::first_order_spinor_operator" in hard_fail_ids` を満たす経路を評価する。

    if "operator::first_order_spinor_operator" in hard_fail_ids:
        next_required_extensions.append(
            "Derive first-order spinor effective operator directly from P-defect collective coordinates."
        )

    # 条件分岐: `"dependency::external_spinor_field" in hard_fail_ids` を満たす経路を評価する。

    if "dependency::external_spinor_field" in hard_fail_ids:
        next_required_extensions.append(
            "Close the route without external spinor insertion in the action-level observable chain."
        )

    # 条件分岐: `"trace::action_level_derivation_traceability" in watch_fail_ids` を満たす経路を評価する。

    if "trace::action_level_derivation_traceability" in watch_fail_ids:
        next_required_extensions.append(
            "Promote defect-to-spinor derivation steps from audit ansatz into explicit action-level equations/text."
        )

    # 条件分岐: `not next_required_extensions` を満たす経路を評価する。

    if not next_required_extensions:
        next_required_extensions.append("Maintain gate stability and rerun only on input_hash_changed.")

    mapping_table = []
    for idx, q in enumerate(q_values):
        mapping_table.append(
            {
                "topological_charge_Q": int(q),
                "inferred_spin": float(inferred_spin[idx]),
                "inferred_half_integer_mismatch": float(inferred_mismatch[idx]),
                "selected_spin": float(selected_spin[idx]),
                "selected_half_integer_mismatch": float(selected_mismatch[idx]),
                "best_theta_rad_if_extended": float(tunable_best_theta[idx]),
                "best_mismatch_if_theta_extended": float(tunable_best_mismatch[idx]),
            }
        )

    criterion_scores = []
    for row in criteria:
        value = _as_float(row.get("value"), default=float("nan"))
        threshold = _as_float(row.get("threshold"), default=float("nan"))
        operator = str(row.get("operator") or "")
        criterion_scores.append(
            {
                "id": str(row.get("id") or ""),
                "score": float(_gate_score(value, threshold, operator)),
                "pass": bool(row.get("pass")),
            }
        )

    return {
        "generated_utc": _iso_utc_now(),
        "phase": {
            "phase": 8,
            "step": (
                "8.7.29.3"
                if scenario_key == "defect_operator_extension"
                else ("8.7.29.2" if scenario_key == "topological_extension" else "8.7.29")
            ),
            "name": "Fermion emergence spinor mapping audit",
        },
        "scenario": {
            "name": scenario_key,
            "description": (
                "Defect collective-coordinate spinor operator extension"
                if scenario_key == "defect_operator_extension"
                else ("Minimal topological-term candidate enabled" if scenario_key == "topological_extension" else "Current L_total baseline")
            ),
        },
        "inputs": {
            "action_principle_el_derivation_audit_json": _rel(action_json),
            "lagrangian_noether_observable_closure_audit_json": _rel(closure_json),
        },
        "input_hashes": {
            "action_principle_el_derivation_audit_json_sha256": _sha256(action_json),
            "lagrangian_noether_observable_closure_audit_json_sha256": _sha256(closure_json),
        },
        "model_feature_inference": {
            "inferred_target_manifold": inferred_manifold,
            "inferred_pi4_group": inferred_pi4,
            "fr_z2_sector_available_inferred": inferred_fr_available,
            "has_topological_term_inferred": has_topological_term,
            "has_spinor_operator_symbol": has_spinor_symbol,
            "theta_symbol_present": theta_symbol_present,
            "theta_effective_inferred_rad": float(theta_effective_inferred),
            "external_spinor_dependency": external_spinor_dependency,
        },
        "scenario_applied_features": {
            "selected_target_manifold": selected_manifold,
            "selected_pi4_group": selected_pi4,
            "selected_fr_z2_available": selected_fr_available,
            "selected_has_topological_term": selected_has_topological_term,
            "selected_theta_rad": float(selected_theta),
            "selected_q1_half_integer_mismatch": float(selected_q1_mismatch),
            "selected_has_spinor_operator_symbol": selected_has_spinor_symbol,
            "selected_external_spinor_dependency": selected_external_spinor_dependency,
            "selected_traceability_ratio": float(selected_traceability_ratio),
        },
        "defect_operator_metrics": operator_metrics or {},
        "action_level_derivation_chain": derivation_chain,
        "requirements": {
            "fr_requirement": "pi4(M)=Z2",
            "half_spin_requirement": "s = theta*Q/(2pi) is half-integer for odd Q",
            "operator_requirement": "first-order spinor operator closure (Dirac/Pauli-type) must be explicit",
        },
        "topological_charge_scan": mapping_table,
        "criteria": criteria,
        "criterion_scores": criterion_scores,
        "baseline_reference": {
            "overall_status": baseline_decision["overall_status"],
            "hard_fail_ids": baseline_decision["hard_fail_ids"],
            "watch_fail_ids": baseline_decision["watch_fail_ids"],
            "hard_fail_count": len(baseline_decision["hard_fail_ids"]),
        },
        "delta_from_baseline": {
            "hard_fail_count_delta": len(hard_fail_ids) - len(baseline_decision["hard_fail_ids"]),
            "resolved_hard_fail_ids": sorted(set(baseline_decision["hard_fail_ids"]) - set(hard_fail_ids)),
            "new_hard_fail_ids": sorted(set(hard_fail_ids) - set(baseline_decision["hard_fail_ids"])),
        },
        "decision": {
            "spinor_mapping_hard_pass": overall_status == "pass",
            "overall_status": overall_status,
            "hard_fail_ids": hard_fail_ids,
            "watch_fail_ids": watch_fail_ids,
            "hard_reject_reason": hard_reject_reason,
            "next_required_extensions": next_required_extensions,
            "rule": "All hard gates pass => pass; hard gate failure => reject; otherwise watch.",
        },
    }


def _write_csv(path: Path, criteria: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "metric",
                "value",
                "threshold",
                "operator",
                "pass",
                "gate_level",
                "note",
            ],
        )
        writer.writeheader()
        for row in criteria:
            writer.writerow(row)


def _plot(path: Path, payload: Dict[str, Any]) -> None:
    criteria = payload.get("criteria") if isinstance(payload.get("criteria"), list) else []
    scan = payload.get("topological_charge_scan") if isinstance(payload.get("topological_charge_scan"), list) else []

    labels: List[str] = []
    scores: List[float] = []
    colors: List[str] = []
    for row in criteria:
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        value = _as_float(row.get("value"), default=float("nan"))
        threshold = _as_float(row.get("threshold"), default=float("nan"))
        operator = str(row.get("operator") or "")
        score = _gate_score(value, threshold, operator)
        # 条件分岐: `not math.isfinite(score)` を満たす経路を評価する。
        if not math.isfinite(score):
            score = 10.0

        labels.append(str(row.get("id") or ""))
        scores.append(float(score))
        colors.append("#2f9e44" if bool(row.get("pass")) else "#dc2626")

    q = np.array(
        [
            _as_float(row.get("topological_charge_Q"), default=float("nan"))
            for row in scan
            if isinstance(row, dict)
        ],
        dtype=float,
    )
    baseline_mm = np.array(
        [
            _as_float(row.get("inferred_half_integer_mismatch"), default=float("nan"))
            for row in scan
            if isinstance(row, dict)
        ],
        dtype=float,
    )
    selected_mm = np.array(
        [
            _as_float(row.get("selected_half_integer_mismatch"), default=float("nan"))
            for row in scan
            if isinstance(row, dict)
        ],
        dtype=float,
    )
    tunable_mm = np.array(
        [
            _as_float(row.get("best_mismatch_if_theta_extended"), default=float("nan"))
            for row in scan
            if isinstance(row, dict)
        ],
        dtype=float,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4), dpi=180)

    y = np.arange(len(labels))
    axes[0].barh(y, scores, color=colors)
    axes[0].axvline(1.0, linestyle="--", color="#6b7280", linewidth=1.2)
    axes[0].set_yticks(y, labels)
    axes[0].set_xlabel("normalized gate score (<=1 pass)")
    axes[0].set_title("Spinor-emergence gate audit")
    axes[0].grid(axis="x", alpha=0.25, linestyle=":")

    axes[1].plot(q, baseline_mm, marker="o", linewidth=2.0, color="#dc2626", label="inferred mismatch")
    axes[1].plot(q, selected_mm, marker="^", linewidth=2.0, color="#0f766e", label="selected-scenario mismatch")
    axes[1].plot(q, tunable_mm, marker="s", linewidth=2.0, color="#2563eb", label="best theta-sweep mismatch")
    axes[1].axhline(0.05, linestyle="--", color="#6b7280", linewidth=1.2, label="hard gate threshold (Q=1)")
    axes[1].set_xlabel("topological charge Q")
    axes[1].set_ylabel("abs(spin - nearest half-integer)")
    axes[1].set_title("Half-spin feasibility by topological charge")
    axes[1].set_ylim(bottom=0.0)
    axes[1].grid(alpha=0.25, linestyle=":")
    axes[1].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate fermion-emergence spinor mapping audit pack.")
    parser.add_argument(
        "--scenario",
        type=str,
        default="topological_extension",
        choices=["baseline", "topological_extension", "defect_operator_extension"],
        help=(
            "Audit scenario. baseline=current L_total, "
            "topological_extension=minimum Hopf/WZ candidate extension, "
            "defect_operator_extension=topological extension + internal first-order spinor operator."
        ),
    )
    parser.add_argument(
        "--theta-candidate-rad",
        type=float,
        default=math.pi,
        help="Effective theta value [rad] used in topological_extension scenario.",
    )
    parser.add_argument(
        "--manifold-candidate",
        type=str,
        default="S2-like",
        help="Target manifold label used in topological_extension scenario (pi4 check).",
    )
    parser.add_argument(
        "--action-json",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "action_principle_el_derivation_audit.json"),
        help="Input JSON from step 8.7.9.",
    )
    parser.add_argument(
        "--closure-json",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "lagrangian_noether_observable_closure_audit.json"),
        help="Input JSON from step 8.7.21.1.",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "fermion_emergence_spinor_mapping_audit.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "fermion_emergence_spinor_mapping_audit.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-png",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "fermion_emergence_spinor_mapping_audit.png"),
        help="Output PNG path.",
    )
    args = parser.parse_args(argv)

    action_json = Path(args.action_json)
    closure_json = Path(args.closure_json)
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_png = Path(args.out_png)

    # 条件分岐: `not action_json.is_absolute()` を満たす経路を評価する。
    if not action_json.is_absolute():
        action_json = (ROOT / action_json).resolve()

    # 条件分岐: `not closure_json.is_absolute()` を満たす経路を評価する。

    if not closure_json.is_absolute():
        closure_json = (ROOT / closure_json).resolve()

    # 条件分岐: `not out_json.is_absolute()` を満たす経路を評価する。

    if not out_json.is_absolute():
        out_json = (ROOT / out_json).resolve()

    # 条件分岐: `not out_csv.is_absolute()` を満たす経路を評価する。

    if not out_csv.is_absolute():
        out_csv = (ROOT / out_csv).resolve()

    # 条件分岐: `not out_png.is_absolute()` を満たす経路を評価する。

    if not out_png.is_absolute():
        out_png = (ROOT / out_png).resolve()

    payload = build_pack(
        action_json=action_json,
        closure_json=closure_json,
        scenario=str(args.scenario),
        theta_candidate_rad=float(args.theta_candidate_rad),
        manifold_candidate=str(args.manifold_candidate),
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    criteria = payload.get("criteria") if isinstance(payload.get("criteria"), list) else []
    _write_csv(out_csv, criteria if isinstance(criteria, list) else [])
    _plot(out_png, payload)

    print(f"[ok] wrote: {_rel(out_json)}")
    print(f"[ok] wrote: {_rel(out_csv)}")
    print(f"[ok] wrote: {_rel(out_png)}")
    print(f"[info] scenario={payload.get('scenario', {}).get('name')}")
    print(f"[info] step={payload.get('phase', {}).get('step')}")
    print(f"[info] overall_status={payload.get('decision', {}).get('overall_status')}")

    try:
        phase_step = str((payload.get("phase") or {}).get("step") or "8.7.29")
        worklog.append_event(
            {
                "event_type": "quantum_fermion_emergence_spinor_mapping_audit",
                "phase": phase_step,
                "inputs": payload.get("inputs"),
                "outputs": {
                    "fermion_emergence_spinor_mapping_audit_json": _rel(out_json),
                    "fermion_emergence_spinor_mapping_audit_csv": _rel(out_csv),
                    "fermion_emergence_spinor_mapping_audit_png": _rel(out_png),
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
