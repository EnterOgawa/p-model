#!/usr/bin/env python3
"""Generate 8.7.56.1599-.1602 exact v^2 subtraction treatment audit artifacts.

This branch follows the retained Case-I classification honestly.
The next same-level computation is not another slogan-level classification,
but the exact treatment of the asymptotic `v^2` subtraction suggested by the
quadratic transverse-projection note.

What this branch must compute:

1. Reconstruct the retained exact vector/Q-ball branch at the restored
   Phase-1-equivalent point.
2. Evaluate the subtracted isotropic profile
   `V_eff^sub(r) = lambda(-f_0(r)^2 + f_L(r)^2)`.
3. Compute the normalized spherical form factor `F_sub(q_theory)` and
   `alpha_sub = F_sub(q_theory)^2 / (4*pi)`.
4. Compare the subtraction result against the retained scalar residual and
   classify the outcome as improve / neutral / worsen.

The overall `lambda` factor cancels in the normalized form factor, so the
actual computation uses the signed radial profile `(-f_0^2 + f_L^2)` directly.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
WORK_HISTORY_RECENT = ROOT / "doc" / "WORK_HISTORY_RECENT.md"
CURRENT_PROBLEM = ROOT / "doc" / "quantum" / "34_trial2_numeric_alpha_current_problem.md"
CURRENT_STATUS = ROOT / "doc" / "quantum" / "36_trial2_numeric_alpha_current_status.md"
UNIFIED_ROADMAP = ROOT / "doc" / "quantum" / "39_trial2_vector_qball_unified_closure_roadmap.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

DIRECTIVE_NOTE = Path(
    r"C:\Users\ogawa\Downloads\trial2_vector_qball_case_i_v2_subtraction_20260328.md"
)
CLASS_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1595_1598_tp_alpha_case_class_declaration_gate_metrics.json"
)
TP_AUDIT_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1591_1594_tp_alpha_audit_declaration_gate_metrics.json"
)
ANCHOR_EVAL = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1483_1486_ell0_anchor_continuation_numeric_evaluation_metrics.json"
)
PHASE1_EVAL = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_phase1_exact_coupled_l0_solver_numeric_evaluation_metrics.json"
)
EXACT_REINJECTION_BRANCH = ROOT / "scripts" / "quantum" / "t2a_1479.py"

STEP_TAG = "8.7.56.1599-1602"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor v^2 subtraction exact treatment audit"
)
STEM = build_compact_artifact_stem(STEP_TAG, "v2_sub_exact_treat", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_case_i_leading_foundation_candidate_v2_subtraction_exact_treatment_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_v2_subtraction_exact_treatment_signed_kernel_worsen_disposition_sync_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_quadratic_transverse_projection_disposition_sync"
)
NEXT_ROUTE = "8.7.56.1603"
TARGET_ALPHA = 1.0 / 137.035999084
NEUTRAL_EPS = 1.0e-9


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を確認する。

def require(path: Path) -> None:
    """Abort when one required input is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8 テキストを読み込む。

def read_text(path: Path) -> str:
    """Read one UTF-8 text file."""
    return path.read_text(encoding="utf-8")


# 関数: UTF-8 JSON を読み込む。

def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: 表示用の相対パスを返す。

def display_path(path: Path) -> str:
    """Return one repo-relative path when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# 関数: 部分文字列に一致する最初の行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line matching one substring."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 標準 metrics row を構成する。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# 関数: 標準 payload を構成する。

def payload(
    step: str,
    name: str,
    inputs: dict,
    rows: list[dict],
    summary: dict,
    decision: dict,
    evidence: dict,
) -> dict:
    """Build one standard payload."""
    return {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": step, "name": name},
        "inputs": inputs,
        "rows": rows,
        "summary": summary,
        "decision": decision,
        "evidence": evidence,
    }


# 関数: JSON/CSV 成果物を書き出す。

def write_artifact(kind: str, data: dict) -> dict[str, str]:
    """Write one JSON payload and one rows CSV."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    paths = build_metrics_paths(PUBLIC_OUT, STEM, kind)
    paths["json"].write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_id", "status", "metric", "value", "note"],
        )
        writer.writeheader()
        writer.writerows(data["rows"])

    return {"json": display_path(paths["json"]), "csv": display_path(paths["csv"])}


# 関数: 真偽値を 0/1 に変換する。

def truth(value: bool) -> float:
    """Convert one boolean into 0/1 float form."""
    return 1.0 if value else 0.0


# 関数: ローカル Python モジュールを動的 import する。

def load_module(path: Path, module_name: str):
    """Dynamically import one local Python module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[fail] unable to import module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 関数: 規格化された球対称 form factor を評価する。

def form_factor(radius: np.ndarray, density: np.ndarray, q_ratio: float) -> tuple[float, float]:
    """Evaluate one normalized spherical form factor."""
    weight = density * (radius**2)
    norm = float(np.trapezoid(weight, radius))
    qx = float(q_ratio) * radius
    sinc = np.ones_like(qx)
    mask = np.abs(qx) > 1.0e-12
    sinc[mask] = np.sin(qx[mask]) / qx[mask]
    numerator = float(np.trapezoid(weight * sinc, radius))
    return float(numerator / norm), float(norm)


# 関数: retained exact branch の radial profile を再構成する。

def solve_exact_profile_with_arrays(pivot, beta: float, amp0: float, amp_l: float) -> dict:
    """Reconstruct the retained exact-branch radial profile with explicit arrays."""
    r0 = 1.0e-4
    y0 = [float(amp0), 0.0, float(amp_l) * r0, float(amp_l)]

    # 関数: current exact coupled pilot ODE を返す。
    def ode(radius: float, y: np.ndarray) -> list[float]:
        """Return the current exact coupled pilot ODE."""
        f0, f0_prime, f_l, f_l_prime = [float(value) for value in y]
        rr = max(float(radius), 1.0e-6)
        rho = math.sqrt(max(f0 * f0 + f_l * f_l, 0.0))
        nonlinear_coeff = 3.0 * rho + rho * rho
        f0_double_prime = (
            -(2.0 / rr) * f0_prime
            - (float(beta * beta) - float(pivot.RADIAL_MASS_SQUARED)) * f0
            - nonlinear_coeff * f0
        )
        f_l_double_prime = (
            -(2.0 / rr) * f_l_prime
            + (2.0 / (rr * rr)) * f_l
            - (float(beta * beta) - float(pivot.LONGITUDINAL_DIRECT_MASS_SQUARED)) * f_l
            - nonlinear_coeff * f_l
        )
        return [f0_prime, f0_double_prime, f_l_prime, f_l_double_prime]

    solution = solve_ivp(
        ode,
        (r0, 25.0),
        y0,
        max_step=0.10,
        rtol=1.0e-7,
        atol=1.0e-9,
    )
    if not solution.success:
        raise SystemExit("[fail] exact branch reconstruction failed during v^2 subtraction audit")

    radius = np.asarray(solution.t, dtype=float)
    f0_values = np.asarray(solution.y[0], dtype=float)
    f_l_values = np.asarray(solution.y[2], dtype=float)
    q_theory = float((1.0 - float(beta) * float(beta)) ** 0.25)
    return {
        "radius": radius,
        "f0": f0_values,
        "fL": f_l_values,
        "q_theory_over_m0": q_theory,
    }


# 関数: residual 比較を improve / neutral / worsen に分類する。

def classify_subtraction_result(
    scalar_residual_rel: float,
    sub_residual_rel: float,
) -> str:
    """Classify the subtraction result against the retained scalar residual."""
    delta = float(sub_residual_rel) - float(scalar_residual_rel)
    if delta < -NEUTRAL_EPS:
        return "improve"

    if abs(delta) <= NEUTRAL_EPS:
        return "neutral"

    return "worsen"


# 関数: `.1599-.1602` を実行する。

def main() -> None:
    """Execute the exact v^2 subtraction treatment audit branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        PART5,
        DIRECTIVE_NOTE,
        CLASS_GATE,
        TP_AUDIT_GATE,
        ANCHOR_EVAL,
        PHASE1_EVAL,
        EXACT_REINJECTION_BRANCH,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    part5_text = read_text(PART5)
    directive_text = read_text(DIRECTIVE_NOTE)

    class_summary = read_json(CLASS_GATE)["summary"]
    tp_summary = read_json(TP_AUDIT_GATE)["summary"]
    anchor_summary = read_json(ANCHOR_EVAL)["summary"]
    phase1_summary = read_json(PHASE1_EVAL)["summary"]

    exact_branch = load_module(EXACT_REINJECTION_BRANCH, "t2a_1479_reuse")
    pivot = exact_branch.load_module(exact_branch.PIVOT_BRANCH, "pivot_branch_reuse")

    phase1_row = anchor_summary["phase1_equivalent_row"]
    beta = float(anchor_summary["beta_1_scalar"])
    amp0 = float(phase1_summary["phase1_best_alpha_candidate"]["amp0"])
    amp_l = float(phase1_row["amp_l"])

    profile = solve_exact_profile_with_arrays(pivot, beta, amp0, amp_l)
    radius = profile["radius"]
    f0_values = profile["f0"]
    f_l_values = profile["fL"]
    q_theory = float(profile["q_theory_over_m0"])

    signed_density = f0_values * f0_values - f_l_values * f_l_values
    subtraction_profile = -f0_values * f0_values + f_l_values * f_l_values
    scalar_profile = f0_values * f0_values

    f_signed, norm_signed = form_factor(radius, signed_density, q_theory)
    f_sub, norm_sub = form_factor(radius, subtraction_profile, q_theory)
    f_scalar, norm_scalar = form_factor(radius, scalar_profile, q_theory)

    alpha_signed = float((f_signed * f_signed) / (4.0 * math.pi))
    alpha_sub = float((f_sub * f_sub) / (4.0 * math.pi))
    alpha_scalar = float((f_scalar * f_scalar) / (4.0 * math.pi))
    residual_signed = float(abs(alpha_signed - TARGET_ALPHA) / TARGET_ALPHA)
    residual_sub = float(abs(alpha_sub - TARGET_ALPHA) / TARGET_ALPHA)
    residual_scalar = float(class_summary["scalar_alpha_residual_rel"])
    residual_gap_vs_scalar = float(residual_sub - residual_scalar)
    residual_ratio_vs_scalar = float(residual_sub / residual_scalar)
    result_case = classify_subtraction_result(residual_scalar, residual_sub)

    profile_identity_max_abs = float(np.max(np.abs(subtraction_profile + signed_density)))
    form_factor_identity_gap = float(abs(f_sub - f_signed))
    alpha_identity_gap = float(abs(alpha_sub - alpha_signed))
    projected_nlo_ceiling = float(class_summary["anisotropic_nlo_ceiling"])
    scalar_residual_rel = residual_scalar

    improve_selected = result_case == "improve"
    neutral_selected = result_case == "neutral"
    worsen_selected = result_case == "worsen"
    subtraction_matches_prior_signed_kernel = bool(profile_identity_max_abs <= 1.0e-12)
    case_i_exact_foundation_supported = bool(improve_selected)
    case_i_candidate_survives_exact_subtraction = bool(not worsen_selected)
    quadratic_tp_disposition_sync_admissible_now = True
    physical_reject_required = False

    inputs = {
        "source_files": {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "ai_context": display_path(AI_CONTEXT),
            "work_history_recent": display_path(WORK_HISTORY_RECENT),
            "current_problem": display_path(CURRENT_PROBLEM),
            "current_status": display_path(CURRENT_STATUS),
            "unified_roadmap": display_path(UNIFIED_ROADMAP),
            "part5": display_path(PART5),
            "directive_note": display_path(DIRECTIVE_NOTE),
            "class_gate": display_path(CLASS_GATE),
            "tp_audit_gate": display_path(TP_AUDIT_GATE),
            "anchor_eval": display_path(ANCHOR_EVAL),
            "phase1_eval": display_path(PHASE1_EVAL),
            "exact_reinjection_branch": display_path(EXACT_REINJECTION_BRANCH),
        },
        "constants": {
            "target_alpha": TARGET_ALPHA,
            "amp0_phase1_equivalent": amp0,
            "amp_l_phase1_equivalent": amp_l,
            "q_theory_over_m0": q_theory,
            "scalar_residual_rel": scalar_residual_rel,
            "projected_nlo_ceiling": projected_nlo_ceiling,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    rows = [
        row(
            "subtraction_profile_matches_signed_density_up_to_global_sign",
            "pass" if subtraction_matches_prior_signed_kernel else "reject",
            "subtraction profile matches prior signed-density kernel up to global sign",
            profile_identity_max_abs,
            "Because V_sub = -f0^2 + fL^2 = -(f0^2 - fL^2), the exact subtraction profile should collapse onto the previously rejected signed-density kernel up to a global sign.",
        ),
        row(
            "f_sub_matches_prior_signed_form_factor",
            "pass" if form_factor_identity_gap <= 1.0e-12 else "reject",
            "F_sub matches prior signed-density form factor",
            form_factor_identity_gap,
            "The normalized form factor is invariant under the global sign flip, so exact subtraction reproduces the prior signed-density blind-vector result.",
        ),
        row(
            "alpha_sub_matches_prior_signed_alpha",
            "pass" if alpha_identity_gap <= 1.0e-12 else "reject",
            "alpha_sub matches prior signed-density alpha",
            alpha_identity_gap,
            "The exact subtraction alpha collapses onto the already retained signed-density/vector no-go value.",
        ),
        row(
            "alpha_sub_at_q_theory",
            "watch",
            "alpha_sub at q_theory",
            alpha_sub,
            "This is the exact v^2-subtracted alpha obtained from the restored exact branch.",
        ),
        row(
            "residual_sub_rel",
            "watch",
            "relative residual of alpha_sub vs target",
            residual_sub,
            "The exact subtraction result must be compared directly against the retained scalar residual before any new disposition is fixed.",
        ),
        row(
            "residual_vs_scalar_gap",
            "reject" if worsen_selected else "pass",
            "residual_sub minus retained scalar residual",
            residual_gap_vs_scalar,
            "Positive values mean the exact subtraction treatment worsens the retained scalar candidate rather than tightening it.",
        ),
        row(
            "residual_vs_scalar_ratio",
            "reject" if worsen_selected else "pass",
            "residual_sub divided by retained scalar residual",
            residual_ratio_vs_scalar,
            "This measures how far the exact subtraction treatment drifts away from the retained scalar candidate.",
        ),
        row(
            "subtraction_result_improve",
            "pass" if improve_selected else "reject",
            "exact subtraction result classified as improve",
            truth(improve_selected),
            "Improve would require the exact subtraction residual to drop below the retained scalar residual.",
        ),
        row(
            "subtraction_result_neutral",
            "pass" if neutral_selected else "reject",
            "exact subtraction result classified as neutral",
            truth(neutral_selected),
            "Neutral would require the exact subtraction residual to stay numerically equal to the retained scalar residual.",
        ),
        row(
            "subtraction_result_worsen",
            "pass" if worsen_selected else "reject",
            "exact subtraction result classified as worsen",
            truth(worsen_selected),
            "Worsen means the exact subtraction treatment reproduces the prior signed-kernel no-go and therefore fails to support exact Case I closure.",
        ),
        row(
            "case_i_exact_foundation_supported",
            "pass" if case_i_exact_foundation_supported else "reject",
            "Case I exact foundation supported after subtraction",
            truth(case_i_exact_foundation_supported),
            "Case I exact foundation would require the exact subtraction result to improve the retained scalar candidate, not destroy it.",
        ),
        row(
            "quadratic_tp_disposition_sync_admissible_now",
            "pass" if quadratic_tp_disposition_sync_admissible_now else "reject",
            "quadratic transverse-projection disposition sync admissible now",
            truth(quadratic_tp_disposition_sync_admissible_now),
            "Once the exact subtraction result is fixed as improve/neutral/worsen, the next honest action is the official quadratic disposition sync.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "selected_subtraction_result_case": result_case,
        "subtraction_profile_matches_prior_signed_kernel": subtraction_matches_prior_signed_kernel,
        "profile_identity_max_abs": profile_identity_max_abs,
        "F_sub_at_q_theory": f_sub,
        "F_signed_at_q_theory": f_signed,
        "F_scalar_at_q_theory": f_scalar,
        "alpha_sub_at_q_theory": alpha_sub,
        "alpha_signed_at_q_theory": alpha_signed,
        "alpha_scalar_proxy_at_q_theory": alpha_scalar,
        "residual_sub_rel": residual_sub,
        "scalar_alpha_residual_rel": scalar_residual_rel,
        "projected_nlo_ceiling": projected_nlo_ceiling,
        "residual_vs_scalar_gap": residual_gap_vs_scalar,
        "residual_vs_scalar_ratio": residual_ratio_vs_scalar,
        "improve_selected": improve_selected,
        "neutral_selected": neutral_selected,
        "worsen_selected": worsen_selected,
        "case_i_exact_foundation_supported": case_i_exact_foundation_supported,
        "case_i_leading_foundation_candidate_survives_exact_subtraction": case_i_candidate_survives_exact_subtraction,
        "quadratic_tp_disposition_sync_admissible_now": quadratic_tp_disposition_sync_admissible_now,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": True,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "formulas": {
            "subtracted_isotropic_profile": "V_eff^sub(r) = lambda(-f_0(r)^2 + f_L(r)^2)",
            "prior_signed_kernel": "rho_signed(r) = f_0(r)^2 - f_L(r)^2",
            "identity": "V_eff^sub(r) = -rho_signed(r)",
            "form_factor": "F(q) = int rho(r) sinc(qr) r^2 dr / int rho(r) r^2 dr",
        },
        "hits": {
            "directive_subtraction_definition": hit(directive_text, "V_{\\rm eff}^{\\rm sub}(r)"),
            "directive_step2": hit(directive_text, "F_{\\rm sub}(q)"),
            "directive_step4": hit(directive_text, "悪化"),
            "current_problem_v2_subtraction": hit(current_problem_text, "v^2 subtraction exact treatment audit"),
            "current_status_v2_subtraction": hit(current_status_text, "v^2 subtraction exact treatment audit"),
            "unified_roadmap_v2_subtraction": hit(
                unified_roadmap_text, "v^2 subtraction exact treatment audit"
            ),
            "part5_v2_subtraction": hit(part5_text, "v^2 subtraction exact treatment audit"),
        },
        "profile_samples": {
            "radius_count": int(radius.size),
            "signed_norm": norm_signed,
            "subtracted_norm": norm_sub,
            "scalar_norm": norm_scalar,
            "phase1_equivalent_amp_l": amp_l,
            "phase1_equivalent_max_abs_ratio": float(phase1_row["max_abs_ratio"]),
        },
        "retained_numeric_state": {
            "scalar_F_exact_at_q_theory": float(tp_summary["scalar_F_exact_at_q_theory"]),
            "scalar_alpha_exact_at_q_theory": float(tp_summary["scalar_alpha_exact_at_q_theory"]),
            "vector_F_at_q_theory": float(phase1_row["F_at_q_theory"]),
            "vector_alpha_at_q_theory": float(phase1_row["alpha_at_q_theory"]),
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
        },
    }

    inventory_paths = write_artifact(
        "inventory",
        payload(
            "8.7.56.1599",
            f"{STEP_NAME} inventory",
            inputs,
            rows,
            summary,
            decision,
            evidence,
        ),
    )
    audit_paths = write_artifact(
        "audit",
        payload(
            "8.7.56.1600",
            f"{STEP_NAME} audit",
            inputs,
            rows,
            summary,
            decision,
            evidence,
        ),
    )
    gate_paths = write_artifact(
        "declaration_gate",
        payload(
            "8.7.56.1601",
            f"{STEP_NAME} declaration gate",
            inputs,
            rows,
            summary,
            decision,
            evidence,
        ),
    )
    route_paths = write_artifact(
        "route_sync",
        payload(
            "8.7.56.1602",
            f"{STEP_NAME} route sync",
            inputs,
            rows,
            summary,
            decision,
            evidence,
        ),
    )

    print("[ok] v^2 subtraction exact-treatment artifacts written:")
    print(f" - {inventory_paths['json']}")
    print(f" - {audit_paths['json']}")
    print(f" - {gate_paths['json']}")
    print(f" - {route_paths['json']}")


if __name__ == "__main__":
    main()
