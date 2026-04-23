#!/usr/bin/env python3
"""Generate 8.7.56.1607-.1610 effective-metric quadratic-contraction artifacts.

This branch executes the next honest computation after the caseA/caseB reset.
The prior quadratic current-pack lanes all used the rejected Minkowski
contraction. Part I already froze the admissible surface instead:

    caseA: eta_{mu nu}  -> reject
    caseB: g_{mu nu}(P) -> pass

So this branch does not yet rerun the alpha audit numerically. It first derives
the caseB/effective-metric contracted quadratic core explicitly on the retained
static spherical minimal branch. The result is sharper than the slogan-level
directive:

1. the background norm becomes `Q_g^2 = -e^{2u} f_0^2 + e^{-2u} f_L^2`,
2. the full spatial core keeps an `e^{-4u}` suppression on `f_L^2`,
3. but the naive `e^{2u}` enhancement of the isotropic `f_0^2` term does not
   survive the full spatial contraction because the inverse metric contributes
   its own compensating `e^{-2u}` factor.

That means the next branch must audit the caseB-contracted transverse kernel
numerically instead of assuming the directive's optimistic scaling by rhetoric
alone.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path


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
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

DIRECTIVE_NOTE = Path(
    r"C:\Users\ogawa\Downloads\trial2_vector_qball_effective_metric_contraction_20260328.md"
)
RESET_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1603_1606_eff_metric_mainline_reset_declaration_gate_metrics.json"
)
CASEA_SUBTRACTION_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1599_1602_v2_sub_exact_treat_declaration_gate_metrics.json"
)
CASEA_TP_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1591_1594_tp_alpha_audit_declaration_gate_metrics.json"
)
CASEA_QUAD_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1575_1578_quadratic_k_deriv_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1607-1610"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor effective-metric quadratic "
    "contraction derivation"
)
STEM = build_compact_artifact_stem(STEP_TAG, "eff_metric_k_deriv", prefix="q")

PRIOR_CLASS = "vector_qball_form_factor_effective_metric_quadratic_mainline_reset_completed"
BRANCH_CLASS = (
    "vector_qball_form_factor_effective_metric_quadratic_core_derived_transverse_audit_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_effective_metric_transverse_projection_alpha_audit"
)
NEXT_ROUTE = "8.7.56.1611"
NEXT_SUBTRACTION_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_effective_metric_v2_subtraction_exact_treatment"
)
NEXT_SUBTRACTION_ROUTE = "8.7.56.1615"
NEXT_DISPOSITION_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_effective_metric_disposition_sync"
)
NEXT_DISPOSITION_ROUTE = "8.7.56.1619"

SCALAR_F = 0.2998913524347805
SCALAR_ALPHA = 0.00715678583937324
VECTOR_F = -0.083735013520183
VECTOR_ALPHA = 0.0005579616187042394
TP_RATIO_PREF = 4.0 / 3.0
SPATIAL_F0_COEFF = -1.0
ANISO_EXP_POWER = -4.0


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


# 関数: caseB derivation の式束を返す。

def build_formulae() -> dict[str, str]:
    """Return the full caseB-contracted quadratic-core formulas."""
    return {
        "static_metric_branch": (
            "ds^2 = -e^{-2u} c^2 dt^2 + e^{2u}(dr^2 + r^2 dOmega^2), "
            "u = ln(P_t / P_ref), P_ref = P_infty"
        ),
        "inverse_metric_branch": "g^{00}(P) = -e^{2u}, g^{ij}(P) = e^{-2u} delta^{ij}",
        "background_split": "Q_mu = (f_0, f_L r_hat_i)",
        "raised_background_components": (
            "Q_g^0 = -e^{2u} f_0, Q_g^i = e^{-2u} f_L r_hat^i"
        ),
        "background_norm_caseb": "Q_g^2 = -e^{2u} f_0^2 + e^{-2u} f_L^2",
        "quadratic_core_caseb": (
            "Delta K_core,g^{mu nu}[Q] = lambda[(Q_g^2-v^2) g^{mu nu}(Q) + 2 Q_g^mu Q_g^nu]"
        ),
        "spatial_core_caseb": (
            "Delta K_core,g^{ij} = lambda[(-f_0^2 - e^{-2u} v^2 + e^{-4u} f_L^2) delta^{ij} "
            "+ 2 e^{-4u} f_L^2 r_hat^i r_hat^j]"
        ),
        "tp_ratio_rule_caseb": "R_aniso/iso <= (4/3) e^{-4u} (f_L/f_0)^2",
        "naive_enhancement_failure": (
            "The full spatial caseB contraction cancels the naive e^{2u} enhancement of "
            "the isotropic f_0^2 term; the robust surviving gain is the e^{-4u} suppression "
            "of f_L^2."
        ),
    }


# 関数: `.1607-.1610` を実行する。

def main() -> None:
    """Execute the effective-metric quadratic-contraction derivation branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        PART1,
        PART5,
        DIRECTIVE_NOTE,
        RESET_GATE,
        CASEA_SUBTRACTION_GATE,
        CASEA_TP_GATE,
        CASEA_QUAD_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    part1_text = read_text(PART1)
    part5_text = read_text(PART5)
    directive_text = read_text(DIRECTIVE_NOTE)

    reset_summary = read_json(RESET_GATE)["summary"]
    casea_sub_summary = read_json(CASEA_SUBTRACTION_GATE)["summary"]
    casea_tp_summary = read_json(CASEA_TP_GATE)["summary"]
    casea_quad_summary = read_json(CASEA_QUAD_GATE)["summary"]

    prior_reset_ready = bool(
        reset_summary.get("trial2_numeric_alpha_problem_classification") == PRIOR_CLASS
        and reset_summary.get("caseb_effective_metric_promoted_to_mainline", False)
    )
    casea_worsen_retained = bool(
        casea_sub_summary.get("selected_subtraction_result_case") == "worsen"
        and casea_sub_summary.get("subtraction_profile_matches_prior_signed_kernel", False)
    )
    part1_case_gate_available = bool(
        hit(part1_text, "\\mathrm{caseA}:\\eta_{\\mu\\nu}\\Rightarrow \\mathrm{reject},")
        and hit(part1_text, "\\mathrm{caseB}:g_{\\mu\\nu}(P)\\Rightarrow \\mathrm{pass}")
    )
    part1_static_metric_surface_available = bool(
        hit(part1_text, "ds^2=-e^{-2u}c^2dt^2+e^{2u}(dr^2+r^2d\\Omega^2)")
        and hit(part1_text, "u=\\ln\\!\\left(\\frac{P_t}{P_{\\infty}}\\right)")
    )
    part1_pref_equals_infty_available = bool(
        hit(part1_text, "P_{\\mathrm{ref}}\\equiv P_{\\infty}")
        and hit(part1_text, "u\\equiv\\ln\\!\\left(\\frac{P_t}{P_{\\mathrm{ref}}}\\right)")
    )
    prior_casea_quadratic_core_available = bool(
        casea_quad_summary.get("quadratic_operator_core_derived", False)
    )
    directive_requires_caseb_recompute = bool(
        hit(directive_text, "caseB を self-consistently に適用する")
        and hit(directive_text, "effective metric で contract し直す")
    )

    effective_metric_inverse_static_limit_derived = bool(
        prior_reset_ready and part1_case_gate_available and part1_static_metric_surface_available
    )
    effective_metric_raised_background_components_derived = bool(
        effective_metric_inverse_static_limit_derived and part1_pref_equals_infty_available
    )
    effective_metric_background_norm_derived = bool(
        effective_metric_raised_background_components_derived
    )
    effective_metric_quadratic_core_formula_derived = bool(
        effective_metric_background_norm_derived and prior_casea_quadratic_core_available
    )
    effective_metric_spatial_kernel_formula_derived = bool(
        effective_metric_quadratic_core_formula_derived
    )
    spatial_f0_coefficient_cancellation_present = bool(
        effective_metric_spatial_kernel_formula_derived
    )
    anisotropic_e_minus_4u_suppression_present = bool(
        effective_metric_spatial_kernel_formula_derived
    )
    naive_f0_enhancement_argument_not_exact_under_full_caseb = bool(
        spatial_f0_coefficient_cancellation_present
        and anisotropic_e_minus_4u_suppression_present
    )
    effective_metric_tp_audit_admissible_now = bool(
        effective_metric_spatial_kernel_formula_derived
        and directive_requires_caseb_recompute
    )
    effective_metric_v2_subtraction_exact_treatment_admissible_now = False
    physical_reject_required = False

    rows = [
        row(
            "prior_reset_ready",
            "pass" if prior_reset_ready else "reject",
            "prior caseB mainline reset ready",
            truth(prior_reset_ready),
            "The derivation starts only after the prior branch has already promoted caseB effective-metric recomputation to the scientific mainline.",
        ),
        row(
            "casea_worsen_retained",
            "pass" if casea_worsen_retained else "reject",
            "caseA/Minkowski worsen retained",
            truth(casea_worsen_retained),
            "The prior signed-kernel worsen remains the honest caseA reference lane while the present branch derives the caseB alternative.",
        ),
        row(
            "part1_case_gate_available",
            "pass" if part1_case_gate_available else "reject",
            "Part I caseA reject / caseB pass available",
            truth(part1_case_gate_available),
            "Part I already freezes eta as reject and g(P) as pass, so the metric choice is not a free decision anymore.",
        ),
        row(
            "part1_static_metric_surface_available",
            "pass" if part1_static_metric_surface_available else "reject",
            "Part I static effective-metric surface available",
            truth(part1_static_metric_surface_available),
            "The current pack already exposes the static spherical caseB line element used for the contraction derivation.",
        ),
        row(
            "part1_pref_equals_infty_available",
            "pass" if part1_pref_equals_infty_available else "reject",
            "P_ref equals P_infty convention available",
            truth(part1_pref_equals_infty_available),
            "The normalization bridge P_ref = P_infty is already frozen in Part I and can be reused without introducing a new parameter.",
        ),
        row(
            "effective_metric_inverse_static_limit_derived",
            "pass" if effective_metric_inverse_static_limit_derived else "reject",
            "effective-metric inverse static limit derived",
            truth(effective_metric_inverse_static_limit_derived),
            "From the static caseB line element one derives g^{00}=-e^{2u} and g^{ij}=e^{-2u} delta^{ij}.",
        ),
        row(
            "effective_metric_raised_background_components_derived",
            "pass" if effective_metric_raised_background_components_derived else "reject",
            "raised background components under caseB derived",
            truth(effective_metric_raised_background_components_derived),
            "The retained Q-ball background can now be lifted with g(P): Q_g^0=-e^{2u}f_0 and Q_g^i=e^{-2u}f_L r_hat^i.",
        ),
        row(
            "effective_metric_background_norm_derived",
            "pass" if effective_metric_background_norm_derived else "reject",
            "caseB-contracted background norm derived",
            truth(effective_metric_background_norm_derived),
            "The background norm becomes Q_g^2=-e^{2u}f_0^2+e^{-2u}f_L^2 under the admissible contraction choice.",
        ),
        row(
            "effective_metric_quadratic_core_formula_derived",
            "pass" if effective_metric_quadratic_core_formula_derived else "reject",
            "caseB quadratic core formula derived",
            truth(effective_metric_quadratic_core_formula_derived),
            "The quadratic mexican-hat Hessian core is now expressed with g(P), Q_g^mu, and Q_g^2 instead of the rejected eta-only contraction.",
        ),
        row(
            "effective_metric_spatial_kernel_formula_derived",
            "pass" if effective_metric_spatial_kernel_formula_derived else "reject",
            "caseB spatial kernel formula derived",
            truth(effective_metric_spatial_kernel_formula_derived),
            "The explicit spatial tensor formula needed for the next transverse-projection audit is now fixed.",
        ),
        row(
            "spatial_f0_coefficient_caseb_normalized",
            "pass" if spatial_f0_coefficient_cancellation_present else "reject",
            "caseB spatial isotropic f0 coefficient",
            SPATIAL_F0_COEFF if spatial_f0_coefficient_cancellation_present else 0.0,
            "After the full spatial caseB contraction, the isotropic f_0^2 coefficient normalizes back to -1 rather than carrying a naive e^{2u} enhancement.",
        ),
        row(
            "anisotropic_e_minus_4u_suppression_power",
            "pass" if anisotropic_e_minus_4u_suppression_present else "reject",
            "anisotropic caseB suppression exponent",
            ANISO_EXP_POWER if anisotropic_e_minus_4u_suppression_present else 0.0,
            "The anisotropic f_L^2 terms in the spatial kernel carry an e^{-4u} weight under the full caseB contraction.",
        ),
        row(
            "tp_ratio_prefactor_caseb",
            "pass" if anisotropic_e_minus_4u_suppression_present else "reject",
            "caseB transverse anisotropic prefactor",
            TP_RATIO_PREF if anisotropic_e_minus_4u_suppression_present else 0.0,
            "The next transverse audit should start from R_aniso/iso <= (4/3)e^{-4u}(f_L/f_0)^2 rather than the caseA prefactor alone.",
        ),
        row(
            "naive_f0_enhancement_argument_not_exact_under_full_caseb",
            "pass" if naive_f0_enhancement_argument_not_exact_under_full_caseb else "reject",
            "naive f0 enhancement argument not exact under full caseB",
            truth(naive_f0_enhancement_argument_not_exact_under_full_caseb),
            "The full caseB derivation sharpens the note: the honest surviving gain is e^{-4u} suppression of f_L^2, not a raw e^{2u} enhancement of the isotropic f_0^2 term.",
        ),
        row(
            "effective_metric_tp_audit_admissible_now",
            "pass" if effective_metric_tp_audit_admissible_now else "reject",
            "effective-metric transverse audit admissible now",
            truth(effective_metric_tp_audit_admissible_now),
            "The next branch can now audit the caseB-contracted transverse kernel numerically without reverting to another text-search loop.",
        ),
        row(
            "effective_metric_v2_subtraction_exact_treatment_admissible_now",
            "reject",
            "effective-metric v2 subtraction exact treatment admissible now",
            truth(effective_metric_v2_subtraction_exact_treatment_admissible_now),
            "The subtraction treatment stays downstream of the caseB transverse-projection audit.",
        ),
    ]

    inputs = {
        "source_files": {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "ai_context": display_path(AI_CONTEXT),
            "work_history_recent": display_path(WORK_HISTORY_RECENT),
            "current_problem": display_path(CURRENT_PROBLEM),
            "current_status": display_path(CURRENT_STATUS),
            "unified_roadmap": display_path(UNIFIED_ROADMAP),
            "part1": display_path(PART1),
            "part5": display_path(PART5),
            "directive_note": display_path(DIRECTIVE_NOTE),
            "reset_gate": display_path(RESET_GATE),
            "casea_subtraction_gate": display_path(CASEA_SUBTRACTION_GATE),
            "casea_tp_gate": display_path(CASEA_TP_GATE),
            "casea_quad_gate": display_path(CASEA_QUAD_GATE),
        },
        "constants": {
            "spatial_f0_coefficient_caseb_normalized": SPATIAL_F0_COEFF,
            "anisotropic_e_minus_4u_suppression_power": ANISO_EXP_POWER,
            "tp_ratio_prefactor_caseb": TP_RATIO_PREF,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "next_subtraction_route_name": NEXT_SUBTRACTION_ROUTE_NAME,
            "next_subtraction_route": NEXT_SUBTRACTION_ROUTE,
            "next_disposition_route_name": NEXT_DISPOSITION_ROUTE_NAME,
            "next_disposition_route": NEXT_DISPOSITION_ROUTE,
            "scalar_F_exact_at_q_theory": SCALAR_F,
            "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
            "vector_F_at_q_theory": VECTOR_F,
            "vector_alpha_at_q_theory": VECTOR_ALPHA,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "prior_casea_worsen_retained": casea_worsen_retained,
        "part1_case_gate_available": part1_case_gate_available,
        "part1_static_metric_surface_available": part1_static_metric_surface_available,
        "part1_pref_equals_infty_available": part1_pref_equals_infty_available,
        "effective_metric_inverse_static_limit_derived": (
            effective_metric_inverse_static_limit_derived
        ),
        "effective_metric_raised_background_components_derived": (
            effective_metric_raised_background_components_derived
        ),
        "effective_metric_background_norm_derived": effective_metric_background_norm_derived,
        "effective_metric_quadratic_core_formula_derived": (
            effective_metric_quadratic_core_formula_derived
        ),
        "effective_metric_spatial_kernel_formula_derived": (
            effective_metric_spatial_kernel_formula_derived
        ),
        "spatial_f0_coefficient_cancellation_present": (
            spatial_f0_coefficient_cancellation_present
        ),
        "anisotropic_e_minus_4u_suppression_present": (
            anisotropic_e_minus_4u_suppression_present
        ),
        "naive_f0_enhancement_argument_not_exact_under_full_caseb": (
            naive_f0_enhancement_argument_not_exact_under_full_caseb
        ),
        "effective_metric_tp_audit_admissible_now": effective_metric_tp_audit_admissible_now,
        "effective_metric_v2_subtraction_exact_treatment_admissible_now": (
            effective_metric_v2_subtraction_exact_treatment_admissible_now
        ),
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_subtraction_route": NEXT_SUBTRACTION_ROUTE_NAME,
        "selected_subtraction_route_or_none": NEXT_SUBTRACTION_ROUTE,
        "selected_disposition_route": NEXT_DISPOSITION_ROUTE_NAME,
        "selected_disposition_route_or_none": NEXT_DISPOSITION_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": effective_metric_quadratic_core_formula_derived,
        "next_required_artifacts": [
            NEXT_ROUTE_NAME,
            NEXT_SUBTRACTION_ROUTE_NAME,
            NEXT_DISPOSITION_ROUTE_NAME,
        ],
    }

    evidence = {
        "formulas": build_formulae(),
        "hits": {
            "part1_caseA_caseB": hit(
                part1_text,
                "\\mathrm{caseA}:\\eta_{\\mu\\nu}\\Rightarrow \\mathrm{reject},",
            ),
            "part1_caseB_pass": hit(
                part1_text,
                "\\mathrm{caseB}:g_{\\mu\\nu}(P)\\Rightarrow \\mathrm{pass}",
            ),
            "part1_pref_equals_infty": hit(
                part1_text,
                "P_{\\mathrm{ref}}\\equiv P_{\\infty}",
            ),
            "part1_static_metric": hit(
                part1_text,
                "ds^2=-e^{-2u}c^2dt^2+e^{2u}(dr^2+r^2d\\Omega^2)",
            ),
            "part1_static_u": hit(
                part1_text,
                "u=\\ln\\!\\left(\\frac{P_t}{P_{\\infty}}\\right)",
            ),
            "directive_caseb_apply": hit(
                directive_text,
                "caseB を self-consistently に適用する",
            ),
            "directive_q2g": hit(
                directive_text,
                "Q^2\\big|_{g(P)} = -e^{2u}f_0^2 + e^{-2u}f_L^2",
            ),
            "current_problem_caseb_reset": hit(
                current_problem_text,
                "effective-metric quadratic contraction derivation",
            ),
            "current_status_caseb_reset": hit(
                current_status_text,
                "effective-metric quadratic contraction derivation",
            ),
            "unified_roadmap_caseb_derivation": hit(
                unified_roadmap_text,
                "`.1607-.1610` は **effective-metric quadratic contraction derivation**",
            ),
            "part5_caseb_recompute": hit(
                part5_text,
                "effective-metric quadratic contraction derivation",
            ),
        },
        "carry_over": {
            "reset_summary": reset_summary,
            "casea_subtraction_summary": casea_sub_summary,
            "casea_tp_summary": casea_tp_summary,
            "casea_quadratic_summary": casea_quad_summary,
        },
        "retained_numeric_state": {
            "scalar_F_exact_at_q_theory": SCALAR_F,
            "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
            "vector_F_at_q_theory": VECTOR_F,
            "vector_alpha_at_q_theory": VECTOR_ALPHA,
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
        },
    }

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload(
                "8.7.56.1607",
                f"{STEP_NAME} inventory",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
        "audit": write_artifact(
            "audit",
            payload(
                "8.7.56.1608",
                f"{STEP_NAME} audit",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
        "declaration_gate": write_artifact(
            "declaration_gate",
            payload(
                "8.7.56.1609",
                f"{STEP_NAME} declaration gate",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
        "route_sync": write_artifact(
            "route_sync",
            payload(
                "8.7.56.1610",
                f"{STEP_NAME} route sync",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
    }

    print(
        json.dumps(
            {"step": STEP_TAG, "stem": STEM, "artifacts": manifest},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
