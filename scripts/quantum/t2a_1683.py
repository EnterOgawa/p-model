#!/usr/bin/env python3
"""Generate 8.7.56.1683-.1686 transverse resolvent-response audit artifacts.

This branch follows the failure-structure derivation of `.1679-.1682`.
There the common missing object was identified as a canonical probe-response
map rather than another local density. The newly isolated candidate was the
vacuum-subtracted transverse susceptibility

    Delta chi_T[Q] = Pi_T (K[Q]^{-1} - K[0]^{-1}) Pi_T.

The present branch must answer a narrower question:

    Can the current frozen-action pack turn Delta chi_T into one canonical
    observable with one canonical alpha read?

The audit uses the already-fixed projected-kernel matrix element
`M_T(q) = <J_perp|Delta K_T[Q]|J_perp>` and the massless transverse vacuum
kernel `K_{T,0}(q) = q^2` to construct three finite current-pack response
variants:

1. one-leg amputation:
   A_1(q) = -q^2 Delta chi_T(q) = M_T(q)/(q^2 + M_T(q))
2. two-leg amputation:
   A_2(q) = -q^4 Delta chi_T(q) = q^2 M_T(q)/(q^2 + M_T(q))
3. static-kernel-scaled proxy:
   A_stat(q) = F_T(q)/(1 + F_T(q)), where F_T(q)=M_T(q)/M_T(0)

The unamputated susceptibility itself keeps a q->0 pole and is therefore not
normalizable as a form factor. If the finite reads above spread widely, the
honest result is not "pick the best one", but "no canonical amputation /
normalization rule is available under the current pack".
"""

from __future__ import annotations

import csv
import json
import math
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
EXPERT_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "54_trial2_numeric_alpha_vector_qball_failure_structure_probe_response_query.md"
)

DERIV_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1679_1682_fail_struct_resolvent_declaration_gate_metrics.json"
)
PROJECTED_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1659_1662_pmu_tresp_pk_audit_declaration_gate_metrics.json"
)
FALLBACK_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1667_1670_fallback_closeout_registry_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1683-1686"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor transverse "
    "resolvent-response observable audit"
)
STEM = build_compact_artifact_stem(STEP_TAG, "tresp_resolvent_audit", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_failure_structure_local_surrogate_logic_falsified_"
    "transverse_resolvent_response_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_transverse_resolvent_response_scheme_dependent_"
    "no_canonical_read_decision_gate_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_resolvent_decision_gate_or_"
    "fallback_return"
)
NEXT_ROUTE = "8.7.56.1687"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_new_action_level_"
    "structure_or_external_input_reactivation_wait_restore"
)
FOLLOWUP_ROUTE = "8.7.56.1691"

TARGET_ALPHA = 1.0 / 137.035999084
BARE_ALPHA = 1.0 / (4.0 * math.pi)
SCALAR_ALPHA = 0.00715678583937324
VECTOR_ALPHA = 0.0005579616187042394


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


# 関数: alpha と target の相対残差を返す。

def alpha_residual_rel(alpha_value: float) -> float:
    """Return one target-relative alpha residual."""
    return float(abs(float(alpha_value) - TARGET_ALPHA) / TARGET_ALPHA)


# 関数: 代表 alpha の cluster spread を返す。

def span_ratio(values: list[float]) -> dict[str, float]:
    """Return span and max/min ratio over strictly positive values."""
    vmin = min(values)
    vmax = max(values)
    return {
        "min": float(vmin),
        "max": float(vmax),
        "span": float(vmax - vmin),
        "ratio": float(vmax / vmin),
    }


# 関数: resolvent-response formula bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return the current-pack response formulas used in this audit."""
    return {
        "vacuum_transverse_kernel": "K_T,0(q) = q^2",
        "projected_kernel_shift": "Delta K_T(q) <-> M_T(q) = <J_perp|Delta K_T[Q]|J_perp>",
        "resolvent_kernel": "Delta chi_T(q) = (q^2 + M_T(q))^{-1} - q^{-2}",
        "unamputated_problem": "Delta chi_T(q) keeps a q -> 0 pole and therefore cannot be normalized directly as a form factor under the current pack.",
        "one_leg_amputation": "A_1(q) = -q^2 Delta chi_T(q) = M_T(q)/(q^2 + M_T(q))",
        "two_leg_amputation": "A_2(q) = -q^4 Delta chi_T(q) = q^2 M_T(q)/(q^2 + M_T(q))",
        "static_scaled_proxy": "A_stat(q) = F_T(q)/(1 + F_T(q)), with F_T(q) = M_T(q)/M_T(0)",
        "decision_rule": "If finite response reads remain strongly scheme-dependent and no canonical amputation rule is selected by the current pack, the honest read is no canonical resolvent observable under the current pack.",
    }


# 関数: `.1683-.1686` を実行する。

def main() -> None:
    """Execute the transverse resolvent-response observable audit."""
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
        EXPERT_NOTE,
        DERIV_GATE,
        PROJECTED_GATE,
        FALLBACK_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    part1_text = read_text(PART1)
    part5_text = read_text(PART5)
    expert_note_text = read_text(EXPERT_NOTE)

    deriv_gate = read_json(DERIV_GATE)
    projected_gate = read_json(PROJECTED_GATE)
    fallback_gate = read_json(FALLBACK_GATE)

    deriv_summary = deriv_gate["summary"]
    projected_summary = projected_gate["summary"]
    fallback_summary = fallback_gate["summary"]

    q_theory = float(projected_gate["inputs"]["constants"]["q_theory_over_m0"])
    q_squared = float(q_theory * q_theory)
    m_q = float(projected_summary["official_projected_kernel_numerator_at_q_theory"])
    m_zero = float(projected_summary["official_projected_kernel_denominator_at_zero"])
    f_t = float(projected_summary["official_projected_kernel_F_at_q_theory"])
    alpha_t = float(projected_summary["official_projected_kernel_alpha_at_q_theory"])

    raw_delta_chi_q = float((1.0 / (q_squared + m_q)) - (1.0 / q_squared))
    raw_q0_pole_present = True
    unamputated_form_factor_normalizable = False

    one_leg_response = float(-(q_squared) * raw_delta_chi_q)
    one_leg_alpha = float((one_leg_response * one_leg_response) / (4.0 * math.pi))
    one_leg_supports_scalar = bool(alpha_residual_rel(one_leg_alpha) <= 0.05)

    two_leg_response = float(-(q_squared * q_squared) * raw_delta_chi_q)
    two_leg_alpha = float((two_leg_response * two_leg_response) / (4.0 * math.pi))
    two_leg_supports_scalar = bool(alpha_residual_rel(two_leg_alpha) <= 0.05)

    static_scaled_response = float(f_t / (1.0 + f_t))
    static_scaled_alpha = float(
        (static_scaled_response * static_scaled_response) / (4.0 * math.pi)
    )
    static_scaled_supports_scalar = bool(alpha_residual_rel(static_scaled_alpha) <= 0.05)

    finite_alpha_values = [one_leg_alpha, two_leg_alpha, static_scaled_alpha]
    spread = span_ratio(finite_alpha_values)
    scheme_dependence_large = bool(spread["ratio"] >= 10.0)
    all_finite_candidates_fail_scalar = bool(
        (not one_leg_supports_scalar)
        and (not two_leg_supports_scalar)
        and (not static_scaled_supports_scalar)
    )
    canonical_plane_wave_source_candidate_available = True
    canonical_source_amputation_rule_available = False
    transverse_resolvent_canonical_observable_available = False
    resolvent_family_failed_or_unavailable_under_current_pack = bool(
        all_finite_candidates_fail_scalar and scheme_dependence_large
    )
    resolvent_decision_gate_admissible_now = True
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
            "part1": display_path(PART1),
            "part5": display_path(PART5),
            "expert_note": display_path(EXPERT_NOTE),
            "derivation_gate": display_path(DERIV_GATE),
            "projected_kernel_gate": display_path(PROJECTED_GATE),
            "fallback_gate": display_path(FALLBACK_GATE),
        },
        "constants": {
            "q_theory_over_m0": q_theory,
            "q_squared_at_q_theory": q_squared,
            "projected_kernel_numerator_at_q_theory": m_q,
            "projected_kernel_denominator_at_zero": m_zero,
            "official_projected_kernel_alpha_at_q_theory": alpha_t,
            "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
            "vector_alpha_at_q_theory": VECTOR_ALPHA,
            "bare_alpha": BARE_ALPHA,
            "target_alpha": TARGET_ALPHA,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "followup_route_name": FOLLOWUP_ROUTE_NAME,
            "followup_route": FOLLOWUP_ROUTE,
        },
    }

    rows = [
        row(
            "transverse_resolvent_surface_untested_prior",
            "pass",
            "transverse resolvent-response surface isolated by prior branch",
            1.0,
            "The branch starts only after the failure-structure derivation has already isolated Delta chi_T as the genuinely untested object.",
        ),
        row(
            "canonical_plane_wave_source_candidate_available",
            "pass" if canonical_plane_wave_source_candidate_available else "reject",
            "canonical plane-wave transverse source candidate available",
            truth(canonical_plane_wave_source_candidate_available),
            "A conserved transverse plane-wave probe is the natural current-pack source candidate because the vacuum transverse mode is already fixed.",
        ),
        row(
            "unamputated_q0_pole_present",
            "pass" if raw_q0_pole_present else "reject",
            "unamputated susceptibility keeps q->0 pole",
            truth(raw_q0_pole_present),
            "Delta chi_T(q) itself behaves like q^-2 at small q, so it cannot be normalized directly into a finite form factor under the current pack.",
        ),
        row(
            "unamputated_form_factor_normalizable",
            "pass" if unamputated_form_factor_normalizable else "reject",
            "unamputated resolvent form factor normalizable",
            truth(unamputated_form_factor_normalizable),
            "This remains false because no current-pack rule removes the q->0 pole without choosing an amputation convention.",
        ),
        row(
            "one_leg_amputated_alpha_at_q_theory",
            "reject" if not one_leg_supports_scalar else "pass",
            "one-leg-amputated resolvent alpha at q_theory",
            one_leg_alpha,
            "Using A_1(q) = -q^2 Delta chi_T(q) saturates near the bare scale and overshoots the target badly.",
        ),
        row(
            "two_leg_amputated_alpha_at_q_theory",
            "reject" if not two_leg_supports_scalar else "pass",
            "two-leg-amputated resolvent alpha at q_theory",
            two_leg_alpha,
            "Using A_2(q) = -q^4 Delta chi_T(q) stays on a small no-go scale and still does not support the scalar candidate.",
        ),
        row(
            "static_scaled_proxy_alpha_at_q_theory",
            "reject" if not static_scaled_supports_scalar else "pass",
            "static-kernel-scaled resolvent proxy alpha at q_theory",
            static_scaled_alpha,
            "Scaling by the static projected-kernel matrix element gives another finite read, but it remains non-canonical and still misses the scalar candidate.",
        ),
        row(
            "all_finite_candidates_fail_scalar",
            "pass" if all_finite_candidates_fail_scalar else "reject",
            "all finite resolvent candidates fail scalar candidate",
            truth(all_finite_candidates_fail_scalar),
            "No finite current-pack resolvent read approaches the retained scalar strong candidate once the amputation ambiguity is made explicit.",
        ),
        row(
            "scheme_alpha_span",
            "watch" if scheme_dependence_large else "pass",
            "alpha span across finite resolvent candidates",
            spread["span"],
            "The finite resolvent reads spread over a wide range, so the current pack does not yet select one canonical response normalization.",
        ),
        row(
            "scheme_alpha_ratio",
            "watch" if scheme_dependence_large else "pass",
            "max/min alpha ratio across finite resolvent candidates",
            spread["ratio"],
            "A large ratio indicates strong scheme dependence rather than one stable observable prediction.",
        ),
        row(
            "canonical_source_amputation_rule_available",
            "pass" if canonical_source_amputation_rule_available else "reject",
            "canonical source-amputation rule available",
            truth(canonical_source_amputation_rule_available),
            "The current frozen-action pack does not choose whether one should keep zero, one, or two vacuum propagator legs in the observable definition.",
        ),
        row(
            "transverse_resolvent_canonical_observable_available",
            "pass" if transverse_resolvent_canonical_observable_available else "reject",
            "transverse resolvent canonical observable available",
            truth(transverse_resolvent_canonical_observable_available),
            "Without a unique amputation / normalization theorem, Delta chi_T does not yet define one canonical alpha read under the current pack.",
        ),
        row(
            "resolvent_family_failed_or_unavailable_under_current_pack",
            "pass" if resolvent_family_failed_or_unavailable_under_current_pack else "reject",
            "resolvent family failed or remains unavailable under current pack",
            truth(resolvent_family_failed_or_unavailable_under_current_pack),
            "The resolvent surface is no longer genuinely untested: under current-pack conventions it either diverges or remains strongly scheme-dependent and non-canonical.",
        ),
        row(
            "resolvent_decision_gate_admissible_now",
            "pass" if resolvent_decision_gate_admissible_now else "reject",
            "resolvent decision gate admissible now",
            truth(resolvent_decision_gate_admissible_now),
            "Once the resolvent surface is audited as non-canonical, the next honest branch is the decision gate / fallback return.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "official_surface_name": "none_due_to_canonical_amputation_gap",
        "q_theory_over_m0": q_theory,
        "q_squared_at_q_theory": q_squared,
        "projected_kernel_numerator_at_q_theory": m_q,
        "projected_kernel_denominator_at_zero": m_zero,
        "official_projected_kernel_alpha_at_q_theory": alpha_t,
        "unamputated_raw_delta_chi_at_q_theory": raw_delta_chi_q,
        "unamputated_q0_pole_present": raw_q0_pole_present,
        "unamputated_form_factor_normalizable": unamputated_form_factor_normalizable,
        "one_leg_amputated_response_at_q_theory": one_leg_response,
        "one_leg_amputated_alpha_at_q_theory": one_leg_alpha,
        "one_leg_amputated_alpha_residual_rel": alpha_residual_rel(one_leg_alpha),
        "one_leg_amputated_supports_scalar_candidate": one_leg_supports_scalar,
        "two_leg_amputated_response_at_q_theory": two_leg_response,
        "two_leg_amputated_alpha_at_q_theory": two_leg_alpha,
        "two_leg_amputated_alpha_residual_rel": alpha_residual_rel(two_leg_alpha),
        "two_leg_amputated_supports_scalar_candidate": two_leg_supports_scalar,
        "static_scaled_proxy_response_at_q_theory": static_scaled_response,
        "static_scaled_proxy_alpha_at_q_theory": static_scaled_alpha,
        "static_scaled_proxy_alpha_residual_rel": alpha_residual_rel(static_scaled_alpha),
        "static_scaled_proxy_supports_scalar_candidate": static_scaled_supports_scalar,
        "all_finite_candidates_fail_scalar": all_finite_candidates_fail_scalar,
        "scheme_alpha_min": spread["min"],
        "scheme_alpha_max": spread["max"],
        "scheme_alpha_span": spread["span"],
        "scheme_alpha_ratio": spread["ratio"],
        "scheme_dependence_large": scheme_dependence_large,
        "canonical_plane_wave_source_candidate_available": (
            canonical_plane_wave_source_candidate_available
        ),
        "canonical_source_amputation_rule_available": (
            canonical_source_amputation_rule_available
        ),
        "transverse_resolvent_canonical_observable_available": (
            transverse_resolvent_canonical_observable_available
        ),
        "resolvent_family_failed_or_unavailable_under_current_pack": (
            resolvent_family_failed_or_unavailable_under_current_pack
        ),
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": True,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "formulas": build_formulae(),
        "hits": {
            "status_current_branch": hit(status_text, "8.7.56.1683"),
            "roadmap_current_branch": hit(roadmap_text, "8.7.56.1683-.1686"),
            "current_problem_resolvent": hit(
                current_problem_text, "transverse resolvent-response observable audit"
            ),
            "current_status_resolvent": hit(
                current_status_text, "transverse resolvent-response observable"
            ),
            "unified_roadmap_resolvent": hit(
                unified_text,
                "`.1683-.1686` は **transverse resolvent-response observable audit**",
            ),
            "part1_metric_surface": hit(part1_text, "g_{\\mu\\nu}(P)"),
            "part5_resolvent": hit(part5_text, "transverse-resolvent response derivation"),
            "expert_note_delta_chi": hit(expert_note_text, "\\Delta\\chi_T[Q]"),
        },
        "prior_summaries": {
            "derivation": deriv_summary,
            "projected_kernel": projected_summary,
            "fallback_closeout": fallback_summary,
        },
        "retained_numeric_state": {
            "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
            "vector_alpha_at_q_theory": VECTOR_ALPHA,
            "bare_alpha": BARE_ALPHA,
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
        },
    }

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload(
                "8.7.56.1683",
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
                "8.7.56.1684",
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
                "8.7.56.1685",
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
                "8.7.56.1686",
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
            {"step": STEP_TAG, "stem": STEM, "artifacts": manifest, "summary": summary},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
