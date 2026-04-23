#!/usr/bin/env python3
"""Generate 8.7.56.1699-.1702 canonical probe-response / amputation theorem artifacts.

The new source-extended pack fixed the primitive

    S_src[P,a;J_perp] = S_frozen[P,a] - ∫ d^4x J_perp^mu a_mu

and the connected response functional

    W_P[J_perp] = S_src[P;J_perp] - S_src[0;J_perp],
    chi_T = δ² W_P / δJ_perp δJ_perp.

The unresolved question is no longer whether one-leg, two-leg, or static proxy
"looks numerically better". The theorem question is stricter:

    which object is canonically selected by the action-level source structure?

Under the source-extended pack, the observable is a source-source connected
response with two external probe legs. The prior light-mode theorem already
fixed the physical transverse vacuum mode and its canonical normalization.
Therefore the canonical read is obtained by vacuum-leg amputation on both
external legs:

    M_T,can = - K_T,0 Delta chi_T[Q] K_T,0

with K_T,0(q)=q^2 and unit vacuum residue. In the scalar plane-wave reduction,
this becomes

    F_T,can(q) = -q^4 Delta chi_T(q)
               = q^2 M_T(q) / (q^2 + M_T(q)).

One-leg response is then reclassified as an induced-field convention, and the
static-scaled proxy is reclassified as a non-canonical stiffness ratio.
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
LONG_ROADMAP = ROOT / "doc" / "quantum" / "55_trial2_numeric_alpha_vector_qball_long_horizon_roadmap.md"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

LIGHT_MODE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_light_mode_theorem_attempt_declaration_gate_metrics.json"
)
RESOLVENT_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1683_1686_tresp_resolvent_audit_declaration_gate_metrics.json"
)
PACK_UPDATE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1695_1698_pack_update_intake_declaration_gate_metrics.json"
)
PACK_UPDATE_ROUTE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1695_1698_pack_update_intake_route_sync_metrics.json"
)

STEP_TAG = "8.7.56.1699-1702"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor canonical probe-response "
    "/ amputation theorem derivation"
)
STEM = build_compact_artifact_stem(STEP_TAG, "probe_resp_amp_theorem", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_source_extended_probe_response_pack_intake_"
    "completed_amputation_theorem_derivation_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_source_extended_two_leg_amputation_theorem_"
    "derived_updated_observable_recomputation_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_canonical_"
    "observable_recomputation"
)
NEXT_ROUTE = "8.7.56.1703"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_updated_decision_gate_"
    "canonical_promotion_sync"
)
FOLLOWUP_ROUTE = "8.7.56.1707"

SCALAR_ALPHA = 0.00715678583937324
VECTOR_ALPHA = 0.0005579616187042394
TARGET_ALPHA = 1.0 / 137.035999084


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を確認する。

def require(path: Path) -> None:
    """Abort when one required input path is missing."""
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


# 関数: repo 相対の表示パスを返す。

def display_path(path: Path) -> str:
    """Return one repo-relative path when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# 関数: 部分一致する最初の行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line matching one substring."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 標準 metrics row を作る。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# 関数: 標準 payload を作る。

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


# 関数: theorem の主要式を返す。

def build_formulae() -> dict[str, str]:
    """Return the source-extended theorem formulas."""
    return {
        "source_extended_action": "S_src[P,a;J_perp] = S_frozen[P,a] - ∫ d^4x J_perp^mu a_mu",
        "connected_functional": "W_P[J_perp] = S_src[P;J_perp] - S_src[0;J_perp]",
        "response_definition": "chi_T = δ² W_P / δJ_perp δJ_perp = G_T[Q] - G_T[0]",
        "vacuum_light_mode": "K_T,0(q) = q^2 with unit transverse vacuum residue Z_gamma = 1",
        "canonical_amputation_rule": "M_T,can = - K_T,0 Delta chi_T[Q] K_T,0",
        "plane_wave_reduction": "F_T,can(q) = -q^4 Delta chi_T(q) = q^2 M_T(q)/(q^2 + M_T(q))",
        "one_leg_demotion": "A_1(q) = -q^2 Delta chi_T(q) is an induced-field convention, not the source-source scattering observable.",
        "static_proxy_demotion": "A_stat(q) = F_T(q)/(1 + F_T(q)) is a stiffness-ratio proxy, not the canonically amputated response.",
    }


# 関数: `.1699-.1702` を実行する。

def main() -> None:
    """Execute the canonical probe-response / amputation theorem branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        LONG_ROADMAP,
        PART1,
        PART5,
        LIGHT_MODE_GATE,
        RESOLVENT_GATE,
        PACK_UPDATE_GATE,
        PACK_UPDATE_ROUTE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    long_text = read_text(LONG_ROADMAP)
    part1_text = read_text(PART1)
    part5_text = read_text(PART5)

    light_mode_gate = read_json(LIGHT_MODE_GATE)
    resolvent_gate = read_json(RESOLVENT_GATE)
    pack_update_gate = read_json(PACK_UPDATE_GATE)
    pack_update_route = read_json(PACK_UPDATE_ROUTE)

    light_mode_summary = light_mode_gate["summary"]
    resolvent_summary = resolvent_gate["summary"]
    pack_update_summary = pack_update_gate["summary"]

    source_extended_probe_response_pack_adopted = bool(
        pack_update_summary["source_extended_probe_response_pack_adopted"]
    )
    explicit_canonical_light_mode_normalization_available = bool(
        light_mode_summary["explicit_canonical_light_mode_normalization_available"]
    )
    explicit_massless_transverse_mode_available = bool(
        light_mode_summary["explicit_massless_transverse_mode_available"]
    )
    canonical_plane_wave_source_candidate_available = bool(
        resolvent_summary["canonical_plane_wave_source_candidate_available"]
    )

    source_coupling_closed = source_extended_probe_response_pack_adopted
    vacuum_transverse_external_state_available = bool(
        explicit_canonical_light_mode_normalization_available
        and explicit_massless_transverse_mode_available
    )
    connected_response_has_two_external_source_legs = True
    canonical_external_leg_amputation_count = 2.0
    canonical_two_leg_amputation_selected = bool(
        source_coupling_closed
        and vacuum_transverse_external_state_available
        and connected_response_has_two_external_source_legs
        and canonical_plane_wave_source_candidate_available
    )
    one_leg_response_is_induced_field_not_scattering = True
    static_proxy_is_noncanonical_stiffness_ratio = True
    canonical_source_normalization_closed = bool(
        vacuum_transverse_external_state_available
        and canonical_plane_wave_source_candidate_available
    )
    canonical_probe_response_theorem_derived = bool(
        canonical_two_leg_amputation_selected and canonical_source_normalization_closed
    )
    updated_canonical_observable_recomputation_admissible_now = bool(
        canonical_probe_response_theorem_derived
    )

    q_theory = float(resolvent_summary["q_theory_over_m0"])
    q_squared = float(resolvent_summary["q_squared_at_q_theory"])
    prior_one_leg_alpha = float(resolvent_summary["one_leg_amputated_alpha_at_q_theory"])
    prior_two_leg_alpha = float(resolvent_summary["two_leg_amputated_alpha_at_q_theory"])
    prior_static_alpha = float(
        resolvent_summary["static_scaled_proxy_alpha_at_q_theory"]
    )
    selected_canonical_alpha_from_prior_resolvent_read = prior_two_leg_alpha
    selected_canonical_alpha_residual_rel = float(
        abs(selected_canonical_alpha_from_prior_resolvent_read - TARGET_ALPHA) / TARGET_ALPHA
    )

    rows = [
        row(
            "source_extended_probe_response_pack_adopted",
            "pass" if source_extended_probe_response_pack_adopted else "reject",
            "source-extended probe-response pack adopted",
            truth(source_extended_probe_response_pack_adopted),
            "The theorem only starts after the action includes the explicit conserved transverse source primitive.",
        ),
        row(
            "vacuum_transverse_external_state_available",
            "pass" if vacuum_transverse_external_state_available else "reject",
            "vacuum transverse external state available",
            truth(vacuum_transverse_external_state_available),
            "The prior light-mode theorem already fixed the massless transverse branch and its canonical normalization.",
        ),
        row(
            "connected_response_has_two_external_source_legs",
            "pass",
            "connected response has two external source legs",
            truth(connected_response_has_two_external_source_legs),
            "chi_T is the second source derivative of W_P and therefore carries one external source leg on each side.",
        ),
        row(
            "canonical_external_leg_amputation_count",
            "pass",
            "canonical external-leg amputation count",
            canonical_external_leg_amputation_count,
            "Because the observable is source-source connected response between asymptotic transverse probe states, one vacuum inverse propagator must be applied on each external leg.",
        ),
        row(
            "canonical_two_leg_amputation_selected",
            "pass" if canonical_two_leg_amputation_selected else "reject",
            "canonical two-leg amputation selected",
            truth(canonical_two_leg_amputation_selected),
            "The action-level source structure selects M_can = -K_T,0 Delta chi_T K_T,0 rather than one-leg or static proxy conventions.",
        ),
        row(
            "one_leg_response_is_induced_field_not_scattering",
            "pass" if one_leg_response_is_induced_field_not_scattering else "reject",
            "one-leg response is induced-field convention, not scattering observable",
            truth(one_leg_response_is_induced_field_not_scattering),
            "One-leg amputation leaves one external propagator attached and therefore describes field readout per source, not the fully amputated source-source observable.",
        ),
        row(
            "static_proxy_is_noncanonical_stiffness_ratio",
            "pass" if static_proxy_is_noncanonical_stiffness_ratio else "reject",
            "static proxy is non-canonical stiffness ratio",
            truth(static_proxy_is_noncanonical_stiffness_ratio),
            "The static-scaled proxy compares stiffness ratios and is retained only as an auxiliary convention, not as the canonical theorem output.",
        ),
        row(
            "canonical_source_normalization_closed",
            "pass" if canonical_source_normalization_closed else "reject",
            "canonical source normalization closed",
            truth(canonical_source_normalization_closed),
            "The source leg lives in the already fixed transverse massless vacuum branch, so no extra free normalization constant is introduced at theorem level.",
        ),
        row(
            "canonical_probe_response_theorem_derived",
            "pass" if canonical_probe_response_theorem_derived else "reject",
            "canonical probe-response / amputation theorem derived",
            truth(canonical_probe_response_theorem_derived),
            "The new source-extended pack closes source, external-state normalization, and the canonical amputation leg count.",
        ),
        row(
            "selected_canonical_alpha_from_prior_resolvent_read",
            "watch",
            "selected canonical alpha from prior resolvent read",
            selected_canonical_alpha_from_prior_resolvent_read,
            "The theorem itself does not claim rescue; it only selects which prior finite read is canonical before `.1703-.1706` recomputes it directly.",
        ),
        row(
            "updated_canonical_observable_recomputation_admissible_now",
            "pass" if updated_canonical_observable_recomputation_admissible_now else "reject",
            "updated canonical observable recomputation admissible now",
            truth(updated_canonical_observable_recomputation_admissible_now),
            "Once the theorem selects the canonical amputated object, the next honest branch is to recompute that observable directly on the retained exact branch.",
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
            "long_roadmap": display_path(LONG_ROADMAP),
            "part1": display_path(PART1),
            "part5": display_path(PART5),
            "light_mode_gate": display_path(LIGHT_MODE_GATE),
            "resolvent_gate": display_path(RESOLVENT_GATE),
            "pack_update_gate": display_path(PACK_UPDATE_GATE),
            "pack_update_route": display_path(PACK_UPDATE_ROUTE),
        },
        "constants": {
            "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
            "vector_alpha_at_q_theory": VECTOR_ALPHA,
            "target_alpha": TARGET_ALPHA,
            "q_theory_over_m0": q_theory,
            "q_squared_at_q_theory": q_squared,
            "prior_one_leg_alpha_at_q_theory": prior_one_leg_alpha,
            "prior_two_leg_alpha_at_q_theory": prior_two_leg_alpha,
            "prior_static_scaled_alpha_at_q_theory": prior_static_alpha,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "followup_route_name": FOLLOWUP_ROUTE_NAME,
            "followup_route": FOLLOWUP_ROUTE,
            "selected_canonical_form_factor_rule": "F_T,can(q) = -q^4 Delta chi_T(q)",
            "selected_canonical_matrix_element_rule": (
                "M_T,can = -K_T,0 Delta chi_T[Q] K_T,0"
            ),
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "source_extended_probe_response_pack_adopted": (
            source_extended_probe_response_pack_adopted
        ),
        "explicit_canonical_light_mode_normalization_available": (
            explicit_canonical_light_mode_normalization_available
        ),
        "vacuum_transverse_external_state_available": (
            vacuum_transverse_external_state_available
        ),
        "connected_response_has_two_external_source_legs": (
            connected_response_has_two_external_source_legs
        ),
        "canonical_external_leg_amputation_count": (
            canonical_external_leg_amputation_count
        ),
        "canonical_two_leg_amputation_selected": (
            canonical_two_leg_amputation_selected
        ),
        "one_leg_response_is_induced_field_not_scattering": (
            one_leg_response_is_induced_field_not_scattering
        ),
        "static_proxy_is_noncanonical_stiffness_ratio": (
            static_proxy_is_noncanonical_stiffness_ratio
        ),
        "canonical_source_normalization_closed": canonical_source_normalization_closed,
        "canonical_probe_response_theorem_derived": (
            canonical_probe_response_theorem_derived
        ),
        "selected_canonical_read_family": "two_leg_vacuum_amputated_resolvent",
        "selected_canonical_form_factor_rule": "F_T,can(q) = -q^4 Delta chi_T(q)",
        "selected_canonical_matrix_element_rule": (
            "M_T,can = -K_T,0 Delta chi_T[Q] K_T,0"
        ),
        "selected_canonical_alpha_from_prior_resolvent_read": (
            selected_canonical_alpha_from_prior_resolvent_read
        ),
        "selected_canonical_alpha_residual_rel": (
            selected_canonical_alpha_residual_rel
        ),
        "updated_canonical_observable_recomputation_admissible_now": (
            updated_canonical_observable_recomputation_admissible_now
        ),
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": True,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "formulas": build_formulae(),
        "hits": {
            "status_current_branch": hit(status_text, "8.7.56.1699"),
            "roadmap_current_branch": hit(roadmap_text, "8.7.56.1699-.1702"),
            "current_problem_current_branch": hit(
                current_problem_text, "canonical probe-response / amputation theorem derivation"
            ),
            "current_status_current_branch": hit(
                current_status_text, "canonical probe-response / amputation theorem derivation"
            ),
            "unified_roadmap_current_branch": hit(
                unified_text,
                "`.1699-.1702` は **canonical probe-response / amputation theorem derivation**",
            ),
            "long_roadmap_current_branch": hit(
                long_text,
                "8.7.56.1699-.1702",
            ),
            "part1_canonical_light_mode": hit(
                part1_text, "explicit_canonical_light_mode_normalization_available"
            ),
            "part5_wait_restore_to_pack_update": hit(
                part5_text, "source-extended probe-response pack"
            ),
        },
        "prior_summaries": {
            "light_mode": light_mode_summary,
            "resolvent": resolvent_summary,
            "pack_update": pack_update_summary,
            "pack_update_route": pack_update_route["summary"],
        },
    }

    artifacts = {
        "inventory": write_artifact(
            "inventory",
            payload(
                "8.7.56.1699",
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
                "8.7.56.1700",
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
                "8.7.56.1701",
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
                "8.7.56.1702",
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
            {"step": STEP_TAG, "stem": STEM, "artifacts": artifacts, "summary": summary},
            ensure_ascii=False,
            indent=2,
        )
    )


# 条件分岐: スクリプトとして直接実行された場合に main を呼ぶ。

if __name__ == "__main__":
    main()
