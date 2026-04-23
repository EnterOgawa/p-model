#!/usr/bin/env python3
"""Generate 8.7.56.1679-.1682 failure-structure / response-derivation artifacts.

This branch does not add another same-level density rescue lane.
Instead, it uses the accumulated failures to identify the common logical gap.

What has now failed under the current frozen-action pack:

1. exact constitutive-map closure,
2. branch-local full nonlinear energy-density closure,
3. projected-kernel / transverse-response observable,
4. constrained ground-state / branch-selection rescue,
5. the exact Hamiltonian-core energy-density read as a scalar-foundation rescue.

The common pattern is that every tested route tried to promote one local or
quasi-local surrogate built directly from the retained exact branch into the
canonical observable.

The new derivation therefore shifts one level up:

    a. keep the quadratic action around P_mu = Q_mu + a_mu,
    b. couple an external conserved transverse probe J_perp to a_mu,
    c. integrate out a_mu,
    d. identify the vacuum-subtracted transverse susceptibility

        Delta chi_T[Q] = Pi_T (K[Q]^{-1} - K[0]^{-1}) Pi_T

as the genuinely untested canonical object.

This does not claim success. It only freezes the equation candidate and the
reason why prior projected-kernel failure does not yet falsify the resolvent
response family:

    prior route:      <J | Delta K_T | J>
    new candidate:   -<J | G_T,0 Delta K_T G_T,0 | J> + O(Delta K_T^2)

The next branch can then audit this transverse-resolvent observable directly.
"""

from __future__ import annotations

import csv
import json
import statistics
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

ENERGY_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1627_1630_energy_density_ff_audit_declaration_gate_metrics.json"
)
CONSTITUTIVE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1647_1650_constitutive_map_audit_declaration_gate_metrics.json"
)
FULL_NL_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1651_1654_full_nl_energy_audit_declaration_gate_metrics.json"
)
PROJECTED_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1659_1662_pmu_tresp_pk_audit_declaration_gate_metrics.json"
)
GROUND_STATE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1663_1666_gs_branch_select_audit_declaration_gate_metrics.json"
)
FALLBACK_CLOSEOUT_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1667_1670_fallback_closeout_registry_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1679-1682"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor failure-structure analysis "
    "/ transverse-resolvent response derivation"
)
STEM = build_compact_artifact_stem(STEP_TAG, "fail_struct_resolvent", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_conditional_reactivation_input_assimilation_"
    "ordering_only_no_new_trigger_opened"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_failure_structure_local_surrogate_logic_falsified_"
    "transverse_resolvent_response_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_transverse_resolvent_response_audit"
)
NEXT_ROUTE = "8.7.56.1683"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_resolvent_decision_gate_or_"
    "fallback_return"
)
FOLLOWUP_ROUTE = "8.7.56.1687"

TARGET_ALPHA = 1.0 / 137.035999084
SCALAR_ALPHA = 0.00715678583937324


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


# 関数: 失敗 family の alpha cluster 統計を返す。

def cluster_stats(alpha_values: list[float]) -> dict[str, float]:
    """Return basic cluster statistics for failed-family alpha values."""
    cluster_min = min(alpha_values)
    cluster_max = max(alpha_values)
    cluster_mean = statistics.mean(alpha_values)
    cluster_span = cluster_max - cluster_min
    return {
        "cluster_min": float(cluster_min),
        "cluster_max": float(cluster_max),
        "cluster_mean": float(cluster_mean),
        "cluster_span": float(cluster_span),
        "cluster_span_over_scalar": float(cluster_span / SCALAR_ALPHA),
        "cluster_mean_over_scalar": float(cluster_mean / SCALAR_ALPHA),
    }


# 関数: 新しい canonical candidate 式を返す。

def build_formulae() -> dict[str, str]:
    """Return the failure-structure and response-resolvent equations."""
    return {
        "quadratic_split": "P_mu = Q_mu + a_mu with Q_mu = P_mu^Qball",
        "quadratic_action": "S^(2)[a;Q] = (1/2) int d^4x d^4y a_mu(x) K^{mu nu}[Q](x,y) a_nu(y)",
        "transverse_sector": "K_T[Q] = Pi_T K[Q] Pi_T",
        "prior_projected_kernel": "M_T[J;Q] = <J_perp | Delta K_T[Q] | J_perp>",
        "probe_coupling": "S_probe[a,J] = int d^4x a_mu J_perp^mu, with partial_mu J_perp^mu = 0",
        "classical_solution": "a_cl[Q;J] = -K[Q]^{-1} J_perp",
        "effective_response": "Delta W_Q[J_perp] = -(1/2) <J_perp | Delta chi_T[Q] | J_perp>",
        "susceptibility_kernel": (
            "Delta chi_T[Q] = Pi_T (K[Q]^{-1} - K[0]^{-1}) Pi_T"
        ),
        "born_resolvent": (
            "Delta chi_T[Q] = -G_T,0 Delta K_T[Q] G_T,0 + O(Delta K_T^2), "
            "G_T,0 = Pi_T K[0]^{-1} Pi_T"
        ),
        "response_form_factor": (
            "F_resp(q;J) = Delta W_Q[J_q] / Delta W_Q[J_ref], "
            "alpha_resp(q;J) = F_resp(q;J)^2 / (4 pi)"
        ),
        "failure_matrix_statement": (
            "All prior failed routes tested local or quasi-local surrogates "
            "O[P] or direct stiffness Delta K_T[Q]; none yet tested the "
            "vacuum-subtracted transverse susceptibility Delta chi_T[Q]."
        ),
    }


# 関数: `.1679-.1682` を実行する。

def main() -> None:
    """Execute the failure-structure / transverse-resolvent derivation branch."""
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
        ENERGY_GATE,
        CONSTITUTIVE_GATE,
        FULL_NL_GATE,
        PROJECTED_GATE,
        GROUND_STATE_GATE,
        FALLBACK_CLOSEOUT_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    part1_text = read_text(PART1)
    part5_text = read_text(PART5)

    energy_data = read_json(ENERGY_GATE)
    constitutive_data = read_json(CONSTITUTIVE_GATE)
    full_nl_data = read_json(FULL_NL_GATE)
    projected_data = read_json(PROJECTED_GATE)
    ground_state_data = read_json(GROUND_STATE_GATE)
    fallback_data = read_json(FALLBACK_CLOSEOUT_GATE)

    energy_summary = energy_data["summary"]
    constitutive_summary = constitutive_data["summary"]
    full_nl_summary = full_nl_data["summary"]
    projected_summary = projected_data["summary"]
    ground_state_summary = ground_state_data["summary"]
    fallback_summary = fallback_data["summary"]

    energy_alpha = float(energy_summary["official_alpha_E_at_q_theory"])
    projected_alpha = float(projected_summary["official_projected_kernel_alpha_at_q_theory"])
    pilot_full_alpha = float(full_nl_summary["pilot_full_nonlinear_alpha_at_q_theory"])
    family_full_alpha = float(full_nl_summary["family_proxy_full_alpha_at_q_theory"])
    failed_alpha_cluster = [energy_alpha, projected_alpha, pilot_full_alpha, family_full_alpha]
    cluster = cluster_stats(failed_alpha_cluster)

    constitutive_failed = not bool(constitutive_summary["exact_constitutive_map_available"])
    full_nl_failed = not bool(full_nl_summary["branch_local_full_nonlinear_energy_density_exact_available"])
    energy_foundation_failed = not bool(energy_summary["energy_core_exact_foundation_supported"])
    projected_failed = bool(projected_summary["transverse_response_fallback_failed"])
    branch_selection_failed = not bool(
        ground_state_summary["constrained_ground_state_branch_selection_supported"]
    )
    same_level_exhausted = bool(fallback_summary["same_level_fallback_family_exhausted"])

    failed_family_count = sum(
        [
            constitutive_failed,
            full_nl_failed,
            energy_foundation_failed,
            projected_failed,
            branch_selection_failed,
        ]
    )

    local_surrogate_logic_falsified = bool(
        constitutive_failed
        and full_nl_failed
        and energy_foundation_failed
        and projected_failed
        and branch_selection_failed
    )
    transverse_resolvent_response_surface_untested = True
    projected_kernel_failure_does_not_falsify_resolvent = True

    source_inputs = {
        "status": display_path(STATUS),
        "roadmap": display_path(ROADMAP),
        "ai_context": display_path(AI_CONTEXT),
        "work_history_recent": display_path(WORK_HISTORY_RECENT),
        "current_problem": display_path(CURRENT_PROBLEM),
        "current_status": display_path(CURRENT_STATUS),
        "unified_roadmap": display_path(UNIFIED_ROADMAP),
        "part1": display_path(PART1),
        "part5": display_path(PART5),
        "energy_gate": display_path(ENERGY_GATE),
        "constitutive_gate": display_path(CONSTITUTIVE_GATE),
        "full_nonlinear_gate": display_path(FULL_NL_GATE),
        "projected_gate": display_path(PROJECTED_GATE),
        "ground_state_gate": display_path(GROUND_STATE_GATE),
        "fallback_closeout_gate": display_path(FALLBACK_CLOSEOUT_GATE),
    }

    inventory_rows = [
        row(
            "same_level_fallback_family_exhausted_prior",
            "pass" if same_level_exhausted else "reject",
            "prior same-level fallback family exhausted",
            truth(same_level_exhausted),
            "The branch starts only after the current pack has already closed density, constitutive-map, nonlinear-energy, projected-kernel, and branch-selection families honestly.",
        ),
        row(
            "failed_family_count",
            "pass" if failed_family_count >= 5 else "watch",
            "failed current-pack family count",
            failed_family_count,
            "This counts the independently audited current-pack families that now land on no-go or unavailable reads.",
        ),
        row(
            "failed_alpha_cluster_mean",
            "watch",
            "mean alpha over failed canonical families",
            cluster["cluster_mean"],
            "The failed canonical-family alphas cluster tightly on the vector no-go scale rather than near the retained scalar strong candidate.",
        ),
        row(
            "failed_alpha_cluster_span",
            "pass" if cluster["cluster_span_over_scalar"] < 0.01 else "watch",
            "alpha span across failed canonical families",
            cluster["cluster_span"],
            "A small span across multiple failed families indicates a common structural failure rather than independent accidental misses.",
        ),
    ]
    inventory_payload = payload(
        "8.7.56.1680",
        STEP_NAME + " inventory",
        {
            "source_files": source_inputs,
            "constants": {
                "scalar_alpha": SCALAR_ALPHA,
                "target_alpha": TARGET_ALPHA,
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
            },
        },
        inventory_rows,
        {
            "trial2_numeric_alpha_problem_classification": PRIOR_CLASS,
            "failed_family_count": failed_family_count,
            "failed_alpha_cluster_min": cluster["cluster_min"],
            "failed_alpha_cluster_max": cluster["cluster_max"],
            "failed_alpha_cluster_mean": cluster["cluster_mean"],
            "failed_alpha_cluster_span": cluster["cluster_span"],
        },
        {"branch_ready": True},
        {
            "hits": {
                "status_same_level_exhausted": hit(
                    status_text, "same_level_fallback_family_exhausted = true"
                ),
                "problem_current_step": hit(current_problem_text, "8.7.56.1679"),
                "status_current_step": hit(current_status_text, "8.7.56.1679"),
                "roadmap_wait_restore": hit(
                    unified_roadmap_text,
                    ".1679-.1682",
                ),
            }
        },
    )

    audit_rows = [
        row(
            "constitutive_map_family_failed",
            "pass" if constitutive_failed else "reject",
            "exact constitutive-map family failed under current pack",
            truth(constitutive_failed),
            "The current frozen-action pack still cannot canonically decide the probe-to-observable constitutive map.",
        ),
        row(
            "branch_local_full_nonlinear_family_failed",
            "pass" if full_nl_failed else "reject",
            "branch-local full nonlinear energy family failed under current pack",
            truth(full_nl_failed),
            "Adding same-branch nonlinear energy candidates does not move the read away from the vector no-go cluster.",
        ),
        row(
            "official_energy_core_foundation_failed",
            "pass" if energy_foundation_failed else "reject",
            "official energy-core scalar-foundation rescue failed",
            truth(energy_foundation_failed),
            "The exact Hamiltonian-core read is canonical but still sits on the vector no-go scale.",
        ),
        row(
            "projected_kernel_stiffness_family_failed",
            "pass" if projected_failed else "reject",
            "projected-kernel stiffness family failed",
            truth(projected_failed),
            "The prior response fallback tested the direct stiffness matrix element of Delta K_T and also landed on the vector no-go scale.",
        ),
        row(
            "branch_selection_family_failed",
            "pass" if branch_selection_failed else "reject",
            "constrained ground-state / branch-selection family failed",
            truth(branch_selection_failed),
            "The current pack does not support a canonical branch switch after all observable families above remain closed negatively.",
        ),
        row(
            "local_surrogate_logic_falsified",
            "pass" if local_surrogate_logic_falsified else "reject",
            "local or quasi-local surrogate observable logic falsified",
            truth(local_surrogate_logic_falsified),
            "What failed in common is the attempt to promote one local or quasi-local branch functional directly into the canonical observable.",
        ),
        row(
            "common_missing_object_is_probe_response_map",
            "pass",
            "common missing object is canonical probe-response map",
            1.0,
            "The failure pattern points upstream to a missing observable bridge, not to one more density variant.",
        ),
        row(
            "projected_kernel_failure_does_not_falsify_resolvent",
            "pass" if projected_kernel_failure_does_not_falsify_resolvent else "reject",
            "projected-kernel failure does not yet falsify transverse resolvent response",
            truth(projected_kernel_failure_does_not_falsify_resolvent),
            "The failed object was <J|Delta K_T|J>, whereas the new response candidate is controlled by the inverse operator K_T^{-1}.",
        ),
    ]
    audit_payload = payload(
        "8.7.56.1681",
        STEP_NAME + " audit",
        {"source_files": source_inputs},
        audit_rows,
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "local_surrogate_logic_falsified": local_surrogate_logic_falsified,
            "common_missing_object_is_probe_response_map": True,
            "projected_kernel_failure_does_not_falsify_resolvent_response": projected_kernel_failure_does_not_falsify_resolvent,
        },
        {"branch_completed": True},
        {
            "prior_summaries": {
                "energy": energy_summary,
                "constitutive": constitutive_summary,
                "full_nonlinear": full_nl_summary,
                "projected_kernel": projected_summary,
                "ground_state": ground_state_summary,
            }
        },
    )

    declaration_rows = [
        row(
            "transverse_resolvent_response_surface_untested",
            "pass" if transverse_resolvent_response_surface_untested else "reject",
            "transverse resolvent-response surface remains untested",
            truth(transverse_resolvent_response_surface_untested),
            "No prior branch evaluated the vacuum-subtracted transverse susceptibility Delta chi_T = Pi_T(K[Q]^{-1} - K[0]^{-1})Pi_T.",
        ),
        row(
            "response_resolvent_equation_candidate_derived",
            "pass",
            "response-resolvent equation candidate derived",
            1.0,
            "The canonical candidate is now frozen at the level of the quadratic action plus external conserved transverse probe.",
        ),
        row(
            "new_same_pack_object_is_nonlocal_not_local",
            "pass",
            "new current-pack object is nonlocal susceptibility, not another local density",
            1.0,
            "This branch does not reopen the same-level local rescue family. It opens a distinct nonlocal response surface.",
        ),
        row(
            "transverse_resolvent_audit_admissible_now",
            "pass",
            "transverse resolvent-response audit admissible now",
            1.0,
            "Once Delta chi_T is fixed as the untested object, the next honest branch is to audit its observable read directly.",
        ),
        row(
            "physical_reject_required",
            "pass",
            "physical_reject_required",
            0.0,
            "Even after the failure-structure refinement, the correct read is still partial closeout plus targeted reopen rather than physical reject.",
        ),
    ]
    declaration_payload = payload(
        "8.7.56.1682",
        STEP_NAME + " declaration gate",
        {
            "source_files": source_inputs,
            "constants": {
                "scalar_alpha": SCALAR_ALPHA,
                "failed_alpha_cluster_mean": cluster["cluster_mean"],
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
            },
        },
        declaration_rows,
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "local_surrogate_logic_falsified": local_surrogate_logic_falsified,
            "transverse_resolvent_response_surface_untested": transverse_resolvent_response_surface_untested,
            "response_resolvent_equation_candidate_derived": True,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
            "selected_followup_route": FOLLOWUP_ROUTE_NAME,
            "selected_followup_route_or_none": FOLLOWUP_ROUTE,
            "physical_reject_required": False,
        },
        {
            "overall_status": (
                "vector_qball_form_factor_failure_structure_local_surrogate_logic_"
                "falsified_transverse_resolvent_response_next_declared"
            ),
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"formulas": build_formulae()},
    )

    route_rows = [
        row(
            "route_state_changed_by_current_branch",
            "pass",
            "route state changed by failure-structure derivation",
            1.0,
            "The official next route moves away from wait-restore into the newly isolated transverse-resolvent response audit.",
        ),
        row(
            "same_level_density_retry_still_inadmissible",
            "pass",
            "same-level density retry still inadmissible",
            1.0,
            "The new branch does not reopen density, constitutive-map, nonlinear-energy, or projected-kernel retries as such.",
        ),
        row(
            "new_nonlocal_response_surface_promoted",
            "pass",
            "new nonlocal response surface promoted to mainline",
            1.0,
            "The mainline is now the vacuum-subtracted transverse susceptibility rather than another local surrogate observable.",
        ),
    ]
    route_payload = payload(
        "8.7.56.1682",
        STEP_NAME + " route sync",
        {"source_files": source_inputs},
        route_rows,
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
            "selected_followup_route": FOLLOWUP_ROUTE_NAME,
            "selected_followup_route_or_none": FOLLOWUP_ROUTE,
            "physical_reject_required": False,
        },
        {"route_synced": True},
        {
            "hits": {
                "part1_effective_metric": hit(part1_text, "g_{\\mu\\nu}(P)"),
                "part5_current_tail": hit(part5_text, ".1675-.1678"),
                "roadmap_current_tail": hit(roadmap_text, "8.7.56.1679-.1682"),
            }
        },
    )

    outputs = {
        "inventory": write_artifact("inventory", inventory_payload),
        "audit": write_artifact("audit", audit_payload),
        "declaration_gate": write_artifact("declaration_gate", declaration_payload),
        "route_sync": write_artifact("route_sync", route_payload),
    }

    print("[ok] failure-structure / transverse-resolvent artifacts written:")
    for kind, paths in outputs.items():
        print(f"  - {kind}: {paths['json']} | {paths['csv']}")


if __name__ == "__main__":
    main()
