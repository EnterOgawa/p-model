#!/usr/bin/env python3
"""Generate 8.7.56.1667-.1670 fallback closeout / reopen registry artifacts.

The current frozen-action pack has now exhausted every same-level fallback
family that was still admissible after the energy-density breakthrough pack
closed as Gate B retain-but-not-promote:

1. density / constitutive-map,
2. branch-local nonlinear energy density,
3. `P_mu` transverse response / projected-kernel observable,
4. constrained ground-state / branch-selection.

None of these lanes canonically promotes the retained scalar strong candidate.
At the same time, the broader scalar-side candidate itself still survives and
`physical_reject_required` remains false.

The honest task of `.1667-.1670` is therefore not another rescue derivation.
It is to freeze the current-pack fallback closeout and its reopen ordering in a
machine-readable form:

- same-level rescue extension is no longer admissible,
- genuinely new action-level structure becomes the primary reopen surface,
- future external input is retained as a reserve side lane,
- scalar-leaning noncanonical evidence remains retained but unpromoted.
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
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
LOCAL_RESPONSE = ROOT / "doc" / "quantum" / "51_trial2_numeric_alpha_vector_qball_branch_local_nonlinear_response.md"

PRIMARY_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1655_1658_primary_decision_gate_declaration_gate_metrics.json"
)
PROJECTED_KERNEL_GATE = (
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
ENERGY_REGISTRY_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1639_1642_energy_density_reopen_registry_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1667-1670"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor fallback closeout / "
    "reopen registry"
)
STEM = build_compact_artifact_stem(STEP_TAG, "fallback_closeout_registry", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_constrained_ground_state_branch_selection_"
    "not_supported_fallback_closeout_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_current_pack_fallback_family_closeout_"
    "reopen_registry_completed"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_fallback_closeout_"
    "advice_pack_refresh"
)
NEXT_ROUTE = "8.7.56.1671"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_new_action_"
    "level_structure_or_external_input_reactivation"
)
FOLLOWUP_ROUTE = "8.7.56.1675"

PRIMARY_REOPEN = "genuinely_new_action_level_structure_beyond_current_frozen_action_pack"
SECONDARY_REOPEN = (
    "exact_constitutive_map_or_branch_local_full_nonlinear_energy_density_"
    "reopen_after_pack_update"
)
RESERVE_REOPEN = "future_external_input_or_expert_input_guiding_new_primary_surface"

SCALAR_ALPHA = 0.00715678583937324
VECTOR_ALPHA = 0.0005579616187042394
ENERGY_CORE_ALPHA = 0.0005422361373947313


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


# 関数: `.1667-.1670` を実行する。

def main() -> None:
    """Execute the fallback closeout / reopen registry branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        PART5,
        LOCAL_RESPONSE,
        PRIMARY_GATE,
        PROJECTED_KERNEL_GATE,
        GROUND_STATE_GATE,
        ENERGY_REGISTRY_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    part5_text = read_text(PART5)
    local_response_text = read_text(LOCAL_RESPONSE)

    primary_summary = read_json(PRIMARY_GATE)["summary"]
    projected_summary = read_json(PROJECTED_KERNEL_GATE)["summary"]
    ground_state_summary = read_json(GROUND_STATE_GATE)["summary"]
    energy_registry_summary = read_json(ENERGY_REGISTRY_GATE)["summary"]

    inventory_ready = all(
        item is not None
        for item in (
            hit(status_text, "fallback closeout / reopen registry"),
            hit(roadmap_text, "8.7.56.1667-.1670"),
            hit(current_problem_text, "fallback closeout / reopen registry"),
            hit(current_status_text, "fallback closeout / reopen registry"),
            hit(
                unified_text,
                "`.1667-.1670` は **fallback closeout / reopen registry**",
            ),
            hit(part5_text, "next mainline は `.1667-.1670` **fallback closeout / reopen registry**"),
            hit(local_response_text, "retained vector no-go scale に張り付いた。"),
        )
    )

    gate_b_retained = bool(
        primary_summary.get("trial2_numeric_alpha_problem_classification")
        == "vector_qball_form_factor_primary_gate_b_retain_not_promote_transverse_response_fallback_next"
        and primary_summary.get("gate_b_retain_not_promote_selected", False)
        and primary_summary.get("primary_breakthrough_pack_failed", False)
    )
    projected_kernel_failed = bool(
        projected_summary.get("trial2_numeric_alpha_problem_classification")
        == "vector_qball_form_factor_p_mu_transverse_response_projected_kernel_tracks_vector_no_go_ground_state_fallback_next"
        and projected_summary.get("transverse_response_fallback_failed", False)
    )
    branch_selection_failed = bool(
        ground_state_summary.get("trial2_numeric_alpha_problem_classification") == PRIOR_CLASS
        and ground_state_summary.get("fallback_family_exhausted", False)
        and not ground_state_summary.get(
            "constrained_ground_state_branch_selection_supported", True
        )
    )
    same_level_fallback_family_exhausted = bool(
        gate_b_retained and projected_kernel_failed and branch_selection_failed
    )
    same_level_rescue_lane_extension_admissible = False

    noncanonical_scalar_leaning_evidence_retained = bool(
        primary_summary.get("secondary_evidence_scalar_leaning", False)
        and not primary_summary.get("secondary_canonical_promotion_supported", True)
        and energy_registry_summary.get(
            "evidence_only_improvement_surfaces_retained", False
        )
    )
    physical_reject_not_selected = bool(
        not primary_summary.get("physical_reject_required", True)
        and not projected_summary.get("physical_reject_required", True)
        and not ground_state_summary.get("physical_reject_required", True)
        and not energy_registry_summary.get("physical_reject_required", True)
    )

    registry_wording_honest = bool(
        inventory_ready
        and same_level_fallback_family_exhausted
        and not same_level_rescue_lane_extension_admissible
        and noncanonical_scalar_leaning_evidence_retained
        and physical_reject_not_selected
    )
    registry_ready = bool(registry_wording_honest)

    rows = [
        row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "fallback closeout inventory ready",
            truth(inventory_ready),
            "Closeout starts only after status, roadmap, current notes, unified roadmap, Part V, and the latest local response note all point to the same `.1667-.1670` branch.",
        ),
        row(
            "gate_b_retained",
            "pass" if gate_b_retained else "reject",
            "Gate B retain-not-promote carried over",
            truth(gate_b_retained),
            "The first-shot breakthrough pack must remain closed as Gate B retain-but-not-promote before the global fallback closeout can be honest.",
        ),
        row(
            "projected_kernel_failed",
            "pass" if projected_kernel_failed else "reject",
            "projected-kernel fallback failed",
            truth(projected_kernel_failed),
            "The projected-kernel fallback remains fixed on the retained vector no-go scale and therefore cannot reopen the pack internally.",
        ),
        row(
            "branch_selection_failed",
            "pass" if branch_selection_failed else "reject",
            "branch-selection fallback failed",
            truth(branch_selection_failed),
            "The constrained ground-state / branch-selection fallback also closes negatively and therefore completes the same-level fallback family.",
        ),
        row(
            "same_level_fallback_family_exhausted",
            "pass" if same_level_fallback_family_exhausted else "reject",
            "same-level fallback family exhausted",
            truth(same_level_fallback_family_exhausted),
            "Density, constitutive-map, nonlinear-energy, projected-kernel, and branch-selection families have now all been tested honestly under the current frozen-action pack.",
        ),
        row(
            "same_level_rescue_lane_extension_admissible",
            "reject",
            "same-level rescue lane extension admissible",
            truth(same_level_rescue_lane_extension_admissible),
            "Once the family is exhausted, inventing another same-level rescue variant is no longer honest under the current pack.",
        ),
        row(
            "noncanonical_scalar_leaning_evidence_retained",
            "pass" if noncanonical_scalar_leaning_evidence_retained else "reject",
            "noncanonical scalar-leaning evidence retained",
            truth(noncanonical_scalar_leaning_evidence_retained),
            "Electric-like and note-gradient surfaces remain retained as evidence-only hints, but they do not reopen the current pack canonically.",
        ),
        row(
            "primary_reopen_surface_fixed",
            "pass",
            "primary reopen surface fixed",
            1.0,
            "The primary reopen surface is genuinely new action-level structure beyond the current frozen-action pack.",
        ),
        row(
            "secondary_reopen_surface_fixed",
            "pass",
            "secondary reopen surface fixed",
            1.0,
            "The first concrete computation to revisit after a pack update remains exact constitutive-map / branch-local full nonlinear energy density reopening.",
        ),
        row(
            "reserve_reopen_surface_fixed",
            "pass",
            "reserve reopen surface fixed",
            1.0,
            "Future external input is retained only as a reserve side lane that may help open the new primary surface.",
        ),
        row(
            "physical_reject_not_selected",
            "pass" if physical_reject_not_selected else "reject",
            "physical reject not selected",
            truth(physical_reject_not_selected),
            "The current closeout is route-local to the frozen-action pack and does not force physical rejection of the retained scalar strong candidate.",
        ),
        row(
            "registry_wording_honest",
            "pass" if registry_wording_honest else "reject",
            "fallback closeout registry wording honest",
            truth(registry_wording_honest),
            "The registry is honest only if the full failed fallback family remains visible, same-level extension is prohibited, and the reopen surfaces are narrowed explicitly.",
        ),
        row(
            "registry_ready",
            "pass" if registry_ready else "reject",
            "fallback closeout registry ready",
            truth(registry_ready),
            "Once the same-level family is exhausted and the reopen ordering is explicit, the current-pack fallback closeout can be frozen machine-readably.",
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
            "part5": display_path(PART5),
            "local_response": display_path(LOCAL_RESPONSE),
            "primary_gate": display_path(PRIMARY_GATE),
            "projected_kernel_gate": display_path(PROJECTED_KERNEL_GATE),
            "ground_state_gate": display_path(GROUND_STATE_GATE),
            "energy_registry_gate": display_path(ENERGY_REGISTRY_GATE),
        },
        "constants": {
            "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
            "vector_alpha_at_q_theory": VECTOR_ALPHA,
            "energy_core_alpha_at_q_theory": ENERGY_CORE_ALPHA,
            "primary_reopen_surface": PRIMARY_REOPEN,
            "secondary_reopen_surface": SECONDARY_REOPEN,
            "reserve_reopen_surface": RESERVE_REOPEN,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "followup_route_name": FOLLOWUP_ROUTE_NAME,
            "followup_route": FOLLOWUP_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "same_level_fallback_family_exhausted": same_level_fallback_family_exhausted,
        "same_level_rescue_lane_extension_admissible": same_level_rescue_lane_extension_admissible,
        "exact_constitutive_map_available": primary_summary[
            "exact_constitutive_map_available"
        ],
        "branch_local_full_nonlinear_energy_density_exact_available": primary_summary[
            "branch_local_full_nonlinear_energy_density_exact_available"
        ],
        "projected_kernel_fallback_failed": projected_kernel_failed,
        "constrained_ground_state_branch_selection_supported": ground_state_summary[
            "constrained_ground_state_branch_selection_supported"
        ],
        "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
        "official_energy_core_alpha_at_q_theory": ENERGY_CORE_ALPHA,
        "official_projected_kernel_alpha_at_q_theory": projected_summary[
            "official_projected_kernel_alpha_at_q_theory"
        ],
        "electric_like_component_alpha_at_q_theory": primary_summary[
            "electric_like_component_alpha_at_q_theory"
        ],
        "note_gradient_alpha_at_q_theory": primary_summary[
            "note_gradient_alpha_at_q_theory"
        ],
        "noncanonical_scalar_leaning_evidence_retained": noncanonical_scalar_leaning_evidence_retained,
        "primary_reopen_surface": PRIMARY_REOPEN,
        "secondary_reopen_surface": SECONDARY_REOPEN,
        "reserve_reopen_surface": RESERVE_REOPEN,
        "fallback_closeout_reopen_registry_wording_honest": registry_wording_honest,
        "fallback_closeout_reopen_registry_ready": registry_ready,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": registry_ready,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "hits": {
            "status_branch_hit": hit(status_text, "fallback closeout / reopen registry"),
            "roadmap_branch_hit": hit(roadmap_text, "8.7.56.1667-.1670"),
            "current_problem_branch_hit": hit(
                current_problem_text, "fallback closeout / reopen registry"
            ),
            "current_status_branch_hit": hit(
                current_status_text, "fallback closeout / reopen registry"
            ),
            "unified_roadmap_branch_hit": hit(
                unified_text,
                "`.1667-.1670` は **fallback closeout / reopen registry**",
            ),
            "part5_branch_hit": hit(
                part5_text, "next mainline は `.1667-.1670` **fallback closeout / reopen registry**"
            ),
            "local_response_hit": hit(
                local_response_text, "retained vector no-go scale に張り付いた。"
            ),
        },
        "carry_over": {
            "primary_gate_summary": primary_summary,
            "projected_kernel_summary": projected_summary,
            "ground_state_summary": ground_state_summary,
            "energy_registry_summary": energy_registry_summary,
        },
        "retained_numeric_state": {
            "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
            "vector_alpha_at_q_theory": VECTOR_ALPHA,
            "energy_core_alpha_at_q_theory": ENERGY_CORE_ALPHA,
            "projected_kernel_alpha_at_q_theory": projected_summary[
                "official_projected_kernel_alpha_at_q_theory"
            ],
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
        },
    }

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload(
                "8.7.56.1667",
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
                "8.7.56.1668",
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
                "8.7.56.1669",
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
                "8.7.56.1670",
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
            {"step": STEP_TAG, "stem": STEM, "manifest": manifest, "summary": summary},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
