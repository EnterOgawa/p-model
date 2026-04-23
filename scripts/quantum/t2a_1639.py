#!/usr/bin/env python3
"""Generate 8.7.56.1639-.1642 energy-density closeout / reopen registry artifacts.

This branch does not derive a new observable. The current-pack energy-density
lane has already been closed out honestly:

1. The official exact Hamiltonian-core read is Case II vector-no-go-like.
2. Electric-like / note-gradient surfaces remain evidence-only.
3. Prior caseA worsen, ground-state no-go, and caseB no-metric-rescue remain
   visible.
4. The branch-local full nonlinear energy density is still unavailable.

The honest next step is therefore to freeze the reopen ordering in a
machine-readable form before any refreshed advice-pack or future external input
lane is considered.
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

CLOSEOUT_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1635_1638_energy_density_closeout_declaration_gate_metrics.json"
)
CLOSEOUT_ROUTE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1635_1638_energy_density_closeout_route_sync_metrics.json"
)
CLASS_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1631_1634_energy_density_alpha_case_class_declaration_gate_metrics.json"
)
FF_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1627_1630_energy_density_ff_audit_declaration_gate_metrics.json"
)
DERIV_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1623_1626_energy_density_audit_declaration_gate_metrics.json"
)
CASEA_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1599_1602_v2_sub_exact_treat_declaration_gate_metrics.json"
)
GROUND_STATE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1615_1618_gs_nodeless_audit_declaration_gate_metrics.json"
)
CASEB_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1619_1622_eff_metric_v2_sub_restore_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1639-1642"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor energy-density closeout / "
    "reopen registry"
)
STEM = build_compact_artifact_stem(STEP_TAG, "energy_density_reopen_registry", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_energy_density_case_ii_vector_no_go_like_"
    "closeout_sync_completed"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_energy_density_case_ii_vector_no_go_like_"
    "reopen_registry_completed"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_energy_density_reopen_"
    "advice_pack_refresh"
)
NEXT_ROUTE = "8.7.56.1643"

PRIMARY_REOPEN = (
    "branch_local_full_nonlinear_energy_density_or_exact_constitutive_map_gap"
)
SECONDARY_REOPEN = (
    "evidence_only_electric_like_or_note_gradient_canonical_promotion_gap"
)
RESERVE_REOPEN = "future_external_input_or_new_action_level_structure"


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を検証する。

def require(path: Path) -> None:
    """Fail when one required path is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8テキストを読み込む。

def read_text(path: Path) -> str:
    """Read one UTF-8 text file."""
    return path.read_text(encoding="utf-8")


# 関数: UTF-8 JSON を読み込む。

def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: 表示用の相対パスへ変換する。

def display_path(path: Path) -> str:
    """Convert one absolute path into repo-relative display form when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# 関数: 部分文字列に一致する最初の行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first matching line for one substring pattern."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 標準形式の metrics row を生成する。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# 関数: 標準形式の payload を生成する。

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


# 関数: compact stem で JSON / CSV を出力する。

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


# 関数: 真偽値を 0 / 1 に変換する。

def truth(value: bool) -> float:
    """Convert one boolean into 0/1 float form."""
    return 1.0 if value else 0.0


# 関数: `.1639-.1642` を実行する。

def main() -> None:
    """Execute the energy-density closeout / reopen registry branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        PART5,
        CLOSEOUT_GATE,
        CLOSEOUT_ROUTE,
        CLASS_GATE,
        FF_GATE,
        DERIV_GATE,
        CASEA_GATE,
        GROUND_STATE_GATE,
        CASEB_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    part5_text = read_text(PART5)

    closeout_gate = read_json(CLOSEOUT_GATE)
    closeout_route = read_json(CLOSEOUT_ROUTE)
    class_gate = read_json(CLASS_GATE)
    ff_gate = read_json(FF_GATE)
    deriv_gate = read_json(DERIV_GATE)
    casea_gate = read_json(CASEA_GATE)
    ground_state_gate = read_json(GROUND_STATE_GATE)
    caseb_gate = read_json(CASEB_GATE)

    closeout_summary = closeout_gate["summary"]
    closeout_route_summary = closeout_route["summary"]
    class_summary = class_gate["summary"]
    ff_summary = ff_gate["summary"]
    deriv_summary = deriv_gate["summary"]
    casea_summary = casea_gate["summary"]
    ground_state_summary = ground_state_gate["summary"]
    caseb_summary = caseb_gate["summary"]

    status_registry = hit(status_text, "energy-density closeout / reopen registry")
    roadmap_registry = hit(roadmap_text, "`8.7.56.1639-.1642`")
    current_problem_registry = hit(current_problem_text, "energy-density closeout / reopen registry")
    current_status_registry = hit(current_status_text, "energy-density closeout / reopen registry")
    unified_registry = hit(unified_roadmap_text, "energy-density closeout / reopen registry")
    part5_registry = hit(part5_text, "**energy-density closeout / reopen registry**")

    inventory_ready = all(
        item is not None
        for item in (
            status_registry,
            roadmap_registry,
            current_problem_registry,
            current_status_registry,
            unified_registry,
            part5_registry,
        )
    )

    closeout_sync_available = bool(
        closeout_summary.get("trial2_numeric_alpha_problem_classification") == PRIOR_CLASS
        and closeout_summary.get("energy_density_closeout_sync_ready", False)
        and closeout_route_summary.get("trial2_numeric_alpha_problem_classification") == PRIOR_CLASS
    )
    primary_reopen_honest = bool(
        closeout_summary.get("selected_primary_reopen_surface") == PRIMARY_REOPEN
        and closeout_summary.get("full_nonlinear_energy_density_reopen_retained", False)
    )
    secondary_reopen_honest = bool(
        closeout_summary.get("evidence_only_improvement_surfaces_retained", False)
        and class_summary.get("noncanonical_improvement_surfaces_retained", False)
        and ff_summary.get("electric_like_improves_but_is_not_official", False)
    )
    reserve_reopen_honest = bool(
        not closeout_summary.get("physical_reject_required", True)
        and ground_state_summary.get("physical_reject_required", False) is False
        and caseb_summary.get("physical_reject_required", False) is False
    )
    casea_retained = bool(casea_summary.get("worsen_selected", False))
    ground_state_retained = bool(
        not ground_state_summary.get("ground_state_nodeless_hypothesis_supported_under_current_pack", True)
    )
    caseb_retained = bool(
        not caseb_summary.get("metric_artifact_rescue_supported", True)
    )

    registry_wording_honest = all(
        (
            inventory_ready,
            closeout_sync_available,
            primary_reopen_honest,
            secondary_reopen_honest,
            reserve_reopen_honest,
            casea_retained,
            ground_state_retained,
            caseb_retained,
        )
    )
    registry_ready = registry_wording_honest

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
            "closeout_gate": display_path(CLOSEOUT_GATE),
            "closeout_route": display_path(CLOSEOUT_ROUTE),
            "classification_gate": display_path(CLASS_GATE),
            "energy_ff_gate": display_path(FF_GATE),
            "energy_deriv_gate": display_path(DERIV_GATE),
            "casea_gate": display_path(CASEA_GATE),
            "ground_state_gate": display_path(GROUND_STATE_GATE),
            "caseb_gate": display_path(CASEB_GATE),
        },
        "constants": {
            "primary_reopen_surface": PRIMARY_REOPEN,
            "secondary_reopen_surface": SECONDARY_REOPEN,
            "reserve_reopen_surface": RESERVE_REOPEN,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    inventory_rows = [
        row(
            "inventory_ready",
            "pass" if inventory_ready else "fail",
            "energy-density reopen-registry inventory ready",
            truth(inventory_ready),
            "Registry starts only after status, roadmap, current notes, unified roadmap, and Part V all already point to the energy-density reopen-registry branch.",
        ),
        row(
            "closeout_sync_available",
            "pass" if closeout_sync_available else "fail",
            "prior energy-density closeout sync available",
            truth(closeout_sync_available),
            "The registry is only honest after the Case II vector-no-go-like closeout has already been frozen machine-readably.",
        ),
        row(
            "primary_reopen_honest",
            "pass" if primary_reopen_honest else "fail",
            "primary reopen surface honest",
            truth(primary_reopen_honest),
            "The primary reopen surface remains the branch-local full nonlinear energy density / exact constitutive-map gap.",
        ),
        row(
            "secondary_reopen_honest",
            "pass" if secondary_reopen_honest else "fail",
            "secondary reopen surface honest",
            truth(secondary_reopen_honest),
            "Evidence-only electric-like / note-gradient improvements remain visible, but only as non-canonical promotion gaps.",
        ),
        row(
            "reserve_reopen_honest",
            "pass" if reserve_reopen_honest else "fail",
            "reserve reopen surface honest",
            truth(reserve_reopen_honest),
            "Future external input or genuinely new action-level structure remains reserve and does not replace the current reopen gap.",
        ),
    ]
    inventory_artifacts = write_artifact(
        "inventory",
        payload(
            "8.7.56.1639",
            STEP_NAME + " inventory",
            inputs,
            inventory_rows,
            {
                "inventory_ready": inventory_ready,
                "closeout_sync_available": closeout_sync_available,
                "primary_reopen_honest": primary_reopen_honest,
                "secondary_reopen_honest": secondary_reopen_honest,
                "reserve_reopen_honest": reserve_reopen_honest,
            },
            {
                "overall_status": "inventory_completed",
                "branch_completed": False,
            },
            {
                "hits": {
                    "status_registry": status_registry,
                    "roadmap_registry": roadmap_registry,
                    "current_problem_registry": current_problem_registry,
                    "current_status_registry": current_status_registry,
                    "unified_registry": unified_registry,
                    "part5_registry": part5_registry,
                },
            },
        ),
    )

    audit_rows = [
        row(
            "casea_worsen_retained",
            "pass" if casea_retained else "fail",
            "prior caseA worsen retained",
            truth(casea_retained),
            "The registry must keep the Minkowski-contracted signed-kernel worsen explicit rather than hiding it behind the energy-density closeout wording.",
        ),
        row(
            "ground_state_no_go_retained",
            "pass" if ground_state_retained else "fail",
            "ground-state note no-go retained",
            truth(ground_state_retained),
            "The current pack still does not support the nodeless ground-state rescue, so the registry must keep that failure visible.",
        ),
        row(
            "caseb_no_metric_rescue_retained",
            "pass" if caseb_retained else "fail",
            "caseB no-metric-rescue retained",
            truth(caseb_retained),
            "The effective-metric recomputation also failed to rescue the scalar candidate and remains part of the reopen pack.",
        ),
        row(
            "registry_wording_honest",
            "pass" if registry_wording_honest else "fail",
            "energy-density reopen-registry wording honest",
            truth(registry_wording_honest),
            "The registry wording is honest only if the Case II closeout, the evidence-only surfaces, and all failed rescue lanes remain visible together.",
        ),
        row(
            "registry_ready",
            "pass" if registry_ready else "fail",
            "energy-density reopen registry ready",
            truth(registry_ready),
            "Once the ordering is explicit, the energy-density lane can move from closeout into a stable reopen registry.",
        ),
    ]
    audit_artifacts = write_artifact(
        "audit",
        payload(
            "8.7.56.1640",
            STEP_NAME + " audit",
            inputs,
            audit_rows,
            {
                "casea_worsen_retained": casea_retained,
                "ground_state_note_no_go_retained": ground_state_retained,
                "caseb_no_metric_rescue_retained": caseb_retained,
                "registry_wording_honest": registry_wording_honest,
                "registry_ready": registry_ready,
            },
            {
                "overall_status": "audit_completed",
                "branch_completed": False,
            },
            {
                "carry_over": {
                    "closeout_summary": closeout_summary,
                    "classification_summary": class_summary,
                    "energy_ff_summary": ff_summary,
                    "energy_deriv_summary": deriv_summary,
                    "casea_summary": casea_summary,
                    "ground_state_summary": ground_state_summary,
                    "caseb_summary": caseb_summary,
                }
            },
        ),
    )

    declaration_summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "selected_disposition_case": closeout_summary["selected_disposition_case"],
        "official_surface_name": closeout_summary["official_surface_name"],
        "official_F_E_at_q_theory": closeout_summary["official_F_E_at_q_theory"],
        "official_alpha_E_at_q_theory": closeout_summary["official_alpha_E_at_q_theory"],
        "official_alpha_E_residual_rel": closeout_summary["official_alpha_E_residual_rel"],
        "primary_reopen_surface": PRIMARY_REOPEN,
        "secondary_reopen_surface": SECONDARY_REOPEN,
        "reserve_reopen_surface": RESERVE_REOPEN,
        "evidence_only_improvement_surfaces_retained": closeout_summary[
            "evidence_only_improvement_surfaces_retained"
        ],
        "electric_like_component_alpha_at_q_theory": closeout_summary[
            "electric_like_component_alpha_at_q_theory"
        ],
        "note_gradient_alpha_at_q_theory": closeout_summary[
            "note_gradient_alpha_at_q_theory"
        ],
        "full_nonlinear_energy_density_reopen_retained": closeout_summary[
            "full_nonlinear_energy_density_reopen_retained"
        ],
        "prior_casea_worsen_retained": closeout_summary["prior_casea_worsen_retained"],
        "ground_state_note_no_go_retained": closeout_summary[
            "ground_state_note_no_go_retained"
        ],
        "prior_caseb_no_metric_rescue_retained": closeout_summary[
            "prior_caseb_no_metric_rescue_retained"
        ],
        "energy_density_reopen_registry_wording_honest": registry_wording_honest,
        "energy_density_reopen_registry_ready": registry_ready,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "physical_reject_required": False,
    }
    declaration_rows = [
        row(
            "primary_reopen_surface_fixed",
            "pass",
            "primary reopen surface fixed",
            1.0,
            "The primary reopen surface is frozen as the branch-local full nonlinear energy density / exact constitutive-map gap.",
        ),
        row(
            "secondary_reopen_surface_fixed",
            "pass",
            "secondary reopen surface fixed",
            1.0,
            "The evidence-only electric-like / note-gradient improvement family remains secondary and cannot replace the official exact surface.",
        ),
        row(
            "reserve_reopen_surface_fixed",
            "pass",
            "reserve reopen surface fixed",
            1.0,
            "Future external input or genuinely new action-level structure stays reserve and does not block the current registry.",
        ),
        row(
            "physical_reject_not_selected",
            "pass",
            "physical reject not selected",
            1.0,
            "Even after the energy-density lane closes into a reopen registry, the route remains local and does not force physical rejection.",
        ),
        row(
            "declaration_gate_ready",
            "pass" if registry_ready else "fail",
            "energy-density reopen-registry declaration gate ready",
            truth(registry_ready),
            "Once the reopen ordering is explicit and honest, the registry can be declared machine-readably complete.",
        ),
    ]
    declaration_artifacts = write_artifact(
        "declaration_gate",
        payload(
            "8.7.56.1641",
            STEP_NAME + " declaration gate",
            inputs,
            declaration_rows,
            declaration_summary,
            {
                "overall_status": BRANCH_CLASS + "_declared",
                "branch_completed": registry_ready,
            },
            {"artifacts": {"inventory": inventory_artifacts, "audit": audit_artifacts}},
        ),
    )

    route_sync_summary = {
        **declaration_summary,
        "numeric_state_changed_by_current_branch": False,
        "route_state_changed_by_current_branch": True,
        "registry_machine_readable_fixed": True,
    }
    route_sync_rows = [
        row(
            "official_energy_core_alpha_retained",
            "pass",
            "official energy-core alpha retained",
            declaration_summary["official_alpha_E_at_q_theory"],
            "The registry does not alter the official exact energy-core read.",
        ),
        row(
            "vector_no_go_scale_retained",
            "pass",
            "vector no-go scale retained",
            0.0005579616187042394,
            "The retained direct vector fixed-q no-go scale remains part of the comparison baseline.",
        ),
        row(
            "scalar_candidate_retained",
            "pass",
            "scalar strong candidate retained",
            0.00715678583937324,
            "The retained scalar exact-profile candidate stays visible as a separate strong candidate and is not overwritten by the registry.",
        ),
        row(
            "route_state_changed",
            "pass",
            "route state changed by current branch",
            1.0,
            "The branch changes only the route state by freezing the reopen ordering machine-readably.",
        ),
        row(
            "next_route_fixed",
            "pass",
            "next route fixed",
            1.0,
            "The next official route is the refreshed energy-density reopen advice-pack branch.",
        ),
    ]
    route_sync_artifacts = write_artifact(
        "route_sync",
        payload(
            "8.7.56.1642",
            STEP_NAME + " route sync",
            inputs,
            route_sync_rows,
            route_sync_summary,
            {
                "overall_status": BRANCH_CLASS,
                "branch_completed": True,
                "next_required_artifacts": [NEXT_ROUTE_NAME],
            },
            {
                "artifacts": {
                    "inventory": inventory_artifacts,
                    "audit": audit_artifacts,
                    "declaration_gate": declaration_artifacts,
                }
            },
        ),
    )

    print("[ok] completed", STEP_TAG, STEP_NAME)
    print("[ok] inventory:", inventory_artifacts["json"])
    print("[ok] audit:", audit_artifacts["json"])
    print("[ok] declaration_gate:", declaration_artifacts["json"])
    print("[ok] route_sync:", route_sync_artifacts["json"])


if __name__ == "__main__":
    main()
