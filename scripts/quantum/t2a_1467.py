#!/usr/bin/env python3
"""Generate 8.7.56.1467-.1470 archive-registry restore artifacts for unified closure."""

from __future__ import annotations

import csv
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
WORK_HISTORY_RECENT = ROOT / "doc" / "WORK_HISTORY_RECENT.md"
PRIMARY_SOURCES = ROOT / "doc" / "PRIMARY_SOURCES.md"
CURRENT_PROBLEM = ROOT / "doc" / "quantum" / "34_trial2_numeric_alpha_current_problem.md"
CURRENT_STATUS = ROOT / "doc" / "quantum" / "36_trial2_numeric_alpha_current_status.md"
EXPERT_SHARE = ROOT / "doc" / "quantum" / "38_trial2_numeric_alpha_vector_qball_exploratory_expert_share.md"
UNIFIED_ROADMAP = ROOT / "doc" / "quantum" / "39_trial2_vector_qball_unified_closure_roadmap.md"
ADVICE_REQUEST = ROOT / "doc" / "quantum" / "40_trial2_numeric_alpha_vector_qball_reopen_advice_request.md"
NEXT_ACTION_INTEGRATION = ROOT / "doc" / "quantum" / "41_trial2_vector_qball_next_action_integration.md"
CASE_GAMMA_ADVICE = ROOT / "doc" / "quantum" / "42_trial2_numeric_alpha_vector_qball_case_gamma_advice_request.md"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

UNIFIED_PLAN = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_unified_closure_plan_20260327.md")
NEXT_STEPS = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_next_steps_20260327.md")
NEXT_ACTION = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_next_action_20260327.md")

FINAL_SUMMARY_INV = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_handoff_final_summary_route_"
    "source_inventory_metrics.json"
)
FINAL_SUMMARY_AUDIT = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_handoff_final_summary_route_"
    "audit_metrics.json"
)
FINAL_SUMMARY_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_handoff_final_summary_route_"
    "declaration_gate_metrics.json"
)
FINAL_SUMMARY_HANDOFF = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_handoff_final_summary_route_"
    "expert_handoff_sync_metrics.json"
)
DIAGNOSTIC_INV = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_perturbative_fl_driven_ode_diagnostic_reopen_review_"
    "source_inventory_metrics.json"
)
DIAGNOSTIC_AUDIT = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_perturbative_fl_driven_ode_diagnostic_reopen_review_"
    "audit_metrics.json"
)
DIAGNOSTIC_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_perturbative_fl_driven_ode_diagnostic_reopen_review_"
    "declaration_gate_metrics.json"
)
DIAGNOSTIC_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_perturbative_fl_driven_ode_diagnostic_reopen_review_"
    "numeric_evaluation_metrics.json"
)
PHASE1_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_phase1_exact_coupled_l0_solver_"
    "numeric_evaluation_metrics.json"
)

SCRIPT_1459 = ROOT / "scripts" / "quantum" / "t2a_1459.py"
SCRIPT_1463 = ROOT / "scripts" / "quantum" / "t2a_1463.py"

PRIOR_CLASS = "vector_qball_form_factor_unified_closure_perturbative_fl_driven_ode_case_gamma_archive_registry_restore_required"
BRANCH_CLASS = "vector_qball_form_factor_unified_closure_future_reopen_trigger_handoff_archive_registry_restore_completed"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_exact_action_level_ell0_operator_derivation"
NEXT_ROUTE = "8.7.56.1471"
STEM = "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_handoff_archive_registry_restore"

PRIMARY_TRIGGER = "exact_action_level_ell0_operator_reopen"
SECONDARY_TRIGGER = "future_source_theorem_reopen"
RESERVE_TRIGGER = "observable_dictionary_exact_charge_current_bridge"


# Function: return the current UTC timestamp string.
def now_iso() -> str:
    """Return the current UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


# Function: fail fast when a required path is missing.

def require(path: Path) -> None:
    """Fail fast when a required path is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# Function: read one UTF-8 text file.

def read_text(path: Path) -> str:
    """Read one UTF-8 text file."""
    return path.read_text(encoding="utf-8")


# Function: read one UTF-8 JSON file.

def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Function: convert one path into repo-relative display form when possible.

def display_path(path: Path) -> str:
    """Convert one path into repo-relative display form when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# Function: return the first matching line for one substring.

def hit(text: str, pattern: str) -> dict | None:
    """Return the first matching line for one substring."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# Function: build one standard metrics row.

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# Function: write one JSON payload and CSV rows table.

def write_artifact(kind: str, data: dict) -> None:
    """Write one JSON payload and CSV rows table."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    json_path = PUBLIC_OUT / f"{STEM}_{kind}_metrics.json"
    csv_path = PUBLIC_OUT / f"{STEM}_{kind}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# Function: build one standard payload object.

def payload(
    step: str,
    name: str,
    inputs: dict,
    rows: list[dict],
    summary: dict,
    decision: dict,
    evidence: dict,
) -> dict:
    """Build one standard payload object."""
    return {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": step, "name": name},
        "inputs": inputs,
        "rows": rows,
        "summary": summary,
        "decision": decision,
        "evidence": evidence,
    }


# Function: build one concise archive-restore sentence.

def build_restore_text(phase1_eval: dict, diagnostic_eval: dict) -> str:
    """Build one concise archive-restore sentence."""
    return (
        "Case gamma is now archived without reviving the wrong-branch rescue: "
        f"max|fL/f0|={diagnostic_eval['diagnostic_max_abs_ratio']}, "
        f"F_diag(q_theory)={diagnostic_eval['diagnostic_F_at_q_theory']}, "
        f"alpha_diag(q_theory)={diagnostic_eval['diagnostic_alpha_at_q_theory']}, "
        f"while the retained scalar baseline stays at F_exact(q_theory)={phase1_eval['F_exact_at_q_theory_scalar']} "
        f"and alpha_exact(q_theory)={phase1_eval['alpha_exact_at_q_theory_scalar']}; future reopen ordering remains "
        f"{PRIMARY_TRIGGER}, {SECONDARY_TRIGGER}, {RESERVE_TRIGGER}, physical_reject_required stays false, and the next computation "
        "mainline begins with exact-action-level ell=0 operator derivation."
    )


# Function: refresh one existing handoff bundle with the current synced files.

def refresh_handoff_bundle(bundle_dir: Path, bundle_zip: Path, files_to_sync: list[Path]) -> dict:
    """Refresh one existing handoff bundle with the current synced files."""
    copied_files: list[Path] = []
    for source in files_to_sync:
        target_path = bundle_dir / source.name
        shutil.copy2(source, target_path)
        copied_files.append(source)

    with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(bundle_dir.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, arcname=file_path.relative_to(bundle_dir))

    with zipfile.ZipFile(bundle_zip, "r") as archive:
        zip_file_count = len(archive.namelist())

    return {
        "copied_files": copied_files,
        "copied_count": len(copied_files),
        "staging_file_count": len(list(bundle_dir.iterdir())),
        "zip_file_count": zip_file_count,
    }


# Function: execute the archive-registry restore branch and hand off to the computation mainline.

def main() -> None:
    """Execute the archive-registry restore branch and hand off to the computation mainline."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        PRIMARY_SOURCES,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        EXPERT_SHARE,
        UNIFIED_ROADMAP,
        ADVICE_REQUEST,
        NEXT_ACTION_INTEGRATION,
        CASE_GAMMA_ADVICE,
        PART1,
        PART3A,
        PART5,
        UNIFIED_PLAN,
        NEXT_STEPS,
        NEXT_ACTION,
        FINAL_SUMMARY_INV,
        FINAL_SUMMARY_AUDIT,
        FINAL_SUMMARY_GATE,
        FINAL_SUMMARY_HANDOFF,
        DIAGNOSTIC_INV,
        DIAGNOSTIC_AUDIT,
        DIAGNOSTIC_GATE,
        DIAGNOSTIC_EVAL,
        PHASE1_EVAL,
        SCRIPT_1459,
        SCRIPT_1463,
        Path(__file__),
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    expert_share_text = read_text(EXPERT_SHARE)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    advice_request_text = read_text(ADVICE_REQUEST)
    next_action_integration_text = read_text(NEXT_ACTION_INTEGRATION)
    case_gamma_advice_text = read_text(CASE_GAMMA_ADVICE)
    part5_text = read_text(PART5)
    next_action_text = read_text(NEXT_ACTION)

    final_summary_inv = read_json(FINAL_SUMMARY_INV)["summary"]
    final_summary_audit = read_json(FINAL_SUMMARY_AUDIT)["summary"]
    final_summary_gate = read_json(FINAL_SUMMARY_GATE)["summary"]
    final_summary_handoff = read_json(FINAL_SUMMARY_HANDOFF)["summary"]
    diagnostic_inv = read_json(DIAGNOSTIC_INV)["summary"]
    diagnostic_audit = read_json(DIAGNOSTIC_AUDIT)["summary"]
    diagnostic_gate = read_json(DIAGNOSTIC_GATE)["summary"]
    diagnostic_eval = read_json(DIAGNOSTIC_EVAL)["summary"]
    phase1_eval = read_json(PHASE1_EVAL)["summary"]

    bundle_zip = ROOT / final_summary_handoff["share_pack_bundle_zip"]
    bundle_dir = bundle_zip.with_suffix("")
    require(bundle_zip)
    require(bundle_dir)

    restore_hits_required = [
        hit(status_text, "handoff archive registry restore"),
        hit(roadmap_text, "`8.7.56.1467-.1470`"),
        hit(roadmap_text, "`8.7.56.1471-.1474`"),
        hit(current_problem_text, "Case γ"),
        hit(unified_roadmap_text, "computation mainline"),
        hit(next_action_integration_text, "wrong-branch suspicion"),
        hit(case_gamma_advice_text, "Case γ"),
        hit(next_action_text, "branch"),
    ]
    restore_hits_optional = [
        hit(current_status_text, "Case γ"),
        hit(expert_share_text, "future reopen ordering"),
        hit(advice_request_text, "Case γ"),
        hit(part5_text, "archive registry restore"),
    ]
    inventory_ready = all(item is not None for item in restore_hits_required)

    share_pack_bundle_available = bool(bundle_zip.exists() and bundle_dir.exists())
    final_summary_ready = bool(
        final_summary_inv["inventory_ready"]
        and final_summary_audit["future_reopen_trigger_handoff_final_summary_wording_honest"]
        and final_summary_gate["future_reopen_trigger_handoff_final_summary_ready"]
        and final_summary_handoff["future_reopen_trigger_handoff_final_summary_handoff_sync_complete"]
    )
    prior_route_ready = bool(
        diagnostic_eval["selected_next_generation_route"]
        == "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_handoff_archive_registry_restore"
    )
    case_gamma_selected = bool(diagnostic_eval["case_gamma_selected"])
    wrong_branch_rescue_not_supported = not bool(diagnostic_eval["wrong_branch_suspicion_supported"])
    perturbative_breakdown_detected = bool(diagnostic_eval["perturbative_breakdown_detected"])
    exact_solver_reinjection_not_required = not bool(diagnostic_eval["exact_solver_reinjection_required"])
    archive_restore_required = bool(diagnostic_eval["handoff_archive_registry_restore_required"])
    case_c_honest_partial_retained = bool(diagnostic_eval["case_c_honest_partial_retained"])
    retained_scalar_candidate_retained = bool(diagnostic_eval["retained_scalar_strong_candidate_retained"])
    physical_reject_not_selected = not bool(diagnostic_eval["physical_reject_required"])
    ordering_retained = bool(
        final_summary_handoff["primary_future_reopen_trigger"] == PRIMARY_TRIGGER
        and final_summary_handoff["secondary_future_reopen_trigger"] == SECONDARY_TRIGGER
        and final_summary_handoff["reserve_future_reopen_trigger"] == RESERVE_TRIGGER
    )
    post_closeout_computation_mainline_armed = bool(
        hit(roadmap_text, "exact-action-level `\\ell=0` operator derivation")
        and hit(roadmap_text, "branch continuation / family map")
        and hit(roadmap_text, "effective source theorem attempt")
        and hit(roadmap_text, "observable dictionary gate")
    )
    restore_text = build_restore_text(phase1_eval, diagnostic_eval)
    restore_word_count = len(restore_text.split())
    restore_concise = restore_word_count <= 100
    wording_honest = all(
        [
            restore_concise,
            final_summary_ready,
            case_gamma_selected,
            wrong_branch_rescue_not_supported,
            perturbative_breakdown_detected,
            exact_solver_reinjection_not_required,
            archive_restore_required,
            case_c_honest_partial_retained,
            retained_scalar_candidate_retained,
            physical_reject_not_selected,
            ordering_retained,
            post_closeout_computation_mainline_armed,
        ]
    )
    archive_restore_ready = all(
        [
            inventory_ready,
            share_pack_bundle_available,
            final_summary_ready,
            prior_route_ready,
            wording_honest,
        ]
    )

    common_inputs = {
        "source_files": {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "ai_context": display_path(AI_CONTEXT),
            "work_history_recent": display_path(WORK_HISTORY_RECENT),
            "primary_sources": display_path(PRIMARY_SOURCES),
            "current_problem_note": display_path(CURRENT_PROBLEM),
            "current_status_note": display_path(CURRENT_STATUS),
            "expert_share_note": display_path(EXPERT_SHARE),
            "unified_closure_roadmap_note": display_path(UNIFIED_ROADMAP),
            "advice_request_note": display_path(ADVICE_REQUEST),
            "next_action_integration_note": display_path(NEXT_ACTION_INTEGRATION),
            "case_gamma_advice_note": display_path(CASE_GAMMA_ADVICE),
            "part1": display_path(PART1),
            "part3a": display_path(PART3A),
            "part5": display_path(PART5),
            "unified_plan_note": display_path(UNIFIED_PLAN),
            "next_steps_note": display_path(NEXT_STEPS),
            "next_action_note": display_path(NEXT_ACTION),
        },
        "source_metrics": {
            "final_summary_inventory": display_path(FINAL_SUMMARY_INV),
            "final_summary_audit": display_path(FINAL_SUMMARY_AUDIT),
            "final_summary_gate": display_path(FINAL_SUMMARY_GATE),
            "final_summary_handoff": display_path(FINAL_SUMMARY_HANDOFF),
            "diagnostic_inventory": display_path(DIAGNOSTIC_INV),
            "diagnostic_audit": display_path(DIAGNOSTIC_AUDIT),
            "diagnostic_gate": display_path(DIAGNOSTIC_GATE),
            "diagnostic_evaluation": display_path(DIAGNOSTIC_EVAL),
            "phase1_eval": display_path(PHASE1_EVAL),
        },
        "scripts": {
            "handoff_final_summary_route": display_path(SCRIPT_1459),
            "case_gamma_diagnostic_review": display_path(SCRIPT_1463),
            "handoff_archive_registry_restore": display_path(Path(__file__)),
        },
        "constants": {
            "prior_classification": PRIOR_CLASS,
            "primary_trigger": PRIMARY_TRIGGER,
            "secondary_trigger": SECONDARY_TRIGGER,
            "reserve_trigger": RESERVE_TRIGGER,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "bundle_zip": display_path(bundle_zip),
        },
    }

    inventory_payload = payload(
        "8.7.56.1467",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure handoff archive-registry restore inventory",
        common_inputs,
        [
            row(
                "inventory_ready",
                "pass" if inventory_ready else "reject",
                "archive-registry restore inventory ready",
                1 if inventory_ready else 0,
                "The restore inventory is ready only if the final-summary handoff, Case gamma diagnostic, current notes, roadmap switch, and paper wording coexist in one pack.",
            ),
            row(
                "final_summary_handoff_available",
                "pass" if final_summary_ready else "reject",
                "future reopen-trigger handoff final summary available for restore",
                1 if final_summary_ready else 0,
                "The archive-registry restore can start only after the prior handoff final-summary route is frozen and synced.",
            ),
            row(
                "case_gamma_diagnostic_available",
                "pass" if archive_restore_required else "reject",
                "Case gamma diagnostic available for restore",
                1 if archive_restore_required else 0,
                "The restore route exists because the perturbative fL diagnostic ended in Case gamma and returned the state to archive-registry restore.",
            ),
            row(
                "share_pack_bundle_available",
                "pass" if share_pack_bundle_available else "reject",
                "canonical share-pack bundle available for restore",
                1 if share_pack_bundle_available else 0,
                "The restore route reuses the canonical expert-share bundle produced by the handoff final-summary branch.",
            ),
            row(
                "post_closeout_computation_mainline_armed",
                "pass" if post_closeout_computation_mainline_armed else "reject",
                "post-closeout computation mainline armed",
                1 if post_closeout_computation_mainline_armed else 0,
                "The restore route is only honest if the archive handoff now points to the exact ell=0 operator computation mainline rather than more wording-only branches.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "prior_problem_classification": PRIOR_CLASS,
            "inventory_ready": inventory_ready,
            "final_summary_handoff_available": final_summary_ready,
            "case_gamma_diagnostic_available": archive_restore_required,
            "share_pack_bundle_available": share_pack_bundle_available,
            "post_closeout_computation_mainline_armed": post_closeout_computation_mainline_armed,
            "share_pack_bundle_zip": display_path(bundle_zip),
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_handoff_archive_registry_restore_inventory_fixed",
            "advance_to_8_7_56_1468": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "restore_hits_required": restore_hits_required,
            "restore_hits_optional": restore_hits_optional,
            "diagnostic_problem_classification": diagnostic_eval["trial2_numeric_alpha_problem_classification"],
            "restore_text": restore_text,
            "restore_word_count": restore_word_count,
        },
    )
    write_artifact("source_inventory", inventory_payload)

    audit_payload = payload(
        "8.7.56.1468",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure handoff archive-registry restore wording audit",
        common_inputs,
        [
            row(
                "archive_registry_restore_wording_honest",
                "pass" if wording_honest else "reject",
                "archive-registry restore wording honest",
                1 if wording_honest else 0,
                "The restore wording is honest only if it preserves Case C honest partial, the retained scalar baseline, the Case gamma diagnostic, the frozen reopen ordering, and physical_reject_required=false without overclaiming a rescue.",
            ),
            row(
                "archive_registry_restore_wording_concise",
                "pass" if restore_concise else "reject",
                "archive-registry restore wording concise",
                1 if restore_concise else 0,
                "The restore wording should compress the frozen Case gamma result into one short sentence before handing control to the operator-derivation computation mainline.",
            ),
            row(
                "case_gamma_retained",
                "pass" if case_gamma_selected else "reject",
                "Case gamma retained in archive-registry restore",
                1 if case_gamma_selected else 0,
                "The restore route must keep the perturbative diagnostic in Case gamma rather than reopening the wrong-branch rescue claim.",
            ),
            row(
                "wrong_branch_rescue_not_supported",
                "pass" if wrong_branch_rescue_not_supported else "reject",
                "wrong-branch rescue not supported",
                1 if wrong_branch_rescue_not_supported else 0,
                "The perturbative diagnostic did not support the wrong-branch rescue hypothesis and the restore wording must preserve that result.",
            ),
            row(
                "retained_scalar_strong_candidate_retained",
                "pass" if retained_scalar_candidate_retained else "reject",
                "retained scalar strong candidate retained in archive-registry restore",
                1 if retained_scalar_candidate_retained else 0,
                "The scalar strong candidate must stay visible so the route does not read like a full physical reject.",
            ),
            row(
                "future_reopen_ordering_retained",
                "pass" if ordering_retained else "reject",
                "future reopen ordering retained in archive-registry restore",
                1 if ordering_retained else 0,
                "The restore wording must preserve the primary / secondary / reserve reopen ordering verbatim.",
            ),
            row(
                "physical_reject_not_selected",
                "pass" if physical_reject_not_selected else "reject",
                "physical reject not selected in archive-registry restore",
                1 if physical_reject_not_selected else 0,
                "The restore route remains a retained reopen state and therefore must keep physical_reject_required=false.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "prior_problem_classification": PRIOR_CLASS,
            "archive_registry_restore_wording_honest": wording_honest,
            "archive_registry_restore_wording_concise": restore_concise,
            "case_gamma_selected": case_gamma_selected,
            "wrong_branch_rescue_not_supported": wrong_branch_rescue_not_supported,
            "retained_scalar_strong_candidate_retained": retained_scalar_candidate_retained,
            "primary_future_reopen_trigger": PRIMARY_TRIGGER,
            "secondary_future_reopen_trigger": SECONDARY_TRIGGER,
            "reserve_future_reopen_trigger": RESERVE_TRIGGER,
            "physical_reject_required": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_handoff_archive_registry_restore_audited",
            "advance_to_8_7_56_1469": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "restore_text": restore_text,
            "restore_word_count": restore_word_count,
            "diagnostic_max_abs_ratio": diagnostic_eval["diagnostic_max_abs_ratio"],
            "diagnostic_F_at_q_theory": diagnostic_eval["diagnostic_F_at_q_theory"],
            "diagnostic_alpha_at_q_theory": diagnostic_eval["diagnostic_alpha_at_q_theory"],
            "scalar_F_exact_at_q_theory": phase1_eval["F_exact_at_q_theory_scalar"],
            "scalar_alpha_exact_at_q_theory": phase1_eval["alpha_exact_at_q_theory_scalar"],
        },
    )
    write_artifact("audit", audit_payload)

    gate_payload = payload(
        "8.7.56.1469",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure handoff archive-registry restore declaration gate",
        common_inputs,
        [
            row(
                "archive_registry_restore_ready",
                "pass" if archive_restore_ready else "reject",
                "archive-registry restore ready",
                1 if archive_restore_ready else 0,
                "The restore route can be declared only after the inventory, wording audit, prior final-summary handoff, Case gamma diagnostic, and frozen reopen ordering all agree.",
            ),
            row(
                "archive_registry_restore_fixed",
                "pass" if archive_restore_ready else "reject",
                "archive-registry restore fixed",
                1 if archive_restore_ready else 0,
                "The declaration gate freezes the archive-registry restore wording and hands the branch off to the exact-action-level ell=0 operator derivation mainline.",
            ),
            row(
                "case_gamma_archive_restored",
                "pass" if archive_restore_required else "reject",
                "Case gamma archive restored",
                1 if archive_restore_required else 0,
                "The declaration gate confirms that Case gamma did not reopen the exact solver and instead returned the state to archive-registry restore.",
            ),
            row(
                "next_computation_mainline_selected",
                "pass" if archive_restore_ready else "reject",
                "exact-action-level ell=0 operator derivation selected as next computation mainline",
                1 if archive_restore_ready else 0,
                "After archive restoration is complete, the next official branch is the exact-action-level ell=0 operator derivation computation route.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "prior_problem_classification": PRIOR_CLASS,
            "archive_registry_restore_ready": archive_restore_ready,
            "primary_future_reopen_trigger": PRIMARY_TRIGGER,
            "secondary_future_reopen_trigger": SECONDARY_TRIGGER,
            "reserve_future_reopen_trigger": RESERVE_TRIGGER,
            "physical_reject_required": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_handoff_archive_registry_restore_gate_closed",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "restore_text": restore_text,
            "diagnostic_case_gamma": case_gamma_selected,
            "post_closeout_computation_mainline_armed": post_closeout_computation_mainline_armed,
        },
    )
    write_artifact("declaration_gate", gate_payload)

    refresh_candidates = [
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        PRIMARY_SOURCES,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        EXPERT_SHARE,
        UNIFIED_ROADMAP,
        ADVICE_REQUEST,
        NEXT_ACTION_INTEGRATION,
        CASE_GAMMA_ADVICE,
        PART1,
        PART3A,
        PART5,
        UNIFIED_PLAN,
        NEXT_STEPS,
        NEXT_ACTION,
        FINAL_SUMMARY_INV,
        FINAL_SUMMARY_AUDIT,
        FINAL_SUMMARY_GATE,
        FINAL_SUMMARY_HANDOFF,
        DIAGNOSTIC_INV,
        DIAGNOSTIC_AUDIT,
        DIAGNOSTIC_GATE,
        DIAGNOSTIC_EVAL,
        PHASE1_EVAL,
        SCRIPT_1459,
        SCRIPT_1463,
        Path(__file__),
        PUBLIC_OUT / f"{STEM}_source_inventory_metrics.json",
        PUBLIC_OUT / f"{STEM}_audit_metrics.json",
        PUBLIC_OUT / f"{STEM}_declaration_gate_metrics.json",
    ]
    refreshed_bundle = refresh_handoff_bundle(bundle_dir, bundle_zip, refresh_candidates)

    handoff_payload = payload(
        "8.7.56.1470",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure handoff archive-registry restore expert handoff sync",
        common_inputs,
        [
            row(
                "archive_registry_restore_handoff_sync_complete",
                "pass" if archive_restore_ready else "reject",
                "archive-registry restore handoff sync complete",
                1 if archive_restore_ready else 0,
                "Handoff sync is complete only when the archive-registry restore route is ready and the canonical bundle has been refreshed with the restore files.",
            ),
            row(
                "share_pack_bundle_retained",
                "pass" if share_pack_bundle_available else "reject",
                "canonical share-pack bundle retained for archive-registry restore",
                1 if share_pack_bundle_available else 0,
                "The archive-registry restore reuses the same canonical expert-share bundle carried by the prior handoff branches.",
            ),
            row(
                "share_pack_bundle_refreshed",
                "pass" if refreshed_bundle["copied_count"] > 0 else "reject",
                "canonical share-pack bundle refreshed with archive-registry restore files",
                refreshed_bundle["copied_count"],
                "The archive-registry restore refreshes the canonical bundle in place so expert readers see the active Case gamma restore wording and the computation-mainline handoff.",
            ),
            row(
                "expert_handoff_note_sync_ready",
                "pass" if inventory_ready else "reject",
                "expert handoff notes sync ready",
                1 if inventory_ready else 0,
                "The expert handoff is ready because the final-summary route, Case gamma diagnostic, restore wording, and current notes now point to the same reopen state.",
            ),
            row(
                "exact_action_level_ell0_operator_derivation_selected",
                "pass" if archive_restore_ready else "reject",
                "exact-action-level ell=0 operator derivation selected",
                1 if archive_restore_ready else 0,
                "After the archive-registry restore is frozen, the route advances to the exact-action-level ell=0 operator derivation computation mainline.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "prior_problem_classification": PRIOR_CLASS,
            "archive_registry_restore_handoff_sync_complete": archive_restore_ready,
            "share_pack_bundle_zip": display_path(bundle_zip),
            "share_pack_bundle_refresh_count": refreshed_bundle["copied_count"],
            "primary_future_reopen_trigger": PRIMARY_TRIGGER,
            "secondary_future_reopen_trigger": SECONDARY_TRIGGER,
            "reserve_future_reopen_trigger": RESERVE_TRIGGER,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_handoff_archive_registry_restore_handoff_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "restore_text": restore_text,
            "refreshed_bundle_copied_files": [display_path(path) for path in refreshed_bundle["copied_files"]],
            "refreshed_bundle_staging_file_count": refreshed_bundle["staging_file_count"],
            "refreshed_bundle_zip_file_count": refreshed_bundle["zip_file_count"],
            "bundle_zip": display_path(bundle_zip),
            "bundle_dir": display_path(bundle_dir),
            "reopen_trigger_ordering": [PRIMARY_TRIGGER, SECONDARY_TRIGGER, RESERVE_TRIGGER],
        },
    )
    write_artifact("expert_handoff_sync", handoff_payload)

    print("[done] 8.7.56.1467-.1470 artifacts generated")
    print(f"[done] restore_word_count={restore_word_count}")
    print(f"[done] bundle_zip={bundle_zip}")
    print(f"[done] next={NEXT_ROUTE_NAME}")


if __name__ == "__main__":
    main()
