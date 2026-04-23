#!/usr/bin/env python3
"""Generate 8.7.56.1459-.1462 unified-closure future reopen-trigger handoff final summary artifacts."""

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
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

UNIFIED_PLAN = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_unified_closure_plan_20260327.md")
NEXT_STEPS = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_next_steps_20260327.md")

HANDOFF_CLOSEOUT_INV = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_handoff_closeout_sync_"
    "source_inventory_metrics.json"
)
HANDOFF_CLOSEOUT_AUDIT = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_handoff_closeout_sync_"
    "audit_metrics.json"
)
HANDOFF_CLOSEOUT_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_handoff_closeout_sync_"
    "declaration_gate_metrics.json"
)
HANDOFF_CLOSEOUT_SYNC = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_handoff_closeout_sync_"
    "expert_handoff_sync_metrics.json"
)
PHASE1_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_phase1_exact_coupled_l0_solver_"
    "numeric_evaluation_metrics.json"
)
PHASE2_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_lambda_rot_form_factor_correction_"
    "numeric_evaluation_metrics.json"
)
PHASE3_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_blind_vector_observable_gate_"
    "numeric_evaluation_metrics.json"
)

SCRIPT_1455 = ROOT / "scripts" / "quantum" / "t2a_1455.py"

PRIOR_CLASS = "vector_qball_form_factor_unified_closure_future_reopen_trigger_handoff_closeout_sync_completed"
BRANCH_CLASS = "vector_qball_form_factor_unified_closure_future_reopen_trigger_handoff_final_summary_completed"
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_handoff_archive_registry"
)
NEXT_ROUTE = "8.7.56.1463"
STEM = "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_handoff_final_summary_route"

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


# Function: build one concise handoff final-summary sentence.

def build_handoff_final_summary_text(phase1_eval: dict, phase3_eval: dict) -> str:
    """Build one concise handoff final-summary sentence."""
    return (
        "Case C remains an honest partial handoff: fixed-q_theory blind vector no-go stays at "
        f"F(q_theory)={phase3_eval['blind_F_at_q_theory']} and alpha(q_theory)={phase3_eval['blind_alpha_at_q_theory']}, "
        "the retained scalar baseline stays at "
        f"F_exact(q_theory)={phase1_eval['F_exact_at_q_theory_scalar']} and alpha_exact(q_theory)="
        f"{phase1_eval['alpha_exact_at_q_theory_scalar']}, and future reopen triggers remain ordered as "
        f"{PRIMARY_TRIGGER}, {SECONDARY_TRIGGER}, {RESERVE_TRIGGER} while physical_reject_required=false."
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


# Function: execute the unified-closure future reopen-trigger handoff final summary route.

def main() -> None:
    """Execute the unified-closure future reopen-trigger handoff final summary route."""
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
        PART1,
        PART3A,
        PART5,
        UNIFIED_PLAN,
        NEXT_STEPS,
        HANDOFF_CLOSEOUT_INV,
        HANDOFF_CLOSEOUT_AUDIT,
        HANDOFF_CLOSEOUT_GATE,
        HANDOFF_CLOSEOUT_SYNC,
        PHASE1_EVAL,
        PHASE2_EVAL,
        PHASE3_EVAL,
        SCRIPT_1455,
        Path(__file__),
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    expert_share_text = read_text(EXPERT_SHARE)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    unified_plan_text = read_text(UNIFIED_PLAN)
    next_steps_text = read_text(NEXT_STEPS)

    handoff_closeout_inv = read_json(HANDOFF_CLOSEOUT_INV)["summary"]
    handoff_closeout_audit = read_json(HANDOFF_CLOSEOUT_AUDIT)["summary"]
    handoff_closeout_gate = read_json(HANDOFF_CLOSEOUT_GATE)["summary"]
    handoff_closeout_sync = read_json(HANDOFF_CLOSEOUT_SYNC)["summary"]
    phase1_eval = read_json(PHASE1_EVAL)["summary"]
    phase2_eval = read_json(PHASE2_EVAL)["summary"]
    phase3_eval = read_json(PHASE3_EVAL)["summary"]

    bundle_zip = ROOT / handoff_closeout_sync["share_pack_bundle_zip"]
    bundle_dir = bundle_zip.with_suffix("")
    require(bundle_zip)
    require(bundle_dir)

    final_summary_hits_required = [
        hit(status_text, "handoff final summary"),
        hit(roadmap_text, "`8.7.56.1459-.1462`"),
        hit(current_problem_text, "future reopen-trigger ordering"),
        hit(current_status_text, "future reopen trigger ordering"),
        hit(expert_share_text, "current official state:"),
        hit(unified_roadmap_text, "future reopen-trigger handoff final summary"),
        hit(part3a_text, "handoff-final-summary"),
        hit(part5_text, "reopen-trigger handoff final summary"),
    ]
    final_summary_hits_optional = [
        hit(unified_plan_text, "Case C"),
        hit(next_steps_text, "source theorem"),
    ]
    inventory_ready = all(item is not None for item in final_summary_hits_required)

    share_pack_bundle_available = bool(bundle_zip.exists() and bundle_dir.exists())
    prior_route_ready = bool(
        handoff_closeout_gate["selected_next_generation_route"]
        == "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_handoff_final_summary_route"
    )
    handoff_closeout_sync_ready = bool(
        handoff_closeout_audit["future_reopen_trigger_handoff_closeout_wording_honest"]
        and prior_route_ready
        and handoff_closeout_sync["share_pack_bundle_refresh_count"] > 0
        and share_pack_bundle_available
    )
    case_c_honest_partial_retained = bool(handoff_closeout_audit["case_c_honest_partial_retained"])
    retained_scalar_candidate_retained = bool(handoff_closeout_audit["retained_scalar_strong_candidate_retained"])
    blind_vector_no_go_retained = bool(handoff_closeout_audit["blind_vector_observable_no_go_retained"])
    physical_reject_not_selected = not bool(handoff_closeout_gate["physical_reject_required"])
    ordering_retained = bool(
        handoff_closeout_audit["primary_future_reopen_trigger"] == PRIMARY_TRIGGER
        and handoff_closeout_audit["secondary_future_reopen_trigger"] == SECONDARY_TRIGGER
        and handoff_closeout_audit["reserve_future_reopen_trigger"] == RESERVE_TRIGGER
        and handoff_closeout_gate["primary_future_reopen_trigger"] == PRIMARY_TRIGGER
        and handoff_closeout_gate["secondary_future_reopen_trigger"] == SECONDARY_TRIGGER
        and handoff_closeout_gate["reserve_future_reopen_trigger"] == RESERVE_TRIGGER
        and handoff_closeout_sync["primary_future_reopen_trigger"] == PRIMARY_TRIGGER
        and handoff_closeout_sync["secondary_future_reopen_trigger"] == SECONDARY_TRIGGER
        and handoff_closeout_sync["reserve_future_reopen_trigger"] == RESERVE_TRIGGER
    )
    final_summary_text = build_handoff_final_summary_text(phase1_eval, phase3_eval)
    final_summary_word_count = len(final_summary_text.split())
    final_summary_concise = final_summary_word_count <= 80
    wording_honest = all(
        [
            final_summary_concise,
            case_c_honest_partial_retained,
            retained_scalar_candidate_retained,
            blind_vector_no_go_retained,
            physical_reject_not_selected,
            ordering_retained,
            handoff_closeout_sync_ready,
        ]
    )
    final_summary_ready = all(
        [
            inventory_ready,
            handoff_closeout_sync_ready,
            share_pack_bundle_available,
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
            "part1": display_path(PART1),
            "part3a": display_path(PART3A),
            "part5": display_path(PART5),
            "unified_plan_note": display_path(UNIFIED_PLAN),
            "next_steps_note": display_path(NEXT_STEPS),
        },
        "source_metrics": {
            "handoff_closeout_inventory": display_path(HANDOFF_CLOSEOUT_INV),
            "handoff_closeout_audit": display_path(HANDOFF_CLOSEOUT_AUDIT),
            "handoff_closeout_gate": display_path(HANDOFF_CLOSEOUT_GATE),
            "handoff_closeout_sync": display_path(HANDOFF_CLOSEOUT_SYNC),
            "phase1_eval": display_path(PHASE1_EVAL),
            "phase2_eval": display_path(PHASE2_EVAL),
            "phase3_eval": display_path(PHASE3_EVAL),
        },
        "scripts": {
            "future_reopen_trigger_handoff_closeout_sync": display_path(SCRIPT_1455),
            "future_reopen_trigger_handoff_final_summary_route": display_path(Path(__file__)),
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
        "8.7.56.1459",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure future reopen-trigger handoff final summary inventory",
        common_inputs,
        [
            row(
                "inventory_ready",
                "pass" if inventory_ready else "reject",
                "future reopen-trigger handoff final summary inventory ready",
                1 if inventory_ready else 0,
                "The handoff final-summary inventory is ready only if the handoff-closeout metrics, canonical bundle, paper wording, and synced notes coexist in one pack.",
            ),
            row(
                "future_reopen_trigger_handoff_closeout_sync_available",
                "pass" if handoff_closeout_sync_ready else "reject",
                "future reopen-trigger handoff closeout sync available for final summary",
                1 if handoff_closeout_sync_ready else 0,
                "The handoff final-summary route starts only after the handoff closeout wording and expert handoff are already frozen.",
            ),
            row(
                "share_pack_bundle_available",
                "pass" if share_pack_bundle_available else "reject",
                "canonical share-pack bundle available for handoff final summary",
                1 if share_pack_bundle_available else 0,
                "The handoff final-summary route reuses the canonical expert-share bundle produced by the prior handoff closeout sync branch.",
            ),
            row(
                "ordering_retained",
                "pass" if ordering_retained else "reject",
                "future reopen ordering retained for handoff final summary",
                1 if ordering_retained else 0,
                "The handoff final-summary route is honest only if it keeps the exact operator / future source theorem / observable dictionary ordering unchanged.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "prior_problem_classification": PRIOR_CLASS,
            "inventory_ready": inventory_ready,
            "future_reopen_trigger_handoff_closeout_sync_available": handoff_closeout_sync_ready,
            "share_pack_bundle_available": share_pack_bundle_available,
            "share_pack_bundle_zip": display_path(bundle_zip),
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_handoff_final_summary_inventory_fixed",
            "advance_to_8_7_56_1460": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "final_summary_hits_required": final_summary_hits_required,
            "final_summary_hits_optional": final_summary_hits_optional,
            "final_summary_text": final_summary_text,
            "final_summary_word_count": final_summary_word_count,
            "phase1_best_vector_alpha": phase1_eval["phase1_best_alpha_candidate"]["alpha_at_q_theory"],
            "phase2_best_additive_alpha": phase2_eval["best_naive_add_3sigma_alpha"],
            "phase3_blind_alpha_at_q_theory": phase3_eval["blind_alpha_at_q_theory"],
        },
    )
    write_artifact("source_inventory", inventory_payload)

    audit_payload = payload(
        "8.7.56.1460",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure future reopen-trigger handoff final summary wording audit",
        common_inputs,
        [
            row(
                "future_reopen_trigger_handoff_final_summary_wording_honest",
                "pass" if wording_honest else "reject",
                "future reopen-trigger handoff final summary wording honest",
                1 if wording_honest else 0,
                "The handoff final-summary wording is honest only if it keeps Case C honest partial, the retained scalar strong baseline, the blind vector no-go, the frozen reopen ordering, and physical_reject_required=false without overreach.",
            ),
            row(
                "future_reopen_trigger_handoff_final_summary_wording_concise",
                "pass" if final_summary_concise else "reject",
                "future reopen-trigger handoff final summary wording concise",
                1 if final_summary_concise else 0,
                "The handoff final-summary route should compress the frozen reopen state into one short handoff sentence rather than reopening theorem-side detail.",
            ),
            row(
                "case_c_honest_partial_retained",
                "pass" if case_c_honest_partial_retained else "reject",
                "Case C honest partial retained in handoff final summary",
                1 if case_c_honest_partial_retained else 0,
                "The handoff final-summary wording must preserve the Case C honest-partial disposition.",
            ),
            row(
                "retained_scalar_strong_candidate_retained",
                "pass" if retained_scalar_candidate_retained else "reject",
                "retained scalar strong candidate retained in handoff final summary",
                1 if retained_scalar_candidate_retained else 0,
                "The retained scalar baseline must stay visible so the handoff does not read like a full reject.",
            ),
            row(
                "future_reopen_ordering_retained",
                "pass" if ordering_retained else "reject",
                "future reopen ordering retained in handoff final summary",
                1 if ordering_retained else 0,
                "The handoff final-summary wording must preserve the primary / secondary / reserve reopen ordering verbatim.",
            ),
            row(
                "blind_vector_observable_no_go_retained",
                "pass" if blind_vector_no_go_retained else "reject",
                "blind vector observable no-go retained in handoff final summary",
                1 if blind_vector_no_go_retained else 0,
                "The fixed-q_theory blind vector failure remains part of the handoff final-summary wording and is not diluted into a generic hold state.",
            ),
            row(
                "physical_reject_not_selected",
                "pass" if physical_reject_not_selected else "reject",
                "physical reject not selected in handoff final summary",
                1 if physical_reject_not_selected else 0,
                "The handoff final-summary wording remains a retained reopen state and must keep physical_reject_required=false.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "prior_problem_classification": PRIOR_CLASS,
            "future_reopen_trigger_handoff_final_summary_wording_honest": wording_honest,
            "future_reopen_trigger_handoff_final_summary_wording_concise": final_summary_concise,
            "case_c_honest_partial_retained": case_c_honest_partial_retained,
            "retained_scalar_strong_candidate_retained": retained_scalar_candidate_retained,
            "blind_vector_observable_no_go_retained": blind_vector_no_go_retained,
            "primary_future_reopen_trigger": PRIMARY_TRIGGER,
            "secondary_future_reopen_trigger": SECONDARY_TRIGGER,
            "reserve_future_reopen_trigger": RESERVE_TRIGGER,
            "physical_reject_required": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_handoff_final_summary_audited",
            "advance_to_8_7_56_1461": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "final_summary_text": final_summary_text,
            "final_summary_word_count": final_summary_word_count,
            "retained_scalar_alpha_exact_at_q_theory": phase1_eval["alpha_exact_at_q_theory_scalar"],
            "best_phase1_vector_alpha": phase1_eval["phase1_best_alpha_candidate"]["alpha_at_q_theory"],
            "phase3_blind_signed_crossing_over_m0": phase3_eval["signed_target_crossing_over_m0"],
        },
    )
    write_artifact("audit", audit_payload)

    gate_payload = payload(
        "8.7.56.1461",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure future reopen-trigger handoff final summary declaration gate",
        common_inputs,
        [
            row(
                "future_reopen_trigger_handoff_final_summary_ready",
                "pass" if final_summary_ready else "reject",
                "future reopen-trigger handoff final summary ready",
                1 if final_summary_ready else 0,
                "The handoff final summary can be declared only after the inventory, wording audit, handoff-closeout sync, and frozen reopen ordering all agree.",
            ),
            row(
                "future_reopen_trigger_handoff_final_summary_fixed",
                "pass" if final_summary_ready else "reject",
                "future reopen-trigger handoff final summary fixed",
                1 if final_summary_ready else 0,
                "The declaration gate freezes the concise handoff final-summary wording for the reopen-trigger state without reopening any vector computation lane.",
            ),
            row(
                "physical_reject_not_selected",
                "pass" if physical_reject_not_selected else "reject",
                "physical reject not selected after handoff final summary",
                1 if physical_reject_not_selected else 0,
                "The handoff final-summary route keeps the retained reopen reading and does not escalate to physical reject.",
            ),
            row(
                "next_route_selected",
                "pass" if final_summary_ready else "reject",
                "future reopen-trigger handoff archive registry selected as next route",
                1 if final_summary_ready else 0,
                "After the handoff final summary is fixed, the next work is an archive registry for the frozen reopen-trigger handoff state.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "prior_problem_classification": PRIOR_CLASS,
            "future_reopen_trigger_handoff_final_summary_ready": final_summary_ready,
            "primary_future_reopen_trigger": PRIMARY_TRIGGER,
            "secondary_future_reopen_trigger": SECONDARY_TRIGGER,
            "reserve_future_reopen_trigger": RESERVE_TRIGGER,
            "physical_reject_required": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_handoff_final_summary_gate_closed",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "final_summary_text": final_summary_text,
            "retained_scalar_alpha_exact_at_q_theory": phase1_eval["alpha_exact_at_q_theory_scalar"],
            "best_phase1_vector_alpha": phase1_eval["phase1_best_alpha_candidate"]["alpha_at_q_theory"],
            "blind_vector_alpha_at_q_theory": phase3_eval["blind_alpha_at_q_theory"],
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
        PART1,
        PART3A,
        PART5,
        UNIFIED_PLAN,
        NEXT_STEPS,
        HANDOFF_CLOSEOUT_INV,
        HANDOFF_CLOSEOUT_AUDIT,
        HANDOFF_CLOSEOUT_GATE,
        HANDOFF_CLOSEOUT_SYNC,
        PHASE1_EVAL,
        PHASE2_EVAL,
        PHASE3_EVAL,
        SCRIPT_1455,
        Path(__file__),
        PUBLIC_OUT / f"{STEM}_source_inventory_metrics.json",
        PUBLIC_OUT / f"{STEM}_audit_metrics.json",
        PUBLIC_OUT / f"{STEM}_declaration_gate_metrics.json",
    ]
    refreshed_bundle = refresh_handoff_bundle(bundle_dir, bundle_zip, refresh_candidates)

    handoff_payload = payload(
        "8.7.56.1462",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure future reopen-trigger handoff final summary expert handoff sync",
        common_inputs,
        [
            row(
                "future_reopen_trigger_handoff_final_summary_handoff_sync_complete",
                "pass" if final_summary_ready else "reject",
                "future reopen-trigger handoff final summary handoff sync complete",
                1 if final_summary_ready else 0,
                "Handoff sync is complete only when the handoff final-summary route is ready and the canonical bundle has been refreshed with the synced summary files.",
            ),
            row(
                "share_pack_bundle_retained",
                "pass" if share_pack_bundle_available else "reject",
                "canonical share-pack bundle retained for handoff final summary",
                1 if share_pack_bundle_available else 0,
                "The handoff final-summary route reuses the same canonical Case C / reopen-trigger bundle.",
            ),
            row(
                "share_pack_bundle_refreshed",
                "pass" if refreshed_bundle["copied_count"] > 0 else "reject",
                "canonical share-pack bundle refreshed with handoff final-summary files",
                refreshed_bundle["copied_count"],
                "The handoff final-summary route refreshes the canonical bundle in place so expert readers see the active handoff-final-summary wording and route state.",
            ),
            row(
                "expert_handoff_note_sync_ready",
                "pass" if inventory_ready else "reject",
                "expert handoff notes sync ready",
                1 if inventory_ready else 0,
                "The expert handoff is ready because the handoff-closeout sync, handoff-final-summary wording, and current notes now point to the same reopen state.",
            ),
            row(
                "future_reopen_trigger_handoff_archive_registry_selected",
                "pass" if final_summary_ready else "reject",
                "future reopen-trigger handoff archive registry selected",
                1 if final_summary_ready else 0,
                "After the handoff final summary is frozen, the route advances to an archive registry for the retained reopen-trigger handoff state.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "prior_problem_classification": PRIOR_CLASS,
            "future_reopen_trigger_handoff_final_summary_handoff_sync_complete": final_summary_ready,
            "share_pack_bundle_zip": display_path(bundle_zip),
            "share_pack_bundle_refresh_count": refreshed_bundle["copied_count"],
            "primary_future_reopen_trigger": PRIMARY_TRIGGER,
            "secondary_future_reopen_trigger": SECONDARY_TRIGGER,
            "reserve_future_reopen_trigger": RESERVE_TRIGGER,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_handoff_final_summary_handoff_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "final_summary_text": final_summary_text,
            "refreshed_bundle_copied_files": [display_path(path) for path in refreshed_bundle["copied_files"]],
            "refreshed_bundle_staging_file_count": refreshed_bundle["staging_file_count"],
            "refreshed_bundle_zip_file_count": refreshed_bundle["zip_file_count"],
            "bundle_zip": display_path(bundle_zip),
            "bundle_dir": display_path(bundle_dir),
            "reopen_trigger_ordering": [PRIMARY_TRIGGER, SECONDARY_TRIGGER, RESERVE_TRIGGER],
        },
    )
    write_artifact("expert_handoff_sync", handoff_payload)

    print("[done] 8.7.56.1459-.1462 artifacts generated")
    print(f"[done] final_summary_word_count={final_summary_word_count}")
    print(f"[done] bundle_zip={bundle_zip}")
    print(f"[done] next={NEXT_ROUTE_NAME}")


if __name__ == "__main__":
    main()
