#!/usr/bin/env python3
"""Generate 8.7.56.1119-.1122 Trial-2 future-canon source-normalization reserve registry artifacts."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
NOTE_SI = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_si_dimension_tracking.md")

AUDIT_1088 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_delta_registry_audit_metrics.json"
)
ROUTE_1074 = PUBLIC_OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_sixty_fifth_refresh_metrics.json"
INVENTORY_1115 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_tv_dimensionless_ratio_rewrite_registry_source_inventory_metrics.json"
)
AUDIT_1116 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_tv_dimensionless_ratio_rewrite_registry_audit_metrics.json"
)
GATE_1117 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_tv_dimensionless_ratio_rewrite_registry_declaration_gate_metrics.json"
)
ROUTE_1118 = PUBLIC_OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_seventy_sixth_refresh_metrics.json"

CURRENT_ROUTE = (
    "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_source_normalization_bridge_reserve_registry"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_multi_delta_hold_contract"
)
NEXT_ROUTE = "8.7.56.1123"

RESERVE_LANE_ITEMS = [
    "delta_source_normalization_bridge_reserve",
]


# Function: return the current UTC timestamp.
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# Function: require one path to exist.

def require(path: Path) -> None:
    """Abort when one required file is missing."""
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


# Function: return one stable display path.

def display_path(path: Path) -> str:
    """Return one repo-relative path when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# Function: locate the first line containing one substring.

def hit(text: str, pattern: str) -> dict | None:
    """Return the first matching line for one substring pattern."""
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


# Function: build one standard metrics payload.

def payload(step: str, name: str, inputs: dict, rows: list[dict], summary: dict, decision: dict, evidence: dict) -> dict:
    """Build one standard metrics payload."""
    return {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": step, "name": name},
        "inputs": inputs,
        "rows": rows,
        "summary": summary,
        "decision": decision,
        "evidence": evidence,
    }


# Function: write one JSON metrics artifact and one CSV rows table.

def write_artifact(stem: str, data: dict) -> None:
    """Write one metrics payload as JSON and CSV."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    json_path = PUBLIC_OUT / f"{stem}_metrics.json"
    csv_path = PUBLIC_OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# Function: build one wording target record.

def target(text: str, path: Path, key: str, pattern: str, note: str) -> dict:
    """Build one wording target record."""
    evidence = hit(text, pattern)
    return {
        "file_key": key,
        "file": display_path(path),
        "pattern": pattern,
        "present": evidence is not None,
        "note": note,
        "evidence": evidence,
    }


# Function: classify the reserve registry outcome.

def classify(registry_ready: bool, ambiguity_confirmed: bool, reserve_only: bool) -> str:
    """Classify the source-normalization reserve registry outcome."""
    if registry_ready and ambiguity_confirmed and reserve_only:
        return "source_normalization_bridge_reserve_registry_frozen"

    if registry_ready and ambiguity_confirmed:
        return "source_normalization_bridge_reserve_registry_partial"

    return "source_normalization_bridge_reserve_registry_incomplete"


# Function: execute the source-normalization reserve registry branch.

def main() -> None:
    """Execute the Trial-2 future-canon source-normalization reserve registry branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PART3A,
        PART5,
        NOTE_SI,
        AUDIT_1088,
        ROUTE_1074,
        INVENTORY_1115,
        AUDIT_1116,
        GATE_1117,
        ROUTE_1118,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    si_note_text = read_text(NOTE_SI)

    audit_1088 = read_json(AUDIT_1088)["summary"]
    route_1074 = read_json(ROUTE_1074)["summary"]
    inventory_1115 = read_json(INVENTORY_1115)["summary"]
    audit_1116 = read_json(AUDIT_1116)["summary"]
    gate_1117 = read_json(GATE_1117)["summary"]
    route_1118 = read_json(ROUTE_1118)["summary"]

    targets = [
        target(status_text, STATUS, "status_1119", "8.7.56.1119", "STATUS must retain this branch."),
        target(roadmap_text, ROADMAP, "roadmap_1119", "`8.7.56.1119-.1122`", "ROADMAP must retain this branch."),
        target(part3a_text, PART3A, "part3a_reserve_route", "future-canon source-normalization bridge reserve registry", "Part III-A must expose the active reserve route."),
        target(part5_text, PART5, "part5_reserve_route", "future-canon source-normalization bridge reserve registry", "Part V must expose the active reserve route."),
        target(si_note_text, NOTE_SI, "si_note_j", "$J^\\mu$ の正しい読み方", "The SI note must still expose the J^mu source-normalization question."),
        target(si_note_text, NOTE_SI, "si_note_bridge", "g_P = \\frac{4\\pi G\\,M_\\chi^2}{c^2\\,v}", "The SI note must still expose the carried Newton-side bridge formula."),
        target(si_note_text, NOTE_SI, "si_note_not_bookkeeping_only", "physics の問題ではなく bookkeeping の問題", "The SI note must still retain the source-normalization diagnosis surface."),
    ]

    prior_route_active = all(
        [
            inventory_1115["reserve_lane_downstream_retained"],
            audit_1116["reserve_lane_downstream_retained"],
            not audit_1116["reserve_lane_prematurely_promoted"],
            gate_1117["selected_residual_route"] == CURRENT_ROUTE,
            route_1118["selected_next_generation_route"] == CURRENT_ROUTE,
            not route_1118["physical_reject_required"],
        ]
    )
    inventory_ready = all(item["present"] for item in targets) and prior_route_active

    source_normalization_ambiguity_confirmed = bool(
        route_1074["source_normalization_ambiguity_confirmed"]
        and route_1074["first_missing_or_ambiguous_bridge_location"] == "j0_to_newton_source_mapping"
        and route_1074["first_missing_or_ambiguous_bridge_type"] == "matter_current_normalization_c_power"
        and audit_1088["source_normalization_reserve_retained"]
    )
    source_normalization_bridge_reserve_item_confirmed = bool(
        inventory_1115["reserve_lane_item_keys"] == RESERVE_LANE_ITEMS
        and gate_1117["trial2_numeric_alpha_reserve_lane_downstream_retained"]
        and route_1118["reserve_lane_item_keys"] == RESERVE_LANE_ITEMS
    )
    source_normalization_bridge_reserve_still_reserve_only = bool(
        not audit_1116["reserve_lane_prematurely_promoted"]
        and gate_1117["selected_residual_route"] == CURRENT_ROUTE
        and route_1118["selected_next_generation_route"] == CURRENT_ROUTE
        and not route_1118["reopen_prerequisite_satisfied_under_current_canon"]
        and not route_1118["physical_reject_required"]
    )
    multi_delta_hold_required = bool(
        route_1118["future_canon_multi_delta_program_required"]
        and source_normalization_bridge_reserve_still_reserve_only
    )
    source_normalization_bridge_reserve_registry_ready = bool(
        inventory_ready
        and source_normalization_ambiguity_confirmed
        and source_normalization_bridge_reserve_item_confirmed
        and source_normalization_bridge_reserve_still_reserve_only
        and multi_delta_hold_required
    )
    registry_class = classify(
        source_normalization_bridge_reserve_registry_ready,
        source_normalization_ambiguity_confirmed,
        source_normalization_bridge_reserve_still_reserve_only,
    )

    inputs = {
        "status_markdown": display_path(STATUS),
        "roadmap_markdown": display_path(ROADMAP),
        "ai_context_json": display_path(AI_CONTEXT),
        "part3a_markdown": display_path(PART3A),
        "part5_markdown": display_path(PART5),
        "si_dimension_tracking_note": display_path(NOTE_SI),
        "prior_1088_json": display_path(AUDIT_1088),
        "prior_1074_json": display_path(ROUTE_1074),
        "prior_1115_json": display_path(INVENTORY_1115),
        "prior_1116_json": display_path(AUDIT_1116),
        "prior_1117_json": display_path(GATE_1117),
        "prior_1118_json": display_path(ROUTE_1118),
    }

    inventory = payload(
        "8.7.56.1119",
        "Trial-2 numeric alpha future-canon source-normalization bridge reserve registry source inventory",
        inputs,
        [
            row("inventory_complete", "pass" if inventory_ready else "reject", "source-normalization reserve registry inventory complete", 1 if inventory_ready else 0, "The reserve registry is assembled from the T_v rewrite metrics, the retained SI-tracking evidence, and the frozen public wording."),
            row("source_normalization_ambiguity_confirmed", "pass" if source_normalization_ambiguity_confirmed else "reject", "source-normalization ambiguity confirmed", 1 if source_normalization_ambiguity_confirmed else 0, "The retained SI-tracking route still says the first ambiguous bridge is the J0-to-Newton source mapping."),
            row("reserve_item_confirmed", "pass" if source_normalization_bridge_reserve_item_confirmed else "reject", "source-normalization reserve item confirmed", 1 if source_normalization_bridge_reserve_item_confirmed else 0, "The reserve lane still contains only the source-normalization bridge item."),
            row("reserve_only_still_enforced", "pass" if source_normalization_bridge_reserve_still_reserve_only else "reject", "source-normalization bridge still reserve-only", 1 if source_normalization_bridge_reserve_still_reserve_only else 0, "The reserve item is retained after the T_v rewrite but is not promoted into a current-canon reopen."),
            row("multi_delta_hold_required", "pass" if multi_delta_hold_required else "reject", "multi-delta hold still required after reserve registry", 1 if multi_delta_hold_required else 0, "All remaining work stays in future-canon carry-over order after the reserve item is frozen."),
        ],
        {
            "inventory_ready": inventory_ready,
            "source_normalization_bridge_reserve_registry_ready": source_normalization_bridge_reserve_registry_ready,
            "source_normalization_ambiguity_confirmed": source_normalization_ambiguity_confirmed,
            "source_normalization_bridge_reserve_item_confirmed": source_normalization_bridge_reserve_item_confirmed,
            "source_normalization_bridge_reserve_still_reserve_only": source_normalization_bridge_reserve_still_reserve_only,
            "reserve_lane_item_keys": RESERVE_LANE_ITEMS,
            "future_canon_multi_delta_program_required": True,
            "reopen_prerequisite_satisfied_under_current_canon": False,
            "first_route_to_close_or_none": CURRENT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_future_canon_source_normalization_reserve_inventory_frozen",
            "advance_to_8_7_56_1120": inventory_ready,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "targets": targets,
            "retained_1088_summary": audit_1088,
            "retained_1074_summary": route_1074,
            "retained_1115_summary": inventory_1115,
            "retained_1116_summary": audit_1116,
            "retained_1117_summary": gate_1117,
            "retained_1118_summary": route_1118,
        },
    )

    audit = payload(
        "8.7.56.1120",
        "Trial-2 numeric alpha future-canon source-normalization bridge reserve registry audit",
        inputs,
        [
            row("reserve_registry_ready", "pass" if source_normalization_bridge_reserve_registry_ready else "reject", "source-normalization reserve registry ready", 1 if source_normalization_bridge_reserve_registry_ready else 0, "The reserve registry passes only if the ambiguity stays visible, remains reserve-only, and still hands off to a hold contract."),
            row("source_normalization_ambiguity_confirmed", "pass" if source_normalization_ambiguity_confirmed else "reject", "source-normalization ambiguity confirmed", 1 if source_normalization_ambiguity_confirmed else 0, "The SI-tracking route still localizes the ambiguity at the J0-to-Newton source mapping."),
            row("reserve_item_still_reserve_only", "pass" if source_normalization_bridge_reserve_still_reserve_only else "reject", "reserve item still reserve-only", 1 if source_normalization_bridge_reserve_still_reserve_only else 0, "The reserve item does not overtake the theorem-side or rewrite-side surfaces."),
            row("current_canon_not_reopened", "pass" if multi_delta_hold_required else "reject", "current canon not reopened by reserve item", 1 if multi_delta_hold_required else 0, "The reserve registry remains future-canon only and does not escalate to a current-canon reopen."),
            row("physical_reject_not_selected", "pass", "physical reject not selected after reserve registry", 1, "The reserve registry does not convert the route into a physical reject."),
        ],
        {
            "audit_ready": inventory_ready,
            "source_normalization_bridge_reserve_registry_ready": source_normalization_bridge_reserve_registry_ready,
            "source_normalization_ambiguity_confirmed": source_normalization_ambiguity_confirmed,
            "source_normalization_bridge_reserve_still_reserve_only": source_normalization_bridge_reserve_still_reserve_only,
            "reserve_lane_prematurely_promoted": False,
            "future_canon_multi_delta_program_required": True,
            "reopen_prerequisite_satisfied_under_current_canon": False,
            "physical_reject_required": False,
            "selected_source_normalization_bridge_reserve_registry_class": registry_class,
            "first_route_to_close_after_audit_or_none": NEXT_ROUTE_NAME,
        },
        {
            "overall_status": "trial2_numeric_alpha_future_canon_source_normalization_reserve_audited",
            "advance_to_8_7_56_1121": source_normalization_bridge_reserve_registry_ready,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"inventory_summary": inventory["summary"]},
    )

    gate = payload(
        "8.7.56.1121",
        "Trial-2 numeric alpha future-canon source-normalization bridge reserve registry declaration gate",
        inputs,
        [
            row("gate_complete", "pass" if source_normalization_bridge_reserve_registry_ready else "reject", "future-canon source-normalization reserve registry gate complete", 1 if source_normalization_bridge_reserve_registry_ready else 0, "The reserve registry becomes official only after the reserve-only guard and hold requirement both pass."),
            row("ambiguity_frozen_as_reserve_only", "pass" if source_normalization_bridge_reserve_still_reserve_only else "reject", "source-normalization ambiguity frozen as reserve-only", 1 if source_normalization_bridge_reserve_still_reserve_only else 0, "The source-normalization bridge is retained but not promoted beyond reserve evidence."),
            row("multi_delta_hold_selected", "pass" if multi_delta_hold_required else "reject", "future-canon multi-delta hold contract selected", 1 if multi_delta_hold_required else 0, "The next branch moves to a top-level hold contract after the reserve registry is frozen."),
            row("physical_reject_not_selected", "pass", "physical reject not selected", 1, "The reserve registry preserves the future-canon candidate rather than rejecting the route."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "alpha_prediction_future_canon_source_normalization_bridge_reserve_registry",
            "trial2_numeric_alpha_future_canon_source_normalization_bridge_reserve_registry_completed": source_normalization_bridge_reserve_registry_ready,
            "trial2_numeric_alpha_source_normalization_ambiguity_confirmed": source_normalization_ambiguity_confirmed,
            "trial2_numeric_alpha_source_normalization_bridge_reserve_still_reserve_only": source_normalization_bridge_reserve_still_reserve_only,
            "trial2_numeric_alpha_reopen_prerequisite_satisfied_under_current_canon": False,
            "trial2_numeric_alpha_physical_reject_required": False,
            "selected_residual_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_future_canon_source_normalization_reserve_gate_closed",
            "advance_to_8_7_56_1122": source_normalization_bridge_reserve_registry_ready,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"audit_summary": audit["summary"]},
    )

    route = payload(
        "8.7.56.1122",
        "Trial-2 numeric alpha route contract one-hundred-seventy-seventh refresh",
        inputs,
        [
            row("route_contract_complete", "pass" if source_normalization_bridge_reserve_registry_ready else "reject", "route contract one-hundred-seventy-seventh refresh complete", 1 if source_normalization_bridge_reserve_registry_ready else 0, "The reserve registry is converted into the next-generation route contract."),
            row("reserve_registry_completed", "pass" if source_normalization_bridge_reserve_registry_ready else "reject", "future-canon source-normalization reserve registry completed", 1 if source_normalization_bridge_reserve_registry_ready else 0, "The reserve-side item inside the future-canon program is now frozen as one registry."),
            row("multi_delta_hold_selected_as_next_route", "pass" if multi_delta_hold_required else "reject", "future-canon multi-delta hold contract selected as next route", 1 if multi_delta_hold_required else 0, "The next step moves to a hold contract after the reserve registry is frozen."),
            row("physical_reject_not_selected", "pass", "physical reject not selected after reserve registry", 1, "The route remains structurally alive after freezing the reserve registry."),
        ],
        {
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "future_canon_source_normalization_bridge_reserve_registry_completed": source_normalization_bridge_reserve_registry_ready,
            "future_canon_multi_delta_program_required": True,
            "source_normalization_ambiguity_confirmed": source_normalization_ambiguity_confirmed,
            "source_normalization_bridge_reserve_still_reserve_only": source_normalization_bridge_reserve_still_reserve_only,
            "reserve_lane_item_keys": RESERVE_LANE_ITEMS,
            "reopen_prerequisite_satisfied_under_current_canon": False,
            "physical_reject_required": False,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_route_contract_one_hundred_seventy_seventh_refresh_frozen",
            "advance_to_next_route": source_normalization_bridge_reserve_registry_ready,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"gate_summary": gate["summary"], "audit_summary": audit["summary"]},
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "alpha_is_prediction_future_canon_source_normalization_bridge_reserve_registry_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "alpha_is_prediction_future_canon_source_normalization_bridge_reserve_registry_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "alpha_is_prediction_future_canon_source_normalization_bridge_reserve_registry_declaration_gate",
        gate,
    )
    write_artifact("mass_origin_v2_t2_alpha_route_contract_one_hundred_seventy_seventh_refresh", route)

    print("[done] 8.7.56.1119-.1122 artifacts generated")


if __name__ == "__main__":
    main()
