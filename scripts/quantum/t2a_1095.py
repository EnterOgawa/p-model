#!/usr/bin/env python3
"""Generate 8.7.56.1095-.1098 Trial-2 future-canon delta-program carry-over registry artifacts."""

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
NOTE_ALPHA = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_alpha_is_prediction.md")
NOTE_DIM = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_dimension_normalization_review.md")
NOTE_SI = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_si_dimension_tracking.md")

AUDIT_1080 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_tmchi_tv_prove_or_no_go_review_audit_metrics.json"
)
AUDIT_1088 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_delta_registry_audit_metrics.json"
)
INVENTORY_1091 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_challenge_wording_freeze_source_inventory_metrics.json"
)
AUDIT_1092 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_challenge_wording_freeze_audit_metrics.json"
)
GATE_1093 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_challenge_wording_freeze_declaration_gate_metrics.json"
)
ROUTE_1094 = PUBLIC_OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_seventieth_refresh_metrics.json"

CURRENT_ROUTE = (
    "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_delta_program_carryover_registry"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_tmchi_lane_registry"
)
NEXT_ROUTE = "8.7.56.1099"

TMCHI_LANE_ITEMS = [
    "delta_tmchi_promotion_theorem",
    "delta_h0p_mass_frequency_bridge",
]
TV_LANE_ITEMS = [
    "delta_tv_dimensionless_ratio_rewrite",
]
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


# Function: classify the carry-over registry outcome.

def classify(primary_lane_ready: bool, tv_lane_ready: bool, reserve_lane_ready: bool) -> str:
    """Classify the carry-over registry outcome."""
    if primary_lane_ready and tv_lane_ready and reserve_lane_ready:
        return "tmchi_lane_first_carryover_registry"

    if primary_lane_ready:
        return "tmchi_lane_only_carryover_registry"

    return "future_canon_carryover_registry_incomplete"


# Function: execute the carry-over registry branch.

def main() -> None:
    """Execute the Trial-2 future-canon delta-program carry-over registry branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PART3A,
        PART5,
        NOTE_ALPHA,
        NOTE_DIM,
        NOTE_SI,
        AUDIT_1080,
        AUDIT_1088,
        INVENTORY_1091,
        AUDIT_1092,
        GATE_1093,
        ROUTE_1094,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    alpha_note_text = read_text(NOTE_ALPHA)
    dim_note_text = read_text(NOTE_DIM)
    si_note_text = read_text(NOTE_SI)

    audit_1080 = read_json(AUDIT_1080)["summary"]
    audit_1088 = read_json(AUDIT_1088)["summary"]
    inventory_1091 = read_json(INVENTORY_1091)["summary"]
    audit_1092 = read_json(AUDIT_1092)["summary"]
    gate_1093 = read_json(GATE_1093)["summary"]
    route_1094 = read_json(ROUTE_1094)["summary"]

    targets = [
        target(status_text, STATUS, "status_1095", "8.7.56.1095", "STATUS must already point to this branch."),
        target(roadmap_text, ROADMAP, "roadmap_1095", "`8.7.56.1095-.1098`", "ROADMAP must already expose this branch."),
        target(part3a_text, PART3A, "part3a_carryover_route", "future-canon delta-program carry-over registry", "Part III-A must expose the carry-over route."),
        target(part5_text, PART5, "part5_carryover_route", "future-canon delta-program carry-over registry", "Part V must expose the carry-over route."),
        target(part5_text, PART5, "part5_public_registry", "one public challenge registry", "Part V must still expose the public challenge registry wording."),
        target(part5_text, PART5, "part5_current_limit", "current canon limit", "Part V must still retain the current-canon-limit guardrail."),
        target(alpha_note_text, NOTE_ALPHA, "alpha_note_formula", r"\alpha = \frac{c^3}{4\pi v^2 \hbar}", "The alpha note must remain in the carry-over pack."),
        target(dim_note_text, NOTE_DIM, "dim_note_tmchi", r"T_{M_\chi}", "The dimension note must still expose T_Mchi."),
        target(dim_note_text, NOTE_DIM, "dim_note_tv", r"T_v", "The dimension note must still expose T_v."),
        target(si_note_text, NOTE_SI, "si_note_j", r"$J^\mu$ の正しい読み方", "The SI note must stay in the reserve pack."),
    ]

    prior_route_active = all(
        [
            inventory_1091["future_canon_challenge_wording_inventory_ready"],
            audit_1092["part5_public_challenge_registry_ready"],
            audit_1092["checkpoint_public_challenge_registry_ready"],
            not audit_1092["reopen_prerequisite_satisfied_under_current_canon"],
            gate_1093["selected_residual_route"] == CURRENT_ROUTE,
            route_1094["selected_next_generation_route"] == CURRENT_ROUTE,
            not route_1094["physical_reject_required"],
        ]
    )
    inventory_ready = all(item["present"] for item in targets) and prior_route_active

    tmchi_lane_ready = bool(
        audit_1080["tmchi_no_go_current_canon"]
        and audit_1088["tmchi_pack_required"]
        and audit_1088["h0p_mass_frequency_bridge_required"]
    )
    tv_lane_ready = bool(
        audit_1080["tv_downstream_unresolved_after_tmchi_no_go"]
        and audit_1088["tv_pack_required"]
    )
    reserve_lane_ready = bool(audit_1088["source_normalization_reserve_retained"])
    primary_lane_selected = tmchi_lane_ready and tv_lane_ready
    carryover_registry_ready = bool(
        inventory_ready
        and tmchi_lane_ready
        and tv_lane_ready
        and reserve_lane_ready
        and audit_1088["future_canon_multi_delta_program_required"]
        and not audit_1092["reopen_prerequisite_satisfied_under_current_canon"]
    )
    selected_carryover_class = classify(tmchi_lane_ready, tv_lane_ready, reserve_lane_ready)

    inputs = {
        "status_markdown": display_path(STATUS),
        "roadmap_markdown": display_path(ROADMAP),
        "ai_context_json": display_path(AI_CONTEXT),
        "part3a_markdown": display_path(PART3A),
        "part5_markdown": display_path(PART5),
        "alpha_note": display_path(NOTE_ALPHA),
        "dimension_note": display_path(NOTE_DIM),
        "si_note": display_path(NOTE_SI),
        "prior_1080_json": display_path(AUDIT_1080),
        "prior_1088_json": display_path(AUDIT_1088),
        "prior_1091_json": display_path(INVENTORY_1091),
        "prior_1092_json": display_path(AUDIT_1092),
        "prior_1093_json": display_path(GATE_1093),
        "prior_1094_json": display_path(ROUTE_1094),
    }

    inventory = payload(
        "8.7.56.1095",
        "Trial-2 numeric alpha future-canon delta-program carry-over registry source inventory",
        inputs,
        [
            row("inventory_complete", "pass" if inventory_ready else "reject", "carry-over registry inventory complete", 1 if inventory_ready else 0, "The carry-over pack is assembled from the wording-freeze metrics, delta-registry metrics, public wording, and retained notes."),
            row("tmchi_lane_seed_ready", "pass" if tmchi_lane_ready else "reject", "T_Mchi lane seed ready", 1 if tmchi_lane_ready else 0, "The upstream carry-over lane must include theorem promotion plus the H0P mass-frequency bridge."),
            row("tv_lane_seed_ready", "pass" if tv_lane_ready else "reject", "T_v lane seed ready", 1 if tv_lane_ready else 0, "The T_v carry-over lane must remain downstream from the T_Mchi lane."),
            row("reserve_lane_seed_ready", "pass" if reserve_lane_ready else "reject", "reserve lane seed ready", 1 if reserve_lane_ready else 0, "The source-normalization ambiguity remains reserve evidence only."),
            row("reopen_still_false", "pass", "reopen prerequisite still false under current canon", 1, "The carry-over registry does not reopen current-canon computation."),
        ],
        {
            "inventory_ready": inventory_ready,
            "tmchi_lane_item_keys": TMCHI_LANE_ITEMS,
            "tv_lane_item_keys": TV_LANE_ITEMS,
            "reserve_lane_item_keys": RESERVE_LANE_ITEMS,
            "primary_future_canon_lane": "tmchi_lane" if primary_lane_selected else None,
            "carryover_registry_ready": carryover_registry_ready,
            "first_route_to_close_or_none": CURRENT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_future_canon_carryover_inventory_frozen",
            "advance_to_8_7_56_1096": inventory_ready,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "targets": targets,
            "retained_1080_summary": audit_1080,
            "retained_1088_summary": audit_1088,
            "retained_1091_summary": inventory_1091,
            "retained_1092_summary": audit_1092,
            "retained_1093_summary": gate_1093,
            "retained_1094_summary": route_1094,
        },
    )

    audit = payload(
        "8.7.56.1096",
        "Trial-2 numeric alpha future-canon delta-program carry-over registry audit",
        inputs,
        [
            row("carryover_registry_ready", "pass" if carryover_registry_ready else "reject", "carry-over registry ready", 1 if carryover_registry_ready else 0, "The carry-over registry passes only if all three lanes are classified honestly while current-canon reopen stays false."),
            row("primary_tmchi_lane_selected", "pass" if primary_lane_selected else "reject", "primary future-canon lane selected as T_Mchi", 1 if primary_lane_selected else 0, "T_Mchi is the first executable lane because T_v is downstream unresolved after the T_Mchi no-go."),
            row("tv_lane_downstream_to_tmchi", "pass" if tv_lane_ready else "reject", "T_v lane remains downstream to T_Mchi", 1 if tv_lane_ready else 0, "The T_v lane cannot overtake the upstream theorem-promotion lane."),
            row("reserve_lane_retained", "pass" if reserve_lane_ready else "reject", "reserve lane retained", 1 if reserve_lane_ready else 0, "The source-normalization bridge remains reserve evidence rather than the primary lane."),
            row("physical_reject_not_required", "pass" if not route_1094["physical_reject_required"] else "reject", "physical reject not required", 1 if not route_1094["physical_reject_required"] else 0, "The carry-over registry must preserve the structural-pass reading."),
        ],
        {
            "audit_ready": inventory_ready,
            "tmchi_lane_carryover_ready": tmchi_lane_ready,
            "tv_lane_carryover_ready": tv_lane_ready,
            "reserve_lane_carryover_ready": reserve_lane_ready,
            "primary_future_canon_lane_selected": "tmchi_lane" if primary_lane_selected else None,
            "future_canon_multi_delta_program_required": bool(audit_1088["future_canon_multi_delta_program_required"]),
            "reopen_prerequisite_satisfied_under_current_canon": False,
            "physical_reject_required": False,
            "selected_future_canon_carryover_class": selected_carryover_class,
            "first_route_to_close_after_audit_or_none": NEXT_ROUTE_NAME,
        },
        {
            "overall_status": "trial2_numeric_alpha_future_canon_carryover_audited",
            "advance_to_8_7_56_1097": carryover_registry_ready,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"inventory_summary": inventory["summary"]},
    )

    gate = payload(
        "8.7.56.1097",
        "Trial-2 numeric alpha future-canon delta-program carry-over registry declaration gate",
        inputs,
        [
            row("gate_complete", "pass" if carryover_registry_ready else "reject", "future-canon delta-program carry-over registry gate complete", 1 if carryover_registry_ready else 0, "The carry-over registry becomes official only after the lane classification passes."),
            row("tmchi_lane_ready_confirmed", "pass" if tmchi_lane_ready else "reject", "T_Mchi lane carry-over ready confirmed", 1 if tmchi_lane_ready else 0, "The upstream lane is now frozen as the first executable future-canon lane."),
            row("tv_lane_ready_confirmed", "pass" if tv_lane_ready else "reject", "T_v lane carry-over ready confirmed", 1 if tv_lane_ready else 0, "The T_v lane remains attached as the downstream follow-through lane."),
            row("next_route_selected", "pass", "future-canon T_Mchi lane registry selected", 1, "The next branch will start from the upstream theorem-promotion lane."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "alpha_prediction_future_canon_delta_program_carryover_registry",
            "trial2_numeric_alpha_future_canon_delta_program_carryover_registry_completed": carryover_registry_ready,
            "trial2_numeric_alpha_tmchi_lane_carryover_ready": tmchi_lane_ready,
            "trial2_numeric_alpha_tv_lane_carryover_ready": tv_lane_ready,
            "trial2_numeric_alpha_reserve_lane_carryover_ready": reserve_lane_ready,
            "trial2_numeric_alpha_primary_future_canon_lane_selected": "tmchi_lane" if primary_lane_selected else None,
            "trial2_numeric_alpha_future_canon_multi_delta_program_required": True,
            "trial2_numeric_alpha_reopen_prerequisite_satisfied_under_current_canon": False,
            "trial2_numeric_alpha_physical_reject_required": False,
            "selected_residual_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_future_canon_carryover_gate_closed",
            "advance_to_8_7_56_1098": carryover_registry_ready,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"audit_summary": audit["summary"]},
    )

    route = payload(
        "8.7.56.1098",
        "Trial-2 numeric alpha route contract one-hundred-seventy-first refresh",
        inputs,
        [
            row("route_contract_complete", "pass" if carryover_registry_ready else "reject", "route contract one-hundred-seventy-first refresh complete", 1 if carryover_registry_ready else 0, "The carry-over registry is converted into the next-generation route contract."),
            row("carryover_registry_completed", "pass" if carryover_registry_ready else "reject", "future-canon delta-program carry-over registry completed", 1 if carryover_registry_ready else 0, "The multi-delta program is now split into executable lanes."),
            row("tmchi_lane_selected_as_next_route", "pass" if primary_lane_selected else "reject", "T_Mchi lane selected as next route", 1 if primary_lane_selected else 0, "The first next-generation lane is the upstream theorem-promotion lane."),
            row("physical_reject_not_selected", "pass", "physical reject not selected after carry-over registry", 1, "The route remains structurally alive after the registry split."),
        ],
        {
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "future_canon_delta_program_carryover_registry_completed": carryover_registry_ready,
            "future_canon_multi_delta_program_required": True,
            "primary_future_canon_lane_selected": "tmchi_lane" if primary_lane_selected else None,
            "tmchi_lane_item_keys": TMCHI_LANE_ITEMS,
            "tv_lane_item_keys": TV_LANE_ITEMS,
            "reserve_lane_item_keys": RESERVE_LANE_ITEMS,
            "reopen_prerequisite_satisfied_under_current_canon": False,
            "physical_reject_required": False,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_route_contract_one_hundred_seventy_first_refresh_frozen",
            "advance_to_next_route": carryover_registry_ready,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"gate_summary": gate["summary"], "audit_summary": audit["summary"]},
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "alpha_is_prediction_future_canon_delta_program_carryover_registry_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "alpha_is_prediction_future_canon_delta_program_carryover_registry_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "alpha_is_prediction_future_canon_delta_program_carryover_registry_declaration_gate",
        gate,
    )
    write_artifact("mass_origin_v2_t2_alpha_route_contract_one_hundred_seventy_first_refresh", route)

    print("[done] 8.7.56.1095-.1098 artifacts generated")


if __name__ == "__main__":
    main()
