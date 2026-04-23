#!/usr/bin/env python3
"""Generate 8.7.56.1115-.1118 Trial-2 future-canon T_v rewrite registry artifacts."""

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

AUDIT_1088 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_delta_registry_audit_metrics.json"
)
INVENTORY_1111 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_h0p_mass_frequency_bridge_registry_source_inventory_metrics.json"
)
AUDIT_1112 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_h0p_mass_frequency_bridge_registry_audit_metrics.json"
)
GATE_1113 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_h0p_mass_frequency_bridge_registry_declaration_gate_metrics.json"
)
ROUTE_1114 = PUBLIC_OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_seventy_fifth_refresh_metrics.json"

CURRENT_ROUTE = (
    "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_tv_dimensionless_ratio_rewrite_registry"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_source_normalization_bridge_reserve_registry"
)
NEXT_ROUTE = "8.7.56.1119"

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


# Function: classify the T_v rewrite registry outcome.

def classify(registry_ready: bool, rewrite_surface_confirmed: bool, reserve_lane_retained: bool) -> str:
    """Classify the T_v rewrite registry outcome."""
    if registry_ready and rewrite_surface_confirmed and reserve_lane_retained:
        return "tv_dimensionless_ratio_rewrite_registry_frozen"

    if registry_ready and rewrite_surface_confirmed:
        return "tv_dimensionless_ratio_rewrite_registry_partial"

    return "tv_dimensionless_ratio_rewrite_registry_incomplete"


# Function: execute the T_v dimensionless-ratio rewrite registry branch.

def main() -> None:
    """Execute the Trial-2 future-canon T_v dimensionless-ratio rewrite registry branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PART3A,
        PART5,
        NOTE_ALPHA,
        NOTE_DIM,
        AUDIT_1088,
        INVENTORY_1111,
        AUDIT_1112,
        GATE_1113,
        ROUTE_1114,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    alpha_note_text = read_text(NOTE_ALPHA)
    dim_note_text = read_text(NOTE_DIM)

    audit_1088 = read_json(AUDIT_1088)["summary"]
    inventory_1111 = read_json(INVENTORY_1111)["summary"]
    audit_1112 = read_json(AUDIT_1112)["summary"]
    gate_1113 = read_json(GATE_1113)["summary"]
    route_1114 = read_json(ROUTE_1114)["summary"]

    targets = [
        target(status_text, STATUS, "status_1115", "8.7.56.1115", "STATUS must retain this branch."),
        target(roadmap_text, ROADMAP, "roadmap_1115", "`8.7.56.1115-.1118`", "ROADMAP must retain this branch."),
        target(part3a_text, PART3A, "part3a_tv_route", "future-canon `T_v` dimensionless-ratio rewrite registry", "Part III-A must expose the active T_v rewrite route."),
        target(part5_text, PART5, "part5_tv_route", "future-canon `T_v` dimensionless-ratio rewrite registry", "Part V must expose the active T_v rewrite route."),
        target(alpha_note_text, NOTE_ALPHA, "alpha_note_formula", r"\alpha = \frac{c^3}{4\pi v^2 \hbar}", "The alpha note must still expose the bare-v alpha formula."),
        target(alpha_note_text, NOTE_ALPHA, "alpha_note_v", r"v = \frac{H_0^{(P)} \cdot M_\chi}{m_0}", "The alpha note must still expose the downstream v relation."),
        target(dim_note_text, NOTE_DIM, "dim_note_tv", r"T_v", "The dimension note must still expose the T_v theorem placeholder."),
        target(dim_note_text, NOTE_DIM, "dim_note_ratio", "dimensionless ratio", "The dimension note must still expose the ratio-normalization requirement."),
    ]

    prior_route_active = all(
        [
            inventory_1111["tv_dimensionless_ratio_rewrite_downstream_retained"],
            audit_1112["tv_dimensionless_ratio_rewrite_downstream_retained"],
            gate_1113["trial2_numeric_alpha_future_canon_h0p_mass_frequency_bridge_registry_completed"],
            gate_1113["selected_residual_route"] == CURRENT_ROUTE,
            route_1114["selected_next_generation_route"] == CURRENT_ROUTE,
            not route_1114["physical_reject_required"],
        ]
    )
    inventory_ready = all(item["present"] for item in targets) and prior_route_active

    tv_dimensionless_ratio_rewrite_surface_confirmed = bool(
        audit_1088["tv_pack_required"]
        and hit(alpha_note_text, r"\alpha = \frac{c^3}{4\pi v^2 \hbar}") is not None
        and hit(alpha_note_text, r"v = \frac{H_0^{(P)} \cdot M_\chi}{m_0}") is not None
        and hit(dim_note_text, r"T_v") is not None
        and hit(dim_note_text, "dimensionless ratio") is not None
    )
    tv_dimensionless_ratio_rewrite_is_first_downstream_item = bool(
        inventory_1111["tv_lane_item_keys"] == TV_LANE_ITEMS
        and audit_1112["tv_dimensionless_ratio_rewrite_downstream_retained"]
        and gate_1113["selected_residual_route"] == CURRENT_ROUTE
        and route_1114["selected_next_generation_route"] == CURRENT_ROUTE
    )
    reserve_lane_downstream_retained = bool(
        inventory_1111["reserve_lane_item_keys"] == RESERVE_LANE_ITEMS
        and audit_1112["reserve_lane_downstream_retained"]
        and gate_1113["trial2_numeric_alpha_reserve_lane_downstream_retained"]
        and route_1114["reserve_lane_downstream_retained"]
    )
    rewrite_still_future_canon_only = bool(
        audit_1088["future_canon_multi_delta_program_required"]
        and not route_1114["reopen_prerequisite_satisfied_under_current_canon"]
        and not route_1114["physical_reject_required"]
    )
    tv_dimensionless_ratio_rewrite_registry_ready = bool(
        inventory_ready
        and tv_dimensionless_ratio_rewrite_surface_confirmed
        and tv_dimensionless_ratio_rewrite_is_first_downstream_item
        and reserve_lane_downstream_retained
        and rewrite_still_future_canon_only
    )
    registry_class = classify(
        tv_dimensionless_ratio_rewrite_registry_ready,
        tv_dimensionless_ratio_rewrite_surface_confirmed,
        reserve_lane_downstream_retained,
    )

    inputs = {
        "status_markdown": display_path(STATUS),
        "roadmap_markdown": display_path(ROADMAP),
        "ai_context_json": display_path(AI_CONTEXT),
        "part3a_markdown": display_path(PART3A),
        "part5_markdown": display_path(PART5),
        "alpha_note": display_path(NOTE_ALPHA),
        "dimension_note": display_path(NOTE_DIM),
        "prior_1088_json": display_path(AUDIT_1088),
        "prior_1111_json": display_path(INVENTORY_1111),
        "prior_1112_json": display_path(AUDIT_1112),
        "prior_1113_json": display_path(GATE_1113),
        "prior_1114_json": display_path(ROUTE_1114),
    }

    inventory = payload(
        "8.7.56.1115",
        "Trial-2 numeric alpha future-canon T_v dimensionless-ratio rewrite registry source inventory",
        inputs,
        [
            row("inventory_complete", "pass" if inventory_ready else "reject", "T_v rewrite registry inventory complete", 1 if inventory_ready else 0, "The T_v rewrite registry is assembled from the H0P bridge registry metrics, retained note surfaces, and frozen public wording."),
            row("tv_rewrite_surface_confirmed", "pass" if tv_dimensionless_ratio_rewrite_surface_confirmed else "reject", "T_v rewrite surface confirmed", 1 if tv_dimensionless_ratio_rewrite_surface_confirmed else 0, "The retained notes still say alpha cannot stay on bare v and must move to a dimensionless ratio form."),
            row("tv_rewrite_first_downstream_item", "pass" if tv_dimensionless_ratio_rewrite_is_first_downstream_item else "reject", "T_v rewrite fixed as first downstream rewrite item", 1 if tv_dimensionless_ratio_rewrite_is_first_downstream_item else 0, "The T_v rewrite remains the next rewrite item after the theorem-side lane and H0P bridge are frozen."),
            row("reserve_lane_downstream_retained", "pass" if reserve_lane_downstream_retained else "reject", "reserve lane retained downstream", 1 if reserve_lane_downstream_retained else 0, "The source-normalization bridge remains reserve evidence only and does not outrank the T_v rewrite."),
            row("rewrite_still_future_canon_only", "pass" if rewrite_still_future_canon_only else "reject", "T_v rewrite still future-canon only", 1 if rewrite_still_future_canon_only else 0, "The rewrite registry does not reopen current-canon computation and remains part of the future-canon multi-delta program."),
        ],
        {
            "inventory_ready": inventory_ready,
            "tv_dimensionless_ratio_rewrite_registry_ready": tv_dimensionless_ratio_rewrite_registry_ready,
            "tv_dimensionless_ratio_rewrite_surface_confirmed": tv_dimensionless_ratio_rewrite_surface_confirmed,
            "tv_dimensionless_ratio_rewrite_is_first_downstream_item": tv_dimensionless_ratio_rewrite_is_first_downstream_item,
            "reserve_lane_item_keys": RESERVE_LANE_ITEMS,
            "reserve_lane_downstream_retained": reserve_lane_downstream_retained,
            "future_canon_multi_delta_program_required": True,
            "reopen_prerequisite_satisfied_under_current_canon": False,
            "first_route_to_close_or_none": CURRENT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_future_canon_tv_rewrite_inventory_frozen",
            "advance_to_8_7_56_1116": inventory_ready,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "targets": targets,
            "retained_1088_summary": audit_1088,
            "retained_1111_summary": inventory_1111,
            "retained_1112_summary": audit_1112,
            "retained_1113_summary": gate_1113,
            "retained_1114_summary": route_1114,
        },
    )

    audit = payload(
        "8.7.56.1116",
        "Trial-2 numeric alpha future-canon T_v dimensionless-ratio rewrite registry audit",
        inputs,
        [
            row("tv_rewrite_registry_ready", "pass" if tv_dimensionless_ratio_rewrite_registry_ready else "reject", "T_v rewrite registry ready", 1 if tv_dimensionless_ratio_rewrite_registry_ready else 0, "The T_v rewrite registry passes only if the rewrite surface stays visible, remains the first downstream rewrite item, and still hands off to reserve evidence."),
            row("tv_rewrite_surface_confirmed", "pass" if tv_dimensionless_ratio_rewrite_surface_confirmed else "reject", "T_v rewrite surface confirmed", 1 if tv_dimensionless_ratio_rewrite_surface_confirmed else 0, "The dimension note still requires a ratio-normalized quantity instead of bare v."),
            row("tv_rewrite_first_downstream_item", "pass" if tv_dimensionless_ratio_rewrite_is_first_downstream_item else "reject", "T_v rewrite confirmed as first downstream rewrite item", 1 if tv_dimensionless_ratio_rewrite_is_first_downstream_item else 0, "The rewrite stays immediately after the H0P bridge in the future-canon order."),
            row("reserve_lane_not_prematurely_promoted", "pass" if reserve_lane_downstream_retained else "reject", "reserve lane not prematurely promoted", 1 if reserve_lane_downstream_retained else 0, "The source-normalization ambiguity remains reserve evidence after the T_v rewrite registry."),
            row("current_canon_not_reopened", "pass" if rewrite_still_future_canon_only else "reject", "current canon not reopened by T_v rewrite alone", 1 if rewrite_still_future_canon_only else 0, "The rewrite registry remains future-canon only and does not escalate to a current-canon reopen."),
        ],
        {
            "audit_ready": inventory_ready,
            "tv_dimensionless_ratio_rewrite_registry_ready": tv_dimensionless_ratio_rewrite_registry_ready,
            "tv_dimensionless_ratio_rewrite_surface_confirmed": tv_dimensionless_ratio_rewrite_surface_confirmed,
            "tv_dimensionless_ratio_rewrite_is_first_downstream_item": tv_dimensionless_ratio_rewrite_is_first_downstream_item,
            "reserve_lane_prematurely_promoted": False,
            "reserve_lane_downstream_retained": reserve_lane_downstream_retained,
            "future_canon_multi_delta_program_required": True,
            "reopen_prerequisite_satisfied_under_current_canon": False,
            "physical_reject_required": False,
            "selected_tv_dimensionless_ratio_rewrite_registry_class": registry_class,
            "first_route_to_close_after_audit_or_none": NEXT_ROUTE_NAME,
        },
        {
            "overall_status": "trial2_numeric_alpha_future_canon_tv_rewrite_audited",
            "advance_to_8_7_56_1117": tv_dimensionless_ratio_rewrite_registry_ready,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"inventory_summary": inventory["summary"]},
    )

    gate = payload(
        "8.7.56.1117",
        "Trial-2 numeric alpha future-canon T_v dimensionless-ratio rewrite registry declaration gate",
        inputs,
        [
            row("gate_complete", "pass" if tv_dimensionless_ratio_rewrite_registry_ready else "reject", "future-canon T_v rewrite registry gate complete", 1 if tv_dimensionless_ratio_rewrite_registry_ready else 0, "The T_v rewrite registry becomes official only after the rewrite surface and downstream reserve guard both pass."),
            row("tv_rewrite_first_downstream_item_frozen", "pass" if tv_dimensionless_ratio_rewrite_is_first_downstream_item else "reject", "T_v rewrite frozen as first downstream rewrite item", 1 if tv_dimensionless_ratio_rewrite_is_first_downstream_item else 0, "The T_v rewrite remains the first rewrite item after the theorem-side lane and H0P bridge."),
            row("reserve_lane_retained_frozen", "pass" if reserve_lane_downstream_retained else "reject", "reserve lane retained after T_v rewrite registry", 1 if reserve_lane_downstream_retained else 0, "The source-normalization bridge remains reserve evidence after the T_v rewrite registry is frozen."),
            row("next_route_selected", "pass", "future-canon source-normalization bridge reserve registry selected", 1, "The next branch moves to the retained reserve lane after freezing the T_v rewrite registry."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "alpha_prediction_future_canon_tv_dimensionless_ratio_rewrite_registry",
            "trial2_numeric_alpha_future_canon_tv_dimensionless_ratio_rewrite_registry_completed": tv_dimensionless_ratio_rewrite_registry_ready,
            "trial2_numeric_alpha_tv_dimensionless_ratio_rewrite_surface_confirmed": tv_dimensionless_ratio_rewrite_surface_confirmed,
            "trial2_numeric_alpha_tv_dimensionless_ratio_rewrite_is_first_downstream_item": tv_dimensionless_ratio_rewrite_is_first_downstream_item,
            "trial2_numeric_alpha_reserve_lane_downstream_retained": reserve_lane_downstream_retained,
            "trial2_numeric_alpha_reopen_prerequisite_satisfied_under_current_canon": False,
            "trial2_numeric_alpha_physical_reject_required": False,
            "selected_residual_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_future_canon_tv_rewrite_gate_closed",
            "advance_to_8_7_56_1118": tv_dimensionless_ratio_rewrite_registry_ready,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"audit_summary": audit["summary"]},
    )

    route = payload(
        "8.7.56.1118",
        "Trial-2 numeric alpha route contract one-hundred-seventy-sixth refresh",
        inputs,
        [
            row("route_contract_complete", "pass" if tv_dimensionless_ratio_rewrite_registry_ready else "reject", "route contract one-hundred-seventy-sixth refresh complete", 1 if tv_dimensionless_ratio_rewrite_registry_ready else 0, "The T_v rewrite registry is converted into the next-generation route contract."),
            row("tv_rewrite_registry_completed", "pass" if tv_dimensionless_ratio_rewrite_registry_ready else "reject", "future-canon T_v dimensionless-ratio rewrite registry completed", 1 if tv_dimensionless_ratio_rewrite_registry_ready else 0, "The rewrite-side item inside the future-canon program is now frozen as one registry."),
            row("reserve_registry_selected_as_next_route", "pass" if reserve_lane_downstream_retained else "reject", "source-normalization bridge reserve registry selected as next route", 1 if reserve_lane_downstream_retained else 0, "The next step moves to the retained reserve lane after the T_v rewrite registry is frozen."),
            row("physical_reject_not_selected", "pass", "physical reject not selected after T_v rewrite registry", 1, "The route remains structurally alive after freezing the T_v rewrite registry."),
        ],
        {
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "future_canon_tv_dimensionless_ratio_rewrite_registry_completed": tv_dimensionless_ratio_rewrite_registry_ready,
            "future_canon_multi_delta_program_required": True,
            "tv_dimensionless_ratio_rewrite_surface_confirmed": tv_dimensionless_ratio_rewrite_surface_confirmed,
            "tv_dimensionless_ratio_rewrite_is_first_downstream_item": tv_dimensionless_ratio_rewrite_is_first_downstream_item,
            "reserve_lane_item_keys": RESERVE_LANE_ITEMS,
            "reserve_lane_downstream_retained": reserve_lane_downstream_retained,
            "reopen_prerequisite_satisfied_under_current_canon": False,
            "physical_reject_required": False,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_route_contract_one_hundred_seventy_sixth_refresh_frozen",
            "advance_to_next_route": tv_dimensionless_ratio_rewrite_registry_ready,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"gate_summary": gate["summary"], "audit_summary": audit["summary"]},
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "alpha_is_prediction_future_canon_tv_dimensionless_ratio_rewrite_registry_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "alpha_is_prediction_future_canon_tv_dimensionless_ratio_rewrite_registry_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "alpha_is_prediction_future_canon_tv_dimensionless_ratio_rewrite_registry_declaration_gate",
        gate,
    )
    write_artifact("mass_origin_v2_t2_alpha_route_contract_one_hundred_seventy_sixth_refresh", route)

    print("[done] 8.7.56.1115-.1118 artifacts generated")


if __name__ == "__main__":
    main()
