#!/usr/bin/env python3
"""Generate 8.7.56.1099-.1102 Trial-2 future-canon T_Mchi lane registry artifacts."""

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
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
NOTE_ALPHA = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_alpha_is_prediction.md")
NOTE_DIM = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_dimension_normalization_review.md")

AUDIT_1080 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_tmchi_tv_prove_or_no_go_review_audit_metrics.json"
)
AUDIT_1088 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_delta_registry_audit_metrics.json"
)
INVENTORY_1095 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_delta_program_carryover_registry_source_inventory_metrics.json"
)
AUDIT_1096 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_delta_program_carryover_registry_audit_metrics.json"
)
GATE_1097 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_delta_program_carryover_registry_declaration_gate_metrics.json"
)
ROUTE_1098 = PUBLIC_OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_seventy_first_refresh_metrics.json"

CURRENT_ROUTE = (
    "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_tmchi_lane_registry"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_tmchi_theorem_bridge_pack_registry"
)
NEXT_ROUTE = "8.7.56.1103"

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


# Function: classify the T_Mchi lane registry outcome.

def classify(tmchi_lane_registry_ready: bool, tv_downstream_retained: bool, reserve_downstream_retained: bool) -> str:
    """Classify the T_Mchi lane registry outcome."""
    if tmchi_lane_registry_ready and tv_downstream_retained and reserve_downstream_retained:
        return "tmchi_lane_registry_frozen"

    if tmchi_lane_registry_ready:
        return "tmchi_lane_registry_partially_frozen"

    return "tmchi_lane_registry_incomplete"


# Function: execute the T_Mchi lane registry branch.

def main() -> None:
    """Execute the Trial-2 future-canon T_Mchi lane registry branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PART1,
        PART3A,
        PART5,
        NOTE_ALPHA,
        NOTE_DIM,
        AUDIT_1080,
        AUDIT_1088,
        INVENTORY_1095,
        AUDIT_1096,
        GATE_1097,
        ROUTE_1098,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    alpha_note_text = read_text(NOTE_ALPHA)
    dim_note_text = read_text(NOTE_DIM)

    audit_1080 = read_json(AUDIT_1080)["summary"]
    audit_1088 = read_json(AUDIT_1088)["summary"]
    inventory_1095 = read_json(INVENTORY_1095)["summary"]
    audit_1096 = read_json(AUDIT_1096)["summary"]
    gate_1097 = read_json(GATE_1097)["summary"]
    route_1098 = read_json(ROUTE_1098)["summary"]

    targets = [
        target(status_text, STATUS, "status_1099", "8.7.56.1099", "STATUS must retain this branch."),
        target(roadmap_text, ROADMAP, "roadmap_1099", "`8.7.56.1099-.1102`", "ROADMAP must retain this branch."),
        target(part1_text, PART1, "part1_lchi", r"\mathcal{L}_{\chi}", "Part I must still expose the scalar kinetic coefficient surface."),
        target(part1_text, PART1, "part1_same_sector_proxy", "same-sector proxy value", "Part I must still expose the same-sector proxy dependency."),
        target(part1_text, PART1, "part1_current_surface", r"J^\mu_{\mathrm{matter}}=(\rho c,\rho \mathbf{v})", "Part I must retain the current surface used by the reserve evidence."),
        target(part3a_text, PART3A, "part3a_tmchi_lane_route", "future-canon `T_{M_\\chi}` lane registry", "Part III-A must expose the active T_Mchi lane route."),
        target(part5_text, PART5, "part5_tmchi_lane_route", "future-canon $T_{M_\\chi}$ lane registry branch `8.7.56.1099-.1102`", "Part V must expose the active T_Mchi lane route."),
        target(alpha_note_text, NOTE_ALPHA, "alpha_note_mchi", r"M_\chi^2 = \frac{c^4}{4\pi G}", "The alpha note must still expose the Newton-side T_Mchi claim."),
        target(alpha_note_text, NOTE_ALPHA, "alpha_note_h0p", r"H_0^{(P)} = \frac{m_0}{\sqrt{Z_P^{\rm grav}}}", "The alpha note must still expose the H0P mass-frequency bridge claim."),
        target(dim_note_text, NOTE_DIM, "dim_note_tmchi", r"T_{M_\chi}", "The dimension note must still expose T_Mchi."),
        target(dim_note_text, NOTE_DIM, "dim_note_case_c", "Case C", "The dimension note must still expose the current-canon no-go classifier."),
    ]

    prior_route_active = all(
        [
            inventory_1095["inventory_ready"],
            inventory_1095["primary_future_canon_lane"] == "tmchi_lane",
            audit_1096["tmchi_lane_carryover_ready"],
            audit_1096["primary_future_canon_lane_selected"] == "tmchi_lane",
            gate_1097["selected_residual_route"] == CURRENT_ROUTE,
            route_1098["selected_next_generation_route"] == CURRENT_ROUTE,
            not route_1098["physical_reject_required"],
        ]
    )
    inventory_ready = all(item["present"] for item in targets) and prior_route_active

    tmchi_promotion_delta_retained = bool(
        audit_1080["tmchi_no_go_current_canon"]
        and audit_1088["tmchi_pack_required"]
        and "delta_tmchi_promotion_theorem" in inventory_1095["tmchi_lane_item_keys"]
    )
    h0p_mass_frequency_bridge_delta_retained = bool(
        audit_1088["h0p_mass_frequency_bridge_required"]
        and "delta_h0p_mass_frequency_bridge" in inventory_1095["tmchi_lane_item_keys"]
    )
    tmchi_lane_bundled_upstream = bool(
        tmchi_promotion_delta_retained
        and h0p_mass_frequency_bridge_delta_retained
        and inventory_1095["primary_future_canon_lane"] == "tmchi_lane"
    )
    tv_lane_downstream_retained = bool(
        audit_1080["tv_downstream_unresolved_after_tmchi_no_go"]
        and "delta_tv_dimensionless_ratio_rewrite" in inventory_1095["tv_lane_item_keys"]
    )
    reserve_lane_downstream_retained = bool(
        audit_1088["source_normalization_reserve_retained"]
        and "delta_source_normalization_bridge_reserve" in inventory_1095["reserve_lane_item_keys"]
    )
    tmchi_lane_registry_ready = bool(
        inventory_ready
        and tmchi_lane_bundled_upstream
        and tv_lane_downstream_retained
        and reserve_lane_downstream_retained
        and audit_1088["future_canon_multi_delta_program_required"]
        and not route_1098["physical_reject_required"]
    )
    tmchi_lane_registry_class = classify(
        tmchi_lane_registry_ready,
        tv_lane_downstream_retained,
        reserve_lane_downstream_retained,
    )

    inputs = {
        "status_markdown": display_path(STATUS),
        "roadmap_markdown": display_path(ROADMAP),
        "ai_context_json": display_path(AI_CONTEXT),
        "part1_markdown": display_path(PART1),
        "part3a_markdown": display_path(PART3A),
        "part5_markdown": display_path(PART5),
        "alpha_note": display_path(NOTE_ALPHA),
        "dimension_note": display_path(NOTE_DIM),
        "prior_1080_json": display_path(AUDIT_1080),
        "prior_1088_json": display_path(AUDIT_1088),
        "prior_1095_json": display_path(INVENTORY_1095),
        "prior_1096_json": display_path(AUDIT_1096),
        "prior_1097_json": display_path(GATE_1097),
        "prior_1098_json": display_path(ROUTE_1098),
    }

    inventory = payload(
        "8.7.56.1099",
        "Trial-2 numeric alpha future-canon T_Mchi lane registry source inventory",
        inputs,
        [
            row("inventory_complete", "pass" if inventory_ready else "reject", "T_Mchi lane registry inventory complete", 1 if inventory_ready else 0, "The T_Mchi lane pack is assembled from the carry-over registry metrics, theorem metrics, public wording, and retained notes."),
            row("tmchi_promotion_delta_retained", "pass" if tmchi_promotion_delta_retained else "reject", "T_Mchi promotion delta retained", 1 if tmchi_promotion_delta_retained else 0, "The first item in the lane remains the missing theorem that promotes M_chi beyond the kinetic-coefficient surface."),
            row("h0p_bridge_delta_retained", "pass" if h0p_mass_frequency_bridge_delta_retained else "reject", "H0P mass-frequency bridge delta retained", 1 if h0p_mass_frequency_bridge_delta_retained else 0, "The second item in the lane remains the H0P mass-frequency bridge."),
            row("tv_lane_downstream_retained", "pass" if tv_lane_downstream_retained else "reject", "T_v lane retained downstream", 1 if tv_lane_downstream_retained else 0, "The downstream T_v lane stays attached but does not enter the upstream bundled lane."),
            row("reserve_lane_downstream_retained", "pass" if reserve_lane_downstream_retained else "reject", "reserve lane retained downstream", 1 if reserve_lane_downstream_retained else 0, "The source-normalization bridge remains reserve evidence only."),
        ],
        {
            "inventory_ready": inventory_ready,
            "tmchi_lane_registry_ready": tmchi_lane_registry_ready,
            "tmchi_lane_item_keys": TMCHI_LANE_ITEMS,
            "tmchi_lane_primary_focus": "delta_tmchi_promotion_theorem",
            "tmchi_lane_secondary_item": "delta_h0p_mass_frequency_bridge",
            "tv_lane_item_keys": TV_LANE_ITEMS,
            "reserve_lane_item_keys": RESERVE_LANE_ITEMS,
            "tmchi_lane_bundled_upstream": tmchi_lane_bundled_upstream,
            "tv_lane_downstream_retained": tv_lane_downstream_retained,
            "reserve_lane_downstream_retained": reserve_lane_downstream_retained,
            "first_route_to_close_or_none": CURRENT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_future_canon_tmchi_lane_inventory_frozen",
            "advance_to_8_7_56_1100": inventory_ready,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "targets": targets,
            "retained_1080_summary": audit_1080,
            "retained_1088_summary": audit_1088,
            "retained_1095_summary": inventory_1095,
            "retained_1096_summary": audit_1096,
            "retained_1097_summary": gate_1097,
            "retained_1098_summary": route_1098,
        },
    )

    audit = payload(
        "8.7.56.1100",
        "Trial-2 numeric alpha future-canon T_Mchi lane registry audit",
        inputs,
        [
            row("tmchi_lane_registry_ready", "pass" if tmchi_lane_registry_ready else "reject", "T_Mchi lane registry ready", 1 if tmchi_lane_registry_ready else 0, "The upstream lane passes only if theorem promotion and the H0P bridge remain bundled while the downstream lanes stay outside the registry."),
            row("tmchi_lane_bundled_upstream", "pass" if tmchi_lane_bundled_upstream else "reject", "T_Mchi lane bundled upstream", 1 if tmchi_lane_bundled_upstream else 0, "The theorem-promotion delta and H0P bridge delta must remain one bundled upstream lane."),
            row("tv_lane_not_prematurely_mixed", "pass" if tv_lane_downstream_retained else "reject", "T_v lane not prematurely mixed into T_Mchi lane", 1 if tv_lane_downstream_retained else 0, "T_v stays downstream because it remained unresolved after the current-canon T_Mchi no-go."),
            row("reserve_lane_not_prematurely_promoted", "pass" if reserve_lane_downstream_retained else "reject", "reserve lane not prematurely promoted", 1 if reserve_lane_downstream_retained else 0, "The source-normalization bridge remains reserve evidence and does not outrank the theorem lane."),
            row("physical_reject_not_required", "pass" if not route_1098["physical_reject_required"] else "reject", "physical reject not required", 1 if not route_1098["physical_reject_required"] else 0, "The lane registry remains inside the structural-pass reading."),
        ],
        {
            "audit_ready": inventory_ready,
            "tmchi_lane_registry_ready": tmchi_lane_registry_ready,
            "tmchi_lane_bundled_upstream": tmchi_lane_bundled_upstream,
            "tmchi_promotion_delta_retained": tmchi_promotion_delta_retained,
            "h0p_mass_frequency_bridge_delta_retained": h0p_mass_frequency_bridge_delta_retained,
            "tv_lane_prematurely_mixed": False,
            "reserve_lane_prematurely_promoted": False,
            "tv_lane_downstream_retained": tv_lane_downstream_retained,
            "reserve_lane_downstream_retained": reserve_lane_downstream_retained,
            "future_canon_multi_delta_program_required": bool(audit_1088["future_canon_multi_delta_program_required"]),
            "reopen_prerequisite_satisfied_under_current_canon": False,
            "physical_reject_required": False,
            "selected_tmchi_lane_registry_class": tmchi_lane_registry_class,
            "first_route_to_close_after_audit_or_none": NEXT_ROUTE_NAME,
        },
        {
            "overall_status": "trial2_numeric_alpha_future_canon_tmchi_lane_audited",
            "advance_to_8_7_56_1101": tmchi_lane_registry_ready,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"inventory_summary": inventory["summary"]},
    )

    gate = payload(
        "8.7.56.1101",
        "Trial-2 numeric alpha future-canon T_Mchi lane registry declaration gate",
        inputs,
        [
            row("gate_complete", "pass" if tmchi_lane_registry_ready else "reject", "future-canon T_Mchi lane registry gate complete", 1 if tmchi_lane_registry_ready else 0, "The T_Mchi lane becomes official only after the upstream bundling and downstream guards both pass."),
            row("tmchi_primary_focus_frozen", "pass" if tmchi_promotion_delta_retained else "reject", "T_Mchi promotion delta frozen as primary focus", 1 if tmchi_promotion_delta_retained else 0, "The next route starts from the theorem-promotion delta rather than from the downstream T_v rewrite."),
            row("h0p_bridge_secondary_focus_frozen", "pass" if h0p_mass_frequency_bridge_delta_retained else "reject", "H0P bridge frozen as secondary item", 1 if h0p_mass_frequency_bridge_delta_retained else 0, "The mass-frequency bridge remains attached inside the same T_Mchi lane."),
            row("next_route_selected", "pass", "future-canon T_Mchi theorem-bridge pack registry selected", 1, "The next branch drills into the two-item T_Mchi lane pack."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "alpha_prediction_future_canon_tmchi_lane_registry",
            "trial2_numeric_alpha_future_canon_tmchi_lane_registry_completed": tmchi_lane_registry_ready,
            "trial2_numeric_alpha_tmchi_lane_bundled_upstream": tmchi_lane_bundled_upstream,
            "trial2_numeric_alpha_tmchi_promotion_delta_retained": tmchi_promotion_delta_retained,
            "trial2_numeric_alpha_h0p_mass_frequency_bridge_delta_retained": h0p_mass_frequency_bridge_delta_retained,
            "trial2_numeric_alpha_tv_lane_downstream_retained": tv_lane_downstream_retained,
            "trial2_numeric_alpha_reserve_lane_downstream_retained": reserve_lane_downstream_retained,
            "trial2_numeric_alpha_reopen_prerequisite_satisfied_under_current_canon": False,
            "trial2_numeric_alpha_physical_reject_required": False,
            "selected_residual_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_future_canon_tmchi_lane_gate_closed",
            "advance_to_8_7_56_1102": tmchi_lane_registry_ready,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"audit_summary": audit["summary"]},
    )

    route = payload(
        "8.7.56.1102",
        "Trial-2 numeric alpha route contract one-hundred-seventy-second refresh",
        inputs,
        [
            row("route_contract_complete", "pass" if tmchi_lane_registry_ready else "reject", "route contract one-hundred-seventy-second refresh complete", 1 if tmchi_lane_registry_ready else 0, "The T_Mchi lane registry is converted into the next-generation route contract."),
            row("tmchi_lane_registry_completed", "pass" if tmchi_lane_registry_ready else "reject", "future-canon T_Mchi lane registry completed", 1 if tmchi_lane_registry_ready else 0, "The primary future-canon lane is now frozen as one upstream executable lane."),
            row("tmchi_theorem_focus_selected", "pass" if tmchi_promotion_delta_retained else "reject", "T_Mchi theorem focus selected", 1 if tmchi_promotion_delta_retained else 0, "The next step will start from the theorem-focused pack ordering inside the lane."),
            row("physical_reject_not_selected", "pass", "physical reject not selected after T_Mchi lane registry", 1, "The route remains structurally alive after the lane registry freeze."),
        ],
        {
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "future_canon_tmchi_lane_registry_completed": tmchi_lane_registry_ready,
            "future_canon_multi_delta_program_required": True,
            "tmchi_lane_item_keys": TMCHI_LANE_ITEMS,
            "tmchi_lane_primary_focus": "delta_tmchi_promotion_theorem",
            "tmchi_lane_secondary_item": "delta_h0p_mass_frequency_bridge",
            "tv_lane_item_keys": TV_LANE_ITEMS,
            "reserve_lane_item_keys": RESERVE_LANE_ITEMS,
            "tv_lane_downstream_retained": tv_lane_downstream_retained,
            "reserve_lane_downstream_retained": reserve_lane_downstream_retained,
            "reopen_prerequisite_satisfied_under_current_canon": False,
            "physical_reject_required": False,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_route_contract_one_hundred_seventy_second_refresh_frozen",
            "advance_to_next_route": tmchi_lane_registry_ready,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"gate_summary": gate["summary"], "audit_summary": audit["summary"]},
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "alpha_is_prediction_future_canon_tmchi_lane_registry_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "alpha_is_prediction_future_canon_tmchi_lane_registry_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "alpha_is_prediction_future_canon_tmchi_lane_registry_declaration_gate",
        gate,
    )
    write_artifact("mass_origin_v2_t2_alpha_route_contract_one_hundred_seventy_second_refresh", route)

    print("[done] 8.7.56.1099-.1102 artifacts generated")


if __name__ == "__main__":
    main()
