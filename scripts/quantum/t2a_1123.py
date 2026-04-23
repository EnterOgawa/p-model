#!/usr/bin/env python3
"""Generate 8.7.56.1123-.1126 Trial-2 future-canon multi-delta hold contract artifacts."""

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
NOTE_DIMENSION = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_dimension_normalization_review.md")
NOTE_SI = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_si_dimension_tracking.md")

AUDIT_1088 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_delta_registry_audit_metrics.json"
)
ROUTE_1110 = PUBLIC_OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_seventy_fourth_refresh_metrics.json"
ROUTE_1114 = PUBLIC_OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_seventy_fifth_refresh_metrics.json"
ROUTE_1118 = PUBLIC_OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_seventy_sixth_refresh_metrics.json"
ROUTE_1122 = PUBLIC_OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_seventy_seventh_refresh_metrics.json"

CURRENT_ROUTE = (
    "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_multi_delta_hold_contract"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_carry_over_share_pack_registry"
)
NEXT_ROUTE = "8.7.56.1127"
TMCHI_NEXT_ROUTE = (
    "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_h0p_mass_frequency_bridge_registry"
)
H0P_NEXT_ROUTE = (
    "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_tv_dimensionless_ratio_rewrite_registry"
)
TV_NEXT_ROUTE = (
    "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_source_normalization_bridge_reserve_registry"
)

ALL_DELTA_ITEMS = [
    "delta_tmchi_promotion_theorem",
    "delta_h0p_mass_frequency_bridge",
    "delta_tv_dimensionless_ratio_rewrite",
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

def payload(
    step: str,
    name: str,
    inputs: dict,
    rows: list[dict],
    summary: dict,
    decision: dict,
    evidence: dict,
) -> dict:
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


# Function: classify the top-level hold-contract outcome.

def classify(hold_ready: bool, all_items_frozen: bool, reopen_false: bool, physical_reject_false: bool) -> str:
    """Classify the multi-delta hold-contract outcome."""
    if hold_ready and all_items_frozen and reopen_false and physical_reject_false:
        return "future_canon_multi_delta_hold_contract_frozen"

    if all_items_frozen and reopen_false and physical_reject_false:
        return "future_canon_multi_delta_hold_contract_partial"

    return "future_canon_multi_delta_hold_contract_incomplete"


# Function: execute the multi-delta hold-contract branch.

def main() -> None:
    """Execute the Trial-2 future-canon multi-delta hold-contract branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PART3A,
        PART5,
        NOTE_ALPHA,
        NOTE_DIMENSION,
        NOTE_SI,
        AUDIT_1088,
        ROUTE_1110,
        ROUTE_1114,
        ROUTE_1118,
        ROUTE_1122,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    alpha_note_text = read_text(NOTE_ALPHA)
    dimension_note_text = read_text(NOTE_DIMENSION)
    si_note_text = read_text(NOTE_SI)

    audit_1088_payload = read_json(AUDIT_1088)
    route_1110 = read_json(ROUTE_1110)["summary"]
    route_1114 = read_json(ROUTE_1114)["summary"]
    route_1118 = read_json(ROUTE_1118)["summary"]
    route_1122 = read_json(ROUTE_1122)["summary"]

    audit_1088 = audit_1088_payload["summary"]
    registry_1088 = audit_1088_payload["evidence"]["registry_summary"]

    targets = [
        target(status_text, STATUS, "status_1123", "8.7.56.1123", "STATUS must retain this branch."),
        target(roadmap_text, ROADMAP, "roadmap_1123", "`8.7.56.1123-.1126`", "ROADMAP must retain this branch."),
        target(part3a_text, PART3A, "part3a_hold_route", "future-canon multi-delta hold contract", "Part III-A must expose the active top-level hold route."),
        target(part5_text, PART5, "part5_hold_route", "future-canon multi-delta hold contract", "Part V must expose the active top-level hold route."),
        target(alpha_note_text, NOTE_ALPHA, "alpha_note_v_line", "v = \\frac{H_0^{(P)} \\cdot M_\\chi}{m_0}", "The alpha-is-prediction note must still expose the carried v-chain."),
        target(alpha_note_text, NOTE_ALPHA, "alpha_note_mchi_line", "M_\\chi = c^2/\\sqrt{4\\pi G}", "The alpha-is-prediction note must still expose the carried M_chi line."),
        target(dimension_note_text, NOTE_DIMENSION, "dimension_note_tmchi", "### $T_{M_\\chi}$", "The dimension-normalization note must still expose the T_Mchi theorem surface."),
        target(dimension_note_text, NOTE_DIMENSION, "dimension_note_tv", "### $T_v$", "The dimension-normalization note must still expose the T_v rewrite surface."),
        target(dimension_note_text, NOTE_DIMENSION, "dimension_note_case_c", "### Case C: $T_{M_\\chi}$ no-go", "The dimension-normalization note must still expose the Case-C fallback surface."),
        target(si_note_text, NOTE_SI, "si_note_j", "$J^\\mu$ の正しい読み方", "The SI tracking note must still expose the J^mu normalization question."),
    ]

    registry_items_frozen = bool(
        registry_1088["future_canon_delta_registry_items"] == ALL_DELTA_ITEMS
        and audit_1088["future_canon_multi_delta_program_required"]
        and not audit_1088["wording_only_reopen_admissible"]
        and not audit_1088["single_delta_patch_admissible"]
    )
    tmchi_item_frozen = bool(
        route_1110["tmchi_promotion_theorem_first_missing_surface_confirmed"]
        and route_1110["selected_next_generation_route"] == TMCHI_NEXT_ROUTE
        and route_1110["h0p_mass_frequency_bridge_same_lane_downstream_retained"]
    )
    h0p_item_frozen = bool(
        route_1114["future_canon_h0p_mass_frequency_bridge_registry_completed"]
        and route_1114["h0p_mass_frequency_bridge_requirement_confirmed"]
        and route_1114["selected_next_generation_route"] == H0P_NEXT_ROUTE
    )
    tv_item_frozen = bool(
        route_1118["future_canon_tv_dimensionless_ratio_rewrite_registry_completed"]
        and route_1118["tv_dimensionless_ratio_rewrite_surface_confirmed"]
        and route_1118["selected_next_generation_route"] == TV_NEXT_ROUTE
    )
    reserve_item_frozen = bool(
        route_1122["future_canon_source_normalization_bridge_reserve_registry_completed"]
        and route_1122["source_normalization_ambiguity_confirmed"]
        and route_1122["selected_next_generation_route"] == CURRENT_ROUTE
    )
    all_four_delta_items_frozen = bool(
        registry_items_frozen and tmchi_item_frozen and h0p_item_frozen and tv_item_frozen and reserve_item_frozen
    )
    reopen_prerequisite_false_retained = bool(
        not route_1110["reopen_prerequisite_satisfied_under_current_canon"]
        and not route_1114["reopen_prerequisite_satisfied_under_current_canon"]
        and not route_1118["reopen_prerequisite_satisfied_under_current_canon"]
        and not route_1122["reopen_prerequisite_satisfied_under_current_canon"]
    )
    physical_reject_false_retained = bool(
        not route_1110["physical_reject_required"]
        and not route_1114["physical_reject_required"]
        and not route_1118["physical_reject_required"]
        and not route_1122["physical_reject_required"]
    )
    hold_policy_frozen = bool(
        all_four_delta_items_frozen
        and audit_1088["future_canon_multi_delta_program_required"]
        and reopen_prerequisite_false_retained
        and physical_reject_false_retained
        and route_1122["selected_next_generation_route"] == CURRENT_ROUTE
    )
    future_canon_candidate_retained = bool(hold_policy_frozen and physical_reject_false_retained)
    carry_over_share_pack_route_required = bool(future_canon_candidate_retained and hold_policy_frozen)
    inventory_ready = bool(all(item["present"] for item in targets) and hold_policy_frozen)
    future_canon_multi_delta_hold_contract_ready = bool(
        inventory_ready
        and all_four_delta_items_frozen
        and reopen_prerequisite_false_retained
        and physical_reject_false_retained
        and carry_over_share_pack_route_required
    )
    hold_contract_class = classify(
        future_canon_multi_delta_hold_contract_ready,
        all_four_delta_items_frozen,
        reopen_prerequisite_false_retained,
        physical_reject_false_retained,
    )

    inputs = {
        "status_markdown": display_path(STATUS),
        "roadmap_markdown": display_path(ROADMAP),
        "ai_context_json": display_path(AI_CONTEXT),
        "part3a_markdown": display_path(PART3A),
        "part5_markdown": display_path(PART5),
        "alpha_is_prediction_note": display_path(NOTE_ALPHA),
        "dimension_normalization_review_note": display_path(NOTE_DIMENSION),
        "si_dimension_tracking_note": display_path(NOTE_SI),
        "prior_1088_json": display_path(AUDIT_1088),
        "prior_1110_json": display_path(ROUTE_1110),
        "prior_1114_json": display_path(ROUTE_1114),
        "prior_1118_json": display_path(ROUTE_1118),
        "prior_1122_json": display_path(ROUTE_1122),
    }

    inventory = payload(
        "8.7.56.1123",
        "Trial-2 numeric alpha future-canon multi-delta hold contract source inventory",
        inputs,
        [
            row("inventory_complete", "pass" if inventory_ready else "reject", "multi-delta hold-contract inventory complete", 1 if inventory_ready else 0, "The top-level hold contract is assembled from the delta-registry audit, the four frozen item routes, the retained note pack, and the frozen public wording."),
            row("all_four_delta_items_frozen", "pass" if all_four_delta_items_frozen else "reject", "all four delta items frozen", 1 if all_four_delta_items_frozen else 0, "The theorem, bridge, rewrite, and reserve items must all remain frozen before the top-level hold contract is honest."),
            row("reopen_prerequisite_false_retained", "pass" if reopen_prerequisite_false_retained else "reject", "current-canon reopen still false", 1 if reopen_prerequisite_false_retained else 0, "The hold contract must not reopen current-canon computation."),
            row("physical_reject_false_retained", "pass" if physical_reject_false_retained else "reject", "physical reject still false", 1 if physical_reject_false_retained else 0, "The hold contract must not escalate the route into a physical reject."),
            row("carry_over_share_pack_route_required", "pass" if carry_over_share_pack_route_required else "reject", "carry-over share-pack route required after hold contract", 1 if carry_over_share_pack_route_required else 0, "Once the top-level hold contract is frozen, the next honest work is canonical carry-over / share-pack synchronization rather than reopen."),
        ],
        {
            "inventory_ready": inventory_ready,
            "future_canon_multi_delta_hold_contract_ready": future_canon_multi_delta_hold_contract_ready,
            "future_canon_delta_registry_items": ALL_DELTA_ITEMS,
            "tmchi_theorem_item_frozen": tmchi_item_frozen,
            "h0p_bridge_item_frozen": h0p_item_frozen,
            "tv_rewrite_item_frozen": tv_item_frozen,
            "source_normalization_reserve_item_frozen": reserve_item_frozen,
            "all_four_delta_items_frozen": all_four_delta_items_frozen,
            "future_canon_candidate_retained": future_canon_candidate_retained,
            "future_canon_multi_delta_program_required": True,
            "reopen_prerequisite_satisfied_under_current_canon": False,
            "physical_reject_required": False,
            "first_route_to_close_or_none": CURRENT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_future_canon_multi_delta_hold_inventory_frozen",
            "advance_to_8_7_56_1124": inventory_ready,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "targets": targets,
            "retained_1088_summary": audit_1088,
            "retained_1088_registry_summary": registry_1088,
            "retained_1110_summary": route_1110,
            "retained_1114_summary": route_1114,
            "retained_1118_summary": route_1118,
            "retained_1122_summary": route_1122,
        },
    )

    audit = payload(
        "8.7.56.1124",
        "Trial-2 numeric alpha future-canon multi-delta hold contract audit",
        inputs,
        [
            row("hold_contract_ready", "pass" if future_canon_multi_delta_hold_contract_ready else "reject", "future-canon multi-delta hold contract ready", 1 if future_canon_multi_delta_hold_contract_ready else 0, "The top-level hold contract passes only if all four delta items stay frozen and both current-canon reopen and physical reject remain false."),
            row("all_four_delta_items_frozen", "pass" if all_four_delta_items_frozen else "reject", "all four delta items remain frozen", 1 if all_four_delta_items_frozen else 0, "The hold contract would be premature if any one frozen item dropped out of the carried pack."),
            row("current_canon_not_reopened", "pass" if reopen_prerequisite_false_retained else "reject", "current canon not reopened by hold contract", 1 if reopen_prerequisite_false_retained else 0, "The hold contract preserves the current-canon theorem-absence limit."),
            row("physical_reject_not_selected", "pass" if physical_reject_false_retained else "reject", "physical reject not selected by hold contract", 1 if physical_reject_false_retained else 0, "The hold contract retains the future-canon candidate instead of rejecting the route."),
            row("carry_over_share_pack_route_selected", "pass" if carry_over_share_pack_route_required else "reject", "carry-over share-pack route selected after hold contract", 1 if carry_over_share_pack_route_required else 0, "The next work after the hold contract is share-pack synchronization for future-canon carry-over."),
        ],
        {
            "audit_ready": inventory_ready,
            "future_canon_multi_delta_hold_contract_ready": future_canon_multi_delta_hold_contract_ready,
            "all_four_delta_items_frozen": all_four_delta_items_frozen,
            "future_canon_candidate_retained": future_canon_candidate_retained,
            "reopen_prerequisite_satisfied_under_current_canon": False,
            "physical_reject_required": False,
            "selected_multi_delta_hold_contract_class": hold_contract_class,
            "first_route_to_close_after_audit_or_none": NEXT_ROUTE_NAME,
        },
        {
            "overall_status": "trial2_numeric_alpha_future_canon_multi_delta_hold_audited",
            "advance_to_8_7_56_1125": future_canon_multi_delta_hold_contract_ready,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"inventory_summary": inventory["summary"]},
    )

    gate = payload(
        "8.7.56.1125",
        "Trial-2 numeric alpha future-canon multi-delta hold contract declaration gate",
        inputs,
        [
            row("gate_complete", "pass" if future_canon_multi_delta_hold_contract_ready else "reject", "future-canon multi-delta hold contract gate complete", 1 if future_canon_multi_delta_hold_contract_ready else 0, "The hold contract becomes official only after the four-item freeze and the carry-over-only policy both pass."),
            row("all_four_delta_items_frozen", "pass" if all_four_delta_items_frozen else "reject", "all four delta items frozen at declaration gate", 1 if all_four_delta_items_frozen else 0, "The declaration gate keeps theorem, bridge, rewrite, and reserve items in one carried contract."),
            row("hold_policy_frozen", "pass" if hold_policy_frozen else "reject", "top-level carry-over hold policy frozen", 1 if hold_policy_frozen else 0, "The declaration gate must make the hold-only policy explicit before any share-pack carry-over can be synchronized."),
            row("next_route_selected", "pass" if carry_over_share_pack_route_required else "reject", "future-canon carry-over share-pack registry selected", 1 if carry_over_share_pack_route_required else 0, "The next branch moves to the canonical carry-over / share-pack registry after freezing the top-level hold contract."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "alpha_prediction_future_canon_multi_delta_hold_contract",
            "trial2_numeric_alpha_future_canon_multi_delta_hold_contract_completed": future_canon_multi_delta_hold_contract_ready,
            "trial2_numeric_alpha_all_four_delta_items_frozen": all_four_delta_items_frozen,
            "trial2_numeric_alpha_future_canon_candidate_retained": future_canon_candidate_retained,
            "trial2_numeric_alpha_hold_policy_frozen": hold_policy_frozen,
            "trial2_numeric_alpha_reopen_prerequisite_satisfied_under_current_canon": False,
            "trial2_numeric_alpha_physical_reject_required": False,
            "selected_residual_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_future_canon_multi_delta_hold_gate_closed",
            "advance_to_8_7_56_1126": future_canon_multi_delta_hold_contract_ready,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"audit_summary": audit["summary"]},
    )

    route = payload(
        "8.7.56.1126",
        "Trial-2 numeric alpha route contract one-hundred-seventy-eighth refresh",
        inputs,
        [
            row("route_contract_complete", "pass" if future_canon_multi_delta_hold_contract_ready else "reject", "route contract one-hundred-seventy-eighth refresh complete", 1 if future_canon_multi_delta_hold_contract_ready else 0, "The top-level hold contract is converted into the next-generation carry-over / share-pack route contract."),
            row("multi_delta_hold_contract_completed", "pass" if future_canon_multi_delta_hold_contract_ready else "reject", "future-canon multi-delta hold contract completed", 1 if future_canon_multi_delta_hold_contract_ready else 0, "The theorem, bridge, rewrite, and reserve items are now frozen together as one carried hold contract."),
            row("carry_over_share_pack_selected_as_next_route", "pass" if carry_over_share_pack_route_required else "reject", "future-canon carry-over share-pack registry selected as next route", 1 if carry_over_share_pack_route_required else 0, "The next step moves to share-pack synchronization for the carried future-canon hold state."),
            row("physical_reject_not_selected", "pass" if physical_reject_false_retained else "reject", "physical reject not selected after hold contract", 1 if physical_reject_false_retained else 0, "The route remains structurally alive after freezing the top-level hold contract."),
        ],
        {
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "future_canon_multi_delta_hold_contract_completed": future_canon_multi_delta_hold_contract_ready,
            "future_canon_delta_registry_items": ALL_DELTA_ITEMS,
            "all_four_delta_items_frozen": all_four_delta_items_frozen,
            "future_canon_candidate_retained": future_canon_candidate_retained,
            "hold_policy_frozen": hold_policy_frozen,
            "reopen_prerequisite_satisfied_under_current_canon": False,
            "physical_reject_required": False,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_route_contract_one_hundred_seventy_eighth_refresh_frozen",
            "advance_to_next_route": future_canon_multi_delta_hold_contract_ready,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"gate_summary": gate["summary"], "audit_summary": audit["summary"]},
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "alpha_is_prediction_future_canon_multi_delta_hold_contract_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "alpha_is_prediction_future_canon_multi_delta_hold_contract_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "alpha_is_prediction_future_canon_multi_delta_hold_contract_declaration_gate",
        gate,
    )
    write_artifact("mass_origin_v2_t2_alpha_route_contract_one_hundred_seventy_eighth_refresh", route)

    print("[done] 8.7.56.1123-.1126 artifacts generated")


if __name__ == "__main__":
    main()
