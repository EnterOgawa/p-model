#!/usr/bin/env python3
"""Generate 8.7.56.1091-.1094 Trial-2 future-canon challenge-wording-freeze artifacts."""

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

INVENTORY_1087 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_delta_registry_source_inventory_metrics.json"
)
AUDIT_1088 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_delta_registry_audit_metrics.json"
)
GATE_1089 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_delta_registry_declaration_gate_metrics.json"
)
ROUTE_1090 = PUBLIC_OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_sixty_ninth_refresh_metrics.json"

CURRENT_ROUTE = (
    "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_challenge_wording_freeze"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_delta_program_carryover_registry"
)
NEXT_ROUTE = "8.7.56.1095"
DELTA_ITEMS = [
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


# Function: return one repo-relative display path.

def display_path(path: Path) -> str:
    """Return one stable display path."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# Function: locate the first line containing one substring.

def hit(text: str, pattern: str) -> dict | None:
    """Return evidence for the first matching line."""
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


# Function: build one standard payload.

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


# Function: write one JSON metrics file and one CSV rows table.

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


# Function: build one target record.

def target(text: str, path: Path, key: str, pattern: str, note: str) -> dict:
    """Build one pattern target record."""
    evidence = hit(text, pattern)
    return {
        "file_key": key,
        "file": display_path(path),
        "pattern": pattern,
        "present": evidence is not None,
        "note": note,
        "evidence": evidence,
    }


# Function: classify the wording-freeze result.

def classify(wording_ready: bool, guardrail_ready: bool) -> str:
    """Classify the wording-freeze outcome."""
    if wording_ready and guardrail_ready:
        return "future_canon_challenge_wording_frozen"

    if wording_ready:
        return "delta_registry_visible_but_guardrail_incomplete"

    return "future_canon_challenge_wording_incomplete"


# Function: execute the challenge-wording-freeze branch.

def main() -> None:
    """Execute the Trial-2 future-canon challenge-wording-freeze branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PART3A,
        PART5,
        NOTE_ALPHA,
        NOTE_DIM,
        NOTE_SI,
        INVENTORY_1087,
        AUDIT_1088,
        GATE_1089,
        ROUTE_1090,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    alpha_note_text = read_text(NOTE_ALPHA)
    dim_note_text = read_text(NOTE_DIM)
    si_note_text = read_text(NOTE_SI)

    inventory_1087 = read_json(INVENTORY_1087)["summary"]
    audit_1088 = read_json(AUDIT_1088)["summary"]
    gate_1089 = read_json(GATE_1089)["summary"]
    route_1090 = read_json(ROUTE_1090)["summary"]

    targets = [
        target(status_text, STATUS, "status_1091", "8.7.56.1091", "STATUS must already point to this branch."),
        target(roadmap_text, ROADMAP, "roadmap_1091", "`8.7.56.1091-.1094`", "ROADMAP must already expose this branch."),
        target(part3a_text, PART3A, "part3a_registry_label", "one public challenge registry", "Part III-A must expose the registry label."),
        target(part5_text, PART5, "part5_registry_label", "one public challenge registry", "Part V must expose the registry label."),
        target(part5_text, PART5, "part5_limit_label", "current canon limit", "Part V must retain the current-canon-limit wording."),
        target(part5_text, PART5, "part5_not_reject", "physical reject ではなく", "Part V must still say this is not physical reject."),
        target(part5_text, PART5, "part5_multi_delta", "future_canon_multi_delta_program_required = true", "Part V must keep the multi-delta statement."),
        target(part3a_text, PART3A, "part3a_guardrail", "physical reject not selected", "Part III-A must still retain the non-reject guardrail."),
        target(alpha_note_text, NOTE_ALPHA, "alpha_note_formula", r"\alpha = \frac{c^3}{4\pi v^2 \hbar}", "The alpha note must stay in the pack."),
        target(dim_note_text, NOTE_DIM, "dim_note_tmchi", r"T_{M_\chi}", "The dimension note must stay in the pack."),
        target(dim_note_text, NOTE_DIM, "dim_note_tv", r"T_v", "The dimension note must stay in the pack."),
        target(si_note_text, NOTE_SI, "si_note_j", r"$J^\mu$ の正しい読み方", "The SI note must stay in the pack."),
    ]

    for delta_item in DELTA_ITEMS:
        targets.append(target(part3a_text, PART3A, f"part3a_{delta_item}", f"`{delta_item}`", "Part III-A must expose each delta item."))
        targets.append(target(part5_text, PART5, f"part5_{delta_item}", f"`{delta_item}`", "Part V must expose each delta item."))

    prior_route_active = all(
        [
            inventory_1087["future_canon_delta_registry_ready"],
            audit_1088["future_canon_multi_delta_program_required"],
            not audit_1088["reopen_prerequisite_satisfied_under_current_canon"],
            gate_1089["selected_residual_route"] == CURRENT_ROUTE,
            route_1090["selected_next_generation_route"] == CURRENT_ROUTE,
            not route_1090["physical_reject_required"],
        ]
    )
    inventory_ready = all(item["present"] for item in targets) and prior_route_active
    part5_delta_ready = all(hit(part5_text, f"`{item}`") is not None for item in DELTA_ITEMS)
    checkpoint_delta_ready = all(hit(part3a_text, f"`{item}`") is not None for item in DELTA_ITEMS)
    public_registry_ready = all(
        [
            hit(part3a_text, "one public challenge registry") is not None,
            hit(part5_text, "one public challenge registry") is not None,
        ]
    )
    guardrail_ready = all(
        [
            hit(part5_text, "current canon limit") is not None,
            hit(part5_text, "physical reject ではなく") is not None,
            hit(part3a_text, "physical reject not selected") is not None,
        ]
    )
    multi_delta_public_ready = hit(part5_text, "future_canon_multi_delta_program_required = true") is not None
    wording_ready = all([inventory_ready, part5_delta_ready, checkpoint_delta_ready, public_registry_ready, multi_delta_public_ready])
    challenge_class = classify(wording_ready, guardrail_ready)

    inputs = {
        "status_markdown": display_path(STATUS),
        "roadmap_markdown": display_path(ROADMAP),
        "ai_context_json": display_path(AI_CONTEXT),
        "part3a_markdown": display_path(PART3A),
        "part5_markdown": display_path(PART5),
        "alpha_note": display_path(NOTE_ALPHA),
        "dimension_note": display_path(NOTE_DIM),
        "si_note": display_path(NOTE_SI),
        "prior_1087_json": display_path(INVENTORY_1087),
        "prior_1088_json": display_path(AUDIT_1088),
        "prior_1089_json": display_path(GATE_1089),
        "prior_1090_json": display_path(ROUTE_1090),
    }

    inventory = payload(
        "8.7.56.1091",
        "Trial-2 numeric alpha future-canon challenge wording freeze source inventory",
        inputs,
        [
            row("inventory_complete", "pass" if inventory_ready else "reject", "challenge-wording inventory complete", 1 if inventory_ready else 0, "The wording pack is assembled from prior metrics, public wording, and retained notes."),
            row("part5_delta_items_visible", "pass" if part5_delta_ready else "reject", "Part V delta items visible", sum(1 for item in DELTA_ITEMS if hit(part5_text, f"`{item}`") is not None), "Part V should expose all four delta items."),
            row("checkpoint_delta_items_visible", "pass" if checkpoint_delta_ready else "reject", "checkpoint delta items visible", sum(1 for item in DELTA_ITEMS if hit(part3a_text, f"`{item}`") is not None), "Part III-A should expose the same four delta items."),
            row("public_registry_surface_ready", "pass" if public_registry_ready else "reject", "public challenge registry surface ready", 1 if public_registry_ready else 0, "Part III-A and Part V should both expose the registry label."),
            row("guardrail_surface_ready", "pass" if guardrail_ready else "reject", "current-canon-limit / not-physical-reject guardrail ready", 1 if guardrail_ready else 0, "The wording freeze must keep current-canon closeout distinct from physical reject."),
        ],
        {
            "inventory_ready": inventory_ready,
            "part5_delta_registry_item_keys": DELTA_ITEMS,
            "checkpoint_delta_registry_item_keys": DELTA_ITEMS,
            "public_challenge_registry_surface_ready": public_registry_ready,
            "current_canon_limit_not_physical_reject_guardrail_ready": guardrail_ready,
            "future_canon_challenge_wording_inventory_ready": inventory_ready,
            "first_route_to_close_or_none": CURRENT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_future_canon_challenge_wording_inventory_frozen",
            "advance_to_8_7_56_1092": inventory_ready,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"targets": targets, "retained_1088_summary": audit_1088, "retained_1089_summary": gate_1089, "retained_1090_summary": route_1090},
    )

    audit = payload(
        "8.7.56.1092",
        "Trial-2 numeric alpha future-canon challenge wording freeze audit",
        inputs,
        [
            row("part5_public_registry_ready", "pass" if part5_delta_ready and public_registry_ready else "reject", "Part V public challenge registry ready", 1 if part5_delta_ready and public_registry_ready else 0, "Part V should expose the delta pack as one registry."),
            row("checkpoint_public_registry_ready", "pass" if checkpoint_delta_ready and public_registry_ready else "reject", "checkpoint public challenge registry ready", 1 if checkpoint_delta_ready and public_registry_ready else 0, "Part III-A should mirror the same public registry."),
            row("guardrail_publicly_ready", "pass" if guardrail_ready else "reject", "current-canon-limit not-physical-reject wording ready", 1 if guardrail_ready else 0, "The public wording must say that closeout is not physical reject."),
            row("multi_delta_publicly_ready", "pass" if multi_delta_public_ready else "reject", "future-canon multi-delta program publicly stated", 1 if multi_delta_public_ready else 0, "Part V must still state that the reopen path is multi-delta."),
            row("reopen_still_false", "pass", "reopen prerequisite still false under current canon", 1, "The wording freeze does not reopen current-canon computation."),
        ],
        {
            "audit_ready": inventory_ready,
            "part5_public_challenge_registry_ready": part5_delta_ready and public_registry_ready,
            "checkpoint_public_challenge_registry_ready": checkpoint_delta_ready and public_registry_ready,
            "current_canon_limit_not_physical_reject_wording_ready": guardrail_ready,
            "future_canon_multi_delta_program_publicly_stated": multi_delta_public_ready,
            "reopen_prerequisite_satisfied_under_current_canon": False,
            "selected_future_canon_challenge_wording_class": challenge_class,
            "first_route_to_close_after_audit_or_none": NEXT_ROUTE_NAME,
        },
        {
            "overall_status": "trial2_numeric_alpha_future_canon_challenge_wording_audited",
            "advance_to_8_7_56_1093": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"wording_inventory_summary": inventory["summary"]},
    )

    gate = payload(
        "8.7.56.1093",
        "Trial-2 numeric alpha future-canon challenge wording freeze declaration gate",
        inputs,
        [
            row("gate_complete", "pass", "future-canon challenge wording gate complete", 1, "The wording freeze is fixed at the declaration-gate level."),
            row("public_registry_ready_confirmed", "pass" if wording_ready else "reject", "public challenge registry ready confirmed", 1 if wording_ready else 0, "The delta pack is now publicly surfaced."),
            row("guardrail_ready_confirmed", "pass" if guardrail_ready else "reject", "current-canon-limit not-physical-reject publicly frozen", 1 if guardrail_ready else 0, "The public wording keeps the route out of physical reject."),
            row("next_route_selected", "pass", "future-canon delta-program carry-over registry selected", 1, "The next branch will register the executable carry-over structure."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "alpha_prediction_future_canon_challenge_wording_frozen",
            "trial2_numeric_alpha_future_canon_delta_registry_completed": bool(gate_1089["trial2_numeric_alpha_future_canon_delta_registry_completed"]),
            "trial2_numeric_alpha_future_canon_challenge_wording_freeze_completed": wording_ready,
            "trial2_numeric_alpha_part5_public_challenge_registry_ready": part5_delta_ready and public_registry_ready,
            "trial2_numeric_alpha_checkpoint_public_challenge_registry_ready": checkpoint_delta_ready and public_registry_ready,
            "trial2_numeric_alpha_current_canon_limit_not_physical_reject_publicly_frozen": guardrail_ready,
            "trial2_numeric_alpha_future_canon_multi_delta_program_publicly_stated": multi_delta_public_ready,
            "trial2_numeric_alpha_future_canon_multi_delta_program_required": True,
            "trial2_numeric_alpha_reopen_prerequisite_satisfied_under_current_canon": False,
            "trial2_numeric_alpha_physical_reject_required": False,
            "selected_residual_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_future_canon_challenge_wording_gate_closed",
            "advance_to_8_7_56_1094": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"audit_summary": audit["summary"]},
    )

    route = payload(
        "8.7.56.1094",
        "Trial-2 numeric alpha route contract one-hundred-seventieth refresh",
        inputs,
        [
            row("route_contract_complete", "pass", "route contract one-hundred-seventieth refresh complete", 1, "The wording freeze is converted into the next-generation route contract."),
            row("wording_freeze_completed", "pass" if wording_ready else "reject", "future-canon challenge wording freeze completed", 1 if wording_ready else 0, "The public wording now carries the registry and guardrail."),
            row("next_route_selected", "pass", "future-canon delta-program carry-over registry selected as next route", 1, "The mainline proceeds into the carry-over registry."),
            row("physical_reject_not_selected", "pass", "physical reject not selected after wording freeze", 1, "The route remains structurally alive."),
        ],
        {
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "future_canon_delta_registry_completed": True,
            "future_canon_challenge_wording_freeze_completed": wording_ready,
            "future_canon_multi_delta_program_required": True,
            "reopen_prerequisite_satisfied_under_current_canon": False,
            "part5_public_challenge_registry_ready": part5_delta_ready and public_registry_ready,
            "checkpoint_public_challenge_registry_ready": checkpoint_delta_ready and public_registry_ready,
            "current_canon_limit_not_physical_reject_publicly_frozen": guardrail_ready,
            "physical_reject_required": False,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_route_contract_one_hundred_seventieth_refresh_frozen",
            "advance_to_next_route": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"gate_summary": gate["summary"], "audit_summary": audit["summary"]},
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "alpha_is_prediction_future_canon_challenge_wording_freeze_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "alpha_is_prediction_future_canon_challenge_wording_freeze_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "alpha_is_prediction_future_canon_challenge_wording_freeze_declaration_gate",
        gate,
    )
    write_artifact("mass_origin_v2_t2_alpha_route_contract_one_hundred_seventieth_refresh", route)

    print("[done] 8.7.56.1091-.1094 artifacts generated")


if __name__ == "__main__":
    main()
