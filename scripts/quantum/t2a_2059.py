#!/usr/bin/env python3
"""Generate 8.7.56.2059-.2062 higher-harmonic loading registry artifacts."""

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

import scripts.quantum.t2a_1843 as sign_base
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
LONG_ROADMAP = ROOT / "doc" / "quantum" / "55_trial2_numeric_alpha_vector_qball_long_horizon_roadmap.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

PRIOR_GATE = PUBLIC_OUT / "q_8_7_56_2055_2058_higher_harmonic_lattice_loading_declaration_gate_metrics.json"

STEP_TAG = "8.7.56.2059-2062"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor higher-harmonic loading decision gate / registry"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "higher_harmonic_loading_registry",
    prefix="q",
)

PRIOR_CLASS = "vector_qball_form_factor_boundary_bulk_lattice_signed_rule_retained_loading_index_theorem_gate_next"
BRANCH_CLASS = "vector_qball_form_factor_boundary_bulk_lattice_signed_rule_partial_retain_loading_index_theorem_or_farther_extension_next"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_exact_loading_index_theorem_or_farther_harmonic_extension_reactivation"
NEXT_ROUTE = "8.7.56.2063"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_bulk_lattice_loading_closeout_registry"
FOLLOWUP_ROUTE = "8.7.56.2067"


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: JSON/CSV artifact を書き出す。

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

    return {
        "json": sign_base.display_path(paths["json"]),
        "csv": sign_base.display_path(paths["csv"]),
    }


# 関数: テキスト中の最初の一致行を返す。

def find_line(text: str, pattern: str) -> dict[str, object] | None:
    """Return the first matching line payload for one text pattern."""
    for line_number, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {
                "pattern": pattern,
                "line": line_number,
                "text": line.strip(),
            }

    return None


# 関数: 使用公式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the registry sync."""
    return {
        "retained_local_jet_theorem": "delta_q,jet = (3/2) (h1 / h0)",
        "bulk_lattice_step": "Delta_box = Delta r_bulk",
        "theorem_base": "delta_q,base^(box) = (Delta_box - (delta_q,jet mod Delta_box)) mod Delta_box",
        "harmonic_loading_lattice": "delta_q^(n,m) = delta_q,base^(box) + m_n Delta_box",
    }


# 関数: `.2059-.2062` を実行する。

def main() -> None:
    """Execute the higher-harmonic loading decision gate / registry sync."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        LONG_ROADMAP,
        PART5,
        PRIOR_GATE,
    ):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    current_problem_text = sign_base.read_text(CURRENT_PROBLEM)
    current_status_text = sign_base.read_text(CURRENT_STATUS)
    unified_text = sign_base.read_text(UNIFIED_ROADMAP)
    long_text = sign_base.read_text(LONG_ROADMAP)
    part5_text = sign_base.read_text(PART5)
    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]

    inventory_ready = bool(prior_summary["farther_harmonic_extension_admissible_now"])
    gate_a_exact_lattice_promotion_selected = bool(prior_summary["exact_loading_index_theorem_available"])
    gate_b_partial_lattice_retain_selected = bool(prior_summary["boundary_bulk_lattice_signed_rule_supported"])
    gate_c_substantive_pack_update_required = False
    same_level_smooth_loading_retry_admissible = False
    same_level_base_scan_retry_admissible = False
    exact_loading_index_theorem_admissible_now = not gate_a_exact_lattice_promotion_selected
    farther_harmonic_extension_admissible_now = bool(prior_summary["farther_harmonic_extension_admissible_now"])
    substantive_pack_update_required_now = False
    physical_reject_required = False

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "higher-harmonic lattice registry inventory ready",
            sign_base.truth(inventory_ready),
            "The registry starts only after the harmonic bulk-lattice audit has shown that the family survives beyond the first translated windows.",
        ),
        sign_base.row(
            "gate_a_exact_lattice_promotion_selected",
            "reject" if not gate_a_exact_lattice_promotion_selected else "pass",
            "Gate A exact lattice promotion selected",
            sign_base.truth(gate_a_exact_lattice_promotion_selected),
            "Gate A would require the loading-index sequence itself to close as a theorem, not only the lattice family.",
        ),
        sign_base.row(
            "gate_b_partial_lattice_retain_selected",
            "pass" if gate_b_partial_lattice_retain_selected else "reject",
            "Gate B partial lattice retain selected",
            sign_base.truth(gate_b_partial_lattice_retain_selected),
            "Once the boundary bulk-lattice survives core, extension, and farther harmonics but the loading-index theorem is still missing, the honest read is partial retain.",
        ),
        sign_base.row(
            "gate_c_substantive_pack_update_required",
            "reject",
            "Gate C substantive pack update required",
            sign_base.truth(gate_c_substantive_pack_update_required),
            "The current pack still contains one unresolved theorem-level surface before any pack update is needed.",
        ),
        sign_base.row(
            "same_level_smooth_loading_retry_admissible",
            "reject",
            "same-level smooth loading retry admissible",
            sign_base.truth(same_level_smooth_loading_retry_admissible),
            "Smooth q-dependent loading is already closed and should not reopen.",
        ),
        sign_base.row(
            "same_level_base_scan_retry_admissible",
            "reject",
            "same-level bulk-lattice base scan retry admissible",
            sign_base.truth(same_level_base_scan_retry_admissible),
            "Once the theorem base nearly saturates the searched best base, another same-level base scan should stay closed.",
        ),
        sign_base.row(
            "exact_loading_index_theorem_admissible_now",
            "pass" if exact_loading_index_theorem_admissible_now else "reject",
            "exact loading-index theorem admissible now",
            sign_base.truth(exact_loading_index_theorem_admissible_now),
            "The next honest surface is a theorem for the harmonic loading indices themselves.",
        ),
        sign_base.row(
            "farther_harmonic_extension_admissible_now",
            "pass" if farther_harmonic_extension_admissible_now else "reject",
            "farther harmonic extension admissible now",
            sign_base.truth(farther_harmonic_extension_admissible_now),
            "The same retained lattice family can still be pushed to farther harmonics while the loading-index theorem remains open.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "bulk_delta_r_over_m0": float(prior_summary["bulk_delta_r_over_m0"]),
        "delta_q_theorem_over_m0": float(prior_summary["delta_q_theorem_over_m0"]),
        "theorem_lattice_base_over_m0": float(prior_summary["theorem_lattice_base_over_m0"]),
        "theorem_vs_searched_base_gap_over_m0": float(prior_summary["theorem_vs_searched_base_gap_over_m0"]),
        "theorem_fit_window_max_mismatch_fraction": float(prior_summary["theorem_fit_window_max_mismatch_fraction"]),
        "theorem_extension_window_max_mismatch_fraction": float(prior_summary["theorem_extension_window_max_mismatch_fraction"]),
        "theorem_farther_window_max_mismatch_fraction": float(prior_summary["theorem_farther_window_max_mismatch_fraction"]),
        "theorem_quantization_max_abs_gap_over_m0": float(prior_summary["theorem_quantization_max_abs_gap_over_m0"]),
        "theorem_quantization_rms_gap_over_m0": float(prior_summary["theorem_quantization_rms_gap_over_m0"]),
        "loading_index_monotone": bool(prior_summary["loading_index_monotone"]),
        "delta_sequence_monotone": bool(prior_summary["delta_sequence_monotone"]),
        "gate_a_exact_lattice_promotion_selected": gate_a_exact_lattice_promotion_selected,
        "gate_b_partial_lattice_retain_selected": gate_b_partial_lattice_retain_selected,
        "gate_c_substantive_pack_update_required": gate_c_substantive_pack_update_required,
        "same_level_smooth_loading_retry_admissible": same_level_smooth_loading_retry_admissible,
        "same_level_base_scan_retry_admissible": same_level_base_scan_retry_admissible,
        "exact_loading_index_theorem_admissible_now": exact_loading_index_theorem_admissible_now,
        "farther_harmonic_extension_admissible_now": farther_harmonic_extension_admissible_now,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "substantive_pack_update_required_now": substantive_pack_update_required_now,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2061",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "status": sign_base.display_path(STATUS),
                "roadmap": sign_base.display_path(ROADMAP),
                "ai_context": sign_base.display_path(AI_CONTEXT),
                "work_history_recent": sign_base.display_path(WORK_HISTORY_RECENT),
                "current_problem": sign_base.display_path(CURRENT_PROBLEM),
                "current_status": sign_base.display_path(CURRENT_STATUS),
                "unified_roadmap": sign_base.display_path(UNIFIED_ROADMAP),
                "long_roadmap": sign_base.display_path(LONG_ROADMAP),
                "part5": sign_base.display_path(PART5),
                "prior_gate": sign_base.display_path(PRIOR_GATE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_higher_harmonic_loading_registry_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2055"),
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2055-.2058"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2055"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2055"),
                "unified_roadmap_hit": find_line(unified_text, ".2055-.2058"),
                "long_roadmap_hit": find_line(long_text, ".2055-.2058"),
                "part5_hit": find_line(part5_text, ".2047-.2054"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_sync_rows = [
        sign_base.row(
            "status_synced",
            "pass",
            "STATUS sync target present",
            sign_base.truth(bool(find_line(status_text, "8.7.56.2055"))),
            "The registry is only valid if the status already points to the harmonic loading branch.",
        ),
        sign_base.row(
            "roadmap_synced",
            "pass",
            "ROADMAP sync target present",
            sign_base.truth(bool(find_line(roadmap_text, "8.7.56.2055-.2058"))),
            "The public roadmap must expose the harmonic loading branch before the registry is fixed.",
        ),
        sign_base.row(
            "long_horizon_synced",
            "pass",
            "long-horizon roadmap sync target present",
            sign_base.truth(bool(find_line(long_text, ".2055-.2058"))),
            "The long-horizon roadmap must also show the same harmonic loading route.",
        ),
    ]
    route_sync_payload = sign_base.payload(
        "8.7.56.2062",
        STEP_NAME + " route sync",
        {
            "source_files": {
                "status": sign_base.display_path(STATUS),
                "roadmap": sign_base.display_path(ROADMAP),
                "current_problem": sign_base.display_path(CURRENT_PROBLEM),
                "current_status": sign_base.display_path(CURRENT_STATUS),
                "unified_roadmap": sign_base.display_path(UNIFIED_ROADMAP),
                "long_roadmap": sign_base.display_path(LONG_ROADMAP),
                "part5": sign_base.display_path(PART5),
                "declaration_gate": declaration_paths["json"],
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
        route_sync_rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_higher_harmonic_loading_registry_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2055"),
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2055-.2058"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2055"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2055"),
                "unified_roadmap_hit": find_line(unified_text, ".2055-.2058"),
                "long_roadmap_hit": find_line(long_text, ".2055-.2058"),
                "part5_hit": find_line(part5_text, ".2047-.2054"),
            },
        },
    )
    write_artifact("route_sync", route_sync_payload)


if __name__ == "__main__":
    main()
