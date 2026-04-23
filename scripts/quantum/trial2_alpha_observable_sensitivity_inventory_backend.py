#!/usr/bin/env python3
"""Inventory current observable-level alpha comparison surfaces.

Purpose:
    Trial-2 mainline is no longer exact-constant collapse. The new honest
    question is whether P-model native alpha can be compared against
    independent observables without reusing CODATA extraction inputs.

    This helper audits the current Part III-B verification surfaces and fixes
    three facts:

    1. which surfaces already expose alpha explicitly inside the current pack,
    2. which of those surfaces are excluded from the primary score because
       they overlap CODATA-input style alpha extractions,
    3. whether any independent primary-ready rerun surface actually exists now.

Inputs:
    - scripts/quantum/de_broglie_precision_alpha_consistency.py
    - scripts/quantum/qed_vacuum_precision.py
    - scripts/quantum/weak_interaction_beta_decay_route_ab_audit.py
    - scripts/quantum/weak_interaction_ckm_first_row_audit.py
    - scripts/quantum/weak_interaction_pmns_first_row_audit.py
    - output/public/quantum/* existing metrics JSONs

Outputs:
    - One in-memory audit pack consumed by `.5887-.5898` wrappers

Assumptions:
    - `alpha_P_frozen` is the canonical native checkpoint
    - direct CODATA-input observables are excluded from the primary score
    - current pack must distinguish "alpha-sensitive in physics" from
      "rerun-ready in current implementation"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

ALPHA_P_FROZEN = 0.007302943961943229
ALPHA_P_4D_CAN = 0.0072988143426522215
ALPHA_P_4D_VERTEX = 0.007299279720153683
ALPHA_CODATA = 0.0072973525643


# 関数: 1 つの artifact path の存在情報を返す。
def build_file_info(path: Path) -> dict:
    """Return one compact file-info dictionary."""
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "exists": bool(path.exists()),
    }


# 関数: JSON artifact を可能なら読み込む。

def read_json_if_exists(path: Path) -> dict | None:
    """Read one UTF-8 JSON file when it exists."""
    if not path.exists():
        return None

    return json.loads(path.read_text(encoding="utf-8"))


# 関数: observable candidate rows を返す。

def build_candidate_rows() -> list[dict]:
    """Return the retained observable-level candidate rows."""
    de_broglie_metrics = PUBLIC_OUT / "de_broglie_precision_alpha_consistency_metrics.json"
    qed_metrics = PUBLIC_OUT / "qed_vacuum_precision_metrics.json"
    weak_ab_metrics = PUBLIC_OUT / "weak_interaction_beta_decay_route_ab_audit.json"
    weak_ckm_metrics = PUBLIC_OUT / "weak_interaction_ckm_first_row_audit.json"
    weak_pmns_metrics = PUBLIC_OUT / "weak_interaction_pmns_first_row_audit.json"

    de_broglie_payload = read_json_if_exists(de_broglie_metrics)
    weak_ab_payload = read_json_if_exists(weak_ab_metrics)
    weak_ckm_payload = read_json_if_exists(weak_ckm_metrics)
    weak_pmns_payload = read_json_if_exists(weak_pmns_metrics)

    recoil_alpha_inv = None
    g2_alpha_inv = None
    recoil_vs_g2_z = None
    if de_broglie_payload is not None:
        recoil_alpha_inv = float(de_broglie_payload["measurements"][0]["alpha_inv"])
        g2_alpha_inv = float(de_broglie_payload["measurements"][1]["alpha_inv"])
        recoil_vs_g2_z = float(de_broglie_payload["derived"]["z_score"])

    weak_ab_hard = None
    if weak_ab_payload is not None:
        weak_ab_hard = bool(weak_ab_payload["decision"]["route_a_hard_pass"])

    weak_ckm_abs_z = None
    if weak_ckm_payload is not None:
        weak_ckm_abs_z = float(weak_ckm_payload["derived"]["abs_z_reported"])

    weak_pmns_abs_z = None
    if weak_pmns_payload is not None:
        selected_dataset = str(weak_pmns_payload["gate"]["dataset"])
        weak_pmns_abs_z = float(
            weak_pmns_payload["derived"][selected_dataset]["abs_z_center_proxy"]
        )

    return [
        {
            "observable_id": "de_broglie_alpha_consistency",
            "label": "de Broglie / recoil-vs-g-2 alpha consistency",
            "sector": "matter-wave precision alpha",
            "script_path": "scripts/quantum/de_broglie_precision_alpha_consistency.py",
            "metrics_path": build_file_info(de_broglie_metrics),
            "alpha_dependency_kind": "explicit_extracted_alpha_comparison",
            "structural_alpha_sensitivity_rank": "very_high",
            "current_alpha_rerun_ready_now": True,
            "codata_input_overlap_now": True,
            "primary_score_admissible_now": False,
            "reserve_diagnostic_now": True,
            "selected_priority_rank": 0,
            "key_metric": recoil_vs_g2_z,
            "key_metric_label": "recoil_vs_g2_z",
            "notes": (
                "This is the only current pack surface that already carries alpha "
                "explicitly as data, but it directly reuses recoil and electron g-2 "
                "alpha determinations and is therefore excluded from the primary score."
            ),
            "observed_values": {
                "alpha_inv_recoil": recoil_alpha_inv,
                "alpha_inv_g2": g2_alpha_inv,
            },
        },
        {
            "observable_id": "qed_vacuum_baseline_pack",
            "label": "QED vacuum baseline pack (Casimir / Lamb / H 1S-2S)",
            "sector": "vacuum / spectroscopy",
            "script_path": "scripts/quantum/qed_vacuum_precision.py",
            "metrics_path": build_file_info(qed_metrics),
            "alpha_dependency_kind": "structurally_alpha_sensitive_but_not_parameterized",
            "structural_alpha_sensitivity_rank": "high",
            "current_alpha_rerun_ready_now": False,
            "codata_input_overlap_now": False,
            "primary_score_admissible_now": False,
            "reserve_diagnostic_now": False,
            "selected_priority_rank": 1,
            "key_metric": None,
            "key_metric_label": "not_available",
            "notes": (
                "Casimir, Lamb, and H 1S-2S are physically relevant high-sensitivity "
                "targets, but the current script fixes baseline scales and safety checks; "
                "it does not yet expose alpha as an actual rerun input."
            ),
            "observed_values": {},
        },
        {
            "observable_id": "weak_beta_decay_route_ab",
            "label": "Weak beta-decay Route A/B pack",
            "sector": "weak sector",
            "script_path": "scripts/quantum/weak_interaction_beta_decay_route_ab_audit.py",
            "metrics_path": build_file_info(weak_ab_metrics),
            "alpha_dependency_kind": "indirect_electroweak_surrogate_no_explicit_alpha",
            "structural_alpha_sensitivity_rank": "medium",
            "current_alpha_rerun_ready_now": False,
            "codata_input_overlap_now": False,
            "primary_score_admissible_now": False,
            "reserve_diagnostic_now": False,
            "selected_priority_rank": 2,
            "key_metric": weak_ab_hard,
            "key_metric_label": "route_ab_hard_pass",
            "notes": (
                "The weak Route A/B pack is independent enough for later observable "
                "comparison, but alpha is not currently exposed as a deterministic rerun "
                "input in the public implementation."
            ),
            "observed_values": {},
        },
        {
            "observable_id": "weak_ckm_first_row",
            "label": "CKM first-row closure",
            "sector": "weak sector",
            "script_path": "scripts/quantum/weak_interaction_ckm_first_row_audit.py",
            "metrics_path": build_file_info(weak_ckm_metrics),
            "alpha_dependency_kind": "independent_but_alpha_inactive",
            "structural_alpha_sensitivity_rank": "low",
            "current_alpha_rerun_ready_now": False,
            "codata_input_overlap_now": False,
            "primary_score_admissible_now": False,
            "reserve_diagnostic_now": False,
            "selected_priority_rank": 3,
            "key_metric": weak_ckm_abs_z,
            "key_metric_label": "abs_z_ckm",
            "notes": (
                "Useful as global weak-sector consistency evidence, but not a direct alpha "
                "comparison surface in the current pack."
            ),
            "observed_values": {},
        },
        {
            "observable_id": "weak_pmns_first_row",
            "label": "PMNS first-row closure",
            "sector": "weak sector",
            "script_path": "scripts/quantum/weak_interaction_pmns_first_row_audit.py",
            "metrics_path": build_file_info(weak_pmns_metrics),
            "alpha_dependency_kind": "independent_but_alpha_inactive",
            "structural_alpha_sensitivity_rank": "low",
            "current_alpha_rerun_ready_now": False,
            "codata_input_overlap_now": False,
            "primary_score_admissible_now": False,
            "reserve_diagnostic_now": False,
            "selected_priority_rank": 4,
            "key_metric": weak_pmns_abs_z,
            "key_metric_label": "abs_z_pmns",
            "notes": (
                "Useful as global weak-sector closure evidence, but it does not currently "
                "carry an explicit alpha lever for Trial-2 observable comparison."
            ),
            "observed_values": {},
        },
    ]


# 関数: sensitivity inventory pack を束ねる。

def build_trial2_alpha_observable_sensitivity_inventory_pack() -> dict:
    """Return one audit pack for observable-level alpha sensitivity."""
    candidates = build_candidate_rows()
    explicit_alpha_count = sum(
        1 for row in candidates if row["alpha_dependency_kind"] == "explicit_extracted_alpha_comparison"
    )
    codata_overlap_count = sum(1 for row in candidates if row["codata_input_overlap_now"])
    independent_count = sum(1 for row in candidates if not row["codata_input_overlap_now"])
    primary_ready_count = sum(1 for row in candidates if row["primary_score_admissible_now"])
    reserve_ready_count = sum(1 for row in candidates if row["reserve_diagnostic_now"])
    rerun_ready_independent_count = sum(
        1
        for row in candidates
        if row["current_alpha_rerun_ready_now"] and not row["codata_input_overlap_now"]
    )

    primary_materialization = next(
        row for row in candidates if row["observable_id"] == "qed_vacuum_baseline_pack"
    )
    secondary_materialization = next(
        row for row in candidates if row["observable_id"] == "weak_beta_decay_route_ab"
    )
    reserve_diagnostic = next(
        row for row in candidates if row["observable_id"] == "de_broglie_alpha_consistency"
    )

    return {
        "alpha_constants": {
            "alpha_P_frozen": ALPHA_P_FROZEN,
            "alpha_P_4D_can": ALPHA_P_4D_CAN,
            "alpha_P_4D_vertex": ALPHA_P_4D_VERTEX,
            "alpha_CODATA": ALPHA_CODATA,
        },
        "candidates": candidates,
        "summary": {
            "total_candidate_surfaces": len(candidates),
            "explicit_alpha_surface_count": explicit_alpha_count,
            "codata_overlap_surface_count": codata_overlap_count,
            "independent_surface_count": independent_count,
            "primary_ready_surface_count": primary_ready_count,
            "reserve_diagnostic_surface_count": reserve_ready_count,
            "rerun_ready_independent_surface_count": rerun_ready_independent_count,
            "selected_primary_materialization_observable_id": primary_materialization[
                "observable_id"
            ],
            "selected_secondary_materialization_observable_id": secondary_materialization[
                "observable_id"
            ],
            "selected_reserve_diagnostic_observable_id": reserve_diagnostic[
                "observable_id"
            ],
        },
        "trial2_alpha_observable_sensitivity_inventory_complete_now": True,
        "trial2_alpha_current_primary_ready_surface_available_now": bool(
            primary_ready_count > 0
        ),
        "trial2_alpha_current_rerun_ready_independent_surface_available_now": bool(
            rerun_ready_independent_count > 0
        ),
        "trial2_alpha_primary_materialization_qed_vacuum_now": True,
        "trial2_alpha_secondary_materialization_weak_sector_now": True,
        "trial2_alpha_reserve_diagnostic_de_broglie_now": True,
    }


# 関数: backend 単体実行時に compact summary を表示する。

def main() -> None:
    """Run the observable-sensitivity inventory directly."""
    pack = build_trial2_alpha_observable_sensitivity_inventory_pack()
    summary = pack["summary"]
    print("[trial2_alpha_observable_sensitivity_inventory_backend]")
    print(
        "  total_candidate_surfaces = "
        f"{summary['total_candidate_surfaces']}"
    )
    print(
        "  explicit_alpha_surface_count = "
        f"{summary['explicit_alpha_surface_count']}"
    )
    print(
        "  primary_ready_surface_count = "
        f"{summary['primary_ready_surface_count']}"
    )
    print(
        "  selected_primary_materialization = "
        f"{summary['selected_primary_materialization_observable_id']}"
    )


if __name__ == "__main__":
    main()
