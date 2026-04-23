#!/usr/bin/env python3
"""Audit alpha-dependency materialization inside current weak-sector packs.

Purpose:
    After the QED-vacuum baseline pack is selected as the primary
    materialization candidate, the weak sector remains the next independent
    comparison surface. This backend fixes whether the current public weak
    scripts already expose one explicit alpha rerun lever.

Inputs:
    - scripts/quantum/weak_interaction_beta_decay_route_ab_audit.py
    - scripts/quantum/weak_interaction_ckm_first_row_audit.py
    - scripts/quantum/weak_interaction_pmns_first_row_audit.py
    - output/public/quantum/weak_interaction_* audit JSONs

Outputs:
    - One in-memory audit pack consumed by `.5903-.5906` wrappers
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


# 関数: JSON artifact を可能なら読み込む。
def read_json_if_exists(path: Path) -> dict | None:
    """Read one UTF-8 JSON payload when it exists."""
    if not path.exists():
        return None

    return json.loads(path.read_text(encoding="utf-8"))


# 関数: weak-sector rows を返す。

def build_weak_sector_rows() -> list[dict]:
    """Return retained weak-sector candidate rows."""
    route_ab_payload = read_json_if_exists(PUBLIC_OUT / "weak_interaction_beta_decay_route_ab_audit.json")
    ckm_payload = read_json_if_exists(PUBLIC_OUT / "weak_interaction_ckm_first_row_audit.json")
    pmns_payload = read_json_if_exists(PUBLIC_OUT / "weak_interaction_pmns_first_row_audit.json")

    route_ab_transition = None
    if isinstance(route_ab_payload, dict):
        decision = route_ab_payload.get("decision")
        if isinstance(decision, dict):
            route_ab_transition = str(decision.get("transition"))

    ckm_abs_z = None
    if isinstance(ckm_payload, dict):
        derived = ckm_payload.get("derived")
        if isinstance(derived, dict):
            ckm_abs_z = float(derived["abs_z_reported"])

    pmns_abs_z = None
    if isinstance(pmns_payload, dict):
        dataset = str(pmns_payload.get("gate", {}).get("dataset"))
        derived = pmns_payload.get("derived")
        if isinstance(derived, dict) and isinstance(derived.get(dataset), dict):
            pmns_abs_z = float(derived[dataset]["abs_z_center_proxy"])

    return [
        {
            "surface_id": "weak_beta_decay_route_ab",
            "label": "Weak beta-decay Route A/B",
            "alpha_dependency_kind": "indirect_electroweak_surrogate_no_public_alpha_input",
            "current_alpha_rerun_ready_now": False,
            "independent_observable_now": True,
            "selected_secondary_target_now": True,
            "primary_score_admissible_now": False,
            "notes": (
                "This pack is independent and already quantitative, but alpha is "
                "not yet exposed as an explicit public rerun lever in the current implementation."
            ),
            "key_metric": route_ab_transition,
            "key_metric_label": "route_ab_transition",
        },
        {
            "surface_id": "weak_ckm_first_row",
            "label": "CKM first-row closure",
            "alpha_dependency_kind": "alpha_inactive_consistency_surface",
            "current_alpha_rerun_ready_now": False,
            "independent_observable_now": True,
            "selected_secondary_target_now": False,
            "primary_score_admissible_now": False,
            "notes": (
                "CKM closure is a useful weak-sector reference surface, but alpha "
                "does not appear as a public rerun variable in the current pack."
            ),
            "key_metric": ckm_abs_z,
            "key_metric_label": "abs_z_ckm",
        },
        {
            "surface_id": "weak_pmns_first_row",
            "label": "PMNS first-row closure",
            "alpha_dependency_kind": "alpha_inactive_consistency_surface",
            "current_alpha_rerun_ready_now": False,
            "independent_observable_now": True,
            "selected_secondary_target_now": False,
            "primary_score_admissible_now": False,
            "notes": (
                "PMNS closure is a useful weak-sector reference surface, but alpha "
                "does not appear as a public rerun variable in the current pack."
            ),
            "key_metric": pmns_abs_z,
            "key_metric_label": "abs_z_pmns",
        },
    ]


# 関数: `.5903-.5906` 用の audit pack を束ねる。

def build_trial2_weak_sector_alpha_materialization_pack() -> dict:
    """Return the retained weak-sector materialization audit pack."""
    rows = build_weak_sector_rows()
    independent_rows = [row for row in rows if row["independent_observable_now"]]
    selected_secondary_rows = [row for row in rows if row["selected_secondary_target_now"]]
    primary_ready_rows = [row for row in rows if row["primary_score_admissible_now"]]
    rerun_ready_rows = [row for row in rows if row["current_alpha_rerun_ready_now"]]

    return {
        "alpha_constants": {
            "alpha_P_frozen": ALPHA_P_FROZEN,
            "alpha_P_4D_can": ALPHA_P_4D_CAN,
            "alpha_P_4D_vertex": ALPHA_P_4D_VERTEX,
            "alpha_CODATA": ALPHA_CODATA,
        },
        "surfaces": rows,
        "summary": {
            "weak_surface_count": len(rows),
            "weak_independent_surface_count": len(independent_rows),
            "weak_selected_secondary_target_count": len(selected_secondary_rows),
            "weak_rerun_ready_count": len(rerun_ready_rows),
            "weak_primary_ready_count": len(primary_ready_rows),
            "selected_secondary_target_ids": [
                str(row["surface_id"]) for row in selected_secondary_rows
            ],
        },
        "trial2_weak_sector_materialization_complete_now": True,
        "trial2_weak_sector_positive_partial_now": bool(len(selected_secondary_rows) > 0),
        "trial2_weak_sector_primary_ready_now": bool(len(primary_ready_rows) > 0),
        "trial2_weak_sector_rerun_ready_now": bool(len(rerun_ready_rows) > 0),
    }


# 関数: backend 単体実行時に compact summary を表示する。

def main() -> None:
    """Run the weak-sector materialization audit directly."""
    pack = build_trial2_weak_sector_alpha_materialization_pack()
    summary = pack["summary"]
    print("[trial2_weak_sector_alpha_materialization_backend]")
    print(f"  weak_surface_count = {summary['weak_surface_count']}")
    print(f"  weak_primary_ready_count = {summary['weak_primary_ready_count']}")
    print(
        "  selected_secondary_target_ids = "
        f"{summary['selected_secondary_target_ids']}"
    )


if __name__ == "__main__":
    main()
