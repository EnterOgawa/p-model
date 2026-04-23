#!/usr/bin/env python3
"""Generate 8.7.56.1727-.1730 pack-update closeout / reopen registry refresh artifacts.

The source-extended probe-response pack has now been run through every honest
same-level follow-up surface available under the updated action:

1. canonical two-leg probe-response / amputation theorem closes externally,
2. canonical recomputation still fails scalar promotion,
3. exact constitutive-map reopen stays unavailable,
4. branch-local full nonlinear energy-density reopen carries over unchanged,
5. inverse local-family audit requires huge or noncanonical coefficients,
6. the latest external note is genuine as a file but opens no new primary
   surface because it mostly restates already executed response-theory routes.

`.1727-.1730` therefore does not seek another rescue. It freezes the updated
pack as an exhausted closeout and refreshes the reopen ordering so that future
reactivation is allowed only after a genuinely new surface arrives.
"""

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

PACK_UPDATE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1695_1698_pack_update_intake_declaration_gate_metrics.json"
)
DECISION_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1707_1710_probe_resp_gate_sync_declaration_gate_metrics.json"
)
CONSTITUTIVE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1711_1714_updpk_const_map_reopen_declaration_gate_metrics.json"
)
NONLINEAR_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1715_1718_updpk_full_nl_reopen_declaration_gate_metrics.json"
)
INVERSE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1719_1722_inv_local_constraint_declaration_gate_metrics.json"
)
INPUT_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1723_1726_ext_input_gate_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1727-1730"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor pack-update closeout / "
    "reopen registry refresh"
)
STEM = build_compact_artifact_stem(STEP_TAG, "updpk_closeout_registry", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_external_input_response_strategy_assimilation_"
    "no_new_primary_surface_pack_update_closeout_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_updated_pack_exhausted_family_closeout_"
    "reopen_registry_refresh_completed"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_new_action_"
    "level_structure_or_exact_probe_response_pack_update_reactivation"
)
NEXT_ROUTE = "8.7.56.1731"
FOLLOWUP_ROUTE_NAME = "none"

PRIMARY_REOPEN = (
    "genuinely_new_action_level_structure_or_new_exact_probe_response_map_"
    "beyond_current_source_extended_pack"
)
SECONDARY_REOPEN = (
    "substantive_pack_update_that_changes_internal_hamiltonian_or_exact_"
    "constitutive_surface_beyond_current_updated_pack"
)
RESERVE_REOPEN = (
    "future_external_input_guiding_new_primary_surface_after_updated_pack_"
    "closeout"
)

SCALAR_ALPHA = 0.00715678583937324
ENERGY_CORE_ALPHA = 0.0005422361373947313
PROJECTED_KERNEL_ALPHA = 0.0005600186431488893


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を確認する。

def require(path: Path) -> None:
    """Abort when one required input file is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8 テキストを読み込む。

def read_text(path: Path) -> str:
    """Read one UTF-8 text file."""
    return path.read_text(encoding="utf-8")


# 関数: UTF-8 JSON を読み込む。

def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: repo 相対の表示パスを返す。

def display_path(path: Path) -> str:
    """Return one repo-relative path when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# 関数: 部分一致する最初の行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line matching one substring."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 標準 metrics row を作る。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# 関数: 標準 payload を作る。

def payload(
    step: str,
    name: str,
    inputs: dict,
    rows: list[dict],
    summary: dict,
    decision: dict,
    evidence: dict,
) -> dict:
    """Build one standard payload."""
    return {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": step, "name": name},
        "inputs": inputs,
        "rows": rows,
        "summary": summary,
        "decision": decision,
        "evidence": evidence,
    }


# 関数: JSON/CSV 成果物を書き出す。

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

    return {"json": display_path(paths["json"]), "csv": display_path(paths["csv"])}


# 関数: 真偽値を 0/1 に変換する。

def truth(value: bool) -> float:
    """Convert one boolean into 0/1 float form."""
    return 1.0 if value else 0.0


# 関数: `.1727-.1730` を実行する。

def main() -> None:
    """Execute the pack-update closeout / reopen registry refresh branch."""
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
        PACK_UPDATE_GATE,
        DECISION_GATE,
        CONSTITUTIVE_GATE,
        NONLINEAR_GATE,
        INVERSE_GATE,
        INPUT_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    long_roadmap_text = read_text(LONG_ROADMAP)
    part5_text = read_text(PART5)

    pack_update_summary = read_json(PACK_UPDATE_GATE)["summary"]
    decision_summary = read_json(DECISION_GATE)["summary"]
    constitutive_summary = read_json(CONSTITUTIVE_GATE)["summary"]
    nonlinear_summary = read_json(NONLINEAR_GATE)["summary"]
    inverse_summary = read_json(INVERSE_GATE)["summary"]
    input_summary = read_json(INPUT_GATE)["summary"]

    inventory_ready = all(
        item is not None
        for item in (
            hit(status_text, "pack-update closeout / reopen registry refresh"),
            hit(roadmap_text, "8.7.56.1727-.1730"),
            hit(current_problem_text, "pack-update closeout / reopen registry refresh"),
            hit(current_status_text, "pack-update closeout / reopen registry refresh"),
            hit(
                unified_text,
                "`.1727-.1730` は **pack-update closeout / reopen registry refresh**",
            ),
            hit(long_roadmap_text, "11. `8.7.56.1727-.1730`"),
            hit(part5_text, "next official branch は"),
        )
    )

    source_extended_pack_adopted = bool(
        pack_update_summary.get("source_extended_probe_response_pack_adopted", False)
        and pack_update_summary.get("new_primary_trigger_opened", False)
    )
    canonical_no_go_closed = bool(
        decision_summary.get("canonical_probe_response_theorem_derived", False)
        and decision_summary.get("updated_canonical_observable_exact_available", False)
        and decision_summary.get("gate_b_retain_canonical_no_go_selected", False)
        and decision_summary.get("updated_pack_canonical_promotion_failed", False)
    )
    constitutive_reopen_failed = bool(
        not constitutive_summary.get(
            "exact_constitutive_map_available_under_updated_pack", True
        )
        and not constitutive_summary.get(
            "updated_pack_adds_internal_branch_to_probe_map", True
        )
    )
    nonlinear_reopen_failed = bool(
        nonlinear_summary.get("updated_pack_nonlinear_reopen_failed", False)
        and not nonlinear_summary.get(
            "branch_local_full_nonlinear_energy_density_exact_available_under_updated_pack",
            True,
        )
    )
    inverse_local_family_failed = bool(
        inverse_summary.get(
            "local_family_rescue_requires_large_or_noncanonical_coefficients", False
        )
    )
    latest_input_opened_no_new_surface = bool(
        input_summary.get("new_external_input_detected", False)
        and input_summary.get("input_is_ordering_or_historical_diagnostic_only", False)
        and not input_summary.get("new_primary_surface_opened", True)
    )
    updated_pack_exhausted_family_fixed = bool(
        source_extended_pack_adopted
        and canonical_no_go_closed
        and constitutive_reopen_failed
        and nonlinear_reopen_failed
        and inverse_local_family_failed
        and latest_input_opened_no_new_surface
    )
    same_level_updated_pack_retry_admissible = False
    physical_reject_not_selected = bool(
        not decision_summary.get("physical_reject_required", True)
        and not constitutive_summary.get("physical_reject_required", True)
        and not nonlinear_summary.get("physical_reject_required", True)
        and not inverse_summary.get("physical_reject_required", True)
        and not input_summary.get("physical_reject_required", True)
    )
    registry_wording_honest = bool(
        inventory_ready
        and updated_pack_exhausted_family_fixed
        and not same_level_updated_pack_retry_admissible
        and physical_reject_not_selected
    )
    registry_ready = bool(registry_wording_honest)

    rows = [
        row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "updated-pack closeout inventory ready",
            truth(inventory_ready),
            "Closeout starts only after status, roadmap, current notes, unified roadmap, long roadmap, and Part V all point to the same `.1727-.1730` branch.",
        ),
        row(
            "source_extended_pack_adopted",
            "pass" if source_extended_pack_adopted else "reject",
            "source-extended pack adopted",
            truth(source_extended_pack_adopted),
            "The updated-pack closeout is meaningful only if the new source-extended action-level primitive was genuinely adopted as a prior pack update.",
        ),
        row(
            "canonical_no_go_closed",
            "pass" if canonical_no_go_closed else "reject",
            "canonical two-leg no-go closed",
            truth(canonical_no_go_closed),
            "The updated pack must already have a canonical external probe-response theorem and still fail scalar promotion before closeout is honest.",
        ),
        row(
            "constitutive_reopen_failed",
            "pass" if constitutive_reopen_failed else "reject",
            "updated-pack constitutive reopen failed",
            truth(constitutive_reopen_failed),
            "The updated pack still adds no internal branch-to-observable bridge, so exact constitutive reopening remains unavailable.",
        ),
        row(
            "nonlinear_reopen_failed",
            "pass" if nonlinear_reopen_failed else "reject",
            "updated-pack nonlinear reopen failed",
            truth(nonlinear_reopen_failed),
            "The branch-local nonlinear-energy family carries over unchanged under the updated pack and still tracks the vector no-go scale.",
        ),
        row(
            "inverse_local_family_failed",
            "pass" if inverse_local_family_failed else "reject",
            "inverse local-family rescue failed",
            truth(inverse_local_family_failed),
            "Local same-branch rescue now requires huge or noncanonical coefficients, so that family closes honestly under the updated pack.",
        ),
        row(
            "latest_input_opened_no_new_surface",
            "pass" if latest_input_opened_no_new_surface else "reject",
            "latest external input opened no new surface",
            truth(latest_input_opened_no_new_surface),
            "The latest response-strategy note is genuine as a file but remains ordering-only because it mainly restates already executed response-theory routes.",
        ),
        row(
            "updated_pack_exhausted_family_fixed",
            "pass" if updated_pack_exhausted_family_fixed else "reject",
            "updated-pack exhausted family fixed",
            truth(updated_pack_exhausted_family_fixed),
            "Canonical response, constitutive reopen, nonlinear reopen, inverse local audit, and later external-input gate now close into one exhausted updated-pack family.",
        ),
        row(
            "same_level_updated_pack_retry_admissible",
            "reject",
            "same-level updated-pack retry admissible",
            truth(same_level_updated_pack_retry_admissible),
            "Once the updated-pack family is exhausted, adding another same-level surrogate rescue is no longer honest.",
        ),
        row(
            "primary_reopen_surface_fixed",
            "pass",
            "primary reopen surface fixed",
            1.0,
            "The primary reopen surface is a genuinely new action-level structure or a new exact probe-response map beyond the current source-extended pack.",
        ),
        row(
            "secondary_reopen_surface_fixed",
            "pass",
            "secondary reopen surface fixed",
            1.0,
            "The secondary reopen surface is a substantive pack update that changes the internal Hamiltonian / constitutive sector rather than merely external-source bookkeeping.",
        ),
        row(
            "reserve_reopen_surface_fixed",
            "pass",
            "reserve reopen surface fixed",
            1.0,
            "Future expert input is retained only as a reserve side lane that may identify the new primary surface after closeout.",
        ),
        row(
            "physical_reject_not_selected",
            "pass" if physical_reject_not_selected else "reject",
            "physical reject not selected",
            truth(physical_reject_not_selected),
            "This closeout remains route-local to the current updated pack and does not force physical rejection of the retained scalar strong candidate.",
        ),
        row(
            "registry_wording_honest",
            "pass" if registry_wording_honest else "reject",
            "updated-pack closeout registry wording honest",
            truth(registry_wording_honest),
            "The registry is honest only if the full updated-pack fail stack remains visible and same-level retry stays blocked.",
        ),
        row(
            "registry_ready",
            "pass" if registry_ready else "reject",
            "updated-pack closeout registry ready",
            truth(registry_ready),
            "Once the updated-pack family is exhausted and the reopen ordering is explicit, the pack-update closeout can be frozen machine-readably.",
        ),
    ]

    inputs = {
        "source_files": {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "ai_context": display_path(AI_CONTEXT),
            "work_history_recent": display_path(WORK_HISTORY_RECENT),
            "current_problem": display_path(CURRENT_PROBLEM),
            "current_status": display_path(CURRENT_STATUS),
            "unified_roadmap": display_path(UNIFIED_ROADMAP),
            "long_roadmap": display_path(LONG_ROADMAP),
            "part5": display_path(PART5),
            "pack_update_gate": display_path(PACK_UPDATE_GATE),
            "decision_gate": display_path(DECISION_GATE),
            "constitutive_gate": display_path(CONSTITUTIVE_GATE),
            "nonlinear_gate": display_path(NONLINEAR_GATE),
            "inverse_gate": display_path(INVERSE_GATE),
            "input_gate": display_path(INPUT_GATE),
        },
        "constants": {
            "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
            "official_energy_core_alpha_at_q_theory": ENERGY_CORE_ALPHA,
            "official_projected_kernel_alpha_at_q_theory": PROJECTED_KERNEL_ALPHA,
            "updated_canonical_alpha_at_q_theory": decision_summary[
                "updated_canonical_alpha_at_q_theory"
            ],
            "pilot_full_nonlinear_alpha_at_q_theory": nonlinear_summary[
                "updated_pack_pilot_full_nonlinear_alpha_at_q_theory"
            ],
            "family_proxy_full_nonlinear_alpha_at_q_theory": nonlinear_summary[
                "updated_pack_family_proxy_full_nonlinear_alpha_at_q_theory"
            ],
            "base_local_scalar_proxy_alpha_at_q_theory": inverse_summary[
                "base_same_branch_scalar_proxy_alpha_at_q_theory"
            ],
            "primary_reopen_surface": PRIMARY_REOPEN,
            "secondary_reopen_surface": SECONDARY_REOPEN,
            "reserve_reopen_surface": RESERVE_REOPEN,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "followup_route_name": FOLLOWUP_ROUTE_NAME,
            "followup_route": None,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "source_extended_probe_response_pack_adopted": source_extended_pack_adopted,
        "canonical_probe_response_theorem_derived": decision_summary[
            "canonical_probe_response_theorem_derived"
        ],
        "updated_canonical_observable_exact_available": decision_summary[
            "updated_canonical_observable_exact_available"
        ],
        "updated_pack_canonical_promotion_failed": decision_summary[
            "updated_pack_canonical_promotion_failed"
        ],
        "exact_constitutive_map_available_under_updated_pack": constitutive_summary[
            "exact_constitutive_map_available_under_updated_pack"
        ],
        "branch_local_full_nonlinear_energy_density_exact_available_under_updated_pack": nonlinear_summary[
            "branch_local_full_nonlinear_energy_density_exact_available_under_updated_pack"
        ],
        "local_family_rescue_requires_large_or_noncanonical_coefficients": inverse_summary[
            "local_family_rescue_requires_large_or_noncanonical_coefficients"
        ],
        "new_primary_surface_opened_by_latest_input": input_summary[
            "new_primary_surface_opened"
        ],
        "updated_pack_exhausted_family_fixed": updated_pack_exhausted_family_fixed,
        "same_level_updated_pack_retry_admissible": same_level_updated_pack_retry_admissible,
        "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
        "official_energy_core_alpha_at_q_theory": ENERGY_CORE_ALPHA,
        "official_projected_kernel_alpha_at_q_theory": PROJECTED_KERNEL_ALPHA,
        "updated_canonical_alpha_at_q_theory": decision_summary[
            "updated_canonical_alpha_at_q_theory"
        ],
        "updated_pack_pilot_full_nonlinear_alpha_at_q_theory": nonlinear_summary[
            "updated_pack_pilot_full_nonlinear_alpha_at_q_theory"
        ],
        "updated_pack_family_proxy_full_nonlinear_alpha_at_q_theory": nonlinear_summary[
            "updated_pack_family_proxy_full_nonlinear_alpha_at_q_theory"
        ],
        "base_same_branch_scalar_proxy_alpha_at_q_theory": inverse_summary[
            "base_same_branch_scalar_proxy_alpha_at_q_theory"
        ],
        "one_parameter_fLsq_coeff_for_target_alpha": inverse_summary[
            "one_parameter_fLsq_coeff_for_target_alpha"
        ],
        "one_parameter_fLsq_coeff_for_scalar_candidate": inverse_summary[
            "one_parameter_fLsq_coeff_for_scalar_candidate"
        ],
        "primary_reopen_surface": PRIMARY_REOPEN,
        "secondary_reopen_surface": SECONDARY_REOPEN,
        "reserve_reopen_surface": RESERVE_REOPEN,
        "updated_pack_closeout_reopen_registry_wording_honest": registry_wording_honest,
        "updated_pack_closeout_reopen_registry_ready": registry_ready,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": None,
        "selected_followup_route_or_none": None,
        "physical_reject_required": False,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": registry_ready,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "hits": {
            "status_branch_hit": hit(
                status_text, "pack-update closeout / reopen registry refresh"
            ),
            "roadmap_branch_hit": hit(roadmap_text, "8.7.56.1727-.1730"),
            "current_problem_branch_hit": hit(
                current_problem_text, "pack-update closeout / reopen registry refresh"
            ),
            "current_status_branch_hit": hit(
                current_status_text, "pack-update closeout / reopen registry refresh"
            ),
            "unified_roadmap_branch_hit": hit(
                unified_text,
                "`.1727-.1730` は **pack-update closeout / reopen registry refresh**",
            ),
            "long_roadmap_branch_hit": hit(
                long_roadmap_text, "11. `8.7.56.1727-.1730`"
            ),
            "part5_branch_hit": hit(part5_text, "next official branch は"),
        },
        "carry_over": {
            "pack_update_summary": pack_update_summary,
            "decision_summary": decision_summary,
            "constitutive_summary": constitutive_summary,
            "nonlinear_summary": nonlinear_summary,
            "inverse_summary": inverse_summary,
            "input_summary": input_summary,
        },
        "retained_numeric_state": {
            "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
            "official_energy_core_alpha_at_q_theory": ENERGY_CORE_ALPHA,
            "official_projected_kernel_alpha_at_q_theory": PROJECTED_KERNEL_ALPHA,
            "updated_canonical_alpha_at_q_theory": decision_summary[
                "updated_canonical_alpha_at_q_theory"
            ],
            "updated_pack_pilot_full_nonlinear_alpha_at_q_theory": nonlinear_summary[
                "updated_pack_pilot_full_nonlinear_alpha_at_q_theory"
            ],
            "updated_pack_family_proxy_full_nonlinear_alpha_at_q_theory": nonlinear_summary[
                "updated_pack_family_proxy_full_nonlinear_alpha_at_q_theory"
            ],
            "base_same_branch_scalar_proxy_alpha_at_q_theory": inverse_summary[
                "base_same_branch_scalar_proxy_alpha_at_q_theory"
            ],
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
        },
    }

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload(
                "8.7.56.1727",
                f"{STEP_NAME} inventory",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
        "audit": write_artifact(
            "audit",
            payload(
                "8.7.56.1728",
                f"{STEP_NAME} audit",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
        "declaration_gate": write_artifact(
            "declaration_gate",
            payload(
                "8.7.56.1729",
                f"{STEP_NAME} declaration gate",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
        "route_sync": write_artifact(
            "route_sync",
            payload(
                "8.7.56.1730",
                f"{STEP_NAME} route sync",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
    }

    print(
        json.dumps(
            {"step": STEP_TAG, "stem": STEM, "manifest": manifest, "summary": summary},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
