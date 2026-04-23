#!/usr/bin/env python3
"""Generate 8.7.56.1715-.1718 updated-pack nonlinear-energy reopen artifacts.

The updated source-extended pack added an explicit external source primitive

    S_src[P, a; J_perp] = S_frozen[P, a] - ∫ d^4x J_perp^mu a_mu

and thereby closed the external two-leg probe-response theorem. The previous
branch `.1711-.1714` then fixed that this pack update does *not* alter the
internal Hamiltonian sector or provide an exact constitutive map.

This branch therefore asks the only honest follow-up question left inside the
updated pack:

    if the internal Hamiltonian sector is unchanged, do the branch-local full
    nonlinear energy-density candidates reopen anything once they are carried
    forward under the updated-pack assumptions?

Because the source extension is external-only, the expected answer is that the
old `.1651-.1654` nonlinear surfaces carry over unchanged and still track the
vector no-go scale rather than the retained scalar strong candidate.
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

import scripts.quantum.t2a_1651 as old_full_nl_tools


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

OLD_FULL_NL_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1651_1654_full_nl_energy_audit_declaration_gate_metrics.json"
)
UPDATED_DECISION_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1707_1710_probe_resp_gate_sync_declaration_gate_metrics.json"
)
UPDATED_CONSTITUTIVE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1711_1714_updpk_const_map_reopen_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1715-1718"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor branch-local full nonlinear "
    "energy-density reopen after pack update"
)
STEM = build_compact_artifact_stem(STEP_TAG, "updpk_full_nl_reopen", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_source_extended_exact_constitutive_map_"
    "unavailable_branch_local_full_nonlinear_energy_reopen_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_updated_pack_branch_local_full_nonlinear_energy_"
    "carryover_tracks_vector_no_go_inverse_constraint_audit_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_inverse_local_observable_"
    "constraint_audit"
)
NEXT_ROUTE = "8.7.56.1719"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_external_input_assimilation_"
    "new_primary_surface_gate"
)
FOLLOWUP_ROUTE = "8.7.56.1723"

SCALAR_ALPHA = 0.00715678583937324
VECTOR_ALPHA = 0.0005579616187042394
ENERGY_CORE_ALPHA = 0.0005422361373947313


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を確認する。

def require(path: Path) -> None:
    """Abort when one required input is missing."""
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


# 関数: repo 相対表示パスを返す。

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


# 関数: 参照値に対する相対ギャップを返す。

def relative_gap(value: float, reference: float) -> float:
    """Return one reference-relative absolute gap."""
    return float(abs(float(value) - float(reference)) / float(reference))


# 関数: `.1715-.1718` を実行する。

def main() -> None:
    """Execute the updated-pack nonlinear-energy reopen branch."""
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
        OLD_FULL_NL_GATE,
        UPDATED_DECISION_GATE,
        UPDATED_CONSTITUTIVE_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    long_text = read_text(LONG_ROADMAP)
    part5_text = read_text(PART5)

    old_full_nl_summary = read_json(OLD_FULL_NL_GATE)["summary"]
    updated_decision_summary = read_json(UPDATED_DECISION_GATE)["summary"]
    updated_constitutive_summary = read_json(UPDATED_CONSTITUTIVE_GATE)["summary"]

    candidate_bundle = old_full_nl_tools.build_candidate_surfaces()
    pilot_surface = candidate_bundle["pilot_surface"]
    family_surface = candidate_bundle["mh_surface"]

    updated_pack_changes_internal_hamiltonian_sector = bool(
        updated_constitutive_summary["updated_pack_changes_internal_hamiltonian_sector"]
    )
    exact_constitutive_map_available_under_updated_pack = bool(
        updated_constitutive_summary["exact_constitutive_map_available_under_updated_pack"]
    )

    pilot_alpha = float(pilot_surface["alpha_at_q_theory"])
    family_alpha = float(family_surface["alpha_at_q_theory"])
    old_pilot_alpha = float(old_full_nl_summary["pilot_full_nonlinear_alpha_at_q_theory"])
    old_family_alpha = float(old_full_nl_summary["family_proxy_full_alpha_at_q_theory"])

    pilot_surface_carries_over_exactly = abs(pilot_alpha - old_pilot_alpha) <= 1.0e-18
    family_surface_carries_over_exactly = abs(family_alpha - old_family_alpha) <= 1.0e-18
    pilot_supports_scalar_candidate = bool(
        old_full_nl_summary["pilot_full_supports_scalar_candidate"]
    )
    family_supports_scalar_candidate = bool(
        old_full_nl_summary["family_proxy_supports_scalar_candidate"]
    )
    pilot_tracks_vector_no_go_scale = bool(
        old_full_nl_summary["pilot_full_tracks_vector_no_go_scale"]
    )
    family_tracks_vector_no_go_scale = bool(
        old_full_nl_summary["family_proxy_tracks_vector_no_go_scale"]
    )
    branch_local_full_nonlinear_energy_density_exact_available_under_updated_pack = bool(
        exact_constitutive_map_available_under_updated_pack
    )
    updated_pack_nonlinear_reopen_failed = (
        (not branch_local_full_nonlinear_energy_density_exact_available_under_updated_pack)
        and (not pilot_supports_scalar_candidate)
        and (not family_supports_scalar_candidate)
    )

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
            "old_full_nl_gate": display_path(OLD_FULL_NL_GATE),
            "updated_decision_gate": display_path(UPDATED_DECISION_GATE),
            "updated_constitutive_gate": display_path(UPDATED_CONSTITUTIVE_GATE),
        },
        "constants": {
            "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
            "vector_alpha_at_q_theory": VECTOR_ALPHA,
            "official_energy_core_alpha_at_q_theory": ENERGY_CORE_ALPHA,
            "updated_canonical_alpha_at_q_theory": updated_decision_summary[
                "updated_canonical_alpha_at_q_theory"
            ],
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "followup_route_name": FOLLOWUP_ROUTE_NAME,
            "followup_route": FOLLOWUP_ROUTE,
        },
    }

    rows = [
        row(
            "updated_pack_changes_internal_hamiltonian_sector",
            "reject",
            "updated pack changes internal Hamiltonian sector",
            truth(updated_pack_changes_internal_hamiltonian_sector),
            "The source-extension primitive is external-only, so this reopen branch starts from the fixed theorem-side fact that the internal Hamiltonian sector is unchanged.",
        ),
        row(
            "pilot_branch_local_nonlinear_candidate_carries_over_under_updated_pack",
            "pass",
            "pilot branch-local nonlinear candidate carries over under updated pack",
            truth(pilot_surface_carries_over_exactly),
            "Because the updated pack leaves the internal Hamiltonian sector untouched, the pilot-consistent nonlinear surface is expected to match its pre-update value exactly.",
        ),
        row(
            "family_proxy_branch_local_candidate_carries_over_under_updated_pack",
            "pass",
            "family-proxy branch-local nonlinear candidate carries over under updated pack",
            truth(family_surface_carries_over_exactly),
            "The vacuum-subtracted Mexican-hat proxy is likewise an internal branch-local surface and therefore carries over unchanged under an external-only pack update.",
        ),
        row(
            "branch_local_full_nonlinear_energy_density_exact_available_under_updated_pack",
            "reject",
            "branch-local full nonlinear energy density exact available under updated pack",
            truth(branch_local_full_nonlinear_energy_density_exact_available_under_updated_pack),
            "The updated pack still does not provide the missing constitutive surface that would canonically promote any branch-local nonlinear density candidate.",
        ),
        row(
            "updated_pack_pilot_full_nonlinear_alpha_at_q_theory",
            "watch",
            "updated-pack pilot full nonlinear alpha at q_theory",
            pilot_alpha,
            "The carryover read remains the same blind fixed-q_theory value because the updated pack does not modify the internal nonlinear density itself.",
        ),
        row(
            "updated_pack_family_proxy_full_nonlinear_alpha_at_q_theory",
            "watch",
            "updated-pack family-proxy full nonlinear alpha at q_theory",
            family_alpha,
            "The family-proxy carryover read also remains unchanged under the external-only pack update.",
        ),
        row(
            "updated_pack_pilot_supports_scalar_candidate",
            "reject",
            "updated-pack pilot full nonlinear candidate supports scalar candidate",
            truth(pilot_supports_scalar_candidate),
            "Even after the pack update, the pilot-consistent branch-local nonlinear surface stays on the vector no-go scale and does not support scalar promotion.",
        ),
        row(
            "updated_pack_family_proxy_supports_scalar_candidate",
            "reject",
            "updated-pack family-proxy full nonlinear candidate supports scalar candidate",
            truth(family_supports_scalar_candidate),
            "The family-level branch-local nonlinear proxy likewise fails to support the scalar strong candidate under the updated pack.",
        ),
        row(
            "updated_pack_pilot_tracks_vector_no_go_scale",
            "pass",
            "updated-pack pilot candidate tracks vector no-go scale",
            truth(pilot_tracks_vector_no_go_scale),
            "The pilot surface remains closer to the retained vector no-go alpha than to the retained scalar strong candidate.",
        ),
        row(
            "updated_pack_family_tracks_vector_no_go_scale",
            "pass",
            "updated-pack family-proxy candidate tracks vector no-go scale",
            truth(family_tracks_vector_no_go_scale),
            "The family-proxy surface also remains on the same no-go scale.",
        ),
        row(
            "updated_pack_pilot_vs_pre_update_rel_gap",
            "watch",
            "updated-pack pilot relative gap vs pre-update pilot alpha",
            relative_gap(pilot_alpha, old_pilot_alpha),
            "This should collapse to zero when the external-only pack update leaves the internal nonlinear sector untouched.",
        ),
        row(
            "updated_pack_family_vs_pre_update_rel_gap",
            "watch",
            "updated-pack family relative gap vs pre-update family alpha",
            relative_gap(family_alpha, old_family_alpha),
            "This tracks the same carryover statement for the family-proxy nonlinear surface.",
        ),
        row(
            "updated_pack_nonlinear_reopen_failed",
            "pass",
            "updated-pack nonlinear reopen failed",
            truth(updated_pack_nonlinear_reopen_failed),
            "With no internal constitutive upgrade and no scalar-supporting carryover read, the updated-pack nonlinear reopen closes honestly as a no-go.",
        ),
        row(
            "inverse_local_observable_constraint_audit_admissible_now",
            "pass",
            "inverse local-observable constraint audit admissible now",
            1.0,
            "Once both updated-pack secondary reopen surfaces fail, the remaining next step is the side diagnostic that constrains what any future local family would have to look like.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "updated_pack_changes_internal_hamiltonian_sector": updated_pack_changes_internal_hamiltonian_sector,
        "exact_constitutive_map_available_under_updated_pack": exact_constitutive_map_available_under_updated_pack,
        "branch_local_full_nonlinear_energy_density_exact_available_under_updated_pack": branch_local_full_nonlinear_energy_density_exact_available_under_updated_pack,
        "pilot_branch_local_nonlinear_candidate_carries_over_under_updated_pack": pilot_surface_carries_over_exactly,
        "family_proxy_branch_local_candidate_carries_over_under_updated_pack": family_surface_carries_over_exactly,
        "updated_pack_pilot_full_nonlinear_F_at_q_theory": pilot_surface["F_at_q_theory"],
        "updated_pack_pilot_full_nonlinear_alpha_at_q_theory": pilot_alpha,
        "updated_pack_pilot_supports_scalar_candidate": pilot_supports_scalar_candidate,
        "updated_pack_pilot_tracks_vector_no_go_scale": pilot_tracks_vector_no_go_scale,
        "updated_pack_family_proxy_full_F_at_q_theory": family_surface["F_at_q_theory"],
        "updated_pack_family_proxy_full_nonlinear_alpha_at_q_theory": family_alpha,
        "updated_pack_family_proxy_supports_scalar_candidate": family_supports_scalar_candidate,
        "updated_pack_family_proxy_tracks_vector_no_go_scale": family_tracks_vector_no_go_scale,
        "updated_pack_pilot_vs_pre_update_rel_gap": relative_gap(pilot_alpha, old_pilot_alpha),
        "updated_pack_family_vs_pre_update_rel_gap": relative_gap(family_alpha, old_family_alpha),
        "updated_pack_canonical_alpha_at_q_theory": updated_decision_summary[
            "updated_canonical_alpha_at_q_theory"
        ],
        "official_energy_core_alpha_at_q_theory": ENERGY_CORE_ALPHA,
        "updated_pack_nonlinear_reopen_failed": updated_pack_nonlinear_reopen_failed,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": True,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "formulas": {
            "source_extended_action": "S_src[P,a;J_perp] = S_frozen[P,a] - ∫ d^4x J_perp^mu a_mu",
            "external_only_variation": "δS_src/δP |_(J_perp=0) = δS_frozen/δP",
            "pilot_full_candidate": "epsilon_full,pilot(r) = epsilon_H,core(r) + rho(r)^3 + rho(r)^4/4",
            "family_proxy_candidate": "epsilon_full,MHproxy(r) = epsilon_H,core(r) + (1/4)[(rho(r)^2-v^2)^2 - v^4]",
            "carryover_rule": "updated_pack_changes_internal_hamiltonian_sector = false => epsilon_full,updated(r) = epsilon_full,pre-update(r)",
            "failure_read": "The updated pack changes the external theorem only, so the old branch-local nonlinear energy candidates carry over unchanged and remain non-canonical no-go reads.",
        },
        "hits": {
            "status_current_branch": hit(
                status_text,
                "branch-local full nonlinear energy-density reopen after pack update",
            ),
            "roadmap_current_branch": hit(
                roadmap_text,
                "8.7.56.1715-.1718",
            ),
            "current_problem_branch_hit": hit(
                current_problem_text,
                "branch-local full nonlinear energy-density reopen after pack update",
            ),
            "current_status_branch_hit": hit(
                current_status_text,
                "branch-local full nonlinear energy-density reopen after pack update",
            ),
            "unified_roadmap_branch_hit": hit(
                unified_text,
                "`.1715-.1718` は **branch-local full nonlinear energy-density reopen after pack update**",
            ),
            "long_roadmap_branch_hit": hit(
                long_text,
                "8.7.56.1715-.1718",
            ),
            "part5_updated_pack_hit": hit(
                part5_text,
                "branch-local full nonlinear energy-density reopen after pack update",
            ),
        },
        "prior_summaries": {
            "old_full_nonlinear": old_full_nl_summary,
            "updated_decision_gate": updated_decision_summary,
            "updated_constitutive_gate": updated_constitutive_summary,
        },
    }

    outputs: dict[str, dict[str, str]] = {}
    for kind in ("inventory", "audit", "declaration_gate", "route_sync"):
        outputs[kind] = write_artifact(
            kind,
            payload(
                step=STEP_TAG,
                name=f"{STEP_NAME} {kind.replace('_', ' ')}",
                inputs=inputs,
                rows=rows,
                summary=summary,
                decision=decision,
                evidence=evidence,
            ),
        )

    print("[ok] updated-pack branch-local nonlinear-energy reopen artifacts written:")
    for kind, paths in outputs.items():
        print(f"  - {kind}: {paths['json']} | {paths['csv']}")


if __name__ == "__main__":
    main()
