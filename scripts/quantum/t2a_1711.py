#!/usr/bin/env python3
"""Generate 8.7.56.1711-.1714 updated-pack constitutive-map reopen artifacts.

The source-extended pack closed the external probe-response / amputation
theorem:

    S_src[P, a; J_perp] = S_frozen[P, a] - ∫ d^4x J_perp^mu a_mu
    chi_T = δ² W_P / δJ_perp δJ_perp
    F_T,can(q) = -q^4 Delta chi_T(q)

That theorem answers which external response object is canonical. It does not
automatically answer a different question:

    does the updated pack now provide an exact constitutive map that turns the
    retained exact branch P_mu^Qball into a canonical observable density or
    microscopic current functional?

This branch therefore audits whether the new source primitive changed only the
external-source bookkeeping, or whether it also added the missing internal
branch-to-observable bridge. If the answer is still negative, the honest next
route is the updated-pack branch-local full nonlinear energy-density reopen.
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
THEOREM_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1699_1702_probe_resp_amp_theorem_declaration_gate_metrics.json"
)
RECOMPUTE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1703_1706_probe_resp_canon_recompute_declaration_gate_metrics.json"
)
DECISION_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1707_1710_probe_resp_gate_sync_declaration_gate_metrics.json"
)
MICRO_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1543_1546_micro_source_fn_deriv_declaration_gate_metrics.json"
)
ENERGY_DERIV_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1623_1626_energy_density_audit_declaration_gate_metrics.json"
)
ENERGY_FF_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1627_1630_energy_density_ff_audit_declaration_gate_metrics.json"
)
OLD_CONSTITUTIVE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1647_1650_constitutive_map_audit_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1711-1714"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor exact constitutive-map "
    "reopen after pack update"
)
STEM = build_compact_artifact_stem(STEP_TAG, "updpk_const_map_reopen", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_source_extended_canonical_two_leg_no_scalar_"
    "promotion_exact_constitutive_map_reopen_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_source_extended_exact_constitutive_map_"
    "unavailable_branch_local_full_nonlinear_energy_reopen_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_branch_local_full_nonlinear_"
    "energy_density_reopen_after_pack_update"
)
NEXT_ROUTE = "8.7.56.1715"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_inverse_local_observable_"
    "constraint_audit"
)
FOLLOWUP_ROUTE = "8.7.56.1719"


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


# 関数: `.1711-.1714` を実行する。

def main() -> None:
    """Execute the updated-pack exact constitutive-map reopen branch."""
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
        THEOREM_GATE,
        RECOMPUTE_GATE,
        DECISION_GATE,
        MICRO_GATE,
        ENERGY_DERIV_GATE,
        ENERGY_FF_GATE,
        OLD_CONSTITUTIVE_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    long_text = read_text(LONG_ROADMAP)
    part5_text = read_text(PART5)

    pack_update_gate = read_json(PACK_UPDATE_GATE)
    theorem_gate = read_json(THEOREM_GATE)
    recompute_gate = read_json(RECOMPUTE_GATE)
    decision_gate = read_json(DECISION_GATE)
    micro_gate = read_json(MICRO_GATE)
    energy_deriv_gate = read_json(ENERGY_DERIV_GATE)
    energy_ff_gate = read_json(ENERGY_FF_GATE)
    old_constitutive_gate = read_json(OLD_CONSTITUTIVE_GATE)

    pack_summary = pack_update_gate["summary"]
    theorem_summary = theorem_gate["summary"]
    recompute_summary = recompute_gate["summary"]
    decision_summary = decision_gate["summary"]
    micro_summary = micro_gate["summary"]
    energy_deriv_summary = energy_deriv_gate["summary"]
    energy_ff_summary = energy_ff_gate["summary"]
    old_constitutive_summary = old_constitutive_gate["summary"]

    source_extended_probe_response_pack_adopted = bool(
        pack_summary["source_extended_probe_response_pack_adopted"]
    )
    canonical_probe_response_theorem_derived = bool(
        theorem_summary["canonical_probe_response_theorem_derived"]
    )
    updated_canonical_observable_exact_available = bool(
        decision_summary["updated_canonical_observable_exact_available"]
    )
    microscopic_chiral_current_constitutive_map_available = bool(
        micro_summary["microscopic_chiral_current_constitutive_map_available"]
    )
    microscopic_pauli_tensor_constitutive_map_available = bool(
        micro_summary["microscopic_pauli_tensor_constitutive_map_available"]
    )
    exact_hamiltonian_core_density_available = bool(
        energy_deriv_summary["exact_hamiltonian_core_density_available"]
    )
    branch_local_full_energy_density_available = bool(
        energy_ff_summary["branch_local_full_energy_density_available"]
    )
    prior_exact_constitutive_map_available = bool(
        old_constitutive_summary["exact_constitutive_map_available"]
    )

    source_extension_changes_external_response_theorem_only = (
        source_extended_probe_response_pack_adopted
        and canonical_probe_response_theorem_derived
        and updated_canonical_observable_exact_available
    )
    updated_pack_adds_internal_branch_to_probe_map = False
    updated_pack_adds_exact_branch_local_energy_map = False
    updated_pack_changes_internal_hamiltonian_sector = False
    canonical_two_leg_rule_is_not_internal_constitutive_map = True
    exact_constitutive_map_available_under_updated_pack = (
        source_extension_changes_external_response_theorem_only
        and updated_pack_adds_internal_branch_to_probe_map
        and microscopic_chiral_current_constitutive_map_available
        and microscopic_pauli_tensor_constitutive_map_available
        and exact_hamiltonian_core_density_available
        and branch_local_full_energy_density_available
        and prior_exact_constitutive_map_available
    )
    branch_local_full_nonlinear_energy_density_reopen_after_pack_update_admissible_now = (
        not exact_constitutive_map_available_under_updated_pack
    )
    inverse_local_observable_constraint_audit_retained = True

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
            "theorem_gate": display_path(THEOREM_GATE),
            "recompute_gate": display_path(RECOMPUTE_GATE),
            "decision_gate": display_path(DECISION_GATE),
            "micro_constitutive_gate": display_path(MICRO_GATE),
            "energy_derivation_gate": display_path(ENERGY_DERIV_GATE),
            "energy_ff_gate": display_path(ENERGY_FF_GATE),
            "old_constitutive_gate": display_path(OLD_CONSTITUTIVE_GATE),
        },
        "constants": {
            "updated_canonical_alpha_at_q_theory": decision_summary[
                "updated_canonical_alpha_at_q_theory"
            ],
            "official_energy_core_alpha_at_q_theory": energy_ff_summary[
                "official_alpha_E_at_q_theory"
            ],
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "followup_route_name": FOLLOWUP_ROUTE_NAME,
            "followup_route": FOLLOWUP_ROUTE,
        },
    }

    rows = [
        row(
            "source_extended_probe_response_pack_adopted",
            "pass",
            "source-extended probe-response pack adopted",
            truth(source_extended_probe_response_pack_adopted),
            "The updated-pack constitutive reopen only starts after the explicit external source primitive has been added to the action.",
        ),
        row(
            "canonical_probe_response_theorem_derived",
            "pass",
            "canonical probe-response theorem derived",
            truth(canonical_probe_response_theorem_derived),
            "The pack update already closes the external two-leg amputation rule and therefore fixes the canonical response observable at theorem level.",
        ),
        row(
            "updated_canonical_observable_exact_available",
            "pass",
            "updated canonical observable exact available",
            truth(updated_canonical_observable_exact_available),
            "The theorem-selected two-leg observable has already been recomputed directly on the retained exact branch.",
        ),
        row(
            "source_extension_changes_external_response_theorem_only",
            "pass",
            "source extension changes external response theorem only",
            truth(source_extension_changes_external_response_theorem_only),
            "The new source primitive changes how external probe legs are normalized and amputated, not the internal branch-local observable content by itself.",
        ),
        row(
            "canonical_two_leg_rule_is_not_internal_constitutive_map",
            "pass",
            "canonical two-leg rule is not internal constitutive map",
            truth(canonical_two_leg_rule_is_not_internal_constitutive_map),
            "F_T,can(q) = -q^4 Delta chi_T(q) closes the external response theorem, but it does not by itself provide a local or microscopic map O_can[P^Qball].",
        ),
        row(
            "updated_pack_adds_internal_branch_to_probe_map",
            "reject",
            "updated pack adds internal branch-to-probe constitutive map",
            truth(updated_pack_adds_internal_branch_to_probe_map),
            "No new branch-local rho_can[f_0,f_L,...] or equivalent exact observable dictionary is introduced by the source-extension primitive alone.",
        ),
        row(
            "updated_pack_changes_internal_hamiltonian_sector",
            "reject",
            "updated pack changes internal Hamiltonian sector",
            truth(updated_pack_changes_internal_hamiltonian_sector),
            "The source term adds external probe coupling, but it does not modify the already-audited internal Hamiltonian / nonlinear density sector.",
        ),
        row(
            "microscopic_chiral_current_constitutive_map_available",
            "reject",
            "microscopic chiral-current constitutive map available under updated pack",
            truth(microscopic_chiral_current_constitutive_map_available),
            "The older microscopic bilinear gap remains open because the source extension does not derive psi_bar gamma^mu (1-gamma^5) psi / 2 from the retained branch.",
        ),
        row(
            "microscopic_pauli_tensor_constitutive_map_available",
            "reject",
            "microscopic Pauli-tensor constitutive map available under updated pack",
            truth(microscopic_pauli_tensor_constitutive_map_available),
            "The updated pack also does not derive a Pauli-tensor constitutive surface for the retained branch.",
        ),
        row(
            "branch_local_full_energy_density_available",
            "reject",
            "exact branch-local full energy density available under updated pack",
            truth(branch_local_full_energy_density_available),
            "The branch-local full nonlinear energy surface is still not canonically fixed at the same level as the external response theorem.",
        ),
        row(
            "exact_constitutive_map_available_under_updated_pack",
            "reject",
            "exact constitutive map available under updated pack",
            truth(exact_constitutive_map_available_under_updated_pack),
            "Because the source extension closes only the external theorem while the internal branch-local and microscopic maps stay absent, the updated pack still does not supply an exact constitutive map.",
        ),
        row(
            "branch_local_full_nonlinear_energy_density_reopen_after_pack_update_admissible_now",
            "pass",
            "branch-local full nonlinear energy-density reopen after pack update admissible now",
            truth(
                branch_local_full_nonlinear_energy_density_reopen_after_pack_update_admissible_now
            ),
            "Once the updated-pack constitutive reopen fails honestly, the next secondary reopen surface is the branch-local full nonlinear energy-density audit under the same updated pack.",
        ),
        row(
            "inverse_local_observable_constraint_audit_retained",
            "pass",
            "inverse local-observable constraint audit retained as side diagnostic",
            truth(inverse_local_observable_constraint_audit_retained),
            "The inverse local-observable audit stays available only as a downstream diagnostic and does not replace the updated-pack nonlinear mainline.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "source_extended_probe_response_pack_adopted": source_extended_probe_response_pack_adopted,
        "canonical_probe_response_theorem_derived": canonical_probe_response_theorem_derived,
        "updated_canonical_observable_exact_available": updated_canonical_observable_exact_available,
        "source_extension_changes_external_response_theorem_only": source_extension_changes_external_response_theorem_only,
        "updated_pack_adds_internal_branch_to_probe_map": updated_pack_adds_internal_branch_to_probe_map,
        "updated_pack_changes_internal_hamiltonian_sector": updated_pack_changes_internal_hamiltonian_sector,
        "canonical_two_leg_rule_is_not_internal_constitutive_map": canonical_two_leg_rule_is_not_internal_constitutive_map,
        "microscopic_chiral_current_constitutive_map_available": microscopic_chiral_current_constitutive_map_available,
        "microscopic_pauli_tensor_constitutive_map_available": microscopic_pauli_tensor_constitutive_map_available,
        "exact_hamiltonian_core_density_available": exact_hamiltonian_core_density_available,
        "branch_local_full_energy_density_available": branch_local_full_energy_density_available,
        "exact_constitutive_map_available_under_updated_pack": exact_constitutive_map_available_under_updated_pack,
        "branch_local_full_nonlinear_energy_density_reopen_after_pack_update_admissible_now": branch_local_full_nonlinear_energy_density_reopen_after_pack_update_admissible_now,
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
            "response_functional": "W_P[J_perp] = S_src[P;J_perp] - S_src[0;J_perp]",
            "susceptibility_definition": "chi_T = δ² W_P / δJ_perp δJ_perp",
            "canonical_response_rule": "F_T,can(q) = -q^4 Delta chi_T(q)",
            "missing_internal_map": "M_T,can(q) ?= ∫ d^3x e^{i q·x} O_can[P^Qball](x)",
            "failure_read": "The updated pack fixes the external response theorem but still does not derive the branch-local or microscopic O_can[P^Qball] functional.",
        },
        "hits": {
            "status_current_branch": hit(
                status_text,
                "exact constitutive-map reopen after pack update",
            ),
            "roadmap_current_branch": hit(
                roadmap_text,
                "8.7.56.1711-.1714",
            ),
            "current_problem_gap_hit": hit(
                current_problem_text,
                "exact constitutive map / canonical observable bridge",
            ),
            "current_status_gap_hit": hit(
                current_status_text,
                "exact constitutive map / canonical observable bridge",
            ),
            "unified_roadmap_current_branch": hit(
                unified_text,
                "`.1711-.1714` は **exact constitutive-map reopen after pack update**",
            ),
            "long_roadmap_current_branch": hit(
                long_text,
                "8.7.56.1711-.1714",
            ),
            "part5_updated_pack_hit": hit(
                part5_text,
                "exact constitutive-map reopen after pack update",
            ),
        },
        "prior_summaries": {
            "pack_update": pack_summary,
            "theorem": theorem_summary,
            "recompute": recompute_summary,
            "decision_gate": decision_summary,
            "micro_constitutive": micro_summary,
            "energy_derivation": energy_deriv_summary,
            "energy_ff": energy_ff_summary,
            "old_constitutive": old_constitutive_summary,
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

    print("[ok] updated-pack exact constitutive-map reopen artifacts written:")
    for kind, paths in outputs.items():
        print(f"  - {kind}: {paths['json']} | {paths['csv']}")


if __name__ == "__main__":
    main()
