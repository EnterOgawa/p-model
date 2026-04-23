#!/usr/bin/env python3
"""Generate 8.7.56.1647-.1650 exact constitutive-map audit artifacts.

This branch asks one narrow question:

Can the current frozen-action pack canonically decide what observable a probe
reads from the restored exact vector branch without introducing new action-level
structure or post-hoc normalization?

The honest audit treats the already-derived Hamiltonian core as exact but does
not automatically promote it to a canonical observable. The branch succeeds only
if an exact constitutive map is already closed under the current pack.
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
LOCAL_RESPONSE = ROOT / "doc" / "quantum" / "50_trial2_vector_qball_breakthrough_instruction_response.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
EXTERNAL_NOTE = Path(
    r"C:\Users\ogawa\Downloads\50_trial2_numeric_alpha_vector_qball_breakthrough_instruction_pack.md"
)
ADVICE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1643_1646_ed_reopen_advice_pack_declaration_gate_metrics.json"
)
ADVICE_ROUTE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1643_1646_ed_reopen_advice_pack_route_sync_metrics.json"
)
REGISTRY_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1639_1642_energy_density_reopen_registry_declaration_gate_metrics.json"
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
MICRO_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1543_1546_micro_source_fn_deriv_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1647-1650"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor exact constitutive-map audit"
STEM = build_compact_artifact_stem(STEP_TAG, "constitutive_map_audit", prefix="q")

PRIOR_CLASS = "vector_qball_form_factor_energy_density_reopen_advice_pack_refresh_completed"
BRANCH_CLASS = (
    "vector_qball_form_factor_exact_constitutive_map_unavailable_"
    "branch_local_full_nonlinear_energy_audit_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_branch_local_full_nonlinear_"
    "energy_density_audit"
)
NEXT_ROUTE = "8.7.56.1651"
FOLLOWUP_ROUTE = "8.7.56.1655"
FALLBACK_ROUTE = "8.7.56.1659"
SECOND_FALLBACK_ROUTE = "8.7.56.1663"

PRIMARY_REOPEN = (
    "branch_local_full_nonlinear_energy_density_or_exact_constitutive_map_gap"
)
SECONDARY_REOPEN = (
    "evidence_only_electric_like_or_note_gradient_canonical_promotion_gap"
)
RESERVE_REOPEN = "future_external_input_or_new_action_level_structure"
SCALAR_ALPHA = 0.00715678583937324
TARGET_ALPHA = 0.0072973525692838015
VECTOR_ALPHA = 0.0005579616187042394


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を検証する。

def require(path: Path) -> None:
    """Fail when one required path is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8テキストを読み込む。

def read_text(path: Path) -> str:
    """Read one UTF-8 text file."""
    return path.read_text(encoding="utf-8")


# 関数: UTF-8 JSON を読み込む。

def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: 表示用の相対パスへ変換する。

def display_path(path: Path) -> str:
    """Convert one absolute path into repo-relative display form when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# 関数: 部分文字列に一致する最初の行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first matching line for one substring pattern."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 標準形式の metrics row を作る。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# 関数: 標準形式の payload を作る。

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


# 関数: compact stem で JSON / CSV を出力する。

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


# 関数: 真偽値を 0 / 1 に変換する。

def truth(value: bool) -> float:
    """Convert one boolean into 0/1 float form."""
    return 1.0 if value else 0.0


# 関数: `.1647-.1650` を実行する。

def main() -> None:
    """Execute the exact constitutive-map audit branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        LOCAL_RESPONSE,
        PART5,
        EXTERNAL_NOTE,
        ADVICE_GATE,
        ADVICE_ROUTE,
        REGISTRY_GATE,
        ENERGY_DERIV_GATE,
        ENERGY_FF_GATE,
        MICRO_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    local_text = read_text(LOCAL_RESPONSE)
    part5_text = read_text(PART5)
    external_text = read_text(EXTERNAL_NOTE)
    ai_context = read_json(AI_CONTEXT)
    advice_gate = read_json(ADVICE_GATE)
    advice_route = read_json(ADVICE_ROUTE)
    registry_gate = read_json(REGISTRY_GATE)
    energy_deriv_gate = read_json(ENERGY_DERIV_GATE)
    energy_ff_gate = read_json(ENERGY_FF_GATE)
    micro_gate = read_json(MICRO_GATE)

    exact_hamiltonian_core_density_available = energy_deriv_gate["summary"][
        "exact_hamiltonian_core_density_available"
    ]
    branch_local_full_energy_density_available = energy_ff_gate["summary"][
        "branch_local_full_energy_density_available"
    ]
    energy_core_exact_foundation_supported = energy_ff_gate["summary"][
        "energy_core_exact_foundation_supported"
    ]
    microscopic_chiral_current_constitutive_map_available = micro_gate["summary"][
        "microscopic_chiral_current_constitutive_map_available"
    ]
    microscopic_pauli_tensor_constitutive_map_available = micro_gate["summary"][
        "microscopic_pauli_tensor_constitutive_map_available"
    ]

    exact_constitutive_map_available = (
        exact_hamiltonian_core_density_available
        and branch_local_full_energy_density_available
        and energy_core_exact_foundation_supported
        and microscopic_chiral_current_constitutive_map_available
        and microscopic_pauli_tensor_constitutive_map_available
    )
    observable_from_current_pack_canonical = exact_constitutive_map_available
    raw_hamiltonian_core_observable_selected = (
        exact_hamiltonian_core_density_available
        and observable_from_current_pack_canonical
        and energy_core_exact_foundation_supported
    )
    branch_local_full_nonlinear_energy_density_followup_admissible_now = (
        not exact_constitutive_map_available
    )
    secondary_canonical_promotion_admissible_now = False
    fallback_not_required_now = branch_local_full_nonlinear_energy_density_followup_admissible_now
    parallel_jeff_lane_retained = "parallel `J_{\\rm eff}`" in local_text or "J_eff / exact current-closure theorem lane" in external_text

    inputs = {
        "status": display_path(STATUS),
        "roadmap": display_path(ROADMAP),
        "ai_context": display_path(AI_CONTEXT),
        "work_history_recent": display_path(WORK_HISTORY_RECENT),
        "current_problem": display_path(CURRENT_PROBLEM),
        "current_status": display_path(CURRENT_STATUS),
        "unified_roadmap": display_path(UNIFIED_ROADMAP),
        "local_response": display_path(LOCAL_RESPONSE),
        "part5": display_path(PART5),
        "external_instruction_pack": display_path(EXTERNAL_NOTE),
        "advice_gate": display_path(ADVICE_GATE),
        "advice_route": display_path(ADVICE_ROUTE),
        "registry_gate": display_path(REGISTRY_GATE),
        "energy_derivation_gate": display_path(ENERGY_DERIV_GATE),
        "energy_ff_gate": display_path(ENERGY_FF_GATE),
        "micro_constitutive_gate": display_path(MICRO_GATE),
    }

    inventory_rows = [
        row(
            "inventory_advice_pack_ready",
            "ok",
            "prior_advice_pack_ready",
            truth(
                advice_gate["decision"]["selected_branch_classification"] == PRIOR_CLASS
            ),
            "The constitutive-map audit starts only after the advice-pack refresh has frozen the breakthrough order.",
        ),
        row(
            "inventory_primary_surface",
            "ok",
            "primary_reopen_surface_retained",
            truth(registry_gate["summary"]["primary_reopen_surface"] == PRIMARY_REOPEN),
            "The primary reopen surface remains the constitutive-map or branch-local nonlinear energy-density gap.",
        ),
        row(
            "inventory_energy_core_exact",
            "ok",
            "exact_hamiltonian_core_density_available",
            truth(exact_hamiltonian_core_density_available),
            "The Hamiltonian core is exact under the current frozen action and remains the starting point of this audit.",
        ),
        row(
            "inventory_micro_gap",
            "ok",
            "prior_micro_constitutive_gap_retained",
            truth(micro_gate["summary"]["constitutive_map_reopen_required"]),
            "The older scalar-to-spinor / bilinear constitutive-map gap stays visible as carry-over evidence.",
        ),
        row(
            "inventory_fallback_order",
            "ok",
            "transverse_response_fallback_prepared",
            truth("8.7.56.1659-.1662" in local_text),
            "The transverse-response route is retained but demoted to fallback before this audit starts.",
        ),
    ]

    audit_rows = [
        row(
            "audit_exact_constitutive_map_available",
            "reject",
            "exact constitutive map available under current pack",
            truth(exact_constitutive_map_available),
            "The current pack still lacks a canonical theorem that turns the restored exact vector branch into a uniquely selected observable readout.",
        ),
        row(
            "audit_observable_canonical",
            "reject",
            "observable from current pack canonical",
            truth(observable_from_current_pack_canonical),
            "Current frozen-action structures do not yet canonically decide whether the probe reads T00, overlap-weighted transverse response, or a transformed constitutive quantity.",
        ),
        row(
            "audit_raw_energy_core_selected",
            "reject",
            "raw Hamiltonian core observable selected canonically",
            truth(raw_hamiltonian_core_observable_selected),
            "The exact Hamiltonian core remains exact but not yet canonically promoted to the physical observable readout.",
        ),
        row(
            "audit_micro_chiral_map",
            "reject",
            "microscopic chiral-current constitutive map available",
            truth(microscopic_chiral_current_constitutive_map_available),
            "The earlier microscopic chiral-current constitutive map gap is still unresolved under the current pack.",
        ),
        row(
            "audit_micro_pauli_map",
            "reject",
            "microscopic Pauli-tensor constitutive map available",
            truth(microscopic_pauli_tensor_constitutive_map_available),
            "The earlier microscopic Pauli-tensor constitutive map gap also remains unresolved.",
        ),
        row(
            "audit_followup_primary",
            "ok",
            "branch-local full nonlinear energy-density followup admissible now",
            truth(branch_local_full_nonlinear_energy_density_followup_admissible_now),
            "Because the exact constitutive map does not close, the next honest primary lane is the branch-local full nonlinear energy-density audit.",
        ),
        row(
            "audit_secondary_gate",
            "reject",
            "secondary canonical-promotion admissible now",
            truth(secondary_canonical_promotion_admissible_now),
            "Secondary canonical-promotion work remains downstream of a primary constitutive closure or primary nonlinear rescue.",
        ),
        row(
            "audit_parallel_jeff",
            "ok",
            "parallel J_eff lane retained",
            truth(parallel_jeff_lane_retained),
            "The J_eff theorem lane is retained only as parallel theory work and does not replace the constitutive-map mainline.",
        ),
    ]

    declaration_rows = [
        row(
            "gate_branch_completed",
            "ok",
            "exact constitutive-map audit completed",
            1.0,
            "The branch completes by answering the canonical closure question honestly under the current frozen pack.",
        ),
        row(
            "gate_branch_class",
            "ok",
            "selected branch classification fixed",
            1.0,
            "The audit freezes the current-pack read as constitutive-map unavailable and routes the work to nonlinear energy-density follow-up.",
        ),
        row(
            "gate_physical_reject",
            "ok",
            "physical_reject_required",
            0.0,
            "Physical reject remains false after the constitutive-map audit.",
        ),
        row(
            "gate_numeric_state",
            "ok",
            "numeric_state_changed_by_current_branch",
            0.0,
            "This branch is a canonical closure audit and does not alter the retained scalar or vector benchmark numbers.",
        ),
        row(
            "gate_route_changed",
            "ok",
            "route_state_changed_by_current_branch",
            1.0,
            "The route state changes because the next mainline moves to branch-local full nonlinear energy density.",
        ),
    ]

    route_rows = [
        row(
            "route_next_official",
            "ok",
            "recommended_next_route_or_none",
            1651.0,
            "The next official branch is the branch-local full nonlinear energy-density audit.",
        ),
        row(
            "route_followup_gate",
            "ok",
            "primary_decision_gate_route",
            1655.0,
            "The primary decision gate remains scheduled after the nonlinear follow-up.",
        ),
        row(
            "route_fallback_one",
            "ok",
            "transverse_response_fallback_route",
            1659.0,
            "The P_mu transverse-response observable remains first fallback only if the instruction pack fails.",
        ),
        row(
            "route_fallback_two",
            "ok",
            "branch_selection_fallback_route",
            1663.0,
            "Ground-state / branch-selection remains second fallback only.",
        ),
        row(
            "route_fallback_not_required_now",
            "ok",
            "fallback not required now",
            truth(fallback_not_required_now),
            "The current primary instruction pack still has one unresolved in-pack branch before any fallback route is allowed.",
        ),
    ]

    evidence = {
        "retained_scalar_exact_alpha": SCALAR_ALPHA,
        "retained_target_alpha": TARGET_ALPHA,
        "retained_vector_no_go_alpha": VECTOR_ALPHA,
        "official_energy_core_alpha": energy_ff_gate["summary"][
            "official_alpha_E_at_q_theory"
        ],
        "primary_reopen_surface": PRIMARY_REOPEN,
        "secondary_reopen_surface": SECONDARY_REOPEN,
        "reserve_reopen_surface": RESERVE_REOPEN,
    }

    inventory_payload = payload(
        STEP_TAG,
        STEP_NAME,
        inputs,
        inventory_rows,
        {
            "source_inventory_completed": True,
            "primary_reopen_surface": PRIMARY_REOPEN,
            "fallback_prepared": True,
        },
        {
            "prior_classification": PRIOR_CLASS,
            "selected_branch_classification": BRANCH_CLASS,
        },
        {
            "status_hit": hit(status_text, "exact constitutive-map audit"),
            "roadmap_hit": hit(roadmap_text, "8.7.56.1647-.1650"),
            "local_response_hit": hit(local_text, "exact constitutive-map audit"),
            "part5_hit": hit(part5_text, "exact constitutive-map audit"),
        },
    )
    audit_payload = payload(
        STEP_TAG,
        STEP_NAME,
        inputs,
        audit_rows,
        {
            "exact_constitutive_map_available": exact_constitutive_map_available,
            "observable_from_current_pack_canonical": observable_from_current_pack_canonical,
            "raw_hamiltonian_core_observable_selected": raw_hamiltonian_core_observable_selected,
            "branch_local_full_nonlinear_energy_density_followup_admissible_now": branch_local_full_nonlinear_energy_density_followup_admissible_now,
            "secondary_canonical_promotion_admissible_now": secondary_canonical_promotion_admissible_now,
        },
        {
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "external_exact_constitutive_hit": hit(
                external_text,
                "exact constitutive map audit",
            ),
            "external_probe_hit": hit(
                external_text,
                "probe が直接読む量は `T^{00}` なのか",
            ),
            "external_response_hit": hit(
                external_text,
                "constitutive response として一段変換された quantity",
            ),
            "ai_context_next": ai_context["next"],
        },
    )
    declaration_payload = payload(
        STEP_TAG,
        STEP_NAME,
        inputs,
        declaration_rows,
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "exact_constitutive_map_available": exact_constitutive_map_available,
            "observable_from_current_pack_canonical": observable_from_current_pack_canonical,
            "raw_hamiltonian_core_observable_selected": raw_hamiltonian_core_observable_selected,
            "branch_local_full_nonlinear_energy_density_followup_admissible_now": branch_local_full_nonlinear_energy_density_followup_admissible_now,
            "physical_reject_required": False,
        },
        {
            "selected_branch_classification": BRANCH_CLASS,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
            "physical_reject_required": False,
        },
        evidence,
    )
    route_payload = payload(
        STEP_TAG,
        STEP_NAME,
        inputs,
        route_rows,
        {
            "next_official_branch": "8.7.56.1651-.1654",
            "primary_decision_gate_branch": "8.7.56.1655-.1658",
            "first_fallback_branch": "8.7.56.1659-.1662",
            "second_fallback_branch": "8.7.56.1663-.1666",
        },
        {
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
            "primary_decision_gate_route": "8.7.56.1655-.1658",
            "fallback_route": "8.7.56.1659-.1662",
            "second_fallback_route": "8.7.56.1663-.1666",
        },
        evidence,
    )

    outputs = {
        "inventory": write_artifact("inventory", inventory_payload),
        "audit": write_artifact("audit", audit_payload),
        "declaration_gate": write_artifact("declaration_gate", declaration_payload),
        "route_sync": write_artifact("route_sync", route_payload),
    }

    print(
        json.dumps(
            {
                "branch_class": BRANCH_CLASS,
                "exact_constitutive_map_available": exact_constitutive_map_available,
                "observable_from_current_pack_canonical": observable_from_current_pack_canonical,
                "raw_hamiltonian_core_observable_selected": raw_hamiltonian_core_observable_selected,
                "next_official_branch": "8.7.56.1651-.1654",
                "selected_next_generation_route": NEXT_ROUTE_NAME,
                "recommended_next_route_or_none": NEXT_ROUTE,
                "outputs": outputs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
