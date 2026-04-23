#!/usr/bin/env python3
"""Generate 8.7.56.1567-.1570 J_eff^0 structure-classification artifacts.

This branch must not fall back into a text-search loop. The previous branch
already fixed the explicit frozen-action split

    J_eff^mu = J_eff,kin^mu + J_eff,stk^mu + J_eff,gf^mu
             + J_eff,matter^mu + J_eff,rot^mu .

The present task is therefore a blind classification problem: under the
current pack, is the exact charge density read honestly as

1. scalar proxy |f_0|^2,
2. naive signed density |f_0|^2 - |f_L|^2,
3. another nonzero combination,
4. zero?

Because the explicit split is fixed, the same-field on-shell self-source is
already known to vanish, and microscopic matter / rotational functionals are
still absent, the honest current-pack classification is Case IV-like zero
under the current pack. A future nonzero reopen is still retained, but it is
explicitly downstream of microscopic functional derivation rather than hidden
inside the present pack.
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
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
DIRECTIVE_NOTE = Path(
    r"C:\Users\ogawa\Downloads\trial2_vector_qball_jeff_final_directive_20260328.md"
)
PRIOR_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1563_1566_direct_jeff_deriv_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1567-1570"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor J_eff^0 structure classification"
)
STEM = build_compact_artifact_stem(STEP_TAG, "jeff_q0_class", prefix="q")

PRIOR_CLASS = "vector_qball_form_factor_direct_jeff_split_derived_charge_density_classification_next"
BRANCH_CLASS = (
    "vector_qball_form_factor_jeff_charge_density_zero_current_pack_disposition_sync_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_jeff_case_iv_zero_current_pack_disposition_sync"
)
NEXT_ROUTE = "8.7.56.1571"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_microscopic_matter_rot_functional_reopen"
)
FOLLOWUP_ROUTE = "8.7.56.1575"


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を確認する。

def require(path: Path) -> None:
    """Abort when one required path is missing."""
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


# 関数: 表示用の相対パスを返す。

def display_path(path: Path) -> str:
    """Return one repo-relative path when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# 関数: 部分文字列に一致する最初の行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line matching one substring."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: metrics row を構成する。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# 関数: 標準 payload を構成する。

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


# 関数: `.1567-.1570` を実行する。

def main() -> None:
    """Execute the J_eff^0 structure-classification branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        PART5,
        DIRECTIVE_NOTE,
        PRIOR_GATE,
    ):
        require(path)

    prior_gate = read_json(PRIOR_GATE)
    prior_summary = prior_gate["summary"]
    prior_formulas = prior_gate["evidence"]["formulas"]

    directive_text = read_text(DIRECTIVE_NOTE)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    part5_text = read_text(PART5)

    prior_split_ready = bool(
        prior_summary.get("trial2_numeric_alpha_problem_classification") == PRIOR_CLASS
        and prior_summary.get("jeff_eff_mu_explicit_form_derived", False)
        and prior_summary.get("jeff_eff_mu_explicit_split_component_count") == 5.0
    )
    same_field_on_shell_zero_retained = bool(
        prior_summary.get("same_field_on_shell_zero_retained", False)
    )
    microscopic_matter_functional_available = bool(
        prior_summary.get("microscopic_matter_functional_available", False)
    )
    microscopic_rotational_functional_available = bool(
        prior_summary.get("microscopic_rotational_functional_available", False)
    )
    massless_transverse_mode_retained = bool(
        prior_summary.get("massless_transverse_mode_retained", False)
    )

    self_source_zero_component_count = 3.0 if same_field_on_shell_zero_retained else 0.0
    missing_functional_component_count = 0.0
    if not microscopic_matter_functional_available:
        missing_functional_component_count += 1.0

    if not microscopic_rotational_functional_available:
        missing_functional_component_count += 1.0

    scalar_proxy_exact_supported = False
    signed_density_exact_supported = False
    other_nonzero_combination_supported = False
    zero_structure_selected_under_current_pack = bool(
        prior_split_ready
        and same_field_on_shell_zero_retained
        and not microscopic_matter_functional_available
        and not microscopic_rotational_functional_available
    )
    conditional_nonzero_reopen_retained = bool(
        zero_structure_selected_under_current_pack and missing_functional_component_count > 0.0
    )
    jeff_eff_charge_density_structure_identified = zero_structure_selected_under_current_pack
    classification_case_i_supported = scalar_proxy_exact_supported
    classification_case_ii_supported = other_nonzero_combination_supported
    classification_case_iii_supported = signed_density_exact_supported
    classification_case_iv_zero_under_current_pack = zero_structure_selected_under_current_pack
    disposition_case_selected = False
    effective_source_theorem_attempt_admissible_now = False
    observable_dictionary_gate_admissible_now = False

    directive_case_i_hit = hit(directive_text, "Case I: J_eff⁰ ≈ |f₀|²")
    directive_case_iii_hit = hit(directive_text, "Case III: J_eff⁰ = |f₀|² − |f_L|²")
    directive_case_iv_hit = hit(directive_text, "Case IV: J_eff = 0")
    problem_zero_hit = hit(
        current_problem_text,
        "same_field_on_shell_zero_retained = true",
    )
    status_zero_hit = hit(
        current_status_text,
        "same_field_on_shell_zero_retained = true",
    )
    roadmap_classification_hit = hit(
        unified_roadmap_text,
        "J_{\\mathrm{eff}}^0",
    )
    part5_state_hit = hit(
        part5_text,
        "vector_qball_form_factor_direct_jeff_split_derived_charge_density_classification_next",
    )

    rows = [
        row(
            "prior_split_ready",
            "pass" if prior_split_ready else "reject",
            "prior explicit J_eff split ready",
            truth(prior_split_ready),
            "The current branch only classifies structure after the five-piece direct J_eff split is fixed.",
        ),
        row(
            "same_field_on_shell_zero_retained",
            "pass" if same_field_on_shell_zero_retained else "reject",
            "same-field on-shell zero retained",
            truth(same_field_on_shell_zero_retained),
            "The explicit self-source already collapses to zero on the same-field on-shell background.",
        ),
        row(
            "massless_transverse_mode_retained",
            "pass" if massless_transverse_mode_retained else "reject",
            "massless transverse mode retained",
            truth(massless_transverse_mode_retained),
            "The classification still refers to the physical transverse light mode only.",
        ),
        row(
            "microscopic_matter_functional_available",
            "pass" if microscopic_matter_functional_available else "reject",
            "microscopic matter functional available",
            truth(microscopic_matter_functional_available),
            "No explicit Q-ball-background matter-current constitutive map is available in the current pack.",
        ),
        row(
            "microscopic_rotational_functional_available",
            "pass" if microscopic_rotational_functional_available else "reject",
            "microscopic rotational functional available",
            truth(microscopic_rotational_functional_available),
            "No explicit reduced rotational-source functional is available on the restored exact vector branch.",
        ),
        row(
            "self_source_zero_component_count",
            "pass" if self_source_zero_component_count == 3.0 else "reject",
            "explicit self-source components fixed to zero",
            self_source_zero_component_count,
            "The kinetic, Stueckelberg, and gauge-fixing sectors are retained only through the same-field on-shell zero identity.",
        ),
        row(
            "missing_functional_component_count",
            "pass" if missing_functional_component_count == 2.0 else "reject",
            "microscopic source components still missing",
            missing_functional_component_count,
            "The matter and rotational slots remain symbolic placeholders rather than nonzero branch-local functionals.",
        ),
        row(
            "scalar_proxy_exact_supported",
            "pass" if scalar_proxy_exact_supported else "reject",
            "scalar-proxy exact structure supported",
            truth(scalar_proxy_exact_supported),
            "The current pack does not derive J_eff^0 as an exact |f_0|^2 leading term.",
        ),
        row(
            "signed_density_exact_supported",
            "pass" if signed_density_exact_supported else "reject",
            "naive signed-density exact structure supported",
            truth(signed_density_exact_supported),
            "The current pack does not derive J_eff^0 as the exact |f_0|^2 - |f_L|^2 combination.",
        ),
        row(
            "other_nonzero_combination_supported",
            "pass" if other_nonzero_combination_supported else "reject",
            "other nonzero structure supported",
            truth(other_nonzero_combination_supported),
            "No exact nonzero alternative combination is opened by the present split plus current-pack functionals.",
        ),
        row(
            "zero_structure_selected_under_current_pack",
            "pass" if zero_structure_selected_under_current_pack else "reject",
            "zero structure selected under current pack",
            truth(zero_structure_selected_under_current_pack),
            "With same-field on-shell zero retained and both microscopic source functionals absent, the honest current-pack read is zero.",
        ),
        row(
            "conditional_nonzero_reopen_retained",
            "pass" if conditional_nonzero_reopen_retained else "reject",
            "conditional nonzero reopen retained",
            truth(conditional_nonzero_reopen_retained),
            "A nonzero classification remains possible only after microscopic matter/rotational functionals are reopened.",
        ),
        row(
            "jeff_eff_charge_density_structure_identified",
            "pass" if jeff_eff_charge_density_structure_identified else "reject",
            "J_eff^0 structure identified",
            truth(jeff_eff_charge_density_structure_identified),
            "The present branch identifies the honest current-pack class as zero under the fixed frozen-action split.",
        ),
        row(
            "classification_case_iv_zero_under_current_pack",
            "pass" if classification_case_iv_zero_under_current_pack else "reject",
            "Case IV-like zero classification under current pack",
            truth(classification_case_iv_zero_under_current_pack),
            "This is not yet the final disposition sync, but the blind structure classification lands on the zero branch.",
        ),
        row(
            "disposition_case_selected",
            "pass" if disposition_case_selected else "reject",
            "disposition case selected",
            truth(disposition_case_selected),
            "Formal Case I-IV disposition is synced in the next branch rather than inside the blind classification branch.",
        ),
        row(
            "effective_source_theorem_attempt_admissible_now",
            "pass" if effective_source_theorem_attempt_admissible_now else "reject",
            "effective source theorem attempt admissible now",
            truth(effective_source_theorem_attempt_admissible_now),
            "Source-theorem work remains downstream of disposition sync and microscopic functional reopen.",
        ),
        row(
            "observable_dictionary_gate_admissible_now",
            "pass" if observable_dictionary_gate_admissible_now else "reject",
            "observable dictionary gate admissible now",
            truth(observable_dictionary_gate_admissible_now),
            "Observable mapping remains downstream of both direct J_eff classification and future microscopic reopen.",
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
            "part5": display_path(PART5),
            "directive_note": display_path(DIRECTIVE_NOTE),
            "prior_gate": display_path(PRIOR_GATE),
        },
        "constants": {
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "followup_route_name": FOLLOWUP_ROUTE_NAME,
            "followup_route": FOLLOWUP_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "jeff_eff_mu_explicit_form_derived": prior_summary.get("jeff_eff_mu_explicit_form_derived", False),
        "jeff_eff_mu_explicit_split_component_count": 5.0,
        "same_field_on_shell_zero_retained": same_field_on_shell_zero_retained,
        "massless_transverse_mode_retained": massless_transverse_mode_retained,
        "microscopic_matter_functional_available": microscopic_matter_functional_available,
        "microscopic_rotational_functional_available": microscopic_rotational_functional_available,
        "scalar_proxy_exact_supported": scalar_proxy_exact_supported,
        "signed_density_exact_supported": signed_density_exact_supported,
        "other_nonzero_combination_supported": other_nonzero_combination_supported,
        "zero_structure_selected_under_current_pack": zero_structure_selected_under_current_pack,
        "conditional_nonzero_reopen_retained": conditional_nonzero_reopen_retained,
        "classification_case_iv_zero_under_current_pack": classification_case_iv_zero_under_current_pack,
        "jeff_eff_charge_density_structure_identified": jeff_eff_charge_density_structure_identified,
        "disposition_case_selected": disposition_case_selected,
        "effective_source_theorem_attempt_admissible_now": effective_source_theorem_attempt_admissible_now,
        "observable_dictionary_gate_admissible_now": observable_dictionary_gate_admissible_now,
        "frozen_action_only_used": True,
        "new_free_parameters_introduced": False,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "scalar_strong_candidate_retained": True,
        "blind_vector_no_go_retained": True,
        "physical_reject_required": False,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": jeff_eff_charge_density_structure_identified,
        "next_required_artifacts": [
            NEXT_ROUTE_NAME,
            FOLLOWUP_ROUTE_NAME,
        ],
    }

    evidence = {
        "formulas": {
            "jeff_charge_density": prior_formulas["jeff_charge_density"],
            "same_field_on_shell": prior_formulas["same_field_on_shell"],
            "classification_rule": (
                "If the explicit self-source is zero on-shell and no microscopic matter/rotational "
                "functional is available, the honest current-pack J_eff^0 class is zero."
            ),
        },
        "hits": {
            "directive_case_i": directive_case_i_hit,
            "directive_case_iii": directive_case_iii_hit,
            "directive_case_iv": directive_case_iv_hit,
            "current_problem_zero": problem_zero_hit,
            "current_status_zero": status_zero_hit,
            "unified_roadmap_classification": roadmap_classification_hit,
            "part5_current_state": part5_state_hit,
        },
        "support_counts": {
            "explicit_split_component_count": 5.0,
            "self_source_zero_component_count": self_source_zero_component_count,
            "missing_functional_component_count": missing_functional_component_count,
            "current_pack_nonzero_support_count": 0.0,
        },
        "retained_numeric_state": {
            "scalar_F_exact_at_q_theory": 0.2998913524347805,
            "scalar_alpha_exact_at_q_theory": 0.00715678583937324,
            "vector_F_at_q_theory": -0.083735013520183,
            "vector_alpha_at_q_theory": 0.0005579616187042394,
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
        },
    }

    inventory_paths = write_artifact(
        "inventory",
        payload("8.7.56.1567", f"{STEP_NAME} inventory", inputs, rows, summary, decision, evidence),
    )
    audit_paths = write_artifact(
        "audit",
        payload("8.7.56.1568", f"{STEP_NAME} audit", inputs, rows, summary, decision, evidence),
    )
    gate_paths = write_artifact(
        "declaration_gate",
        payload(
            "8.7.56.1569",
            f"{STEP_NAME} declaration gate",
            inputs,
            rows,
            summary,
            decision,
            evidence,
        ),
    )
    route_paths = write_artifact(
        "route_sync",
        payload("8.7.56.1570", f"{STEP_NAME} route sync", inputs, rows, summary, decision, evidence),
    )

    print("[ok] J_eff^0 structure-classification artifacts written:")
    print(f" - {inventory_paths['json']}")
    print(f" - {audit_paths['json']}")
    print(f" - {gate_paths['json']}")
    print(f" - {route_paths['json']}")


if __name__ == "__main__":
    main()
