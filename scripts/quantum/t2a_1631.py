#!/usr/bin/env python3
"""Generate 8.7.56.1631-.1634 energy-density alpha case-classification artifacts.

This branch does not introduce a new observable. It classifies the already
audited exact Hamiltonian-core energy-density form factor into the honest case
under the current pack.

The inputs are already fixed:

1. The exact Hamiltonian core is
   `epsilon_H,core = |F_0r^(P)|^2 + m_0^2 f_0^2`.
2. Its official exact form factor on the retained branch is
   `F_E(q_theory) = -0.0825465944966888`,
   `alpha_E = 0.0005422361373947313`.
3. That exact read lands near the retained vector no-go scale, not the
   retained scalar strong candidate.

The only question here is therefore:

- should the energy-density branch be read as a scalar-rescue case,
  a vector-no-go-like case, or an "other / evidence-only" case?

This branch fixes the honest official answer and sends the route to closeout.
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

ENERGY_FF_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1627_1630_energy_density_ff_audit_declaration_gate_metrics.json"
)
ENERGY_DERIV_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1623_1626_energy_density_audit_declaration_gate_metrics.json"
)
CASEA_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1599_1602_v2_sub_exact_treat_declaration_gate_metrics.json"
)
CASEB_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1619_1622_eff_metric_v2_sub_restore_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1631-1634"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor energy-density alpha case classification"
)
STEM = build_compact_artifact_stem(STEP_TAG, "energy_density_alpha_case_class", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_energy_density_form_factor_no_scalar_rescue_"
    "case_classification_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_energy_density_case_ii_vector_no_go_like_"
    "disposition_sync_next"
)
SELECTED_CASE = "case_ii_vector_no_go_like_under_current_pack"
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_energy_density_"
    "disposition_sync_closeout"
)
NEXT_ROUTE = "8.7.56.1635"
DOWNSTREAM_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_energy_density_"
    "closeout_reopen_registry"
)
DOWNSTREAM_ROUTE = "8.7.56.1639"


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


# 関数: 標準 metrics row を構成する。

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


# 関数: `.1631-.1634` を実行する。

def main() -> None:
    """Execute the energy-density alpha case-classification branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        PART5,
        ENERGY_FF_GATE,
        ENERGY_DERIV_GATE,
        CASEA_GATE,
        CASEB_GATE,
    ):
        require(path)

    energy_summary = read_json(ENERGY_FF_GATE)["summary"]
    deriv_summary = read_json(ENERGY_DERIV_GATE)["summary"]
    casea_summary = read_json(CASEA_GATE)["summary"]
    caseb_summary = read_json(CASEB_GATE)["summary"]

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    part5_text = read_text(PART5)

    official_alpha = float(energy_summary["official_alpha_E_at_q_theory"])
    official_residual = float(energy_summary["official_alpha_E_residual_rel"])
    official_f_value = float(energy_summary["official_F_E_at_q_theory"])
    core_vs_vector_alpha_rel_gap = float(energy_summary["core_vs_vector_alpha_rel_gap"])
    core_vs_scalar_alpha_rel_gap = float(energy_summary["core_vs_scalar_alpha_rel_gap"])
    core_vs_vector_form_factor_gap = float(energy_summary["core_vs_vector_form_factor_gap"])
    core_vs_scalar_form_factor_gap = float(energy_summary["core_vs_scalar_form_factor_gap"])
    electric_like_alpha = float(energy_summary["electric_like_component_alpha_at_q_theory"])
    note_gradient_alpha = float(energy_summary["note_gradient_alpha_at_q_theory"])
    electric_like_subleading = bool(energy_summary["electric_like_component_subleading"])
    electric_like_improves = bool(energy_summary["electric_like_improves_but_is_not_official"])
    note_gradient_improves = bool(
        energy_summary["note_gradient_alpha_at_q_theory"] > official_alpha
    )
    branch_local_full_energy_available = bool(
        energy_summary["branch_local_full_energy_density_available"]
    )
    energy_tracks_vector = bool(energy_summary["energy_core_tracks_vector_no_go_scale"])
    energy_supports_scalar = bool(energy_summary["energy_core_supports_scalar_candidate"])
    energy_exact_foundation = bool(energy_summary["energy_core_exact_foundation_supported"])

    radial_mass_fraction = float(deriv_summary["radial_mass_term_fraction"])
    electric_like_fraction = float(deriv_summary["electric_like_term_fraction"])
    casea_sub_residual = float(casea_summary["residual_sub_rel"])
    caseb_sub_residual = float(caseb_summary["full_sub_residual_rel"])

    case_i_scalar_rescue_selected = bool(
        energy_supports_scalar and energy_exact_foundation
    )
    case_ii_vector_no_go_like_selected = bool(
        (not case_i_scalar_rescue_selected)
        and energy_tracks_vector
        and (not energy_supports_scalar)
        and core_vs_vector_alpha_rel_gap < 0.05
        and core_vs_vector_form_factor_gap < 0.01
        and core_vs_vector_alpha_rel_gap < core_vs_scalar_alpha_rel_gap
        and core_vs_vector_form_factor_gap < core_vs_scalar_form_factor_gap
    )
    case_iii_noncanonical_improvement_selected = bool(
        (not case_i_scalar_rescue_selected)
        and (not case_ii_vector_no_go_like_selected)
        and electric_like_improves
    )
    exact_core_official_read_fixed = bool(
        case_i_scalar_rescue_selected
        or case_ii_vector_no_go_like_selected
        or case_iii_noncanonical_improvement_selected
    )
    noncanonical_improvement_surfaces_retained = bool(
        electric_like_improves or note_gradient_improves
    )
    full_nonlinear_energy_density_reopen_retained = not branch_local_full_energy_available
    prior_casea_worsen_retained = bool(casea_summary["worsen_selected"])
    prior_caseb_no_metric_rescue_retained = not bool(
        caseb_summary["metric_artifact_rescue_supported"]
    )
    energy_density_disposition_sync_closeout_admissible_now = bool(
        case_ii_vector_no_go_like_selected
    )
    energy_density_closeout_reopen_registry_admissible_now = bool(
        energy_density_disposition_sync_closeout_admissible_now
    )
    physical_reject_required = False

    rows = [
        row(
            "case_i_scalar_rescue_selected",
            "pass" if case_i_scalar_rescue_selected else "reject",
            "Case I scalar-rescue selected",
            truth(case_i_scalar_rescue_selected),
            "The exact Hamiltonian-core read would need to support the retained scalar strong candidate directly to select Case I.",
        ),
        row(
            "case_ii_vector_no_go_like_selected",
            "pass" if case_ii_vector_no_go_like_selected else "reject",
            "Case II vector-no-go-like selected",
            truth(case_ii_vector_no_go_like_selected),
            "The exact official energy-core observable sits near the retained vector no-go scale and far from the retained scalar candidate, so the honest current-pack classification is vector-no-go-like.",
        ),
        row(
            "official_energy_core_alpha_at_q_theory",
            "watch",
            "official energy-core alpha at q_theory",
            official_alpha,
            "This is the exact official read now being classified.",
        ),
        row(
            "official_energy_core_residual_rel",
            "reject" if not energy_supports_scalar else "pass",
            "official energy-core residual relative to target",
            official_residual,
            "The official exact energy-core observable remains far from the target and therefore cannot be called a scalar rescue.",
        ),
        row(
            "core_vs_vector_alpha_rel_gap",
            "pass" if case_ii_vector_no_go_like_selected else "watch",
            "official alpha relative gap vs retained vector no-go",
            core_vs_vector_alpha_rel_gap,
            "The exact official read is close to the retained vector no-go scale.",
        ),
        row(
            "core_vs_scalar_alpha_rel_gap",
            "reject",
            "official alpha relative gap vs retained scalar strong candidate",
            core_vs_scalar_alpha_rel_gap,
            "The exact official read remains far from the retained scalar strong candidate.",
        ),
        row(
            "core_vs_vector_form_factor_gap",
            "pass" if case_ii_vector_no_go_like_selected else "watch",
            "official form-factor gap vs retained vector no-go",
            core_vs_vector_form_factor_gap,
            "The exact official energy-core form factor nearly coincides with the retained vector no-go read.",
        ),
        row(
            "core_vs_scalar_form_factor_gap",
            "reject",
            "official form-factor gap vs retained scalar strong candidate",
            core_vs_scalar_form_factor_gap,
            "The form-factor distance to the retained scalar strong candidate is much larger than the distance to the retained vector no-go read.",
        ),
        row(
            "electric_like_subleading",
            "pass" if electric_like_subleading else "watch",
            "electric-like evidence surface is norm-subleading",
            electric_like_fraction,
            "The electric-like contribution improves numerically but cannot outrank the exact Hamiltonian core because its norm weight remains subleading.",
        ),
        row(
            "noncanonical_improvement_surfaces_retained",
            "watch" if noncanonical_improvement_surfaces_retained else "reject",
            "noncanonical improvement surfaces retained as evidence only",
            truth(noncanonical_improvement_surfaces_retained),
            "Electric-like and note-gradient surfaces are retained as evidence-only improvements, not as official rescue observables.",
        ),
        row(
            "electric_like_alpha_at_q_theory",
            "watch",
            "electric-like evidence alpha at q_theory",
            electric_like_alpha,
            "This surface improves numerically but remains non-canonical under the current pack.",
        ),
        row(
            "note_gradient_alpha_at_q_theory",
            "watch",
            "note-gradient evidence alpha at q_theory",
            note_gradient_alpha,
            "This evidence-only surface improves over the official exact core but still stays far from the target and is not exact under the current pack.",
        ),
        row(
            "full_nonlinear_energy_density_reopen_retained",
            "pass" if full_nonlinear_energy_density_reopen_retained else "reject",
            "branch-local full nonlinear energy density reopen retained",
            truth(full_nonlinear_energy_density_reopen_retained),
            "The branch-local exact nonlinear energy density is still unavailable, so reopen remains the honest follow-up surface even after the current classification.",
        ),
        row(
            "energy_density_disposition_sync_closeout_admissible_now",
            "pass" if energy_density_disposition_sync_closeout_admissible_now else "reject",
            "energy-density disposition sync / closeout admissible now",
            truth(energy_density_disposition_sync_closeout_admissible_now),
            "Once the exact official read is classified as vector-no-go-like, the honest next action is to synchronize the branch-level closeout wording.",
        ),
        row(
            "energy_density_closeout_reopen_registry_admissible_now",
            "pass" if energy_density_closeout_reopen_registry_admissible_now else "reject",
            "energy-density closeout reopen registry admissible now",
            truth(energy_density_closeout_reopen_registry_admissible_now),
            "After closeout, the next useful route is a reopen registry that keeps the remaining branch-local nonlinear energy-density gap explicit.",
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
            "energy_ff_gate": display_path(ENERGY_FF_GATE),
            "energy_deriv_gate": display_path(ENERGY_DERIV_GATE),
            "casea_gate": display_path(CASEA_GATE),
            "caseb_gate": display_path(CASEB_GATE),
        }
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "selected_classification_case": SELECTED_CASE,
        "official_surface_name": "epsilon_H_core",
        "official_F_E_at_q_theory": official_f_value,
        "official_alpha_E_at_q_theory": official_alpha,
        "official_alpha_E_residual_rel": official_residual,
        "case_i_scalar_rescue_selected": case_i_scalar_rescue_selected,
        "case_ii_vector_no_go_like_selected": case_ii_vector_no_go_like_selected,
        "case_iii_noncanonical_improvement_selected": case_iii_noncanonical_improvement_selected,
        "energy_core_tracks_vector_no_go_scale": energy_tracks_vector,
        "energy_core_supports_scalar_candidate": energy_supports_scalar,
        "energy_core_exact_foundation_supported": energy_exact_foundation,
        "core_vs_vector_alpha_rel_gap": core_vs_vector_alpha_rel_gap,
        "core_vs_scalar_alpha_rel_gap": core_vs_scalar_alpha_rel_gap,
        "core_vs_vector_form_factor_gap": core_vs_vector_form_factor_gap,
        "core_vs_scalar_form_factor_gap": core_vs_scalar_form_factor_gap,
        "radial_mass_term_fraction": radial_mass_fraction,
        "electric_like_term_fraction": electric_like_fraction,
        "electric_like_component_alpha_at_q_theory": electric_like_alpha,
        "note_gradient_alpha_at_q_theory": note_gradient_alpha,
        "noncanonical_improvement_surfaces_retained": noncanonical_improvement_surfaces_retained,
        "full_nonlinear_energy_density_reopen_retained": full_nonlinear_energy_density_reopen_retained,
        "prior_casea_worsen_retained": prior_casea_worsen_retained,
        "prior_caseb_no_metric_rescue_retained": prior_caseb_no_metric_rescue_retained,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_registry_route": DOWNSTREAM_ROUTE_NAME,
        "recommended_followup_registry_route_or_none": DOWNSTREAM_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": exact_core_official_read_fixed,
        "next_required_artifacts": [NEXT_ROUTE_NAME, DOWNSTREAM_ROUTE_NAME],
    }

    evidence = {
        "hits": {
            "status_energy_branch": hit(status_text, "energy-density alpha case classification"),
            "roadmap_energy_branch": hit(roadmap_text, "energy-density alpha case classification"),
            "current_problem_energy_branch": hit(current_problem_text, "energy-density alpha case classification"),
            "current_status_energy_branch": hit(current_status_text, "energy-density alpha case classification"),
            "unified_roadmap_energy_branch": hit(
                unified_roadmap_text,
                "`.1631-.1634` は **energy-density alpha case classification**",
            ),
            "part5_energy_branch": hit(part5_text, "**energy-density alpha case classification**"),
        },
        "official_read": {
            "official_F_E_at_q_theory": official_f_value,
            "official_alpha_E_at_q_theory": official_alpha,
            "official_alpha_E_residual_rel": official_residual,
        },
        "comparison_surfaces": {
            "electric_like_component_alpha_at_q_theory": electric_like_alpha,
            "note_gradient_alpha_at_q_theory": note_gradient_alpha,
            "casea_subtraction_residual_rel": casea_sub_residual,
            "caseb_subtraction_residual_rel": caseb_sub_residual,
        },
        "retained_numeric_state": {
            "scalar_alpha_exact_at_q_theory": 0.00715678583937324,
            "vector_alpha_at_q_theory": 0.0005579616187042394,
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
        },
    }

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload(
                "8.7.56.1631",
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
                "8.7.56.1632",
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
                "8.7.56.1633",
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
                "8.7.56.1634",
                f"{STEP_NAME} route sync",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
    }

    print(json.dumps({"step": STEP_TAG, "stem": STEM, "artifacts": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
