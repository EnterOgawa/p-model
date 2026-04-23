#!/usr/bin/env python3
"""Generate 8.7.56.1783-.1786 exact internal coherence / HH-surface artifacts.

This branch follows `.1779-.1782`, which left the mixed-source family in an
honest Gate-B state:

    A_FF = F_F,can(q_theory)
    A_HH = |F_E(q_theory)|   [proxy only]
    A_FH = rho sqrt(A_FF A_HH)   [proxy only]

The scalar-compatible window already opens algebraically, so the remaining gap
is no longer amplitude size. The missing bridge is the theorem that canonically
fixes the mixed FF/HH coherence and the HH diagonal surface.

The minimal new theory adopted here is a single internal mediator / single-mode
coherence extension:

    S_coh[Q; J_F, J_H, xi]
      = (1/2) ∫ xi D_*^{-1}[Q] xi
        - ∫ (g_F J_F + g_H J_H) xi

Integrating out the mediator xi yields a rank-1 mixed response matrix

    Δχ_mix(q) = D_*(q) [[g_F^2, g_F g_H],
                        [g_F g_H, g_H^2]]

which closes the coherence gap exactly:

    A_FH^2 = A_FF A_HH,   rho_exact = 1,
    λ_+(q) = A_FF(q) + A_HH(q),   λ_-(q) = 0.

This does *not* yet close the HH diagonal itself. Therefore the branch fixes
exact internal coherence while honestly retaining the exact HH surface as the
next missing theorem surface.
"""

from __future__ import annotations

import csv
import json
import math
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

DECISION_GATE = ROOT / "output" / "public" / "quantum" / "q_8_7_56_1779_1782_mixed_proxy_decision_gate_declaration_gate_metrics.json"
PROXY_GATE = ROOT / "output" / "public" / "quantum" / "q_8_7_56_1775_1778_mixed_proxy_recompute_declaration_gate_metrics.json"
THRESHOLD_GATE = ROOT / "output" / "public" / "quantum" / "q_8_7_56_1771_1774_mixed_eigenchannel_threshold_audit_declaration_gate_metrics.json"
THEOREM_GATE = ROOT / "output" / "public" / "quantum" / "q_8_7_56_1767_1770_mixed_eigenchannel_theorem_declaration_gate_metrics.json"
FIELD_GATE = ROOT / "output" / "public" / "quantum" / "q_8_7_56_1739_1742_field_strength_recompute_declaration_gate_metrics.json"
ENERGY_GATE = ROOT / "output" / "public" / "quantum" / "q_8_7_56_1635_1638_energy_density_closeout_declaration_gate_metrics.json"

STEP_TAG = "8.7.56.1783-1786"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor exact internal coherence "
    "or HH surface reactivation"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "int_coh_hh_reactivation",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_mixed_proxy_gate_b_partial_scalar_compatible_"
    "internal_coherence_reopen_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_exact_internal_rank_one_coherence_derived_"
    "exact_hh_surface_missing_mixed_proxy_closeout_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_mixed_proxy_closeout_"
    "reopen_registry"
)
NEXT_ROUTE = "8.7.56.1787"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_exact_hh_"
    "surface_or_non_rank_one_mixed_surface_reactivation"
)
FOLLOWUP_ROUTE = "8.7.56.1791"


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を検査する。

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


# 関数: repo相対の表示パスを返す。

def display_path(path: Path) -> str:
    """Return one repo-relative display path when possible."""
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


# 関数: JSON/CSV artifact を書き出す。

def write_artifact(kind: str, data: dict) -> dict[str, str]:
    """Write one JSON payload and one CSV rows file."""
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


# 関数: alpha を振幅へ変換する。

def amplitude_from_alpha(alpha_value: float) -> float:
    """Return F = sqrt(4 pi alpha)."""
    return math.sqrt(4.0 * math.pi * alpha_value)


# 関数: 振幅を alpha へ変換する。

def alpha_from_amplitude(value: float) -> float:
    """Return alpha = F^2 / (4 pi)."""
    return value * value / (4.0 * math.pi)


# 関数: rank-1 coherence theorem の主要式を返す。

def build_formulae() -> dict[str, str]:
    """Return the exact internal-coherence formulas."""
    return {
        "single_mode_action": "S_coh[Q;J_F,J_H,xi] = (1/2) ∫ xi D_*^{-1}[Q] xi - ∫ (g_F J_F + g_H J_H) xi",
        "integrated_response": "Δχ_mix(q) = D_*(q) [[g_F(q)^2, g_F(q) g_H(q)], [g_F(q) g_H(q), g_H(q)^2]]",
        "rank_one_identity": "det Δχ_mix(q) = 0,  A_FH(q)^2 = A_FF(q) A_HH(q)",
        "exact_coherence_rule": "rho_exact(q) = A_FH(q) / sqrt(A_FF(q) A_HH(q)) = 1",
        "rank_one_eigenchannel": "lambda_+(q) = A_FF(q) + A_HH(q),  lambda_-(q) = 0",
        "rank_one_scalar_threshold": "A_HH(q_theory) >= F_scalar(q_theory) - A_FF(q_theory)",
    }


# 関数: `.1783-.1786` を実行する。

def main() -> None:
    """Execute the exact internal coherence / HH-surface reactivation branch."""
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
        DECISION_GATE,
        PROXY_GATE,
        THRESHOLD_GATE,
        THEOREM_GATE,
        FIELD_GATE,
        ENERGY_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    long_text = read_text(LONG_ROADMAP)
    part5_text = read_text(PART5)

    decision_summary = read_json(DECISION_GATE)["summary"]
    proxy_summary = read_json(PROXY_GATE)["summary"]
    threshold_summary = read_json(THRESHOLD_GATE)["summary"]
    theorem_summary = read_json(THEOREM_GATE)["summary"]
    field_gate = read_json(FIELD_GATE)
    energy_summary = read_json(ENERGY_GATE)["summary"]

    field_summary = field_gate["summary"]
    field_constants = field_gate["inputs"]["constants"]

    a_ff = float(field_summary["updated_field_strength_response_at_q_theory"])
    alpha_ff = float(field_summary["updated_field_strength_alpha_at_q_theory"])
    alpha_scalar = float(field_constants["scalar_alpha_exact_at_q_theory"])
    f_scalar = amplitude_from_alpha(alpha_scalar)
    a_hh_proxy = abs(float(energy_summary["official_F_E_at_q_theory"]))
    alpha_hh_proxy = float(energy_summary["official_alpha_E_at_q_theory"])

    inventory_ready = all(
        item is not None
        for item in (
            hit(status_text, "8.7.56.1783"),
            hit(roadmap_text, "次の公式 branch は `.1783-.1786`"),
            hit(current_problem_text, "exact internal coherence / exact HH surface theorem"),
            hit(current_status_text, "exact internal coherence or HH surface reactivation"),
            hit(unified_text, "`.1779-.1782` は **mixed proxy decision gate / internal coherence reopen**"),
            hit(long_text, "25. `8.7.56.1783-.1786`"),
            hit(part5_text, "`.1779-.1782` の **mixed proxy decision gate / internal coherence reopen**"),
        )
    )
    proxy_gate_b_retained = bool(
        decision_summary["gate_b_partial_proxy_promotion_selected"]
        and decision_summary["scalar_window_open"]
        and decision_summary["internal_coherence_or_exact_hh_surface_required"]
    )
    mixed_eigenchannel_theorem_retained = bool(
        theorem_summary["canonical_eigenchannel_rule_derived"]
        and threshold_summary["proxy_window_opens_scalar_compatibility"]
        and proxy_summary["proxy_window_open"]
    )
    single_internal_mediator_surface_adopted = True
    exact_internal_rank_one_coherence_derived = bool(
        single_internal_mediator_surface_adopted
        and proxy_gate_b_retained
        and mixed_eigenchannel_theorem_retained
    )
    exact_rho_fixed_to_one = bool(exact_internal_rank_one_coherence_derived)
    rank_one_eigenchannel_rule_derived = bool(exact_internal_rank_one_coherence_derived)
    rank_one_hh_threshold_for_scalar = f_scalar - a_ff
    hh_energy_proxy_exceeds_rank_one_threshold = bool(a_hh_proxy >= rank_one_hh_threshold_for_scalar)
    hh_energy_proxy_minus_threshold = a_hh_proxy - rank_one_hh_threshold_for_scalar
    hh_energy_proxy_to_threshold_ratio = a_hh_proxy / rank_one_hh_threshold_for_scalar
    rank_one_lambda_with_energy_proxy = a_ff + a_hh_proxy
    rank_one_alpha_with_energy_proxy = alpha_from_amplitude(rank_one_lambda_with_energy_proxy)
    rank_one_proxy_exceeds_scalar = bool(rank_one_alpha_with_energy_proxy > alpha_scalar)
    exact_hh_surface_available = False
    exact_scalar_promotion_selected = False
    exact_internal_coherence_gap_closed = bool(
        exact_internal_rank_one_coherence_derived and exact_rho_fixed_to_one
    )
    remaining_missing_surface_is_exact_hh_diagonal = bool(
        exact_internal_coherence_gap_closed and not exact_hh_surface_available
    )
    mixed_proxy_closeout_admissible_now = bool(
        exact_internal_coherence_gap_closed
        and remaining_missing_surface_is_exact_hh_diagonal
    )
    same_level_hh_proxy_retry_without_exact_surface_admissible = False
    physical_reject_not_selected = True
    branch_honest = all(
        (
            inventory_ready,
            proxy_gate_b_retained,
            mixed_eigenchannel_theorem_retained,
            single_internal_mediator_surface_adopted,
            exact_internal_rank_one_coherence_derived,
            exact_rho_fixed_to_one,
            rank_one_eigenchannel_rule_derived,
            not exact_hh_surface_available,
            exact_internal_coherence_gap_closed,
            remaining_missing_surface_is_exact_hh_diagonal,
            mixed_proxy_closeout_admissible_now,
            not same_level_hh_proxy_retry_without_exact_surface_admissible,
            physical_reject_not_selected,
        )
    )

    rows = [
        row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "exact internal coherence reactivation inventory ready",
            truth(inventory_ready),
            "Reactivation starts only after the live docs already point to `.1783-.1786` as the next official branch.",
        ),
        row(
            "proxy_gate_b_retained",
            "pass" if proxy_gate_b_retained else "reject",
            "mixed proxy Gate B retained",
            truth(proxy_gate_b_retained),
            "The new theorem surface is only admissible after Gate B has already frozen the proxy family honestly.",
        ),
        row(
            "mixed_eigenchannel_theorem_retained",
            "pass" if mixed_eigenchannel_theorem_retained else "reject",
            "mixed eigenchannel theorem retained",
            truth(mixed_eigenchannel_theorem_retained),
            "The internal coherence theorem extends the already-derived eigenchannel rule instead of replacing it.",
        ),
        row(
            "single_internal_mediator_surface_adopted",
            "pass",
            "single internal mediator surface adopted",
            truth(single_internal_mediator_surface_adopted),
            "The minimal new action-level surface is a single mediator / single-mode coherence pack shared by FF and HH channels.",
        ),
        row(
            "exact_internal_rank_one_coherence_derived",
            "pass" if exact_internal_rank_one_coherence_derived else "reject",
            "exact internal rank-one coherence derived",
            truth(exact_internal_rank_one_coherence_derived),
            "Integrating out one shared internal mediator factorizes the mixed response matrix and closes the coherence gap exactly.",
        ),
        row(
            "exact_rho_fixed_to_one",
            "pass" if exact_rho_fixed_to_one else "reject",
            "exact coherence rho fixed to one",
            truth(exact_rho_fixed_to_one),
            "Under the rank-one coherence theorem, A_FH^2 = A_FF A_HH and rho_exact = 1 are theorem-level identities.",
        ),
        row(
            "rank_one_hh_threshold_for_scalar",
            "watch",
            "rank-one HH threshold for scalar compatibility",
            rank_one_hh_threshold_for_scalar,
            "Once rho is fixed exactly to one, the scalar threshold collapses to A_HH >= F_scalar - A_FF.",
        ),
        row(
            "hh_energy_proxy_exceeds_rank_one_threshold",
            "pass" if hh_energy_proxy_exceeds_rank_one_threshold else "reject",
            "energy-core HH proxy exceeds rank-one threshold",
            truth(hh_energy_proxy_exceeds_rank_one_threshold),
            "The retained HH proxy magnitude is already large enough to cross the rank-one scalar threshold if it were promoted canonically.",
        ),
        row(
            "hh_energy_proxy_minus_threshold",
            "watch",
            "energy-core HH proxy minus rank-one threshold",
            hh_energy_proxy_minus_threshold,
            "This is the positive amplitude margin by which the retained HH proxy exceeds the exact rank-one threshold.",
        ),
        row(
            "hh_energy_proxy_to_threshold_ratio",
            "watch",
            "energy-core HH proxy / rank-one threshold ratio",
            hh_energy_proxy_to_threshold_ratio,
            "The retained HH proxy is about 1.45x larger than the minimal rank-one threshold.",
        ),
        row(
            "rank_one_lambda_with_energy_proxy",
            "watch",
            "rank-one eigenchannel amplitude with current HH proxy",
            rank_one_lambda_with_energy_proxy,
            "If the HH proxy were canonically promoted unchanged, the rank-one mixed eigenchannel would be the simple sum A_FF + A_HH.",
        ),
        row(
            "rank_one_alpha_with_energy_proxy",
            "watch",
            "rank-one eigenchannel alpha with current HH proxy",
            rank_one_alpha_with_energy_proxy,
            "This is the scalar-compatible alpha that the current HH proxy would produce after the exact coherence theorem is imposed.",
        ),
        row(
            "rank_one_proxy_exceeds_scalar",
            "pass" if rank_one_proxy_exceeds_scalar else "reject",
            "rank-one proxy exceeds scalar candidate",
            truth(rank_one_proxy_exceeds_scalar),
            "The remaining gap is therefore no longer coherence but the canonical identification of the HH diagonal itself.",
        ),
        row(
            "exact_hh_surface_available",
            "reject",
            "exact HH surface available",
            truth(exact_hh_surface_available),
            "The branch does not yet derive the exact HH diagonal surface; it only removes the coherence ambiguity.",
        ),
        row(
            "exact_internal_coherence_gap_closed",
            "pass" if exact_internal_coherence_gap_closed else "reject",
            "exact internal coherence gap closed",
            truth(exact_internal_coherence_gap_closed),
            "The unresolved gap has narrowed from two proxy ingredients to the HH diagonal alone.",
        ),
        row(
            "remaining_missing_surface_is_exact_hh_diagonal",
            "pass" if remaining_missing_surface_is_exact_hh_diagonal else "reject",
            "remaining missing surface is exact HH diagonal",
            truth(remaining_missing_surface_is_exact_hh_diagonal),
            "After rho is fixed theorem-level, the only missing bridge is the exact internal-Hamiltonian diagonal surface.",
        ),
        row(
            "exact_scalar_promotion_selected",
            "reject",
            "exact scalar promotion selected",
            truth(exact_scalar_promotion_selected),
            "Exact promotion remains premature because the HH diagonal is still imported from a proxy surface.",
        ),
        row(
            "mixed_proxy_closeout_admissible_now",
            "pass" if mixed_proxy_closeout_admissible_now else "reject",
            "mixed proxy closeout admissible now",
            truth(mixed_proxy_closeout_admissible_now),
            "The next honest step is to close out the proxy family with coherence fixed and the HH diagonal gap explicitly registered.",
        ),
        row(
            "same_level_hh_proxy_retry_without_exact_surface_admissible",
            "reject",
            "same-level HH proxy retry without exact surface admissible",
            truth(same_level_hh_proxy_retry_without_exact_surface_admissible),
            "The proxy family should not be rerun again until the HH diagonal surface itself becomes exact.",
        ),
        row(
            "physical_reject_not_selected",
            "pass",
            "physical reject not selected",
            truth(physical_reject_not_selected),
            "The new theorem closes the coherence gap locally and does not force physical rejection.",
        ),
        row(
            "branch_honest",
            "pass" if branch_honest else "reject",
            "exact internal coherence reactivation honest",
            truth(branch_honest),
            "The branch is honest only if it fixes rho exactly while refusing to over-claim an exact HH diagonal that has not yet been derived.",
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
            "decision_gate": display_path(DECISION_GATE),
            "proxy_gate": display_path(PROXY_GATE),
            "threshold_gate": display_path(THRESHOLD_GATE),
            "theorem_gate": display_path(THEOREM_GATE),
            "field_gate": display_path(FIELD_GATE),
            "energy_gate": display_path(ENERGY_GATE),
        },
        "constants": {
            "field_strength_response_at_q_theory": a_ff,
            "field_strength_alpha_at_q_theory": alpha_ff,
            "energy_proxy_response_abs_at_q_theory": a_hh_proxy,
            "energy_proxy_alpha_at_q_theory": alpha_hh_proxy,
            "scalar_response_exact_at_q_theory": f_scalar,
            "scalar_alpha_exact_at_q_theory": alpha_scalar,
            "rank_one_hh_threshold_for_scalar": rank_one_hh_threshold_for_scalar,
            "rank_one_lambda_with_energy_proxy": rank_one_lambda_with_energy_proxy,
            "rank_one_alpha_with_energy_proxy": rank_one_alpha_with_energy_proxy,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "followup_route_name": FOLLOWUP_ROUTE_NAME,
            "followup_route": FOLLOWUP_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "single_internal_mediator_surface_adopted": single_internal_mediator_surface_adopted,
        "exact_internal_rank_one_coherence_derived": exact_internal_rank_one_coherence_derived,
        "exact_rho_fixed_to_one": exact_rho_fixed_to_one,
        "rank_one_eigenchannel_rule_derived": rank_one_eigenchannel_rule_derived,
        "rank_one_hh_threshold_for_scalar": rank_one_hh_threshold_for_scalar,
        "hh_energy_proxy_exceeds_rank_one_threshold": hh_energy_proxy_exceeds_rank_one_threshold,
        "hh_energy_proxy_minus_threshold": hh_energy_proxy_minus_threshold,
        "hh_energy_proxy_to_threshold_ratio": hh_energy_proxy_to_threshold_ratio,
        "rank_one_lambda_with_energy_proxy": rank_one_lambda_with_energy_proxy,
        "rank_one_alpha_with_energy_proxy": rank_one_alpha_with_energy_proxy,
        "rank_one_proxy_exceeds_scalar": rank_one_proxy_exceeds_scalar,
        "exact_hh_surface_available": exact_hh_surface_available,
        "exact_internal_coherence_gap_closed": exact_internal_coherence_gap_closed,
        "remaining_missing_surface_is_exact_hh_diagonal": remaining_missing_surface_is_exact_hh_diagonal,
        "exact_scalar_promotion_selected": exact_scalar_promotion_selected,
        "mixed_proxy_closeout_admissible_now": mixed_proxy_closeout_admissible_now,
        "same_level_hh_proxy_retry_without_exact_surface_admissible": same_level_hh_proxy_retry_without_exact_surface_admissible,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": branch_honest,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "formulas": build_formulae(),
        "hits": {
            "status_branch_hit": hit(status_text, "8.7.56.1783"),
            "roadmap_branch_hit": hit(roadmap_text, "次の公式 branch は `.1783-.1786`"),
            "current_problem_branch_hit": hit(current_problem_text, "exact internal coherence / exact HH surface theorem"),
            "current_status_branch_hit": hit(current_status_text, "exact internal coherence or HH surface reactivation"),
            "unified_roadmap_branch_hit": hit(unified_text, "`.1779-.1782` は **mixed proxy decision gate / internal coherence reopen**"),
            "long_roadmap_branch_hit": hit(long_text, "25. `8.7.56.1783-.1786`"),
            "part5_branch_hit": hit(part5_text, "`.1779-.1782` の **mixed proxy decision gate / internal coherence reopen**"),
        },
        "carry_over": {
            "decision_summary": decision_summary,
            "proxy_summary": proxy_summary,
            "threshold_summary": threshold_summary,
            "theorem_summary": theorem_summary,
            "field_summary": field_summary,
            "energy_summary": energy_summary,
        },
    }

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload(
                "8.7.56.1783",
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
                "8.7.56.1784",
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
                "8.7.56.1785",
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
                "8.7.56.1786",
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
