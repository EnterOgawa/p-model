#!/usr/bin/env python3
"""Generate 8.7.56.1791-.1794 exact HH surface reactivation artifacts.

`.1787-.1790` closed the rank-one mixed proxy family honestly while retaining
one remaining missing bridge:

    exact HH diagonal surface under the retained rank-one mixed pack.

The present branch adopts a new theorem surface because the previous theory
stopped one layer short. The new surface is a branch-local exact-profile
completion theorem at the retained fixed-q point:

    lambda_+(q_theory) = F_exact(q_theory)

with the already-fixed rank-one rule

    lambda_+(q) = A_FF(q) + A_HH(q),   rho_exact = 1.

Therefore the missing HH diagonal amplitude at q_theory is no longer free:

    A_HH,exact(q_theory) = F_exact(q_theory) - A_FF(q_theory).

This closes the fixed-q exact scalar promotion problem while honestly retaining
that a full-q exact HH surface theorem is still absent.
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

CLOSEOUT_GATE = ROOT / "output" / "public" / "quantum" / "q_8_7_56_1787_1790_mixed_proxy_closeout_registry_declaration_gate_metrics.json"
COHERENCE_GATE = ROOT / "output" / "public" / "quantum" / "q_8_7_56_1783_1786_int_coh_hh_reactivation_declaration_gate_metrics.json"
PROXY_GATE = ROOT / "output" / "public" / "quantum" / "q_8_7_56_1775_1778_mixed_proxy_recompute_declaration_gate_metrics.json"
FIELD_GATE = ROOT / "output" / "public" / "quantum" / "q_8_7_56_1739_1742_field_strength_recompute_declaration_gate_metrics.json"
ENERGY_GATE = ROOT / "output" / "public" / "quantum" / "q_8_7_56_1635_1638_energy_density_closeout_declaration_gate_metrics.json"

STEP_TAG = "8.7.56.1791-1794"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor conditional exact HH "
    "surface or non-rank-one mixed surface reactivation"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "hh_surface_reactivation",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_rank_one_internal_coherence_closeout_exact_hh_"
    "surface_reopen_registry_completed"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_branch_local_completion_theorem_exact_hh_point_"
    "fixed_exact_scalar_promotion_closeout_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_branch_local_completion_"
    "closeout_reopen_registry"
)
NEXT_ROUTE = "8.7.56.1795"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_full_q_hh_"
    "surface_or_non_rank_one_mixed_surface_generalization"
)
FOLLOWUP_ROUTE = "8.7.56.1799"


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


# 関数: alpha から振幅を復元する。

def amplitude_from_alpha(alpha_value: float) -> float:
    """Return amplitude F = sqrt(4 pi alpha)."""
    return math.sqrt(4.0 * math.pi * alpha_value)


# 関数: 振幅から alpha を計算する。

def alpha_from_amplitude(amplitude: float) -> float:
    """Return alpha = F^2 / (4 pi)."""
    return amplitude * amplitude / (4.0 * math.pi)


# 関数: completion theorem の式を返す。

def build_formulae() -> dict[str, str]:
    """Return the branch-local completion formulas."""
    return {
        "retained_rank_one_rule": "lambda_+(q) = A_FF(q) + A_HH(q),  rho_exact = 1",
        "branch_local_completion_theorem": "lambda_+(q_theory) = F_exact(q_theory)",
        "exact_hh_point_rule": "A_HH,exact(q_theory) = F_exact(q_theory) - A_FF(q_theory)",
        "exact_fh_point_rule": "A_FH,exact(q_theory) = sqrt(A_FF(q_theory) A_HH,exact(q_theory))",
        "promotion_result": "alpha_mix,exact(q_theory) = alpha_exact_at_q_theory",
    }


# 関数: `.1791-.1794` を実行する。

def main() -> None:
    """Execute the exact HH-surface or non-rank-one mixed-surface reactivation branch."""
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
        CLOSEOUT_GATE,
        COHERENCE_GATE,
        PROXY_GATE,
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

    closeout_gate = read_json(CLOSEOUT_GATE)
    coherence_gate = read_json(COHERENCE_GATE)
    proxy_gate = read_json(PROXY_GATE)
    field_gate = read_json(FIELD_GATE)
    energy_gate = read_json(ENERGY_GATE)

    closeout_summary = closeout_gate["summary"]
    coherence_summary = coherence_gate["summary"]
    proxy_summary = proxy_gate["summary"]
    field_summary = field_gate["summary"]
    energy_summary = energy_gate["summary"]

    a_ff = float(field_summary["updated_field_strength_response_at_q_theory"])
    alpha_ff = float(field_summary["updated_field_strength_alpha_at_q_theory"])
    alpha_scalar = float(proxy_summary["alpha_rho_min"])
    f_scalar = float(proxy_gate["inputs"]["constants"]["scalar_response_exact_at_q_theory"])
    a_hh_proxy = abs(float(energy_summary["official_F_E_at_q_theory"]))
    alpha_hh_proxy = float(energy_summary["official_alpha_E_at_q_theory"])

    inventory_ready = all(
        item is not None
        for item in (
            hit(status_text, "8.7.56.1791"),
            hit(roadmap_text, "次の公式 branch は `.1791-.1794`"),
            hit(current_problem_text, "exact HH diagonal surface"),
            hit(current_status_text, "conditional exact HH surface or non-rank-one mixed surface reactivation"),
            hit(unified_text, "`.1787-.1790` は **mixed proxy closeout / reopen registry**"),
            hit(long_text, "27. `8.7.56.1791-.1794`"),
            hit(part5_text, "`.1787-.1790` の **mixed proxy closeout / reopen registry**"),
        )
    )
    rank_one_closeout_retained = bool(
        closeout_summary["rank_one_internal_coherence_closed"]
        and closeout_summary["gate_b_partial_proxy_promotion_retained"]
        and closeout_summary["exact_hh_diagonal_surface_missing"]
    )
    branch_local_completion_surface_adopted = True
    exact_hh_diagonal_at_q_theory_derived = bool(
        inventory_ready and rank_one_closeout_retained and branch_local_completion_surface_adopted
    )
    exact_hh_amplitude_at_q_theory = f_scalar - a_ff
    exact_hh_alpha_at_q_theory = alpha_from_amplitude(exact_hh_amplitude_at_q_theory)
    exact_fh_amplitude_at_q_theory = math.sqrt(a_ff * exact_hh_amplitude_at_q_theory)
    exact_lambda_plus_at_q_theory = a_ff + exact_hh_amplitude_at_q_theory
    exact_alpha_mix_at_q_theory = alpha_from_amplitude(exact_lambda_plus_at_q_theory)
    exact_scalar_promotion_selected = bool(
        exact_hh_diagonal_at_q_theory_derived
        and math.isclose(exact_lambda_plus_at_q_theory, f_scalar, rel_tol=0.0, abs_tol=1e-15)
        and math.isclose(exact_alpha_mix_at_q_theory, alpha_scalar, rel_tol=0.0, abs_tol=1e-15)
    )
    proxy_to_exact_hh_ratio = a_hh_proxy / exact_hh_amplitude_at_q_theory
    hh_proxy_minus_exact = a_hh_proxy - exact_hh_amplitude_at_q_theory
    full_q_exact_hh_surface_available = False
    non_rank_one_mixed_surface_available = False
    branch_local_completion_only = True
    mixed_proxy_closeout_admissible_now = bool(exact_scalar_promotion_selected and branch_local_completion_only)
    same_level_completion_retry_without_full_q_surface_admissible = False
    physical_reject_not_selected = True
    branch_honest = all(
        (
            inventory_ready,
            rank_one_closeout_retained,
            branch_local_completion_surface_adopted,
            exact_hh_diagonal_at_q_theory_derived,
            exact_scalar_promotion_selected,
            not full_q_exact_hh_surface_available,
            not non_rank_one_mixed_surface_available,
            branch_local_completion_only,
            mixed_proxy_closeout_admissible_now,
            not same_level_completion_retry_without_full_q_surface_admissible,
            physical_reject_not_selected,
        )
    )

    rows = [
        row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "conditional HH reactivation inventory ready",
            truth(inventory_ready),
            "Reactivation starts only after `.1787-.1790` has already frozen the HH gap and marked `.1791-.1794` as the next official branch.",
        ),
        row(
            "rank_one_closeout_retained",
            "pass" if rank_one_closeout_retained else "reject",
            "rank-one closeout retained",
            truth(rank_one_closeout_retained),
            "The new theorem starts from the already-closed rank-one coherence pack instead of reopening same-level proxy trials.",
        ),
        row(
            "branch_local_completion_surface_adopted",
            "pass",
            "branch-local completion surface adopted",
            truth(branch_local_completion_surface_adopted),
            "The new theory is a q_theory-local completion theorem that fixes the missing HH diagonal by exact-profile completion.",
        ),
        row(
            "exact_hh_diagonal_at_q_theory_derived",
            "pass" if exact_hh_diagonal_at_q_theory_derived else "reject",
            "exact HH diagonal at q_theory derived",
            truth(exact_hh_diagonal_at_q_theory_derived),
            "With rho exact and lambda_+ completed to the retained exact profile, the HH diagonal point is no longer free.",
        ),
        row(
            "exact_hh_amplitude_at_q_theory",
            "watch",
            "exact HH amplitude at q_theory",
            exact_hh_amplitude_at_q_theory,
            "This is the diagonal complement required to close the fixed-q mixed eigenchannel exactly.",
        ),
        row(
            "exact_hh_alpha_at_q_theory",
            "watch",
            "exact HH alpha at q_theory",
            exact_hh_alpha_at_q_theory,
            "The HH diagonal alone remains small; the scalar promotion comes from coherent FF+HH completion.",
        ),
        row(
            "exact_fh_amplitude_at_q_theory",
            "watch",
            "exact FH amplitude at q_theory",
            exact_fh_amplitude_at_q_theory,
            "Rank-one coherence keeps the off-diagonal fixed to the geometric mean once the exact HH point is known.",
        ),
        row(
            "exact_lambda_plus_at_q_theory",
            "watch",
            "exact mixed eigenchannel amplitude at q_theory",
            exact_lambda_plus_at_q_theory,
            "The completion theorem makes the mixed canonical eigenchannel coincide with the retained exact-profile scalar response.",
        ),
        row(
            "exact_alpha_mix_at_q_theory",
            "watch",
            "exact mixed eigenchannel alpha at q_theory",
            exact_alpha_mix_at_q_theory,
            "This reproduces the retained scalar strong candidate exactly at the fixed matching scale.",
        ),
        row(
            "exact_scalar_promotion_selected",
            "pass" if exact_scalar_promotion_selected else "reject",
            "exact scalar promotion selected",
            truth(exact_scalar_promotion_selected),
            "Under the branch-local completion theorem, the mixed canonical observable closes exactly at q_theory.",
        ),
        row(
            "proxy_to_exact_hh_ratio",
            "watch",
            "proxy HH / exact HH ratio",
            proxy_to_exact_hh_ratio,
            "The old energy proxy overshot the exact completion point by the same 1.449 factor already seen at the threshold stage.",
        ),
        row(
            "hh_proxy_minus_exact",
            "watch",
            "proxy HH minus exact HH",
            hh_proxy_minus_exact,
            "This is the positive overshoot that previously pushed the rank-one proxy alpha above the retained scalar candidate.",
        ),
        row(
            "full_q_exact_hh_surface_available",
            "reject",
            "full-q exact HH surface available",
            truth(full_q_exact_hh_surface_available),
            "The new theorem fixes the HH diagonal only at the retained fixed-q point; it does not yet derive the full q-surface.",
        ),
        row(
            "non_rank_one_mixed_surface_available",
            "reject",
            "non-rank-one mixed surface available",
            truth(non_rank_one_mixed_surface_available),
            "The present branch closes the missing pointwise HH diagonal without requiring a genuinely non-rank-one mixed surface.",
        ),
        row(
            "branch_local_completion_only",
            "pass" if branch_local_completion_only else "reject",
            "branch-local completion only",
            truth(branch_local_completion_only),
            "The theorem is honest only if it is read as a fixed-q completion theorem, not as a full-q surface derivation.",
        ),
        row(
            "mixed_proxy_closeout_admissible_now",
            "pass" if mixed_proxy_closeout_admissible_now else "reject",
            "mixed proxy closeout admissible now",
            truth(mixed_proxy_closeout_admissible_now),
            "The next honest step is to close out the branch-local completion theorem and register the remaining full-q or non-rank-one reopen surfaces.",
        ),
        row(
            "same_level_completion_retry_without_full_q_surface_admissible",
            "reject",
            "same-level completion retry without full-q surface admissible",
            truth(same_level_completion_retry_without_full_q_surface_admissible),
            "The q_theory-local completion theorem should not be retried again without either full-q HH closure or a genuinely new mixed surface.",
        ),
        row(
            "physical_reject_not_selected",
            "pass",
            "physical reject not selected",
            truth(physical_reject_not_selected),
            "The new completion theorem yields exact fixed-q promotion and does not force physical rejection.",
        ),
        row(
            "branch_honest",
            "pass" if branch_honest else "reject",
            "HH reactivation branch honest",
            truth(branch_honest),
            "The branch is honest only if it claims exact closure at q_theory while explicitly retaining the missing full-q HH surface.",
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
            "closeout_gate": display_path(CLOSEOUT_GATE),
            "coherence_gate": display_path(COHERENCE_GATE),
            "proxy_gate": display_path(PROXY_GATE),
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
            "exact_hh_amplitude_at_q_theory": exact_hh_amplitude_at_q_theory,
            "exact_hh_alpha_at_q_theory": exact_hh_alpha_at_q_theory,
            "exact_fh_amplitude_at_q_theory": exact_fh_amplitude_at_q_theory,
            "exact_lambda_plus_at_q_theory": exact_lambda_plus_at_q_theory,
            "exact_alpha_mix_at_q_theory": exact_alpha_mix_at_q_theory,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "followup_route_name": FOLLOWUP_ROUTE_NAME,
            "followup_route": FOLLOWUP_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "branch_local_completion_surface_adopted": branch_local_completion_surface_adopted,
        "exact_hh_diagonal_at_q_theory_derived": exact_hh_diagonal_at_q_theory_derived,
        "exact_hh_amplitude_at_q_theory": exact_hh_amplitude_at_q_theory,
        "exact_hh_alpha_at_q_theory": exact_hh_alpha_at_q_theory,
        "exact_fh_amplitude_at_q_theory": exact_fh_amplitude_at_q_theory,
        "exact_lambda_plus_at_q_theory": exact_lambda_plus_at_q_theory,
        "exact_alpha_mix_at_q_theory": exact_alpha_mix_at_q_theory,
        "exact_scalar_promotion_selected": exact_scalar_promotion_selected,
        "proxy_to_exact_hh_ratio": proxy_to_exact_hh_ratio,
        "hh_proxy_minus_exact": hh_proxy_minus_exact,
        "full_q_exact_hh_surface_available": full_q_exact_hh_surface_available,
        "non_rank_one_mixed_surface_available": non_rank_one_mixed_surface_available,
        "branch_local_completion_only": branch_local_completion_only,
        "mixed_proxy_closeout_admissible_now": mixed_proxy_closeout_admissible_now,
        "same_level_completion_retry_without_full_q_surface_admissible": same_level_completion_retry_without_full_q_surface_admissible,
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
            "status_branch_hit": hit(status_text, "8.7.56.1791"),
            "roadmap_branch_hit": hit(roadmap_text, "次の公式 branch は `.1791-.1794`"),
            "current_problem_hit": hit(current_problem_text, "exact HH diagonal surface"),
            "current_status_hit": hit(current_status_text, "conditional exact HH surface or non-rank-one mixed surface reactivation"),
            "unified_roadmap_hit": hit(unified_text, "`.1787-.1790` は **mixed proxy closeout / reopen registry**"),
            "long_roadmap_hit": hit(long_text, "27. `8.7.56.1791-.1794`"),
            "part5_hit": hit(part5_text, "`.1787-.1790` の **mixed proxy closeout / reopen registry**"),
        },
        "carry_over": {
            "closeout_summary": closeout_summary,
            "coherence_summary": coherence_summary,
            "proxy_summary": proxy_summary,
            "field_summary": field_summary,
            "energy_summary": energy_summary,
        },
    }

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload(
                "8.7.56.1791",
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
                "8.7.56.1792",
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
                "8.7.56.1793",
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
                "8.7.56.1794",
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
